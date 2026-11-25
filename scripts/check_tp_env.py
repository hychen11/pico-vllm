"""M8 step 0: can this machine run tensor parallelism? (no project code changes)

    uv run python scripts/check_tp_env.py

1. per GPU: name, compute capability, FA2 varlen + kvcache, Triton KV store
2. gloo on CPU, 2 processes: all_reduce / gather (what the CPU layer tests use)
3. NCCL across the GPUs, 2 processes: all_reduce / gather correctness, then
   latency and bandwidth at the message sizes Qwen3-0.6B TP=2 sends per layer
"""

from __future__ import annotations

import os
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------- 1. per-GPU kernels


def check_kernels(dev: int) -> None:
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    print(f"\n[cuda:{dev}] {name} sm_{cap[0]}{cap[1]}")
    torch.cuda.set_device(dev)
    d = f"cuda:{dev}"
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        q = torch.randn(8, 4, 128, device=d, dtype=torch.bfloat16)
        kc = torch.randn(2, 256, 2, 128, device=d, dtype=torch.bfloat16)  # 2 blocks
        cu = torch.tensor([0, 8], device=d, dtype=torch.int32)
        bt = torch.tensor([[0, 1]], device=d, dtype=torch.int32)
        flash_attn_varlen_func(q, kc, kc, cu, cu, 8, 8, causal=True, block_table=bt)
        flash_attn_with_kvcache(q[:1].unsqueeze(1), kc, kc,
                                cache_seqlens=torch.tensor([8], device=d, dtype=torch.int32),
                                block_table=bt, causal=True)
        torch.cuda.synchronize(dev)
        print("  FA2 varlen + kvcache : OK")
    except Exception as e:  # noqa: BLE001
        print(f"  FA2                  : FAIL  {type(e).__name__}: {str(e).splitlines()[0]}")
    try:
        from pico_vllm.kernels.kv_store import store_kvcache

        k = torch.randn(3, 2, 128, device=d, dtype=torch.bfloat16)
        cache = torch.zeros(2, 4, 2, 128, device=d, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(cache)
        store_kvcache(k, k, cache, v_cache, torch.tensor([5, -1, 2], device=d))
        torch.cuda.synchronize(dev)
        ok = torch.equal(cache.view(8, 2, 128)[5], k[0])
        print(f"  Triton KV store      : {'OK' if ok else 'WRONG RESULT'}")
    except Exception as e:  # noqa: BLE001
        print(f"  Triton KV store      : FAIL  {type(e).__name__}: {str(e).splitlines()[0]}")


# ------------------------------------------------------------ 2./3. collectives


def worker(rank: int, world: int, backend: str, port: int, q: mp.Queue) -> None:
    try:
        dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{port}",
                                world_size=world, rank=rank)
        if backend == "nccl":
            torch.cuda.set_device(rank)  # rank i -> cuda:i (cuda:0 = 5090)
            dev = torch.device("cuda", rank)
        else:
            dev = torch.device("cpu")

        # correctness: all_reduce sums, gather concatenates on rank 0
        x = torch.full((4,), float(rank + 1), device=dev)
        dist.all_reduce(x)
        assert torch.equal(x.cpu(), torch.full((4,), float(sum(range(1, world + 1)))))
        y = torch.full((2,), float(rank), device=dev)
        out = [torch.empty_like(y) for _ in range(world)] if rank == 0 else None
        dist.gather(y, out, dst=0)
        if rank == 0:
            assert torch.equal(torch.cat(out).cpu(), torch.tensor([0., 0., 1., 1.]))
        dist.barrier()
        result = {"rank": rank, "ok": True}

        if backend == "nccl":
            # Qwen3-0.6B TP=2 per layer: 2 all_reduces of [num_tokens, 1024] bf16.
            # decode with 32 seqs -> 64 KiB; prefill with 4096 tokens -> 8 MiB.
            timings = {}
            for label, n_tok in [("decode 32 seqs (64 KiB)", 32),
                                 ("prefill 4096 tok (8 MiB)", 4096)]:
                t = torch.randn(n_tok, 1024, device=dev, dtype=torch.bfloat16)
                for _ in range(5):
                    dist.all_reduce(t)
                torch.cuda.synchronize(dev)
                iters = 200 if n_tok == 32 else 20
                start = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(t)
                torch.cuda.synchronize(dev)
                us = (time.perf_counter() - start) / iters * 1e6
                timings[label] = (us, t.numel() * 2 / us / 1e3)  # GB/s of payload
            result["timings"] = timings
        dist.destroy_process_group()
        q.put(result)
    except Exception:  # noqa: BLE001
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def check_collectives(backend: str, world: int = 2) -> None:
    print(f"\n[{backend}] {world} processes")
    ctx = mp.get_context("spawn")  # never fork after CUDA is initialized
    q = ctx.Queue()
    port = free_port()
    ps = [ctx.Process(target=worker, args=(r, world, backend, port, q)) for r in range(world)]
    for p in ps:
        p.start()
    results = sorted((q.get(timeout=120) for _ in ps), key=lambda r: r["rank"])
    for p in ps:
        p.join(timeout=30)
    for r in results:
        if not r["ok"]:
            print(f"  rank {r['rank']} FAIL\n{r['error']}")
            return
    print("  all_reduce + gather  : OK")
    if "timings" in results[0]:
        for label, (us, gbps) in results[0]["timings"].items():
            print(f"  all_reduce {label:26s}: {us:8.1f} us  ({gbps:.2f} GB/s)")
        # 28 layers x 2 all_reduces per step
        dec_us = results[0]["timings"]["decode 32 seqs (64 KiB)"][0]
        print(f"  -> per decode step, 56 all_reduces ~ {56 * dec_us / 1e3:.1f} ms")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL "
          f"{'.'.join(map(str, torch.cuda.nccl.version()))}, "
          f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '(unset: FASTEST_FIRST)')}")
    for i in range(torch.cuda.device_count()):
        check_kernels(i)
    check_collectives("gloo")
    if torch.cuda.device_count() >= 2:
        check_collectives("nccl")
    else:
        print("\n[nccl] skipped: fewer than 2 GPUs visible")
