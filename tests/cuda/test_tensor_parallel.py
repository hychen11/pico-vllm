"""Tensor parallelism on two GPUs (M8 step 7), with the real Qwen3-0.6B.

The pytest process holds the session's world_size=1 group and one process holds
one group, so every TP=2 engine runs in a spawned child (whose rank 0 spawns rank 1
in turn). Results come back through a file.

- ``test_tp2_steps_match_tp1``: the child drives a TP=2 engine step by step through
  chunked prefill, a prefix-cache hit, decode and preemption, checks the M7
  invariants after every step, and records each step's batch, logits and greedy
  tokens. This process then replays the same steps on a TP=1 model, feeding it the
  child's tokens (teacher forcing), and compares the logits. float32: in bf16 the
  different summation order of all_reduce flips near-tied tokens, see
  docs/m8-tp-guide.md (step 5).
- ``test_tp2_llm_generates_and_exits_cleanly``: ``LLM(..., tensor_parallel_size=2)``
  generates sensible text in bf16; ``exit`` stops the worker, removes the shared
  memory and destroys the group; a second engine in the same process works too.
"""

from __future__ import annotations

import gc
import os
import traceback
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.model,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs"),
]

BS = 16
BUDGET = 64  # max_num_batched_tokens: prompts longer than this are chunked
MAX_SEQS = 4
NUM_BLOCKS = 20  # tight: the four seqs need ~25 blocks at the end -> preemption
MAX_TOKENS = 24
LATE_STEP = 6  # D joins here, after A's prefix blocks are hashed


@pytest.fixture
def model_path() -> str:
    path = os.environ.get("PICO_VLLM_QWEN_MODEL")
    if not path:
        pytest.skip("set PICO_VLLM_QWEN_MODEL to a local Qwen3 dir")
    return os.path.expanduser(path)


def _child(fn: Callable, args: tuple, out: str) -> None:
    try:
        result = ("ok", fn(*args))
    except BaseException:  # noqa: BLE001
        result = ("error", traceback.format_exc())
    torch.save(result, out)


def run_in_child(fn: Callable, *args: Any, out: str, timeout: float = 600) -> Any:
    """``fn(*args)`` in a fresh spawned process; returns its result or raises with
    its traceback. ``fn`` must be module-level (spawn pickles it by name).

    Empties this process's CUDA cache first: earlier tests' models are freed but
    their memory stays reserved by the caching allocator, and the child sizes its
    KV cache from what the driver reports free (``gpu_memory_utilization``)."""
    gc.collect()
    torch.cuda.empty_cache()
    p = mp.get_context("spawn").Process(target=_child, args=(fn, args, out))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        pytest.fail(f"child still running after {timeout}s")
    assert os.path.exists(out), f"child died without a result (exitcode {p.exitcode})"
    status, result = torch.load(out, weights_only=False)
    assert status == "ok", f"child failed:\n{result}"
    return result


def prompts(vocab: int) -> tuple[list[list[int]], list[int]]:
    """A, B, C at the start, D at LATE_STEP. A: chunked (78 > BUDGET); B: chunked over
    three steps; D: shares A's first 3 full blocks (prefix hit)."""
    g = torch.Generator().manual_seed(0)

    def rand(n: int) -> list[int]:
        return torch.randint(0, vocab, (n,), generator=g).tolist()

    prefix = rand(3 * BS)
    a, b, c, d = prefix + rand(30), rand(150), rand(40), prefix + rand(20)
    return [a, b, c, d], [0, 0, 0, LATE_STEP]


def drive(scheduler: Any, run: Callable, vocab: int, record: Callable) -> list[Any]:
    """Step ``scheduler`` to the end. ``run(seqs, is_prefill)`` -> (logits, tokens);
    ``record(step, batch, is_prefill, logits, tokens)``. Same on both sides, so the
    same tokens give the same batches."""
    from pico_vllm.sequence import Sequence

    ids, add_at = prompts(vocab)
    seqs = [Sequence(p) for p in ids]
    added: list[Any] = []  # what the invariants may look at
    step = 0
    while step == 0 or not scheduler.is_finished() or step <= LATE_STEP:
        for s, at in zip(seqs, add_at, strict=True):
            if at == step:
                scheduler.add(s)
                added.append(s)
        if scheduler.is_finished():
            step += 1
            continue
        batch, is_prefill = scheduler.schedule()
        key = [(seqs.index(q), q.num_cached_tokens, q.num_scheduled_tokens) for q in batch]
        logits, tokens = run(batch, is_prefill)
        scheduler.postprocess(batch, tokens)
        for q in batch:
            if q.num_completion_tokens >= MAX_TOKENS:
                scheduler.finish(q)
        record(step, key, is_prefill, logits, tokens, added)
        step += 1
    return seqs


def tp2_steps(model_path: str) -> dict[str, Any]:
    """Child: TP=2 engine, record every step."""
    from conftest import check_scheduler_invariants
    from pico_vllm.config import Config
    from pico_vllm.core.paged_kv_cache import kv_block_bytes
    from pico_vllm.engine.llm_engine import LLMEngine

    probe = Config(model_path)  # only for the shapes
    hf = probe.hf_config
    block_bytes = kv_block_bytes(hf.num_hidden_layers, BS, hf.num_key_value_heads // 2,
                                 hf.head_dim, torch.float32)
    engine = LLMEngine.from_config(Config(
        model_path, tensor_parallel_size=2, use_flash=False, dtype="float32",
        kvcache_block_size=BS, kvcache_capacity_bytes=NUM_BLOCKS * block_bytes,
        max_num_seqs=MAX_SEQS, max_num_batched_tokens=BUDGET, max_model_len=512,
    ))
    sched, runner = engine.scheduler, engine.model_runner
    counts = {"preempts": 0, "chunks": 0, "prefix_hits": 0}
    preempt, allocate = sched.preempt, sched.block_manager.allocate

    def counting_preempt(seq: Any) -> None:
        counts["preempts"] += 1
        preempt(seq)

    def counting_allocate(seq: Any, num_cached_blocks: int) -> None:
        counts["prefix_hits"] += num_cached_blocks > 0
        allocate(seq, num_cached_blocks)

    sched.preempt = counting_preempt
    sched.block_manager.allocate = counting_allocate

    def run(batch: list, is_prefill: bool) -> tuple[torch.Tensor, list[int]]:
        counts["chunks"] += is_prefill and any(
            q.num_cached_tokens + q.num_scheduled_tokens < q.num_tokens for q in batch)
        logits = runner.call("run", batch, is_prefill)
        return logits, logits.argmax(-1).tolist()

    steps: list[dict[str, Any]] = []

    def record(step: int, key: list, is_prefill: bool, logits: torch.Tensor,
               tokens: list[int], seqs: list) -> None:
        check_scheduler_invariants(sched, seqs)
        steps.append(dict(step=step, key=key, is_prefill=is_prefill,
                          logits=logits.float().cpu(), tokens=tokens))

    try:
        drive(sched, run, hf.vocab_size, record)
        return dict(steps=steps, counts=counts,
                    num_blocks=runner.kv_cache.k_cache.shape[1])
    finally:
        engine.exit()


def test_tp2_steps_match_tp1(model_path: str, tmp_path: Any,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    from pico_vllm.config import Config
    from pico_vllm.core.paged_kv_cache import PagedKVCache
    from pico_vllm.core.scheduler import Scheduler
    from pico_vllm.engine.llm_engine import build_model
    from pico_vllm.sequence import Sequence
    from pico_vllm.worker.model_runner import ModelRunner

    ref = run_in_child(tp2_steps, model_path, out=str(tmp_path / "tp2.pt"))
    counts = ref["counts"]
    # The hard paths really ran (otherwise the comparison proves little).
    assert ref["num_blocks"] == NUM_BLOCKS
    assert counts["chunks"] >= 2 and counts["prefix_hits"] >= 1 and counts["preempts"] >= 1

    monkeypatch.setattr(Sequence, "block_size", BS)
    config = Config(model_path, use_flash=False, dtype="float32", device="cuda:0",
                    kvcache_block_size=BS, max_model_len=512)
    hf = config.hf_config
    model = build_model(config)
    cache = PagedKVCache(hf.num_hidden_layers, NUM_BLOCKS, BS, hf.num_key_value_heads,
                         hf.head_dim, dtype=torch.float32, device=config.device)
    runner = ModelRunner(model, cache, BS, use_flash=False)
    scheduler = Scheduler(MAX_SEQS, BUDGET, NUM_BLOCKS, BS)

    it = iter(ref["steps"])
    worst = 0.0

    def run(batch: list, is_prefill: bool) -> tuple[torch.Tensor, list[int]]:
        expected = next(it)
        return runner.run(batch, is_prefill), expected["tokens"]  # teacher forcing

    def record(step: int, key: list, is_prefill: bool, logits: torch.Tensor,
               tokens: list[int], seqs: list) -> None:
        nonlocal worst
        expected = ref["steps"][len(seen)]
        seen.append(step)
        assert (step, key, is_prefill) == (
            expected["step"], expected["key"], expected["is_prefill"]), "batches diverged"
        got = logits.float().cpu()
        worst = max(worst, (got - expected["logits"]).abs().max().item())
        # fp32: ~3e-5 seen on Qwen3-0.6B
        torch.testing.assert_close(got, expected["logits"], atol=1e-3, rtol=1e-4)

    seen: list[int] = []
    seqs = drive(scheduler, run, hf.vocab_size, record)
    assert len(seen) == len(ref["steps"])
    assert all(q.is_finished for q in seqs)
    print(f"{len(seen)} steps, {counts}, max |logit diff| TP=2 vs TP=1: {worst:.2e}")


def tp2_generate_twice(model_path: str) -> list[dict[str, Any]]:
    """Child: two TP=2 LLMs one after the other; generate, exit, check the cleanup."""
    import torch.distributed as dist

    from pico_vllm.llm import LLM
    from pico_vllm.sampling_params import SamplingParams

    results = []
    for _ in range(2):
        llm = LLM(model_path, tensor_parallel_size=2, use_flash=False,
                  max_num_seqs=8, max_num_batched_tokens=1024, max_model_len=512,
                  gpu_memory_utilization=0.5)
        engine = llm.engine
        shm = "/dev/shm/" + engine.model_runner.shm.name
        assert os.path.exists(shm) and dist.get_world_size() == 2
        out = llm.generate(["The capital of France is"],
                           SamplingParams(temperature=0.0, max_tokens=8))
        procs = list(engine.processes)
        engine.exit()
        results.append(dict(text=out[0]["text"], exitcodes=[p.exitcode for p in procs],
                            shm_left=os.path.exists(shm),
                            group_left=dist.is_initialized()))
    return results


def test_tp2_llm_generates_and_exits_cleanly(model_path: str, tmp_path: Any) -> None:
    results = run_in_child(tp2_generate_twice, model_path, out=str(tmp_path / "gen.pt"))
    for r in results:
        assert "Paris" in r["text"], r["text"]
        assert r["exitcodes"] == [0]
        assert not r["shm_left"] and not r["group_left"]
