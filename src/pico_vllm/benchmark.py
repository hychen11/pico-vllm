"""Fixed throughput workload for M6 (CUDA fast path) before/after comparisons.

Runs the real ``LLMEngine`` step loop on a local Qwen3 checkpoint (``--model``,
default ``$PICO_VLLM_QWEN_MODEL``). Without one it falls back to a randomly
initialised Qwen3 with Qwen3-0.6B shapes: throughput depends only on shapes, and
on Qwen3-0.6B both give the same numbers within 1%. Prompts are seeded random
token ids and every request runs to exactly ``output_len`` tokens (``ignore_eos``).

Two passes, each on a fresh scheduler + KV cache (so pass 2 gets no prefix-cache
hits from pass 1):

1. ``max_tokens=1``: prefill only -> prefill tok/s;
2. ``max_tokens=output_len``: prefill + decode -> decode tok/s is
   ``num_seqs * (output_len - 1) / (t_full - t_prefill)``.

    uv run python -m pico_vllm.benchmark --model ~/models/Qwen3-0.6B
    uv run python -m pico_vllm.benchmark --num-seqs 64 --max-output-len 256
    uv run python -m pico_vllm.benchmark --model ~/models/Qwen3-0.6B --no-flash --tp 2

TP>1 (M8), as nano-vllm: one ``LLMEngine.from_config`` (it spawns the workers);
each pass only gets a fresh scheduler, which is enough to drop the prefix cache.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from pico_vllm.config import DTYPES, Config, resolve_use_flash
from pico_vllm.core.paged_kv_cache import PagedKVCache, num_blocks_from_capacity
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.engine.llm_engine import LLMEngine
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence
from pico_vllm.utils.loader import init_on, load_model
from pico_vllm.worker.model_runner import ModelRunner
from pico_vllm.worker.process_group import init_process_group


def qwen3_0_6b_config(**overrides: Any) -> Any:
    """``transformers.Qwen3Config`` with Qwen3-0.6B shapes (``overrides`` for tiny tests)."""
    from transformers import Qwen3Config

    shapes: dict[str, Any] = dict(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )
    shapes.update(overrides)
    return Qwen3Config(**shapes)


@dataclass
class BenchmarkResult:
    num_seqs: int
    prompt_tokens: int
    output_tokens: int
    prefill_seconds: float
    total_seconds: float
    # What the run actually used (read off the runner), not what was asked for:
    # e.g. no FlashAttention -> no CUDA Graph, and TP>1 forces eager.
    triton_kv_store: bool = False
    flash: bool = False
    cuda_graph: bool = False

    @property
    def prefill_tok_s(self) -> float:
        return self.prompt_tokens / self.prefill_seconds

    @property
    def decode_tok_s(self) -> float:
        # Pass 1 already produced each seq's first token; the rest is decode.
        decode_tokens = self.output_tokens - self.num_seqs
        return decode_tokens / max(self.total_seconds - self.prefill_seconds, 1e-9)

    @property
    def total_tok_s(self) -> float:
        return (self.prompt_tokens + self.output_tokens) / self.total_seconds

    def __str__(self) -> str:
        return (
            f"seqs={self.num_seqs} prompt_tokens={self.prompt_tokens} "
            f"output_tokens={self.output_tokens}\n"
            f"prefill: {self.prefill_seconds:7.2f}s  {self.prefill_tok_s:10.1f} tok/s\n"
            f"decode:  {self.total_seconds - self.prefill_seconds:7.2f}s  "
            f"{self.decode_tok_s:10.1f} tok/s\n"
            f"total:   {self.total_seconds:7.2f}s  {self.total_tok_s:10.1f} tok/s"
        )


def build_engine(
        model: Qwen3ForCausalLM,
        hf_config: Any,
        *,
        kvcache_capacity_bytes: int,
        block_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        use_triton: bool | None = None,
        use_flash: bool | None = None,
        use_cuda_graph: bool = True,
        max_model_len: int = 4096,
) -> LLMEngine:
    """Wrap an already-placed ``model`` in a fresh scheduler, KV cache and engine."""
    param = next(model.parameters())
    head_dim = hf_config.head_dim
    num_kv_heads = hf_config.num_key_value_heads // dist.get_world_size()  # this rank's
    num_blocks = num_blocks_from_capacity(
        kvcache_capacity_bytes, hf_config.num_hidden_layers, block_size,
        num_kv_heads, head_dim, param.dtype,
    )
    kv_cache = PagedKVCache(
        hf_config.num_hidden_layers, num_blocks, block_size, num_kv_heads,
        head_dim, dtype=param.dtype, device=param.device, use_triton=use_triton,
    )
    Sequence.block_size = block_size
    scheduler = Scheduler(max_num_seqs, max_num_batched_tokens, num_blocks, block_size)
    use_flash = resolve_use_flash(use_flash, param.device, param.dtype, block_size)
    runner = ModelRunner(model, kv_cache, block_size, use_flash=use_flash)
    if use_flash and kv_cache.use_triton and use_cuda_graph:
        runner.capture_cudagraphs(max_num_seqs, max_model_len)
    return LLMEngine(scheduler, runner, eos_token_id=None, max_model_len=max_model_len)


def generate(engine: LLMEngine, prompts: list[list[int]], sp: SamplingParams) -> int:
    """Run every prompt to completion; return the number of generated tokens."""
    for prompt in prompts:
        engine.add_request(prompt, sp)
    output_tokens = 0
    while not engine.is_finished():
        output_tokens += sum(seq.num_completion_tokens for seq in engine.step())
    return output_tokens


def make_prompts(
        num_seqs: int, min_len: int, max_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(rng.randint(min_len, max_len))]
        for _ in range(num_seqs)
    ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
        *,
        num_seqs: int = 32,
        min_input_len: int = 128,
        max_input_len: int = 512,
        output_len: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        kvcache_capacity_bytes: int = 4 * 2**30,
        block_size: int = 256,
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 16384,
        seed: int = 0,
        hf_config: Any = None,
        model_path: str | None = None,
        warmup: bool = True,
        use_triton: bool | None = None,
        use_flash: bool | None = None,
        use_cuda_graph: bool = True,
        tensor_parallel_size: int = 1,
) -> BenchmarkResult:
    """``model_path``: load that checkpoint (and its config); otherwise random weights
    with ``hf_config`` (default: Qwen3-0.6B shapes). ``tensor_parallel_size`` > 1
    needs ``model_path``; ``use_triton`` / ``use_cuda_graph`` then do not apply."""
    tp = tensor_parallel_size
    if tp > 1 and model_path is None:
        raise ValueError("tensor_parallel_size > 1 needs model_path")
    if model_path is not None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
    elif hf_config is None:
        hf_config = qwen3_0_6b_config()
    torch.manual_seed(seed)
    dev = torch.device(device)
    if tp > 1:  # as nano-vllm: the model lives in the workers, so one engine
        engine = LLMEngine.from_config(Config(
            model_path, tensor_parallel_size=tp, device=dev, dtype=dtype,
            use_flash=use_flash, kvcache_capacity_bytes=kvcache_capacity_bytes,
            kvcache_block_size=block_size, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ))
    else:
        init_process_group(1, dev)  # the layers read the TP rank/size from it
        with init_on(dev, dtype):
            model = Qwen3ForCausalLM(hf_config)
        if model_path is not None:
            load_model(model, model_path)
        model.eval()

    def fresh_engine() -> LLMEngine:
        if tp > 1:  # a fresh scheduler drops the prefix cache of the last pass
            num_blocks = engine.model_runner.kv_cache.k_cache.shape[1]
            engine.scheduler = Scheduler(max_num_seqs, max_num_batched_tokens,
                                         num_blocks, block_size)
            return engine
        return build_engine(
            model, hf_config, kvcache_capacity_bytes=kvcache_capacity_bytes,
            block_size=block_size, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens, use_triton=use_triton,
            use_flash=use_flash, use_cuda_graph=use_cuda_graph,
        )

    prompts = make_prompts(num_seqs, min_input_len, max_input_len, hf_config.vocab_size, seed)
    prompt_tokens = sum(len(p) for p in prompts)

    def timed(max_tokens: int) -> tuple[float, int, ModelRunner]:
        engine = fresh_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        _synchronize(dev)
        start = time.perf_counter()
        output_tokens = generate(engine, prompts, sp)
        _synchronize(dev)
        return time.perf_counter() - start, output_tokens, engine.model_runner

    try:
        if warmup:
            warm = make_prompts(2, 16, 64, hf_config.vocab_size, seed + 1)
            generate(fresh_engine(), warm,
                     SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
        prefill_seconds, _, _ = timed(1)
        total_seconds, output_tokens, runner = timed(output_len)
    finally:
        if tp > 1:
            engine.exit()
    return BenchmarkResult(num_seqs, prompt_tokens, output_tokens, prefill_seconds,
                           total_seconds, triton_kv_store=runner.kv_cache.use_triton,
                           flash=runner.use_flash, cuda_graph=bool(runner.graphs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=os.environ.get("PICO_VLLM_QWEN_MODEL"),
                        help="local Qwen3 checkpoint dir (default: $PICO_VLLM_QWEN_MODEL); "
                             "without one, random weights with Qwen3-0.6B shapes")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--kv-cache-gib", type=float, default=4.0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-triton", action="store_true",
                        help="force the eager KV-cache write (A/B against the Triton kernel)")
    parser.add_argument("--no-flash", action="store_true",
                        help="force the eager attention (A/B against FlashAttention 2)")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="launch decode kernels one by one (A/B against CUDA Graph)")
    parser.add_argument("--tp", type=int, default=1,
                        help="tensor_parallel_size (needs --model; rank i uses cuda:i, "
                             "pass --no-flash unless every GPU runs FlashAttention 2)")
    args = parser.parse_args(argv)

    result = run_benchmark(
        num_seqs=args.num_seqs,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        output_len=args.max_output_len,
        device=args.device,
        dtype=DTYPES[args.dtype],
        kvcache_capacity_bytes=int(args.kv_cache_gib * 2**30),
        block_size=args.block_size,
        seed=args.seed,
        model_path=os.path.expanduser(args.model) if args.model else None,
        use_triton=False if args.no_triton else None,
        use_flash=False if args.no_flash else None,
        use_cuda_graph=not args.no_cuda_graph,
        tensor_parallel_size=args.tp,
    )
    name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"
    weights = args.model or "random (Qwen3-0.6B shapes)"
    print(f"device={name} dtype={args.dtype} triton_kv_store={result.triton_kv_store} "
          f"flash={result.flash} cuda_graph={result.cuda_graph} "
          f"tp={args.tp} weights={weights}")
    print(result)


if __name__ == "__main__":
    main()
