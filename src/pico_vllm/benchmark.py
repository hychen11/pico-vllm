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
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from pico_vllm.config import DTYPES, resolve_use_flash
from pico_vllm.core.paged_kv_cache import PagedKVCache, num_blocks_from_capacity
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.engine.llm_engine import LLMEngine
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence
from pico_vllm.utils.loader import init_on, load_model
from pico_vllm.worker.model_runner import ModelRunner


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
) -> LLMEngine:
    """Wrap an already-placed ``model`` in a fresh scheduler, KV cache and engine."""
    param = next(model.parameters())
    head_dim = hf_config.head_dim
    num_blocks = num_blocks_from_capacity(
        kvcache_capacity_bytes, hf_config.num_hidden_layers, block_size,
        hf_config.num_key_value_heads, head_dim, param.dtype,
    )
    kv_cache = PagedKVCache(
        hf_config.num_hidden_layers, num_blocks, block_size, hf_config.num_key_value_heads,
        head_dim, dtype=param.dtype, device=param.device, use_triton=use_triton,
    )
    Sequence.block_size = block_size
    scheduler = Scheduler(max_num_seqs, max_num_batched_tokens, num_blocks, block_size)
    use_flash = resolve_use_flash(use_flash, param.device, param.dtype, block_size)
    runner = ModelRunner(model, kv_cache, block_size, use_flash=use_flash)
    return LLMEngine(scheduler, runner, eos_token_id=None)


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
) -> BenchmarkResult:
    """``model_path``: load that checkpoint (and its config); otherwise random weights
    with ``hf_config`` (default: Qwen3-0.6B shapes)."""
    if model_path is not None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)
    elif hf_config is None:
        hf_config = qwen3_0_6b_config()
    torch.manual_seed(seed)
    dev = torch.device(device)
    with init_on(dev, dtype):
        model = Qwen3ForCausalLM(hf_config)
    if model_path is not None:
        load_model(model, model_path)
    model.eval()

    def fresh_engine() -> LLMEngine:
        return build_engine(
            model, hf_config, kvcache_capacity_bytes=kvcache_capacity_bytes,
            block_size=block_size, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens, use_triton=use_triton,
            use_flash=use_flash,
        )

    prompts = make_prompts(num_seqs, min_input_len, max_input_len, hf_config.vocab_size, seed)
    prompt_tokens = sum(len(p) for p in prompts)

    if warmup:
        warm = make_prompts(2, 16, 64, hf_config.vocab_size, seed + 1)
        generate(fresh_engine(), warm,
                 SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))

    def timed(max_tokens: int) -> tuple[float, int]:
        engine = fresh_engine()
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        _synchronize(dev)
        start = time.perf_counter()
        output_tokens = generate(engine, prompts, sp)
        _synchronize(dev)
        return time.perf_counter() - start, output_tokens

    prefill_seconds, _ = timed(1)
    total_seconds, output_tokens = timed(output_len)
    return BenchmarkResult(num_seqs, prompt_tokens, output_tokens, prefill_seconds,
                           total_seconds)


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
    )
    name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"
    weights = args.model or "random (Qwen3-0.6B shapes)"
    print(f"device={name} dtype={args.dtype} triton_kv_store={not args.no_triton} "
          f"flash={not args.no_flash} weights={weights}")
    print(result)


if __name__ == "__main__":
    main()
