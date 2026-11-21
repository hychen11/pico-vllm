"""Generate with a local Qwen3 checkpoint (M5).

    uv run hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
    uv run python example.py ~/models/Qwen3-0.6B
"""

from __future__ import annotations

import os
import sys
import time

from pico_vllm.llm import LLM
from pico_vllm.sampling_params import SamplingParams


def main(model_path: str) -> None:
    """Build an ``LLM``, generate for a few prompts, and print prompt -> completion."""
    llm = LLM(
        os.path.expanduser(model_path),
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        kvcache_block_size=16,
        kvcache_capacity_bytes=512 * 2**20,
    )
    prompts = [
        "The capital of France is",
        "List three prime numbers:",
        "def fibonacci(n):",
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=24),  # greedy
        SamplingParams(temperature=0.6, max_tokens=24),
        SamplingParams(temperature=0.0, max_tokens=48),
    ]

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    for prompt, out in zip(prompts, outputs, strict=True):
        print(f"--- {prompt!r}\n{out['text']!r}\n")
    num_tokens = sum(len(out["token_ids"]) for out in outputs)
    print(f"{num_tokens} tokens in {elapsed:.1f}s ({num_tokens / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python example.py /path/to/Qwen3-dir")
    main(sys.argv[1])
