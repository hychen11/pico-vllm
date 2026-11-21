"""The M6 benchmark workload runs end to end and counts tokens correctly."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from pico_vllm.benchmark import make_prompts, qwen3_0_6b_config, run_benchmark  # noqa: E402

pytestmark = pytest.mark.integration

TINY = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
)


def _run(device: str) -> None:
    result = run_benchmark(
        num_seqs=5, min_input_len=3, max_input_len=20, output_len=6, device=device,
        dtype=torch.float32, kvcache_capacity_bytes=2**20, block_size=4,
        max_num_batched_tokens=32, hf_config=qwen3_0_6b_config(**TINY),
    )
    prompts = make_prompts(5, 3, 20, TINY["vocab_size"], seed=0)
    assert result.prompt_tokens == sum(len(p) for p in prompts)
    assert result.output_tokens == 5 * 6  # ignore_eos: every seq runs to output_len
    assert result.prefill_seconds > 0 and result.total_seconds > 0
    assert result.prefill_tok_s > 0 and result.decode_tok_s > 0


def test_make_prompts_is_seeded() -> None:
    a = make_prompts(4, 2, 9, 50, seed=1)
    assert a == make_prompts(4, 2, 9, 50, seed=1)
    assert all(2 <= len(p) <= 9 and all(0 <= t < 50 for t in p) for p in a)


def test_benchmark_cpu() -> None:
    _run("cpu")


@pytest.mark.cuda
@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_benchmark_cuda() -> None:
    _run("cuda")
