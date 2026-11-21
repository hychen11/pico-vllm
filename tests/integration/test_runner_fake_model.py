"""Scheduler + ModelRunner + tiny random Qwen3 (M4).

The key check: logits produced through the paged KV cache (prefill, chunked
prefill, prefix-cache hits, step-by-step decode) equal a full recomputation of
the whole sequence without any cache.
"""

from __future__ import annotations

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.sequence import Sequence
from pico_vllm.utils.context import get_context
from pico_vllm.worker.model_runner import ModelRunner

transformers = pytest.importorskip("transformers")
from pico_vllm.models.qwen3 import Qwen3ForCausalLM  # noqa: E402

pytestmark = pytest.mark.integration

BS, NUM_BLOCKS, LAYERS, KV_HEADS, HEAD_DIM, VOCAB = 4, 32, 2, 2, 16, 128


@pytest.fixture(autouse=True)
def _small_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sequence.block_size is a class attribute; it must match the scheduler's.
    monkeypatch.setattr(Sequence, "block_size", BS)


@pytest.fixture(scope="module")
def model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


def make_runner(model: Qwen3ForCausalLM) -> ModelRunner:
    cache = PagedKVCache(LAYERS, NUM_BLOCKS, BS, KV_HEADS, HEAD_DIM)
    return ModelRunner(model, cache, BS)


def make_scheduler(budget: int = 64) -> Scheduler:
    return Scheduler(
        max_num_seqs=4, max_num_batched_tokens=budget, num_blocks=NUM_BLOCKS, block_size=BS
    )


def full_last_logits(model: Qwen3ForCausalLM, token_ids: list[int]) -> torch.Tensor:
    """Reference: run the whole sequence with no context (no KV cache)."""
    with torch.inference_mode():
        ids = torch.tensor(token_ids)
        hidden = model(ids, torch.arange(len(token_ids)))
        return model.compute_logits(hidden)[-1]


def drive(
    scheduler: Scheduler, runner: ModelRunner, model: Qwen3ForCausalLM, steps: int
) -> int:
    """Play the engine (M5) role for ``steps`` steps with greedy sampling.

    Whenever a seq's last scheduled token is its last token (prompt fully
    prefilled, or a decode step), check its logits against full recomputation.
    Returns how many such checks were made.
    """
    checks = 0
    for _ in range(steps):
        seqs, is_prefill = scheduler.schedule()
        logits = runner.run(seqs, is_prefill)
        assert logits.shape == (len(seqs), VOCAB)
        sampled = []
        for seq, row in zip(seqs, logits, strict=True):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                torch.testing.assert_close(
                    row, full_last_logits(model, seq.token_ids), atol=1e-4, rtol=1e-4
                )
                checks += 1
            sampled.append(int(row.argmax()))
        scheduler.postprocess(seqs, sampled)
    return checks


def prompt(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


# --- prepare_inputs metadata ----------------------------------------------------


def test_prepare_inputs_prefill_with_prefix_hit(model: Qwen3ForCausalLM) -> None:
    """The example from utils/context.py: A fresh 5-token prompt, B with 4 cached tokens."""
    runner = make_runner(model)
    a = Sequence(list(range(100, 105)))
    a.block_table, a.num_scheduled_tokens = [3, 8], 5
    b = Sequence(list(range(200, 206)))
    b.block_table, b.num_cached_tokens, b.num_scheduled_tokens = [5, 9], 4, 2

    input_ids, positions, ctx = runner.prepare_inputs([a, b], is_prefill=True)

    assert input_ids.tolist() == [100, 101, 102, 103, 104, 204, 205]
    assert positions.tolist() == [0, 1, 2, 3, 4, 4, 5]
    assert ctx.slot_mapping.tolist() == [12, 13, 14, 15, 32, 36, 37]
    assert ctx.cu_seqlens_q.tolist() == [0, 5, 7]
    assert ctx.context_lens == [5, 6]
    assert ctx.block_tables == [[3, 8], [5, 9]]
    assert ctx.is_prefill and ctx.kv_cache is runner.kv_cache


def test_prepare_inputs_decode(model: Qwen3ForCausalLM) -> None:
    runner = make_runner(model)
    a = Sequence(list(range(100, 106)))
    a.block_table, a.num_cached_tokens, a.num_scheduled_tokens = [3, 8], 5, 1
    b = Sequence(list(range(200, 207)))
    b.block_table, b.num_cached_tokens, b.num_scheduled_tokens = [5, 9], 6, 1

    input_ids, positions, ctx = runner.prepare_inputs([a, b], is_prefill=False)

    assert input_ids.tolist() == [105, 206]
    assert positions.tolist() == [5, 6]
    assert ctx.slot_mapping.tolist() == [33, 38]
    assert ctx.cu_seqlens_q.tolist() == [0, 1, 2]
    assert ctx.context_lens == [6, 7]
    assert not ctx.is_prefill


# --- run --------------------------------------------------------------------------


def test_run_resets_context_even_on_error(model: Qwen3ForCausalLM) -> None:
    class Boom(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    runner = make_runner(model)
    runner.model = Boom()
    seq = Sequence([1, 2, 3])
    seq.block_table, seq.num_scheduled_tokens = [0], 3
    with pytest.raises(RuntimeError, match="boom"):
        runner.run([seq], is_prefill=True)
    assert get_context().kv_cache is None


def test_two_sequences_prefill_then_decode_match_full_recompute(
    model: Qwen3ForCausalLM,
) -> None:
    scheduler, runner = make_scheduler(), make_runner(model)
    scheduler.add(Sequence(prompt(9, seed=1)))
    scheduler.add(Sequence(prompt(6, seed=2)))
    # step 1: one prefill batch with both seqs; steps 2-4: three decode steps each.
    assert drive(scheduler, runner, model, steps=4) == 8


def test_chunked_prefill_matches_full_recompute(model: Qwen3ForCausalLM) -> None:
    scheduler, runner = make_scheduler(budget=4), make_runner(model)
    seq = Sequence(prompt(10, seed=3))
    scheduler.add(seq)
    # 10-token prompt in chunks of 4, 4, 2; then three decode steps.
    assert drive(scheduler, runner, model, steps=6) == 4
    assert seq.num_tokens == 10 + 4


def test_prefix_cache_hit_matches_full_recompute(model: Qwen3ForCausalLM) -> None:
    scheduler, runner = make_scheduler(), make_runner(model)
    first = Sequence(prompt(9, seed=4))
    scheduler.add(first)
    drive(scheduler, runner, model, steps=1)
    scheduler.finish(first)

    # Shares the first two full blocks (8 tokens) with ``first``.
    second = Sequence(first.token_ids[:8] + [5, 6, 7])
    scheduler.add(second)
    seqs, is_prefill = scheduler.schedule()
    assert seqs == [second] and is_prefill
    assert second.num_cached_tokens == 8  # prefix hit: only 3 tokens are computed
    assert second.num_scheduled_tokens == 3

    logits = runner.run(seqs, is_prefill)
    torch.testing.assert_close(
        logits[0], full_last_logits(model, second.token_ids), atol=1e-4, rtol=1e-4
    )
    scheduler.postprocess(seqs, [int(logits[0].argmax())])
    assert drive(scheduler, runner, model, steps=2) == 2
