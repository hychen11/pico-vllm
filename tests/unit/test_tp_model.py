"""The whole Qwen3 under TP=2 (M8 step 4) matches TP=1, on 2 CPU ranks over gloo.

Both TP sizes load the same HF checkpoint tensors and run the same steps:
1. a plain forward (no KV cache) of one prompt;
2. ``ModelRunner`` + paged KV cache (each rank holds only its KV heads) driven by
   a ``Scheduler``: a chunked prefill, a second prompt, then decode steps that
   cross block boundaries. Appended tokens are fixed, so every run sees the same
   batches.
TP=1 runs in this process (world_size=1 session group); TP=2 in spawned ranks.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.distributed as dist
from tp_spawn import run_ranks

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sequence import Sequence
from pico_vllm.utils.loader import load_weights
from pico_vllm.worker.model_runner import ModelRunner

transformers = pytest.importorskip("transformers")

BLOCK_SIZE = 4
STEPS = 8


def tiny_config(tie: bool) -> Any:
    return transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=128, tie_word_embeddings=tie,
    )


def drive(rank: int, config: Any, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Build the model at the current world size, load ``state``, run both checks."""
    model = Qwen3ForCausalLM(config).eval()
    load_weights(model, state.items())
    ids = torch.arange(23) * 7 % config.vocab_size
    with torch.inference_mode():
        hidden = model(ids, torch.arange(len(ids)))  # no context: plain causal attention
        logits = model.compute_logits(hidden[-3:])

    Sequence.block_size = BLOCK_SIZE  # a class attribute: spawned ranks start at 256
    num_blocks = 32
    kv_heads = config.num_key_value_heads // dist.get_world_size()
    cache = PagedKVCache(config.num_hidden_layers, num_blocks, BLOCK_SIZE, kv_heads,
                         config.head_dim)
    runner = ModelRunner(model, cache, BLOCK_SIZE, use_flash=False)
    # 16-token budget: the 21-token prompt is prefilled in two chunks.
    scheduler = Scheduler(4, 16, num_blocks, BLOCK_SIZE)
    for n in (21, 6):
        scheduler.add(Sequence((torch.arange(n) * 5 % config.vocab_size).tolist()))
    steps = []
    for step in range(STEPS):
        seqs, is_prefill = scheduler.schedule()
        out = runner.run(seqs, is_prefill)
        steps.append((is_prefill, [s.num_scheduled_tokens for s in seqs], out))
        scheduler.postprocess(seqs, [(step * 5 + 3 * i) % config.vocab_size
                                     for i in range(len(seqs))])
    return {"hidden": hidden, "logits": logits, "steps": steps,
            "kv_cache_shape": tuple(cache.k_cache.shape)}


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
def test_tp2_model_matches_tp1(tie: bool) -> None:
    torch.manual_seed(0)
    config = tiny_config(tie)
    state = {k: v.clone() for k, v in
             transformers.Qwen3ForCausalLM(config).float().state_dict().items()}

    ref = drive(0, config, state)  # TP=1, this process
    ranks = run_ranks(drive, config, state)

    full_kv = ref["kv_cache_shape"]
    for rank, got in enumerate(ranks):
        # Each rank caches half the KV heads; everything else in the shape is equal.
        assert got["kv_cache_shape"][-2] * 2 == full_kv[-2]
        # Row-parallel layers all_reduce, so every rank ends with the full hidden.
        _close(got["hidden"], ref["hidden"])
        if rank == 0:
            _close(got["logits"], ref["logits"])
        else:
            assert got["logits"] is None  # the LM head gathers to rank 0 only

        assert [s[:2] for s in got["steps"]] == [s[:2] for s in ref["steps"]]
        for (_, _, out), (_, _, ref_out) in zip(got["steps"], ref["steps"], strict=True):
            if rank == 0:
                _close(out, ref_out)
            else:
                assert out is None

    # The run covered a chunked prefill and several decode steps.
    kinds = [(p, n) for p, n, _ in ref["steps"]]
    assert kinds[0] == (True, [16]) and kinds[1][0] is True
    assert sum(not p for p, _ in kinds) >= 4
