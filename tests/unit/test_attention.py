"""Tests for the eager paged-attention reference (M4)."""

import pytest
import torch
import torch.nn.functional as F

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.layers.attention import Attention, causal_attention
from pico_vllm.utils.context import Context, reset_context, set_context
from pico_vllm.worker.model_runner import slot_mapping_for

HEADS, KV_HEADS, D, BS = 4, 2, 8, 4
SCALE = D**-0.5


def rand_qkv(q_len: int, kv_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(q_len, HEADS, D),
        torch.randn(kv_len, KV_HEADS, D),
        torch.randn(kv_len, KV_HEADS, D),
    )


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full causal attention (q_len == kv_len) via PyTorch, GQA by repeat_interleave."""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, scale=SCALE
    )
    return out.transpose(0, 1)


@pytest.fixture(autouse=True)
def _clean_context():
    reset_context()
    yield
    reset_context()


def test_causal_attention_matches_sdpa_when_no_cache() -> None:
    torch.manual_seed(0)
    q, k, v = rand_qkv(6, 6)
    out = causal_attention(q, k, v, SCALE)
    assert out.shape == (6, HEADS, D)
    torch.testing.assert_close(out, sdpa_reference(q, k, v))


@pytest.mark.parametrize("q_len", [1, 2, 5])
def test_causal_attention_suffix_queries_match_full(q_len: int) -> None:
    """Queries that are the last q_len of kv_len tokens (prefix hit / chunk / decode)."""
    torch.manual_seed(0)
    q, k, v = rand_qkv(7, 7)
    full = sdpa_reference(q, k, v)
    torch.testing.assert_close(causal_attention(q[-q_len:], k, v, SCALE), full[-q_len:])


def test_gqa_heads_share_kv_in_groups() -> None:
    """Q heads 2h and 2h+1 use KV head h: identical queries in a group give identical outputs."""
    torch.manual_seed(0)
    q, k, v = rand_qkv(3, 3)
    q[:, 1] = q[:, 0]
    q[:, 3] = q[:, 2]
    out = causal_attention(q, k, v, SCALE)
    torch.testing.assert_close(out[:, 0], out[:, 1])
    torch.testing.assert_close(out[:, 2], out[:, 3])
    assert not torch.allclose(out[:, 0], out[:, 2])


def test_attention_without_context_is_plain_causal() -> None:
    torch.manual_seed(0)
    q, k, v = rand_qkv(5, 5)
    attn = Attention(HEADS, KV_HEADS, D, SCALE, layer_idx=0)
    torch.testing.assert_close(attn(q, k, v), sdpa_reference(q, k, v))


def test_attention_with_context_batches_two_sequences() -> None:
    """Seq A: fresh 5-token prefill. Seq B: 4 tokens cached, 2 new. Each sees only itself."""
    torch.manual_seed(0)
    layer = 1
    cache = PagedKVCache(2, 16, BS, KV_HEADS, D)
    attn = Attention(HEADS, KV_HEADS, D, SCALE, layer_idx=layer)

    qa, ka, va = rand_qkv(5, 5)
    qb, kb, vb = rand_qkv(6, 6)
    table_a, table_b = [3, 8], [5, 9]
    # B's first 4 tokens were computed by an earlier step.
    cache.write(layer, kb[:4], vb[:4], torch.tensor(slot_mapping_for(table_b, BS, 0, 4)))

    slots = slot_mapping_for(table_a, BS, 0, 5) + slot_mapping_for(table_b, BS, 4, 6)
    set_context(
        Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 5, 7]),
            slot_mapping=torch.tensor(slots),
            context_lens=[5, 6],
            block_tables=[table_a, table_b],
            kv_cache=cache,
        )
    )
    out = attn(torch.cat([qa, qb[4:]]), torch.cat([ka, kb[4:]]), torch.cat([va, vb[4:]]))

    assert out.shape == (7, HEADS, D)
    torch.testing.assert_close(out[:5], sdpa_reference(qa, ka, va))
    torch.testing.assert_close(out[5:], sdpa_reference(qb, kb, vb)[4:])
    # This step's K/V landed in the cache, other layers untouched.
    torch.testing.assert_close(cache.read(layer, table_b, 6)[0], kb)
    assert not cache.k_cache[0].any()
