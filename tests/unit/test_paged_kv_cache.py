"""Tests for PagedKVCache, slot mapping, and KV-cache sizing (M4)."""

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache, num_blocks_from_capacity
from pico_vllm.worker.model_runner import slot_mapping_for

BS, LAYERS, KV_HEADS, HEAD_DIM = 4, 2, 2, 3


def make_cache(num_blocks: int = 16) -> PagedKVCache:
    return PagedKVCache(LAYERS, num_blocks, BS, KV_HEADS, HEAD_DIM)


def rand_kv(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n, KV_HEADS, HEAD_DIM), torch.randn(n, KV_HEADS, HEAD_DIM)


def slots(block_table: list[int], start: int, end: int) -> torch.Tensor:
    return torch.tensor(slot_mapping_for(block_table, BS, start, end))


# --- slot mapping ---------------------------------------------------------------


def test_slot_mapping_crosses_physical_blocks() -> None:
    # pos 0..3 live in block 12 (slots 48..51), pos 4..5 in block 7 (slots 28, 29).
    assert slot_mapping_for([12, 7], BS, 0, 6) == [48, 49, 50, 51, 28, 29]
    assert slot_mapping_for([12, 7], BS, 5, 6) == [29]


def test_slot_mapping_starts_after_cached_prefix() -> None:
    assert slot_mapping_for([12, 7], BS, 4, 6) == [28, 29]


def test_slot_mapping_empty_range() -> None:
    assert slot_mapping_for([12, 7], BS, 3, 3) == []


# --- write / read ---------------------------------------------------------------


def test_cache_shape_and_zero_init() -> None:
    cache = make_cache(num_blocks=5)
    assert cache.k_cache.shape == (LAYERS, 5, BS, KV_HEADS, HEAD_DIM)
    assert cache.v_cache.shape == cache.k_cache.shape
    assert not cache.k_cache.any() and not cache.v_cache.any()


def test_write_then_read_returns_tokens_in_order() -> None:
    cache = make_cache()
    table = [12, 7]
    k, v = rand_kv(6)
    cache.write(0, k, v, slots(table, 0, 6))
    rk, rv = cache.read(0, table, 6)
    assert torch.equal(rk, k) and torch.equal(rv, v)


def test_incremental_writes_match_one_shot_write() -> None:
    """Prefill 5 tokens, then decode 1: same cache content as writing all 6 at once."""
    table = [12, 7]
    k, v = rand_kv(6)
    one_shot, stepwise = make_cache(), make_cache()
    one_shot.write(0, k, v, slots(table, 0, 6))
    stepwise.write(0, k[:5], v[:5], slots(table, 0, 5))
    stepwise.write(0, k[5:], v[5:], slots(table, 5, 6))
    assert torch.equal(stepwise.k_cache, one_shot.k_cache)
    assert torch.equal(stepwise.v_cache, one_shot.v_cache)


def test_read_truncates_to_context_len() -> None:
    cache = make_cache()
    table = [3, 9]
    k, v = rand_kv(8)
    cache.write(1, k, v, slots(table, 0, 8))
    rk, _ = cache.read(1, table, 5)
    assert torch.equal(rk, k[:5])


def test_layers_are_isolated() -> None:
    cache = make_cache()
    k, v = rand_kv(4)
    cache.write(1, k, v, slots([2], 0, 4))
    assert not cache.k_cache[0].any() and not cache.v_cache[0].any()


def test_sequences_are_isolated() -> None:
    cache = make_cache()
    ka, va = rand_kv(5)
    kb, vb = rand_kv(5)
    cache.write(0, ka, va, slots([1, 2], 0, 5))
    cache.write(0, kb, vb, slots([3, 4], 0, 5))
    assert torch.equal(cache.read(0, [1, 2], 5)[0], ka)
    assert torch.equal(cache.read(0, [3, 4], 5)[0], kb)


def test_shared_prefix_block_is_read_by_both_sequences() -> None:
    """Block 5 holds a shared full prefix; each seq has its own tail block."""
    cache = make_cache()
    prefix_k, prefix_v = rand_kv(4)
    cache.write(0, prefix_k, prefix_v, slots([5], 0, 4))
    ka, va = rand_kv(2)
    kb, vb = rand_kv(1)
    cache.write(0, ka, va, slots([5, 8], 4, 6))
    cache.write(0, kb, vb, slots([5, 9], 4, 5))

    rka, _ = cache.read(0, [5, 8], 6)
    rkb, _ = cache.read(0, [5, 9], 5)
    assert torch.equal(rka[:4], prefix_k) and torch.equal(rkb[:4], prefix_k)
    assert torch.equal(rka[4:], ka) and torch.equal(rkb[4:], kb)


# --- sizing ---------------------------------------------------------------------


def test_num_blocks_for_qwen3_0_6b() -> None:
    # 2 (K+V) * 28 layers * 256 tokens * 8 kv heads * 128 dim * 2 bytes = 28 MiB per block.
    assert num_blocks_from_capacity(4 * 2**30, 28, 256, 8, 128, torch.bfloat16) == 146


def test_num_blocks_rounds_down() -> None:
    per_block = 2 * LAYERS * BS * KV_HEADS * HEAD_DIM * 4  # float32
    assert num_blocks_from_capacity(per_block, LAYERS, BS, KV_HEADS, HEAD_DIM) == 1
    assert num_blocks_from_capacity(per_block - 1, LAYERS, BS, KV_HEADS, HEAD_DIM) == 0
    assert num_blocks_from_capacity(3 * per_block + 7, LAYERS, BS, KV_HEADS, HEAD_DIM) == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_num_blocks_fits_in_capacity(dtype: torch.dtype) -> None:
    capacity = 10_000
    n = num_blocks_from_capacity(capacity, LAYERS, BS, KV_HEADS, HEAD_DIM, dtype)
    cache = PagedKVCache(LAYERS, n, BS, KV_HEADS, HEAD_DIM, dtype=dtype)
    used = cache.k_cache.nbytes + cache.v_cache.nbytes
    assert used <= capacity
    one_more = PagedKVCache(LAYERS, n + 1, BS, KV_HEADS, HEAD_DIM, dtype=dtype)
    assert one_more.k_cache.nbytes + one_more.v_cache.nbytes > capacity
