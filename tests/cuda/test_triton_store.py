"""Triton KV-store kernel matches the eager ``PagedKVCache.write`` exactly (M6 step 1)."""

from __future__ import annotations

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.kernels import kv_store
from pico_vllm.worker.model_runner import slot_mapping_for

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and kv_store.HAS_TRITON), reason="requires CUDA + triton"
    ),
]

LAYERS, NUM_BLOCKS, BS = 2, 16, 4


def pair(kv_heads: int, head_dim: int, dtype: torch.dtype) -> tuple[PagedKVCache, PagedKVCache]:
    def make(use_triton: bool) -> PagedKVCache:
        return PagedKVCache(LAYERS, NUM_BLOCKS, BS, kv_heads, head_dim, dtype=dtype,
                            device="cuda", use_triton=use_triton)

    return make(True), make(False)


def assert_same(fast: PagedKVCache, eager: PagedKVCache) -> None:
    # A pure copy: bit-exact, not just close.
    assert torch.equal(fast.k_cache, eager.k_cache)
    assert torch.equal(fast.v_cache, eager.v_cache)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("kv_heads", "head_dim"), [(2, 16), (8, 128), (2, 3)])
def test_matches_eager_across_blocks(kv_heads: int, head_dim: int, dtype: torch.dtype) -> None:
    # (2, 3): row of 6 elements, not a power of two -> exercises the mask.
    fast, eager = pair(kv_heads, head_dim, dtype)
    torch.manual_seed(0)
    # Two sequences in scattered physical blocks; seq b starts mid-block (cached prefix).
    slots = slot_mapping_for([12, 7, 3], BS, 0, 10) + slot_mapping_for([5, 9], BS, 2, 7)
    slot_mapping = torch.tensor(slots, device="cuda")
    for layer in range(LAYERS):
        k = torch.randn(len(slots), kv_heads, head_dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        fast.write(layer, k, v, slot_mapping)
        eager.write(layer, k, v, slot_mapping)
    assert_same(fast, eager)
    # Untouched slots stay zero: other blocks / layers are not clobbered.
    written = torch.zeros(NUM_BLOCKS * BS, dtype=torch.bool, device="cuda")
    written[slot_mapping] = True
    flat = fast.k_cache[0].view(NUM_BLOCKS * BS, -1)
    assert flat[~written].abs().sum() == 0


def test_strided_v_from_fused_qkv() -> None:
    """``v`` in Qwen3Attention is a slice of the fused qkv output: rows are strided."""
    kv_heads, head_dim, q_size = 2, 16, 64
    fast, eager = pair(kv_heads, head_dim, torch.bfloat16)
    n, kv_size = 5, kv_heads * head_dim
    qkv = torch.randn(n, q_size + 2 * kv_size, device="cuda", dtype=torch.bfloat16)
    _, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    k, v = k.view(n, kv_heads, head_dim), v.view(n, kv_heads, head_dim)
    assert not v.is_contiguous()
    slot_mapping = torch.tensor(slot_mapping_for([2, 11], BS, 1, 6), device="cuda")
    fast.write(1, k, v, slot_mapping)
    eager.write(1, k, v, slot_mapping)
    assert_same(fast, eager)


def test_decode_step_one_token_per_seq() -> None:
    fast, eager = pair(8, 128, torch.bfloat16)
    slot_mapping = torch.tensor([3, 17, 42, 63], device="cuda")
    k = torch.randn(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fast.write(0, k, v, slot_mapping)
    eager.write(0, k, v, slot_mapping)
    assert_same(fast, eager)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
def test_cache_on_non_current_device() -> None:
    """Triton launches on the *current* device; a cache on another GPU must still work."""
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    assert torch.cuda.current_device() != device.index

    def make(use_triton: bool) -> PagedKVCache:
        return PagedKVCache(LAYERS, NUM_BLOCKS, BS, 8, 128, dtype=torch.bfloat16,
                            device=device, use_triton=use_triton)

    fast, eager = make(True), make(False)
    slot_mapping = torch.tensor(slot_mapping_for([12, 7, 3], BS, 0, 10), device=device)
    k = torch.randn(10, 8, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fast.write(1, k, v, slot_mapping)
    eager.write(1, k, v, slot_mapping)
    assert_same(fast, eager)
    assert torch.cuda.current_device() != device.index  # current device is restored


def test_auto_selection() -> None:
    assert PagedKVCache(1, 2, BS, 1, 4, device="cuda").use_triton
    assert not PagedKVCache(1, 2, BS, 1, 4, device="cpu").use_triton
    with pytest.raises(ValueError):
        PagedKVCache(1, 2, BS, 1, 4, device="cpu", use_triton=True)
