"""Triton kernel that scatters one step's K/V rows into the paged KV cache (M6 step 1).

Same semantics as the eager ``PagedKVCache.write``: token ``i``'s ``[num_kv_heads,
head_dim]`` row goes to flat slot ``slot_mapping[i]`` of one layer's cache. One
program copies one token's K and V row, so a step is a single kernel launch instead
of two advanced-indexing ``index_put_`` calls.

Triton ships with Linux CUDA torch wheels but not with CPU-only torch (CI), so the
import is optional; ``HAS_TRITON`` tells callers whether ``store_kvcache`` can run.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the installed torch wheel
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _store_kvcache_kernel(
            k_ptr,
            k_row_stride,
            v_ptr,
            v_row_stride,
            k_cache_ptr,
            v_cache_ptr,
            slot_mapping_ptr,
            ROW: tl.constexpr,  # num_kv_heads * head_dim
            BLOCK: tl.constexpr,  # next power of two >= ROW (tl.arange needs one)
    ):
        i = tl.program_id(0).to(tl.int64)
        slot = tl.load(slot_mapping_ptr + i)
        offs = tl.arange(0, BLOCK)
        mask = offs < ROW
        k = tl.load(k_ptr + i * k_row_stride + offs, mask=mask)
        v = tl.load(v_ptr + i * v_row_stride + offs, mask=mask)
        tl.store(k_cache_ptr + slot * ROW + offs, k, mask=mask)
        tl.store(v_cache_ptr + slot * ROW + offs, v, mask=mask)


def _row_major(x: torch.Tensor) -> torch.Tensor:
    """Rows may be strided (``v`` is a slice of the fused qkv output), but each row's
    ``[num_kv_heads, head_dim]`` must be contiguous for the flat ``offs`` indexing."""
    _, num_kv_heads, head_dim = x.shape
    if x.stride(2) == 1 and x.stride(1) == head_dim:
        return x
    return x.contiguous()


def store_kvcache(
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
) -> None:
    """k, v: ``[num_tokens, num_kv_heads, head_dim]`` on CUDA;
    k_cache, v_cache: one layer, contiguous ``[num_blocks, block_size, num_kv_heads,
    head_dim]``; slot_mapping: ``[num_tokens]`` int64 flat slots."""
    num_tokens, num_kv_heads, head_dim = k.shape
    if num_tokens == 0:
        return
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k.dtype == k_cache.dtype and v.dtype == v_cache.dtype
    assert slot_mapping.numel() == num_tokens
    k, v = _row_major(k), _row_major(v)
    row = num_kv_heads * head_dim
    # Triton compiles for, and launches on, the *current* CUDA device, not the device of
    # its arguments. Make that the cache's device so e.g. a cache on cuda:1 works while
    # cuda:0 is current (otherwise: "no kernel image is available" or a wrong-GPU launch).
    with torch.cuda.device(k_cache.device):
        _store_kvcache_kernel[(num_tokens,)](
            k, k.stride(0), v, v.stride(0), k_cache, v_cache, slot_mapping,
            ROW=row, BLOCK=triton.next_power_of_2(row),
        )
