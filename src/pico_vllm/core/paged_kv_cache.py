"""Physical storage for the paged KV cache.

``BlockManager`` (M2) only hands out block *ids*; this module owns the tensors
those ids index into. A token's KV lives at a flat *slot*::

    slot = block_table[pos // block_size] * block_size + pos % block_size

Copy-on-write is not needed: the prefix cache only shares *full* blocks, and a
full block is never written again, so a write never lands in a shared block.
"""

from __future__ import annotations

import torch

from pico_vllm.kernels import kv_store


class PagedKVCache:
    """K/V tensors shaped ``[num_layers, num_blocks, block_size, num_kv_heads, head_dim]``."""

    def __init__(
            self,
            num_layers: int,
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_dim: int,
            dtype: torch.dtype = torch.float32,
            device: torch.device | str = "cpu",
            use_triton: bool | None = None,
    ) -> None:
        """Allocate ``self.k_cache`` and ``self.v_cache`` (zeros) with the shape above.

        ``use_triton``: write through the Triton kernel (M6). ``None`` = whenever the
        cache is on CUDA and Triton is importable; ``False`` forces the eager path.
        """
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        available = self.k_cache.is_cuda and kv_store.HAS_TRITON
        if use_triton and not available:
            raise ValueError("use_triton=True needs a CUDA cache and an importable triton")
        self.use_triton = available if use_triton is None else use_triton

    def write(
            self, layer: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store this step's K/V of one layer.

        k, v: ``[num_tokens, num_kv_heads, head_dim]``; slot_mapping: ``[num_tokens]``.
        Token ``i`` goes to flat slot ``slot_mapping[i]``.
        """
        if self.use_triton:
            kv_store.store_kvcache(
                k, v, self.k_cache[layer], self.v_cache[layer], slot_mapping
            )
            return
        num_tokens, num_kv_heads, head_dim = k.shape
        self.k_cache[layer].view(-1, num_kv_heads, head_dim)[slot_mapping] = k
        self.v_cache[layer].view(-1, num_kv_heads, head_dim)[slot_mapping] = v

    def read(
            self, layer: int, block_table: list[int], context_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sequence's K/V of one layer, contiguous in token order.

        Gathers the blocks in ``block_table`` and truncates to ``context_len`` tokens.
        Returns k, v: ``[context_len, num_kv_heads, head_dim]``.
        """
        blocks = torch.tensor(block_table, dtype=torch.long, device=self.k_cache.device)
        k = self.k_cache[layer, blocks].flatten(0, 1)[:context_len]
        v = self.v_cache[layer, blocks].flatten(0, 1)[:context_len]
        return k, v


def num_blocks_from_capacity(
        capacity_bytes: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
) -> int:
    """How many KV blocks (K and V, all layers) fit in ``capacity_bytes``.

    Dynamic sizing from free GPU memory is deferred to M6.
    """
    block_capacity = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
    return capacity_bytes // block_capacity
