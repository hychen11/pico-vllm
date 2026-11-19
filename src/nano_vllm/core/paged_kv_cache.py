"""Physical storage for the paged KV cache.

``BlockManager`` (M2) only hands out block *ids*; this module owns the tensors
those ids index into. A token's KV lives at a flat *slot*::

    slot = block_table[pos // block_size] * block_size + pos % block_size

Copy-on-write is not needed: the prefix cache only shares *full* blocks, and a
full block is never written again, so a write never lands in a shared block.
"""

from __future__ import annotations

import torch


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
    ) -> None:
        """Allocate ``self.k_cache`` and ``self.v_cache`` (zeros) with the shape above."""
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def write(
            self, layer: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store this step's K/V of one layer.

        k, v: ``[num_tokens, num_kv_heads, head_dim]``; slot_mapping: ``[num_tokens]``.
        Token ``i`` goes to flat slot ``slot_mapping[i]``.
        """
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
