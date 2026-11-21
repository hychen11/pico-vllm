"""Per-step attention metadata shared between ModelRunner and Attention.

ModelRunner calls ``set_context`` before the forward pass and ``reset_context``
after it; every ``Attention`` layer reads it with ``get_context``. A global avoids
threading these tensors through every layer's ``forward`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pico_vllm.core.paged_kv_cache import PagedKVCache


@dataclass
class Context:
    """Metadata for one batched forward pass over many sequences.

    A step is all-prefill or all-decode (see ``Scheduler.schedule``); both share one
    layout. Tokens of all sequences are flattened into one ``[num_tokens]`` batch.
    A new Context is built every step: only these small index tensors are created,
    and ``kv_cache`` is a reference to the one shared cache, never a copy.

    Example (block_size=4), a prefill step with two sequences:

    - A: new 5-token prompt, block_table=[3, 8], nothing cached -> computes a0..a4
    - B: 6-token prompt, block_table=[5, 9], first block hit the prefix cache
      (b0..b3 already in block 5) -> computes only b4, b5

    ::

        row i:          0   1   2   3   4 |  5   6
        token:         a0  a1  a2  a3  a4 | b4  b5
        positions:      0   1   2   3   4 |  4   5
        slot_mapping:  12  13  14  15  32 | 36  37    # block * 4 + pos % 4

        cu_seqlens_q = [0, 5, 7]          # A = rows 0:5, B = rows 5:7
        context_lens = [5, 6]             # KV to read: A a0..a4, B b0..b5
        block_tables = [[3, 8], [5, 9]]

    The next (decode) step computes one token per sequence (a5, b6)::

        positions = [5, 6], slot_mapping = [33, 38]
        cu_seqlens_q = [0, 1, 2], context_lens = [6, 7]
    """

    is_prefill: bool = False
    # [num_seqs + 1]: query tokens of seq i are flat rows cu_seqlens_q[i]:cu_seqlens_q[i+1].
    cu_seqlens_q: torch.Tensor | None = None
    # [num_tokens]: flat KV slot to write each query token's K/V into.
    slot_mapping: torch.Tensor | None = None
    # Per seq: total tokens with KV after this step (cached + scheduled).
    context_lens: list[int] = field(default_factory=list)
    # Per seq: physical block ids (Sequence.block_table).
    block_tables: list[list[int]] = field(default_factory=list)
    kv_cache: PagedKVCache | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(context: Context) -> None:
    global _CONTEXT
    _CONTEXT = context


def reset_context() -> None:
    """Restore an empty ``Context()``."""
    global _CONTEXT
    _CONTEXT = Context()
