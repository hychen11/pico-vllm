"""Paged attention, eager PyTorch reference (no FlashAttention until M6).

TODO(M4): make ``Qwen3Attention`` (models/qwen3.py) call this module instead of
its single-sequence ``_attention``; it needs a ``layer_idx`` to address the cache.
"""

from __future__ import annotations

import torch
from torch import nn

from nano_vllm.utils.context import get_context


class Attention(nn.Module):
    def __init__(
            self, num_heads: int, num_kv_heads: int, head_dim: int, scale: float, layer_idx: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.layer_idx = layer_idx

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: ``[num_tokens, num_heads, D]``; k, v: ``[num_tokens, num_kv_heads, D]``.

        With ``get_context()``:
        1. ``kv_cache.write(layer_idx, k, v, slot_mapping)``;
        2. for each sequence i: take its query rows (``cu_seqlens_q``), ``kv_cache.read``
           its full context, and run ``causal_attention``.
        Returns ``[num_tokens, num_heads, D]``.
        """
        context = get_context()
        kv_cache = context.kv_cache
        if kv_cache is None:
            # No step context (e.g. M3 tests): one sequence, no cache, plain causal attention.
            return causal_attention(q, k, v, self.scale)
        # Write first: this step's queries must also see this step's own K/V.
        kv_cache.write(self.layer_idx, k, v, context.slot_mapping)
        cu = context.cu_seqlens_q.tolist()
        outputs = []
        for i in range(len(context.block_tables)):
            q_i = q[cu[i]: cu[i + 1]]
            k_i, v_i = kv_cache.read(
                self.layer_idx, context.block_tables[i], context.context_lens[i]
            )
            outputs.append(causal_attention(q_i, k_i, v_i, self.scale))
        return torch.cat(outputs, dim=0)


def causal_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Attention for one sequence whose queries are the *last* ``len(q)`` of ``len(k)`` tokens.

    q: ``[q_len, num_heads, D]``; k, v: ``[kv_len, num_kv_heads, D]`` with ``q_len <= kv_len``.
    Query j sits at position ``kv_len - q_len + j`` and may attend to keys ``0..that``.
    (Plain ``is_causal=True`` is only right when ``q_len == kv_len``.)
    GQA: repeat each KV head ``num_heads // num_kv_heads`` times.
    Returns ``[q_len, num_heads, D]``.
    """
    n = q.shape[1] // k.shape[1]
    q_len = q.shape[0]
    kv_len = k.shape[0]
    # because GQA, need repeat kv header
    k = k.repeat_interleave(n, dim=1)
    v = v.repeat_interleave(n, dim=1)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    scores = q @ k.transpose(-1, -2) * scale

    q_pos = torch.arange(q_len, device=q.device) + (kv_len - q_len)
    k_pos = torch.arange(kv_len, device=q.device)
    mask = k_pos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    return out.transpose(0, 1)
