"""Rotary position embedding (RoPE), neox/rotate-half style as used by Qwen3."""

import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs ``(x[i], x[i + D/2])`` by the per-position angle."""
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, base: float) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freqs = torch.outer(torch.arange(max_position, dtype=torch.float), inv_freq)
        # [max_position, head_dim]: first half cos, second half sin.
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [N]; q: [N, num_heads, D]; k: [N, num_kv_heads, D]."""
        cos, sin = self.cos_sin_cache[positions].unsqueeze(-2).chunk(2, dim=-1)
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
