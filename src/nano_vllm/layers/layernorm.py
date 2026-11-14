"""Root-mean-square layer normalization."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last dim, computed in float32."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(orig_dtype)
