"""Activation functions."""

import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """SwiGLU gate: split the last dim into (gate, up) and return ``silu(gate) * up``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
