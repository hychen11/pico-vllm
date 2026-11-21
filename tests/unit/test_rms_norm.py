"""Tests for the M3 RMSNorm and SwiGLU activation layers."""

import math

import pytest
import torch
import torch.nn.functional as F

from pico_vllm.layers.activation import SiluAndMul
from pico_vllm.layers.layernorm import RMSNorm


def test_rms_norm_hand_computed_values() -> None:
    norm = RMSNorm(2, eps=0.0)
    x = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    # rms([3, 4]) = sqrt(12.5); rms([1, 1]) = 1.
    r = math.sqrt(12.5)
    expected = torch.tensor([[3.0 / r, 4.0 / r], [1.0, 1.0]])
    torch.testing.assert_close(norm(x), expected)


def test_rms_norm_weight_scales_each_channel() -> None:
    norm = RMSNorm(2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, -0.5]))
    out = norm(torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(out, torch.tensor([[2.0, -0.5]]))


def test_rms_norm_eps_is_added_inside_the_sqrt() -> None:
    norm = RMSNorm(2, eps=1.0)
    out = norm(torch.tensor([[1.0, 1.0]]))
    expected = 1.0 / math.sqrt(1.0 + 1.0)
    torch.testing.assert_close(out, torch.full((1, 2), expected))
    # eps keeps a zero input finite.
    assert torch.equal(RMSNorm(4, eps=1e-6)(torch.zeros(1, 4)), torch.zeros(1, 4))


def test_rms_norm_normalizes_last_dim_only() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8)
    out = RMSNorm(8, eps=0.0)(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.pow(2).mean(-1), torch.ones(3, 5))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rms_norm_preserves_low_precision_dtype(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    norm = RMSNorm(16).to(dtype)
    x = torch.randn(4, 16).to(dtype)
    out = norm(x)
    assert out.dtype == dtype
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_silu_and_mul_equals_silu_gate_times_up() -> None:
    torch.manual_seed(0)
    gate, up = torch.randn(3, 6), torch.randn(3, 6)
    out = SiluAndMul()(torch.cat([gate, up], dim=-1))
    assert out.shape == (3, 6)
    torch.testing.assert_close(out, F.silu(gate) * up)


def test_silu_and_mul_hand_computed_value() -> None:
    out = SiluAndMul()(torch.tensor([[0.0, 1.0, 5.0, 2.0]]))
    silu1 = 1.0 / (1.0 + math.exp(-1.0))
    torch.testing.assert_close(out, torch.tensor([[0.0, 2.0 * silu1]]))
