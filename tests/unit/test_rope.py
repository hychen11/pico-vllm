"""Tests for the M3 rotary position embedding (neox / rotate-half style)."""

import math

import pytest
import torch

from pico_vllm.layers.rotary_embedding import RotaryEmbedding


def test_position_zero_is_identity() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, max_position=16, base=10000.0)
    q, k = torch.randn(1, 4, 8), torch.randn(1, 2, 8)
    q_out, k_out = rope(torch.tensor([0]), q, k)
    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)


def test_rotation_preserves_per_pair_norm() -> None:
    torch.manual_seed(0)
    d = 8
    rope = RotaryEmbedding(head_dim=d, max_position=64, base=10000.0)
    q, k = torch.randn(5, 3, d), torch.randn(5, 1, d)
    q_out, _ = rope(torch.tensor([0, 1, 7, 20, 63]), q, k)

    def pair_norm(x: torch.Tensor) -> torch.Tensor:
        return x[..., : d // 2] ** 2 + x[..., d // 2 :] ** 2

    torch.testing.assert_close(pair_norm(q_out), pair_norm(q))


def test_hand_computed_value_for_head_dim_four() -> None:
    # head_dim=4, base=100: inv_freq = [1, 100^-0.5] = [1, 0.1]; pairs are (x0, x2), (x1, x3).
    rope = RotaryEmbedding(head_dim=4, max_position=8, base=100.0)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    out, _ = rope(torch.tensor([1]), x, x.clone())
    a, b = 1.0, 0.1
    expected = torch.tensor(
        [[[
            1.0 * math.cos(a) - 3.0 * math.sin(a),
            2.0 * math.cos(b) - 4.0 * math.sin(b),
            3.0 * math.cos(a) + 1.0 * math.sin(a),
            4.0 * math.cos(b) + 2.0 * math.sin(b),
        ]]]
    )
    torch.testing.assert_close(out, expected)


def test_dot_product_depends_only_on_relative_position() -> None:
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=16, max_position=128, base=10000.0)
    q, k = torch.randn(1, 1, 16), torch.randn(1, 1, 16)

    def score(m: int, n: int) -> torch.Tensor:
        q_m, _ = rope(torch.tensor([m]), q, k)
        _, k_n = rope(torch.tensor([n]), q, k)
        return (q_m * k_n).sum()

    ref = score(5, 2)
    for m, n in [(3, 0), (10, 7), (100, 97)]:
        torch.testing.assert_close(score(m, n), ref, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(score(6, 2), ref, atol=1e-4)


def test_preserves_input_dtype() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position=8, base=10000.0)
    q = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    q_out, k_out = rope(torch.tensor([1, 2]), q, q)
    assert q_out.dtype == k_out.dtype == torch.bfloat16


def test_matches_transformers_qwen3_rope() -> None:
    pytest.importorskip("transformers")
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(0)
    head_dim, n, base = 16, 6, 1000000.0
    config = Qwen3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
        max_position_embeddings=64, rope_parameters={"rope_type": "default", "rope_theta": base},
    )
    positions = torch.tensor([0, 1, 2, 9, 30, 63])
    q, k = torch.randn(n, 4, head_dim), torch.randn(n, 2, head_dim)

    ours_q, ours_k = RotaryEmbedding(head_dim, 64, base)(positions, q, k)

    hf_rope = Qwen3RotaryEmbedding(config)
    cos, sin = hf_rope(q, positions.unsqueeze(0))  # [1, N, D]
    # HF layout is [B, heads, N, D].
    hf_q, hf_k = apply_rotary_pos_emb(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), cos, sin
    )
    torch.testing.assert_close(ours_q, hf_q[0].transpose(0, 1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ours_k, hf_k[0].transpose(0, 1), atol=1e-5, rtol=1e-5)
