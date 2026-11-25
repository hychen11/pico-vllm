"""Tensor-parallel layers (M8 steps 2-3) on 2 CPU ranks over gloo.

Every rank builds the same full weights and input from one seed, loads the full
weights through ``weight_loader`` (the layer keeps its own slice), and compares
against the unsplit math. fp32, so only summation-order rounding differs.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from tp_spawn import run_ranks

from pico_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from pico_vllm.layers.linear import (
    ColumnParallelLinear,
    MergedLinear,
    QKVLinear,
    RowParallelLinear,
)

TP = 2
H, D, NH, NKV = 32, 4, 4, 2  # hidden, head_dim, q heads, kv heads
INTER = 24
VOCAB = 10  # ids 0-4 live on rank 0, 5-9 on rank 1


def _load(layer: nn.Linear, name: str, full: torch.Tensor, shard_id: object = None) -> None:
    param = getattr(layer, name)
    if shard_id is None:
        param.weight_loader(param, full)
    else:
        param.weight_loader(param, full, shard_id)


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def check_column(rank: int) -> None:
    w, b = torch.randn(INTER, H), torch.randn(INTER)
    x = torch.randn(5, H)
    layer = ColumnParallelLinear(H, INTER, bias=True)
    assert layer.weight.shape == (INTER // TP, H)
    _load(layer, "weight", w)
    _load(layer, "bias", b)
    _close(layer(x), (x @ w.T + b).chunk(TP, dim=-1)[rank])


def check_column_then_row(rank: int) -> None:
    """The Megatron pair: column output feeds row input directly, one all_reduce."""
    w1, w2 = torch.randn(INTER, H), torch.randn(H, INTER)
    x = torch.randn(5, H)
    col, row = ColumnParallelLinear(H, INTER), RowParallelLinear(INTER, H)
    _load(col, "weight", w1)
    _load(row, "weight", w2)
    _close(row(col(x)), x @ w1.T @ w2.T)


def check_qkv(rank: int) -> None:
    wq, wk, wv = torch.randn(NH * D, H), torch.randn(NKV * D, H), torch.randn(NKV * D, H)
    bq, bk, bv = torch.randn(NH * D), torch.randn(NKV * D), torch.randn(NKV * D)
    x = torch.randn(5, H)
    layer = QKVLinear(H, D, NH, NKV, bias=True)
    q_local, kv_local = NH * D // TP, NKV * D // TP
    assert layer.weight.shape == (q_local + 2 * kv_local, H)
    for sid, w, b in (("q", wq, bq), ("k", wk, bk), ("v", wv, bv)):
        _load(layer, "weight", w, sid)
        _load(layer, "bias", b, sid)
    q, k, v = layer(x).split([q_local, kv_local, kv_local], dim=-1)
    # This rank's heads of each shard: q heads [rank*NH/TP, ...), kv heads likewise.
    for out, w, b in ((q, wq, bq), (k, wk, bk), (v, wv, bv)):
        _close(out, (x @ w.T + b).chunk(TP, dim=-1)[rank])


def check_merged(rank: int) -> None:
    gate, up = torch.randn(INTER, H), torch.randn(INTER, H)
    x = torch.randn(5, H)
    layer = MergedLinear(H, [INTER, INTER])
    _load(layer, "weight", gate, 0)
    _load(layer, "weight", up, 1)
    g, u = layer(x).split([INTER // TP] * 2, dim=-1)
    _close(g, (x @ gate.T).chunk(TP, dim=-1)[rank])
    _close(u, (x @ up.T).chunk(TP, dim=-1)[rank])


def check_row(rank: int) -> None:
    w, b = torch.randn(H, INTER), torch.randn(H)
    x = torch.randn(5, INTER)
    layer = RowParallelLinear(INTER, H, bias=True)
    assert layer.weight.shape == (H, INTER // TP)
    _load(layer, "weight", w)
    _load(layer, "bias", b)
    # Input = this rank's slice, as the column-parallel layer before it produces.
    _close(layer(x.chunk(TP, dim=-1)[rank]), x @ w.T + b)  # bias added exactly once


def check_mlp_pair(rank: int) -> None:
    """gate_up (column) -> SiLU*mul (local) -> down (row): no talk until the end."""
    gate, up, down = torch.randn(INTER, H), torch.randn(INTER, H), torch.randn(H, INTER)
    x = torch.randn(5, H)
    gate_up, down_proj = MergedLinear(H, [INTER, INTER]), RowParallelLinear(INTER, H)
    _load(gate_up, "weight", gate, 0)
    _load(gate_up, "weight", up, 1)
    _load(down_proj, "weight", down)
    g, u = gate_up(x).chunk(2, dim=-1)
    out = down_proj(torch.nn.functional.silu(g) * u)
    _close(out, (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T)


def check_embedding(rank: int) -> None:
    table = torch.randn(VOCAB, H)
    ids = torch.tensor([0, 9, 4, 5, 5, 2])  # both halves, boundary ids, a repeat
    emb = VocabParallelEmbedding(VOCAB, H)
    assert emb.weight.shape == (VOCAB // TP, H)
    _load(emb, "weight", table)
    # Every rank gets every row: its own lookups plus the others' via all_reduce.
    _close(emb(ids), table[ids])


def check_lm_head(rank: int) -> None:
    table = torch.randn(VOCAB, H)
    x = torch.randn(3, H)
    head = ParallelLMHead(VOCAB, H)
    _load(head, "weight", table)
    logits = head(x)
    if rank == 0:
        _close(logits, x @ table.T)  # all V/tp slices, in rank order
    else:
        assert logits is None


def check_tied_head(rank: int) -> None:
    """The tie shares one split parameter: loading lm_head fills the embedding."""
    table = torch.randn(VOCAB, H)
    emb, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    head.weight = emb.weight
    _load(head, "weight", table)
    ids = torch.arange(VOCAB)
    hidden = emb(ids)
    _close(hidden, table)
    logits = head(hidden)
    if rank == 0:
        _close(logits, table @ table.T)


CHECKS = [check_column, check_column_then_row, check_qkv, check_merged, check_row,
          check_mlp_pair, check_embedding, check_lm_head, check_tied_head]


@pytest.mark.parametrize("check", CHECKS, ids=lambda f: f.__name__)
def test_tp2_matches_unsplit(check) -> None:  # noqa: ANN001
    run_ranks(check)


def test_single_rank_is_plain() -> None:
    """world_size=1 (the session group): full shapes, loaders copy unchanged."""
    torch.manual_seed(0)
    w, x = torch.randn(H, INTER), torch.randn(5, INTER)
    row = RowParallelLinear(INTER, H)
    _load(row, "weight", w)
    _close(row(x), x @ w.T)
    qkv = QKVLinear(H, D, NH, NKV, bias=False)
    assert qkv.weight.shape == ((NH + 2 * NKV) * D, H)
    table, ids = torch.randn(VOCAB, H), torch.tensor([0, 9, 3])
    emb, head = VocabParallelEmbedding(VOCAB, H), ParallelLMHead(VOCAB, H)
    _load(emb, "weight", table)
    _load(head, "weight", table)
    _close(emb(ids), table[ids])
    _close(head(table[ids]), table[ids] @ table.T)


def test_divide_rejects_uneven_split() -> None:
    from pico_vllm.layers.linear import divide

    assert divide(3072, 2) == 1536
    with pytest.raises(AssertionError, match="not divisible"):
        divide(10, 4)
