"""Tensor-parallel linear layers and their checkpoint shard loaders (M8).

Megatron-style pairs: a column-parallel layer (``_PackedLinear``: qkv_proj,
gate_up_proj) splits the output dim, so each rank holds 1/tp of the heads or of
the MLP intermediate dim and needs no communication; the row-parallel layer after
it (``RowParallelLinear``: o_proj, down_proj) splits the input dim and sums the
ranks' partial outputs with one all_reduce.

Every layer attaches ``weight_loader`` to its parameters; the checkpoint loader
passes it the full HF tensor and the layer copies in this rank's slice. With one
rank (world_size=1) all of this reduces to plain ``nn.Linear``.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class ColumnParallelLinear(nn.Linear):
    """Splits the output dim: each rank holds rows ``[rank-th 1/tp]`` of the weight
    (and bias) and produces that 1/tp of the output. No communication: the next
    layer is row parallel and takes exactly this slice as its input."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, self.tp_size), bias=bias)
        for param in self.parameters():
            param.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        loaded_weight = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
        param.data.copy_(loaded_weight)
    

class _PackedLinear(ColumnParallelLinear):
    """A column-parallel layer whose output dim is the concatenation of named shards.

    Each rank holds 1/tp of every shard (not 1/tp of the concatenation, which would
    give rank 0 all of q and rank 1 only k and v). The checkpoint stores the shards
    as separate tensors, so ``weight_loader`` takes a ``shard_id`` too.
    """

    def __init__(self, input_size: int, shard_sizes: dict, bias: bool) -> None:
        super().__init__(input_size, sum(shard_sizes.values()), bias=bias)
        self.shard_offsets = {}  # shard id -> (offset, size) in the local weight
        offset = 0
        for shard_id, full_size in shard_sizes.items():
            size = divide(full_size, self.tp_size)
            self.shard_offsets[shard_id] = (offset, size)
            offset += size

    def weight_loader(  # type: ignore[override]
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: object
    ) -> None:
        offset, size = self.shard_offsets[shard_id]
        loaded_weight = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        target = param.data.narrow(0, offset, size)
        assert target.shape == loaded_weight.shape, (target.shape, loaded_weight.shape)
        target.copy_(loaded_weight)


class QKVLinear(_PackedLinear):
    """Fused Q/K/V projection; shard ids are ``"q"``, ``"k"``, ``"v"``."""

    def __init__(
        self, hidden_size: int, head_dim: int, num_heads: int, num_kv_heads: int, bias: bool
    ) -> None:
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        super().__init__(hidden_size, {"q": q_size, "k": kv_size, "v": kv_size}, bias)


class MergedLinear(_PackedLinear):
    """Several projections of the same input; shard ids are ``0, 1, ...``."""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False) -> None:
        super().__init__(input_size, dict(enumerate(output_sizes)), bias)


class RowParallelLinear(nn.Linear):
    """Splits the input dim: each rank holds ``weight[:, rank-th 1/tp]`` and gets the
    matching 1/tp of the input (the column-parallel layer's output), computes a
    partial sum, and ``all_reduce`` adds the ranks' parts. The bias is not split;
    only rank 0 adds it, so it is counted once in the sum."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        super().__init__(divide(input_size, self.tp_size), output_size, bias=bias)
        for param in self.parameters():
            param.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.dim() == 2:  # weight [out, in]: this rank's input columns
            loaded_weight = loaded_weight.chunk(self.tp_size, dim=1)[self.tp_rank]
        assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
