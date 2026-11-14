"""Linear layers with packed (merged) weights and checkpoint shard loaders.

Each packed layer attaches ``weight_loader(param, loaded_weight, shard_id)`` to its
parameters so the checkpoint loader can copy separate HF tensors into one slice.
"""

import torch
from torch import nn


class _PackedLinear(nn.Linear):
    """A linear layer whose output dim is the concatenation of named shards."""

    def __init__(self, input_size: int, shard_sizes: dict, bias: bool) -> None:
        super().__init__(input_size, sum(shard_sizes.values()), bias=bias)
        self.shard_offsets = {}
        offset = 0
        for shard_id, size in shard_sizes.items():
            self.shard_offsets[shard_id] = (offset, size)
            offset += size
        for param in self.parameters():
            param.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: object
    ) -> None:
        offset, size = self.shard_offsets[shard_id]
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
