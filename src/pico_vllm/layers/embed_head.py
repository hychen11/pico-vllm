"""Vocab-parallel embedding and LM head (M8 step 3), as in nano-vllm.

Both split the vocabulary: rank r holds rows ``[r * V/tp, (r+1) * V/tp)`` of the
``[V, hidden]`` table. A tied LM head shares the embedding's parameter, which is
split the same way, so the tie still holds per rank.

- Embedding: each token id lives on exactly one rank. The others look up a dummy
  row and zero it, so an ``all_reduce`` (sum) leaves every rank the true row.
- LM head: each rank scores its V/tp words; the slices are ``gather``-ed to rank 0
  and concatenated (only rank 0 samples). Other ranks return ``None``.

With one rank both are a plain embedding / ``x @ weight.T``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from pico_vllm.layers.linear import divide


class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        per_rank = divide(num_embeddings, self.tp_size)
        super().__init__(per_rank, embedding_dim)
        self.vocab_start = per_rank * self.tp_rank
        self.vocab_end = self.vocab_start + per_rank
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        loaded_weight = loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(x, self.weight)
        mask = (x >= self.vocab_start) & (x < self.vocab_end)  # ids this rank holds
        y = F.embedding(mask * (x - self.vocab_start), self.weight)  # others -> row 0
        y = mask.unsqueeze(-1) * y  # ... whose result is zeroed
        dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def reset_parameters(self) -> None:
        """``nn.Linear``'s init, as the LM head had before M8: the embedding's
        N(0, 1) would make random-weight logits ~30x larger (checkpoints overwrite it)."""
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:  # type: ignore[override]
        """``x``: ``[N, hidden]``. Rank 0: logits ``[N, V]``; other ranks: ``None``."""
        logits = F.linear(x, self.weight)  # [N, V/tp]: this rank's words
        if self.tp_size == 1:
            return logits
        parts = [torch.empty_like(logits) for _ in range(self.tp_size)] \
            if self.tp_rank == 0 else None
        dist.gather(logits, parts, dst=0)
        return torch.cat(parts, dim=-1) if parts is not None else None
