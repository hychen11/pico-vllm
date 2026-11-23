"""Turn a scheduled batch of sequences into model inputs and run one step.

Each step computes, for every scheduled seq, the tokens
``token_ids[num_cached_tokens : num_cached_tokens + num_scheduled_tokens]``.
Decode is just the case ``num_scheduled_tokens == 1``, so prefill (including
prefix-cache hits and chunked prefill) and decode share one input layout.
"""

from __future__ import annotations

import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sequence import Sequence
from pico_vllm.utils.context import Context, reset_context, set_context


def slot_mapping_for(block_table: list[int], block_size: int, start: int, end: int) -> list[int]:
    """Flat KV slots for token positions ``start..end-1`` of one sequence."""
    slot_mapping = []
    for pos in range(start, end):
        idx = pos // block_size
        slot_mapping.append(block_table[idx] * block_size + pos % block_size)
    return slot_mapping


class ModelRunner:
    def __init__(
            self, model: Qwen3ForCausalLM, kv_cache: PagedKVCache, block_size: int,
            use_flash: bool = False,
    ) -> None:
        self.model = model
        self.use_flash = use_flash
        self.kv_cache = kv_cache
        self.block_size = block_size

    def prepare_inputs(
            self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Context]:
        """Build flat ``input_ids`` and ``positions`` (``[num_tokens]``) and the step Context."""
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        cu_seqlens_q: list[int] = [0]
        cu_seqlens_k: list[int] = [0]
        max_seqlen_q = 0
        context_lens: list[int] = []
        block_tables: list[list[int]] = []
        # Prefill and decode share this loop; decode is num_scheduled_tokens == 1.
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(slot_mapping_for(seq.block_table, self.block_size, start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + (end - start))
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, end - start)
            context_lens.append(end)  # cached + this step: all KV attention must read
            block_tables.append(seq.block_table)

        device = self.kv_cache.k_cache.device

        def as_tensor(xs: list, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.tensor(xs, dtype=dtype, device=device)

        # FA2 wants a rectangular int32 block table; pad rows with -1 (never read).
        max_num_blocks = max((len(bt) for bt in block_tables), default=0)
        block_table = [bt + [-1] * (max_num_blocks - len(bt)) for bt in block_tables]

        context = Context(
            is_prefill=is_prefill,
            cu_seqlens_q=as_tensor(cu_seqlens_q, torch.int32),
            slot_mapping=as_tensor(slot_mapping),
            context_lens=context_lens,
            block_tables=block_tables,
            kv_cache=self.kv_cache,
            cu_seqlens_k=as_tensor(cu_seqlens_k, torch.int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max(context_lens, default=0),
            block_table=as_tensor(block_table, torch.int32),
            use_flash=self.use_flash,
        )
        return as_tensor(input_ids), as_tensor(positions), context

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        """One model step. Returns logits ``[num_seqs, vocab]`` at each seq's last query token.

        ``prepare_inputs`` → ``set_context`` → model forward → ``reset_context`` (always,
        even on error) → ``compute_logits`` on the last row of each sequence.
        """
        input_ids, positions, context = self.prepare_inputs(seqs, is_prefill)
        set_context(context)
        try:
            hidden_states = self.model(input_ids, positions)
        finally:
            reset_context()
        last = context.cu_seqlens_q[1:] - 1
        logits = self.model.compute_logits(hidden_states[last])
        return logits
