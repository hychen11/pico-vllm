"""Size the paged KV cache from free GPU memory (M6 step 5).

Same budget as nano-vllm::

    kv_bytes = total * gpu_memory_utilization - used - activation_peak

- ``total`` / ``used``: ``torch.cuda.mem_get_info`` after the warm-up, so ``used``
  covers this process (CUDA context, weights) and every other process on the GPU;
- ``activation_peak``: the most memory one step allocates on top of that, measured by
  ``profile_activation_bytes`` running the two largest steps the scheduler can
  produce: a prefill that fills the token budget, and a decode that fills
  ``max_num_seqs``.

The warm-up writes its K/V into a temporary cache. That cache is allocated before
the peak counter is reset and freed afterwards, so it is not counted twice.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.sampler import sample
from pico_vllm.sequence import Sequence
from pico_vllm.worker.model_runner import ModelRunner


def warmup_shape(
        max_num_batched_tokens: int, max_model_len: int, max_num_seqs: int
) -> tuple[int, int]:
    """``(num_seqs, seq_len)`` of the warm-up prefill: as many ``max_model_len`` prompts
    as one step's token budget holds (at least one, at most ``max_num_seqs``)."""
    seq_len = min(max_model_len, max_num_batched_tokens)
    num_seqs = min(max(max_num_batched_tokens // seq_len, 1), max_num_seqs)
    return num_seqs, seq_len


@torch.inference_mode()
def profile_activation_bytes(
        model: Any,
        hf_config: Any,
        *,
        block_size: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        max_num_seqs: int,
        use_flash: bool,
) -> int:
    """Peak bytes allocated by one worst-case step beyond what was allocated before it.

    ``model`` must already be on its CUDA device in its final dtype.
    """
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_seqs, seq_len = warmup_shape(max_num_batched_tokens, max_model_len, max_num_seqs)
    blocks_per_seq = math.ceil(seq_len / block_size)

    cache = PagedKVCache(
        hf_config.num_hidden_layers, num_seqs * blocks_per_seq, block_size,
        hf_config.num_key_value_heads, head_dim, dtype=dtype, device=device,
    )
    runner = ModelRunner(model, cache, block_size, use_flash=use_flash)
    seqs = []
    for i in range(num_seqs):
        seq = Sequence([0] * seq_len)
        seq.block_table = list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq))
        seq.num_scheduled_tokens = seq_len
        seqs.append(seq)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    # 1. Largest prefill: the whole token budget, e.g. 4 prompts x 4096 tokens.
    logits = runner.run(seqs, is_prefill=True)
    sample(logits, torch.zeros(num_seqs, device=device))
    del logits

    # 2. Largest decode: max_num_seqs sequences, one token each, every one with a
    # full-length context. They reuse the KV the prefill just wrote: decode seq i
    # points at prefill seq i % num_seqs's blocks (all prompts are identical, so
    # this is the sharing a prefix-cache hit would give). Each recomputes its last
    # token and writes it back to the same slot; the values don't matter here.
    decode_seqs = []
    for i in range(max_num_seqs):
        seq = Sequence([0] * seq_len)
        seq.block_table = seqs[i % num_seqs].block_table
        seq.num_cached_tokens = seq_len - 1
        seq.num_scheduled_tokens = 1
        decode_seqs.append(seq)
    logits = runner.run(decode_seqs, is_prefill=False)
    sample(logits, torch.zeros(max_num_seqs, device=device))

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    del logits, runner, cache
    torch.cuda.empty_cache()
    return peak


def num_blocks_from_memory(
        total_bytes: int,
        used_bytes: int,
        activation_bytes: int,
        block_bytes: int,
        gpu_memory_utilization: float,
) -> int:
    """KV blocks that fit in ``total * utilization - used - activation`` (never negative)."""
    available = int(total_bytes * gpu_memory_utilization) - used_bytes - activation_bytes
    return max(available, 0) // block_bytes


def num_blocks_from_gpu_memory(
        device: torch.device,
        activation_bytes: int,
        block_bytes: int,
        gpu_memory_utilization: float,
) -> int:
    """``num_blocks_from_memory`` with the device's current ``mem_get_info``."""
    free, total = torch.cuda.mem_get_info(device)
    return num_blocks_from_memory(
        total, total - free, activation_bytes, block_bytes, gpu_memory_utilization
    )
