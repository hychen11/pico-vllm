"""The step loop that ties Scheduler, ModelRunner, and the sampler together (M5).

One ``step()``::

    scheduler.schedule() -> model_runner.run() -> sample() -> scheduler.postprocess()
    -> finish seqs that hit EOS or max_tokens (frees their blocks)

This is the ``drive()`` loop from ``tests/integration/test_runner_fake_model.py``
plus stopping rules.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

from pico_vllm.config import DEFAULT_CPU_KVCACHE_BYTES, Config
from pico_vllm.core.paged_kv_cache import PagedKVCache, kv_block_bytes
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sampler import sample
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence
from pico_vllm.utils.loader import init_on, load_model
from pico_vllm.worker.memory import num_blocks_from_gpu_memory, profile_activation_bytes
from pico_vllm.worker.model_runner import ModelRunner


def build_model(config: Config) -> Qwen3ForCausalLM:
    """Native Qwen3 with checkpoint weights, on ``config.device`` in ``config.dtype``."""
    with init_on(config.device, config.dtype):
        model = Qwen3ForCausalLM(config.hf_config)
    load_model(model, config.model)  # copy_ casts checkpoint dtype/device as needed
    return model.eval()


def num_kv_blocks(config: Config, model: Qwen3ForCausalLM) -> int:
    """KV blocks for this engine.

    - ``kvcache_capacity_bytes`` set: as many blocks as fit in it;
    - otherwise on CUDA: warm up, then fill ``gpu_memory_utilization`` of the GPU
      (``worker.memory``);
    - otherwise (CPU): ``DEFAULT_CPU_KVCACHE_BYTES``.
    Raises ``ValueError`` when not even one block fits.
    """
    hf = config.hf_config
    block_bytes = kv_block_bytes(
        hf.num_hidden_layers, config.kvcache_block_size, hf.num_key_value_heads, hf.head_dim,
        config.dtype,
    )
    capacity = config.kvcache_capacity_bytes
    if capacity is None and config.device.type == "cuda":
        activation = profile_activation_bytes(
            model, hf, block_size=config.kvcache_block_size,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=config.max_model_len, max_num_seqs=config.max_num_seqs,
            use_flash=bool(config.use_flash),
        )
        num_blocks = num_blocks_from_gpu_memory(
            config.device, activation, block_bytes, config.gpu_memory_utilization
        )
        free, total = torch.cuda.mem_get_info(config.device)
        source = (
            f"gpu_memory_utilization={config.gpu_memory_utilization} "
            f"({(total - free) / 2**30:.2f} of {total / 2**30:.2f} GiB already in use, "
            f"{activation / 2**30:.2f} GiB reserved for activations)"
        )
    else:
        capacity = DEFAULT_CPU_KVCACHE_BYTES if capacity is None else capacity
        num_blocks = capacity // block_bytes
        source = f"kvcache_capacity_bytes={capacity}"
    if num_blocks <= 0:
        raise ValueError(
            f"no KV-cache block ({block_bytes} bytes) fits with {source}; raise it, or "
            "lower max_num_batched_tokens / max_num_seqs / kvcache_block_size"
        )
    return num_blocks


class LLMEngine:
    def __init__(
            self,
            scheduler: Scheduler,
            model_runner: ModelRunner,
            eos_token_id: int | None,
            tokenizer: Any | None = None,
            max_model_len: int | None = None,
    ) -> None:
        """Wire already-built components together.

        Taking components (rather than a model path) lets integration tests build
        an engine around a tiny random Qwen3 with token-id prompts and no tokenizer.
        Use ``from_config`` for a real checkpoint.

        ``max_model_len``: ``add_request`` rejects requests whose prompt plus
        ``max_tokens`` exceeds it (``None``: no limit). CUDA Graph buffers are sized
        for it, so no sequence may grow past it.
        """
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len

    @classmethod
    def from_config(cls, config: Config) -> LLMEngine:
        """Build every component for the local checkpoint ``config.model``.

        - ``build_model(config)``: weights on ``config.device`` in ``config.dtype``;
        - ``PagedKVCache`` on the same device/dtype, sized by ``num_kv_blocks``;
        - ``Scheduler`` and ``PagedKVCache`` must share ``num_blocks`` and block size;
        - set ``Sequence.block_size = config.kvcache_block_size`` (class attribute!);
        - ``AutoTokenizer.from_pretrained(config.model)``; EOS id from the tokenizer.
        """
        if config.device.type == "cuda":
            # An earlier engine in this process may still be waiting for gc (its
            # modules form reference cycles). Release it before loading weights:
            # otherwise they land in its cached KV segment, which then can never be
            # returned, and the auto-sized cache shrinks with every new engine.
            gc.collect()
            torch.cuda.empty_cache()
        model = build_model(config)

        hf = config.hf_config
        dtype = config.dtype
        block_size = config.kvcache_block_size
        num_blocks = num_kv_blocks(config, model)

        kv_cache = PagedKVCache(
            hf.num_hidden_layers, num_blocks, block_size, hf.num_key_value_heads, hf.head_dim,
            dtype=dtype, device=config.device,
        )
        Sequence.block_size = block_size
        scheduler = Scheduler(
            config.max_num_seqs, config.max_num_batched_tokens, num_blocks, block_size
        )
        # Config.__post_init__ already resolved use_flash to a bool.
        runner = ModelRunner(model, kv_cache, block_size, use_flash=bool(config.use_flash))
        # After the KV cache (as nano-vllm): graphs record its addresses, and their
        # memory pool comes out of the headroom gpu_memory_utilization leaves.
        if config.use_flash and kv_cache.use_triton and not config.enforce_eager:
            runner.capture_cudagraphs(config.max_num_seqs, config.max_model_len)
        # Imported lazily: transformers is an optional dependency (``--extra model``).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model)
        eos_token_id = tokenizer.eos_token_id
        return cls(scheduler, runner, eos_token_id, tokenizer, config.max_model_len)

    def add_request(
            self, prompt: str | list[int], sampling_params: SamplingParams
    ) -> Sequence:
        """Queue one request. A ``str`` prompt is encoded with the tokenizer;
        a ``list[int]`` is used as token ids directly. Returns the new Sequence.

        Raises ``ValueError`` if prompt + ``max_tokens`` exceeds ``max_model_len``."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        total = len(prompt) + sampling_params.max_tokens
        if self.max_model_len is not None and total > self.max_model_len:
            raise ValueError(
                f"prompt ({len(prompt)} tokens) + max_tokens "
                f"({sampling_params.max_tokens}) = {total} exceeds max_model_len "
                f"({self.max_model_len})"
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self) -> list[Sequence]:
        """Run one scheduler + model step; return the seqs that finished in it.

        After ``postprocess``, a seq is done when its newly appended token is EOS
        (unless ``ignore_eos``) or ``num_completion_tokens >= max_tokens``; call
        ``scheduler.finish(seq)`` so its blocks are freed.
        Only check seqs that actually appended a token this step: a non-final
        chunked-prefill step appends nothing.
        """
        seqs, is_prefill = self.scheduler.schedule()
        logits = self.model_runner.run(seqs, is_prefill)
        temperatures = torch.tensor(
            [seq.sampling_params.temperature for seq in seqs], device=logits.device
        )
        token_ids = sample(logits, temperatures)
        num_tokens_before = [seq.num_tokens for seq in seqs]
        self.scheduler.postprocess(seqs, token_ids)
        outputs = []
        for seq, before in zip(seqs, num_tokens_before, strict=True):
            if seq.num_tokens == before:
                continue
            if self._should_stop(seq):
                self.scheduler.finish(seq)
                outputs.append(seq)
        return outputs

    def is_finished(self) -> bool:
        """True when no request is waiting or running."""
        return self.scheduler.is_finished()

    def _should_stop(self, seq: Sequence) -> bool:
        """EOS / max_tokens rule for a seq that just appended a token."""
        param = seq.sampling_params
        if seq.num_completion_tokens >= param.max_tokens:
            return True
        return not param.ignore_eos and seq.last_token == self.eos_token_id
