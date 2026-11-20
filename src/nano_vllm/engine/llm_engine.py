"""The step loop that ties Scheduler, ModelRunner, and the sampler together (M5).

One ``step()``::

    scheduler.schedule() -> model_runner.run() -> sample() -> scheduler.postprocess()
    -> finish seqs that hit EOS or max_tokens (frees their blocks)

This is the ``drive()`` loop from ``tests/integration/test_runner_fake_model.py``
plus stopping rules.
"""

from __future__ import annotations

from typing import Any

import torch

from nano_vllm.config import Config
from nano_vllm.core.paged_kv_cache import PagedKVCache, num_blocks_from_capacity
from nano_vllm.core.scheduler import Scheduler
from nano_vllm.models.qwen3 import Qwen3ForCausalLM
from nano_vllm.sampler import sample
from nano_vllm.sampling_params import SamplingParams
from nano_vllm.sequence import Sequence
from nano_vllm.utils.loader import load_model
from nano_vllm.worker.model_runner import ModelRunner


class LLMEngine:
    def __init__(
            self,
            scheduler: Scheduler,
            model_runner: ModelRunner,
            eos_token_id: int | None,
            tokenizer: Any | None = None,
    ) -> None:
        """Wire already-built components together.

        Taking components (rather than a model path) lets integration tests build
        an engine around a tiny random Qwen3 with token-id prompts and no tokenizer.
        Use ``from_config`` for a real checkpoint.
        """
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config: Config) -> LLMEngine:
        """Build every component for the local checkpoint ``config.model``.

        - ``Qwen3ForCausalLM(config.hf_config)`` + ``load_model(model, config.model)``;
        - ``PagedKVCache`` sized by ``num_blocks_from_capacity`` (pick a capacity);
        - ``Scheduler`` and ``PagedKVCache`` must share ``num_blocks`` and block size;
        - set ``Sequence.block_size = config.kvcache_block_size`` (class attribute!);
        - ``AutoTokenizer.from_pretrained(config.model)``; EOS id from the tokenizer.
        """

        model = Qwen3ForCausalLM(config.hf_config)
        load_model(model, config.model)
        model.eval()

        hf = config.hf_config
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        dtype = next(model.parameters()).dtype
        block_size = config.kvcache_block_size
        num_blocks = num_blocks_from_capacity(
            config.kvcache_capacity_bytes, hf.num_hidden_layers, block_size,
            hf.num_key_value_heads, head_dim, dtype,
        )

        kv_cache = PagedKVCache(
            hf.num_hidden_layers, num_blocks, block_size, hf.num_key_value_heads, head_dim,
            dtype=dtype,
        )
        Sequence.block_size = block_size
        scheduler = Scheduler(
            config.max_num_seqs, config.max_num_batched_tokens, num_blocks, block_size
        )
        runner = ModelRunner(model, kv_cache, block_size)
        # Imported lazily: transformers is an optional dependency (``--extra model``).
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model)
        eos_token_id = tokenizer.eos_token_id
        return cls(scheduler, runner, eos_token_id, tokenizer)

    def add_request(
            self, prompt: str | list[int], sampling_params: SamplingParams
    ) -> Sequence:
        """Queue one request. A ``str`` prompt is encoded with the tokenizer;
        a ``list[int]`` is used as token ids directly. Returns the new Sequence."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
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
