"""Offline generation API (M5).

    llm = LLM("/path/to/Qwen3-0.6B", max_model_len=1024)
    outputs = llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=32))
    outputs[0]["text"], outputs[0]["token_ids"]
"""

from __future__ import annotations

from typing import Any

from pico_vllm.config import Config
from pico_vllm.engine.llm_engine import LLMEngine
from pico_vllm.sampling_params import SamplingParams


class LLM:
    def __init__(self, model: str, **kwargs: Any) -> None:
        """Build ``Config(model, **kwargs)`` and ``LLMEngine.from_config``.

        kwargs: ``max_model_len``, ``max_num_seqs``, ``max_num_batched_tokens``, ...
        """
        config = Config(model=model, **kwargs)
        self.engine = LLMEngine.from_config(config)

    def generate(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict[str, Any]]:
        """Generate a completion for every prompt.

        One ``SamplingParams`` applies to all prompts; a list gives one per prompt.
        ``add_request`` each prompt, ``step()`` until ``is_finished()``, then return
        ``{"text": str | None, "token_ids": list[int]}`` per prompt **in input order**
        (seqs finish out of order; map them back by ``seq_id``).
        ``text`` is ``None`` when the engine has no tokenizer.
        """
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params, strict=True):
            self.engine.add_request(prompt, sp)

        tokenizer = self.engine.tokenizer
        outputs: dict[int, dict[str, Any]] = {}
        while not self.engine.is_finished():
            for seq in self.engine.step():
                # Only the generated part; seq.token_ids also contains the prompt.
                token_ids = seq.completion_token_ids
                text = tokenizer.decode(token_ids) if tokenizer is not None else None
                outputs[seq.seq_id] = {"text": text, "token_ids": token_ids}

        # seq_ids grow in add_request order, so sorting restores input order.
        return [outputs[seq_id] for seq_id in sorted(outputs)]
