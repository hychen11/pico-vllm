"""Engine configuration and Qwen3 checkpoint config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """Settings for one engine instance backed by a local Qwen3 checkpoint directory."""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    hf_config: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be a local directory: {self.model!r}")
        # Imported lazily: transformers is an optional dependency (``--extra model``).
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


def get_rope_theta(hf_config: Any) -> float:
    """Return the RoPE base, rejecting RoPE scaling variants (deferred to Track B).

    transformers v5 stores it in ``rope_parameters``; v4 used ``rope_theta``/``rope_scaling``.
    """
    rope = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None)
    if rope:
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"RoPE scaling {rope_type!r} is not supported")
        if "rope_theta" in rope:
            return float(rope["rope_theta"])
    return float(hf_config.rope_theta)
