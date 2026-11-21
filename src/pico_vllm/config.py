"""Engine configuration and Qwen3 checkpoint config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Config:
    """Settings for one engine instance backed by a local Qwen3 checkpoint directory."""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    # Bytes for the whole paged KV cache (K and V, all layers); sizes num_blocks.
    # Fixed until M6, which sizes the cache from free GPU memory instead.
    kvcache_capacity_bytes: int = 2 * 2**30
    # "auto": "cuda" when available, else "cpu". Resolved to a ``torch.device``.
    device: str | torch.device = "auto"
    # "auto": the checkpoint dtype on CUDA (bfloat16 if unset), float32 on CPU.
    # Otherwise "float32" / "bfloat16" / "float16" or a ``torch.dtype``.
    dtype: str | torch.dtype = "auto"
    hf_config: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be a local directory: {self.model!r}")
        # Imported lazily: transformers is an optional dependency (``--extra model``).
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.device = resolve_device(self.device)
        self.dtype = resolve_dtype(self.dtype, self.device, self.hf_config)


def resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype, device: torch.device, hf_config: Any) -> torch.dtype:
    """``auto`` keeps CPU in float32 (exact M5 parity, and bf16 CPU matmuls are slow)."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        if device.type == "cpu":
            return torch.float32
        # transformers v5 stores it as ``dtype``; v4 as ``torch_dtype``.
        ckpt = getattr(hf_config, "dtype", None) or getattr(hf_config, "torch_dtype", None)
        if isinstance(ckpt, str):
            ckpt = _DTYPES.get(ckpt)
        return ckpt if isinstance(ckpt, torch.dtype) else torch.bfloat16
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be 'auto', {sorted(_DTYPES)} or a torch.dtype: {dtype!r}")
    return _DTYPES[dtype]


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
