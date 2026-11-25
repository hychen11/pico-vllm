"""Engine configuration and Qwen3 checkpoint config loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

DEFAULT_CPU_KVCACHE_BYTES = 2 * 2**30

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Config:
    """Settings for one engine instance backed by a local Qwen3 checkpoint directory."""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    # Bytes for the whole paged KV cache (K and V, all layers); sizes num_blocks.
    # None: on CUDA, fill ``gpu_memory_utilization`` of the GPU after a warm-up
    # (``worker.memory``); on CPU, ``DEFAULT_CPU_KVCACHE_BYTES``.
    kvcache_capacity_bytes: int | None = None
    # Fraction of the GPU's total memory this engine may use (weights, activations,
    # KV cache; memory other processes hold counts against it too). Ignored when
    # ``kvcache_capacity_bytes`` is set or on CPU.
    gpu_memory_utilization: float = 0.9
    # "auto": "cuda" when available, else "cpu". Resolved to a ``torch.device``.
    device: str | torch.device = "auto"
    # "auto": the checkpoint dtype on CUDA (bfloat16 if unset), float32 on CPU.
    # Otherwise "float32" / "bfloat16" / "float16" or a ``torch.dtype``.
    dtype: str | torch.dtype = "auto"
    # FlashAttention 2 (M6). None: on whenever the setup supports it (see
    # ``resolve_use_flash``); True: require it; False: force the eager attention.
    use_flash: bool | None = None
    # True: never use CUDA Graphs. Otherwise decode steps replay captured graphs
    # whenever FlashAttention and the Triton KV write are on (M6 step 6).
    enforce_eager: bool = False
    # Number of GPUs to split each layer across (M8, single machine; rank i uses
    # cuda:i). > 1 forces ``enforce_eager``. ``use_flash`` is not changed: pass
    # False when not every GPU can run FlashAttention 2.
    tensor_parallel_size: int = 1
    hf_config: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1]: {self.gpu_memory_utilization}"
            )
        if self.kvcache_capacity_bytes is not None and self.kvcache_capacity_bytes <= 0:
            raise ValueError(
                f"kvcache_capacity_bytes must be positive: {self.kvcache_capacity_bytes}"
            )
        if not os.path.isdir(self.model):
            raise ValueError(f"model must be a local directory: {self.model!r}")
        # Imported lazily: transformers is an optional dependency (``--extra model``).
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        self.device = resolve_device(self.device)
        tp, hf = self.tensor_parallel_size, self.hf_config
        if tp < 1 or hf.num_attention_heads % tp or hf.num_key_value_heads % tp:
            raise ValueError(f"tensor_parallel_size={tp} must be >= 1 and divide "
                             f"num_attention_heads and num_key_value_heads")
        if tp > 1:
            self.enforce_eager = True  # graph capture under TP would hang (ranks not in lockstep)
        self.dtype = resolve_dtype(self.dtype, self.device, self.hf_config)
        self.use_flash = resolve_use_flash(
            self.use_flash, self.device, self.dtype, self.kvcache_block_size
        )


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
            ckpt = DTYPES.get(ckpt)
        return ckpt if isinstance(ckpt, torch.dtype) else torch.bfloat16
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be 'auto', {sorted(DTYPES)} or a torch.dtype: {dtype!r}")
    return DTYPES[dtype]


def resolve_use_flash(
        use_flash: bool | None, device: torch.device, dtype: torch.dtype, block_size: int
) -> bool:
    """FA2 needs CUDA, fp16/bf16, an importable ``flash_attn``, and a paged-KV block
    size that is a multiple of 256. ``None`` turns it on exactly when all hold."""
    from pico_vllm.layers.attention import HAS_FLASH_ATTN

    supported = (
        torch.device(device).type == "cuda"
        and dtype in (torch.float16, torch.bfloat16)
        and HAS_FLASH_ATTN
        and block_size % 256 == 0
    )
    if use_flash is None:
        return supported
    if use_flash and not supported:
        raise ValueError(
            "use_flash=True needs a CUDA device, float16/bfloat16, an importable "
            f"flash_attn and block_size % 256 == 0 (got {device}, {dtype}, {block_size})"
        )
    return use_flash


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
