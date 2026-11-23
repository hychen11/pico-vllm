"""Load Hugging Face Qwen3 checkpoint tensors into the native model."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from glob import glob
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # type annotations only
    from pico_vllm.models.qwen3 import Qwen3ForCausalLM


@contextmanager
def init_on(device: torch.device | str, dtype: torch.dtype) -> Iterator[None]:
    """Construct modules with parameters allocated directly on ``device`` in ``dtype``.

    Avoids building an fp32 CPU copy first and then ``.to(device, dtype)``. Buffers
    created with an explicit dtype (RoPE's float32 ``cos_sin_cache``) keep it, whereas
    ``module.to(bfloat16)`` would downcast them too.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(old_dtype)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    assert param.shape == loaded_weight.shape, (param.shape, loaded_weight.shape)
    param.data.copy_(loaded_weight)


def load_weights(
    model: Qwen3ForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Copy ``(checkpoint_name, tensor)`` pairs into ``model``.

    ``model.packed_modules_mapping`` maps a checkpoint module name (e.g. ``q_proj``) to
    ``(packed_module_name, shard_id)``; matching tensors go into that shard of the packed
    parameter. Every other name must match a model parameter exactly.
    Returns the names of the model parameters that received data.
    """
    mapping = model.packed_modules_mapping
    # remove_duplicate=False keeps tied names such as ``lm_head.weight``.
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded: set[str] = set()
    with torch.no_grad():
        for name, tensor in weights:
            for ckpt_module, (packed_module, shard_id) in mapping.items():
                if f".{ckpt_module}." in name:
                    packed_name = name.replace(f".{ckpt_module}.", f".{packed_module}.")
                    param = params[packed_name]
                    param.weight_loader(param, tensor, shard_id)
                    loaded.add(packed_name)
                    break
            else:
                if name not in params:
                    raise KeyError(f"unexpected checkpoint tensor: {name}")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
    return loaded


def load_model(model: Qwen3ForCausalLM, path: str) -> None:
    """Load every ``*.safetensors`` file in the checkpoint directory ``path``."""
    from safetensors import safe_open

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {path!r}")
    loaded: set[str] = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            loaded |= load_weights(model, ((name, f.get_tensor(name)) for name in f.keys()))
    # A tied parameter counts as loaded if any of its names was loaded.
    loaded_ids = {id(p) for n, p in model.named_parameters(remove_duplicate=False) if n in loaded}
    missing = [n for n, p in model.named_parameters() if id(p) not in loaded_ids]
    if missing:
        raise ValueError(f"checkpoint is missing tensors: {missing}")
