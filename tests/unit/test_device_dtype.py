"""Config device/dtype resolution and placing the native model on it (M6 step 0)."""

from __future__ import annotations

import pytest
import torch

from nano_vllm.config import Config, resolve_dtype
from nano_vllm.engine.llm_engine import build_model
from nano_vllm.models.qwen3 import Qwen3ForCausalLM
from nano_vllm.utils.loader import init_on

transformers = pytest.importorskip("transformers")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def tiny_hf_config(**overrides: object) -> transformers.Qwen3Config:
    kwargs = dict(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    kwargs.update(overrides)
    return transformers.Qwen3Config(**kwargs)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, object]:
    """A tiny random HF Qwen3 saved as safetensors; returns (dir, hf_model)."""
    path = tmp_path_factory.mktemp("tiny_qwen3")
    config = tiny_hf_config()
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = transformers.Qwen3ForCausalLM(config).float().eval()
    hf.save_pretrained(path)
    return str(path), hf


# --- resolution -------------------------------------------------------------------


def test_explicit_device_and_dtype(checkpoint) -> None:
    config = Config(checkpoint[0], device="cpu", dtype="bfloat16")
    assert config.device == torch.device("cpu")
    assert config.dtype is torch.bfloat16


def test_auto_on_cpu_is_float32(checkpoint) -> None:
    config = Config(checkpoint[0], device="cpu")
    assert config.dtype is torch.float32


def test_auto_on_cuda_follows_checkpoint() -> None:
    cuda = torch.device("cuda")
    assert resolve_dtype("auto", cuda, tiny_hf_config(dtype="bfloat16")) is torch.bfloat16
    assert resolve_dtype("auto", cuda, tiny_hf_config(dtype="float16")) is torch.float16
    assert resolve_dtype("auto", cuda, tiny_hf_config()) is torch.bfloat16  # unset


def test_auto_device(checkpoint) -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Config(checkpoint[0]).device.type == expected


def test_rejects_unknown_dtype(checkpoint) -> None:
    with pytest.raises(ValueError):
        Config(checkpoint[0], dtype="int8")


# --- placement --------------------------------------------------------------------


def test_init_on_restores_default_dtype() -> None:
    with init_on("cpu", torch.bfloat16):
        model = Qwen3ForCausalLM(tiny_hf_config())
    assert torch.get_default_dtype() is torch.float32
    assert all(p.dtype is torch.bfloat16 for p in model.parameters())
    # Explicit-float32 buffers are not downcast (``model.to(bf16)`` would do that).
    assert model.model.layers[0].self_attn.rotary_emb.cos_sin_cache.dtype is torch.float32


def _check_build(path: str, hf: object, device: str, dtype: torch.dtype, atol: float) -> None:
    model = build_model(Config(path, device=device, dtype=dtype))
    assert all(p.device.type == device and p.dtype is dtype for p in model.parameters())
    ids = torch.tensor([3, 17, 99, 4, 64, 127, 0, 42])
    with torch.inference_mode():
        ref = hf(input_ids=ids[None]).logits[0]
        out = model.compute_logits(model(ids.to(device), torch.arange(8, device=device)))
    torch.testing.assert_close(out.float().cpu(), ref, atol=atol, rtol=0)


def test_build_model_cpu_float32_matches_hf(checkpoint) -> None:
    _check_build(*checkpoint, "cpu", torch.float32, atol=1e-4)


@pytest.mark.cuda
@needs_cuda
def test_build_model_cuda_bfloat16_matches_hf(checkpoint) -> None:
    _check_build(*checkpoint, "cuda", torch.bfloat16, atol=5e-2)
