"""Numerical parity of the native Qwen3 model against transformers' reference."""

import pytest
import torch

from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.utils.loader import load_model, load_weights

transformers = pytest.importorskip("transformers")

PROMPT = torch.tensor([3, 17, 99, 4, 64, 127, 0, 42, 8])


def build_hf(**overrides: object) -> "transformers.Qwen3ForCausalLM":
    kwargs = dict(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False, attention_bias=False,
    )
    kwargs.update(overrides)
    config = transformers.Qwen3Config(**kwargs)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = transformers.Qwen3ForCausalLM(config).float().eval()
    # Randomize every parameter (norm weights and biases default to 1/0).
    with torch.no_grad():
        for p in hf.parameters():
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if p.dim() == 1 and p.mean() == 1 else 0))
    return hf


def hf_logits(hf: "transformers.Qwen3ForCausalLM") -> torch.Tensor:
    with torch.no_grad():
        return hf(input_ids=PROMPT.unsqueeze(0)).logits[0]


def native_logits(model: Qwen3ForCausalLM) -> torch.Tensor:
    with torch.no_grad():
        return model.compute_logits(model(PROMPT, torch.arange(len(PROMPT))))


CASES = {
    "untied": dict(tie_word_embeddings=False),
    "tied": dict(tie_word_embeddings=True),
    "attention_bias": dict(attention_bias=True),
    "tied_bias": dict(tie_word_embeddings=True, attention_bias=True),
}


@pytest.mark.parametrize("overrides", CASES.values(), ids=CASES.keys())
def test_logits_match_transformers(overrides: dict) -> None:
    hf = build_hf(**overrides)
    model = Qwen3ForCausalLM(hf.config).eval()
    load_weights(model, hf.state_dict().items())
    if hf.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(native_logits(model), hf_logits(hf), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("overrides", CASES.values(), ids=CASES.keys())
def test_load_model_from_safetensors_matches_transformers(tmp_path, overrides: dict) -> None:
    pytest.importorskip("safetensors")
    hf = build_hf(**overrides)
    hf.save_pretrained(tmp_path)  # writes model.safetensors (+ config.json)
    assert list(tmp_path.glob("*.safetensors"))
    model = Qwen3ForCausalLM(transformers.AutoConfig.from_pretrained(tmp_path)).eval()
    load_model(model, str(tmp_path))
    torch.testing.assert_close(native_logits(model), hf_logits(hf), atol=1e-4, rtol=1e-4)


def test_load_model_from_raw_safetensors_file(tmp_path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    hf = build_hf(tie_word_embeddings=True)
    state = {k: v for k, v in hf.state_dict().items() if k != "lm_head.weight"}
    safetensors_torch.save_file({k: v.contiguous() for k, v in state.items()},
                                str(tmp_path / "model-00001.safetensors"))
    model = Qwen3ForCausalLM(hf.config).eval()
    load_model(model, str(tmp_path))
    torch.testing.assert_close(native_logits(model), hf_logits(hf), atol=1e-4, rtol=1e-4)


def test_load_model_requires_safetensors_files(tmp_path) -> None:
    model = Qwen3ForCausalLM(build_hf().config)
    with pytest.raises(FileNotFoundError):
        load_model(model, str(tmp_path))


def test_load_model_rejects_missing_tensors(tmp_path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    hf = build_hf()
    state = {k: v.contiguous() for k, v in hf.state_dict().items() if k != "model.norm.weight"}
    safetensors_torch.save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="model.norm.weight"):
        load_model(Qwen3ForCausalLM(hf.config), str(tmp_path))
