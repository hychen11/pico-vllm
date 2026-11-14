"""Tests for the M3 native Qwen3 model: config, weight mapping, shapes, tying."""

from types import SimpleNamespace

import pytest
import torch

from nano_vllm.config import Config, get_rope_theta
from nano_vllm.models.qwen3 import Qwen3ForCausalLM
from nano_vllm.utils.loader import load_weights

transformers = pytest.importorskip("transformers")

VOCAB, HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 128, 64, 96, 4, 2, 16


def tiny_config(**overrides: object) -> "transformers.Qwen3Config":
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER, num_hidden_layers=2,
        num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    kwargs.update(overrides)
    return transformers.Qwen3Config(**kwargs)


def test_qkv_weights_load_into_their_slices() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config(attention_bias=True))
    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    q, k, v = (torch.randn(s, HIDDEN) for s in (q_size, kv_size, kv_size))
    qb, kb, vb = (torch.randn(s) for s in (q_size, kv_size, kv_size))
    prefix = "model.layers.1.self_attn"
    load_weights(model, [
        (f"{prefix}.q_proj.weight", q), (f"{prefix}.k_proj.weight", k),
        (f"{prefix}.v_proj.weight", v), (f"{prefix}.q_proj.bias", qb),
        (f"{prefix}.k_proj.bias", kb), (f"{prefix}.v_proj.bias", vb),
    ])
    qkv = model.model.layers[1].self_attn.qkv_proj
    assert qkv.weight.shape == (q_size + 2 * kv_size, HIDDEN)
    assert torch.equal(qkv.weight, torch.cat([q, k, v]))
    assert torch.equal(qkv.bias, torch.cat([qb, kb, vb]))


def test_gate_up_weights_load_into_their_slices() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config())
    gate, up = torch.randn(INTER, HIDDEN), torch.randn(INTER, HIDDEN)
    load_weights(model, [
        ("model.layers.0.mlp.up_proj.weight", up),
        ("model.layers.0.mlp.gate_proj.weight", gate),
    ])
    gate_up = model.model.layers[0].mlp.gate_up_proj.weight
    assert gate_up.shape == (2 * INTER, HIDDEN)
    assert torch.equal(gate_up[:INTER], gate)
    assert torch.equal(gate_up[INTER:], up)


def test_load_weights_rejects_unexpected_names_and_shapes() -> None:
    model = Qwen3ForCausalLM(tiny_config())
    with pytest.raises(KeyError):
        load_weights(model, [("model.layers.0.self_attn.rotary.weight", torch.zeros(1))])
    with pytest.raises(AssertionError):
        load_weights(model, [("model.norm.weight", torch.zeros(HIDDEN + 1))])
    with pytest.raises(AssertionError):
        load_weights(model, [("model.layers.0.self_attn.k_proj.weight", torch.zeros(1, HIDDEN))])


def test_gqa_intermediate_and_output_shapes() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    attn = model.model.layers[0].self_attn
    seen: dict = {}
    hooks = [
        attn.q_norm.register_forward_hook(lambda m, i, o: seen.__setitem__("q", o.shape)),
        attn.k_norm.register_forward_hook(lambda m, i, o: seen.__setitem__("k", o.shape)),
        attn.o_proj.register_forward_hook(
            lambda m, i, o: seen.update(o_in=i[0].shape, o_out=o.shape)
        ),
    ]
    n = 5
    with torch.no_grad():
        hidden = model(torch.arange(n), torch.arange(n))
    for h in hooks:
        h.remove()
    assert seen["q"] == (n, HEADS, HEAD_DIM)
    assert seen["k"] == (n, KV_HEADS, HEAD_DIM)
    assert seen["o_in"] == (n, HEADS * HEAD_DIM)
    assert seen["o_out"] == (n, HIDDEN)
    assert hidden.shape == (n, HIDDEN)


def test_forward_produces_finite_logits() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    ids = torch.tensor([1, 7, 42, 127, 3, 0])
    with torch.no_grad():
        logits = model.compute_logits(model(ids, torch.arange(len(ids))))
    assert logits.shape == (len(ids), VOCAB)
    assert torch.isfinite(logits).all()


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    a, b = torch.tensor([5, 9, 11, 2]), torch.tensor([5, 9, 11, 100])
    with torch.no_grad():
        ha, hb = model(a, torch.arange(4)), model(b, torch.arange(4))
    torch.testing.assert_close(ha[:3], hb[:3])
    assert not torch.allclose(ha[3], hb[3])


def test_tied_embeddings_share_parameter() -> None:
    model = Qwen3ForCausalLM(tiny_config(tie_word_embeddings=True))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    load_weights(model, [("lm_head.weight", torch.ones(VOCAB, HIDDEN))])
    assert torch.equal(model.model.embed_tokens.weight, torch.ones(VOCAB, HIDDEN))


def test_untied_embeddings_are_separate() -> None:
    model = Qwen3ForCausalLM(tiny_config(tie_word_embeddings=False))
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    load_weights(model, [("lm_head.weight", torch.ones(VOCAB, HIDDEN))])
    assert not torch.equal(model.model.embed_tokens.weight, torch.ones(VOCAB, HIDDEN))


def test_get_rope_theta_reads_rope_parameters() -> None:
    config = tiny_config(rope_parameters={"rope_type": "default", "rope_theta": 1234.0})
    assert get_rope_theta(config) == 1234.0
    legacy = SimpleNamespace(rope_theta=5000.0, rope_scaling=None)
    assert get_rope_theta(legacy) == 5000.0


def test_get_rope_theta_rejects_scaled_rope() -> None:
    yarn = SimpleNamespace(rope_parameters={"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0})
    with pytest.raises(NotImplementedError):
        get_rope_theta(yarn)
    legacy = SimpleNamespace(rope_theta=1e6, rope_scaling={"type": "linear", "factor": 2.0})
    with pytest.raises(NotImplementedError):
        get_rope_theta(legacy)


def test_config_loads_local_dir(tmp_path) -> None:
    tiny_config(max_position_embeddings=1024).save_pretrained(tmp_path)
    config = Config(str(tmp_path))
    assert config.hf_config.model_type == "qwen3"
    assert config.hf_config.num_key_value_heads == KV_HEADS
    assert config.max_model_len == 1024  # clamped to max_position_embeddings


def test_config_rejects_non_directory(tmp_path) -> None:
    file = tmp_path / "config.json"
    file.write_text("{}")
    with pytest.raises(ValueError):
        Config(str(file))
    with pytest.raises(ValueError):
        Config(str(tmp_path / "missing"))
