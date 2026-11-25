"""Config validation for tensor_parallel_size (M8)."""

from __future__ import annotations

import pytest
import torch

from pico_vllm.config import Config

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """4 query heads, 2 KV heads: tp=2 splits both, tp=4 only the query heads."""
    path = tmp_path_factory.mktemp("tiny_qwen3_tp_config")
    transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64,
    ).save_pretrained(path)
    return str(path)


@pytest.fixture
def two_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


def test_default_is_one_and_keeps_graphs(checkpoint_dir: str) -> None:
    config = Config(checkpoint_dir, device="cpu")
    assert config.tensor_parallel_size == 1
    assert config.enforce_eager is False


@pytest.mark.parametrize("tp", [0, -1, 3, 4])  # 3: splits neither; 4: query heads, not KV
def test_rejects_bad_size(checkpoint_dir: str, tp: int) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        Config(checkpoint_dir, device="cpu", tensor_parallel_size=tp)


@pytest.mark.parametrize("use_flash", [False, None])
def test_tp_forces_eager_and_leaves_use_flash(checkpoint_dir: str, two_gpus: None,
                                              use_flash: bool | None) -> None:
    config = Config(checkpoint_dir, device="cuda", tensor_parallel_size=2,
                    kvcache_block_size=256, use_flash=use_flash)
    assert config.enforce_eager is True
    single = Config(checkpoint_dir, device="cuda", kvcache_block_size=256, use_flash=use_flash)
    assert config.use_flash == single.use_flash  # TP does not touch use_flash
