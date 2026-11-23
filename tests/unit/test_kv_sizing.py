"""KV-cache sizing: fixed capacity, CPU default, and the GPU-memory budget (M6 step 5).

GPU-free: the budget arithmetic and the engine's choice between the three sources
(``mem_get_info`` and the warm-up are faked). The real warm-up runs in
``tests/cuda/test_kv_sizing_cuda.py``.
"""

from __future__ import annotations

import pytest
import torch

from pico_vllm.config import DEFAULT_CPU_KVCACHE_BYTES, Config
from pico_vllm.core.paged_kv_cache import kv_block_bytes
from pico_vllm.engine import llm_engine
from pico_vllm.worker.memory import num_blocks_from_memory, warmup_shape

transformers = pytest.importorskip("transformers")

GiB = 2**30


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Only config.json: ``Config`` reads it; ``num_kv_blocks`` needs no weights."""
    path = tmp_path_factory.mktemp("tiny_qwen3_config")
    transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64,
    ).save_pretrained(path)
    return str(path)


def block_bytes(config: Config) -> int:
    hf = config.hf_config
    return kv_block_bytes(hf.num_hidden_layers, config.kvcache_block_size,
                          hf.num_key_value_heads, hf.head_dim, config.dtype)


# --- arithmetic -----------------------------------------------------------------


def test_kv_block_bytes_qwen3_0_6b() -> None:
    # 2 (K+V) * 28 layers * 256 tokens * 8 kv heads * 128 dim * 2 bytes = 28 MiB.
    assert kv_block_bytes(28, 256, 8, 128, torch.bfloat16) == 28 * 2**20


def test_num_blocks_from_memory() -> None:
    # 32 GiB * 0.9 = 28.8 GiB; minus 2 GiB used and 0.8 GiB activations = 26 GiB.
    total, used, act = 32 * GiB, 2 * GiB, int(0.8 * GiB)
    assert num_blocks_from_memory(total, used, act, GiB, 0.9) == 26
    assert num_blocks_from_memory(total, used, act, GiB, 1.0) == 29


def test_num_blocks_from_memory_never_negative() -> None:
    # Another process already holds more than the budget.
    assert num_blocks_from_memory(32 * GiB, 30 * GiB, GiB, 2**20, 0.9) == 0


@pytest.mark.parametrize(
    ("budget", "max_len", "max_seqs", "expected"),
    [
        (16384, 4096, 512, (4, 4096)),   # four full-length prompts
        (16384, 40960, 512, (1, 16384)),  # one prompt, cut to the budget
        (16384, 100, 512, (163, 100)),
        (16384, 100, 8, (8, 100)),       # capped by max_num_seqs
    ],
)
def test_warmup_shape(budget: int, max_len: int, max_seqs: int,
                      expected: tuple[int, int]) -> None:
    num_seqs, seq_len = warmup_shape(budget, max_len, max_seqs)
    assert (num_seqs, seq_len) == expected
    assert num_seqs * seq_len <= budget


# --- Config ---------------------------------------------------------------------


@pytest.mark.parametrize("util", [0.0, -0.1, 1.5])
def test_rejects_bad_gpu_memory_utilization(checkpoint_dir: str, util: float) -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        Config(checkpoint_dir, device="cpu", gpu_memory_utilization=util)


def test_rejects_non_positive_capacity(checkpoint_dir: str) -> None:
    with pytest.raises(ValueError, match="kvcache_capacity_bytes"):
        Config(checkpoint_dir, device="cpu", kvcache_capacity_bytes=0)


# --- which source sizes the cache -----------------------------------------------


@pytest.fixture
def no_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not profile")

    monkeypatch.setattr(llm_engine, "profile_activation_bytes", fail)
    monkeypatch.setattr(llm_engine, "num_blocks_from_gpu_memory", fail)


def test_cpu_default_capacity(checkpoint_dir: str, no_gpu_probe: None) -> None:
    config = Config(checkpoint_dir, device="cpu", kvcache_block_size=16)
    assert config.kvcache_capacity_bytes is None
    n = llm_engine.num_kv_blocks(config, model=None)  # type: ignore[arg-type]
    assert n == DEFAULT_CPU_KVCACHE_BYTES // block_bytes(config)


def test_explicit_capacity_wins_on_cuda(checkpoint_dir: str, no_gpu_probe: None) -> None:
    config = Config(checkpoint_dir, device="cpu", kvcache_block_size=16,
                    kvcache_capacity_bytes=10 * 2**20)
    config.device = torch.device("cuda")  # no GPU needed: nothing is profiled
    n = llm_engine.num_kv_blocks(config, model=None)  # type: ignore[arg-type]
    assert n == 10 * 2**20 // block_bytes(config)


def test_cuda_auto_uses_memory_budget(
        checkpoint_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(checkpoint_dir, device="cpu", kvcache_block_size=16,
                    gpu_memory_utilization=0.5, max_num_batched_tokens=64)
    config.device = torch.device("cuda")
    calls = {}

    def fake_profile(model: object, hf: object, **kwargs: object) -> int:
        calls["profile"] = kwargs
        return 123

    def fake_budget(device: torch.device, activation: int, bb: int, util: float) -> int:
        calls["budget"] = (device, activation, bb, util)
        return 7

    monkeypatch.setattr(llm_engine, "profile_activation_bytes", fake_profile)
    monkeypatch.setattr(llm_engine, "num_blocks_from_gpu_memory", fake_budget)
    assert llm_engine.num_kv_blocks(config, model=None) == 7  # type: ignore[arg-type]
    assert calls["profile"]["max_num_batched_tokens"] == 64
    assert calls["budget"] == (torch.device("cuda"), 123, block_bytes(config), 0.5)


def test_no_block_fits_raises(checkpoint_dir: str) -> None:
    config = Config(checkpoint_dir, device="cpu", kvcache_block_size=16,
                    kvcache_capacity_bytes=1)
    with pytest.raises(ValueError, match="no KV-cache block"):
        llm_engine.num_kv_blocks(config, model=None)  # type: ignore[arg-type]
