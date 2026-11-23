"""The real warm-up and GPU-memory sizing on CUDA (M6 step 5)."""

from __future__ import annotations

import gc
import os

import pytest
import torch

from pico_vllm.engine.llm_engine import num_kv_blocks
from pico_vllm.llm import LLM
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.utils.loader import init_on
from pico_vllm.worker.memory import profile_activation_bytes

transformers = pytest.importorskip("transformers")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def gpu_used_fraction() -> float:
    free, total = torch.cuda.mem_get_info()
    return 1 - free / total


# Auto sizing needs headroom below gpu_memory_utilization; another process may hold it.
needs_free_gpu = pytest.mark.skipif(
    torch.cuda.is_available() and gpu_used_fraction() > 0.4,
    reason="GPU memory is mostly held by other processes",
)


def tiny_hf_config() -> transformers.Qwen3Config:
    return transformers.Qwen3Config(
        vocab_size=1024, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny_qwen3_cuda")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(tiny_hf_config()).save_pretrained(path)
    return str(path)


def test_profile_frees_its_temporary_cache() -> None:
    hf = tiny_hf_config()
    with init_on(torch.device("cuda"), torch.bfloat16):
        model = Qwen3ForCausalLM(hf).eval()
    # Collect earlier tests' garbage first: the warm-up allocates enough Python
    # objects to trigger gc itself, which would free it mid-test.
    gc.collect()
    before = torch.cuda.memory_allocated()
    small = profile_activation_bytes(model, hf, block_size=16, max_num_batched_tokens=256,
                                     max_model_len=256, max_num_seqs=4, use_flash=False)
    assert torch.cuda.memory_allocated() == before
    assert small > 0
    # More tokens per step and more decode rows -> a larger peak.
    large = profile_activation_bytes(model, hf, block_size=16, max_num_batched_tokens=2048,
                                     max_model_len=256, max_num_seqs=256, use_flash=False)
    assert large > small


@needs_free_gpu
def test_auto_sized_cache_fits_the_budget(checkpoint: str) -> None:
    util = 0.9
    llm = LLM(checkpoint, device="cuda", kvcache_block_size=16, max_model_len=256,
              max_num_batched_tokens=512, max_num_seqs=8, gpu_memory_utilization=util)
    cache = llm.engine.model_runner.kv_cache
    num_blocks = cache.k_cache.shape[1]
    assert llm.engine.scheduler.block_manager.num_blocks == num_blocks
    free, total = torch.cuda.mem_get_info()
    assert total - free <= util * total + 64 * 2**20  # caching-allocator slack

    outputs = llm.generate([[1, 2, 3], [4, 5, 6, 7]],
                           SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    assert [len(o["token_ids"]) for o in outputs] == [8, 8]


@needs_free_gpu
def test_lower_utilization_gives_fewer_blocks(checkpoint: str) -> None:
    from pico_vllm.config import Config
    from pico_vllm.engine.llm_engine import build_model

    hi = Config(checkpoint, device="cuda", kvcache_block_size=16, max_model_len=256,
                max_num_batched_tokens=512, max_num_seqs=8, gpu_memory_utilization=0.9)
    lo = Config(checkpoint, device="cuda", kvcache_block_size=16, max_model_len=256,
                max_num_batched_tokens=512, max_num_seqs=8, gpu_memory_utilization=0.5)
    model = build_model(hi)
    assert num_kv_blocks(lo, model) < num_kv_blocks(hi, model)


@needs_free_gpu
def test_second_engine_gets_the_same_blocks(checkpoint: str) -> None:
    """A deleted engine waiting for gc must not shrink the next one's cache."""
    kwargs = dict(device="cuda", kvcache_block_size=16, max_model_len=256,
                  max_num_batched_tokens=512, max_num_seqs=8, gpu_memory_utilization=0.5)
    counts = []
    for _ in range(3):
        llm = LLM(checkpoint, **kwargs)
        counts.append(llm.engine.model_runner.kv_cache.k_cache.shape[1])
        del llm
    assert counts[0] == counts[1] == counts[2]


MODEL_PATH = os.environ.get("PICO_VLLM_QWEN_MODEL")


@pytest.mark.model
@needs_free_gpu
@pytest.mark.skipif(not MODEL_PATH, reason="set PICO_VLLM_QWEN_MODEL to a local Qwen3 dir")
def test_qwen3_auto_sizing_generates() -> None:
    llm = LLM(MODEL_PATH, device="cuda", max_model_len=1024)
    num_blocks = llm.engine.model_runner.kv_cache.k_cache.shape[1]
    assert num_blocks > 0
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=8))
    assert out[0]["token_ids"]
