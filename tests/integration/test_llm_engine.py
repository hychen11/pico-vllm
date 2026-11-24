"""LLMEngine / LLM.generate end to end (M5).

Most tests build an engine around a tiny random Qwen3 with token-id prompts and
no tokenizer: ``LLMEngine(scheduler, runner, eos_token_id)``. The last test runs a
real checkpoint and is skipped unless ``PICO_VLLM_QWEN_MODEL`` is set.
"""

from __future__ import annotations

import os

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.engine.llm_engine import LLMEngine
from pico_vllm.llm import LLM
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence, SequenceStatus
from pico_vllm.worker.model_runner import ModelRunner

transformers = pytest.importorskip("transformers")
from pico_vllm.models.qwen3 import Qwen3ForCausalLM  # noqa: E402

pytestmark = pytest.mark.integration

BS, NUM_BLOCKS, LAYERS, KV_HEADS, HEAD_DIM, VOCAB = 4, 64, 2, 2, 16, 128
GREEDY = 0.0


@pytest.fixture(autouse=True)
def _small_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sequence.block_size is a class attribute; it must match the scheduler's.
    monkeypatch.setattr(Sequence, "block_size", BS)


@pytest.fixture(scope="module")
def model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


def make_engine(
    model: Qwen3ForCausalLM, eos_token_id: int | None = None, budget: int = 64
) -> LLMEngine:
    scheduler = Scheduler(
        max_num_seqs=8, max_num_batched_tokens=budget, num_blocks=NUM_BLOCKS, block_size=BS
    )
    runner = ModelRunner(model, PagedKVCache(LAYERS, NUM_BLOCKS, BS, KV_HEADS, HEAD_DIM), BS)
    return LLMEngine(scheduler, runner, eos_token_id)


def make_llm(engine: LLMEngine) -> LLM:
    """An ``LLM`` around a prebuilt engine (``LLM.__init__`` needs a checkpoint dir)."""
    llm = LLM.__new__(LLM)
    llm.engine = engine
    return llm


def prompt(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


def greedy_reference(model: Qwen3ForCausalLM, prompt_ids: list[int], n: int) -> list[int]:
    """Greedy decode by recomputing the whole sequence each step (no KV cache)."""
    ids = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(n):
            hidden = model(torch.tensor(ids), torch.arange(len(ids)))
            ids.append(int(model.compute_logits(hidden)[-1].argmax()))
    return ids[len(prompt_ids) :]


def num_free_blocks(engine: LLMEngine) -> int:
    return len(engine.scheduler.block_manager.free_block_ids)


# --- greedy correctness ---------------------------------------------------------


def test_two_requests_interleave_and_match_greedy_reference(model: Qwen3ForCausalLM) -> None:
    prompts = [prompt(9, seed=1), prompt(6, seed=2)]
    llm = make_llm(make_engine(model))
    outputs = llm.generate(prompts, SamplingParams(temperature=GREEDY, max_tokens=5))

    assert len(outputs) == 2
    for p, out in zip(prompts, outputs, strict=True):
        assert out["text"] is None  # no tokenizer
        assert out["token_ids"] == greedy_reference(model, p, 5)


def test_chunked_prefill_matches_greedy_reference(model: Qwen3ForCausalLM) -> None:
    p = prompt(10, seed=3)
    llm = make_llm(make_engine(model, budget=4))  # prefill in chunks of 4, 4, 2
    [out] = llm.generate([p], SamplingParams(temperature=GREEDY, max_tokens=4))
    assert out["token_ids"] == greedy_reference(model, p, 4)


# --- stopping rules -------------------------------------------------------------


@pytest.mark.parametrize("max_tokens", [1, 3, 7])
def test_stops_after_exactly_max_tokens_and_frees_blocks(
    model: Qwen3ForCausalLM, max_tokens: int
) -> None:
    engine = make_engine(model)
    seq = engine.add_request(prompt(6, seed=4), SamplingParams(temperature=GREEDY,
                                                                 max_tokens=max_tokens))
    finished: list[Sequence] = []
    steps = 0
    while not engine.is_finished():
        finished += engine.step()
        steps += 1
        assert steps <= max_tokens + 1, "engine did not stop"

    assert finished == [seq]
    assert seq.status == SequenceStatus.FINISHED
    assert seq.num_completion_tokens == max_tokens
    assert num_free_blocks(engine) == NUM_BLOCKS


def test_step_returns_each_finished_seq_once(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    short = engine.add_request(prompt(5, seed=5), SamplingParams(temperature=GREEDY,
                                                                   max_tokens=2))
    long = engine.add_request(prompt(5, seed=6), SamplingParams(temperature=GREEDY,
                                                                  max_tokens=4))
    per_step = []
    while not engine.is_finished():
        per_step.append(engine.step())
    # prefill makes token 1; decodes make tokens 2..4.
    assert per_step == [[], [short], [], [long]]


def test_stops_on_eos(model: Qwen3ForCausalLM) -> None:
    p = prompt(7, seed=7)
    ref = greedy_reference(model, p, 6)
    eos = ref[2]  # greedy decoding produces this as its 3rd token
    stop_at = ref.index(eos) + 1  # first occurrence (could be earlier than index 2)

    engine = make_engine(model, eos_token_id=eos)
    [out] = make_llm(engine).generate([p], SamplingParams(temperature=GREEDY, max_tokens=6))
    assert out["token_ids"] == ref[:stop_at]
    assert out["token_ids"][-1] == eos
    assert num_free_blocks(engine) == NUM_BLOCKS


def test_ignore_eos_runs_to_max_tokens(model: Qwen3ForCausalLM) -> None:
    p = prompt(7, seed=7)
    ref = greedy_reference(model, p, 6)
    engine = make_engine(model, eos_token_id=ref[0])  # EOS would stop after 1 token
    params = SamplingParams(temperature=GREEDY, max_tokens=6, ignore_eos=True)
    [out] = make_llm(engine).generate([p], params)
    assert out["token_ids"] == ref


def test_prompt_ending_in_eos_does_not_stop_during_chunked_prefill(
    model: Qwen3ForCausalLM,
) -> None:
    """A non-final chunk appends nothing; its last_token is still a prompt token."""
    eos = 99
    p = prompt(9, seed=8) + [eos]
    engine = make_engine(model, eos_token_id=eos, budget=4)
    params = SamplingParams(temperature=GREEDY, max_tokens=3, ignore_eos=False)
    [out] = make_llm(engine).generate([p], params)
    assert len(out["token_ids"]) >= 1  # it generated after the whole prompt was prefilled
    assert out["token_ids"] == greedy_reference(model, p, len(out["token_ids"]))


# --- generate() API -------------------------------------------------------------


def test_generate_returns_input_order_when_finishing_out_of_order(
    model: Qwen3ForCausalLM,
) -> None:
    prompts = [prompt(6, seed=9), prompt(6, seed=10), prompt(6, seed=11)]
    params = [
        SamplingParams(temperature=GREEDY, max_tokens=6),  # finishes last
        SamplingParams(temperature=GREEDY, max_tokens=1),  # finishes first
        SamplingParams(temperature=GREEDY, max_tokens=3),
    ]
    outputs = make_llm(make_engine(model)).generate(prompts, params)
    assert [len(o["token_ids"]) for o in outputs] == [6, 1, 3]
    for p, sp, out in zip(prompts, params, outputs, strict=True):
        assert out["token_ids"] == greedy_reference(model, p, sp.max_tokens)


def test_add_request_rejects_requests_longer_than_max_model_len(
        model: Qwen3ForCausalLM,
) -> None:
    engine = make_engine(model)
    engine.max_model_len = 20
    engine.add_request(prompt(12, seed=7), SamplingParams(max_tokens=8))  # 20: fits
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        engine.add_request(prompt(13, seed=7), SamplingParams(max_tokens=8))


def test_generate_rejects_mismatched_params_list(model: Qwen3ForCausalLM) -> None:
    llm = make_llm(make_engine(model))
    with pytest.raises(ValueError):
        llm.generate([prompt(4, seed=12), prompt(4, seed=13)], [SamplingParams()])


def test_sampled_requests_finish_with_valid_tokens(model: Qwen3ForCausalLM) -> None:
    prompts = [prompt(5, seed=14), prompt(8, seed=15)]
    params = SamplingParams(temperature=1.0, max_tokens=4)
    outputs = make_llm(make_engine(model)).generate(prompts, params)
    for out in outputs:
        assert len(out["token_ids"]) == 4
        assert all(0 <= t < VOCAB for t in out["token_ids"])


# --- real checkpoint --------------------------------------------------------------

MODEL_PATH = os.environ.get("PICO_VLLM_QWEN_MODEL")


@pytest.mark.model
@pytest.mark.skipif(not MODEL_PATH, reason="set PICO_VLLM_QWEN_MODEL to a local Qwen3 dir")
def test_greedy_generate_matches_transformers() -> None:
    max_tokens = 16
    prompts = ["The capital of France is", "def fibonacci(n):"]
    # Exact token match with HF needs the same numerics: pin CPU + float32.
    llm = LLM(MODEL_PATH, max_model_len=256, kvcache_block_size=16,
              kvcache_capacity_bytes=256 * 2**20, device="cpu", dtype="float32")
    outputs = llm.generate(prompts, SamplingParams(temperature=GREEDY, max_tokens=max_tokens,
                                                   ignore_eos=True))

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()
    for text, out in zip(prompts, outputs, strict=True):
        ids = tokenizer(text, return_tensors="pt").input_ids
        with torch.inference_mode():
            ref = hf.generate(ids, max_new_tokens=max_tokens, min_new_tokens=max_tokens,
                              do_sample=False)
        assert out["token_ids"] == ref[0, ids.shape[1] :].tolist()
        assert out["text"] == tokenizer.decode(out["token_ids"])
