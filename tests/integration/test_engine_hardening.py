"""LLMEngine correctness under pressure, abort, long prompts and sampling edges (M7).

The reference for every greedy request is ``greedy_reference``: recompute the
whole sequence each step with no KV cache. Whatever the engine does on the way
(batching, chunked prefill, prefix-cache hits, preemption and recompute, aborts
of other requests), each greedy output must equal that reference.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.engine.llm_engine import LLMEngine
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence, SequenceStatus
from pico_vllm.worker.model_runner import ModelRunner

transformers = pytest.importorskip("transformers")
from pico_vllm.models.qwen3 import Qwen3ForCausalLM  # noqa: E402

pytestmark = pytest.mark.integration

BS, LAYERS, KV_HEADS, HEAD_DIM, VOCAB = 4, 2, 2, 16, 128
GREEDY = 0.0


@pytest.fixture(autouse=True)
def _small_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
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


class Recorder:
    """Counts the hard paths an engine run went through, so tests can't go vacuous."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.preempts = self.chunks = self.prefix_hits = 0
        bm = scheduler.block_manager
        preempt, schedule, allocate = scheduler.preempt, scheduler.schedule, bm.allocate

        def counting_preempt(seq: Sequence) -> None:
            self.preempts += 1
            preempt(seq)

        def counting_schedule() -> tuple[list[Sequence], bool]:
            seqs, is_prefill = schedule()
            if is_prefill:
                for q in seqs:
                    self.chunks += q.num_cached_tokens + q.num_scheduled_tokens < q.num_tokens
            return seqs, is_prefill

        def counting_allocate(seq: Sequence, num_cached_blocks: int) -> None:
            self.prefix_hits += num_cached_blocks > 0  # once per admission, not per chunk
            allocate(seq, num_cached_blocks)

        scheduler.preempt = counting_preempt  # type: ignore[method-assign]
        scheduler.schedule = counting_schedule  # type: ignore[method-assign]
        bm.allocate = counting_allocate  # type: ignore[method-assign]


def make_engine(
        model: Qwen3ForCausalLM, num_blocks: int = 64, budget: int = 64, max_seqs: int = 8,
        eos_token_id: int | None = None, max_model_len: int | None = None,
) -> LLMEngine:
    scheduler = Scheduler(max_seqs, budget, num_blocks, BS)
    runner = ModelRunner(model, PagedKVCache(LAYERS, num_blocks, BS, KV_HEADS, HEAD_DIM), BS)
    return LLMEngine(scheduler, runner, eos_token_id, max_model_len=max_model_len)


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
    return ids[len(prompt_ids):]


def run(engine: LLMEngine, check: Callable[..., None] | None = None,
        seqs: list[Sequence] | None = None, max_steps: int = 1000) -> list[Sequence]:
    """Step until finished; return seqs in the order step() reported them."""
    done: list[Sequence] = []
    for _ in range(max_steps):
        if engine.is_finished():
            return done
        done.extend(engine.step())
        if check is not None:
            check(engine.scheduler, seqs)
    raise AssertionError("engine did not finish")


# ------------------------------------------------------------- parity under pressure


@pytest.mark.parametrize(
    ("num_blocks", "budget", "max_seqs"),
    [
        (64, 64, 8),  # roomy: plain batching
        (12, 64, 8),  # tight memory: preemption and recompute
        (64, 6, 8),  # tight budget: chunked prefill
        (12, 7, 3),  # all of it at once
    ],
)
def test_many_requests_match_one_by_one_reference(
        model: Qwen3ForCausalLM, check_invariants: Callable[..., None],
        num_blocks: int, budget: int, max_seqs: int,
) -> None:
    """Batched generation must not let sequences leak into each other: every request
    matches its own no-cache greedy reference, whatever the scheduler did."""
    engine = make_engine(model, num_blocks, budget, max_seqs)
    rec = Recorder(engine.scheduler)
    base = prompt(12, seed=100)
    prompts = [
        base[:9] + prompt(3, seed=1),  # shares two full blocks with base
        prompt(7, seed=2),
        base,
        base[:8] + prompt(5, seed=3),
        prompt(13, seed=4),
        base[:4] + prompt(2, seed=5),
    ]
    max_tokens = [8, 10, 6, 9, 7, 12]
    seqs = [engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=m))
            for p, m in zip(prompts, max_tokens, strict=True)]
    done = run(engine, check_invariants, seqs)

    assert sorted(q.seq_id for q in done) == sorted(q.seq_id for q in seqs)
    for q, p, m in zip(seqs, prompts, max_tokens, strict=True):
        assert q.completion_token_ids == greedy_reference(model, p, m), f"seq {q.seq_id}"
    assert len(engine.scheduler.block_manager.free_block_ids) == num_blocks
    if num_blocks == 12:
        assert rec.preempts > 0
    if budget < 10:
        assert rec.chunks > 0


def test_prefix_cache_hits_produce_the_reference(model: Qwen3ForCausalLM) -> None:
    """Second wave of requests reuses the first wave's freed-but-hashed blocks."""
    engine = make_engine(model)
    rec = Recorder(engine.scheduler)
    base = prompt(14, seed=7)
    sp = SamplingParams(temperature=GREEDY, max_tokens=5)
    engine.add_request(base, sp)
    run(engine)
    second = [base[:12] + prompt(2, seed=8), base, base[:8] + [1]]
    seqs = [engine.add_request(p, sp) for p in second]
    run(engine)
    assert rec.prefix_hits == 3
    for q, p in zip(seqs, second, strict=True):
        assert q.completion_token_ids == greedy_reference(model, p, 5)


# ---------------------------------------------------------------------------- abort


def test_abort_waiting_request_before_it_runs(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    sp = SamplingParams(temperature=GREEDY, max_tokens=4)
    keep = engine.add_request(prompt(6, seed=1), sp)
    gone = engine.add_request(prompt(6, seed=2), sp)
    engine.abort_request(gone)
    assert gone.status == SequenceStatus.FINISHED and gone.num_completion_tokens == 0
    done = run(engine)
    assert done == [keep]
    assert keep.completion_token_ids == greedy_reference(model, prompt(6, seed=1), 4)


def test_abort_running_request_mid_decode(
        model: Qwen3ForCausalLM, check_invariants: Callable[..., None],
) -> None:
    engine = make_engine(model)
    sp = SamplingParams(temperature=GREEDY, max_tokens=10)
    a = engine.add_request(prompt(6, seed=1), sp)
    b = engine.add_request(prompt(9, seed=2), sp)
    for _ in range(4):  # prefill + 3 decode steps
        engine.step()
    assert b.status == SequenceStatus.RUNNING and b.block_table
    engine.abort_request(b)
    check_invariants(engine.scheduler, [a, b])
    assert b.num_completion_tokens == 4 and not b.block_table
    done = run(engine, check_invariants, [a, b])
    assert done == [a]
    assert a.completion_token_ids == greedy_reference(model, prompt(6, seed=1), 10)
    assert len(engine.scheduler.block_manager.free_block_ids) == 64


def test_abort_halfway_through_chunked_prefill(
        model: Qwen3ForCausalLM, check_invariants: Callable[..., None],
) -> None:
    engine = make_engine(model, budget=5)
    long_prompt = prompt(17, seed=3)
    seq = engine.add_request(long_prompt, SamplingParams(temperature=GREEDY, max_tokens=4))
    engine.step()
    engine.step()  # 10 of 17 prompt tokens computed
    assert seq.status == SequenceStatus.WAITING and seq.num_cached_tokens == 10
    engine.abort_request(seq)
    check_invariants(engine.scheduler, [seq])
    assert engine.is_finished()
    assert engine.step() == []
    # The two full blocks it computed stay hashed: the same prompt hits them and
    # still produces the reference.
    rec = Recorder(engine.scheduler)
    again = engine.add_request(long_prompt, SamplingParams(temperature=GREEDY, max_tokens=4))
    run(engine)
    assert rec.prefix_hits == 1
    assert again.completion_token_ids == greedy_reference(model, long_prompt, 4)


def test_abort_seq_sharing_prefix_keeps_the_other_correct(
        model: Qwen3ForCausalLM, check_invariants: Callable[..., None],
) -> None:
    engine = make_engine(model)
    base = prompt(12, seed=5)
    sp = SamplingParams(temperature=GREEDY, max_tokens=8)
    first = engine.add_request(base, sp)
    engine.step()  # first's full blocks are hashed
    second = engine.add_request(base[:8] + prompt(3, seed=6), sp)
    engine.step()  # second shares first's two full blocks
    shared = set(first.block_table) & set(second.block_table)
    assert len(shared) == 2
    engine.abort_request(first)
    for b in shared:  # still held by second
        assert engine.scheduler.block_manager.blocks[b].ref_cnt == 1
    run(engine, check_invariants, [first, second])
    assert second.completion_token_ids == greedy_reference(
        model, base[:8] + prompt(3, seed=6), 8)


def test_abort_preempted_request(
        model: Qwen3ForCausalLM, check_invariants: Callable[..., None],
) -> None:
    engine = make_engine(model, num_blocks=6)
    rec = Recorder(engine.scheduler)
    sp = SamplingParams(temperature=GREEDY, max_tokens=12)
    seqs = [engine.add_request(prompt(7, seed=s), sp) for s in range(3)]
    for _ in range(30):
        engine.step()
        preempted = [q for q in engine.scheduler.waiting if q.num_completion_tokens > 0]
        if preempted:
            break
    assert rec.preempts > 0 and preempted
    victim = preempted[0]
    engine.abort_request(victim)
    check_invariants(engine.scheduler, seqs)
    done = run(engine, check_invariants, seqs)
    assert victim not in done
    for q in seqs:
        if q is not victim:
            assert q.completion_token_ids == greedy_reference(model, q.token_ids[:7], 12)


def test_abort_finished_request_is_a_noop(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    seq = engine.add_request(prompt(5, seed=1), SamplingParams(temperature=GREEDY, max_tokens=2))
    run(engine)
    tokens = list(seq.token_ids)
    engine.abort_request(seq)
    engine.abort_request(seq)
    assert seq.token_ids == tokens and seq.status == SequenceStatus.FINISHED


# -------------------------------------------------------------- request validation


def test_add_request_rejects_empty_prompt(model: Qwen3ForCausalLM) -> None:
    with pytest.raises(ValueError, match="at least one token"):
        make_engine(model).add_request([], SamplingParams())


def test_request_larger_than_kv_pool_is_rejected_not_deadlocked(
        model: Qwen3ForCausalLM,
) -> None:
    """Regression: such a request used to be accepted, then trip an assertion in
    Scheduler._schedule_decode once it outgrew the pool while running alone."""
    engine = make_engine(model, num_blocks=2)  # 8 KV slots
    with pytest.raises(ValueError, match="needs 3 KV blocks"):
        engine.add_request(prompt(7, seed=1), SamplingParams(max_tokens=3))  # 9 positions
    with pytest.raises(ValueError, match="needs 3 KV blocks"):
        engine.add_request(prompt(9, seed=1), SamplingParams(max_tokens=1))
    assert engine.is_finished()


def test_request_that_exactly_fills_the_kv_pool_runs(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model, num_blocks=2)
    p = prompt(7, seed=1)
    seq = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=2))  # 8 positions
    run(engine)
    assert seq.completion_token_ids == greedy_reference(model, p, 2)


def test_step_on_finished_engine_returns_nothing(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    assert engine.step() == []
    engine.add_request(prompt(3, seed=1), SamplingParams(max_tokens=1))
    run(engine)
    assert engine.step() == []


# ------------------------------------------------------------------- long prompts


@pytest.mark.parametrize("n", [4, 8, 16])  # prompt is exactly whole blocks
def test_prompt_of_whole_blocks(model: Qwen3ForCausalLM, n: int) -> None:
    """The last full block is never a cache hit (its last token must run to give
    logits); with an identical earlier prompt the rest is."""
    engine = make_engine(model)
    rec = Recorder(engine.scheduler)
    p = prompt(n, seed=n)
    sp = SamplingParams(temperature=GREEDY, max_tokens=5)
    first = engine.add_request(p, sp)
    run(engine)
    second = engine.add_request(p, sp)
    run(engine)
    expected = greedy_reference(model, p, 5)
    assert first.completion_token_ids == expected == second.completion_token_ids
    assert rec.prefix_hits == (1 if n > BS else 0)


def test_prompt_many_times_the_token_budget(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model, budget=3)
    rec = Recorder(engine.scheduler)
    p = prompt(40, seed=9)
    seq = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=4))
    other = engine.add_request(prompt(5, seed=10), SamplingParams(temperature=GREEDY,
                                                                  max_tokens=4))
    run(engine)
    assert rec.chunks >= 13
    assert seq.completion_token_ids == greedy_reference(model, p, 4)
    assert other.completion_token_ids == greedy_reference(model, prompt(5, seed=10), 4)


def test_prompt_up_to_max_model_len(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model, max_model_len=32)
    p = prompt(29, seed=11)
    seq = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=3))  # 32
    run(engine)
    assert seq.completion_token_ids == greedy_reference(model, p, 3)
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request(prompt(30, seed=11), SamplingParams(max_tokens=3))


def test_single_token_prompt(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    seq = engine.add_request([5], SamplingParams(temperature=GREEDY, max_tokens=6))
    run(engine)
    assert seq.completion_token_ids == greedy_reference(model, [5], 6)


# ------------------------------------------------------------ sampling boundaries


def test_max_tokens_one_finishes_at_prefill(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    sp = SamplingParams(temperature=GREEDY, max_tokens=1)
    seqs = [engine.add_request(prompt(n, seed=n), sp) for n in (3, 6, 9)]
    done = engine.step()  # one prefill step finishes all of them
    assert sorted(q.seq_id for q in done) == [q.seq_id for q in seqs]
    assert engine.is_finished()
    for q in seqs:
        assert q.completion_token_ids == greedy_reference(model, q.token_ids[:-1], 1)


def test_eos_as_first_token(model: Qwen3ForCausalLM) -> None:
    p = prompt(6, seed=12)
    first = greedy_reference(model, p, 1)[0]
    engine = make_engine(model, eos_token_id=first)
    seq = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=10))
    assert engine.step() == [seq]
    assert seq.completion_token_ids == [first]


def test_eos_and_ignore_eos_in_one_batch(model: Qwen3ForCausalLM) -> None:
    p = prompt(6, seed=13)
    ref = greedy_reference(model, p, 8)
    eos = ref[2]
    engine = make_engine(model, eos_token_id=eos)
    stops = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=8))
    keeps = engine.add_request(p, SamplingParams(temperature=GREEDY, max_tokens=8,
                                                 ignore_eos=True))
    run(engine)
    assert stops.completion_token_ids == ref[: ref.index(eos) + 1]
    assert keeps.completion_token_ids == ref


def test_greedy_rows_unaffected_by_sampled_rows(model: Qwen3ForCausalLM) -> None:
    engine = make_engine(model)
    torch.manual_seed(0)
    greedy = [engine.add_request(prompt(5, seed=s), SamplingParams(temperature=GREEDY,
                                                                   max_tokens=6))
              for s in range(3)]
    sampled = [engine.add_request(prompt(5, seed=s), SamplingParams(temperature=t,
                                                                    max_tokens=6))
               for s, t in [(10, 1.0), (11, 1e-30), (12, 50.0)]]
    run(engine)
    for s, q in enumerate(greedy):
        assert q.completion_token_ids == greedy_reference(model, prompt(5, seed=s), 6)
    for q in sampled:
        assert q.num_completion_tokens == 6
        assert all(0 <= t < VOCAB for t in q.completion_token_ids)
    # temperature 1e-30 is greedy in all but name
    assert sampled[1].completion_token_ids == greedy_reference(model, prompt(5, seed=11), 6)
