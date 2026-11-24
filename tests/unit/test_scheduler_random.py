"""Randomized state-machine tests for Scheduler + BlockManager (M7).

A fake engine drives the scheduler with random operations: add a request (often
sharing a prefix with earlier ones), abort a random live request, or run one step
with a fake model that samples random tokens (sometimes EOS). Pools are small so
preemption, chunked prefill and prefix-cache hits all happen. After every
operation ``check_scheduler_invariants`` (tests/conftest.py) checks the queues,
block ref counts, free list and hash index. At the end every request must finish.

Each case is a fixed seed, so a failure reproduces exactly.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.scheduler import Scheduler
from pico_vllm.sampling_params import SamplingParams
from pico_vllm.sequence import Sequence, SequenceStatus

VOCAB = 50
EOS = 0


class FakeEngine:
    """LLMEngine's step loop and stopping rules around a model that samples randomly."""

    def __init__(self, rng: random.Random, block_size: int, num_blocks: int, budget: int,
                 max_seqs: int) -> None:
        self.rng = rng
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.s = Scheduler(max_seqs, budget, num_blocks, block_size)
        self.seqs: list[Sequence] = []
        self.prompts: dict[int, list[int]] = {}
        self.aborted: set[int] = set()
        self.prefixes: list[list[int]] = []
        self.stats = {"preempt": 0, "chunk": 0, "prefix_hit": 0, "abort": 0}
        preempt = self.s.preempt

        def counting_preempt(seq: Sequence) -> None:
            self.stats["preempt"] += 1
            preempt(seq)

        self.s.preempt = counting_preempt  # type: ignore[method-assign]

    def add(self) -> None:
        rng = self.rng
        capacity = self.num_blocks * self.block_size
        max_tokens = rng.randint(1, 8)
        # As LLMEngine.add_request: prompt + max_tokens - 1 positions must fit the pool.
        max_prompt = min(4 * self.block_size + 3, capacity + 1 - max_tokens)
        n = rng.randint(1, max_prompt)
        if self.prefixes and rng.random() < 0.6:  # reuse a prefix: prefix-cache hits
            prefix = rng.choice(self.prefixes)[:n]
            prompt = prefix + [rng.randrange(1, VOCAB) for _ in range(n - len(prefix))]
        else:
            prompt = [rng.randrange(1, VOCAB) for _ in range(n)]
        self.prefixes.append(prompt)
        seq = Sequence(prompt, SamplingParams(temperature=1.0, max_tokens=max_tokens,
                                              ignore_eos=rng.random() < 0.3))
        self.prompts[seq.seq_id] = list(prompt)
        self.seqs.append(seq)
        self.s.add(seq)

    def abort(self) -> None:
        live = [q for q in self.seqs if not q.is_finished]
        if live:
            seq = self.rng.choice(live)
            self.s.abort(seq)
            self.aborted.add(seq.seq_id)
            self.stats["abort"] += 1

    def step(self) -> None:
        if self.s.is_finished():
            return
        seqs, is_prefill = self.s.schedule()
        assert seqs, "a non-empty scheduler must schedule something"
        assert len(seqs) <= self.s.max_num_seqs
        assert len({q.seq_id for q in seqs}) == len(seqs), "seq scheduled twice"
        assert not any(q.is_finished for q in seqs), "finished seq scheduled"
        if is_prefill:
            assert sum(q.num_scheduled_tokens for q in seqs) <= self.s.max_num_batched_tokens
            for q in seqs:
                assert len(q.block_table) == q.num_blocks
                assert q.num_cached_tokens + q.num_scheduled_tokens <= q.num_tokens
                self.stats["prefix_hit"] += q.num_cached_tokens > 0
                self.stats["chunk"] += q.num_cached_tokens + q.num_scheduled_tokens < q.num_tokens
        else:
            for q in seqs:
                assert q.num_scheduled_tokens == 1
                assert len(q.block_table) == q.num_blocks, "decode token has no KV slot"
        before = [q.num_tokens for q in seqs]
        tokens = [EOS if self.rng.random() < 0.1 else self.rng.randrange(1, VOCAB) for _ in seqs]
        self.s.postprocess(seqs, tokens)
        for q, n in zip(seqs, before, strict=True):
            if q.num_tokens == n:  # chunked prefill step: nothing sampled
                continue
            p = q.sampling_params
            if q.num_completion_tokens >= p.max_tokens or (
                    not p.ignore_eos and q.last_token == EOS):
                self.s.finish(q)


CASES = [
    # (block_size, num_blocks, budget, max_seqs)
    (4, 64, 64, 8),  # roomy
    (4, 6, 64, 8),  # tight memory: preemption
    (4, 64, 5, 8),  # tight token budget: chunked prefill
    (4, 8, 7, 3),  # everything tight
    (1, 12, 6, 4),  # block_size 1 (regression: decode never allocated)
    (2, 10, 3, 2),
    (8, 4, 9, 16),
]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(("block_size", "num_blocks", "budget", "max_seqs"), CASES)
def test_random_operations_keep_invariants_and_finish(
        monkeypatch: pytest.MonkeyPatch, check_invariants: Callable[..., None], seed: int,
        block_size: int, num_blocks: int, budget: int, max_seqs: int,
) -> None:
    monkeypatch.setattr(Sequence, "block_size", block_size)
    rng = random.Random(seed)
    eng = FakeEngine(rng, block_size, num_blocks, budget, max_seqs)

    for _ in range(150):
        r = rng.random()
        if r < 0.25:
            eng.add()
        elif r < 0.3:
            eng.abort()
        else:
            eng.step()
        check_invariants(eng.s, eng.seqs)
        for q in eng.seqs:  # generation only appends; the prompt never changes
            assert q.token_ids[:q.num_prompt_tokens] == eng.prompts[q.seq_id]

    # Liveness: with no new requests, everything finishes in a bounded number of steps.
    for _ in range(2000):
        if eng.s.is_finished():
            break
        eng.step()
        check_invariants(eng.s, eng.seqs)
    assert eng.s.is_finished()
    for q in eng.seqs:
        assert q.status == SequenceStatus.FINISHED
        if q.seq_id not in eng.aborted:
            p = q.sampling_params
            assert 1 <= q.num_completion_tokens <= p.max_tokens
            if q.num_completion_tokens < p.max_tokens:
                assert not p.ignore_eos and q.last_token == EOS
    bm = eng.s.block_manager
    assert len(bm.free_block_ids) == num_blocks and not bm.used_block_ids


def test_random_cases_exercise_the_hard_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against the randomized test going vacuous: across the seeds, preemption,
    chunked prefill, prefix-cache hits and aborts all actually happen."""
    totals = {"preempt": 0, "chunk": 0, "prefix_hit": 0, "abort": 0}
    for block_size, num_blocks, budget, max_seqs in CASES:
        monkeypatch.setattr(Sequence, "block_size", block_size)
        for seed in range(8):
            rng = random.Random(seed)
            eng = FakeEngine(rng, block_size, num_blocks, budget, max_seqs)
            for _ in range(150):
                r = rng.random()
                if r < 0.25:
                    eng.add()
                elif r < 0.3:
                    eng.abort()
                else:
                    eng.step()
            for k, v in eng.stats.items():
                totals[k] += v
    assert all(v >= 20 for v in totals.values()), totals


# ------------------------------------------------------------- BlockManager alone


@pytest.mark.parametrize("seed", range(20))
def test_block_manager_random_alloc_free_append(
        monkeypatch: pytest.MonkeyPatch, seed: int,
) -> None:
    """Random allocate / decode-append / hash / free directly on a BlockManager.

    Prompts come from a few shared prefixes. After every operation: free and used
    partition the pool, each block's ref_cnt equals the number of live tables that
    hold it, and a prefix hit only ever returns blocks caching the same tokens.
    """
    bs = 4
    monkeypatch.setattr(Sequence, "block_size", bs)
    rng = random.Random(seed)
    bm = BlockManager(num_blocks=12, block_size=bs)
    bases = [[rng.randrange(VOCAB) for _ in range(16)] for _ in range(3)]
    live: list[Sequence] = []

    def check() -> None:
        refs = [0] * bm.num_blocks
        for q in live:
            for b in q.block_table:
                refs[b] += 1
        free = list(bm.free_block_ids)
        assert len(set(free)) == len(free)
        assert set(free) | bm.used_block_ids == set(range(bm.num_blocks))
        assert set(free).isdisjoint(bm.used_block_ids)
        for b, block in enumerate(bm.blocks):
            assert block.ref_cnt == refs[b]
        for q in live:
            for i, b in enumerate(q.block_table):
                if bm.blocks[b].hash != -1:
                    assert bm.blocks[b].token_ids == q.block(i)

    for _ in range(300):
        r = rng.random()
        if r < 0.4:
            base = rng.choice(bases)
            n = rng.randint(1, len(base))
            seq = Sequence(base[:n] + [rng.randrange(VOCAB) for _ in range(rng.randint(0, 3))])
            cached = bm.can_allocate(seq)
            if cached != -1:
                bm.allocate(seq, cached)
                assert seq.num_cached_tokens == cached * bs < seq.num_tokens
                # What the prefill step computes, then registers.
                seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
                bm.hash_blocks(seq)
                seq.num_cached_tokens, seq.num_scheduled_tokens = seq.num_tokens, 0
                live.append(seq)
        elif r < 0.75 and live:
            seq = rng.choice(live)  # one decode step: append a token, give it a slot
            seq.append_token(rng.randrange(VOCAB))  # sampled; has no KV yet
            if bm.can_append(seq):
                bm.may_append(seq)
                assert len(seq.block_table) == seq.num_blocks
                seq.num_scheduled_tokens = 1
                bm.hash_blocks(seq)
                seq.num_cached_tokens, seq.num_scheduled_tokens = seq.num_tokens, 0
            else:
                live.remove(seq)
                bm.deallocate(seq)
        elif live:
            seq = live.pop(rng.randrange(len(live)))
            bm.deallocate(seq)
            assert not seq.block_table
        check()
    for seq in live:
        bm.deallocate(seq)
    live.clear()
    check()
    assert len(bm.free_block_ids) == bm.num_blocks
