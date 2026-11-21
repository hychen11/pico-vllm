"""Tests for Scheduler: queues, token budget, chunked prefill, decode, preemption."""

import pytest

from pico_vllm.core.scheduler import Scheduler
from pico_vllm.sequence import Sequence, SequenceStatus

BS = 4  # small block size so tests stay readable
TOKEN = 7  # fake sampled token returned by the "model"


@pytest.fixture(autouse=True)
def _small_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Sequence, "block_size", BS)


def make_scheduler(num_blocks: int = 64, budget: int = 64, max_seqs: int = 8) -> Scheduler:
    return Scheduler(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=budget,
        num_blocks=num_blocks,
        block_size=BS,
    )


def make_seq(n: int, start: int = 0) -> Sequence:
    return Sequence(list(range(start, start + n)))


def step(s: Scheduler) -> tuple[list[Sequence], bool]:
    """One engine step with a fake model that samples TOKEN for every seq."""
    seqs, is_prefill = s.schedule()
    s.postprocess(seqs, [TOKEN] * len(seqs))
    return seqs, is_prefill


def num_generated(seq: Sequence) -> int:
    return seq.num_tokens - seq.num_prompt_tokens


# ------------------------------------------------------------ add / is_finished


def test_add_puts_seq_in_waiting() -> None:
    s = make_scheduler()
    assert s.is_finished()
    seq = make_seq(5)
    s.add(seq)
    assert list(s.waiting) == [seq]
    assert not s.running
    assert not s.is_finished()


# ---------------------------------------------------------------------- prefill


def test_prefill_moves_waiting_to_running() -> None:
    s = make_scheduler()
    a, b = make_seq(5), make_seq(3, start=100)
    s.add(a)
    s.add(b)

    seqs, is_prefill = s.schedule()

    assert is_prefill
    assert seqs == [a, b]
    assert not s.waiting
    assert list(s.running) == [a, b]
    for seq in seqs:
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_scheduled_tokens == len(seq)
        assert len(seq.block_table) == seq.num_blocks


def test_prefill_respects_token_budget() -> None:
    s = make_scheduler(budget=10)
    seqs_in = [make_seq(4, start=i * 100) for i in range(3)]
    for seq in seqs_in:
        s.add(seq)

    seqs, is_prefill = s.schedule()

    assert is_prefill
    assert sum(q.num_scheduled_tokens for q in seqs) <= 10
    # 4 + 4 fit; the third (4 > 2 remaining) is not the first seq, so no chunking.
    assert seqs == seqs_in[:2]
    assert list(s.waiting) == [seqs_in[2]]
    assert seqs_in[2].block_table == []  # failed admission leaves no state


def test_prefill_respects_max_num_seqs() -> None:
    s = make_scheduler(max_seqs=2)
    for i in range(3):
        s.add(make_seq(2, start=i * 100))

    seqs, _ = s.schedule()

    assert len(seqs) == 2
    assert len(s.waiting) == 1


def test_prefill_stops_when_out_of_blocks() -> None:
    s = make_scheduler(num_blocks=3)
    a, b = make_seq(8), make_seq(8, start=100)  # 2 blocks each
    s.add(a)
    s.add(b)

    seqs, _ = s.schedule()

    assert seqs == [a]
    assert list(s.waiting) == [b]
    assert b.block_table == []


def test_prefill_has_priority_over_decode() -> None:
    s = make_scheduler()
    a = make_seq(5)
    s.add(a)
    step(s)  # a prefilled, now running

    b = make_seq(3, start=100)
    s.add(b)
    seqs, is_prefill = s.schedule()

    assert is_prefill
    assert seqs == [b]  # a waits this step


# -------------------------------------------------------------- chunked prefill


def test_chunked_prefill_splits_long_prompt() -> None:
    s = make_scheduler(budget=6)
    seq = make_seq(10)
    s.add(seq)

    seqs, is_prefill = step(s)
    assert is_prefill and seqs == [seq]
    # First chunk: 6 tokens computed, seq still waiting, no token appended.
    assert seq.num_cached_tokens == 6
    assert seq.num_tokens == 10
    assert seq.status == SequenceStatus.WAITING
    assert list(s.waiting) == [seq]

    seqs, is_prefill = step(s)
    assert is_prefill and seqs == [seq]
    # Last chunk: prompt done, moved to running, first token appended.
    assert seq.num_cached_tokens == 10
    assert seq.num_tokens == 11
    assert seq.last_token == TOKEN
    assert seq.status == SequenceStatus.RUNNING
    assert list(s.running) == [seq]


def test_only_first_seq_may_be_chunked() -> None:
    s = make_scheduler(budget=6)
    long, short = make_seq(10), make_seq(3, start=100)
    s.add(long)
    s.add(short)

    seqs, _ = step(s)
    assert seqs == [long]  # 6 of 10

    seqs, _ = step(s)
    assert seqs == [long]  # remaining 4; short (3) does not fit in the last 2

    seqs, _ = step(s)
    assert seqs == [short]


# ---------------------------------------------------------------- postprocess


def test_postprocess_advances_progress() -> None:
    s = make_scheduler()
    seq = make_seq(5)
    s.add(seq)

    step(s)

    assert seq.num_cached_tokens == 5
    assert seq.num_scheduled_tokens == 0
    assert seq.num_tokens == 6


def test_postprocess_registers_prefix_cache() -> None:
    s = make_scheduler()
    a = make_seq(10)
    s.add(a)
    step(s)  # blocks 0 and 1 of a are full and hashed

    b = make_seq(10)
    s.add(b)
    seqs, _ = s.schedule()

    assert seqs == [b]
    assert b.block_table[:2] == a.block_table[:2]
    assert b.num_cached_tokens == 2 * BS
    assert b.num_scheduled_tokens == 10 - 2 * BS


# ----------------------------------------------------------------------- decode


def test_decode_schedules_one_token_per_running_seq() -> None:
    s = make_scheduler()
    a, b = make_seq(5), make_seq(6, start=100)
    s.add(a)
    s.add(b)
    step(s)

    seqs, is_prefill = s.schedule()

    assert not is_prefill
    assert seqs == [a, b]
    assert all(q.num_scheduled_tokens == 1 for q in seqs)
    assert list(s.running) == [a, b]  # order kept


def test_decode_respects_max_num_seqs_and_keeps_rest_running() -> None:
    s = make_scheduler(max_seqs=2)
    seqs_in = [make_seq(3, start=i * 100) for i in range(3)]
    for seq in seqs_in:
        s.add(seq)
    step(s)  # prefills 2 (max_seqs)
    step(s)  # prefills the 3rd

    seqs, is_prefill = s.schedule()

    assert not is_prefill
    assert seqs == seqs_in[:2]
    assert list(s.running) == seqs_in  # unscheduled seq is not dropped


def test_decode_allocates_block_at_boundary() -> None:
    s = make_scheduler()
    seq = make_seq(4)  # exactly one full block
    s.add(seq)
    step(s)  # appends 5th token
    assert len(seq.block_table) == 1

    s.schedule()

    assert len(seq.block_table) == 2


def test_decode_steps_generate_tokens() -> None:
    s = make_scheduler()
    seq = make_seq(3)
    s.add(seq)
    for _ in range(5):
        step(s)
    assert num_generated(seq) == 5
    assert len(seq.block_table) == seq.num_blocks


# ------------------------------------------------------------------- preemption


def test_preempt_resets_seq_and_puts_it_at_front_of_waiting() -> None:
    s = make_scheduler()
    a, b = make_seq(5), make_seq(5, start=100)
    s.add(a)
    step(s)
    s.add(b)  # b is waiting
    s.running.remove(a)  # callers pop the victim from running first

    s.preempt(a)

    assert list(s.waiting) == [a, b]
    assert a.status == SequenceStatus.WAITING
    assert a.block_table == []
    assert a.num_cached_tokens == 0


def test_decode_preempts_latest_running_seq() -> None:
    # 6 blocks, 5 seqs of one full block each -> 1 free block after prefill.
    s = make_scheduler(num_blocks=6)
    a, b, c, d, e = (make_seq(4, start=i * 100) for i in range(5))
    for seq in (a, b, c, d, e):
        s.add(seq)
    step(s)  # every seq now has 5 tokens and needs a new block

    seqs, is_prefill = s.schedule()

    assert not is_prefill
    # a takes the free block. b needs one -> preempt the newest (e).
    # c needs one -> preempt the newest left (d). Older seqs keep running.
    assert seqs == [a, b, c]
    assert list(s.running) == [a, b, c]
    # Each preempted seq goes to the front, so the last one preempted is first.
    assert list(s.waiting) == [d, e]
    assert d.block_table == [] and e.block_table == []


def test_decode_preempts_self_when_alone() -> None:
    s = make_scheduler(num_blocks=2)
    a, b = make_seq(4), make_seq(4, start=100)
    s.add(a)
    s.add(b)
    step(s)  # both full, 0 free blocks, both need a new block

    seqs, _ = s.schedule()

    # a pops, b is preempted to free a block; a gets it.
    assert seqs == [a]
    assert list(s.waiting) == [b]
    assert list(s.running) == [a]


# ----------------------------------------------------------------------- finish


def test_finish_releases_blocks_and_removes_from_running() -> None:
    s = make_scheduler(num_blocks=4)
    seq = make_seq(5)
    s.add(seq)
    step(s)

    s.finish(seq)

    assert seq.status == SequenceStatus.FINISHED
    assert seq not in s.running
    assert seq not in s.waiting
    assert seq.block_table == []
    assert len(s.block_manager.free_block_ids) == 4
    assert s.is_finished()


def test_finished_seq_is_never_rescheduled() -> None:
    s = make_scheduler()
    a, b = make_seq(5), make_seq(5, start=100)
    s.add(a)
    s.add(b)
    step(s)
    s.finish(a)

    for _ in range(5):
        seqs, _ = step(s)
        assert a not in seqs


# ------------------------------------------------------------------ end to end


@pytest.mark.parametrize(
    ("num_blocks", "budget", "max_seqs"),
    [
        (64, 64, 8),  # plenty of everything
        (64, 6, 8),  # tight token budget -> chunked prefill
        (6, 64, 8),  # tight memory -> preemption
        (64, 64, 2),  # small batch
    ],
)
def test_all_seqs_complete(num_blocks: int, budget: int, max_seqs: int) -> None:
    s = make_scheduler(num_blocks=num_blocks, budget=budget, max_seqs=max_seqs)
    gen = 6
    seqs_in = [make_seq(n, start=i * 100) for i, n in enumerate([3, 9, 5, 4])]
    for seq in seqs_in:
        s.add(seq)

    finished: set[int] = set()
    for _ in range(500):
        if s.is_finished():
            break
        seqs, is_prefill = s.schedule()
        assert len({q.seq_id for q in seqs}) == len(seqs), "seq scheduled twice"
        assert not finished & {q.seq_id for q in seqs}, "finished seq rescheduled"
        assert len(seqs) <= max_seqs
        if is_prefill:
            assert sum(q.num_scheduled_tokens for q in seqs) <= budget
        s.postprocess(seqs, [TOKEN] * len(seqs))
        for q in seqs:
            if q.status == SequenceStatus.RUNNING and num_generated(q) >= gen:
                s.finish(q)
                finished.add(q.seq_id)

        bm = s.block_manager
        assert set(bm.free_block_ids).isdisjoint(bm.used_block_ids)
        assert len(bm.free_block_ids) + len(bm.used_block_ids) == num_blocks

    assert s.is_finished()
    assert all(num_generated(q) == gen for q in seqs_in)
    assert all(q.status == SequenceStatus.FINISHED for q in seqs_in)
    assert len(s.block_manager.free_block_ids) == num_blocks
