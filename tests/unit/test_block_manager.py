"""Tests for KV-cache block bookkeeping in BlockManager."""

import pytest

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.sequence import Sequence

BS = 4  # small block size so tests stay readable


@pytest.fixture(autouse=True)
def _small_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sequence and BlockManager must agree on block_size.
    monkeypatch.setattr(Sequence, "block_size", BS)


def make_seq(n: int, start: int = 0) -> Sequence:
    return Sequence(list(range(start, start + n)))


def register_full_blocks(bm: BlockManager, seq: Sequence) -> None:
    """Hash every full block of seq and register it, as hash_blocks will do."""
    h = -1
    for i, block_id in enumerate(seq.block_table):
        token_ids = seq.block(i)
        if len(token_ids) < BS:
            break
        h = bm.compute_hash(token_ids, h)
        bm.blocks[block_id].update(h, token_ids)
        bm.hash_to_block_id[h] = block_id


def assert_invariants(bm: BlockManager) -> None:
    free = set(bm.free_block_ids)
    assert len(free) == len(bm.free_block_ids), "duplicate id in free list"
    assert free.isdisjoint(bm.used_block_ids)
    assert free | bm.used_block_ids == set(range(bm.num_blocks))
    for b in bm.blocks:
        if b.block_id in bm.used_block_ids:
            assert b.ref_cnt > 0
        else:
            assert b.ref_cnt == 0


def prefill(bm: BlockManager, seq: Sequence) -> int:
    n = bm.can_allocate(seq)
    assert n != -1, "expected enough free blocks"
    bm.allocate(seq, n)
    return n


# ---------------------------------------------------------------- compute_hash


def test_compute_hash_is_deterministic() -> None:
    assert BlockManager.compute_hash([1, 2, 3, 4]) == BlockManager.compute_hash([1, 2, 3, 4])


def test_compute_hash_depends_on_tokens() -> None:
    assert BlockManager.compute_hash([1, 2, 3, 4]) != BlockManager.compute_hash([1, 2, 3, 5])


def test_compute_hash_depends_on_prefix() -> None:
    a = BlockManager.compute_hash([1, 2, 3, 4])
    with_a = BlockManager.compute_hash([5, 6, 7, 8], a)
    with_other = BlockManager.compute_hash([5, 6, 7, 8], a + 1)
    no_prefix = BlockManager.compute_hash([5, 6, 7, 8])
    assert len({with_a, with_other, no_prefix}) == 3


def test_compute_hash_returns_non_negative_int() -> None:
    h = BlockManager.compute_hash([1, 2, 3, 4])
    assert isinstance(h, int) and h >= 0


# ------------------------------------------------- _allocate / _deallocate_block


def test_allocate_block_takes_head_of_free_list() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    block_id = bm._allocate_block()
    assert block_id == 0
    assert list(bm.free_block_ids) == [1, 2]
    assert bm.used_block_ids == {0}
    assert bm.blocks[0].ref_cnt == 1
    assert_invariants(bm)


def test_allocate_block_clears_its_stale_hash() -> None:
    bm = BlockManager(num_blocks=2, block_size=BS)
    h = bm.compute_hash([1, 2, 3, 4])
    bm.blocks[0].update(h, [1, 2, 3, 4])
    bm.hash_to_block_id[h] = 0

    bm._allocate_block()

    assert bm.blocks[0].hash == -1
    assert bm.blocks[0].token_ids == []
    assert h not in bm.hash_to_block_id


def test_allocate_block_keeps_hash_owned_by_other_block() -> None:
    bm = BlockManager(num_blocks=2, block_size=BS)
    h = bm.compute_hash([1, 2, 3, 4])
    bm.blocks[0].update(h, [1, 2, 3, 4])
    bm.hash_to_block_id[h] = 1  # mapping now points elsewhere

    bm._allocate_block()

    assert bm.hash_to_block_id[h] == 1


def test_deallocate_block_appends_to_tail_and_keeps_hash() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    block_id = bm._allocate_block()
    h = bm.compute_hash([1, 2, 3, 4])
    bm.blocks[block_id].update(h, [1, 2, 3, 4])
    bm.hash_to_block_id[h] = block_id
    bm.blocks[block_id].ref_cnt = 0

    bm._deallocate_block(block_id)

    assert list(bm.free_block_ids) == [1, 2, 0]
    assert bm.blocks[block_id].hash == h
    assert bm.hash_to_block_id[h] == block_id
    assert_invariants(bm)


def test_deallocate_block_rejects_referenced_block() -> None:
    bm = BlockManager(num_blocks=2, block_size=BS)
    block_id = bm._allocate_block()  # ref_cnt == 1
    with pytest.raises(AssertionError):
        bm._deallocate_block(block_id)


# ----------------------------------------------------------------- can_allocate


@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "ok"),
    [
        (5, 3, True),   # needs 2, has 3
        (5, 2, True),   # needs 2, has exactly 2
        (5, 1, False),  # needs 2, has 1
        (8, 2, True),   # exactly 2 full blocks
        (9, 2, False),  # needs 3
    ],
)
def test_can_allocate_capacity(num_tokens: int, num_blocks: int, ok: bool) -> None:
    bm = BlockManager(num_blocks=num_blocks, block_size=BS)
    result = bm.can_allocate(make_seq(num_tokens))
    assert (result != -1) is ok
    if ok:
        assert result == 0  # nothing cached yet


def test_can_allocate_does_not_mutate_state() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    seq = make_seq(6)
    bm.can_allocate(seq)
    assert list(bm.free_block_ids) == [0, 1, 2, 3]
    assert bm.used_block_ids == set()
    assert seq.block_table == []


def test_can_allocate_counts_shared_prefix() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(10)  # blocks: [0-3] [4-7] [8,9]
    prefill(bm, first)
    register_full_blocks(bm, first)

    second = make_seq(10)
    assert bm.can_allocate(second) == 2


def test_can_allocate_never_caches_last_block() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(8)  # two full blocks, both registered
    prefill(bm, first)
    register_full_blocks(bm, first)

    # Same 8 tokens: only block 0 may count, the last block must be recomputed.
    assert bm.can_allocate(make_seq(8)) == 1


def test_can_allocate_stops_at_first_miss() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(12)
    prefill(bm, first)
    register_full_blocks(bm, first)

    # Block 0 differs, so block 1 must not count even though it matches.
    tokens = list(range(12))
    tokens[0] = 999
    assert bm.can_allocate(Sequence(tokens)) == 0


def test_can_allocate_rejects_hash_hit_with_different_tokens() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(8)
    prefill(bm, first)
    register_full_blocks(bm, first)
    bm.blocks[first.block_table[0]].token_ids = [7, 7, 7, 7]  # fake a collision

    assert bm.can_allocate(make_seq(8)) == 0


def test_can_allocate_shared_used_block_needs_no_free_block() -> None:
    bm = BlockManager(num_blocks=3, block_size=BS)
    first = make_seq(5)  # uses 2 blocks, block 0 full
    prefill(bm, first)
    register_full_blocks(bm, first)
    assert len(bm.free_block_ids) == 1

    # Shares block 0 (in use), needs only 1 new block.
    assert bm.can_allocate(make_seq(5)) == 1


def test_can_allocate_free_cached_block_still_needs_free_slot() -> None:
    bm = BlockManager(num_blocks=2, block_size=BS)
    first = make_seq(5)
    prefill(bm, first)
    register_full_blocks(bm, first)
    bm.deallocate(first)

    # Block 0 is free but cached: reviving it still consumes a free slot.
    assert bm.can_allocate(make_seq(5)) == 1
    assert bm.can_allocate(make_seq(9)) == -1


# --------------------------------------------------------------------- allocate


def test_allocate_fresh_sequence() -> None:
    bm = BlockManager(num_blocks=5, block_size=BS)
    seq = make_seq(9)
    prefill(bm, seq)

    assert seq.block_table == [0, 1, 2]
    assert seq.num_cached_tokens == 0
    assert all(bm.blocks[b].ref_cnt == 1 for b in seq.block_table)
    assert list(bm.free_block_ids) == [3, 4]
    assert_invariants(bm)


def test_allocate_shares_used_prefix_block() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(10)
    prefill(bm, first)
    register_full_blocks(bm, first)

    second = make_seq(10)
    prefill(bm, second)

    assert second.block_table[:2] == first.block_table[:2]
    assert second.block_table[2] != first.block_table[2]
    assert second.num_cached_tokens == 2 * BS
    assert bm.blocks[first.block_table[0]].ref_cnt == 2
    assert bm.blocks[first.block_table[1]].ref_cnt == 2
    assert bm.blocks[first.block_table[2]].ref_cnt == 1
    assert_invariants(bm)


def test_allocate_revives_freed_cached_block() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    first = make_seq(5)
    prefill(bm, first)
    register_full_blocks(bm, first)
    cached_id = first.block_table[0]
    bm.deallocate(first)

    second = make_seq(5)
    prefill(bm, second)

    assert second.block_table[0] == cached_id
    assert second.num_cached_tokens == BS
    assert bm.blocks[cached_id].ref_cnt == 1
    assert cached_id in bm.used_block_ids
    assert cached_id not in bm.free_block_ids
    assert_invariants(bm)


# ------------------------------------------------------------------- deallocate


def test_deallocate_frees_all_blocks_and_clears_seq() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    seq = make_seq(9)
    prefill(bm, seq)

    bm.deallocate(seq)

    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    assert bm.used_block_ids == set()
    # Freed in reverse order, appended to the tail.
    assert list(bm.free_block_ids) == [3, 2, 1, 0]
    assert_invariants(bm)


def test_deallocate_keeps_block_shared_by_other_seq() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(10)
    prefill(bm, first)
    register_full_blocks(bm, first)
    second = make_seq(10)
    prefill(bm, second)
    private_id = first.block_table[2]

    bm.deallocate(first)

    for b in second.block_table:
        assert b in bm.used_block_ids
        assert bm.blocks[b].ref_cnt == 1
    assert private_id not in bm.used_block_ids  # private block freed
    assert_invariants(bm)


def test_full_round_trip_returns_to_empty() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    seqs = [make_seq(10), make_seq(10), make_seq(7, start=100)]
    for s in seqs:
        prefill(bm, s)
        register_full_blocks(bm, s)
    for s in seqs:
        bm.deallocate(s)

    assert bm.used_block_ids == set()
    assert len(bm.free_block_ids) == bm.num_blocks
    assert_invariants(bm)


# ------------------------------------------------------ can_append / may_append


@pytest.mark.parametrize(
    ("num_tokens", "needs_block"),
    [
        (4, False),  # last block exactly full, no token appended yet
        (5, True),   # 5th token opens block 1
        (6, False),
        (8, False),
        (9, True),
    ],
)
def test_can_append_with_no_free_blocks(num_tokens: int, needs_block: bool) -> None:
    # A decode seq: its first num_tokens - 1 tokens have blocks, the last was just
    # sampled and is about to run.
    bm = BlockManager(num_blocks=8, block_size=BS)
    seq = make_seq(num_tokens - 1)
    prefill(bm, seq)
    seq.append_token(99)
    while bm.free_block_ids:  # exhaust the pool
        bm._allocate_block()
    assert bm.can_append(seq) is not needs_block


def test_block_size_one_allocates_a_block_every_decode_step() -> None:
    # Regression: the old rule ``len(seq) % block_size == 1`` is never true for
    # block_size 1, so decode never allocated and slot_mapping ran off the table.
    bm = BlockManager(num_blocks=8, block_size=1)
    seq = make_seq(3)
    seq.block_size = 1  # instance attribute: this seq only
    prefill(bm, seq)
    for token in range(3):
        seq.append_token(token)
        assert bm.can_append(seq)
        bm.may_append(seq)
        assert len(seq.block_table) == seq.num_tokens
        assert_invariants(bm)


def test_can_append_with_free_block() -> None:
    bm = BlockManager(num_blocks=1, block_size=BS)
    assert bm.can_append(make_seq(5))


def test_decode_grows_block_table_at_boundaries() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    seq = make_seq(3)
    prefill(bm, seq)
    assert len(seq.block_table) == 1

    for token in range(100, 110):  # grow from 3 to 13 tokens
        seq.append_token(token)
        assert bm.can_append(seq)
        bm.may_append(seq)
        assert len(seq.block_table) == seq.num_blocks
        assert_invariants(bm)

    assert len(seq.block_table) == 4  # 13 tokens -> 4 blocks


def test_may_append_is_noop_inside_block() -> None:
    bm = BlockManager(num_blocks=4, block_size=BS)
    seq = make_seq(5)
    prefill(bm, seq)
    seq.append_token(42)  # len 6, still in block 1

    bm.may_append(seq)

    assert len(seq.block_table) == 2
    assert list(bm.free_block_ids) == [2, 3]


# ------------------------------------------------------------------ hash_blocks


def test_hash_blocks_after_prefill_matches_chained_hash() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    seq = make_seq(10)
    prefill(bm, seq)
    seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens

    bm.hash_blocks(seq)

    h0 = bm.compute_hash(seq.block(0))
    h1 = bm.compute_hash(seq.block(1), h0)
    assert bm.blocks[seq.block_table[0]].hash == h0
    assert bm.blocks[seq.block_table[1]].hash == h1
    assert bm.blocks[seq.block_table[2]].hash == -1  # partial block not hashed
    assert bm.hash_to_block_id[h0] == seq.block_table[0]
    assert bm.hash_to_block_id[h1] == seq.block_table[1]
    assert bm.blocks[seq.block_table[1]].token_ids == seq.block(1)


def test_hash_blocks_enables_prefix_hit() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    first = make_seq(10)
    prefill(bm, first)
    first.num_scheduled_tokens = len(first)
    bm.hash_blocks(first)

    assert bm.can_allocate(make_seq(10)) == 2


def test_hash_blocks_noop_when_no_block_completed() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    seq = make_seq(5)
    prefill(bm, seq)
    seq.num_cached_tokens = 5
    seq.append_token(99)  # len 6, block 1 still partial
    seq.num_scheduled_tokens = 1

    bm.hash_blocks(seq)

    assert bm.blocks[seq.block_table[1]].hash == -1
    assert bm.hash_to_block_id == {}


def test_hash_blocks_during_decode_chains_from_previous_block() -> None:
    bm = BlockManager(num_blocks=10, block_size=BS)
    seq = make_seq(6)
    prefill(bm, seq)
    seq.num_scheduled_tokens = len(seq)
    bm.hash_blocks(seq)  # hashes block 0
    seq.num_cached_tokens, seq.num_scheduled_tokens = len(seq), 0

    for token in (100, 101):  # fill block 1 (len 8)
        seq.append_token(token)
        bm.may_append(seq)
        seq.num_scheduled_tokens = 1
        bm.hash_blocks(seq)
        seq.num_cached_tokens += 1
        seq.num_scheduled_tokens = 0

    h0 = bm.blocks[seq.block_table[0]].hash
    expected = bm.compute_hash(seq.block(1), h0)
    assert bm.blocks[seq.block_table[1]].hash == expected
    assert bm.hash_to_block_id[expected] == seq.block_table[1]
