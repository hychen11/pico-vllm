"""Shared pytest fixtures and deterministic test defaults."""

from __future__ import annotations

import random

import pytest


@pytest.fixture(scope="session", autouse=True)
def _single_rank_group():  # noqa: ANN202
    """One world_size=1 process group for the whole session (M8 step 1).

    The model's layers read the TP rank/size from torch.distributed, as in
    nano-vllm, and many tests build models directly instead of through
    ``LLMEngine.from_config``. gloo works for CPU and CUDA tensors here, since
    world_size=1 never communicates. Engines see the group and reuse it.
    """
    import torch.distributed as dist

    from pico_vllm.worker.process_group import free_port

    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{free_port()}",
                                world_size=1, rank=0)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _seed_python_random() -> None:
    """Keep tests deterministic before PyTorch-based tests are introduced."""
    random.seed(0)


def check_scheduler_invariants(scheduler, seqs) -> None:  # noqa: ANN001
    """Assert the bookkeeping invariants that must hold between engine steps.

    ``seqs``: every Sequence ever added to ``scheduler`` (finished/aborted included).
    Used by the randomized scheduler tests and the engine hardening tests (M7).
    """
    from pico_vllm.sequence import SequenceStatus

    bm = scheduler.block_manager
    waiting, running = list(scheduler.waiting), list(scheduler.running)
    waiting_ids, running_ids = {id(q) for q in waiting}, {id(q) for q in running}
    assert len(waiting_ids) == len(waiting), "seq twice in waiting"
    assert len(running_ids) == len(running), "seq twice in running"
    assert waiting_ids.isdisjoint(running_ids), "seq in both queues"

    # Block pool: free and used partition all blocks; ref_cnt says which is which.
    free = list(bm.free_block_ids)
    assert len(set(free)) == len(free), "duplicate id in free list"
    assert set(free).isdisjoint(bm.used_block_ids)
    assert set(free) | bm.used_block_ids == set(range(bm.num_blocks))

    refs = [0] * bm.num_blocks
    for q in seqs:
        if q.status == SequenceStatus.FINISHED:
            assert id(q) not in waiting_ids | running_ids, "finished seq still queued"
            assert not q.block_table, "finished seq still holds blocks"
            continue
        if q.status == SequenceStatus.WAITING:
            assert id(q) in waiting_ids, "WAITING seq not in waiting"
        else:
            assert q.status == SequenceStatus.RUNNING and id(q) in running_ids
        assert q.num_scheduled_tokens == 0, "scheduled tokens left after a step"
        assert len(set(q.block_table)) == len(q.block_table), "block twice in one table"
        for b in q.block_table:
            refs[b] += 1
        # Every hashed block a seq holds caches exactly that seq's tokens: what
        # makes a prefix-cache hit safe.
        for i, b in enumerate(q.block_table):
            if bm.blocks[b].hash != -1:
                assert bm.blocks[b].token_ids == q.block(i), "hashed block / tokens mismatch"
    for q in running:
        # Between steps a running seq has KV for all but its newest token.
        assert q.num_cached_tokens == q.num_tokens - 1
        assert len(q.block_table) == -(-(q.num_tokens - 1) // bm.block_size)
    for i, q in enumerate(waiting):
        if q.block_table:  # only the head can be halfway through a chunked prefill
            assert i == 0, "non-head waiting seq holds blocks"
            assert len(q.block_table) == q.num_blocks
            assert 0 < q.num_cached_tokens < q.num_tokens
        else:
            assert q.num_cached_tokens == 0

    for b, block in enumerate(bm.blocks):
        assert block.ref_cnt == refs[b], f"block {b}: ref_cnt {block.ref_cnt} != {refs[b]}"
    for h, b in bm.hash_to_block_id.items():
        assert bm.blocks[b].hash == h, "hash index points at a block with another hash"


@pytest.fixture
def check_invariants():  # noqa: ANN201
    return check_scheduler_invariants
