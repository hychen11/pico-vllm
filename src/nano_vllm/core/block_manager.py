"""Paged KV-cache block bookkeeping.

The actual KV cache is NOT stored here. It is one large GPU tensor allocated
by the model runner, roughly shaped like:

    [2 (K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]

and it holds the K/V vectors of every cached token at every layer.

This module only does the bookkeeping for that tensor, like a page table:

- ``Block`` is the metadata for one physical slot ``kv_cache[:, :, block_id]``.
- ``BlockManager`` tracks which blocks are free, used, or shared, and fills in
  each sequence's ``block_table`` (logical block index -> physical block_id).
- ``Sequence`` owns the token ids.

At forward time, token ``i`` of a sequence is written to slot
``block_table[i // block_size] * block_size + i % block_size`` (slot_mapping),
and paged attention reads the history through ``block_tables``.
"""

from collections import deque

import numpy as np
import xxhash

from nano_vllm.sequence import Sequence


class Block:
    """Metadata for one physical KV-cache block (no K/V data lives here).

    Attributes:
        block_id: Index into the ``num_blocks`` dim of the KV-cache tensor.
        ref_cnt: Number of sequences currently using this block. It can be > 1
            when a prefix is shared. The block is freed when it drops to 0.
        hash: Content hash of a full block, used for prefix caching.
            -1 means the block is not hashed (not full or not cached yet).
        token_ids: Copy of the tokens in this block. Used only to verify a
            prefix-cache hit against hash collisions. ``Sequence`` owns the
            real token ids.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Called when the block is allocated to a new owner."""
        self.ref_cnt = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Allocates, frees, and shares KV-cache blocks across sequences."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # A freed block keeps its hash, so a later prefix hit may need to pull
        # it out of the middle of this deque with remove(), which is O(n).
        # The free list holds both never-hashed blocks and freed blocks that
        # keep their hash for reuse.
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, 'little'))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_cnt == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_cnt == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # Skip the last block: it is either partial (never hashed) or, if full,
        # must still be recomputed. KV cache stores only K/V, not hidden states,
        # so the last token must run through the model to produce logits for
        # sampling. Excluding it guarantees at least one token is computed.
        for i in range(num_new_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # not hit cache, and need to allocate new block
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # if block id in used_block_ids, just ref+=1
        # if block id hit in free_block_ids, need move from free_block_ids
        if num_new_blocks > len(self.free_block_ids):
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_cnt += 1
            else:
                block.ref_cnt = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        for _ in range(num_cached_blocks, seq.num_blocks):
            block_id = self._allocate_block()
            seq.block_table.append(block_id)
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # only append on first token
        return len(self.free_block_ids) >= (len(seq)%self.block_size==1)

    def may_append(self, seq: Sequence) -> None:
        if len(seq)%self.block_size==1:
            block_id = self._allocate_block()
            seq.block_table.append(block_id)

    def hash_blocks(self, seq: Sequence) -> None:
        """
        After each model step, register the hashes of blocks that became
        full in this step, so later requests with the same prefix can find
        and reuse them.
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


"""
prefill check:can_allocate op:allocate
decode  check:can_append  op:may_append
"""