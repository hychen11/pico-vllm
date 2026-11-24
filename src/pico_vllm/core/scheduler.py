"""Decide which sequences run in each model step.

The scheduler only does bookkeeping: it moves sequences between queues,
reserves KV blocks through the BlockManager, and returns a batch. It never
calls model code.

Each step is either a prefill batch or a decode batch, never both:

- prefill: check ``can_allocate``, then ``allocate``
- decode:  check ``can_append``, then ``may_append``
"""

from collections import deque

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(
            self,
            max_num_seqs: int,
            max_num_batched_tokens: int,
            num_blocks: int,
            block_size: int,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.block_manager = BlockManager(num_blocks, block_size)
        # Both queues: left = old (served first), right = new.
        # waiting: add() appends new requests on the right; prefill takes from
        #   the left (popleft). preempt() puts seqs back on the left, so they
        #   resume before newer requests.
        self.waiting: deque[Sequence] = deque()
        # running: finished prefills are appended on the right; decode takes
        #   from the left (popleft). When out of blocks, the newest seq on the
        #   right is preempted (pop), so older seqs keep running.
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Return (scheduled seqs, is_prefill). Try prefill first, else decode.

        Must not be called when ``is_finished()``. Every seq must fit the KV pool on
        its own (``LLMEngine.add_request`` checks this); otherwise a seq that cannot
        run even alone makes the step empty, which fails an assertion.
        """
        seqs = self._schedule_prefill()
        if seqs:
            return seqs, True
        return self._schedule_decode(), False

    def _schedule_prefill(self) -> list[Sequence]:
        """Move waiting seqs to running within max_num_seqs and the token budget.

        A prompt longer than the remaining budget is chunked: schedule only
        part of it and set ``seq.num_scheduled_tokens`` accordingly.
        """
        num_batched_tokens = 0
        scheduled_seqs = []
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # Only the first seq in a step may be chunked.
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        """Schedule one token per running seq, preempting others when out of blocks."""
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # cannot append, continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # extendleft will expend one by one!
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence) -> None:
        """Free seq's blocks and put it back at the front of waiting."""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        # preempt append left, so can be fetch first!
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """Update seqs after a model step.

        Hash newly full blocks, advance ``num_cached_tokens`` and reset
        ``num_scheduled_tokens``. Append the sampled token only when the whole
        prompt has been prefilled (a chunked prefill step samples nothing).
        Deciding when a seq is done (EOS / max_tokens) is the engine's job (M5).
        """
        for seq, token_id in zip(seqs, token_ids, strict=True):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # continue because if prefill, return token is already known, dont need append!
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            #append_token new token，have no KV cache, only next step decode
            seq.append_token(token_id)

    def finish(self, seq: Sequence) -> None:
        """Mark seq FINISHED, release its blocks, and remove it from running."""
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)

    def abort(self, seq: Sequence) -> None:
        """Drop seq wherever it is: waiting (incl. halfway through a chunked prefill
        or after preemption) or running. Frees its blocks and marks it FINISHED.
        Full blocks it already computed stay hashed, so later prefix hits still work.
        A seq that is already finished is left alone."""
        if seq.is_finished:
            return
        if seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        else:
            self.running.remove(seq)
        seq.status = SequenceStatus.FINISHED
        seq.num_scheduled_tokens = 0
        self.block_manager.deallocate(seq)
