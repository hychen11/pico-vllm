"""Turn a scheduled batch of sequences into model inputs and run one step.

Each step computes, for every scheduled seq, the tokens
``token_ids[num_cached_tokens : num_cached_tokens + num_scheduled_tokens]``.
Decode is just the case ``num_scheduled_tokens == 1``, so prefill (including
prefix-cache hits and chunked prefill) and decode share one input layout.

CUDA Graph (M6 step 6): ``capture_cudagraphs`` records one decode step per batch
size in ``graph_batch_sizes``. A decode step that fits then copies its inputs into
fixed buffers and replays the smallest graph that holds it, instead of launching
every kernel from Python. See docs/concepts/decode-attention-and-cuda-graph.md.

Tensor parallelism (M8 step 5), as in nano-vllm: ``from_config(config, rank, ...)``
builds one rank's runner. Ranks > 0 run it in their own process and never return
from it: they stay in ``loop()``, running every ``call`` rank 0 broadcasts through
shared memory, until ``"exit"``.
"""

from __future__ import annotations

import math
import pickle
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from pico_vllm.core.paged_kv_cache import PagedKVCache
from pico_vllm.models.qwen3 import Qwen3ForCausalLM
from pico_vllm.sequence import Sequence
from pico_vllm.utils.context import Context, reset_context, set_context
from pico_vllm.worker.process_group import init_process_group

if TYPE_CHECKING:
    from pico_vllm.config import Config

# Room for one pickled ``(method, args)`` call; a decode step's args carry every
# scheduled seq's token ids. /dev/shm pages are only allocated when written.
SHM_BYTES = 2**24


def slot_mapping_for(block_table: list[int], block_size: int, start: int, end: int) -> list[int]:
    """Flat KV slots for token positions ``start..end-1`` of one sequence."""
    slot_mapping = []
    for pos in range(start, end):
        idx = pos // block_size
        slot_mapping.append(block_table[idx] * block_size + pos % block_size)
    return slot_mapping


class ModelRunner:
    def __init__(
            self, model: Qwen3ForCausalLM, kv_cache: PagedKVCache, block_size: int,
            use_flash: bool = False,
    ) -> None:
        self.model = model
        self.use_flash = use_flash
        self.kv_cache = kv_cache
        self.block_size = block_size
        # Filled by capture_cudagraphs (called by LLMEngine); empty = always eager.
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}  # batch size -> graph
        self.graph_batch_sizes: list[int] = []  # ascending
        self.graph_vars: dict[str, torch.Tensor] = {}  # fixed buffers the graphs read
        # TP (set by from_config): rank, broadcast channel, and whether exit()
        # destroys the process group.
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.shm: SharedMemory | None = None
        self.events: list[Any] = []  # rank 0: one per worker
        self.event: Any = None  # ranks > 0: their own
        self.owns_group = False

    @classmethod
    def from_config(
            cls, config: Config, rank: int = 0, event: Any = None, port: int | None = None,
    ) -> ModelRunner:
        """Build rank ``rank``'s runner for the checkpoint ``config.model``.

        Rank 0 (in the engine's process): ``event`` is the list of the workers'
        Events; returns the runner. Rank > 0 (the target of a spawned process):
        ``event`` is its own Event; never returns until rank 0 calls ``"exit"``.
        ``port``: where all ranks meet (TP>1); ``None`` picks one (TP=1).

        Every rank runs the same collectives in the same order: KV sizing runs the
        model forward (an all_reduce per layer), then the ranks agree on num_blocks.
        """
        # Lazy: llm_engine imports this module.
        from pico_vllm.engine.llm_engine import build_model, num_kv_blocks

        tp = config.tensor_parallel_size
        if rank > 0:
            config.device = torch.device("cuda", rank)  # this process's pickled copy
            torch.cuda.set_device(config.device)
            # spawn re-imports every module: class attributes are back to defaults.
            Sequence.block_size = config.kvcache_block_size
        owns_group = init_process_group(tp, config.device, port=port, rank=rank)

        model = build_model(config)
        num_blocks = num_kv_blocks(config, model)
        if tp > 1:
            # One block table for all ranks: none may use more blocks than the
            # smallest cache holds (the 4060's, here).
            t = torch.tensor([num_blocks], device=config.device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            num_blocks = int(t.item())
        hf = config.hf_config
        kv_cache = PagedKVCache(  # this rank's KV heads only
            hf.num_hidden_layers, num_blocks, config.kvcache_block_size,
            hf.num_key_value_heads // tp, hf.head_dim,
            dtype=config.dtype, device=config.device,
        )
        # Config.__post_init__ already resolved use_flash to a bool.
        self = cls(model, kv_cache, config.kvcache_block_size,
                   use_flash=bool(config.use_flash))
        self.owns_group = owns_group
        # After the KV cache (as nano-vllm): graphs record its addresses, and their
        # memory pool comes out of the headroom gpu_memory_utilization leaves.
        # (TP>1 forces enforce_eager.)
        if config.use_flash and kv_cache.use_triton and not config.enforce_eager:
            self.capture_cudagraphs(config.max_num_seqs, config.max_model_len)

        if tp > 1:
            name = f"pico_vllm_{port}"  # the port is unique to this engine
            if rank == 0:
                self.shm = SharedMemory(name=name, create=True, size=SHM_BYTES)
                self.events = event
                dist.barrier()  # the workers may attach now
            else:
                dist.barrier()
                # track=False: rank 0 created it and unlinks it.
                self.shm = SharedMemory(name=name, track=False)
                self.event = event
                self.loop()
        return self

    def write_shm(self, method: str, *args: Any) -> None:
        """Rank 0: 4-byte length + ``pickle.dumps((method, args))``, then wake all."""
        data = pickle.dumps((method, args))
        if 4 + len(data) > self.shm.size:
            raise RuntimeError(f"call {method!r} pickles to {len(data)} bytes; "
                               f"the shared memory holds {self.shm.size - 4}")
        self.shm.buf[:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4:4 + len(data)] = data
        for event in self.events:
            event.set()

    def read_shm(self) -> tuple[str, tuple]:
        """Ranks > 0: wait for rank 0's next call and return it."""
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], "little")
        method, args = pickle.loads(self.shm.buf[4:4 + n])
        self.event.clear()
        return method, args

    def loop(self) -> None:
        """Ranks > 0: run rank 0's calls until ``"exit"``."""
        while True:
            method, args = self.read_shm()
            self.call(method, *args)
            if method == "exit":
                break

    def call(self, method: str, *args: Any) -> Any:
        """``self.<method>(*args)`` on every rank; rank 0 broadcasts it first.

        Rank 0 may only overwrite the buffer once every worker has read it. ``run``
        guarantees that (its all_reduces wait for every rank), and so does
        ``exit`` (a barrier), the only calls the engine makes.
        """
        if self.rank == 0 and self.shm is not None:  # TP>1, workers not exited yet
            self.write_shm(method, *args)
        return getattr(self, method)(*args)

    def exit(self) -> None:
        """Release the shared memory and, if this runner created it, the group.
        Every rank runs it (via ``call("exit")``), so the collectives here match."""
        if self.shm is not None:
            self.shm.close()
            dist.barrier()  # every rank has closed it before rank 0 unlinks
            if self.rank == 0:
                self.shm.unlink()
            self.shm = None
        if self.owns_group and dist.is_initialized():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.destroy_process_group()
            self.owns_group = False

    def prepare_inputs(
            self, seqs: list[Sequence], is_prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Context]:
        """Build flat ``input_ids`` and ``positions`` (``[num_tokens]``) and the step Context."""
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        cu_seqlens_q: list[int] = [0]
        cu_seqlens_k: list[int] = [0]
        max_seqlen_q = 0
        context_lens: list[int] = []
        block_tables: list[list[int]] = []
        # Prefill and decode share this loop; decode is num_scheduled_tokens == 1.
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(slot_mapping_for(seq.block_table, self.block_size, start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + (end - start))
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, end - start)
            context_lens.append(end)  # cached + this step: all KV attention must read
            block_tables.append(seq.block_table)

        device = self.kv_cache.k_cache.device

        def as_tensor(xs: list, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.tensor(xs, dtype=dtype, device=device)

        # FA2 wants a rectangular int32 block table; pad rows with -1 (never read).
        max_num_blocks = max((len(bt) for bt in block_tables), default=0)
        block_table = [bt + [-1] * (max_num_blocks - len(bt)) for bt in block_tables]

        context = Context(
            # shared
            is_prefill=is_prefill,
            use_flash=self.use_flash,
            kv_cache=self.kv_cache,
            slot_mapping=as_tensor(slot_mapping),
            cu_seqlens_q=as_tensor(cu_seqlens_q, torch.int32),
            # eager
            context_lens=context_lens,
            block_tables=block_tables,
            # FA2
            block_table=as_tensor(block_table, torch.int32),
            cu_seqlens_k=as_tensor(cu_seqlens_k, torch.int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max(context_lens, default=0),
            cache_seqlens=as_tensor(context_lens, torch.int32),
        )
        return as_tensor(input_ids), as_tensor(positions), context

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor | None:
        """One model step. Returns logits ``[num_seqs, vocab]`` at each seq's last query
        token on rank 0, ``None`` on the other TP ranks (the LM head gathers to rank 0).

        Decode steps replay a captured graph (``_replay``) when there are any. Every
        decode step fits one: the scheduler runs at most ``max_num_seqs`` seqs and
        ``LLMEngine.add_request`` keeps every seq within ``max_model_len``.
        Otherwise: ``set_context`` → model forward → ``reset_context`` (always).
        """
        input_ids, positions, context = self.prepare_inputs(seqs, is_prefill)
        if is_prefill or not self.graphs:  # no graphs: never captured, i.e. eager
            set_context(context)
            try:
                hidden_states = self.model(input_ids, positions)
            finally:
                reset_context()
            last = context.cu_seqlens_q[1:] - 1
            return self.model.compute_logits(hidden_states[last])
        return self.model.compute_logits(self._replay(input_ids, positions, context))

    @torch.inference_mode()
    def capture_cudagraphs(self, max_num_seqs: int, max_model_len: int) -> None:
        """Record a decode step for batch sizes 1, 2, 4, 8, 16, 32, ..., and
        ``max_num_seqs`` itself, so every decode batch the scheduler can produce has a
        graph. ``max_num_seqs`` / ``max_model_len`` must be the scheduler's / engine's.
        Needs ``use_flash`` and the Triton KV write.

        Every graph reads the same fixed buffers (``graph_vars``), sized for the largest
        batch; ``block_table`` is ``ceil(max_model_len / block_size)`` wide. Recording
        goes from the largest size down and all graphs share one memory pool, so the
        smaller ones reuse the largest one's intermediate tensors.
        """
        assert self.use_flash, "CUDA Graph decode needs the FlashAttention path"
        # Padding rows carry slot -1. Only the Triton write skips them; the eager
        # write would index -1, i.e. overwrite the cache's last slot.
        assert self.kv_cache.use_triton, "CUDA Graph decode needs the Triton KV write"
        device = self.kv_cache.k_cache.device
        max_bs = max_num_seqs  # nano-vllm caps it at 512; then bigger batches would crash
        sizes = [bs for bs in [1, 2, 4, 8, *range(16, max_bs + 1, 16)] if bs <= max_bs]
        if sizes[-1] != max_bs:
            sizes.append(max_bs)  # e.g. max_num_seqs=20: 17..20 seqs need a graph too
        self.graph_batch_sizes = sizes
        max_blocks_per_seq = math.ceil(max_model_len / self.block_size)

        def buffer(*shape: int, dtype: torch.dtype, fill: int = 0) -> torch.Tensor:
            return torch.full(shape, fill, dtype=dtype, device=device)

        self.graph_vars = {
            "input_ids": buffer(max_bs, dtype=torch.long),
            "positions": buffer(max_bs, dtype=torch.long),
            "slot_mapping": buffer(max_bs, dtype=torch.long, fill=-1),  # -1: no KV write
            "cache_seqlens": buffer(max_bs, dtype=torch.int32),
            "block_table": buffer(max_bs, max_blocks_per_seq, dtype=torch.int32, fill=-1),
            "outputs": torch.zeros(max_bs, self.model.lm_head.embedding_dim,
                                   dtype=self.kv_cache.k_cache.dtype, device=device),
        }
        # 1. Warm up every size, then 2. capture them.
        self._warmup_cudagraphs()
        # Largest first, all graphs sharing one memory pool.
        pool = None
        for bs in reversed(sizes):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool):
                self._decode_on_graph_buffers(bs)
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph
        torch.cuda.synchronize(device)

    def _warmup_cudagraphs(self) -> None:
        """Run every graph batch size once for real, before any capture.

        First calls initialize things that must not happen during capture: cuBLAS
        handles and workspaces, the Triton JIT compile, FA2 / allocator first-time
        allocations. Without this, capture fails with "operation failed due to a
        previous error during capture". The buffers hold placeholders (slot -1,
        length 0), so this writes no KV and reads none.
        """
        for bs in self.graph_batch_sizes:
            self._decode_on_graph_buffers(bs)
        torch.cuda.synchronize(self.kv_cache.k_cache.device)

    def _decode_on_graph_buffers(self, bs: int) -> None:
        """One decode forward over the first ``bs`` rows of ``graph_vars``, writing
        ``outputs[:bs]``. Used for both the warm-up and the capture."""
        v = self.graph_vars
        set_context(Context(
            is_prefill=False, use_flash=True, kv_cache=self.kv_cache,
            slot_mapping=v["slot_mapping"][:bs], block_table=v["block_table"][:bs],
            cache_seqlens=v["cache_seqlens"][:bs],
        ))
        try:
            v["outputs"][:bs] = self.model(v["input_ids"][:bs], v["positions"][:bs])
        finally:
            reset_context()

    def _replay(
            self, input_ids: torch.Tensor, positions: torch.Tensor, context: Context
    ) -> torch.Tensor:
        """Copy this decode step into the graph buffers, replay the smallest graph that
        holds it, and return hidden states ``[num_seqs, hidden]``.

        Rows past ``num_seqs`` are padding: slot -1 (the KV write skips them), length 0
        and block ids -1 (attention reads nothing). Their outputs are discarded.
        """
        n, width = context.block_table.shape
        bs = next(b for b in self.graph_batch_sizes if b >= n)
        v = self.graph_vars
        v["input_ids"][:n] = input_ids
        v["positions"][:n] = positions
        # Reset padding first: rows left over from a bigger earlier step must not
        # write KV or read another sequence's blocks.
        v["slot_mapping"].fill_(-1)
        v["slot_mapping"][:n] = context.slot_mapping
        v["cache_seqlens"].zero_()
        v["cache_seqlens"][:n] = context.cache_seqlens
        v["block_table"].fill_(-1)
        v["block_table"][:n, :width] = context.block_table
        self.graphs[bs].replay()
        return v["outputs"][:n]
