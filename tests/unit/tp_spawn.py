"""Run a function on N CPU ranks (gloo) in fresh processes; used by the TP tests.

The pytest process already holds the session's world_size=1 group (conftest), and
one process holds one group, so TP>1 code has to run in spawned processes.
"""

from __future__ import annotations

import io
import traceback
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pico_vllm.worker.process_group import free_port


def _worker(rank: int, world: int, port: int, fn: Callable, args: tuple, q: Any) -> None:
    try:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                world_size=world, rank=rank)
        torch.manual_seed(0)  # same "full" weights and inputs on every rank
        result = fn(rank, *args)
        dist.destroy_process_group()
        # As bytes: a tensor put on a torch.multiprocessing queue is only a handle to
        # this process's shared memory, which is gone once this process exits.
        buf = io.BytesIO()
        torch.save(result, buf)
        q.put((rank, buf.getvalue(), None))
    except Exception:  # noqa: BLE001
        q.put((rank, None, traceback.format_exc()))


def run_ranks(fn: Callable, *args: Any, world: int = 2, timeout: float = 120) -> list[Any]:
    """``fn(rank, *args)`` on ranks 0..world-1; returns their results by rank.

    ``fn`` must be a module-level function (spawn pickles it by name). Raises with
    every failing rank's traceback.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, fn, args, q))
             for r in range(world)]
    for p in procs:
        p.start()
    try:
        out = {}
        for _ in procs:
            rank, payload, error = q.get(timeout=timeout)
            result = None if error else torch.load(io.BytesIO(payload), weights_only=False)
            out[rank] = (result, error)
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()
    errors = [f"rank {r}:\n{e}" for r, (_, e) in sorted(out.items()) if e]
    assert not errors, "\n".join(errors)
    return [out[r][0] for r in range(world)]
