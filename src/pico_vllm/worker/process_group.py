"""The torch.distributed process group (M8).

As in nano-vllm, a group always exists, even for one GPU (world_size=1), so the
parallel layers can call ``dist.get_rank()`` / ``dist.get_world_size()`` directly.
Single machine only: every rank meets at 127.0.0.1.
"""

from __future__ import annotations

import socket
from datetime import timedelta

import torch
import torch.distributed as dist

# How long a rank waits in a collective (or for the others to join) before the
# group aborts. If one rank dies (e.g. a TP worker raises), the others would
# otherwise sit in all_reduce for NCCL's default 10 minutes, looking hung.
TIMEOUT = timedelta(seconds=60)


def free_port() -> int:
    """A TCP port nobody is using right now (binding port 0 lets the OS pick one)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_process_group(
        tp_size: int, device: torch.device, port: int | None = None, rank: int = 0,
) -> bool:
    """Make sure a process group of ``tp_size`` exists, with this process as ``rank``.

    ``port``: where all ranks meet on 127.0.0.1. TP>1 must pass the one port that
    every rank was given; ``None`` (one rank) picks a free one.

    Returns True if this call created it (the caller then destroys it on exit), False
    if one already existed (another engine, or the test session's group). Raises
    ``RuntimeError`` if the existing group has a different world size: one process
    holds one group, so a different TP size needs a fresh process.
    """
    if dist.is_initialized():
        if dist.get_world_size() != tp_size:
            raise RuntimeError(
                f"process group has world_size {dist.get_world_size()}, need {tp_size}; "
                f"start tensor_parallel_size={tp_size} in a fresh process"
            )
        return False
    backend = "nccl" if device.type == "cuda" else "gloo"
    port = free_port() if port is None else port
    dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{port}",
                            world_size=tp_size, rank=rank, timeout=TIMEOUT)
    return True
