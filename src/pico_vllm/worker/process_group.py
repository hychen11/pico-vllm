"""The torch.distributed process group (M8).

As in nano-vllm, a group always exists, even for one GPU (world_size=1), so the
parallel layers can call ``dist.get_rank()`` / ``dist.get_world_size()`` directly.
Single machine only: every rank meets at 127.0.0.1.
"""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist


def free_port() -> int:
    """A TCP port nobody is using right now (binding port 0 lets the OS pick one)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_process_group(tp_size: int, device: torch.device) -> bool:
    """Make sure a process group of ``tp_size`` exists, with this process as rank 0.

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
    dist.init_process_group(backend, init_method=f"tcp://127.0.0.1:{free_port()}",
                            world_size=tp_size, rank=0)
    return True
