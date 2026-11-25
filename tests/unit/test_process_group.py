"""``init_process_group``: create the group once, reuse it, refuse a different size."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
import torch.distributed as dist

from pico_vllm.worker.process_group import free_port, init_process_group


def test_free_port_is_bindable() -> None:
    import socket

    port = free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_reuses_the_existing_group() -> None:
    assert dist.is_initialized()  # the session group from conftest
    assert init_process_group(1, torch.device("cpu")) is False
    assert init_process_group(1, torch.device("cpu")) is False  # second engine: same
    assert dist.get_rank() == 0 and dist.get_world_size() == 1


def test_rejects_a_different_world_size() -> None:
    with pytest.raises(RuntimeError, match="world_size 1, need 2"):
        init_process_group(2, torch.device("cpu"))


def test_creates_the_group_when_none_exists() -> None:
    # A fresh interpreter: this pytest process already holds the session group.
    code = textwrap.dedent("""
        import torch, torch.distributed as dist
        from pico_vllm.worker.process_group import init_process_group
        assert not dist.is_initialized()
        assert init_process_group(1, torch.device("cpu")) is True
        assert dist.get_backend() == "gloo" and dist.get_world_size() == 1
        assert init_process_group(1, torch.device("cpu")) is False
        dist.destroy_process_group()
    """)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
