"""Focused coverage for tests/support/capture_server.py's own cleanup
guarantees -- not CaptureServer itself (see tests/capture_server/test_server.py
for that), but the shared test helper wrapping it."""

from __future__ import annotations

import asyncio
import socket

import pytest

from tests.support.capture_server import start_capture_server

pytestmark = pytest.mark.asyncio


async def test_start_cleans_up_run_task_when_bind_fails_before_any_listener_is_acquired(unused_tcp_port):
    """A deterministic bind failure -- the port is already occupied by a
    plain listening socket, not a DNS-dependent bad hostname -- happens
    before websockets.serve ever acquires a listener here, so there is
    nothing for this specific case to leak on that front. What this case
    does prove: the failed startup's run() task must not survive as a
    still-pending task -- it was actually cancelled and awaited, not just
    abandoned to finish on its own later."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", unused_tcp_port))
    blocker.listen(1)
    try:
        tasks_before = asyncio.all_tasks()

        with pytest.raises(OSError):
            await start_capture_server(unused_tcp_port)

        leaked = asyncio.all_tasks() - tasks_before
        assert all(t.done() for t in leaked)
    finally:
        blocker.close()
