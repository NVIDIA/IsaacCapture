"""Shared fixture for a real, running CaptureServer with mock acquisition.
Normal CaptureClient integration tests use this instead of reimplementing
server behavior. A deliberate protocol violation the real server cannot be
configured to produce uses its own narrow, test-local peer instead (see
test_client.py's lifecycle-ordering/replay section and its two
never-confirms-Stop fixtures)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.support.capture_server import start_capture_server, stop_capture_server


@pytest.fixture
async def real_mock_server(unused_tcp_port: int) -> AsyncIterator[str]:
    """Yields the ws:// URL of a running production CaptureServer with mock acquisition."""
    server = await start_capture_server(unused_tcp_port)
    try:
        yield f"ws://127.0.0.1:{unused_tcp_port}"
    finally:
        await stop_capture_server(server)
