"""Deterministic tests for CaptureStatusMonitor against the real mock
CaptureServer transport -- not a hand-rolled test double, the actual
production CaptureServer running with its mock acquisition composition.
Covers: reaching BEGUN, explicit stop, BUSY rejection, connection refused,
stop-before-start, prompt stop on pre-acceptance failure, and the
decimal-string timestamp encoding.
"""

from __future__ import annotations

import asyncio

import pytest

from capture_server import protocol as p
from egocap_web.capture_status import CaptureStatusMonitor
from egocap_web.config import CaptureStatusConfig
from tests.support.capture_server import poll_until as _poll_until
from tests.support.capture_server import start_capture_server as _start_server
from tests.support.capture_server import stop_capture_server as _stop_server

pytestmark = pytest.mark.asyncio


def _config(port: int, **overrides) -> CaptureStatusConfig:
    return CaptureStatusConfig(capture_server_url=f"ws://127.0.0.1:{port}", **overrides)


async def test_monitor_reaches_begun_against_real_mock_server(unused_tcp_port):
    server = await _start_server(unused_tcp_port)
    try:
        monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        await monitor.start()
        await _poll_until(lambda: monitor.snapshot().running)

        snap = monitor.snapshot()
        assert snap.connecting is False
        assert snap.mode == p.CaptureMode.MOCK
        assert snap.session_id is not None
        assert snap.terminal is False
        assert snap.status is not None  # a real StatusSnapshot was observed

        await monitor.stop()
    finally:
        await _stop_server(server)


async def test_monitor_reports_terminal_on_explicit_stop(unused_tcp_port):
    server = await _start_server(unused_tcp_port)
    try:
        monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        await monitor.start()
        await _poll_until(lambda: monitor.snapshot().running)

        await monitor.stop()

        snap = monitor.snapshot()
        assert snap.terminal is True
        assert snap.running is False
        assert snap.stop_reason == p.StopReason.REQUESTED
    finally:
        await _stop_server(server)


async def test_on_stopped_forwards_capture_lost_to_the_recording_consumer(unused_tcp_port):
    """The one terminal signal a recording consumer needs: CaptureStatusMonitor's
    own real on_stopped must reach RecordingCallbacks.on_capture_lost, so
    an active take loses its capture source for real, not silently."""

    class _RecordingStub:
        def __init__(self) -> None:
            self.lost_reasons: list[str] = []

        def on_capture_frame(self, frame) -> None:
            pass

        def on_capture_image(self, image) -> None:
            pass

        def on_capture_lost(self, reason: str) -> None:
            self.lost_reasons.append(reason)

    server = await _start_server(unused_tcp_port)
    try:
        recording = _RecordingStub()
        monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        monitor.set_recording(recording)
        await monitor.start()
        await _poll_until(lambda: monitor.snapshot().running)

        await monitor.stop()

        assert len(recording.lost_reasons) == 1
        assert recording.lost_reasons[0]
    finally:
        await _stop_server(server)


async def test_monitor_reports_busy_rejection_honestly(unused_tcp_port):
    """A second attempt against an already-owned connection is a real
    BUSY/second-connection rejection, not something the monitor papers
    over or retries into ownership it was never granted."""
    server = await _start_server(unused_tcp_port)
    try:
        first = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        await first.start()
        await _poll_until(lambda: first.snapshot().running)

        second = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b2")
        await second.start()
        await _poll_until(lambda: second.snapshot().connecting is False)

        snap = second.snapshot()
        assert snap.error_message is not None
        assert snap.terminal is False
        assert snap.session_id is None

        await first.stop()
    finally:
        await _stop_server(server)


async def test_monitor_reports_connection_refused_honestly(unused_tcp_port):
    """No server is listening on this port at all -- the monitor must
    surface that as a local diagnostic, not hang connecting() forever."""
    monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
    await monitor.start()
    await _poll_until(lambda: monitor.snapshot().connecting is False)

    snap = monitor.snapshot()
    assert snap.error_message is not None
    assert snap.running is False
    assert snap.session_id is None


async def test_stop_immediately_after_start_never_connects(unused_tcp_port):
    """stop() called with no intervening await after start() must still
    prevent connection -- CaptureClient.stop() alone would no-op here
    because the startup task has not run at all yet (see capture_status.py's
    _stop_requested)."""
    server = await _start_server(unused_tcp_port)
    try:
        monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        await monitor.start()
        await monitor.stop()  # no await in between -- the race this guards against

        snap = monitor.snapshot()
        assert snap.connecting is False
        assert snap.running is False
        assert snap.session_id is None
        assert snap.error_message is not None

        # No capture attempt was ever made, so the server must have accepted
        # nobody -- a second monitor should be free to connect immediately.
        second = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b2")
        await second.start()
        await _poll_until(lambda: second.snapshot().running)
        await second.stop()
    finally:
        await _stop_server(server)


async def test_stop_returns_promptly_after_pre_acceptance_failure(unused_tcp_port):
    """A pre-acceptance failure (connection refused here) never produces
    on_stopped -- stop() must not wait out the full confirmation timeout
    for a callback that is never coming."""
    monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
    await monitor.start()
    await _poll_until(lambda: monitor.snapshot().connecting is False)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await monitor.stop()
    assert loop.time() - started < 1.0


async def test_response_json_uses_decimal_string_timestamp(unused_tcp_port):
    server = await _start_server(unused_tcp_port)
    try:
        monitor = CaptureStatusMonitor(_config(unused_tcp_port), backend_instance_id="b1")
        await monitor.start()
        await _poll_until(lambda: monitor.snapshot().status is not None)

        body = monitor.snapshot().to_json()
        assert isinstance(body["status"]["updated_at_ns"], str)
        assert body["status"]["updated_at_ns"].isdigit()
        assert body["mode"] == int(p.CaptureMode.MOCK)
        assert body["schema_version"] == 1

        await monitor.stop()
    finally:
        await _stop_server(server)
