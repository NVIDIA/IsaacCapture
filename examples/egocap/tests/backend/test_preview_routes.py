"""Tests for WS /api/session and GET /stream{stream_id} against the real
mock CaptureServer transport. Covers: bootstrap-before-data ordering,
BEGUN-before-tracking, binary header correctness, and MJPEG identity
binding/error codes.

Uses the app's own lifespan (via TestClient(app) as a context manager,
exactly like production) to create CaptureStatusMonitor, rather than
constructing one directly in the test and assigning it to app.state --
TestClient runs the ASGI app on its own event loop, and a monitor built on
a different loop (e.g. the test function's own pytest-asyncio loop) would
hang the first time a route awaited one of its asyncio primitives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import struct
import threading
import time

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import URL
from starlette.requests import ClientDisconnect
from starlette.websockets import WebSocketDisconnect, WebSocketState

from egocap_web.app import app
from egocap_web.capture_status import CaptureStatusMonitor
from egocap_web.config import CaptureStatusConfig
from egocap_web.preview_protocol import BINARY_HEADER_SIZE, BINARY_KIND_TRACKING
from egocap_web.preview_routes import _BoundedMjpegResponse, _trusted_hostnames, capture_session
from egocap_web.recording_command import RecordingContext
from egocap_web.recording_session import RecordingSessionCoordinator
from tests.support.capture_server import RunningServer, start_capture_server, stop_capture_server

# tests.support.capture_server's own worst-case startup/shutdown bound
# (close()'s ~12s plus the run-task join's 5s). asyncio.wait_for already
# cancels and awaits a timed-out coroutine before raising -- including
# whatever cleanup its own `finally` performs -- so this one bound is all
# the normal path needs; no separate custom cancel-and-wait wrapper.
_REAL_SERVER_CALL_BOUND_S = 20.0
# A last-resort bound around the cross-thread Future.result() call itself,
# comfortably above the deadline above. This is not expected to fire under
# normal operation (the inner wait_for deadline fires first and already
# waits out cleanup); if it ever does fire, that is a genuine unresolved
# hang on the loop thread, not evidence that cleanup completed.
_OUTER_SAFETY_BOUND_S = _REAL_SERVER_CALL_BOUND_S + 20.0


class _ServerThread:
    """Runs the mock CaptureServer on its own persistent event loop in a
    background thread. TestClient(app) runs the FastAPI app on a *different*
    event loop (its own anyio portal thread) -- the two only ever talk over
    a real ws:// TCP connection, never share asyncio objects, so this is
    safe. (asyncio.run() per start/stop call is NOT safe here: it tears
    down and recreates the loop the server's own internals are bound to.)
    """

    def __init__(self, port: int) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        try:
            self.server: RunningServer = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(start_capture_server(port), timeout=_REAL_SERVER_CALL_BOUND_S), self._loop
            ).result(timeout=_OUTER_SAFETY_BOUND_S)
        except BaseException:
            # Startup itself failed -- this thread/loop never got a
            # RunningServer to tear down via stop(), so they are this
            # constructor's own responsibility to retire, not left running.
            self._retire_loop()
            raise

    def stop(self) -> None:
        """Delegates to the one shared real-server teardown helper, on this
        thread's own event loop -- the loop/thread themselves are always
        torn down afterward, even if that teardown call itself failed or
        timed out, so a close failure can never leave this thread running."""
        try:
            asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(stop_capture_server(self.server), timeout=_REAL_SERVER_CALL_BOUND_S), self._loop
            ).result(timeout=_OUTER_SAFETY_BOUND_S)
        finally:
            self._retire_loop()

    def _retire_loop(self) -> None:
        """By the time this runs, the submitted call's own asyncio.wait_for
        has already cancelled and awaited its coroutine's completion on the
        loop -- this is only responsible for stopping the loop and joining
        the thread, not for waiting out any remaining cleanup itself."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()


def _request_one_camera(monkeypatch, tmp_path):
    """Configure a station YAML requesting one camera."""
    path = tmp_path / "station.yaml"
    path.write_text("video_sources:\n- type: oak_camera\n  name: head_human_camera\n", encoding="utf-8")
    monkeypatch.setenv("EGOCAP_STATION_CONFIG", str(path))


def test_server_thread_stops_its_background_thread_after_startup_failure(unused_tcp_port):
    """A deterministic startup failure -- the port is already occupied,
    before websockets.serve ever acquires a listener here -- must still
    leave _ServerThread's background thread actually stopped, not
    abandoned mid-cleanup, not merely raise."""
    threads_before = set(threading.enumerate())
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", unused_tcp_port))
    blocker.listen(1)
    try:
        with pytest.raises(OSError):
            _ServerThread(unused_tcp_port)
    finally:
        blocker.close()

    # The background thread's own loop was actually stopped, not
    # abandoned mid-cleanup -- no new thread survives this call.
    _poll_until_sync(lambda: set(threading.enumerate()) <= threads_before, timeout_s=10.0)


def _poll_until_sync(predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


@pytest.mark.parametrize(
    "origin",
    [None, "null", "http://evil.example", "http://testserver.evil.example", "http://testserver:9999"],
    ids=["missing", "literal_null", "foreign", "suffix_trick", "wrong_port"],
)
def test_a_connection_with_a_missing_or_unlisted_origin_is_rejected_before_accept(
    unused_tcp_port, monkeypatch, origin
):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            headers = {} if origin is None else {"origin": origin}
            with (
                pytest.raises(WebSocketDisconnect) as exc_info,
                client.websocket_connect("/api/session", headers=headers),
            ):
                pass
            assert exc_info.value.code == 1008
            # No admission slot was consumed by a connection rejected before accept().
            assert monitor._ws_viewer_count == 0
    finally:
        server.stop()


def test_a_connection_from_an_explicitly_configured_extra_origin_is_accepted(unused_tcp_port, monkeypatch):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_ALLOWED_ORIGINS", "https://example.com, https://other.example")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            with client.websocket_connect("/api/session", headers={"origin": "https://example.com"}) as ws:
                first = json.loads(ws.receive_text())
                assert first["type"] == 3
            _poll_until_sync(lambda: monitor._ws_viewer_count == 0)
    finally:
        server.stop()


def test_trusted_hostnames_entries_are_lowercased_and_unbracketed(monkeypatch):
    # websocket.url.hostname (what this is matched against) is already
    # lowercased and bracket-stripped by urllib -- a configured entry like
    # "Station.local" or "[::1]" must normalize the same way, or it would
    # silently never match.
    monkeypatch.setenv("EGOCAP_TRUSTED_HOSTNAMES", "Station.Lab.Local, [::1], 127.0.0.1")
    assert _trusted_hostnames() == frozenset({"station.lab.local", "::1", "127.0.0.1"})


def test_a_same_origin_host_not_in_the_configured_trust_set_is_rejected(unused_tcp_port, monkeypatch):
    # Simulates DNS rebinding: Host and Origin agree with each other
    # (exactly what _expected_origin would accept unpinned), but Host
    # itself ("testserver", TestClient's default) is not in the deployment's
    # configured trusted set -- EGOCAP_TRUSTED_HOSTNAMES being set at all
    # should refuse to derive trust from an unlisted Host.
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_TRUSTED_HOSTNAMES", "station.lab.local")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            with (
                pytest.raises(WebSocketDisconnect) as exc_info,
                client.websocket_connect("/api/session", headers={"origin": "http://testserver"}),
            ):
                pass
            assert exc_info.value.code == 1008
            assert monitor._ws_viewer_count == 0
    finally:
        server.stop()


def test_a_same_origin_host_in_the_configured_trust_set_is_accepted(unused_tcp_port, monkeypatch):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_TRUSTED_HOSTNAMES", "testserver, station.lab.local")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                first = json.loads(ws.receive_text())
                assert first["type"] == 3
            _poll_until_sync(lambda: monitor._ws_viewer_count == 0)
    finally:
        server.stop()


def test_an_explicit_extra_origin_still_works_with_a_trust_set_configured(unused_tcp_port, monkeypatch):
    # EGOCAP_ALLOWED_ORIGINS entries are literal Origin matches the operator
    # configured directly -- they must not be affected by EGOCAP_TRUSTED_HOSTNAMES,
    # which only pins the separate Host-derived same-origin fallback.
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.setenv("EGOCAP_ALLOWED_ORIGINS", "https://example.com")
    monkeypatch.setenv("EGOCAP_TRUSTED_HOSTNAMES", "station.lab.local")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            with client.websocket_connect("/api/session", headers={"origin": "https://example.com"}) as ws:
                first = json.loads(ws.receive_text())
                assert first["type"] == 3
            _poll_until_sync(lambda: monitor._ws_viewer_count == 0)
    finally:
        server.stop()


def test_ws_bootstrap_then_running_descriptor_bearing_snapshot(unused_tcp_port, monkeypatch):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                first = json.loads(ws.receive_text())
                assert first["version"] == 1
                assert first["type"] == 3

                for _ in range(100):
                    msg = json.loads(ws.receive_text())
                    if "capture" not in msg:
                        continue  # a proactively-pushed recording_snapshot/reply, not a capture snapshot
                    status = msg["capture"]["status"]
                    if status is not None and status["running"]:
                        assert msg["capture"]["binary_generation"] == 1
                        assert len(msg["capture"]["started"]["streams"]) >= 1
                        break
                else:
                    raise AssertionError("never saw a running, descriptor-bearing snapshot")
            _poll_until_sync(lambda: monitor._ws_viewer_count == 0)
    finally:
        server.stop()


def test_ws_tracking_binary_message_has_correct_header(unused_tcp_port, monkeypatch):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                for _ in range(200):
                    message = ws.receive()
                    if "bytes" in message and message["bytes"] is not None:
                        payload = message["bytes"]
                        assert len(payload) >= BINARY_HEADER_SIZE
                        generation, kind, reserved, stream_id, sequence, cap_ns, recv_ns = struct.unpack(
                            "<IBBHQQQ", payload[:BINARY_HEADER_SIZE]
                        )
                        assert generation == 1
                        assert kind == BINARY_KIND_TRACKING
                        assert reserved == 0
                        assert stream_id == 0
                        assert sequence >= 0
                        assert cap_ns > 0
                        assert recv_ns > 0
                        break
                else:
                    raise AssertionError("no binary tracking message received")
    finally:
        server.stop()


def _recv_recording_reply(ws, request_id: str) -> dict:
    """Reads text messages until the correlated recording_reply for
    ``request_id`` arrives, skipping ordinary preview snapshots/heartbeats
    in between."""
    for _ in range(200):
        message = ws.receive()
        text = message.get("text")
        if text is None:
            continue
        parsed = json.loads(text)
        if parsed.get("type") == "recording_reply" and parsed.get("request_id") == request_id:
            return parsed
    raise AssertionError(f"no recording_reply for {request_id} received")


def _recv_recording_snapshot(ws) -> dict:
    """Reads text messages until a proactively-pushed recording_snapshot message arrives."""
    for _ in range(200):
        message = ws.receive()
        text = message.get("text")
        if text is None:
            continue
        parsed = json.loads(text)
        if parsed.get("type") == "recording_snapshot":
            return parsed
    raise AssertionError("no recording_snapshot message received")


def test_recording_snapshot_is_pushed_proactively_on_connect_with_no_command_sent(unused_tcp_port, monkeypatch):
    """A freshly-connecting viewer must learn the current recording context
    without having to send a command first -- otherwise there is nothing to
    initialize the frontend's recordingSnapshot from on connect."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            batch_id = client.app.state.recording_context.batch_id
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                pushed = _recv_recording_snapshot(ws)
                assert pushed["snapshot"]["batch_id"] == batch_id
                assert pushed["snapshot"]["episode_phase"] == "IDLE"
    finally:
        server.stop()


def test_a_second_viewer_sees_the_first_viewers_command_without_sending_one_itself(
    unused_tcp_port, monkeypatch, tmp_path
):
    """Regression test for multi-tab correctness: a second, passive viewer
    must see the FIRST viewer's start_episode as a proactive push, not a
    stale/absent recording context -- otherwise it could act on an episode
    that already changed state from its point of view."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    _request_one_camera(monkeypatch, tmp_path)
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            client.app.state.recording_context.coordinator.set_task("Pour water.")
            batch_id = client.app.state.recording_context.batch_id
            with (
                client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws1,
                client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws2,
            ):
                _recv_recording_snapshot(ws1)  # bootstrap push on each connection
                _recv_recording_snapshot(ws2)
                # start_episode requires real, fresh readiness -- a camera
                # descriptor accepted and an actual preview sample for both
                # tracking and the configured camera.
                _poll_until_sync(
                    lambda: (
                        monitor.preview.tracking() is not None
                        and monitor.camera_descriptors
                        and monitor.preview.camera("head_human_camera") is not None
                    )
                )

                ws1.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r1",
                            "action": "start_episode",
                            "payload": {"batch_id": batch_id},
                        }
                    )
                )
                reply = _recv_recording_reply(ws1, "r1")
                assert reply["outcome"] == "ok"
                episode_id = reply["snapshot"]["current_episode_id"]

                # ws2 never sent a command -- it must still see the change.
                # Mid-operation transitions are now ALSO proactively pushed
                # (STARTING, then RECORDING once activation completes) --
                # read until the final RECORDING push lands, not just the
                # first one.
                pushed = _recv_recording_snapshot(ws2)
                while pushed["snapshot"]["episode_phase"] != "RECORDING":
                    pushed = _recv_recording_snapshot(ws2)
                assert pushed["snapshot"]["current_episode_id"] == episode_id
    finally:
        server.stop()


def test_a_readiness_change_is_pushed_with_no_command_sent(unused_tcp_port, monkeypatch):
    """A readiness flip (a camera going stale or coming up) fires no callback and follows no command; a viewer
    that was already connected must still be told, or Record stays disabled until a reload."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            coordinator = client.app.state.recording_context.coordinator
            coordinator.set_task("Pour water.")
            state = {"ready": False}
            coordinator._readiness_check = lambda: (state["ready"], None if state["ready"] else "no camera")
            # The first look always counts as a change; take it now so the flip below is the only one.
            client.app.state.recording_context.snapshot()
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                assert _recv_recording_snapshot(ws)["snapshot"]["ready_to_record"] is False
                state["ready"] = True  # no command is sent
                pushed = _recv_recording_snapshot(ws)
                assert pushed["snapshot"]["ready_to_record"] is True
    finally:
        server.stop()


def test_recording_context_survives_a_reconnect_not_scoped_per_connection(unused_tcp_port, monkeypatch, tmp_path):
    """Regression test for the exact bug of a per-WebSocket-connection
    RecordingContext: committing a task and starting an episode on one
    connection, then reconnecting, must see that same session/task/episode
    state -- not a freshly reset context -- because RecordingContext is
    owned once for the whole app lifetime (see its own docstring)."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    _request_one_camera(monkeypatch, tmp_path)
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            # The real backend-assigned batch id -- never a fixed string.
            batch_id = client.app.state.recording_context.batch_id
            assert isinstance(batch_id, str) and batch_id != ""

            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r1",
                            "action": "set_task",
                            "payload": {"batch_id": batch_id, "task": "Pour water into the cup."},
                        }
                    )
                )
                reply = _recv_recording_reply(ws, "r1")
                assert reply["outcome"] == "ok"

                # start_episode requires real, fresh readiness -- a camera
                # descriptor accepted and an actual preview sample for both
                # tracking and the configured camera.
                _poll_until_sync(
                    lambda: (
                        monitor.preview.tracking() is not None
                        and monitor.camera_descriptors
                        and monitor.preview.camera("head_human_camera") is not None
                    )
                )

                ws.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r2",
                            "action": "start_episode",
                            "payload": {"batch_id": batch_id},
                        }
                    )
                )
                reply = _recv_recording_reply(ws, "r2")
                assert reply["outcome"] == "ok"
                episode_id = reply["snapshot"]["current_episode_id"]
                # Let real tracking/camera frames actually flow into the
                # take before stopping it -- an empty take fails QC
                # (pico_controller/camera are required sources) and would
                # never report KEPT regardless of this test's own concern.
                time.sleep(0.3)
                assert episode_id is not None

            # A fresh connection, same app -- must see the SAME context, not a reset one.
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws2:
                ws2.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r3",
                            "action": "stop_episode",
                            "payload": {"batch_id": batch_id, "episode_id": episode_id},
                        }
                    )
                )
                reply = _recv_recording_reply(ws2, "r3")
                assert reply["outcome"] == "ok"
                assert reply["snapshot"]["task_text"] == "Pour water into the cup."
                assert reply["snapshot"]["saved_count"] == 1
    finally:
        server.stop()


def test_new_batch_rotates_identity_and_a_stale_concurrent_request_is_rejected(unused_tcp_port, monkeypatch):
    """Two tabs both see the old batch; one requests new_batch and succeeds,
    the other's concurrent new_batch request against the now-stale id is
    rejected -- the same stale-target pattern as the other commands."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            old_batch_id = client.app.state.recording_context.batch_id
            with (
                client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws1,
                client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws2,
            ):
                _recv_recording_snapshot(ws1)
                _recv_recording_snapshot(ws2)

                ws1.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r1",
                            "action": "new_batch",
                            "payload": {"batch_id": old_batch_id},
                        }
                    )
                )
                reply1 = _recv_recording_reply(ws1, "r1")
                assert reply1["outcome"] == "ok"
                new_batch_id = reply1["snapshot"]["batch_id"]
                assert new_batch_id != old_batch_id

                # ws2 never sent anything -- it must observe the rotated batch via the proactive push.
                pushed = _recv_recording_snapshot(ws2)
                assert pushed["snapshot"]["batch_id"] == new_batch_id

                # ws2's own concurrent request against the now-stale id is rejected.
                ws2.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r2",
                            "action": "new_batch",
                            "payload": {"batch_id": old_batch_id},
                        }
                    )
                )
                reply2 = _recv_recording_reply(ws2, "r2")
                assert reply2["outcome"] == "rejected"
    finally:
        server.stop()


def test_empty_task_text_gets_a_rejected_reply_and_never_closes_the_socket(unused_tcp_port, monkeypatch):
    """Regression test: an ordinary empty task-text edit must stay a normal
    rejected business outcome, never a 1003 protocol-close of the whole
    connection (which would also kill tracking/camera/MJPEG for that
    viewer, not just the task edit)."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            batch_id = client.app.state.recording_context.batch_id
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                ws.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "operator_token": "<tester>",
                            "control_token": client.app.state.recording_context.control_token,
                            "request_id": "r1",
                            "action": "set_task",
                            "payload": {"batch_id": batch_id, "task": ""},
                        }
                    )
                )
                reply = _recv_recording_reply(ws, "r1")
                assert reply["outcome"] == "rejected"
                assert reply["reason"] is not None

                # The socket is still alive -- a routine snapshot still
                # arrives, proving tracking/camera delivery was never torn
                # down by this.
                for _ in range(50):
                    message = ws.receive()
                    if message.get("text") is not None:
                        break
                else:
                    raise AssertionError("socket appears closed: no further message received")
    finally:
        server.stop()


def test_a_wrong_control_token_is_an_ordinary_rejection_leaves_the_socket_connected_and_is_not_echoed(
    unused_tcp_port, monkeypatch
):
    """A business-rule rejection (wrong control_token), not a protocol
    close -- same reasoning as the empty-task-text case above -- and the
    rejected reply must never echo the credential the client sent back to
    it."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            batch_id = client.app.state.recording_context.batch_id
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                real_token = client.app.state.recording_context.control_token
                wrong_token = real_token[:-1] + ("y" if real_token[-1] != "y" else "z")
                ws.send_text(
                    json.dumps(
                        {
                            "type": "recording_command",
                            "control_token": wrong_token,
                            "request_id": "r1",
                            "action": "set_task",
                            "payload": {"batch_id": batch_id, "task": "New task"},
                        }
                    )
                )
                reply = _recv_recording_reply(ws, "r1")
                assert reply["outcome"] == "rejected"
                assert wrong_token not in json.dumps(reply)

                for _ in range(50):
                    message = ws.receive()
                    if message.get("text") is not None:
                        break
                else:
                    raise AssertionError("socket appears closed: no further message received")
    finally:
        server.stop()


@pytest.mark.parametrize(
    "build_malformed_message",
    [
        # "payload" is missing entirely -- a structural protocol violation.
        lambda token: {
            "type": "recording_command",
            "control_token": token,
            "request_id": "r1",
            "action": "set_task",
        },
        # The real token placed in `action` instead of `control_token` --
        # the exact leak vector parse_recording_command's old unknown-action
        # error used to echo via `{action!r}` in the raised exception text.
        lambda token: {"type": "recording_command", "request_id": "r1", "action": token, "payload": {}},
    ],
    ids=["missing_payload", "token_in_action_field"],
)
def test_a_malformed_command_carrying_a_real_control_token_never_logs_it(
    unused_tcp_port, monkeypatch, caplog, build_malformed_message
):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with caplog.at_level(logging.WARNING), TestClient(app) as client:
            real_token = client.app.state.recording_context.control_token
            with client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws:
                ws.send_text(json.dumps(build_malformed_message(real_token)))
                for _ in range(200):  # snapshots may still arrive before the close does
                    message = ws.receive()
                    if message["type"] == "websocket.close":
                        assert message["code"] == 1003
                        break
                else:
                    raise AssertionError("the socket was not closed")
    finally:
        server.stop()
    # Not just "the secret is absent" (which would also pass if nothing
    # logged at all) -- the expected fixed-category warning actually fired.
    assert "recording command rejected: malformed or invalid command" in caplog.text
    assert real_token not in caplog.text


_OVERSIZED = {
    "type": "recording_command",
    "request_id": "r1",
    "action": "start_episode",
    "payload": {"x": "y" * 9000},
}
_UNKNOWN_ACTION = {"type": "recording_command", "request_id": "r1", "action": "no_such_action", "payload": {}}


@pytest.mark.parametrize(
    ("frame", "binary"),
    [
        (b"\x00\x01 not text", True),
        ("this is not json", False),
        (json.dumps(_UNKNOWN_ACTION), False),
        (json.dumps(_OVERSIZED), False),
    ],
)
def test_a_malformed_binary_or_oversized_recording_command_closes_the_socket_with_1003(
    unused_tcp_port, monkeypatch, frame, binary
):
    """The receiver only asks for the close; the single sender performs it, with the protocol-error code."""
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with (
            TestClient(app) as client,
            client.websocket_connect("/api/session", headers={"origin": "http://testserver"}) as ws,
        ):
            (ws.send_bytes if binary else ws.send_text)(frame)
            for _ in range(200):  # snapshots may still arrive before the close does
                message = ws.receive()
                if message["type"] == "websocket.close":
                    assert message["code"] == 1003
                    break
            else:
                raise AssertionError("the socket was not closed")
    finally:
        server.stop()


def test_mjpeg_unknown_stream_returns_404(unused_tcp_port, monkeypatch):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    try:
        with TestClient(app) as client:
            resp = client.get("/stream1?backend_instance_id=b1&session_id=x")
            assert resp.status_code == 404
    finally:
        server.stop()


def test_mjpeg_superseded_backend_identity_returns_409(unused_tcp_port, monkeypatch, tmp_path):
    server = _ServerThread(unused_tcp_port)
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    _request_one_camera(monkeypatch, tmp_path)
    try:
        with TestClient(app) as client:
            monitor = client.app.state.monitor
            _poll_until_sync(lambda: monitor.native.started is not None)
            resp = client.get(
                f"/stream{monitor.camera_descriptors[0].stream_id}?backend_instance_id=WRONG&session_id=x"
            )
            assert resp.status_code == 409
    finally:
        server.stop()


# A happy-path MJPEG test (GET /stream{id} returning a real multipart
# response body) is deliberately not automated here: driving an infinite
# multipart/x-mixed-replace StreamingResponse through starlette's sync
# TestClient hung in this environment even before headers were returned,
# for reasons unrelated to the route's own logic (the 404/409 error-path
# tests above, which return an ordinary JSONResponse from the same route
# function, work fine). This is verified instead by the real end-to-end
# smoke test against a live uvicorn process documented in the
# implementation report, which confirms actual JPEG bytes arriving over
# the wire via curl.


class _FakeApp:
    def __init__(self, monitor: CaptureStatusMonitor) -> None:
        # Never opened in this test -- only recording_context.snapshot_message()
        # is exercised here, so a real-but-unopened coordinator is enough.
        coordinator = RecordingSessionCoordinator(
            lambda: (_ for _ in ()).throw(AssertionError("not expected to open")),
            queue_capacity=1,
            readiness_check=lambda: (False, "not configured"),
        )
        context = RecordingContext(coordinator=coordinator, control_token="<unused>")
        self.state = type("_State", (), {"monitor": monitor, "recording_context": context})()


class _HangingWebSocket:
    """A fake WebSocket whose accept/receive succeed, but whose every send
    and close call blocks forever -- used to prove capture_session's finally
    chain still completes and releases the admission slot within the send
    timeout, never depending on close() actually succeeding."""

    def __init__(self, app: _FakeApp) -> None:
        self.app = app
        self.client_state = WebSocketState.CONNECTED
        # Enough of a real WebSocket's surface for capture_session's own
        # Origin check to pass -- this test is about the hang/timeout
        # behavior downstream of admission, not about origin rejection.
        self.headers = {"origin": "http://testserver"}
        self.url = URL("ws://testserver/api/session")

    async def accept(self) -> None:
        return None

    async def receive(self):
        # Simulated "no message, no disconnect yet" -- the receiver task is
        # cancelled by _serve_session's own finally, not by this returning.
        await asyncio.sleep(3600)

    async def send_text(self, _data: str) -> None:
        await asyncio.sleep(3600)

    async def send_bytes(self, _data: bytes) -> None:
        await asyncio.sleep(3600)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_ws_route_completes_and_releases_slot_when_send_and_close_both_hang():
    """Reproduces the bug a bounded SEND_TIMEOUT_S alone does not fix: if
    the route's own close() call (in _serve_session's finally) has no
    timeout, a slow/unresponsive peer can block it forever, which blocks
    capture_session's own finally from ever running release_ws_viewer().
    Both _bounded_close's timeout and the outer admission accounting are
    exercised here with no real network/process involved."""
    monitor = CaptureStatusMonitor(CaptureStatusConfig(), backend_instance_id="b1")
    fake_ws = _HangingWebSocket(_FakeApp(monitor))

    assert monitor._ws_viewer_count == 0
    # Expected real duration is ~SEND_TIMEOUT_S (blocked send) +
    # ~SEND_TIMEOUT_S (blocked close), each independently bounded; this
    # wait_for is only a generous outer guard against a true regression
    # (an actually-unbounded close) hanging the whole test suite.
    await asyncio.wait_for(capture_session(fake_ws), timeout=10.0)
    assert monitor._ws_viewer_count == 0


@pytest.mark.asyncio
async def test_mjpeg_response_releases_admission_when_initial_send_hangs():
    """A peer that never reads the response (http.response.start itself
    blocks) must still release the per-camera viewer admission -- this is
    the failure-before-the-body-generator-starts case a release placed
    only in the generator's own finally would miss."""
    released = []

    async def never_finishes_body():
        yield b"unused"

    async def hanging_send(_message) -> None:
        await asyncio.sleep(3600)

    response = _BoundedMjpegResponse(
        never_finishes_body(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        release=lambda: released.append(True),
    )

    async def no_receive():
        await asyncio.sleep(3600)

    # scope["asgi"]["spec_version"] >= 2.4 (what real uvicorn negotiates)
    # sends the response body directly rather than racing it against
    # listen_for_disconnect -- the race would otherwise let a `receive`
    # that returns immediately short-circuit this test before the send
    # ever actually blocks.
    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}

    # _BoundedMjpegResponse's own per-send timeout (SEND_TIMEOUT_S=2.0) is
    # what should actually fire here, not this outer guard -- the guard is
    # only a generous safety net against a true regression hanging the
    # whole test suite. TimeoutError is a stdlib OSError subclass, so
    # starlette's own spec_version>=2.4 path converts it to ClientDisconnect
    # before it reaches this caller -- either is proof the send didn't hang.
    with pytest.raises((TimeoutError, ClientDisconnect)):
        await asyncio.wait_for(response(scope, no_receive, hanging_send), timeout=8.0)
    assert released == [True]
