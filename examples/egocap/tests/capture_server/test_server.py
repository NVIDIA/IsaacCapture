"""Deterministic, real-socket tests for the production CaptureServer --
not mocked transports, an actual websockets listener and an actual
CaptureClient talking to it. Ordinary tests start it via
tests/support/capture_server.py's shared startup/teardown helpers; a
test specifically about CaptureServer's own run()/close() internals still
constructs it directly, with its own inline readiness wait and teardown.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
import websockets
from websockets.exceptions import ConnectionClosedError

from capture_server import protocol as p
from capture_server.acquisition import Acquisition
from capture_server.acquisition._thread_runner import ThreadRunner
from capture_server.client import CaptureClient
from capture_server.launch import build_mock_source_factory
from capture_server.server import CaptureServer, ServerConfig
from tests.support.capture_server import start_capture_server, stop_capture_server

pytestmark = pytest.mark.asyncio


async def test_full_lifecycle_real_client_against_real_server(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        events: list[tuple] = []
        started: list = []
        frames: list = []
        stopped: list = []
        finished = asyncio.Event()

        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS),
            cameras=({"name": "camera_left", "type": p.CameraType.OAK_D_COLOR},),
            on_started=lambda e: (started.append(e), events.append(("started", e.running, len(e.streams)))),
            on_status=lambda e: events.append(("status", e.type.name, e.running)),
            on_capture_frame=lambda f: frames.append(f),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        await asyncio.sleep(0.5)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)

        assert events[0] == ("started", False, 2)
        assert ("status", "BEGUN", True) in events
        assert len(frames) > 0
        assert all(len(f.payload) == 1995 for f in frames)
        # The actual object-identity check: every frame's description is
        # the SAME object CaptureStarted.streams carried for stream_id 0,
        # not a rebuilt copy per frame.
        tracking_descriptor = next(s for s in started[0].streams if s.stream_id == 0)
        assert all(f.description is tracking_descriptor for f in frames)
        assert frames[0].sequence == 0
        assert [f.sequence for f in frames] == list(range(len(frames)))

        result = stopped[0]
        assert result.running is False
        assert result.reason == p.StopReason.REQUESTED
        assert result.error_message is None
        counters = {c.stream_id: c for c in result.streams}
        for counter in counters.values():
            assert counter.produced == counter.sent + counter.dropped
        # sent (server-side successful writes) can be slightly >= len(frames)
        # received by on_capture_frame: the client itself suppresses any
        # frame arriving after its own stop() request.
        assert counters[0].sent >= len(frames)
    finally:
        await stop_capture_server(running)


async def test_camera_only_capture_has_no_tracking_stream(unused_tcp_port):
    """Camera-only (no tracking.sources, no plugin request) never enters
    IC/OpenXR at all -- status must describe only the camera, with NO
    native provider/device inventory whatsoever."""
    running = await start_capture_server(unused_tcp_port)
    try:
        started = []
        statuses = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(),
            cameras=({"name": "camera_left", "type": p.CameraType.OAK_D_COLOR},),
            on_started=lambda e: started.append(e),
            on_status=lambda e: statuses.append(e),
            on_stopped=lambda r: finished.set(),
        )
        assert len(started[0].streams) == 1
        await asyncio.sleep(0.2)
        begun = next((s for s in statuses if s.status is not None), None)
        assert begun is not None, "expected at least one SessionStatus with a status snapshot"
        assert not begun.status.providers, "no native provider inventory for a camera-only attempt"
        assert not begun.status.devices
        assert len(begun.status.cameras) == 1
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)


async def test_enabled_plugin_with_empty_tracking_sources_is_rejected(unused_tcp_port):
    """The real IC output-graph constructor rejects an empty output
    mapping outright (a constraint the mock now also enforces, see
    mock_isaac_capture.OutputCombiner) -- so "an enabled plugin with zero
    tracking.sources selected" has no valid native graph to build at all,
    not merely an odd-but-supportable combination. Rejected up front as
    INVALID_CONFIG, before ever reaching source construction."""
    running = await start_capture_server(unused_tcp_port)
    try:
        begin = p.BeginRequest(
            session_id="empty-sources-with-plugin-session",
            tracking=p.TrackingConfig(sources=()),
            plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True),),
        )
        config = p.effective_config(begin)
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(raw)
        assert data.get("type") == int(p.MessageType.REQUEST_REJECTED)
        assert data.get("reason") == int(p.RejectionReason.INVALID_CONFIG)
    finally:
        await stop_capture_server(running)


async def test_plugin_request_independent_of_hands_subscription_still_enters_ic(unused_tcp_port):
    """Plugin requests are independent of output subscriptions: an
    enabled Manus plugin alongside a NONEMPTY
    tracking.sources that does NOT include HANDS still needs IC entered
    for plugin discovery to run, and Manus's native inventory shows up in
    status -- selecting HANDS never implies requesting Manus, and
    requesting Manus never implies selecting HANDS."""
    running = await start_capture_server(unused_tcp_port)
    try:
        started = []
        statuses = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            cameras=({"name": "camera_left", "type": p.CameraType.OAK_D_COLOR},),
            plugins=(p.PluginType.MANUS,),
            on_started=lambda e: started.append(e),
            on_status=lambda e: statuses.append(e),
            on_stopped=lambda r: finished.set(),
        )
        assert len(started[0].streams) == 2  # tracking + the one camera
        await asyncio.sleep(0.2)
        begun = next((s for s in statuses if s.status is not None), None)
        assert begun is not None, "expected at least one SessionStatus with a status snapshot"
        provider_types = {pr.type for pr in begun.status.providers}
        assert p.ProviderType.OPENXR_RUNTIME in provider_types
        assert p.ProviderType.PLUGIN in provider_types
        manus = next(pr for pr in begun.status.providers if pr.plugin == p.PluginType.MANUS)
        assert manus.status == p.ProviderState.AVAILABLE
        device_ids = {d.id for d in begun.status.devices}
        assert "device-headset" in device_ids
        # Manus's hand devices appear even though HANDS was never selected.
        assert "device-hand-left" in device_ids
        assert "device-hand-right" in device_ids
        assert len(begun.status.cameras) == 1
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)


async def test_required_plugin_fails_establishment_when_undiscoverable(unused_tcp_port):
    """A required Manus plugin that can't be discovered must fail session
    establishment with IC_ERROR -- the acceptance criterion the launch.py
    IC-construction fix exists for, now exercised with a NONEMPTY
    tracking.sources (HEAD) alongside the plugin, since zero sources with
    an enabled plugin is rejected before establishment is ever attempted
    (see test_enabled_plugin_with_empty_tracking_sources_is_rejected)."""
    running = await start_capture_server(unused_tcp_port, ic_faults={"manus_plugin_load_failed"})
    try:
        begin = p.BeginRequest(
            session_id="required-plugin-undiscoverable-session",
            tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
            plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True, required=True),),
        )
        config = p.effective_config(begin)
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )

            async def _await_stopped() -> dict:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    data = json.loads(raw)
                    if data.get("type") == int(p.MessageType.SESSION_STOPPED):
                        return data

            # One overall bound on the whole wait, not 5s per individual
            # recv() -- a regression that never ends the session (the
            # exact bug this test exists to catch) must fail fast here,
            # not hang past any single test runner's own global timeout.
            stopped_event = await asyncio.wait_for(_await_stopped(), timeout=10)
        assert stopped_event["reason"] == int(p.StopReason.IC_ERROR)
        assert stopped_event["error_message"] is not None
    finally:
        await stop_capture_server(running)


async def test_second_owner_connection_is_rejected_with_1013(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        finished = asyncio.Event()
        await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda r: finished.set())

        with pytest.raises(ConnectionClosedError) as excinfo:
            async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws2:
                await ws2.recv()
        assert excinfo.value.rcvd.code == 1013

        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)


@pytest.mark.parametrize(
    ("fault_name", "expected_reason"),
    [
        ("startup_error", p.StopReason.IC_ERROR),
        ("camera_required_failure", p.StopReason.CAMERA_ERROR),
        ("fatal_acquisition_error", p.StopReason.IC_EXITED),
    ],
)
async def test_named_fault_produces_correct_stop_reason(unused_tcp_port, fault_name, expected_reason):
    running = await start_capture_server(unused_tcp_port, ic_faults={fault_name}, camera_faults={fault_name})
    try:
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        cameras = (
            ({"name": "camera_left", "type": p.CameraType.OAK_D_COLOR},)
            if fault_name == "camera_required_failure"
            else ()
        )
        await client.start(
            sources=(p.SourceType.HEAD,),
            cameras=cameras,
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=10)
        assert stopped[0].reason == expected_reason
        assert stopped[0].error_message is not None
    finally:
        await stop_capture_server(running)


async def test_provider_failed_fault_emits_session_status(unused_tcp_port):
    """The Manus provider only appears in status at all when
    the session actually requested the Manus plugin -- this fault
    simulates that REQUESTED plugin failing, so the request must be
    explicit here, not implied merely by selecting HANDS."""
    running = await start_capture_server(unused_tcp_port, ic_faults={"provider_failed"})
    try:
        statuses = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            plugins=(p.PluginType.MANUS,),
            on_status=lambda e: statuses.append(e),
            on_stopped=lambda r: finished.set(),
        )
        await asyncio.sleep(0.3)
        failed = [
            s
            for s in statuses
            if s.status is not None and any(pr.status == p.ProviderState.FAILED for pr in s.status.providers)
        ]
        assert failed, "expected at least one SessionStatus reflecting the failed provider"
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)


async def test_camera_optional_failure_skips_frames_without_ending_session(unused_tcp_port):
    running = await start_capture_server(
        unused_tcp_port,
        camera_faults={"camera_optional_failure"},
        config=ServerConfig(camera_required=False),
    )
    try:
        images = []
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            cameras=({"name": "camera_left", "type": p.CameraType.OAK_D_COLOR},),
            on_capture_image=lambda i: images.append(i),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        await asyncio.sleep(0.4)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert len(images) == 0  # every tick failed for this always-on fault
        assert stopped[0].reason == p.StopReason.REQUESTED  # session was not ended by the fault
    finally:
        await stop_capture_server(running)


async def test_connect_delay_can_be_stopped_before_begun(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port, config=ServerConfig(connect_delay=5.0))
    try:
        events = []
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_status=lambda e: events.append(e),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        await asyncio.sleep(0.1)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert not any(e.type == p.MessageType.SESSION_BEGUN for e in events)
        assert stopped[0].reason == p.StopReason.REQUESTED
    finally:
        await stop_capture_server(running)


async def test_begin_rejected_busy_for_different_session_id(unused_tcp_port):
    """Drive both Begins from one raw socket (never through CaptureClient,
    which owns its own exclusive receive loop and cannot safely share reads
    with test code on the same connection)."""
    import json

    from capture_server.protocol import (
        PROTOCOL_VERSION,
        BeginRequest,
        MessageType,
        TrackingConfig,
        effective_config,
    )

    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            first = BeginRequest(tracking=TrackingConfig(sources=(p.SourceType.HEAD,)))
            await ws.send(
                json.dumps(
                    {
                        "type": int(MessageType.BEGIN),
                        "version": PROTOCOL_VERSION,
                        "session_id": first.session_id or "session-a",
                        **effective_config(first).to_json(),
                    }
                )
            )
            init_raw = await asyncio.wait_for(ws.recv(), timeout=2)
            assert isinstance(p.parse_server_message(json.loads(init_raw)), p.SessionInit)

            other = BeginRequest(tracking=TrackingConfig(sources=(p.SourceType.HEAD,)))
            await ws.send(
                json.dumps(
                    {
                        "type": int(MessageType.BEGIN),
                        "version": PROTOCOL_VERSION,
                        "session_id": "a-different-session-id",
                        **effective_config(other).to_json(),
                    }
                )
            )
            # Session A's own lifecycle messages (SessionStarting, ...) keep
            # arriving interleaved on the same connection; skip them until
            # the rejection for session B's Begin shows up.
            rejected = None
            for _ in range(50):
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
                if isinstance(raw, bytes):
                    continue  # a binary tracking/image frame, not a JSON event
                event = p.parse_server_message(json.loads(raw))
                if isinstance(event, p.RequestRejected):
                    rejected = event
                    break
            assert rejected is not None, "RequestRejected for the busy Begin never arrived"
            assert rejected.reason == p.RejectionReason.BUSY
    finally:
        await stop_capture_server(running)


async def test_stop_for_unknown_session_is_rejected(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        import json

        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(
                json.dumps({"type": int(p.MessageType.STOP), "version": p.PROTOCOL_VERSION, "session_id": "nope"})
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            rejected = p.parse_server_message(json.loads(raw))
            assert isinstance(rejected, p.RequestRejected)
            assert rejected.reason == p.RejectionReason.UNKNOWN_SESSION
    finally:
        await stop_capture_server(running)


async def test_malformed_json_closes_with_1002(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send("not valid json{{{")
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
    finally:
        await stop_capture_server(running)


async def test_binary_message_from_client_closes_with_1002(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(b"\x00\x01\x02\x03")
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
    finally:
        await stop_capture_server(running)


async def test_unexpected_message_type_closes_with_1002(unused_tcp_port):
    import json

    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await ws.send(json.dumps({"type": int(p.MessageType.SESSION_INIT), "version": 1}))
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
    finally:
        await stop_capture_server(running)


async def test_begin_with_overlong_session_id_closes_with_1002(unused_tcp_port):
    """Begin's envelope validation must apply the exact same canonical
    session_id rule Stop already does (see protocol.require_envelope) --
    an overlong ID is an ENVELOPE problem (closes 1002), not a
    domain-level configuration issue (RequestRejected)."""
    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            overlong_id = "s" * (p._MAX_SESSION_ID_LEN + 1)
            begin = p.BeginRequest(session_id=None, tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": overlong_id,
                        **p.effective_config(begin).to_json(),
                    }
                )
            )
            with pytest.raises(ConnectionClosedError) as excinfo:
                await asyncio.wait_for(ws.recv(), timeout=2)
            assert excinfo.value.rcvd.code == 1002
    finally:
        await stop_capture_server(running)


async def test_identical_begin_replay_resends_cached_session_init(unused_tcp_port):
    """An identical-ID Begin replay must replay the cached SessionInit
    plus the current progress/established status -- not silently do
    nothing."""
    import json

    from capture_server.protocol import (
        PROTOCOL_VERSION,
        BeginRequest,
        MessageType,
        TrackingConfig,
        effective_config,
    )

    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            begin = BeginRequest(
                session_id="replay-session", tracking=TrackingConfig(sources=(p.SourceType.HEAD,))
            )
            begin_json = {
                "type": int(MessageType.BEGIN),
                "version": PROTOCOL_VERSION,
                "session_id": begin.session_id,
                **effective_config(begin).to_json(),
            }
            await ws.send(json.dumps(begin_json))
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            original_init = p.parse_server_message(json.loads(raw))
            assert isinstance(original_init, p.SessionInit)

            # Identical replay: never a RequestRejected, and the SAME
            # SessionInit content replayed, not silently ignored. Other
            # legitimate lifecycle messages (SessionStarting progress) can
            # still arrive interleaved, so scan for it rather than assuming
            # it's the very next message.
            await ws.send(json.dumps(begin_json))
            replayed_init = None
            for _ in range(50):
                raw2 = await asyncio.wait_for(ws.recv(), timeout=2)
                if isinstance(raw2, bytes):
                    continue
                event = p.parse_server_message(json.loads(raw2))
                assert not isinstance(event, p.RequestRejected), "identical replay must not be rejected"
                if isinstance(event, p.SessionInit):
                    replayed_init = event
                    break
            assert replayed_init is not None, "replayed SessionInit never arrived"
            assert replayed_init.to_json() == original_init.to_json()
    finally:
        await stop_capture_server(running)


async def _begin_and_stop_on_one_connection(ws, session_id: str) -> None:
    """Drive one full Begin -> ... -> Stop -> SessionStopped cycle on an
    already-open raw socket, skipping intermediate lifecycle/data messages,
    so the connection (and its _last_completed cache) stays the SAME one
    throughout -- completed-ID tracking is scoped to the current owner
    connection, not global across different connections."""
    import json

    from capture_server.protocol import (
        PROTOCOL_VERSION,
        BeginRequest,
        MessageType,
        TrackingConfig,
        effective_config,
    )

    begin = BeginRequest(session_id=session_id, tracking=TrackingConfig(sources=(p.SourceType.HEAD,)))
    await ws.send(
        json.dumps(
            {
                "type": int(MessageType.BEGIN),
                "version": PROTOCOL_VERSION,
                "session_id": begin.session_id,
                **effective_config(begin).to_json(),
            }
        )
    )
    await ws.send(
        json.dumps({"type": int(MessageType.STOP), "version": PROTOCOL_VERSION, "session_id": session_id})
    )
    for _ in range(50):
        raw = await asyncio.wait_for(ws.recv(), timeout=2)
        if isinstance(raw, bytes):
            continue
        event = p.parse_server_message(json.loads(raw))
        if isinstance(event, p.SessionStopped):
            return
    raise AssertionError("SessionStopped never arrived")


async def test_reused_completed_session_id_is_rejected(unused_tcp_port):
    import json

    from capture_server.protocol import (
        PROTOCOL_VERSION,
        BeginRequest,
        MessageType,
        TrackingConfig,
        effective_config,
    )

    running = await start_capture_server(unused_tcp_port)
    try:
        session_id = "reuse-me"
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await _begin_and_stop_on_one_connection(ws, session_id)

            begin = BeginRequest(session_id=session_id, tracking=TrackingConfig(sources=(p.SourceType.HEAD,)))
            await ws.send(
                json.dumps(
                    {
                        "type": int(MessageType.BEGIN),
                        "version": PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **effective_config(begin).to_json(),
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            rejected = p.parse_server_message(json.loads(raw))
            assert isinstance(rejected, p.RequestRejected)
            assert rejected.reason == p.RejectionReason.INVALID_CONFIG
    finally:
        await stop_capture_server(running)


async def test_duplicate_stop_replays_cached_terminal_reply(unused_tcp_port):
    import json

    from capture_server.protocol import PROTOCOL_VERSION, MessageType

    running = await start_capture_server(unused_tcp_port)
    try:
        session_id = "dup-stop-session"
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            await _begin_and_stop_on_one_connection(ws, session_id)

            await ws.send(
                json.dumps({"type": int(MessageType.STOP), "version": PROTOCOL_VERSION, "session_id": session_id})
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            replayed = p.parse_server_message(json.loads(raw))
            assert isinstance(replayed, p.SessionStopped)
            assert replayed.session_id == session_id
            assert replayed.reason == p.StopReason.REQUESTED
    finally:
        await stop_capture_server(running)


async def test_provider_recovered_fault_reports_normal_status(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port, ic_faults={"provider_recovered"})
    try:
        statuses = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            plugins=(p.PluginType.MANUS,),
            on_status=lambda e: statuses.append(e),
            on_stopped=lambda r: finished.set(),
        )
        await asyncio.sleep(0.2)
        begun = [e for e in statuses if e.status is not None]
        assert begun
        assert any(pr.plugin == p.PluginType.MANUS for pr in begun[0].status.providers), (
            "Manus was explicitly requested -- it must appear in the inventory"
        )
        assert all(pr.status == p.ProviderState.AVAILABLE for pr in begun[0].status.providers)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        await stop_capture_server(running)


async def test_sample_freeze_fault_repeats_tracking_values(unused_tcp_port):
    running = await start_capture_server(unused_tcp_port, ic_faults={"sample_freeze"})
    try:
        frames = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_capture_frame=lambda f: frames.append(f),
            on_stopped=lambda r: finished.set(),
        )
        await asyncio.sleep(0.3)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)

        assert len(frames) >= 2
        positions = {tuple(f.payload[1:13]) for f in frames}
        assert len(positions) == 1, "frozen head position bytes should be identical across frames"
    finally:
        await stop_capture_server(running)


async def test_race_stop_never_schedules_new_source_work_once_stop_is_already_requested():
    """_race_stop must check the stop gate BEFORE
    scheduling any new source coroutine -- not schedule it and then
    immediately abandon it in the background. Deterministic: stop is
    already requested BEFORE _race_stop is ever called, so the coroutine
    passed to it must never actually start running its body at all."""
    from capture_server.server import _Session, _StopDuringStartup

    started = False

    async def _never_runs() -> None:
        nonlocal started
        started = True

    begin = p.BeginRequest(session_id="race-stop-session", tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    session = _Session(begin, (), object(), 0.0, p.CaptureMode.MOCK, None, None)
    session.request_stop()  # stop is already known before _race_stop runs

    with pytest.raises(_StopDuringStartup):
        await session._race_stop(_never_runs())

    assert started is False, "the coroutine must never have been scheduled/started at all"


async def test_blocked_source_entry_still_lets_stop_end_the_session(unused_tcp_port):
    """``_Session._race_stop`` must stop WAITING on a blocked ``enter()``
    the moment Stop is requested, even though it cannot force the
    underlying call itself to return -- see ThreadRunner's docstrings for
    why that distinction is unavoidable. Also confirms the
    consequence of that: ``_finalize``'s subsequent ``close()`` cannot
    confirm the still-stuck worker thread genuinely released anything
    within its bound, so ``cleanup_ok`` must come back False and the
    terminal reason must be IC_ERROR, never REQUESTED -- a timed-out wait
    is not proof of safe cleanup."""
    entry_started = threading.Event()
    # A releasable gate, not a fixed sleep -- Python cannot forcibly kill
    # a thread (see ThreadRunner.close's docstring), so the background
    # thread stays genuinely blocked only until the test's own `finally`
    # releases it, rather than leaving a timed sleep running on past the
    # test (and than requiring that duration to "outlast" every bound
    # below, which was a fragile timing assumption).
    release_entry = threading.Event()

    class _BlockedEntryTracking:
        def __init__(self) -> None:
            # Shrink the release-confirmation bound so this test stays fast.
            self._runner = ThreadRunner(release_timeout_s=0.5)

        async def enter(self) -> None:
            def _block() -> None:
                # Signals genuine entry into the blocking call BEFORE
                # blocking, so the test can wait for real in-progress
                # blocking, not just "the executor job was scheduled".
                entry_started.set()
                release_entry.wait()

            await self._runner.run(_block)

        async def next_tracking_sample(self) -> dict[str, Any]:
            raise AssertionError("never reached: entry never completed")

        async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
            return p.StatusSnapshot(updated_at_ns=updated_at_ns)

        async def close(self) -> bool:
            return await self._runner.close(None)

    server = CaptureServer(
        ServerConfig(port=unused_tcp_port),
        source_factory=lambda begin, cameras: Acquisition(tracking=_BlockedEntryTracking(), cameras=()),
    )
    asyncio.ensure_future(server.run())
    for _ in range(200):
        if server._ws_server is not None:
            break
        await asyncio.sleep(0.01)
    try:
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        # Wait for genuine evidence that enter() is actually inside its
        # blocking call before stopping -- otherwise this could pass via
        # the unrelated pre-entry Stop path without ever proving
        # interruption of an already-in-progress blocked call.
        await asyncio.wait_for(asyncio.to_thread(entry_started.wait, 5), timeout=5)
        assert entry_started.is_set()
        await client.stop()
        # Bounded well under the entry gate's own lifetime (which doesn't
        # resolve until this test's own finally releases it) -- if
        # _race_stop regressed back to a direct `await self._source.enter()`,
        # this would hang instead of completing quickly (close()'s own
        # bound, not entry's, dominates: race_stop itself returns almost
        # immediately).
        await asyncio.wait_for(finished.wait(), timeout=5)
        # The worker thread was still inside enter() when close() ran and
        # gave up waiting (its shortened 0.5s bound) -- that is a timeout,
        # not confirmed release, so this must be IC_ERROR, never REQUESTED.
        assert stopped[0].reason == p.StopReason.IC_ERROR
    finally:
        release_entry.set()  # let the stuck worker thread actually finish
        server.request_stop()
        await server.close()


async def test_completed_session_tombstones_are_bounded(unused_tcp_port, monkeypatch):
    """Completed-session-ID memory must evict the oldest entry once past
    its bound, not grow unboundedly or remember only the single most
    recent completion."""
    import capture_server.server as server_module

    monkeypatch.setattr(server_module, "_COMPLETED_TOMBSTONE_CAPACITY", 2)
    running = await start_capture_server(unused_tcp_port)
    try:
        # Drive raw Begin/Stop cycles on one shared connection directly
        # (rather than one CaptureClient per cycle, which would each open
        # its own connection) so all three completions land in this one
        # CaptureServer's _completed window.
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            completed_ids = []
            for i in range(3):
                begin = p.BeginRequest(
                    session_id=f"tombstone-{i}",
                    tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
                )
                config = p.effective_config(begin)
                await ws.send(
                    json.dumps(
                        {
                            "type": int(p.MessageType.BEGIN),
                            "version": p.PROTOCOL_VERSION,
                            "session_id": begin.session_id,
                            **config.to_json(),
                        }
                    )
                )
                # Drain until SessionBegun (type 4), then Stop and drain to SessionStopped (type 6).
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    if json.loads(raw).get("type") == int(p.MessageType.SESSION_BEGUN):
                        break
                await ws.send(
                    json.dumps(
                        {
                            "type": int(p.MessageType.STOP),
                            "version": p.PROTOCOL_VERSION,
                            "session_id": begin.session_id,
                        }
                    )
                )
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    if json.loads(raw).get("type") == int(p.MessageType.SESSION_STOPPED):
                        break
                completed_ids.append(begin.session_id)

            # The oldest of the three (capacity 2) must have been evicted:
            # retrying its own (now-forgotten) session_id is no longer
            # recognized as a completed one, so it's accepted as fresh
            # rather than rejected as reused.
            evicted_id = completed_ids[0]
            begin = p.BeginRequest(session_id=evicted_id, tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
            config = p.effective_config(begin)
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            assert json.loads(raw).get("type") == int(p.MessageType.SESSION_INIT)
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.STOP),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": evicted_id,
                    }
                )
            )
    finally:
        await stop_capture_server(running)


async def test_new_owner_cannot_replay_or_stop_a_stale_owners_session(unused_tcp_port):
    """A session left active (admission-blocking) after its owner
    disconnects with a failed resource release must not be
    replayable/stoppable by a DIFFERENT, later-connecting owner that
    happens to reuse its session_id -- owner identity, not just
    session_id, must gate replay/Stop matching."""

    class _FailingCloseTracking:
        """Wraps the real mock tracking adapter so enter()/
        next_tracking_sample()/status() behave normally -- only close()
        is overridden to simulate a resource release that always fails."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def enter(self) -> None:
            await self._inner.enter()

        async def next_tracking_sample(self) -> dict[str, Any]:
            return await self._inner.next_tracking_sample()

        async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
            return await self._inner.status(updated_at_ns)

        async def close(self) -> bool:
            raise RuntimeError("simulated: resource release always fails")

    shared_session_id = "reused-session-id-across-owners"
    _real_factory = build_mock_source_factory(scenario="demo", seed=1)

    def _failing_close_source_factory(begin: p.BeginRequest, cameras: Any) -> Acquisition:
        real = _real_factory(begin, cameras)
        return Acquisition(tracking=_FailingCloseTracking(real.tracking), cameras=real.cameras)

    server = CaptureServer(
        ServerConfig(port=unused_tcp_port),
        source_factory=_failing_close_source_factory,
    )
    asyncio.ensure_future(server.run())
    for _ in range(200):
        if server._ws_server is not None:
            break
        await asyncio.sleep(0.01)
    try:
        # Owner A: start the session, then disconnect without a clean Stop
        # -- _handle_connection's finally still runs request_stop()+cleanup,
        # which fails (close() raises), leaving it the active session.
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws_a:
            begin = p.BeginRequest(
                session_id=shared_session_id,
                tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
            )
            config = p.effective_config(begin)
            await ws_a.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )
            while True:
                raw = await asyncio.wait_for(ws_a.recv(), timeout=5)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                if json.loads(raw).get("type") == int(p.MessageType.SESSION_BEGUN):
                    break
        # ws_a is now closed (the `async with` block exited); give the
        # server's own disconnect handling a moment to run its (failing)
        # cleanup.
        for _ in range(200):
            if server._active_session is not None and server._active_session.done.is_set():
                break
            await asyncio.sleep(0.01)
        assert server._active_session is not None, "failed release must keep admission blocked"
        assert server._active_session.cleanup_ok is False

        # Owner B: a brand new connection reusing the exact same session_id.
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws_b:
            await ws_b.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )
            raw = await asyncio.wait_for(ws_b.recv(), timeout=5)
            event = json.loads(raw)
            # Must be rejected as BUSY (admission still station-wide
            # blocked), never treated as a replay of owner A's session.
            assert event.get("type") == int(p.MessageType.REQUEST_REJECTED)
            assert event.get("reason") == int(p.RejectionReason.BUSY)

            await ws_b.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.STOP),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                    }
                )
            )
            raw = await asyncio.wait_for(ws_b.recv(), timeout=5)
            event = json.loads(raw)
            # Must never receive owner A's stale terminal reply.
            assert event.get("type") == int(p.MessageType.REQUEST_REJECTED)
            assert event.get("reason") == int(p.RejectionReason.UNKNOWN_SESSION)
    finally:
        server.request_stop()
        await server.close()


async def test_mock_acquisition_status_reflects_accepted_cameras():
    """Status must build its camera inventory from the cameras actually
    accepted for THIS session, not a fixed default. A
    camera_optional_failure poll failure degrades whichever camera it
    actually hits -- the mock has no notion of required/optional (mock
    camera failure has no knowledge of the protocol's required-camera
    StopReason); the required/optional distinction is a
    session-termination CONSEQUENCE decided by Acquisition.enter (see
    test_named_fault_produces_correct_stop_reason's camera_required_failure
    case), not a difference in whether a given poll degrades."""
    cameras = (
        p.CameraStreamConfig(
            stream_id=1,
            name="cam_required",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=1280,
            height=720,
            fps=30.0,
            jpeg_quality=80,
        ),
        p.CameraStreamConfig(
            stream_id=2,
            name="cam_optional",
            type=p.CameraType.OAK_D_COLOR,
            required=False,
            width=1280,
            height=720,
            fps=30.0,
            jpeg_quality=80,
        ),
    )
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))

    factory = build_mock_source_factory(scenario="demo", seed=1)
    acq = factory(begin, cameras)
    await acq.enter()
    status = await acq.status(1)
    assert {(c.name, c.stream_id) for c in status.cameras} == {("cam_required", 1), ("cam_optional", 2)}
    assert all(c.status == p.DeviceState.CONNECTED for c in status.cameras)
    await acq.close()

    faulty_factory = build_mock_source_factory(scenario="demo", seed=1, camera_faults={"camera_optional_failure"})
    faulty_acq = faulty_factory(begin, cameras)
    await faulty_acq.enter()
    for camera in faulty_acq.cameras:
        try:
            await camera.next_camera_jpeg()
        except Exception:
            pass
    by_name = {c.name: c for c in (await faulty_acq.status(1)).cameras}
    assert by_name["cam_required"].status == p.DeviceState.DEGRADED
    assert by_name["cam_optional"].status == p.DeviceState.DEGRADED
    await faulty_acq.close()


async def test_mock_acquisition_status_omits_manus_unless_requested():
    """Without a Manus request, the server does not launch Manus merely
    because HANDS was selected -- a missing/unrequested plugin has NO
    inventory entry at all, not a synthesized disconnected one."""
    factory = build_mock_source_factory(scenario="demo", seed=1)

    without_manus = factory(p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HANDS,))), ())
    await without_manus.enter()
    status = await without_manus.status(1)
    assert not any(pr.plugin == p.PluginType.MANUS for pr in status.providers)
    manus_provider_ids = {pr.id for pr in status.providers if pr.plugin == p.PluginType.MANUS}
    assert not any(d.provider_id in manus_provider_ids for d in status.devices)
    await without_manus.close()

    with_manus = factory(
        p.BeginRequest(
            tracking=p.TrackingConfig(sources=(p.SourceType.HANDS,)),
            plugins=(p.PluginRequest(plugin=p.PluginType.MANUS, enabled=True),),
        ),
        (),
    )
    await with_manus.enter()
    status_with = await with_manus.status(1)
    assert any(pr.plugin == p.PluginType.MANUS for pr in status_with.providers)
    await with_manus.close()


async def test_mock_source_factory_respects_disabled_manus_request():
    """A Manus plugin request with enabled=False must not make Manus
    appear in status either -- "enabled: True permits launching the
    requested plugin; false skips it.\" Retargeted to the acquisition
    boundary: Manus presence is now keyed to whether the session selected
    HANDS, not to a separate manus_requested flag -- CaptureServer only
    ever forwards an enabled plugin request to the source_factory in the
    first place (see CaptureServer._handle_begin), so a disabled request
    reaching here is equivalent to no HANDS selection at all."""
    factory = build_mock_source_factory(scenario="demo", seed=1)
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    acq = factory(begin, ())
    assert acq.tracking is None


async def test_terminal_reply_preserves_status_observed_before_a_pre_begun_failure(
    unused_tcp_port,
):
    """_last_sent_status must be updated uniformly at every
    status read, not only once SessionBegun is reached -- otherwise a
    failure between STARTING_SOURCES and Begun loses the last genuinely
    observed native status from the terminal reply entirely."""

    class _FailsFirstTrackingStepTracking:
        async def enter(self) -> None:
            pass

        async def next_tracking_sample(self) -> dict[str, Any]:
            raise RuntimeError("simulated: first tracking step fails before Begun")

        async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
            return p.StatusSnapshot(updated_at_ns=updated_at_ns)

        async def close(self) -> bool:
            return True

    server = CaptureServer(
        ServerConfig(port=unused_tcp_port),
        source_factory=lambda begin, cameras: Acquisition(tracking=_FailsFirstTrackingStepTracking(), cameras=()),
    )
    asyncio.ensure_future(server.run())
    for _ in range(200):
        if server._ws_server is not None:
            break
        await asyncio.sleep(0.01)
    try:
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert stopped[0].reason == p.StopReason.IC_ERROR
        assert stopped[0].status is not None, (
            "status observed during STARTING_SOURCES must survive into the terminal reply"
        )
    finally:
        server.request_stop()
        await server.close()


async def test_quiesce_waits_for_held_send_then_discards_data_queued_behind_it():
    """A slow in-flight binary send must hold the quiesce
    acknowledgment pending for its entire duration -- not resolve early --
    and whatever frames/status got queued BEHIND it while held must be
    discarded, never sent, once disposal finally runs."""
    from capture_server.server import _Outbox

    send_started = asyncio.Event()
    release_send = asyncio.Event()
    sent_binaries: list[bytes] = []
    on_sent_calls: list[int] = []

    async def _send_json(data: dict[str, Any]) -> None:
        pass

    async def _send_binary(message: bytes) -> None:
        if not send_started.is_set():
            send_started.set()
            await release_send.wait()  # hold exactly the first send in flight
        sent_binaries.append(message)

    async def _close_transport() -> None:
        pass

    outbox = _Outbox(send_json=_send_json, send_binary=_send_binary, close_transport=_close_transport)
    outbox.bind_session([1], on_sent=on_sent_calls.append, on_fatal=lambda msg: None)
    outbox.start()
    try:
        outbox.publish_frame(1, b"frame-0")
        await asyncio.wait_for(send_started.wait(), timeout=1)
        outbox.publish_frame(1, b"frame-1")  # queued behind the held send
        outbox.publish_status({"n": "status-1"})  # also queued behind it

        quiesce_task = asyncio.ensure_future(outbox.quiesce())
        await asyncio.sleep(0.05)
        assert not quiesce_task.done(), "the ack must stay pending for the whole held-send duration"

        release_send.set()
        await asyncio.wait_for(quiesce_task, timeout=1)

        assert on_sent_calls == [1], "exactly one completion, for the item that was genuinely in flight"
        assert sent_binaries == [b"frame-0"], "frame-1, queued behind the held send, must never be sent"
        assert outbox._latest_status is None, "status queued behind the held send must be discarded too"
    finally:
        release_send.set()
        await outbox.stop_and_join()


async def test_held_send_failure_during_quiesce_fires_on_fatal_exactly_once():
    """If the held send ultimately FAILS (not just completes),
    exactly one on_fatal notification must fire -- not one per item that
    was queued behind it -- and the connection must reject any further
    publish after that (no reuse of a connection whose writer has died)."""
    from capture_server.server import _Outbox

    send_started = asyncio.Event()
    release_send = asyncio.Event()
    fatal_messages: list[str] = []
    on_sent_calls: list[int] = []

    async def _send_json(data: dict[str, Any]) -> None:
        pass

    async def _send_binary(message: bytes) -> None:
        send_started.set()
        await release_send.wait()
        raise OSError("simulated: held send ultimately fails")

    async def _close_transport() -> None:
        pass

    outbox = _Outbox(send_json=_send_json, send_binary=_send_binary, close_transport=_close_transport)
    outbox.bind_session([1], on_sent=on_sent_calls.append, on_fatal=fatal_messages.append)
    outbox.start()
    try:
        outbox.publish_frame(1, b"frame-0")  # becomes the held, ultimately-failing send
        await asyncio.wait_for(send_started.wait(), timeout=1)
        outbox.publish_frame(1, b"frame-1")  # queued behind it -- must never be sent
        outbox.publish_status({"n": "status-1"})

        quiesce_task = asyncio.ensure_future(outbox.quiesce())
        await asyncio.sleep(0.05)
        assert not quiesce_task.done()
        assert fatal_messages == []

        release_send.set()
        await asyncio.wait_for(quiesce_task, timeout=1)

        assert on_sent_calls == [], "the held send failed -- on_sent must never fire for it"
        assert len(fatal_messages) == 1, "exactly one on_fatal, regardless of how much was queued behind it"
        assert outbox.closed is True
        assert outbox.publish_control({"n": "after-failure"}) is False, (
            "no further reuse of a connection whose writer has already died"
        )
    finally:
        release_send.set()
        await outbox.stop_and_join()


async def test_quiesce_after_writer_already_exited_resolves_immediately():
    """quiesce() called AFTER the
    writer has already exited (e.g. an earlier send failed and closed the
    connection) must resolve immediately, never hang waiting on a writer
    that will never run again."""
    from capture_server.server import _Outbox

    async def _send_json(data: dict[str, Any]) -> None:
        raise OSError("simulated: immediate send failure")

    async def _send_binary(message: bytes) -> None:
        pass

    close_called = asyncio.Event()

    async def _close_transport() -> None:
        close_called.set()

    outbox = _Outbox(send_json=_send_json, send_binary=_send_binary, close_transport=_close_transport)
    outbox.bind_session([], on_sent=lambda sid: None, on_fatal=lambda msg: None)
    outbox.start()
    try:
        assert outbox.publish_control({"n": 0}) is True
        await asyncio.wait_for(close_called.wait(), timeout=1)  # the writer has already exited
        await asyncio.wait_for(outbox.quiesce(), timeout=1)
    finally:
        await outbox.stop_and_join()


async def test_repeated_quiesce_calls_are_idempotent():
    """Calling quiesce() more than
    once against the same bound attempt must not re-arm an
    already-resolved acknowledgment and wait anew -- every call after the
    first must also resolve immediately."""
    from capture_server.server import _Outbox

    async def _send_json(data: dict[str, Any]) -> None:
        pass

    async def _send_binary(message: bytes) -> None:
        pass

    async def _close_transport() -> None:
        pass

    outbox = _Outbox(send_json=_send_json, send_binary=_send_binary, close_transport=_close_transport)
    outbox.bind_session([], on_sent=lambda sid: None, on_fatal=lambda msg: None)
    outbox.start()
    try:
        await asyncio.wait_for(outbox.quiesce(), timeout=1)
        await asyncio.wait_for(outbox.quiesce(), timeout=1)
        await asyncio.wait_for(outbox.quiesce(), timeout=1)
    finally:
        await outbox.stop_and_join()


async def test_outbox_control_overflow_closes_connection_not_just_session():
    """Control-queue overflow must close the whole connection
    (the same fatal condition as any other send failure), not merely end
    whichever session happens to be active -- drive the Outbox directly
    (not through a real socket) so backlog can be forced deterministically
    by holding the very first send in flight. The overflow itself is
    recorded immediately (no further enqueue succeeds), but the actual
    on_fatal/close notification is deferred until that held send resolves
    -- the same "a send already in flight is allowed to finish" allowance
    as every other fatal condition, never an abort mid-flight."""
    from capture_server.server import _CONTROL_QUEUE_CAPACITY, _Outbox

    close_called = asyncio.Event()
    fatal_messages: list[str] = []
    release_first_send = asyncio.Event()

    async def _send_json(data: dict[str, Any]) -> None:
        await release_first_send.wait()  # held "in flight" until the test releases it

    async def _close_transport() -> None:
        close_called.set()

    outbox = _Outbox(
        send_json=_send_json,
        send_binary=lambda message: asyncio.sleep(0),
        close_transport=_close_transport,
    )
    outbox.bind_session([], on_sent=lambda sid: None, on_fatal=fatal_messages.append)
    outbox.start()
    try:
        assert outbox.publish_control({"n": 0}) is True
        await asyncio.sleep(0.05)  # let the writer actually dequeue item 0 into the held send
        last_result = True
        for i in range(1, _CONTROL_QUEUE_CAPACITY + 2):
            last_result = outbox.publish_control({"n": i})
        assert last_result is False, "enqueue past capacity must report failure"
        assert outbox.closed is True
        assert fatal_messages == [], "notification must wait for the held send, not fire mid-flight"
        release_first_send.set()
        await asyncio.wait_for(close_called.wait(), timeout=1)
        assert fatal_messages and "overflow" in fatal_messages[-1]
    finally:
        release_first_send.set()
        await outbox.stop_and_join()


async def test_produced_equals_sent_plus_dropped_across_a_genuine_eviction():
    """Confirmed final counters satisfy produced = sent + dropped.
    Exercises the exact mechanism _Session's own counters rely
    on (_Outbox.publish_frame's bounded drop-oldest queue, on_sent) to
    force a GENUINE admitted-then-evicted drop, not just the trivially-
    true `dropped := produced - sent` identity on a run that never
    actually overflows -- distinct from an upstream source omission (a
    None poll, which this package's producer loops never even pass to
    publish_frame/produced in the first place, see _produce_camera's
    ``if result is None: continue``): only an item that WAS admitted
    (produced incremented, handed to the outbox) and then genuinely
    evicted by the bounded queue counts as dropped."""
    from capture_server.server import _QUEUE_CAPACITY, _Outbox

    sent_ids: list[int] = []
    fatal_messages: list[str] = []
    release_first_send = asyncio.Event()

    async def _send_binary(_message: bytes) -> None:
        await release_first_send.wait()  # held "in flight" until the test releases it

    outbox = _Outbox(
        send_json=lambda data: asyncio.sleep(0),
        send_binary=_send_binary,
        close_transport=lambda: asyncio.sleep(0),
    )
    outbox.bind_session([0], on_sent=sent_ids.append, on_fatal=fatal_messages.append)
    outbox.start()
    try:
        produced = 0

        def _produce() -> None:
            nonlocal produced
            outbox.publish_frame(0, b"frame")
            produced += 1

        _produce()  # dequeued into the held send below
        await asyncio.sleep(0.05)  # let the writer actually pick it up and block on it

        # Publish more than the per-stream queue's capacity while the
        # writer is stuck -- the oldest of these must be evicted.
        extra = _QUEUE_CAPACITY + 2
        for _ in range(extra):
            _produce()

        release_first_send.set()
        # The writer now drains: the one held send, plus whatever survived
        # eviction in the bounded queue (capacity _QUEUE_CAPACITY).
        expected_sent = 1 + _QUEUE_CAPACITY
        for _ in range(100):
            if len(sent_ids) >= expected_sent:
                break
            await asyncio.sleep(0.01)
        assert len(sent_ids) == expected_sent
        assert fatal_messages == []

        sent = len(sent_ids)
        dropped = produced - sent  # the exact derivation _Session._finalize uses
        assert produced == sent + dropped  # holds by construction
        # The real point: dropped is not zero here -- a genuine eviction
        # happened, distinct from a vacuous run where nothing ever
        # overflowed.
        assert dropped == extra - _QUEUE_CAPACITY
        assert dropped > 0
    finally:
        release_first_send.set()
        await outbox.stop_and_join()


async def test_owner_disconnect_reports_owner_disconnected_reason(unused_tcp_port):
    """An owner disconnecting without a clean Stop must produce
    a terminal SessionStopped modeling OWNER_DISCONNECTED, not a bare
    REQUESTED -- checked via the connection's own completed-tombstone
    cache since the owner is, by construction, no longer connected to
    receive the reply itself."""
    running = await start_capture_server(unused_tcp_port)
    try:
        session_id = "owner-disconnect-session"
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            begin = p.BeginRequest(session_id=session_id, tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
            config = p.effective_config(begin)
            await ws.send(
                json.dumps(
                    {
                        "type": int(p.MessageType.BEGIN),
                        "version": p.PROTOCOL_VERSION,
                        "session_id": begin.session_id,
                        **config.to_json(),
                    }
                )
            )
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                if json.loads(raw).get("type") == int(p.MessageType.SESSION_BEGUN):
                    break
        # ws is now closed (no Stop was ever sent) -- the server's own
        # disconnect handling in _handle_connection's finally must have
        # requested the stop itself, with the modeled cause.
        for _ in range(200):
            if session_id in running.server._completed:
                break
            await asyncio.sleep(0.01)
        assert session_id in running.server._completed, "owner-disconnect cleanup never completed"
        assert running.server._completed[session_id].reason == p.StopReason.OWNER_DISCONNECTED
        # Normal requested Stop, clean shutdown and owner loss all use
        # null -- owner loss is a benign cause, not a diagnostic to
        # report as error_message.
        assert running.server._completed[session_id].error_message is None
    finally:
        await stop_capture_server(running)


async def test_server_shutdown_reports_server_shutdown_reason(unused_tcp_port):
    """CaptureServer.close() must stop an active session with a
    modeled SERVER_SHUTDOWN cause, not a bare REQUESTED."""
    running = await start_capture_server(unused_tcp_port)
    try:
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        running.server.request_stop()
        await running.server.close()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert stopped[0].reason == p.StopReason.SERVER_SHUTDOWN
        # Normal requested Stop, clean shutdown and owner loss all use
        # null -- a clean shutdown is a benign cause, not a diagnostic to
        # report as error_message.
        assert stopped[0].error_message is None
    finally:
        await stop_capture_server(running)


async def test_new_connection_rejected_during_shutdown(unused_tcp_port):
    """Admission must close immediately once request_stop() has
    been called, even before close() gets around to actually closing the
    listener -- otherwise a brand new connection could slip in through
    that gap."""
    running = await start_capture_server(unused_tcp_port)
    running.server.request_stop()
    try:
        with pytest.raises(ConnectionClosedError) as excinfo:
            async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
                await ws.recv()
        assert excinfo.value.rcvd.code == 1012
    finally:
        await running.server.close()


async def test_close_always_closes_listener_even_if_session_stop_times_out(unused_tcp_port, monkeypatch):
    """close() must close the listener regardless of whether
    waiting for the active session to settle times out -- previously an
    uncaught TimeoutError from that wait skipped _ws_server.close()
    entirely, leaving the listener (and the port) held open."""
    import capture_server.server as server_module

    class _NeverSettlesSession:
        def __init__(self) -> None:
            self.done = asyncio.Event()  # deliberately never set

        def request_stop(self, reason: p.StopReason | None = None, message: str | None = None) -> None:
            pass

    monkeypatch.setattr(server_module, "_SOURCE_RELEASE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(server_module, "_SEND_DEADLINE_S", 0.05)
    running = await start_capture_server(unused_tcp_port)
    running.server._active_session = _NeverSettlesSession()
    # Bounded well under what an uncaught-TimeoutError hang would look
    # like -- this is the regression check itself.
    await asyncio.wait_for(running.server.close(), timeout=2)
    with pytest.raises(OSError):
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}", open_timeout=1):
            pass


async def test_identical_begin_replay_after_begun_also_replays_current_status(unused_tcp_port):
    """Replaying an identical Begin against an already-Begun
    session must resend the cached SessionBegun AND the session's current
    (last-observed) status as a SessionStatus -- not just the unchanged
    Begun alone."""
    running = await start_capture_server(unused_tcp_port)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{unused_tcp_port}") as ws:
            begin = p.BeginRequest(
                session_id="replay-status-session",
                tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)),
            )
            begin_json = {
                "type": int(p.MessageType.BEGIN),
                "version": p.PROTOCOL_VERSION,
                "session_id": begin.session_id,
                **p.effective_config(begin).to_json(),
            }
            await ws.send(json.dumps(begin_json))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                event = p.parse_server_message(json.loads(raw))
                if isinstance(event, p.SessionBegun):
                    break

            await ws.send(json.dumps(begin_json))
            replayed_begun = None
            replayed_status = None
            for _ in range(50):
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
                if isinstance(raw, (bytes, bytearray)):
                    continue
                event = p.parse_server_message(json.loads(raw))
                assert not isinstance(event, p.RequestRejected), "identical replay must not be rejected"
                if isinstance(event, p.SessionBegun) and replayed_begun is None:
                    replayed_begun = event
                    continue
                if replayed_begun is not None and isinstance(event, p.SessionStatus):
                    replayed_status = event
                    break
            assert replayed_begun is not None, "replayed SessionBegun never arrived"
            assert replayed_status is not None, "replayed SessionStatus never arrived"
            assert replayed_status.session_id == begin.session_id
    finally:
        await stop_capture_server(running)


async def test_source_failure_with_empty_exception_message_still_sends_terminal(unused_tcp_port):
    """A source failure whose exception has NO message text (e.g.
    ``RuntimeError()`` with no args) must still produce a valid,
    delivered terminal reply -- ``protocol.SessionStopped`` rejects an
    empty (but non-None) ``error_message``, so without normalizing to a
    nonempty fallback, ``_finalize``'s one terminal constructor would
    itself raise while building it, silently losing the terminal reply
    entirely (``on_stopped`` would simply never fire)."""

    class _EmptyMessageFailureTracking:
        async def enter(self) -> None:
            pass

        async def next_tracking_sample(self) -> dict[str, Any]:
            raise RuntimeError  # deliberately empty message: str(exc) == ""

        async def status(self, updated_at_ns: int) -> p.StatusSnapshot:
            return p.StatusSnapshot(updated_at_ns=updated_at_ns)

        async def close(self) -> bool:
            return True

    server = CaptureServer(
        ServerConfig(port=unused_tcp_port),
        source_factory=lambda begin, cameras: Acquisition(tracking=_EmptyMessageFailureTracking(), cameras=()),
    )
    asyncio.ensure_future(server.run())
    for _ in range(200):
        if server._ws_server is not None:
            break
        await asyncio.sleep(0.01)
    try:
        stopped = []
        finished = asyncio.Event()
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped.append(r), finished.set()),
        )
        # If the terminal reply were silently lost, this would time out --
        # on_stopped must still fire.
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert stopped[0].reason == p.StopReason.IC_ERROR
        # Guaranteed nonempty despite the underlying exception's own empty
        # str() -- falls back to the exception's type name.
        assert stopped[0].error_message
        assert "RuntimeError" in stopped[0].error_message
    finally:
        server.request_stop()
        await server.close()
