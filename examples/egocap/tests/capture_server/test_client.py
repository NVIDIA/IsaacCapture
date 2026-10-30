import asyncio
import dataclasses
import json

import numpy as np
import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from capture_server import client as c
from capture_server import protocol as p
from capture_server.client import (
    CaptureClient,
    CaptureStopped,
    RequestRejectedError,
    StoppedBeforeAcceptedError,
)
from capture_server.server import ServerConfig
from tests.support.capture_server import poll_until, start_capture_server, stop_capture_server

# ---------------------------------------------------------------------------
# End-to-end happy path and basic validation, against the real mock
# CaptureServer (its "demo" scenario, via the real_mock_server fixture).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_stop_happy_path_fires_callbacks_in_order(real_mock_server):
    events: list[str] = []
    frames: list[c.CaptureFrame] = []
    stopped_result: list[CaptureStopped] = []
    finished = asyncio.Event()

    client = CaptureClient(real_mock_server)
    started_streams: list[tuple] = []

    def on_started(event: c.CaptureStarted) -> None:
        events.append("init")
        assert event.running is False
        assert event.mode == p.CaptureMode.MOCK
        assert len(event.streams) == 1
        started_streams.append(event.streams)

    def on_status(event: c.CaptureStatus) -> None:
        if event.type == c.CaptureStatusType.STARTING:
            events.append("starting")
            assert event.running is False
        elif event.type == c.CaptureStatusType.BEGUN:
            events.append("begun")
            assert event.running is True

    def on_frame(frame: c.CaptureFrame) -> None:
        events.append("frame")
        frames.append(frame)

    def on_stopped(result: CaptureStopped) -> None:
        events.append("stopped")
        assert result.running is False
        stopped_result.append(result)
        finished.set()

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_started=on_started,
        on_status=on_status,
        on_capture_frame=on_frame,
        on_stopped=on_stopped,
    )
    assert events[0] == "init"  # on_started fires before start() returns

    await poll_until(lambda: len(frames) >= 1)  # wait for an actual real frame, not a timing guess
    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)
    await client.stop()  # idempotent once already stopped

    # At least one "starting" precedes "begun", which precedes at least one
    # real frame, which precedes the final "stopped" -- exact counts/timing
    # of real progress updates are the real server's business, not ours.
    assert "starting" in events
    assert events.index("starting") < events.index("begun")
    assert events.index("begun") < events.index("frame")
    assert events[-1] == "stopped"
    assert events.count("frame") >= 1

    frame = frames[0]
    assert frame.description is started_streams[0][0]  # exact same object, not a rebuilt copy
    assert isinstance(frame.payload, bytes)
    assert len(frame.payload) == frame.description.payload_size
    decoded = p.decode_tracking_frame(frame.payload)
    assert bool(decoded["head.present"]) is True

    result = stopped_result[0]
    assert result.reason == p.StopReason.REQUESTED
    assert result.error_message is None
    assert len(result.streams) == 1
    counters = result.streams[0]
    assert counters.stream_id == 0
    assert counters.produced == counters.sent  # the server's own accounting is always consistent
    assert counters.dropped == 0
    # Frames the server sent after this client requested Stop are suppressed
    # client-side (see test_process_message_discards_data_while_stopping),
    # so on_frame call count can legitimately trail the server's sent count.
    assert 1 <= events.count("frame") <= counters.sent


def test_capture_status_type_values_are_distinct_from_wire_message_type():
    # CaptureStatusType is an independent client-facing enum, deliberately
    # numbered differently from the wire MessageType it's derived from
    # (SessionStarting=3/SessionBegun=4/SessionStatus=5).
    assert int(c.CaptureStatusType.STARTING) == 1
    assert int(c.CaptureStatusType.BEGUN) == 2
    assert int(c.CaptureStatusType.STATUS_CHANGE) == 3


@pytest.mark.asyncio
async def test_capture_status_fields_match_type(real_mock_server):
    statuses = []
    begun_seen = asyncio.Event()
    finished = asyncio.Event()
    client = CaptureClient(real_mock_server)

    def on_status(event: c.CaptureStatus) -> None:
        statuses.append(event)
        if event.type == c.CaptureStatusType.BEGUN:
            begun_seen.set()

    await client.start(sources=(p.SourceType.HEAD,), on_status=on_status, on_stopped=lambda _: finished.set())
    await asyncio.wait_for(begun_seen.wait(), timeout=5)  # the real server reaches BEGUN asynchronously
    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)

    by_type = {s.type: s for s in statuses}
    starting = by_type[c.CaptureStatusType.STARTING]
    assert starting.running is False
    assert starting.startup_stage is not None
    assert starting.message is not None

    begun = by_type[c.CaptureStatusType.BEGUN]
    assert begun.running is True
    assert begun.startup_stage is None
    assert begun.message is None
    assert begun.status is not None


@pytest.mark.asyncio
async def test_stop_during_capture_ends_session(real_mock_server):
    finished = asyncio.Event()
    stopped_result: list[CaptureStopped] = []

    def on_stopped(result: CaptureStopped) -> None:
        stopped_result.append(result)
        finished.set()

    client = CaptureClient(real_mock_server)
    await client.start(sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS), on_stopped=on_stopped)
    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)

    assert stopped_result[0].reason == p.StopReason.REQUESTED


@pytest.mark.asyncio
async def test_start_while_active_raises_runtime_error(real_mock_server):
    finished = asyncio.Event()
    client = CaptureClient(real_mock_server)
    await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _: finished.set())

    with pytest.raises(RuntimeError, match="active attempt continues"):
        await client.start(sources=(p.SourceType.HEAD,))

    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)


@pytest.mark.asyncio
async def test_restart_after_completion_succeeds(real_mock_server):
    """A fresh start() after a
    prior attempt's on_stopped has fully fired must succeed with a new
    session, not be blocked by leftover state from the previous one."""
    client = CaptureClient(real_mock_server)
    first_finished = asyncio.Event()
    await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _: first_finished.set())
    first_session = client._attempt.session_id
    await client.stop()
    await asyncio.wait_for(first_finished.wait(), timeout=5)
    assert client._attempt is None

    second_finished = asyncio.Event()
    second_stopped: list[CaptureStopped] = []
    await client.start(
        sources=(p.SourceType.HEAD,),
        on_stopped=lambda r: (second_stopped.append(r), second_finished.set()),
    )
    assert client._attempt.session_id != first_session
    await client.stop()
    await asyncio.wait_for(second_finished.wait(), timeout=5)
    assert second_stopped[0].reason == p.StopReason.REQUESTED


@pytest.mark.asyncio
async def test_stop_is_idempotent_when_idle():
    client = CaptureClient("ws://127.0.0.1:1")
    await client.stop()  # no connection ever opened; must not raise
    await client.stop()


@pytest.mark.asyncio
async def test_concurrent_stop_calls_are_idempotent(real_mock_server):
    finished = asyncio.Event()
    client = CaptureClient(real_mock_server)
    await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _: finished.set())

    await asyncio.gather(client.stop(), client.stop())  # second call is a no-op
    await asyncio.wait_for(finished.wait(), timeout=5)


@pytest.mark.asyncio
async def test_second_connection_closes_1013_while_owner_active(real_mock_server):
    """The real server admits exactly one owner connection; a second TCP
    connection is closed with 1013 immediately, before any Begin is even
    read. BUSY-on-a-second-Begin-on-the-SAME-connection is a different,
    raw-server-policy case already covered by
    test_begin_rejected_busy_for_different_session_id in test_server.py --
    this test is specifically about CaptureClient's own public start() API
    surfacing that 1013 close as a real exception with the real code, not
    a fabricated one."""
    first_finished = asyncio.Event()
    first_client = CaptureClient(real_mock_server)
    await first_client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _: first_finished.set())
    try:
        second_client = CaptureClient(real_mock_server)
        with pytest.raises(ConnectionClosed) as excinfo:
            await second_client.start(sources=(p.SourceType.HEAD,))
        assert excinfo.value.rcvd.code == 1013
    finally:
        await first_client.stop()
        await asyncio.wait_for(first_finished.wait(), timeout=5)


@pytest.mark.asyncio
async def test_start_raises_request_rejected_error_for_unknown_camera(unused_tcp_port):
    """RequestRejectedError is CaptureClient's own public-API path for a
    RequestRejected arriving during the pre-acceptance handshake -- an
    unknown camera name is INVALID_CONFIG on the very first Begin, real
    behavior the server itself produces, not a fabricated rejection. Needs
    its own server (not the shared real_mock_server fixture) because this
    particular rejection only fires when the server is configured with a
    concrete known_cameras allowlist."""
    server = await start_capture_server(unused_tcp_port, config=ServerConfig(known_cameras=()))
    try:
        client = CaptureClient(f"ws://127.0.0.1:{unused_tcp_port}")
        with pytest.raises(RequestRejectedError) as excinfo:
            await client.start(
                sources=(p.SourceType.HEAD,),
                cameras=({"name": "no_such_camera", "type": p.CameraType.OAK_D_COLOR},),
            )
        assert excinfo.value.reason == p.RejectionReason.INVALID_CONFIG
        assert "unknown camera" in excinfo.value.error_message
    finally:
        await stop_capture_server(server)


def test_constructor_rejects_invalid_url():
    with pytest.raises(ValueError, match="ws://"):
        CaptureClient("not-a-url")
    with pytest.raises(ValueError, match="ws://"):
        CaptureClient("")


@pytest.mark.asyncio
async def test_start_rejects_invalid_selections():
    # Validated before any connection attempt, so no server is needed.
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match="invalid CaptureClient.start arguments"):
        await client.start(sources=(p.SourceType.HEAD, p.SourceType.HEAD))  # duplicate


@pytest.mark.asyncio
async def test_start_rejects_non_callable_callback():
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(ValueError, match="must be callable"):
        await client.start(sources=(p.SourceType.HEAD,), on_started="not-callable")


# ---------------------------------------------------------------------------
# Small pure-function/internal-helper coverage.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_awaits_an_async_callback():
    seen = []

    async def on_frame(value):
        await asyncio.sleep(0)
        seen.append(value)

    await c._invoke(on_frame, "frame-1")
    assert seen == ["frame-1"]
    await c._invoke(None, "frame-2")  # None callback is a no-op


def test_from_json_text_rejects_invalid_json():
    with pytest.raises(p.ProtocolError, match="not valid JSON"):
        c._from_json_text("not json")


def test_stream_registry_registers_camera_streams():
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    camera_stream = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[camera_stream],
    )
    registry = c._StreamRegistry.from_session_init(init_event)
    assert registry.cameras[1] is camera_stream


# ---------------------------------------------------------------------------
# _dispatch_binary / _process_message: direct, attempt-level unit coverage
# that doesn't need a real socket.
# ---------------------------------------------------------------------------


def _tracking_stream() -> p.TrackingStream:
    return p.tracking_stream_descriptor((p.SourceType.HEAD,), 60.0)


def _begun_attempt(**extra) -> c._Attempt:
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={})
    attempt.begun = True
    for key, value in extra.items():
        setattr(attempt, key, value)
    return attempt


@pytest.mark.asyncio
async def test_dispatch_binary_rejects_unregistered_camera():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(callbacks={"on_capture_image": lambda img: None})
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 100, 200)

    with pytest.raises(p.ProtocolError, match="unregistered camera"):
        await client._dispatch_binary(attempt, header, b"fake-jpeg-bytes")


@pytest.mark.asyncio
async def test_dispatch_binary_dispatches_a_registered_camera():
    seen = []
    stream = p.CameraStream(
        stream_id=1,
        kind=p.StreamKind.IMAGE,
        name="camera_left",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=4,
        height=3,
        fps=30.0,
        codec=p.ImageCodec.JPEG,
        jpeg_quality=90,
        orientation=p.ImageOrientation(),
    )
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(
        callbacks={"on_capture_image": lambda img: seen.append(img)},
        streams=c._StreamRegistry(cameras={1: stream}),
    )
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 100, 200)

    await client._dispatch_binary(attempt, header, b"fake-jpeg-bytes")

    assert len(seen) == 1
    assert seen[0].jpeg == b"fake-jpeg-bytes"
    assert seen[0].name == "camera_left"
    assert seen[0].expected_size == (4, 3)


# ---------------------------------------------------------------------------
# CameraImage.decode().
# ---------------------------------------------------------------------------


def test_camera_image_decode_round_trips_a_solid_color():
    from PIL import Image

    buf = __import__("io").BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(buf, format="JPEG", quality=100)
    jpeg_bytes = buf.getvalue()

    image = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=jpeg_bytes,
    )
    decoded = image.decode()
    assert decoded.shape == (3, 4, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_allclose(decoded[0, 0], [10, 20, 30], atol=5)
    decoded[0, 0, 0] = 255  # the returned array is writable

    sized = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=jpeg_bytes,
        expected_size=(4, 3),
    )
    assert sized.decode() is not None
    mismatched = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=jpeg_bytes,
        expected_size=(99, 99),
    )
    with pytest.raises(p.ProtocolError, match="expected"):
        mismatched.decode()


def test_camera_image_decode_rejects_invalid_jpeg():
    image = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=b"not a jpeg",
    )
    with pytest.raises(p.ProtocolError, match="not a valid JPEG"):
        image.decode()


def test_camera_image_decode_rejects_truncated_jpeg():
    from PIL import Image

    buf = __import__("io").BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(buf, format="JPEG", quality=100)
    truncated = buf.getvalue()[:-20]
    image = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=truncated,
    )
    with pytest.raises(p.ProtocolError):
        image.decode()


def test_camera_image_decode_rejects_non_jpeg_format():
    from PIL import Image

    buf = __import__("io").BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(buf, format="PNG")
    image = c.CameraImage(
        session_id="session-1",
        stream_id=1,
        sequence=0,
        capture_time_ns=100,
        receive_time_ns=200,
        name="camera_left",
        jpeg=buf.getvalue(),
    )
    with pytest.raises(p.ProtocolError, match="expected JPEG"):
        image.decode()


@pytest.mark.asyncio
async def test_process_message_rejects_binary_data_before_session_begun():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={})
    header = p.FrameMetadata(p.BinaryKind.CAPTURE_IMAGE, 1, 0, 100, 200)
    raw = p.encode_header(header, b"fake-jpeg-bytes")

    with pytest.raises(p.ProtocolError, match="before SessionBegun"):
        await client._process_message(attempt, raw, stopping=False)


@pytest.mark.asyncio
async def test_dispatch_binary_rejects_non_monotonic_sequence():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(
        callbacks={"on_capture_frame": lambda f: None}, streams=c._StreamRegistry(tracking=_tracking_stream())
    )
    payload = b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE
    first = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 5, 0, 0), payload)
    await client._process_message(attempt, first, stopping=False)
    repeated = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 5, 0, 0), payload)
    with pytest.raises(p.ProtocolError, match="sequence went backward"):
        await client._process_message(attempt, repeated, stopping=False)


@pytest.mark.asyncio
async def test_dispatch_binary_rejects_capture_frame_on_nonzero_stream():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(callbacks={"on_capture_frame": lambda f: None})
    payload = b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE
    raw = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 1, 0, 0, 0), payload)
    with pytest.raises(p.ProtocolError, match="stream_id must be 0"):
        await client._process_message(attempt, raw, stopping=False)


@pytest.mark.asyncio
async def test_dispatch_binary_rejects_capture_frame_with_no_registered_tracking_stream():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(callbacks={"on_capture_frame": lambda f: None})
    payload = b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE
    raw = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 0, 0, 0), payload)
    with pytest.raises(p.ProtocolError, match="no tracking stream was registered"):
        await client._process_message(attempt, raw, stopping=False)


@pytest.mark.asyncio
async def test_dispatch_binary_rejects_capture_frame_with_wrong_payload_length():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(
        callbacks={"on_capture_frame": lambda f: None},
        streams=c._StreamRegistry(tracking=_tracking_stream()),
    )
    short_payload = b"\x00" * (p.IC_TRACKING_V1_PAYLOAD_SIZE - 1)
    raw = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 0, 0, 0), short_payload)
    with pytest.raises(p.ProtocolError, match="payload is .* expected"):
        await client._process_message(attempt, raw, stopping=False)


@pytest.mark.asyncio
async def test_process_message_discards_data_while_stopping():
    seen = []
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(
        callbacks={"on_capture_frame": lambda f: seen.append(f)},
        streams=c._StreamRegistry(tracking=_tracking_stream()),
    )
    attempt.stop_requested.set()
    payload = b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE
    raw = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 0, 0, 0), payload)

    result = await client._process_message(attempt, raw, stopping=True)  # must not raise or dispatch

    assert result is None
    assert seen == []


@pytest.mark.asyncio
async def test_process_message_suppresses_binary_on_live_stop_flag_even_if_stale_param_false():
    """Regression for the stale-``stopping``-boolean finding: suppression
    must react to ``attempt.stop_requested`` being set, not to whatever
    value the caller's loop happened to sample before its last await."""
    seen = []
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt(
        callbacks={"on_capture_frame": lambda f: seen.append(f)},
        streams=c._StreamRegistry(tracking=_tracking_stream()),
    )
    attempt.stop_requested.set()
    payload = b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE
    raw = p.encode_header(p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 0, 0, 0), payload)

    # stopping=False here simulates a stale pre-await sample -- the fix
    # must check attempt.stop_requested live rather than trust this param.
    result = await client._process_message(attempt, raw, stopping=False)

    assert result is None
    assert seen == []


# ---------------------------------------------------------------------------
# _validate_layout: unchanged signature/behavior from before this redesign.
# ---------------------------------------------------------------------------


def test_validate_layout_skips_camera_only_sessions():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[],
    )
    client._validate_layout(begin, init_event)  # no tracking stream to validate -- must not raise


def test_validate_layout_rejects_omitted_requested_tracking_stream():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[],
    )
    with pytest.raises(p.ProtocolError, match="omits the tracking stream"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_unrequested_tracking_stream():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    stream = p.tracking_stream_descriptor((p.SourceType.HEAD,), 60.0)
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[stream],
    )
    with pytest.raises(p.ProtocolError, match="not requested"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_missing_requested_camera():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[],
    )
    with pytest.raises(p.ProtocolError, match="camera streams do not match"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_unexpected_camera():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=()))
    camera_stream = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[camera_stream],
    )
    with pytest.raises(p.ProtocolError, match="camera streams do not match"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_mismatched_camera_type():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    camera_stream = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    forged = dataclasses.replace(camera_stream, type=None)
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[forged],
    )
    with pytest.raises(p.ProtocolError, match="type does not match"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_mismatched_effective_config():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,), rate_hz=60.0))
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[p.tracking_stream_descriptor((p.SourceType.HEAD,), 60.0)],
    )
    forged_tracking = dataclasses.replace(init_event.effective_config.tracking, rate_hz=30.0)
    forged_config = dataclasses.replace(init_event.effective_config, tracking=forged_tracking)
    forged_init = dataclasses.replace(init_event, effective_config=forged_config)
    with pytest.raises(p.ProtocolError, match="effective_config does not match"):
        client._validate_layout(begin, forged_init)


def test_validate_layout_rejects_duplicate_camera_stream_names():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
    )
    stream_a = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    stream_b = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=2,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[stream_a, stream_b],
    )
    with pytest.raises(p.ProtocolError, match="duplicate camera stream names"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_duplicate_camera_stream_ids():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(
            p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),
            p.CameraRequest(name="camera_right", type=p.CameraType.OAK_D_COLOR),
        ),
    )
    stream_left = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_left",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    stream_right = p.camera_stream_descriptor(
        p.CameraStreamConfig(
            stream_id=1,
            name="camera_right",
            type=p.CameraType.OAK_D_COLOR,
            required=True,
            width=4,
            height=3,
            fps=30.0,
            jpeg_quality=90,
        )
    )
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[stream_left, stream_right],
    )
    with pytest.raises(p.ProtocolError, match="duplicate camera stream_id"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_more_than_one_tracking_stream():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    stream = p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[stream, stream],
    )
    with pytest.raises(p.ProtocolError, match="more than one"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_tracking_stream_sources_inconsistent_with_effective_config():
    """effective_config matching Begin is not enough
    -- the tracking stream descriptor carries its own copy of sources/
    rate_hz, and a server could let those disagree with effective_config
    while effective_config itself still matches what Begin requested."""
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD, p.SourceType.CONTROLLERS)))
    # The stream descriptor's own sources disagree with effective_config.tracking.sources
    # (which still correctly reflects begin -- only the stream itself is forged).
    mismatched_stream = p.tracking_stream_descriptor((p.SourceType.HEAD,), begin.tracking.rate_hz)
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[mismatched_stream],
    )
    with pytest.raises(p.ProtocolError, match="tracking stream sources do not match"):
        client._validate_layout(begin, init_event)


def test_validate_layout_rejects_tracking_stream_rate_inconsistent_with_effective_config():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,), rate_hz=60.0))
    mismatched_stream = p.tracking_stream_descriptor(begin.tracking.sources, 30.0)
    init_event = p.session_init_event(
        session_id="session-1",
        mode=p.CaptureMode.MOCK,
        begin=begin,
        streams=[mismatched_stream],
    )
    with pytest.raises(p.ProtocolError, match="tracking stream rate_hz does not match"):
        client._validate_layout(begin, init_event)


@pytest.mark.asyncio
async def test_start_rejects_mismatched_tracking_layout():
    async def _bad_layout_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        bad_stream = p.TrackingStream(
            stream_id=0,
            kind=p.StreamKind.TRACKING,
            name="tracking",
            schema=p.TrackingSchema.IC_TRACKING_V1,
            sources=begin.tracking.sources,
            rate_hz=begin.tracking.rate_hz,
            pose_representation=p.PoseRepresentation.POSITION_QUATERNION,
            quaternion_order=p.QuaternionOrder.XYZW,
            position_unit=p.PositionUnit.METERS,
            axes=p.Axes.RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD,
            reference_space="session_floor",
            hand_joint_order=p.HandJointOrder.OPENXR_26,
            payload_size=1,
            fields=(),
        )
        init_event = p.session_init_event(
            session_id=begin.session_id,
            mode=p.CaptureMode.MOCK,
            begin=begin,
            streams=[bad_stream],
        )
        await ws.send(json.dumps(init_event.to_json()))

    server = await websockets.serve(_bad_layout_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        with pytest.raises(p.ProtocolError, match="does not match"):
            await client.start(sources=(p.SourceType.HEAD,))
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Pre-acceptance failure modes: timeout, OSError, rejected/malformed
# messages before a validated SessionInit. "Stop during handshake" is
# covered separately below, at every phase.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_raises_timeout_error_on_connect(monkeypatch):
    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(c, "_ACCEPT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(c.websockets, "connect", _hang)
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(TimeoutError):
        await client.start(sources=(p.SourceType.HEAD,))
    assert client._attempt is None


@pytest.mark.asyncio
async def test_start_raises_oserror_on_connect(monkeypatch):
    async def _raise(*_args, **_kwargs):
        raise ConnectionRefusedError("nobody is listening")

    monkeypatch.setattr(c.websockets, "connect", _raise)
    client = CaptureClient("ws://127.0.0.1:1")
    with pytest.raises(OSError):
        await client.start(sources=(p.SourceType.HEAD,))
    assert client._attempt is None


@pytest.mark.asyncio
async def test_start_raises_timeout_error_awaiting_session_init(monkeypatch):
    async def _never_replies(ws) -> None:
        await ws.recv()  # Begin
        await asyncio.sleep(0.5)

    monkeypatch.setattr(c, "_ACCEPT_TIMEOUT_S", 0.05)
    server = await websockets.serve(_never_replies, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        with pytest.raises(TimeoutError):
            await client.start(sources=(p.SourceType.HEAD,))
        assert client._attempt is None
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_start_rejects_malformed_pre_init_messages():
    async def _bad_handler_unexpected_type(ws) -> None:
        await ws.recv()
        event = p.SessionStarting(session_id="x", stage=p.StartupStage.INITIALIZING, message="hi", status=None)
        await ws.send(json.dumps(event.to_json()))

    server = await websockets.serve(_bad_handler_unexpected_type, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        with pytest.raises(p.ProtocolError):
            await client.start(sources=(p.SourceType.HEAD,))
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_start_rejects_binary_before_init():
    async def _bad_handler_binary_before_init(ws) -> None:
        await ws.recv()
        header = p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, 0, 0, 0)
        await ws.send(p.encode_header(header, b"\x00" * p.IC_TRACKING_V1_PAYLOAD_SIZE))

    server = await websockets.serve(_bad_handler_binary_before_init, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        with pytest.raises(p.ProtocolError):
            await client.start(sources=(p.SourceType.HEAD,))
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_receive_until_init_rejects_unexpected_first_message_same_session():
    async def _handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        starting = p.SessionStarting(
            session_id=begin.session_id,
            stage=p.StartupStage.INITIALIZING,
            message="hi",
            status=None,
        )
        await ws.send(json.dumps(starting.to_json()))

    server = await websockets.serve(_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        with pytest.raises(p.ProtocolError, match="expected SessionInit first"):
            await client.start(sources=(p.SourceType.HEAD,))
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Six essential structural checks for Stop-during-handshake behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_during_handshake_connect_aborts_without_on_stopped(monkeypatch):
    """Stop while the handshake is blocked on connect() itself (not merely
    checked once afterward) must abort the attempt as a cancellation, with
    no on_stopped -- no session was ever accepted."""

    async def _hang_connect(*_args, **_kwargs):
        await asyncio.sleep(100)

    monkeypatch.setattr(c.websockets, "connect", _hang_connect)
    client = CaptureClient("ws://127.0.0.1:1")
    on_stopped_calls = []
    start_task = asyncio.ensure_future(
        client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda r: on_stopped_calls.append(r))
    )
    await asyncio.sleep(0.01)  # let start() actually begin connecting
    await client.stop()

    with pytest.raises(StoppedBeforeAcceptedError):
        await asyncio.wait_for(start_task, timeout=5)
    assert client._attempt is None
    assert on_stopped_calls == []


@pytest.mark.asyncio
async def test_stop_during_handshake_begin_send_aborts_without_on_stopped():
    """Stop while the handshake is blocked sending Begin -- the old design
    only checked Stop right after connect, not during the send or the
    SessionInit wait."""
    release = asyncio.Event()

    async def _handler(ws) -> None:
        await ws.recv()  # Begin arrives but the handler never replies
        await release.wait()

    server = await websockets.serve(_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        on_stopped_calls = []
        start_task = asyncio.ensure_future(
            client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda r: on_stopped_calls.append(r))
        )
        await asyncio.sleep(0.05)  # let Begin actually get sent, now blocked awaiting SessionInit
        await client.stop()

        with pytest.raises(StoppedBeforeAcceptedError):
            await asyncio.wait_for(start_task, timeout=5)
        assert client._attempt is None
        assert on_stopped_calls == []
        release.set()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stop_from_callback_completes_with_one_on_stopped(real_mock_server):
    """Calling stop() from inside a callback (on_started) must still result
    in exactly one on_stopped, confirmed by the server."""
    client = CaptureClient(real_mock_server)
    init_calls = []
    stopped_result = []
    finished = asyncio.Event()

    async def _stopping_on_init(event):
        init_calls.append(event)
        await client.stop()

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_started=_stopping_on_init,
        on_stopped=lambda r: (stopped_result.append(r), finished.set()),
    )
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert len(init_calls) == 1
    assert len(stopped_result) == 1
    assert stopped_result[0].reason == p.StopReason.REQUESTED


@pytest.mark.asyncio
async def test_normal_terminal_reply_confirms_stop(real_mock_server):
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()
    await client.start(
        sources=(p.SourceType.HEAD,), on_stopped=lambda r: (stopped_result.append(r), finished.set())
    )
    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert stopped_result[0].reason == p.StopReason.REQUESTED
    assert stopped_result[0].error_message is None


@pytest.mark.asyncio
async def test_callback_failure_attempts_stop_then_completes(real_mock_server):
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()

    def _raising_on_frame(_frame):
        raise ValueError("on_capture_frame boom")

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_capture_frame=_raising_on_frame,
        on_stopped=lambda r: (stopped_result.append(r), finished.set()),
    )
    await asyncio.wait_for(finished.wait(), timeout=5)
    result = stopped_result[0]
    assert result.reason == p.StopReason.REQUESTED  # the mock server still confirmed Stop
    assert "on_capture_frame boom" in result.error_message


@pytest.mark.asyncio
async def test_first_callback_failure_diagnostic_is_preserved_over_a_second():
    """Once a callback has failed, stopping begins and lifecycle messages
    still dispatch normally (only binary data is suppressed) -- so a
    second callback failure during the drain must not overwrite the first
    attempt's diagnostic, which is the real cause."""

    async def _two_statuses_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id,
            mode=p.CaptureMode.MOCK,
            begin=begin,
            streams=streams,
        )
        await ws.send(json.dumps(init_event.to_json()))
        begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
        await ws.send(json.dumps(begun.to_json()))
        for ns in (1, 2):
            status = p.SessionStatus(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=ns))
            await ws.send(json.dumps(status.to_json()))
        await ws.recv()  # Stop
        stopped = p.SessionStopped(
            session_id=begin.session_id,
            reason=p.StopReason.REQUESTED,
            error_message=None,
            status=None,
        )
        await ws.send(json.dumps(stopped.to_json()))

    server = await websockets.serve(_two_statuses_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        stopped_result = []
        finished = asyncio.Event()
        calls = []

        def _raising_on_status(event):
            calls.append(event)
            raise ValueError(f"status {event.status.updated_at_ns} boom")

        await client.start(
            sources=(p.SourceType.HEAD,),
            on_status=_raising_on_status,
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=5)
        # on_status now also receives SessionBegun (updated_at_ns=0), so it
        # is the first of the three calls to raise; the two SessionStatus
        # events (updated_at_ns=1, 2) still dispatch despite stopping.
        assert len(calls) == 3
        result = stopped_result[0]
        assert "status 0 boom" in result.error_message
        assert "status 1 boom" not in result.error_message
        assert "status 2 boom" not in result.error_message
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_on_init_callback_failure_attempts_stop_then_completes(real_mock_server):
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()

    def _raising_on_init(_event):
        raise ValueError("on_started boom")

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_started=_raising_on_init,
        on_stopped=lambda r: (stopped_result.append(r), finished.set()),
    )
    await asyncio.wait_for(finished.wait(), timeout=5)
    result = stopped_result[0]
    assert result.reason == p.StopReason.REQUESTED
    assert "on_started boom" in result.error_message


@pytest.mark.asyncio
async def test_on_init_cancelled_error_still_reaches_on_stopped(real_mock_server):
    """A CancelledError raised by application work an
    async on_started callback itself awaited (not an external task.cancel() on
    the supervisor) must be treated like any other callback failure --
    readiness still settles and on_stopped still fires exactly once,
    instead of escaping uncaught before finalization ever runs."""
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()

    async def _cancelled_on_init(_event):
        raise asyncio.CancelledError("on_started's own awaited work was cancelled")

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_started=_cancelled_on_init,
        on_stopped=lambda r: (stopped_result.append(r), finished.set()),
    )
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert len(stopped_result) == 1
    assert "on_started's own awaited work was cancelled" in stopped_result[0].error_message
    assert client._attempt is None


@pytest.mark.asyncio
async def test_run_attempt_finalizes_even_if_run_until_stopped_itself_raises(monkeypatch, real_mock_server):
    """Defensive guarantee: if _run_until_stopped somehow raises instead of
    returning a CaptureStopped (a bug, not a normal operating condition --
    every realistic failure path already returns a result), _run_attempt's
    try/finally must still reach _finalize with an unconfirmed result,
    rather than leaving the attempt stuck with no on_stopped."""
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()

    async def _boom(_attempt, _diagnostic):
        raise RuntimeError("simulated bug inside _run_until_stopped")

    monkeypatch.setattr(client, "_run_until_stopped", _boom)

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_stopped=lambda r: (stopped_result.append(r), finished.set()),
    )
    await asyncio.wait_for(finished.wait(), timeout=5)
    assert len(stopped_result) == 1
    assert stopped_result[0].reason is None  # never confirmed by the server
    assert client._attempt is None


@pytest.mark.asyncio
async def test_cancelling_start_before_acceptance_then_retrying_succeeds(monkeypatch, real_mock_server):
    """start()'s pre-acceptance CancelledError must
    propagate only after the supervisor's own cleanup has actually
    finished, so an immediate retry -- on the SAME client, whose
    self._attempt is the thing that would race -- doesn't hit 'attempt
    already active' against cleanup that is still in flight."""

    async def _hang_connect(*_args, **_kwargs):
        await asyncio.sleep(100)

    real_connect = c.websockets.connect
    monkeypatch.setattr(c.websockets, "connect", _hang_connect)
    client = CaptureClient(real_mock_server)
    start_task = asyncio.ensure_future(client.start(sources=(p.SourceType.HEAD,)))
    await asyncio.sleep(0.01)  # let start() actually begin connecting

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=5)

    # The supervisor's pre-acceptance cleanup must have already finished by
    # the time the above cancellation reached us -- not just eventually.
    assert client._attempt is None

    monkeypatch.setattr(c.websockets, "connect", real_connect)  # restore for the retry below
    # Must succeed immediately on the SAME client, not raise "attempt
    # already active".
    await client.start(sources=(p.SourceType.HEAD,))
    await client.stop()


async def _send_normal_handshake(ws, begin: p.BeginRequest) -> None:
    """Sends a real Begin/Init/Starting/Begun handshake for the given
    request -- shared by the two narrow ignore-Stop fixtures below, which
    both need a genuinely accepted session before exercising a Stop
    confirmation the real server can never withhold."""
    streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
    init_event = p.session_init_event(
        session_id=begin.session_id, mode=p.CaptureMode.MOCK, begin=begin, streams=streams
    )
    await ws.send(json.dumps(init_event.to_json()))
    starting = p.SessionStarting(
        session_id=begin.session_id, stage=p.StartupStage.STARTING_SOURCES, message="starting", status=None
    )
    await ws.send(json.dumps(starting.to_json()))
    begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
    await ws.send(json.dumps(begun.to_json()))


@pytest.mark.asyncio
async def test_missing_terminal_reply_completes_unconfirmed_after_deadline(monkeypatch):
    """A peer that accepts a session normally but never confirms Stop --
    real behavior the production server can never exhibit (it always
    eventually replies) -- exercises the client's own stop() confirmation
    deadline, via a narrow test-local peer rather than a general mock
    server."""
    monkeypatch.setattr(c, "_STOP_TIMEOUT_S", 0.05)

    async def _ignores_stop_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        await _send_normal_handshake(ws, begin)
        async for _raw in ws:
            pass  # deliberately never replies to Stop

    server = await websockets.serve(_ignores_stop_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        stopped_result = []
        finished = asyncio.Event()
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
        result = stopped_result[0]
        assert result.reason is None  # the server never confirmed
        assert "Timed out" in result.error_message
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_data_callback_failure_with_no_stop_confirmation_is_unconfirmed(monkeypatch):
    """Same narrow never-confirms-Stop peer as above, but this time Stop is
    triggered by a failing on_capture_frame callback rather than an
    explicit client.stop() call -- confirms the retried-Stop-then-deadline
    path is reached from a callback failure too, not only an explicit
    stop()."""
    monkeypatch.setattr(c, "_STOP_TIMEOUT_S", 0.05)

    async def _sends_frames_then_ignores_stop_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        await _send_normal_handshake(ws, begin)

        values = {f.name: np.zeros(f.shape, dtype=f.numpy_dtype) for f in p.IC_TRACKING_V1_FIELDS}
        values["head.orientation"] = np.array([0, 0, 0, 1], dtype=np.float32)
        payload = p.encode_tracking_frame(values)

        async def _send_frames() -> None:
            for seq in range(5):
                header = p.FrameMetadata(p.BinaryKind.CAPTURE_FRAME, 0, seq, seq * 16_000_000, seq * 16_000_000)
                await ws.send(p.encode_header(header, payload))
                await asyncio.sleep(0.005)

        sender = asyncio.ensure_future(_send_frames())
        try:
            async for _raw in ws:
                pass  # deliberately never replies to Stop
        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

    server = await websockets.serve(_sends_frames_then_ignores_stop_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        stopped_result = []
        finished = asyncio.Event()

        def _raising_on_frame(_frame):
            raise ValueError("on_capture_frame boom")

        await client.start(
            sources=(p.SourceType.HEAD,),
            on_capture_frame=_raising_on_frame,
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=5)
        result = stopped_result[0]
        assert result.reason is None  # the server never confirmed the retried Stop
        assert "on_capture_frame boom" in result.error_message
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_on_stopped_callback_failure_goes_to_loop_exception_handler(real_mock_server):
    client = CaptureClient(real_mock_server)
    finished = asyncio.Event()
    handled = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: (handled.append(context), finished.set()))

    def _raising_on_stopped(_result):
        raise ValueError("on_stopped boom")

    try:
        await client.start(sources=(p.SourceType.HEAD,), on_stopped=_raising_on_stopped)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert handled and "on_stopped boom" in str(handled[0]["exception"])
    finally:
        loop.set_exception_handler(None)


# ---------------------------------------------------------------------------
# Lifecycle-ordering/replay enforcement -- unit-level checks against
# CaptureClient's own validation functions directly, not a synthetic peer.
# These are genuine CaptureClient contract violations the real server
# cannot be configured to produce; test_session_status_before_begun_ends_session_unconfirmed
# below still needs a narrow local peer, since SessionStatus-before-Begun
# is a message-ordering violation that only shows up through the live
# receive loop. nonmonotonic-sequence and unregistered-camera-image cases
# are already covered unit-level by test_dispatch_binary_rejects_non_monotonic_sequence
# and test_dispatch_binary_rejects_unregistered_camera above.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_accepts_a_matching_duplicate_session_init():
    """Through _process_message's own SessionInit dispatch (not calling
    _check_duplicate_init directly), so this also covers that dispatch
    branch -- not just the validator it delegates to."""
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
    init_event = p.session_init_event(
        session_id="session-1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams
    )
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={}, init_json=init_event.to_json())

    result = await client._process_message(attempt, json.dumps(init_event.to_json()), stopping=False)
    assert result is None  # accepted silently, not a terminal reply


def test_check_duplicate_init_rejects_a_mismatched_replay():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
    init_event = p.session_init_event(
        session_id="session-1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams
    )
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={}, init_json=init_event.to_json())

    other_streams = [p.tracking_stream_descriptor((p.SourceType.HEAD, p.SourceType.CONTROLLERS), 30.0)]
    mismatched = p.session_init_event(
        session_id="session-1", mode=p.CaptureMode.MOCK, begin=begin, streams=other_streams
    )
    with pytest.raises(p.ProtocolError, match="does not match"):
        client._check_duplicate_init(attempt, mismatched)


@pytest.mark.asyncio
async def test_process_message_accepts_a_matching_duplicate_session_begun():
    """Through _process_message's own already-begun SessionBegun dispatch
    branch (not calling _check_duplicate_begun directly), so this also
    covers that dispatch branch -- not just the validator it delegates to."""
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    begun = p.SessionBegun(session_id="session-1", status=p.StatusSnapshot(updated_at_ns=0))
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={}, begun=True, begun_event=begun)

    result = await client._process_message(attempt, json.dumps(begun.to_json()), stopping=False)
    assert result is None  # accepted silently, not a terminal reply


def test_check_duplicate_begun_rejects_a_mismatched_replay():
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    begun = p.SessionBegun(session_id="session-1", status=p.StatusSnapshot(updated_at_ns=0))
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={}, begun_event=begun)

    mismatched = p.SessionBegun(session_id="session-1", status=p.StatusSnapshot(updated_at_ns=999))
    with pytest.raises(p.ProtocolError, match="original establishment"):
        client._check_duplicate_begun(attempt, mismatched)


@pytest.mark.asyncio
async def test_process_message_rejects_session_starting_after_begun():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt()
    starting = p.SessionStarting(
        session_id="session-1", stage=p.StartupStage.STARTING_SOURCES, message="late", status=None
    )
    with pytest.raises(p.ProtocolError, match="after SessionBegun"):
        await client._process_message(attempt, json.dumps(starting.to_json()), stopping=False)


@pytest.mark.asyncio
async def test_process_message_rejects_a_message_for_a_different_session_id():
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _begun_attempt()
    wrong = p.SessionStatus(session_id="some-other-session", status=p.StatusSnapshot(updated_at_ns=1))
    with pytest.raises(p.ProtocolError, match="expected"):
        await client._process_message(attempt, json.dumps(wrong.to_json()), stopping=False)


@pytest.mark.asyncio
async def test_session_status_before_begun_ends_session_unconfirmed():
    async def _status_before_begun_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id,
            mode=p.CaptureMode.MOCK,
            begin=begin,
            streams=streams,
        )
        await ws.send(json.dumps(init_event.to_json()))
        status = p.SessionStatus(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=1))
        await ws.send(json.dumps(status.to_json()))

    server = await websockets.serve(_status_before_begun_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        stopped_result = []
        finished = asyncio.Event()
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=5)
        assert stopped_result[0].reason is None
        assert "before SessionBegun" in stopped_result[0].error_message
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_rejected_and_status_messages_during_capture_do_not_end_session():
    async def _status_and_rejected_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id, mode=p.CaptureMode.MOCK, begin=begin, streams=streams
        )
        await ws.send(json.dumps(init_event.to_json()))
        begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
        await ws.send(json.dumps(begun.to_json()))
        rejected = p.RequestRejected(
            session_id="some-other-session",
            request_type=p.MessageType.STOP,
            reason=p.RejectionReason.UNKNOWN_SESSION,
            error_message="unknown session",
        )
        await ws.send(json.dumps(rejected.to_json()))
        status = p.SessionStatus(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=1))
        await ws.send(json.dumps(status.to_json()))
        await ws.recv()  # Stop
        stopped = p.SessionStopped(
            session_id=begin.session_id, reason=p.StopReason.REQUESTED, error_message=None, status=None
        )
        await ws.send(json.dumps(stopped.to_json()))

    server = await websockets.serve(_status_and_rejected_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        statuses = []
        finished = asyncio.Event()

        await client.start(
            sources=(p.SourceType.HEAD,),
            on_status=lambda s: statuses.append(s),
            on_stopped=lambda _: finished.set(),
        )
        await asyncio.sleep(0.05)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)

        # on_status receives SessionBegun and the one SessionStatus; the
        # RequestRejected for another session's id never ends the session
        # and is not delivered to on_status at all.
        assert len(statuses) == 2
        assert [s.type for s in statuses] == [
            c.CaptureStatusType.BEGUN,
            c.CaptureStatusType.STATUS_CHANGE,
        ]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_transport_loss_mid_capture_reports_unconfirmed_stop():
    async def _drop_connection_after_begun_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id, mode=p.CaptureMode.MOCK, begin=begin, streams=streams
        )
        await ws.send(json.dumps(init_event.to_json()))
        begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
        await ws.send(json.dumps(begun.to_json()))
        await ws.close()  # simulate an abrupt transport/owner loss mid-capture

    server = await websockets.serve(_drop_connection_after_begun_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        stopped_result = []
        finished = asyncio.Event()

        await client.start(
            sources=(p.SourceType.HEAD,),
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
        await asyncio.wait_for(finished.wait(), timeout=5)

        assert stopped_result[0].reason is None  # unconfirmed, not a fabricated server failure
        assert stopped_result[0].status == p.StatusSnapshot(updated_at_ns=0)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stop_after_transport_already_closed_is_silent():
    async def _drop_connection_after_begun_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id, mode=p.CaptureMode.MOCK, begin=begin, streams=streams
        )
        await ws.send(json.dumps(init_event.to_json()))
        begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
        await ws.send(json.dumps(begun.to_json()))
        await ws.close()

    server = await websockets.serve(_drop_connection_after_begun_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        finished = asyncio.Event()

        await client.start(sources=(p.SourceType.HEAD,), on_stopped=lambda _: finished.set())
        await asyncio.wait_for(finished.wait(), timeout=5)  # transport already dropped by now

        await client.stop()  # must not raise even though the attempt is long gone
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_session_starting_status_is_cached():
    async def _handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id,
            mode=p.CaptureMode.MOCK,
            begin=begin,
            streams=streams,
        )
        await ws.send(json.dumps(init_event.to_json()))
        starting = p.SessionStarting(
            session_id=begin.session_id,
            stage=p.StartupStage.STARTING_SOURCES,
            message="hi",
            status=p.StatusSnapshot(updated_at_ns=7),
        )
        await ws.send(json.dumps(starting.to_json()))
        await ws.recv()  # Stop
        stopped = p.SessionStopped(
            session_id=begin.session_id,
            reason=p.StopReason.REQUESTED,
            error_message=None,
            status=None,
        )
        await ws.send(json.dumps(stopped.to_json()))

    server = await websockets.serve(_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        finished = asyncio.Event()
        starting_seen = asyncio.Event()
        # start() returns once on_started has been attempted, which can race
        # ahead of the receive loop processing the next message (an
        # already-sent SessionStarting) -- wait for on_status itself
        # rather than asserting on internal state right after start().
        await client.start(
            sources=(p.SourceType.HEAD,),
            on_status=lambda _e: starting_seen.set(),
            on_stopped=lambda _: finished.set(),
        )
        await asyncio.wait_for(starting_seen.wait(), timeout=5)
        assert client._attempt.last_status == p.StatusSnapshot(updated_at_ns=7)
        await client.stop()
        await asyncio.wait_for(finished.wait(), timeout=5)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_cancelling_start_after_acceptance_still_completes_via_on_stopped(real_mock_server):
    """Cancelling start()'s own await (not the supervisor task) after
    SessionInit was already accepted must still drive the attempt to a
    single on_stopped completion."""
    client = CaptureClient(real_mock_server)
    stopped_result = []
    finished = asyncio.Event()
    init_seen = asyncio.Event()

    start_task = asyncio.ensure_future(
        client.start(
            sources=(p.SourceType.HEAD,),
            on_started=lambda _e: init_seen.set(),
            on_stopped=lambda r: (stopped_result.append(r), finished.set()),
        )
    )
    await asyncio.wait_for(init_seen.wait(), timeout=5)
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    await asyncio.wait_for(finished.wait(), timeout=5)
    assert client._attempt is None


@pytest.mark.asyncio
async def test_session_begun_after_stop_requested_still_fires_once():
    """A SessionBegun that legitimately arrives after stop() was already
    called still fires on_begun exactly once -- only binary data
    (tracking frames/camera images) is suppressed once Stop has been
    requested, not JSON lifecycle messages. The attempt still completes
    normally via the server's confirmed SessionStopped."""
    begun_sent = asyncio.Event()

    async def _slow_begun_handler(ws) -> None:
        begin = p.parse_begin(json.loads(await ws.recv()))
        streams = [p.tracking_stream_descriptor(begin.tracking.sources, begin.tracking.rate_hz)]
        init_event = p.session_init_event(
            session_id=begin.session_id,
            mode=p.CaptureMode.MOCK,
            begin=begin,
            streams=streams,
        )
        await ws.send(json.dumps(init_event.to_json()))
        await ws.recv()  # Stop, sent by the client before SessionBegun below
        begun = p.SessionBegun(session_id=begin.session_id, status=p.StatusSnapshot(updated_at_ns=0))
        await ws.send(json.dumps(begun.to_json()))
        begun_sent.set()
        stopped = p.SessionStopped(
            session_id=begin.session_id,
            reason=p.StopReason.REQUESTED,
            error_message=None,
            status=None,
        )
        await ws.send(json.dumps(stopped.to_json()))

    server = await websockets.serve(_slow_begun_handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        client = CaptureClient(f"ws://127.0.0.1:{port}")
        begun_calls = []
        finished = asyncio.Event()

        await client.start(
            sources=(p.SourceType.HEAD,),
            on_status=lambda e: begun_calls.append(e) if e.type == c.CaptureStatusType.BEGUN else None,
            on_stopped=lambda _: finished.set(),
        )
        await client.stop()
        await asyncio.wait_for(begun_sent.wait(), timeout=5)
        await asyncio.wait_for(finished.wait(), timeout=5)

        assert len(begun_calls) == 1  # still dispatched, even though it arrived after stop()
    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# _run_until_stopped's send/drain branches, with a fully-controlled stub
# transport instead of racing real timeouts.
# ---------------------------------------------------------------------------


class _StubTransport:
    """A minimal controllable stand-in for a websockets connection.
    ``recv_sequence`` is a list of either a str/bytes message to return or
    an Exception instance/class to raise, consumed in order."""

    def __init__(self, recv_sequence=(), *, send_raises=None, close_raises=None):
        self._recv_sequence = list(recv_sequence)
        self._send_raises = send_raises
        self._close_raises = close_raises
        self.sent = []

    async def send(self, data):
        if self._send_raises is not None:
            raise self._send_raises
        self.sent.append(data)

    async def recv(self):
        item = self._recv_sequence.pop(0)
        if isinstance(item, Exception) or (isinstance(item, type) and issubclass(item, Exception)):
            raise item
        return item

    async def close(self):
        if self._close_raises is not None:
            raise self._close_raises


def _stopping_attempt(ws, session_id="session-1") -> c._Attempt:
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    attempt = c._Attempt(session_id=session_id, begin=begin, callbacks={}, ws=ws)
    attempt.stop_requested.set()
    return attempt


@pytest.mark.asyncio
async def test_run_until_stopped_returns_unconfirmed_when_send_fails(monkeypatch):
    from websockets.exceptions import ConnectionClosedOK

    monkeypatch.setattr(c, "_STOP_TIMEOUT_S", 1.0)
    client = CaptureClient("ws://127.0.0.1:1")
    ws = _StubTransport(send_raises=ConnectionClosedOK(None, None), recv_sequence=[ConnectionClosedOK(None, None)])
    attempt = _stopping_attempt(ws)

    result = await client._run_until_stopped(attempt, None)

    assert result.reason is None


@pytest.mark.asyncio
async def test_run_until_stopped_skips_bad_messages_then_confirms(monkeypatch):
    monkeypatch.setattr(c, "_STOP_TIMEOUT_S", 5.0)
    stopped = p.SessionStopped(
        session_id="session-1",
        reason=p.StopReason.REQUESTED,
        error_message=None,
        status=None,
    )
    other_session = p.SessionStatus(session_id="other", status=p.StatusSnapshot(updated_at_ns=0))
    ws = _StubTransport(
        recv_sequence=[
            b"\x00binary-data-is-skipped",
            "not valid json",
            json.dumps(other_session.to_json()),
            json.dumps(stopped.to_json()),
        ]
    )
    client = CaptureClient("ws://127.0.0.1:1")
    attempt = _stopping_attempt(ws)

    result = await client._run_until_stopped(attempt, None)

    assert result is not None
    assert result.reason == p.StopReason.REQUESTED


@pytest.mark.asyncio
async def test_close_ws_swallows_connection_closed():
    from websockets.exceptions import ConnectionClosedOK

    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    attempt = c._Attempt(
        session_id="session-1",
        begin=begin,
        callbacks={},
        ws=_StubTransport(close_raises=ConnectionClosedOK(None, None)),
    )
    await client._close_ws(attempt)
    assert attempt.ws is None


# ---------------------------------------------------------------------------
# The _race helper itself.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_race_returns_coro_result_when_it_wins_first():
    async def _quick():
        return "done"

    stop_event = asyncio.Event()
    result = await CaptureClient._race(_quick(), stop_event, timeout=5.0)
    assert result == "done"


@pytest.mark.asyncio
async def test_race_raises_stop_won_race_when_stop_wins():
    """Deliberately NOT asyncio.CancelledError -- see
    StoppedBeforeAcceptedError's docstring for why _race must not raise a
    bare CancelledError in a task nobody actually cancelled."""

    async def _hang():
        await asyncio.sleep(100)

    stop_event = asyncio.Event()
    stop_event.set()
    with pytest.raises(c._StopWonRace):
        await CaptureClient._race(_hang(), stop_event, timeout=5.0)


@pytest.mark.asyncio
async def test_race_raises_timeout_error_when_deadline_wins():
    async def _hang():
        await asyncio.sleep(100)

    stop_event = asyncio.Event()
    with pytest.raises(TimeoutError):
        await CaptureClient._race(_hang(), stop_event, timeout=0.01)


@pytest.mark.asyncio
async def test_race_propagates_coro_exception():
    async def _raise():
        raise ValueError("boom")

    stop_event = asyncio.Event()
    with pytest.raises(ValueError, match="boom"):
        await CaptureClient._race(_raise(), stop_event, timeout=5.0)


# ---------------------------------------------------------------------------
# Client-facing SessionInit/TrackingStream/FieldDescriptor conversion.
# ---------------------------------------------------------------------------


def _wire_session_init_for(sources: tuple[p.SourceType, ...]) -> p.SessionInit:
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=sources), session_id="s-1")
    streams = [p.tracking_stream_descriptor(sources, begin.tracking.rate_hz)]
    return p.session_init_event(session_id="s-1", mode=p.CaptureMode.MOCK, begin=begin, streams=streams)


def test_to_client_session_init_passes_tracking_stream_through_unchanged():
    wire = _wire_session_init_for((p.SourceType.HEAD,))
    client_init = c._to_client_started(wire)

    assert isinstance(client_init, c.CaptureStarted)
    assert client_init.running is False
    assert client_init.session_id == wire.session_id
    assert client_init.mode == wire.mode
    assert client_init.effective_config is wire.effective_config
    assert client_init.clock is wire.clock
    assert len(client_init.streams) == 1

    # No conversion/copy -- CaptureStarted.streams carries the exact wire
    # SessionInit.streams objects, since FrameDescription/FieldDescriptor
    # (protocol.TrackingStream/FieldDescriptor) already describe exactly
    # what applications need to decode CaptureFrame.payload themselves.
    stream = client_init.streams[0]
    assert stream is wire.streams[0]
    assert isinstance(stream, c.FrameDescription)
    assert stream.stream_id == wire.streams[0].stream_id
    assert stream.sources == wire.streams[0].sources
    assert stream.rate_hz == wire.streams[0].rate_hz
    assert stream.payload_size == p.ic_tracking_v1_layout().payload_size

    for field in stream.fields:
        assert isinstance(field, c.FieldDescriptor)
        assert isinstance(field.dtype, p.WireDType)
        assert isinstance(field.offset, int)


def test_to_client_session_init_field_descriptor_table_matches_guide_examples():
    wire = _wire_session_init_for((p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS))
    stream = c._to_client_started(wire).streams[0]
    by_name = {f.name: f for f in stream.fields}

    expected = {
        "head.present": (p.WireDType.UINT8, (), 0),
        "head.position": (p.WireDType.FLOAT32_LE, (3,), 1),
        "head.orientation": (p.WireDType.FLOAT32_LE, (4,), 13),
        "head.is_valid": (p.WireDType.UINT8, (), 29),
        "controller_left.trigger_value": (p.WireDType.FLOAT64_LE, (), 146),
        "hand_left.joint_positions": (p.WireDType.FLOAT32_LE, (26, 3), 278),
        "hand_left.joint_orientations": (p.WireDType.FLOAT32_LE, (26, 4), 590),
        "hand_left.joint_radii": (p.WireDType.FLOAT32_LE, (26,), 1006),
        "hand_left.joint_valid": (p.WireDType.UINT8, (26,), 1110),
    }
    for name, (dtype, shape, offset) in expected.items():
        assert by_name[name].dtype == dtype
        assert by_name[name].shape == shape
        assert by_name[name].offset == offset


def test_to_client_session_init_passes_camera_streams_through_unchanged():
    begin = p.BeginRequest(
        tracking=p.TrackingConfig(sources=()),
        cameras=(p.CameraRequest(name="camera_left", type=p.CameraType.OAK_D_COLOR),),
        session_id="s-1",
    )
    camera_stream = p.CameraStream(
        stream_id=1,
        kind=p.StreamKind.IMAGE,
        name="camera_left",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=1920,
        height=1080,
        fps=60.0,
        codec=p.ImageCodec.JPEG,
        jpeg_quality=90,
        orientation=p.ImageOrientation(rotation_degrees=0, mirrored=False),
        intrinsics=None,
    )
    wire = p.session_init_event(session_id="s-1", mode=p.CaptureMode.MOCK, begin=begin, streams=[camera_stream])
    client_init = c._to_client_started(wire)
    assert client_init.streams == (camera_stream,)


@pytest.mark.asyncio
async def test_on_init_receives_client_facing_session_init(real_mock_server):
    client = CaptureClient(real_mock_server)
    received: list[c.CaptureStarted] = []
    finished = asyncio.Event()

    await client.start(
        sources=(p.SourceType.HEAD,),
        on_started=lambda e: received.append(e),
        on_stopped=lambda _: finished.set(),
    )
    await client.stop()
    await asyncio.wait_for(finished.wait(), timeout=5)

    assert len(received) == 1
    init_event = received[0]
    assert isinstance(init_event, c.CaptureStarted)
    assert init_event.running is False
    stream = init_event.streams[0]
    assert isinstance(stream, c.FrameDescription)
    assert all(isinstance(f.dtype, p.WireDType) for f in stream.fields)
