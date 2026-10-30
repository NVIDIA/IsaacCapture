# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""WS /api/session and GET /stream{stream_id} routes. Split out of app.py
for readability -- this module owns viewer task lifetime and transport
details; app.py keeps the FastAPI app object, lifespan and static hosting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocketState

from .capture_status import CaptureStatusMonitor
from .preview_protocol import (
    CAMERA_SAMPLE_HZ,
    HEARTBEAT_INTERVAL_S,
    MAX_JSON_SNAPSHOT_BYTES,
    MAX_RECORDING_COMMAND_BYTES,
    PENDING_REPLY_QUEUE_SIZE,
    SEND_TIMEOUT_S,
    SENDER_INTERVAL_S,
    TRACKING_SAMPLE_HZ,
    build_snapshot,
    pack_tracking_header,
)
from .recording_command import (
    RecordingCommandError,
    RecordingContext,
    handle_recording_command,
    parse_recording_command,
)

_logger = logging.getLogger(__name__)

router = APIRouter()

_TRACKING_SAMPLE_INTERVAL_S = 1.0 / TRACKING_SAMPLE_HZ
_CAMERA_SAMPLE_INTERVAL_S = 1.0 / CAMERA_SAMPLE_HZ
# Preview-side staleness bound for a cached camera image, independent of
# the frontend's own display-side freshness rule -- a request for an image
# this old is refused rather than served as if it were live.
_CAMERA_STALE_S = 2.0


async def _bounded_close(websocket: WebSocket, *, code: int = 1000, reason: str = "") -> None:
    """Closes a WebSocket with the same send bound as any other write, so a
    blocked close can never hold a route's finally block open indefinitely
    (which would in turn delay viewer-admission release)."""
    try:
        async with asyncio.timeout(SEND_TIMEOUT_S):
            await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001 -- best-effort close, including TimeoutError
        pass


def _current_generation(monitor: CaptureStatusMonitor) -> int | None:
    """Binary generation to advertise, or None before acceptance/after terminal."""
    # One attempt per process: once accepted, this connection's generation
    # is always 1 until the attempt reaches terminal. There is no runtime
    # descriptor replacement within a single backend lifetime.
    if monitor.native.started is None or monitor.native.stopped is not None:
        return None
    return 1


class _ViewerTermination:
    """Mutable box the receiver uses to request the close code/reason the
    sender (the one actual writer for this viewer) should perform. The
    receiver never calls `websocket.close()`/`_bounded_close()` itself --
    two tasks closing the same socket could race with the sender's own
    in-flight `send_text`/`send_bytes`, which is exactly what the
    one-writer rule exists to prevent. Close reason text is kept short and
    fixed (WebSocket control frames cap reason text at 125 bytes); the real
    diagnostic detail is logged separately, never embedded in the close
    frame itself."""

    def __init__(self) -> None:
        """Starts with no close requested yet."""
        self.code: int | None = None
        self.reason: str = ""

    def request(self, *, code: int, reason: str) -> None:
        """Records the close this viewer should receive; first request wins."""
        if self.code is None:
            self.code = code
            self.reason = reason


async def _drain_receiver(
    websocket: WebSocket,
    disconnected: asyncio.Event,
    reply_queue: asyncio.Queue[dict[str, Any]],
    termination: _ViewerTermination,
    recording_context: RecordingContext,
) -> None:
    """Detects closure and accepts exactly one legitimate inbound message
    shape: a small, bounded recording-command JSON text message, replied to
    (via ``reply_queue``, drained by the existing single sender) with a
    real, correlated outcome from the process-wide ``recording_context`` shared
    by every viewer -- see ``recording_command.py``. Anything else (binary, oversized,
    malformed, or an unrecognized action) is still a protocol error, exactly
    as before -- recorded via ``termination`` for the sender to actually
    close on, never closed here."""
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            text = message.get("text")
            if text is None or len(text.encode("utf-8")) > MAX_RECORDING_COMMAND_BYTES:
                _logger.warning("recording command rejected: missing or oversized text message")
                termination.request(code=1003, reason="unexpected or oversized message")
                break
            try:
                raw = json.loads(text)
                if not isinstance(raw, dict) or raw.get("type") != "recording_command":
                    raise RecordingCommandError("expected a recording_command message")
                command = parse_recording_command(raw)
            except (ValueError, RecordingCommandError):
                # Never log the raw text or the exception's own message: a
                # malformed command can still carry a real control_token
                # (recording_command.py), including via other fields like
                # `action` -- a fixed category is the only line that's
                # guaranteed safe regardless of what a future validator
                # message happens to embed.
                _logger.warning("recording command rejected: malformed or invalid command")
                termination.request(code=1003, reason="malformed recording command")
                break
            # request_id is an arbitrary client-supplied string (parse_recording_command only
            # requires it nonempty) -- never logged verbatim, same reasoning as the malformed-
            # command case above: it could carry a real control_token hoping it leaks out here.
            _logger.info("recording command received: action=%s", command.action)
            reply = await handle_recording_command(command, recording_context)
            try:
                reply_queue.put_nowait(reply)
            except asyncio.QueueFull:
                # An overloaded/slow viewer whose replies aren't being
                # drained fast enough -- close it rather than grow the
                # queue unboundedly or silently drop the reply.
                _logger.warning("recording reply queue full; closing viewer")
                termination.request(code=1013, reason="reply backlog too large")
                break
    except WebSocketDisconnect:
        pass
    finally:
        disconnected.set()


def _build_snapshot_now(
    monitor: CaptureStatusMonitor, *, revision: int, now: float, visualization: dict | None = None
) -> dict:
    """Assembles one JSON-ready preview snapshot from the monitor's current
    native/local facts and latest preview samples, plus the optional 3D-view
    metadata the application advertises (web presentation, not a capture fact)."""
    status_obj = monitor.snapshot()
    snapshot = build_snapshot(
        backend_instance_id=status_obj.backend_instance_id,
        revision=revision,
        station_label=status_obj.station_label,
        profile_label=status_obj.profile_label,
        connecting=status_obj.connecting,
        native=monitor.native,
        error_message=status_obj.error_message,
        binary_generation=_current_generation(monitor),
        tracking_slot=monitor.preview.tracking(),
        tracking_diagnostic=monitor.preview.tracking_diagnostic(),
        camera_slots_by_name={cam.name: monitor.preview.camera(cam.name) for cam in monitor.camera_descriptors},
        camera_diagnostics={
            cam.name: d
            for cam in monitor.camera_descriptors
            if (d := monitor.preview.camera_diagnostic(cam.name)) is not None
        },
        now_monotonic=now,
    )
    if visualization is not None:
        snapshot["visualization"] = visualization
    return snapshot


def _visualization_for(websocket: WebSocket) -> dict | None:
    """The 3D-view metadata for this viewer, using the host name the browser itself used to reach the backend."""
    scene = getattr(websocket.app.state, "visualization", None)
    return None if scene is None else scene.projection(websocket.url.hostname)


def _expected_origin(websocket: WebSocket) -> str:
    """The backend's own browser-facing origin, serialized the way a real
    browser's Origin header does: ws<->http, wss<->https, default port
    omitted, IPv6 host re-bracketed (``websocket.url.hostname`` strips the
    brackets a real Origin header keeps)."""
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    default_port = 443 if scheme == "https" else 80
    port = websocket.url.port
    hostname = websocket.url.hostname or ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is None or port == default_port:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _configured_allowed_origins() -> frozenset[str]:
    """Explicitly configured additional origins (comma-separated), for a
    deployment where the frontend is hosted separately from this backend
    or reached through a reverse proxy that changes the browser-visible
    origin -- the default same-origin check alone can't cover that case."""
    raw = os.environ.get("EGOCAP_ALLOWED_ORIGINS", "")
    return frozenset(entry.strip() for entry in raw.split(",") if entry.strip())


def _trusted_hostnames() -> frozenset[str] | None:
    """Pins the same-origin fallback below to a configured set of
    hostnames/addresses, or ``None`` if ``EGOCAP_TRUSTED_HOSTNAMES`` is
    unset. Unpinned, ``_expected_origin`` derives the "backend's own"
    origin from the same Host header the browser sent -- which a
    DNS-rebinding page can also control (point a hostname at this
    backend's IP, open a WebSocket, and the browser's self-consistent
    Host/Origin pair passes without ever proving it reached the real
    station). Unset is a quick local/dev default, not a safe one for a
    deployment reachable from an untrusted network; set this alongside (or
    instead of) ``EGOCAP_ALLOWED_ORIGINS`` there."""
    raw = os.environ.get("EGOCAP_TRUSTED_HOSTNAMES")
    if raw is None:
        return None
    # Matched against websocket.url.hostname, which urllib already lowercases
    # and strips the brackets from (e.g. "[::1]" -> "::1") -- normalize
    # configured entries the same way, so "Station.local" or "[::1]" here
    # isn't silently never matched.
    return frozenset(entry.strip().strip("[]").lower() for entry in raw.split(",") if entry.strip())


def _origin_allowed(websocket: WebSocket) -> bool:
    """Rejects a missing, ``null``, or unlisted Origin header -- this is
    what actually prevents an unrelated page from opening this channel in
    the operator's browser (the separate operator-name field in
    recording_command.py is attribution, not access control: it does not
    block anything by itself). No substring matching, no wildcard, no
    trusting a forwarded header -- an exact match against an explicitly
    configured origin, or against the backend's own derived origin (only
    when the request's Host is itself trusted, see ``_trusted_hostnames``)."""
    origin = websocket.headers.get("origin")
    if not origin or origin == "null":
        return False
    if origin in _configured_allowed_origins():
        return True
    trusted_hostnames = _trusted_hostnames()
    if trusted_hostnames is not None and (websocket.url.hostname or "") not in trusted_hostnames:
        return False
    return origin == _expected_origin(websocket)


@router.websocket("/api/session")
async def capture_session(websocket: WebSocket) -> None:
    """Admits one preview viewer and owns its admission slot for the
    accept-through-serve lifetime, regardless of how that lifetime ends."""
    if not _origin_allowed(websocket):
        # Denying before accept() -- the ASGI server answers the handshake
        # itself (HTTP 403), never an accepted-then-immediately-closed
        # connection, and no admission slot is consumed for a request that
        # was never going to be a real viewer.
        await _bounded_close(websocket, code=1008, reason="origin not allowed")
        return
    monitor: CaptureStatusMonitor = websocket.app.state.monitor
    recording_context: RecordingContext = websocket.app.state.recording_context
    if not monitor.try_admit_ws_viewer():
        await _bounded_close(websocket, code=1013, reason="too many viewers")
        return

    # One try/finally owns admission across accept-through-serve: a plain
    # except Exception around accept() alone would miss asyncio.CancelledError
    # (a BaseException, not an Exception), leaking the slot on cancellation
    # during accept.
    try:
        await websocket.accept()
        await _serve_session(websocket, monitor, recording_context)
    finally:
        monitor.release_ws_viewer()


async def _serve_session(
    websocket: WebSocket, monitor: CaptureStatusMonitor, recording_context: RecordingContext
) -> None:
    """Runs one viewer's bounded snapshot/heartbeat/tracking/reply sender
    loop until disconnect, a protocol violation, or an oversized snapshot.
    This loop (and only this loop) ever writes to the socket -- recording
    replies queued by the receiver are drained and sent here too, ahead of
    sampled tracking, so one write path stays the single source of
    outbound ordering. ``recording_context`` is the one process-lifetime
    instance shared by every viewer (see its own docstring), not created here."""
    disconnected = asyncio.Event()
    reply_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=PENDING_REPLY_QUEUE_SIZE)
    termination = _ViewerTermination()
    receiver_task = asyncio.create_task(
        _drain_receiver(websocket, disconnected, reply_queue, termination, recording_context)
    )

    # The revision this viewer has actually been sent a matching snapshot
    # for -- not just the monitor's current revision. A binary sample is
    # only eligible once this equals monitor.revision again after the send
    # completes: if the native observation advanced while the snapshot
    # write was in flight (an awaited call), that mismatch is detected
    # below and this loop sends a fresh snapshot before any binary, rather
    # than trusting post-await state the viewer was never actually shown.
    last_sent_revision: int | None = None
    last_snapshot_monotonic = 0.0
    last_tracking_sequence: int | None = None
    last_tracking_sent_monotonic = 0.0
    # Mirrors last_sent_revision's role, for RecordingContext instead of
    # the capture monitor -- starts at None so the very first loop
    # iteration always pushes the current recording context, giving a
    # freshly (re)connecting viewer its bootstrap without having to send a
    # command first. Also what lets this SAME viewer notice a different
    # viewer's command (shared recording_context, see its own docstring).
    last_sent_recording_revision: int | None = None

    try:
        while not disconnected.is_set():
            now = time.monotonic()
            revision = monitor.revision
            must_send_snapshot = (
                last_sent_revision is None
                or revision != last_sent_revision
                or (now - last_snapshot_monotonic) >= HEARTBEAT_INTERVAL_S
            )
            if must_send_snapshot:
                snapshot = _build_snapshot_now(
                    monitor, revision=revision, now=now, visualization=_visualization_for(websocket)
                )
                body = json.dumps(snapshot).encode("utf-8")
                if len(body) > MAX_JSON_SNAPSHOT_BYTES:
                    await _bounded_close(websocket, code=1009, reason="snapshot too large")
                    return
                try:
                    async with asyncio.timeout(SEND_TIMEOUT_S):
                        await websocket.send_text(body.decode("utf-8"))
                except (TimeoutError, WebSocketDisconnect):
                    return
                last_sent_revision = revision
                last_snapshot_monotonic = time.monotonic()

            # Recording-command replies are control traffic, sent ahead of
            # sampled tracking this same iteration. Drains only the number
            # of replies actually queued at the START of this turn, not
            # "until empty" -- the receiver can keep enqueueing more while
            # each send is awaited, and an unbounded drain here would let a
            # viewer that keeps sending commands starve its own snapshot/
            # tracking delivery indefinitely.
            for _ in range(reply_queue.qsize()):
                reply = reply_queue.get_nowait()
                try:
                    async with asyncio.timeout(SEND_TIMEOUT_S):
                        await websocket.send_text(json.dumps(reply))
                except (TimeoutError, WebSocketDisconnect):
                    return

            # Proactive recording-context push: bootstraps a freshly
            # (re)connecting viewer without it having to send a command
            # first, and propagates another viewer's command to this one
            # too, since recording_context is shared process-wide. The
            # revision to mark as sent is captured BEFORE the send, not
            # re-read afterward -- a mutation during the awaited send must
            # still be picked up by a later iteration, not be mistaken for
            # already having been delivered.
            recording_context.refresh()
            pending_recording_revision = recording_context.revision
            if pending_recording_revision != last_sent_recording_revision:
                try:
                    async with asyncio.timeout(SEND_TIMEOUT_S):
                        await websocket.send_text(json.dumps(recording_context.snapshot_message()))
                except (TimeoutError, WebSocketDisconnect):
                    return
                last_sent_recording_revision = pending_recording_revision

            # Only eligible if the viewer's last-confirmed-sent snapshot
            # still matches the monitor's current revision -- see
            # last_sent_revision's docstring above.
            if last_sent_revision == monitor.revision:
                generation = _current_generation(monitor)
                native_running = monitor.native.status is not None and monitor.native.status.running
                if generation is not None and native_running:
                    tracking_descriptor = monitor.tracking_descriptor
                    tracking_slot = monitor.preview.tracking()
                    if (
                        tracking_descriptor is not None
                        and tracking_slot is not None
                        and tracking_slot.sequence != last_tracking_sequence
                        and (time.monotonic() - last_tracking_sent_monotonic) >= _TRACKING_SAMPLE_INTERVAL_S
                    ):
                        header = pack_tracking_header(
                            generation=generation,
                            stream_id=tracking_descriptor.stream_id,
                            sequence=tracking_slot.sequence,
                            capture_time_ns=tracking_slot.capture_time_ns,
                            receive_time_ns=tracking_slot.receive_time_ns,
                        )
                        try:
                            async with asyncio.timeout(SEND_TIMEOUT_S):
                                await websocket.send_bytes(header + tracking_slot.payload)
                        except (TimeoutError, WebSocketDisconnect):
                            return
                        # If the revision changed while this send was in
                        # flight, the next loop iteration's must_send_snapshot
                        # check (revision != last_sent_revision) catches it
                        # and sends a fresh snapshot before any further binary.
                        last_tracking_sequence = tracking_slot.sequence
                        last_tracking_sent_monotonic = time.monotonic()

            try:
                await asyncio.wait_for(disconnected.wait(), timeout=SENDER_INTERVAL_S)
            except TimeoutError:
                pass
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 -- best-effort cleanup
            pass
        # The one actual close call for this viewer, regardless of why this
        # loop ended -- a receiver-requested protocol/overflow close uses
        # its own code/reason; an ordinary disconnect/teardown uses the
        # default. Never closed from _drain_receiver itself (see
        # _ViewerTermination's own docstring).
        if websocket.client_state != WebSocketState.DISCONNECTED:
            if termination.code is not None:
                await _bounded_close(websocket, code=termination.code, reason=termination.reason)
            else:
                await _bounded_close(websocket)


def _multipart_jpeg(jpeg: bytes, sequence: int, capture_time_ns: int, receive_time_ns: int) -> bytes:
    """Wraps one unchanged JPEG in a multipart/x-mixed-replace part."""
    headers = (
        f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
        f"X-Capture-Sequence: {sequence}\r\nX-Capture-Time-Ns: {capture_time_ns}\r\n"
        f"X-Receive-Time-Ns: {receive_time_ns}\r\n\r\n"
    ).encode("ascii")
    return headers + jpeg + b"\r\n"


def _is_fresh(slot, *, now: float) -> bool:
    """Whether a preview slot's backend-local receipt age is within the camera staleness bound."""
    return slot is not None and (now - slot.received_monotonic) <= _CAMERA_STALE_S


class _BoundedMjpegResponse(StreamingResponse):
    """Bounds every ASGI send (headers and each multipart chunk) with the
    same send timeout the WS sender uses, and releases the caller-supplied
    viewer admission exactly once on every exit path -- including a
    failure before the body generator ever starts (e.g. the initial
    ``http.response.start`` send), which a release placed only inside the
    generator's own ``finally`` would miss."""

    def __init__(self, *args, release, **kwargs) -> None:
        """Stores the caller's viewer-admission release callback."""
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope, receive, send) -> None:
        """Serves the response with every ASGI send bounded and admission
        released exactly once, however the response ends."""

        async def bounded_send(message) -> None:
            """Bounds one ASGI send with the shared send timeout."""
            async with asyncio.timeout(SEND_TIMEOUT_S):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            self._release()


@router.get("/stream{stream_id}")
async def camera_stream(stream_id: int, request: Request) -> Response:
    """Validates the requested camera/identity and streams its latest JPEGs
    as MJPEG, or returns 404/409/503 for an unselected/superseded/stale one."""
    monitor: CaptureStatusMonitor = request.app.state.monitor
    camera = monitor.camera_descriptor_by_stream_id(stream_id)
    if camera is None:
        return JSONResponse({"detail": "unknown camera stream"}, status_code=404)

    backend_instance_id = request.query_params.get("backend_instance_id")
    session_id = request.query_params.get("session_id")
    if backend_instance_id != monitor.snapshot().backend_instance_id:
        return JSONResponse({"detail": "superseded backend identity"}, status_code=409)
    if monitor.native.started is None or session_id != monitor.native.started.session_id:
        return JSONResponse({"detail": "superseded capture session identity"}, status_code=409)
    if monitor.native.status is None or not monitor.native.status.running:
        return JSONResponse({"detail": "capture is not running"}, status_code=503)
    now = time.monotonic()
    initial_slot = monitor.preview.camera(camera.name)
    if not _is_fresh(initial_slot, now=now):
        return JSONResponse({"detail": "no fresh cached image for this camera"}, status_code=503)
    if not monitor.try_admit_mjpeg_viewer(camera.name):
        return JSONResponse({"detail": "too many viewers for this camera"}, status_code=503)

    released = False

    def release_once() -> None:
        """Releases this viewer's MJPEG admission slot, idempotently."""
        nonlocal released
        if not released:
            released = True
            monitor.release_mjpeg_viewer(camera.name)

    async def generate():
        """Yields each new JPEG for the selected camera until the source
        goes stale, the session ends, or the viewer disconnects."""
        last_sequence: int | None = None
        # The cache has no new-frame signal, so it is polled -- but faster than the configured rate.
        # Sleeping one full period per poll delivered ~56 of 60 frames (period plus loop overhead) and
        # bunched the rest whenever a frame landed just after a poll. The configured rate is instead
        # enforced as a token bucket on what is SENT: one token per period, at most two banked, so a
        # 60 Hz source at a 60 Hz setting passes every frame despite its arrival jitter, while a 10 Hz
        # setting still decimates a 60 Hz source to ten evenly spaced frames per second.
        poll_s = min(_CAMERA_SAMPLE_INTERVAL_S / 4.0, 0.004)
        tokens = 1.0
        last_refill = time.monotonic()
        while True:
            if monitor.native.started is None or session_id != monitor.native.started.session_id:
                return
            if monitor.native.stopped is not None:
                return
            now = time.monotonic()
            slot = monitor.preview.camera(camera.name)
            if not _is_fresh(slot, now=now):
                # The source this response was serving went stale
                # mid-stream -- end it rather than let an old cache entry
                # keep being presented as live MJPEG.
                return
            tokens = min(2.0, tokens + (now - last_refill) / _CAMERA_SAMPLE_INTERVAL_S)
            last_refill = now
            if slot.sequence != last_sequence and tokens >= 1.0:
                tokens -= 1.0
                last_sequence = slot.sequence
                yield _multipart_jpeg(slot.jpeg, slot.sequence, slot.capture_time_ns, slot.receive_time_ns)
            await asyncio.sleep(poll_s)

    return _BoundedMjpegResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
        release=release_once,
    )
