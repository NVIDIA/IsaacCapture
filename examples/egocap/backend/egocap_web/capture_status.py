# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureClient callbacks and the one immutable status value the HTTP
endpoint serves.

``CaptureStatusMonitor`` owns exactly one CaptureClient attempt for the
whole backend process lifetime: it starts the attempt once at application
startup (an owned task, so FastAPI can serve the page while connection is
still in progress) and republishes an immutable snapshot every time a
lifecycle callback fires. The HTTP endpoint only ever reads the latest
published snapshot -- it never opens a connection, starts/stops capture or
polls hardware per request.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Protocol

from websockets.exceptions import WebSocketException

from capture_server import protocol as p
from capture_server.client import (
    CameraImage,
    CaptureClient,
    CaptureFrame,
    CaptureStarted,
    CaptureStatus,
    CaptureStatusType,
    CaptureStopped,
    RequestRejectedError,
    StoppedBeforeAcceptedError,
)

from .capture_preview import LatestPreviewCache
from .config import CaptureStatusConfig
from .preview_protocol import MAX_MJPEG_VIEWERS_PER_CAMERA, MAX_WS_VIEWERS, NativeFacts

_logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class RecordingCallbacks(Protocol):
    """The synchronous hooks a recording consumer needs from accepted
    native capture lifecycle events -- duck-typed (``RecordingSessionCoordinator``
    satisfies this without either module importing the other's concrete
    type) so this module stays decoupled from recording specifics; the two
    frame/image hooks only ever forward the SAME already-validated
    callback values preview already consumes, never a second capture
    connection. ``on_capture_lost`` forwards the one terminal signal
    (see ``_on_stopped``) a recording consumer needs to know the
    attempt itself ended, independent of any frame/image ever arriving
    again -- an active take must never silently admit a normal Stop once
    its real capture source is gone."""

    def on_capture_frame(self, frame: CaptureFrame) -> None: ...
    def on_capture_image(self, image: CameraImage) -> None: ...
    def on_capture_lost(self, reason: str) -> None: ...


# How long stop() may take to actually confirm on_stopped delivery before
# this reports completion as unconfirmed rather than hanging application
# shutdown indefinitely on a native resource that failed to release.
_STOP_CONFIRMATION_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class CaptureStatusResponse:
    """The one value GET /api/capture-status returns. Every field here is
    an observed fact from a real CaptureClient callback or local connection
    state -- nothing is inferred/fabricated. See ``frontend/README.md``'s
    "Response fields" table for the meaning of each member."""

    schema_version: int
    backend_instance_id: str
    station_label: str
    profile_label: str
    mode: p.CaptureMode | None
    session_id: str | None
    connecting: bool
    status_type: CaptureStatusType | None
    running: bool
    startup_stage: p.StartupStage | None
    message: str | None
    terminal: bool
    stop_reason: p.StopReason | None
    error_message: str | None
    status: p.StatusSnapshot | None

    def to_json(self) -> dict:
        """JSON-serializable form. ``status.updated_at_ns`` is a decimal
        string, not a JSON number, to avoid JavaScript's float-precision
        loss on large nanosecond timestamps -- every other integer here
        stays within JSON's safe-integer range."""
        return {
            "schema_version": self.schema_version,
            "backend_instance_id": self.backend_instance_id,
            "station_label": self.station_label,
            "profile_label": self.profile_label,
            "mode": None if self.mode is None else int(self.mode),
            "session_id": self.session_id,
            "connecting": self.connecting,
            "status_type": None if self.status_type is None else int(self.status_type),
            "running": self.running,
            "startup_stage": None if self.startup_stage is None else int(self.startup_stage),
            "message": self.message,
            "terminal": self.terminal,
            "stop_reason": None if self.stop_reason is None else int(self.stop_reason),
            "error_message": self.error_message,
            "status": None if self.status is None else _status_snapshot_to_json(self.status),
        }


def _status_snapshot_to_json(status: p.StatusSnapshot) -> dict:
    """JSON form of a StatusSnapshot for the response, adjusted for what a browser can decode."""
    data = status.to_json()
    # The wire protocol's own to_json() keeps this an ordinary JSON integer
    # (correct for a 64-bit server/client exchange); the browser decodes
    # JSON numbers as float64, which loses precision above 2**53. Convert
    # only at this HTTP boundary, not in the shared protocol serializer.
    data["updated_at_ns"] = str(data["updated_at_ns"])
    return data


@dataclass(frozen=True)
class _LocalFacts:
    """Transport-local facts the HTTP response needs that are not part of
    a native CaptureClient callback: whether the pre-acceptance attempt is
    still in flight, and a local diagnostic (a pre-acceptance failure, or
    an unconfirmed-shutdown note from ``stop()``'s own timeout). Paired
    with ``NativeFacts`` by ``_derive_response`` below -- the only two
    inputs ``CaptureStatusResponse`` is ever built from."""

    connecting: bool = True
    error_message: str | None = None
    # Set only by stop()'s own confirmation timeout: on_stopped never fired
    # within the bound, so native.status's last-seen `running` can no
    # longer be trusted as current -- force the response to stop claiming
    # a native attempt is still running once this backend has given up
    # waiting for it to confirm otherwise.
    stop_timed_out: bool = False


def _derive_response(
    *,
    schema_version: int,
    backend_instance_id: str,
    station_label: str,
    profile_label: str,
    native: NativeFacts,
    local: _LocalFacts,
) -> CaptureStatusResponse:
    """The one place CaptureStatusResponse is built, from native +
    transport-local facts -- every callback updates ``native``/``local``
    and calls this, rather than hand-replacing a second, independently
    maintained response value field-by-field."""
    mode = native.started.mode if native.started is not None else None
    if native.stopped is not None:
        session_id = native.stopped.session_id
    elif native.status is not None:
        session_id = native.status.session_id
    elif native.started is not None:
        session_id = native.started.session_id
    else:
        session_id = None

    if native.stopped is not None:
        # Terminal: nothing from the pre-stop CaptureStatus (status_type/
        # startup_stage/message) is still meaningful, and running is
        # always false -- see _on_stopped.
        return CaptureStatusResponse(
            schema_version=schema_version,
            backend_instance_id=backend_instance_id,
            station_label=station_label,
            profile_label=profile_label,
            mode=mode,
            session_id=session_id,
            connecting=False,
            status_type=None,
            running=False,
            startup_stage=None,
            message=None,
            terminal=True,
            stop_reason=native.stopped.reason,
            error_message=local.error_message or native.stopped.error_message,
            status=native.latest_status,
        )

    status = native.status
    return CaptureStatusResponse(
        schema_version=schema_version,
        backend_instance_id=backend_instance_id,
        station_label=station_label,
        profile_label=profile_label,
        mode=mode,
        session_id=session_id,
        connecting=local.connecting,
        status_type=status.type if status is not None else None,
        running=False if local.stop_timed_out else (status.running if status is not None else False),
        startup_stage=status.startup_stage if status is not None else None,
        message=status.message if status is not None else None,
        terminal=False,
        stop_reason=None,
        error_message=local.error_message,
        status=native.latest_status,
    )


class CaptureStatusMonitor:
    """Owns one CaptureClient attempt and the latest published status.

    Deliberately no statechart/transition table: each callback is a short,
    synchronous replacement of the immutable current value; this increment
    deliberately has no reducers and no state machine.
    """

    def __init__(
        self,
        config: CaptureStatusConfig,
        *,
        backend_instance_id: str,
        recording: RecordingCallbacks | None = None,
    ) -> None:
        self._config = config
        self._client = CaptureClient(config.capture_server_url)
        # Fans each accepted callback to preview (below) AND recording
        # consumption -- both share this one native connection; this
        # module never creates a second CaptureClient. None (the default)
        # means no recording consumer exists yet -- see set_recording(),
        # used when app.py constructs the monitor before the real
        # recording consumer (which itself needs this monitor to
        # construct).
        self._recording = recording
        self._schema_version = SCHEMA_VERSION
        self._backend_instance_id = backend_instance_id
        self._station_label = config.station_label
        self._profile_label = config.profile_label
        self._startup_task: asyncio.Task[None] | None = None
        # The one owner value for every native CaptureClient callback fact
        # (preview_protocol.NativeFacts): replaced atomically per callback
        # via dataclasses.replace(). Paired with ``_local`` (transport-local
        # facts a native callback never carries), these two values are the
        # only inputs _derive_response()/snapshot() ever read -- there is
        # no second, independently hand-updated CaptureStatusResponse.
        self.native = NativeFacts()
        self._local = _LocalFacts()
        # Bumped only on a meaningful lifecycle/status/descriptor change
        # (on_started/on_status/on_stopped and the two never-attempted
        # paths) -- sample arrival/heartbeats never advance it, matching the
        # preview snapshot's revision contract. Also doubles as the "what
        # native observation has this viewer actually been sent" marker
        # for WS binary-send eligibility -- see preview_routes.py.
        self.revision: int = 0
        self.preview = LatestPreviewCache()
        self._ws_viewer_count = 0
        self._mjpeg_viewer_counts: dict[str, int] = {}
        # Set once on_stopped actually fires -- OR once _run_attempt reaches
        # a terminal outcome that will never produce on_stopped at all (never
        # accepted, see _run_attempt) -- so an explicit stop() can confirm
        # real completion rather than trusting CaptureClient.stop()'s own
        # return value (it only requests cancellation, it does not wait for
        # it) or waiting the full timeout for a callback that is never coming.
        self._stopped_event = asyncio.Event()
        # Set synchronously by stop(), checked by _run_attempt before it
        # calls CaptureClient.start() -- closes the race where stop() runs
        # before the startup task has had a chance to run at all (so
        # CaptureClient.stop() would find no attempt to cancel yet): once
        # this is set, _run_attempt never connects in the first place.
        self._stop_requested = asyncio.Event()

    def try_admit_ws_viewer(self) -> bool:
        """Reserves a WS viewer slot if under MAX_WS_VIEWERS; returns whether it was admitted."""
        if self._ws_viewer_count >= MAX_WS_VIEWERS:
            return False
        self._ws_viewer_count += 1
        return True

    def release_ws_viewer(self) -> None:
        """Releases one previously-admitted WS viewer slot."""
        self._ws_viewer_count = max(0, self._ws_viewer_count - 1)

    def try_admit_mjpeg_viewer(self, camera_name: str) -> bool:
        """Reserves an MJPEG viewer slot for one camera if under the per-camera cap."""
        count = self._mjpeg_viewer_counts.get(camera_name, 0)
        if count >= MAX_MJPEG_VIEWERS_PER_CAMERA:
            return False
        self._mjpeg_viewer_counts[camera_name] = count + 1
        return True

    def release_mjpeg_viewer(self, camera_name: str) -> None:
        """Releases one previously-admitted MJPEG viewer slot for one camera."""
        count = self._mjpeg_viewer_counts.get(camera_name, 0)
        self._mjpeg_viewer_counts[camera_name] = max(0, count - 1)

    def snapshot(self) -> CaptureStatusResponse:
        """Returns the current immutable value, derived fresh from
        ``self.native``/``self._local``. Never awaits, never touches the
        network -- exactly what the HTTP endpoint needs per request."""
        return _derive_response(
            schema_version=self._schema_version,
            backend_instance_id=self._backend_instance_id,
            station_label=self._station_label,
            profile_label=self._profile_label,
            native=self.native,
            local=self._local,
        )

    @property
    def tracking_descriptor(self) -> p.TrackingStream | None:
        """The accepted tracking stream descriptor, if any."""
        if self.native.started is None:
            return None
        for stream in self.native.started.streams:
            if isinstance(stream, p.TrackingStream):
                return stream
        return None

    @property
    def camera_descriptors(self) -> tuple[p.CameraStream, ...]:
        """The accepted camera stream descriptors, if any."""
        if self.native.started is None:
            return ()
        return tuple(s for s in self.native.started.streams if isinstance(s, p.CameraStream))

    def camera_descriptor_by_stream_id(self, stream_id: int) -> p.CameraStream | None:
        """The accepted camera descriptor advertising this stream ID, if any."""
        for cam in self.camera_descriptors:
            if cam.stream_id == stream_id:
                return cam
        return None

    def set_recording(self, recording: RecordingCallbacks) -> None:
        """Attaches the recording consumer after construction -- used when
        that consumer itself needs a reference to this monitor to build
        (e.g. its config/readiness closures read ``camera_descriptors``),
        so it cannot exist yet when this monitor is constructed."""
        self._recording = recording

    async def start(self) -> None:
        """Starts the one owned CaptureClient attempt in its own task, so
        the caller (FastAPI's lifespan) returns immediately and the
        application can serve the status page while connection/acceptance
        is still in progress."""
        self._startup_task = asyncio.create_task(self._run_attempt())

    async def _run_attempt(self) -> None:
        """Runs the one CaptureClient attempt, unless stop() already ran; a pre-acceptance failure is
        reported as a local diagnostic (terminal stays false) instead of leaving the page connecting."""
        if self._stop_requested.is_set():
            # stop() already ran before this task got a turn to execute at
            # all -- never connect in the first place (there is nothing for
            # CaptureClient.stop() to cancel yet, and starting now would mean
            # acquiring a capture attempt after shutdown already asked not
            # to). No attempt was ever made, so no on_stopped will ever fire.
            self._local = _LocalFacts(connecting=False, error_message="stopped before the capture attempt started")
            self.revision += 1
            self._stopped_event.set()
            return
        try:
            await self._client.start(
                sources=self._config.sources,
                cameras=self._config.cameras,
                plugins=self._config.plugins,
                on_started=self._on_started,
                on_status=self._on_status,
                on_capture_frame=self._on_capture_frame,
                on_capture_image=self._on_capture_image,
                on_stopped=self._on_stopped,
            )
        except (
            RequestRejectedError,
            StoppedBeforeAcceptedError,
            OSError,
            WebSocketException,
            p.ProtocolError,
        ) as exc:
            # A real pre-acceptance failure: rejected Begin, explicit stop
            # before acceptance, a transport-level failure (refused/DNS),
            # a handshake failure (refused HTTP upgrade, the server closing
            # the socket outright for a second owner connection -- code 1013
            # never produces a RequestRejected JSON reply, only a close), or
            # a malformed first message. Report it honestly as a local
            # diagnostic rather than leaving the page stuck on "Connecting"
            # forever. terminal stays False -- no SessionStopped was ever
            # actually received from the server. No attempt was ever
            # accepted, so on_stopped will never fire for this one either.
            self._local = _LocalFacts(connecting=False, error_message=str(exc) or type(exc).__name__)
            self.revision += 1
            self._stopped_event.set()

    def _on_started(self, started: CaptureStarted) -> None:
        """The attempt was accepted: publish its mode and session id (running stays false until BEGUN)."""
        # Accepted, but not yet BEGUN -- do not set running True here even
        # though CaptureStarted.running is always False already; this is
        # the one place "accepted" becomes visible to the browser.
        self.native = replace(self.native, started=started)
        self._local = replace(self._local, connecting=False)
        self.revision += 1

    def _on_status(self, status: CaptureStatus) -> None:
        """A STARTING, BEGUN or STATUS_CHANGE callback: replace the native status fields with what it reported."""
        self.native = replace(
            self.native,
            status=status,
            latest_status=status.status if status.status is not None else self.native.latest_status,
        )
        self.revision += 1

    def _on_capture_frame(self, frame: CaptureFrame) -> None:
        """Retains one accepted tracking frame in the latest-sample preview
        cache, and (if a recording consumer exists) also offers it to
        recording -- both consumers share this one validated callback,
        never a second capture connection."""
        # Verify current session association: a frame from a superseded
        # attempt (there is only ever one per process, but the callback
        # contract still documents this check) must never land in the
        # preview cache.
        if self.native.started is None or frame.session_id != self.native.started.session_id:
            return
        self.preview.set_tracking(
            sequence=frame.sequence,
            capture_time_ns=frame.capture_time_ns,
            receive_time_ns=frame.receive_time_ns,
            payload=frame.payload,
        )
        if self._recording is not None:
            try:
                self._recording.on_capture_frame(frame)
            except Exception:  # noqa: BLE001 -- a recording-side bug must never disrupt capture/preview itself
                _logger.exception("recording on_capture_frame raised")

    def _on_capture_image(self, image: CameraImage) -> None:
        """Retains one accepted camera image in the latest-sample preview
        cache, and (if a recording consumer exists) also offers it to
        recording."""
        if self.native.started is None or image.session_id != self.native.started.session_id:
            return
        if image.name not in {c.name for c in self.camera_descriptors}:
            return
        self.preview.set_camera(
            image.name,
            sequence=image.sequence,
            capture_time_ns=image.capture_time_ns,
            receive_time_ns=image.receive_time_ns,
            jpeg=image.jpeg,
        )
        if self._recording is not None:
            try:
                self._recording.on_capture_image(image)
            except Exception:  # noqa: BLE001 -- a recording-side bug must never disrupt capture/preview itself
                _logger.exception("recording on_capture_image raised")

    def _on_stopped(self, stopped: CaptureStopped) -> None:
        """The accepted attempt ended, confirmed by the server or not: mark
        native facts terminal, and (if a recording consumer exists)
        forward this as the one terminal capture-loss signal -- an active
        take must never silently admit a normal Stop once its real
        capture source is gone."""
        self.native = replace(
            self.native,
            stopped=stopped,
            latest_status=stopped.status if stopped.status is not None else self.native.latest_status,
        )
        self.revision += 1
        self.preview.clear()
        self._stopped_event.set()
        if self._recording is not None:
            reason = stopped.error_message or (
                stopped.reason.name if stopped.reason is not None else "unconfirmed"
            )
            try:
                self._recording.on_capture_lost(reason)
            except Exception:  # noqa: BLE001 -- a recording-side bug must never disrupt capture shutdown
                _logger.exception("recording on_capture_lost raised")

    async def stop(self) -> None:
        """Requests cancellation and waits, bounded, for the actual
        terminal callback -- CaptureClient.stop() returning only means the
        request was sent, not that on_stopped fired. If the bound expires,
        report the attempt's completion as unconfirmed rather than
        pretending resources were released."""
        # Set before anything else, synchronously: _run_attempt checks this
        # before it calls CaptureClient.start(), so a stop() that runs
        # before the startup task has executed at all still prevents it
        # from connecting -- see _run_attempt.
        self._stop_requested.set()
        await self._client.stop()
        try:
            await asyncio.wait_for(self._stopped_event.wait(), timeout=_STOP_CONFIRMATION_TIMEOUT_S)
        except TimeoutError:
            if self.native.stopped is None:
                self._local = replace(
                    self._local,
                    stop_timed_out=True,
                    error_message=(
                        self._local.error_message
                        or "shutdown: stop() did not confirm termination within "
                        f"{_STOP_CONFIRMATION_TIMEOUT_S:.0f}s"
                    ),
                )
        if self._startup_task is not None:
            # The attempt's own task should already be finishing/finished
            # by the time on_stopped (or this timeout) is reached; surface
            # any exception it still raised rather than silently dropping it.
            try:
                await asyncio.wait_for(self._startup_task, timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
