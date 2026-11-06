# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Async client for the CaptureServer wire protocol: a two-call
``start()``/``stop()`` facade with five lifecycle/data callbacks, built
on top of ``capture_server.protocol``'s message encode/decode.
``on_status`` delivers a single ``CaptureStatus`` struct per callback,
consolidating the wire's ``SessionStarting | SessionBegun |
SessionStatus`` union into one consistent shape distinguished by its
``type``.

This module owns the WebSocket transport (``websockets``); it is the only
place in this package that imports it. Scope note: this is a client-side
state machine only -- it has no IsaacCapture/OAK-D adapter and no
production-grade backpressure/drop-oldest queueing; those remain later,
separate work.

Ownership model: ``start()`` creates exactly one supervisor task
(``_run_attempt``) per attempt, and that task is the sole owner of the
attempt's socket, acceptance state, stream registry, protocol reads/writes,
callback dispatch and finalization -- everything for that attempt lives on
one ``_Attempt`` object the supervisor exclusively mutates. Nothing else
ever reaches into those fields or calls ``task.cancel()`` on the
supervisor: ``start()`` only waits on the attempt's readiness event, and
``stop()`` only sets the attempt's stop flag. The supervisor observes that
flag cooperatively at its own network-wait boundaries (see ``_race``) and
reacts to it there, rather than being torn down by an external cancellation
mid-cleanup. This replaces an earlier design with several independently
cancelling pieces (a receive loop, a deadline watchdog, a separate cleanup
retry) that went through repeated review rounds without converging -- see
this commit's own message for why.

Completion contract: every accepted attempt (one that received SessionInit)
calls ``on_stopped`` exactly once, from ``_finalize``, which is the single
path into transport close + callback invocation + clearing the active
attempt. A callback other than ``on_stopped`` raising, a transport failure,
a lost Stop reply, or an explicit ``stop()`` call all end up at that one
path.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import IntEnum
from typing import Any, ClassVar, TypeVar
from urllib.parse import urlsplit

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from . import protocol as p

# FrameDescription/FieldDescriptor are the same objects
# capture_server.protocol uses internally to decode the wire SessionInit --
# applications decode CaptureFrame.payload themselves using this
# byte-level information, so there is no separate client-facing shape
# to maintain.
FrameDescription = p.TrackingStream
FieldDescriptor = p.FieldDescriptor

__all__ = [
    "CalibrationEndedError",
    "CalibrationStopped",
    "CaptureClient",
    "DeviceQueryError",
    "ManusPluginConfig",
    "CaptureFrame",
    "CaptureStarted",
    "CaptureStatus",
    "CaptureStatusType",
    "CaptureStopped",
    "CameraImage",
    "FieldDescriptor",
    "FrameDescription",
    "RequestRejectedError",
    "StoppedBeforeAcceptedError",
]

T = TypeVar("T")
Callback = Callable[[T], "None | Awaitable[None]"]

_DEFAULT_RATE_HZ = 60.0
_ACCEPT_TIMEOUT_S = 10.0
_STOP_TIMEOUT_S = 10.0
# A device query is one short exchange; the server bounds its own scan well inside this.
_QUERY_TIMEOUT_S = 20.0
_CLOSE_TIMEOUT_S = 2.0
# The documented transport ceiling is a complete JPEG (2 MiB) plus the
# 32-byte binary header; websockets' own default max_size (~1 MiB) would
# silently reject a legal camera message.
_MAX_MESSAGE_BYTES = 2 * 1024 * 1024 + p.HEADER_SIZE

_CALLBACK_NAMES = (
    "on_started",
    "on_status",
    "on_capture_frame",
    "on_capture_image",
    "on_stopped",
)


class _StopWonRace(Exception):
    """Internal-only signal that ``_race``'s ``stop_event`` won -- never
    raised as ``asyncio.CancelledError`` (see ``StoppedBeforeAcceptedError``
    for why) and never meant to cross out of this module. Both ``_race``
    callers consume it themselves: the pre-acceptance handshake lets it
    propagate up to ``_run_attempt``, which stores it on ``start_exception``
    for ``start()`` to translate into ``StoppedBeforeAcceptedError``; the
    post-acceptance receive loop catches it directly and just re-enters as
    "now stopping"."""


@dataclass(frozen=True)
class CaptureStarted:
    """The value delivered to ``on_started``. Same shape as ``protocol.SessionInit`` plus
    ``running`` (always False: IC startup has been initiated, the capture
    session has not begun). ``streams`` entries are the exact ``FrameDescription``/
    ``CameraStream`` objects the wire SessionInit carried -- no copy or
    conversion, since ``protocol.TrackingStream`` already describes decoded
    field names/wire dtypes/shapes/offsets with no further client-facing
    shape needed."""

    session_id: str
    running: bool
    mode: p.CaptureMode
    effective_config: p.CaptureConfiguration
    clock: p.CaptureClock
    streams: tuple[FrameDescription | p.CameraStream, ...]


def _to_client_started(wire: p.SessionInit) -> CaptureStarted:
    return CaptureStarted(
        session_id=wire.session_id,
        running=False,
        mode=wire.mode,
        effective_config=wire.effective_config,
        clock=wire.clock,
        streams=wire.streams,
    )


class CaptureStatusType(IntEnum):
    """The ``CaptureStatus.type`` discriminator. Deliberately distinct numeric values
    from the wire ``MessageType`` (SessionStarting=3/SessionBegun=4/
    SessionStatus=5): this is an independent, client-facing enum, not a
    restatement of the wire discriminator."""

    STARTING = 1
    BEGUN = 2
    STATUS_CHANGE = 3


@dataclass(frozen=True)
class CaptureStatus:
    """The one value delivered to ``on_status``. Replaces the earlier raw
    ``SessionStarting | SessionBegun | SessionStatus`` union with a single
    consistent struct: ``startup_stage``/``message`` are populated only for
    STARTING (None otherwise); ``status`` can be None only for STARTING
    (before any observation exists) and is always present for BEGUN/
    STATUS_CHANGE. No public wire ``MessageType``/``version`` member."""

    session_id: str
    type: CaptureStatusType
    running: bool
    startup_stage: p.StartupStage | None
    message: str | None
    status: p.StatusSnapshot | None


def _to_client_status(
    event: p.SessionStarting | p.SessionBegun | p.SessionStatus,
) -> CaptureStatus:
    if isinstance(event, p.SessionStarting):
        return CaptureStatus(
            session_id=event.session_id,
            type=CaptureStatusType.STARTING,
            running=False,
            startup_stage=event.stage,
            message=event.message,
            status=event.status,
        )
    status_type = CaptureStatusType.BEGUN if isinstance(event, p.SessionBegun) else CaptureStatusType.STATUS_CHANGE
    return CaptureStatus(
        session_id=event.session_id,
        type=status_type,
        running=True,
        startup_stage=None,
        message=None,
        status=event.status,
    )


@dataclass(frozen=True)
class CaptureFrame:
    """The value delivered to ``on_capture_frame``. ``payload`` is the raw, unchanged tracking blob; the
    client performs no numeric unpacking. ``description`` is the exact
    ``FrameDescription`` object delivered through ``CaptureStarted.streams``
    for this stream, not a rebuilt copy -- see ``_StreamRegistry``, which
    caches that same object for reuse here."""

    session_id: str
    description: FrameDescription
    stream_id: int
    sequence: int
    capture_time_ns: int
    receive_time_ns: int
    payload: bytes


@dataclass(frozen=True)
class CameraImage:
    """The value delivered to ``on_capture_image``. Stream/sequence/timestamp fields are direct attributes,
    not a nested metadata/header object. ``decode()`` is documented as
    taking no arguments; ``expected_size`` is bound internally from the
    image's registered CameraStream at dispatch time, not part of the
    public member list."""

    session_id: str
    stream_id: int
    sequence: int
    capture_time_ns: int
    receive_time_ns: int
    name: str
    jpeg: bytes
    expected_size: tuple[int, int] | None = dataclass_field(default=None, repr=False)

    def decode(self) -> np.ndarray:
        """Decode ``jpeg`` into a new, writable RGB uint8 array of shape
        [height, width, 3], validated against the registered camera's
        dimensions when known."""
        from PIL import Image, UnidentifiedImageError  # lazy: only needed to decode images.

        try:
            with Image.open(io.BytesIO(self.jpeg)) as img:
                if img.format != "JPEG":
                    raise p.ProtocolError(f"camera {self.name!r} image is {img.format}, expected JPEG")
                if self.expected_size is not None and img.size != self.expected_size:
                    raise p.ProtocolError(
                        f"camera {self.name!r} image is {img.size}, expected {self.expected_size}"
                    )
                return np.array(img.convert("RGB"), dtype=np.uint8)
        except UnidentifiedImageError as exc:
            raise p.ProtocolError(f"camera {self.name!r} image is not a valid JPEG") from exc
        except OSError as exc:
            raise p.ProtocolError(f"camera {self.name!r} image could not be decoded: {exc}") from exc


class RequestRejectedError(Exception):
    """The server rejected Begin before accepting a session.
    ``str(error)`` returns ``error_message``."""

    def __init__(self, session_id: str, reason: p.RejectionReason, error_message: str) -> None:
        super().__init__(error_message)
        self.session_id = session_id
        self.reason = reason
        self.error_message = error_message


class StoppedBeforeAcceptedError(Exception):
    """``start()`` raises this when an explicit ``stop()`` call ends the
    attempt before acceptance. Deliberately NOT ``asyncio.CancelledError``, even though
    ``stop()`` is documented as "requests cancellation of the client's
    current session": that phrase describes a domain-level session
    outcome, not a runtime cancellation of the task running ``start()``.
    The task awaiting ``start()`` was never actually cancelled (nothing
    here calls ``task.cancel()`` on it) -- raising a real CancelledError
    into it anyway would be a bare CancelledError in a task nobody
    cancelled, which can desync an enclosing ``asyncio.TaskGroup``'s or
    ``asyncio.timeout()``'s own cancel-scope bookkeeping for callers that
    wrap ``start()`` in one. Genuine external cancellation of ``start()``'s
    own await (the caller's own task actually being cancelled) is a
    separate case and still raises real ``asyncio.CancelledError``, as
    documented."""


@dataclass(frozen=True)
class CaptureStopped:
    """The client's completion value. A non-None ``reason`` confirms a
    server terminal reply; None means unconfirmed local completion, never a
    fabricated server failure. ``running`` is always False: the attempt has
    ended."""

    session_id: str
    running: bool
    reason: p.StopReason | None
    error_message: str | None
    status: p.StatusSnapshot | None
    streams: tuple[p.StreamCounters, ...] | None


@dataclass(frozen=True)
class ManusPluginConfig:
    """The Manus plugin and, optionally, MCAL files to load for each glove.

    The names are bare ``.mcal`` basenames of files in the server's managed MCAL folder, as listed by
    :meth:`CaptureClient.query_devices`; they are never paths on this machine. A name only selects a file:
    whether the glove accepted it is reported by the plugin's own per-hand status.
    """

    left_calibration: str | None = None
    right_calibration: str | None = None
    plugin: ClassVar[p.PluginType] = p.PluginType.MANUS

    def __post_init__(self) -> None:
        """Reject names the protocol cannot carry, and one file chosen for both hands."""
        for name in (self.left_calibration, self.right_calibration):
            if name is not None and not p.is_mcal_name(name):
                raise ValueError(f"calibration file must be an .mcal basename, got {name!r}")
        if self.left_calibration is not None and self.left_calibration == self.right_calibration:
            raise ValueError("left_calibration and right_calibration must name different files")


class DeviceQueryError(Exception):
    """The device list could not be retrieved (transport, timeout, protocol, or a server without the query).

    Never reported as an empty list: an answer with no MCAL files is a READY entry with none.
    """


class CalibrationEndedError(Exception):
    """A pending calibration confirmation was released because the calibration ended first."""


@dataclass(frozen=True)
class CalibrationStopped:
    """The client's calibration completion value.

    A non-None ``reason`` confirms the server's terminal message. ``None`` means the connection or protocol
    failed before one arrived: the server-side outcome is unknown, no files are offered, and the gloves may
    still be in use until the server reports otherwise.
    """

    session_id: str
    reason: p.CalibrationStopReason | None
    result: p.CalibrationResult | None
    error_message: str | None


@dataclass
class _CalibrationAttempt:
    """All state of one ``start_calibration`` call, owned by its supervisor task."""

    session_id: str
    sides: tuple[p.HandSide, ...]
    callbacks: dict[str, Callback[Any] | None]
    ws: Any = None
    stop_requested: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    # Set when on_started has been attempted, or the attempt failed before acceptance.
    ready: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    accepted: bool = False
    # Set once the attempt is finalizing, before its pending confirmations are failed, so none can register after.
    ended: bool = False
    start_exception: BaseException | None = None
    events: asyncio.Queue[tuple[str, Any] | None] = dataclass_field(default_factory=asyncio.Queue)
    pending: dict[str, asyncio.Future[None]] = dataclass_field(default_factory=dict)
    accepted_tokens: set[str] = dataclass_field(default_factory=set)
    stop_sent: bool = False
    callback_error: BaseException | None = None
    task: asyncio.Task[None] | None = None


def _is_awaitable(value: Any) -> bool:
    return inspect.isawaitable(value)


async def _invoke(callback: Callback[T] | None, value: T) -> None:
    if callback is None:
        return
    result = callback(value)
    if _is_awaitable(result):
        _ = await result  # intentionally discarded: this runs the callback for its side effects only


@dataclass
class _StreamRegistry:
    """Caches SessionInit's stream descriptors for decoding binary
    messages -- a tracking stream (ID 0) and up to three camera streams."""

    tracking: p.TrackingStream | None = None
    cameras: dict[int, p.CameraStream] = dataclass_field(default_factory=dict)

    @classmethod
    def from_session_init(cls, event: p.SessionInit) -> _StreamRegistry:
        registry = cls()
        for stream in event.streams:
            if isinstance(stream, p.TrackingStream):
                registry.tracking = stream
            else:
                registry.cameras[stream.stream_id] = stream
        return registry


@dataclass
class _Attempt:
    """All state belonging to one ``start()`` call. Exclusively created,
    read and mutated by that attempt's single supervisor task
    (``_run_attempt``) -- no other code path touches these fields, so a
    stale reference from an earlier, already-finished attempt can never
    affect a later one."""

    session_id: str
    begin: p.BeginRequest
    callbacks: dict[str, Callback[Any] | None]
    ws: Any = None
    streams: _StreamRegistry = dataclass_field(default_factory=_StreamRegistry)
    init_json: dict[str, Any] | None = None
    begun: bool = False
    begun_event: p.SessionBegun | None = None
    last_status: p.StatusSnapshot | None = None
    sequences: dict[int, int] = dataclass_field(default_factory=dict)
    stop_requested: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    # Set once by the supervisor when on_started has been attempted (whether it
    # raised or not) or when the attempt failed before acceptance -- this is
    # exactly what start() waits for.
    ready: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    accepted: bool = False
    # Set only for a pre-acceptance failure; start() raises this and never
    # calls on_stopped, since no session was ever accepted.
    start_exception: BaseException | None = None
    # A strong reference to the supervisor task -- explicit ownership, and
    # lets start()'s pre-acceptance cancellation handler wait for the
    # supervisor's own cleanup to finish (see start()) without ever calling
    # task.cancel() on it.
    task: asyncio.Task[None] | None = None


class CaptureClient:
    """Two-operation async client: ``start()`` opens a session and delivers
    its lifecycle through callbacks; ``stop()`` requests cancellation."""

    def __init__(self, url: str) -> None:
        if not isinstance(url, str) or not url or not url.startswith(("ws://", "wss://")):
            raise ValueError("CaptureClient url must be a nonempty ws:// or wss:// string")
        self._url = url
        self._attempt: _Attempt | None = None
        self._calibration: _CalibrationAttempt | None = None

    async def start(
        self,
        *,
        sources: Sequence[p.SourceType],
        cameras: Sequence[p.CameraSelection] = (),
        plugins: Sequence[p.PluginType | ManusPluginConfig] = (),
        on_started: Callback[CaptureStarted] | None = None,
        on_status: Callback[CaptureStatus] | None = None,
        on_capture_frame: Callback[CaptureFrame] | None = None,
        on_capture_image: Callback[CameraImage] | None = None,
        on_stopped: Callback[CaptureStopped] | None = None,
    ) -> None:
        if self._attempt is not None or self._calibration is not None:
            raise RuntimeError(
                "CaptureClient.start called while an attempt is active; the active attempt continues"
            )
        callbacks = {
            "on_started": on_started,
            "on_status": on_status,
            "on_capture_frame": on_capture_frame,
            "on_capture_image": on_capture_image,
            "on_stopped": on_stopped,
        }
        for name in _CALLBACK_NAMES:
            value = callbacks[name]
            if value is not None and not callable(value):
                raise ValueError(f"CaptureClient.start {name} must be callable or None")
        begin = self._build_begin_request(sources=sources, cameras=cameras, plugins=plugins)

        attempt = _Attempt(session_id=begin.session_id, begin=begin, callbacks=callbacks)
        self._attempt = attempt
        attempt.task = asyncio.ensure_future(self._run_attempt(attempt))

        try:
            await attempt.ready.wait()
        except asyncio.CancelledError:
            # start()'s own caller cancelled this await -- this does NOT
            # touch the supervisor task (a separate task entirely; nothing
            # here calls task.cancel() on it). Signal it to stop the same
            # way an explicit stop() would, then -- only for the
            # pre-acceptance case -- wait (bounded, without cancelling the
            # supervisor: asyncio.wait never cancels what it's waiting
            # for) for the supervisor to
            # actually finish tearing the handshake down before this
            # exception reaches our own caller, so an immediate retry
            # doesn't race "attempt already active" against our own
            # not-yet-finished cleanup. After acceptance, the supervisor's
            # completion is already asynchronous via on_stopped -- that
            # existing behavior is unchanged here.
            attempt.stop_requested.set()
            if not attempt.accepted and attempt.task is not None:
                await asyncio.wait({attempt.task}, timeout=_ACCEPT_TIMEOUT_S)
            raise

        if attempt.start_exception is not None:
            if isinstance(attempt.start_exception, _StopWonRace):
                # Translate the internal race-loss signal into the public,
                # deliberately-not-CancelledError exception -- see
                # StoppedBeforeAcceptedError's docstring for why.
                raise StoppedBeforeAcceptedError(str(attempt.start_exception)) from None
            raise attempt.start_exception
        # Otherwise accepted: on_started has been attempted, return normally.

    async def stop(self) -> None:
        attempt = self._attempt
        if attempt is None:
            return  # idle, or the previous attempt already finished
        attempt.stop_requested.set()  # idempotent: a no-op if already set

    # -- calibration -------------------------------------------------------

    async def start_calibration(
        self,
        *,
        sides: Sequence[p.HandSide],
        on_started: Callback[p.CalibrationStarted] | None = None,
        on_status: Callback[p.CalibrationStatus] | None = None,
        on_stopped: Callback[CalibrationStopped] | None = None,
    ) -> None:
        """Start a Manus glove calibration on the server; returns once the server accepts it.

        Acceptance does not mean the gloves are connected or the calibration finished. Every accepted
        attempt ends with exactly one ``on_stopped`` call, including after a lost connection (then with
        ``reason=None``). A rejected start raises :class:`RequestRejectedError` and calls no callback. Callbacks
        run one at a time on this loop; never wait inside one for a later callback.
        """
        if self._attempt is not None or self._calibration is not None:
            raise RuntimeError(
                "CaptureClient.start_calibration called while an attempt is active; the active attempt continues"
            )
        try:
            request = p.StartCalibrationRequest(session_id=str(uuid.uuid4()), sides=tuple(sides))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid CaptureClient.start_calibration arguments: {exc}") from exc
        callbacks = {"on_started": on_started, "on_status": on_status, "on_stopped": on_stopped}
        for name, value in callbacks.items():
            if value is not None and not callable(value):
                raise ValueError(f"CaptureClient.start_calibration {name} must be callable or None")
        attempt = _CalibrationAttempt(session_id=request.session_id, sides=request.sides, callbacks=callbacks)
        self._calibration = attempt
        attempt.task = asyncio.ensure_future(self._run_calibration(attempt, request))
        try:
            await attempt.ready.wait()
        except asyncio.CancelledError:
            # Same contract as start(): signal a stop, let the supervisor tidy up, never cancel it.
            attempt.stop_requested.set()
            if not attempt.accepted and attempt.task is not None:
                await asyncio.wait({attempt.task}, timeout=_ACCEPT_TIMEOUT_S)
            raise
        if attempt.start_exception is not None:
            raise attempt.start_exception

    async def confirm_calibration_step(self, *, step_token: str) -> None:
        """Start the step offered under ``step_token``; returns once the worker accepted it.

        Acceptance means the SDK accepted starting the step, not that the step or calibration finished. A
        repeat of an accepted token returns at once and never starts another step. A stale or unoffered token
        raises :class:`RequestRejectedError` and leaves the attempt unchanged; if the calibration ends first,
        :class:`CalibrationEndedError` is raised.
        """
        attempt = self._calibration
        if attempt is None or not attempt.accepted:
            raise RuntimeError("no accepted calibration is active")
        if not isinstance(step_token, str) or not step_token:
            raise ValueError("step_token must be a nonempty string")
        if step_token in attempt.accepted_tokens:
            return
        future = attempt.pending.get(step_token)
        if future is None:
            # Finalizing may already have failed the pending confirmations; a new one would never be resolved.
            if attempt.ended:
                raise CalibrationEndedError("the calibration ended before the step was accepted")
            future = asyncio.get_running_loop().create_future()
            attempt.pending[step_token] = future
            request = p.ConfirmCalibrationStepRequest(session_id=attempt.session_id, step_token=step_token)
            await self._send_calibration(attempt, request.to_json())
            # The send can yield while the attempt ends; finalizing then no longer sees this future.
            if attempt.ended and not future.done():
                attempt.pending.pop(step_token, None)
                future.set_exception(CalibrationEndedError("the calibration ended before the step was accepted"))
        # Shielded: a cancelled caller must not cancel the future a retry (or the reader) still uses.
        await asyncio.shield(future)

    async def stop_calibration(self) -> None:
        """Ask the server to cancel the active calibration. Idempotent; wait for ``on_stopped`` for the result."""
        attempt = self._calibration
        if attempt is None:
            return
        attempt.stop_requested.set()
        if attempt.accepted and not attempt.stop_sent:
            attempt.stop_sent = True
            await self._send_calibration(
                attempt, p.StopCalibrationRequest(session_id=attempt.session_id).to_json()
            )

    async def _send_calibration(self, attempt: _CalibrationAttempt, data: dict[str, Any]) -> None:
        """Send one request; a dead transport is reported by the reader as an unconfirmed ending."""
        ws = attempt.ws
        if ws is None:
            return
        try:
            await asyncio.wait_for(ws.send(_to_json_text(data)), timeout=_STOP_TIMEOUT_S)
        except (ConnectionClosed, OSError, asyncio.TimeoutError):
            pass

    async def _run_calibration(self, attempt: _CalibrationAttempt, request: p.StartCalibrationRequest) -> None:
        """The sole owner of one calibration attempt, from connect through finalization."""
        try:
            started = await self._calibration_handshake(attempt, request)
        except BaseException as exc:  # noqa: BLE001 -- reported through start_exception, not raised here
            await self._close_calibration_ws(attempt)
            attempt.start_exception = (
                StoppedBeforeAcceptedError(str(exc)) if isinstance(exc, _StopWonRace) else exc
            )
            attempt.ready.set()
            self._calibration = None
            return
        attempt.accepted = True
        dispatcher = asyncio.ensure_future(self._dispatch_calibration(attempt))
        attempt.events.put_nowait(("on_started", started))
        terminal: p.CalibrationStopped | None = None
        diagnostic: BaseException | None = None
        try:
            if attempt.stop_requested.is_set():
                await self.stop_calibration()
            terminal = await self._read_calibration(attempt)
        except BaseException as exc:  # noqa: BLE001 -- becomes the unconfirmed ending's diagnostic
            diagnostic = exc
        finally:
            stopped = self._calibration_stopped(attempt, terminal, diagnostic)
            attempt.ended = True
            for future in attempt.pending.values():
                if not future.done():
                    future.set_exception(
                        CalibrationEndedError("the calibration ended before the step was accepted")
                    )
            attempt.pending.clear()
            # The connection is closed and the client idle before on_stopped runs, so a caller (including the
            # callback itself) can start the next attempt at once: the server frees its owner slot only when
            # this connection is gone.
            await self._close_calibration_ws(attempt)
            if self._calibration is attempt:
                self._calibration = None
            attempt.events.put_nowait(("on_stopped", stopped))
            attempt.events.put_nowait(None)
            await asyncio.gather(dispatcher, return_exceptions=True)

    async def _calibration_handshake(
        self, attempt: _CalibrationAttempt, request: p.StartCalibrationRequest
    ) -> p.CalibrationStarted:
        """Connect, send StartCalibration, and wait for acceptance or rejection."""
        attempt.ws = await self._race(
            websockets.connect(self._url, compression=None, max_size=_MAX_MESSAGE_BYTES),
            attempt.stop_requested,
            _ACCEPT_TIMEOUT_S,
        )
        await self._race(
            attempt.ws.send(_to_json_text(request.to_json())), attempt.stop_requested, _ACCEPT_TIMEOUT_S
        )
        raw = await self._race(attempt.ws.recv(), attempt.stop_requested, _ACCEPT_TIMEOUT_S)
        if isinstance(raw, bytes):
            raise p.ProtocolError("received a binary message before CalibrationStarted")
        event = p.parse_server_message(_from_json_text(raw))
        if isinstance(event, p.RequestRejected):
            raise RequestRejectedError(event.session_id, event.reason, event.error_message)
        if not isinstance(event, p.CalibrationStarted) or event.session_id != attempt.session_id:
            raise p.ProtocolError(f"expected CalibrationStarted first, got {type(event).__name__}")
        if event.sides != attempt.sides:
            raise p.ProtocolError("CalibrationStarted names different sides than requested")
        return event

    async def _read_calibration(self, attempt: _CalibrationAttempt) -> p.CalibrationStopped | None:
        """Read until the server's terminal message; None when the transport ended without one."""
        while True:
            try:
                raw = await attempt.ws.recv()
            except ConnectionClosed:
                return None
            if isinstance(raw, bytes):
                raise p.ProtocolError("received a binary message during a calibration")
            event = p.parse_server_message(_from_json_text(raw))
            if event.session_id != attempt.session_id:
                raise p.ProtocolError(f"received a message for session {event.session_id!r}, expected another")
            if isinstance(event, p.CalibrationStopped):
                return event
            if isinstance(event, p.CalibrationStatus):
                attempt.events.put_nowait(("on_status", event))
            elif isinstance(event, p.CalibrationStepAccepted):
                attempt.accepted_tokens.add(event.step_token)
                future = attempt.pending.pop(event.step_token, None)
                if future is not None and not future.done():
                    future.set_result(None)
            elif isinstance(event, p.RequestRejected):
                self._calibration_rejected(attempt, event)
            elif not isinstance(event, p.CalibrationStarted):
                raise p.ProtocolError(f"unexpected {type(event).__name__} during a calibration")

    @staticmethod
    def _calibration_rejected(attempt: _CalibrationAttempt, event: p.RequestRejected) -> None:
        """A refused confirmation fails its caller; the attempt itself continues unchanged."""
        if event.request_type != p.MessageType.CONFIRM_CALIBRATION_STEP:
            return
        error = RequestRejectedError(event.session_id, event.reason, event.error_message)
        tokens = [event.step_token] if event.step_token is not None else list(attempt.pending)
        for token in tokens:
            future = attempt.pending.pop(token, None)
            if future is not None and not future.done():
                future.set_exception(error)

    async def _dispatch_calibration(self, attempt: _CalibrationAttempt) -> None:
        """Deliver callbacks one at a time, in order, apart from the reader so a callback can await a confirm."""
        while True:
            item = await attempt.events.get()
            if item is None:
                return
            name, value = item
            try:
                await _invoke(attempt.callbacks[name], value)
            except (Exception, asyncio.CancelledError) as exc:
                if name == "on_stopped":
                    # Reported to the loop, never repeated: there is no later callback to carry it.
                    asyncio.get_running_loop().call_exception_handler(
                        {"message": "on_stopped callback raised", "exception": exc}
                    )
                elif attempt.callback_error is None:
                    # A failing callback cancels the calibration; the terminal value carries the diagnostic.
                    attempt.callback_error = exc
                    attempt.stop_requested.set()
                    await self.stop_calibration()
            finally:
                if name == "on_started":
                    attempt.ready.set()

    @staticmethod
    def _calibration_stopped(
        attempt: _CalibrationAttempt, terminal: p.CalibrationStopped | None, diagnostic: BaseException | None
    ) -> CalibrationStopped:
        """The server's terminal message, or an honest unconfirmed value; files only from a confirmed COMPLETED."""
        notes = [f"a callback raised: {attempt.callback_error!r}"] if attempt.callback_error is not None else []
        if terminal is not None:
            message = "; ".join(([terminal.error_message] if terminal.error_message else []) + notes) or None
            return CalibrationStopped(attempt.session_id, terminal.reason, terminal.result, message)
        why = f"protocol error: {diagnostic}" if diagnostic is not None else "the connection ended"
        message = "; ".join([f"{why}; the server's outcome is unknown", *notes])
        return CalibrationStopped(attempt.session_id, None, None, message)

    async def _close_calibration_ws(self, attempt: _CalibrationAttempt) -> None:
        ws, attempt.ws = attempt.ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT_S)
            except BaseException:  # noqa: BLE001 -- best-effort cleanup, must never raise or hang
                pass

    # -- the supervisor ----------------------------------------------------

    async def _run_attempt(self, attempt: _Attempt) -> None:
        """The sole owner of this attempt from connect through
        finalization. Never externally cancelled; reacts to
        ``attempt.stop_requested`` cooperatively via ``_race``."""
        try:
            init_event = await self._handshake(attempt)
        except BaseException as exc:  # noqa: BLE001 -- reported via start_exception, not raised here
            attempt.start_exception = exc
            attempt.ready.set()
            self._attempt = None
            return

        attempt.accepted = True
        attempt.streams = _StreamRegistry.from_session_init(init_event)
        attempt.init_json = init_event.to_json()
        diagnostic: BaseException | None = None
        try:
            await _invoke(attempt.callbacks["on_started"], _to_client_started(init_event))
        except (Exception, asyncio.CancelledError) as exc:
            # A callback's own failure -- including CancelledError raised by
            # application work the callback itself awaited (e.g. a Future it
            # created that got cancelled). This is not an external
            # task.cancel() on the supervisor; the supervisor was never
            # touched. Treat it like any other callback exception rather
            # than letting it escape before readiness/cleanup ever happen.
            diagnostic = exc
        attempt.ready.set()  # on_started attempted (whether it raised or not); start() may return now

        # Finalization (transport close + on_stopped + clearing the active
        # attempt) is a hard guarantee from this point on: whatever happens
        # in _run_until_stopped, the finally below always reaches it.
        stopped: CaptureStopped | None = None
        try:
            stopped = await self._run_until_stopped(attempt, diagnostic)
        finally:
            if stopped is None:
                stopped = self._unconfirmed(attempt, diagnostic)
            await self._finalize(attempt, stopped)

    async def _handshake(self, attempt: _Attempt) -> p.SessionInit:
        """Connect, send Begin, and wait for a validated SessionInit -- the
        whole pre-acceptance sequence. Every phase is interruptible by
        ``attempt.stop_requested`` (not just a single checkpoint after
        connect), via ``_race``. Raises for any failure in this phase,
        including a stop() request; the caller treats every exception here
        uniformly as "never accepted"."""
        try:
            attempt.ws = await self._race(
                websockets.connect(self._url, compression=None, max_size=_MAX_MESSAGE_BYTES),
                attempt.stop_requested,
                _ACCEPT_TIMEOUT_S,
            )
            await self._race(
                attempt.ws.send(_to_json_text(self._begin_json(attempt.begin))),
                attempt.stop_requested,
                _ACCEPT_TIMEOUT_S,
            )
            init_event = await self._race(
                self._receive_until_init(attempt), attempt.stop_requested, _ACCEPT_TIMEOUT_S
            )
            self._validate_layout(attempt.begin, init_event)
        except BaseException:
            await self._close_ws(attempt)
            raise
        return init_event

    async def _receive_until_init(self, attempt: _Attempt) -> p.SessionInit:
        while True:
            raw = await attempt.ws.recv()
            if isinstance(raw, bytes):
                raise p.ProtocolError("received a binary message before SessionInit")
            event = p.parse_server_message(_from_json_text(raw))
            if isinstance(event, p.RequestRejected):
                raise RequestRejectedError(event.session_id, event.reason, event.error_message)
            if event.session_id != attempt.session_id:
                raise p.ProtocolError(
                    f"received {type(event).__name__} for session {event.session_id!r}, "
                    f"expected {attempt.session_id!r}"
                )
            if isinstance(event, p.SessionInit):
                return event
            raise p.ProtocolError(f"expected SessionInit first, got {type(event).__name__}")

    @staticmethod
    async def _race(coro: Awaitable[Any], stop_event: asyncio.Event, timeout: float | None) -> Any:
        """Await ``coro``, interruptible by ``stop_event`` being set or
        ``timeout`` elapsing (``None`` means no deadline, only
        interruptible by ``stop_event``) -- the one network-wait mechanism
        every handshake phase and the post-acceptance receive loop (before
        it starts draining for a terminal reply) route through, instead of
        each place rolling its own timeout/cancellation handling.
        Whichever of ``coro``/the stop-wait loses the race is properly
        cancelled and awaited (joined) before returning, so nothing is
        left running in the background.

        Returns ``coro``'s result if it wins. Raises ``_StopWonRace`` if
        ``stop_event`` wins (deliberately NOT ``asyncio.CancelledError`` --
        see ``StoppedBeforeAcceptedError``), or ``TimeoutError`` if the
        deadline elapses first."""
        work: asyncio.Future = asyncio.ensure_future(coro)
        waiter: asyncio.Future = asyncio.ensure_future(stop_event.wait())
        try:
            done, pending = await asyncio.wait(
                {work, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            work.cancel()
            waiter.cancel()
            await asyncio.gather(work, waiter, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if work in done:
            return work.result()
        if waiter in done:
            waiter.result()  # never raises: Event.wait() only returns True
            raise _StopWonRace("CaptureClient.stop() was called")
        raise TimeoutError(f"timed out after {timeout}s")

    async def _run_until_stopped(self, attempt: _Attempt, diagnostic: BaseException | None) -> CaptureStopped:
        """The post-acceptance phase: dispatch data/status callbacks until
        a terminal SessionStopped, a transport failure, or Stop (requested
        externally, or because a callback above just raised into
        ``diagnostic``) ends the attempt. Once stopping, Stop is sent
        exactly once and everything else is drained/discarded under ONE
        absolute deadline covering both that send and the wait for the
        matching terminal reply -- not two deadlines racing each other."""
        if diagnostic is not None:
            attempt.stop_requested.set()
        loop = asyncio.get_running_loop()
        deadline: float | None = None

        while True:
            stopping = attempt.stop_requested.is_set()
            if stopping and deadline is None:
                deadline = loop.time() + _STOP_TIMEOUT_S
                try:
                    await asyncio.wait_for(
                        attempt.ws.send(_to_json_text(self._stop_json(attempt))),
                        timeout=max(0.0, deadline - loop.time()),
                    )
                except (ConnectionClosed, OSError, TimeoutError):
                    # Best-effort: a blocked or failed send must not abandon
                    # the attempt -- stop() returns None after sending or
                    # scheduling the request. Still drain for a possible
                    # terminal reply under the same deadline.
                    pass

            if stopping:
                timeout = max(0.0, deadline - loop.time())
                if timeout == 0.0:
                    return self._unconfirmed(attempt, diagnostic, timed_out=True)
                try:
                    raw = await asyncio.wait_for(attempt.ws.recv(), timeout=timeout)
                except TimeoutError:
                    return self._unconfirmed(attempt, diagnostic, timed_out=True)
                except (ConnectionClosed, OSError) as exc:
                    return self._unconfirmed(attempt, diagnostic or exc)
            else:
                # Not yet stopping: no deadline, but still interruptible by
                # stop_requested the moment it's set, rather than only
                # being noticed once some other message happens to arrive.
                try:
                    raw = await self._race(attempt.ws.recv(), attempt.stop_requested, None)
                except _StopWonRace:
                    continue  # stop_requested just got set; re-enter the loop as stopping
                except (ConnectionClosed, OSError) as exc:
                    return self._unconfirmed(attempt, diagnostic or exc)

            try:
                result = await self._process_message(attempt, raw, stopping)
            except p.ProtocolError as exc:
                if stopping:
                    # A malformed/unexpected message while draining for the
                    # terminal reply doesn't abort the search for it or
                    # override a real diagnostic -- same as the old design's
                    # dedicated drain loop skipped these.
                    continue
                return self._unconfirmed(attempt, diagnostic or exc)
            except (Exception, asyncio.CancelledError) as exc:
                # A data/status callback's own failure -- including a
                # CancelledError raised by application work the callback
                # itself awaited, same reasoning as the on_started handling
                # above: this is not an external cancellation of the
                # supervisor.
                if diagnostic is None:
                    diagnostic = exc
                attempt.stop_requested.set()
                continue  # next iteration enters the stopping branch above

            if result is not None:  # a confirmed SessionStopped
                return self._merge_diagnostic(result, diagnostic) if diagnostic else result

    async def _process_message(self, attempt: _Attempt, raw: bytes | str, stopping: bool) -> CaptureStopped | None:
        """Handle one received message. Returns a ``CaptureStopped`` only
        for a confirmed terminal ``SessionStopped``; otherwise None. Binary
        data (tracking frames/camera images) is suppressed once Stop is
        requested, but JSON lifecycle messages (SessionStarting/SessionBegun/SessionStatus)
        are still dispatched and validated normally; a legitimate
        SessionBegun that happens to arrive after Stop was requested still
        fires on_begun once. Any resulting ProtocolError while ``stopping``
        (the caller's loop-iteration snapshot, used only for the checks
        below that don't need to react within the same await) is tolerated
        by the caller rather than aborting the drain."""
        if isinstance(raw, bytes):
            # Checked live here, not via the ``stopping`` parameter: Stop
            # can be requested while the recv()/_race that produced this
            # message was already in flight, so a sampled-before-the-await
            # boolean could be stale by the time a binary frame is actually
            # being dispatched.
            if attempt.stop_requested.is_set():
                return None
            if not attempt.begun:
                raise p.ProtocolError("received binary data before SessionBegun")
            header, payload = p.decode_header(raw)
            self._check_sequence(attempt, header)
            await self._dispatch_binary(attempt, header, payload)
            return None

        event = p.parse_server_message(_from_json_text(raw))
        if isinstance(event, p.RequestRejected):
            # A rejection names the *rejected* request's session_id, which
            # can legitimately differ from the active one (a retried Stop
            # for a stale id, a BUSY reply to another Begin attempt). A
            # reply for another ID must not reset the active session, and
            # a rejection never ends the active session.
            return None
        if event.session_id != attempt.session_id:
            if stopping:
                return None  # ignore stray messages for another session while draining
            raise p.ProtocolError(
                f"received {type(event).__name__} for session {event.session_id!r}, "
                f"expected {attempt.session_id!r}"
            )

        if isinstance(event, p.SessionStopped):
            return CaptureStopped(
                session_id=event.session_id,
                running=False,
                reason=event.reason,
                error_message=event.error_message,
                status=event.status,
                streams=event.streams,
            )

        if isinstance(event, p.SessionInit):
            self._check_duplicate_init(attempt, event)
            return None
        if isinstance(event, p.SessionStarting):
            if attempt.begun:
                raise p.ProtocolError("received SessionStarting after SessionBegun")
            if event.status is not None:
                attempt.last_status = event.status
            await _invoke(attempt.callbacks["on_status"], _to_client_status(event))
            return None
        if isinstance(event, p.SessionBegun):
            if not attempt.begun:
                attempt.begun = True
                attempt.begun_event = event  # the raw parsed event, for to_json() replay checks
                attempt.last_status = event.status
                await _invoke(attempt.callbacks["on_status"], _to_client_status(event))
            else:
                self._check_duplicate_begun(attempt, event)
            return None
        if isinstance(event, p.SessionStatus):
            if not attempt.begun:
                raise p.ProtocolError("received SessionStatus before SessionBegun")
            attempt.last_status = event.status
            await _invoke(attempt.callbacks["on_status"], _to_client_status(event))
            return None
        # parse_server_message's _JSON_PARSERS_BY_TYPE only ever returns one
        # of the types already handled above -- this branch guards that
        # invariant rather than silently ignoring an unknown message type.
        raise AssertionError(f"unhandled server message type: {type(event).__name__}")  # pragma: no cover

    async def _finalize(self, attempt: _Attempt, stopped: CaptureStopped) -> None:
        """The one path into transport close, on_stopped invocation and
        clearing the active attempt -- reached from every ending, whether
        a confirmed SessionStopped, a transport failure, a lost Stop reply,
        or a callback's own failure. ``self._attempt`` is cleared only
        after on_stopped has actually returned, so a concurrent start()
        correctly sees the attempt as still active until this finishes."""
        await self._close_ws(attempt)
        try:
            await _invoke(attempt.callbacks["on_stopped"], stopped)
        except Exception as exc:  # noqa: BLE001 -- on_stopped itself must not be re-invoked
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "CaptureClient on_stopped callback raised",
                    "exception": exc,
                }
            )
        finally:
            self._attempt = None

    async def query_devices(self) -> tuple[p.DeviceEntry, ...]:
        """List the devices the server offers, without taking the control connection.

        Uses its own short-lived connection to the server's read-only ``/query`` route, so it works before
        capture and while capturing (including from a callback), and needs no glove or session. The Manus
        entry lists the MCAL files a :class:`ManusPluginConfig` can select. Concurrent calls are independent.
        Raises :class:`DeviceQueryError` when no answer is obtained, including when the server is too old to
        know the query.
        """
        # The query route is at the root path of the same listener; a station URL with a path prefix is not
        # supported.
        url = urlsplit(self._url)._replace(path="/query", query="", fragment="").geturl()
        request = p.QueryDevicesRequest(request_id=f"devices-{uuid.uuid4()}")
        try:
            async with asyncio.timeout(_QUERY_TIMEOUT_S):
                async with websockets.connect(url, compression=None, max_size=_MAX_MESSAGE_BYTES) as ws:
                    await ws.send(_to_json_text(request.to_json()))
                    raw = await ws.recv()
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd is not None else None
            raise DeviceQueryError(
                f"the server closed the device query (close code {code}); it may not support device queries"
            ) from None
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
            raise DeviceQueryError(f"the device query failed: {str(exc) or type(exc).__name__}") from None
        if isinstance(raw, bytes):
            raise DeviceQueryError("the server answered the device query with a binary message")
        try:
            reply = p.DevicesReply.from_json(_from_json_text(raw))
        except p.ProtocolError as exc:
            raise DeviceQueryError(f"the server's device reply was invalid: {exc}") from None
        # A reply for another request is never accepted as this query's answer.
        if reply.request_id != request.request_id:
            raise DeviceQueryError("the server's reply does not answer this device query")
        return reply.devices

    # -- message/result construction ---------------------------------------

    def _build_begin_request(
        self,
        *,
        sources: Sequence[p.SourceType],
        cameras: Sequence[p.CameraSelection],
        plugins: Sequence[p.PluginType | ManusPluginConfig],
    ) -> p.BeginRequest:
        try:
            tracking = p.TrackingConfig(sources=tuple(sources), rate_hz=_DEFAULT_RATE_HZ)
            plugin_requests = tuple(
                p.PluginRequest(
                    plugin=plugin.plugin,
                    left_calibration=plugin.left_calibration,
                    right_calibration=plugin.right_calibration,
                )
                if isinstance(plugin, ManusPluginConfig)
                else p.PluginRequest(plugin=plugin)
                for plugin in plugins
            )
            camera_requests = tuple(p.CameraRequest(name=c["name"], type=c["type"]) for c in cameras)
            return p.BeginRequest(
                tracking=tracking,
                plugins=plugin_requests,
                cameras=camera_requests,
                session_id=str(uuid.uuid4()),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid CaptureClient.start arguments: {exc}") from exc

    def _begin_json(self, begin: p.BeginRequest) -> dict[str, Any]:
        config = p.effective_config(begin)
        return {
            "type": int(p.MessageType.BEGIN),
            "version": p.PROTOCOL_VERSION,
            "session_id": begin.session_id,
            **config.to_json(),
        }

    def _stop_json(self, attempt: _Attempt) -> dict[str, Any]:
        return {
            "type": int(p.MessageType.STOP),
            "version": p.PROTOCOL_VERSION,
            "session_id": attempt.session_id,
        }

    def _validate_layout(self, begin: p.BeginRequest, init_event: p.SessionInit) -> None:
        """Reject a SessionInit whose stream registry does not match what
        Begin actually requested, or whose tracking layout does not match
        this client's own known-correct ``ic_tracking_v1`` field table --
        the client decodes exactly one fixed schema and trusts no other
        shape or an unrequested/missing stream."""
        expected_config = p.effective_config(begin)
        if init_event.effective_config.to_json() != expected_config.to_json():
            raise p.ProtocolError("SessionInit's effective_config does not match what Begin requested")

        tracking_streams = [s for s in init_event.streams if isinstance(s, p.TrackingStream)]
        camera_streams = [s for s in init_event.streams if isinstance(s, p.CameraStream)]

        tracking_requested = len(begin.tracking.sources) > 0
        if tracking_requested and not tracking_streams:
            raise p.ProtocolError("SessionInit omits the tracking stream despite tracking.sources being requested")
        if not tracking_requested and tracking_streams:
            raise p.ProtocolError("SessionInit advertises a tracking stream that was not requested")
        if len(tracking_streams) > 1:
            raise p.ProtocolError("SessionInit advertised more than one tracking stream")
        _known_layout = p.ic_tracking_v1_layout()
        if tracking_streams and (
            tracking_streams[0].payload_size != _known_layout.payload_size
            or tracking_streams[0].fields != _known_layout.fields
        ):
            raise p.ProtocolError(
                "SessionInit's tracking layout does not match the client's known ic_tracking_v1 field table"
            )
        if tracking_streams:
            # The tracking stream descriptor carries its own copy of
            # sources/rate_hz, separate from effective_config.tracking --
            # the earlier effective_config equality check above does not by
            # itself guarantee these agree with each other.
            stream = tracking_streams[0]
            accepted_tracking = init_event.effective_config.tracking
            if set(stream.sources) != set(accepted_tracking.sources):
                raise p.ProtocolError(
                    "SessionInit's tracking stream sources do not match effective_config.tracking"
                )
            if stream.rate_hz != accepted_tracking.rate_hz:
                raise p.ProtocolError(
                    "SessionInit's tracking stream rate_hz does not match effective_config.tracking"
                )

        advertised_names = [s.name for s in camera_streams]
        if len(advertised_names) != len(set(advertised_names)):
            # Checked before building a name-keyed dict below: two
            # descriptors sharing a name but different stream_ids would
            # otherwise silently collapse into one entry and the
            # duplicate would escape every check that follows.
            raise p.ProtocolError("SessionInit has duplicate camera stream names")

        requested_cameras = {c.name: c.type for c in begin.cameras}
        advertised_cameras = {s.name: s.type for s in camera_streams}
        if advertised_cameras.keys() != requested_cameras.keys():
            raise p.ProtocolError("SessionInit's camera streams do not match the requested camera selections")
        for name, requested_type in requested_cameras.items():
            if advertised_cameras[name] != requested_type:
                raise p.ProtocolError(f"SessionInit's camera {name!r} type does not match the requested type")
        stream_ids = [s.stream_id for s in camera_streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise p.ProtocolError("SessionInit has duplicate camera stream_id values")

    def _check_duplicate_init(self, attempt: _Attempt, event: p.SessionInit) -> None:
        """A replayed Begin on the same session re-sends SessionInit; the
        client re-validates it against the cached descriptor (cached as
        JSON on the attempt at acceptance) rather than re-running on_started
        or resetting sequences: identical replayed metadata is validated
        without repeating on_started or on_begun or resetting sequences.
        Compared via to_json() rather
        than dataclass equality: CameraStream.intrinsics can hold a NumPy
        array, whose ``==`` is elementwise and not directly usable as a
        bool."""
        if event.to_json() != attempt.init_json:
            raise p.ProtocolError("duplicate SessionInit does not match the active session's configuration")

    def _check_duplicate_begun(self, attempt: _Attempt, event: p.SessionBegun) -> None:
        """A replayed SessionBegun is validated against the establishment
        snapshot cached when SessionBegun first arrived -- not against
        ``attempt.last_status``, which later SessionStatus messages keep
        overwriting -- so "the same begun event replayed" means matching
        the original establishment, the same replay contract
        ``_check_duplicate_init`` already applies to SessionInit."""
        if event.to_json() != attempt.begun_event.to_json():
            raise p.ProtocolError("duplicate SessionBegun does not match the session's original establishment")

    def _unconfirmed(
        self, attempt: _Attempt, diagnostic: BaseException | None, *, timed_out: bool = False
    ) -> CaptureStopped:
        if diagnostic is None and timed_out:
            diagnostic = TimeoutError("Timed out waiting for SessionStopped")
        return CaptureStopped(
            session_id=attempt.session_id,
            running=False,
            reason=None,
            error_message=str(diagnostic) if diagnostic is not None else None,
            status=attempt.last_status,
            streams=None,
        )

    def _merge_diagnostic(self, confirmed: CaptureStopped, diagnostic: BaseException) -> CaptureStopped:
        message = confirmed.error_message
        message = f"{message}; also: {diagnostic}" if message else str(diagnostic)
        return CaptureStopped(
            session_id=confirmed.session_id,
            running=False,
            reason=confirmed.reason,
            error_message=message,
            status=confirmed.status,
            streams=confirmed.streams,
        )

    def _check_sequence(self, attempt: _Attempt, header: p.FrameMetadata) -> None:
        expected_min = attempt.sequences.get(header.stream_id, 0)
        if header.sequence < expected_min:
            raise p.ProtocolError(
                f"stream {header.stream_id} sequence went backward: {header.sequence} < {expected_min}"
            )
        attempt.sequences[header.stream_id] = header.sequence + 1

    async def _dispatch_binary(self, attempt: _Attempt, header: p.FrameMetadata, payload: bytes) -> None:
        if header.kind == p.BinaryKind.CAPTURE_FRAME:
            if header.stream_id != 0:
                raise p.ProtocolError(f"CaptureFrame stream_id must be 0, got {header.stream_id}")
            description = attempt.streams.tracking
            if description is None:
                raise p.ProtocolError("received CaptureFrame but no tracking stream was registered")
            if len(payload) != description.payload_size:
                raise p.ProtocolError(
                    f"CaptureFrame payload is {len(payload)} bytes, expected {description.payload_size}"
                )
            frame = CaptureFrame(
                session_id=attempt.session_id,
                description=description,  # the exact object from CaptureStarted.streams
                stream_id=header.stream_id,
                sequence=header.sequence,
                capture_time_ns=header.capture_time_ns,
                receive_time_ns=header.receive_time_ns,
                payload=payload,
            )
            await _invoke(attempt.callbacks["on_capture_frame"], frame)
        else:
            stream = attempt.streams.cameras.get(header.stream_id)
            if stream is None:
                raise p.ProtocolError(f"CaptureImage for unregistered camera stream_id {header.stream_id}")
            image = CameraImage(
                session_id=attempt.session_id,
                stream_id=header.stream_id,
                sequence=header.sequence,
                capture_time_ns=header.capture_time_ns,
                receive_time_ns=header.receive_time_ns,
                name=stream.name,
                jpeg=payload,
                expected_size=(stream.width, stream.height),
            )
            await _invoke(attempt.callbacks["on_capture_image"], image)

    async def _close_ws(self, attempt: _Attempt) -> None:
        ws, attempt.ws = attempt.ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=_CLOSE_TIMEOUT_S)
            except BaseException:  # noqa: BLE001 -- best-effort cleanup, must never raise or hang
                pass


def _to_json_text(data: dict[str, Any]) -> str:
    return json.dumps(data)


def _from_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise p.ProtocolError(f"received text message is not valid JSON: {exc}") from exc
