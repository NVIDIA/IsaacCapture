# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The production CaptureServer: a real WebSocket listener implementing
the wire protocol end to end -- SessionInit through SessionStopped
sequencing, bounded queues, fair sending, cancellation and counter
accounting.

This is the ONE server implementation, not a separate "mock server". Every
piece of session/transport logic here is data-source-agnostic: ``_Session``
is constructed with an :class:`~capture_server.acquisition.Acquisition` --
a fixed composition of one optional IsaacCapture adapter and zero or more
DepthAI adapters (:mod:`capture_server.acquisition`), each owning its own
acquisition cadence and native-to-protocol mapping. ``CaptureServer``'s
``source_factory`` constructor argument is this server's only wiring
point for WHICH concrete adapters (and, for the mock ones, which
scenario/seed/fault options) it is constructed with --
:mod:`capture_server.launch` builds the mock or the real (IsaacCapture and
OAK-D) one from ``ServerConfig.mock``, and either plugs into this same
unchanged session/listener code -- only the acquisition source differs,
not the serving machinery. This module itself never imports
:mod:`capture_server.mock`.

Lifecycle convention: explicit ``request_stop()`` / ``run()`` / ``close()``
methods, driven by a CLI that installs SIGINT/SIGTERM handlers calling
``request_stop()`` rather than a bare ``asyncio.run()``.

Architecture: one owner WebSocket connection at a time (a second connection
is rejected with close code 1013). One receiver reads client JSON; one
sender drains a small control queue before round-robining the tracking
queue and each camera queue (each bounded, drop-oldest). One ``_Session``
owns everything for one accepted Begin through its terminal SessionStopped,
mirroring ``capture_server.client``'s single-supervisor-per-attempt design.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import ServerConnection

from . import protocol as p
from .acquisition import Acquisition, CameraSourceError, SourceExited

__all__ = ["CaptureServer", "ServerConfig", "main"]

# Inbound (client -> server) messages are only ever small Begin/Stop JSON,
# bounded by a 64 KiB control ceiling. The server is the one SENDING
# the much larger camera-image frames, so there is no corresponding
# outbound ceiling to declare here -- websockets.serve's max_size governs
# what this server will ACCEPT from the client, not what it sends.
_MAX_INBOUND_BYTES = 64 * 1024
_QUEUE_CAPACITY = 3
# Control messages are required lifecycle transitions, never silently
# droppable like a replaceable data sample -- overflowing this bound means
# something is structurally wrong (a stuck writer, a runaway producer of
# control traffic), so it terminates the connection rather than quietly
# discarding a control message.
_CONTROL_QUEUE_CAPACITY = 64
_SEND_DEADLINE_S = 2.0
_CLOSE_TIMEOUT_S = 2.0
_SOURCE_RELEASE_TIMEOUT_S = 5.0
_COMPLETED_TOMBSTONE_CAPACITY = 16


class _StopDuringStartup(Exception):
    """Internal control-flow signal only, never a recorded diagnostic:
    Stop was requested while a startup call (e.g. source entry) was still
    in flight. Caught by ``_Session._run_unprotected`` and treated as a
    clean requested stop (``_fatal_reason`` stays None), not a failure."""


_T = TypeVar("_T")


def _diagnostic_text(exc: BaseException) -> str:
    """A guaranteed-nonempty diagnostic string for ``exc``. ``str(exc)``
    can legitimately be empty (``RuntimeError()`` with no args, some
    ``SourceError`` subclasses) -- but ``protocol.SessionStopped``
    requires ``error_message`` to be nonempty or None, so an empty
    ``str(exc)`` used as-is would make ``_finalize``'s one terminal
    constructor raise while building the terminal reply, which would
    silently lose it entirely (``done`` still gets set, but the client
    never learns this attempt actually failed). Falls back to the
    exception's type name so every recorded diagnostic is always usable."""
    return str(exc) or type(exc).__name__


@dataclass(frozen=True)
class StationCamera:
    """Server-owned settings of one physical camera, keyed by the logical name clients request.

    ``device_id`` is the camera's stable physical ID (an OAK-D's MXID), never an enumeration index.
    """

    device_id: str
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    jpeg_quality: int = 80
    required: bool = True
    # Cap on the USB link speed the device is booted at (DepthAI UsbSpeed name, e.g. "HIGH" = USB2);
    # None leaves the SDK default. Some host ports/cables cannot boot an OAK-D at SuperSpeed.
    max_usb_speed: str | None = None

    def __post_init__(self) -> None:
        """Reject an empty device ID, non-positive/non-finite size or fps, or a JPEG quality outside 1..100."""
        if not self.device_id:
            raise ValueError("StationCamera.device_id must be a nonempty physical device ID")
        if self.width <= 0 or self.height <= 0 or not 0 < self.fps < float("inf"):
            raise ValueError(f"StationCamera size/fps must be positive, got {self.width}x{self.height}@{self.fps}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"StationCamera.jpeg_quality must be in 1..100, got {self.jpeg_quality}")


@dataclass
class ServerConfig:
    """CLI-configured, process-wide server-local settings, independent of
    the client protocol. The one mode decision lives here: ``mock`` selects
    the acquisition implementation once at setup, and the capture mode the
    protocol reports is derived from the same flag. Nothing else is
    mock-specific (seed/scenario/fault selection live entirely in
    :mod:`capture_server.launch`'s mock ``source_factory`` composition --
    mock-only seed/scenarios/faults never live in generic ServerConfig or
    session logic)."""

    # True selects the existing mock acquisition, False the real one. Chosen
    # at server setup, never by a client's Begin. The default keeps library
    # callers and tests on the mock; the CLI always passes it explicitly.
    mock: bool = True
    host: str = "127.0.0.1"
    port: int = 8082
    connect_delay: float = 0.0
    # Station acquisition policy -- whether a configured camera's terminal
    # failure ends the whole session (the `required` camera policy).
    # Independent of fault injection: a fault decides whether a camera
    # actually fails this session; this decides the consequence if it does.
    camera_required: bool = True
    # Station's known logical camera names -- None accepts any name (the
    # current permissive default, since this package has no real station
    # registry yet); set to reject an unknown camera name as INVALID_CONFIG
    # instead of inventing settings for it.
    known_cameras: tuple[str, ...] | None = None
    # Real acquisition: logical camera name -> its station settings. When set, only these names are
    # accepted and each stream uses its entry; None keeps the mock's fixed 1280x720@30 defaults.
    cameras: dict[str, StationCamera] | None = None
    # Operator-facing note appended to the startup banner, saying where the served data comes from. None
    # prints the mode's default (the mock's SYNTHETIC-data warning, nothing for live); replay sets it so
    # recorded data is never announced as synthetic.
    data_note: str | None = None

    def __post_init__(self) -> None:
        """Reject one physical camera assigned to more than one logical name."""
        if self.cameras is not None:
            ids = [c.device_id for c in self.cameras.values()]
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            if duplicates:
                raise ValueError(f"physical camera(s) assigned to more than one logical name: {duplicates}")


@dataclass
class _StreamCounters:
    produced: int = 0
    sent: int = 0
    dropped: int = 0

    def to_protocol(self, stream_id: int) -> p.StreamCounters:
        return p.StreamCounters(stream_id=stream_id, produced=self.produced, sent=self.sent, dropped=self.dropped)


class _BoundedDropOldestQueue:
    """A bounded byte-message queue that drops its oldest entry (not the
    newest) when full. Callers only ever call ``get()`` after confirming (via
    ``bool(queue)``) it's non-empty, so this does no waiting of its own --
    there is no suspension logic here at all, hence a plain sync method,
    not a coroutine. The Outbox's single shared wake Event is what a
    caller actually blocks on between checks. Per-item drop accounting
    lives entirely in ``_Session``'s derived ``dropped = produced - sent``,
    not here -- this class only implements the eviction behavior itself."""

    def __init__(self, capacity: int) -> None:
        self._items: list[bytes] = []
        self._capacity = capacity

    def put(self, item: bytes) -> None:
        if len(self._items) >= self._capacity:
            self._items.pop(0)
        self._items.append(item)

    def get(self) -> bytes:
        return self._items.pop(0)

    def __bool__(self) -> bool:
        return bool(self._items)

    def drain(self) -> int:
        """Discard every remaining item (production has stopped; these
        will never be sent) and return how many there were."""
        count = len(self._items)
        self._items.clear()
        return count


def _now_ns() -> int:
    return time.monotonic_ns()


def _status_excluding_timestamp(status: p.StatusSnapshot) -> dict[str, Any]:
    """``StatusSnapshot.to_json()`` without ``updated_at_ns`` -- a changed
    timestamp alone must not trigger a SessionStatus."""
    data = status.to_json()
    data.pop("updated_at_ns")
    return data


class _Outbox:
    """The ONE outbound structure and its single writer task for one owner
    CONNECTION's whole lifetime -- constructed once in
    ``CaptureServer._handle_connection``, reused across every ``_Session``
    that runs on that connection, not reconstructed per session. No other
    code path anywhere ever calls ``ws.send()`` on the owner connection:
    control messages (SessionInit/Starting/Begun/Stopped/RequestRejected,
    including replays) are a bounded ordered FIFO always drained ahead of
    data; SessionStatus observations coalesce to the latest value only (an
    unsent one is superseded by a newer one, never queued as a backlog);
    tracking/camera frames use the existing bounded drop-oldest per-stream
    queues, round-robined one item per outer-loop pass. One deadline and
    one failure-close policy covers EVERY send without exception: a timed
    out or failed send closes the transport immediately and the writer
    stops -- the connection's state is no longer trusted, so nothing,
    not even a terminal reply, is attempted on it again. Control-queue
    overflow closes the connection outright, the same as any other send
    failure, rather than only ending whichever session happens to be
    active."""

    def __init__(
        self,
        send_json: Callable[[dict[str, Any]], Any],
        send_binary: Callable[[bytes], Any],
        close_transport: Callable[[], Any],
    ) -> None:
        self._send_json = send_json
        self._send_binary = send_binary
        self._close_transport = close_transport
        self._control: list[dict[str, Any]] = []
        self._latest_status: dict[str, Any] | None = None
        self._frame_queues: dict[int, _BoundedDropOldestQueue] = {}
        self._on_sent: Callable[[int], None] | None = None
        self._on_fatal: Callable[[str], None] | None = None
        self._wake = asyncio.Event()
        self.closed = False
        self.fatal_message: str | None = None
        # Guards the actual close_transport()/on_fatal notification so it
        # fires exactly once no matter which path first called
        # _mark_fatal -- see _run's finally, the one place that performs
        # it.
        self._transport_closed = False
        self._writer_task: asyncio.Task | None = None
        # An ending session's explicit quiesce handshake -- see quiesce()
        # and _run()'s own handling of _quiesce_requested. NOT a lock: a
        # lock's release timing cannot express "I promise not to look at
        # the queue again", only mutual exclusion during whatever window
        # each side happens to hold it, which is why this is a flag the
        # writer itself checks before selecting, plus an ack it sets only
        # once it has actually stopped selecting and discarded whatever
        # was still pending.
        self._quiesce_requested = False
        self._quiesce_ack = asyncio.Event()
        self._quiesce_ack.set()  # idle: no session is currently quiescing

    def bind_session(
        self,
        stream_ids: list[int],
        *,
        on_sent: Callable[[int], None],
        on_fatal: Callable[[str], None],
    ) -> None:
        """Reset per-session frame-queue registration and callbacks for a
        newly-starting session, called once per accepted Begin before that
        session publishes anything. Queued control/status from a REPLAY of
        the PREVIOUS session (there is at most one active session at a
        time, so this only ever matters across a Begin/Stop/Begin sequence
        on one connection) is intentionally left alone -- only frame
        routing and callbacks are session-specific."""
        self._frame_queues = {sid: _BoundedDropOldestQueue(_QUEUE_CAPACITY) for sid in stream_ids}
        self._on_sent = on_sent
        self._on_fatal = on_fatal
        # A fresh session must never inherit a previous session's pending
        # quiesce state or stale latest-status observation. This attempt
        # gets its OWN acknowledgment, created PENDING (not set) -- the
        # writer resolves it exactly once, either when this attempt's own
        # disposal genuinely completes or when the writer itself exits
        # (see _run's quiesce branch and its finally). quiesce() itself
        # only ever waits on it, never clears/recreates it -- a resolved
        # acknowledgment must stay resolved for the rest of this attempt's
        # lifetime, so repeated quiesce() calls (idempotent by design)
        # return the same completion instead of waiting again on an ack a
        # since-exited writer will never set a second time.
        self._quiesce_requested = False
        self._quiesce_ack = asyncio.Event()
        self._latest_status = None

    def start(self) -> None:
        self._writer_task = asyncio.ensure_future(self._run())

    async def stop_and_join(self) -> None:
        """Connection teardown: the connection itself is going away (the
        owner loop ended, cleanly or not) -- no more sends are attempted
        after this, full stop, same as any other closed condition. This is
        NOT the same concept as one session's data being quiesced at its
        own ordinary end (see ``_Session._blocked``): that still lets
        required control messages for the still-good connection flow
        normally through this same Outbox; only an actually-dead
        connection stops selection entirely."""
        self.closed = True
        self._wake.set()
        if self._writer_task is not None:
            await asyncio.gather(self._writer_task, return_exceptions=True)

    def _mark_fatal(self, message: str) -> None:
        """The ONE place any failure -- a send failure, control-queue
        overflow, or an unexpected exception anywhere in the writer -- is
        RECORDED. Does not itself close the transport or notify
        ``on_fatal``; that happens exactly once, from inside the writer's
        own task (``_run``'s ``finally``), regardless of which caller/task
        first detected the failure. That keeps the actual close an
        explicitly observed obligation of the one writer task (which
        ``stop_and_join`` already awaits), never a detached fire-and-forget
        task nobody is tracking."""
        if not self.closed:
            self.closed = True
            self.fatal_message = message
        self._wake.set()

    def publish_control(self, data: dict[str, Any]) -> bool:
        """Enqueue one required lifecycle/terminal/rejection message.
        Returns False without enqueuing if the connection is already
        closed. Overflow closes the connection outright (the same fatal
        condition as a failed send), rather than silently dropping a
        required message or merely ending whichever session happens to be
        active."""
        if self.closed:
            return False
        if len(self._control) >= _CONTROL_QUEUE_CAPACITY:
            self._mark_fatal("control queue overflow")
            return False
        self._control.append(data)
        self._wake.set()
        return True

    def publish_status(self, data: dict[str, Any]) -> None:
        """Coalesced: only ever the latest observation is sent, silently
        superseding any earlier one this writer hasn't gotten to yet."""
        if self.closed:
            return
        self._latest_status = data
        self._wake.set()

    def publish_frame(self, stream_id: int, message: bytes) -> None:
        if self.closed:
            return
        queue = self._frame_queues.get(stream_id)
        if queue is not None:
            queue.put(message)
            self._wake.set()

    async def quiesce(self) -> None:
        """Called by an ending session's own ``_finalize`` (safe to call
        more than once -- idempotent). After this returns, the writer
        guarantees it will never again SELECT this session's status or
        frame data: it is the writer itself that notices the request (no
        later than the start of its next selection pass), stops
        selecting, discards whatever of that data it still held pending,
        and only THEN resolves the acknowledgment this method awaits -- so
        there is no window in which the caller can freeze counters while
        the writer is still free to dequeue one more item it just hasn't
        gotten to yet. This method NEVER clears or recreates that
        acknowledgment -- it was created once, already pending, when this
        attempt was bound (see ``bind_session``), and the writer resolves
        it exactly once for this attempt's whole lifetime, either via
        genuine disposal or via its own exit (``_run``'s ``finally``, for
        the ordinary case where a send failure has already closed the
        writer by the time this runs). Clearing it here would instead
        re-arm an already-resolved acknowledgment that a since-exited
        writer will never set again, hanging forever. Repeated calls
        therefore just await the same (eventually-resolved) completion
        rather than waiting anew each time. A send already in flight at
        the moment this is first requested is allowed to finish (that one
        item was already irrevocably underway); required control messages
        (including this session's own terminal reply, published after
        this returns) keep draining normally throughout."""
        self._quiesce_requested = True
        self._wake.set()
        await self._quiesce_ack.wait()

    async def _send_one(
        self, send_call: Callable[[], Awaitable[Any]], *, on_success: Callable[[], None] | None = None
    ) -> None:
        """The ONE place that actually performs a send and applies the
        ONE deadline/failure-close policy -- no other wait_for wraps a
        send anywhere else, so there is exactly one layer of timeout for
        every message this connection ever sends. Only ever called from
        the single writer task (``_run``), so no mutual exclusion is
        needed here -- see ``quiesce`` for how an ending session's data is
        kept from racing this."""
        try:
            await asyncio.wait_for(send_call(), timeout=_SEND_DEADLINE_S)
        except (ConnectionClosed, OSError, TimeoutError) as exc:
            self._mark_fatal(_diagnostic_text(exc))
            return
        if on_success is not None:
            on_success()

    async def _run(self) -> None:
        """Select and send exactly one item per pass. The ONE guard at the
        top is the ONLY thing that ever stops selection ENTIRELY: once the
        CONNECTION itself is closed (a send failure, overflow, an
        unexpected writer exception, or ``stop_and_join``), nothing more
        is ever sent, full stop, even if items remain queued -- an
        uncertain/dead transport must never be written to again, not even
        to "just finish draining the backlog first".

        A session's own ordinary end is a narrower, different stop:
        ``quiesce()`` only turns off selection of THAT session's
        status/frame data, and its disposal step runs UNCONDITIONALLY at
        the top of every pass -- before control is even checked -- so a
        quiesce caller is never stuck waiting behind an entire control
        backlog draining first; a send already in flight when quiesce is
        requested is the only thing allowed to finish before disposal
        runs. The still-good connection keeps draining control messages
        normally throughout, including that session's own terminal reply.

        ANY exception escaping a pass -- not just the three specific
        send-failure types ``_send_one`` already distinguishes -- is
        treated as a fatal connection failure the same way a failed send
        is: a writer that silently died here with no such boundary would
        otherwise leave queues filling forever with nothing to ever
        notice. The outer ``finally`` is the ONE place that actually
        closes the transport and notifies ``on_fatal``, exactly once
        (guarded by ``_transport_closed``) and only for a genuinely fatal
        close (``fatal_message`` set) -- an ordinary ``stop_and_join``
        teardown has no fatal_message and intentionally does neither,
        since the transport is already going away on its own, torn down
        by whoever is calling ``stop_and_join``. This keeps the actual
        close an explicitly observed obligation of this one writer task,
        which ``stop_and_join`` already awaits -- never a detached
        fire-and-forget task nobody is tracking. The ``finally`` also
        unconditionally settles any pending quiesce acknowledgment on the
        way out, regardless of which path ends this loop -- a quiesce
        caller must never hang waiting on a writer that has stopped
        running."""
        rr_index = 0
        try:
            while True:
                if self.closed:
                    return
                try:
                    if self._quiesce_requested and not self._quiesce_ack.is_set():
                        self._latest_status = None
                        for queue in self._frame_queues.values():
                            queue.drain()
                        self._quiesce_ack.set()
                    if self._control:
                        data = self._control.pop(0)
                        await self._send_one(lambda data=data: self._send_json(data))
                        continue
                    if self._quiesce_requested:
                        # Nothing left to select for this attempt -- wait
                        # for more control (its own terminal reply) or a
                        # new attempt binding.
                        self._wake.clear()
                        await self._wake.wait()
                        continue
                    if self._latest_status is not None:
                        data, self._latest_status = self._latest_status, None
                        await self._send_one(lambda data=data: self._send_json(data))
                        continue
                    sent = False
                    stream_ids = list(self._frame_queues)
                    for _ in stream_ids:
                        stream_id = stream_ids[rr_index % len(stream_ids)] if stream_ids else None
                        rr_index = (rr_index + 1) % len(stream_ids) if stream_ids else 0
                        if stream_id is None:
                            break
                        queue = self._frame_queues[stream_id]
                        if not queue:
                            continue
                        message = queue.get()
                        on_sent = self._on_sent
                        await self._send_one(
                            lambda message=message: self._send_binary(message),
                            on_success=(lambda sid=stream_id: on_sent(sid)) if on_sent is not None else None,
                        )
                        sent = True
                        break  # one item per pass; closed is re-checked at the top before selecting another
                    if not sent:
                        self._wake.clear()
                        await self._wake.wait()
                except Exception as exc:  # noqa: BLE001 -- any unexpected writer failure, not just a send's own three types, must still reach the one failure-close policy
                    self._mark_fatal(_diagnostic_text(exc))
        finally:
            # Local notification (on_fatal, the quiesce acknowledgment)
            # happens FIRST and unconditionally, before even attempting
            # the transport close handshake -- that handshake depends on
            # the REMOTE peer cooperating and can stall independently of
            # anything local, and local session Stop/quiesce/source
            # cleanup must never be gated on a stalled peer. The close
            # attempt itself still happens (best-effort, exceptions
            # suppressed), just strictly after signaling locally, not
            # before.
            do_close = self.closed and self.fatal_message is not None and not self._transport_closed
            if do_close:
                self._transport_closed = True
                if self._on_fatal is not None:
                    self._on_fatal(self.fatal_message)
            self._quiesce_ack.set()
            if do_close:
                with contextlib.suppress(Exception):
                    await self._close_transport()


# Stop causes that are merely descriptive of HOW an attempt was asked to
# end, never of something having gone wrong -- see _Session._record_failure.
_BENIGN_STOP_REASONS = frozenset(
    {p.StopReason.REQUESTED, p.StopReason.OWNER_DISCONNECTED, p.StopReason.SERVER_SHUTDOWN}
)


class _Session:
    """Owns one accepted Begin through its terminal SessionStopped --
    schema resolution, startup progress, tracking/camera production,
    bounded queues and cleanup. One task (``run``) is the sole owner,
    mirroring ``capture_server.client``'s supervisor design. Takes an
    already-constructed :class:`~capture_server.acquisition.Acquisition`;
    knows nothing about where its data actually comes from. Each adapter
    inside it owns its own acquisition cadence (see
    ``_produce_tracking``/``_produce_camera`` below)."""

    def __init__(
        self,
        begin: p.BeginRequest,
        cameras: tuple[p.CameraStreamConfig, ...],
        source: Acquisition,
        connect_delay: float,
        mode: p.CaptureMode,
        outbox: _Outbox,
        owner_ws: ServerConnection,
    ) -> None:
        self.session_id = begin.session_id
        self.begin = begin
        # Which owner connection started this session -- a NEW owner
        # connecting later (because the old one disconnected while this
        # session's resource release was stuck/failed, which correctly
        # keeps admission blocked station-wide) must never be able to
        # trigger this session's replay path or receive its stale terminal
        # reply; CaptureServer checks this identity before treating any
        # Begin/Stop as touching THIS session, not just the session_id.
        self.owner_ws = owner_ws
        self._cameras = cameras
        self._source = source
        self._connect_delay = connect_delay
        self._mode = mode
        # The ONE connection-owned Outbox -- shared, not reconstructed per
        # session; see _Outbox's own docstring. This session only ever
        # publishes into it, never calls send_json/send_binary/ws.send
        # directly, and binds its own frame-queue registration/callbacks
        # to it once startup resolves the accepted streams.
        self._outbox = outbox
        self.stop_requested = asyncio.Event()
        self.done = asyncio.Event()
        self.terminal: p.SessionStopped | None = None
        self.init_event: p.SessionInit | None = None
        # Cached EXACT objects first sent, for identical-Begin replay --
        # client._check_duplicate_begun compares the replay byte-for-byte
        # against what it originally received, so a replay must resend
        # these unchanged, never rebuilt with a fresher observation.
        self._begun_event: p.SessionBegun | None = None
        self._last_starting_event: p.SessionStarting | None = None
        self._has_tracking = bool(begin.tracking.sources)
        # Only an advertised stream gets a counter -- a camera-only session
        # (no tracking.sources) must not report a phantom stream_id=0.
        self._counters: dict[int, _StreamCounters] = {
            **({0: _StreamCounters()} if self._has_tracking else {}),
            **{c.stream_id: _StreamCounters() for c in cameras},
        }
        self._sequences: dict[int, int] = {0: 0, **{c.stream_id: 0 for c in cameras}}
        # Authoritative only after done is set: whether _finalize's source
        # release genuinely succeeded. done.set() alone is not proof of
        # that -- CaptureServer only reopens admission when this is True.
        self.cleanup_ok = False
        # Set by any failing phase (connect-delay stop, entry, first step,
        # a producer, the writer, the owner connection, server shutdown);
        # None means a plain requested Stop. Only _finalize ever reads
        # these to build the one terminal result. The FIRST cause/message
        # recorded always wins -- see request_stop().
        self._fatal_reason: p.StopReason | None = None
        self._fatal_message: str | None = None
        self._producer_tasks: list[asyncio.Task] = []
        self._status_task: asyncio.Task | None = None
        self._last_sent_status: p.StatusSnapshot | None = None
        self._blocked = False  # flips True at the start of _finalize's quiesce step

    def _tracking_stream(self) -> p.TrackingStream | None:
        if not self._has_tracking:
            return None
        return p.tracking_stream_descriptor(self.begin.tracking.sources, self.begin.tracking.rate_hz)

    def _publish_control(self, data: dict[str, Any]) -> None:
        """Enqueue into the shared Outbox's control FIFO. A False return
        means the connection is already closed or just overflowed -- the
        Outbox itself already handled closing the transport and reporting
        the fatal condition (via the on_fatal hook bound in
        ``_bind_outbox``); nothing further is needed here."""
        if self._blocked:
            return
        self._outbox.publish_control(data)

    def _publish_status(self, data: dict[str, Any]) -> None:
        if self._blocked:
            return
        self._outbox.publish_status(data)

    def _publish_frame(self, stream_id: int, message: bytes) -> None:
        if self._blocked:
            return
        self._counters[stream_id].produced += 1
        self._outbox.publish_frame(stream_id, message)

    def _on_sent(self, stream_id: int) -> None:
        self._counters[stream_id].sent += 1

    def _record_failure(self, reason: p.StopReason, message: str) -> None:
        """The ONE place every GENUINE operation failure (as opposed to a
        preset benign stop cause -- see ``request_stop``) gets recorded,
        applying one consistent precedence rule everywhere a failure can
        occur: the FIRST true failure recorded is primary and is never
        silently replaced by a later one -- a later failure's text is
        appended instead, so it is never lost even though the original
        reason/primary message stays what gets reported. A failure IS
        allowed to replace a merely benign stop cause
        (REQUESTED/OWNER_DISCONNECTED/SERVER_SHUTDOWN) recorded before any
        real failure occurred -- that benign cause was never actually why
        this attempt ended once a genuine failure shows up, and reporting
        it anyway would silently hide a real problem (e.g. a source
        release failure reported as nothing more than an ordinary
        shutdown)."""
        if self._fatal_reason is None or self._fatal_reason in _BENIGN_STOP_REASONS:
            self._fatal_reason = reason
            self._fatal_message = message
        elif self._fatal_message != message:
            self._fatal_message = f"{self._fatal_message}; additionally: {message}"

    def _on_fatal(self, message: str) -> None:
        self._record_failure(p.StopReason.IC_ERROR, message)
        self.request_stop()

    def _bind_outbox(self, stream_ids: list[int]) -> None:
        self._outbox.bind_session(stream_ids, on_sent=self._on_sent, on_fatal=self._on_fatal)

    async def run(self) -> None:
        """The whole session lifecycle: SessionInit through SessionStopped.
        Always settles ``self.done`` with a non-None ``self.terminal``, via
        the one ``_finalize`` sequence, regardless of how/where this
        attempt ended."""
        try:
            await self._run_unprotected()
        finally:
            try:
                await self._finalize()
            finally:
                # Outermost finally: the attempt task settles no matter what
                # finalization itself does. This is NOT proof that source
                # resources were actually released -- see self.cleanup_ok,
                # which _finalize sets explicitly and which gates whether
                # CaptureServer reopens admission for a new session.
                self.done.set()

    async def _race_stop(self, coro: Awaitable[_T]) -> _T:
        """Await ``coro``, but stop waiting (raising ``_StopDuringStartup``)
        as soon as Stop is requested, instead of blocking until ``coro``
        itself finishes. This does NOT actually interrupt ``coro``'s
        underlying work -- a blocking native call on a dedicated thread
        cannot be forcibly stopped (see ``ThreadRunner.close``'s
        docstring for the same limitation) -- it only stops THIS session
        from hanging on it, so startup can be abandoned and cleanup can
        proceed instead of the whole attempt freezing until a blocked
        ``enter()`` call happens to return on its own.

        The gate is checked HERE, once, before any new source work is
        ever scheduled -- not at each individual startup call site. If
        Stop is already known (requested before this attempt's task even
        got a turn to run, or between two earlier startup calls), ``coro``
        is never scheduled at all: ``ensure_future`` is what actually
        submits a source call onto its worker thread (see
        ``ThreadRunner``), so skipping it here means native entry/step
        genuinely never starts for an attempt already known to be
        stopping, rather than starting it and immediately abandoning it.
        ``coro`` was already constructed by the caller (e.g.
        ``self._source.enter()``) before being passed in, so it is
        explicitly closed rather than silently dropped, to finalize
        cleanly without Python warning about an unawaited coroutine."""
        if self.stop_requested.is_set():
            if asyncio.iscoroutine(coro):
                coro.close()
            raise _StopDuringStartup()
        stop_task = asyncio.ensure_future(self.stop_requested.wait())
        work_task = asyncio.ensure_future(coro)
        done, _pending = await asyncio.wait({stop_task, work_task}, return_when=asyncio.FIRST_COMPLETED)
        if work_task in done:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
            return work_task.result()
        # Stop won the race: leave work_task running in the background
        # (nothing here can force its underlying thread to stop) and
        # swallow whatever it eventually resolves to, so an unretrieved
        # exception isn't logged for work this session no longer cares
        # about.
        work_task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        raise _StopDuringStartup()

    async def _run_unprotected(self) -> None:
        """The ONE session-boundary conversion point: runs the actual
        startup/production body and converts ANY exception that escapes it
        -- typed source failures as well as a genuinely unexpected one from
        anywhere else in that body (a status() read, a descriptor call, the
        source factory, ...) -- into the existing ``_fatal_reason``/
        ``_fatal_message`` terminal-failure machinery, never letting one
        escape into the detached session task where nothing is watching
        for it. A more specific diagnostic already recorded by the body
        itself (or by a producer/status task observed live) is never
        overwritten by this catch-all. Never sends a terminal reply itself
        -- ``run``'s finally always routes to the one ``_finalize``
        sequence regardless of how this returns."""
        try:
            await self._run_unprotected_body()
        except _StopDuringStartup:
            pass  # a clean requested stop, not a failure -- _fatal_reason stays None
        except SourceExited as exc:
            self._record_failure(p.StopReason.IC_EXITED, _diagnostic_text(exc))
        except CameraSourceError as exc:
            self._record_failure(p.StopReason.CAMERA_ERROR, f"{exc.camera_name}: {_diagnostic_text(exc)}")
        except Exception as exc:  # noqa: BLE001 -- any other unexpected failure is still fatal, not REQUESTED
            self._record_failure(p.StopReason.IC_ERROR, _diagnostic_text(exc))

    async def _run_unprotected_body(self) -> None:
        """Everything up through starting production tasks and waiting for
        Stop. Raises on any source/startup failure instead of catching it
        itself -- ``_run_unprotected`` is the one place that converts an
        exception here into a recorded diagnostic."""
        tracking_stream = self._tracking_stream()
        camera_streams = tuple(p.camera_stream_descriptor(c) for c in self._cameras)
        streams: list[p.TrackingStream | p.CameraStream] = list(camera_streams)
        if tracking_stream is not None:
            streams.insert(0, tracking_stream)

        init = p.session_init_event(session_id=self.session_id, mode=self._mode, begin=self.begin, streams=streams)
        self.init_event = init  # cached for CaptureServer._handle_begin's replay path
        # The shared connection Outbox/writer is already running (started
        # once in CaptureServer._handle_connection, for the whole
        # connection's lifetime) -- bind this session's own frame-stream
        # routing/callbacks to it, replacing whatever a PREVIOUS session on
        # this same connection had bound.
        self._bind_outbox([s.stream_id for s in streams])
        self._publish_control(init.to_json())

        starting = p.SessionStarting(
            session_id=self.session_id,
            stage=p.StartupStage.INITIALIZING,
            message="Initializing acquisition sources",
            status=None,
        )
        self._last_starting_event = starting
        self._publish_control(starting.to_json())
        if self._connect_delay > 0:
            try:
                await asyncio.wait_for(self.stop_requested.wait(), timeout=self._connect_delay)
                return
            except TimeoutError:
                pass

        # Interruptible by Stop during a blocked entry -- a real/slow native
        # entry call must not freeze the whole attempt until it happens to
        # return; see _race_stop's own docstring for what "interruptible"
        # does and doesn't mean here. Every startup call on the source --
        # not just entry -- uses the same race_stop policy, for the same
        # reason: a blocked status read or first step must not strand Stop
        # any more than a blocked entry would.
        await self._race_stop(self._source.enter())

        # _last_sent_status is ONE last-observed-snapshot value, updated
        # uniformly at every point this session reads native status --
        # starting, begun, and later via _poll_status -- never gated on
        # having reached Begun first. If startup fails between this point
        # and Begun (e.g. the first tracking step raises), _finalize's
        # terminal reply must still carry the last genuinely observed
        # status rather than losing it entirely.
        self._last_sent_status = await self._race_stop(self._source.status(_now_ns()))
        starting = p.SessionStarting(
            session_id=self.session_id,
            stage=p.StartupStage.STARTING_SOURCES,
            message="Starting acquisition sources",
            status=self._last_sent_status,
        )
        self._last_starting_event = starting
        self._publish_control(starting.to_json())

        if tracking_stream is not None:
            # SessionBegun requires a successful first step, not just
            # successful entry: native context entry and the first step
            # must both have succeeded for tracking requests.
            await self._race_stop(self._source.next_tracking_sample())

        self._last_sent_status = await self._race_stop(self._source.status(_now_ns()))
        begun_event = p.SessionBegun(session_id=self.session_id, status=self._last_sent_status)
        self._begun_event = begun_event  # cached verbatim for replay
        self._publish_control(begun_event.to_json())

        self._producer_tasks = (
            [asyncio.ensure_future(self._produce_tracking(tracking_stream))] if tracking_stream is not None else []
        )
        self._producer_tasks += [asyncio.ensure_future(self._produce_camera(camera)) for camera in self._cameras]
        self._status_task = asyncio.ensure_future(self._poll_status())

        # Wait for Stop OR for any producer/status task to end on its own.
        # Each of those tasks already records _fatal_reason and calls
        # request_stop() itself on every failure it anticipates, so this is
        # not the primary detection path for ordinary source/encode
        # failures -- it is what keeps a genuinely unanticipated exception
        # (a bug in the task's own code, not a source/encode failure) from
        # leaving the session hung forever, by observing that task ending
        # live, the moment it happens, rather than only incidentally
        # noticing it later during cleanup.
        stop_task = asyncio.ensure_future(self.stop_requested.wait())
        watched = [stop_task, *self._producer_tasks, self._status_task]
        done, _pending = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is stop_task or task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                self._record_failure(
                    p.StopReason.IC_ERROR, f"session task failed unexpectedly: {_diagnostic_text(exc)}"
                )
            self.request_stop()
        if not stop_task.done():
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task
        await self.stop_requested.wait()

    async def _finalize(self) -> None:
        """The ONE finalization sequence, reached from every ending (plain
        Stop, connect-delay Stop, startup failure, a mid-capture producer
        failure, a connection-level send failure observed via on_fatal),
        in this exact order:
        1. Stop/join all producers and the source.
        2. Quiesce: block this session's own further publication, then
           await the shared Outbox's explicit quiesce handshake (see
           ``_Outbox.quiesce``) -- an acknowledgment the WRITER itself
           sets only after it has stopped selecting this session's
           status/frames and discarded whatever was still pending, not
           something inferred from lock-release timing. The connection
           itself is NOT torn down here; only this session's own data
           production/accounting stops.
        3. (folded into step 2 -- the writer discards this session's
           queued-but-unsent data as part of the same handshake.)
        4. Freeze accounting (dropped = produced - sent, derived once).
        5. Release the source, recording whether that genuinely succeeded.
        6. Build and send the terminal SessionStopped through the SAME
           Outbox used for everything else, from the last cached status
           observation -- never a fresh call into a source that may
           already have failed. If the connection is already closed (a
           prior send failure), this publish is a harmless no-op -- see
           ``_Outbox.publish_control``.
        Nothing outside this method ever sends SessionStopped or
        constructs ``self.terminal``. There is no separate fallback path:
        a failure at any step here is recorded (fatal reason/message,
        ``self.cleanup_ok``) and the sequence continues through to
        attempting one terminal reply, rather than branching into
        different cleanup code for different failure types."""
        # 1. Stop/join producers and the source -- no partial tick, no
        # further publication can originate from them after this point. Any
        # exception here was already observed LIVE, the moment it happened,
        # by _run_unprotected's asyncio.wait over the same tasks (not
        # discovered only now) -- gather(return_exceptions=True) here is
        # purely to join/cancel, not a second detection path.
        for task in self._producer_tasks:
            task.cancel()
        if self._producer_tasks:
            await asyncio.gather(*self._producer_tasks, return_exceptions=True)
        if self._status_task is not None:
            self._status_task.cancel()
            await asyncio.gather(self._status_task, return_exceptions=True)

        # 2. Quiesce. _blocked stops any FURTHER publication from this
        # session (producers are already cancelled above, so nothing new
        # should arrive anyway -- this also covers the fatal/overflow path,
        # which can fire from inside the Outbox itself). quiesce() is the
        # explicit handshake: the writer itself stops selecting this
        # session's status/frames and discards whatever was still pending
        # BEFORE it sets the acknowledgment this awaits, so step 4's
        # counter-freezing can never race a dequeue the writer hasn't
        # reported yet (not an assumption inferred from lock timing).
        self._blocked = True
        await self._outbox.quiesce()

        # 3. (handled inside quiesce() above -- the writer already
        # discarded whatever of this session's data was still queued.)

        # 4. Freeze accounting. dropped is DERIVED, not separately tracked
        # through any cancellation path: now that production and sending
        # have both genuinely and fully stopped (step 2's barrier), produced
        # - sent IS exactly what was never sent, whether evicted by
        # drop-oldest or left queued when quiescing began.
        for counters in self._counters.values():
            counters.dropped = counters.produced - counters.sent

        # 5. Release the source. close() surfaces its own
        # failures (see ThreadRunner.close), so a real release failure is
        # observed here and recorded rather than silently assumed to have
        # worked. self.cleanup_ok is the one authoritative signal of
        # whether it is safe to admit a new session on this connection --
        # done.set() (run()'s outermost finally) only means this attempt's
        # task settled, not that resources were actually released.
        # _record_failure's precedence means a genuine release failure
        # here CAN replace a merely benign stop cause already recorded
        # (REQUESTED/OWNER_DISCONNECTED/SERVER_SHUTDOWN) -- a resource
        # that failed to release is the real reason this attempt ended
        # badly, not whatever benign cause asked it to stop in the first
        # place, and reporting only the benign cause would hide that.
        try:
            self.cleanup_ok = await self._source.close()
        except Exception as exc:  # noqa: BLE001 -- a genuine release failure, not assumed success
            self.cleanup_ok = False
            self._record_failure(p.StopReason.IC_ERROR, f"source release failed: {_diagnostic_text(exc)}")
        else:
            if not self.cleanup_ok:
                self._record_failure(p.StopReason.IC_ERROR, "source release did not complete within the bound")

        # 6. Build and publish the terminal result, from the last cached
        # status observation -- never a fresh call into a source that may
        # already be broken. Goes through the same _publish_control as
        # every other control message -- if the connection is already
        # closed (a prior send failure), this is a harmless no-op; the
        # Outbox already reported that failure via on_fatal when it
        # happened.
        reason = self._fatal_reason if self._fatal_reason is not None else p.StopReason.REQUESTED
        self.terminal = p.SessionStopped(
            session_id=self.session_id,
            reason=reason,
            error_message=self._fatal_message,
            status=self._last_sent_status,
            streams=tuple(c.to_protocol(sid) for sid, c in sorted(self._counters.items())),
        )
        # _blocked only gates THIS session's non-terminal publication (see
        # _publish_control); the terminal reply is this session's own
        # explicit exception to that, published directly against the
        # Outbox rather than through _publish_control's _blocked check.
        self._outbox.publish_control(self.terminal.to_json())

    async def _poll_status(self) -> None:
        """Generic status observation: read the source's current status on
        an interval and, if it genuinely changed, publish ONE SessionStatus
        -- the shared mechanism by which any source's changing native state
        (mock fault or live observation) reaches the client. Never
        publishes unchanged snapshots as a heartbeat."""
        while not self.stop_requested.is_set():
            await asyncio.sleep(0.1)
            if self.stop_requested.is_set():
                return
            try:
                current = await self._source.status(_now_ns())
            except Exception as exc:  # noqa: BLE001 -- an observed, not silently-vanishing, failure
                self._record_failure(p.StopReason.IC_ERROR, f"status observation failed: {_diagnostic_text(exc)}")
                self.request_stop()
                return
            if _status_excluding_timestamp(current) != _status_excluding_timestamp(self._last_sent_status):
                self._last_sent_status = current
                self._publish_status(p.SessionStatus(session_id=self.session_id, status=current).to_json())

    def request_stop(self, reason: p.StopReason | None = None, message: str | None = None) -> None:
        """Request this attempt stop. ``reason``/``message`` are an
        OPTIONAL preset BENIGN cause (OWNER_DISCONNECTED, SERVER_SHUTDOWN)
        -- set only if nothing at all has been recorded yet, so a genuine
        failure (or an earlier preset) already in place is never
        overwritten by a later one here. Unlike ``_record_failure``, this
        never REPLACES an existing benign cause with another benign one --
        only an actual failure, recorded through ``_record_failure``, is
        ever allowed to upgrade a benign cause after the fact. A plain
        ``request_stop()`` with no arguments (an ordinary client Stop)
        never sets a preset cause, so ``_finalize`` defaults to REQUESTED
        for it exactly as before."""
        if reason is not None and self._fatal_reason is None:
            self._fatal_reason = reason
            self._fatal_message = message
        self.stop_requested.set()

    async def _produce_tracking(self, stream: p.TrackingStream | None) -> None:
        """Repeatedly publish whatever the tracking adapter produces --
        pacing/deadline-skip discipline now lives entirely in
        ``IsaacCaptureAdapter.next_tracking_sample`` -- exactly one
        acquisition cadence owner. This loop only retains publication/
        sequence/counter responsibilities."""
        if stream is None:
            return
        while not self.stop_requested.is_set():
            try:
                capture_time_ns, fields = await self._source.next_tracking_sample()
                payload = p.encode_tracking_frame(fields)
            except SourceExited as exc:
                self._record_failure(p.StopReason.IC_EXITED, _diagnostic_text(exc))
                self.request_stop()
                return
            except Exception as exc:  # noqa: BLE001 -- any other source/encode failure
                self._record_failure(p.StopReason.IC_ERROR, _diagnostic_text(exc))
                self.request_stop()
                return
            seq = self._sequences[0]
            self._sequences[0] += 1
            header = p.FrameMetadata(
                kind=p.BinaryKind.CAPTURE_FRAME,
                stream_id=0,
                sequence=seq,
                # capture_time_ns is the real host-sampling timestamp the
                # adapter stamped at its step() call, not now() at publish
                # time: sample_time_semantics=HOST_POLL means the
                # timestamp marks the host sampling step, not the time the
                # server later finishes assembling the frame for
                # transport. receive_time_ns legitimately IS
                # assembly/publish time -- the two are distinguished on
                # purpose.
                capture_time_ns=capture_time_ns,
                receive_time_ns=_now_ns(),
            )
            self._publish_frame(0, p.encode_header(header, payload))

    async def _produce_camera(self, camera: p.CameraStreamConfig) -> None:
        """Repeatedly publish whatever the camera adapter produces --
        pacing/deadline-skip discipline now lives entirely in
        ``DepthAIAdapter.next_camera_jpeg``. ``None`` means an empty poll,
        not a failure -- polling an empty queue must not manufacture a new
        frame -- this loop just skips publishing that tick, same as before
        for an optional camera's skipped frame."""
        while not self.stop_requested.is_set():
            try:
                result = await self._source.next_camera_jpeg(camera.stream_id)
            except CameraSourceError as exc:
                if camera.required:
                    self._record_failure(p.StopReason.CAMERA_ERROR, f"{exc.camera_name}: {_diagnostic_text(exc)}")
                    self.request_stop()
                    return
                continue  # optional camera: skip this frame, keep going
            except Exception as exc:  # noqa: BLE001 -- an unexpected encode/source failure
                if camera.required:
                    self._record_failure(p.StopReason.CAMERA_ERROR, _diagnostic_text(exc))
                    self.request_stop()
                    return
                continue
            if result is None:
                continue  # empty poll: not a failure, nothing new to publish this tick
            capture_time_ns, jpeg = result
            seq = self._sequences[camera.stream_id]
            self._sequences[camera.stream_id] += 1
            header = p.FrameMetadata(
                kind=p.BinaryKind.CAPTURE_IMAGE,
                stream_id=camera.stream_id,
                sequence=seq,
                # capture_time_ns is the real host-sampling timestamp the
                # adapter stamped at its tryGet() poll, not now() at
                # publish time -- see the tracking-frame branch above for
                # the same spec requirement.
                capture_time_ns=capture_time_ns,
                receive_time_ns=_now_ns(),
            )
            self._publish_frame(camera.stream_id, p.encode_header(header, jpeg))


class CaptureServer:
    """The real WebSocket listener: accepts exactly one owner connection,
    dispatches Begin/Stop, and runs sessions via :class:`_Session`. The
    acquisition implementation is chosen once, here at construction, from
    ``config.mock`` (through :func:`capture_server.launch.build_source_factory`);
    everything else is shared, mode-agnostic implementation. This module
    itself never imports :mod:`capture_server.mock`. ``source_factory`` is an
    explicit override seam for launch composition that binds mock-only
    options (the CLI) and for tests: a factory built by
    :mod:`capture_server.launch` that disagrees with ``config.mock`` is
    rejected, so the reported mode cannot differ from the data published.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        *,
        source_factory: Callable[[p.BeginRequest, tuple[p.CameraStreamConfig, ...]], Acquisition] | None = None,
    ) -> None:
        self.config = config or ServerConfig()
        if source_factory is None:
            # The single selection point: mock or real, from the setup flag, before any session.
            from .launch import build_source_factory

            source_factory = build_source_factory(self.config)
        declared_mock = getattr(source_factory, "mock", None)
        if declared_mock is not None and declared_mock != self.config.mock:
            raise ValueError(
                f"source_factory is for {'mock' if declared_mock else 'real'} acquisition "
                f"but ServerConfig.mock={self.config.mock}"
            )
        # The reported capture mode always agrees with the selected setup.
        self._mode = p.CaptureMode.MOCK if self.config.mock else p.CaptureMode.LIVE
        self._source_factory = source_factory
        self._stop_event = asyncio.Event()
        self._ws_server: websockets.WebSocketServer | None = None
        self._owner: ServerConnection | None = None
        # The ONE connection-owned Outbox, constructed once per owner
        # connection and reused across every session that runs on it --
        # see _Outbox's own docstring. No code path anywhere calls
        # ws.send() directly; everything (rejections, replays, session
        # lifecycle, status, frames) goes through this.
        self._outbox: _Outbox | None = None
        self._active_session: _Session | None = None
        # Scoped to the CURRENT owner connection, reset in _handle_connection
        # -- a session ID completed under a since-disconnected owner must
        # not leak into a brand new connection's admission decisions.
        # Bounded tombstone history (not just the single most recent
        # completion): a client can legitimately retry a Stop for an
        # attempt that finished one or two Begin/Stop cycles ago, not only
        # the literal last one, so this needs a small window, not a single
        # value.
        self._completed: collections.OrderedDict[str, p.SessionStopped] = collections.OrderedDict()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _camera_stream_config(self, stream_id: int, camera: Any) -> p.CameraStreamConfig:
        """The server-owned settings for one requested camera: its station entry, else the mock defaults."""
        station = (self.config.cameras or {}).get(camera.name)
        return p.CameraStreamConfig(
            stream_id=stream_id,
            name=camera.name,
            type=camera.type,
            required=station.required if station else self.config.camera_required,
            width=station.width if station else 1280,
            height=station.height if station else 720,
            fps=station.fps if station else 30.0,
            jpeg_quality=station.jpeg_quality if station else 80,
        )

    async def run(self) -> None:
        self._ws_server = await websockets.serve(
            self._handle_connection,
            self.config.host,
            self.config.port,
            compression=None,
            # Inbound (client->server) messages are only ever small JSON
            # Begin/Stop commands, bounded by a 64 KiB control ceiling, not
            # the 2 MiB+32B camera-image ceiling that bounds outbound
            # server->client binary frames instead.
            max_size=_MAX_INBOUND_BYTES,
            # websockets' own default (10s) is longer than this server's
            # existing bounded-shutdown budget (close()'s active-session
            # wait plus _CLOSE_TIMEOUT_S) combined -- a stalled peer could
            # silently blow past every bound this module otherwise states,
            # purely inside the close handshake itself. Set explicitly to
            # stay consistent with those existing numbers.
            close_timeout=_CLOSE_TIMEOUT_S,
            # This endpoint serves native clients only (CaptureClient, replay
            # tools) -- browser access goes through the web backend's own
            # /api/session proxy instead, which is same-origin by construction.
            # origins=[None] rejects any handshake that carries an Origin
            # header (i.e. any browser-initiated connection) while still
            # accepting the missing-Origin native clients this server
            # actually has. This is origin filtering, not authentication: a
            # non-browser client can still omit/spoof Origin, so a remotely
            # exposed deployment still needs its own network/auth/TLS
            # controls; the loopback default remains the real boundary.
            origins=[None],
        )
        mode_label = "MOCK" if self._mode == p.CaptureMode.MOCK else "LIVE"
        default_note = (
            " -- SYNTHETIC data, no real IsaacCapture/OAK-D hardware in use." if mode_label == "MOCK" else ""
        )
        note = self.config.data_note if self.config.data_note is not None else default_note
        print(
            f"[capture-server] {mode_label} CaptureServer listening on ws://{self.config.host}:{self.config.port}/"
            + note,
            file=sys.stderr,
            flush=True,
        )
        await self._stop_event.wait()

    async def close(self) -> None:
        """Stop any active session (deliberately with a modeled
        SERVER_SHUTDOWN cause, not a bare request_stop()), then ALWAYS
        close the listener afterward regardless of whether that session
        settled within the timeout -- these are two independent
        obligations, not one followed by the other only on success. A
        session's own source release is separately bounded by
        ``_SOURCE_RELEASE_TIMEOUT_S``; the wait here must be at least that
        generous plus margin for its own send deadline, or this routinely
        times out on exactly the slow-but-still-succeeding releases that
        bound is meant to tolerate, which previously left
        ``self._ws_server.close()`` unreached entirely.

        Sets the shutdown flag itself (idempotent with ``request_stop``)
        rather than relying on every caller to have called
        ``request_stop()`` first -- this is the one close/admission owner,
        so shutdown state must be true the moment this starts, not only
        when the CLI's own signal-handler path happens to have set it."""
        self._stop_event.set()
        try:
            if self._active_session is not None:
                # No message: SessionStopped.error_message is null for a
                # clean shutdown, same as REQUESTED/OWNER_DISCONNECTED --
                # these are benign stop causes, not error diagnostics.
                self._active_session.request_stop(p.StopReason.SERVER_SHUTDOWN)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._active_session.done.wait(),
                        timeout=_SOURCE_RELEASE_TIMEOUT_S + _SEND_DEADLINE_S + 1.0,
                    )
        finally:
            if self._ws_server is not None:
                self._ws_server.close()
                await asyncio.wait_for(self._ws_server.wait_closed(), timeout=_CLOSE_TIMEOUT_S)

    async def _handle_connection(self, ws: ServerConnection) -> None:
        if self._stop_event.is_set():
            # Shutdown is already underway -- close new admission
            # immediately rather than leaving a gap between request_stop()
            # being called and close() actually closing the listener, in
            # which a brand new connection could otherwise slip in.
            await ws.close(code=1012, reason="server is shutting down")
            return
        if self._owner is not None:
            await ws.close(code=1013, reason="owner connection already active")
            return
        self._owner = ws
        # One Outbox for this connection's whole lifetime -- constructed
        # here, reused across every session that runs on it, torn down in
        # this method's own finally. These closures are the Outbox's only
        # raw primitives: no other code anywhere calls ws.send()/ws.close()
        # for data/control traffic.
        outbox = _Outbox(
            send_json=lambda data: ws.send(json.dumps(data)),
            send_binary=lambda message: ws.send(message),
            close_transport=lambda: ws.close(code=1011, reason="writer could not settle in time"),
        )
        outbox.start()
        self._outbox = outbox
        self._completed.clear()
        try:
            async for raw in ws:
                await self._handle_message(ws, raw)
        except ConnectionClosed:
            pass  # normal client-initiated close
        finally:
            # Connection-level bookkeeping release is its own guaranteed
            # finally, independent of whatever the session-stop handling
            # above does or raises.
            try:
                # Capture the actual session OBJECT once, up front -- never
                # re-read self._active_session after awaiting. _run_session
                # (a separate concurrently-scheduled task) races this same
                # done.wait() and can itself clear self._active_session to
                # None the moment session.run() returns, so dereferencing
                # self._active_session again afterward can see None even
                # though it was non-None a moment ago.
                session = self._active_session
                if session is not None:
                    # No message: see the matching SERVER_SHUTDOWN comment
                    # in close() -- owner loss is also a benign stop cause,
                    # error_message stays null.
                    session.request_stop(p.StopReason.OWNER_DISCONNECTED)
                    await session.done.wait()
                    # self._active_session is the ONE authoritative admission
                    # state: "an attempt is running" and "the last attempt's
                    # resource release failed" are both represented by it
                    # staying non-None. An owner disconnecting must not clear
                    # it on cleanup failure -- unreleased hardware is a
                    # station-wide fact, not something scoped to one owner
                    # connection, so it has to keep blocking a brand new
                    # connection's admission too, not just this one's.
                    # The identity check (not just "cleanup_ok") guards
                    # against _run_session's own finally having already
                    # cleared/replaced self._active_session concurrently.
                    if session.cleanup_ok and self._active_session is session:
                        self._active_session = None
            finally:
                # The connection is already gone by this point (the async
                # for loop above only ends when it is) -- stop_and_join
                # tears down this Outbox's writer task so it never leaks,
                # even if nothing was ever queued for it to send.
                await outbox.stop_and_join()
                self._owner = None
                self._outbox = None

    async def _handle_message(self, ws: ServerConnection, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            await ws.close(code=1002, reason="client must not send binary messages")
            return
        try:
            data = json.loads(raw)
        except ValueError:
            await ws.close(code=1002, reason="malformed JSON")
            return
        if not isinstance(data, dict) or "type" not in data:
            await ws.close(code=1002, reason="missing envelope type")
            return
        if data.get("type") == int(p.MessageType.BEGIN):
            await self._handle_begin(ws, data)
        elif data.get("type") == int(p.MessageType.STOP):
            await self._handle_stop(ws, data)
        else:
            await ws.close(code=1002, reason="unexpected message type from client")

    def _reject(
        self,
        session_id: str,
        request_type: p.MessageType,
        reason: p.RejectionReason,
        message: str,
    ) -> None:
        """Enqueued through the shared Outbox's control FIFO, same as
        every other required message -- not a direct send. There is only
        ever one owner connection's Outbox bound at a time, so no
        connection identity needs to be passed in."""
        if self._outbox is None:
            return
        self._outbox.publish_control(
            p.RequestRejected(
                session_id=session_id,
                request_type=request_type,
                reason=reason,
                error_message=message,
            ).to_json(),
        )

    async def _handle_begin(self, ws: ServerConnection, data: dict[str, Any]) -> None:
        # Envelope-level problems (missing/invalid type, version, session_id
        # -- including an overlong one) close 1002; a STRUCTURALLY VALID
        # envelope whose configuration is merely unsupported gets
        # RequestRejected/INVALID_CONFIG instead. Routed through the exact
        # same ``require_envelope`` that ``parse_stop`` uses (see its own
        # docstring) rather than a separately hand-maintained check, so
        # Begin and Stop can never quietly define "valid envelope"
        # differently from each other again.
        try:
            session_id = p.require_envelope(data, p.MessageType.BEGIN)
        except p.ProtocolError:
            await ws.close(code=1002, reason="invalid Begin envelope")
            return

        # Shutdown admission gate: checked here, not just for brand NEW
        # connections in _handle_connection -- an EXISTING idle owner can
        # otherwise still send a Begin after request_stop()/close() has
        # already begun, racing shutdown's own wait-for-session-completion.
        # Also rejects outright if this connection's Outbox/writer has
        # already ended (a send failure can close the writer while the
        # read side of this same socket is still alive) -- binding a new
        # session onto a dead writer would create a fresh quiesce
        # acknowledgment that writer will never resolve, hanging that
        # session's own _finalize forever.
        if self._stop_event.is_set() or self._outbox is None or self._outbox.closed:
            self._reject(
                session_id,
                p.MessageType.BEGIN,
                p.RejectionReason.BUSY,
                "server is shutting down" if self._stop_event.is_set() else "this connection is no longer usable",
            )
            return
        try:
            begin = p.parse_begin(data)
        except p.ProtocolError as exc:
            self._reject(session_id, p.MessageType.BEGIN, p.RejectionReason.INVALID_CONFIG, str(exc))
            return

        known = self.config.known_cameras
        if known is None and self.config.cameras is not None:
            known = tuple(self.config.cameras)
        if known is not None:
            unknown = [c.name for c in begin.cameras if c.name not in known]
            if unknown:
                self._reject(
                    begin.session_id,
                    p.MessageType.BEGIN,
                    p.RejectionReason.INVALID_CONFIG,
                    f"unknown camera name(s): {unknown}",
                )
                return

        # An enabled plugin needs IC actually entered for discovery to run
        # at all -- but the real IC graph-construction API rejects an
        # empty output mapping outright, so "a plugin with zero tracking
        # sources selected" has no valid native graph to build in the
        # first place. Plugin requests ARE independent of output
        # subscriptions otherwise (a session can enable Manus without
        # selecting HANDS, and vice versa) -- this rejects only the
        # specific zero-sources-at-all combination, not any particular
        # source choice.
        if begin.tracking.sources == () and any(plugin.enabled for plugin in begin.plugins):
            self._reject(
                begin.session_id,
                p.MessageType.BEGIN,
                p.RejectionReason.INVALID_CONFIG,
                "an enabled plugin requires at least one tracking.sources entry",
            )
            return

        if self._active_session is not None:
            active = self._active_session
            # Owner identity, not just session_id, gates replay: a session
            # can remain the active/admission-blocking one after its owner
            # disconnected (its resource release failed) -- a DIFFERENT,
            # newly-connected owner reusing that same session_id must never
            # be treated as replaying it (would publish into that old,
            # orphaned owner's outbox, which nobody is listening to) or be
            # able to receive its stale terminal reply.
            if begin.session_id == active.session_id and active.owner_ws is ws:
                # Identical-ID replay: do not reallocate resources or
                # reset sequences -- replay the cached SessionInit, then
                # the current progress/established status.
                if begin != active.begin:
                    self._reject(
                        begin.session_id,
                        p.MessageType.BEGIN,
                        p.RejectionReason.INVALID_CONFIG,
                        "Begin configuration differs from the active session",
                    )
                    return
                # Resend the EXACT cached objects originally sent, not
                # rebuilt with a fresher status -- CaptureClient's own
                # duplicate-SessionInit/SessionBegun validation compares a
                # replay byte-for-byte against what it first received, so
                # a rebuilt replay with a new timestamp/snapshot would be
                # rejected by a real client. Any newer observation is
                # published separately through the normal SessionStatus
                # path, not folded into this replay. Route through the
                # active session's own Outbox (not a direct ws.send) so
                # this doesn't race that session's single writer.
                if active.init_event is not None:
                    active._publish_control(active.init_event.to_json())
                if active._begun_event is not None:
                    active._publish_control(active._begun_event.to_json())
                    # Begun itself is replayed byte-for-byte, unchanged --
                    # but a replay must also carry current PROGRESS, which
                    # for an established session means its latest observed
                    # status, sent separately as its own SessionStatus
                    # rather than folded into Begun.
                    if active._last_sent_status is not None:
                        active._publish_status(
                            p.SessionStatus(
                                session_id=active.session_id, status=active._last_sent_status
                            ).to_json()
                        )
                elif active._last_starting_event is not None:
                    # SessionStarting already carries its own inline status
                    # snapshot -- no separate SessionStatus replay needed.
                    active._publish_control(active._last_starting_event.to_json())
                return
            # KNOWN LIMITATION: if the active session's own resource
            # release failed/timed out (cleanup_ok is False, see
            # _Session.finalize), the server rejects BUSY for every future
            # Begin from any connection -- self._active_session is
            # server-wide, not scoped to the connection that owned it --
            # including after the underlying release might have genuinely
            # finished in the background, since nothing here re-awaits it.
            # Restarting the server is the documented recovery (also with
            # real hardware: a timed-out native release does not prove the
            # device was released); retrying the release before answering
            # BUSY is deferred until a stuck release has been observed.
            self._reject(
                begin.session_id,
                p.MessageType.BEGIN,
                p.RejectionReason.BUSY,
                "another session is already active",
            )
            return

        if begin.session_id in self._completed:
            self._reject(
                begin.session_id,
                p.MessageType.BEGIN,
                p.RejectionReason.INVALID_CONFIG,
                "session_id was already used by a completed session",
            )
            return

        # required is a fixed station acquisition policy, independent of any
        # mock fault -- whether a camera's failure proves fatal is decided
        # here once per camera, and the source (mock or live) separately
        # decides whether that camera actually fails this session, via the
        # adapter's enter()/next_camera_jpeg() raising, not by this flag.
        cameras = tuple(self._camera_stream_config(i + 1, c) for i, c in enumerate(begin.cameras))
        try:
            source = self._source_factory(begin, cameras)
        except Exception as exc:  # noqa: BLE001 -- a deliberate rejection, not an unobserved crash
            self._reject(
                begin.session_id,
                p.MessageType.BEGIN,
                p.RejectionReason.INVALID_CONFIG,
                f"could not construct an acquisition source: {exc}",
            )
            return
        assert self._outbox is not None  # bound in _handle_connection before any message is handled
        session = _Session(
            begin,
            cameras,
            source,
            self.config.connect_delay,
            self._mode,
            self._outbox,
            ws,
        )
        self._active_session = session
        asyncio.ensure_future(self._run_session(session))

    async def _run_session(self, session: _Session) -> None:
        try:
            await session.run()
        finally:
            if session.terminal is not None:
                self._completed[session.session_id] = session.terminal
                self._completed.move_to_end(session.session_id)
                while len(self._completed) > _COMPLETED_TOMBSTONE_CAPACITY:
                    self._completed.popitem(last=False)
            # Admission reopens only when the session's own cleanup
            # genuinely succeeded -- a failed source release leaves this
            # connection unable to admit a new session rather than silently
            # reusing hardware/state that was never actually released.
            if self._active_session is session and session.cleanup_ok:
                self._active_session = None

    async def _handle_stop(self, ws: ServerConnection, data: dict[str, Any]) -> None:
        try:
            stop = p.parse_stop(data)
        except p.ProtocolError:
            await ws.close(code=1002, reason="invalid Stop envelope")
            return

        # Owner identity gates this too, same reasoning as the Begin replay
        # path above: a stale active session left blocking admission after
        # its original owner disconnected must not be stoppable/replayable
        # by a DIFFERENT, newly-connected owner that happens to reuse its
        # session_id.
        if (
            self._active_session is not None
            and self._active_session.session_id == stop.session_id
            and self._active_session.owner_ws is ws
        ):
            if self._active_session.done.is_set():
                # The attempt itself already finished (its own writer has
                # stopped), but it's still held as the active session
                # because its resource release failed -- request_stop()
                # here would be a no-op no one is listening to. Replay the
                # terminal it already produced instead, same as a retry
                # against a cleanly-completed session would get. Through
                # the shared Outbox, not a direct send.
                if self._active_session.terminal is not None and self._outbox is not None:
                    self._outbox.publish_control(self._active_session.terminal.to_json())
                return
            self._active_session.request_stop()
            return
        if stop.session_id in self._completed:
            if self._outbox is not None:
                self._outbox.publish_control(self._completed[stop.session_id].to_json())
            return
        self._reject(
            stop.session_id,
            p.MessageType.STOP,
            p.RejectionReason.UNKNOWN_SESSION,
            "no active or recently completed session with that ID",
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    # server.py never imports capture_server.mock: the mock-only options are
    # plain values here and are validated against the mock's scenarios only
    # when the mock is the selected setup (see _mock_only_options_used).
    parser = argparse.ArgumentParser(prog="capture-server")
    # At most one non-real source can be chosen; with neither flag the server runs real acquisition.
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--mock",
        action="store_true",
        help="run with the synthetic mock source (without --mock/--replay the server runs real acquisition)",
    )
    source.add_argument(
        "--replay",
        default=None,
        metavar="EPISODE_DIR",
        help="play back a recorded episode directory (Zarr tracking groups plus *.mp4 cameras) instead of "
        "acquiring; reported to clients as MOCK mode because the data is recorded, not live",
    )
    source.add_argument(
        "--gmsl-ipc",
        default=None,
        metavar="SOCKET_PATH",
        help="live station: read GMSL camera frames from the camera receiver's local frame feed at this Unix "
        "socket (tracking still comes from IsaacCapture via ISAAC_ROOT); cameras are named with --gmsl-camera",
    )
    parser.add_argument(
        "--gmsl-camera",
        action="append",
        default=[],
        metavar="NAME=STREAM[:WIDTHxHEIGHT]",
        help="with --gmsl-ipc: logical camera NAME served from feed STREAM, optionally downscaled to WIDTHxHEIGHT "
        "(the wire protocol caps one image at 2 MiB; a full-resolution head camera can exceed it); repeatable",
    )
    parser.add_argument(
        "--gmsl-fps",
        type=float,
        default=60.0,
        help="with --gmsl-ipc: the feed's frame rate, used to pace and time out polls",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--seed", type=int, default=_MOCK_DEFAULTS["seed"], help="mock only")
    parser.add_argument("--scenario", default=_MOCK_DEFAULTS["scenario"], help="mock only")
    parser.add_argument("--connect-delay", type=float, default=_MOCK_DEFAULTS["connect_delay"], help="mock only")
    parser.add_argument(
        "--fault",
        action="append",
        default=[],
        help="mock only: named fault to inject (repeatable), see mock_isaac_capture.KNOWN_FAULTS "
        "and mock_depthai.KNOWN_FAULTS",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=DEVICE_ID",
        help="real only: station camera, logical NAME -> stable physical DEVICE_ID (OAK-D MXID); repeatable",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="real only")
    parser.add_argument("--camera-height", type=int, default=720, help="real only")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="real only")
    parser.add_argument("--camera-quality", type=int, default=80, help="real only: JPEG quality 1..100")
    parser.add_argument(
        "--camera-usb-speed", default=None, help="real only: cap the camera USB speed, e.g. HIGH (USB2) or SUPER"
    )
    return parser


def _station_cameras(args: argparse.Namespace) -> dict[str, StationCamera] | None:
    """The ``--camera NAME=DEVICE_ID`` entries sharing the global size/fps/quality; None when none given."""
    if not args.camera:
        return None
    cameras: dict[str, StationCamera] = {}
    for entry in args.camera:
        name, sep, device_id = entry.partition("=")
        if not sep or not name:
            raise ValueError(f"--camera must be NAME=DEVICE_ID, got {entry!r}")
        if name in cameras:
            raise ValueError(f"--camera name {name!r} given twice")
        cameras[name] = StationCamera(
            device_id,
            args.camera_width,
            args.camera_height,
            args.camera_fps,
            args.camera_quality,
            max_usb_speed=args.camera_usb_speed,
        )
    return cameras


# Defaults of the mock-only CLI options; any other value with real acquisition is an error.
_MOCK_DEFAULTS = {"seed": 0, "scenario": "demo", "connect_delay": 0.0}


def _mock_only_options_used(args: argparse.Namespace) -> list[str]:
    """Names of the mock-only options given a non-default value."""
    used = [
        f"--{name.replace('_', '-')}" for name, default in _MOCK_DEFAULTS.items() if getattr(args, name) != default
    ]
    return used + (["--fault"] if args.fault else [])


def _replay_setup(args: argparse.Namespace) -> tuple[ServerConfig, Any]:
    """Config and source factory for replaying a recorded episode.

    Recorded data is served under ``mock=True`` and reported as MOCK: the protocol has no replay mode, and
    LIVE would misrepresent the source. The episode's cameras become the station cameras, so each stream
    gets its recorded size and rate and any other camera name is rejected.
    """
    from .launch import build_replay_source_factory, replay_station_cameras
    from .replay import load_episode
    from .replay.episode import CONTROLLER_GROUP, HAND_GROUP

    episode = load_episode(args.replay)
    streams = sorted(
        ([CONTROLLER_GROUP] if episode.controllers is not None else [])
        + ([HAND_GROUP] if episode.hands is not None else [])
        + list(episode.cameras)
    )
    config = ServerConfig(
        mock=True,
        host=args.host,
        port=args.port,
        cameras=replay_station_cameras(episode),
        data_note=(
            f" -- REPLAY of recorded episode {episode.path} (streams: {', '.join(streams)}; loops every "
            f"{episode.period:.1f}s). Recorded data, not live hardware; reported to clients as MOCK mode."
        ),
    )
    return config, build_replay_source_factory(episode)


def _gmsl_cameras(entries: list[str]) -> dict[str, Any]:
    """Parse ``--gmsl-camera NAME=STREAM[:WIDTHxHEIGHT]`` entries; a malformed or repeated one is an error."""
    from .launch import GmslCamera

    cameras: dict[str, Any] = {}
    for entry in entries:
        name, sep, rest = entry.partition("=")
        stream, _, size_text = rest.partition(":")
        if not sep or not name or not stream:
            raise ValueError(f"--gmsl-camera must be NAME=STREAM[:WIDTHxHEIGHT], got {entry!r}")
        if name in cameras:
            raise ValueError(f"--gmsl-camera name {name!r} given twice")
        size: tuple[int, int] | None = None
        if size_text:
            width_text, x, height_text = size_text.partition("x")
            valid = x and width_text.isdigit() and height_text.isdigit() and int(width_text) * int(height_text) > 0
            if not valid:
                raise ValueError(f"--gmsl-camera size must be WIDTHxHEIGHT, got {size_text!r} in {entry!r}")
            size = (int(width_text), int(height_text))
        cameras[name] = GmslCamera(stream, size)
    return cameras


def _gmsl_setup(args: argparse.Namespace) -> tuple[ServerConfig, Any]:
    """Config and source factory for the live station: GMSL cameras from the frame feed plus real tracking.

    The feed is probed once here so every configured camera's stream and size is known (and any unknown
    stream rejected) before a session asks for it; sessions then subscribe on their own.
    """
    from .gmsl_ipc import probe_feed
    from .launch import build_gmsl_source_factory, gmsl_station_cameras

    cameras = _gmsl_cameras(args.gmsl_camera)
    if not cameras:
        raise ValueError("--gmsl-ipc needs at least one --gmsl-camera NAME=STREAM[:WIDTHxHEIGHT]")
    catalog = probe_feed(args.gmsl_ipc, tuple(dict.fromkeys(c.stream for c in cameras.values())))
    stations = gmsl_station_cameras(cameras, catalog, fps=args.gmsl_fps, jpeg_quality=args.camera_quality)
    config = ServerConfig(
        mock=False,
        host=args.host,
        port=args.port,
        cameras=stations,
        data_note=(
            " -- GMSL cameras from the frame feed at "
            f"{args.gmsl_ipc}: "
            + ", ".join(
                f"{name}<-{cam.stream} {stations[name].width}x{stations[name].height}"
                for name, cam in cameras.items()
            )
        ),
    )
    return config, build_gmsl_source_factory(args.gmsl_ipc, cameras, stations)


async def _run_cli(args: argparse.Namespace) -> int:
    from .launch import build_source_factory

    try:
        if args.replay is not None:
            config, source_factory = _replay_setup(args)
        elif args.gmsl_ipc is not None:
            config, source_factory = _gmsl_setup(args)
        else:
            config = ServerConfig(
                mock=args.mock,
                host=args.host,
                port=args.port,
                connect_delay=args.connect_delay,
                cameras=_station_cameras(args),
            )
            # The one place the implementation is chosen, before any session is accepted.
            source_factory = build_source_factory(
                config, scenario=args.scenario, seed=args.seed, faults=frozenset(args.fault)
            )
    except (ValueError, NotImplementedError, OSError, RuntimeError) as exc:
        # A malformed episode is a ValueError (EpisodeFormatError); a missing directory an OSError; an
        # unreachable or malformed frame feed a FeedError (RuntimeError).
        print(f"capture-server: {exc}", file=sys.stderr)
        return 2
    server = CaptureServer(config, source_factory=source_factory)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(sig, server.request_stop)

    try:
        await server.run()
    finally:
        await server.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.mock and (used := _mock_only_options_used(args)):
        parser.error(f"{', '.join(used)} only apply with --mock; real acquisition and replay have no such options")
    if (args.mock or args.replay is not None or args.gmsl_ipc is not None) and args.camera:
        parser.error(
            "--camera only applies to real OAK-D acquisition; mock cameras use fixed settings, "
            "replay cameras come from the episode and GMSL cameras are named with --gmsl-camera"
        )
    if args.gmsl_ipc is None and args.gmsl_camera:
        parser.error("--gmsl-camera only applies with --gmsl-ipc")
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover - exercised by running the module, not pytest
    raise SystemExit(main())
