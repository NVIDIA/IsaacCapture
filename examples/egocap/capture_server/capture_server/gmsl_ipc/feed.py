# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Client for the station camera receiver's local frame feed: a Unix socket carrying JSON-lines
notifications about GMSL camera frames that live in GPU ring buffers exported as CUDA IPC handles.

The receiver is a separate process on this host. It decodes the GMSL camera streams into
GPU-resident rings (one slot per in-flight frame, NV12 or RGBA8) and publishes one JSON line per
frame naming the stream, the slot it landed in, a running sequence number and timestamps. It is
a broadcast publisher -- several consumers may subscribe at once and a consumer that stops
reading is dropped by the publisher, never blocked -- so attaching here never disturbs the
recorder that may also be consuming the same feed.

Handshake (publisher protocol version 2): the publisher sends ``hello`` (stream catalog without
memory handles), the client answers ``subscribe`` with the stream names it wants, and the
publisher replies ``subscribed`` with those streams' slots (shape, strides, byte size and the
base64 CUDA IPC handle of each slot) or ``error``. Every later line is a ``frame`` notification.

Reading the slot bytes is isolated behind :class:`SlotMapper` so the protocol client, the frame
source and their tests need no GPU: the production mapper (:mod:`.cuda_slots`) opens the IPC
handles with CuPy, tests substitute an in-process mapper.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

__all__ = [
    "FeedError",
    "FeedClient",
    "FrameNotice",
    "SlotMapper",
    "SlotReader",
    "StreamInfo",
    "probe_feed",
]

#: Publisher protocol version this client speaks; older publishers sent slots inside ``hello`` and had no
#: subscription step, a layout this client does not support.
_PROTOCOL_VERSION = 2
#: Bound on one JSON line; a frame notice with full receiver metadata is a few KiB, so this is generous.
_MAX_LINE_BYTES = 1 << 20


class FeedError(RuntimeError):
    """A protocol or connection failure talking to the frame feed."""


@dataclass(frozen=True)
class StreamInfo:
    """One stream of the publisher's catalog, as advertised in ``hello``/``subscribed``."""

    name: str
    width: int
    height: int
    pixel_format: str
    gpu_id: int
    slot_count: int
    #: Raw ``slots`` entries from ``subscribed`` (shape/strides/nbytes/handle); empty from a bare ``hello``.
    slots: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FrameNotice:
    """One ``frame`` notification: which slot of which stream now holds frame ``seq``."""

    stream: str
    slot: int
    slot_generation: int
    seq: int
    #: Publisher-side nanosecond timestamp of the frame (the receiver's clock, see ``metadata``).
    timestamp_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SlotReader(Protocol):
    """Reads the pixel bytes of one stream's ring slots into host memory."""

    def read(self, slot: int) -> np.ndarray:
        """A host copy of ``slot``'s bytes with the slot's advertised shape (rows x pitch, uint8)."""

    def close(self) -> None:
        """Release whatever the mapping holds; idempotent."""


class SlotMapper(Protocol):
    """Opens a :class:`SlotReader` for a subscribed stream."""

    def open(self, stream: StreamInfo) -> SlotReader:
        """Map every slot of ``stream``; raises if the memory cannot be opened on this host."""


class _LineSocket:
    """Blocking JSON-lines reader/writer over one connected socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def send(self, message: dict[str, Any]) -> None:
        """Write one JSON object as a line."""
        self._sock.sendall((json.dumps(message) + "\n").encode("utf-8"))

    def recv(self) -> dict[str, Any] | None:
        """The next complete JSON line, or None when the publisher closed the connection."""
        while b"\n" not in self._buf:
            # An unbounded buffer would let a misbehaving publisher exhaust memory before any newline.
            if len(self._buf) > _MAX_LINE_BYTES:
                raise FeedError("frame feed sent an oversized line")
            chunk = self._sock.recv(65536)
            if not chunk:
                return None
            self._buf.extend(chunk)
        line, _, rest = bytes(self._buf).partition(b"\n")
        self._buf = bytearray(rest)
        return json.loads(line)


def _stream_info(entry: dict[str, Any]) -> StreamInfo:
    """A catalog entry as the publisher sends it, validated into :class:`StreamInfo`."""
    try:
        return StreamInfo(
            name=str(entry["name"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            pixel_format=str(entry["pixel_format"]).lower(),
            gpu_id=int(entry.get("gpu_id", 0)),
            slot_count=int(entry.get("slot_count", len(entry.get("slots", [])))),
            slots=tuple(dict(s) for s in entry.get("slots", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedError(f"malformed stream catalog entry {entry!r}: {exc}") from exc


def _handshake(line: _LineSocket, streams: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, StreamInfo]]:
    """Run hello/subscribe/subscribed and return the hello message plus the subscribed catalog."""
    hello = line.recv()
    if hello is None or hello.get("type") != "hello":
        raise FeedError(f"expected a hello message from the frame feed, got {hello!r}")
    # A publisher without subscriptions would never answer ``subscribe``; fail here with a clear reason
    # instead of hanging on the reply.
    if int(hello.get("version", 1)) < _PROTOCOL_VERSION and "subscriptions" not in set(
        hello.get("features") or []
    ):
        raise FeedError(f"frame feed protocol version {hello.get('version')!r} lacks subscriptions")
    line.send({"type": "subscribe", "streams": list(streams)})
    reply = line.recv()
    if reply is None:
        raise FeedError("frame feed closed the connection before confirming the subscription")
    if reply.get("type") == "error":
        raise FeedError(str(reply.get("error") or "subscription rejected"))
    if reply.get("type") != "subscribed":
        raise FeedError(f"expected a subscribed message from the frame feed, got {reply!r}")
    catalog = {info.name: info for info in (_stream_info(e) for e in reply.get("streams", []))}
    return hello, catalog


def probe_feed(
    socket_path: str, streams: tuple[str, ...] = (), *, timeout_s: float = 5.0
) -> dict[str, StreamInfo]:
    """Connect once, subscribe to ``streams`` (empty = all), return the catalog and disconnect.

    Used at server setup to learn each stream's size and pixel format before any session asks for it,
    so unknown names and sizes are rejected up front rather than at the first Begin.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        try:
            sock.connect(socket_path)
        except OSError as exc:
            raise FeedError(f"cannot connect to the frame feed at {socket_path}: {exc}") from exc
        _, catalog = _handshake(_LineSocket(sock), streams)
        return catalog
    finally:
        sock.close()


class FeedClient:
    """One subscription to the frame feed, keeping the newest notice per stream.

    The reader thread owns the socket: it drains notifications continuously (so the publisher never
    sees this consumer as slow) and keeps only the latest :class:`FrameNotice` per stream -- the
    camera adapter polls for the newest frame and skips any it did not get to, so older notices have
    no consumer here. Slot bytes are read on the caller's thread through the stream's
    :class:`SlotReader`.
    """

    def __init__(
        self,
        socket_path: str,
        streams: tuple[str, ...],
        *,
        slot_mapper: SlotMapper,
        connect_timeout_s: float = 5.0,
    ) -> None:
        if not streams:
            raise ValueError("FeedClient needs at least one stream name")
        self._socket_path = socket_path
        self._streams = tuple(streams)
        self._mapper = slot_mapper
        self._connect_timeout_s = connect_timeout_s
        self._sock: socket.socket | None = None
        self._readers: dict[str, SlotReader] = {}
        self._catalog: dict[str, StreamInfo] = {}
        self._latest: dict[str, FrameNotice] = {}
        self._lock = threading.Lock()
        # Woken on every notice and on failure/close so a per-stream encoder can sleep between frames
        # instead of polling: at 60 Hz the gap between notices is the whole frame budget.
        self._changed = threading.Condition(self._lock)
        self._closed = threading.Event()
        self._failure: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def catalog(self) -> dict[str, StreamInfo]:
        """The subscribed streams' catalog entries (valid after :meth:`connect`)."""
        return dict(self._catalog)

    def connect(self) -> None:
        """Connect, subscribe, map every subscribed stream's slots and start the reader thread."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        try:
            try:
                sock.connect(self._socket_path)
            except OSError as exc:
                raise FeedError(f"cannot connect to the frame feed at {self._socket_path}: {exc}") from exc
            line = _LineSocket(sock)
            _, catalog = _handshake(line, self._streams)
            missing = [name for name in self._streams if name not in catalog]
            if missing:
                raise FeedError(f"frame feed did not confirm stream(s) {missing}")
            # Map before publishing anything: a stream whose memory cannot be opened fails the whole
            # connect, so a caller never sees a half-usable client.
            readers = {name: self._mapper.open(catalog[name]) for name in self._streams}
        except Exception:
            sock.close()
            raise
        self._sock = sock
        self._catalog = catalog
        self._readers = readers
        # The reader thread blocks in recv(); a short timeout lets it notice close() promptly.
        sock.settimeout(0.5)
        self._thread = threading.Thread(
            target=self._reader_loop, args=(line,), name="gmsl-feed-reader", daemon=True
        )
        self._thread.start()

    def _reader_loop(self, line: _LineSocket) -> None:
        """Drain notifications until closed; a publisher failure is recorded for the next poll to raise."""
        try:
            while not self._closed.is_set():
                try:
                    message = line.recv()
                except socket.timeout:
                    continue
                if message is None:
                    raise FeedError("frame feed closed the connection")
                if message.get("type") != "frame":
                    continue
                notice = FrameNotice(
                    stream=str(message["stream"]),
                    slot=int(message["slot"]),
                    slot_generation=int(message.get("slot_generation", 0)),
                    seq=int(message["seq"]),
                    timestamp_ns=int(message["timestamp_ns"]),
                    metadata=dict(message.get("metadata") or {}),
                )
                with self._changed:
                    self._latest[notice.stream] = notice
                    self._changed.notify_all()
        except Exception as exc:  # noqa: BLE001 -- any reader failure is surfaced to the polling adapter
            if not self._closed.is_set():
                with self._changed:
                    self._failure = str(exc) or type(exc).__name__
                    self._changed.notify_all()

    def latest(self, stream: str) -> FrameNotice | None:
        """The newest notice for ``stream`` (not consumed), or None before the first frame.

        Raises :class:`FeedError` once the reader thread has failed, so the adapter reports the camera
        as failed instead of waiting on a dead feed.
        """
        with self._lock:
            if self._failure is not None:
                raise FeedError(self._failure)
            return self._latest.get(stream)

    def wait_for_newer(self, stream: str, seq: int | None, timeout_s: float) -> FrameNotice | None:
        """Block until ``stream`` has a notice newer than ``seq`` (None = any), the feed fails, or the feed
        is closed; returns the notice, or None on timeout/close.

        Raises :class:`FeedError` on a reader failure, like :meth:`latest`. Waiting on the reader's
        condition instead of polling ``latest`` is what lets an encoder react within a millisecond of
        a frame being announced rather than at the next poll tick.
        """
        deadline = time.monotonic() + timeout_s
        with self._changed:
            while True:
                if self._failure is not None:
                    raise FeedError(self._failure)
                notice = self._latest.get(stream)
                if notice is not None and (seq is None or notice.seq > seq):
                    return notice
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed.is_set():
                    return None
                self._changed.wait(remaining)

    def read_slot(self, stream: str, slot: int) -> np.ndarray:
        """Host copy of ``slot`` of ``stream`` (rows x pitch, uint8) through the stream's slot reader."""
        return self._readers[stream].read(slot)

    def close(self) -> None:
        """Stop the reader, unmap the slots and close the socket; idempotent."""
        self._closed.set()
        # Release anyone blocked in wait_for_newer() so encoder threads can exit promptly.
        with self._changed:
            self._changed.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        for reader in self._readers.values():
            reader.close()
        self._readers = {}
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already shut down/broken -- close() below still runs
            self._sock.close()
            self._sock = None
