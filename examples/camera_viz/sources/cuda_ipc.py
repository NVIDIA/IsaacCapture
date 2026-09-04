# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero-copy CUDA frame source over a Unix socket.

Consumes frames published by the sensing plugin's ``CudaIpcPublisher``
(``src/plugins/sensing/core/cuda_ipc_publisher.cpp``). The producer exports one
CUDA allocation holding a ring of RGBA8 slots; this source maps it once and
hands the renderer a CuPy view straight onto producer memory. No encode, no
decode, no host round-trip — the only per-frame traffic on the socket is a
24-byte ready message.

Unlike every other source here, the pixels are *not* ours: a slot stays valid
only until we release it. ``latest()`` therefore releases the previously
returned slot, which is safe because the viz layer copies during ``submit()``
before the next poll comes round.

Wire format mirrors ``core/cuda_ipc_protocol.hpp``; the struct format strings
below are that file's layout and must change with it.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from typing import Optional

from pipeline import Frame, FrameSource, SourceSpec

from ._cuda_vmm import CudaDriverError, driver
from ._helpers import notify, notify_verbose

# core/cuda_ipc_protocol.hpp. '<' pins little-endian and kills native padding;
# the C++ side is checked against these sizes by static_assert.
_HELLO_FMT = "<8I3Q"
_HELLO_SIZE = struct.calcsize(_HELLO_FMT)
_FRAME_FMT = "<2I2Q"
_FRAME_SIZE = struct.calcsize(_FRAME_FMT)
_RELEASE_FMT = "<2I"

_HELLO_MAGIC = 0x44554349
_FRAME_MAGIC = 0x4D524649
_RELEASE_MAGIC = 0x4C455249
_PROTOCOL_VERSION = 1
_FORMAT_RGBA8 = 0

# A producer that answers but disagrees (geometry, protocol version, pixel
# format) is a config error, not a race, so back off hard. Retrying at the
# normal cadence would evict a correctly-configured consumer once a second,
# since the producer serves whoever connected last.
_REJECT_DELAY_S = 5.0

# A retired mapping is freed only after the renderer has provably stopped
# touching it. The mailbox is cleared first, so one poll interval is enough;
# a quarter second is many intervals and still bounds the memory held.
_RETIRE_GRACE_S = 0.25


class CudaIpcSource(FrameSource):
    """RGBA8 frames mapped directly from a producer process's CUDA memory."""

    _kind = "cuda_ipc"

    def __init__(
        self,
        name: str,
        socket_path: str,
        width: int,
        height: int,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "cuda_ipc source requires CuPy (cupy-cuda12x). "
                "Install via `uv pip install cupy-cuda12x`."
            ) from e

        self._cp = cp
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._socket_path = socket_path
        self._reconnect_delay_s = float(reconnect_delay_s)

        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()

        # Mapping state, replaced wholesale on every (re)connect.
        self._slots: list = []
        self._ptr = 0
        self._size = 0
        self._handle = 0
        self._device_id = 0
        self._ctx = 0

        # Mailbox. ``_pending`` is published-but-unconsumed, ``_inflight`` is
        # the slot the renderer received last and may still be reading.
        self._lock = threading.Lock()
        self._pending: Optional[tuple] = None
        self._inflight: Optional[int] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_report_s = 0.0

    # ── FrameSource interface ─────────────────────────────────────────

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"cuda_ipc_{self._spec.name}",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                notify(self._kind, "reader thread did not exit; leaking mapping")
                return
            self._thread = None
        self._teardown()

    def latest(self) -> Optional[Frame]:
        """Newest published frame, or None.

        The returned Frame *borrows* producer memory: this call releases the
        slot handed out by the previous call, so the previous Frame is invalid
        once this returns. Copy before polling again; never hold two.
        """
        with self._lock:
            if self._pending is None:
                return None
            slot, sequence, timestamp_ns = self._pending
            self._pending = None
            previous, self._inflight = self._inflight, slot
            slots = self._slots
            if slot >= len(slots):
                return None  # torn down between publish and poll
            image = slots[slot]

        # The renderer asking for a new frame means it is done with the last
        # one, so its slot can go back to the producer.
        if previous is not None and previous != slot:
            self._release(previous)

        self._frame_count += 1
        now = time.monotonic()
        if now - self._last_report_s >= 5.0:
            notify_verbose(self._kind, f"{self._frame_count} frames, seq={sequence}")
            self._last_report_s = now

        # stream=0: the producer synchronized its copy stream before telling us
        # the slot was ready, so the pixels are complete on any stream we read from.
        return Frame(
            image=image,
            timestamp_ns=timestamp_ns,
            source_id=self._spec.name,
            stream=0,
        )

    # ── reader thread ─────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        opening_notified = False
        while not self._stop.is_set():
            try:
                self._connect()
            except (ValueError, CudaDriverError) as e:
                # The producer is there and we could not agree with it. Say so
                # every time: unlike a missing socket, this will not fix itself.
                self._teardown()
                notify(self._kind, f"rejected producer on {self._socket_path}: {e}")
                self._stop.wait(timeout=_REJECT_DELAY_S)
                continue
            except OSError as e:
                self._teardown()
                if not opening_notified:
                    notify(
                        self._kind, f"waiting for producer on {self._socket_path} ({e})"
                    )
                    opening_notified = True
                self._stop.wait(timeout=self._reconnect_delay_s)
                continue

            notify(
                self._kind,
                f"attached to {self._socket_path} "
                f"({self._spec.width}x{self._spec.height}, {len(self._slots)} slots)",
            )
            opening_notified = False
            self._pump()
            self._teardown()

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(self._socket_path)
        # Short timeout so stop() is responsive while blocked in recv.
        sock.settimeout(0.5)

        fd = -1
        try:
            msg, fds, _flags, _addr = socket.recv_fds(sock, _HELLO_SIZE, 1)
            if not fds:
                raise ValueError("producer sent no memory handle")
            fd = fds[0]
            if len(msg) < _HELLO_SIZE:
                # Ancillary data rides the first byte, so a short read still
                # carries the fd; finish the header before using it.
                msg += _recv_exact(sock, _HELLO_SIZE - len(msg))
            self._handshake(sock, msg, fd)
        except BaseException:
            sock.close()
            raise
        finally:
            # cuMemImportFromShareableHandle dups what it needs; holding the fd
            # open past the import would pin the producer's allocation after it
            # exits.
            if fd >= 0:
                os.close(fd)

    def _handshake(self, sock: socket.socket, msg: bytes, fd: int) -> None:
        (
            magic,
            version,
            width,
            height,
            pixel_format,
            slot_count,
            device_id,
            _sensor_id,
            pitch,
            slot_stride,
            total_bytes,
        ) = struct.unpack(_HELLO_FMT, msg)

        if magic != _HELLO_MAGIC:
            raise ValueError(f"bad handshake magic 0x{magic:08x}")
        if version != _PROTOCOL_VERSION:
            raise ValueError(
                f"producer speaks protocol v{version}, this source speaks "
                f"v{_PROTOCOL_VERSION} — rebuild both sides"
            )
        if pixel_format != _FORMAT_RGBA8:
            raise ValueError(
                f"unsupported pixel format {pixel_format} (expected RGBA8)"
            )
        if (width, height) != (self._spec.width, self._spec.height):
            raise ValueError(
                f"producer serves {width}x{height} but the config declares "
                f"{self._spec.width}x{self._spec.height}"
            )
        if pitch != width * 4:
            raise ValueError(f"expected tightly packed rows, got pitch {pitch}")
        # These size the mapping and the per-slot views, so a producer that
        # disagrees with itself must be rejected here rather than turned into
        # out-of-bounds device pointers.
        if not 2 <= slot_count <= 64:
            raise ValueError(f"implausible slot count {slot_count}")
        if slot_stride < height * pitch:
            raise ValueError(
                f"slot stride {slot_stride} is smaller than a {width}x{height} frame"
            )
        if slot_count * slot_stride > total_bytes:
            raise ValueError(
                f"{slot_count} slots of {slot_stride} B overrun the "
                f"{total_bytes} B allocation"
            )

        cp = self._cp
        # Import into the primary context, which is the one CuPy allocates and
        # launches on; a pointer from any other context would fault on use.
        cp.cuda.Device(device_id).use()
        drv = driver()
        self._ctx = drv.primary_ctx_retain(device_id)
        drv.ctx_set_current(self._ctx)
        self._device_id = device_id

        self._handle = drv.import_fd(fd)
        self._ptr = drv.map_readwrite(self._handle, total_bytes, device_id)
        self._size = total_bytes

        # One CuPy view per slot, built once. ``owner=self`` keeps this source
        # alive for as long as any handed-out frame references its memory.
        self._slots = []
        for i in range(slot_count):
            mem = cp.cuda.UnownedMemory(
                self._ptr + i * slot_stride,
                height * pitch,
                self,
                device_id,
            )
            self._slots.append(
                cp.ndarray(
                    (height, width, 4),
                    dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(mem, 0),
                )
            )

        self._sock = sock

    def _pump(self) -> None:
        """Read ready messages until the producer goes away or we're stopped."""
        sock = self._sock
        assert sock is not None
        while not self._stop.is_set():
            try:
                header = _recv_exact(sock, _FRAME_SIZE)
            except socket.timeout:
                continue
            except OSError as e:
                notify(self._kind, f"producer connection lost ({e})")
                return
            except EOFError:
                notify(self._kind, "producer disconnected")
                return

            magic, slot, sequence, timestamp_ns = struct.unpack(_FRAME_FMT, header)
            if magic != _FRAME_MAGIC or slot >= len(self._slots):
                notify(self._kind, "protocol desync; reconnecting")
                return

            with self._lock:
                superseded = self._pending
                self._pending = (slot, sequence, timestamp_ns)

            # A frame the renderer never picked up is dropped here rather than
            # queued: this is a mailbox, and a stale frame is worth less than
            # the slot it occupies.
            if superseded is not None and superseded[0] != slot:
                self._release(superseded[0])

    # ── plumbing ──────────────────────────────────────────────────────

    def _release(self, slot: int) -> None:
        sock = self._sock
        if sock is None:
            return
        payload = struct.pack(_RELEASE_FMT, _RELEASE_MAGIC, slot)
        try:
            with self._send_lock:
                sock.sendall(payload)
        except OSError:
            # Producer is gone; _pump will notice and reconnect.
            pass

    def _teardown(self) -> None:
        with self._lock:
            self._pending = None
            self._inflight = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        # The context is retained before the mapping exists, so release it even
        # when a connect failed partway — otherwise every retry adds a
        # reference the primary context is never destroyed under.
        if not self._ctx:
            return

        if self._ptr:
            # Drop the views before unmapping, then wait out any submit()
            # already in flight on the render thread — freeing under it would
            # fault.
            self._slots = []
            time.sleep(_RETIRE_GRACE_S)

        drv = driver()
        drv.ctx_set_current(self._ctx)
        failures = drv.unmap(self._ptr, self._size, self._handle)
        if failures:
            notify(self._kind, f"leaked shared mapping: {', '.join(failures)} failed")
        self._slots = []
        self._ptr = 0
        self._size = 0
        self._handle = 0
        drv.primary_ctx_release(self._device_id)
        self._ctx = 0


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    """Read exactly ``count`` bytes. SOCK_STREAM may split any message.

    The socket carries a short timeout so the reader stays responsive to
    stop(). That timeout may only surface *between* messages: abandoning a
    half-read one would leave the stream misaligned, so once any byte has
    arrived this keeps waiting for the rest.
    """
    chunks = []
    remaining = count
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            if remaining == count:
                raise
            continue
        if not chunk:
            raise EOFError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
