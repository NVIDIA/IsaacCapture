# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reader for the SENSING plugin's existing H.264 file output, decoded to RGB on the CPU.

The producer (``camera_plugin_sensing``) writes each sensor's H.264 Annex-B elementary stream to the ``output=``
path of its ``--add-stream``. This module tails that file with one persistent parser and one persistent software
decoder per sensor: every complete access unit is decoded IN ORDER (ordinary H.264 P-frames reference earlier
frames, so units cannot be skipped before decoding), and only the newest DECODED frame is kept for delivery. The
same code serves the real Jetson encoder and the native mock; neither is special-cased.

Everything is blocking and synchronous: run it on one dedicated worker (see ``ThreadRunner``). The parser and
decoder are PyAV's, created lazily so ordinary OAK-only runs do not load them until a SENSING stream attaches.
"""

from __future__ import annotations

import collections
import ctypes
import errno
import functools
import os
import re
import threading
import time
from typing import Any, Callable

import numpy as np

__all__ = ["SensingStreamError", "SensingStreamReader"]

# Bytes consumed from the file before the already-read prefix is handed back to the filesystem. The producer
# writes at the encoder bitrate for the whole session, so without this the file grows without bound.
_PUNCH_CHUNK_BYTES = 8 * 1024 * 1024
_READ_BYTES = 4 * 1024 * 1024
# Work done per poll is capped by bytes (a producer appending faster than we parse cannot hold the worker) and
# by the poll's own deadline; the rest of a backlog is simply read on the next poll.
_MAX_POLL_BYTES = 32 * 1024 * 1024
_MIN_WORK_S = 0.2
# Bytes fed to the parser without it completing a single access unit before the stream is declared bad: a
# malformed stream must not make the parser retain input without bound.
_MAX_UNIT_BYTES = 32 * 1024 * 1024
# Complete but not yet decoded stream bytes the reader may hold. A decoder that falls this far behind resumes at
# the newest restart point (an IDR unit carrying its own SPS/PPS; nothing before it is needed), or ends the stream.
_MAX_PENDING_BYTES = 64 * 1024 * 1024
# A failed stream's file may stay unreclaimed (a failing fallocate) only this far before the producer stops.
_MAX_UNRECLAIMED_BYTES = 64 * 1024 * 1024
# Time allowed to read, decode and flush whatever the producer left behind when it ended.
_FINISH_BUDGET_S = 30.0
_POLL_INTERVAL_S = 0.005
_FALLOC_FL_KEEP_SIZE = 0x01
_FALLOC_FL_PUNCH_HOLE = 0x02


# NAL units read straight from the Annex-B bytes (a start code, then a header byte whose low five bits are the
# type): an IDR slice (5), a sequence parameter set (7) and a picture parameter set (8), for any nal_ref_idc.
_IDR_NAL = re.compile(rb"\x00\x00\x01[\x25\x45\x65]")
_SPS_NAL = re.compile(rb"\x00\x00\x01[\x27\x47\x67]")
_PPS_NAL = re.compile(rb"\x00\x00\x01[\x28\x48\x68]")


def _is_restart_point(packet: Any) -> bool:
    """Whether decoding can safely begin at this access unit with nothing decoded before it.

    That needs an IDR slice (no earlier frame referenced) AND in-band SPS and PPS in the same unit: an IDR alone
    says nothing about whether the decoder has the parameter sets it refers to. Read from the bytes themselves,
    never from a parser flag.
    """
    data = bytes(packet)
    return all(nal.search(data) is not None for nal in (_IDR_NAL, _SPS_NAL, _PPS_NAL))


class SensingStreamError(Exception):
    """The producer's stream is unreadable, undecodable, or disagrees with the expected geometry."""


@functools.cache
def _bind_fallocate() -> Callable[[int, int, int, int], int]:
    """The C library's ``fallocate`` with explicit argument types, bound once; ``-1`` plus ``errno`` on failure."""
    fallocate = ctypes.CDLL(None, use_errno=True).fallocate
    fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    return fallocate


class SensingStreamReader:
    """One sensor's H.264 file: attach, then repeatedly take the newest decoded frame as an RGB array.

    A parse, decode or geometry failure is terminal: later reads raise and a new attempt builds a new reader.
    The returned array is owned by the reader and valid until the next read.
    """

    def __init__(
        self,
        path: str,
        *,
        width: int,
        height: int,
        full_range: bool = False,
        fallocate: Callable[[int, int, int, int], int] | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        """Record the expected geometry and colour range; the file is opened later, in :meth:`connect`.

        ``stop`` is set (from any thread) when the stream is being shut down: every wait and loop here checks it
        at least every fraction of a second and ends with :class:`SensingStreamError`, so Stop never queues
        behind a long startup, read or final flush.
        """
        self._path = path
        self._width, self._height = width, height
        self._full_range = full_range
        self._fd: int | None = None
        self._offset = 0
        self._punched = 0
        self._closed = False
        self._parser: Any = None
        self._decoder: Any = None
        self._failed = False
        # Complete access units (PyAV packets) waiting to be decoded, in stream order, and their total size.
        self._pending: collections.deque[Any] = collections.deque()
        self._pending_bytes = 0
        self._newest: Any = None  # newest decoded frame not yet delivered
        self._finished = False
        self._fed_since_packet = 0
        # Injectable so tests can make the native call fail; production binds the C library's once.
        self._fallocate = fallocate
        self._stop = stop

    def _check_stopped(self) -> None:
        """End the current operation, with the reason, if the stream is being shut down."""
        if self._stop is not None and self._stop.is_set():
            raise SensingStreamError("stopped")

    def _wait(self, seconds: float) -> None:
        """Sleep for ``seconds``, or less if the stream is shut down meanwhile."""
        if self._stop is not None:
            self._stop.wait(seconds)
        else:
            time.sleep(seconds)

    def connect(self, timeout_s: float) -> None:
        """Wait up to ``timeout_s`` for the producer to create the file, then open it.

        Raises ``FileNotFoundError`` (an ``OSError``) if it never appears, and :class:`SensingStreamError`
        if it exists but cannot be opened for reading and space reclamation: the stream file is a bounded
        spool, so a file we cannot reclaim from is an error, never a quietly unbounded one.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_stopped()
            try:
                # Read-write: handing consumed bytes back (hole punching) needs a writable descriptor.
                self._fd = os.open(self._path, os.O_RDWR)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise
                self._wait(_POLL_INTERVAL_S * 4)
            except OSError as e:
                raise SensingStreamError(f"cannot open {self._path} for reading and reclamation: {e}") from e
        if self._fallocate is None:
            self._fallocate = _bind_fallocate()
        import av  # noqa: PLC0415 -- lazy: only a SENSING stream needs the decoder

        self._parser = av.CodecContext.create("h264", "r")
        self._decoder = av.CodecContext.create("h264", "r")
        # One thread: frame threading delays output by several frames, which a live preview should not pay.
        self._decoder.thread_count = 1

    def _drain_file(self, deadline: float) -> None:
        """Read newly written bytes within this poll's work bounds and queue the complete access units."""
        assert self._fd is not None
        # A snapshot of the size, not "until EOF": a producer that keeps appending cannot extend this poll.
        end = min(os.fstat(self._fd).st_size, self._offset + _MAX_POLL_BYTES)
        while self._offset < end:
            chunk = os.pread(self._fd, min(_READ_BYTES, end - self._offset), self._offset)
            if not chunk:
                break
            self._offset += len(chunk)
            self._fed_since_packet += len(chunk)
            # The parser holds a unit back until it sees the next one begin, so a half-written frame is never
            # returned; a temporary end of file on a growing file just means waiting for more.
            self._queue(self._parser.parse(chunk))
            if self._fed_since_packet > _MAX_UNIT_BYTES:
                raise SensingStreamError(f"no complete access unit in {self._fed_since_packet} bytes of stream")
            if time.monotonic() >= deadline:
                break  # always one chunk of progress, then yield to the caller's deadline
        self._reclaim()

    def _queue(self, packets: list[Any]) -> None:
        """Append parsed access units for decoding, and bound what is held when the decoder is behind."""
        for packet in packets:
            self._pending.append(packet)
            self._pending_bytes += packet.size
            self._fed_since_packet = 0
        if self._pending_bytes <= _MAX_PENDING_BYTES:
            return
        # Overloaded: resume at the newest restart point, dropping everything older undecoded. This is the only
        # place units are skipped before decoding, and it is safe only where nothing earlier is needed: an IDR
        # unit that also carries its SPS and PPS.
        newest_key = max((i for i, pk in enumerate(self._pending) if _is_restart_point(pk)), default=None)
        if newest_key is None:
            raise SensingStreamError(
                f"decoder cannot keep up: {self._pending_bytes} bytes undecoded and no restart point to resume at"
            )
        for _ in range(newest_key):
            self._pending_bytes -= self._pending.popleft().size
        if self._pending_bytes > _MAX_PENDING_BYTES:
            raise SensingStreamError(
                f"decoder cannot keep up: {self._pending_bytes} bytes undecoded even from the newest restart point"
            )

    def _decode_pending(self, deadline: float) -> None:
        """Decode queued units in order until none are left or ``deadline`` passes (one unit of progress at least).

        A decode that returns nothing is the decoder buffering, not an error; every returned frame is checked
        and the newest one is kept.
        """
        while self._pending:
            self._check_stopped()
            packet = self._pending.popleft()
            self._pending_bytes -= packet.size
            for frame in self._decoder.decode(packet):
                self._newest = self._checked(frame)
            if time.monotonic() >= deadline:
                break

    def _checked(self, frame: Any) -> Any:
        """The decoded frame, if it has the geometry the station and the producer agreed on."""
        if (frame.width, frame.height) != (self._width, self._height):
            raise SensingStreamError(
                f"decoded {frame.width}x{frame.height}, expected {self._width}x{self._height}"
            )
        return frame

    def _deliver(self) -> tuple[np.ndarray, int] | None:
        """Convert the newest undelivered decoded frame to RGB, or None if there is none."""
        frame, self._newest = self._newest, None
        if frame is None:
            return None
        selected_ns = time.monotonic_ns()
        # The producer converts with BT.601; the range follows its --full-range flag. Stated explicitly so
        # colours never depend on a decoder's guess for an unspecified stream.
        rgb = frame.reformat(
            format="rgb24", src_colorspace="ITU601", src_color_range="JPEG" if self._full_range else "MPEG"
        ).to_ndarray()
        return rgb, selected_ns

    def _reclaim(self) -> None:
        """Hand the consumed prefix back to the filesystem once enough has accumulated; failure is an error.

        Space reclamation is what keeps a long session bounded, so an unsupported or failing call ends the
        stream with the reason instead of letting the file grow silently; the reclaimed offset advances only
        after the call succeeds.
        """
        pending = self._offset - self._punched
        if pending < _PUNCH_CHUNK_BYTES:
            return
        assert self._fd is not None and self._fallocate is not None
        ctypes.set_errno(0)
        if self._fallocate(self._fd, _FALLOC_FL_PUNCH_HOLE | _FALLOC_FL_KEEP_SIZE, self._punched, pending) != 0:
            code = ctypes.get_errno() or errno.EIO
            raise SensingStreamError(f"cannot reclaim consumed stream bytes: {os.strerror(code)} (errno {code})")
        self._punched = self._offset

    def read_latest(self, timeout_s: float) -> tuple[np.ndarray, int] | None:
        """Wait up to ``timeout_s`` for a decoded frame, then return ``(rgb, selected_ns)`` of the newest one.

        Every complete unit is decoded in order; frames decoded but not yet delivered are dropped in favour of
        the newest. ``selected_ns`` is host monotonic time when that frame is selected (after decoding; the
        protocol's HOST_POLL semantics). Returns None when no frame was produced in time; never fabricates one.
        Any failure makes the reader terminal and is raised as :class:`SensingStreamError`.
        """
        if self._failed:
            raise SensingStreamError("reader failed earlier; a new connection is required")
        if self._finished:
            raise SensingStreamError("stream already finished; the decoder has been flushed")
        if self._fd is None:
            raise SensingStreamError("reader is not connected")
        deadline = time.monotonic() + max(timeout_s, 0.0)
        try:
            while True:
                self._check_stopped()
                # Reading may use a little longer than a zero-timeout poll so it can still make progress.
                work_deadline = max(deadline, time.monotonic() + _MIN_WORK_S)
                self._drain_file(work_deadline)
                self._decode_pending(work_deadline)
                if self._newest is not None:
                    return self._deliver()
                if time.monotonic() >= deadline:
                    return None
                self._wait(_POLL_INTERVAL_S)
        except SensingStreamError:
            self._fail()
            raise
        except Exception as e:  # noqa: BLE001 -- parser/decoder/OS errors all end this stream the same way
            self._fail()
            raise SensingStreamError(f"{type(e).__name__}: {e}") from e

    def _check_finish_budget(self, deadline: float) -> None:
        """Fail the stream, with the reason, if finishing was cancelled or has used up its time budget."""
        self._check_stopped()
        if time.monotonic() >= deadline:
            raise SensingStreamError(
                f"could not finish decoding the remaining stream within {_FINISH_BUDGET_S:g} s"
            )

    def finish(self) -> tuple[np.ndarray, int] | None:
        """Flush the parser and decoder once and return the final frame; call only after the producer has ended.

        A temporary end of file on a growing file is not the end of the stream, so this is never called for it.
        Reads whatever the producer left in the file first, then drains the parser and the decoder's delayed
        output. Returns None if no frame remains. Terminal: the reader cannot be read again.
        """
        if self._failed or self._fd is None or self._finished:
            return None
        self._finished = True
        try:
            far = time.monotonic() + _FINISH_BUDGET_S
            # Everything the producer left behind is read and decoded, in order, before the flush: flushing with
            # units still pending would drop or reorder them. The budget bounds the whole sequence.
            while self._offset < os.fstat(self._fd).st_size or self._pending:
                self._check_finish_budget(far)
                self._drain_file(far)
                self._decode_pending(far)
            self._queue(self._parser.parse(None))  # flush the parser's held-back last unit
            while self._pending:
                self._check_finish_budget(far)
                self._decode_pending(far)
            for frame in self._decoder.decode(None):  # flush the decoder's delayed output
                self._newest = self._checked(frame)
            return self._deliver()
        except SensingStreamError:
            self._fail()
            raise
        except Exception as e:  # noqa: BLE001 -- parser/decoder/OS errors all end this stream the same way
            self._fail()
            raise SensingStreamError(f"{type(e).__name__}: {e}") from e

    def _fail(self) -> None:
        """Make the reader terminal. The descriptor stays open so :meth:`reclaim_all` can keep reclaiming."""
        self._failed = True
        self._pending.clear()
        self._pending_bytes = 0

    def reclaim_all(self) -> None:
        """For a stream that has failed: keep handing the file's bytes back to the filesystem, without decoding.

        The shared producer keeps appending to every sensor's file for the whole session, so a failed stream's
        file would otherwise grow in memory (tmpfs) until Stop. A file the producer created after this stream
        failed (before it ever attached) is opened here. A single failing call is simply tried again on the next
        tick, but the spool is bounded: once more than ``_MAX_UNRECLAIMED_BYTES`` cannot be handed back, this
        raises :class:`SensingStreamError` so the caller can stop the producer rather than let it fill memory.
        """
        self._failed = True
        if self._fd is None and not self._closed:
            try:
                self._fd = os.open(self._path, os.O_RDWR)
            except FileNotFoundError:
                return  # the producer has not created the file yet; nothing is accumulating
            except OSError:
                self._check_unreclaimed(self._path_size(), "cannot open it")
                return
        if self._fd is None:
            return
        if self._fallocate is None:
            self._fallocate = _bind_fallocate()
        size = os.fstat(self._fd).st_size
        if size <= self._punched:
            return
        ctypes.set_errno(0)
        mode = _FALLOC_FL_PUNCH_HOLE | _FALLOC_FL_KEEP_SIZE
        try:
            if self._fallocate(self._fd, mode, self._punched, size - self._punched) == 0:
                self._punched = size
                return
            reason = os.strerror(ctypes.get_errno()) or "fallocate failed"
        except OSError as e:
            reason = str(e)
        self._check_unreclaimed(size - self._punched, reason)

    def _path_size(self) -> int:
        try:
            return os.stat(self._path).st_size
        except OSError:
            return 0

    def _check_unreclaimed(self, unreclaimed: int, reason: str) -> None:
        if unreclaimed > _MAX_UNRECLAIMED_BYTES:
            raise SensingStreamError(
                f"{self._path} holds {unreclaimed >> 20} MiB that cannot be handed back ({reason}); "
                "the spool is unbounded"
            )

    def close(self) -> None:
        """Close the file; nothing else is held (the producer owns the file's lifetime and directory)."""
        self._closed = True
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)
