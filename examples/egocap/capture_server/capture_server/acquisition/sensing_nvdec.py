# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware H.264 decoding for the SENSING reader: Jetson NVDEC driven through a GStreamer subprocess.

A 5 MP stream at 60 fps is far beyond one CPU core of software decoding, so the reader hands the access units to
``nvv4l2decoder`` instead and takes planar YUV frames back. It is a subprocess rather than a library binding
because the Jetson decoder is only reachable through NVIDIA's GStreamer/V4L2 stack, which neither the PyAV wheel
(its ``h264_v4l2m2m`` cannot open the NVDEC node) nor the system ffmpeg wrapper serve well: the ffmpeg wrapper
holds about 0.6 s of frames internally, while ``nvv4l2decoder disable-dpb=true`` delivers each frame in tens of
milliseconds. ``disable-dpb`` is correct only for streams without frame reordering; the producer encodes with no
B-frames by construction (``setNumBFrames(0)`` in the Jetson encoder, mirrored by the mock encoder).

Everything here is thread-based and bounded: three frame buffers, a capped input queue, and a close that returns
within a second. The reader owns when frames are taken, so "newest decoded frame wins" holds exactly as it does
for the software path.
"""

from __future__ import annotations

import collections
import fcntl
import functools
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence

import numpy as np

__all__ = ["HardwareDecodeError", "HardwareDecoder", "hardware_decoder_unavailable_reason", "pipeline_command"]

_log = logging.getLogger("isaaccapture.capture_server.sensing_nvdec")

_DEVICE_NODE = "/dev/v4l2-nvdec"
_GST_LAUNCH = "gst-launch-1.0"
_GST_INSPECT = "gst-inspect-1.0"
_DECODER_ELEMENT = "nvv4l2decoder"
_PROBE_TIMEOUT_S = 10.0
# Bytes the decoder process may have queued on its input before the caller is asked to wait. The pipe itself
# holds up to _STDIN_PIPE_BYTES; the rest waits in this process, where the reader's own overload policy sees it.
_STDIN_PIPE_BYTES = 1 << 20  # the largest size an unprivileged process may request (/proc/sys/fs/pipe-max-size)
_WRITE_CHUNK_BYTES = 1 << 20
# Shutdown bounds: the whole SENSING chain must release inside the server's 5 s budget (see sensing.py).
_CLOSE_GRACE_S = 0.4
_JOIN_S = 0.3
_STDERR_TAIL_BYTES = 4096
# One frame being filled, one waiting to be taken, one held by the consumer while it converts it.
_FRAME_BUFFERS = 3
_F_SETPIPE_SZ = 1031  # Linux fcntl command; not exposed by the fcntl module on every Python


class HardwareDecodeError(Exception):
    """The hardware decoder process failed, exited early, or produced output that does not fit the geometry."""


@functools.cache
def hardware_decoder_unavailable_reason() -> str | None:
    """Why NVDEC decoding cannot be used on this machine, or ``None`` when it can; cached for the process.

    Checks the decoder device node, the GStreamer tools, and that the ``nvv4l2decoder`` element is installed,
    so "auto" falls back to software on a desktop or a Jetson without the NVIDIA GStreamer plugins.
    """
    # The node exists only on Jetson platforms with an NVDEC engine.
    if not os.path.exists(_DEVICE_NODE):
        return f"{_DEVICE_NODE} is missing (this is not a Jetson with NVDEC)"
    for tool in (_GST_LAUNCH, _GST_INSPECT):
        if shutil.which(tool) is None:
            return f"{tool} is not installed"
    try:
        probe = subprocess.run(  # noqa: S603 -- fixed argument list, no shell
            [_GST_INSPECT, "--exists", _DECODER_ELEMENT],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not query GStreamer for {_DECODER_ELEMENT}: {e}"
    if probe.returncode != 0:
        return f"the GStreamer element {_DECODER_ELEMENT} is not installed"
    return None


def pipeline_command(*, low_delay: bool = True) -> list[str]:
    """The ``gst-launch-1.0`` argument list: Annex-B on stdin, tightly packed I420 frames on stdout.

    ``nvvidconv`` only converts NV12 to planar I420 (no scaling and no colour-matrix change), so the reader's own
    explicit BT.601 conversion still decides the colours. ``low_delay`` turns off the decoder's reorder buffer:
    required for a live preview, valid only for streams without B-frames.

    The one-buffer leaky queue ahead of the converter is what keeps the decoder at full speed. Reading a 7.6 MB
    frame through a pipe takes many GIL acquisitions, and when other Python threads (JPEG encoding) hold the GIL
    the reads fall behind; without the queue that backpressure stops the decoder and the input backs up
    (measured: 11 fps decode against 124 fps with the queue). With it, frames the reader is too slow to take are
    dropped inside GStreamer before the costly copy, so the reader always sees the newest one.
    """
    decoder = [_DECODER_ELEMENT, "enable-max-performance=1"]
    if low_delay:
        decoder.append("disable-dpb=true")
    return [
        _GST_LAUNCH,
        "-q",
        "fdsrc",
        "fd=0",
        "blocksize=4194304",
        "!",
        "h264parse",
        "!",
        *decoder,
        "!",
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw,format=I420",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


class HardwareDecoder:
    """One decoder process: feed Annex-B access units in, take the newest decoded I420 frame out.

    Frames are ``(height * 3 // 2, width)`` uint8 arrays (planar Y, U, V), the layout PyAV's ``yuv420p`` frames
    are built from. Not thread-safe for its own callers: ``start``/``write``/``pump``/``close_input``/``close``
    belong to the reader's single worker; only the internal output and stderr threads run beside it.
    """

    def __init__(self, width: int, height: int, *, command: Sequence[str] | None = None) -> None:
        """Record the geometry; the process starts in :meth:`start`.

        ``command`` replaces the GStreamer pipeline (tests inject a stand-in that follows the same pipe contract).
        """
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"I420 needs positive even dimensions, got {width}x{height}")
        self._width, self._height = width, height
        self._frame_bytes = width * height * 3 // 2
        self._command = list(command) if command is not None else pipeline_command()
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffers = [np.empty((height * 3 // 2, width), dtype=np.uint8) for _ in range(_FRAME_BUFFERS)]
        self._lock = threading.Condition()
        self._ready: int | None = None  # newest complete frame not yet taken
        self._held: int | None = None  # frame the consumer is converting
        self._frames_seen = 0
        self._reader_done = threading.Event()
        self._reader_error: Exception | None = None
        self._input: collections.deque[bytes] = collections.deque()
        self._input_offset = 0  # bytes of _input[0] already written
        self._input_bytes = 0
        self._input_closed = False
        self._closed = False
        self._stderr_tail = bytearray()
        self._threads: list[threading.Thread] = []

    @property
    def frames_received(self) -> int:
        """Frames that reached this process, including ones replaced by a newer frame before being taken.

        The decoder may decode more: its leaky queue drops frames this process is too slow to read.
        """
        with self._lock:
            return self._frames_seen

    @property
    def pending_input_bytes(self) -> int:
        """Bytes handed to :meth:`write` that the decoder process has not yet accepted."""
        return self._input_bytes

    def start(self) -> None:
        """Launch the decoder process and its output and diagnostics threads."""
        self._proc = proc = subprocess.Popen(  # noqa: S603 -- fixed argument list, no shell
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        # Writes must never block the reader's worker: a stalled decoder would otherwise outlast Stop. A larger
        # pipe lets a whole IDR access unit go in one call; failing to resize it only costs more small writes.
        os.set_blocking(proc.stdin.fileno(), False)
        try:
            fcntl.fcntl(proc.stdin.fileno(), _F_SETPIPE_SZ, _STDIN_PIPE_BYTES)
        except OSError:
            _log.debug("could not enlarge the decoder input pipe", exc_info=True)
        self._threads = [
            threading.Thread(target=self._read_frames, name="nvdec-frames", daemon=True),
            threading.Thread(target=self._drain_stderr, name="nvdec-stderr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def write(self, data: bytes) -> None:
        """Queue one access unit for the decoder and push as much input as the pipe accepts right now."""
        if self._input_closed or self._closed:
            raise HardwareDecodeError("the decoder input is already closed")
        self._input.append(data)
        self._input_bytes += len(data)
        self.pump()

    def pump(self) -> None:
        """Push queued input into the decoder's pipe without blocking; what does not fit stays queued."""
        proc = self._proc
        if proc is None or proc.stdin is None or self._input_closed:
            return
        fd = proc.stdin.fileno()
        while self._input:
            head = memoryview(self._input[0])[self._input_offset :]
            try:
                written = os.write(fd, head[:_WRITE_CHUNK_BYTES])
            except BlockingIOError:
                return  # the decoder has not consumed what it already has
            except BrokenPipeError as e:
                raise HardwareDecodeError(
                    f"the decoder closed its input unexpectedly; {self._diagnostics()}"
                ) from e
            self._input_bytes -= written
            self._input_offset += written
            if self._input_offset >= len(self._input[0]):
                self._input.popleft()
                self._input_offset = 0

    def close_input(self) -> bool:
        """Signal end of stream once all queued input is written; ``False`` while some is still queued.

        Closing the input makes the decoder flush its remaining frames and exit, which ends :meth:`wait_finished`.
        """
        self.pump()
        if self._input:
            return False
        proc = self._proc
        if proc is not None and proc.stdin is not None and not self._input_closed:
            self._input_closed = True
            proc.stdin.close()
        return True

    def take_frame(self) -> np.ndarray | None:
        """The newest complete frame, or ``None``; the array stays valid until :meth:`release_frame`.

        Taking marks the frame as held so the output thread cannot overwrite it while it is converted; every
        successful take must be followed by :meth:`release_frame`.
        """
        with self._lock:
            if self._ready is None:
                return None
            if self._held is not None:
                raise RuntimeError("release_frame() must be called before taking another frame")
            self._held, self._ready = self._ready, None
            return self._buffers[self._held]

    def release_frame(self) -> None:
        """Let the output thread reuse the frame returned by the last :meth:`take_frame`."""
        with self._lock:
            self._held = None

    @property
    def has_frame(self) -> bool:
        """Whether a complete frame is waiting to be taken."""
        with self._lock:
            return self._ready is not None

    def wait_finished(self, timeout_s: float) -> bool:
        """Wait for the decoder to end its output after :meth:`close_input`; ``False`` if it did not in time."""
        return self._reader_done.wait(timeout_s)

    def check(self) -> None:
        """Raise :class:`HardwareDecodeError` if the decoder failed or exited before its input was closed."""
        if self._reader_error is not None:
            raise HardwareDecodeError(f"{self._reader_error}; {self._diagnostics()}") from self._reader_error
        # The output ending while input is still open means the process died; a closed input ends it on purpose.
        if self._reader_done.is_set() and not self._input_closed:
            code = self._proc.poll() if self._proc is not None else None
            raise HardwareDecodeError(f"the decoder exited unexpectedly (code {code}); {self._diagnostics()}")

    def close(self) -> None:
        """Stop the process and its threads within a fraction of a second; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        # Closing the input lets a healthy decoder exit by itself; the signals cover a stalled one. Both waits are
        # short because Stop has to finish inside the server's release budget.
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(_CLOSE_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for thread in self._threads:
            thread.join(_JOIN_S)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

    def _claim_free_buffer(self) -> int:
        """The index of a buffer neither waiting to be taken nor held; one always exists with three buffers."""
        with self._lock:
            return next(i for i in range(_FRAME_BUFFERS) if i not in (self._ready, self._held))

    def _read_frames(self) -> None:
        """Output thread: read whole frames from the process, keeping only the newest one available."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                index = self._claim_free_buffer()
                flat = memoryview(self._buffers[index]).cast("B")
                got = 0
                # A pipe read may return less than a frame; the frame is complete only when every byte is here.
                while got < self._frame_bytes:
                    count = proc.stdout.readinto(flat[got:])
                    if not count:
                        break
                    got += count
                if got == 0:
                    return  # clean end of stream
                if got < self._frame_bytes:
                    raise HardwareDecodeError(
                        f"decoder output ended mid-frame ({got} of {self._frame_bytes} bytes; "
                        f"expected {self._width}x{self._height} I420)"
                    )
                with self._lock:
                    # A frame the consumer never took is replaced: the preview wants the newest, not every frame.
                    self._ready = index
                    self._frames_seen += 1
                    self._lock.notify_all()
        except Exception as e:  # noqa: BLE001 -- any read failure ends this decoder; check() reports it
            self._reader_error = e
        finally:
            self._reader_done.set()
            with self._lock:
                self._lock.notify_all()

    def _drain_stderr(self) -> None:
        """Diagnostics thread: keep the last few KiB of the process's error output for failure messages."""
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            while chunk := proc.stderr.read(1024):
                self._stderr_tail += chunk
                del self._stderr_tail[:-_STDERR_TAIL_BYTES]
        except (OSError, ValueError):
            pass  # the stream was closed under us by close(); nothing more to collect

    def _diagnostics(self) -> str:
        """The decoder's most recent error output, for the message of a failure."""
        text = bytes(self._stderr_tail).decode("utf-8", errors="replace").strip()
        return f"decoder output: {text[-500:]!r}" if text else "the decoder printed no error output"
