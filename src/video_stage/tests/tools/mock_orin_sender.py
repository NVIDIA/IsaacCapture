#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mock of the Orin producer's direct sender, behaving as the real one does.

It mirrors DirectSender in src/plugins/sensing/core/direct_sender.cpp and what feeds it:
  - the stream: H.264 as the Jetson encoder emits it (no AUD, SPS+PPS before every IDR, no B-frames, constant bitrate,
    an IDR every fps*5 frames), made once with ffmpeg if the source file is missing;
  - frames paced at the publication rate, with source frame ids that step by 2 (60 fps capture, every second capture
    published) and a stamp on the Orin's own monotonic clock (offset from this machine's, optionally drifting), as the
    frame-ready time of each unit;
  - two TCP connections opened by the Orin side, "DATA" then "CTRL" (4-byte hellos), retried every second; on every
    (re)connect the queue is cleared and the stream starts at a keyframe;
  - a bounded queue: a unit that does not fit is dropped, and so is everything after it until the next keyframe;
  - a control thread blocked in recv: 'K' makes the encoder produce a keyframe at the next unit, 'Q' is answered with 'R'
    with t2 stamped when the first byte arrives and t3 just before the reply is sent, on the Orin's clock.
One sender per --port, all in one process on one clock, like the producer's cameras. Runs until stopped or --duration.
"""

from __future__ import annotations

import argparse
import collections
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HEADER = struct.Struct("!4sHHIHHQQQQIIQ")
STREAM = struct.Struct("!HHBBHIIIIQHHI")
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII")
CODEC_H264 = 2
BITRATE = "8M"  # the producer's default target


# ----------------------------------------------------------------------------------------------------------------------
# The stream
# ----------------------------------------------------------------------------------------------------------------------


def make_stream(
    path: Path, width: int, height: int, fps: int, seconds: int = 12
) -> None:
    """Encode a test pattern the way the Jetson encoder does: IDR period fps*5, no B-frames, CBR, no AUDs."""
    gop = fps * 5
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}",
        "-t", str(seconds), "-c:v", "libx264", "-preset", "veryfast", "-profile:v", "high", "-bf", "0", "-g", str(gop),
        "-x264opts", f"keyint={gop}:min-keyint={gop}:scenecut=0:aud=0:repeat-headers=1:nal-hrd=cbr",
        "-b:v", BITRATE, "-maxrate", BITRATE, "-bufsize", "267k", "-pix_fmt", "yuv420p", "-f", "h264", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


_TRAILING_NALS = (10, 11, 12)


def _nals(data: bytes) -> list[tuple[int, int, int]]:
    """(offset of the start code, NAL type, offset of the NAL header byte) for every NAL unit."""
    out = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= len(data):
            break
        start = j - 1 if j > 0 and data[j - 1] == 0 else j
        out.append((start, data[j + 3] & 0x1F, j + 3))
        i = j + 3
    return out


def _first_mb_in_slice(data: bytes, header_at: int) -> int:
    """The slice's first_mb_in_slice, an Exp-Golomb code right after the NAL header byte."""
    bits = int.from_bytes(data[header_at + 1 : header_at + 5], "big")
    zeros = 0
    while zeros < 15 and not (bits >> (31 - zeros)) & 1:
        zeros += 1
    code = (bits >> (32 - (2 * zeros + 1))) & ((1 << (2 * zeros + 1)) - 1)
    return code - 1


def split_access_units(data: bytes) -> list[tuple[bytes, bool]]:
    """Split Annex-B data into (access unit, is IDR). A unit begins at the first NAL after a picture's slices, or at a
    picture whose first slice follows another picture directly; later slices of one picture stay with it."""
    starts = []
    previous_vcl = False
    for position, (start, nal_type, header) in enumerate(_nals(data)):
        vcl = nal_type in (1, 5)
        if position == 0:
            begin = True
        elif vcl:
            begin = previous_vcl and _first_mb_in_slice(data, header) == 0
        else:
            begin = previous_vcl and nal_type not in _TRAILING_NALS
        if begin:
            starts.append(start)
        # End-of-sequence, end-of-stream and filler data (constant-bitrate padding) belong to the picture before them.
        previous_vcl = vcl or (previous_vcl and nal_type in _TRAILING_NALS)
    starts.append(len(data))
    units = []
    for a, b in zip(starts, starts[1:]):
        au = data[a:b]
        units.append((au, any(t == 5 for _, t, _ in _nals(au))))
    return units


def record(
    au: bytes,
    sequence: int,
    frame: int,
    stamp_ns: int,
    width: int,
    height: int,
    fps: int,
) -> bytes:
    descriptor_bytes = STREAM.size + SAMPLE.size
    header = HEADER.pack(
        b"AVS1",
        1,
        0,
        HEADER.size,
        1,
        1,
        sequence,
        1,
        0,
        0,
        descriptor_bytes,
        len(au),
        0,
    )
    stream = STREAM.pack(
        0, 1, CODEC_H264, 1, 0, width, height, fps, 1, sequence, 0, 1, 0
    )
    sample = SAMPLE.pack(
        0, 0, 0, 0x0005, 0, 0, 0, frame, 0, stamp_ns, 0, 0, 0, 0, len(au)
    )
    body = header + stream + sample + au
    return struct.pack("!I", len(body)) + body


# ----------------------------------------------------------------------------------------------------------------------
# The Orin's clock
# ----------------------------------------------------------------------------------------------------------------------


class OrinClock:
    """Monotonic time on a machine with another boot time, optionally running a little fast or slow."""

    def __init__(self, offset_ns: int, drift_ppm: float) -> None:
        self._t0 = time.monotonic_ns()
        self._offset = offset_ns
        self._drift = drift_ppm * 1e-6

    def now_ns(self) -> int:
        local = time.monotonic_ns()
        return local + self._offset + int((local - self._t0) * self._drift)


# ----------------------------------------------------------------------------------------------------------------------
# DirectSender (mirrors direct_sender.cpp)
# ----------------------------------------------------------------------------------------------------------------------


class DirectSender:
    def __init__(
        self,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: int,
        clock: OrinClock,
        max_queued_bytes: int,
    ):
        self.host, self.port, self.width, self.height, self.fps = (
            host,
            port,
            width,
            height,
            fps,
        )
        self.clock = clock
        self.max_queued_bytes = max_queued_bytes
        self._cond = threading.Condition()
        self._queue: collections.deque[bytes] = collections.deque()
        self._queued_bytes = 0
        self._need_keyframe = True
        self._stop = False
        self._sequence = 0
        self._keyframe_requested = False
        self.sent = 0
        self.dropped = 0
        self.keyframe_requests = 0
        self.clock_answers = 0
        self.connects = 0
        self._live: list[socket.socket] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Cancels any connect or send in progress and waits for the worker thread; a stuck thread is an error."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
            live = list(self._live)
        for s in live:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError(f"sender for port {self.port} did not stop")

    def take_keyframe_request(self) -> bool:
        with self._cond:
            requested, self._keyframe_requested = self._keyframe_requested, False
            return requested

    def submit(self, au: bytes, keyframe: bool, frame_id: int, stamp_ns: int) -> None:
        with self._cond:
            if self._need_keyframe:
                if not keyframe:
                    return  # a delta frame cannot start a stream; the keyframe request is pending
                self._need_keyframe = False
            if self._queued_bytes + len(au) > self.max_queued_bytes:
                # Dropping any frame breaks the chain after it: discard up to the next keyframe, and ask for one.
                self.dropped += 1
                self._need_keyframe = True
                self._keyframe_requested = True
                return
            self._sequence += 1
            rec = record(
                au,
                self._sequence,
                frame_id,
                stamp_ns,
                self.width,
                self.height,
                self.fps,
            )
            self._queued_bytes += len(rec)
            self._queue.append(rec)
            self._cond.notify()

    # --- connection --------------------------------------------------------------------------------------------------

    def _connect(self) -> tuple[socket.socket, socket.socket] | None:
        socks = []
        try:
            for hello in (b"DATA", b"CTRL"):
                s = socket.create_connection((self.host, self.port), timeout=3)
                with self._cond:
                    if self._stop:
                        s.close()
                        raise OSError("stopping")
                    self._live.append(s)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                s.sendall(hello)
                socks.append(s)
        except OSError:
            for s in socks:
                s.close()
            with self._cond:
                self._live = []
            return None
        return socks[0], socks[1]

    def _control(self, ctrl: socket.socket, state: dict) -> None:
        """Blocks in recv so t2 is the arrival of a request, not the next time the sender looks at the socket."""
        try:
            while True:
                first = ctrl.recv(1)
                if not first:
                    break
                t2 = self.clock.now_ns()
                if first == b"K":
                    with self._cond:
                        self.keyframe_requests += 1
                        self._keyframe_requested = True
                elif first == b"Q":
                    rest = b""
                    while len(rest) < 12:
                        chunk = ctrl.recv(12 - len(rest))
                        if not chunk:
                            raise OSError("closed inside a request")
                        rest += chunk
                    seq, t1 = struct.unpack("<Iq", rest)
                    t3 = self.clock.now_ns()
                    ctrl.sendall(struct.pack("<cIqqq", b"R", seq, t1, t2, t3))
                    self.clock_answers += 1
        except OSError:
            pass
        state["control_ended"] = True

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stop:
                    return
            pair = self._connect()
            if pair is None:
                with self._cond:
                    if self._cond.wait_for(lambda: self._stop, timeout=1.0):
                        return
                continue
            data, ctrl = pair
            self.connects += 1
            state = {"control_ended": False}
            control = threading.Thread(
                target=self._control, args=(ctrl, state), daemon=True
            )
            control.start()
            with self._cond:
                # The PC needs a keyframe first: drop anything queued before the connection and ask for one.
                self._queue.clear()
                self._queued_bytes = 0
                self._need_keyframe = True
                self._keyframe_requested = True
            try:
                while True:
                    with self._cond:
                        self._cond.wait_for(
                            lambda: self._stop or self._queue, timeout=0.2
                        )
                        if self._stop:
                            return
                        if not self._queue:
                            if state["control_ended"]:
                                break
                            continue
                        rec = self._queue.popleft()
                        self._queued_bytes -= len(rec)
                    data.sendall(rec)
                    self.sent += 1
            except OSError:
                pass
            finally:
                for s in (ctrl, data):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    s.close()
                with self._cond:
                    self._live = []
                control.join(timeout=2)


# ----------------------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--host", default="127.0.0.1", help="the PC running video_stage_worker"
    )
    ap.add_argument(
        "--port",
        type=int,
        action="append",
        required=True,
        help="one per camera (repeat)",
    )
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1984)
    ap.add_argument("--fps", type=int, default=30, help="publication rate")
    ap.add_argument(
        "--source",
        default="orin_like.h264",
        help="H.264 file; made with ffmpeg if it does not exist",
    )
    ap.add_argument(
        "--frame-id-step",
        type=int,
        default=2,
        help="source frame ids advance by this per unit",
    )
    ap.add_argument(
        "--clock-offset-ms",
        type=float,
        default=5000.0,
        help="this clock's boot-time offset from the PC's",
    )
    ap.add_argument("--clock-drift-ppm", type=float, default=0.0)
    ap.add_argument("--max-queued-mb", type=float, default=16.0)
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 means until stopped",
    )
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(
            f"making {source} ({args.width}x{args.height}@{args.fps}, Jetson-style H.264)",
            flush=True,
        )
        make_stream(source, args.width, args.height, args.fps)
    units = split_access_units(source.read_bytes())
    keyframes = [i for i, (_, key) in enumerate(units) if key]
    if not keyframes:
        sys.exit(f"{source} has no IDR access units")

    clock = OrinClock(int(args.clock_offset_ms * 1e6), args.clock_drift_ppm)
    senders = [
        DirectSender(
            args.host,
            p,
            args.width,
            args.height,
            args.fps,
            clock,
            int(args.max_queued_mb * (1 << 20)),
        )
        for p in args.port
    ]
    period = 1.0 / args.fps
    cursors = [0] * len(senders)
    frame_id = 0
    started = time.monotonic()
    next_tick = started
    next_report = started + 5
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_tick += period
            frame_id += args.frame_id_step
            stamp = clock.now_ns()
            for n, s in enumerate(senders):
                # A keyframe request makes that camera's encoder emit an IDR; the file's next keyframe stands in for it.
                if s.take_keyframe_request():
                    cursors[n] = next(
                        (k for k in keyframes if k >= cursors[n]), keyframes[0]
                    )
                au, key = units[cursors[n] % len(units)]
                cursors[n] = (cursors[n] + 1) % len(units)
                s.submit(au, key, frame_id, stamp)
            if time.monotonic() >= next_report:
                next_report += 5
                print(
                    " | ".join(
                        f"port {s.port}: sent {s.sent} dropped {s.dropped} K {s.keyframe_requests} Q {s.clock_answers}"
                        f" connects {s.connects}"
                        for s in senders
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        for s in senders:
            s.stop()
    for s in senders:
        print(
            f"port {s.port}: sent {s.sent}, dropped {s.dropped}, keyframe requests {s.keyframe_requests}, "
            f"clock answers {s.clock_answers}, connections {s.connects}"
        )


if __name__ == "__main__":
    main()
