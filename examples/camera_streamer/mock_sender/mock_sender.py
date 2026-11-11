#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera-free stand-in for argus_sender: serves AVS1 HEVC over TCP.

Like the real sender it is the TCP server, accepts one receiver at a time,
rebuilds its stream state on every connection and starts each stream at an IDR.
Imagery is a looped test pattern with a burned-in frame counter. All timing
fields come from a synthetic clock (see --clock-offset-ms) and say nothing about
real sensor timing: SOF/TSC and Argus frame timestamps are never marked valid.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("isaaccapture.camera_streamer.mock_sender")

HEADER = struct.Struct("!4sHHIHHQQQQIIQ")  # 64 bytes
STREAM = struct.Struct("!HHBBHIIIIQHHI")  # 40 bytes
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII")  # 72 bytes
CODEC_HEVC = 1
PIXEL_NV12 = 1
STREAM_PRESENT = 0x0001
SAMPLE_PRESENT = 0x0001
SAMPLE_SENDER_TIMING_VALID = 0x0008
BUNDLE_SENDER_CLOCK_VALID = 0x0001  # realtime domain; TAI and TSC flags stay clear
LOOP_SECONDS = 12
GOP_SECONDS = 2

_PARAMETER_SETS_AND_PREFIX = (32, 33, 34, 35, 39)  # VPS, SPS, PPS, AUD, prefix SEI


@dataclass(frozen=True)
class StreamSpec:
    name: str
    width: int
    height: int
    fps: int


def parse_stream_spec(text: str) -> StreamSpec:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+):(\d+)x(\d+)@(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected NAME:WxH@FPS, got {text!r}")
    name, width, height, fps = match.groups()
    spec = StreamSpec(name, int(width), int(height), int(fps))
    if spec.width % 2 or spec.height % 2 or spec.fps <= 0:
        raise argparse.ArgumentTypeError(f"invalid geometry or rate in {text!r}")
    return spec


@dataclass(frozen=True)
class Pause:
    stream: str
    start_s: float
    duration_s: float


def parse_pause(text: str) -> Pause:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)@([0-9.]+)\+([0-9.]+)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected NAME@START+DURATION, got {text!r}")
    return Pause(match[1], float(match[2]), float(match[3]))


# --- HEVC test stream -------------------------------------------------------


def make_stream_file(path: Path, spec: StreamSpec) -> None:
    """Encode a looping pattern with an IDR every GOP_SECONDS, no B-frames, AUDs and
    repeated parameter sets, the shape the Jetson encoder produces."""
    gop = spec.fps * GOP_SECONDS
    label = f"{spec.name} %{{n}}"
    size = max(spec.height // 12, 16)
    draw = (
        f"drawtext=text='{label}':x=16:y=16:fontsize={size}:"
        "fontcolor=white:box=1:boxcolor=black@0.7"
    )
    params = (
        f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:repeat-headers=1:"
        "aud=1:rc-lookahead=0:no-open-gop=1:log-level=error"
    )
    bitrate = max(1_000_000, int(spec.width * spec.height * spec.fps * 0.1))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={spec.width}x{spec.height}:rate={spec.fps}",
        "-vf", draw, "-t", str(LOOP_SECONDS), "-c:v", "libx265", "-preset", "ultrafast",
        "-x265-params", params, "-b:v", str(bitrate), "-pix_fmt", "yuv420p", "-f", "hevc",
        str(path),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


def _nals(data: bytes) -> list[tuple[int, int, int]]:
    """(start-code offset, NAL type, NAL header offset) for every NAL unit."""
    out = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0 or j + 4 >= len(data):
            break
        start = j - 1 if j > 0 and data[j - 1] == 0 else j
        out.append((start, (data[j + 3] >> 1) & 0x3F, j + 3))
        i = j + 3
    return out


def split_access_units(data: bytes) -> list[tuple[bytes, bool]]:
    """Split Annex-B HEVC into (access unit, is IRAP) pairs."""
    nals = _nals(data)
    starts = []
    previous_vcl = False
    for position, (start, nal_type, header) in enumerate(nals):
        vcl = nal_type < 32
        if position == 0:
            begin = True
        elif vcl:
            begin = previous_vcl and bool(data[header + 2] & 0x80)
        else:
            begin = previous_vcl and nal_type in _PARAMETER_SETS_AND_PREFIX
        if begin:
            starts.append(start)
        previous_vcl = vcl
    starts.append(len(data))
    units = []
    for a, b in zip(starts, starts[1:]):
        au = data[a:b]
        units.append((au, any(16 <= t <= 21 for _, t, _ in _nals(au))))
    return units


# --- AVS1 records -----------------------------------------------------------


class MockClock:
    """Synthetic sender clock: this host's wall clock plus an offset and a drift."""

    def __init__(self, offset_ns: int, drift_ppm: float) -> None:
        self._t0 = time.monotonic_ns()
        self._offset = offset_ns
        self._drift = drift_ppm * 1e-6

    def now_ns(self) -> int:
        elapsed = time.monotonic_ns() - self._t0
        return time.time_ns() + self._offset + int(elapsed * self._drift)


@dataclass
class _Cursor:
    spec: StreamSpec
    units: list[tuple[bytes, bool]]
    position: int = 0
    frame_number: int = 0
    stream_sequence: int = 0
    need_idr: bool = True
    paused: bool = False
    base_phase: int = field(default=0)

    def next_unit(self) -> bytes | None:
        """The next access unit, or None while waiting for an IDR."""
        while True:
            au, idr = self.units[self.position]
            self.position = (self.position + 1) % len(self.units)
            if self.need_idr and not idr:
                continue
            self.need_idr = False
            return au


def build_record(
    cursors: list[_Cursor],
    payloads: list[bytes | None],
    bundle_sequence: int,
    session_id: int,
    ready_ns: int,
    enqueue_ns: int,
) -> bytes:
    """One length-prefixed AVS1 record: every stream has a descriptor, present or not."""
    streams, samples, body = [], [], b""
    for index, (cursor, payload) in enumerate(zip(cursors, payloads)):
        present = payload is not None
        streams.append(
            STREAM.pack(
                index, STREAM_PRESENT if present else 0, CODEC_HEVC, PIXEL_NV12, 0,
                cursor.spec.width, cursor.spec.height, cursor.spec.fps, 1,
                cursor.stream_sequence if present else 0, index, 1, 0,
            )
        )  # fmt: skip
        if present:
            flags = SAMPLE_PRESENT | SAMPLE_SENDER_TIMING_VALID
            samples.append(
                SAMPLE.pack(
                    index, 0, 0, flags, 0, index, 0, cursor.frame_number, 0, 0,
                    ready_ns, ready_ns, ready_ns, len(body), len(payload),
                )
            )  # fmt: skip
            body += payload
        else:
            samples.append(
                SAMPLE.pack(index, 0, 0, 0, 0, index, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )
    descriptors = b"".join(streams) + b"".join(samples)
    header = HEADER.pack(
        b"AVS1", 1, BUNDLE_SENDER_CLOCK_VALID, HEADER.size, len(streams), len(samples),
        bundle_sequence, session_id, ready_ns, enqueue_ns, len(descriptors), len(body), 0,
    )  # fmt: skip
    record = header + descriptors + body
    return struct.pack("!I", len(record)) + record


# --- server -----------------------------------------------------------------


class MockSender:
    def __init__(
        self,
        specs: list[StreamSpec],
        sources: dict[StreamSpec, Path],
        clock: MockClock,
        pauses: list[Pause],
        disconnects_s: list[float],
    ) -> None:
        self.specs = specs
        self.clock = clock
        self.pauses = pauses
        self.disconnects_s = sorted(disconnects_s)
        self._units = {
            spec: split_access_units(sources[spec].read_bytes()) for spec in specs
        }
        self.base_fps = max(spec.fps for spec in specs)
        self.stop_event = threading.Event()
        self.bundles_sent = 0

    def serve(self, listener: socket.socket, duration_s: float = 0.0) -> None:
        """Accept receivers one at a time until stopped or duration_s elapses."""
        listener.settimeout(0.5)
        started = time.monotonic()
        while not self.stop_event.is_set():
            if duration_s and time.monotonic() - started >= duration_s:
                return
            try:
                conn, peer = listener.accept()
            except socket.timeout:
                continue
            logger.info("receiver connected from %s:%d", *peer[:2])
            try:
                self._stream_to(conn)
            except OSError as exc:
                logger.info("receiver connection ended: %s", exc)
            finally:
                conn.close()

    def _stream_to(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        cursors = [_Cursor(spec, self._units[spec]) for spec in self.specs]
        session_id = random.getrandbits(62) | 1
        disconnects = list(self.disconnects_s)
        start = time.monotonic()
        tick = 0
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - start
            if disconnects and elapsed >= disconnects[0]:
                logger.info("scripted disconnect at %.1fs", disconnects.pop(0))
                return
            ready_ns = self.clock.now_ns()
            payloads: list[bytes | None] = []
            for cursor in cursors:
                paused = any(
                    p.stream == cursor.spec.name
                    and p.start_s <= elapsed < p.start_s + p.duration_s
                    for p in self.pauses
                )
                if paused and not cursor.paused:
                    logger.info("pausing %s", cursor.spec.name)
                if cursor.paused and not paused:
                    logger.info("resuming %s at next IDR", cursor.spec.name)
                    cursor.need_idr = True
                cursor.paused = paused
                due = (tick + 1) * cursor.spec.fps // self.base_fps > (
                    tick * cursor.spec.fps // self.base_fps
                )
                payload = None
                if due and not paused:
                    payload = cursor.next_unit()
                    if payload is not None:
                        cursor.stream_sequence += 1
                        cursor.frame_number += 1
                payloads.append(payload)
            tick += 1
            record = build_record(
                cursors, payloads, tick, session_id, ready_ns, self.clock.now_ns()
            )
            conn.sendall(record)
            self.bundles_sent += 1
            deadline = start + tick / self.base_fps
            if self.stop_event.wait(max(0.0, deadline - time.monotonic())):
                return


def ensure_sources(specs: list[StreamSpec], cache_dir: Path) -> dict[StreamSpec, Path]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg with libx265 and drawtext is required to build the test streams"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for spec in specs:
        path = cache_dir / f"{spec.name}_{spec.width}x{spec.height}_{spec.fps}.hevc"
        if not path.exists():
            logger.info("encoding %s", path.name)
            make_stream_file(path, spec)
        sources[spec] = path
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--stream", type=parse_stream_spec, action="append", metavar="NAME:WxH@FPS",
        help="mono HEVC stream, in receiver config order; repeat. "
        "Default: head:2560x1984@30 wrist1:640x480@30 wrist2:640x480@30",
    )  # fmt: skip
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "camera_streamer_mock_sender",
    )
    parser.add_argument("--clock-offset-ms", type=float, default=5000.0)
    parser.add_argument("--clock-drift-ppm", type=float, default=0.0)
    parser.add_argument(
        "--pause", type=parse_pause, action="append", default=[], metavar="NAME@START+SECONDS",
        help="stop one stream for a while; it resumes at an IDR",
    )  # fmt: skip
    parser.add_argument(
        "--disconnect-at", type=float, action="append", default=[], metavar="SECONDS",
        help="close the receiver connection this long after it connects",
    )  # fmt: skip
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    specs = args.stream or [
        StreamSpec("head", 2560, 1984, 30),
        StreamSpec("wrist1", 640, 480, 30),
        StreamSpec("wrist2", 640, 480, 30),
    ]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        parser.error("stream names must be unique")
    for pause in args.pause:
        if pause.stream not in names:
            parser.error(f"--pause names unknown stream {pause.stream!r}")
    sender = MockSender(
        specs,
        ensure_sources(specs, args.cache_dir),
        MockClock(int(args.clock_offset_ms * 1e6), args.clock_drift_ppm),
        args.pause,
        args.disconnect_at,
    )
    listener = socket.create_server((args.bind, args.port), reuse_port=False)
    logger.info(
        "mock sender listening on %s:%d (synthetic clock)", args.bind, args.port
    )
    try:
        sender.serve(listener, args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
