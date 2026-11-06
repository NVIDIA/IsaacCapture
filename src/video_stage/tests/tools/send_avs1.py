#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for the Orin sender: sends an H.264 Annex-B file as AVS1 records over TCP, paced at a frame rate.

The file must delimit access units with AUD NAL units (x264: `-x264opts aud=1`). Each record carries the frame
number and a CLOCK_MONOTONIC stamp, as the Orin sender does. Exits after one pass, or loops with --loops.
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time

HEADER = struct.Struct("!4sHHIHHQQQQIIQ")
STREAM = struct.Struct("!HHBBHIIIIQHHI")
SAMPLE = struct.Struct("!HBBHHIIQQQQQQII")
CODEC_H264 = 2


def split_access_units(data: bytes) -> list[bytes]:
    """Split Annex-B data at AUD NAL units (type 9)."""
    starts = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= len(data):
            break
        if data[j + 3] & 0x1F == 9:
            starts.append(j - 1 if j > 0 and data[j - 1] == 0 else j)
        i = j + 3
    if not starts:
        raise ValueError("no AUD NAL units found; encode with -x264opts aud=1")
    starts.append(len(data))
    return [data[a:b] for a, b in zip(starts, starts[1:])]


def record(au: bytes, sequence: int, frame: int, stamp_ns: int, width: int, height: int, fps: int) -> bytes:
    descriptor_bytes = STREAM.size + SAMPLE.size
    header = HEADER.pack(b"AVS1", 1, 0, HEADER.size, 1, 1, sequence, 1, 0, 0, descriptor_bytes, len(au), 0)
    stream = STREAM.pack(0, 1, CODEC_H264, 1, 0, width, height, fps, 1, sequence, 0, 1, 0)
    sample = SAMPLE.pack(0, 0, 0, 0x0005, 0, 0, 0, frame, 0, stamp_ns, 0, 0, 0, 0, len(au))
    body = header + stream + sample + au
    return struct.pack("!I", len(body)) + body


def control_loop(ctrl: socket.socket, offset_ns: int, stats: dict) -> None:
    """Answer each 13-byte 'Q' with a 29-byte 'R' on a clock offset by `offset_ns`; count 'K' keyframe requests."""
    buf = b""
    while True:
        try:
            first = ctrl.recv(4096)
        except OSError:
            return
        if not first:
            return
        t2 = time.monotonic_ns() + offset_ns
        buf += first
        while buf:
            if buf[0:1] == b"K":
                stats["K"] += 1
                buf = buf[1:]
            elif buf[0:1] == b"Q":
                if len(buf) < 13:
                    break
                _, seq, t1 = struct.unpack("<cIq", buf[:13])
                buf = buf[13:]
                t3 = time.monotonic_ns() + offset_ns
                try:
                    ctrl.sendall(struct.pack("<cIqqq", b"R", seq, t1, t2, t3))
                except OSError:
                    return
                stats["Q"] += 1
                t2 = time.monotonic_ns() + offset_ns
            else:
                buf = buf[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", help="H.264 Annex-B file with AUDs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--clock-offset-ms", type=float, default=5000.0, help="simulated boot-time offset of this clock")
    args = ap.parse_args()

    aus = split_access_units(open(args.file, "rb").read())
    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(b"DATA")
    ctrl = socket.create_connection((args.host, args.port))
    ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ctrl.sendall(b"CTRL")
    stats = {"Q": 0, "K": 0}
    reader = threading.Thread(target=control_loop, args=(ctrl, int(args.clock_offset_ms * 1e6), stats), daemon=True)
    reader.start()
    period = 1.0 / args.fps
    frame = 0
    next_send = time.monotonic()
    for _ in range(args.loops):
        for au in aus:
            delay = next_send - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            frame += 1
            sock.sendall(record(au, frame, frame, time.monotonic_ns(), args.width, args.height, args.fps))
            next_send += period
    sock.shutdown(socket.SHUT_WR)
    sock.settimeout(30)
    try:
        while sock.recv(4096):
            pass
    except OSError:
        pass
    sock.close()
    ctrl.close()
    print(f"sent {frame} access units; answered {stats['Q']} clock requests; {stats['K']} keyframe requests")


if __name__ == "__main__":
    main()
