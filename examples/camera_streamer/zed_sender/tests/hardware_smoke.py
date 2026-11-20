#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in live camera test: AVS1 interoperability, paired eyes and reconnects."""

import argparse
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "camera_receiver"))
# The sibling receiver is a standalone source tree, so its path must be set before importing.
from avs_wire import parse_avs1_record  # noqa: E402


def receive(sock, count):
    result = bytearray()
    while len(result) < count:
        chunk = sock.recv(count - len(result))
        if not chunk:
            raise RuntimeError("sender closed the connection")
        result.extend(chunk)
    return result


def connect(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sender exited: {process.returncode}")
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=0.3)
            sock.settimeout(10)
            return sock
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("sender did not listen within 10 seconds")


def session(process, args, directory, expected, index):
    arrivals, ages, frame_numbers = [], [], []
    byte_count, previous, session_id = 0, 0, None
    with (
        connect(process, args.port) as sock,
        (directory / f"{index}-left.h265").open("wb") as left,
        (directory / f"{index}-right.h265").open("wb") as right,
    ):
        for i in range(args.frames):
            (size,) = struct.unpack("!I", receive(sock, 4))
            assert 0 < size < 32 * 1024 * 1024, size
            bundle = parse_avs1_record(receive(sock, size))
            arrivals.append(time.monotonic())
            ages.append((time.time_ns() - bundle.sender_capture_ready_ns) / 1e6)
            byte_count += size + 4 if i else 0
            if session_id is None:
                session_id = bundle.session_id
            assert bundle.session_id == session_id
            assert bundle.sequence > previous
            previous = bundle.sequence
            assert bundle.flags == 1 and bundle.sender_clock_domain == "realtime"
            assert bundle.sender_capture_ready_tsc_ns is None
            assert bundle.sender_tcp_enqueue_ns >= bundle.sender_capture_ready_ns > 0
            assert len(bundle.streams) == 1
            stream = bundle.streams[0]
            assert stream.index == 0 and stream.present and stream.stereo
            assert stream.codec == 1 and stream.pixel_format == 1
            assert (
                stream.width,
                stream.height,
                stream.fps_numerator,
                stream.fps_denominator,
                stream.rotation,
            ) == expected
            assert len(stream.samples) == 2
            assert stream.samples[0].frame_number == stream.samples[1].frame_number
            frame_numbers.append(stream.samples[0].frame_number)
            for role, sample, target in zip((1, 2), stream.samples, (left, right)):
                assert sample.role == role and sample.flags == 9
                assert sample.sof_tsc_ns is None and sample.frame_timestamp_ns is None
                assert (
                    0
                    < sample.sender_capture_ready_ns
                    <= sample.sender_process_submit_ns
                    <= sample.sender_process_done_ns
                )
                assert bytes(sample.payload[:4]).startswith((b"\0\0\1", b"\0\0\0\1"))
                target.write(sample.payload)
    span = arrivals[-1] - arrivals[0]
    return dict(
        session_id=session_id,
        frames=args.frames,
        fps=(args.frames - 1) / span,
        wire_mbps=byte_count * 8 / span / 1e6,
        source_sequence_gaps=sum(
            ((b - a) % 2**32) - 1 for a, b in zip(frame_numbers, frame_numbers[1:])
        ),
        capture_ready_to_receive_ms=dict(
            mean=sum(ages) / len(ages),
            p95=sorted(ages)[int((len(ages) - 1) * 0.95)],
            max=max(ages),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sender", type=Path, default=ROOT / "zed_sender/build/zed_sender"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "zed_sender/configs/zed.yaml"
    )
    parser.add_argument("--port", type=int, default=55010)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.frames < 2 or args.sessions < 1:
        parser.error("need at least two frames and one session")
    c = yaml.safe_load(args.config.read_text())["output"]
    expected = c["width"], c["height"], c["fps"], 1, c.get("rotation", 0)
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="zed-smoke-"))
    directory.mkdir(parents=True, exist_ok=True)
    summary = dict(
        ok=False, config=str(args.config), output_dir=str(directory), sessions=[]
    )
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    with (directory / "sender.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(args.sender.resolve()),
                "--config",
                str(args.config.resolve()),
                "--port",
                str(args.port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for index in range(args.sessions):
                summary["sessions"].append(
                    session(process, args, directory, expected, index)
                )
            assert len({s["session_id"] for s in summary["sessions"]}) == args.sessions
            summary["ok"] = True
        except Exception as error:
            summary["error"] = repr(error)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    rendered = json.dumps(summary, indent=2)
    (directory / "summary.json").write_text(rendered + "\n")
    print(rendered)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
