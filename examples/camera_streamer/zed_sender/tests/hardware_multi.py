#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in multi-camera ZED test: one sender, per-camera AVS1 bundles over loopback.

Reports each configured stream, including ones that never delivered a frame.
"""

import argparse
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time

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


def stream_name(stream):
    return f"cam{stream.samples[0].device_index}"


def run(process, args, directory):
    streams = {}
    previous_bundle, session_id, layout = 0, None, None
    with connect(process, args.port) as sock:
        for _ in range(args.frames):
            (size,) = struct.unpack("!I", receive(sock, 4))
            assert 0 < size < 32 * 1024 * 1024, size
            bundle = parse_avs1_record(receive(sock, size))
            now, wall = time.monotonic(), time.time_ns()
            session_id = session_id or bundle.session_id
            assert bundle.session_id == session_id
            assert bundle.sequence == previous_bundle + 1, "bundle sequence gap"
            previous_bundle = bundle.sequence
            assert bundle.sender_tcp_enqueue_ns >= bundle.sender_capture_ready_ns > 0
            # Every bundle describes all cameras the same way and carries exactly one.
            described = [
                (stream_name(s), s.stereo, s.width, s.height, s.fps_numerator)
                for s in bundle.streams
            ]
            layout = layout or described
            assert described == layout, "stream layout changed within a session"
            present = [s for s in bundle.streams if s.present]
            assert len(present) == 1, "each bundle carries one camera"
            stream = present[0]
            state = streams.setdefault(
                stream_name(stream),
                dict(arrivals=[], ages=[], numbers=[], bytes=0, sequence=0, files=[]),
            )
            assert stream.sequence == state["sequence"] + 1, "stream sequence gap"
            state["sequence"] = stream.sequence
            assert len(stream.samples) == (2 if stream.stereo else 1)
            roles = (1, 2) if stream.stereo else (0,)
            if not state["files"]:
                suffixes = ("left", "right") if stream.stereo else ("mono",)
                state["files"] = [
                    (directory / f"{stream_name(stream)}-{suffix}.h265").open("wb")
                    for suffix in suffixes
                ]
            for role, sample, target in zip(roles, stream.samples, state["files"]):
                assert sample.role == role and sample.flags == 9
                assert (
                    0
                    < sample.sender_capture_ready_ns
                    <= sample.sender_process_submit_ns
                    <= sample.sender_process_done_ns
                )
                assert bytes(sample.payload[:4]).startswith((b"\0\0\1", b"\0\0\0\1"))
                target.write(sample.payload)
            state["bytes"] += size + 4 if state["arrivals"] else 0
            state["arrivals"].append(now)
            state["ages"].append((wall - bundle.sender_capture_ready_ns) / 1e6)
            state["numbers"].append(stream.samples[0].frame_number)
    report = {}
    for name, stereo, width, height, fps in layout:
        state = streams.get(name)
        entry = dict(stereo=stereo, width=width, height=height, target_fps=fps)
        if not state or len(state["arrivals"]) < 2:
            entry.update(
                frames=len(state["arrivals"]) if state else 0, delivering=False
            )
        else:
            for handle in state["files"]:
                handle.close()
            span = state["arrivals"][-1] - state["arrivals"][0]
            ages = sorted(state["ages"])
            entry.update(
                frames=len(state["arrivals"]),
                delivering=True,
                fps=(len(state["arrivals"]) - 1) / span,
                wire_mbps=state["bytes"] * 8 / span / 1e6,
                source_sequence_gaps=sum(
                    ((b - a) % 2**32) - 1
                    for a, b in zip(state["numbers"], state["numbers"][1:])
                ),
                capture_ready_to_receive_ms=dict(
                    mean=sum(ages) / len(ages),
                    p95=ages[int((len(ages) - 1) * 0.95)],
                    max=ages[-1],
                ),
            )
        report[name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sender", type=Path, default=ROOT / "zed_sender/build/zed_sender"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "zed_sender/configs/zed_multi.yaml"
    )
    parser.add_argument("--port", type=int, default=55030)
    parser.add_argument(
        "--frames", type=int, default=300, help="total bundles across all cameras"
    )
    parser.add_argument(
        "--require",
        default="",
        help="comma-separated stream names that must deliver frames, e.g. cam0,cam2",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="zed-multi-"))
    directory.mkdir(parents=True, exist_ok=True)
    summary = dict(ok=False, config=str(args.config), output_dir=str(directory))
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
            summary["streams"] = run(process, args, directory)
            required = [name for name in args.require.split(",") if name]
            missing = [
                name
                for name in required
                if not summary["streams"].get(name, {}).get("delivering")
            ]
            assert not missing, f"required streams did not deliver: {missing}"
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
