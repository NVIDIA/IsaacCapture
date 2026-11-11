#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""camera_ipc_probe must read GPU frames published by the real CUDA IPC publisher.

A separate publisher process (CUDA IPC handles cannot be opened by the process
that exported them) serves two NV12 streams through ``CudaIpcFramePublisher``;
the probe binary is run against its socket and its stdout is checked.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
_EXAMPLES = _HERE.parents[3]
_RECEIVER_DIR = _EXAMPLES / "camera_streamer" / "camera_receiver"
_DEFAULT_PROBE = _HERE.parents[2] / "build" / "probe" / "camera_ipc_probe"

# name -> (width, height). Each NV12 buffer must hold at least the 4096 bytes
# the probe checksums.
_STREAMS = {"head": (64, 48), "wrist": (96, 48)}
_RING = 3
_SESSION_ID = 424242
_SOF_BASE_NS = 1_000_000_000
_SOF_STEP_NS = 33_000_000
_CHECKSUM_BYTES = 4096
_WAIT_S = 20.0

_FRAME_RE = re.compile(
    r"^(?P<name>\S+) slot=(?P<slot>\d+) gen=(?P<gen>\d+) "
    r"stream_sequence=(?P<seq>-?\d+) session=(?P<session>\d+) "
    r"sensor_sof_ns=(?P<sof>\S+) checksum=(?P<sum>\d+)$"
)
_STREAM_RE = re.compile(
    r"^stream (?P<name>\S+) (?P<w>\d+)x(?P<h>\d+) (?P<fmt>\S+) "
    r"slots=(?P<slots>\d+) layout=(?P<layout>.*)$"
)


def _slot_value(slot: int) -> int:
    return 10 * slot + 3


def _expected_checksum(slot: int) -> int:
    return _CHECKSUM_BYTES * _slot_value(slot)


def _probe_path() -> Path:
    return Path(os.environ.get("CAMERA_IPC_PROBE", str(_DEFAULT_PROBE)))


def _skip_reason() -> str | None:
    probe = _probe_path()
    if not probe.is_file() or not os.access(probe, os.X_OK):
        return f"probe binary not found or not executable: {probe}"
    try:
        import cupy as cp
    except ImportError:
        return "CuPy is not importable"
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return "no CUDA device"
    except Exception as exc:  # CUDA runtime errors vary by driver state
        return f"CUDA unavailable: {exc}"
    return None


def _publisher_main(socket_path: str) -> None:
    """Child process: serve the streams and publish frames until terminated."""
    import cupy as cp

    sys.path.insert(0, str(_RECEIVER_DIR))
    from cuda_ipc import CudaIpcFramePublisher

    publisher = CudaIpcFramePublisher(socket_path=socket_path, owner_name="test")
    rings = {}
    for name, (width, height) in _STREAMS.items():
        y_bytes = width * height
        layout = {
            "planes": [
                {
                    "name": "y",
                    "offset": 0,
                    "width": width,
                    "height": height,
                    "pitch": width,
                },
                {
                    "name": "uv",
                    "offset": y_bytes,
                    "width": width,
                    "height": height // 2,
                    "pitch": width,
                },
            ]
        }
        buffers = [
            cp.full((y_bytes * 3 // 2,), _slot_value(i), dtype=cp.uint8)
            for i in range(_RING)
        ]
        rings[name] = buffers
        publisher.register_stream(name, width, height, "nv12", 0, buffers, layout)
    cp.cuda.Device(0).synchronize()
    publisher.start()

    stopping = False

    def _stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    print("READY", flush=True)

    seq = 0
    while not stopping:
        for name in _STREAMS:
            sof_ns = _SOF_BASE_NS + seq * _SOF_STEP_NS
            publisher.publish(
                name,
                slot=seq % _RING,
                slot_generation=seq // _RING + 1,
                seq=seq,
                timestamp_ns=sof_ns,
                metadata={
                    "session_id": _SESSION_ID,
                    "stream_name": name,
                    "stream_sequence": seq,
                    "sensor_sof_ns": sof_ns,
                    "sof_tsc_ns": sof_ns - 5,
                    "sof_tsc_valid": True,
                },
            )
        seq += 1
        time.sleep(0.02)
    publisher.stop()


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class IpcClientProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="ipc_probe_test_")
        cls.socket_path = os.path.join(cls._tmp.name, "ipc.sock")
        cls._publisher = subprocess.Popen(
            [sys.executable, str(_HERE), "--publisher", cls.socket_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.addClassCleanup(_terminate, cls._publisher)
        deadline = time.monotonic() + _WAIT_S
        line = ""
        while time.monotonic() < deadline:
            line = cls._publisher.stdout.readline().strip()
            if line == "READY" or cls._publisher.poll() is not None:
                break
        if line != "READY":
            _terminate(cls._publisher)
            err = cls._publisher.stderr.read()
            raise RuntimeError(f"publisher did not become ready: {err}")

    def run_probe(self, *args: str, socket_path: str | None = None, frames: int = 8):
        cmd = [
            str(_probe_path()),
            "--socket",
            socket_path or self.socket_path,
            "--frames",
            str(frames),
            *args,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_WAIT_S, check=False
        )

    def parse(self, stdout: str):
        streams, frames = {}, []
        for line in stdout.splitlines():
            if line.startswith("stream "):
                m = _STREAM_RE.match(line)
                self.assertIsNotNone(m, line)
                streams[m["name"]] = m
            else:
                m = _FRAME_RE.match(line)
                self.assertIsNotNone(m, f"unparseable frame line: {line!r}")
                frames.append(m)
        return streams, frames

    def check_frames(self, frames) -> None:
        for m in frames:
            seq = int(m["seq"])
            self.assertEqual(int(m["slot"]), seq % _RING, m.string)
            self.assertEqual(int(m["gen"]), seq // _RING + 1, m.string)
            self.assertEqual(int(m["session"]), _SESSION_ID, m.string)
            self.assertEqual(int(m["sof"]), _SOF_BASE_NS + seq * _SOF_STEP_NS, m.string)
            self.assertEqual(
                int(m["sum"]), _expected_checksum(int(m["slot"])), m.string
            )

    def test_all_streams(self) -> None:
        result = self.run_probe(frames=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        streams, frames = self.parse(result.stdout)
        self.assertEqual(set(streams), set(_STREAMS))
        for name, (w, h) in _STREAMS.items():
            m = streams[name]
            self.assertEqual((int(m["w"]), int(m["h"])), (w, h))
            self.assertEqual(m["fmt"], "nv12")
            self.assertEqual(int(m["slots"]), _RING)
            self.assertIn('"uv"', m["layout"])
            self.assertIn('"y"', m["layout"])
        self.assertEqual(len(frames), 12)
        self.assertEqual({m["name"] for m in frames}, set(_STREAMS))
        self.check_frames(frames)

    def test_single_stream_subscription(self) -> None:
        result = self.run_probe("--stream", "head", frames=6)
        self.assertEqual(result.returncode, 0, result.stderr)
        streams, frames = self.parse(result.stdout)
        self.assertEqual(set(streams), {"head"})
        self.assertEqual(len(frames), 6)
        self.assertEqual({m["name"] for m in frames}, {"head"})
        self.check_frames(frames)

    def test_unknown_stream_fails_clearly(self) -> None:
        result = self.run_probe("--stream", "nonexistent")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonexistent", result.stderr)
        self.assertNotRegex(result.stdout, _FRAME_RE.pattern)

    def test_missing_socket_fails_clearly(self) -> None:
        missing = os.path.join(self._tmp.name, "no_such.sock")
        result = self.run_probe(socket_path=missing)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(missing, result.stderr)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--publisher":
        _publisher_main(sys.argv[2])
    else:
        unittest.main()
