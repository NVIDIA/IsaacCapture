#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mock_sender -> camera_receiver (NVDEC) -> CUDA IPC, read with the consumer's client.

Run with the receiver venv's interpreter: ``camera_receiver/.venv/bin/python -m
unittest tests.test_mock_to_ipc_pipeline``.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mock_sender"))
sys.path.insert(0, str(ROOT / "camera_receiver"))
sys.path.insert(0, str(ROOT / "cuda_ipc_consumer"))

try:
    import cupy as cp
    import yaml

    _HAVE_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # missing module or no driver
    cp = None
    yaml = None
    _HAVE_CUDA = False

_CODEC = list((ROOT / "camera_receiver" / "codec").glob("_camera_receiver_codec*.so"))
if _HAVE_CUDA:
    from consumer import CudaIpcClient  # noqa: E402

import mock_sender as ms  # noqa: E402

STREAMS = [("head", 640, 480), ("wrist", 320, 240)]
SOF_LAG_NS = 20_000_000
FRAMES = 60
SESSION_TIMEOUT_S = 30.0
# NVDEC returns no picture for the first access unit of a session, so the first
# published stream_sequence is 2.
_FIRST_SEQ = (1, 2)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _luma_stats(slot, width: int, height: int) -> tuple[float, float]:
    """Mean and spread of the NV12 luma plane, computed on the GPU."""
    luma = slot.array.reshape(-1)[: width * height].astype(cp.float32)
    return float(luma.mean()), float(luma.std())


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg with libx265 is required")
@unittest.skipUnless(_HAVE_CUDA, "a CUDA device with cupy and pyyaml is required")
@unittest.skipUnless(_CODEC, "the native camera_receiver codec module is not built")
class MockToIpcPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._dir.name)
        specs = [ms.StreamSpec(n, w, h, 30) for n, w, h in STREAMS]
        ms.ensure_sources(specs, cls.tmp / "cache")

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def setUp(self):
        self.procs: list[tuple[str, subprocess.Popen, Path]] = []
        self.client = None
        self.addCleanup(self._cleanup)

    def _spawn(self, name: str, argv: list[str]) -> None:
        log = self.tmp / f"{self._testMethodName}.{name}.log"
        with open(log, "wb") as out:
            proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT)
        self.procs.append((name, proc, log))

    def _cleanup(self):
        if self.client is not None:
            self.client.close()
        for _, proc, _ in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _logs(self) -> str:
        parts = []
        for name, _, log in self.procs:
            tail = log.read_text(errors="replace").splitlines()[-25:]
            parts.append(f"--- {name} ---\n" + "\n".join(tail))
        return "\n".join(parts)

    def _start_pipeline(self, *mock_args: str) -> None:
        port = _free_port()
        sock = str(self.tmp / "ipc.sock")
        config = {
            "version": 1,
            "receiver": {
                "host": "127.0.0.1",
                "port": port,
                "gpu_id": 0,
                "stats_log_interval_s": 1,
                "receive_fps_warning_threshold": 0,
            },
            "ipc": {
                "socket_path": sock,
                "owner_name": "mock_to_ipc_test",
                "queue_size": 1024,
            },
            "streams": [
                {
                    "index": i,
                    "name": n,
                    "type": "mono",
                    "width": w,
                    "height": h,
                    "output_format": "nv12",
                    "ring_depth": 16,
                    "ipc": {"mono": n},
                }
                for i, (n, w, h) in enumerate(STREAMS)
            ],
        }
        cfg_path = self.tmp / f"{self._testMethodName}.yaml"
        cfg_path.write_text(yaml.safe_dump(config))
        mock = [
            sys.executable,
            str(ROOT / "mock_sender" / "mock_sender.py"),
            "--port",
            str(port),
            "--cache-dir",
            str(self.tmp / "cache"),
            "--duration",
            "60",
        ]
        for n, w, h in STREAMS:
            mock += ["--stream", f"{n}:{w}x{h}@30"]
        self._spawn("mock_sender", mock + list(mock_args))
        self._spawn(
            "camera_receiver",
            [
                sys.executable,
                str(ROOT / "camera_receiver" / "camera_receiver.py"),
                "--config",
                str(cfg_path),
            ],
        )
        deadline = time.monotonic() + SESSION_TIMEOUT_S
        while not Path(sock).exists():
            if time.monotonic() > deadline or any(p.poll() for _, p, _ in self.procs):
                self.fail(f"IPC socket never appeared\n{self._logs()}")
            time.sleep(0.1)
        self.client = CudaIpcClient(sock)
        self.client.connect()

    def _collect(self, until, timeout_s: float = SESSION_TIMEOUT_S):
        """Frames as (stream, metadata, luma stats) until ``until(frames)`` is true."""
        dims = {n: (w, h) for n, w, h in STREAMS}
        frames: list[tuple[str, dict, tuple[float, float]]] = []
        deadline = time.monotonic() + timeout_s
        while not until(frames):
            if time.monotonic() > deadline:
                self.fail(f"timed out with {len(frames)} frames\n{self._logs()}")
            messages = self.client.read_messages()
            if messages is None:
                self.fail(f"IPC socket closed\n{self._logs()}")
            for msg in messages:
                if msg.get("type") != "frame":
                    continue
                name = msg["stream"]
                slot = self.client.slots[(name, int(msg["slot"]))]
                frames.append((name, msg["metadata"], _luma_stats(slot, *dims[name])))
        return frames

    def test_frames_and_metadata(self):
        self._start_pipeline()
        info = self.client.stream_info
        for n, w, h in STREAMS:
            self.assertEqual((info[n]["width"], info[n]["height"]), (w, h))
            self.assertEqual(info[n]["pixel_format"], "nv12")

        def enough(frames):
            return all(sum(f[0] == n for f in frames) >= FRAMES for n, _, _ in STREAMS)

        frames = self._collect(enough)
        for n, w, h in STREAMS:
            mine = [(m, s) for name, m, s in frames if name == n][:FRAMES]
            with self.subTest(stream=n):
                seqs = [m["stream_sequence"] for m, _ in mine]
                self.assertIn(seqs[0], _FIRST_SEQ, self._logs())
                self.assertEqual(seqs, list(range(seqs[0], seqs[0] + len(seqs))))
                numbers = [m["frame_number"] for m, _ in mine]
                self.assertEqual(numbers, sorted(set(numbers)))
                self.assertEqual(len({m["session_id"] for m, _ in mine}), 1)
                self.assertNotEqual(mine[0][0]["session_id"], 0)
                for m, _ in mine:
                    self.assertEqual((m["width"], m["height"]), (w, h))
                    self.assertEqual(m["ipc_pixel_format"], "nv12")
                    self.assertEqual(m["sender_wall_clock"], "realtime")
                    self.assertIsNotNone(m["sensor_sof_ns"])
                    self.assertEqual(
                        m["sensor_sof_ns"],
                        m["sender_capture_ready_ns"] - SOF_LAG_NS,
                    )
                self.assertTrue(all(s[1] > 1.0 for _, s in mine), "constant luma")
                means = [s[0] for _, s in mine]
                self.assertTrue(
                    any(
                        abs(means[i + 10] - means[i]) > 0.01
                        for i in range(len(means) - 10)
                    ),
                    f"pattern does not move: {means}",
                )

    def test_reconnect_starts_new_session(self):
        self._start_pipeline("--disconnect-at", "3")

        def changed(frames):
            return len(
                {m["session_id"] for n, m, _ in frames if n == "head"}
            ) >= 2 and (
                sum(
                    m["session_id"] == frames[-1][1]["session_id"] for _, m, _ in frames
                )
                >= 10
            )

        frames = [f for f in self._collect(changed, timeout_s=45.0) if f[0] == "head"]
        sessions = list(dict.fromkeys(m["session_id"] for _, m, _ in frames))
        first = [m for _, m, _ in frames if m["session_id"] == sessions[0]]
        second = [(m, s) for _, m, s in frames if m["session_id"] == sessions[1]]
        self.assertNotEqual(sessions[0], sessions[1])
        self.assertGreater(len(first), 30, "frames before the disconnect")
        # Decoders restart at the sender's IDR, so the counters restart and the
        # first published picture is real.
        seqs = [m["stream_sequence"] for m, _ in second[:10]]
        self.assertIn(seqs[0], _FIRST_SEQ, self._logs())
        self.assertEqual(seqs, list(range(seqs[0], seqs[0] + len(seqs))))
        self.assertGreater(second[0][1][1], 1.0, "first frame after reconnect blank")


if __name__ == "__main__":
    unittest.main()
