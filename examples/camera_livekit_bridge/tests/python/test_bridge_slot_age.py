# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge drops a picture whose notice is older than the ring's wrap time.

A separate publisher process serves one NV12 stream through the real CUDA IPC
publisher, stamping each notice with ``time.monotonic_ns()`` (fresh) or ten
seconds earlier (stale). The bridge publishes to a local livekit-server; no
subscriber is needed, because the check is on the bridge's own stderr and
``[stats]`` line.

Run: python3 test_bridge_slot_age.py -v   (about 30 s; needs a CUDA device).

Environment overrides: LIVEKIT_SERVER_BIN, CAMERA_LIVEKIT_BRIDGE_BIN,
CUPY_PYTHON (interpreter with CuPy for the publisher process). Skipped, never
faked, when a prerequisite is missing.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from test_bridge_pipeline import (
    BRIDGE_BIN,
    RECEIVER_DIR,
    RECEIVER_PY,
    SERVER_BIN,
    _cuda_present,
    _Proc,
    _python_has,
    free_port,
    LIVEKIT_ISSUER,
    SIGNING_KEY,
    make_token,
)

FPS = 30
RING = 6
WIDTH, HEIGHT = 320, 240
STREAM, TRACK = "cam", "cam_track"
STALE_AGE_NS = 10_000_000_000
PUBLISHER_IDENTITY = "bridge-pub"
RUN_S = 8.0
STATS_TIMEOUT_S = 20.0

STATS_RE = re.compile(
    r"\[stats\] " + TRACK + r": (?P<fps>[\d.]+) fps, (?P<frames>\d+) frames, "
    r"(?P<keyframes>\d+) keyframes, (?P<skipped>\d+) skipped, "
    r"(?P<rejected>\d+) rejected, (?P<overruns>\d+) slot overruns, "
    r"(?P<stale>\d+) stale slot reads,"
)


def _cupy_python() -> Path:
    override = os.environ.get("CUPY_PYTHON")
    if override:
        return Path(override)
    if _python_has(RECEIVER_PY, "cupy"):
        return RECEIVER_PY
    return Path(sys.executable)


def _skip_reason() -> str | None:
    if not SERVER_BIN.is_file():
        return f"livekit-server not found ({SERVER_BIN}); set LIVEKIT_SERVER_BIN"
    if not BRIDGE_BIN.is_file():
        return f"camera_livekit_bridge not built ({BRIDGE_BIN}); set CAMERA_LIVEKIT_BRIDGE_BIN"
    if not _cuda_present():
        return "no CUDA device (nvidia-smi -L lists no GPU)"
    if not _python_has(_cupy_python(), "cupy"):
        return f"{_cupy_python()} lacks CuPy; set CUPY_PYTHON"
    return None


def _publisher_main(socket_path: str, mode: str) -> None:
    """Child process: serve one stream and announce frames until terminated."""
    import cupy as cp

    sys.path.insert(0, str(RECEIVER_DIR))
    from cuda_ipc import CudaIpcFramePublisher

    publisher = CudaIpcFramePublisher(socket_path=socket_path, owner_name="slot_age")
    y_bytes = WIDTH * HEIGHT
    layout = {
        "planes": [
            {
                "name": "y",
                "offset": 0,
                "width": WIDTH,
                "height": HEIGHT,
                "pitch": WIDTH,
            },
            {
                "name": "uv",
                "offset": y_bytes,
                "width": WIDTH,
                "height": HEIGHT // 2,
                "pitch": WIDTH,
            },
        ]
    }
    buffers = [
        cp.full((y_bytes * 3 // 2,), 100 + i, dtype=cp.uint8) for i in range(RING)
    ]
    publisher.register_stream(STREAM, WIDTH, HEIGHT, "nv12", 0, buffers, layout)
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
        now = time.monotonic_ns()
        publisher.publish(
            STREAM,
            slot=seq % RING,
            slot_generation=seq // RING + 1,
            seq=seq,
            timestamp_ns=now - STALE_AGE_NS if mode == "stale" else now,
            metadata={"session_id": 1, "stream_name": STREAM, "stream_sequence": seq},
        )
        seq += 1
        time.sleep(1.0 / FPS)
    publisher.stop()


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class BridgeSlotAgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="bridge_slot_age_"))
        cls.addClassCleanup(shutil.rmtree, cls._tmp, ignore_errors=True)
        lk_port = free_port(socket.SOCK_STREAM)
        cls.url = f"ws://127.0.0.1:{lk_port}"
        cls.room = f"slot-age-{uuid.uuid4().hex[:10]}"
        server = _Proc(
            "livekit",
            [
                str(SERVER_BIN),
                "--dev",
                "--keys",
                f"{LIVEKIT_ISSUER}: {SIGNING_KEY}",
                "--bind",
                "127.0.0.1",
                "--node-ip",
                "127.0.0.1",
                "--port",
                str(lk_port),
                "--rtc.tcp_port",
                str(free_port(socket.SOCK_STREAM)),
                "--udp-port",
                str(free_port(socket.SOCK_DGRAM)),
            ],
            cls._tmp,
        )
        cls.addClassCleanup(server.stop)
        server.wait_for("starting LiveKit server", 20)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                socket.create_connection(("127.0.0.1", lk_port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("livekit-server never accepted connections")
        token = cls._tmp / "pub.token"
        token.write_text(make_token(cls.room, PUBLISHER_IDENTITY, True))
        cls.token = token

    def _run(self, mode: str) -> str:
        """Run publisher + bridge for RUN_S; return the bridge's output."""
        sock = self._tmp / f"{mode}.sock"
        publisher = _Proc(
            f"publisher_{mode}",
            [
                str(_cupy_python()),
                str(Path(__file__).resolve()),
                "--publisher",
                str(sock),
                mode,
            ],
            self._tmp,
        )
        self.addCleanup(publisher.stop)
        publisher.wait_for("READY", 60)
        bridge = _Proc(
            f"bridge_{mode}",
            [
                str(BRIDGE_BIN),
                "--livekit-url",
                self.url,
                "--livekit-token-file",
                str(self.token),
                "--socket",
                str(sock),
                "--fps",
                str(FPS),
                "--stream",
                f"{STREAM}={TRACK}",
            ],
            self._tmp,
        )
        self.addCleanup(bridge.stop)
        bridge.wait_for("publishing", 60)
        time.sleep(RUN_S)
        bridge.wait_for("[stats]", STATS_TIMEOUT_S)
        text = bridge.text()
        bridge.stop()
        return text

    def _stats(self, text: str) -> dict[str, int]:
        matches = list(STATS_RE.finditer(text))
        self.assertTrue(matches, f"no [stats] line for {TRACK}:\n{text[-3000:]}")
        last = matches[-1]
        return {k: int(float(v)) for k, v in last.groupdict().items()}

    def test_stale_notice_is_dropped_loudly(self):
        text = self._run("stale")
        self.assertIn("SLOT OVERWRITE RISK", text)
        stats = self._stats(text)
        self.assertGreater(stats["stale"], 0, text[-3000:])
        self.assertEqual(stats["frames"], 0, "a stale picture was published")

    def test_fresh_notice_is_published(self):
        text = self._run("fresh")
        self.assertNotIn("SLOT OVERWRITE RISK", text)
        stats = self._stats(text)
        self.assertEqual(stats["stale"], 0, text[-3000:])
        self.assertGreater(stats["frames"], 0, text[-3000:])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--publisher":
        _publisher_main(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
