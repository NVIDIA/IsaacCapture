# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge encodes only pictures whose slot write generation is unchanged.

A separate publisher process serves one NV12 stream through the real CUDA IPC
publisher. Each notice carries the generation returned by ``begin_write``; the
"stale" notices are announced after the same slot's marker has advanced again,
which is what a rewrite that began after the announcement looks like. The
bridge publishes to a local livekit-server; no subscriber is needed, because
the checks are on the bridge's own stderr and ``[stats]`` line. A small relay
between the bridge and the real publisher covers a publisher that lacks the
feature and a damaged marker descriptor.

Run: python3 test_bridge_slot_marker.py -v   (about 60 s; needs a CUDA device).

Environment overrides: LIVEKIT_SERVER_BIN, CAMERA_LIVEKIT_BRIDGE_BIN,
CUPY_PYTHON (interpreter with CuPy for the publisher process). Skipped, never
faked, when a prerequisite is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
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
    make_token,
)

FPS = 30
RING = 6
WIDTH, HEIGHT = 320, 240
STREAM, TRACK = "cam", "cam_track"
PUBLISHER_IDENTITY = "bridge-pub"
RUN_S = 6.0
STATS_TIMEOUT_S = 20.0
EXIT_TIMEOUT_S = 30.0

STATS_RE = re.compile(
    r"\[stats\] " + TRACK + r": (?P<fps>[\d.]+) fps, (?P<frames>\d+) frames, "
    r"(?P<keyframes>\d+) keyframes, (?P<skipped>\d+) skipped, "
    r"(?P<rejected>\d+) rejected, (?P<overruns>\d+) slot overruns,"
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
    """Child process: serve one stream and announce frames until terminated.

    ``valid`` announces the generation of the write just made. ``stale``
    advances the slot's marker a second time before announcing, so every
    notice names a generation the slot has already left. ``mixed`` alternates.
    """
    import cupy as cp

    sys.path.insert(0, str(RECEIVER_DIR))
    from cuda_ipc import CudaIpcFramePublisher

    publisher = CudaIpcFramePublisher(socket_path=socket_path, owner_name="slot_marker")
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
        slot = seq % RING
        generation = publisher.begin_write(STREAM, slot)
        if mode == "stale" or (mode == "mixed" and seq % 2 == 1):
            publisher.begin_write(STREAM, slot)
        publisher.publish(
            STREAM,
            slot=slot,
            slot_generation=generation,
            seq=seq,
            timestamp_ns=time.monotonic_ns(),
            metadata={"session_id": 1, "stream_name": STREAM, "stream_sequence": seq},
        )
        seq += 1
        time.sleep(1.0 / FPS)
    publisher.stop()


class _Relay:
    """Forwards one bridge connection to a real publisher, editing its replies.

    The handshake is ``hello``, ``subscribe``, ``subscribed``; ``edit`` is
    applied to the parsed ``subscribed`` message and ``edit_hello`` to the
    ``hello``. Either may be None to forward unchanged.
    """

    def __init__(self, path: Path, upstream: Path, edit_hello=None, edit=None):
        self.path = path
        self._upstream = upstream
        self._edit_hello = edit_hello
        self._edit = edit
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @staticmethod
    def _line(sock: socket.socket) -> bytes:
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise ConnectionError("peer closed")
            data += chunk
        return data

    def _serve(self) -> None:
        try:
            down, _ = self._server.accept()
            up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            up.connect(str(self._upstream))
            with down, up:
                hello = json.loads(self._line(up))
                if self._edit_hello:
                    self._edit_hello(hello)
                down.sendall(json.dumps(hello).encode() + b"\n")
                up.sendall(self._line(down))
                subscribed = json.loads(self._line(up))
                if self._edit:
                    self._edit(subscribed)
                down.sendall(json.dumps(subscribed).encode() + b"\n")
                time.sleep(2.0)
        except (OSError, ConnectionError, ValueError):
            pass

    def close(self) -> None:
        self._server.close()


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class BridgeSlotMarkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="bridge_slot_marker_"))
        cls.addClassCleanup(shutil.rmtree, cls._tmp, ignore_errors=True)
        lk_port = free_port(socket.SOCK_STREAM)
        cls.url = f"ws://127.0.0.1:{lk_port}"
        cls.room = f"slot-marker-{uuid.uuid4().hex[:10]}"
        server = _Proc(
            "livekit",
            [
                str(SERVER_BIN),
                "--dev",
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

    def _start_publisher(self, mode: str) -> Path:
        sock = self._tmp / f"{mode}_{uuid.uuid4().hex[:6]}.sock"
        publisher = _Proc(
            f"publisher_{mode}",
            [str(_cupy_python()), str(Path(__file__).resolve()), "--publisher"]
            + [str(sock), mode],
            self._tmp,
        )
        self.addCleanup(publisher.stop)
        publisher.wait_for("READY", 60)
        return sock

    def _bridge(self, name: str, sock: Path) -> _Proc:
        bridge = _Proc(
            name,
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
        return bridge

    def _run(self, mode: str) -> str:
        """Run publisher + bridge for RUN_S; return the bridge's output."""
        bridge = self._bridge(f"bridge_{mode}", self._start_publisher(mode))
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

    def _refused(self, sock: Path, name: str) -> str:
        """Run the bridge against ``sock``; it must exit non-zero; return its output."""
        bridge = self._bridge(name, sock)
        try:
            bridge.proc.wait(timeout=EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.fail(f"bridge did not exit:\n{bridge.text()[-3000:]}")
        self.assertNotEqual(bridge.proc.returncode, 0, bridge.text()[-3000:])
        text = bridge.text()
        self.assertNotRegex(text, r"\d+ frames", "the bridge streamed anyway")
        return text

    def test_unchanged_marker_is_encoded(self):
        text = self._run("valid")
        self.assertNotIn("SLOT OVERWRITE", text)
        stats = self._stats(text)
        self.assertEqual(stats["overruns"], 0, text[-3000:])
        self.assertGreater(stats["frames"], 0, text[-3000:])

    def test_advanced_marker_is_dropped_loudly(self):
        text = self._run("stale")
        self.assertIn("SLOT OVERWRITE", text)
        stats = self._stats(text)
        self.assertGreater(stats["overruns"], 0, text[-3000:])
        self.assertEqual(stats["frames"], 0, "an overwritten picture was encoded")

    def test_only_the_overwritten_pictures_are_dropped(self):
        text = self._run("mixed")
        self.assertIn("SLOT OVERWRITE", text)
        stats = self._stats(text)
        self.assertGreater(stats["overruns"], 0, text[-3000:])
        self.assertGreater(stats["frames"], 0, text[-3000:])

    def test_publisher_without_write_generations_is_refused(self):
        upstream = self._start_publisher("valid")

        def drop_feature(hello: dict) -> None:
            hello["features"] = [
                f for f in hello["features"] if f != "slot_write_generation"
            ]

        relay = _Relay(self._tmp / "nofeature.sock", upstream, edit_hello=drop_feature)
        self.addCleanup(relay.close)
        text = self._refused(relay.path, "bridge_nofeature")
        self.assertIn("does not provide slot write generations", text)

    def test_missing_marker_descriptor_is_refused(self):
        upstream = self._start_publisher("valid")
        relay = _Relay(
            self._tmp / "nodesc.sock",
            upstream,
            edit=lambda msg: msg.pop("slot_markers", None),
        )
        self.addCleanup(relay.close)
        text = self._refused(relay.path, "bridge_nodesc")
        self.assertIn("did not offer slot write generations", text)

    def test_wrong_epoch_descriptor_is_refused(self):
        upstream = self._start_publisher("valid")

        def wrong_epoch(msg: dict) -> None:
            msg["slot_markers"]["epoch"] += 1

        relay = _Relay(self._tmp / "epoch.sock", upstream, edit=wrong_epoch)
        self.addCleanup(relay.close)
        text = self._refused(relay.path, "bridge_epoch")
        self.assertIn("[error]", text)
        self.assertIn("not the advertised region", text)

    def test_garbled_descriptors_are_refused(self):
        edits = {
            "layout": lambda d: d.update(layout_version=d["layout_version"] + 1),
            "size": lambda d: d.update(size=d["size"] + 8),
            "name": lambda d: d.update(name="/no_such_marker_region"),
            "fields": lambda d: d.pop("epoch"),
            "type": lambda d: d.update(size="large"),
        }
        upstream = self._start_publisher("valid")
        for label, edit in edits.items():
            with self.subTest(label):
                relay = _Relay(
                    self._tmp / f"garbled_{label}.sock",
                    upstream,
                    edit=lambda msg, edit=edit: edit(msg["slot_markers"]),
                )
                self.addCleanup(relay.close)
                text = self._refused(relay.path, f"bridge_garbled_{label}")
                self.assertIn("[error]", text)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--publisher":
        _publisher_main(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
