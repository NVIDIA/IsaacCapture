# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge exits promptly on SIGTERM while frame notices keep arriving.

A separate publisher process announces frames in a tight loop, so the bridge's
notice queue never empties. The bridge's notice-drain loops must still notice a
stop request: after SIGTERM it has to exit with status 0 within a few seconds.

Run: python3 test_bridge_shutdown.py -v   (about 15 s; needs a CUDA device).

Environment overrides and skip rules are those of test_bridge_slot_age.py.
"""

from __future__ import annotations

import json
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from test_bridge_pipeline import (
    BRIDGE_BIN,
    RECEIVER_DIR,
    SERVER_BIN,
    _Proc,
    free_port,
    LIVEKIT_ISSUER,
    SIGNING_KEY,
    make_token,
)
from test_bridge_slot_age import (
    HEIGHT,
    PUBLISHER_IDENTITY,
    RING,
    STREAM,
    TRACK,
    WIDTH,
    _cupy_python,
    _skip_reason,
)

FPS = 30
FLOOD_S = 3.0
EXIT_DEADLINE_S = 5.0
NOTICES_PER_BLOCK = 2000


def _publisher_main(socket_path: str) -> None:
    """Child process: serve one stream and announce frames as fast as possible."""
    import cupy as cp

    sys.path.insert(0, str(RECEIVER_DIR))
    from cuda_ipc import CudaIpcFramePublisher

    publisher = CudaIpcFramePublisher(socket_path=socket_path, owner_name="shutdown")
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

    # The publisher's own broadcast thread is slower than the bridge's reader, so the
    # queue would drain between notices. Write one pre-encoded block of notices straight
    # to the subscribed socket in a loop: the writer is then almost always waiting on
    # the reader rather than the reverse. The notices go stale, which the bridge
    # tolerates; only the unbroken supply matters here.
    block = b"".join(
        json.dumps(
            {
                "type": "frame",
                "stream": STREAM,
                "slot": i % RING,
                "slot_generation": i // RING + 1,
                "seq": i,
                "timestamp_ns": time.monotonic_ns(),
                "metadata": {"session_id": 1, "stream_name": STREAM},
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for i in range(NOTICES_PER_BLOCK)
    )
    while not publisher._clients and not stopping:
        time.sleep(0.05)
    while not stopping:
        for client in list(publisher._clients):
            try:
                client.sock.sendall(block)
            except OSError:
                return
    publisher.stop()


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class BridgeShutdownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="bridge_shutdown_"))
        cls.addClassCleanup(shutil.rmtree, cls._tmp, ignore_errors=True)
        lk_port = free_port(socket.SOCK_STREAM)
        cls.url = f"ws://127.0.0.1:{lk_port}"
        cls.room = f"shutdown-{uuid.uuid4().hex[:10]}"
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

    def test_sigterm_exits_while_notices_keep_arriving(self):
        sock = self._tmp / "flood.sock"
        publisher = _Proc(
            "publisher_flood",
            [
                str(_cupy_python()),
                str(Path(__file__).resolve()),
                "--publisher",
                str(sock),
            ],
            self._tmp,
        )
        self.addCleanup(publisher.stop)
        publisher.wait_for("READY", 60)
        bridge = _Proc(
            "bridge_flood",
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
        time.sleep(FLOOD_S)
        self.assertIsNone(bridge.proc.poll(), bridge.text()[-3000:])

        start = time.monotonic()
        bridge.proc.send_signal(signal.SIGTERM)
        try:
            code = bridge.proc.wait(timeout=EXIT_DEADLINE_S)
        except subprocess.TimeoutExpired:
            self.fail(
                f"the bridge did not exit within {EXIT_DEADLINE_S} s of SIGTERM "
                f"while notices kept arriving:\n{bridge.text()[-3000:]}"
            )
        self.assertEqual(code, 0, bridge.text()[-3000:])
        self.assertLess(time.monotonic() - start, EXIT_DEADLINE_S)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--publisher":
        _publisher_main(sys.argv[2])
    else:
        unittest.main()
