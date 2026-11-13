# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End to end: mock sender -> camera_receiver -> camera_livekit_bridge -> LiveKit -> subscriber.

Run: python3 test_bridge_pipeline.py -v   (about 40 s; needs a CUDA device).

Environment overrides: LIVEKIT_SERVER_BIN, CAMERA_LIVEKIT_BRIDGE_BIN,
LIVEKIT_PYTHON (interpreter with the `livekit` SDK and numpy, used for the
subscriber). The test is skipped, never faked, when a prerequisite is missing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
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

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2]
MOCK_DIR = EXAMPLES / "camera_streamer" / "mock_sender"
RECEIVER_DIR = EXAMPLES / "camera_streamer" / "camera_receiver"

SERVER_BIN = Path(
    os.environ.get(
        "LIVEKIT_SERVER_BIN",
        shutil.which("livekit-server")
        or Path.home() / "github/livekit-server-1.13.10/livekit-server",
    )
)
BRIDGE_BIN = Path(
    os.environ.get(
        "CAMERA_LIVEKIT_BRIDGE_BIN",
        EXAMPLES / "camera_livekit_bridge/build/bridge/camera_livekit_bridge",
    )
)
RECEIVER_PY = RECEIVER_DIR / ".venv" / "bin" / "python"
SUBSCRIBER_PY = Path(os.environ.get("LIVEKIT_PYTHON", sys.executable))

API_KEY, API_SECRET = "devkey", "secret"
SOURCE_FPS = 30
# name, source width, source height, track name, bridge divisor
STREAMS = [
    ("head", 640, 496, "head_track", 2),
    ("wrist", 320, 240, "wrist_track", 1),
]
COLLECT_S = 8.0
PUBLISHER_IDENTITY = "bridge-pub"

# Runs under LIVEKIT_PYTHON; prints one JSON document with per-track samples.
SUBSCRIBER_SCRIPT = r"""
import asyncio, json, sys, time
from livekit import rtc

url, token_file, n_tracks, collect_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
token = open(token_file).read().strip()


async def main():
    room = rtc.Room()
    tracks = {}
    tasks = []

    async def read(name, identity, track):
        t = tracks.setdefault(name, {"identity": identity, "frames": []})
        first = None
        async for event in rtc.VideoStream(track):
            now = time.time()
            first = first or now
            if now - first > collect_s:
                break
            m = event.metadata
            t["frames"].append({
                "t": now,
                "w": event.frame.width,
                "h": event.frame.height,
                "id": m.frame_id if m is not None and m.HasField("frame_id") else None,
                "ts": m.user_timestamp if m is not None and m.HasField("user_timestamp") else None,
                "ud": bytes(m.user_data).hex() if m is not None and m.HasField("user_data") else None,
            })

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            tasks.append(asyncio.create_task(read(publication.name, participant.identity, track)))

    await room.connect(url, token)
    deadline = time.time() + 25 + collect_s
    while time.time() < deadline:
        await asyncio.sleep(0.25)
        if len(tasks) >= n_tracks and all(t.done() for t in tasks):
            break
    keyframe_api = any(hasattr(rtc.VideoStream, n) for n in ("request_keyframe", "send_pli"))
    await room.disconnect()
    print("RESULT " + json.dumps({"tracks": tracks, "keyframe_api": keyframe_api}))


asyncio.run(main())
"""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_token(room: str, identity: str, publish: bool) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    claims = {
        "iss": API_KEY,
        "sub": identity,
        "nbf": now,
        "exp": now + 3600,
        "name": identity,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": publish,
            "canSubscribe": not publish,
            "canPublishData": publish,
        },
    }
    body = _b64(json.dumps(claims).encode())
    sig = hmac.new(API_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256)
    return f"{header}.{body}.{_b64(sig.digest())}"


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _python_has(python: Path, module: str) -> bool:
    if not python.exists():
        return False
    try:
        return (
            subprocess.run(
                [str(python), "-c", f"import {module}"],
                capture_output=True,
                check=False,
                timeout=60,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _cuda_present() -> bool:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        out = subprocess.run(
            [smi, "-L"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def _skip_reason() -> str | None:
    if not SERVER_BIN.is_file():
        return f"livekit-server not found ({SERVER_BIN}); set LIVEKIT_SERVER_BIN"
    if not BRIDGE_BIN.is_file():
        return f"camera_livekit_bridge not built ({BRIDGE_BIN}); set CAMERA_LIVEKIT_BRIDGE_BIN"
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is required by the mock sender"
    if not RECEIVER_PY.is_file():
        return f"receiver venv missing ({RECEIVER_PY})"
    if not _cuda_present():
        return "no CUDA device (nvidia-smi -L lists no GPU)"
    if not _python_has(SUBSCRIBER_PY, "livekit.rtc") or not _python_has(
        SUBSCRIBER_PY, "numpy"
    ):
        return f"{SUBSCRIBER_PY} lacks the livekit SDK or numpy; set LIVEKIT_PYTHON"
    return None


class _Proc:
    """A child process with its output captured to a file and scanned for markers."""

    def __init__(self, name: str, argv: list[str], logdir: Path, cwd=None):
        self.name = name
        self.log_path = logdir / f"{name}.log"
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            argv,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            start_new_session=True,
        )

    def text(self) -> str:
        self._log.flush()
        return self.log_path.read_text(errors="replace")

    def wait_for(self, marker: str, timeout: float) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if marker in self.text():
                return
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"{self.name} exited ({self.proc.returncode}) before {marker!r}:\n"
                    f"{self.text()[-3000:]}"
                )
            time.sleep(0.2)
        raise AssertionError(
            f"{self.name}: no {marker!r} within {timeout}s:\n{self.text()[-3000:]}"
        )

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class BridgePipelineTest(unittest.TestCase):
    result: dict

    @classmethod
    def setUpClass(cls):
        cls._procs: list[_Proc] = []
        cls._tmp = Path(tempfile.mkdtemp(prefix="bridge_pipeline_"))
        try:
            cls.result = cls._run_pipeline()
        except BaseException:
            cls._stop_all()
            raise
        cls._stop_all()

    @classmethod
    def _stop_all(cls):
        for proc in reversed(cls._procs):
            proc.stop()
        cls._procs.clear()

    @classmethod
    def _spawn(cls, name, argv, cwd=None) -> _Proc:
        proc = _Proc(name, argv, cls._tmp, cwd)
        cls._procs.append(proc)
        return proc

    @classmethod
    def _run_pipeline(cls) -> dict:
        tmp = cls._tmp
        room = f"bridge-pipeline-{uuid.uuid4().hex[:10]}"
        lk_port, lk_tcp, lk_udp = (
            free_port(socket.SOCK_STREAM),
            free_port(socket.SOCK_STREAM),
            free_port(socket.SOCK_DGRAM),
        )
        mock_port = free_port(socket.SOCK_STREAM)
        sock = tmp / "receiver.sock"
        url = f"ws://127.0.0.1:{lk_port}"

        server = cls._spawn(
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
                str(lk_tcp),
                "--udp-port",
                str(lk_udp),
            ],
        )
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

        mock_cmd = [
            sys.executable,
            str(MOCK_DIR / "mock_sender.py"),
            "--port",
            str(mock_port),
            "--duration",
            "120",
            "--cache-dir",
            str(Path(tempfile.gettempdir()) / "camera_livekit_bridge_test_cache"),
        ]
        for name, w, h, _, _ in STREAMS:
            mock_cmd += ["--stream", f"{name}:{w}x{h}@{SOURCE_FPS}"]
        mock = cls._spawn("mock_sender", mock_cmd)
        mock.wait_for("mock sender listening", 120)

        cfg = tmp / "receiver.yaml"
        lines = [
            "version: 1",
            "receiver:",
            "  host: 127.0.0.1",
            f"  port: {mock_port}",
            "  gpu_id: 0",
            "  stats_log_interval_s: 5",
            "ipc:",
            f"  socket_path: {sock}",
            f"  owner_name: bridge_pipeline_{os.getpid()}",
            "  queue_size: 1024",
            "streams:",
        ]
        for i, (name, w, h, _, _) in enumerate(STREAMS):
            lines += [
                f"  - index: {i}",
                f"    name: {name}",
                "    type: mono",
                f"    width: {w}",
                f"    height: {h}",
                "    output_format: nv12",
                "    ring_depth: 6",
                "    ipc:",
                f"      mono: {name}",
            ]
        cfg.write_text("\n".join(lines) + "\n")
        receiver = cls._spawn(
            "camera_receiver",
            [
                str(RECEIVER_PY),
                str(RECEIVER_DIR / "camera_receiver.py"),
                "--config",
                str(cfg),
                "--port",
                str(mock_port),
                "--socket",
                str(sock),
            ],
            cwd=RECEIVER_DIR,
        )
        receiver.wait_for("camera_receiver running", 60)
        receiver.wait_for("connected to AVS1 sender", 30)

        pub_token = tmp / "pub.token"
        pub_token.write_text(make_token(room, PUBLISHER_IDENTITY, True))
        bridge_cmd = [
            str(BRIDGE_BIN),
            "--livekit-url",
            url,
            "--livekit-token-file",
            str(pub_token),
            "--socket",
            str(sock),
            "--fps",
            str(SOURCE_FPS),
        ]
        for name, _, _, track, div in STREAMS:
            bridge_cmd += ["--stream", f"{name}={track}:{div}"]
        bridge = cls._spawn("bridge", bridge_cmd)
        bridge.wait_for("publishing", 60)

        sub_token = tmp / "sub.token"
        sub_token.write_text(make_token(room, "test-subscriber", False))
        script = tmp / "subscriber.py"
        script.write_text(SUBSCRIBER_SCRIPT)
        try:
            out = subprocess.run(
                [
                    str(SUBSCRIBER_PY),
                    str(script),
                    url,
                    str(sub_token),
                    str(len(STREAMS)),
                    str(COLLECT_S),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=COLLECT_S + 60,
            )
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(f"subscriber timed out: {exc.stdout}") from exc
        marker = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
        if not marker:
            raise AssertionError(
                f"subscriber produced no result (rc={out.returncode}):\n"
                f"{out.stdout[-2000:]}\n{out.stderr[-2000:]}\nbridge:\n{bridge.text()[-2000:]}"
            )
        return json.loads(marker[-1][len("RESULT ") :])

    def _frames(self, track: str) -> list[dict]:
        tracks = self.result["tracks"]
        self.assertIn(track, tracks, f"track {track} missing; got {sorted(tracks)}")
        return tracks[track]["frames"]

    def test_tracks_and_sizes(self):
        self.assertEqual(sorted(self.result["tracks"]), sorted(s[3] for s in STREAMS))
        for _name, w, h, track, div in STREAMS:
            info = self.result["tracks"][track]
            self.assertEqual(info["identity"], PUBLISHER_IDENTITY)
            frames = self._frames(track)
            self.assertTrue(frames, f"{track} delivered no frames")
            sizes = {(f["w"], f["h"]) for f in frames}
            # The first frames can be pre-adaptation; the steady state must match.
            steady = {(f["w"], f["h"]) for f in frames[len(frames) // 4 :]}
            self.assertEqual(steady, {(w // div, h // div)}, f"{track}: {sizes}")

    def test_frame_ids_and_rate(self):
        for _, _, _, track, _ in STREAMS:
            frames = self._frames(track)
            ids = [f["id"] for f in frames]
            self.assertNotIn(None, ids, f"{track}: frame without frame_id")
            steps = [b - a for a, b in zip(ids, ids[1:])]
            self.assertTrue(all(s > 0 for s in steps), f"{track}: ids not increasing")
            consecutive = sum(1 for s in steps if s == 1)
            self.assertGreaterEqual(
                consecutive / max(len(steps), 1), 0.8, f"{track}: steps {steps[:60]}"
            )
            span = frames[-1]["t"] - frames[0]["t"]
            rate = (len(frames) - 1) / span
            self.assertTrue(
                0.7 * SOURCE_FPS <= rate <= 1.15 * SOURCE_FPS,
                f"{track}: {rate:.1f} fps over {span:.1f}s",
            )

    def test_user_timestamp_is_capture_time(self):
        for _, _, _, track, _ in STREAMS:
            frames = self._frames(track)
            t0 = frames[0]["t"]
            late = [f for f in frames if f["t"] - t0 >= 1.0]
            self.assertGreater(len(late), 30, f"{track}: too few frames after 1 s")
            for f in late:
                self.assertIsNotNone(f["ts"], f"{track}: frame without user_timestamp")
                age_ms = (f["t"] * 1e6 - f["ts"]) / 1000.0
                self.assertTrue(
                    0 <= age_ms <= 1500,
                    f"{track}: frame {f['id']} age {age_ms:.1f} ms",
                )

    def test_user_data_is_sensor_counter(self):
        for _, _, _, track, _ in STREAMS:
            for f in self._frames(track):
                self.assertIsNotNone(f["ud"], f"{track}: frame without user_data")
                raw = bytes.fromhex(f["ud"])
                self.assertEqual(len(raw), 8, f"{track}: user_data {raw!r}")
                self.assertNotEqual(int.from_bytes(raw, "little"), 0)

    def test_subscriber_keyframe_request(self):
        if not self.result["keyframe_api"]:
            self.skipTest(
                "the livekit Python SDK exposes no subscriber keyframe request"
            )


if __name__ == "__main__":
    unittest.main()
