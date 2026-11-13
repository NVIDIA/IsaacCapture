# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge timing against egocap's own clock-sync and recording code.

Mock sender -> camera_receiver -> camera_livekit_bridge -> LiveKit -> egocap's
ClockSyncClient / LiveKitRecordingFeed (run in a subprocess under egocap's
interpreter). The bridge is restarted per `--timestamp-basis`; everything
else is shared.

Run: python3 test_bridge_timing.py -v   (about 40 s; needs a CUDA device).

Environment overrides: LIVEKIT_SERVER_BIN, CAMERA_LIVEKIT_BRIDGE_BIN,
EGOCAP_DIR (an egocap checkout with the timing code, the directory holding
backend/ and capture_server/), EGOCAP_PYTHON (interpreter with egocap's
dependencies and the livekit SDK). Skipped, never faked, when one is missing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from test_bridge_pipeline import (
    BRIDGE_BIN,
    MOCK_DIR,
    RECEIVER_DIR,
    RECEIVER_PY,
    SERVER_BIN,
    SOURCE_FPS,
    _cuda_present,
    _Proc,
    free_port,
    API_KEY,
    API_SECRET,
    _b64,
    make_token,
)

# name, source width, source height, track name, bridge divisor
STREAMS = [
    ("head", 640, 496, "head_track", 2),
    ("wrist", 320, 240, "wrist_track", 1),
]
BASES = ("monotonic_us",)
COLLECT_S = 10.0
MAX_AGE_MS = 1500.0
OFFSET_TOLERANCE_NS = 2_000_000

# Defaults: the egocap example in this repository and the current interpreter; override with the environment.
EGOCAP_DIR = Path(
    os.environ.get("EGOCAP_DIR", Path(__file__).resolve().parents[3] / "egocap")
)
EGOCAP_PY = Path(os.environ.get("EGOCAP_PYTHON", sys.executable))

# Runs under EGOCAP_PY with PYTHONPATH at egocap's packages; prints one RESULT line.
DRIVER_SCRIPT = r"""
import asyncio, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from capture_server.livekit_streams import load_livekit_streams
from egocap_web.livekit_receiver import LiveKitFrameReceiver
from egocap_web.livekit_recording import (
    LiveKitRecordingFeed, unrecordable_reason, uncertainty_budget_ns)

yaml_path, token_file, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])


class Ingress:
    # Keeps timestamps only: holding decoded frames would pin the SDK's buffers.
    def __init__(self):
        self.samples = {}

    def offer(self, sample):
        self.samples.setdefault(sample.source, []).append(
            (sample.capture_time, sample.receive_time))
        return True


async def main():
    streams = load_livekit_streams(Path(yaml_path))
    reason = unrecordable_reason(SimpleNamespace(streams=streams), "http://127.0.0.1:8765")
    receiver = LiveKitFrameReceiver(
        streams, open(token_file).read().strip(), clock_sync=True)
    await receiver.start()
    client = receiver.clock_sync
    exchanges = {"added": 0, "rejected_by_estimator": 0}
    real_add = client.estimator.add

    def counting_add(exchange):
        exchanges["added"] += 1
        why = real_add(exchange)
        if why is not None:
            exchanges["rejected_by_estimator"] += 1
        return why

    client.estimator.add = counting_add
    ingress = Ingress()
    # As LiveKitRecordingSource.prepare() does: the feed starts only once a mapping exists.
    t0 = time.monotonic()
    while client.mapping() is None and time.monotonic() - t0 < 10:
        await asyncio.sleep(0.05)
    first_mapping_s = time.monotonic() - t0
    receiver.clear_queued()
    feed = LiveKitRecordingFeed(
        receiver, streams, ingress, {s.stream_id: s.name for s in streams})
    feed.start()
    maps, end = [], time.monotonic() + seconds
    while time.monotonic() < end and feed.failure is None:
        await asyncio.sleep(0.5)
        m = client.mapping()
        maps.append(None if m is None else [m.epoch, m.offset_ns, m.rtt_ns, m.uncertainty_ns])
    out = {
        "first_mapping_s": first_mapping_s,
        "unrecordable_reason": reason,
        "budget_ns": uncertainty_budget_ns(),
        "feed_failure": feed.failure,
        "loss": feed.loss(),
        "exchanges": exchanges,
        "client_rejected": client.rejected,
        "discontinuity": client.discontinuity,
        "epochs": sorted({h.epoch for h in client.estimator.history}),
        "mappings": maps,
        "ages_ms": {n: [(r - c) * 1000 for c, r in s] for n, s in ingress.samples.items()},
    }
    await feed.stop()
    await receiver.stop()
    print("RESULT " + json.dumps(out))


asyncio.run(main())
"""


def make_receiver_token(room: str, identity: str) -> str:
    """Subscribe and publish-data: clock sync sends its requests as data messages."""
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    claims = {
        "iss": API_KEY, "sub": identity, "nbf": now, "exp": now + 3600, "name": identity,
        "video": {
            "room": room, "roomJoin": True, "canPublish": False,
            "canSubscribe": True, "canPublishData": True,
        },
    }  # fmt: skip
    body = _b64(json.dumps(claims).encode())
    sig = hmac.new(API_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256)
    return f"{header}.{body}.{_b64(sig.digest())}"


def _egocap_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [
        "backend",
        "capture_server",
        "recording",
        "../../src/plugins/manus_calibration/python",
    ]
    env["PYTHONPATH"] = os.pathsep.join(str(EGOCAP_DIR / p) for p in parts)
    return env


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
    if not (EGOCAP_DIR / "backend/egocap_web/livekit_clock_sync.py").is_file():
        return f"egocap timing checkout not found ({EGOCAP_DIR}); set EGOCAP_DIR"
    if not EGOCAP_PY.is_file():
        return f"egocap interpreter missing ({EGOCAP_PY}); set EGOCAP_PYTHON"
    probe = subprocess.run(
        [str(EGOCAP_PY), "-c", "import livekit.rtc, egocap_web.livekit_recording"],
        capture_output=True,
        env=_egocap_env(),
        check=False,
        timeout=120,
    )
    if probe.returncode != 0:
        return (
            "egocap code or livekit SDK not importable under EGOCAP_PYTHON: "
            + (probe.stderr.decode(errors="replace")[-300:])
        )
    return None


def _streams_yaml(url: str, room: str, publisher: str, basis: str) -> str:
    lines = [
        "livekit:",
        f"  url: {url}",
        f"  room: {room}",
        f"  publisher: {publisher}",
        f"  timestamp_basis: {basis}",
        "  streams:",
    ]
    for name, w, h, track, div in STREAMS:
        lines.append(
            f"    - {{name: {name}, track: {track}, width: {w // div}, height: {h // div}, fps: {SOURCE_FPS}}}"
        )
    return "\n".join(lines) + "\n"


def _result(stdout: str, what: str, extra: str = "") -> dict:
    marker = [ln for ln in stdout.splitlines() if ln.startswith("RESULT ")]
    if not marker:
        raise AssertionError(f"{what} produced no result:\n{stdout[-2000:]}\n{extra}")
    return json.loads(marker[-1][len("RESULT ") :])


def _summ(ages: list[float]) -> str:
    return f"n={len(ages)} min={min(ages):.1f} median={statistics.median(ages):.1f} max={max(ages):.1f} ms"


def _report(r: dict) -> str:
    maps = [m for m in r["mappings"] if m is not None]
    last = maps[-1] if maps else None
    ages = {n: _summ(a) for n, a in r["ages_ms"].items() if a}
    return (
        f"first_mapping_s={r['first_mapping_s']:.2f} replies={r['exchanges']} mappings={len(maps)} last(epoch,offset_ns,rtt_ns,unc_ns)={last} "
        f"discontinuity={r['discontinuity']} feed_failure={r['feed_failure']!r} ages={ages}"
    )


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class BridgeTimingTest(unittest.TestCase):
    results: dict[str, dict]

    @classmethod
    def setUpClass(cls):
        cls._procs: list[_Proc] = []
        cls._tmp = Path(tempfile.mkdtemp(prefix="bridge_timing_"))
        cls.results = {}
        try:
            cls._run_all()
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
    def _start_chain(cls, room: str) -> tuple[str, Path]:
        tmp = cls._tmp
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
                str(SERVER_BIN), "--dev", "--bind", "127.0.0.1", "--node-ip", "127.0.0.1",
                "--port", str(lk_port), "--rtc.tcp_port", str(lk_tcp), "--udp-port", str(lk_udp),
            ],
        )  # fmt: skip
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
            sys.executable, str(MOCK_DIR / "mock_sender.py"),
            "--port", str(mock_port), "--duration", "400",
            "--cache-dir", str(Path(tempfile.gettempdir()) / "camera_livekit_bridge_test_cache"),
        ]  # fmt: skip
        for name, w, h, _, _ in STREAMS:
            mock_cmd += ["--stream", f"{name}:{w}x{h}@{SOURCE_FPS}"]
        mock = cls._spawn("mock_sender", mock_cmd)
        mock.wait_for("mock sender listening", 120)

        lines = [
            "version: 1", "receiver:", "  host: 127.0.0.1", f"  port: {mock_port}", "  gpu_id: 0",
            "  stats_log_interval_s: 5", "ipc:", f"  socket_path: {sock}",
            f"  owner_name: bridge_timing_{os.getpid()}", "  queue_size: 1024", "streams:",
        ]  # fmt: skip
        for i, (name, w, h, _, _) in enumerate(STREAMS):
            lines += [
                f"  - index: {i}", f"    name: {name}", "    type: mono", f"    width: {w}",
                f"    height: {h}", "    output_format: nv12", "    ring_depth: 6", "    ipc:",
                f"      mono: {name}",
            ]  # fmt: skip
        cfg = tmp / "receiver.yaml"
        cfg.write_text("\n".join(lines) + "\n")
        receiver = cls._spawn(
            "camera_receiver",
            [
                str(RECEIVER_PY), str(RECEIVER_DIR / "camera_receiver.py"),
                "--config", str(cfg), "--port", str(mock_port), "--socket", str(sock),
            ],
            cwd=RECEIVER_DIR,
        )  # fmt: skip
        receiver.wait_for("camera_receiver running", 60)
        receiver.wait_for("connected to AVS1 sender", 30)
        return url, sock

    @classmethod
    def _run_basis(cls, url: str, sock: Path, room: str, basis: str) -> dict:
        tmp = cls._tmp
        identity = f"bridge-{basis}"
        pub_token = tmp / f"pub-{basis}.token"
        pub_token.write_text(make_token(room, identity, True))
        cmd = [
            str(BRIDGE_BIN), "--livekit-url", url, "--livekit-token-file", str(pub_token),
            "--socket", str(sock), "--fps", str(SOURCE_FPS), "--timestamp-basis", basis,
        ]  # fmt: skip
        for name, _, _, track, div in STREAMS:
            cmd += ["--stream", f"{name}={track}:{div}"]
        n_before = len(cls._procs)
        bridge = cls._spawn(f"bridge_{basis}", cmd)
        try:
            bridge.wait_for("publishing", 60)
            yaml_path = tmp / f"streams-{basis}.yaml"
            yaml_path.write_text(_streams_yaml(url, room, identity, basis))
            # Clock sync sends data messages, so the receiver needs publish-data as well as subscribe.
            token = tmp / f"recv-{basis}.token"
            token.write_text(make_receiver_token(room, f"timing-{basis}"))
            out = subprocess.run(
                [str(EGOCAP_PY), "-c", DRIVER_SCRIPT, str(yaml_path), str(token), str(COLLECT_S)],
                capture_output=True, text=True, check=False, env=_egocap_env(),
                timeout=COLLECT_S + 90,
            )  # fmt: skip
            result = _result(
                out.stdout,
                f"egocap driver ({basis})",
                out.stderr[-2000:] + bridge.text()[-2000:],
            )
        finally:
            # One identity at a time: stop only this bridge before the next starts.
            for proc in reversed(cls._procs[n_before:]):
                proc.stop()
            del cls._procs[n_before:]
        return result

    @classmethod
    def _run_all(cls):
        room = f"bridge-timing-{uuid.uuid4().hex[:10]}"
        url, sock = cls._start_chain(room)
        for basis in BASES:
            cls.results[basis] = cls._run_basis(url, sock, room, basis)
            print(f"\n[{basis}] {_report(cls.results[basis])}", file=sys.stderr)

    def _check_feed(self, basis: str):
        r = self.results[basis]
        self.assertIsNone(
            r["feed_failure"], f"{basis}: feed failed: {r['feed_failure']}"
        )
        self.assertEqual(sorted(r["ages_ms"]), sorted(s[0] for s in STREAMS))
        for name, ages in r["ages_ms"].items():
            self.assertGreater(
                len(ages), 30, f"{basis}/{name}: too few recorded frames"
            )
            self.assertTrue(
                all(0 <= a <= MAX_AGE_MS for a in ages),
                f"{basis}/{name}: age out of range; {_summ(ages)}; "
                f"over limit: {[round(a) for a in ages if a > MAX_AGE_MS][:10]}",
            )

    def _check_clock(self, basis: str):
        r = self.results[basis]
        self.assertGreaterEqual(
            r["exchanges"]["added"], 5, f"{basis}: few replies: {r['exchanges']}"
        )
        maps = [m for m in r["mappings"] if m is not None]
        self.assertTrue(maps, f"{basis}: estimator never produced a mapping")
        self.assertFalse(r["discontinuity"], f"{basis}: clock discontinuity")
        self.assertEqual(r["epochs"], [1], f"{basis}: epochs {r['epochs']}")
        budget = r["budget_ns"]
        for epoch, offset, rtt, unc in maps:
            self.assertEqual(epoch, 1)
            self.assertLess(
                unc, budget, f"{basis}: uncertainty {unc} ns over budget {budget} ns"
            )
            self.assertLess(
                abs(offset),
                OFFSET_TOLERANCE_NS,
                f"{basis}: offset {offset} ns (rtt {rtt} ns)",
            )

    def test_monotonic_us_clock_and_feed(self):
        self._check_clock("monotonic_us")
        self._check_feed("monotonic_us")


@unittest.skipUnless(
    BRIDGE_BIN and Path(str(BRIDGE_BIN)).exists(), "camera_livekit_bridge not built"
)
class BridgeCaptureBasisCliTest(unittest.TestCase):
    """Startup refusals; the bridge exits before connecting, so no servers or GPU."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="bridge_cli_")
        self.addCleanup(self._td.cleanup)
        self._tmp_dir = Path(self._td.name)

    def _run_bridge_cli(self, name: str, *extra: str):
        token = self._tmp_dir / f"{name}.token"
        token.write_text(make_token("unused", f"bridge-{name}", True))
        cmd = [
            str(BRIDGE_BIN), "--livekit-url", "ws://127.0.0.1:1", "--livekit-token-file", str(token),
            "--socket", str(self._tmp_dir / "none.sock"), "--stream", "head=head_track",
            "--timestamp-basis", "capture_monotonic_us", *extra,
        ]  # fmt: skip
        started = time.monotonic()
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=30
        )
        self.assertLess(time.monotonic() - started, 15)
        return out

    def test_capture_monotonic_us_requires_chrony_sender(self):
        out = self._run_bridge_cli("no-sender")
        self.assertEqual(out.returncode, 2, out.stdout + out.stderr)
        self.assertIn("--chrony-sender", out.stderr)

    def test_capture_monotonic_us_requires_sender_refid(self):
        out = self._run_bridge_cli("no-refid", "--chrony-sender", "127.0.0.1")
        self.assertEqual(out.returncode, 2, out.stdout + out.stderr)
        self.assertIn("--chrony-sender-refid", out.stderr)


if __name__ == "__main__":
    unittest.main()
