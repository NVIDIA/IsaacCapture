# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture time through a measured sender clock, end to end.

mock sender (own clock, ground truth, UDP clock responder) -> camera_receiver ->
camera_livekit_bridge (`--timestamp-basis capture_monotonic_us --sender-clock
127.0.0.1:PORT`) -> LiveKit -> a
Python subscriber. Each published frame is joined to the mock's ground truth on
(stream, stream_sequence, session_id); the published `user_timestamp` is compared
with the true capture instant on this host's CLOCK_MONOTONIC. The bridge must
publish an accurate capture time or none at all: a wrong claim anywhere fails.

Run: LIVEKIT_PYTHON=/path/to/python python3 test_capture_time.py -v
(about 2.5 minutes; needs a CUDA device, ffmpeg and the receiver venv).

Environment overrides: LIVEKIT_SERVER_BIN, CAMERA_LIVEKIT_BRIDGE_BIN,
LIVEKIT_PYTHON (interpreter with the `livekit` SDK and numpy, used for the
subscriber). Skipped, never faked, when a prerequisite is missing.
"""

from __future__ import annotations

import json
import re
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
    SUBSCRIBER_PY,
    _Proc,
    _skip_reason,
    free_port,
    make_token,
)

# name, source width, source height, track name, bridge divisor
STREAMS = [
    ("head", 640, 496, "head_track", 2),
    ("wrist", 320, 240, "wrist_track", 1),
]
TRACK_STREAM = {track: name for name, _, _, track, _ in STREAMS}

# Accuracy bounds, on a live run of the accuracy scenario the error was
# median +0.3 us, p95 22 us, max 37 us. The bounds sit well above that and well
# below anything a consumer would notice.
MEDIAN_BOUND_NS = 100_000
P95_BOUND_NS = 500_000
MAX_BOUND_NS = 2_000_000

# Runs under LIVEKIT_PYTHON. Waits for the first frame, then records every frame
# (track, arrival on CLOCK_MONOTONIC, frame_id, user_timestamp) for run_s.
SUBSCRIBER_SCRIPT = r"""
import asyncio, json, sys, time
from livekit import rtc

url, token_file, run_s, out_file = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
token = open(token_file).read().strip()


async def main():
    room = rtc.Room()
    rows = []
    state = {"end": None}

    async def read(name, track):
        async for event in rtc.VideoStream(track, capacity=8):
            now = time.monotonic_ns()
            if state["end"] is None:
                state["end"] = time.monotonic() + run_s
            m = event.metadata
            rows.append([
                name, now,
                m.frame_id if m is not None and m.HasField("frame_id") else None,
                m.user_timestamp if m is not None and m.HasField("user_timestamp") else None,
            ])

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(read(publication.name, track))

    await room.connect(url, token)
    hard_end = time.monotonic() + 120 + run_s
    while time.monotonic() < hard_end:
        await asyncio.sleep(0.25)
        if state["end"] is not None and time.monotonic() > state["end"]:
            break
    await room.disconnect()
    with open(out_file, "w") as f:
        json.dump(rows, f)
    print("RESULT ok", len(rows))


asyncio.run(main())
"""


def _pctl(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[
        min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    ]


class Run:
    """One scenario's frames joined to ground truth, plus the bridge log."""

    def __init__(self, rows, truth_lines, bridge_log: str):
        self.bridge_log = bridge_log
        # session_id -> start (first ready instant, the mock's connection start)
        self.sessions: dict[int, float] = {}
        truth = {}
        by_key: dict[tuple, list[tuple[int, dict]]] = {}
        for line in truth_lines:
            r = json.loads(line)
            sid = r["session_id"]
            start = self.sessions.get(sid)
            if start is None or r["ready_pc_mono_ns"] < start:
                self.sessions[sid] = r["ready_pc_mono_ns"]
            truth[(r["stream"], r["stream_sequence"], sid)] = r
            by_key.setdefault((r["stream"], r["stream_sequence"]), []).append((sid, r))
        self.frames: list[dict] = []
        self.unjoined = 0
        for track, arrival, frame_id, ts in rows:
            stream = TRACK_STREAM.get(track)
            if stream is None or frame_id is None:
                self.unjoined += 1
                continue
            # The sequence repeats across sessions; the session whose frame was
            # ready most recently before this arrival is the one delivered.
            cands = [
                (arrival - r["ready_pc_mono_ns"], sid, r)
                for sid, r in by_key.get((stream, frame_id), [])
                if arrival >= r["ready_pc_mono_ns"]
            ]
            if not cands:
                self.unjoined += 1
                continue
            _, sid, r = min(cands, key=lambda c: c[0])
            self.frames.append(
                {
                    "stream": stream,
                    "session": sid,
                    "id": frame_id,
                    "ts": ts,
                    "arrival": arrival,
                    "t_rel": (r["ready_pc_mono_ns"] - self.sessions[sid]) / 1e9,
                    "err_ns": None
                    if ts is None
                    else ts * 1000 - r["true_capture_pc_mono_ns"],
                    "age_ms": (arrival - r["true_capture_pc_mono_ns"]) / 1e6,
                }
            )
        self.frames.sort(key=lambda f: f["arrival"])

    def session_order(self) -> list[int]:
        return sorted(self.sessions, key=self.sessions.get)

    def select(self, session=None, lo=None, hi=None, stream=None) -> list[dict]:
        return [
            f
            for f in self.frames
            if (session is None or f["session"] == session)
            and (stream is None or f["stream"] == stream)
            and (lo is None or f["t_rel"] >= lo)
            and (hi is None or f["t_rel"] < hi)
        ]

    def errors(self, frames: list[dict]) -> list[float]:
        return sorted(f["err_ns"] for f in frames if f["err_ns"] is not None)

    def summary(self, frames: list[dict] | None = None) -> str:
        frames = self.frames if frames is None else frames
        e = self.errors(frames)
        spans = [f["t_rel"] for f in frames if f["ts"] is None]
        withheld = len(spans)
        if spans:
            withheld = f"{withheld} (t_rel {min(spans):.2f}..{max(spans):.2f} s)"
        if not e:
            return f"frames={len(frames)} withheld={withheld} no claims"
        return (
            f"frames={len(frames)} withheld={withheld} claims={len(e)} error_us "
            f"median={statistics.median(e) / 1e3:.1f} "
            f"p95_abs={_pctl(sorted(abs(x) for x in e), 0.95) / 1e3:.1f} "
            f"max_abs={max(abs(x) for x in e) / 1e3:.1f}"
        )


def _stat_counts(log: str, reason: str) -> int:
    """Largest cumulative count the bridge's stats lines reported for a withhold reason."""
    best = 0
    for line in log.splitlines():
        marker = f", {reason} "
        if "[stats]" in line and marker in line:
            tail = line.split(marker, 1)[1].split(",", 1)[0]
            if tail.strip().isdigit():
                best = max(best, int(tail.strip()))
    return best


def _qualified_count(log: str) -> int:
    """Largest cumulative qualified-claim count in the bridge's stats lines."""
    found = [int(m) for m in re.findall(r"capture claims: (\d+) qualified", log)]
    return max(found, default=0)


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class CaptureTimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="capture_time_"))
        cls._procs: list[_Proc] = []
        lk_port, lk_tcp, lk_udp = (
            free_port(socket.SOCK_STREAM),
            free_port(socket.SOCK_STREAM),
            free_port(socket.SOCK_DGRAM),
        )
        cls.url = f"ws://127.0.0.1:{lk_port}"
        try:
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
        except BaseException:
            cls._stop_all()
            raise

    @classmethod
    def tearDownClass(cls):
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
    def _scenario(cls, name: str, mock_args: list[str], run_s: float) -> Run:
        """One mock, receiver, bridge and subscriber against the shared server."""
        tmp = cls._tmp / name
        tmp.mkdir()
        n_before = len(cls._procs)
        room = f"capture-time-{name}-{uuid.uuid4().hex[:10]}"
        mock_port = free_port(socket.SOCK_STREAM)
        clock_port = free_port(socket.SOCK_DGRAM)
        sock = tmp / "receiver.sock"
        truth_file = tmp / "ground_truth.jsonl"

        def spawn(label, argv, cwd=None):
            return cls._spawn(f"{name}_{label}", argv, cwd)

        try:
            cfg = tmp / "receiver.yaml"
            lines = [
                "version: 1", "receiver:", "  host: 127.0.0.1", f"  port: {mock_port}", "  gpu_id: 0",
                "  stats_log_interval_s: 5", "ipc:", f"  socket_path: {sock}",
                f"  owner_name: capture_time_{name}_{uuid.uuid4().hex[:6]}", "  queue_size: 1024", "streams:",
            ]  # fmt: skip
            for i, (sname, w, h, _, _) in enumerate(STREAMS):
                lines += [
                    f"  - index: {i}", f"    name: {sname}", "    type: mono", f"    width: {w}",
                    f"    height: {h}", "    output_format: nv12", "    ring_depth: 6", "    ipc:",
                    f"      mono: {sname}",
                ]  # fmt: skip
            cfg.write_text("\n".join(lines) + "\n")

            # The receiver and bridge wait first and the mock starts last, so the
            # mock's clock (session start, --clock-mute) begins as the receiver connects.
            receiver = spawn(
                "receiver",
                [
                    str(RECEIVER_PY), str(RECEIVER_DIR / "camera_receiver.py"),
                    "--config", str(cfg), "--port", str(mock_port), "--socket", str(sock),
                ],
                cwd=RECEIVER_DIR,
            )  # fmt: skip
            receiver.wait_for("camera_receiver running", 60)

            pub_token = tmp / "pub.token"
            pub_token.write_text(make_token(room, f"capture-time-bridge-{name}", True))
            bridge_cmd = [
                str(BRIDGE_BIN), "--livekit-url", cls.url, "--livekit-token-file", str(pub_token),
                "--socket", str(sock), "--fps", str(SOURCE_FPS),
                "--timestamp-basis", "capture_monotonic_us",
                "--sender-clock", f"127.0.0.1:{clock_port}",
            ]  # fmt: skip
            for sname, _, _, track, div in STREAMS:
                bridge_cmd += ["--stream", f"{sname}={track}:{div}"]
            bridge = spawn("bridge", bridge_cmd)

            sub_token = tmp / "sub.token"
            sub_token.write_text(make_token(room, f"capture-time-sub-{name}", False))
            script = tmp / "subscriber.py"
            script.write_text(SUBSCRIBER_SCRIPT)
            out_file = tmp / "frames.json"
            sub_log = open(tmp / "subscriber.log", "wb")
            sub = subprocess.Popen(
                [str(SUBSCRIBER_PY), str(script), cls.url, str(sub_token), str(run_s), str(out_file)],
                stdout=sub_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )  # fmt: skip
            try:
                mock_cmd = [
                    sys.executable, str(MOCK_DIR / "mock_sender.py"),
                    "--port", str(mock_port), "--duration", str(run_s + 60),
                    "--cache-dir", str(Path(tempfile.gettempdir()) / "camera_livekit_bridge_test_cache"),
                    "--timing-ext", "--ground-truth", str(truth_file),
                    "--clock-port", str(clock_port),
                ]  # fmt: skip
                for sname, w, h, _, _ in STREAMS:
                    mock_cmd += ["--stream", f"{sname}:{w}x{h}@{SOURCE_FPS}"]
                mock = spawn("mock", mock_cmd + mock_args)
                mock.wait_for("mock sender listening", 120)
                receiver.wait_for("connected to AVS1 sender", 60)
                bridge.wait_for("publishing", 60)
                try:
                    sub.wait(timeout=run_s + 150)
                except subprocess.TimeoutExpired as exc:
                    raise AssertionError("subscriber timed out") from exc
            finally:
                if sub.poll() is None:
                    sub.terminate()
                    try:
                        sub.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        sub.kill()
                sub_log.close()
            if not out_file.is_file():
                raise AssertionError(
                    f"subscriber wrote no frames:\n{(tmp / 'subscriber.log').read_text()[-2000:]}\n"
                    f"bridge:\n{bridge.text()[-2000:]}"
                )
            rows = json.loads(out_file.read_text())
            truth_lines = (
                truth_file.read_text().splitlines() if truth_file.is_file() else []
            )
            run = Run(rows, truth_lines, bridge.text())
            print(
                f"\n[{name}] {run.summary()} unjoined={run.unjoined}", file=sys.stderr
            )
            return run
        finally:
            for proc in reversed(cls._procs[n_before:]):
                proc.stop()
            del cls._procs[n_before:]

    # ---- shared checks ------------------------------------------------------

    def _no_wrong_claim(self, run: Run, label: str):
        wrong = [
            f
            for f in run.frames
            if f["err_ns"] is not None and abs(f["err_ns"]) > MAX_BOUND_NS
        ]
        self.assertFalse(
            wrong,
            f"{label}: {len(wrong)} frames carry a wrong capture time; first: "
            f"{[(f['stream'], f['id'], round(f['t_rel'], 2), round(f['err_ns'] / 1e3)) for f in wrong[:8]]} us",
        )

    def _frame_ids_continuous(self, run: Run, label: str):
        """Frames keep their source stream_sequence (they join to ground truth) without gaps from renumbering."""
        self.assertLess(
            run.unjoined,
            0.05 * max(len(run.frames), 1) + 5,
            f"{label}: {run.unjoined} frames did not join to ground truth",
        )
        for sid in run.session_order():
            for stream in TRACK_STREAM.values():
                frames = run.select(session=sid, stream=stream)
                if len(frames) < 30:
                    continue
                ids = [f["id"] for f in frames]
                steps = [b - a for a, b in zip(ids, ids[1:])]
                self.assertTrue(
                    all(s > 0 for s in steps),
                    f"{label}: {stream} ids not increasing: {steps[:40]}",
                )
                self.assertGreaterEqual(
                    sum(1 for s in steps if s == 1) / len(steps), 0.8,
                    f"{label}: {stream} ids skip: {steps[:60]}",
                )  # fmt: skip

    def _claims_are_never_ready_time(self, run: Run, label: str):
        """A claim is the mapped capture time, never the frame-ready time standing in for it."""
        for f in run.frames:
            if f["ts"] is None:
                continue
            # Ready time is the sof lag (20 ms) after capture: a substituted claim would show that much error.
            self.assertLess(
                abs(f["err_ns"]),
                MAX_BOUND_NS,
                f"{label}: frame {f['id']} claim off by {f['err_ns']} ns",
            )

    def _assert_all_withheld_without_stamp(self, frames: list[dict], label: str):
        stamped = [f for f in frames if f["ts"] is not None]
        self.assertFalse(
            stamped,
            f"{label}: {len(stamped)} of {len(frames)} frames in the window carry a timestamp: "
            f"{[(f['stream'], f['id'], round(f['t_rel'], 2)) for f in stamped[:8]]}",
        )

    # ---- scenarios ----------------------------------------------------------

    def test_1_accuracy(self):
        run = self._scenario(
            "accuracy", ["--sm-offset-ms", "1234", "--sm-drift-ppm", "50"], 20.0
        )
        self._frame_ids_continuous(run, "accuracy")
        self.assertGreaterEqual(len(run.frames), 100, run.summary())
        claims = run.errors(run.frames)
        self.assertGreaterEqual(
            len(claims), 100, f"too few claimed frames: {run.summary()}"
        )
        # Warm-up frames at the start have no claim; after it nearly every frame has one.
        steady = run.select(lo=3.0)
        stamped = [f for f in steady if f["ts"] is not None]
        self.assertGreaterEqual(len(stamped), 0.9 * len(steady), run.summary(steady))
        self.assertLess(abs(statistics.median(claims)), MEDIAN_BOUND_NS, run.summary())
        self.assertLess(
            _pctl(sorted(abs(x) for x in claims), 0.95), P95_BOUND_NS, run.summary()
        )
        self.assertLess(max(abs(x) for x in claims), MAX_BOUND_NS, run.summary())
        self._claims_are_never_ready_time(run, "accuracy")

    def test_2_step_recovery(self):
        step_s = 8.0
        # After a sender-clock step, frames mapped before the next exchange may carry the old
        # offset: a bounded exposure of one exchange interval (~200 ms) plus an RTT.
        exposure_s = 0.4
        run = self._scenario(
            "step", ["--sm-offset-ms", "1234", "--sm-step", f"{step_s:g}:250"], 24.0
        )
        self._frame_ids_continuous(run, "step")
        sid = run.session_order()[0]
        per_stream = {}
        for stream in TRACK_STREAM.values():
            frames = run.select(session=sid, stream=stream)
            wrong = [
                f
                for f in frames
                if f["err_ns"] is not None and abs(f["err_ns"]) > MAX_BOUND_NS
            ]
            outside = [
                f for f in wrong if not step_s <= f["t_rel"] <= step_s + exposure_s
            ]
            self.assertFalse(
                outside,
                f"{stream}: wrong claims outside the exposure window: "
                f"{[(f['id'], round(f['t_rel'], 2), round(f['err_ns'] / 1e3)) for f in outside[:8]]} us",
            )
            self.assertLess(
                len(wrong),
                15,
                f"{stream}: {len(wrong)} wrong claims inside the exposure window",
            )
            near = run.select(session=sid, stream=stream, lo=step_s, hi=step_s + 3.0)
            withheld = [f for f in near if f["ts"] is None]
            per_stream[stream] = (len(wrong), len(withheld))
            self.assertGreater(
                len(withheld),
                0,
                f"{stream}: nothing withheld after the step; {run.summary(near)}",
            )
            self.assertLess(
                len(withheld),
                3 * SOURCE_FPS,
                f"{stream}: {len(withheld)} withheld, more than 3 s worth",
            )
            before = run.select(session=sid, stream=stream, lo=3.0, hi=step_s - 0.5)
            self.assertTrue(
                before and all(f["ts"] is not None for f in before),
                f"{stream}: claims withheld well before the step; {run.summary(before)}",
            )
            after = run.select(session=sid, stream=stream, lo=step_s + 3.0)
            self.assertGreater(
                len(after), 60, f"{stream}: too few frames after the step"
            )
            self.assertTrue(
                all(f["ts"] is not None for f in after),
                f"{stream}: no recovery; {run.summary(after)}",
            )
            self.assertLess(max(abs(f["err_ns"]) for f in after), MAX_BOUND_NS)
            print(
                f"[step] {stream}: wrong in window {len(wrong)}, withheld {len(withheld)}, "
                f"{run.summary(after)}",
                file=sys.stderr,
            )

    def test_3_restart(self):
        run = self._scenario(
            "restart", ["--sm-offset-ms", "1234", "--disconnect-at", "6"], 22.0
        )
        self._frame_ids_continuous(run, "restart")
        self._no_wrong_claim(run, "restart")
        order = run.session_order()
        self.assertGreaterEqual(
            len(order), 2, f"the receiver did not reconnect with a new session: {order}"
        )
        for sid in order[1:]:
            for stream in TRACK_STREAM.values():
                warm = run.select(session=sid, stream=stream, lo=0.0, hi=2.0)
                if not warm:
                    continue
                self.assertTrue(
                    any(f["ts"] is None for f in warm),
                    f"{stream}: session {sid} claimed from its first frame, no warm-up; {run.summary(warm)}",
                )
                gap = [
                    f for f in run.select(session=sid, stream=stream) if f["ts"] is None
                ]
                self.assertLess(
                    len(gap), 4 * SOURCE_FPS, f"{stream}: warm-up gap too long"
                )
                later = run.select(session=sid, stream=stream, lo=3.0)
                if len(later) >= 20:
                    stamped = [f for f in later if f["ts"] is not None]
                    self.assertGreaterEqual(
                        len(stamped),
                        0.9 * len(later),
                        f"{stream}: no recovery; {run.summary(later)}",
                    )
        resumed = [
            f
            for sid in order[1:]
            for f in run.select(session=sid, lo=3.0)
            if f["ts"] is not None
        ]
        self.assertGreater(len(resumed), 20, "no claims resumed after a restart")
        print(
            f"[restart] sessions={len(order)} " + run.summary(resumed), file=sys.stderr
        )

    def test_4_stale_mapping(self):
        mute_start, mute_s = 5.0, 9.0
        run = self._scenario(
            "stale",
            ["--sm-offset-ms", "1234", "--clock-mute", f"{mute_start:g}+{mute_s:g}"],
            25.0,
        )
        self._frame_ids_continuous(run, "stale")
        self._no_wrong_claim(run, "stale")
        sid = run.session_order()[0]
        # The mapping goes stale 5 s after the last reply (about mute_start) and replies return at mute_start + mute_s.
        # The mute is timed from the mock's start, so it begins up to about a second before the session's t_rel 0 reference;
        # the window is inset to stay inside the stale interval for any such gap.
        stale_lo, stale_hi = mute_start + 5.0 + 0.5, mute_start + mute_s - 1.5
        window = run.select(session=sid, lo=stale_lo, hi=stale_hi)
        self.assertGreater(len(window), 30, run.summary(window))
        self._assert_all_withheld_without_stamp(window, "stale window")
        after = run.select(session=sid, lo=mute_start + mute_s + 4.0)
        self.assertGreater(len(after), 30, run.summary(after))
        stamped = [f for f in after if f["ts"] is not None]
        self.assertGreaterEqual(
            len(stamped), 0.9 * len(after), f"no recovery; {run.summary(after)}"
        )
        self.assertGreater(
            _stat_counts(run.bridge_log, "no_mapping"),
            0,
            f"no no_mapping in:\n{run.bridge_log[-3000:]}",
        )
        print(
            f"[stale] window {run.summary(window)}; after {run.summary(after)}",
            file=sys.stderr,
        )

    def test_5_loss_and_reorder(self):
        run = self._scenario(
            "lossy",
            [
                "--sm-offset-ms", "1234", "--sm-drift-ppm", "50",
                "--clock-loss", "0.3", "--clock-reorder", "0.2",
            ],
            20.0,
        )  # fmt: skip
        self._frame_ids_continuous(run, "lossy")
        claims = run.errors(run.frames)
        self.assertGreaterEqual(
            len(claims), 100, f"too few claimed frames: {run.summary()}"
        )
        steady = run.select(lo=5.0)
        stamped = [f for f in steady if f["ts"] is not None]
        self.assertGreaterEqual(len(stamped), 0.9 * len(steady), run.summary(steady))
        self.assertLess(abs(statistics.median(claims)), MEDIAN_BOUND_NS, run.summary())
        self.assertLess(
            _pctl(sorted(abs(x) for x in claims), 0.95), P95_BOUND_NS, run.summary()
        )
        self.assertLess(max(abs(x) for x in claims), 3_000_000, run.summary())
        self.assertGreater(
            _qualified_count(run.bridge_log),
            100,
            f"no qualified claims in:\n{run.bridge_log[-3000:]}",
        )

    def test_6_invalid_records(self):
        start_s, dur_s = 5.0, 3.0
        run = self._scenario(
            "invalid",
            ["--sm-offset-ms", "1234", "--timing-invalid", f"{start_s:g}+{dur_s:g}"],
            16.0,
        )
        self._frame_ids_continuous(run, "invalid")
        self._no_wrong_claim(run, "invalid")
        sid = run.session_order()[0]
        # Inset 0.1 s: a frame ready exactly at a window edge may fall on either side.
        inside = run.select(session=sid, lo=start_s + 0.1, hi=start_s + dur_s - 0.1)
        self.assertGreater(len(inside), 30, run.summary(inside))
        self._assert_all_withheld_without_stamp(inside, "invalid window")
        before = run.select(session=sid, lo=3.0, hi=start_s - 0.1)
        self.assertGreater(len(before), 30, run.summary(before))
        claimed = [f for f in before if f["ts"] is not None]
        self.assertGreaterEqual(
            len(claimed),
            0.9 * len(before),
            f"claims missing before the window; {run.summary(before)}",
        )
        after = run.select(session=sid, lo=start_s + dur_s + 2.0)
        self.assertGreater(len(after), 30, run.summary(after))
        stamped = [f for f in after if f["ts"] is not None]
        self.assertGreaterEqual(
            len(stamped), 0.9 * len(after), f"no recovery; {run.summary(after)}"
        )
        self.assertGreater(
            _stat_counts(run.bridge_log, "sender_invalid"),
            0,
            f"no sender_invalid in:\n{run.bridge_log[-3000:]}",
        )
        print(
            f"[invalid] inside {run.summary(inside)}; after {run.summary(after)}",
            file=sys.stderr,
        )

    def test_7_refusal_without_sender_clock(self):
        token = self._tmp / "refuse.token"
        token.write_text(make_token("unused", "capture-time-refuse", True))
        cmd = [
            str(BRIDGE_BIN), "--livekit-url", "ws://127.0.0.1:1", "--livekit-token-file", str(token),
            "--socket", str(self._tmp / "none.sock"), "--stream", "head=head_track",
            "--timestamp-basis", "capture_monotonic_us",
        ]  # fmt: skip
        started = time.monotonic()
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=30
        )
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(out.returncode, 2, out.stdout + out.stderr)
        self.assertIn("--sender-clock", out.stderr)
        self.assertIn("measured mapping", out.stderr)


if __name__ == "__main__":
    unittest.main()
