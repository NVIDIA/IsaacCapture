#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sensing_mock_producer end to end, through the management channel and a real video_stage_worker.

Binaries come from SENSING_MOCK_PRODUCER and VIDEO_STAGE_WORKER, else tmp/build-vs under the repository root; the
input stream from SENSING_E2E_INPUT, else tmp/vs/orin_like.h264. The test is skipped when any of them is missing.
The worker writes to files (no LiveKit), on ports 5730 and 5731.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "tools"))

import orin_control  # noqa: E402

LEFT_PORT, RIGHT_PORT = 5730, 5731
CLOSED_PORT = 5799  # nothing listens here


def _binary(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or REPO / "tmp" / "build-vs" / default)


PRODUCER = _binary(
    "SENSING_MOCK_PRODUCER", "src/sensing_control/mock_producer/sensing_mock_producer"
)
WORKER = _binary("VIDEO_STAGE_WORKER", "src/video_stage/app/video_stage_worker")
INPUT = Path(
    os.environ.get("SENSING_E2E_INPUT") or REPO / "tmp" / "vs" / "orin_like.h264"
)


def _missing() -> str | None:
    for what, path in (
        ("sensing_mock_producer", PRODUCER),
        ("video_stage_worker", WORKER),
        ("input stream", INPUT),
    ):
        if not path.is_file():
            return f"{what} not found at {path} (set SENSING_MOCK_PRODUCER, VIDEO_STAGE_WORKER, SENSING_E2E_INPUT)"
    return None


def _wait(predicate, timeout: float, interval: float = 0.1):
    """Poll until predicate() returns something truthy; return it, or None on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


@unittest.skipIf(_missing() is not None, _missing() or "")
class MockProducerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.work = Path(tempfile.mkdtemp(prefix="sensing_e2e_", dir=REPO / "tmp"))
        cls.token = "e2e-token-" + os.urandom(8).hex()
        token_file = cls.work / "token.txt"
        token_file.write_text(cls.token + "\n")
        token_file.chmod(0o600)
        cls.token_file = token_file
        cls.out = {
            "left": cls.work / "e2e_left.h264",
            "right": cls.work / "e2e_right.h264",
        }
        cls.procs: list[subprocess.Popen] = []
        cls.started = time.monotonic()

        producer_log = (cls.work / "producer.log").open("w")
        cls.producer = subprocess.Popen(
            [
                str(PRODUCER),
                "--control-port=0",
                f"--control-token-file={token_file}",
                f"--stream=sensing-left={INPUT}",
                f"--stream=sensing-right={INPUT}",
                "--simulate-fault-seconds=3",
                "--retry-seconds=1",
            ],
            stdout=producer_log,
            stderr=subprocess.STDOUT,
        )
        cls.procs.append(cls.producer)

        def announced_port() -> int | None:
            for line in (
                (cls.work / "producer.log").read_text(errors="replace").splitlines()
            ):
                if "management channel on" in line:
                    return int(line.rsplit(":", 1)[1])
            return None

        port = _wait(announced_port, 10)
        if port is None:
            cls.tearDownClass()
            raise RuntimeError(
                "the producer did not announce its management channel:\n"
                + cls.log("producer.log")
            )
        cls.port = port

        worker_log = (cls.work / "worker.log").open("w")
        cls.worker = subprocess.Popen(
            [
                str(WORKER),
                f"--stream={LEFT_PORT}:file={cls.out['left']}",
                f"--stream={RIGHT_PORT}:file={cls.out['right']}",
            ],
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        cls.procs.append(cls.worker)

    @classmethod
    def tearDownClass(cls) -> None:
        for proc in cls.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in cls.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    @classmethod
    def log(cls, name: str) -> str:
        path = cls.work / name
        return path.read_text(errors="replace")[-3000:] if path.exists() else ""

    # -- helpers ----------------------------------------------------------------------------------------------

    def connect(self, token: str | None = None) -> orin_control.Connection:
        return orin_control.Connection("127.0.0.1", self.port, token or self.token)

    def request(self, message: dict) -> dict:
        conn = self.connect()
        try:
            return conn.request(message)
        finally:
            conn._sock.close()

    def status(self) -> dict:
        reply = self.request({"type": "get_status"})
        self.assertTrue(reply["ok"], reply)
        return reply["status"]

    def set_target(self, stream: str, host: str, port: int = 0) -> dict:
        return self.request(
            {"type": "set_direct_target", "stream": stream, "host": host, "port": port}
        )

    def wait_streaming(self) -> dict:
        status = _wait(lambda: (s := self.status())["state"] == "streaming" and s, 30)
        self.assertIsNotNone(
            status, "the producer never reached streaming:\n" + self.log("producer.log")
        )
        return status

    def stream_of(self, status: dict, name: str) -> dict:
        return next(s for s in status["streams"] if s["stream"] == name)

    def wait_stable(self, path: Path) -> int:
        """The size once it stops changing."""
        last = [-1]

        def stable() -> int | None:
            size = _size(path)
            if size == last[0]:
                return size or None
            last[0] = size
            time.sleep(0.5)
            return None

        value = _wait(stable, 15, interval=0.0)
        self.assertIsNotNone(value, f"{path.name} never stopped growing")
        return value

    # -- the scenario, in order -------------------------------------------------------------------------------

    def test_1_fault_then_streaming(self) -> None:
        # The simulated unpowered camera is reported in plain words first.
        fault = _wait(lambda: (s := self.status())["state"] == "camera_fault" and s, 3)
        if fault is None and time.monotonic() - self.started > 3:
            self.skipTest("the simulated fault window had already passed")
        self.assertIsNotNone(
            fault, "never saw camera_fault:\n" + self.log("producer.log")
        )
        self.assertIn("power cable", fault["fault"]["message"])
        self.assertEqual(fault["fault"]["code"], "camera_start_failed")
        self.assertEqual(
            [s["stream"] for s in fault["streams"]], ["sensing-left", "sensing-right"]
        )

        streaming = self.wait_streaming()
        self.assertNotIn("fault", streaming)
        self.assertGreaterEqual(
            time.monotonic() - self.started,
            2.5,
            "the fault window was shorter than asked",
        )

    def test_2_set_target_streams_to_the_worker(self) -> None:
        self.wait_streaming()
        left = self.out["left"]
        reply = self.set_target("sensing-left", "127.0.0.1", LEFT_PORT)
        self.assertTrue(reply["ok"], reply)
        target = self.stream_of(reply["status"], "sensing-left")["direct"]
        self.assertEqual((target["host"], target["port"]), ("127.0.0.1", LEFT_PORT))

        grew = _wait(lambda: _size(left) > 100_000, 20)
        self.assertTrue(
            grew, "the worker's file did not grow:\n" + self.log("worker.log")
        )
        before = _size(left)
        self.assertTrue(
            _wait(lambda: _size(left) > before + 100_000, 10),
            "the file stopped growing",
        )

        connected = _wait(
            lambda: self.stream_of(self.status(), "sensing-left")["direct"][
                "connected"
            ],
            10,
        )
        self.assertTrue(
            connected, "the status never reported the direct link as connected"
        )
        self.assertFalse(
            self.stream_of(self.status(), "sensing-right")["direct"]["host"]
        )

        if shutil.which("ffprobe"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "csv=p=0",
                    str(left),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertGreater(int(probe.stdout.strip().strip(",") or 0), 0)

    def test_3_second_stream_is_independent(self) -> None:
        self.wait_streaming()
        right = self.out["right"]
        self.assertTrue(self.set_target("sensing-right", "127.0.0.1", RIGHT_PORT)["ok"])
        self.assertTrue(
            _wait(lambda: _size(right) > 100_000, 20), self.log("worker.log")
        )

    def test_4_empty_host_stops_the_link(self) -> None:
        self.wait_streaming()
        left = self.out["left"]
        reply = self.set_target("sensing-left", "")
        self.assertTrue(reply["ok"], reply)
        direct = self.stream_of(reply["status"], "sensing-left")["direct"]
        self.assertEqual((direct["host"], direct["port"]), ("", 0))

        settled = self.wait_stable(left)
        time.sleep(1.5)
        self.assertEqual(
            _size(left), settled, "the file kept growing after the link was turned off"
        )
        self.assertFalse(
            self.stream_of(self.status(), "sensing-left")["direct"]["connected"]
        )
        # The other stream is unaffected.
        right = self.out["right"]
        before = _size(right)
        self.assertTrue(
            _wait(lambda: _size(right) > before, 10), "the right stream stopped too"
        )

    def test_5_errors(self) -> None:
        self.wait_streaming()
        with self.assertRaises(orin_control.ChannelError) as refused:
            self.connect(token="not-the-token")
        self.assertIn("401", str(refused.exception))

        reply = self.set_target("sensing-up", "127.0.0.1", LEFT_PORT)
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["error"]["code"], "unknown_stream")
        self.assertIn("sensing-left", reply["error"]["message"])

        self.assertEqual(
            self.set_target("sensing-left", "bad host", 1)["error"]["code"],
            "invalid_host",
        )
        self.assertEqual(
            self.set_target("sensing-left", "127.0.0.1", 0)["error"]["code"],
            "invalid_port",
        )

        # The command-line tool agrees.
        cli = [
            sys.executable,
            str(HERE.parent / "tools" / "orin_control.py"),
            f"127.0.0.1:{self.port}",
            "--token-file",
            str(self.token_file),
        ]
        ok = subprocess.run(
            cli + ["status"], capture_output=True, text=True, timeout=20, check=False
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("state: streaming", ok.stdout)
        bad = subprocess.run(
            cli + ["set-target", "nope", "127.0.0.1", "5730"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        self.assertEqual(bad.returncode, 1)
        self.assertIn("unknown_stream", bad.stderr)

    def test_6_unreachable_target_reports_the_error(self) -> None:
        status = self.wait_streaming()
        # Every stream carries the key, empty while nothing is wrong.
        for stream in status["streams"]:
            self.assertIn("error", stream["direct"])
        self.assertEqual(self.stream_of(status, "sensing-right")["direct"]["error"], "")

        watcher = self.connect()
        reply = self.set_target("sensing-left", "127.0.0.1", CLOSED_PORT)
        self.assertTrue(reply["ok"], reply)
        self.assertIn(
            "error", self.stream_of(reply["status"], "sensing-left")["direct"]
        )

        # The first error to appear after the retarget must already name the new destination.
        wanted = f"cannot connect to 127.0.0.1:{CLOSED_PORT}"

        def error_text() -> str | None:
            error = self.stream_of(self.status(), "sensing-left")["direct"]["error"]
            return error or None

        error = _wait(error_text, 15)
        self.assertIsNotNone(
            error, "direct.error never appeared:\n" + self.log("producer.log")
        )
        self.assertTrue(error.startswith(wanted), error)
        direct = self.stream_of(self.status(), "sensing-left")["direct"]
        self.assertFalse(direct["connected"])
        self.assertEqual((direct["host"], direct["port"]), ("127.0.0.1", CLOSED_PORT))
        # The other stream is unaffected.
        self.assertEqual(
            self.stream_of(self.status(), "sensing-right")["direct"]["error"], ""
        )
        # The failure is also pushed once to connected clients as an error event.
        events = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                message = watcher.receive(timeout=1.0)
            except (TimeoutError, OSError):
                continue
            if message.get("event") == "error":
                events.append(message)
                if message["error"]["code"] == "direct_link_failed":
                    break
        watcher._sock.close()
        direct_events = [
            e for e in events if e["error"]["code"] == "direct_link_failed"
        ]
        self.assertEqual(len(direct_events), 1, events)
        self.assertIn("sensing-left", direct_events[0]["error"]["message"])
        self.assertIn("cannot connect to", direct_events[0]["error"]["message"])

        # Turning the link off clears the error.
        self.assertTrue(self.set_target("sensing-left", "")["ok"])
        self.assertTrue(
            _wait(
                lambda: (
                    self.stream_of(self.status(), "sensing-left")["direct"]["error"]
                    == ""
                ),
                5,
            ),
            "the error was not cleared",
        )

    def test_9_producer_stops_on_sigint(self) -> None:
        self.producer.send_signal(signal.SIGINT)
        self.assertEqual(self.producer.wait(timeout=15), 0, self.log("producer.log"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
