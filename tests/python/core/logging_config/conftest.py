# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Child environments, log-file lookup, and a real leader receiver.

Much of this suite runs in child processes rather than in fixtures. ``install()``
runs once per process and C++'s ``local_sinks()`` resolves the environment once
per process, so a permutation of the six variables *is* a process; there is no
in-process way to ask either half the same question twice. This module holds the
pieces every such test needs: an environment with the contract variables removed
(pytest inherits a live set of them, having become a leader on import), the file
naming both halves promise, and the leader side of the forwarding socket.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The whole external contract, as the root AGENTS.md lists it; a variable
#: missing here leaks from the calling shell into every child a test starts.
CONTRACT_ENV_VARS = (
    "ISAACCAPTURE_LOGGING",
    "ISAACCAPTURE_LOG_DIR",
    "ISAACCAPTURE_LOG_LEVEL",
    "ISAACCAPTURE_LOG_SOCKET",
    "ISAACCAPTURE_NATIVE_CAPTURE",
    "ISAACCAPTURE_NATIVE_CAPTURE_FILE",
)

#: What ``LINE_FORMAT`` renders, whichever half rendered it. ``test_routing``
#: holds this against a line from each.
LINE_RE = re.compile(
    r"^\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(?P<msecs>\d{3})\] "
    r"\[(?P<level>[A-Z]+ *)\] \[(?P<name>[^]]+)\] \[pid:(?P<pid>\d+)\] (?P<message>.*)$"
)

#: A C++ console line: ``LINE_FORMAT`` without the date, which only files keep.
CPP_CONSOLE_LINE_RE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})\.(?P<msecs>\d{3})\] "
    r"\[(?P<level>[A-Z]+ *)\] \[(?P<name>[^]]+)\] \[pid:(?P<pid>\d+)\] (?P<message>.*)$"
)

_EMITTER = os.environ.get("LOG_BRIDGE_TEST_EMITTER", "")

#: The standalone C++ emitter is built by tests/cpp/core/log_bridge and located
#: through the CMake test wiring, so a bare `pytest` run skips what needs it.
requires_emitter = pytest.mark.skipif(
    not (_EMITTER and Path(_EMITTER).is_file()),
    reason="LOG_BRIDGE_TEST_EMITTER is set by the CMake test wiring",
)


def emitter_path() -> str:
    return _EMITTER


def clean_env(log_dir: Path | str, **overrides: str) -> dict[str, str]:
    """This process's environment with the contract variables replaced."""
    env = {k: v for k, v in os.environ.items() if k not in CONTRACT_ENV_VARS}
    env["ISAACCAPTURE_LOG_DIR"] = str(log_dir)
    env.update(overrides)
    return env


def run_python(code: str, env: dict[str, str], *args: str, timeout: float = 180):
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_emitter(args: list[str], env: dict[str, str], timeout: float = 60):
    return subprocess.run(
        [emitter_path(), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _matching(log_dir: Path, pattern: str) -> list[Path]:
    if not log_dir.is_dir():
        return []
    return sorted(p for p in log_dir.iterdir() if re.fullmatch(pattern, p.name))


def session_logs(log_dir: Path) -> list[Path]:
    """``<timestamp>.isaaccapture.<pid>.log`` -- the Python half's own files."""
    return _matching(log_dir, r"[\d-]+\.isaaccapture\.\d+\.log")


def cpp_logs(log_dir: Path) -> list[Path]:
    """``<timestamp>.isaaccapture.<pid>.cpp[-N].log`` -- what local_sinks() opens."""
    return _matching(log_dir, r"[\d-]+\.isaaccapture\.\d+\.cpp(-\d+)?\.log")


def capture_logs(log_dir: Path) -> list[Path]:
    """``<timestamp>.isaaccapture.<pid>.native.log`` -- raw fd 1/2 output."""
    return _matching(log_dir, r"[\d-]+\.isaaccapture\.\d+\.native\.log")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_all(paths: list[Path]) -> str:
    return "".join(read(p) for p in paths)


def wait_until(predicate, timeout: float = 15.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class RecordCollector(logging.Handler):
    """Every record the receiver re-emits into this process's logger tree."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def named(self, name: str) -> list[logging.LogRecord]:
        return [r for r in list(self.records) if r.name == name]

    def wait_for(self, name: str, count: int, timeout: float = 60.0):
        wait_until(lambda: len(self.named(name)) >= count, timeout)
        return self.named(name)


@dataclass
class Receiver:
    """The leader half of the forwarding transport, for a child to ship to."""

    path: str
    collector: RecordCollector
    server: object
    _stopped: bool = False

    def stop(self) -> None:
        """Take the leader away mid-run, as a killed session leader does."""
        if self._stopped:
            return
        self._stopped = True
        self.server.shutdown()
        self.server.server_close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


@pytest.fixture
def receiver():
    """A leader's receiver built from the production server and handler.

    Bound under a fresh short directory, not ``tmp_path``: pytest's is long
    enough to reach the 103-byte ``sun_path`` limit the code refuses at.
    """
    from isaaccapture.logging_config import _forwarding

    directory = tempfile.mkdtemp(prefix="itlog", dir="/tmp")
    path = os.path.join(directory, "r.sock")
    server = _forwarding.ThreadingUnixStreamServer(path, _forwarding.RequestHandler)
    collector = RecordCollector()
    root = logging.getLogger("isaaccapture")
    root.addHandler(collector)
    threading.Thread(
        target=server.serve_forever, name="test-log-receiver", daemon=True
    ).start()
    instance = Receiver(path, collector, server)
    try:
        yield instance
    finally:
        root.removeHandler(collector)
        instance.stop()
        shutil.rmtree(directory, ignore_errors=True)


# A session leader is a process, not an object: install() decides leader or child
# once, from the environment it started with.
_LEADER_CODE = """
import json, os, sys, time
import isaaccapture  # noqa: F401 -- importing it is what installs everything
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "socket": os.environ.get("ISAACCAPTURE_LOG_SOCKET"),
            "capture": os.environ.get("ISAACCAPTURE_NATIVE_CAPTURE_FILE"),
            "pid": os.getpid(),
        },
        handle,
    )
time.sleep(600)
"""


@dataclass
class Leader:
    """A live session leader: real console, real rotating file, real receiver."""

    process: subprocess.Popen
    log_dir: Path
    socket: str
    capture: str
    pid: int
    console_path: Path

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=60)

    def logged(self) -> str:
        return read_all(session_logs(self.log_dir))

    def console(self) -> str:
        return read(self.console_path) if self.console_path.is_file() else ""

    def wait_for(self, needle: str, timeout: float = 30.0) -> str:
        wait_until(lambda: needle in self.logged(), timeout)
        return self.logged()


@pytest.fixture
def leader(tmp_path):
    log_dir = tmp_path / "leader-logs"
    log_dir.mkdir()
    report = tmp_path / "leader.json"
    console_path = tmp_path / "leader.console"
    with open(console_path, "wb") as console:
        process = subprocess.Popen(
            [sys.executable, "-c", _LEADER_CODE, str(report)],
            env=clean_env(log_dir),
            stdout=console,
            stderr=subprocess.STDOUT,
        )
    try:
        started = wait_until(
            lambda: report.is_file() and report.stat().st_size > 0, timeout=180
        )
        assert started, f"the leader never reported: {read(console_path)}"
        published = json.loads(report.read_text(encoding="utf-8"))
        assert published["socket"], "the leader could not start a receiver"
        yield Leader(
            process,
            log_dir,
            published["socket"],
            published["capture"],
            published["pid"],
            console_path,
        )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=60)
