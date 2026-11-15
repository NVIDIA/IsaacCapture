# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A robot app, a robot link service and local clients joined by a real link.

For the tests marked "link". The robot app dials 127.0.0.1 on a port the system
picked; a service that is started again listens on that port again. No dial is
made sooner than RETRY_S after the last one.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from isaaccapture_examples.agentic_remote.link_service.listen import RobotListener
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalCarrier,
    connect_local,
)
from isaaccapture_examples.agentic_remote.link_service.service import RobotLinkService
from isaaccapture_examples.agentic_remote.protocol.link_carrier import LinkCarrier
from isaaccapture_examples.agentic_remote.protocol.messages import Result
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.connection import RobotDialer
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from repo_paths import repo_root

from conftest import AppHarness

# Real seconds a test gives another thread, or a process, before it fails.
WAIT_S = 10.0
# From a failed dial, or a lost robot connection, to the next dial.
RETRY_S = 0.3
# One dial's time limit.
CONNECT_S = 2.0
# The link's peer timeout: without a link message for this long a link dies.
PEER_TIMEOUT_S = 3.0
HOST = "127.0.0.1"
SERVICE_MODULE = "isaaccapture_examples.agentic_remote.link_service"
READY = "robot link service: waits for the robot app on "


def until(done: Callable[[], Any], what: str, wait_s: float = WAIT_S) -> None:
    """Wait in real time until done() holds."""
    end = time.monotonic() + wait_s
    while not done():
        assert time.monotonic() < end, f"{what} did not come true"
        time.sleep(0.005)


class Station:
    """One local client: its end of the local API and a WorkstationClient on it."""

    def __init__(self, end: LocalCarrier, ws: WorkstationClient, role: str) -> None:
        self.end, self.ws, self.role = end, ws, role

    def close(self) -> None:
        self.ws.close()
        self.end.close()

    def kill(self) -> None:
        """Close the socket at once, as the end of the client's process does."""
        self.end.close()
        self.ws.close()

    def final(self, command_id: str) -> Result:
        """A command's final RESULT and, from the Thor, the ROBOT_STATE showing it.

        With that state the client knows the robot app's clock as it stands
        after the command. A RESULT can arrive before an earlier ROBOT_STATE.
        """
        result = self.ws.wait_result(command_id, timeout_s=WAIT_S)
        self.ws.wait_state(
            lambda s: (s.last_result or {}).get("command_id") == command_id, WAIT_S
        )
        return result


class Stack:
    """The robot app of a harness, dialing a robot link service over a real link.

    The harness is built unconnected (connected=False) and started. The service
    runs in this process unless a test puts a ServiceProcess in its place.
    """

    def __init__(self, harness: AppHarness, folder: Path) -> None:
        self.h = harness
        self.path = str(folder / "s.sock")
        assert len(self.path) < 100, f"socket path too long: {self.path}"
        self.port = 0  # the service's TCP port, once one was picked
        self.listener: RobotListener | None = None
        self.service: RobotLinkService | None = None
        self.dialer: RobotDialer | None = None
        self.stations: list[Station] = []
        # While True no dial is made: each one fails before it opens a connection.
        self.held = False
        self._dialers: list[RobotDialer] = []

    # The service

    def start_service(self, **options: Any) -> RobotLinkService:
        """Start a service in this process, on the port of the one before it."""
        self.listener = RobotListener(HOST, self.port)
        self.port = self.listener.port
        options = {"status_period_s": 0.05, "poll_s": 0.01, **options}
        self.service = RobotLinkService(
            self.listener.accept, socket_path=self.path, **options
        )
        self.service.start()
        return self.service

    def stop_service(self) -> None:
        """Close the service of this process and stop listening."""
        self.service.close()
        self.listener.close()
        self.service = self.listener = None

    # The robot app's side

    def dial(self, harness: AppHarness | None = None) -> RobotDialer:
        """Start a dialer for the harness's app; it dials 127.0.0.1 on self.port."""
        if harness is not None:
            self.h = harness
        assert self.port, "no service was started"
        self.dialer = RobotDialer(
            self.h.app, self._connect, SystemClock(), retry_s=RETRY_S
        )
        self._dialers.append(self.dialer)
        self.dialer.start()
        return self.dialer

    def _connect(self) -> LinkCarrier:
        if self.held:
            raise ConnectionError("the test holds the dialing")
        return LinkCarrier.connect(f"{HOST}:{self.port}", CONNECT_S, name="robot-app")

    def connection_events(self) -> int:
        """How many "connection" events, made or lost, the app has emitted."""
        return len(self.h.heard("connection"))

    def cut(self) -> None:
        """Close the robot app's end of the link; wait until the app handled it."""
        events = self.connection_events()
        self.dialer.carrier.close()
        self.h.wait_local_events("connection", events + 1)

    # Local clients

    def client(self, role: str, name: str = "") -> Station:
        """Attach a client in this role; its commands quote the app's numbers."""
        end = connect_local(self.path, role, name or f"test-{role}", WAIT_S)
        ws = WorkstationClient(end, self.h.clock, name=role, quotable=self._current)
        station = Station(end, ws, role)
        self.stations.append(station)
        return station

    def connected(self, station: Station, session_id: str | None = None) -> None:
        """Wait until the service tells this client that the robot connection is up.

        Then the client has the WELCOME and a ROBOT_STATE of the app's session.
        """
        session_id = session_id or self.h.app.session_id
        station.end.wait_status(
            lambda s: s.robot_connected and s.session_id == session_id, WAIT_S
        )
        station.ws.wait_until(
            lambda: (
                station.ws.welcome is not None
                and station.ws.welcome.session_id == session_id
            ),
            WAIT_S,
            "the WELCOME",
        )
        station.ws.wait_state(
            lambda s: s.session_id == session_id and self._current(s), WAIT_S
        )

    def agent(self) -> Station:
        """An agent client that is the harness's client, connected to the robot."""
        station = self.client("agent")
        self.connected(station)
        self.h.client = station.ws
        return station

    def _current(self, state: Any) -> bool:
        return self.h._shows_the_apps_numbers(state)

    def close(self) -> None:
        for station in self.stations:
            station.close()
        for dialer in self._dialers:
            dialer.close()
        if self.service is not None:
            self.stop_service()


class Program:
    """A program of the example in its own process: python -m <module> <args>.

    Its standard input is a pipe the test types into.
    """

    def __init__(self, module: str, *args: str) -> None:
        self.lines: list[str] = []  # what the process wrote, both streams
        self._new: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self.process = subprocess.Popen(
            [sys.executable, "-m", module, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=program_env(),
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self._new.put(line.rstrip("\n"))
        self._new.put(None)

    def type(self, line: str) -> None:
        """Type one line at the program's standard input."""
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def wait_line(self, part: str, wait_s: float = WAIT_S) -> str:
        """The next line the process writes that contains `part`."""
        end = time.monotonic() + wait_s
        while True:
            try:
                line = self._new.get(timeout=max(0.0, end - time.monotonic()))
            except queue.Empty:
                line = None
            assert line is not None, f"no line with {part!r}; got {self.lines}"
            self.lines.append(line)
            if part in line:
                return line

    def signal(self, number: int) -> None:
        self.process.send_signal(number)

    def wait_exit(self) -> int:
        """Wait for the process to end; its exit code. Every line is in `lines`."""
        code = self.process.wait(WAIT_S)
        self._reader.join(WAIT_S)
        while (line := self._new.get()) is not None:
            self.lines.append(line)
        self._new.put(None)
        return code

    def close(self) -> None:
        """End the process if it still runs: SIGKILL, also when it is stopped."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
            self.process.wait(WAIT_S)
        self._reader.join(WAIT_S)
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass


class ServiceProcess(Program):
    """The robot link service as a program: python -m ...link_service --config."""

    def __init__(self, folder: Path, path: str, port: int = 0) -> None:
        """The process listens on 127.0.0.1 and `port`; 0 lets the system pick."""
        self.path = path
        self.port = port
        self.config = folder / f"link_service_{port}.yaml"
        settings = {
            "listen": {"address": HOST, "port": port},
            "allowed_peers": [HOST],
            "socket_path": path,
            "log_level": "info",
        }
        self.config.write_text(yaml.safe_dump(settings), encoding="utf-8")
        super().__init__(SERVICE_MODULE, "--config", str(self.config))

    def wait_ready(self) -> int:
        """Wait until the service listens; the TCP port it listens on."""
        line = self.wait_line(READY)
        endpoint = line.removeprefix(READY).split()[0]
        self.port = int(endpoint.rpartition(":")[2])
        return self.port


def program_env() -> dict[str, str]:
    """The environment of a program of the example: this process's import path."""
    import isaaccapture

    package = repo_root() / "examples" / "agentic_remote" / "python"
    build = Path(isaaccapture.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(package), str(build)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
