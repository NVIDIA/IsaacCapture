# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces for the calibration integration tests: a scripted fake worker launched as a real process, raw
websocket helpers, and a driver that confirms prompts through the real CaptureClient API."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import websockets

from capture_server import protocol as p
from capture_server.calibration import CalibrationConfig
from capture_server.client import CalibrationStopped, CaptureClient
from capture_server.server import ServerConfig
from tests.support.capture_server import RunningServer, start_capture_server, stop_capture_server

FAKE_WORKER = Path(__file__).with_name("calibration_fake_worker.py")
LEFT, RIGHT = p.HandSide.LEFT, p.HandSide.RIGHT
BOTH = (LEFT, RIGHT)
CONTINUOUS_STEPS = [
    {
        "title": "Make a fist",
        "instruction": "Lift up your hand and make a fist.",
        "time": 0,
        "continuous": True,
        "estimated": 3.0,
    },
    {
        "title": "Pinch",
        "instruction": "Pinch your thumb and index finger.",
        "time": 0,
        "continuous": True,
        "estimated": 2.0,
    },
]


@dataclasses.dataclass
class Station:
    """A running CaptureServer with the fake calibration worker configured."""

    url: str
    port: int
    directory: Path
    log: Path
    running: RunningServer
    worker_path: Path

    @property
    def server(self):
        return self.running.server

    def commands(self) -> list[str]:
        """The command types that reached the fake worker, in order."""
        return self.log.read_text().split() if self.log.exists() else []


def install_fake_worker(tmp_path: Path, monkeypatch, scenario: dict | None) -> tuple[Path, Path]:
    """A wrapper executable that runs the fake worker with ``scenario``; returns (executable, command log)."""
    log = tmp_path / "worker-commands.log"
    monkeypatch.setenv("CAL_FAKE_SCENARIO", json.dumps(scenario or {}))
    monkeypatch.setenv("CAL_FAKE_LOG", str(log))
    wrapper = tmp_path / "fake-calibrator"
    wrapper.write_text(
        f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(FAKE_WORKER)!r}, run_name='__main__')\n"
    )
    wrapper.chmod(0o755)
    return wrapper, log


StationFactory = Callable[..., Awaitable[Station]]


async def start_station(
    tmp_path: Path,
    monkeypatch,
    port: int,
    *,
    scenario: dict | None = None,
    calibration: bool = True,
    config: ServerConfig | None = None,
) -> Station:
    worker, log = install_fake_worker(tmp_path, monkeypatch, scenario)
    directory = tmp_path / "managed-mcal"
    base = config or ServerConfig()
    if calibration:
        base = dataclasses.replace(base, calibration=CalibrationConfig(worker_path=worker, directory=directory))
    running = await start_capture_server(port, config=base)
    return Station(f"ws://127.0.0.1:{port}", port, directory, log, running, worker)


async def stop_station(station: Station) -> None:
    await stop_capture_server(station.running)


# ---------------------------------------------------------------------------
# Raw websocket
# ---------------------------------------------------------------------------


async def connect(url: str):
    return await websockets.connect(url)


async def send(ws, message: dict | Any) -> None:
    await ws.send(json.dumps(message.to_json() if hasattr(message, "to_json") else message))


async def recv_event(ws, timeout: float = 5.0):
    """The next JSON message parsed into its typed event; binary capture frames are skipped."""
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(raw, bytes):
            continue
        return p.parse_server_message(json.loads(raw))


async def recv_until(ws, predicate, timeout: float = 10.0):
    """Read events until one matches ``predicate``; returns (match, everything read so far)."""
    seen: list = []

    async def _read():
        while True:
            event = await recv_event(ws, timeout)
            seen.append(event)
            if predicate(event):
                return event

    return await asyncio.wait_for(_read(), timeout), seen


def is_type(*types):
    return lambda event: isinstance(event, types)


def prompt_status(event) -> bool:
    return (
        isinstance(event, p.CalibrationStatus)
        and event.phase is p.CalibrationPhase.AWAITING_STEP
        and event.step is not None
    )


async def start_over_raw(ws, session_id: str = "cal-1", sides=(LEFT,)):
    await send(ws, p.StartCalibrationRequest(session_id, tuple(sides)))
    return await recv_event(ws)


async def next_prompt(ws, timeout: float = 10.0) -> p.CalibrationStatus:
    status, _ = await recv_until(ws, prompt_status, timeout)
    return status


async def terminal_of(ws, timeout: float = 15.0) -> p.CalibrationStopped:
    stopped, _ = await recv_until(ws, is_type(p.CalibrationStopped), timeout)
    return stopped


# ---------------------------------------------------------------------------
# Client driver
# ---------------------------------------------------------------------------


class CalibrationRun:
    """Runs one calibration through the real CaptureClient, confirming each prompt unless told not to."""

    def __init__(self, url: str, *, auto_confirm: bool = True) -> None:
        self.client = CaptureClient(url)
        self.auto_confirm = auto_confirm
        self.started: list[p.CalibrationStarted] = []
        self.statuses: list[p.CalibrationStatus] = []
        self.stopped: list[CalibrationStopped] = []
        self.finished = asyncio.Event()
        self.confirmed: list[str] = []
        self.errors: list[BaseException] = []
        self._seen: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    def prompts(self) -> list[p.CalibrationStep]:
        """Each distinct offered step, in order."""
        out: dict[str, p.CalibrationStep] = {}
        for status in self.statuses:
            if status.step is not None and status.phase is p.CalibrationPhase.AWAITING_STEP:
                out.setdefault(status.step.token, status.step)
        return list(out.values())

    def _on_status(self, status: p.CalibrationStatus) -> None:
        self.statuses.append(status)
        step = status.step
        if not self.auto_confirm or step is None or status.phase is not p.CalibrationPhase.AWAITING_STEP:
            return
        if step.token in self._seen:
            return
        self._seen.add(step.token)
        task = asyncio.ensure_future(self._confirm(step.token))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _confirm(self, token: str) -> None:
        try:
            await self.client.confirm_calibration_step(step_token=token)
            self.confirmed.append(token)
        except Exception as exc:  # noqa: BLE001 -- recorded for the test to assert on
            self.errors.append(exc)

    async def start(self, sides=(LEFT,)) -> None:
        await self.client.start_calibration(
            sides=sides,
            on_started=self.started.append,
            on_status=self._on_status,
            on_stopped=lambda s: (self.stopped.append(s), self.finished.set()),
        )

    async def wait(self, timeout: float = 20.0) -> CalibrationStopped:
        await asyncio.wait_for(self.finished.wait(), timeout)
        return self.stopped[0]

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self.client.stop_calibration()
        for task in list(self._tasks):
            task.cancel()
