# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A scripted stand-in for ``manus_glove_calibrator`` speaking the worker contract, for CaptureServer tests.

Behavior comes from the JSON in ``CAL_FAKE_SCENARIO``; ``CAL_FAKE_LOG`` names a file that gets one line per
command the worker reads, so a test can count what reached it. Run as ``python calibration_fake_worker.py
--ipc-fd=N``. Scenario keys:

``steps``          list of ``{"title", "instruction", "time", "continuous", "estimated"}`` (default: a hold and
                   a 0.2 s motion). ``time`` > 0 is a timed motion; ``continuous`` models a negative SDK time.
``slow_confirm``   seconds to wait before accepting a confirmed step (still honors cancel).
``slow_cancel``    seconds to take stopping after a cancel.
``fail_at_step``   ``[side, index]``: report a failed result at that step after accepting it.
``unreleased``     true: every result reports ``released: false`` (the SDK could not be shut down).
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import sys
import time

SCENARIO = json.loads(os.environ.get("CAL_FAKE_SCENARIO", "{}"))
LOG_PATH = os.environ.get("CAL_FAKE_LOG")
STEPS = SCENARIO.get(
    "steps",
    [
        {"title": "Flat hand", "instruction": "Lay your hand flat.", "time": 0.0},
        {"title": "Wave", "instruction": "Wave your fingers.", "time": 0.2},
    ],
)
RELEASED = not SCENARIO.get("unreleased", False)
GLOVE_IDS = {"left": 11, "right": 22}


class Cancelled(Exception):
    pass


class Channel:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.attempt: str | None = None

    def send(self, message: dict) -> None:
        self.sock.sendall((json.dumps({"v": 1, **message}) + "\n").encode())

    def status(self, phase: str, gloves: list) -> None:
        self.send({"type": "status", "attempt": self.attempt, "phase": phase, "gloves": gloves})

    def poll(self, timeout: float) -> dict | None:
        while b"\n" not in self.buffer:
            ready, _, _ = select.select([self.sock], [], [], timeout)
            if not ready:
                return None
            chunk = self.sock.recv(4096)
            if not chunk:
                raise Cancelled
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        command = json.loads(line)
        if LOG_PATH:
            with open(LOG_PATH, "a", encoding="utf-8") as log:
                log.write(command["type"] + "\n")
        return command

    def sleep(self, seconds: float) -> None:
        """Sleep while still honoring a cancel; other commands read meanwhile are only logged."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            command = self.poll(0.05)
            if command is not None and command["type"] == "cancel":
                raise Cancelled

    def read(self, *types: str) -> dict:
        while True:
            command = self.poll(30.0)
            if command is None:
                raise SystemExit("fake worker timed out waiting for a command")
            if command["type"] == "cancel":
                raise Cancelled
            if command["type"] in types:
                return command


def prompt(channel: Channel, attempt: str, side: str, index: int, step: dict) -> str:
    token = f"{side}-{index}"
    channel.send(
        {
            "type": "prompt",
            "attempt": attempt,
            "token": token,
            "side": side,
            "glove_id": GLOVE_IDS[side],
            "index": index,
            "count": len(STEPS),
            "title": step["title"],
            "instruction": step["instruction"],
            "mode": "motion" if step["time"] > 0 else "hold",
            "duration_s": float(step["time"]),
            "continuous": bool(step.get("continuous", False)),
            "estimated_s": float(step.get("estimated", 0.0)),
        }
    )
    return token


def confirm(channel: Channel, attempt: str, token: str, step: dict) -> None:
    while True:
        command = channel.read("confirm")
        if command["token"] != token:
            channel.send(
                {
                    "type": "rejected",
                    "attempt": attempt,
                    "command": "confirm",
                    "reason": "stale_token",
                    "message": "not the offered prompt",
                    "token": command["token"],
                }
            )
            continue
        if SCENARIO.get("slow_confirm"):
            channel.sleep(SCENARIO["slow_confirm"])
        channel.send({"type": "step_accepted", "attempt": attempt, "token": token})
        break
    if step["time"] > 0:
        channel.status("capturing_step", [])
        elapsed = 0.0
        while elapsed < step["time"]:
            channel.sleep(0.1)
            elapsed = min(step["time"], elapsed + 0.1)
            channel.send({"type": "step_progress", "attempt": attempt, "token": token, "elapsed_s": elapsed})


def export(channel: Channel, attempt: str, out_dir: str, side: str) -> dict:
    data = (b"MCAL-" + side.encode()) * 64
    name = f"{attempt}_{side}.mcal"
    path = os.path.join(out_dir, name)
    with open(path + ".part", "wb") as handle:
        handle.write(data)
    os.rename(path + ".part", path)
    event = {
        "type": "exported",
        "attempt": attempt,
        "side": side,
        "glove_id": GLOVE_IDS[side],
        "file": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    channel.send(event)
    return event


def run(channel: Channel, start: dict) -> int:
    attempt = start["attempt"]
    channel.attempt = attempt
    sides = start["sides"]
    gloves = [{"side": s, "glove_id": GLOVE_IDS[s]} for s in sides]
    channel.status("discovering", [])
    channel.status("discovering", gloves)
    exported = []
    for side in sides:
        for index, step in enumerate(STEPS):
            token = prompt(channel, attempt, side, index, step)
            channel.status("awaiting_step", gloves)
            confirm(channel, attempt, token, step)
            if SCENARIO.get("fail_at_step") == [side, index]:
                channel.send(
                    {
                        "type": "result",
                        "attempt": attempt,
                        "outcome": "failed",
                        "released": RELEASED,
                        "error": {
                            "code": "step_failed",
                            "message": "the SDK refused the step",
                            "side": side,
                            "step": index,
                        },
                    }
                )
                return 1
        exported.append(export(channel, attempt, start["out_dir"], side))
    channel.status("saving", gloves)
    files = [{k: e[k] for k in ("side", "file", "bytes", "sha256")} for e in exported]
    channel.send(
        {"type": "result", "attempt": attempt, "outcome": "completed", "released": RELEASED, "files": files}
    )
    return 0


def main() -> int:
    fd = next(int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--ipc-fd="))
    channel = Channel(socket.socket(fileno=fd))
    channel.send({"type": "ready", "worker_version": "fake-1", "sdk_version": None})
    try:
        start = channel.read("start")
        return run(channel, start)
    except Cancelled:
        time.sleep(float(SCENARIO.get("slow_cancel", 0)))
        channel.status("stopping", [])
        channel.send({"type": "result", "attempt": channel.attempt, "outcome": "cancelled", "released": RELEASED})
        return 0


if __name__ == "__main__":
    sys.exit(main())
