# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted stand-in for ``manus_glove_calibrator`` that speaks the worker contract.

Behavior comes from the JSON object in ``FAKE_WORKER_SCENARIO``; ``FAKE_WORKER_LOG`` names a file
that receives one line per command the fake reads, so tests can count what reached the worker.
A step may set ``"continuous": true`` with ``"estimated": SECONDS`` (and ``"time": 0``) to model the SDK's
continuous step, which is a hold-paced prompt that carries an estimated duration.
``reject_start`` refuses the start command and stays idle; ``idle_cancel_result`` makes SIGTERM before ``start``
report a cancelled result with no attempt id.
Hang scenarios keep the process alive and silent (``ignore_term`` makes it survive SIGTERM too):
``hang_after_start``, ``hang_after_discovering``, ``hang_on_confirm`` and ``hang_after_last_accept``.
``chatty_discovery`` and ``chatty_finish`` (seconds) keep emitting status events while the worker is busy.
Every ``result`` carries ``released``: true unless the scenario sets ``not_released``, which models a worker
whose SDK calibration could not be stopped or whose SDK could not be shut down.
Run as ``python fake_worker.py --ipc-fd=N``.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import socket
import sys
import time

SCENARIO = json.loads(os.environ.get("FAKE_WORKER_SCENARIO", "{}"))
LOG_PATH = os.environ.get("FAKE_WORKER_LOG")
STEPS = SCENARIO.get(
    "steps",
    [
        {"title": "Flat hand", "instruction": "Lay your hand flat.", "time": 0.0},
        {"title": "Wave", "instruction": "Wave your fingers.", "time": 0.4},
    ],
)
GLOVE_IDS = {"left": 11, "right": 22}


class Cancelled(Exception):
    pass


def released() -> bool:
    """The `released` flag every result carries."""
    return not SCENARIO.get("not_released")


def hang() -> None:
    """Stay alive and silent, as a worker blocked inside the SDK would."""
    time.sleep(60)


def chatter(channel: "Channel", phase: str, seconds: float, gloves: list) -> None:
    """Emit a status event every 0.2 s for ``seconds``, modelling a slow but live worker."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        channel.status(phase, gloves)
        channel.sleep(0.2)


# Set once the first confirmation has been refused by the `reject_first_confirm` scenario.
_REJECTED_ONCE = False


def flood(total: int) -> None:
    """Write about ``total`` bytes of short lines plus one overlong line to stdout and stderr."""
    line = "x" * 99 + "\n"
    # Looks like an IPC message; it travels on stdout, which is never parsed.
    decoy = '{"v": 1, "type": "result", "attempt": null, "outcome": "failed", "error": {"code": "internal", "message": "decoy"}}\n'
    for stream in (sys.stdout, sys.stderr):
        stream.write(decoy)
        for _ in range(total // len(line)):
            stream.write(line)
        stream.write("y" * 5000 + "\n")
        stream.flush()


def main() -> int:
    fd = next(
        int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--ipc-fd=")
    )
    sock = socket.socket(fileno=fd)
    channel = Channel(sock)
    print("fake worker log line on stdout", flush=True)
    print("fake worker log line on stderr", file=sys.stderr, flush=True)
    if SCENARIO.get("noise_bytes"):
        flood(SCENARIO["noise_bytes"])
    if SCENARIO.get("exit_before_ready"):
        return 3
    if SCENARIO.get("ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if SCENARIO.get("term_exits_zero"):
        # A worker that shuts down "cleanly" when terminated, so its exit code alone looks like success.
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    if SCENARIO.get("idle_cancel_result"):
        # Like the real worker: SIGTERM before `start` reports a cancelled result with no attempt id.
        def report_cancelled(*_):
            channel.send(
                {
                    "type": "result",
                    "attempt": None,
                    "outcome": "cancelled",
                    "released": True,
                }
            )
            os._exit(0)

        signal.signal(signal.SIGTERM, report_cancelled)
    if SCENARIO.get("never_ready"):
        time.sleep(60)
        return 0
    if SCENARIO.get("garbage_ready"):
        channel.raw(b"this is not json\n")
        time.sleep(60)
        return 0
    channel.send({"type": "ready", "worker_version": "fake-1", "sdk_version": None})
    start = channel.read_command("start")
    attempt = start["attempt"]
    channel.attempt = attempt
    if SCENARIO.get("reject_start"):
        # The real worker refuses start (for example an unusable directory) and stays idle.
        channel.send(
            {
                "type": "rejected",
                "attempt": attempt,
                "command": "start",
                "reason": "invalid",
                "message": "staging_dir is not a writable directory",
            }
        )
        time.sleep(60)
        return 0
    try:
        return run_attempt(channel, attempt, start)
    except Cancelled:
        return cancelled(channel, attempt)


def run_attempt(channel: "Channel", attempt: str, start: dict) -> int:
    sides = start["sides"]
    staging = start["staging_dir"]
    if SCENARIO.get("hang_after_start"):
        hang()
    if SCENARIO.get("malformed_line"):
        channel.raw(b"{bad json\n")
        time.sleep(60)
    if SCENARIO.get("oversized"):
        channel.raw(b'{"v":1,"type":"status","pad":"' + b"x" * 70000 + b'"}\n')
        time.sleep(60)
    if SCENARIO.get("unknown_event"):
        channel.send({"type": "mystery", "attempt": attempt})
        time.sleep(60)
    if SCENARIO.get("spurious_accept"):
        channel.send(
            {"type": "step_accepted", "attempt": attempt, "token": "never-offered"}
        )
        time.sleep(60)
    if SCENARIO.get("unrequested_side_prompt"):
        other = "right" if "right" not in sides else "left"
        channel.send(
            {
                "type": "prompt",
                "attempt": attempt,
                "token": "x-0",
                "side": other,
                "glove_id": GLOVE_IDS[other],
                "index": 0,
                "count": 1,
                "title": "t",
                "instruction": "i",
                "mode": "hold",
                "duration_s": 0.0,
                "continuous": False,
                "estimated_s": 0.0,
            }
        )
        time.sleep(60)
    if SCENARIO.get("wrong_attempt"):
        channel.send(
            {
                "type": "status",
                "attempt": "other-attempt",
                "phase": "discovering",
                "gloves": [],
            }
        )
        time.sleep(60)
    channel.status("discovering", [])
    if SCENARIO.get("hang_after_discovering"):
        hang()
    if SCENARIO.get("chatty_discovery"):
        chatter(channel, "discovering", SCENARIO["chatty_discovery"], [])
    if SCENARIO.get("discover_delay"):
        channel.sleep(SCENARIO["discover_delay"])
    gloves = [{"side": s, "glove_id": GLOVE_IDS[s]} for s in sides]
    channel.status("discovering", gloves)
    exported = []
    for side in sides:
        for index, step in enumerate(STEPS):
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
                    "duration_s": step["time"],
                    "continuous": bool(step.get("continuous", False)),
                    "estimated_s": float(step.get("estimated", 0.0)),
                }
            )
            channel.status("awaiting_step", gloves)
            if SCENARIO.get("crash_after_prompt"):
                os._exit(9)
            confirm_step(channel, attempt, token, step)
            last = side == sides[-1] and index == len(STEPS) - 1
            if last and SCENARIO.get("hang_after_last_accept"):
                hang()
            if last and SCENARIO.get("chatty_finish"):
                chatter(channel, "saving", SCENARIO["chatty_finish"], gloves)
            if SCENARIO.get("fail_at_step") == [side, index]:
                return fail(
                    channel,
                    attempt,
                    "step_failed",
                    "the SDK refused the step",
                    side,
                    index,
                )
        if SCENARIO.get("save_delay"):
            channel.status("saving", gloves)
            channel.sleep(SCENARIO["save_delay"])
        exported.append(export(channel, attempt, staging, side))
        if SCENARIO.get("fail_after_first_export"):
            return fail(
                channel, attempt, "export_failed", "second glove failed", None, None
            )
    channel.status("saving", gloves)
    if SCENARIO.get("no_result"):
        return 0
    files = [{k: e[k] for k in ("side", "file", "bytes", "sha256")} for e in exported]
    if SCENARIO.get("result_mismatch"):
        files[0]["bytes"] += 1
    if SCENARIO.get("partial_result"):
        files = files[:1]
    channel.send(
        {
            "type": "result",
            "attempt": attempt,
            "outcome": "completed",
            "files": files,
            "released": released(),
        }
    )
    if SCENARIO.get("extra_after_result"):
        channel.status("saving", gloves)
    if SCENARIO.get("hang_after_result"):
        time.sleep(60)
    return int(SCENARIO.get("exit_code_after_result", 0))


def confirm_step(channel: "Channel", attempt: str, token: str, step: dict) -> None:
    """Wait for the confirmation of ``token``, refusing anything else, then run the step."""
    global _REJECTED_ONCE
    accepted = False
    while not accepted:
        command = channel.read_command("confirm", "cancel")
        if command["type"] == "cancel":
            if SCENARIO.get("ignore_cancel"):
                time.sleep(60)
            raise Cancelled
        if SCENARIO.get("hang_on_confirm"):
            hang()
        if SCENARIO.get("reject_first_confirm") and not _REJECTED_ONCE:
            _REJECTED_ONCE = True
            channel.send(
                {
                    "type": "rejected",
                    "attempt": attempt,
                    "command": "confirm",
                    "reason": "unexpected",
                    "message": "not now",
                    "token": command["token"],
                }
            )
            continue
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
        channel.send({"type": "step_accepted", "attempt": attempt, "token": token})
        accepted = True
    if step["time"] > 0:
        elapsed = 0.0
        channel.status("capturing_step", [])
        if SCENARIO.get("progress_overrun"):
            channel.send(
                {
                    "type": "step_progress",
                    "attempt": attempt,
                    "token": token,
                    "elapsed_s": step["time"] + 5,
                }
            )
            time.sleep(60)
        while elapsed < step["time"]:
            channel.sleep(0.1)
            elapsed = min(step["time"], elapsed + 0.1)
            channel.send(
                {
                    "type": "step_progress",
                    "attempt": attempt,
                    "token": token,
                    "elapsed_s": elapsed,
                }
            )


def export(channel: "Channel", attempt: str, staging: str, side: str) -> dict:
    empty = SCENARIO.get("empty_export") or SCENARIO.get("empty_side") == side
    data = b"" if empty else (b"MCAL-" + side.encode()) * 64
    name = f"{attempt}_{side}.mcal"
    path = os.path.join(staging, name)
    if SCENARIO.get("symlink_side") == side:
        os.symlink(SCENARIO["symlink_target"], path)
    elif SCENARIO.get("directory_side") == side:
        os.mkdir(path)
    elif not SCENARIO.get("missing_file"):
        with open(path + ".part", "wb") as handle:
            handle.write(data)
        os.rename(path + ".part", path)
    corrupt = SCENARIO.get("bad_sha") or SCENARIO.get("bad_sha_side") == side
    digest = hashlib.sha256(b"different" if corrupt else data).hexdigest()
    reported = (
        SCENARIO.get("reported_name", name)
        if SCENARIO.get("name_side", side) == side
        else name
    )
    event = {
        "type": "exported",
        "attempt": attempt,
        "side": side,
        "glove_id": GLOVE_IDS[side],
        "file": reported,
        "bytes": len(data),
        "sha256": digest,
    }
    if not SCENARIO.get("no_export_events"):
        channel.send(event)
    return event


def fail(channel: "Channel", attempt: str, code: str, message: str, side, step) -> int:
    error = {"code": code, "message": message}
    if side is not None:
        error.update(side=side, step=step)
    channel.send(
        {
            "type": "result",
            "attempt": attempt,
            "outcome": "failed",
            "error": error,
            "released": released(),
        }
    )
    return 1


def cancelled(channel: "Channel", attempt: str) -> int:
    # A worker that needs time to stop its SDK after a cancel.
    time.sleep(float(SCENARIO.get("slow_cancel", 0)))
    channel.status("stopping", [])
    channel.send(
        {
            "type": "result",
            "attempt": attempt,
            "outcome": "cancelled",
            "released": released(),
        }
    )
    return 0


class Channel:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.attempt: str | None = None

    def raw(self, data: bytes) -> None:
        self.sock.sendall(data)

    def send(self, message: dict) -> None:
        self.raw((json.dumps({"v": 1, **message}) + "\n").encode())

    def status(self, phase: str, gloves: list) -> None:
        self.send(
            {
                "type": "status",
                "attempt": self.attempt,
                "phase": phase,
                "gloves": gloves,
            }
        )

    def poll_command(self, timeout: float) -> dict | None:
        """Return the next buffered or arriving command, or None when ``timeout`` passes."""
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

    def read_command(self, *types: str) -> dict:
        while True:
            command = self.poll_command(30.0)
            if command is None:
                raise SystemExit("fake worker timed out waiting for a command")
            if command["type"] == "cancel" and "cancel" not in types:
                if SCENARIO.get("ignore_cancel"):
                    continue
                raise Cancelled
            if command["type"] in types:
                return command

    def sleep(self, seconds: float) -> None:
        """Sleep while still honoring a cancel command."""
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            command = self.poll_command(min(0.05, remaining))
            if (
                command is not None
                and command["type"] == "cancel"
                and not SCENARIO.get("ignore_cancel")
            ):
                raise Cancelled


if __name__ == "__main__":
    sys.exit(main())
