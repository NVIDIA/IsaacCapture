# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND.

One operator command, then exit: the smallest form of the operator console.
It attaches to the robot link service in the role "console", sends at most
one link message and prints what the robot app answered. rest, open, release
and shutdown are sent only after a "yes" typed at this terminal. Every check
of the command itself is the robot app's: it releases torque only with the
arm measured at rest, whatever is typed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections.abc import Callable, Sequence
from typing import Any

from ..link_service.local import LocalCarrier, connect_local
from ..protocol.local_messages import ROLE_CONSOLE
from ..protocol.messages import Result, RobotState
from ..testing.client import ACCEPTED, WorkstationClient

EXIT_OK = 0
EXIT_FAILED = 1  # refused, not answered, not confirmed, or no robot link service

PROGRAM = "console command"
SENDER = "console"
STOP_REASON = "stop typed at the operator console"
UNDELIVERED = "STOP was not delivered"

# What must be typed "yes" to, per command, before anything is sent.
CONFIRMATIONS = {
    "rest": (
        "Return to rest: the arm moves by itself to its rest pose. Is the "
        "path clear? Type yes to start: "
    ),
    "open": (
        "Open the jaws: whatever they hold is let go. Type yes to open the jaws: "
    ),
    "release": (
        "Release torque: the robot app does it only with the arm measured at "
        "rest and the jaws empty; the motors are then off. Type yes to release: "
    ),
    "shutdown": (
        "Shut the robot app down: it returns the arm to rest if it is away "
        "from it (is the path clear?), releases torque at rest and ends. Type "
        "yes to shut down: "
    ),
}
COMMANDS = ("state", "stop", "reset", *CONFIRMATIONS)
# How long each command waits for its final RESULT, in s.
RESULT_WAIT_S = {
    "reset": 10.0,
    "open": 60.0,
    "release": 30.0,
    "rest": 180.0,
    "shutdown": 240.0,
}
STATE_WAIT_S = 3.0
STOP_WAIT_S = 5.0
_LOOK_S = 0.1

Say = Callable[[str], None]
Ask = Callable[[str], str]


class NotAsked(RuntimeError):
    """Nobody can be asked: there is no terminal to type at."""


def _say(text: str) -> None:
    print(text, flush=True)


def terminal_ask(prompt: str) -> str:
    """What is typed at this process's controlling terminal; never a pipe."""
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        raise NotAsked("this command needs a terminal to type the confirmation at")
    try:
        with (
            open("/dev/tty", encoding="utf-8") as typed,
            open("/dev/tty", "w", encoding="utf-8") as shown,
        ):
            shown.write(prompt)
            shown.flush()
            return typed.readline()
    except OSError as problem:
        raise NotAsked(f"no controlling terminal ({problem})") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.agent.console_cmd",
        description=(
            "Send one operator command to the robot app through the robot link "
            "service and print the answer. rest, open, release and shutdown ask "
            "for a typed yes first."
        ),
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--socket",
        default=None,
        metavar="PATH",
        help="the robot link service's local socket (default: its own default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="S",
        help="how long to wait for the answer (default: by command)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, *, ask: Ask = terminal_ask, say: Say = _say
) -> int:
    """Run one command; returns the exit code."""
    args = _parser().parse_args(argv)
    if args.timeout is not None and not args.timeout > 0:
        say(f"{PROGRAM}: --timeout must be above 0")
        return EXIT_FAILED
    command = args.command
    prompt = CONFIRMATIONS.get(command)
    if prompt is not None:
        # Asked before anything is attached or read: the numbers the command
        # quotes are then those of the moment it is sent.
        try:
            answer = ask(prompt)
        except (NotAsked, EOFError, KeyboardInterrupt) as problem:
            say(f"{command} not confirmed ({problem or 'interrupted'}); nothing sent")
            return EXIT_FAILED
        if str(answer).strip().lower() != "yes":
            say(f"{command} not confirmed; nothing sent")
            return EXIT_FAILED
    try:
        carrier = connect_local(args.socket, ROLE_CONSOLE, "console_cmd", 5.0)
    except OSError as problem:  # also LocalRefused and TimeoutError
        say(f"{PROGRAM}: cannot attach to the robot link service: {problem}")
        return EXIT_FAILED
    client = WorkstationClient(
        carrier, types.SimpleNamespace(monotonic=time.monotonic), name=SENDER
    )
    try:
        return _run(command, carrier, client, args.timeout, say)
    except KeyboardInterrupt:
        say(
            f"{PROGRAM}: interrupted. What was sent is not withdrawn; run the "
            "command stop to stop the arm"
        )
        return EXIT_FAILED
    finally:
        client.close()
        carrier.close()


def _run(
    command: str,
    carrier: LocalCarrier,
    client: WorkstationClient,
    timeout_s: float | None,
    say: Say,
) -> int:
    state = _newest_state(carrier, client)
    if command == "state":
        say(f"robot link service: {_status_text(carrier)}")
        say(_state_text(state) if state else "ROBOT_STATE: none has arrived")
        return EXIT_OK if state is not None else EXIT_FAILED
    if command == "stop":
        return _stop(carrier, client, state, timeout_s or STOP_WAIT_S, say)
    if state is None:
        say(
            f"{command} not sent: no ROBOT_STATE to quote ({_status_text(carrier)}); "
            "nothing sent"
        )
        return EXIT_FAILED
    numbers = {
        "session_id": state.session_id,
        "owner_number": state.owner_number,
        "stop_number": state.stop_number,
    }
    if command == "reset":
        command_id = client.reset(**numbers)
    elif command == "rest":
        command_id = client.rest(confirmed=True, **numbers)
    elif command == "open":
        command_id = client.open_jaws(confirmed=True, **numbers)
    elif command == "release":
        command_id = client.release(**numbers)
    else:
        command_id = client.shutdown(confirmed=True, **numbers)
    wait_s = timeout_s or RESULT_WAIT_S[command]
    result = _final(carrier, client, command_id, wait_s, say)
    if result is None:
        say(f"{command}: no final RESULT within {wait_s:g} s ({_status_text(carrier)})")
        return EXIT_FAILED
    say(f"{command}: {_result_text(result)}")
    return EXIT_OK if result.status == "completed" else EXIT_FAILED


def _newest_state(
    carrier: LocalCarrier, client: WorkstationClient
) -> RobotState | None:
    """The newest ROBOT_STATE; waits for the first one while the robot is there."""
    end = time.monotonic() + STATE_WAIT_S
    while client.state is None and carrier.alive and time.monotonic() < end:
        status = carrier.status
        if status is not None and not status.robot_connected:
            break
        time.sleep(_LOOK_S / 2)
    return client.state


def _stop(
    carrier: LocalCarrier,
    client: WorkstationClient,
    before: RobotState | None,
    wait_s: float,
    say: Say,
) -> int:
    """STOP has no RESULT: a higher stop number in ROBOT_STATE answers it."""
    heard = len(carrier.notices)
    client.stop(STOP_REASON, SENDER)
    end = time.monotonic() + wait_s
    while True:
        for notice in carrier.notices[heard:]:
            if notice.text.startswith(UNDELIVERED):
                say(f"stop: {notice.text}")
                return EXIT_FAILED
        state = client.state
        if state is not None and (
            (before is None and state.stop_level != "none")
            or (before is not None and state.session_id != before.session_id)
            or (before is not None and state.stop_number > before.stop_number)
        ):
            say(f"stop: taken. {_state_text(state)}")
            return EXIT_OK
        if not carrier.alive:
            say(f"stop: the robot link service is gone ({carrier.close_reason})")
            return EXIT_FAILED
        if time.monotonic() >= end:
            say(
                f"stop: sent, but no ROBOT_STATE showed a higher stop number "
                f"within {wait_s:g} s ({_status_text(carrier)})"
            )
            return EXIT_FAILED
        time.sleep(_LOOK_S / 2)


def _final(
    carrier: LocalCarrier,
    client: WorkstationClient,
    command_id: str,
    wait_s: float,
    say: Say,
) -> Result | None:
    """The final RESULT of this command id; None when none came."""
    end = time.monotonic() + wait_s
    told = 0
    while True:
        results = client.results(command_id)
        for result in results[told:]:
            if result.status == ACCEPTED:
                say(f"accepted; waiting for the end ({wait_s:g} s at most)")
        told = len(results)
        for result in results:
            if result.status != ACCEPTED:
                return result
        if not carrier.alive or time.monotonic() >= end:
            return None
        time.sleep(_LOOK_S / 2)


def _status_text(carrier: LocalCarrier) -> str:
    status = carrier.status
    if not carrier.alive:
        return f"the robot link service is gone: {carrier.close_reason}"
    if status is None:
        return "no LOCAL_STATUS"
    if status.robot_connected:
        robot = f"robot connection up, session {status.session_id or '?'}"
    else:
        robot = f"robot connection lost ({status.close_reason or 'not connected'})"
    clients = ", ".join(
        f"{count} {role}" for role, count in sorted(status.roles.items())
    )
    stop = "; a STOP is not confirmed yet" if status.stop_unconfirmed else ""
    return f"{robot}; clients: {clients or 'none'}{stop}"


def _state_text(state: RobotState) -> str:
    wait = state.assistance_wait_s
    fault = f" ({state.fault_reason})" if state.fault_reason else ""
    move = f", move {state.move_index}/{state.move_total}" if state.move_id else ""
    return (
        f"ROBOT_STATE: control owner {state.owner} (owner number "
        f"{state.owner_number}), stop number {state.stop_number}, stop level "
        f"{state.stop_level}{fault}, motion {state.motion}{move}, jaws "
        f"{state.gripper.load}, torque {'on' if state.torque_on else 'off'}, "
        f"assistance wait {f'{wait:.0f} s left' if wait is not None else 'none'}"
    )


def _result_text(result: Result) -> str:
    text = f"RESULT {result.status}"
    if result.reason:
        text += f": {result.reason}"
    detail: Any = getattr(result, "detail", None)
    if detail:
        text += f" {json.dumps(detail, sort_keys=True, default=str)}"
    return text


if __name__ == "__main__":
    sys.exit(main())
