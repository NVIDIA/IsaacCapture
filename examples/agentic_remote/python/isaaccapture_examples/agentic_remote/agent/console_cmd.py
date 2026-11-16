# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND.

One operator command, then exit. It attaches to the robot link service in the
role "console", reads the ROBOT_STATE, asks for the typed confirmation, sends
at most one link message and prints what the robot app answered. A stop is
sent at once after the attach: it waits for no ROBOT_STATE.

This file is also the one place that sends RESET, OPEN, RELEASE, SHUTDOWN and
the operator's rest: the operator console (console/) calls question() and
send_command(). Every check of the command itself is the robot app's: it
releases torque only with the arm measured at rest, whatever is typed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import time
import types
from collections.abc import Callable, Sequence
from typing import Any

from ..link_service.local import LocalCarrier, connect_local, send_raw_stop
from ..protocol.confirm import AGENT_ENV as AGENT_ENV  # re-exported
from ..protocol.confirm import AGENT_ENV_PREFIXES as AGENT_ENV_PREFIXES
from ..protocol.confirm import PATH_IS_CLEAR, YES, agent_environment, confirms
from ..protocol.local_messages import ROLE_CONSOLE
from ..protocol.messages import Result, RobotState
from ..testing.client import ACCEPTED, WorkstationClient

EXIT_OK = 0
EXIT_FAILED = 1  # refused, not answered, not confirmed, or no robot link service

PROGRAM = "console command"
SENDER = "console"
STOP_REASON = "stop typed at the operator console"
UNDELIVERED = "STOP was not delivered"
# A raw STOP (link_service.local.send_raw_stop) is never known to be delivered.
RAW_STOP_WRITTEN = "STOP written; not confirmed: look at the Thor terminal"
RAW_STOP_NOT_WRITTEN = "STOP NOT written"
NOT_CONFIRMED = "not confirmed; nothing sent"

ASK_REST = (
    "Return to rest: the arm moves by itself to its rest pose. Is the path "
    f'clear? Type "{PATH_IS_CLEAR}" to start: '
)
ASK_OPEN = (
    f'Open the jaws: whatever they hold is let go. Type "{YES}" to open the jaws: '
)
ASK_RELEASE = (
    "Release torque: the robot app does it only with the arm measured at "
    f'rest and the jaws empty; the motors are then off. Type "{YES}" to release: '
)
ASK_SHUTDOWN_MOVES = (
    "Shut the robot app down: the arm is away from rest, so it first moves by "
    "itself to its rest pose; then torque is released and the robot app ends. "
    f'Is the path clear? Type "{PATH_IS_CLEAR}" to shut down: '
)
ASK_SHUTDOWN_AT_REST = (
    "Shut the robot app down: the arm is at rest; torque is released there and "
    f'the robot app ends. Type "{YES}" to shut down: '
)
# ROBOT_STATE.motion with the arm at rest: a shutdown then moves nothing.
AT_REST = ("rest_hold", "rest_unpowered")
# The commands that ask a question before anything is sent.
CONFIRMED = ("rest", "open", "release", "shutdown")
COMMANDS = ("state", "stop", "reset", *CONFIRMED)
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
# How long the attach may take; a stop that is not attached by then is
# written raw.
ATTACH_S = 5.0
STOP_ATTACH_S = 0.5
_LOOK_S = 0.1

Say = Callable[[str], None]
Ask = Callable[[str], str]


class NotAsked(RuntimeError):
    """Nobody can be asked: there is no terminal to type at."""


class NotConfirmed(ValueError):
    """The phrase typed is not the one the command's question asks for."""


def _say(text: str) -> None:
    print(text, flush=True)


def question(command: str, state: RobotState | None) -> tuple[str, str] | None:
    """(prompt, phrase to type) for a command in this state of the arm; None if none.

    A shutdown asks for "path is clear" unless the state shows the arm at rest.
    """
    if command == "rest":
        return ASK_REST, PATH_IS_CLEAR
    if command == "open":
        return ASK_OPEN, YES
    if command == "release":
        return ASK_RELEASE, YES
    if command == "shutdown":
        if state is not None and state.motion in AT_REST:
            return ASK_SHUTDOWN_AT_REST, YES
        return ASK_SHUTDOWN_MOVES, PATH_IS_CLEAR
    return None


def send_command(
    command: str,
    client: WorkstationClient,
    state: RobotState,
    *,
    phrase_typed: str,
    command_id: str | None = None,
) -> str:
    """Send reset, rest, open, release or shutdown; returns the command id.

    command_id None lets the client make one; a caller that must know the id
    before the send passes client.new_id().

    The link message quotes the session id, owner number and stop number of
    `state`: pass the ROBOT_STATE that was shown with the question, so the
    robot app refuses an answer typed for another state of the arm.
    NotConfirmed, with nothing sent, unless phrase_typed is the phrase that
    question(command, state) asks for. A SHUTDOWN carries confirmed=True only
    for a typed "path is clear".
    """
    asked = question(command, state)
    if asked is not None and not confirms(phrase_typed, asked[1]):
        raise NotConfirmed(f"{command} {NOT_CONFIRMED}")
    numbers = {
        "session_id": state.session_id,
        "owner_number": state.owner_number,
        "stop_number": state.stop_number,
    }
    if command_id is not None:
        numbers["command_id"] = command_id
    return _run(command, client, numbers, confirms(phrase_typed, PATH_IS_CLEAR))


def _run(
    command: str, client: WorkstationClient, numbers: dict[str, Any], path_clear: bool
) -> str:
    """The one send of each command; test_no_limp.py names this function."""
    if command == "reset":
        return client.reset(**numbers)
    if command == "rest":
        return client.rest(confirmed=True, **numbers)
    if command == "open":
        return client.open_jaws(confirmed=True, **numbers)
    if command == "release":
        return client.release(**numbers)
    if command == "shutdown":
        return client.shutdown(confirmed=path_clear, **numbers)
    raise ValueError(f"{command!r} is not a command that is sent with numbers")


def terminal_ask(prompt: str) -> str:
    """What is typed at this process's controlling terminal; never a pipe.

    NotAsked without a terminal and inside an agent's environment. What was
    typed before the question is shown is dropped.
    """
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        raise NotAsked("this command needs a terminal to type the confirmation at")
    if agent_environment() is not None:
        raise NotAsked(
            "the confirmation is typed by the operator, outside an agent environment"
        )
    try:
        with (
            open("/dev/tty", encoding="utf-8") as typed,
            open("/dev/tty", "w", encoding="utf-8") as shown,
        ):
            termios.tcflush(typed.fileno(), termios.TCIFLUSH)
            shown.write(prompt)
            shown.flush()
            return typed.readline()
    except (OSError, termios.error) as problem:
        raise NotAsked(f"no controlling terminal ({problem})") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.agent.console_cmd",
        description=(
            "Send one operator command to the robot app through the robot link "
            "service and print the answer. rest, open, release and shutdown ask "
            f'for a typed confirmation first: "{PATH_IS_CLEAR}" before the arm '
            f'moves, "{YES}" otherwise.'
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
    attach_s = STOP_ATTACH_S if command == "stop" else ATTACH_S
    try:
        carrier = connect_local(args.socket, ROLE_CONSOLE, "console_cmd", attach_s)
    except OSError as problem:  # also LocalRefused and TimeoutError
        why = f"cannot attach to the robot link service: {problem}"
        if command != "stop":
            say(f"{PROGRAM}: {why}")
            return EXIT_FAILED
        # A STOP goes out all the same; nothing can confirm it.
        raw = send_raw_stop(
            args.socket, reason=STOP_REASON, sender=SENDER, name="console_cmd"
        )
        say(f"stop: {why}. {raw_stop_text(raw.written, raw.notice)}")
        return EXIT_FAILED
    first: list[RobotState] = []  # the first ROBOT_STATE received, once it is

    def note(message: object) -> None:
        if isinstance(message, RobotState) and not first:
            first.append(message)

    client = WorkstationClient(
        carrier,
        types.SimpleNamespace(monotonic=time.monotonic),
        name=SENDER,
        listeners=[note],
    )
    try:
        if command == "stop":
            return _stop(carrier, client, first, args.timeout or STOP_WAIT_S, say)
        return _one(command, carrier, client, args.timeout, ask, say)
    except KeyboardInterrupt:
        say(
            f"{PROGRAM}: interrupted. What was sent is not withdrawn; run the "
            "command stop to stop the arm"
        )
        return EXIT_FAILED
    finally:
        client.close()
        carrier.close()


def _one(
    command: str,
    carrier: LocalCarrier,
    client: WorkstationClient,
    timeout_s: float | None,
    ask: Ask,
    say: Say,
) -> int:
    """Any command but stop: read the state, ask, send, wait for the end."""
    state = _newest_state(carrier, client)
    if command == "state":
        say(f"robot link service: {status_text(carrier)}")
        say(state_text(state) if state else "ROBOT_STATE: none has arrived")
        return EXIT_OK if state is not None else EXIT_FAILED
    if state is None:
        say(
            f"{command} not sent: no ROBOT_STATE to quote ({status_text(carrier)}); "
            "nothing sent"
        )
        return EXIT_FAILED
    asked = question(command, state)
    typed = ""
    if asked is not None:
        # The state the answer is given for; the command quotes its numbers.
        say(state_text(state))
        try:
            typed = ask(asked[0])
        except (NotAsked, EOFError, KeyboardInterrupt) as problem:
            say(f"{command} not confirmed ({problem or 'interrupted'}); nothing sent")
            return EXIT_FAILED
        if not confirms(typed, asked[1]):
            say(f"{command} {NOT_CONFIRMED}")
            return EXIT_FAILED
    command_id = send_command(command, client, state, phrase_typed=typed)
    wait_s = timeout_s or RESULT_WAIT_S[command]
    result = _final(carrier, client, command_id, wait_s, say)
    if result is None:
        say(f"{command}: no final RESULT within {wait_s:g} s ({status_text(carrier)})")
        return EXIT_FAILED
    say(f"{command}: {result_text(result)}")
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
    first: Sequence[RobotState],
    wait_s: float,
    say: Say,
) -> int:
    """STOP has no RESULT: a higher stop number in ROBOT_STATE answers it.

    It is sent before any ROBOT_STATE is waited for. Later states are measured
    on the first one this client received, which `first` holds once it came:
    the service hands a new client its newest state before it reads that
    client's STOP.
    """
    heard = len(carrier.notices)
    client.stop(STOP_REASON, SENDER)
    end = time.monotonic() + wait_s
    while True:
        for notice in carrier.notices[heard:]:
            if notice.text.startswith(UNDELIVERED):
                say(f"stop: {notice.text}")
                return EXIT_FAILED
        state = client.state
        if state is not None and first:
            before = first[0]
            if new_session(before, state):
                say(
                    f"stop: NOT CONFIRMED: {new_session_text(state)}; look at "
                    "the Thor terminal"
                )
                return EXIT_FAILED
            elif stop_shows(before, state):
                say(f"stop: taken. {state_text(state)}")
                return EXIT_OK
        if not carrier.alive:
            say(f"stop: the robot link service is gone ({carrier.close_reason})")
            return EXIT_FAILED
        if time.monotonic() >= end:
            say(
                f"stop: sent, but no ROBOT_STATE showed a higher stop number "
                f"within {wait_s:g} s ({status_text(carrier)})"
            )
            return EXIT_FAILED
        time.sleep(_LOOK_S / 2)


def stop_shows(before: RobotState | None, state: RobotState) -> bool:
    """Whether `state` shows a STOP sent after the ROBOT_STATE `before`.

    Only a higher stop number of the same session does. With no `before`
    nothing does: a stop level alone can be that of an earlier STOP.
    """
    return (
        before is not None
        and state.session_id == before.session_id
        and state.stop_number > before.stop_number
    )


def new_session(before: RobotState | None, state: RobotState) -> bool:
    """Whether `state` is of another robot app session than `before`.

    A new session's stop number starts over: it never confirms a STOP.
    """
    return before is not None and state.session_id != before.session_id


def new_session_text(state: RobotState) -> str:
    return (
        f"the robot app shows a new session ({state.session_id}), which says "
        "nothing about this STOP"
    )


def raw_stop_text(written: bool, notice: str) -> str:
    """What to say about a raw STOP: written or not, never "delivered"."""
    if not written:
        return f"{RAW_STOP_NOT_WRITTEN}: {notice}"
    said = f" (the robot link service said: {notice})" if notice else ""
    return f"{RAW_STOP_WRITTEN}{said}"


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


def status_text(carrier: LocalCarrier) -> str:
    """The newest LOCAL_STATUS of a carrier in words."""
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


def state_text(state: RobotState) -> str:
    """A ROBOT_STATE in words: who owns the arm, the stop, the motion, the jaws."""
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


def result_text(result: Result) -> str:
    """A RESULT in words: its status, its reason and its detail."""
    text = f"RESULT {result.status}"
    if result.reason:
        text += f": {result.reason}"
    detail: Any = getattr(result, "detail", None)
    if detail:
        text += f" {json.dumps(detail, sort_keys=True, default=str)}"
    return text


if __name__ == "__main__":
    sys.exit(main())
