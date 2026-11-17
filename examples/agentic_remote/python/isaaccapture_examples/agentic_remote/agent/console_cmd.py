# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND.

One operator command, then exit. It attaches to the robot link service in the
role "console", reads the ROBOT_STATE, asks for the typed confirmation, sends
at most one link message and prints what the robot app answered. A stop is
sent at once after the attach: it waits for no ROBOT_STATE.

This file is also the one place that sends RESET, OPEN (the typed open and the
typed close), RELEASE, SHUTDOWN, the operator's rest and the typed takeover and
hand-back: the operator console (console/) calls question(), jaws_refused()
and send_command(). Every check of the command itself is the robot app's: it
releases torque only with the arm measured at rest, whatever is typed here.

The robot app proves the jaws empty by measurement before a return to rest, a
torque release, a shutdown and a hand-back. When it refuses one of them only
for that, the operator is asked, and the typed word "jaws are empty" goes
with the command, sent again, in place of the proof.

A takeover makes a person the control owner; the hand-back ends it at rest.
Both are asked with the right controller's buttons B and A. takeover and
handback are their typed fallback, which the robot link service forwards
only where its file allows it (operator.typed_takeover).
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

from ..console.status import motors_text
from ..link_service.local import LocalCarrier, connect_local, send_raw_stop
from ..protocol.confirm import AGENT_ENV as AGENT_ENV  # re-exported
from ..protocol.confirm import AGENT_ENV_PREFIXES as AGENT_ENV_PREFIXES
from ..protocol.confirm import (
    JAWS_ARE_EMPTY,
    PATH_IS_CLEAR,
    YES,
    agent_environment,
    confirms,
)
from ..protocol.local_messages import ROLE_CONSOLE
from ..protocol.messages import (
    CLOSE_JAWS,
    OPEN_JAWS,
    REQUEST_HANDBACK,
    REQUEST_HUMAN,
    REQUEST_MOTORS_ON,
    Result,
    RobotState,
)
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
# Said when this program stops waiting for a command's final RESULT. These
# commands carry no expiry: the robot app does one that reaches it later,
# unless a STOP or a change of control owner came before it.
STILL_LIVE = (
    "What was sent is not withdrawn: the robot app may still do it. Run the "
    "command stop before anyone goes near the arm"
)

ASK_REST = (
    "Return to rest: the arm moves by itself to its rest pose; there rest is "
    "measured, the jaws close and the motors go off. Is the path "
    f'clear? Type "{PATH_IS_CLEAR}" to start: '
)
# Asked before the motors are switched on again after a rest or a release.
ASK_ON = (
    "Motors on again: the robot app connects the arm as at its start, and "
    "only an arm that lies at rest. Stay beside the e-stop. "
    f'Type "{YES}" to switch the motors on: '
)
ASK_OPEN = (
    f'Open the jaws: whatever they hold is let go. Type "{YES}" to open the jaws: '
)
# Asked before a typed close only while the jaws are not proven empty.
ASK_CLOSE = (
    "Close the jaws: the robot app does not know them empty ({jaws}), and they "
    f'close on whatever they hold. Type "{YES}" to close the jaws: '
)
# Asked after the robot app refused a rest, a release, a shutdown or a
# hand-back because the jaws are not proven empty. The typed word goes with
# the command, which is sent again.
ASK_JAWS_EMPTY = (
    "{command} was refused because the jaws are not proven empty. Look at the "
    f"jaws. If they hold something: stop, open, {YES}. If they are empty: type "
    f'"{JAWS_ARE_EMPTY}" and {{command}} is asked again on your word: '
)
JAWS_NOT_CONFIRMED = "not confirmed that the jaws are empty; nothing sent"
# ROBOT_STATE.gripper.load of jaws proven empty.
JAWS_EMPTY = "empty"
# The commands the robot app refuses while the jaws are not proven empty, and
# what its refusal then says (rest.py).
JAWS_ASKED = ("rest", "release", "shutdown", "handback")
JAWS_NOT_PROVEN = "the jaws are not proven empty"
ASK_RELEASE = (
    "Release torque: the robot app does it only with the arm measured at "
    "rest and the jaws empty; it closes the jaws and the motors are then off. "
    f'Type "{YES}" to release: '
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
ASK_TAKEOVER = (
    "Take over: a person becomes the control owner, as with right B held 1 s; "
    "the robot app does it only while nobody owns the arm (type stop first). "
    "The arm then follows the right controller while its grip is squeezed. "
    f'Type "{YES}" to take over: '
)
ASK_HANDBACK = (
    "Hand back: the takeover ends and nobody owns the arm, as with right A "
    "held 1 s; the robot app does it only with the arm measured at rest and "
    f'the jaws empty. Type "{YES}" to hand back: '
)
# ROBOT_STATE.motion with the arm at rest: a shutdown then moves nothing.
AT_REST = ("rest_hold", "rest_unpowered", "release_failed")
# The OWNER_REQUEST each typed fallback sends, and the reason it carries.
TYPED_REQUESTS = {"takeover": REQUEST_HUMAN, "handback": REQUEST_HANDBACK}
# Every command that is an OWNER_REQUEST written here: those two, and "on",
# which switches the motors on again after a rest or a release.
OWNER_WORDS = {**TYPED_REQUESTS, "on": REQUEST_MOTORS_ON}
TYPED_REASON = "typed at the operator console"
# How the robot link service's refusal of a typed fallback begins where its
# file does not allow it, and what the operator is told then.
NOT_ALLOWED = f"the {ROLE_CONSOLE} role may not send "
USE_BUTTONS = "use right B / right A on the controller"
# The commands that ask a question before anything is sent. A close asks only
# while the jaws are not proven empty.
CONFIRMED = ("rest", "open", "close", "release", "on", "shutdown", *TYPED_REQUESTS)
COMMANDS = ("state", "stop", "reset", *CONFIRMED)
# The OPEN each typed jaw command sends: its action.
JAW_ACTIONS = {"open": OPEN_JAWS, "close": CLOSE_JAWS}
# How long each command waits for its final RESULT, in s.
RESULT_WAIT_S = {
    "reset": 10.0,
    "on": 60.0,
    "open": 60.0,
    "close": 60.0,
    "release": 30.0,
    "rest": 180.0,
    "shutdown": 240.0,
    "takeover": 10.0,
    "handback": 30.0,
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
    A close asks only while the state shows the jaws not proven empty.
    """
    if command == "rest":
        return ASK_REST, PATH_IS_CLEAR
    if command == "open":
        return ASK_OPEN, YES
    if command == "close":
        jaws = _jaws_not_empty(state)
        return None if jaws is None else (ASK_CLOSE.format(jaws=jaws), YES)
    if command == "release":
        return ASK_RELEASE, YES
    if command == "on":
        return ASK_ON, YES
    if command == "shutdown":
        if state is not None and state.motion in AT_REST:
            return ASK_SHUTDOWN_AT_REST, YES
        return ASK_SHUTDOWN_MOVES, PATH_IS_CLEAR
    if command == "takeover":
        return ASK_TAKEOVER, YES
    if command == "handback":
        return ASK_HANDBACK, YES
    return None


def _jaws_not_empty(state: RobotState | None) -> str | None:
    """How ROBOT_STATE shows jaws that are not proven empty; None when they are."""
    if state is None or state.gripper.load == JAWS_EMPTY:
        return None
    text = f"jaws {state.gripper.load}"
    return f"{text}, may be holding" if state.gripper.may_be_holding else text


def jaws_refused(command: str, result: Result) -> tuple[str, str] | None:
    """(prompt, phrase to type) after a refusal for jaws not proven empty, or None.

    For a command of JAWS_ASKED whose RESULT says that it was refused, or
    failed, because the jaws are not proven empty. The typed phrase is the
    operator's word that they are empty: send_command(..., jaws_empty=True)
    sends the command again with it, and the robot app takes it in place of
    its own proof.
    """
    if command not in JAWS_ASKED or result.status not in ("refused", "failed"):
        return None
    if JAWS_NOT_PROVEN not in result.reason:
        return None
    return ASK_JAWS_EMPTY.format(command=command), JAWS_ARE_EMPTY


def buttons_hint(command: str, result: Result) -> str:
    """What follows a RESULT's text: the buttons to use, or "".

    The buttons, where the robot link service refused a typed takeover or
    hand-back because its file does not allow it.
    """
    refused = result.status == "refused" and result.reason.startswith(NOT_ALLOWED)
    return f"; {USE_BUTTONS}" if refused and command in TYPED_REQUESTS else ""


def send_command(
    command: str,
    client: WorkstationClient,
    state: RobotState,
    *,
    phrase_typed: str,
    command_id: str | None = None,
    jaws_empty: bool = False,
) -> str:
    """Send reset, rest, open, close, release, shutdown, takeover or handback.

    Returns the command id. command_id None lets the client make one; a
    caller that must know the id before the send passes client.new_id().

    The link message quotes the session id, owner number and stop number of
    `state`: pass the ROBOT_STATE that was shown with the question, so the
    robot app refuses an answer typed for another state of the arm.
    NotConfirmed, with nothing sent, unless phrase_typed is the phrase that
    question(command, state) asks for. A SHUTDOWN carries confirmed=True only
    for a typed "path is clear". jaws_empty says that the operator typed the
    phrase jaws_refused() asked for: it goes with a command of JAWS_ASKED as
    the operator's word that the jaws are empty.
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
    if jaws_empty and command in JAWS_ASKED:
        numbers["jaws_empty"] = True
    typed = {
        "path_clear": confirms(phrase_typed, PATH_IS_CLEAR),
        "yes": confirms(phrase_typed, YES),
    }
    return _run(command, client, numbers, typed)


def _run(
    command: str,
    client: WorkstationClient,
    numbers: dict[str, Any],
    typed: dict[str, bool],
) -> str:
    """The one send of each command; test_no_limp.py names this function.

    numbers also holds jaws_empty for a command the operator's word goes
    with. typed says which confirmation was typed.
    """
    if command == "reset":
        return client.reset(**numbers)
    if command == "rest":
        return client.rest(confirmed=True, **numbers)
    if command in JAW_ACTIONS:
        # An opening was confirmed; a close says whether it was asked about.
        confirmed = command == "open" or typed["yes"]
        return client.open_jaws(
            confirmed=confirmed, action=JAW_ACTIONS[command], **numbers
        )
    if command == "release":
        return client.release(**numbers)
    if command == "shutdown":
        return client.shutdown(confirmed=typed["path_clear"], **numbers)
    if command in OWNER_WORDS:
        return client.request_owner(
            OWNER_WORDS[command], reason=TYPED_REASON, **numbers
        )
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
            "service and print the answer. rest, open, release, shutdown, "
            "takeover and handback ask for a typed confirmation first: "
            f'"{PATH_IS_CLEAR}" before the arm moves by itself, "{YES}" '
            "otherwise. close asks only while the jaws are not proven empty. "
            "Where the robot app would refuse a command because the jaws are "
            f'not proven empty, a typed "{JAWS_ARE_EMPTY}" is asked for and '
            "sent with it. takeover and handback are sent on only where the "
            "robot link service's file allows it; else use right B and right A."
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
    """Any command but stop: read the state, ask, send, wait for the end.

    A command refused only because the jaws are not proven empty is sent once
    more, on the operator's typed word that they are empty.
    """
    state = _newest_state(carrier, client)
    if command == "state":
        say(f"robot link service: {status_text(carrier)}")
        say(state_text(state) if state else "ROBOT_STATE: none has arrived")
        return EXIT_OK if state is not None else EXIT_FAILED
    jaws_empty = False
    while True:
        if state is None:
            say(
                f"{command} not sent: no ROBOT_STATE to quote "
                f"({status_text(carrier)}); nothing sent"
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
                say(
                    f"{command} not confirmed ({problem or 'interrupted'}); "
                    "nothing sent"
                )
                return EXIT_FAILED
            if not confirms(typed, asked[1]):
                say(f"{command} {NOT_CONFIRMED}")
                return EXIT_FAILED
        command_id = send_command(
            command, client, state, phrase_typed=typed, jaws_empty=jaws_empty
        )
        wait_s = timeout_s or RESULT_WAIT_S[command]
        result = _final(carrier, client, command_id, wait_s, say)
        if result is None:
            say(
                f"{command}: no final RESULT within {wait_s:g} s "
                f"({status_text(carrier)}). {STILL_LIVE}"
            )
            return EXIT_FAILED
        say(f"{command}: {result_text(result)}{buttons_hint(command, result)}")
        about_jaws = None if jaws_empty else jaws_refused(command, result)
        if about_jaws is None:
            return EXIT_OK if result.status == "completed" else EXIT_FAILED
        try:
            word = ask(about_jaws[0])
        except (NotAsked, EOFError, KeyboardInterrupt):
            return EXIT_FAILED
        if not confirms(word, about_jaws[1]):
            say(f"{command}: {JAWS_NOT_CONFIRMED}")
            return EXIT_FAILED
        # The word is given for the arm as it is now: the command is asked anew.
        jaws_empty, state = True, client.state


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
        f"{state.gripper.load}, {motors_text(state)}, "
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
