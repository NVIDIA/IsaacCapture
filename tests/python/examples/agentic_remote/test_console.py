# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator console: typed commands in, lines out, STOP always.

Most tests run the console against a real robot link service whose robot side
the test scripts (test_console_cmd.Console), so every link message the service
forwarded is seen. Two put it in front of the real robot app on the simulated
arm, and the terminal tests run it in a child process on a pseudo-terminal.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.agent import console_cmd
from isaaccapture_examples.agentic_remote.console import console as console_module
from isaaccapture_examples.agentic_remote.console import status as status_module
from isaaccapture_examples.agentic_remote.console.console import (
    ATTACH_EVERY_S,
    ATTACH_TRIES,
    BANNER,
    EXIT_OK,
    EXIT_UNUSABLE,
    QUIT_NOTE,
    STOP_CTRL_C,
    Console,
    main,
)
from isaaccapture_examples.agentic_remote.console.status import (
    event_bells,
    help_block,
    status_key,
    status_line,
    takeover_text,
    welcome_line,
)
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalRefused,
    RawStop,
    pack_record,
)
from isaaccapture_examples.agentic_remote.link_service.service import STOP_UNDELIVERED
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair, send
from isaaccapture_examples.agentic_remote.protocol.confirm import PATH_IS_CLEAR
from isaaccapture_examples.agentic_remote.protocol.local_messages import LocalStatus
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_MOVE_ID,
    Event,
    GripperState,
    Open,
    OwnerRequest,
    Release,
    Reset,
    Result,
    RobotState,
    Shutdown,
    Stop,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app import app as app_module
from isaaccapture_examples.agentic_remote.robot_app import waits
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from isaaccapture_examples.agentic_remote.robot_app.text import LINE_KINDS
from isaaccapture_examples.agentic_remote.testing import client as client_module
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from link_bench import Bench, robot_state, until, welcome
from test_console_cmd import Console as ScriptedRobot
from test_console_cmd import OddService, completed, odd_services, on_a_terminal

PACKAGE = Path(console_module.__file__).resolve().parent
EXAMPLE = PACKAGE.parents[3]
# Real seconds a test gives another thread before it fails.
WAIT_S = 10.0
FULL = "too many local clients are attached"
ASK_REST = 'Type "path is clear" to start'
ASK_YES = 'Type "yes" to '
TYPED_STOP = Stop(reason=console_cmd.STOP_REASON, sender="console")
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
# A takeover makes a person the control owner; the hand-back ends it at rest;
# from the one to the other it is open. The human program is the workstation
# program that reads the person's controller. The typed takeover and
# hand-back, and the OWNER_REQUEST each sends:
TYPED = [("takeover", "human"), ("handback", "handback")]
POSSIBLE = "takeover: possible now: hold right B for 1 s with the grip released"
NO_PROGRAM = "takeover: not possible: no human program is attached: start run_human.sh"
NO_DATA = (
    "takeover open: no controller data. To end it: type rest, then path is "
    "clear; then start the human program and hold right A, or see the README, "
    '"A takeover that cannot be ended with right A"'
)
SWITCHED_OFF = "human driving is switched off in this robot app's file (human.enabled)"
LOST = "takeover: not possible: the robot connection is lost"
TAKE_OVER_NOTE = "hold right B for 1 s with the grip released to take over"
# LOCAL_STATUS with and without a human program attached.
WITH_PROGRAM = LocalStatus(True, "", "s1", None, False, {"console": 1, "human": 1})
WITHOUT_PROGRAM = LocalStatus(True, "", "s1", None, False, {"console": 1})


def human_facts(**changes: Any) -> dict[str, Any]:
    """ROBOT_STATE's health["human"] with nothing in a takeover's way."""
    facts = {
        "enabled": True,
        "takeover_open": False,
        "driving": False,
        "needs_release": True,
        "source": "teleop",
        "seq": 7,
        "round_trip_s": 0.02,
        "note": "",
        "ended": "",
        "refusal": "",
        "takeover": "",
    }
    assert set(changes) <= set(facts)
    return {**facts, **changes}


def with_human(
    state: RobotState | None = None,
    *,
    stage: str = "ready",
    age: float | None = 0.03,
    contact_hold: bool = False,
    **facts: Any,
) -> RobotState:
    """A ROBOT_STATE of a robot app with the human path: fresh controller data."""
    return dataclasses.replace(
        robot_state() if state is None else state,
        teleop_stage=stage,
        controller_age_s=age,
        health={"human": human_facts(**facts), "contact_hold": contact_hold},
    )


class Desk:
    """An operator console that runs on a thread; the test types and reads."""

    def __init__(self, path: str | None, **options: Any) -> None:
        self.lines = TypedLines()
        self.said: list[str] = []
        self._heard = threading.Condition()
        self.console = Console(self.lines, self._hear, socket_path=path, **options)
        self.code: int | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.wait(BANNER)

    def _run(self) -> None:
        self.code = self.console.run()

    def _hear(self, line: str) -> None:
        with self._heard:
            self.said.append(line)
            self._heard.notify_all()

    def type(self, *lines: str) -> None:
        for line in lines:
            self.lines.type(line + "\n")

    def wait(self, text: str, count: int = 1) -> str:
        """Wait for the count-th line that holds `text`; returns it."""

        def found() -> list[str]:
            return [line for line in self.said if text in line]

        with self._heard:
            assert self._heard.wait_for(lambda: len(found()) >= count, WAIT_S), (
                f"{len(found())} of {count} lines with {text!r}; said: {self.said}"
            )
            return found()[count - 1]

    def count(self, text: str) -> int:
        with self._heard:
            return sum(text in line for line in self.said)

    def sees(self, shown: Callable[[RobotState], bool]) -> None:
        """Wait until the console's newest ROBOT_STATE is one `shown` accepts."""

        def has() -> bool:
            state = self.console._state
            return state is not None and shown(state)

        until(has, "the console's ROBOT_STATE")

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def end(self) -> int | None:
        """End the typed input, as a closed terminal does; the exit code."""
        self.lines.end()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the console did not end"
        return self.code

    def close(self) -> None:
        if self._thread.is_alive():
            self.end()
        self.console.close()


def stopped(robot: ScriptedRobot) -> RobotState:
    """The ROBOT_STATE a STOP is answered with: the next stop number, a hold."""
    robot.state = dataclasses.replace(
        robot.state,
        seq=robot.state.seq + 1,
        stop_number=robot.state.stop_number + 1,
        stop_level="stop",
        motion="hold",
    )
    return robot.state


@pytest.fixture
def bench(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Bench]:
    made = Bench(tmp_path_factory.mktemp("oc"))
    yield made
    made.close()


@pytest.fixture
def robot(bench: Bench) -> Iterator[ScriptedRobot]:
    """The robot side: completes every command and shows every STOP."""
    made = ScriptedRobot(bench)
    made.reply = lambda message: (
        [stopped(made)] if isinstance(message, Stop) else completed(message)
    )
    yield made
    made.close()


@pytest.fixture
def desks() -> Iterator[Callable[..., Desk]]:
    made: list[Desk] = []

    def make(path: str | None, **options: Any) -> Desk:
        made.append(Desk(path, **options))
        return made[-1]

    yield make
    for desk in made:
        desk.close()


@pytest.fixture
def desk(robot: ScriptedRobot, desks: Callable[..., Desk]) -> Desk:
    """A console attached to the service of `robot`, on the machine's clock."""
    made = desks(robot.bench.path)
    made.wait("attached to the robot link service: robot connection up")
    made.sees(lambda state: True)  # a command typed before it is not sent
    return made


@pytest.fixture
def typed_robot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ScriptedRobot]:
    """The robot side of a service whose file allows the typed takeover."""
    allowing = Bench(tmp_path_factory.mktemp("ot"), typed_takeover=True)
    made = ScriptedRobot(allowing)
    yield made
    made.close()
    allowing.close()


# The wording table


def test_rest_asks_for_path_is_clear_and_yes_sends_nothing(
    robot: ScriptedRobot, desk: Desk
) -> None:
    desk.type("rest")
    desk.wait(ASK_REST)
    desk.type("yes")
    desk.wait("rest not confirmed; nothing sent")
    assert robot.settled() == []
    desk.type("rest")
    desk.wait(ASK_REST, 2)
    desk.type("  Path is  CLEAR ")
    desk.wait('rest: RESULT completed {"owner": "nobody"}')
    (message,) = robot.settled()
    assert isinstance(message, OwnerRequest)
    assert (message.request, message.confirmed) == ("rest", True)
    assert (message.session_id, message.owner_number, message.stop_number) == (
        "s1",
        4,
        3,
    )
    # The state the answer is given for stands right above the question.
    asked = desk.said.index(desk.wait(ASK_REST))
    assert desk.said[asked - 1] == console_cmd.state_text(robot.state)


def test_shutdown_at_rest_asks_for_yes_and_is_sent_unconfirmed(
    robot: ScriptedRobot, desk: Desk
) -> None:
    reason = "confirm the path is clear"  # what the Thor says if the arm must move
    robot.reply = lambda message: [Result(message.command_id, "refused", reason)]
    desk.type("shutdown")
    assert "the arm is at rest" in desk.wait(f"{ASK_YES}shut down")
    desk.type("yes")
    desk.wait(f"shutdown: RESULT refused: {reason}")
    (message,) = robot.settled()
    assert isinstance(message, Shutdown) and message.confirmed is False


def test_shutdown_away_from_rest_asks_for_path_is_clear(
    robot: ScriptedRobot, desk: Desk
) -> None:
    robot.show(motion="hold")
    desk.sees(lambda state: state.motion == "hold")
    desk.type("shutdown")
    desk.wait('Type "path is clear" to shut down')
    desk.type("yes")
    desk.wait("shutdown not confirmed; nothing sent")
    assert robot.settled() == []
    desk.type("shutdown")
    desk.wait('Type "path is clear" to shut down', 2)
    desk.type(PATH_IS_CLEAR)
    desk.wait("shutdown: RESULT completed")
    (message,) = robot.settled()
    assert isinstance(message, Shutdown) and message.confirmed is True


@pytest.mark.parametrize(
    ("command", "kind"), [("open", Open), ("release", Release), ("reset", Reset)]
)
def test_open_and_release_ask_for_yes_and_reset_asks_nothing(
    robot: ScriptedRobot, desk: Desk, command: str, kind: type
) -> None:
    desk.type(command)
    if command != "reset":
        desk.wait(ASK_YES)
        desk.type("path is clear")  # not what was asked
        desk.wait(f"{command} not confirmed; nothing sent")
        assert robot.settled() == []
        desk.type(command)
        desk.wait(ASK_YES, 2)
        desk.type("yes")
    desk.wait(f"{command}: RESULT completed")
    (message,) = robot.settled()
    assert type(message) is kind and desk.count(ASK_YES) == 2 * (command != "reset")


def test_a_stop_typed_as_the_answer_is_a_stop_and_confirms_nothing(
    robot: ScriptedRobot, desk: Desk
) -> None:
    desk.type("open")
    desk.wait(ASK_YES)
    desk.type("stop")
    desk.wait("stop: taken.")
    desk.wait("not confirmed; nothing sent")
    assert robot.settled() == [TYPED_STOP]


@pytest.mark.parametrize(("command", "sends"), TYPED)
def test_takeover_and_handback_ask_yes_and_send_the_owner_request(
    typed_robot: ScriptedRobot, desks: Callable[..., Desk], command: str, sends: str
) -> None:
    robot = typed_robot
    desk = desks(robot.bench.path)
    desk.wait("attached to the robot link service: robot connection up")
    desk.sees(lambda state: True)
    desk.type("help", command)
    named = desk.wait(f"{command}  ")
    assert named.endswith("(only if the robot link service's file allows it)")
    desk.wait(ASK_YES)
    desk.type("path is clear")  # not what was asked
    desk.wait(f"{command} not confirmed; nothing sent")
    assert robot.settled() == []
    desk.type(command)
    prompt = desk.wait(ASK_YES, 2)
    asks, phrase = console_cmd.question(command, robot.state)
    assert (prompt, phrase) == (asks.rstrip(), "yes")
    desk.type(" YES ")
    desk.wait(f"{command}: RESULT completed")
    (message,) = robot.settled()
    assert message == OwnerRequest(
        message.command_id,
        "s1",
        4,
        3,
        request=sends,
        reason="typed at the operator console",
    )
    # The state the answer is given for stands right above the question.
    asked = desk.said.index(prompt)
    assert desk.said[asked - 1] == console_cmd.state_text(robot.state)
    # A refusal of the robot app is printed as it is: it names no buttons.
    reason = "control owner is agent, not nobody"
    robot.reply = lambda message: [Result(message.command_id, "refused", reason)]
    desk.type(command)
    desk.wait(ASK_YES, 3)
    desk.type("yes")
    refusal = desk.wait(f"{command}: RESULT refused: ")
    assert refusal == f"{command}: RESULT refused: {reason}"


@pytest.mark.parametrize(("command", "sends"), TYPED)
def test_a_refusal_by_the_service_is_shown_with_the_buttons_to_use(
    robot: ScriptedRobot, desk: Desk, command: str, sends: str
) -> None:
    # The service of `robot` is made as its file ships: the setting is off.
    desk.type(command)
    desk.wait(ASK_YES)
    desk.type("yes")
    assert desk.wait(f"{command}: RESULT refused: ") == (
        f"{command}: RESULT refused: the console role may not send "
        f'OWNER_REQUEST "{sends}"; use right B / right A on the controller'
    )
    assert robot.settled() == []  # nothing of it reached the robot app


def test_quit_and_the_end_of_the_input_leave_the_console_and_send_nothing(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    for leave in ("quit", "the end of the input"):
        desk = desks(robot.bench.path)
        desk.wait("attached to the robot link service")
        if leave == "quit":
            desk.type("quit", "stop")  # nothing typed after it is served
        assert desk.end() == EXIT_OK
        assert desk.said[-1] == QUIT_NOTE and "the robot app runs on" in QUIT_NOTE
    assert robot.settled() == []
    until(lambda: not robot.bench.service._clients, "the consoles to detach")


def test_help_state_and_an_unknown_command_send_nothing(
    robot: ScriptedRobot, desk: Desk
) -> None:
    # `state` typed before the first ROBOT_STATE answers "none has arrived".
    desk.sees(lambda state: True)
    desk.type("help", "state", "move 1 2 3", "")
    assert "nothing is sent and the robot app runs on" in desk.wait("quit  ")
    desk.wait("robot link service: robot connection up, session s1; clients: 1 console")
    assert desk.wait("ROBOT_STATE: ") == console_cmd.state_text(robot.state)
    desk.wait("unknown command 'move 1 2 3'; type help")
    assert robot.settled() == []
    # The first line after the WELCOME names the robot and its driver.
    assert desk.wait("WELCOME: ") == (
        "WELCOME: robot sim-arm, profile sim-6dof, session s1, driver not "
        "declared (treated as real), camera calibration none"
    )


# On the real robot app, on the simulated arm


class Direct:
    """A robot connection handed to the console as its carrier: no service."""

    def __init__(self, end: Any) -> None:
        self._end = end
        self.notices: list[Any] = []
        self.close_reason = ""
        self.status = LocalStatus(True, "", "", None, False, {"console": 1})
        self.send_reliable = end.send_reliable
        self.send_latest = end.send_latest
        self.receive = end.receive
        self.close = end.close

    @property
    def alive(self) -> bool:
        return self._end.alive


@pytest.fixture
def on_the_app(
    make_robot_app: Callable[..., Any], desks: Callable[..., Desk]
) -> tuple[Any, Desk, Any]:
    """(the robot app's harness, a console on its robot connection, that end)."""
    harness = make_robot_app(connected=False)
    robot_end, console_end = loopback_pair()
    assert harness.app.attach_carrier(robot_end)
    desk = desks(None, attach=lambda *args: Direct(console_end))
    desk.wait("WELCOME: ")
    return harness, desk, console_end


def test_a_command_quotes_the_state_shown_with_its_question(
    on_the_app: tuple[Any, Desk, Any],
) -> None:
    h, desk, end = on_the_app
    h.hold_at(RAISED)
    desk.sees(lambda state: state.motion == "hold")
    desk.type("rest")
    desk.wait(ASK_REST)
    # While the operator types, the agent takes the arm and gives it back: the
    # control owner is nobody again, under a later owner number.
    for request in ("agent", "nobody"):
        seen = h.app.snapshot()
        asked = OwnerRequest(
            f"other-{request}",
            h.app.session_id,
            seen.owner_number,
            seen.stop_number,
            request=request,
        )
        send(end, asked)
        before = seen.owner_number
        until(lambda n=before: h.app.snapshot().owner_number > n, request)
    desk.type(PATH_IS_CLEAR)
    refusal = desk.wait("rest: RESULT refused: ")
    assert "owner number" in refusal, refusal
    h.idle()
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    # Asked again, with the state as it is now, the same rest runs.
    desk.sees(lambda state: state.owner_number == h.app.snapshot().owner_number)
    desk.type("rest")
    desk.wait(ASK_REST, 2)
    desk.type(PATH_IS_CLEAR)
    desk.wait("rest: RESULT completed")
    assert h.arm.sent and h.arm.torque_on()


def test_stop_is_taken_while_a_rest_runs(on_the_app: tuple[Any, Desk, Any]) -> None:
    h, desk, _ = on_the_app
    h.hold_at(RAISED)
    desk.sees(lambda state: state.motion == "hold")
    reached, release = h.hold_move_after(6)
    desk.type("rest")
    desk.wait(ASK_REST)
    desk.type(PATH_IS_CLEAR)
    desk.wait("rest: RESULT accepted")
    assert reached.wait(WAIT_S)  # the return is under way: 6 setpoints are out
    desk.type("stop")  # the prompt is free: the rest has not ended
    desk.wait("stop: sent")
    assert desk.count("rest: RESULT") == 1
    desk.wait("stop: taken.")
    release.set()
    desk.wait("rest: RESULT cancelled")
    h.idle()
    assert len(h.arm.sent) == 6  # nothing was sent after the STOP
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.motion_state is MotionState.HOLD
    assert h.app.snapshot().stop_number == 1


@pytest.mark.parametrize(("command", "sends"), TYPED)
def test_a_typed_takeover_the_robot_app_refuses_rings_three_bells(
    on_the_app: tuple[Any, Desk, Any], command: str, sends: str
) -> None:
    # No service here: the robot app itself answers, and human driving is off.
    h, desk, _ = on_the_app
    desk.sees(lambda state: True)
    # The takeover line says so before it names the human program that is
    # missing here too: starting one would change nothing.
    desk.type("state")
    assert desk.wait("takeover: ") == f"takeover: not possible: {SWITCHED_OFF}"
    desk.type(command)
    desk.wait(ASK_YES)
    desk.type("yes")
    refusal = desk.wait(f"{command}: RESULT refused: ")
    assert "use right B" not in refusal
    event = desk.wait(f"[refused] owner request '{sends}' refused: ")
    assert event.startswith("\a\a\a[refused] ") and event.count("\a") == 3
    h.idle()
    assert h.app.snapshot().owner == "nobody" and h.arm.sent == []


# Ctrl+C


def test_ctrl_c_withdraws_the_question_and_sends_one_stop(
    robot: ScriptedRobot, desk: Desk
) -> None:
    desk.type("rest")
    desk.wait(ASK_REST)
    desk.console.interrupt()
    desk.wait("question taken back (Ctrl+C); nothing sent for it")
    desk.wait("stop: taken.")
    # What is typed next is a new command, not the answer.
    desk.type(PATH_IS_CLEAR)
    desk.wait("unknown command 'path is clear'; type help")
    assert robot.settled() == [Stop(reason=STOP_CTRL_C, sender="console")]


def test_ctrl_c_never_ends_the_console(robot: ScriptedRobot, desk: Desk) -> None:
    for number in (1, 2, 3):
        desk.console.interrupt()
        desk.wait("stop: taken.", number)
    assert desk.running and desk.count("question taken back") == 0
    desk.type("reset")
    desk.wait("reset: RESULT completed")
    assert [type(message) for message in robot.settled()] == [Stop, Stop, Stop, Reset]
    assert desk.running


def console_at_a_terminal(socket_path: str, agent_marker: str = "") -> str:
    """The console program on this process's terminal; for on_a_terminal()."""
    said: list[str] = []
    for name in list(os.environ):
        if console_cmd.agent_environment({name: os.environ[name]}):
            del os.environ[name]
    if agent_marker:
        os.environ[agent_marker] = "1"

    def say(line: str) -> None:
        said.append(line)
        print(line, flush=True)  # on the terminal, where the test reads it

    code = main(["--socket", socket_path], say=say)
    return f"exit {code}: {said}"


def test_ctrl_c_at_the_terminal_is_a_stop_and_the_console_runs_on(
    tmp_path: Path,
) -> None:
    # No service: the STOP is tried raw, and said not to be written.
    answer, shown = on_a_terminal(
        lambda: console_at_a_terminal(str(tmp_path / "none.sock")),
        typed_after=b"\x03",
        marker=b"operator console: type stop",
        then=(b"STOP NOT written", b"state\nquit\n"),
    )
    assert shown and "exit 0: " in answer
    said = [
        answer.index(text)
        for text in (
            "stop: STOP NOT written: no robot link service at ",
            "ROBOT_STATE: none has arrived",  # served after the Ctrl+C
            QUIT_NOTE,
        )
    ]
    assert said == sorted(said)


# A terminal, and a person at it


def test_the_console_asks_nothing_without_a_terminal_or_inside_an_agent_environment(
    robot: ScriptedRobot,
) -> None:
    attached: list[Any] = []
    said: list[str] = []
    # Under pytest there is no terminal on stdin.
    code = main(["--socket", robot.bench.path], say=said.append, attach=attached.append)
    assert code == EXIT_UNUSABLE == 2
    assert said == [
        "operator console: it needs a terminal on standard input and output; "
        "nothing sent"
    ]
    for marker in ("CLAUDECODE", "OPENCLAW_HOME"):
        answer, shown = on_a_terminal(
            lambda marker=marker: console_at_a_terminal(robot.bench.path, marker),
            b"release\nyes\n",
            b"yes\n",
            ASK_YES.encode(),
        )
        assert not shown  # no question was put
        assert answer == repr(
            "exit 2: ['operator console: it is typed at by the operator, outside "
            f"an agent environment ({marker} is set); nothing sent']"
        )
    assert attached == [] and robot.settled() == []
    assert not robot.bench.service._clients


def test_what_is_typed_before_a_question_does_not_answer_it(
    robot: ScriptedRobot,
) -> None:
    def run() -> str:
        return console_at_a_terminal(robot.bench.path)

    # Typed once a status line shows a ROBOT_STATE. The yes typed ahead of the
    # question is dropped; the no typed at it counts.
    has_state = b" | owner nobody #4 | "
    answer, shown = on_a_terminal(
        run,
        typed_after=b"release\nyes\n",
        marker=has_state,
        then=(ASK_YES.encode(), b"no\nquit\n"),
    )
    assert shown and "exit 0: " in answer
    assert "release not confirmed; nothing sent" in answer
    assert robot.settled() == []
    answer, shown = on_a_terminal(
        run,
        typed_after=b"release\n",
        marker=has_state,
        then=(ASK_YES.encode(), b"yes\nquit\n"),
    )
    assert shown and "release: sent" in answer
    assert [type(message) for message in robot.settled()] == [Release]


def test_a_stop_typed_before_a_question_is_still_a_stop(robot: ScriptedRobot) -> None:
    # Both lines are in the terminal before the console reads the first one.
    answer, shown = on_a_terminal(
        lambda: console_at_a_terminal(robot.bench.path),
        typed_after=b"release\nstop\n",
        marker=b" | owner nobody #4 | ",
        then=(b"stop: taken.", b"quit\n"),
    )
    assert shown and "exit 0: " in answer
    assert "stop was typed before the question: the question is taken back" in answer
    assert "not confirmed; nothing sent" in answer
    assert robot.settled() == [TYPED_STOP]  # one STOP, and no RELEASE


def test_unusable_arguments_end_with_exit_code_2() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--no-such-option"], lines=iter(()), say=lambda line: None)
    assert raised.value.code == 2


# Turned away


def test_the_console_started_while_turned_away_can_still_stop(
    tmp_path_factory: pytest.TempPathFactory, desks: Callable[..., Desk]
) -> None:
    full = Bench(tmp_path_factory.mktemp("of"), max_clients=1)
    try:
        thor = full.robot()
        viewer = full.client("viewer")  # the one client slot
        desk = desks(full.path, tick_s=None)  # no attempt but the typed ones
        desk.wait(f"cannot attach to the robot link service: {FULL}")
        desk.wait("trying again every 2 s, 30 times at most")
        desk.type("rest", "state", "stop")
        desk.wait("rest: not attached: stop (written raw, not confirmed), attach,")
        line = desk.wait("STOP written; not confirmed: look at the Thor terminal")
        assert line.startswith("\astop: ") and "delivered" not in line
        assert thor.next_message() == TYPED_STOP
        thor.nothing()  # one STOP, and no rest
        # Ctrl+C is the same raw STOP.
        desk.console.interrupt()
        desk.wait("STOP written; not confirmed", 2)
        assert thor.next_message() == Stop(reason=STOP_CTRL_C, sender="console")
        # Once there is room, attach takes it and stop goes through the client.
        viewer.close()
        until(lambda: not full.service._clients, "the viewer to leave")
        desk.type("attach", "stop")
        desk.wait("attached to the robot link service: robot connection up")
        desk.wait("stop: sent; waiting for the robot app to show it")
        assert thor.next_message() == TYPED_STOP
        thor.nothing()  # never both ways for one stop
    finally:
        full.close()


def test_attach_attempts_are_paced_and_counted(desks: Callable[..., Desk]) -> None:
    clock = FakeClock()
    attempts: list[float] = []

    def attach(path: str | None, role: str, name: str, timeout_s: float) -> Any:
        assert (role, name) == ("console", "console")
        attempts.append(clock.monotonic())
        raise LocalRefused(FULL)

    desk = desks(None, clock=clock, tick_s=None, attach=attach)
    desk.wait("trying again every 2 s, 30 times at most")
    began = clock.monotonic()
    for _ in range(1000):  # 250 s of the console's clock
        clock.advance(0.25)
        desk.console.tick()
    assert ATTACH_TRIES == 30 and ATTACH_EVERY_S == 2.0
    assert attempts == [began + 2.0 * number for number in range(31)]
    desk.wait("no further attempt by itself; type attach to try again")
    assert desk.count("cannot attach to the robot link service") == 1
    # After that only the typed attach tries, once each time.
    desk.type("attach")
    desk.wait("cannot attach to the robot link service", 2)
    for _ in range(100):
        clock.advance(0.25)
        desk.console.tick()
    assert len(attempts) == 32


# The display


def test_a_help_request_prints_the_reason_the_jaws_and_the_countdown(
    robot: ScriptedRobot, desk: Desk
) -> None:
    holding = GripperState(0.4, 1.0, True, True, 0.4, True, "loaded")
    robot.show(gripper=holding, motion="hold", assistance_wait_s=118.0)
    desk.sees(lambda state: state.assistance_wait_s == 118.0)
    text = "HELP NEEDED: help needed: no grasp found; the arm holds and nobody owns it"
    data = {"reason": "no grasp found", "timed_return": True, "assistance_s": 120.0}
    robot.thor.send(Event(1, "help_needed", text, data))
    robot.thor.send(Event(2, "countdown", "110 s until the arm moves to rest"))
    desk.wait("[countdown] 110 s until the arm moves to rest")
    first = desk.said.index(f"[help_needed] {text}")
    assert desk.said[first + 1 : first + 9] == [
        "HELP NEEDED: no grasp found",
        "  jaws: loaded; they may be holding something",
        "  time left: 118 s; then the arm moves to rest by itself",
        "  you can type:",
        "    stop   a person's STOP ends the wait; the arm holds until someone acts",
        '    rest   return the arm to rest now; asks for a typed "path is clear"',
        "  or give a new task at the agent program's terminal",
        f"  {NO_PROGRAM}",
    ]
    # The status line keeps the wait and the jaws.
    line = desk.wait(" | wait 118 s | ")
    assert " | hold | jaws loaded, may be holding | torque on | " in line


def test_the_help_block_without_a_state_or_a_timed_return() -> None:
    event = Event(1, "help_needed", "HELP NEEDED: x", {"timed_return": False})
    assert help_block(event, None)[:3] == [
        "HELP NEEDED: HELP NEEDED: x",
        "  jaws: not known (no ROBOT_STATE)",
        "  time left: not known; then the arm keeps holding",
    ]
    waiting = dataclasses.replace(robot_state(), assistance_wait_s=None)
    event = Event(1, "help_needed", "", {"reason": "a\ab", "assistance_s": 120.0})
    assert help_block(event, waiting)[:3] == [
        "HELP NEEDED: a b",
        "  jaws: unknown",
        "  time left: 120 s; the bells say what the arm does then",
    ]


# A person at the controller


def test_takeover_text_repeats_the_thors_reason() -> None:
    for why in (
        SWITCHED_OFF,
        "fault-stop latched: motor fault",
        "the motors are off; restart the robot app",
        "no fresh controller data: stage waiting_for_headset",
        "a synthetic controller may drive only a simulated arm; this robot app "
        "declares driver real",
        "the live view is not fresh: no VIEW_STATUS has arrived; open the live "
        "view page in a visible browser tab",
    ):
        refused = with_human(takeover=why)
        assert takeover_text(refused, WITH_PROGRAM) == f"takeover: not possible: {why}"
    # A robot app that reports nothing of the human path, and none at all.
    assert takeover_text(robot_state(), WITH_PROGRAM) == (
        "takeover: not possible: this robot app does not report it"
    )
    odd = dataclasses.replace(robot_state(), health={"human": {"takeover": None}})
    assert takeover_text(odd, WITH_PROGRAM).endswith("does not report it")
    assert takeover_text(None, WITH_PROGRAM) == (
        "takeover: not possible: no ROBOT_STATE"
    )
    # What cannot be printed in the Thor's text becomes a space.
    assert takeover_text(with_human(takeover="a\ab"), WITH_PROGRAM).endswith(": a b")


def test_takeover_text_says_possible_when_all_is_there() -> None:
    assert takeover_text(with_human(), WITH_PROGRAM) == POSSIBLE
    # Right B is a STOP first: whoever owns the arm, a takeover can follow.
    owned = with_human(robot_state(owner="agent"))
    assert takeover_text(owned, WITH_PROGRAM) == POSSIBLE
    # The console keeps no limit of its own: the Thor's word alone decides.
    old = dataclasses.replace(with_human(stage="stale", age=9.0), view_age_s=None)
    assert takeover_text(old, WITH_PROGRAM) == POSSIBLE


def test_takeover_text_names_a_missing_human_program_first() -> None:
    # Before the Thor's reason, before a robot app that reports nothing and
    # before a missing ROBOT_STATE.
    for state in (with_human(takeover=SWITCHED_OFF), robot_state(), None):
        assert takeover_text(state, WITHOUT_PROGRAM) == NO_PROGRAM
    none = dataclasses.replace(WITH_PROGRAM, roles={"console": 1, "human": 0})
    assert takeover_text(with_human(), none) == NO_PROGRAM
    assert takeover_text(with_human(), WITHOUT_PROGRAM) == NO_PROGRAM
    # Not attached to the service: who is attached is not known.
    assert takeover_text(with_human(takeover=SWITCHED_OFF), None) == (
        f"takeover: not possible: {SWITCHED_OFF}"
    )
    assert takeover_text(with_human(), None) == POSSIBLE
    # Human driving is off in the robot app's file: a human program would
    # change nothing, so the Thor's word stands before the missing program.
    off = with_human(enabled=False, takeover=SWITCHED_OFF)
    assert takeover_text(off, WITHOUT_PROGRAM) == (
        f"takeover: not possible: {SWITCHED_OFF}"
    )
    # Never "possible" without a human program, whatever the Thor says.
    assert takeover_text(with_human(enabled=False), WITHOUT_PROGRAM) == NO_PROGRAM


def test_takeover_text_says_a_lost_robot_connection_before_an_old_state() -> None:
    # The console keeps its last ROBOT_STATE: it says nothing of the arm now.
    lost = dataclasses.replace(WITH_PROGRAM, robot_connected=False)
    for state in (with_human(), with_human(takeover=SWITCHED_OFF), robot_state()):
        assert takeover_text(state, lost) == LOST
    # Before the missing human program and the missing ROBOT_STATE too.
    alone = dataclasses.replace(WITHOUT_PROGRAM, robot_connected=False)
    assert takeover_text(with_human(), alone) == LOST
    assert takeover_text(None, alone) == LOST
    # A takeover that was open when the robot connection was lost stays so.
    held = dataclasses.replace(robot_state(owner="human"), motion="hold")
    for state in (held, robot_state()):
        taken = with_human(state, takeover_open=True)
        assert takeover_text(taken, lost) == (
            "takeover open: the robot connection is lost"
        )


def test_an_open_takeover_without_a_headset_tells_the_operator_how_to_end_it() -> None:
    held = dataclasses.replace(robot_state(owner="human"), motion="hold")
    why = "no fresh controller data: stage no_controller"
    for stage in ("", "stale", "no_controller", "no_server"):
        for start in (held, robot_state(owner="human"), robot_state()):
            lost = with_human(start, stage=stage, takeover_open=True, takeover=why)
            assert takeover_text(lost, WITH_PROGRAM) == NO_DATA
    # A human program that left: its last stage is still shown for a while.
    left = with_human(held, takeover_open=True)
    assert takeover_text(left, WITHOUT_PROGRAM) == NO_DATA
    # With the controller there the line says what the takeover needs next.
    assert takeover_text(left, WITH_PROGRAM) == "takeover open: a person owns the arm"
    stopped = dataclasses.replace(held, owner="nobody")
    assert takeover_text(with_human(stopped, takeover_open=True), WITH_PROGRAM) == (
        "takeover open: nobody owns the arm: hold right B for 1 s to take it again"
    )
    for owner in ("human", "nobody"):
        at_rest = with_human(robot_state(owner=owner), takeover_open=True)
        assert takeover_text(at_rest, WITH_PROGRAM) == (
            "takeover open: at rest: hold right A for 1 s to give the arm back"
        )
    # A return to rest under way: the recovery owns the arm until it ends.
    returning = dataclasses.replace(held, owner="recovery", motion="moving")
    assert takeover_text(with_human(returning, takeover_open=True), None) == (
        "takeover open: recovery owns the arm"
    )
    # Nobody owns the arm and the Thor would refuse right B: its word, no
    # button. A person who owns the arm needs no new takeover.
    view = (
        "the live view is not fresh: the view is not live; open the live view "
        "page in a visible browser tab"
    )
    facts = {"takeover_open": True, "takeover": view}
    assert takeover_text(with_human(stopped, **facts), WITH_PROGRAM) == (
        f"takeover open: nobody owns the arm: a new takeover is not possible: {view}"
    )
    assert takeover_text(with_human(held, **facts), WITH_PROGRAM) == (
        "takeover open: a person owns the arm"
    )
    # The hand-back at rest does not need the live view.
    assert takeover_text(with_human(robot_state(), **facts), WITH_PROGRAM) == (
        "takeover open: at rest: hold right A for 1 s to give the arm back"
    )
    # A fault-stop refuses the takeover and the hand-back alike, away from
    # rest and at rest, with a controller and without: reset comes first.
    facts["takeover"] = "fault-stop latched: motor 3 over temperature"
    for start in (stopped, robot_state()):
        for stage in ("ready", "no_controller"):
            fault = dataclasses.replace(start, stop_level="fault")
            latched = with_human(fault, stage=stage, **facts)
            assert takeover_text(latched, WITH_PROGRAM) == (
                "takeover open: a fault-stop is latched: type reset first"
            )


def test_the_help_block_says_whether_a_takeover_is_possible(
    robot: ScriptedRobot, desk: Desk
) -> None:
    event = Event(1, "help_needed", "HELP NEEDED: x", {"reason": "no grasp found"})
    # The block's last line, below what can be typed.
    block = help_block(event, with_human(), WITH_PROGRAM)
    assert len(block) == 8 and block[-1] == f"  {POSSIBLE}"
    why = "the live view is not fresh: the view is not live"
    block = help_block(event, with_human(takeover=why), WITH_PROGRAM)
    assert block[-1] == f"  takeover: not possible: {why}"
    # At the console the line counts who is attached to the service: no human
    # program here, whatever the robot app says.
    facts = {"human": human_facts()}
    robot.show(teleop_stage="ready", controller_age_s=0.03, health=facts)
    desk.sees(lambda state: "human" in state.health)
    robot.thor.send(event)
    desk.wait(f"  {NO_PROGRAM}")
    first = desk.said.index("[help_needed] HELP NEEDED: x")
    assert desk.said[first + 8] == f"  {NO_PROGRAM}"
    # Once one is attached, the robot app's word is repeated.
    robot.bench.client("human")
    until(
        lambda: desk.console._station.carrier.status.roles.get("human") == 1,
        "the human program at the console",
    )
    robot.thor.send(dataclasses.replace(event, seq=2))
    desk.wait(f"  {POSSIBLE}")


def test_state_prints_the_takeover_line(
    robot: ScriptedRobot, desks: Callable[..., Desk], tmp_path: Path
) -> None:
    desk = desks(robot.bench.path, clock=FakeClock(), tick_s=None)
    desk.wait("attached to the robot link service")
    desk.sees(lambda state: True)
    desk.type("state")
    desk.wait(NO_PROGRAM)
    # It is the last line, below the state and the status line.
    assert desk.said[-3] == console_cmd.state_text(robot.state)
    assert " | owner nobody #4 | " in desk.said[-2] and desk.said[-1] == NO_PROGRAM
    # An open takeover: how it ends.
    facts = {"human": human_facts(takeover_open=True)}
    robot.show(owner="human", motion="hold", health=facts)
    desk.sees(lambda state: state.owner == "human")
    desk.type("state")
    desk.wait(NO_DATA)
    assert " | owner human #4 | " in desk.said[-2] and desk.said[-1] == NO_DATA
    # Not attached: nothing is known of the robot.
    alone = desks(str(tmp_path / "none.sock"), tick_s=None)
    alone.wait("cannot attach to")
    alone.type("state")
    alone.wait("takeover: not possible: no ROBOT_STATE")
    assert alone.said[-1] == "takeover: not possible: no ROBOT_STATE"


def test_the_status_line_shows_the_headset_the_controller_age_and_the_note() -> None:
    ready = with_human(note=TAKE_OVER_NOTE)
    assert status_line(None, ready, WITH_PROGRAM, 0.04).endswith(
        " | permit none | view none | headset ready | controller 0.03 s | "
        f"note: {TAKE_OVER_NOTE} | state 0.04 s"
    )
    # A stage without controller data has no age to show.
    waiting = with_human(stage="waiting_for_headset", age=None)
    assert status_line(None, waiting, WITH_PROGRAM, 0.04).endswith(
        " | view none | headset waiting_for_headset | state 0.04 s"
    )
    # A person owns the arm, after a contact, before the first new squeeze.
    owned = with_human(
        robot_state(owner="human"),
        age=0.127,
        takeover_open=True,
        contact_hold=True,
        note="release the grip, then squeeze it to move\a",
    )
    assert status_line(None, owned, WITH_PROGRAM, 0.04).endswith(
        " | view none | headset ready | controller 0.13 s | takeover open | "
        "needs release | contact hold | note: release the grip, then squeeze "
        "it to move  | state 0.04 s"
    )
    # Under any other owner the needs-release flag asks nothing of anybody.
    assert "needs release" not in status_line(None, ready, WITH_PROGRAM, 0.04)
    released = with_human(robot_state(owner="human"), needs_release=False)
    assert "needs release" not in status_line(None, released, WITH_PROGRAM, 0.04)
    # A robot app without the human path: the line is as it was.
    assert status_line(None, robot_state(), WITH_PROGRAM, 0.04).endswith(
        " | torque on | permit none | view none | state 0.04 s"
    )


def test_a_human_drive_shows_as_moving_and_human_driving() -> None:
    driven = dataclasses.replace(
        robot_state(owner="human", owner_number=6),
        motion="moving",
        move_id=HUMAN_MOVE_ID,
        move_index=14,
        move_total=0,
    )
    facts = {"takeover_open": True, "needs_release": False}
    driving = with_human(driven, driving=True, **facts)
    line = status_line(None, driving, WITH_PROGRAM, 0.02)
    assert " | owner human #6 | stop none #0 | moving 14/0 | jaws " in line
    assert line.endswith(
        " | headset ready | controller 0.03 s | human driving | takeover open | "
        "state 0.02 s"
    )
    # The grip is released: the arm holds and the person keeps it.
    held = dataclasses.replace(driven, motion="hold", move_id="", move_index=0)
    line = status_line(None, with_human(held, **facts), WITH_PROGRAM, 0.02)
    assert " | hold | jaws " in line and "human driving" not in line
    assert " | controller 0.03 s | takeover open | state 0.02 s" in line


def test_a_new_status_line_is_printed_when_the_note_or_the_takeover_changes(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    # The key: each of these facts alone makes a new line due; an age does not.
    shown = with_human()
    for change in (
        {"stage": "stale"},
        {"contact_hold": True},
        {"enabled": False},
        {"takeover_open": True},
        {"driving": True},
        {"needs_release": False},
        {"note": "squeeze the grip to move"},
        {"ended": "contact on joint index 2"},
        {"takeover": "no fresh controller data: stage stale"},
    ):
        changed = status_key(with_human(**change), WITH_PROGRAM)
        assert changed != status_key(shown, WITH_PROGRAM), change
    assert status_key(with_human(age=0.9), WITH_PROGRAM) == (
        status_key(shown, WITH_PROGRAM)
    )
    # At the console, on its own clock.
    clock = FakeClock()
    desk = desks(robot.bench.path, clock=clock, tick_s=None)
    desk.sees(lambda state: True)

    def lines(age: float = 0.03, **facts: Any) -> int:
        """Show these facts a second later; the status lines printed so far."""
        health = {"human": human_facts(**facts)}
        robot.show(teleop_stage="ready", controller_age_s=age, health=health)
        desk.sees(lambda state: state.seq == robot.state.seq)
        clock.advance(1.0)
        desk.console.tick()
        assert desk.console._flush()
        return desk.count(" | owner nobody #4 | ")

    assert lines(note=TAKE_OVER_NOTE) == 1
    assert f" | controller 0.03 s | note: {TAKE_OVER_NOTE} | state " in desk.said[-1]
    assert lines(age=0.05, note=TAKE_OVER_NOTE) == 1  # only an age moved
    squeeze = "squeeze the grip to move"
    assert lines(note=squeeze) == 2
    assert f" | note: {squeeze} | state " in desk.said[-1]
    # Why no takeover is possible is not in the line; `state` prints it.
    stale = "the live view is not fresh: 0.612 s old, the limit is 0.5 s"
    assert lines(note=squeeze, takeover=stale) == 3
    # The seconds in that reason move all the time: no line for them.
    assert lines(note=squeeze, takeover=stale.replace("0.612", "0.731")) == 3
    assert lines(note=squeeze) == 4


def test_the_status_line_says_no_human_program_when_human_driving_is_on_and_none_is_attached() -> (
    None
):
    on = with_human()
    line = status_line(None, on, WITHOUT_PROGRAM, 0.04)
    # In place of the stage, which outlasts a human program that left.
    assert line.endswith(
        " | view none | no human program | controller 0.03 s | state 0.04 s"
    )
    assert "headset" not in line
    attached = status_line(None, on, WITH_PROGRAM, 0.04)
    assert " | headset ready | " in attached and "no human program" not in attached
    # Also with a takeover open, whatever the setting says.
    taken = with_human(enabled=False, stage="", age=None, takeover_open=True)
    assert status_line(None, taken, WITHOUT_PROGRAM, 0.04).endswith(
        " | view none | no human program | takeover open | state 0.04 s"
    )
    # Not with human driving off and no takeover open.
    off = with_human(enabled=False)
    line = status_line(None, off, WITHOUT_PROGRAM, 0.04)
    assert " | headset ready | " in line and "no human program" not in line
    # Not without a LOCAL_STATUS: who is attached is not known then.
    line = status_line(None, on, None, 0.04)
    assert " | headset ready | " in line and "no human program" not in line
    # Not for a robot app without the human path.
    assert "no human program" not in status_line(
        None, robot_state(), WITHOUT_PROGRAM, 0.04
    )
    # A human program that attaches or leaves makes a new line due.
    assert status_key(on, WITHOUT_PROGRAM) != status_key(on, WITH_PROGRAM)


def test_the_bell_event_rings_the_bell(robot: ScriptedRobot, desk: Desk) -> None:
    robot.thor.send(Event(1, "bell", "ARM WILL MOVE TO REST in 10 s"))
    robot.thor.send(Event(2, "countdown", "10 s\a left"))
    robot.thor.send(Event(3, "expired", "the assistance wait ran out"))
    desk.wait("[expired] the assistance wait ran out")
    assert "\a[bell] ARM WILL MOVE TO REST in 10 s" in desk.said
    # Only the kind rings: a bell character in a text is printed as a space.
    assert "[countdown] 10 s  left" in desk.said
    assert sum("\a" in line for line in desk.said) == 1


def test_the_bells_tell_a_drive_end_a_takeover_and_a_refusal_apart(
    robot: ScriptedRobot, desk: Desk
) -> None:
    # The wearer of the headset cannot read: one bell for the end of a human
    # drive, two for a takeover that began or ended, three for a refused one.
    heard = [
        (
            "human",
            "human drive ended: controller data stale (0.31 s old); the arm "
            "holds; release the grip and squeeze again",
            1,
        ),
        (
            "owner",
            "TAKEOVER: a person owns the arm, owner number 7; release the grip, "
            "then squeeze it to move",
            2,
        ),
        ("owner", "takeover ended: the arm is given back; nobody owns it", 2),
        (
            "refused",
            "owner request 'human' refused: no fresh controller data: stage stale",
            3,
        ),
        ("refused", "owner request 'handback' refused: no takeover is open", 3),
        # No bell: another owner change, another refusal, another kind.
        ("owner", "the agent owns the arm, owner number 8", 0),
        ("refused", "owner request 'agent' refused: takeover not finished", 0),
        ("refused", "MOVE refused: stale stop number", 0),
        ("note", "TAKEOVER: no owner event", 0),
    ]
    for seq, (kind, text, bells) in enumerate(heard, 1):
        event = Event(seq, kind, text)
        assert event_bells(event) == bells
        robot.thor.send(event)
    desk.wait("[note] TAKEOVER: no owner event")
    for kind, text, bells in heard:
        (line,) = (line for line in desk.said if line.endswith(f"[{kind}] {text}"))
        assert line == "\a" * bells + f"[{kind}] {text}"
    assert event_bells(Event(1, "bell", "ARM WILL MOVE TO REST in 10 s")) == 1


def test_a_refused_takeover_rings_through_a_flood_of_other_refusals(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    clock = FakeClock()
    desk = desks(robot.bench.path, clock=clock, tick_s=None)
    desk.wait("attached to the robot link service")
    # More kinds of refusal in one second than the console tells apart.
    for seq in range(1, LINE_KINDS + 3):
        robot.thor.send(Event(seq, "refused", f"MOVE refused: reason number {seq}"))
    taken = "owner request 'human' refused: no fresh controller data: stage stale"
    for seq in range(100, 103):  # the same refusal three times in that second
        robot.thor.send(Event(seq, "refused", taken))
    back = "owner request 'handback' refused: no takeover is open"
    robot.thor.send(Event(103, "refused", back))
    robot.thor.send(Event(104, "note", "the end"))
    desk.wait("[note] the end")
    assert desk.count("[refused] MOVE refused: reason number") == LINE_KINDS
    # Each of the two rings three times, once: its repeats are counted.
    assert [line for line in desk.said if "owner request" in line] == [
        f"\a\a\a[refused] {taken}",
        f"\a\a\a[refused] {back}",
    ]
    clock.advance(1.5)  # the second is over
    desk.console.tick()
    desk.wait("2 more refusals of other kinds")
    assert desk.wait("2 more like: ") == f"2 more like: [refused] {taken}"
    assert sum(line.count("\a") for line in desk.said) == 6


def test_a_refusal_of_another_client_is_shown(robot: ScriptedRobot, desk: Desk) -> None:
    robot.thor.send(Result("agent-7", "completed"))
    robot.thor.send(Result("agent-8", "refused", "the permit is too old"))
    desk.wait("(another client) RESULT refused: the permit is too old")
    assert desk.count("agent-7") == 0 and desk.count("RESULT completed") == 0
    # Its own refusal is under the command's name.
    reason = "no fault-stop is latched"
    robot.reply = lambda message: [Result(message.command_id, "refused", reason)]
    desk.type("reset")
    assert desk.wait(f"RESULT refused: {reason}") == f"reset: RESULT refused: {reason}"


def test_repeated_refusals_are_counted_not_printed(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    clock = FakeClock()
    desk = desks(robot.bench.path, clock=clock, tick_s=None)
    desk.wait("attached to the robot link service")
    refused = "[refused] MOVE refused: stale stop number"
    for seq in range(1, 6):
        robot.thor.send(Event(seq, "refused", "MOVE refused: stale stop number"))
        robot.thor.send(Result(f"agent-{seq}", "refused", "stale stop number"))
    robot.thor.send(Event(6, "note", "the end"))
    desk.wait("[note] the end")
    assert desk.count(refused) == 1
    assert desk.count("(another client) RESULT refused: stale stop number") == 1
    clock.advance(1.5)  # the second is over
    desk.console.tick()
    assert desk.wait("4 more like: [refused]") == f"4 more like: {refused}"
    desk.wait("4 more like: (another client) RESULT refused: stale stop number")
    assert desk.count(refused) == 2  # the line itself and its count


def test_a_lost_robot_connection_is_announced_once(
    robot: ScriptedRobot, desk: Desk
) -> None:
    desk.wait("robot connection up | clients 1 console | ")
    robot.thor.carrier.close()
    line = desk.wait("ROBOT CONNECTION LOST: ")
    assert line.startswith("\aROBOT CONNECTION LOST: ")
    assert line.endswith("; the robot holds on its own")
    desk.wait(" | clients 1 console | ", 2)  # a status line after the loss
    assert "robot connection lost (" in desk.wait("robot connection lost (")
    # Nothing that quotes numbers is sent now, and a STOP is said undelivered.
    desk.type("rest", "stop")
    desk.wait("rest: no ROBOT_STATE to quote (robot connection lost")
    assert desk.wait("stop: STOP was not delivered") == f"\astop: {STOP_UNDELIVERED}"
    # The last ROBOT_STATE is kept and shown, and says nothing of a takeover.
    desk.type("state")
    assert desk.wait("takeover: ") == LOST
    time.sleep(0.3)
    assert desk.count("ROBOT CONNECTION LOST") == 1


def test_a_stop_no_robot_state_shows_is_said_loudly_and_is_not_sent_again(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    clock = FakeClock()
    raw: list[Any] = []

    def raw_stop(*args: Any, **fields: Any) -> RawStop:
        raw.append(fields)
        return RawStop(True, "")

    desk = desks(robot.bench.path, clock=clock, tick_s=None, raw_stop=raw_stop)
    desk.wait("attached to the robot link service")
    robot.reply = lambda message: []  # the robot app shows nothing
    desk.type("stop")
    desk.wait("stop: sent; waiting for the robot app to show it")
    clock.advance(4.9)
    desk.console.tick()
    assert desk.count("STOP NOT CONFIRMED") == 0
    clock.advance(0.2)
    desk.console.tick()
    assert desk.wait("STOP NOT CONFIRMED") == (
        "\aSTOP NOT CONFIRMED: no ROBOT_STATE showed a higher stop number within "
        "5 s; use the STOP key at the Thor terminal"
    )
    assert robot.settled() == [TYPED_STOP] and raw == []


def test_a_dead_robot_link_service_is_said_and_stop_is_written_raw(
    tmp_path_factory: pytest.TempPathFactory, desks: Callable[..., Desk]
) -> None:
    bench = Bench(tmp_path_factory.mktemp("od"))
    raw: list[dict[str, Any]] = []

    def raw_stop(path: str | None, **fields: Any) -> RawStop:
        raw.append({"path": path, **fields})
        return RawStop(False, "nobody listens")

    try:
        thor = bench.robot()
        desk = desks(bench.path, tick_s=None, raw_stop=raw_stop)
        desk.wait("attached to the robot link service")
        desk.sees(lambda state: True)
        desk.type("stop")  # attached: through the client, and it waits
        desk.wait("stop: sent")
        assert thor.next_message() == TYPED_STOP
    finally:
        bench.close()
    until(lambda: not desk.console._station.carrier.alive, "the carrier to die")
    desk.console.tick()
    assert desk.wait("STOP NOT CONFIRMED") == (
        "\aSTOP NOT CONFIRMED: the robot link service is gone; type stop again or "
        "use the STOP key at the Thor terminal"
    )
    assert desk.wait("ROBOT LINK SERVICE GONE: ").startswith("\a")
    # What was shown is gone with the attachment: nothing can quote it.
    desk.type("rest", "state", "stop")
    desk.wait("rest: not attached: ")
    desk.wait("ROBOT_STATE: none has arrived")
    assert desk.wait("STOP NOT written") == (
        "\astop: STOP NOT written: nobody listens; use the STOP key at the Thor "
        "terminal"
    )
    assert raw == [
        {
            "path": bench.path,
            "reason": console_cmd.STOP_REASON,
            "sender": "console",
            "name": "console",
        }
    ]


def state_for_the_line() -> RobotState:
    permit = {
        "move_id": "m-7",
        "through_index": 62,
        "evidence_age_s": 0.214,
        "expires_in_s": 1.9,
    }
    return dataclasses.replace(
        robot_state(owner="agent", owner_number=12, stop_number=3),
        motion="moving",
        move_id="m-7",
        move_index=40,
        move_total=95,
        move_stage="sending",
        gripper=GripperState(1.0, 1.0, True, True, 1.0, False, "empty"),
        view_age_s=0.31,
        health={"permit": permit, "permit_refusal": ""},
    )


def test_status_line_names_owner_stop_permit_view_and_ages() -> None:
    state = state_for_the_line()
    of_the_state = (
        "owner agent #12 | stop none #3 | moving 40/95 sending | jaws empty | "
        "torque on | permit 62, evidence 0.21 s | view 0.3 s | state 0.04 s"
    )
    assert status_line(None, state, None, 0.04) == (
        f"not attached to the robot link service | {of_the_state}"
    )
    status = LocalStatus(True, "", "s1", None, False, {"console": 2, "agent": 1}, True)
    declared = dataclasses.replace(welcome(), driver="sim")
    assert status_line(declared, state, status, 0.04) == (
        "robot connection up | clients 1 agent, 2 console | a STOP is not "
        f"confirmed yet | driver sim | {of_the_state}"
    )
    # A fault-stop, a wait, an expired permit, a refused one, no view, no age.
    health = {
        "permit": {"through_index": 9, "evidence_age_s": 4.0, "expires_in_s": -0.5},
        "permit_refusal": "the evidence is 4.0 s old",
    }
    faulted = dataclasses.replace(
        state,
        stop_level="fault",
        fault_reason="motor fault",
        motion="hold",
        move_id="",
        assistance_wait_s=41.6,
        torque_on=False,
        view_age_s=None,
        health=health,
    )
    lost = LocalStatus(False, "peer timed out", "s1", None, False, {"console": 1})
    assert status_line(welcome(), faulted, lost, None) == (
        "robot connection lost (peer timed out) | clients 1 console | driver not "
        "declared (treated as real) | owner agent #12 | stop fault #3 (motor "
        "fault) | hold | jaws empty | torque off | wait 42 s | permit 9, "
        "evidence 4.00 s, expired | permit refused: the evidence is 4.0 s old | "
        "view none | state age unknown"
    )
    assert status_line(None, None, lost, None) == (
        "robot connection lost (peer timed out) | clients 1 console | no ROBOT_STATE"
    )
    no_permit = dataclasses.replace(state, health={})
    assert " | permit none | " in status_line(None, no_permit, None, 1.0)
    # A new line is due for a change of these facts, not for the ages.
    older = dataclasses.replace(state, seq=9, move_index=41, view_age_s=0.9)
    assert status_key(older, status) == status_key(state, status)
    assert status_key(faulted, status) != status_key(state, status)
    assert status_key(state, lost) != status_key(state, status)
    assert welcome_line(dataclasses.replace(welcome(), driver="apai-mock")) == (
        "WELCOME: robot sim-arm, profile sim-6dof, session s1, driver apai-mock, "
        "camera calibration none"
    )


def test_a_status_line_is_printed_on_a_change_and_every_ten_seconds(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    clock = FakeClock()
    desk = desks(robot.bench.path, clock=clock, tick_s=None)
    desk.sees(lambda state: True)
    owner = " | owner nobody #4 | stop none #3 | rest_hold | "
    desk.console.tick()
    desk.wait(owner)
    for _ in range(9):
        clock.advance(1.0)
        desk.console.tick()
    assert desk.count(owner) == 1  # nothing changed
    clock.advance(1.0)
    desk.console.tick()
    assert desk.wait(owner, 2).endswith(" | state 10.00 s")
    robot.show(owner="agent", owner_number=5)
    desk.sees(lambda state: state.owner == "agent")
    clock.advance(1.0)
    desk.console.tick()
    desk.wait(" | owner agent #5 | ")


# What confirms a STOP, and what a STOP never waits behind


def test_a_new_session_does_not_confirm_a_stop(
    robot: ScriptedRobot, desk: Desk
) -> None:
    fresh = dataclasses.replace(
        robot_state(seq=50, session_id="s2", owner_number=0, stop_number=0),
        motion="hold",
    )
    # The robot side drops the STOP and shows a restarted robot app.
    robot.reply = lambda message: (
        [welcome("s2"), fresh] if isinstance(message, Stop) else []
    )
    desk.type("stop")
    assert desk.wait("STOP NOT CONFIRMED") == (
        "\aSTOP NOT CONFIRMED: the robot app shows a new session (s2), which "
        "says nothing about this STOP; type stop again or use the STOP key at "
        "the Thor terminal"
    )
    time.sleep(0.2)
    assert desk.count("stop: taken.") == 0
    # The next STOP is measured against the new session.
    robot.state = fresh
    robot.reply = lambda message: [stopped(robot)]
    desk.type("stop")
    assert "stop number 1, stop level stop" in desk.wait("stop: taken.")


def test_a_stop_sent_before_any_state_waits_for_a_higher_stop_number(
    bench: Bench, desks: Callable[..., Desk]
) -> None:
    thor = bench.robot(None)  # connected; no WELCOME and no ROBOT_STATE yet
    desk = desks(bench.path)
    desk.wait("attached to the robot link service")
    desk.type("stop")
    desk.wait("stop: sent; waiting for the robot app to show it")
    assert thor.next_message() == TYPED_STOP
    thor.send(welcome("s1"))
    old = dataclasses.replace(
        robot_state(seq=1, stop_number=7), stop_level="stop", motion="hold"
    )
    thor.send(old)  # a stop of an earlier STOP: it answers nothing
    desk.sees(lambda state: state.stop_number == 7)
    time.sleep(0.2)
    assert desk.count("stop: taken.") == 0
    thor.send(dataclasses.replace(old, seq=2, stop_number=8))
    assert "stop number 8, stop level stop" in desk.wait("stop: taken.")


def test_stops_typed_again_keep_the_first_deadline(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    clock = FakeClock()
    desk = desks(robot.bench.path, clock=clock, tick_s=None)
    desk.wait("attached to the robot link service")
    desk.sees(lambda state: True)
    robot.reply = lambda message: []  # the robot app shows nothing
    desk.type("stop")
    desk.wait("stop: sent; waiting for the robot app to show it")
    clock.advance(4.9)
    desk.console.tick()
    desk.type("stop")
    desk.wait("stop: sent; waiting for the robot app to show it", 2)
    assert desk.count("STOP NOT CONFIRMED") == 0
    clock.advance(0.2)  # 5.1 s after the first STOP, 0.2 s after the second
    desk.console.tick()
    desk.wait("STOP NOT CONFIRMED")
    assert robot.settled() == [TYPED_STOP, TYPED_STOP]  # each one was sent
    # A STOP after that waits its own 5 s.
    desk.type("stop")
    desk.wait("stop: sent; waiting for the robot app to show it", 3)
    clock.advance(4.9)
    desk.console.tick()
    assert desk.count("STOP NOT CONFIRMED") == 1
    clock.advance(0.2)
    desk.console.tick()
    desk.wait("STOP NOT CONFIRMED", 2)


def test_a_command_before_the_first_state_answers_at_once_and_a_stop_is_not_held_up(
    bench: Bench, desks: Callable[..., Desk]
) -> None:
    thor = bench.robot(None)  # connected; no ROBOT_STATE yet
    desk = desks(bench.path)
    desk.wait("attached to the robot link service")
    until(
        lambda: desk.console._station.carrier.status.robot_connected,
        "robot connected",
    )
    began = time.monotonic()
    desk.type("reset", "stop")
    desk.wait("stop: sent; waiting for the robot app to show it")
    took = time.monotonic() - began
    line = desk.wait("reset: no ROBOT_STATE has arrived")
    assert line.endswith("; nothing sent; type reset again once one is shown")
    assert took < 1.0, f"the typed stop went out {took:.2f} s after it was typed"
    assert thor.next_message() == TYPED_STOP
    thor.nothing()  # and no RESET


class Stuck(Direct):
    """A carrier whose sends other than a STOP wait until `free` is set."""

    def __init__(self, end: Any) -> None:
        super().__init__(end)
        self.free = threading.Event()
        self.waiting = threading.Event()
        self.send_reliable = self._send

    def _send(self, msg_type: int, payload: bytes) -> None:
        if msg_type != Stop.TYPE:
            self.waiting.set()
            assert self.free.wait(WAIT_S)
        self._end.send_reliable(msg_type, payload)


def test_a_ctrl_c_stop_does_not_wait_behind_a_command_whose_send_is_stuck(
    desks: Callable[..., Desk],
) -> None:
    robot_end, console_end = loopback_pair()
    carrier = Stuck(console_end)
    desk = desks(None, attach=lambda *args: carrier)
    desk.wait("attached to the robot link service")
    send(robot_end, robot_state(seq=1))
    desk.sees(lambda state: True)
    try:
        desk.type("reset")
        assert carrier.waiting.wait(WAIT_S)  # the RESET is in its send
        began = time.monotonic()
        desk.console.interrupt()
        got = robot_end.receive(1.0)
        took = time.monotonic() - began
        assert got is not None, "the STOP waited behind the stuck send"
        assert got[0] == Stop.TYPE and took < 1.0
        assert desk.count("reset: sent") == 0
    finally:
        carrier.free.set()
    desk.wait("reset: sent; its RESULT is printed when it comes")
    # Its RESULT is still the console's own.
    command = decode(*robot_end.receive(WAIT_S))
    assert isinstance(command, Reset)
    send(robot_end, Result(command.command_id, "completed"))
    desk.wait("reset: RESULT completed")


class Dying(Direct):
    """A carrier that dies under the send of a STOP; the STOP is not out."""

    def __init__(self, end: Any) -> None:
        super().__init__(end)
        self.dead = False
        self.send_reliable = self._send

    def _send(self, msg_type: int, payload: bytes) -> None:
        if msg_type == Stop.TYPE:
            self.dead = True
            return
        self._end.send_reliable(msg_type, payload)

    @property
    def alive(self) -> bool:
        return not self.dead and self._end.alive


def test_a_stop_whose_carrier_dies_under_the_send_is_written_raw(
    desks: Callable[..., Desk],
) -> None:
    raw: list[dict[str, Any]] = []

    def raw_stop(path: str | None, **fields: Any) -> RawStop:
        raw.append(fields)
        return RawStop(True, "")

    robot_end, console_end = loopback_pair()
    desk = desks(None, attach=lambda *args: Dying(console_end), raw_stop=raw_stop)
    desk.wait("attached to the robot link service")
    desk.type("stop")
    line = desk.wait("stop: ")
    assert line == "\astop: STOP written; not confirmed: look at the Thor terminal"
    assert raw == [
        {"reason": console_cmd.STOP_REASON, "sender": "console", "name": "console"}
    ]
    assert desk.count("stop: sent") == 0 and robot_end.receive(0.2) is None


@pytest.fixture
def odd_service(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[bool], OddService]]:
    yield from odd_services(tmp_path_factory)


def test_a_typed_stop_does_not_wait_behind_an_attempt_to_attach(
    odd_service: Callable[[bool], OddService], desks: Callable[..., Desk]
) -> None:
    service = odd_service(False)  # takes the connection and answers nothing
    desk = desks(service.path)
    typed = time.monotonic()
    desk.type("stop")  # while the first attempt to attach waits its 2 s
    at, _ = service.wait_stop(1)
    assert at - typed < 0.5, f"the STOP was written {at - typed:.2f} s after the stop"
    desk.wait("STOP written; not confirmed: look at the Thor terminal")
    # A typed attach only asks for an attempt: the stop after it is read at once.
    desk.wait("cannot attach to the robot link service: ")
    typed = time.monotonic()
    desk.type("attach", "stop")
    at, _ = service.wait_stop(2)
    assert at - typed < 0.5, f"the STOP was written {at - typed:.2f} s after the stop"
    desk.wait("STOP written; not confirmed", 2)


FILLER = Event(seq=1, kind="note", text="x" * 60)


def unread(carrier: Any) -> None:
    """Fill the carrier's socket until a send of its own would wait."""
    record = pack_record(FILLER.TYPE, FILLER.to_payload())
    while select.select([], [carrier._sock], [], 0)[1]:
        carrier._sock.send(record, socket.MSG_DONTWAIT)


@pytest.mark.parametrize("behind", ["its own send", "a reset that is stuck"])
def test_a_ctrl_c_stop_does_not_wait_for_a_service_that_stopped_reading(
    odd_service: Callable[[bool], OddService], desks: Callable[..., Desk], behind: str
) -> None:
    service = odd_service(True)  # answers the attach, then reads nothing of it
    desk = desks(service.path)
    desk.wait("attached to the robot link service: robot connection up")
    desk.sees(lambda state: True)
    carrier = desk.console._station.carrier
    unread(carrier)
    if behind == "a reset that is stuck":
        desk.type("reset")
        until(carrier._send_lock.locked, "the RESET in its send")
    pressed = time.monotonic()
    desk.console.interrupt()
    at, connection = service.wait_stop(1)
    took = at - pressed
    assert took < 0.8, f"the STOP reached the service {took:.2f} s after the Ctrl+C"
    assert connection == 2  # written raw: nothing of it went into the stuck socket
    desk.wait("\astop: STOP written; not confirmed: look at the Thor terminal")
    assert desk.count("stop: sent") == 0
    desk.wait("ROBOT LINK SERVICE GONE: ")
    time.sleep(0.3)
    assert len(service.stops()) == 1  # one STOP, never both ways


def test_a_command_left_open_is_dropped_when_the_robot_connection_is_lost(
    desks: Callable[..., Desk],
) -> None:
    robot_end, console_end = loopback_pair()
    carrier = Direct(console_end)
    desk = desks(None, attach=lambda *args: carrier)
    desk.wait("attached to the robot link service")
    send(robot_end, welcome())
    send(robot_end, robot_state(seq=1))
    desk.sees(lambda state: True)
    desk.wait("robot connection up | ")  # the console has seen the robot up
    desk.type("reset")
    command = decode(*robot_end.receive(WAIT_S))
    send(robot_end, Result(command.command_id, "accepted"))
    desk.wait("reset: RESULT accepted")
    client = desk.console._station.client
    assert [r.status for r in client.results(command.command_id)] == ["accepted"]
    # The robot connection is lost, and no RESULT ends the command.
    carrier.status = LocalStatus(False, "link lost", "s1", None, False, {"console": 1})
    desk.wait("ROBOT CONNECTION LOST: link lost")
    assert desk.wait("reset: no RESULT will come") == (
        "reset: no RESULT will come: the robot connection was lost"
    )
    # Neither the console nor its client keeps anything of that command.
    assert desk.console._mine == {} and client._results == {}


# Ctrl+C while the console closes


@pytest.mark.parametrize("attached", [False, True])
def test_ctrl_c_while_the_console_closes_is_still_a_stop(
    tmp_path: Path, bench: Bench, desks: Callable[..., Desk], attached: bool
) -> None:
    raw: list[dict[str, Any]] = []

    def raw_stop(path: str | None, **fields: Any) -> RawStop:
        raw.append({"path": path, **fields})
        return RawStop(True, "")

    path = bench.path if attached else str(tmp_path / "none.sock")
    if attached:
        bench.robot()
    desk = desks(path, raw_stop=raw_stop)
    desk.wait("attached to" if attached else "cannot attach to")
    console = desk.console
    detach = console._detach

    def detach_with_ctrl_c() -> None:
        console.interrupt()  # what the signal handler does
        detach()

    console._detach = detach_with_ctrl_c
    desk.type("quit")
    assert desk.end() == EXIT_OK
    assert raw == [
        {"path": path, "reason": STOP_CTRL_C, "sender": "console", "name": "console"}
    ]
    assert desk.said[-2:] == [
        QUIT_NOTE,
        "\astop: STOP written; not confirmed: look at the Thor terminal",
    ]
    assert not console._threads[1].is_alive()  # the console ended


def test_ctrl_c_while_the_console_closes_reaches_the_robot(
    robot: ScriptedRobot, desk: Desk
) -> None:
    detach = desk.console._detach

    def detach_with_ctrl_c() -> None:
        desk.console.interrupt()
        detach()

    desk.console._detach = detach_with_ctrl_c
    desk.type("quit")
    assert desk.end() == EXIT_OK
    assert robot.settled() == [Stop(reason=STOP_CTRL_C, sender="console")]
    assert "STOP written; not confirmed" in desk.said[-1]


def test_ctrl_c_served_while_the_console_closes_goes_raw_and_once(
    robot: ScriptedRobot, desks: Callable[..., Desk]
) -> None:
    raw: list[dict[str, Any]] = []

    def raw_stop(path: str | None, **fields: Any) -> RawStop:
        raw.append(fields)
        return RawStop(True, "")

    desk = desks(robot.bench.path, raw_stop=raw_stop)
    desk.wait("attached to the robot link service: robot connection up")
    desk.sees(lambda state: True)
    console = desk.console
    detach = console._detach

    def detach_once_the_ctrl_c_was_served() -> None:
        # The console closes and is still attached while the stop thread runs.
        console.interrupt()
        until(lambda: raw or robot.forwarded, "the STOP")
        detach()

    console._detach = detach_once_the_ctrl_c_was_served
    desk.type("quit")
    assert desk.end() == EXIT_OK
    # Nothing goes through a client that is being closed.
    assert robot.settled() == []
    assert raw == [{"reason": STOP_CTRL_C, "sender": "console", "name": "console"}]
    assert desk.said[-2:] == [
        QUIT_NOTE,
        "\astop: STOP written; not confirmed: look at the Thor terminal",
    ]


class Held(Direct):
    """A carrier whose STOP goes out and whose send then waits for `go`."""

    def __init__(self, end: Any) -> None:
        super().__init__(end)
        self.go = threading.Event()
        self.out = threading.Event()
        self.send_reliable = self._send

    def _send(self, msg_type: int, payload: bytes) -> None:
        self._end.send_reliable(msg_type, payload)
        if msg_type == Stop.TYPE:
            self.out.set()
            assert self.go.wait(WAIT_S)


def test_a_stop_that_went_out_just_before_quit_is_still_said(
    desks: Callable[..., Desk],
) -> None:
    raw: list[Any] = []
    robot_end, console_end = loopback_pair()
    carrier = Held(console_end)
    desk = desks(
        None, attach=lambda *args: carrier, raw_stop=lambda *a, **k: raw.append(k)
    )
    desk.wait("attached to the robot link service")
    console = desk.console
    detach = console._detach

    def detach_after_the_stop_thread_went_on() -> None:
        carrier.go.set()  # the send of the STOP returns: the console closes now
        end = time.monotonic() + 0.5
        while desk.count("stop: sent") == 0 and time.monotonic() < end:
            time.sleep(0.01)
        detach()

    console._detach = detach_after_the_stop_thread_went_on
    console.interrupt()
    assert carrier.out.wait(WAIT_S)  # the STOP is out, its line is not said yet
    desk.type("quit")
    assert desk.end() == EXIT_OK
    assert decode(*robot_end.receive(WAIT_S)) == Stop(STOP_CTRL_C, "console")
    assert robot_end.receive(0.2) is None and raw == []
    assert desk.said[-2:] == [
        QUIT_NOTE,
        "\astop: sent; not confirmed: the console ends, look at the Thor terminal",
    ]


def test_ctrl_c_after_quit_while_the_ticker_still_attaches_is_a_stop(
    odd_service: Callable[[bool], OddService], desks: Callable[..., Desk]
) -> None:
    service = odd_service(False)  # every attempt to attach waits its 2 s
    desk = desks(service.path)
    desk.wait("trying again every 2 s")
    # The ticker has just begun its next attempt.
    service.wait_connections(service.connections + 1)
    desk.type("quit")
    desk.wait(QUIT_NOTE)
    time.sleep(0.2)
    assert desk.running  # the console waits for its ticker
    desk.console.interrupt()
    assert desk.end() == EXIT_OK
    (stop,) = service.stops()
    assert stop[1] > 2  # written raw, on a connection of its own
    assert desk.said[-2:] == [
        QUIT_NOTE,
        "\astop: STOP written; not confirmed: look at the Thor terminal",
    ]


def test_ctrl_c_right_after_the_stop_thread_ended_is_a_stop(
    tmp_path: Path, desks: Callable[..., Desk]
) -> None:
    raw: list[dict[str, Any]] = []

    def raw_stop(path: str | None, **fields: Any) -> RawStop:
        raw.append(fields)
        return RawStop(True, "")

    desk = desks(str(tmp_path / "none.sock"), raw_stop=raw_stop)
    desk.wait("cannot attach to")
    console = desk.console
    stop_thread = console._threads[1]
    join = stop_thread.join

    def join_then_ctrl_c(timeout: float | None = None) -> None:
        join(timeout)
        if not stop_thread.is_alive() and not raw:
            console.interrupt()  # nobody reads the interrupts any more

    stop_thread.join = join_then_ctrl_c
    desk.type("quit")
    assert desk.end() == EXIT_OK
    assert raw == [{"reason": STOP_CTRL_C, "sender": "console", "name": "console"}]
    assert desk.said[-2:] == [
        QUIT_NOTE,
        "\astop: STOP written; not confirmed: look at the Thor terminal",
    ]
    assert console._interrupts.empty()


# RESULTs are not kept


def test_results_of_other_clients_are_dropped_once_shown(
    robot: ScriptedRobot, desk: Desk
) -> None:
    client = desk.console._station.client
    for index in range(40):
        robot.thor.send(Result(f"agent-move-{index}", "accepted"))
        robot.thor.send(Result(f"agent-move-{index}", "completed"))
        time.sleep(0.005)
    robot.thor.send(Result("agent-last", "refused", "the permit is too old"))
    desk.wait("(another client) RESULT refused: the permit is too old")
    assert client._results == {}
    # Its own command: kept while it is open, dropped with its final RESULT.
    robot.reply = lambda message: [Result(message.command_id, "accepted")]
    desk.type("reset")
    desk.wait("reset: RESULT accepted")
    (command_id,) = desk.console._mine
    assert [r.status for r in client.results(command_id)] == ["accepted"]
    robot.thor.send(Result(command_id, "completed"))
    desk.wait("reset: RESULT completed")
    until(lambda: client._results == {}, "the RESULTs to be dropped")
    assert desk.console._mine == {}


def test_the_clients_result_store_is_bounded() -> None:
    robot_end, client_end = loopback_pair()
    client = WorkstationClient(client_end, FakeClock())
    kept = client_module.RESULTS_KEPT
    try:
        send(robot_end, Result("open-1", "accepted"))  # the oldest; not ended
        for index in range(kept + 50):
            send(robot_end, Result(f"other-{index}", "accepted"))
            send(robot_end, Result(f"other-{index}", "completed"))
        for _ in range(client_module.RESULTS_PER_COMMAND + 20):
            send(robot_end, Result("chatty", "accepted"))
        send(robot_end, Result("last", "completed"))
        client.wait_result("last")
        assert len(client._results) == kept
        # The ended commands went first: the open one is still there.
        assert [r.status for r in client.results("open-1")] == ["accepted"]
        assert client.results("other-0") == []
        assert len(client.results("chatty")) == client_module.RESULTS_PER_COMMAND
    finally:
        client.close()
        robot_end.close()
        client_end.close()


# The source


def test_the_event_kinds_are_the_robot_apps() -> None:
    assert status_module.KIND_BELL == waits.BELL
    assert status_module.KIND_REFUSED == app_module.EVENT_REFUSED
    assert status_module.KIND_HELP_NEEDED == app_module.EVENT_HELP_NEEDED
    assert status_module.KIND_HUMAN == app_module.EVENT_HUMAN
    assert status_module.KIND_OWNER == app_module.EVENT_OWNER
    # So are the texts that ring twice and the words of ROBOT_STATE it reads.
    taken = Event(1, app_module.EVENT_OWNER, app_module.TAKEN_OVER.format(number=7))
    assert event_bells(taken) == 2
    assert event_bells(Event(1, app_module.EVENT_OWNER, app_module.HANDED_BACK)) == 2
    assert status_module._STAGE_READY == app_module.STAGE_READY
    assert status_module._REST_HOLD == MotionState.REST_HOLD.value
    assert status_module._OWNER_HUMAN == Owner.HUMAN.value
    assert status_module._OWNER_NOBODY == Owner.NOBODY.value
    assert status_module._STOP_FAULT == StopLevel.FAULT.value
    assert SWITCHED_OFF == app_module.HUMAN_OFF


def test_the_console_holds_no_copy_of_a_send() -> None:
    files = sorted(PACKAGE.glob("*.py"))
    assert [path.name for path in files] == [
        "__init__.py",
        "__main__.py",
        "console.py",
        "status.py",
    ]
    calls: list[str] = []
    names: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls += [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
    sending = {
        "reset",
        "rest",
        "release",
        "shutdown",
        "open_jaws",
        "request_owner",
        "send",
        "send_reliable",
        "send_latest",
        "move",
        "run_move",
        "permit",
        "gripper",
        "checker_state",
    }
    assert not [call for call in calls if call.rsplit(".", 1)[-1] in sending]
    # STOP is the one link message it sends itself, at one place.
    assert [call for call in calls if call.endswith(".stop")] == ["station.client.stop"]
    assert not names & {"Reset", "Release", "Open", "Shutdown", "OwnerRequest", "Stop"}
    assert {"send_command", "question"} <= names


def test_the_console_imports_without_apai_torch_or_isaac_capture() -> None:
    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.console import "
        "__main__, console, status\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in "
        "('apai', 'torch', 'cupy', 'warp', 'numpy', 'yaml', 'isaaccapture')]\n"
        "assert not bad, bad\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("AGENTIC_REMOTE_APAI_SRC", "PYTHONPATH")
    }
    env["PYTHONPATH"] = str(EXAMPLE / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr


def test_run_console_sh_execs_the_console_with_the_examples_python_path() -> None:
    script = EXAMPLE / "run_console.sh"
    text = script.read_text(encoding="utf-8")
    assert os.access(script, os.X_OK)
    assert 'exec "${python}" -m isaaccapture_examples.agentic_remote.console "$@"' in (
        text
    )
    assert 'export PYTHONPATH="${here}/python' in text
    # Standard input is not a terminal here: the program says so and sends nothing.
    done = subprocess.run(
        [str(script), "--socket", "/nonexistent/robot_link.sock"],
        check=False,
        env={**os.environ, "AGENTIC_REMOTE_CONSOLE_PYTHON": sys.executable},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 2, done.stderr
    assert done.stdout.startswith("operator console: it needs a terminal")
