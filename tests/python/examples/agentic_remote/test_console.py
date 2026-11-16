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
    help_block,
    status_key,
    status_line,
    welcome_line,
)
from isaaccapture_examples.agentic_remote.link_service.local import (
    LocalRefused,
    RawStop,
)
from isaaccapture_examples.agentic_remote.link_service.service import STOP_UNDELIVERED
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair, send
from isaaccapture_examples.agentic_remote.protocol.confirm import PATH_IS_CLEAR
from isaaccapture_examples.agentic_remote.protocol.local_messages import LocalStatus
from isaaccapture_examples.agentic_remote.protocol.messages import (
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
)
from isaaccapture_examples.agentic_remote.robot_app import app as app_module
from isaaccapture_examples.agentic_remote.robot_app import waits
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState
from isaaccapture_examples.agentic_remote.robot_app.terminal import TypedLines
from link_bench import Bench, robot_state, until, welcome
from test_console_cmd import Console as ScriptedRobot
from test_console_cmd import completed, on_a_terminal

PACKAGE = Path(console_module.__file__).resolve().parent
EXAMPLE = PACKAGE.parents[3]
# Real seconds a test gives another thread before it fails.
WAIT_S = 10.0
FULL = "too many local clients are attached"
ASK_REST = 'Type "path is clear" to start'
ASK_YES = 'Type "yes" to '
TYPED_STOP = Stop(reason=console_cmd.STOP_REASON, sender="console")
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]


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
    return made


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
        typed_after=b"\x03state\nquit\n",
        marker=b"operator console: type stop",
    )
    assert shown and "exit 0: " in answer
    assert "stop: STOP NOT written: no robot link service at " in answer
    assert "ROBOT_STATE: none has arrived" in answer  # served after the Ctrl+C
    assert QUIT_NOTE in answer


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

    # The yes typed ahead of the question is dropped; the no typed at it counts.
    answer, shown = on_a_terminal(
        run, b"release\nyes\n", b"no\nquit\n", ASK_YES.encode()
    )
    assert shown and "exit 0: " in answer
    assert "release not confirmed; nothing sent" in answer
    assert robot.settled() == []
    answer, shown = on_a_terminal(run, b"release\n", b"yes\nquit\n", ASK_YES.encode())
    assert shown and "release: sent" in answer
    assert [type(message) for message in robot.settled()] == [Release]


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
    assert desk.said[first + 1 : first + 8] == [
        "HELP NEEDED: no grasp found",
        "  jaws: loaded; they may be holding something",
        "  time left: 118 s; then the arm moves to rest by itself",
        "  you can type:",
        "    stop   a person's STOP ends the wait; the arm holds until someone acts",
        '    rest   return the arm to rest now; asks for a typed "path is clear"',
        "  or give a new task at the agent program's terminal",
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


def test_the_bell_event_rings_the_bell(robot: ScriptedRobot, desk: Desk) -> None:
    robot.thor.send(Event(1, "bell", "ARM WILL MOVE TO REST in 10 s"))
    robot.thor.send(Event(2, "countdown", "10 s\a left"))
    robot.thor.send(Event(3, "expired", "the assistance wait ran out"))
    desk.wait("[expired] the assistance wait ran out")
    assert "\a[bell] ARM WILL MOVE TO REST in 10 s" in desk.said
    # Only the kind rings: a bell character in a text is printed as a space.
    assert "[countdown] 10 s  left" in desk.said
    assert sum("\a" in line for line in desk.said) == 1


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


# The source


def test_the_event_kinds_are_the_robot_apps() -> None:
    assert status_module.KIND_BELL == waits.BELL
    assert status_module.KIND_REFUSED == app_module.EVENT_REFUSED
    assert status_module.KIND_HELP_NEEDED == app_module.EVENT_HELP_NEEDED


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
