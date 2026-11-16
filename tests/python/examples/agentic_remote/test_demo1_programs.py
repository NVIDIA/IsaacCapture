# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The programs of demo 1 together, on one machine.

Demo 1: the agent completes a task while an operator watches the live view and
can type at the operator console. Here the robot link service listens on
127.0.0.1 and a port the system picked, and the robot app dials it over a real
link. Attached to the service at once: the agent program (launcher.main, in
this process, on APAI's mock camera), the operator console (the Console class
the console program runs, its lines typed by the test) and the live view (the
view program's main(), on a thread). The robot app runs on APAI's mock arm in
real time, and on the simulated arm where no agent is needed.

The last test starts the robot link service, the robot app, the live view, the
operator console (on a pseudo-terminal) and the console command as processes of
their own and ends each of them.
"""

from __future__ import annotations

import os
import pty
import re
import signal
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent import console_cmd, launcher, task_gate
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from isaaccapture_examples.agentic_remote.protocol.confirm import PATH_IS_CLEAR
from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState
from isaaccapture_examples.agentic_remote.view import __main__ as view_main
from isaaccapture_examples.agentic_remote.view.model import NO_PICTURE
from link_stack import HOST, Program, ServiceProcess, Stack, program_env
from test_agent_end_to_end import RUN_BUDGET_S, ObstacleGrid, Scene, raise_arm
from test_console import ASK_REST, QUIT_NOTE, Desk
from test_console_cmd import await_read
from test_launcher import PICK, QUICK, scene_robot_config
from test_link_programs import ROBOT_MODULE, ROBOT_YAML
from test_view_program import Browser, Running, StandInSource, write_config

from conftest import apai_runtime_env

apai = pytest.mark.apai
link = pytest.mark.link

# Real seconds a test gives a program before it fails.
WAIT_S = 10.0
# Real seconds a return to rest may take here.
REST_WAIT_S = 30.0
# From rest on the simulated arm: 25 setpoints, 0.5 s.
REST = [0.0] * 6
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
VIEW_MODULE = "isaaccapture_examples.agentic_remote.view"
CONSOLE_MODULE = "isaaccapture_examples.agentic_remote.console"
CONSOLE_CMD_MODULE = "isaaccapture_examples.agentic_remote.agent.console_cmd"
# A status line of the console that shows a view age.
VIEW_AGE = re.compile(r" \| view (\d+\.\d) s \| ")


class TimedDesk(Desk):
    """Desk of test_console.py that also notes when each line was printed."""

    def __init__(self, path: str | None, **options: Any) -> None:
        self.at: list[float] = []  # time.monotonic() of each line of .said
        super().__init__(path, **options)

    def _hear(self, line: str) -> None:
        with self._heard:
            self.at.append(time.monotonic())
        super()._hear(line)

    def wait_long(self, text: str, wait_s: float) -> str:
        """As wait(), for something that takes longer than a thread's turn."""

        def found() -> list[str]:
            return [line for line in self.said if text in line]

        with self._heard:
            assert self._heard.wait_for(lambda: bool(found()), wait_s), (
                f"no line with {text!r} within {wait_s:g} s; said: {self.said}"
            )
            return found()[0]

    def when(self, text: str) -> float:
        """When the first line that holds `text` was printed."""
        with self._heard:
            return next(at for at, line in zip(self.at, self.said) if text in line)

    def holding(self, text: str) -> list[str]:
        """The lines printed so far that hold `text`."""
        with self._heard:
            return [line for line in self.said if text in line]


class ConsoleProcess:
    """The operator console program in its own process, on a pseudo-terminal.

    python -m ...console --socket PATH, outside an agent's environment; the
    test types at the terminal and reads what the program printed on it.
    """

    def __init__(self, path: str) -> None:
        self.said: list[str] = []
        self._heard = threading.Condition()
        self._master, terminal = pty.openpty()
        attributes = termios.tcgetattr(terminal)
        attributes[3] &= ~termios.ECHO  # what the test types is not shown back
        termios.tcsetattr(terminal, termios.TCSANOW, attributes)
        env = {
            name: value
            for name, value in program_env().items()
            if console_cmd.agent_environment({name: value}) is None
        }
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", CONSOLE_MODULE, "--socket", path],
                stdin=terminal,
                stdout=terminal,
                stderr=terminal,
                env=env,
                start_new_session=True,
            )
        finally:
            os.close(terminal)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        pending = b""
        while True:
            try:
                data = os.read(self._master, 4096)
            except OSError:  # the program closed the terminal
                data = b""
            if not data:
                return
            *lines, pending = (pending + data).split(b"\n")
            with self._heard:
                self.said += [
                    line.decode(errors="replace").rstrip("\r") for line in lines
                ]
                self._heard.notify_all()

    def type(self, line: str) -> None:
        os.write(self._master, line.encode() + b"\n")

    def answer(self, question: str, line: str) -> None:
        """Type `line` once the question is shown and the program reads.

        The console drops what was typed before it reads the answer.
        """
        self.wait(question)
        await_read(self.process.pid, WAIT_S, unknown_s=0.5)
        self.type(line)

    def wait(self, text: str, wait_s: float = WAIT_S) -> str:
        """Wait for the first line that holds `text`; returns it."""

        def found() -> list[str]:
            return [line for line in self.said if text in line]

        with self._heard:
            assert self._heard.wait_for(lambda: bool(found()), wait_s), (
                f"no line with {text!r} within {wait_s:g} s; said: {self.said}"
            )
            return found()[0]

    def wait_exit(self) -> int:
        """Wait for the process to end; its exit code. Every line is in `said`."""
        code = self.process.wait(WAIT_S)
        self._reader.join(WAIT_S)
        return code

    def close(self) -> None:
        """End the process if it still runs, and the terminal."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
            self.process.wait(WAIT_S)
        if self._master >= 0:
            # The reader ends at the closed terminal.
            os.close(self._master)
            self._master = -1
        self._reader.join(WAIT_S)


class Tab:
    """What an open, visible browser tab does, a few times and with pauses.

    Each turn it asks for the picture and acknowledges the one it got; every
    third turn it reads the page's status. At most `turns` turns, pause_s
    apart: under three requests per second.
    """

    def __init__(self, port: int, turns: int = 16, pause_s: float = 1.5) -> None:
        self.statuses: list[dict[str, Any]] = []
        self.acknowledged = 0
        self._browser = Browser(port)
        self._turns, self._pause_s = turns, pause_s
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for turn in range(self._turns):
            seq, _ = self._browser.frame()
            if seq is not None:
                self._browser.shown(seq)
                self.acknowledged += 1
            if turn % 3 == 2:
                self.statuses.append(self._browser.status())
            if self._done.wait(self._pause_s):
                return

    def close(self) -> dict[str, Any]:
        """End the tab; the page's status at that moment."""
        self._done.set()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the tab did not end"
        status = self._browser.status()
        self._browser.close()
        return status


class Desks:
    """The consoles and views of one test; each is ended with the test."""

    def __init__(self, path: str, folder: Path) -> None:
        self.path, self.folder = path, folder
        self._desks: list[TimedDesk] = []
        self._views: list[Running] = []
        self._tabs: list[Tab] = []

    def console(self) -> TimedDesk:
        """An operator console attached to the service, with the robot there."""
        desk = TimedDesk(self.path)
        self._desks.append(desk)
        # The robot app may still be dialing: its first ROBOT_STATE says it is there.
        desk.wait("attached to the robot link service: ")
        desk.sees(lambda state: True)
        return desk

    def view(self, *args: str, source: Any = None, **blocks: dict[str, Any]) -> Running:
        """The live view's main() on a thread, attached to the service."""
        config = write_config(self.folder, self.path, **blocks)
        view = Running(config, *args, source=source)
        self._views.append(view)
        view.ready()
        return view

    def tab(self, view: Running, **options: Any) -> Tab:
        tab = Tab(view.program.port, **options)
        self._tabs.append(tab)
        return tab

    def close(self) -> None:
        for tab in self._tabs:
            tab._done.set()
            tab._thread.join(WAIT_S)
        for view in self._views:
            view.end()
        for desk in self._desks:
            desk.close()


@pytest.fixture(autouse=True)
def view_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """The view's main() attaches no log handler in a test."""
    monkeypatch.setattr(view_main, "set_console_level", lambda level: None)


@pytest.fixture
def demo(
    make_realtime_apai: Callable[..., Any],
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Scene, Desks]]:
    """The mock scene's robot app behind the service, and what attaches to it.

    As service_scene of test_agent_end_to_end.py: the robot app on APAI's mock
    arm declares the driver "apai-mock" and dials the service once.
    """
    harness = make_realtime_apai(config=scene_robot_config(), connected=False)
    raise_arm(harness)
    # A short path: a socket path holds 107 bytes at most.
    folder = tmp_path_factory.mktemp("d1")
    stack = Stack(harness, folder)
    desks = Desks(stack.path, folder)
    try:
        stack.start_service()
        stack.dial()
        stack.connected(stack.client("tool"))
        with apai_runtime_env(tmp_path) as state_folder:
            scene = Scene(harness, state_folder)
            scene.socket_path, scene.stack = stack.path, stack
            yield scene, desks
    finally:
        desks.close()
        stack.close()


@pytest.fixture
def sim_demo(
    make_realtime_robot_app: Callable[..., Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Stack, Desks]]:
    """The robot app on the simulated arm (driver "sim") behind the service."""
    folder = tmp_path_factory.mktemp("d1")
    stack = Stack(make_realtime_robot_app(connected=False), folder)
    desks = Desks(stack.path, folder)
    try:
        stack.start_service()
        stack.dial()
        yield stack, desks
    finally:
        desks.close()
        stack.close()


def owner_lines(desk: TimedDesk) -> list[str]:
    """The owners the console's status lines showed, changes only.

    A status line is printed at most once a second, so an owner of less than
    that can be missing; the [owner] lines name every change.
    """
    owners: list[str] = []
    for line in desk.holding(" | owner "):
        owner = line.split(" | owner ")[1].split(" #")[0]
        if not owners or owners[-1] != owner:
            owners.append(owner)
    return owners


# The agent program, the console and the view at once


@apai
@link
def test_demo_1_on_the_mock_arm_with_console_and_view(
    demo: tuple[Scene, Desks],
) -> None:
    scene, desks = demo
    trace, watcher = scene.trace, scene.stack.stations[0]
    desk = desks.console()
    # A view with pictures, and a tab that acknowledges them.
    view = desks.view(source=StandInSource(), status={"ack_timeout_s": 4.0})
    assert watcher.ws.state.view_age_s is None
    # The tab acknowledges until the test ends it: RUN_BUDGET_S at most.
    tab = desks.tab(view, turns=40)
    # The view's age reaches ROBOT_STATE and the console's status line.
    live = watcher.ws.wait_state(lambda s: s.view_age_s is not None, WAIT_S)
    assert 0.0 <= live.view_age_s < 4.0
    desk.sees(lambda state: state.view_age_s is not None)

    scene.typed = [QUICK, PICK]
    scene.answers = ["yes", "yes"]
    began = time.monotonic()
    code = scene.run("--interactive")
    took = time.monotonic() - began
    assert took < RUN_BUDGET_S
    tab_ended = time.monotonic()
    page = tab.close()

    # The tasks: the first succeeded; the pick and place moved the arm and
    # ended with a help request (the mock scene cannot confirm the place).
    opened, picked = scene.reports()
    assert opened["success"] is True and picked["success"] is False
    (result,) = scene.results("pick_and_place")
    assert result["ok"] is True and result["picked"] == "red object"
    assert code == launcher.EXIT_FAILED
    assert trace.owners == ["nobody", "agent", "nobody", "agent", "nobody"]
    # Every joint target was a setpoint of an agent MOVE under a PERMIT.
    assert len(trace.targets) == sum(trace.moves.values()) > 500
    trace.check_every_target_was_permitted()
    trace.check_permits_follow_the_check_ahead()

    # The console printed each owner change as the robot app's EVENT, and its
    # status lines showed the agent as owner with a move under a permit.
    granted = desk.holding("[owner] new task: the agent owns the arm, owner number ")
    assert len(granted) == 2
    desk.wait("HELP NEEDED: task ended without success: ")
    desk.wait(" | owner nobody #4 | ")  # the status line after the last change
    assert "agent" in owner_lines(desk) and owner_lines(desk)[-1] == "nobody"
    moving = [
        line for line in desk.holding(" | owner agent #") if " sending | " in line
    ]
    assert moving and all(" | permit " in line for line in moving)
    # While the tab acknowledged a picture every 1.5 s, the view age on the
    # console stayed under the 4 s after which the view is no longer live. A
    # line printed after the tab ended can show an older acknowledgement.
    ages = [
        float(found.group(1))
        for at, line in zip(desk.at, desk.said)
        if at <= tab_ended and (found := VIEW_AGE.search(line))
    ]
    assert ages and all(0.0 <= age < 4.0 for age in ages)
    assert desk.count("ROBOT CONNECTION LOST") == 0

    # The view's page showed the move while it ran, with the agent as owner.
    shown = [status["state"] for status in tab.statuses if status["state"]]
    during = [state for state in shown if state["move_id"]]
    assert during and {state["move_id"] for state in during} <= set(trace.moves)
    assert all(state["owner"] == "agent" for state in during)
    assert all(state["permit"] is not None for state in during)
    assert tab.acknowledged >= 10
    assert page["robot"]["driver"] == "apai-mock" and page["robot"]["connected"]
    # The agent program's own client may still be counted for a moment.
    clients = {**page["robot"]["clients"], "agent": 0}
    assert clients == {"agent": 0, "console": 1, "tool": 1, "viewer": 1}
    assert page["state"]["owner"] == "nobody" and page["state"]["move_id"] == ""

    # The view only looked: no STOP, and the arm holds with its motors on.
    assert scene.h.app.snapshot().stop_number == 0 and scene.h.heard("stop") == []
    scene.holds()
    print(
        f"measured: run {took:.1f} s, {len(trace.targets)} joint targets in "
        f"{len(trace.moves)} moves, {len(trace.agent_permits)} PERMITs; console "
        f"{len(desk.said)} lines, view age on the console {min(ages):.1f} to "
        f"{max(ages):.1f} s; {tab.acknowledged} pictures acknowledged, "
        f"{len(during)} of {len(tab.statuses)} page reads during a move"
    )


@apai
@link
def test_a_console_stop_during_the_agents_move_ends_the_task_and_the_next_task_runs(
    demo: tuple[Scene, Desks],
) -> None:
    scene, desks = demo
    trace = scene.trace
    desk = desks.console()
    view = desks.view("--source", "none")
    typed_at: list[float] = []

    def press(number: int) -> None:
        if number == 40:  # the first move is under way
            typed_at.append(time.monotonic())
            desk.type("stop")

    scene.h.spy.after_target.append(press)
    scene.typed = [PICK, QUICK]
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive")

    # The console's STOP ended the first move where it stood.
    taken = desk.wait("stop: taken.")
    assert "stop number 1, stop level stop" in taken
    (stop,) = scene.h.heard("stop")
    assert stop[1].startswith(f"STOP from console: {console_cmd.STOP_REASON}")
    first = next(iter(trace.moves))
    ended = trace.results[first]
    assert ended.status == "cancelled"
    assert ended.reason == f"STOP from console: {console_cmd.STOP_REASON}"
    sent = ended.detail["sent"]
    assert 40 <= sent <= 60 < trace.moves[first]
    trace.check_every_target_was_permitted()
    # The console showed the STOP as the robot app's EVENT, and the stop
    # number stays in its status lines.
    desk.wait("[stop] STOP from console: ")
    assert " | owner nobody #4 | " in desk.wait(" | stop none #1 | ")
    # Nobody owned the arm after it; the next task took it without a restart.
    stopped, opened = scene.reports()
    assert stopped["success"] is False and opened["success"] is True
    assert not scene.said(task_gate.RESTART)
    assert scene.order == [
        "agent:completed",
        "help_needed:refused",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
    ]
    assert len(desk.holding("[owner] new task: the agent owns the arm")) == 2
    assert code == launcher.EXIT_FAILED
    assert scene.h.app.snapshot().stop_number == 1
    scene.holds()
    # A view without pictures is never live: no view age anywhere.
    assert all(" | view none | " in line for line in desk.holding(" | owner "))
    browser = Browser(view.program.port)
    page = browser.status()
    browser.close()
    assert page["picture"]["overlay"] == NO_PICTURE and page["view_status"]
    assert page["view_status"]["live"] is False
    assert page["state"]["stop_number"] == 1 and page["state"]["view_age_s"] is None
    print(
        f"measured: stop typed at joint target 40, the move ended after {sent}; "
        f"typed to 'stop: taken' {(desk.when('stop: taken.') - typed_at[0]) * 1e3:.0f} ms"
    )


@apai
@link
def test_a_console_rest_after_the_task_returns_the_arm(
    demo: tuple[Scene, Desks],
) -> None:
    scene, desks = demo
    h = scene.h
    desk = desks.console()
    assert scene.run("--task", QUICK) == launcher.EXIT_OK
    desk.sees(lambda state: state.owner == "nobody" and state.motion == "hold")
    sent = len(h.spy.targets)
    # "yes" is not the confirmation of a return to rest.
    desk.type("rest")
    desk.wait(ASK_REST)
    desk.type("yes")
    desk.wait("rest not confirmed; nothing sent")
    desk.type("rest")
    desk.wait(ASK_REST, 2)
    typed = time.monotonic()
    desk.type(PATH_IS_CLEAR)
    done = desk.wait_long("rest: RESULT completed", REST_WAIT_S)
    took = desk.when("rest: RESULT completed") - typed
    assert "rest: RESULT completed" in done
    h.idle()
    # The arm is at rest, holding, with its motors on; recovery moved it.
    assert h.app.motion_state is MotionState.REST_HOLD
    assert len(h.spy.targets) > sent
    assert h.spy.released_at == [] and h.arm.torque_on()
    desk.sees(lambda state: state.motion == "rest_hold" and state.owner == "nobody")
    assert desk.holding(" | owner recovery #")
    print(
        f"measured: rest from the console took {took:.1f} s, "
        f"{len(h.spy.targets) - sent} joint targets"
    )


@apai
@link
def test_an_obstacle_costs_no_restart(demo: tuple[Scene, Desks]) -> None:
    scene, desks = demo
    trace = scene.trace
    desk = desks.console()
    ObstacleGrid.made.clear()
    scene.grid = ObstacleGrid
    grids: dict[str, np.ndarray] = {}

    def on_attach(made: AgentLink) -> None:
        send_move = made.send_move

        def first_move(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            if not grids:
                # A box where the gripper tip is at the end of the first move.
                kinematics = trace.built["harness"].kin
                tip = kinematics.fk(np.asarray(q[-1], dtype=float))[:3, 3]
                grids["clear"] = ObstacleGrid.made[-1].grid
                grids["box"] = ObstacleGrid.made[-1].with_box(tip, 0.04)
            send_move(move_id, start, tol, q, *rest)

        made.send_move = first_move

    def switch_in(number: int) -> None:
        if number == 5:  # the arm is under way; the grid's next query shows it
            ObstacleGrid.made[-1].grid = grids["box"]

    def on_ask() -> None:
        if len(scene.asked) == 2:
            # A person took the obstacle away before typing yes again.
            ObstacleGrid.made[-1].grid = grids["clear"]

    scene.on_attach = on_attach
    scene.on_ask = on_ask
    scene.h.spy.after_target.append(switch_in)
    scene.typed = [PICK, PICK]
    scene.answers = ["yes", "yes"]
    began = time.monotonic()
    code = scene.run("--interactive")
    took = time.monotonic() - began
    assert took < RUN_BUDGET_S

    # The first task: the check ahead refused a setpoint and the arm held.
    first, second = scene.results("pick_and_place")
    assert first["ok"] is False and first["motion_blocked"] is True
    first_move = next(iter(trace.moves))
    assert trace.results[first_move].status == "refused"
    # Seen from the console: a help request with its reason, no STOP, and the
    # agent as owner again for the second task of the same program.
    desk.wait("HELP NEEDED: blocked: arm stop latched")
    assert len(desk.holding("[owner] new task: the agent owns the arm")) == 2
    help_at = desk.when("HELP NEEDED: blocked: arm stop latched")
    again_at = [
        at
        for at, line in zip(desk.at, desk.said)
        if "[owner] new task: the agent owns the arm" in line
    ][1]
    assert again_at > help_at
    assert desk.holding("[stop] ") == [] and desk.holding("[fault] ") == []
    assert all(" | stop none #0 | " in line for line in desk.holding(" | owner "))
    # No restart: one run of the program, and the second task moved the arm.
    assert not scene.said(task_gate.RESTART)
    assert scene.said("the skill stop of the last task is cleared")
    assert second["ok"] is True and second["picked"] == "red object"
    later = [move_id for move_id in trace.moves if move_id != first_move]
    assert len(later) >= 5
    assert all(trace.results[move_id].status == "completed" for move_id in later)
    trace.check_every_target_was_permitted()
    assert code == launcher.EXIT_FAILED  # neither place could be confirmed
    scene.holds()
    print(
        f"measured: run {took:.1f} s; {len(trace.targets)} joint targets in "
        f"{len(trace.moves)} moves, the first refused after "
        f"{trace.results[first_move].detail['sent']} setpoints"
    )


# The robot app on the simulated arm: no agent program is needed


@link
def test_the_agent_program_gone_shows_the_help_request_and_the_countdown_on_the_console(
    sim_demo: tuple[Stack, Desks],
) -> None:
    stack, desks = sim_demo
    h = stack.h
    agent = stack.agent()
    desk = desks.console()
    h.new_task()
    moved = agent.ws.run_move(PATH, WAIT_S, start_q=REST)
    assert moved.status == "completed", moved.reason
    sent = len(h.sent())
    agent.kill()  # the agent program's socket closes while it owns the arm

    # The help request, in the service's words, and what the operator can do.
    desk.wait("HELP NEEDED: agent program gone")
    block = desk.said.index("HELP NEEDED: agent program gone")
    assert desk.said[block + 1] == "  jaws: empty"
    left = desk.said[block + 2]
    assert left.startswith("  time left: ") and left.endswith(
        " s; then the arm moves to rest by itself"
    )
    assert "    stop   a person's STOP ends the wait" in desk.said[block + 4]
    # The assistance wait counts down on the console.
    desk.sees(lambda state: state.assistance_wait_s is not None)
    first = desk.console._state.assistance_wait_s
    assert 110.0 < first <= 120.0
    desk.wait(" | wait ")
    desk.sees(
        lambda state: (
            state.assistance_wait_s is not None
            and state.assistance_wait_s <= first - 1.0
        )
    )
    # A typed stop ends the wait; nothing moved, the motors are on.
    desk.type("stop")
    desk.wait("stop: taken.")
    desk.sees(lambda state: state.assistance_wait_s is None)
    state = desk.console._state
    assert (state.owner, state.stop_number, state.stop_level) == ("nobody", 1, "stop")
    h.idle()
    assert len(h.sent()) == sent
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.heard("rest") == []


@link
def test_the_console_and_the_view_survive_a_restarted_robot_app(
    sim_demo: tuple[Stack, Desks], make_realtime_robot_app: Callable[..., Any]
) -> None:
    stack, desks = sim_demo
    first = stack.h
    desk = desks.console()
    view = desks.view("--source", "none")
    other = stack.client("console")
    stack.connected(other)
    old = other.ws.state
    numbers = {
        "session_id": old.session_id,
        "owner_number": old.owner_number,
        "stop_number": old.stop_number,
    }
    desk.wait(f"session {first.app.session_id}, driver sim")
    # The console announces a loss only after a tick of its own saw the
    # robot connection up.
    desk.wait("robot connection up | clients ")

    stack.dialer.close()
    first.app.close()
    desk.wait("ROBOT CONNECTION LOST: ")
    assert first.arm.torque_on() and first.arm.released_at == []

    # A new robot app dials the same service: a new session id.
    second = make_realtime_robot_app(connected=False)
    assert second.app.session_id != old.session_id
    stack.dial(second)
    stack.connected(other)
    desk.wait(f"session {second.app.session_id}, driver sim")
    assert desk.count("WELCOME: ") == 2
    desk.sees(lambda state: state.session_id == second.app.session_id)
    # The view shows the new session too.
    browser = Browser(view.program.port)
    try:
        page = browser.status()
        for _ in range(20):
            if page["robot"]["session_id"] == second.app.session_id:
                break
            time.sleep(0.25)
            page = browser.status()
    finally:
        browser.close()
    assert page["robot"]["session_id"] == second.app.session_id
    assert page["robot"]["connected"] is True and page["robot"]["driver"] == "sim"
    # A command with the numbers of the old session is refused.
    refused = other.ws.wait_result(other.ws.rest(**numbers), timeout_s=WAIT_S)
    assert refused.status == "refused" and refused.reason.startswith("session id ")
    desk.wait("session id ")  # the console shows the other client's refusal
    # The console's own commands quote the new session: its stop is taken.
    desk.type("stop")
    desk.wait("stop: taken.")
    assert second.app.snapshot().stop_number == 1
    second.idle()
    assert second.sent() == [] and second.arm.torque_on()


# The programs as processes of their own


def robot_app_arguments(driver: str, folder: Path) -> list[str]:
    """The robot app's arguments for the simulated arm or APAI's mock arm."""
    if driver == "sim":
        return ["--config", ROBOT_YAML, "--arm", "sim"]
    # The shipped file with the human limit set at the speed cap of APAI's
    # rebot_rs model, which the mock arm drives.
    text = Path(ROBOT_YAML).read_text(encoding="utf-8")
    fitted = folder / "robot_app_apai.yaml"
    fitted.write_text(
        text.replace("max_joint_speed: 1.5", "max_joint_speed: 0.8"), encoding="utf-8"
    )
    return ["--config", str(fitted), "--arm", "apai", "--driver", "mock"]


@link
@pytest.mark.parametrize("driver", ["sim", pytest.param("apai-mock", marks=apai)])
def test_the_programs_of_demo_1_start_as_processes_and_end_cleanly(
    tmp_path_factory: pytest.TempPathFactory, driver: str
) -> None:
    folder = tmp_path_factory.mktemp("d1")
    path = str(folder / "s.sock")
    programs: list[Program | ConsoleProcess] = []
    try:
        service = ServiceProcess(folder, path)
        programs.append(service)
        port = service.wait_ready()
        app = Program(
            ROBOT_MODULE, *robot_app_arguments(driver, folder), "--connect",
            f"{HOST}:{port}",
        )  # fmt: skip
        programs.append(app)
        app.wait_line(f"dialing the robot link service at {HOST}:{port}")
        view = Program(
            VIEW_MODULE, "--config", str(write_config(folder, path)), "--source", "none"
        )
        programs.append(view)
        line = view.wait_line("live view: open http://127.0.0.1:")
        view_port = int(line.split("http://127.0.0.1:")[1].split("/")[0])

        # The console program, at a terminal, names the driver the robot app
        # declares.
        desk = ConsoleProcess(path)
        programs.append(desk)
        desk.wait("attached to the robot link service: ")
        assert f", driver {driver}, " in desk.wait("WELCOME: ")
        desk.wait(f" | clients 1 console, 1 viewer | driver {driver} | owner nobody #")

        # The console command, twice: it reads the state, and its STOP is taken.
        state = Program(CONSOLE_CMD_MODULE, "--socket", path, "state")
        programs.append(state)
        assert state.wait_exit() == 0
        assert any(
            line.startswith("ROBOT_STATE: control owner nobody") for line in state.lines
        )
        stop = Program(CONSOLE_CMD_MODULE, "--socket", path, "stop")
        programs.append(stop)
        assert stop.wait_exit() == 0
        assert any(line.startswith("stop: taken. ") for line in stop.lines)
        app.wait_line("[stop] STOP from console: ")
        desk.wait("[stop] STOP from console: ")
        # A rest typed at the console, with the arm at rest already.
        desk.type("rest")
        desk.answer(ASK_REST, PATH_IS_CLEAR)
        desk.wait("rest: RESULT completed", REST_WAIT_S)

        # The page of the view process shows the same robot.
        browser = Browser(view_port)
        try:
            page = browser.status()
            for _ in range(20):
                if page["state"] and page["state"]["stop_number"] == 1:
                    break
                time.sleep(0.25)
                page = browser.status()
        finally:
            browser.close()
        assert page["state"]["stop_number"] == 1 and page["state"]["torque_on"]
        assert page["robot"]["driver"] == driver and page["robot"]["connected"]
        assert page["picture"]["overlay"] == NO_PICTURE

        # Each program ends by its own way out.
        app.type("quit")  # at rest: nothing is asked
        assert app.wait_exit() == 0
        assert "quit: completed; the robot app closes" in app.lines
        desk.wait("ROBOT CONNECTION LOST: ")
        view.signal(signal.SIGTERM)
        assert view.wait_exit() == 0
        desk.type("quit")
        assert desk.wait_exit() == 0
        assert desk.said[-1] == QUIT_NOTE
        service.signal(signal.SIGTERM)
        assert service.wait_exit() == 0
    finally:
        for program in programs:
            program.close()
