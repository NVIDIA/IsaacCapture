# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo 1 by hand on one machine with the mock agent: its two files and its recipe.

The recipe starts five programs on one machine: the robot link service, the
robot app on APAI's software arm (the mock arm), the operator console, the live
view and the agent program on APAI's mock scene. The robot app runs on
configs/robot_app_mock.yaml and the agent program on configs/agent_mock.yaml,
whose obstacle grid is the clear grid: a stand-in for the grid bridge (the
program that keeps the obstacle grid) which reports nothing near.

The last test runs the recipe's commands as processes of their own, the console
and the agent program each at a pseudo-terminal, and types what the recipe
types.
"""

from __future__ import annotations

import dataclasses
import fcntl
import importlib
import json
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

import pytest
import test_launcher
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from isaaccapture_examples.agentic_remote.link_service.local import connect_local
from isaaccapture_examples.agentic_remote.protocol.confirm import (
    PATH_IS_CLEAR,
    YES,
    agent_environment,
)
from isaaccapture_examples.agentic_remote.robot_app import config as robot_config
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient
from isaaccapture_examples.agentic_remote.view.model import NO_PICTURE
from link_stack import HOST, READY, SERVICE_MODULE, Program, program_env, until
from test_agent_end_to_end import RUN_BUDGET_S, Trace
from test_console import ASK_REST, ASK_YES
from test_demo1_programs import VIEW_MODULE, ConsoleProcess
from test_launcher import PICK, QUICK, Stage, scene_robot_config, settings_file
from test_link_programs import ROBOT_MODULE
from test_view_program import Browser

from conftest import apai_runtime_env

apai = pytest.mark.apai
link = pytest.mark.link
# The fixture of test_launcher.py, for the tests of this file.
make_stage = test_launcher.make_stage

EXAMPLE = Path(agent_config.__file__).resolve().parents[4]
CONFIGS = EXAMPLE / "configs"
ROBOT_MOCK = CONFIGS / "robot_app_mock.yaml"
AGENT_MOCK = CONFIGS / "agent_mock.yaml"
# Every key agent_mock.yaml gives; all others take their default.
AGENT_MOCK_KEYS: dict[tuple[str, str], Any] = {
    ("link", "expected_profile_id"): "apai-rs-model",
    ("link", "calibration_id"): "apai-mock-calibration",
    ("link", "connect_timeout_s"): 5.0,
    ("thor", "agent_max_joint_speed"): 1.2,
    ("task", "max_joint_speed"): 1.2,
    ("task", "grid_wait_s"): 10.0,
    ("camera", "allow_invented_times"): True,
    ("experiment", "stack"): "minimal",
    ("experiment", "grid"): "clear",
}
INVENTED_WARNING = "WARNING: the camera frames of --camera mock carry invented"
CLEAR_WARNING = (
    "WARNING: experiment.grid is clear: the obstacle grid is a stand-in with "
    "nothing near; no move is checked against obstacles. For a simulated or mock "
    "arm only: the program ends unless the robot app declares a simulated driver"
)
# Real seconds a test gives a program before it fails.
WAIT_S = 10.0
# Real seconds the agent program may take to start: APAI's runtime is built.
START_WAIT_S = 60.0
# Real seconds a return to rest from the mock scene's home pose may take.
REST_WAIT_S = 60.0
# A status line of the console that shows a held PERMIT.
PERMIT_SHOWN = re.compile(r" \| permit \d+, evidence \d+\.\d\d s")
# What APAI's task report says of the mock scene's place in a run that starts
# without stored beliefs of APAI, as the first run on a state folder does.
UNCONFIRMED = (
    "pick_and_place#0 unverified: red object matches the drop point the skill "
    "itself reported, but only in the belief channel it also wrote -- no "
    "independent confirmation"
)
# The part of a help request's reason that comes before the run folder's name;
# the reason is cut at 256 characters.
NOT_CONFIRMED = (
    "task ended without success: mock LLM wiring check: observed the scene and "
    "stopped [verification] could not confirm: "
)
# How the console's last line begins after a typed quit.
CONSOLE_ENDS = "the operator console ends; nothing was sent for that"


def leaves(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    """Each value of nested mappings and lists, by its path of keys and indexes."""
    if isinstance(value, dict):
        items: Any = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return {path: value}
    found: dict[tuple[Any, ...], Any] = {}
    for key, item in items:
        found.update(leaves(item, (*path, key)))
    return found


def same(ours: Any, theirs: Any) -> bool:
    """Equal; two numbers as pytest.approx compares them."""

    def number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if number(ours) and number(theirs):
        return ours == pytest.approx(theirs)
    return type(ours) is type(theirs) and ours == theirs


def fitted(
    shipped: Path, folder: Path, changes: dict[str, str], more: str = ""
) -> Path:
    """A copy of a shipped file in `folder`: each text replaced once, `more` added."""
    text = shipped.read_text(encoding="utf-8")
    for old, new in changes.items():
        assert text.count(old) == 1, f"{shipped.name} holds {old!r} not once"
        text = text.replace(old, new)
    copy = folder / shipped.name
    copy.write_text(text + more, encoding="utf-8")
    return copy


def check_nothing_confirmed(report: dict[str, Any]) -> None:
    """This task report's pick-and-place ran to its end and nothing confirmed it.

    The mock scene confirms neither the grasp nor the place, so the task ends
    without success.
    """
    (carried,) = (
        row["result"] for row in report["tool_log"] if row["tool"] == "pick_and_place"
    )
    assert carried["ok"] is True and carried["grip_verified"] is False
    assert report["success"] is False
    (why,) = report["unverified"]
    assert why.endswith(UNCONFIRMED)


def off_rest(state: Any, rest: list[float]) -> float:
    """How far the measured pose of a ROBOT_STATE lies from `rest`, in rad.

    0.0 for a ROBOT_STATE without a valid reading of every joint.
    """
    joints = sorted(state.joints, key=lambda sample: sample.joint)
    if len(joints) != len(rest) or not all(sample.valid for sample in joints):
        return 0.0
    return max(abs(sample.value - was) for sample, was in zip(joints, rest))


# The two files


def test_the_two_mock_files_load_and_agree_with_each_other() -> None:
    robot = robot_config.load_config(ROBOT_MOCK)
    settings = agent_config.load_settings(AGENT_MOCK)
    # What the agent program compares with the robot app's WELCOME.
    assert settings.link.expected_profile_id == robot.profile_id
    assert settings.link.calibration_id == robot.camera_calibration_id
    # The agent file's copies of the Thor's settings.
    agent = robot.limits.agent
    assert settings.thor.agent_max_joint_speed == agent.max_joint_speed == 1.2
    assert settings.task.max_joint_speed <= agent.max_joint_speed
    assert settings.thor.first_wait_s == robot.permits.first_wait_s
    assert settings.thor.settle_timeout_s == robot.executor.settle_timeout_s
    # No setpoint of the agent goes to the motors without a PERMIT, and the
    # clear grid stands in only for the minimal stack.
    assert robot.permits.required_for_agent is True
    assert settings.minimal and settings.experiment.grid == "clear"
    # The robot app serves demo 2 as well: a person may take the arm over,
    # and may leave it only where a typed rest is taken.
    assert robot.human.enabled is True and robot.human.require_live_view is False
    human, rest = robot.limits.human, robot.rest
    assert all(low >= edge for low, edge in zip(human.joint_min, rest.region_min))
    assert all(high <= edge for high, edge in zip(human.joint_max, rest.region_max))
    # Neither file names a place of its own: the state folder is the example's.
    assert settings.state_dir == EXAMPLE / ".state"
    assert settings.link.socket_path == ""
    # The agent file says what a grid bridge and the clear grid are where it
    # first names each.
    comments = AGENT_MOCK.read_text(encoding="utf-8").replace("#", " ")
    words = " ".join(comments.split())
    for term, defined in {
        "grid bridge": "grid bridge (the program that keeps the obstacle grid)",
        "clear grid": "clear grid: a stand-in for the grid bridge that reports",
    }.items():
        assert words[words.index(term) :].startswith(defined), term


@apai
def test_the_mock_files_load_and_the_robot_file_is_the_mock_scenes(
    tmp_path: Path,
) -> None:
    loaded = robot_config.load_config(ROBOT_MOCK)
    scene = scene_robot_config()
    ours = leaves(dataclasses.asdict(loaded))
    theirs = leaves(dataclasses.asdict(scene))
    assert set(ours) == set(theirs)
    # (the mock scene's value, the file's) of every key that differs.
    differs = {
        path: (theirs[path], ours[path])
        for path in ours
        if not same(ours[path], theirs[path])
    }
    assert differs == {
        ("human", "enabled"): (False, True),
        ("human", "require_live_view"): (True, False),
        ("executor", "max_setpoints"): (3000, 3300),
    }
    # A loaded file must hold the longest return to rest; the mock scene's own
    # 3000 setpoints do not.
    assert robot_config.longest_return_setpoints(loaded) == 3271
    short = fitted(ROBOT_MOCK, tmp_path, {"max_setpoints: 3300": "max_setpoints: 3000"})
    with pytest.raises(robot_config.ConfigError) as raised:
        robot_config.load_config(short)
    assert str(raised.value) == (
        f"{short}: executor.max_setpoints must be at least 3271, got 3000: a "
        "return to rest from the far end of rest.region_min..region_max (largest "
        "joint travel 3.14 rad at rest.recovery_speed 0.1 rad/s) would be refused "
        "for its length"
    )
    with pytest.raises(robot_config.ConfigError, match="at least 3271, got 3000"):
        robot_config._check_return_fits(scene)

    # agent_mock.yaml: the settings the mock scene's tests run on, with the
    # clear grid in place of their stand-in for the grid bridge.
    settings = agent_config.load_settings(AGENT_MOCK)
    written = yaml.safe_load(AGENT_MOCK.read_text(encoding="utf-8"))
    assert leaves(written) == AGENT_MOCK_KEYS
    for (block, key), value in AGENT_MOCK_KEYS.items():
        assert getattr(getattr(settings, block), key) == value, (block, key)
    tested = agent_config.load_settings(settings_file(tmp_path))
    ours, theirs = leaves(settings.as_dict()), leaves(tested.as_dict())
    assert {path for path in ours if ours[path] != theirs[path]} == {
        ("path",),
        ("state_dir",),
        ("experiment", "grid"),
    }
    # It runs against a simulated driver and no other.
    invented = agent_config.invented_times(settings, "mock")
    assert invented is True
    for driver in ("apai-mock", "sim"):
        agent_config.check_robot_driver(settings, driver, invented)
    for driver in ("real", ""):
        with pytest.raises(
            agent_config.ConfigError,
            match=r"^experiment\.grid is clear, but the robot app declares the driver",
        ):
            agent_config.check_robot_driver(settings, driver, invented)


# The agent program in this process, on the two files


@apai
def test_a_mock_task_runs_in_process_on_the_shipped_files(
    make_stage: Callable[..., Stage],
) -> None:
    from isaaccapture_examples.agentic_remote.agent.clear_grid import ClearGrid

    occupancy = importlib.import_module("apai.perception.occupancy")
    theirs = occupancy.OccupancyClient
    stage = make_stage(config=robot_config.load_config(ROBOT_MOCK))
    assert stage.h.config.human.enabled is True
    trace = Trace(stage.h)
    # The shipped file with one key more: the state folder, which the file
    # leaves at the example's own .state.
    file = fitted(AGENT_MOCK, stage.folder, {}, f"\nstate_dir: {stage.folder}/state\n")

    def attach(socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
        made = stage.attach(socket_path, timeout_s=timeout_s)
        trace.watch(made)
        return made

    stage.typed = [QUICK, PICK]
    stage.answers = ["yes", "yes"]
    arguments = ["--config", str(file), "--camera", "mock", "--interactive"]
    began = time.monotonic()
    # No stand-in of the test: the launcher itself puts the clear grid in.
    with trace.hooked():
        code = launcher.main(
            [*arguments, "--run-dir", str(stage.run_dir)],
            attach=attach,
            ask=stage.ask,
            say=stage.lines.append,
            task_input=stage.next_line,
        )
    assert time.monotonic() - began < RUN_BUDGET_S

    # Both warnings, before anything is attached; the driver is the mock arm's.
    invented, clear = stage.said("WARNING: ")
    assert invented.startswith(INVENTED_WARNING) and clear == CLEAR_WARNING
    connected = stage.said("robot connection up")[0]
    assert "driver apai-mock, session" in connected
    lines = stage.lines
    assert lines.index(invented) < lines.index(clear) < lines.index(connected)
    # APAI's obstacle map ran on the clear grid, required and fed by the mock
    # camera's frames; APAI's client is its own again.
    grid = trace.built["grid"]
    assert isinstance(grid._client, ClearGrid) and grid.status["backend"] == "clear"
    assert grid.required is True and grid.refreshes >= 1
    assert stage.said("obstacle grid: live")
    assert occupancy.OccupancyClient is theirs

    # The first task opened the jaws and succeeded.
    opened, picked = stage.reports()
    assert opened["success"] is True and stage.results("open_gripper")[0]["ok"]
    assert stage.said("task completed: the arm holds where it is")
    # The second sent MOVEs, and every joint target the mock arm took was a
    # setpoint of an agent MOVE under a PERMIT that rested on a camera frame.
    assert stage.motion.count("send_move") == len(trace.moves) >= 5
    assert len(trace.targets) == sum(trace.moves.values()) > 500
    assert stage.h.config.permits.required_for_agent is True
    trace.check_every_target_was_permitted()
    trace.check_permits_follow_the_check_ahead()
    assert trace.agent_permits and trace.thor_permits
    # The mock scene confirms neither the grasp nor the place: a help request,
    # exit code 1.
    check_nothing_confirmed(picked)
    assert stage.order == [
        "agent:completed",
        "nobody:completed",
        "agent:completed",
        "help_needed:completed",
        "close:nobody",
    ]
    (asked,) = stage.said("HELP NEEDED: ")
    assert asked.startswith(f"HELP NEEDED: {NOT_CONFIRMED}")
    assert code == launcher.EXIT_FAILED
    # The arm holds with its motors on; nothing was written beside the state.
    assert stage.h.spy.released_at == [] and stage.h.arm.torque_on()
    assert (stage.folder / "state" / ".gitignore").read_text() == "*\n"


# The recipe as processes


def _take_terminal() -> None:
    """In the new process, before its program starts: the terminal becomes its own.

    The process leads a new session; its standard input, a pseudo-terminal,
    becomes the session's controlling terminal, which /dev/tty then names.
    """
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class AgentTerminal:
    """The agent program, started with run_agent.sh, at a terminal of its own.

    bash run_agent.sh --config PATH --camera mock --interactive, in the
    example's folder and outside an agent's environment. The pseudo-terminal
    is the program's controlling terminal: it reads each typed yes from
    /dev/tty, as at a person's terminal. The test types at it and reads what
    the program wrote.
    """

    def __init__(self, config: Path) -> None:
        self.text = ""  # everything the program wrote so far
        self._at = 0  # where in .text the next wait() begins
        self._heard = threading.Condition()
        self._master, terminal = pty.openpty()
        attributes = termios.tcgetattr(terminal)
        attributes[3] &= ~termios.ECHO  # what the test types is not shown back
        termios.tcsetattr(terminal, termios.TCSANOW, attributes)
        env = {
            name: value
            for name, value in program_env().items()
            if agent_environment({name: value}) is None
        }
        env["AGENTIC_REMOTE_AGENT_PYTHON"] = sys.executable
        env["APAI_REQUIRE_CUDA"] = "0"
        command = ["bash", str(EXAMPLE / "run_agent.sh"), "--config", str(config)]
        try:
            self.process = subprocess.Popen(
                [*command, "--camera", "mock", "--interactive"],
                stdin=terminal,
                stdout=terminal,
                stderr=terminal,
                env=env,
                cwd=EXAMPLE,
                start_new_session=True,
                # Runs in the new process before bash starts: one ioctl.
                preexec_fn=_take_terminal,  # noqa: PLW1509
            )
        finally:
            os.close(terminal)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        while True:
            try:
                data = os.read(self._master, 4096)
            except OSError:  # the program closed the terminal
                data = b""
            if not data:
                return
            with self._heard:
                self.text += data.decode(errors="replace").replace("\r", "")
                self._heard.notify_all()

    def wait(self, part: str, wait_s: float = WAIT_S) -> None:
        """Wait until the program wrote `part` after what the last wait() found."""
        with self._heard:
            assert self._heard.wait_for(
                lambda: part in self.text[self._at :], wait_s
            ), f"no {part!r} within {wait_s:g} s; written: {self.text[self._at :]!r}"
            self._at = self.text.index(part, self._at) + len(part)

    def type(self, line: str) -> None:
        os.write(self._master, line.encode() + b"\n")

    def answer(self, question: str, line: str, wait_s: float = WAIT_S) -> None:
        """Type `line` half a second after the question is shown.

        The program drops what was typed before it shows the question.
        """
        self.wait(question, wait_s)
        time.sleep(0.5)
        self.type(line)

    def end_input(self) -> None:
        """Ctrl+D at an empty line: the end of the typed input."""
        os.write(self._master, b"\x04")

    def wait_exit(self, wait_s: float = WAIT_S) -> int:
        """Wait for the program to end; its exit code. All it wrote is in .text."""
        code = self.process.wait(wait_s)
        self._reader.join(WAIT_S)
        return code

    def close(self) -> None:
        """End the program if it still runs, and the terminal."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGKILL)
            self.process.wait(WAIT_S)
        if self._master >= 0:
            os.close(self._master)  # the reader ends at the closed terminal
            self._master = -1
        self._reader.join(WAIT_S)


class Watcher:
    """A local client of the test in the role "tool": it reads ROBOT_STATE only."""

    def __init__(self, path: str) -> None:
        self.end = connect_local(path, "tool", "recipe-test", WAIT_S)
        self.ws = WorkstationClient(self.end, SystemClock(), name="tool")

    def pose(self) -> list[float]:
        return self.ws.measured_pose()

    def close(self) -> None:
        self.ws.close()
        self.end.close()


def shown(desk: ConsoleProcess, *parts: str, wait_s: float = WAIT_S) -> str:
    """Wait for a line of the console that holds every part; returns the first."""

    def found() -> list[str]:
        return [line for line in list(desk.said) if all(p in line for p in parts)]

    until(lambda: bool(found()), f"a console line with {parts}", wait_s)
    return found()[0]


@pytest.fixture
def recipe_folder(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path]:
    """A short folder for the socket and the files; APAI's stores in tmp_path.

    While it is open the programs started see no setting of a developer's
    shell for APAI, and APAI's grasp server port is one nobody listens on.
    """
    with apai_runtime_env(tmp_path):
        yield tmp_path_factory.mktemp("d1")  # a socket path holds 107 bytes


@link
@apai
def test_the_recipe_of_demo_1_runs_as_processes_with_the_agent_at_a_terminal(
    recipe_folder: Path,
) -> None:
    """Steps 1 to 9 of the recipe. What differs from the recipe by hand:

    the service, the view and the agent program run on copies of their files
    that name a port the system picks, a socket and a state folder of the test;
    the console and the view are started with python -m, not by their scripts;
    a local client in the role "tool" reads ROBOT_STATE; no browser is used.
    """
    folder = recipe_folder
    path = str(folder / "s.sock")
    state = f"state_dir: {folder}/state\n"
    # The shipped files; a port the system picks, the socket and the state
    # folder of the test.
    service_file = fitted(
        CONFIGS / "link_service.yaml",
        folder,
        {"port: 47610": "port: 0", '\nsocket_path: ""': f"\nsocket_path: {path}"},
    )
    view_file = fitted(
        CONFIGS / "view.yaml",
        folder,
        {
            "port: 47620": "port: 0",
            '\nsocket_path: ""': f"\nsocket_path: {path}",
            "\nstate_dir: ../.state\n": f"\n{state}",
        },
    )
    agent_file = fitted(
        AGENT_MOCK, folder, {"\nlink:\n": f"\nlink:\n  socket_path: {path}\n"}, state
    )
    programs: list[Any] = []
    try:
        # 1. The robot link service.
        service = Program(SERVICE_MODULE, "--config", str(service_file))
        programs.append(service)
        endpoint = service.wait_line(READY).removeprefix(READY).split()[0]
        assert endpoint.startswith(f"{HOST}:")
        # 2. The robot app on APAI's mock arm, with the shipped file.
        app = Program(
            ROBOT_MODULE, "--config", str(ROBOT_MOCK), "--arm", "apai",
            "--driver", "mock", "--connect", endpoint,
        )  # fmt: skip
        programs.append(app)
        app.wait_line("robot app: session ")
        # 3. The operator console.
        desk = ConsoleProcess(path)
        programs.append(desk)
        assert ", driver apai-mock, " in desk.wait("WELCOME: ")
        shown(desk, " | driver apai-mock | ", " | owner nobody #0 | ")
        # 4. The live view, without a camera.
        view = Program(VIEW_MODULE, "--config", str(view_file), "--source", "none")
        programs.append(view)
        line = view.wait_line("live view: open http://127.0.0.1:")
        view_port = int(line.split("http://127.0.0.1:")[1].split("/")[0])
        browser = Browser(view_port)
        try:
            page = browser.status()
        finally:
            browser.close()
        assert page["picture"]["overlay"] == NO_PICTURE
        watcher = Watcher(path)
        programs.append(watcher)
        rest = watcher.pose()
        assert rest == pytest.approx([0.0] * 6, abs=0.05)

        # 5. The agent program at its terminal.
        agent = AgentTerminal(agent_file)
        programs.append(agent)
        agent.wait(INVENTED_WARNING, START_WAIT_S)
        agent.wait(CLEAR_WARNING)
        agent.wait("robot connection up: ", START_WAIT_S)
        agent.wait(" driver apai-mock, session ")
        agent.wait("obstacle grid: live", START_WAIT_S)
        agent.wait("ready. Ctrl+C at this terminal is STOP")
        # 6. The first task: the jaws open and the task succeeds.
        agent.wait("task> ")
        agent.type(QUICK)
        agent.answer("Type yes to proceed", YES, START_WAIT_S)
        agent.wait("task completed: the arm holds where it is", START_WAIT_S)
        agent.wait("success: True")  # APAI's task report
        desk.wait("[owner] new task: the agent owns the arm, owner number 1")
        desk.wait("[owner] the agent gave the arm back: task completed")
        shown(desk, " | owner nobody #2 | ")
        # 7. The second task moves the arm under permits and ends in a help
        # request: the mock scene confirms neither the grasp nor the place.
        agent.wait("task> ")
        agent.type(PICK)
        agent.answer("Type yes to proceed", YES, START_WAIT_S)
        # During the task a move of the agent has the arm away from rest.
        during = watcher.ws.wait_state(
            lambda s: (
                s.owner == "agent" and s.motion == "moving" and off_rest(s, rest) > 0.5
            ),
            RUN_BUDGET_S,
        )
        assert (during.owner_number, during.torque_on) == (3, True)
        agent.wait(f"HELP NEEDED: {NOT_CONFIRMED}", RUN_BUDGET_S)
        agent.wait("success: False")  # APAI's task report, which says why in full
        agent.wait(UNCONFIRMED)
        agent.wait("task> ")
        desk.wait("[owner] new task: the agent owns the arm, owner number 3")
        desk.wait(f"HELP NEEDED: {NOT_CONFIRMED}")
        moving = [
            line
            for line in list(desk.said)
            if " | owner agent #3 | " in line and " sending | " in line
        ]
        assert moving and all(PERMIT_SHOWN.search(line) for line in moving)
        # The arm holds where the task left it, with its motors on.
        raised = watcher.pose()
        assert max(abs(at - was) for at, was in zip(raised, rest)) > 0.5
        waiting = watcher.ws.wait_state(lambda s: s.assistance_wait_s is not None)
        assert (waiting.owner, waiting.torque_on) == ("nobody", True)
        shown(desk, " | owner nobody #4 | ", " | wait ")
        # 8. A typed stop ends the wait; a typed rest returns the arm.
        desk.type("stop")
        desk.wait("stop: taken.")
        watcher.ws.wait_state(lambda s: s.assistance_wait_s is None)
        desk.type("rest")
        desk.answer(ASK_REST, PATH_IS_CLEAR)
        desk.wait("rest: RESULT completed", REST_WAIT_S)
        rested = watcher.ws.wait_state(lambda s: s.motion == "rest_hold")
        assert rested.torque_on is True
        assert watcher.pose() == pytest.approx(rest, abs=0.05)

        # 9. The end of every program, each by its own way out.
        agent.end_input()
        assert agent.wait_exit() == launcher.EXIT_FAILED
        assert "1 task(s) of this session ended without success" in agent.text
        # The run folder's task reports say why the second one did.
        (reports,) = (folder / "state" / "runs").glob("*/task-reports.jsonl")
        opened, picked = map(json.loads, reports.read_text().splitlines())
        assert opened["success"] is True
        check_nothing_confirmed(picked)
        desk.type("shutdown")
        desk.answer(f"{ASK_YES}shut down", YES)
        assert app.wait_exit() == 0
        desk.wait("ROBOT CONNECTION LOST: ")
        desk.type("quit")
        assert desk.wait_exit() == 0
        # The closing line; what it says of the robot app is the console's own.
        assert desk.said[-1].startswith(CONSOLE_ENDS)
        watcher.close()
        for program in (view, service):
            program.signal(signal.SIGINT)  # Ctrl+C at its terminal
            assert program.wait_exit() == 0
        print(
            f"measured: pose after the second task {[round(q, 2) for q in raised]}; "
            f"{len(moving)} status lines of the console showed a move under a permit"
        )
    finally:
        for program in programs:
            program.close()
