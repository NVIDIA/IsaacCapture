# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rest, motors off and motors on again, end to end across the robot link.

The workstation side is APAI's own runtime with its rest tool
(apai.skills.rest_pose.move_to_rest), its SafeArm and its checker, on the arm
adapter, behind the task gate as the agent program wires them. The Thor side
is the robot app on APAI's reBot RS driver with the rebot_rs arm profile; the
driver stands on a stand-in motor layer that takes every command at once and
records it. Between them: a loopback carrier, and the real robot link service
on a link of 127.0.0.1.

The rest tool folds the arm onto its rest pose, where two joints lie on their
stops, measures rest, closes the jaws with the driver's close and asks the
robot app for the motors off. The robot app switches them off only at
measured rest. "Standby" is APAI's word for an arm whose motors are off until
its next motion.

The tests marked on_the_rig build the same stack from the rig's robot file
and the rig's ready pose, which lies 0.025 rad above the two stops, with a
motor that ends a few mrad from every command.

They need an APAI checkout (AGENTIC_REMOTE_APAI_SRC).
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import task_gate
from isaaccapture_examples.agentic_remote.agent.arm_adapter import (
    ArmSettings,
    arm_factory,
)
from isaaccapture_examples.agentic_remote.agent.checkahead import (
    CheckAhead,
    fold_margin,
)
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import (
    REQUEST_MOTORS_OFF,
    REQUEST_MOTORS_ON,
    Result,
)
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Refusal,
)
from repo_paths import repo_root
from test_apai_backend import MotorSdk, make_config
from test_arm_adapter import PERIOD_S, EveryFrame

from conftest import AppHarness, FakeGridClient, apai_runtime_env, import_experiment

try:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
except ModuleNotFoundError as missing:
    # Without an APAI checkout conftest.py skips every test here: all are
    # marked "apai". They must still be collected.
    if missing.name != "apai":
        raise
    apai_backend = None

pytestmark = pytest.mark.apai
link = pytest.mark.link
experiment = pytest.mark.experiment

# Real seconds a test gives another thread before it fails, and a whole
# return to rest: the fold alone takes four steps of 2.5 s.
WAIT_S = 10.0
REST_WAIT_S = 60.0
# Joint speed of the agent and of the return to rest in these tests, rad/s.
SPEED = 0.4
REST = [0.0] * 6  # the rest pose: the second and third joint lie on their stops
# A pose raised off the stops, 0.1 rad from the profile's ready pose.
RAISED = [0.1, 1.1, 1.1, 0.0, 0.1, 0.0]
FOLDS = 4  # the fold steps of APAI's rest tool
# Real seconds the arm is left in standby: longer than the workstation's
# obstacle grid takes to refresh, and than the robot app's watch of a hold.
STANDBY_S = 1.0
# The rig's robot file, and the file of the experiment that holds its ready pose.
ROBOT_RIG = repo_root() / "examples/agentic_remote/configs/robot_app_rebot.yaml"
RIG_ARM = "configs/thor_operation.yaml"
# Motors of the rig that end this far from every command, in rad. The first
# three leave a joint nearer the rest pose than its next fold waypoint; the
# last leaves the second joint about 3 mrad below its stop when the last
# fold step starts.
SETTLES = [
    pytest.param({"motor4": -0.003}, id="a-wrist-joint-3-mrad-off"),
    pytest.param({"motor1": +0.006}, id="the-base-joint-6-mrad-off"),
    pytest.param({"motor6": -0.016}, id="the-last-joint-16-mrad-off"),
    pytest.param({"motor2": -0.009}, id="a-stop-joint-below-its-stop"),
]
# A test on the stack of the rig, once per entry of SETTLES.
on_the_rig = pytest.mark.parametrize("settles", SETTLES, indirect=True)


def rest_pose() -> Any:
    """APAI's rest tool: the functional code."""
    return importlib.import_module("apai.skills.rest_pose")


class Motors(MotorSdk):
    """The stand-in motor layer of the Thor, shared by every driver built on it.

    `events` are, in order: ("joints", target) for every joint target, the
    name of every jaw routine of the driver that was called, and
    ("disconnect", measured pose) for every torque cut. `floor` keeps a
    motor from going below a value: a joint that stops short of its stop.
    `settles` is how far a motor ends from every command, in rad.
    """

    def __init__(self, settles: dict[str, float] | None = None) -> None:
        super().__init__()
        self.events: list[Any] = []
        self.floor: dict[str, float] = {}
        self.settles = dict(settles or {})
        send = self.arm.send_mit

        def send_mit(q: Any, *args: Any, **kwargs: Any) -> None:
            self.events.append(("joints", [float(value) for value in q]))
            send(q, *args, **kwargs)
            for name, off in self.settles.items():
                self.arm._mm[name].value += off
            for name, least in self.floor.items():
                motor = self.arm._mm[name]
                motor.value = max(motor.value, least)

        self.arm.send_mit = send_mit

    def pose(self) -> list[float]:
        return [self.arm._mm[name].value for name in self.arm.joint_names]

    def set_pose(self, pose: list[float]) -> None:
        for name, value in zip(self.arm.joint_names, pose):
            self.arm._mm[name].value = float(value)

    def disconnect(self) -> None:
        self.events.append(("disconnect", self.pose()))
        super().disconnect()

    def targets(self) -> list[list[float]]:
        return [event[1] for event in self.events if event[0] == "joints"]

    def cuts(self) -> list[list[float]]:
        return [event[1] for event in self.events if event[0] == "disconnect"]

    def names(self) -> list[str]:
        """The events without the joint targets: jaw routines and torque cuts."""
        return [
            e if isinstance(e, str) else e[0] for e in self.events if e[0] != "joints"
        ]


class Thor:
    """The robot app on APAI's reBot RS driver, as its program builds it.

    It has the functional rest's route and can build the driver anew, so a
    typed rest ends with the motors off and they can be switched on again.
    With rig_home it is the rig's: that ready pose and the rig's robot file.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rig_home: list[float] | None = None,
        settles: dict[str, float] | None = None,
    ) -> None:
        import apai.control.rs_gripper as rs_gripper

        self.rig = rig_home is not None
        ready = {"arm": {"home_q": list(rig_home)}} if self.rig else None
        self.profile = apai_backend.load_apai_profile("rebot_rs", ready)
        self.config = thor_config(self.profile, rig=self.rig)
        self.speed = self.config.limits.agent.max_joint_speed
        self.clock = SystemClock()
        self.motors = Motors(settles)
        self.drivers: list[Any] = []  # APAI's driver objects, one per connect
        monkeypatch.setattr(sys, "path", list(sys.path))  # the driver's SDK path
        # The gripper guard's waits pass without real time.
        monkeypatch.setattr(rs_gripper, "time", self.motors.guard_time)
        checkers = apai_backend.build_apai_checkers(
            self.config, self.profile, self.clock
        )
        self.home = [float(value) for value in self.profile.arm_cfg.home_q]
        self.h = AppHarness(
            self.config,
            self._port(),
            checkers,
            self.clock,
            connected=False,
            step_period_s=None,
            app_options={
                "rest_route": rest_module.functional_route(self.home),
                "rebuild_arm": self._port,
            },
        )
        self.h.start()

    def _port(self) -> Any:
        """A new arm port on the driver; its connect() takes the motor layer."""
        port = apai_backend.build_apai_arm(self.config, self.profile, self.clock)
        driver, motors = port.backend, self.motors

        def connect() -> None:
            # The state RebotRSArm.connect leaves behind, without the bus.
            driver._arm = motors
            driver._stopped = False
            driver._mit_kp = np.full(6, 30.0)
            driver._mit_kd = np.ones(6)
            driver._read_failures = 0
            driver._last_q = driver._read_positions()
            driver._last_q_t = time.monotonic()
            driver._last_cmd_q = None
            driver._last_cmd_t = None

        driver.connect = connect
        for name in ("close_gripper_torque", "close_gripper_two_stage"):
            driver.__dict__[name] = self._noted(name, getattr(driver, name))
        self.drivers.append(driver)
        return port

    def _noted(self, name: str, routine: Callable) -> Callable:
        def noted(*args: Any, **kwargs: Any) -> Any:
            self.motors.events.append(name)
            return routine(*args, **kwargs)

        return noted

    def hold_at(self, pose: list[float]) -> None:
        """Put the arm at this pose, holding, as after a move that ended there."""
        self.motors.set_pose(pose)
        self.h.app._motion.set(MotionState.HOLD)

    def said(self, kind: str) -> list[str]:
        """What the Thor terminal was told in events of this kind."""
        return [text for _, text, _ in self.h.heard(kind)]

    def close(self) -> None:
        self.h.close()


def thor_config(profile: Any, rig: bool = False) -> RobotAppConfig:
    """The robot app's file for APAI's arm model, the return to rest at SPEED.

    With rig it is the rig's robot file as it stands.
    """
    if rig:
        return apai_backend.config_for_profile(load_config(ROBOT_RIG), profile)
    base = make_config()
    limits = base.limits
    assert limits.agent.max_joint_speed == SPEED
    recovery = dataclasses.replace(limits.recovery, max_joint_speed=SPEED)
    base = dataclasses.replace(
        base,
        limits=dataclasses.replace(limits, recovery=recovery),
        rest=dataclasses.replace(base.rest, recovery_speed=SPEED),
    )
    return apai_backend.config_for_profile(base, profile)


def scene(folder: Path, speed: float = SPEED, home: list[float] | None = None) -> Any:
    """APAI's configuration of the rebot_rs arm on the robot link, a mock camera.

    speed is the Thor's agent speed limit; home replaces the ready pose.
    """
    config = importlib.import_module("apai.config")
    loaded = config.load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    data = loaded.as_dict()
    data["memory"]["experience_path"] = str(folder / "experience.json")
    views = [data, *(arm["resolved"] for arm in data["arms"] if "resolved" in arm)]
    for view in views:
        view["occupancy"]["enabled"] = True
        view["occupancy"]["required"] = True
        view["arm"]["type"] = "robot_link"
        view["safety"]["max_joint_vel"] = speed
        if home is not None:
            view["arm"]["home_q"] = list(home)
    for arm in data["arms"]:
        arm["type"] = "robot_link"
        if home is not None and "home_q" in arm:
            arm["home_q"] = list(home)
    return type(loaded)(data)


class Agent:
    """The workstation side: APAI's runtime on the arm adapter, behind the gate.

    task() runs tools as one task of the agent program: the task gate asks
    the operator, takes the arm, runs each tool and gives the arm back.
    `asked` are the gate's questions; `answer` is what the operator types.
    """

    def __init__(self, thor: Thor, made: AgentLink, folder: Path) -> None:
        demo = importlib.import_module("apai.apps.demo")
        occupancy = importlib.import_module("apai.perception.occupancy")
        runtime_module = importlib.import_module("apai.skills.runtime")
        self.thor, self.link = thor, made
        # (move id, last setpoint) of every MOVE the link sent, in order.
        self.moves: list[tuple[str, list[float]]] = []
        send_move = made.send_move

        def noted_move(move_id: str, start: Any, tol: Any, q: Any, *more: Any) -> None:
            self.moves.append((move_id, [float(value) for value in q[-1]]))
            send_move(move_id, start, tol, q, *more)

        made.send_move = noted_move
        self.asked: list[str] = []
        self.lines: list[str] = []
        self.answer: Callable[[str], str] = lambda prompt: "yes"
        settings = ArmSettings(
            expected_profile_id=thor.config.profile_id,
            calibration_id=thor.config.camera_calibration_id,
            allow_unnamed_calibration=True,
            agent_max_joint_speed=thor.speed,
        )
        with (
            mock.patch.object(occupancy, "OccupancyClient", FakeGridClient),
            mock.patch.object(demo, "make_arm", arm_factory(made, settings)),
        ):
            self.runtime, self.raw = demo.build_runtime(
                scene(folder, thor.speed, thor.home if thor.rig else None),
                folder / "run",
                lazy_arm=False,
            )
        self.safe = self.runtime.arm
        self.adapter, self.harness = self.safe.raw, self.safe.harness
        assert self.harness.occupancy.wait_until_usable(WAIT_S)
        self.adapter.bind_checker(
            CheckAhead(self.harness, EveryFrame(), period_s=PERIOD_S)
        )
        self.gate = task_gate.TaskGate(
            made,
            None,
            types.SimpleNamespace(max_joint_speed=thor.speed, end_wait_s=WAIT_S),
            ask=self._ask,
            say=self.lines.append,
            motion_skills=frozenset(runtime_module._MOTION_SKILLS),
        )
        # As the launcher wires them (agent_hooks).
        execute = self.runtime.execute
        self.runtime.execute = lambda name, args: self.gate.execute(execute, name, args)
        self.gate.bind_arm(self.adapter, self.harness)

    def _ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answer(prompt)

    def task(self, *tools: Any, text: str = "a task") -> list[dict]:
        """Run these tools as one task; each is a name or (name, arguments)."""
        results: list[dict] = []

        def run(_text: str) -> Any:
            for tool in tools:
                if callable(tool):
                    results.append(tool())
                    continue
                name, args = (tool, {}) if isinstance(tool, str) else tool
                results.append(self.runtime.execute(name, args))
            return types.SimpleNamespace(success=True, cancelled=False, summary="")

        self.gate.bind_run_task(run)(text)
        return results

    def state(self, shows: Callable[[Any], bool], what: str) -> Any:
        """The first ROBOT_STATE the link shows for which shows() holds."""
        end = time.monotonic() + WAIT_S
        while True:
            state = self.link.state()
            if shows(state):
                return state
            assert time.monotonic() < end, f"no ROBOT_STATE showed {what}: {state}"
            self.link.wait_change(0.05)

    def close(self) -> None:
        demo = importlib.import_module("apai.apps.demo")
        try:
            demo.shutdown_runtime(self.runtime, self.raw)
        finally:
            self.link.close()


@pytest.fixture
def settles(request: pytest.FixtureRequest) -> dict[str, float] | None:
    """How far each motor ends from every command, for a test on_the_rig."""
    return getattr(request, "param", None)


@pytest.fixture
def thor(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    settles: dict[str, float] | None,
) -> Iterator[Thor]:
    """The robot app at rest on the rebot_rs profile, with no robot connection.

    For a test on_the_rig: with the rig's robot file and ready pose. After
    the test: every torque cut the motor layer saw was at measured rest.
    """
    rig_home = None
    if settles is not None:
        folder = request.getfixturevalue("experiment_dir")
        arm = yaml.safe_load((folder / RIG_ARM).read_text(encoding="utf-8"))
        rig_home = [float(value) for value in arm["home_q"]]
    made = Thor(monkeypatch, rig_home, settles)
    try:
        yield made
    finally:
        made.close()
    tolerance = made.config.rest.rest_tol_rad
    for pose in made.motors.cuts():
        assert pose == pytest.approx(REST, abs=tolerance), "torque cut away from rest"


@pytest.fixture
def loopback(thor: Thor, tmp_path: Path) -> Iterator[Agent]:
    """The agent's side on a loopback carrier to the robot app."""
    robot_end, agent_end = loopback_pair()
    assert thor.h.app.attach_carrier(robot_end)
    with apai_runtime_env(tmp_path) as folder:
        agent = Agent(thor, AgentLink(agent_end), folder)
        try:
            yield agent
        finally:
            agent.close()
            robot_end.close()


@pytest.fixture
def stack(thor: Thor, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """The robot app behind the real robot link service, on a link of 127.0.0.1."""
    from link_stack import Stack

    # A short path: a socket path holds 107 bytes at most.
    made = Stack(thor.h, tmp_path_factory.mktemp("ls"))
    try:
        made.start_service()
        made.dial()
        made.console = made.client("console")
        made.connected(made.console)
        yield made
    finally:
        made.close()


@pytest.fixture
def served(thor: Thor, stack: Any, tmp_path: Path) -> Iterator[Agent]:
    """The agent's side as a client of the robot link service."""
    with apai_runtime_env(tmp_path) as folder:
        agent = Agent(thor, AgentLink.attach(stack.path, timeout_s=WAIT_S), folder)
        try:
            yield agent
        finally:
            agent.close()


@pytest.fixture(params=["loopback", pytest.param("service", marks=link)])
def agent(request: pytest.FixtureRequest) -> Agent:
    """The agent's side on a loopback carrier, and again behind the service."""
    name = "loopback" if request.param == "loopback" else "served"
    return request.getfixturevalue(name)


def unpowered(state: Any) -> bool:
    """ROBOT_STATE: at rest with the motors off, and no fault-stop."""
    return (
        state.motion == MotionState.REST_UNPOWERED.value
        and state.torque_on is False
        and state.stop_level == "none"
        and state.fault_reason == ""
    )


def position(targets: list[list[float]], goal: list[float]) -> int:
    """The index of the last joint target the motors got that is this pose."""
    found = [
        index
        for index, target in enumerate(targets)
        if target == pytest.approx(goal, abs=1e-6)
    ]
    assert found, f"the motors never got the joint target {goal}"
    return found[-1]


def check_rest_then_close_then_cut(thor: Thor, close: str) -> None:
    """The order at the motors: the last joint target, the jaw close, the cut."""
    motors = thor.motors
    assert motors.names() == [close, "disconnect"]
    closed = motors.events.index(close)
    # No joint target after the jaws began to close, and the last was rest.
    assert all(event[0] != "joints" for event in motors.events[closed:])
    assert motors.events[-1][0] == "disconnect"
    assert motors.targets()[-1] == pytest.approx(REST, abs=1e-9)
    assert motors.cuts() == [pytest.approx(REST, abs=thor.config.rest.rest_tol_rad)]


# APAI's rest tool, from a raised pose


def test_apais_rest_tool_folds_the_arm_and_the_motors_go_off_at_measured_rest(
    agent: Agent, thor: Thor
) -> None:
    functional = rest_pose()
    thor.hold_at(RAISED)
    session = thor.h.app.session_id
    (result,) = agent.task(task_gate.REST_SKILL)
    # APAI's own account: up to the ready pose, the fold, rest by measurement,
    # the jaws closed, the motors off.
    assert result.get("ok", True) is True, result
    assert (result["at"], result["route"]) == ("rest", ["initial_pose", "fold"])
    assert result["max_error_rad"] <= functional.REST_TOL_RAD
    assert (result["gripper"], result["torque"]) == ("closed", "released")
    assert result["safe_to_power_off"] is True and result["arm_state"] == "standby"
    # The functional route crossed the link: one MOVE per leg, each onto its
    # goal, and the motors got each goal as a joint target.
    waypoints = functional.fold_waypoints(thor.home, REST)
    goals = [thor.home, *waypoints]
    assert len(waypoints) == FOLDS
    assert [goal for _, goal in agent.moves] == [
        pytest.approx(goal, abs=1e-3) for goal in goals
    ]
    targets = thor.motors.targets()
    reached = [position(targets, goal) for _, goal in agent.moves]
    assert reached == sorted(reached) and reached[-1] == len(targets) - 1
    # At the motors: rest, then the driver's close, then the torque cut.
    check_rest_then_close_then_cut(thor, "close_gripper_torque")
    assert thor.said("release")
    # ROBOT_STATE says rest with the motors off; nothing is latched. The state
    # that shows the arm given back may come after the first one at rest.
    state = agent.state(
        lambda state: unpowered(state) and state.owner == "nobody",
        "rest with the motors off and nobody as owner",
    )
    assert state.session_id == session
    assert [sample.value for sample in state.joints] == pytest.approx(REST, abs=0.05)
    assert agent.adapter.connected is False
    assert agent.gate.requests == [
        ("agent", "completed", ""),
        ("nobody", "completed", ""),
    ]
    assert len(agent.asked) == 1  # the task's typed yes
    # And so it stays: the unpowered arm raises no fault-stop afterwards.
    time.sleep(STANDBY_S)
    assert unpowered(agent.link.state()) and len(thor.drivers) == 1


def test_after_standby_the_motors_come_on_again_and_the_next_move_runs(
    agent: Agent, thor: Thor
) -> None:
    """Motors on again, and a move, without a new start of the robot app.

    In standby the workstation's obstacle grid has no arm pose to mask the
    arm out of the camera's depth with, so its refreshes fail and it refuses
    every check. APAI's own arm waits, as it wakes, for the grid's next
    refresh with a pose; the first motion after standby must run here too.
    """
    session = thor.h.app.session_id
    # The arm lies at rest: the rest tool closes the jaws, the motors go off.
    (result,) = agent.task(task_gate.REST_SKILL)
    assert (result["at"], result["torque"]) == ("rest", "released"), result
    assert result["moved"] is False and thor.motors.targets() == []
    agent.state(unpowered, "rest with the motors off")
    time.sleep(STANDBY_S)
    assert agent.harness.occupancy.usable() is False  # refused while in standby

    (home,) = agent.task("move_home", text="raise the arm")
    assert home.get("ok", True) is True, home
    # The functional question came first, then the request across the link.
    question = agent.asked[-1]
    assert task_gate.MOTORS_OFF_NOTE in question
    assert task_gate.CONNECT_QUESTION in question
    assert (REQUEST_MOTORS_ON, "completed", "") in agent.gate.requests
    state = agent.state(lambda s: s.torque_on and s.owner == "nobody", "motors on")
    assert state.session_id == session and state.stop_level == "none"
    assert [sample.value for sample in state.joints] == pytest.approx(
        thor.home, abs=0.05
    )
    assert thor.motors.targets()[-1] == pytest.approx(thor.home, abs=1e-6)
    assert len(thor.drivers) == 2 and thor.h.app.running  # a new driver, one app
    assert thor.motors.names().count("disconnect") == 1
    assert thor.said("rest")[-1].startswith(rest_module.MOTORS_ON_AGAIN)


def test_a_step_inside_the_margin_at_the_rest_poses_stops_passes_both_checks(
    loopback: Agent, thor: Thor
) -> None:
    """The fold drives onto two joint stops, inside the joint margin.

    The joint margin is the band a checker keeps clear of every joint stop.
    APAI's rest tool streams its fold with the margin at 0; the check ahead
    and the Thor's agent checker take the margin away by one rule.
    """
    agent = loopback
    workstation, robot = agent.harness, thor.h.checkers.agent
    margin = workstation.limits.joint_margin
    assert margin == robot.harness.limits.joint_margin > 0.0
    low, high = (np.asarray(side, float) for side in workstation.kin.joint_limits)
    assert low[1] == low[2] == REST[1]  # the rest pose lies on these two stops
    check = agent.adapter._check
    step = 0.002  # rad a period: a quarter of the speed limit

    def folded(by: float) -> np.ndarray:
        """The pose with the two joints this far above their stops."""
        return np.array([0.0, by, by, 0.0, 0.0, 0.0])

    # Steps inside the margin at the two stops: the checker's own vet refuses
    # them, and both sides take the margin away for them. The second ends on
    # the rest pose. The third also moves another joint away from the rest
    # pose, as a fold step does from a pose a motor settled off its command.
    into, onto = (folded(margin - 0.001), folded(margin - 0.001 - step))
    last = (folded(step), np.array(REST))
    away = onto.copy()
    away[0] = step
    under, back = folded(0.001), folded(0.001)
    under[1], back[1] = -0.003, -0.003 + step
    robot.feed_watchdog()
    robot.begin_motion(robot.halt_count())
    try:
        for start, target in ((into, onto), last, (into, away)):
            assert "outside [" in workstation.vet_step(start, target, PERIOD_S)
            assert fold_margin(workstation, start, target) == 0.0
            assert check._vet_step(start, target) is None
            robot.check_setpoint(start, target, PERIOD_S)
        # A joint that reads below its stop may come back up. The agent's
        # envelope of this robot file ends at the stop; the rig's ends below it.
        assert "outside [" in workstation.vet_step(under, back, PERIOD_S)
        assert fold_margin(workstation, under, back) == 0.0
        assert robot._fold_margin(under, back) == 0.0
        assert check._vet_step(under, back) is None
        with pytest.raises(Refusal, match="outside the agent envelope"):
            robot.check_setpoint(under, back, PERIOD_S)
        # A step keeps the margin at a stop the rest pose does not lie on,
        # past the rest value, and when it is not a pose.
        clear, top = np.array(thor.home), np.array(thor.home)
        clear[4], top[4] = high[4] - margin - 0.001, high[4] - margin + 0.001
        past = np.array(REST)
        past[1] = -0.001
        for start, target in ((clear, top), (np.array(REST), past), (under, under)):
            assert fold_margin(workstation, start, target) is None
            assert "outside [" in check._vet_step(start, target)
            with pytest.raises(Refusal, match="outside"):
                robot.check_setpoint(start, target, PERIOD_S)
        broken = onto.copy()
        broken[0] = np.nan
        assert fold_margin(workstation, into, broken) is None
        assert "non-finite" in check._vet_step(into, broken)
        with pytest.raises(Refusal, match="finite"):
            robot.check_setpoint(into, broken, PERIOD_S)
        # The rule adds nothing outside the margin.
        assert fold_margin(workstation, folded(0.3), folded(0.3 - step)) is None
        robot.check_setpoint(folded(0.3), folded(0.3 - step), PERIOD_S)
    finally:
        robot.end_motion()
    # One rule on both sides: steps around the stops get the same answer.
    rng = np.random.default_rng(0)
    answers = []
    for _ in range(2000):
        stop = np.where(rng.random(6) < 0.5, low, high) + rng.uniform(-0.03, 0.03, 6)
        start = np.where(rng.random(6) < 0.3, stop, np.array(thor.home))
        target = start + rng.uniform(-2 * step, 2 * step, 6)
        answers.append(fold_margin(workstation, start, target))
        assert answers[-1] == robot._fold_margin(start, target)
    assert answers.count(0.0) > 50 and answers.count(None) > 50
    # A person and the Thor's own return to rest keep their rules.
    for other in (thor.h.checkers.human, thor.h.checkers.recovery):
        for start, target in ((into, onto), last, (into, away), (under, back)):
            assert other._fold_margin(start, target) is None


def test_the_rest_tool_is_refused_with_a_held_load_in_apais_words(
    loopback: Agent, thor: Thor
) -> None:
    agent = loopback
    thor.hold_at(RAISED)
    agent.runtime.held_object = "cube"
    sent = len(thor.motors.events)
    (result,) = agent.task("move_to_rest")
    assert result["ok"] is False and result["refused"] == "holding"
    assert result["error"].startswith("it is holding 'cube': put it down first")
    assert (result["moved"], result["torque"]) == (False, "on")
    # Nothing moved and nothing was cut: the arm holds where it was.
    assert len(thor.motors.events) == sent and thor.motors.pose() == RAISED
    state = agent.state(lambda s: s.owner == "nobody", "the arm given back")
    assert state.torque_on and state.motion == MotionState.HOLD.value
    assert REQUEST_MOTORS_OFF not in [name for name, _, _ in agent.gate.requests]


def test_a_rest_that_is_not_measured_leaves_the_arm_holding_with_the_motors_on(
    loopback: Agent, thor: Thor
) -> None:
    agent, functional = loopback, rest_pose()
    thor.hold_at(thor.home)
    # The second joint stops 0.06 rad short of its stop: outside the tolerance.
    short = functional.REST_TOL_RAD + 0.01
    thor.motors.floor["motor2"] = short
    asked: list[Result] = []

    def ask_for_the_motors_off() -> dict:
        # What the tool's last step would send, sent although rest is not proven.
        asked.append(
            agent.link.request_owner(
                REQUEST_MOTORS_OFF, quote=agent.link.pinned, timeout_s=WAIT_S
            )
        )
        return {}

    result, _ = agent.task("move_to_rest", ask_for_the_motors_off)
    assert result["ok"] is False and result["at"] == "not_rest"
    assert (result["torque"], result["safe_to_power_off"]) == ("on", False)
    assert f"measured {short:.3f} rad from the rest pose" in result["error"]
    assert result["route"] == ["fold"]  # it began at the ready pose
    # The tool sent no request for the motors off, and the Thor refuses one.
    assert agent.gate.requests == [
        ("agent", "completed", ""),
        ("nobody", "completed", ""),
    ]
    (refused,) = asked
    assert refused.status == "refused"
    assert refused.reason.startswith("the arm is not at rest: joint index 1 is at")
    # The arm holds at the end of the fold with the motors on; no jaw close.
    assert thor.motors.names() == []
    state = agent.state(lambda s: s.owner == "nobody", "the arm given back")
    assert state.torque_on and state.motion == MotionState.HOLD.value
    assert state.stop_level == "none"
    assert agent.adapter.connected is True
    assert thor.motors.pose() == pytest.approx([0.0, short, 0.0, 0.0, 0.0, 0.0])


@link
def test_a_stop_in_the_middle_of_the_fold_wins(
    served: Agent, thor: Thor, stack: Any
) -> None:
    agent = served
    thor.hold_at(thor.home)
    results: list[dict] = []
    worker = threading.Thread(
        target=lambda: results.extend(agent.task("move_to_rest")), daemon=True
    )
    worker.start()

    def in_the_second_fold_step() -> bool:
        if len(agent.moves) != 2:
            return False
        state = agent.link.state()
        return state.move_id == agent.moves[-1][0] and state.move_index >= 20

    end = time.monotonic() + REST_WAIT_S
    while not in_the_second_fold_step():
        assert time.monotonic() < end, "the fold did not reach its second step"
        time.sleep(0.01)
    quoted = agent.link.pinned  # the numbers the task's commands quote
    assert quoted is not None
    stack.console.ws.stop("pressed during the fold", "console")
    worker.join(REST_WAIT_S)
    assert not worker.is_alive()
    (result,) = results
    assert result["ok"] is False and result["at"] == "not_rest"
    assert result["torque"] == "on" and "the motion stopped" in result["error"]
    # The arm holds between the ready pose and rest with the motors on; the
    # STOP made nobody the owner, and nothing was sent to the jaws or cut.
    state = agent.state(lambda s: s.stop_level == "stop", "the STOP")
    assert state.owner == "nobody" and state.torque_on
    assert state.motion == MotionState.HOLD.value
    assert thor.motors.names() == []
    held = thor.motors.pose()
    assert 0.3 < held[1] < 0.9 and held == pytest.approx(thor.motors.targets()[-1])
    sent = len(thor.motors.events)
    # What the stopped task could still send quotes numbers the STOP ended;
    # and with the numbers of now the motors stay on away from measured rest.
    late = agent.link.request_owner(REQUEST_MOTORS_OFF, quote=quoted, timeout_s=WAIT_S)
    assert late.status == "refused" and "stop number" in late.reason
    anew = agent.link.request_owner(REQUEST_MOTORS_OFF, timeout_s=WAIT_S)
    assert anew.status == "refused"
    assert anew.reason.startswith("the arm is not at rest: joint index 1 is at")
    time.sleep(0.3)
    assert len(thor.motors.events) == sent and thor.motors.pose() == held


# The operator's typed rest at the console, with no agent program


@link
def test_the_operators_typed_rest_folds_closes_the_jaws_and_cuts_the_motors(
    thor: Thor, stack: Any
) -> None:
    console, functional = stack.console.ws, rest_pose()
    thor.hold_at(RAISED)
    session = thor.h.app.session_id
    # Only the typed phrase starts it: the console sends it as `confirmed`.
    unconfirmed = console.wait_result(console.rest(confirmed=False), timeout_s=WAIT_S)
    assert unconfirmed.status == "refused" and thor.motors.events == []
    done = console.wait_result(console.rest(confirmed=True), timeout_s=REST_WAIT_S)
    assert (done.status, done.detail) == (
        "completed",
        {"outcome": rest_module.AT_REST, "motors": "off"},
    )
    # The functional route, each leg told at the Thor terminal.
    told = [text for text in thor.said("rest") if "moves for" in text]
    assert [text.split(": ")[1] for text in told] == [
        rest_module.LEG_UP,
        *[
            rest_module.LEG_FOLD.format(step=step, steps=FOLDS)
            for step in range(1, FOLDS + 1)
        ],
    ]
    goals = [thor.home, *functional.fold_waypoints(thor.home, REST)]
    reached = [
        min(
            max(abs(a - b) for a, b in zip(target, goal))
            for target in thor.motors.targets()
        )
        for goal in goals
    ]
    assert max(reached) <= functional.TRACK_TOL_RAD
    # Rest by measurement, the driver's close, the torque cut: in that order.
    assert "return to rest: the arm is at rest, motors on" in thor.said("rest")
    assert thor.said(rest_module.EVENT_JAWS) == [
        rest_module.CLOSING_JAWS,
        rest_module.JAWS_CLOSED,
    ]
    check_rest_then_close_then_cut(thor, "close_gripper_torque")
    state = console.wait_state(unpowered, WAIT_S)
    assert state.owner == "nobody"

    # The console's `on`: the motors are on again in the same session.
    on = console.wait_result(
        console.request_owner(REQUEST_MOTORS_ON), timeout_s=REST_WAIT_S
    )
    assert (on.status, on.detail["motors"]) == ("completed", "on")
    state = console.wait_state(lambda s: s.torque_on, WAIT_S)
    assert state.motion == MotionState.REST_HOLD.value
    assert state.session_id == session and state.stop_level == "none"
    assert len(thor.drivers) == 2 and thor.motors.names().count("disconnect") == 1
    # With the motors on there is nothing to switch on.
    again = console.wait_result(
        console.request_owner(REQUEST_MOTORS_ON), timeout_s=WAIT_S
    )
    assert (again.status, again.reason) == ("refused", rest_module.MOTORS_ARE_ON)


# On the rig: the ready pose lies 0.025 rad above the two stops, so the fold
# is inside the joint margin from its first step, and each fold step starts
# from the pose the motors settled at.


def on_the_rigs_stack(thor: Thor) -> None:
    """The stack is the rig's: its robot file's speeds and its ready pose."""
    rig = load_config(ROBOT_RIG)
    assert thor.rig and thor.speed == rig.limits.agent.max_joint_speed
    assert thor.config.rest.recovery_speed == rig.rest.recovery_speed
    margin = thor.h.checkers.agent.harness.limits.joint_margin
    assert 0.0 < thor.home[1] - REST[1] < margin + 0.01
    thor.hold_at(list(thor.home))


def settled_off(thor: Thor) -> None:
    """Rest was measured with every motor as far off as the test made it."""
    off = [thor.motors.settles.get(name, 0.0) for name in thor.motors.arm.joint_names]
    assert thor.motors.pose() == pytest.approx(off, abs=1e-6)
    assert any(off) and thor.motors.targets()[-1] == pytest.approx(REST, abs=1e-9)
    assert thor.motors.cuts() == [pytest.approx(off, abs=1e-6)]
    assert thor.motors.names() == ["close_gripper_torque", "disconnect"]


@experiment
@on_the_rig
def test_on_the_rig_apais_rest_tool_folds_onto_rest_with_a_motor_that_settles_off(
    loopback: Agent, thor: Thor
) -> None:
    """The fold as the launcher runs it: each step with the joint margin at 0.

    The experiment's motion adapters, which the launcher enters for every
    task of the full stack, make APAI's rest tool stream every fold step so.
    """
    agent, functional = loopback, rest_pose()
    on_the_rigs_stack(thor)
    with import_experiment("thor_motion").motion_adapters(agent.runtime.cfg):
        assert [functional._margin(i, FOLDS) for i in range(FOLDS)] == [0.0] * FOLDS
        (result,) = agent.task(task_gate.REST_SKILL)
    assert result.get("ok", True) is True, result
    assert (result["at"], result["route"]) == ("rest", ["fold"]), result
    assert (result["gripper"], result["torque"]) == ("closed", "released")
    assert len(agent.moves) == FOLDS
    assert agent.harness.violations == []  # no check refused a setpoint
    settled_off(thor)
    agent.state(unpowered, "rest with the motors off")


@link
@experiment
@on_the_rig
def test_on_the_rig_the_operators_typed_rest_folds_with_a_motor_that_settles_off(
    thor: Thor, stack: Any
) -> None:
    console = stack.console.ws
    on_the_rigs_stack(thor)
    done = console.wait_result(console.rest(confirmed=True), timeout_s=REST_WAIT_S)
    assert (done.status, done.detail) == (
        "completed",
        {"outcome": rest_module.AT_REST, "motors": "off"},
    )
    told = [text for text in thor.said("rest") if "moves for" in text]
    assert [text.split(": ")[1] for text in told] == [
        rest_module.LEG_FOLD.format(step=step, steps=FOLDS)
        for step in range(1, FOLDS + 1)
    ]
    settled_off(thor)
    console.wait_state(unpowered, WAIT_S)
