# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RobotLinkArm: APAI's arm on the robot link, and its permit loop.

Three rigs. SimRig runs the real permit loop against the real robot app on the
machine's clock (the simulated arm): a move takes its real length. ScriptedRig
has a robot side the test scripts, for the RESULTs and states the real app
gives only on a fault. The last tests put the adapter under APAI's own SafeArm
and SafetyHarness, built by APAI for its mock scene, against the robot app on
APAI's mock arm.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.agent import arm_adapter, checkahead
from isaaccapture_examples.agentic_remote.agent.arm_adapter import (
    AGENT_TERMINAL,
    ArmSettings,
    RobotLinkArm,
    RobotLinkError,
    arm_factory,
)
from isaaccapture_examples.agentic_remote.agent.checkahead import (
    CheckAhead,
    FrameTimes,
)
from isaaccapture_examples.agentic_remote.agent.link_client import (
    LOST,
    AgentLink,
    Pinned,
)
from isaaccapture_examples.agentic_remote.agent.setpoints import sample_move
from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MOVE_ARMED,
    MOVE_SENDING,
    CheckerState,
    Gripper,
    GripperState,
    JointSample,
    Move,
    Permit,
    Result,
    Stop,
    Welcome,
)
from isaaccapture_examples.agentic_remote.robot_app import executor
from isaaccapture_examples.agentic_remote.robot_app import sim as sim_arm
from link_bench import FakeThor, decode, robot_state, until

from conftest import FakeGridClient, apai_runtime_env

apai = pytest.mark.apai

PERIOD_S = 0.02
N_JOINTS = 6
# The jaws of the test profiles, in profile units (rad of the jaw motor).
JAWS = {"open_pos": 6.2, "closed_pos": 0.0}
PROFILE = {"type": "robot_link", "n_joints": N_JOINTS, "gripper": JAWS}
# A goal 0.08 rad away: 25 setpoints in 0.5 s, peak 0.3 rad/s.
SHORT = [0.08, 0.0, 0.0, 0.0, 0.0, 0.0]
# 60 setpoints in 1.2 s, peak 0.31 rad/s: long enough to step in mid-move.
LONG = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]


def types() -> Any:
    return importlib.import_module("apai.types")


class Grid:
    """What CheckAhead reads of APAI's obstacle grid."""

    def __init__(self) -> None:
        self._refresh_lock = threading.RLock()
        self._last_refresh: float | None = None
        self.refreshes = 0
        self.cache_clears = 0


class Checker:
    """Stands in for the workstation checker (APAI's SafetyHarness).

    It records every check with the camera-frame time of the grid it ran on,
    and refuses a pose `blocked` names: an obstacle the test puts in the path.
    """

    def __init__(self, log: list[tuple]) -> None:
        self.occupancy = Grid()
        self.violations: list[str] = []
        self.estopped = False
        self.halted: str | None = None
        self._grasp_exempt: tuple | None = None
        self.approved_commands = 0
        self.blocked: Callable[[np.ndarray], str | None] = lambda q: None
        # Called with the count of callback calls so far, before each one.
        self.before_approve: Callable[[int], None] = lambda count: None
        self.cleared: list[str] = []
        self._log = log

    def vet_step(self, q_prev: Any, q_next: Any, dt: float) -> str | None:
        assert dt == PERIOD_S
        frame_t = self.occupancy._last_refresh
        self._log.append(("vet", np.array(q_next), np.array(q_prev), frame_t))
        return self.blocked(np.asarray(q_next))

    def approve(self, q_prev: Any, q_next: Any, dt: float) -> None:
        assert dt == PERIOD_S
        self.before_approve(self.approved_commands)
        frame_t = self.occupancy._last_refresh
        self._log.append(("approve", np.array(q_next), np.array(q_prev), frame_t))
        reason = self.blocked(np.asarray(q_next))
        if reason is not None:
            raise types().SafetyViolation(reason)
        self.approved_commands += 1

    def clear_grasp_exemption(self) -> None:
        self._grasp_exempt = None
        self.cleared.append("grasp")

    def clear_contact_zones(self, reason: str = "cleared") -> int:
        self.cleared.append(f"contact:{reason}")
        return 2


class Rig:
    """An adapter on a link, with a scripted checker and a real check ahead."""

    link: AgentLink

    def _build(self, link: AgentLink, **settings: Any) -> None:
        self.link = link
        self.log: list[tuple] = []  # checks and link messages, in order
        self.permit_times: list[float] = []  # when each PERMIT was sent
        self.checker = Checker(self.log)
        self.times = FrameTimes()
        self.check = CheckAhead(self.checker, self.times, period_s=PERIOD_S)
        self.settings = ArmSettings(
            **{
                "expected_profile_id": "sim-6dof",
                "allow_unnamed_calibration": True,
                **settings,
            }
        )
        self.adapter = RobotLinkArm(PROFILE, link=link, settings=self.settings)
        self._spy("send_move", "move")
        self._spy("send_permit", "permit")
        self._spy("gripper", "gripper")
        self._spy("checker_state", "checker_state")
        self._spy("stop", "stop")
        self._camera: threading.Thread | None = None
        self._camera_off = threading.Event()

    def _spy(self, name: str, kind: str) -> None:
        real = getattr(self.link, name)

        def spied(*args: Any, **kwargs: Any) -> Any:
            seen = None
            if kind == "permit":
                state = self.shown()
                if state is not None and state.move_id == args[0]:
                    seen = state.move_index
                self.permit_times.append(time.monotonic())
            self.log.append((kind, *args, seen))
            return real(*args, **kwargs)

        setattr(self.link, name, spied)

    def shown(self) -> Any:
        """The newest ROBOT_STATE, None when there is none or the link is lost."""
        try:
            return self.link.state()
        except RuntimeError:
            return None

    def ready(self) -> RobotLinkArm:
        self.adapter.connect()
        self.adapter.bind_checker(self.check)
        return self.adapter

    def new_grid(self) -> float:
        """Install a grid from a camera frame captured now; its evidence time."""
        grid = self.checker.occupancy
        frame_t, wall = time.monotonic(), time.time()
        self.times.note(frame_t, wall)
        with grid._refresh_lock:
            grid._last_refresh = frame_t
            grid.refreshes += 1
        return wall

    @contextmanager
    def camera(self, every_s: float = 0.1) -> Iterator[None]:
        """New grids at this pace while open, as perception makes them."""

        def run() -> None:
            while not self._camera_off.wait(every_s):
                self.new_grid()

        self._camera_off.clear()
        thread = threading.Thread(target=run, name="test-camera", daemon=True)
        thread.start()
        try:
            yield
        finally:
            self._camera_off.set()
            thread.join(5.0)

    def move(self, goal: Any, duration_s: float, **keywords: Any) -> bool:
        """adapter.stream_to on a fresh grid, as SafeArm calls it."""
        self.new_grid()
        keywords.setdefault("approve", self.checker.approve)
        return self.adapter.stream_to(np.asarray(goal, float), duration_s, **keywords)

    def of(self, kind: str) -> list[tuple]:
        return [event for event in list(self.log) if event[0] == kind]

    def setpoints(self) -> list[np.ndarray]:
        """The setpoints of the last MOVE sent."""
        return [np.asarray(row) for row in self.of("move")[-1][4]]

    def index_of(self, q: np.ndarray) -> int:
        return next(i for i, row in enumerate(self.setpoints()) if (row == q).all())

    def approved(self) -> list[int]:
        """The setpoint index of every call of the checker callback, in order."""
        return [self.index_of(event[1]) for event in self.of("approve")]

    def permits(self) -> list[tuple[int, float]]:
        """(through_index, evidence_wall_s) of every PERMIT sent, in order."""
        return [(event[2], event[3]) for event in self.of("permit")]

    def obstacle_beyond(self, joint0: float, reason: str = "obstacle ahead") -> None:
        """From now on the checker refuses every pose with joint 0 past this."""
        self.checker.blocked = lambda q: reason if q[0] > joint0 else None


class SimRig(Rig):
    """The real robot app on the simulated arm, in real time."""

    def __init__(self, harness: Any, **settings: Any) -> None:
        self.h = harness
        self.robot_end, self.agent_end = loopback_pair()
        self._build(AgentLink(self.agent_end), **settings)
        assert harness.app.attach_carrier(self.robot_end)
        self.ready()

    def own(self) -> Pinned:
        """The agent becomes control owner and the adapter's task opens."""
        result = self.link.request_owner("agent", timed_return=False)
        assert result.status == "completed", result.reason
        number = result.detail["owner_number"]

        def shown() -> bool:
            state = self.link.state()
            return state.owner == "agent" and state.owner_number == number

        until(shown, "the agent as control owner")
        state = self.link.state()
        pinned = Pinned(state.session_id, state.owner_number, state.stop_number)
        self.adapter.open_task(pinned)
        return pinned

    def close(self) -> None:
        self.link.close()
        self.robot_end.close()


@pytest.fixture
def make_sim(make_realtime_robot_app: Callable[..., Any]) -> Iterator[Callable]:
    made: list[SimRig] = []

    def make(**settings: Any) -> SimRig:
        rig = SimRig(make_realtime_robot_app(connected=False), **settings)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


@pytest.fixture
def sim(make_sim: Callable[..., SimRig]) -> SimRig:
    """A robot app at rest with the agent as control owner."""
    rig = make_sim()
    rig.own()
    return rig


def a_welcome(**fields: Any) -> Welcome:
    return Welcome(
        **{
            "session_id": "s1",
            "robot_id": "sim-arm",
            "profile_id": "sim-6dof",
            "joint_names": [f"j{i}" for i in range(N_JOINTS)],
            "joint_min": [-2.0] * N_JOINTS,
            "joint_max": [2.0] * N_JOINTS,
            "rest_q": [0.0] * N_JOINTS,
            "setpoint_period_s": PERIOD_S,
            "max_move_setpoints": 3000,
            "limit_set_versions": {"agent": "sim-1"},
            **fields,
        }
    )


class ScriptedThor:
    """A robot side the test scripts, on the machine's clocks.

    It publishes a ROBOT_STATE every 0.01 s with fresh joint readings, admits
    every MOVE and shows it as armed, and ends a move the way the test says.
    """

    def __init__(self) -> None:
        self.thor = FakeThor("test")
        self.agent_end = self.thor.service_end
        self.lock = threading.RLock()
        self.q = [0.0] * N_JOINTS
        self.fields: dict[str, Any] = {"owner": "agent", "torque_on": True}
        self.jaws = GripperState(1.0, None, False, True, 1.0, False, "empty")
        self.health: dict[str, Any] = {"reading_age_s": 0.0}
        self.invalid_joint: int | None = None
        self.publishing = True
        self.arming = True  # show an admitted move as armed
        self.admit: Result | None = None  # the first RESULT of a MOVE, if not None
        # What ends the move at its first PERMIT: (status, reason, detail, end_q).
        self.at_first_permit: tuple | None = None
        self.on_cancel: tuple | None = (
            "refused",
            "permit reaches setpoint -1, not 3",
            {"outcome": "refused", "sent": 3, "total": 25},
            None,
        )
        self.jaw_reply: Callable[[Gripper], Result] = lambda m: Result(
            m.command_id, "completed", "", None, {"outcome": "reached", "opening": 0.5}
        )
        self.moves: list[Move] = []
        self.permits: list[Permit] = []
        self.grippers: list[Gripper] = []
        self.stops: list[Stop] = []
        self.checker_states: list[CheckerState] = []
        self.others: list[Any] = []
        self._seq = 0
        self._move: Move | None = None
        self._stage = ""
        self._index = 0
        self._off = threading.Event()
        self.thor.send(a_welcome())
        self._thread = threading.Thread(target=self._run, name="thor", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._off.set()
        self._thread.join(5.0)
        self.thor.carrier.close()

    def end_move(
        self,
        status: str,
        reason: str = "",
        detail: dict | None = None,
        end_q: list[float] | None = None,
    ) -> None:
        with self.lock:
            move, self._move, self._stage, self._index = self._move, None, "", 0
        assert move is not None, "no move is running"
        self.thor.send(Result(move.command_id, status, reason, end_q, detail or {}))

    def progress(self, index: int) -> None:
        with self.lock:
            self._index = index

    def _run(self) -> None:
        while not self._off.wait(0.01):
            while (item := self.thor.carrier.receive(0)) is not None:
                self._take(decode(*item))
            if self.publishing:
                self._publish()

    def _publish(self) -> None:
        with self.lock:
            self._seq += 1
            mono, wall = time.monotonic(), time.time()
            move = self._move
            state = dataclasses.replace(
                robot_state(seq=self._seq),
                clock_mono_s=mono,
                clock_wall_s=wall,
                joints=[
                    JointSample(j, self.q[j], mono, mono, j != self.invalid_joint)
                    for j in range(N_JOINTS)
                ],
                gripper=self.jaws,
                health=dict(self.health),
                move_id="" if move is None else move.command_id,
                move_index=self._index,
                move_total=0 if move is None else len(move.setpoints),
                move_stage=self._stage,
                **self.fields,
            )
        self.thor.send(state)

    def _take(self, message: Any) -> None:
        if isinstance(message, Move):
            self.moves.append(message)
            if self.admit is not None:
                answer = dataclasses.replace(self.admit, command_id=message.command_id)
                self.thor.send(answer)
                return
            self.thor.send(Result(message.command_id, "accepted"))
            with self.lock:
                self._move, self._index = message, 0
                self._stage = MOVE_ARMED if self.arming else ""
        elif isinstance(message, Permit):
            first = not any(p.move_id == message.move_id for p in self.permits)
            self.permits.append(message)
            if self._move is None or self._move.command_id != message.move_id:
                return
            with self.lock:
                self._stage = MOVE_SENDING
            if message.through_index < 0:
                if self.on_cancel is not None:
                    self.end_move(*self.on_cancel)
            elif first and self.at_first_permit is not None:
                self.end_move(*self.at_first_permit)
        elif isinstance(message, Gripper):
            self.grippers.append(message)
            self.thor.send(self.jaw_reply(message))
        elif isinstance(message, Stop):
            self.stops.append(message)
        elif isinstance(message, CheckerState):
            self.checker_states.append(message)
            self.thor.send(Result(message.command_id, "completed"))
        else:
            self.others.append(message)


class ScriptedRig(Rig):
    PIN = Pinned("s1", 1, 0)

    def __init__(self, **settings: Any) -> None:
        self.thor = ScriptedThor()
        self._build(AgentLink(self.thor.agent_end), **settings)

    def own(self) -> RobotLinkArm:
        self.ready()
        self.adapter.open_task(self.PIN)
        return self.adapter

    def close(self) -> None:
        self.link.close()
        self.thor.close()


@pytest.fixture
def make_scripted() -> Iterator[Callable[..., ScriptedRig]]:
    made: list[ScriptedRig] = []

    def make(**settings: Any) -> ScriptedRig:
        made.append(ScriptedRig(**settings))
        return made[-1]

    yield make
    for rig in made:
        rig.close()


@pytest.fixture
def scripted(make_scripted: Callable[..., ScriptedRig]) -> ScriptedRig:
    """A scripted robot side with the agent as owner and the task open."""
    rig = make_scripted()
    rig.own()
    return rig


def in_thread(call: Callable[[], Any]) -> threading.Thread:
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread


# The module and the class


def test_the_module_imports_without_apai() -> None:
    package = Path(arm_adapter.__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.agent import arm_adapter\n"
        "arm_adapter.ArmSettings()\n"
        "assert arm_adapter.RobotLinkArm.motion_rate_hz == 50.0\n"
        "assert 'apai' not in sys.modules\n"
        "assert not any(n == 'isaaccapture' or n.startswith('isaaccapture.') "
        "for n in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(package), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def link_calls(source: str) -> set[str]:
    """Every method the source calls on a link (self._link.x(...), link.x(...))."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = ast.unparse(node.func.value)
            if target in ("self._link", "link"):
                found.add(node.func.attr)
    return found


def test_the_adapter_sends_only_these_link_messages() -> None:
    """Nothing it can send releases the motors: no RELEASE, SHUTDOWN, RESET, OPEN."""
    source = Path(arm_adapter.__file__).read_text(encoding="utf-8")
    sending = {"send_move", "send_permit", "gripper", "checker_state", "stop"}
    reading = {
        "wait_ready",
        "state",
        "state_arrived_s",
        "wait_change",
        "wait_first",
        "wait_final",
        "final_result",
        "wait_reflected",
        "forget",
        "lost_reason",
        "new_id",
        "pin",
        "unpin",
    }
    assert link_calls(source) <= sending | reading
    assert sending <= link_calls(source)
    for name in ("Release", "Shutdown", "Reset", "Open", "OwnerRequest"):
        assert name not in source, name
    assert "request_owner" not in source and "._client" not in source
    assert "isaaccapture." not in source.replace("isaaccapture.examples", "")


def test_the_first_values_are_the_contracts() -> None:
    settings = ArmSettings()
    assert (settings.reach_setpoints, settings.renew_s, settings.life_s) == (
        25,
        0.05,
        1.0,
    )
    assert (settings.start_tol_rad, settings.fresh_wait_s) == (0.02, 0.6)
    assert (settings.start_grid_age_s, settings.start_grid_wait_s) == (1.0, 3.0)
    assert (settings.expires_in_s, settings.cancel_wait_s) == (2.0, 2.0)
    assert (settings.fresh_s, settings.result_timeout_s) == (1.0, 12.0)
    assert (settings.state_max_age_s, settings.connect_timeout_s) == (0.5, 10.0)
    # The Thor checks setpoint 0 from its own pose: half a period's travel.
    assert settings.start_tol(0.02) == pytest.approx(0.004)
    slow = dataclasses.replace(settings, agent_max_joint_speed=0.1)
    assert slow.start_tol(0.02) == pytest.approx(0.001)
    fast = dataclasses.replace(settings, agent_max_joint_speed=5.0)
    assert fast.start_tol(0.02) == 0.02
    assert arm_adapter.HOLD_FAILED in executor.HOLD_FAILED
    assert arm_adapter.LOST == "robot connection lost"


@apai
def test_the_arm_is_an_apai_driver_with_the_teardown_flags(
    scripted: ScriptedRig,
) -> None:
    arm_base = importlib.import_module("apai.control.arm_base")
    adapter = scripted.adapter
    assert isinstance(adapter, arm_base.ArmBase) and isinstance(adapter, RobotLinkArm)
    assert type(adapter).__name__ == "RobotLinkArm"
    assert not type(adapter).__abstractmethods__
    assert adapter.n_joints == 6 and type(adapter.motion_rate_hz) is float
    # APAI neither parks this arm nor cuts its torque.
    assert adapter.shutdown_park_required is False
    assert adapter.disconnect_preserves_drive_state is True
    assert adapter.disconnect_cuts_torque is False
    assert adapter.hold_key is None
    assert adapter.connect_cuts_torque is False
    assert adapter.connected_on_confirmation is False
    assert adapter.close_stages_skipped == 0 and adapter._stopped is False
    for name in (
        "motion_status",
        "ensure_motion_ready",
        "check_resume_ready",
        "physics_engine",
        "validate_simulation_clock",
        "reset_props",
        "attachment_feedback_version",
        "stream_profile",
        "close_gripper_torque",
    ):
        assert not hasattr(adapter, name), name
    stream = inspect.signature(arm_base.ArmBase.stream_to).parameters
    ours = inspect.signature(RobotLinkArm.stream_to).parameters
    assert list(ours) == list(stream)
    assert [p.default for p in ours.values()] == [p.default for p in stream.values()]
    assert (
        list(inspect.signature(RobotLinkArm.stream_path).parameters)[2:]
        == list(stream)[2:]
    )


@apai
def test_the_factory_builds_the_adapter_for_its_type_and_apais_arm_otherwise(
    scripted: ScriptedRig,
) -> None:
    config = importlib.import_module("apai.config")
    mock_arm = importlib.import_module("apai.control.mock_arm")
    make_arm = arm_factory(scripted.link, scripted.settings)
    built = make_arm(config.Cfg(PROFILE), kinematics="kin")
    assert isinstance(built, RobotLinkArm) and built.kin == "kin"
    assert make_arm.made == [built]
    assert built._opening(6.2) == 1.0 and built._opening(3.1) == pytest.approx(0.5)
    other = make_arm(config.Cfg({"type": "mock", "n_joints": 6}))
    assert isinstance(other, mock_arm.MockArm) and make_arm.made == [built]


# connect(), disconnect(), get_state()


@apai
def test_connect_sends_nothing_and_disconnect_sends_nothing(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted()
    assert rig.adapter.connected is False
    rig.ready()
    assert rig.adapter.connected is True
    rig.adapter.disconnect()
    assert rig.adapter.connected is False
    time.sleep(0.05)
    thor = rig.thor
    assert not (thor.moves or thor.permits or thor.grippers or thor.stops)
    assert not (thor.checker_states or thor.others)
    assert rig.link.pinned is None  # connect() takes no ownership


@apai
@pytest.mark.parametrize(
    ("welcome", "settings", "text"),
    [
        ({"protocol_version": 2}, {}, "protocol version 2"),
        ({"joint_names": ["a", "b"]}, {}, "has 2 joints, the arm profile 6"),
        ({"setpoint_period_s": 0.01}, {}, "setpoint period is 0.01 s"),
        ({"profile_id": "other"}, {}, "profile is 'other', expected 'sim-6dof'"),
        ({}, {"allow_unnamed_calibration": False}, "names no camera calibration"),
        (
            {"camera_calibration_id": "stereo-a+mount-b"},
            {"calibration_id": "stereo-c+mount-d"},
            "camera calibration is 'stereo-a.mount-b', this program's 'stereo-c",
        ),
    ],
)
def test_connect_refuses_a_robot_app_that_does_not_fit(
    make_scripted: Callable[..., ScriptedRig],
    welcome: dict[str, Any],
    settings: dict[str, Any],
    text: str,
) -> None:
    rig = make_scripted(**settings)
    rig.thor.thor.send(a_welcome(**welcome))
    until(lambda: rig.link.welcome == a_welcome(**welcome), "the WELCOME")
    with pytest.raises(RobotLinkError, match=text):
        rig.adapter.connect()
    assert rig.adapter.connected is False


@apai
def test_connect_takes_a_named_calibration_that_is_the_same(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted(calibration_id="stereo-a+mount-b")
    named = a_welcome(camera_calibration_id="stereo-a+mount-b")
    rig.thor.thor.send(named)
    until(lambda: rig.link.welcome == named, "the WELCOME")
    rig.adapter.connect()
    assert rig.adapter.connected is True


@apai
def test_connect_refuses_motors_that_are_off_and_a_robot_app_that_is_not_there(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted()
    rig.thor.fields["torque_on"] = False
    until(lambda: getattr(rig.shown(), "torque_on", None) is False, "motors off")
    with pytest.raises(RobotLinkError, match="motors off"):
        rig.adapter.connect()
    silent = make_scripted(connect_timeout_s=0.2)
    silent.thor.invalid_joint = 0
    time.sleep(0.1)
    began = time.monotonic()
    with pytest.raises(RobotLinkError, match="not ready"):
        silent.adapter.connect()
    assert time.monotonic() - began < 1.0


@apai
def test_connect_warns_when_the_workstation_range_is_wider(
    make_scripted: Callable[..., ScriptedRig], caplog: pytest.LogCaptureFixture
) -> None:
    wide = (-2.0, -2.5, -2.0, -2.0, -2.0, -2.0)
    rig = make_scripted(joint_min=wide, joint_max=(2.0,) * 6)
    with caplog.at_level(logging.WARNING):
        rig.adapter.connect()
    assert rig.adapter.connected is True
    assert "joint index 1" in caplog.text


@apai
def test_get_state_is_the_newest_published_pose_and_never_waits(
    scripted: ScriptedRig,
) -> None:
    scripted.thor.q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    scripted.thor.health = {"reading_age_s": 0.12}
    until(
        lambda: (
            scripted.link.state().health.get("reading_age_s") == 0.12
            and scripted.link.state().joints[5].value == 0.6
        ),
        "the new pose",
    )
    began = time.monotonic()
    state = scripted.adapter.get_state()
    assert time.monotonic() - began < 0.05
    assert isinstance(state, types().RobotState)
    assert state.q == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # The local arrival time of that state minus the age of its readings.
    assert state.t == pytest.approx(time.monotonic() - 0.12, abs=0.05)
    assert (state.gripper_pos, state.gripper_valid) == (0.0, False)


@apai
@pytest.mark.parametrize(
    ("spoil", "text"),
    [
        (lambda thor: setattr(thor, "invalid_joint", 3), "no valid reading"),
        (lambda thor: thor.health.clear(), "None s old"),
        (lambda thor: thor.health.update(reading_age_s=0.6), "0.6 s old"),
        (lambda thor: setattr(thor, "publishing", False), "ROBOT_STATE is"),
        (lambda thor: thor.thor.carrier.break_link(), "robot link service is gone"),
    ],
)
def test_get_state_raises_when_the_feedback_cannot_be_trusted(
    scripted: ScriptedRig, spoil: Callable[[ScriptedThor], None], text: str
) -> None:
    spoil(scripted.thor)
    time.sleep(0.6)  # past state_max_age_s for the state that stopped coming
    with pytest.raises(RuntimeError, match=text) as raised:
        scripted.adapter.get_state()
    assert "robot link feedback lost" in str(raised.value)
    assert not isinstance(raised.value, types().SafetyViolation)


def jaw_state(opening: Any, t: Any, valid: bool, target_t: Any, now: float) -> Any:
    return dataclasses.replace(
        robot_state(),
        clock_mono_s=now,
        target_t=target_t,
        gripper=GripperState(opening, t, valid, True, opening, False, "empty"),
    )


@apai
def test_a_jaw_reading_older_than_the_last_setpoint_or_than_fresh_s_is_unknown(
    scripted: ScriptedRig,
) -> None:
    reading = scripted.adapter._jaw_reading
    # Half open is 3.1 rad of the profile's 6.2 rad travel.
    assert reading(jaw_state(0.5, 99.5, True, None, 100.0)) == (3.1, True)
    assert reading(jaw_state(0.5, 99.5, True, 99.5, 100.0)) == (3.1, True)
    assert reading(jaw_state(0.5, 99.5, True, 99.6, 100.0)) == (0.0, False)  # setpoint
    assert reading(jaw_state(0.5, 98.9, True, None, 100.0)) == (0.0, False)  # too old
    assert reading(jaw_state(0.5, 99.5, False, None, 100.0)) == (0.0, False)
    assert reading(jaw_state(None, None, False, None, 100.0)) == (0.0, False)
    assert reading(jaw_state(0.5, None, True, None, 100.0)) == (0.0, False)
    # Through get_state(): known right after a jaw reading, unknown a second on.
    thor = scripted.thor
    thor.jaws = GripperState(0.5, time.monotonic(), True, True, 0.5, False, "empty")
    until(lambda: scripted.adapter.get_state().gripper_valid, "the jaw reading")
    assert scripted.adapter.get_state().gripper_pos == pytest.approx(3.1)
    time.sleep(1.05)
    state = scripted.adapter.get_state()
    assert (state.gripper_pos, state.gripper_valid) == (0.0, False)


@apai
def test_the_mask_pose_is_the_published_pose_with_the_jaws_unknown(
    scripted: ScriptedRig,
) -> None:
    robot_mask = importlib.import_module("apai.perception.robot_mask")
    thor = scripted.thor
    thor.jaws = GripperState(0.5, time.monotonic(), True, True, 0.5, False, "empty")
    thor.q = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    until(lambda: scripted.adapter.get_state().gripper_valid, "the jaw reading")
    state = scripted.adapter.body_mask_state()
    assert state.gripper_valid is False
    assert scripted.adapter.body_mask_q() == pytest.approx(thor.q)
    thor.publishing = False
    time.sleep(0.6)
    with pytest.raises(robot_mask.BodyMaskUnavailable, match="feedback lost"):
        scripted.adapter.body_mask_state()


# Calls that are refused outright


@apai
def test_calls_the_robot_link_does_not_take_are_refused_and_send_nothing(
    scripted: ScriptedRig,
) -> None:
    adapter = scripted.adapter
    with pytest.raises(types().SafetyViolation, match="joint paths are not supported"):
        adapter.stream_path([SHORT], 1.0, approve=scripted.checker.approve)
    with pytest.raises(RuntimeError, match="whole moves"):
        adapter.send_joint_target(np.zeros(6))
    with pytest.raises(RuntimeError, match="start a new task"):
        adapter.resume()
    with pytest.raises(TypeError):  # SafeArm then retries without the hint
        adapter.stream_to(np.zeros(6), 1.0, approve=None, bias_compensate=np.zeros(6))
    with pytest.raises(TypeError):
        adapter.stream_to(np.zeros(6), 1.0, feedback_guard=lambda state: None)
    assert not scripted.thor.moves and not scripted.of("move")


@apai
@pytest.mark.parametrize(
    ("keywords", "goal", "text"),
    [
        ({"approve": None}, SHORT, "needs the checker callback"),
        ({"rate_hz": 100.0}, SHORT, "50 setpoints a second, not 100.0"),
        ({"settle_tol": 0.01}, SHORT, "are the robot app's"),
        ({"settle_timeout_s": 1.0}, SHORT, "are the robot app's"),
        ({}, [0.1, 0.0], "6 finite joint values"),
        ({}, [0.1, 0.0, 0.0, 0.0, 0.0, float("nan")], "6 finite joint values"),
    ],
)
def test_a_move_is_refused_before_anything_is_read(
    scripted: ScriptedRig, keywords: dict[str, Any], goal: list[float], text: str
) -> None:
    with pytest.raises(types().SafetyViolation, match=text) as raised:
        scripted.move(goal, 0.5, **keywords)
    assert "no motion sent" in str(raised.value)
    assert not scripted.of("vet") and not scripted.of("approve")
    assert not scripted.thor.moves


@apai
def test_motion_and_jaws_need_an_open_task_a_bound_checker_and_ownership(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    violation = types().SafetyViolation
    rig = make_scripted()
    rig.adapter.connect()
    with pytest.raises(violation, match="does not own the arm; motion needs a"):
        rig.move(SHORT, 0.5)
    with pytest.raises(violation, match="does not own the arm"):
        rig.adapter.set_gripper(3.0)
    with pytest.raises(violation, match="does not own the arm"):
        rig.adapter.close_gripper_two_stage()
    rig.adapter.open_task(ScriptedRig.PIN)
    with pytest.raises(violation, match="no check ahead is bound"):
        rig.move(SHORT, 0.5)
    rig.adapter.bind_checker(rig.check)
    rig.thor.fields["owner"] = "nobody"
    until(lambda: rig.link.lost_reason() is not None, "the lost ownership")
    with pytest.raises(violation, match="control owner is nobody, not the agent"):
        rig.move(SHORT, 0.5)
    assert rig.adapter._stopped is True  # the mirror of the Thor's state
    with pytest.raises(violation, match="arm stop latched"):
        rig.adapter.set_gripper(3.0)
    rig.adapter.close_task()
    assert rig.link.pinned is None
    with pytest.raises(violation):
        rig.move(SHORT, 0.5)
    assert not rig.thor.moves and not rig.thor.grippers
    assert not rig.of("vet") and not rig.of("approve")


# A move against the real robot app, in real time


@apai
def test_setpoints_and_period_are_what_the_thor_accepts(sim: SimRig) -> None:
    before = sim.checker.approved_commands
    assert sim.move(SHORT, 0.5) is True
    rows = sample_move(np.zeros(6), SHORT, 0.5, 50.0)
    assert len(rows) == 25
    # Every setpoint reached the motors, once, in order, value for value.
    assert sim.h.sent() == [row.tolist() for row in rows]
    _, move_id, start, tol, _, region, evidence, expires, _ = sim.of("move")[0]
    assert start == pytest.approx(np.zeros(6)) and region is None
    assert tol == pytest.approx(0.004) and expires == 2.0
    assert sim.times.lookup(sim.checker.occupancy._last_refresh) == evidence
    assert sim.checker.approved_commands == before + 25
    state = sim.adapter.get_state()  # the end of the move is already shown
    assert state.q == pytest.approx(SHORT, abs=0.03)
    assert sim.link.state().owner == "agent" and sim.adapter._stopped is False
    assert sim.link.final_result(move_id) is None  # forgotten
    stats = sim.adapter.last_move
    assert stats["setpoints"] == stats["approved"] == 25
    assert stats["armed_after_s"] < 0.5 and stats["first_permit_after_s"] < 0.6
    assert stats["max_pass_s"] < 0.2
    assert "cancelled" not in stats


@apai
def test_the_callback_clears_each_setpoint_once_in_order_ahead_of_its_permit(
    sim: SimRig,
) -> None:
    """No PERMIT covers a setpoint the callback has not cleared (5.3)."""
    with sim.camera(0.1):
        assert sim.move(LONG, 1.2) is True
    count = len(sim.setpoints())
    assert count == 60
    assert sim.approved() == list(range(count))  # once each, in order
    cleared = -1
    first = True
    for event in sim.log:
        if event[0] == "approve":
            cleared = sim.index_of(event[1])
        elif event[0] == "permit":
            through = event[2]
            assert 0 <= through <= cleared  # never past what the callback cleared
            assert through <= count - 1
            if first:
                assert through == 25  # setpoints 0..reach
                first = False
    assert sim.permits()[-1][0] == count - 1
    assert len(sim.permits()) >= 10  # renewed every pass, also with nothing new
    # The callback saw each setpoint from the one before it, the first from
    # the bound start pose.
    previous = np.zeros(6)
    for event in sim.of("approve"):
        assert event[2] == pytest.approx(previous)
        previous = event[1]
    assert sim.h.sent() == [row.tolist() for row in sim.setpoints()]


@apai
def test_every_permitted_setpoint_was_checked_on_the_grid_the_permit_quotes(
    sim: SimRig,
) -> None:
    with sim.camera(0.08):
        assert sim.move(LONG, 1.2) is True
    evidences = {evidence for _, evidence in sim.permits()}
    assert len(evidences) >= 5  # the grid changed under the move
    checked: set[tuple[int, float]] = set()  # (setpoint, evidence of its check)
    for event in sim.log:
        if event[0] in ("vet", "approve") and sim.of("move"):
            evidence = sim.times.lookup(event[3])
            checked.add((sim.index_of(event[1]), evidence))
        elif event[0] == "permit":
            _, _, through, evidence, _, sent = event
            assert sent is not None
            for index in range(sent, through + 1):
                assert (index, evidence) in checked, (index, through, sent)
    quoted = [evidence for _, evidence in sim.permits()]
    assert quoted == sorted(quoted)  # never older than the one before


@apai
def test_the_permit_loop_keeps_its_timings_against_the_real_executor(
    sim: SimRig,
) -> None:
    """Run with -s for the numbers; the bounds are wide for a loaded machine."""
    with sim.camera(0.1):
        assert sim.move(LONG, 1.2) is True
    stats = dict(sim.adapter.last_move)
    gaps = np.diff(sim.permit_times)
    print(
        f"\nmove of 60 setpoints: armed {stats['armed_after_s'] * 1e3:.0f} ms after "
        f"the MOVE left, first PERMIT {stats['first_permit_after_s'] * 1e3:.0f} ms, "
        f"{stats['permits']} permits, gap median {np.median(gaps) * 1e3:.0f} ms "
        f"max {gaps.max() * 1e3:.0f} ms, slowest pass "
        f"{stats['max_pass_s'] * 1e3:.1f} ms"
    )
    # The Thor waits 0.5 s from "armed" for the first PERMIT.
    assert stats["first_permit_after_s"] - stats["armed_after_s"] < 0.25
    assert stats["armed_after_s"] < 0.4
    assert gaps.max() < 0.3  # under the reach of 0.5 s and the life of 1 s
    assert stats["max_pass_s"] < 0.1

    # A refusal mid-move: how soon the arm stands.
    sim.permit_times.clear()

    def switch_in(count: int) -> None:
        if count == 60 + 30:
            sim.obstacle_beyond(-9.0)

    sim.checker.before_approve = switch_in
    with pytest.raises(types().SafetyViolation, match="obstacle ahead"):
        sim.move(np.zeros(6), 1.2)
    stats = dict(sim.adapter.last_move)
    sim.h.idle()
    sent = len(sim.h.sent()) - 60
    print(
        f"cancel: RESULT {stats['cancel_to_result_s'] * 1e3:.0f} ms after the cancel "
        f"PERMIT; the Thor had reported {stats['sent_seen']} setpoints sent and "
        f"sent {sent} in all"
    )
    assert stats["cancel_to_result_s"] < 0.3
    assert sent - stats["sent_seen"] <= 6  # one pass, the link and one setpoint


@apai
def test_a_refusal_of_the_whole_move_sends_nothing_and_clears_nothing(
    sim: SimRig,
) -> None:
    sim.obstacle_beyond(0.05)
    with pytest.raises(checkahead.CheckRefused, match="obstacle ahead") as raised:
        sim.move(SHORT, 0.5)
    assert isinstance(raised.value, types().SafetyViolation)
    assert not sim.of("move") and not sim.of("permit") and not sim.of("approve")
    assert sim.checker.approved_commands == 0 and sim.h.sent() == []
    assert sim.adapter._stopped is False
    # The path is clear again: the next move runs.
    sim.checker.blocked = lambda q: None
    assert sim.move(SHORT, 0.5) is True


@apai
def test_an_old_grid_refuses_the_move_before_anything_is_sent(
    make_sim: Callable[..., SimRig],
) -> None:
    rig = make_sim(start_grid_age_s=0.05, start_grid_wait_s=0.2)
    rig.own()
    rig.new_grid()
    time.sleep(0.1)
    with pytest.raises(types().SafetyViolation, match="no camera frame newer than"):
        rig.adapter.stream_to(np.asarray(SHORT), 0.5, approve=rig.checker.approve)
    assert not rig.of("move") and not rig.of("approve")


@apai
def test_a_refusal_by_the_thor_before_it_armed_reaches_no_callback(
    make_sim: Callable[..., SimRig],
) -> None:
    """Lead's decision D1: such a refusal leaves APAI's counters unchanged."""
    violation = types().SafetyViolation
    # At admission: the move has expired when it arrives.
    late = make_sim(expires_in_s=-1.0)
    late.own()
    with pytest.raises(violation, match="robot app refused the move: the move exp"):
        late.move(SHORT, 0.5)
    assert late.checker.approved_commands == 0
    assert not late.of("approve") and not late.of("permit")
    assert late.h.sent() == [] and late.adapter._stopped is False
    assert late.link.state().owner == "agent"


@apai
def test_a_start_pose_the_thor_does_not_measure_is_refused_before_it_arms(
    sim: SimRig,
) -> None:
    real = sim.link.send_move

    def shifted(move_id: str, start: Any, *rest: Any) -> None:
        real(move_id, np.asarray(start) + 0.006, *rest)  # over the 0.004 rad sent

    sim.link.send_move = shifted
    with pytest.raises(types().SafetyViolation, match="start pose: joint index 0"):
        sim.move(SHORT, 0.5)
    assert sim.checker.approved_commands == 0
    assert not sim.of("approve") and not sim.of("permit") and sim.h.sent() == []


@apai
def test_the_first_step_keeps_room_under_the_thors_speed_limit(
    make_sim: Callable[..., SimRig],
) -> None:
    """The Thor checks setpoint 0 from its own pose, up to the tolerance away."""
    rig = make_sim(agent_max_joint_speed=0.001)
    rig.own()
    with pytest.raises(types().SafetyViolation, match=r"leaves 0\.0000 rad for it"):
        rig.move(SHORT, 0.04)  # two setpoints: the first is half the travel
    assert not rig.of("move") and not rig.of("approve")


@apai
def test_a_move_longer_than_the_thor_takes_is_refused(scripted: ScriptedRig) -> None:
    with pytest.raises(types().SafetyViolation, match=r"3001 setpoints.*at most 3000"):
        scripted.move(SHORT, 60.03)
    assert not scripted.thor.moves and not scripted.of("vet")


@apai
def test_an_obstacle_switched_into_the_path_ends_the_permits_at_once(
    sim: SimRig,
) -> None:
    """The callback refuses mid-move: cancel permit, hold, the agent keeps the arm."""

    def switch_in(count: int) -> None:
        if count == 30:
            sim.obstacle_beyond(0.0)

    sim.checker.before_approve = switch_in
    with pytest.raises(types().SafetyViolation, match="obstacle ahead") as raised:
        sim.move(LONG, 1.2)
    assert type(raised.value) is types().SafetyViolation  # the callback's own
    assert sim.approved() == list(range(31))  # the 31st call refused
    assert sim.checker.approved_commands == 30
    permits = sim.permits()
    assert permits[-1][0] == -1  # the cancel
    assert permits[-1][1] == permits[-2][1]  # the evidence of the PERMIT before
    assert max(through for through, _ in permits) <= 29
    assert [through for through, _ in permits].count(-1) == 1  # nothing after it
    assert 0 < len(sim.h.sent()) < 30  # the arm stopped short of the refusal
    state = sim.link.state()
    assert state.owner == "agent" and state.stop_level == "none"
    assert sim.adapter._stopped is False
    assert sim.adapter.last_move["cancel_to_result_s"] < 0.5
    # A new move from the measured pose runs.
    sim.checker.blocked = lambda q: None
    sim.checker.before_approve = lambda count: None
    assert sim.move(sim.adapter.get_state().q + 0.02, 0.5) is True


@apai
def test_a_new_grid_that_refuses_a_cleared_setpoint_cancels_the_move(
    sim: SimRig,
) -> None:
    reached, release = sim.h.hold_move_after(8)
    release.set()

    def new_obstacle() -> None:
        assert reached.wait(5.0)  # setpoints up to 25 are cleared, 8 are sent
        sim.obstacle_beyond(float(sim.setpoints()[20][0]))
        sim.new_grid()

    thread = in_thread(new_obstacle)
    with pytest.raises(checkahead.CheckRefused) as raised:
        sim.move(LONG, 1.2)
    thread.join(5.0)
    assert raised.value.index == 21 and raised.value.count == 60
    assert "check ahead refused setpoint 21 of 60: obstacle ahead" in str(raised.value)
    assert sim.checker.violations[-1] == "obstacle ahead"
    permits = sim.permits()
    assert permits[-1][0] == -1 and permits[-1][1] == permits[-2][1]
    # The cancel quotes the grid before the obstacle: the Thor takes it.
    assert permits[-1][1] < sim.times.lookup(sim.checker.occupancy._last_refresh)
    sim.h.idle()
    assert len(sim.h.sent()) <= 21  # the arm never reached the refused setpoint
    assert sim.link.state().owner == "agent"


@apai
@pytest.mark.parametrize(
    ("latch", "error", "text"),
    [
        (lambda c: setattr(c, "estopped", True), "SafetyViolation", "e-stop latched"),
        (lambda c: setattr(c, "halted", "wrong cup"), "MotionHalted", "halted: wrong"),
    ],
)
def test_a_latched_or_halted_checker_ends_the_permits_at_once(
    sim: SimRig, latch: Callable[[Checker], None], error: str, text: str
) -> None:
    def later(count: int) -> None:
        if count == 59:  # every setpoint but the last is cleared: nothing to refuse
            latch(sim.checker)

    sim.checker.before_approve = later
    began = time.monotonic()
    with pytest.raises(getattr(types(), error), match=text) as raised:
        sim.move(LONG, 1.2)
    assert type(raised.value) is getattr(types(), error)
    permits = sim.permits()
    assert permits[-1][0] == -1 and permits[-1][1] == permits[-2][1]
    assert len(sim.h.sent()) < 60 and time.monotonic() - began < 1.2 + 0.6
    assert sim.link.state().owner == "agent"
    assert not sim.of("stop")  # a cancel by permit, never a STOP


@apai
def test_a_gate_seen_only_by_the_loop_ends_a_move_with_nothing_left_to_clear(
    sim: SimRig,
) -> None:
    """Asked every pass: near the end of a move no callback is left to notice."""
    done = threading.Event()

    def later(count: int) -> None:
        if count == 59:
            done.set()

    sim.checker.before_approve = later

    def halt() -> None:
        done.wait(5.0)
        time.sleep(0.15)  # the last setpoint is cleared by now
        sim.checker.halted = "look again"

    thread = in_thread(halt)
    with pytest.raises(types().MotionHalted, match="halted: look again"):
        sim.move(LONG, 1.2)
    thread.join(5.0)
    assert sim.approved() == list(range(60))
    assert sim.permits()[-1][0] == -1
    assert len(sim.h.sent()) < 60


@apai
def test_stop_from_another_thread_ends_the_move_and_is_a_persons_stop(
    sim: SimRig,
) -> None:
    """D7: stop() outside a skill is a STOP from "agent_terminal"."""
    reached, release = sim.h.hold_move_after(15)
    release.set()

    def stop() -> None:
        assert reached.wait(5.0)
        sim.adapter.stop()

    thread = in_thread(stop)
    with pytest.raises(types().SafetyViolation) as raised:
        sim.move(LONG, 1.2)
    thread.join(5.0)
    assert "arm stop latched" in str(raised.value) or "stopped the move" in str(
        raised.value
    )
    (stop_event,) = sim.of("stop")
    assert stop_event[2] == AGENT_TERMINAL
    assert len(sim.h.sent()) < 60 and sim.adapter._stopped is True
    until(lambda: sim.link.state().owner == "nobody", "the owner nobody")
    assert sim.link.state().stop_level in ("stop", "none")
    # Everything after it is refused before sending.
    sent = len(sim.of("move"))
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        sim.move(SHORT, 0.5)
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        sim.adapter.set_gripper(3.0)
    assert len(sim.of("move")) == sent and not sim.of("gripper")


@apai
def test_a_skill_stop_sends_no_stop_and_the_agent_keeps_the_arm(sim: SimRig) -> None:
    numbers = sim.link.state().stop_number
    with sim.adapter.skill_scope():
        sim.adapter.stop()
        sim.adapter.stop()
    time.sleep(0.15)
    assert not sim.of("stop")
    state = sim.link.state()
    assert state.owner == "agent" and state.stop_number == numbers
    assert sim.adapter._stopped is True
    with pytest.raises(types().SafetyViolation, match="arm stop latched; no motion"):
        sim.move(SHORT, 0.5)
    assert not sim.of("move")
    # The scope is this thread's: the same call from another thread is a STOP.
    with sim.adapter.skill_scope():
        in_thread(sim.adapter.stop).join(5.0)
    assert [event[2] for event in sim.of("stop")] == [AGENT_TERMINAL]
    # And outside the scope on this thread too.
    sim.adapter.stop()
    assert len(sim.of("stop")) == 2


@apai
def test_stop_never_raises(make_scripted: Callable[..., ScriptedRig]) -> None:
    rig = make_scripted()
    rig.link.close()
    rig.adapter.stop()
    assert rig.adapter._stopped is True
    broken = make_scripted()
    broken.link.stop = mock.Mock(side_effect=RuntimeError("no socket"))
    broken.adapter.stop()
    assert broken.adapter._stopped is True


@apai
def test_a_stop_at_the_thor_ends_the_move_and_latches_the_adapter(sim: SimRig) -> None:
    """The mirror: ownership lost shows only in ROBOT_STATE."""
    reached, release = sim.h.hold_move_after(15)
    release.set()

    def operator() -> None:
        assert reached.wait(5.0)
        sim.h.app.stop("operator", "thor_terminal")

    thread = in_thread(operator)
    with pytest.raises(types().MotionHalted) as raised:
        sim.move(LONG, 1.2)
    thread.join(5.0)
    assert "the robot app stopped the move" in str(
        raised.value
    ) or "no longer owns the arm" in str(raised.value)
    assert sim.adapter._stopped is True and not sim.of("stop")
    assert len(sim.h.sent()) < 60


@apai
def test_old_numbers_are_never_requoted(sim: SimRig) -> None:
    """After a STOP and a new owner the adapter sends nothing with fresh numbers."""
    old = sim.link.pinned
    sim.h.app.stop("operator", "thor_terminal")
    until(lambda: sim.link.state().owner == "nobody", "the STOP")
    again = sim.link.request_owner("agent", timed_return=False)
    assert again.status == "completed", again.reason
    until(lambda: sim.link.state().owner == "agent", "the new owner")
    new = sim.link.state()
    assert (new.owner_number, new.stop_number) != (old.owner_number, old.stop_number)
    for _ in range(2):
        with pytest.raises(types().SafetyViolation):
            sim.move(SHORT, 0.5)
        with pytest.raises(types().SafetyViolation):
            sim.adapter.set_gripper(3.0)
        with pytest.raises(types().SafetyViolation):
            sim.adapter.close_gripper_two_stage()
        sim.checker.clear_contact_zones()
        # Even with the adapter's own latch cleared by hand.
        sim.adapter._stopped = False
    assert sim.link.pinned == old
    assert not sim.of("move") and not sim.of("gripper") and not sim.of("permit")
    assert sim.h.sent() == []
    # The hook's CHECKER_STATE went out with the old numbers and was refused.
    for event in sim.of("checker_state"):
        assert event[1] == "clear_contact_spots"
    # Only a new task takes the new numbers.
    sim.adapter.close_task()
    pinned = Pinned(new.session_id, new.owner_number, new.stop_number)
    sim.adapter.open_task(pinned)
    assert sim.link.pinned == pinned != old
    assert sim.move(SHORT, 0.5) is True
    move = sim.of("move")[-1]
    assert sim.h.sent() == [np.asarray(row).tolist() for row in move[4]]


@apai
def test_a_lost_link_ends_the_move_and_the_arm_holds(sim: SimRig) -> None:
    reached, release = sim.h.hold_move_after(15)
    release.set()

    def cut() -> None:
        assert reached.wait(5.0)
        sim.robot_end.break_link()

    thread = in_thread(cut)
    with pytest.raises(RobotLinkError, match="robot link service is gone") as raised:
        sim.move(LONG, 1.2)
    thread.join(5.0)
    assert not isinstance(raised.value, types().SafetyViolation)
    assert sim.adapter._stopped is True
    sim.h.idle()
    assert len(sim.h.sent()) < 60  # the robot app ended the move by itself
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        sim.move(SHORT, 0.5)


@apai
def test_a_typeerror_after_the_move_left_is_not_one_safearm_retries_on(
    sim: SimRig,
) -> None:
    def broken(count: int) -> None:
        if count == 30:
            raise TypeError("a bug in a callback")

    sim.checker.before_approve = broken
    monitor = mock.Mock(started=False, label="move_joints")
    with pytest.raises(RobotLinkError, match="a bug in a callback") as raised:
        sim.move(LONG, 1.2, contact_monitor=monitor)
    assert not isinstance(raised.value, TypeError)
    assert monitor.started is True  # set before the MOVE left
    assert sim.permits()[-1][0] == -1 and len(sim.h.sent()) < 60
    assert len(sim.of("move")) == 1


@apai
def test_a_signal_on_the_skill_thread_cancels_the_move_and_goes_on_up(
    sim: SimRig,
) -> None:
    class SignalRequest(BaseException):  # as apai.apps.signal_stop's
        pass

    signal = SignalRequest("ctrl-c")

    def interrupt(count: int) -> None:
        if count == 30:
            raise signal

    sim.checker.before_approve = interrupt
    with pytest.raises(SignalRequest) as raised:
        sim.move(LONG, 1.2)
    assert raised.value is signal
    permits = sim.permits()
    assert permits[-1][0] == -1 and permits[-1][1] == permits[-2][1]
    sim.h.idle()
    assert len(sim.h.sent()) < 60
    assert sim.link.state().owner == "agent"  # the STOP is APAI's next call


@apai
def test_a_contact_on_the_thor_reaches_apai_as_contact(sim: SimRig) -> None:
    contact = importlib.import_module("apai.safety.contact")
    monitor = contact.ContactMonitor(
        contact.ContactMonitorConfig(enabled=True), label="move_joints"
    )
    sim.h.arm.block_joint(0, 0.05)  # something stops the first joint
    with pytest.raises(types().ContactDetected) as raised:
        sim.move([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], 2.4, contact_monitor=monitor)
    text = str(raised.value)
    assert text.startswith("contact detected by the robot app: contact: joint index 0")
    event = raised.value.event
    assert event is monitor.tripped
    # The simulated arm's text is not APAI's: the stop still arrives.
    assert event["joint"] == "unknown" and event["parsed"] is False
    assert event["phase"] == "stream" and event["motion"] == "move_joints"
    assert event["measured_q"] == pytest.approx(event["hold_q"])
    assert event["hold_q"][0] == pytest.approx(0.05, abs=0.02)
    assert monitor.started is True and monitor.commands == len(sim.h.sent()) > 0
    assert monitor.hold == "measured pose (reported by the robot app)"
    summary = monitor.summary()  # what record_contact_motion formats
    assert summary["tripped"] == "joint unknown tracking error 0.000 rad (stream)"
    assert sim.link.state().owner == "agent" and sim.adapter._stopped is False


# The final RESULT of a MOVE, on a scripted robot side


@apai
def test_a_completed_move_returns_true_and_one_not_settled_false(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor
    thor.at_first_permit = ("completed", "", {"outcome": "completed", "sent": 25}, [])
    assert scripted.move(SHORT, 0.5) is True
    thor.at_first_permit = (
        "failed",
        "joint index 0 is not within 0.03 rad of the last setpoint after 2 s",
        {"outcome": "not_settled", "sent": 25, "total": 25},
        None,
    )
    assert scripted.move(SHORT, 0.5) is False
    assert len(thor.moves) == 2 and scripted.adapter._stopped is False
    move = thor.moves[0]
    assert (move.session_id, move.owner_number, move.stop_number) == ("s1", 1, 0)
    assert move.period_s == PERIOD_S and move.start_tol_rad == pytest.approx(0.004)
    assert move.grasp_region is None
    assert move.expires_mono_s == pytest.approx(time.monotonic() + 2.0, abs=1.0)
    first = next(p for p in thor.permits if p.move_id == move.command_id)
    assert (first.session_id, first.owner_number, first.stop_number) == ("s1", 1, 0)
    assert first.through_index == 24
    assert first.expires_mono_s == pytest.approx(time.monotonic() + 1.0, abs=1.0)


@apai
@pytest.mark.parametrize(
    ("ending", "error", "text", "latched"),
    [
        (
            ("failed", "setpoint 7 is 0.120 s late", {"outcome": "late", "sent": 7}),
            "SafetyViolation",
            "ended the move late at setpoint 7 of 25: setpoint 7 is 0.120 s late",
            False,
        ),
        (
            ("failed", "fault: joint feedback lost: joint index 2", {"sent": 9}),
            "RobotLinkError",
            "fault: joint feedback lost",
            True,
        ),
        (
            ("failed", "robot app error in a move: KeyError('x')", {}),
            "RobotLinkError",
            "robot app error in a move",
            True,
        ),
        (("failed", LOST, {}), "RobotLinkError", LOST, True),
        (("refused", LOST, {}), "RobotLinkError", LOST, True),
        (
            (
                "refused",
                "permit expired; its last permit was refused: permit expired",
                {"outcome": "refused", "sent": 12, "total": 25},
            ),
            "SafetyViolation",
            "the robot app ended the move at setpoint 12 of 25: permit expired",
            False,
        ),
        (
            ("refused", "no permit for this move (waited 0.5 s)", {}),
            "SafetyViolation",
            "ended the move at setpoint 0 of 25: no permit for this move",
            False,
        ),
        (
            ("cancelled", "help requested", {"outcome": "stopped", "sent": 4}),
            "MotionHalted",
            "the robot app stopped the move: help requested",
            False,
        ),
        (("failed", "something new", {"outcome": "new"}), "RobotLinkError", "new", 0),
    ],
)
def test_the_final_result_of_a_move_becomes_what_apai_expects(
    scripted: ScriptedRig,
    caplog: pytest.LogCaptureFixture,
    ending: tuple,
    error: str,
    text: str,
    latched: bool,
) -> None:
    scripted.thor.at_first_permit = (*ending, None)
    kind = RobotLinkError if error == "RobotLinkError" else getattr(types(), error)
    with caplog.at_level(logging.WARNING), pytest.raises(kind, match=text) as raised:
        scripted.move(SHORT, 0.5)
    assert type(raised.value) is kind
    assert scripted.adapter._stopped is bool(latched)
    assert [p.through_index for p in scripted.thor.permits if p.through_index < 0] == []
    warned = "refused after the first permit with nothing sent" in caplog.text
    assert warned is ("waited 0.5 s" in text or "no permit for this move" in text)


@apai
def test_a_cancelled_move_latches_the_adapter_when_the_ownership_is_gone(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor

    def stopped() -> None:
        thor.fields.update(owner="nobody", stop_number=1, stop_level="stop")

    thor.at_first_permit = ("cancelled", "stop: operator", {"sent": 2}, None)
    real = thor.end_move
    thor.end_move = lambda *ending: (stopped(), thor._publish(), real(*ending))
    with pytest.raises(types().MotionHalted, match="stopped the move: stop: operator"):
        scripted.move(SHORT, 0.5)
    assert scripted.adapter._stopped is True


@apai
def test_apais_contact_text_is_parsed_into_the_monitors_record(
    scripted: ScriptedRig,
) -> None:
    contact = importlib.import_module("apai.safety.contact")
    monitor = contact.ContactMonitor(
        contact.ContactMonitorConfig(enabled=True), label="move_joints"
    )
    text = (
        "contact detected at joint 3: tracking error 0.214 rad for 85 ms -- "
        "something is blocking the arm"
    )
    end_q = [0.01, 0.0, 0.2, 0.0, 0.0, 0.0]
    scripted.thor.at_first_permit = (
        "failed",
        "contact",
        {"outcome": "contact", "sent": 11, "total": 25, "contact": text},
        end_q,
    )
    with pytest.raises(types().ContactDetected) as raised:
        scripted.move(SHORT, 0.5, contact_monitor=monitor)
    assert str(raised.value) == text
    assert monitor.tripped == {
        "joint": 3,
        "error_rad": 0.214,
        "duration_ms": 85.0,
        "phase": "stream",
        "motion": "move_joints",
        "measured_q": end_q,
        "hold_q": end_q,
    }
    assert raised.value.event is monitor.tripped
    assert monitor.commands == 11
    assert monitor.summary()["tripped"] == "joint 3 tracking error 0.214 rad (stream)"
    # A hold that failed: no hold pose is claimed. No monitor: raised the same.
    scripted.thor.at_first_permit = (
        "failed",
        "contact; the hold at the measured pose failed: read failed; the last "
        "setpoint stands",
        {"outcome": "contact", "sent": 11, "contact": text},
        end_q,
    )
    with pytest.raises(types().ContactDetected) as again:
        scripted.move(SHORT, 0.5)
    assert "hold_q" not in again.value.event
    assert again.value.event["measured_q"] == end_q


@apai
def test_an_early_end_without_contact_is_marked_on_the_monitor(
    scripted: ScriptedRig,
) -> None:
    monitor = mock.Mock(started=False, label="move_joints", hold=None, tripped=None)
    scripted.thor.at_first_permit = (
        "refused",
        "permit expired",
        {"outcome": "refused", "sent": 5},
        None,
    )
    with pytest.raises(types().SafetyViolation):
        scripted.move(SHORT, 0.5, contact_monitor=monitor)
    assert monitor.hold == "last command (stream aborted)"
    assert monitor.commands == 5 and monitor.tripped is None


@apai
@pytest.mark.parametrize(
    ("admit", "error", "text", "latched"),
    [
        (
            Result("", "refused", "busy: a move is running"),
            "SafetyViolation",
            "robot app refused the move: busy: a move is running",
            False,
        ),
        (Result("", "refused", LOST), "RobotLinkError", LOST, True),
        (
            Result("", "cancelled", "stop: operator"),
            "MotionHalted",
            "the robot app stopped the move: stop: operator",
            False,
        ),
    ],
)
def test_a_move_that_ends_at_admission_reaches_no_callback(
    scripted: ScriptedRig, admit: Result, error: str, text: str, latched: bool
) -> None:
    scripted.thor.admit = admit
    kind = RobotLinkError if error == "RobotLinkError" else getattr(types(), error)
    with pytest.raises(kind, match=text):
        scripted.move(SHORT, 0.5)
    assert not scripted.of("approve") and not scripted.thor.permits
    assert scripted.adapter._stopped is latched


@apai
def test_a_move_the_thor_never_arms_gets_no_permit(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted(arm_wait_s=0.3)
    rig.own()
    rig.thor.arming = False
    began = time.monotonic()
    with pytest.raises(RobotLinkError, match="did not arm the move"):
        rig.move(SHORT, 0.5)
    assert 0.3 <= time.monotonic() - began < 1.5
    assert not rig.of("approve") and not rig.thor.permits
    # A newest-wins state can skip "armed": "sending" counts as armed too.
    rig.thor.arming = True
    real = rig.thor._take

    def sending(message: Any) -> None:
        real(message)
        if isinstance(message, Move):
            rig.thor._stage = MOVE_SENDING

    rig.thor._take = sending
    rig.thor.at_first_permit = ("completed", "", {"sent": 25}, None)
    assert rig.move(SHORT, 0.5) is True


@apai
def test_no_answer_to_a_move_is_a_fault_of_the_link(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted(admit_wait_s=0.2)
    rig.own()
    rig.thor._take = lambda message: rig.thor.others.append(message)  # says nothing
    with pytest.raises(RobotLinkError, match=r"no RESULT for the MOVE within 0\.2 s"):
        rig.move(SHORT, 0.5)
    assert not rig.of("approve") and not rig.of("permit")


@apai
def test_a_stale_state_ends_the_permits_at_once(scripted: ScriptedRig) -> None:
    thor = scripted.thor
    thor.on_cancel = None  # the robot side has gone quiet
    silent = threading.Event()

    def quiet() -> None:
        until(lambda: len(thor.permits) >= 3, "three permits")
        thor.publishing = False
        silent.set()

    thread = in_thread(quiet)
    began = time.monotonic()
    with pytest.raises(RobotLinkError, match=r"ROBOT_STATE is .* s old") as raised:
        scripted.move(SHORT, 0.5)
    thread.join(5.0)
    assert "the permits were stopped" in str(raised.value)
    assert scripted.adapter._stopped is True
    # Permits went on for state_max_age_s and no longer, then the cancel.
    last, cancel = thor.permits[-2], thor.permits[-1]
    assert cancel.through_index == -1
    assert cancel.evidence_wall_s == last.evidence_wall_s
    assert time.monotonic() - began < 0.5 + 0.5 + scripted.settings.cancel_wait_s + 1.0


@apai
def test_lost_ownership_ends_the_permits_at_once_with_a_cancel(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor

    def takeover() -> None:
        until(lambda: len(thor.permits) >= 3, "three permits")
        thor.fields.update(owner="human", owner_number=2)

    thread = in_thread(takeover)
    with pytest.raises(types().MotionHalted, match="control owner is human") as raised:
        scripted.move(SHORT, 0.5)
    thread.join(5.0)
    assert "the agent no longer owns the arm" in str(raised.value)
    last, cancel = thor.permits[-2], thor.permits[-1]
    assert cancel.through_index == -1 and last.through_index >= 0
    assert cancel.evidence_wall_s == last.evidence_wall_s
    assert (cancel.owner_number, cancel.stop_number) == (1, 0)  # the pinned ones
    assert scripted.adapter._stopped is True and thor.stops == []


@apai
def test_the_stop_flag_ends_the_permits_at_once_with_a_cancel(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor

    def stop() -> None:
        until(lambda: len(thor.permits) >= 3, "three permits")
        scripted.adapter.stop()

    thread = in_thread(stop)
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        scripted.move(SHORT, 0.5)
    thread.join(5.0)
    last, cancel = thor.permits[-2], thor.permits[-1]
    assert cancel.through_index == -1 and last.through_index >= 0
    assert cancel.evidence_wall_s == last.evidence_wall_s
    until(lambda: len(thor.stops) == 1, "the STOP")
    assert thor.stops[0].sender == "agent_terminal" and thor.stops[0].reason


@apai
def test_a_lost_connection_in_a_result_is_treated_as_a_lost_link(
    scripted: ScriptedRig,
) -> None:
    """The service's own RESULT with reason LOST, mid-move."""
    thor = scripted.thor

    def lose() -> None:
        until(lambda: len(thor.permits) >= 2, "two permits")
        thor.end_move("failed", LOST)

    thread = in_thread(lose)
    with pytest.raises(RobotLinkError, match=LOST):
        scripted.move(SHORT, 0.5)
    thread.join(5.0)
    assert scripted.adapter._stopped is True
    assert scripted.link.lost_reason() is not None
    with pytest.raises(types().SafetyViolation, match="arm stop latched"):
        scripted.move(SHORT, 0.5)


@apai
def test_a_camera_frame_time_that_goes_back_ends_the_permits(
    scripted: ScriptedRig,
) -> None:
    """The Thor ignores a PERMIT on older evidence, and a cancel after it."""
    thor = scripted.thor

    def older() -> None:
        until(lambda: len(thor.permits) >= 2, "two permits")
        grid = scripted.checker.occupancy
        frame_t = time.monotonic()
        scripted.times.note(frame_t, thor.permits[0].evidence_wall_s - 0.5)
        with grid._refresh_lock:
            grid._last_refresh = frame_t
            grid.refreshes += 1

    thread = in_thread(older)
    with pytest.raises(types().SafetyViolation, match="camera-frame time went back"):
        scripted.move(SHORT, 0.5)
    thread.join(5.0)
    evidences = [p.evidence_wall_s for p in thor.permits]
    assert evidences == sorted(evidences) and thor.permits[-1].through_index == -1
    assert thor.permits[-1].evidence_wall_s == thor.permits[-2].evidence_wall_s


@apai
def test_a_move_without_a_result_ends_at_its_deadline(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted(first_wait_s=0.0, settle_timeout_s=0.0, cancel_wait_s=0.2)
    rig.own()
    rig.thor.on_cancel = None
    began = time.monotonic()
    with pytest.raises(RobotLinkError, match="no RESULT for the move"):
        rig.move([0.004, 0, 0, 0, 0, 0], 0.04)  # 2 setpoints: 3.044 s to the deadline
    assert 3.0 < time.monotonic() - began < 4.5
    assert rig.adapter._stopped is True
    assert rig.thor.permits[-1].through_index == -1


@apai
def test_a_refusal_before_any_permit_sends_no_cancel(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    """Nothing to cancel: the Thor refuses the move when its first wait ends."""
    rig = make_scripted(cancel_wait_s=0.3)
    rig.own()
    rig.checker.before_approve = lambda count: rig.obstacle_beyond(-1.0)
    with pytest.raises(types().SafetyViolation, match="obstacle ahead"):
        rig.move(SHORT, 0.5)
    assert rig.thor.permits == [] and len(rig.thor.moves) == 1
    assert rig.checker.approved_commands == 0


@apai
def test_the_grasp_region_of_the_checker_goes_with_the_move(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor
    thor.at_first_permit = ("completed", "", {"sent": 25}, None)
    scripted.checker._grasp_exempt = (np.array([0.31, -0.02, 9.0]), 0.07, -0.06)
    assert scripted.move(SHORT, 0.5) is True
    assert thor.moves[-1].grasp_region == messages.GraspRegion(
        (0.31, -0.02), 0.07, -0.06, None
    )
    scripted.checker._grasp_exempt = (np.array([0.31, -0.02]), 0.07, -0.06, 0.12)
    assert scripted.move(SHORT, 0.5) is True
    assert thor.moves[-1].grasp_region.z_max == 0.12


# CHECKER_STATE


@apai
def test_closing_the_grasp_region_is_told_to_the_thor_once_per_region(
    scripted: ScriptedRig,
) -> None:
    thor, checker = scripted.thor, scripted.checker
    thor.at_first_permit = ("completed", "", {"sent": 25}, None)
    checker.clear_grasp_exemption()  # no MOVE of this task carried a region
    assert checker.cleared == ["grasp"]
    checker._grasp_exempt = (np.array([0.3, 0.0]), 0.07, -0.06)
    assert scripted.move(SHORT, 0.5) is True
    assert checker.clear_grasp_exemption() is None
    assert checker._grasp_exempt is None  # APAI's own method ran first
    until(lambda: len(thor.checker_states) == 1, "the CHECKER_STATE")
    sent = thor.checker_states[0]
    assert sent.action == "close_grasp"
    assert (sent.session_id, sent.owner_number, sent.stop_number) == ("s1", 1, 0)
    checker.clear_grasp_exemption()  # nothing new to close
    time.sleep(0.1)
    assert len(thor.checker_states) == 1


@apai
def test_clearing_the_contact_spots_is_told_to_the_thor_inside_a_task(
    scripted: ScriptedRig, caplog: pytest.LogCaptureFixture
) -> None:
    thor, checker = scripted.thor, scripted.checker
    assert checker.clear_contact_zones("reset_scene") == 2  # APAI's own answer
    assert checker.cleared == ["contact:reset_scene"]
    until(lambda: len(thor.checker_states) == 1, "the CHECKER_STATE")
    assert thor.checker_states[0].action == "clear_contact_spots"
    # Binding again does not send it twice.
    scripted.adapter.bind_checker(scripted.check)
    checker.clear_contact_zones()
    until(lambda: len(thor.checker_states) == 2, "the second CHECKER_STATE")
    time.sleep(0.1)
    assert len(thor.checker_states) == 2
    # Outside a task nothing is sent, and a failing send is never raised.
    scripted.adapter.close_task()
    checker.clear_contact_zones()
    checker.clear_grasp_exemption()
    scripted.adapter.open_task(ScriptedRig.PIN)
    scripted.link.close()
    with caplog.at_level(logging.WARNING):
        assert checker.clear_contact_zones() == 2
    assert "CHECKER_STATE clear_contact_spots was not sent" in caplog.text
    assert len(thor.checker_states) == 2


@apai
def test_a_refused_checker_state_is_logged_and_never_raised(
    scripted: ScriptedRig, caplog: pytest.LogCaptureFixture
) -> None:
    thor = scripted.thor
    real = thor._take

    def refuse(message: Any) -> None:
        if isinstance(message, CheckerState):
            thor.checker_states.append(message)
            thor.thor.send(Result(message.command_id, "refused", "owner number 1"))
        else:
            real(message)

    thor._take = refuse
    scripted.checker.clear_contact_zones()
    until(lambda: len(thor.checker_states) == 1, "the CHECKER_STATE")
    time.sleep(0.1)
    with caplog.at_level(logging.WARNING):
        scripted.adapter.close_task()
    assert "CHECKER_STATE clear_contact_spots with refused: owner number 1" in (
        caplog.text
    )


@apai
def test_the_hooks_reach_the_real_robot_app(sim: SimRig) -> None:
    sim.checker._grasp_exempt = (np.array([0.3, 0.0]), 0.07, -0.02)
    assert sim.move(SHORT, 0.5) is True
    region = sim.of("move")[0][5]
    assert region == messages.GraspRegion((0.3, 0.0), 0.07, -0.02, None)
    sim.checker.clear_grasp_exemption()
    sim.checker.clear_contact_zones("reset_scene")
    ids = [event[1] for event in sim.of("checker_state")]
    assert ids == ["close_grasp", "clear_contact_spots"]
    waiting = [command_id for _, command_id in sim.adapter._checker_ids]
    for command_id in waiting:
        assert sim.link.wait_final(command_id, 5.0).status == "completed"


# Jaws


@apai
def test_jaw_commands_reach_the_thor_in_its_units(sim: SimRig) -> None:
    adapter = sim.adapter
    assert adapter.set_gripper(3.1, effort=0.4) is None  # half of 6.2 rad
    _, opening, effort, mode, timeout_s, _ = sim.of("gripper")[0]
    assert opening == pytest.approx(0.5) and (effort, mode) == (0.4, "plain")
    assert timeout_s == 12.0
    # The next state read already shows the reading the command ended with.
    state = adapter.get_state()
    assert state.gripper_valid is True
    assert state.gripper_pos == pytest.approx(3.1, abs=0.2)
    # Clamped to the travel and to an effort of 0..1.
    adapter.set_gripper(9.0, effort=3.0)
    assert sim.of("gripper")[-1][1:3] == (1.0, 1.0)
    adapter.set_gripper(-1.0, effort=-1.0)
    assert sim.of("gripper")[-1][1:3] == (0.0, 0.0)


@apai
def test_the_grasp_close_stalls_on_the_object_and_never_opens(sim: SimRig) -> None:
    adapter = sim.adapter
    adapter.set_gripper(6.2)
    sim.h.arm.jaw_object(0.5)
    measured = adapter.close_gripper_two_stage(
        width_frac_stage1=0.5, width_frac_stage2=0.85, effort=0.6
    )
    _, opening, effort, mode, _, _ = sim.of("gripper")[-1]
    assert opening == pytest.approx(0.15) and effort == 0.6
    assert mode == "close_until_contact"
    assert measured == pytest.approx(3.1, abs=0.3)  # stalled on the object
    assert adapter.close_stages_skipped == 0
    assert sim.h.arm.jaw_stiffness == sim_arm.LIGHT  # not at the closing stiffness
    # A second close that is not more closed sends nothing.
    sent = len(sim.of("gripper"))
    assert adapter.close_gripper_two_stage(width_frac_stage2=0.85) is None
    assert adapter.close_gripper_two_stage(width_frac_stage2=0.5) is None
    assert adapter.close_stages_skipped == 2 and len(sim.of("gripper")) == sent
    # A plain close on the object ends short: None, as APAI's driver returns.
    adapter.set_gripper(6.2)
    assert adapter.set_gripper(0.0) is None


def jaw_result(status: str, reason: str = "", **detail: Any) -> Callable:
    return lambda m: Result(m.command_id, status, reason, None, detail)


@apai
@pytest.mark.parametrize(
    ("reply", "error", "text", "latched"),
    [
        (
            jaw_result("failed", "the jaws stalled short", outcome="stalled"),
            "GripperFault",
            "the jaws were blocked while opening",
            False,
        ),
        (
            jaw_result("failed", "the jaw command timed out", outcome="timeout"),
            "GripperFault",
            "the jaws were blocked while opening",
            False,
        ),
        (
            jaw_result("refused", "the jaws are not referenced", outcome="refused"),
            "GripperFault",
            "the jaws are not referenced",
            False,
        ),
        (
            jaw_result(
                "refused",
                "the jaws could not be read, so no jaw command was sent: timeout",
            ),
            "GripperFault",
            "could not be read",
            False,
        ),
        (
            jaw_result("refused", "busy: a return to rest is running"),
            "SafetyViolation",
            "robot app refused the jaw command: busy: a return to rest",
            False,
        ),
        (
            jaw_result("cancelled", "dropped by a stop", outcome="cancelled"),
            "MotionHalted",
            "the robot app stopped the jaw command: dropped by a stop",
            False,
        ),
        (
            jaw_result("failed", "fault: gripper fault: over temperature"),
            "RobotLinkError",
            "fault: gripper fault",
            True,
        ),
        (jaw_result("failed", LOST), "RobotLinkError", LOST, True),
        (
            jaw_result("failed", "jaw command error: KeyError()", outcome="failed"),
            "RobotLinkError",
            "the jaw command failed: jaw command error",
            True,
        ),
    ],
)
def test_the_result_of_a_jaw_command_becomes_what_apai_expects(
    scripted: ScriptedRig, reply: Callable, error: str, text: str, latched: bool
) -> None:
    scripted.thor.jaw_reply = reply
    scripted.thor.jaws = GripperState(0.2, None, False, True, 0.2, False, "empty")
    until(lambda: scripted.link.state().gripper.last_target == 0.2, "the jaws")
    kind = RobotLinkError if error == "RobotLinkError" else getattr(types(), error)
    with pytest.raises(kind, match=text) as raised:
        scripted.adapter.set_gripper(6.2)  # an opening command
    assert type(raised.value) is kind
    assert scripted.adapter._stopped is latched
    if error == "GripperFault":
        assert "GripperFault" in repr(raised.value)  # what APAI's result code reads


@apai
def test_a_close_that_ends_short_is_none_and_a_grasp_close_that_times_out_a_fault(
    scripted: ScriptedRig,
) -> None:
    thor = scripted.thor
    thor.jaw_reply = jaw_result("failed", "the jaws stalled short", outcome="stalled")
    assert scripted.adapter.set_gripper(0.0) is None  # a closing command
    thor.jaw_reply = jaw_result(
        "failed", "the jaw command timed out", outcome="timeout"
    )
    with pytest.raises(types().GripperFault, match="the jaw command timed out"):
        scripted.adapter.close_gripper_two_stage()
    thor.jaw_reply = jaw_result("completed", "", outcome="stalled", opening=None)
    assert scripted.adapter.close_gripper_two_stage() is None
    assert scripted.adapter.close_stages_skipped == 0
    assert [g.mode for g in thor.grippers] == ["plain"] + ["close_until_contact"] * 2
    assert all(g.opening == pytest.approx(0.15) for g in thor.grippers[1:])


@apai
def test_no_answer_to_a_jaw_command_latches_the_adapter(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    rig = make_scripted(result_timeout_s=0.3)
    rig.own()
    rig.thor._take = lambda message: rig.thor.others.append(message)
    with pytest.raises(RobotLinkError, match="no RESULT for the jaw command within"):
        rig.adapter.set_gripper(3.0)
    assert rig.adapter._stopped is True


# Under APAI's own SafeArm and SafetyHarness


def robot_link_scene(folder: Path) -> Any:
    """APAI's mock scene with the arm of type robot_link and a required grid."""
    config = importlib.import_module("apai.config")
    loaded = config.load_demo_config(camera="mock", arm="mock", llm="mock")
    data = loaded.as_dict()
    data["memory"]["experience_path"] = str(folder / "experience.json")
    views = [data, *(arm["resolved"] for arm in data["arms"] if "resolved" in arm)]
    for view in views:
        view["occupancy"]["enabled"] = True
        view["occupancy"]["required"] = True
        view["arm"]["type"] = "robot_link"
        # SafeArm then hands every move a contact monitor.
        view["safety"]["contact_monitor"] = {"enabled": True}
    for arm in data["arms"]:
        arm["type"] = "robot_link"
    return type(loaded)(data)


class EveryFrame(FrameTimes):
    """Evidence for any camera frame: its time moved onto the wall clock.

    APAI's mock camera stamps its frames on this machine's monotonic clock,
    and in these tests the Thor's clocks are this machine's.
    """

    def lookup(self, frame_t: float) -> float | None:
        known = super().lookup(frame_t)
        if known is None:  # one evidence time per camera frame, kept
            known = time.time() - (time.monotonic() - float(frame_t))
            self.note(frame_t, known)
        return known


class ApaiRig:
    """The adapter built by APAI's own runtime, on the robot app with APAI's arm."""

    def __init__(self, harness: Any, folder: Path) -> None:
        demo = importlib.import_module("apai.apps.demo")
        occupancy = importlib.import_module("apai.perception.occupancy")
        self.h = harness
        self.robot_end, self.agent_end = loopback_pair()
        self.link = AgentLink(self.agent_end)
        assert harness.app.attach_carrier(self.robot_end)
        settings = ArmSettings(
            expected_profile_id="apai-rs-model",
            calibration_id="apai-mock-calibration",
        )
        self.make_arm = arm_factory(self.link, settings)
        self.moves: list[tuple] = []
        self.stops: list[tuple] = []
        for name, kept in (("send_move", self.moves), ("stop", self.stops)):
            setattr(self.link, name, self._spy(getattr(self.link, name), kept))
        with (
            mock.patch.object(occupancy, "OccupancyClient", FakeGridClient),
            mock.patch.object(demo, "make_arm", self.make_arm),
        ):
            self.runtime, self.raw_arm = demo.build_runtime(
                robot_link_scene(folder), folder / "run"
            )
        self.safe = self.runtime.arm
        self.adapter = self.safe.raw
        self.harness = self.safe.harness
        assert self.harness.occupancy.wait_until_usable(10.0)
        self.adapter.bind_checker(
            CheckAhead(self.harness, EveryFrame(), period_s=PERIOD_S)
        )

    @staticmethod
    def _spy(real: Callable, kept: list[tuple]) -> Callable:
        def spied(*args: Any, **kwargs: Any) -> Any:
            kept.append(args)
            return real(*args, **kwargs)

        return spied

    def own(self) -> None:
        result = self.link.request_owner("agent", timed_return=False)
        assert result.status == "completed", result.reason
        until(lambda: self.link.state().owner == "agent", "the agent as owner")
        state = self.link.state()
        self.adapter.open_task(
            Pinned(state.session_id, state.owner_number, state.stop_number)
        )

    def close(self) -> None:
        demo = importlib.import_module("apai.apps.demo")
        try:
            demo.shutdown_runtime(self.runtime, self.raw_arm)
        finally:
            self.link.close()
            self.robot_end.close()


@pytest.fixture
def apai_rig(
    make_realtime_apai: Callable[..., Any], tmp_path: Path
) -> Iterator[ApaiRig]:
    import test_apai_app

    from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState

    # Started at rest, so the jaws are referenced; then put off the joint
    # stops, holding, where APAI's checker lets the arm move.
    harness = make_realtime_apai(connected=False)
    harness.spy.backend._q[:] = test_apai_app.RAISED
    harness.app._motion.set(MotionState.HOLD)
    with apai_runtime_env(tmp_path) as folder:
        rig = ApaiRig(harness, folder)
        try:
            yield rig
        finally:
            rig.close()


def nearby(rig: ApaiRig, joint: int = 0, by: float = 0.05) -> np.ndarray:
    q = np.asarray(rig.safe.get_state().q, dtype=float).copy()
    q[joint] += by
    return q


@apai
def test_apai_builds_the_adapter_and_moves_the_arm_through_it(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    arm_base = importlib.import_module("apai.control.arm_base")
    assert isinstance(rig.adapter, RobotLinkArm)
    assert isinstance(rig.adapter, arm_base.ArmBase)
    assert rig.adapter.connected is True and rig.make_arm.made == [rig.adapter]
    rig.own()
    goal = nearby(rig)
    before = rig.harness.approved_commands
    begun = rig.harness.motions_begun
    assert rig.safe.move_joints(goal, duration_s=0.5) is True
    (move,) = rig.moves
    count = len(move[3])
    assert count >= 25  # SafeArm stretched the move under its speed cap
    # APAI's counters: one per setpoint the callback cleared, one motion begun.
    assert rig.harness.approved_commands == before + count
    assert rig.harness.motions_begun == begun + 1
    assert rig.safe.get_state().q == pytest.approx(goal, abs=0.03)
    # The motors got exactly the setpoints of the MOVE, in order.
    sent = [np.asarray(q) for q in rig.h.spy.targets[-count:]]
    for target, row in zip(sent, move[3]):
        assert target == pytest.approx(np.asarray(row))
    assert rig.adapter.last_move["approved"] == count


@apai
def test_a_refusal_before_the_first_permit_leaves_apais_counters_unchanged(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    rig.own()
    skills = importlib.import_module("apai.skills.runtime")
    before = rig.harness.approved_commands
    # The whole-move check: a goal outside the workstation checker's limits.
    with pytest.raises(types().SafetyViolation) as whole:
        rig.safe.move_joints(nearby(rig, by=3.0), duration_s=1.0)
    assert rig.moves == []
    # The Thor, before it armed the move: the start pose it measures differs.
    real = rig.link.send_move

    def shifted(move_id: str, start: Any, *rest: Any) -> None:
        real(move_id, np.asarray(start) + 0.006, *rest)

    rig.link.send_move = shifted
    with pytest.raises(types().SafetyViolation, match="start pose") as unarmed:
        rig.safe.move_joints(nearby(rig), duration_s=0.5)
    assert rig.harness.approved_commands == before
    # What APAI's skills ask before they stop and latch.
    reader = mock.Mock(arm=rig.safe)
    for raised in (whole.value, unarmed.value):
        assert skills.SkillRuntime._refused_before_motion(reader, raised, before)
    assert rig.harness.estopped is False and rig.adapter._stopped is False
    rig.link.send_move = real
    assert rig.safe.move_joints(nearby(rig), duration_s=0.5) is True


@apai
def test_a_hint_the_adapter_does_not_take_is_dropped_and_the_move_runs_once(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    rig.own()
    assert rig.safe.move_joints(
        nearby(rig), duration_s=0.5, bias_compensate=np.zeros(6)
    )
    assert len(rig.moves) == 1
    with pytest.raises(types().SafetyViolation, match="no unguarded retry"):
        rig.safe.move_joints(nearby(rig), duration_s=0.5, feedback_guard=lambda s: 0)
    assert len(rig.moves) == 1


@apai
def test_a_halt_during_a_move_ends_it_and_the_agent_keeps_the_arm(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    rig.own()
    goal = nearby(rig, by=0.3)

    def halt() -> None:
        until(lambda: rig.link.state().move_index >= 10, "ten setpoints sent")
        rig.harness.halt("wrong object")

    thread = in_thread(halt)
    with pytest.raises(types().MotionHalted, match="halted: wrong object"):
        rig.safe.move_joints(goal, duration_s=1.5)
    thread.join(5.0)
    rig.h.idle()
    state = rig.link.state()
    assert state.owner == "agent" and state.stop_level == "none"
    assert rig.stops == [] and rig.adapter._stopped is False
    assert abs(rig.safe.get_state().q[0] - goal[0]) > 0.05  # it stopped short
    # A halt is of that motion only: the next move runs.
    assert rig.safe.move_joints(nearby(rig, by=0.02), duration_s=0.5) is True


@apai
def test_a_skill_stop_through_safearm_latches_apai_and_sends_no_stop(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    rig.own()
    with rig.adapter.skill_scope():
        rig.safe.stop()
    assert rig.harness.estopped is True and rig.adapter._stopped is True
    time.sleep(0.15)
    assert rig.stops == [] and rig.link.state().owner == "agent"
    with pytest.raises(types().SafetyViolation, match="e-stop latched"):
        rig.safe.move_joints(nearby(rig), duration_s=0.5)
    assert rig.moves == []


@apai
def test_a_contact_reaches_apais_checker_through_the_monitor(
    apai_rig: ApaiRig,
) -> None:
    rig = apai_rig
    rig.own()
    start = float(rig.safe.get_state().q[0])
    rig.h.spy.backend.block_joint(0)  # the first joint stays where it is
    with pytest.raises(types().ContactDetected) as raised:
        rig.safe.move_joints(nearby(rig, by=0.4), duration_s=2.0)
    assert "contact detected at joint 1" in str(raised.value)
    event = rig.harness.last_contact  # filed by record_contact_motion
    assert event is not None and event["joint"] == 1 and event["phase"] == "stream"
    assert event["hold_q"][0] == pytest.approx(start, abs=0.02)
    assert len(rig.harness.contact_zones) == 1  # the spot is fenced on this side
    assert rig.link.state().owner == "agent" and rig.adapter._stopped is False
    # reset_scene clears the spots on both sides.
    rig.harness.clear_contact_zones("reset_scene")
    (command_id,) = [c for _, c in rig.adapter._checker_ids]
    assert rig.link.wait_final(command_id, 5.0).status == "completed"


@apai
def test_apais_jaw_calls_reach_the_thors_jaws(apai_rig: ApaiRig) -> None:
    rig = apai_rig
    rig.own()
    # APAI's mock jaws: open at 0.0, closed at 1.0 (profile units).
    rig.safe.set_gripper(0.4)
    state = rig.safe.get_state()
    assert state.gripper_valid is True
    assert state.gripper_pos == pytest.approx(0.4, abs=0.05)
    # The grasp close on an object at half travel: it stalls there.
    rig.h.spy.backend.object_stop_frac = 0.5
    measured = rig.adapter.close_gripper_two_stage(
        width_frac_stage1=0.5, width_frac_stage2=0.85, effort=0.6
    )
    assert measured == pytest.approx(0.5, abs=0.05)
    assert rig.adapter.close_stages_skipped == 0
    # After an arm move the jaw reading is older than the last setpoint: unknown.
    assert rig.safe.move_joints(nearby(rig, by=0.02), duration_s=0.5) is True
    assert rig.safe.get_state().gripper_valid is False


@apai
def test_apais_shutdown_sends_no_motion_and_no_torque_cut(apai_rig: ApaiRig) -> None:
    rig = apai_rig
    rig.own()
    assert rig.safe.move_joints(nearby(rig), duration_s=0.5) is True
    sent = len(rig.h.spy.targets)
    rig.adapter.close_task()
    rig.safe.disconnect()
    assert rig.adapter.connected is False
    time.sleep(0.2)
    assert len(rig.h.spy.targets) == sent and len(rig.moves) == 1
    assert rig.link.state().torque_on is True
