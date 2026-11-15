# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app on APAI's own code: its mock arm and three of its safety harnesses.

A workstation client sends link messages over the loopback carrier. The evidence
is what the mock arm was asked to do, the harnesses' own counters and what the
client received. Needs an APAI checkout (AGENTIC_REMOTE_APAI_SRC).
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion, Result
from isaaccapture_examples.agentic_remote.robot_app.app import MOTORS_OFF
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    Limits,
    LimitSet,
    RestConfig,
    RobotAppConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import CheckerSet
from conftest import AppHarness

try:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
except ModuleNotFoundError as missing:
    # Without an APAI checkout conftest.py skips every test here: all are marked
    # "apai". They must still be collected, or pytest exits with "no tests ran".
    if missing.name != "apai":
        raise
    apai_backend = None

pytestmark = pytest.mark.apai

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
STEP_S = 0.02

REST = [0.0] * 6  # on the stops of the second and third joint
HOME = [0.0, 1.2, 1.2, 0.0, 0.0, 0.0]
# The gripper tip about 3 cm above the table, and about 1 cm above it.
ABOVE = [0.0, 1.62, 0.7, 0.0, 0.5, 0.0]
BELOW = [0.0, 1.68, 0.7, 0.0, 0.5, 0.0]
RAISED = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]  # off the joint stops
# The gripper tip is 9 cm from where it is at RAISED.
HIGHER = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]

# The mock jaws' light holding effort: the driver's hold stiffness over its
# closing stiffness.
LIGHT = 1.0 / 6.0
JAWS_CLOSED = 1.0  # the mock profile counts the jaw position from 0 open

FRESH_READ, JAW_READ, OTHER_READ = "fresh read", "jaws", "other"


def moved(q: Sequence[float], joint: int, value: float) -> list[float]:
    """q with one joint index set to value."""
    pose = list(q)
    pose[joint] = value
    return pose


def straight(
    start: Sequence[float], goal: Sequence[float], count: int
) -> list[list[float]]:
    """`count` setpoints at one speed from start to goal; the last is goal."""
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    rows = [(a + (b - a) * (index / count)).tolist() for index in range(1, count)]
    return [*rows, [float(value) for value in goal]]


# 25 setpoints at 0.2 rad/s: the second and third joint leave their stops.
LIFT = straight(REST, RAISED, 25)
# 60 setpoints: the first joint turns at 0.35 rad/s, under the agent's 0.4.
SWEEP = straight(RAISED, moved(RAISED, 0, 0.42), 60)
# 80 setpoints at 0.25 rad/s.
CLIMB = straight(RAISED, HIGHER, 80)


def make_config(profile: apai_backend.ApaiProfile) -> RobotAppConfig:
    """A configuration whose joint limits are those of APAI's arm model.

    The agent's and the human's joint box is narrower on the first joint. The
    human joint speed is the mock profile's speed cap; a faster set is refused.
    """
    low, high = (
        (np.asarray(side, dtype=float) + 0.0).tolist()  # 0.0 for the model's -0.0
        for side in profile.kinematics.joint_limits
    )
    box_low, box_high = [-2.0, *low[1:]], [2.0, *high[1:]]
    return RobotAppConfig(
        robot_id="apai-mock-arm",
        profile_id="apai-rs-model",
        n_joints=6,
        joint_names=tuple(f"joint{i}" for i in range(1, 7)),
        limits=Limits(
            agent=LimitSet(
                name="agent",
                version="agent-1",
                joint_min=box_low,
                joint_max=box_high,
                max_joint_speed=0.4,
                envelope_min=(-1.0, 0.0, 0.0, -1.0, -1.0, -1.5),
                envelope_max=(1.0, 2.5, 2.5, 1.0, 1.0, 1.5),
            ),
            human=LimitSet(
                name="human",
                version="human-1",
                joint_min=box_low,
                joint_max=box_high,
                max_joint_speed=1.2,
                max_step_rad=0.1,
            ),
            recovery=LimitSet(
                name="recovery",
                version="recovery-1",
                joint_min=low,
                joint_max=high,
                max_joint_speed=0.1,
            ),
        ),
        rest=RestConfig(rest_q=REST, region_min=low, region_max=high),
    )


def with_config(config: RobotAppConfig, **blocks: dict[str, Any]) -> RobotAppConfig:
    """The configuration with fields of its blocks replaced: permits={...}."""
    changed = {
        block: dataclasses.replace(getattr(config, block), **fields)
        for block, fields in blocks.items()
    }
    return dataclasses.replace(config, **changed)


class BackendSpy:
    """Records what the robot app asks of APAI's mock arm.

    Every call is passed on unchanged, except the one joint target or the joint
    reads a test tells it to fail.
    """

    # What the robot app must never ask of a driver. The mock arm has no
    # hard_estop(); the spy adds one that only records the call.
    FORBIDDEN = ("stop", "hard_estop", "resume", "stream_to", "stream_path")
    _JAW_CALLS = (
        "gripper_reference",
        "gripper_read",
        "gripper_command",
        "gripper_light_hold",
    )

    def __init__(self, arm: apai_backend.ApaiArm, checkers: CheckerSet) -> None:
        self.backend = backend = arm.backend
        self._harnesses = [checker.harness for checker in checkers.all()]
        self._thread = threading.local()
        self.calls: list[str] = []  # the driver methods called, in order
        self.targets: list[list[float]] = []  # every joint target the driver took
        # Per joint target: the agent's, the human's and the recovery harness's
        # count of approved commands at the moment it was sent.
        self.approved: list[tuple[int, ...]] = []
        self.jaw_commands: list[tuple[float, float]] = []  # (position, effort)
        # For each get_state(): FRESH_READ inside the driver's fresh joint read,
        # JAW_READ inside a jaw call of the arm port, OTHER_READ anywhere else.
        self.state_reads: list[str] = []
        self.released_at: list[list[float]] = []  # the pose at each disconnect
        # Called after a joint target with its number, after a jaw command with
        # its position and effort; on the thread that sent it.
        self.after_target: list[Callable[[int], None]] = []
        self.after_jaw_command: list[Callable[[float, float], None]] = []
        self.fail_target: int | None = None  # the joint target the driver refuses
        self.fail_reads = False  # every fresh joint read fails
        self._real = {
            name: getattr(backend, name)
            for name in (
                "send_joint_target",
                "set_gripper",
                "get_state",
                "read_positions_for_monitor",
                "disconnect",
            )
        }
        backend.send_joint_target = self._send_joint_target
        backend.set_gripper = self._set_gripper
        backend.get_state = self._get_state
        backend.read_positions_for_monitor = self._read_positions_for_monitor
        backend.disconnect = self._disconnect
        for name in ("connect", *self.FORBIDDEN):
            setattr(backend, name, self._recorded(name))
        for name in self._JAW_CALLS:
            setattr(arm, name, self._as_jaw_call(getattr(arm, name)))

    def pose(self) -> list[float]:
        """The mock arm's measured pose, read without leaving a record."""
        return self._real["get_state"]().q.tolist()

    def place(self, pose: Sequence[float]) -> None:
        """Put the mock arm at a pose; the robot app sees no command."""
        self._real["send_joint_target"](np.array(pose, dtype=float))

    def _recorded(self, name: str) -> Callable[..., Any]:
        real = getattr(self.backend, name, None)

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return None if real is None else real(*args, **kwargs)

        return call

    def _as_jaw_call(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def jaw_call(*args: Any, **kwargs: Any) -> Any:
            with self._reading(JAW_READ):
                return call(*args, **kwargs)

        return jaw_call

    @contextlib.contextmanager
    def _reading(self, what: str) -> Iterator[None]:
        before = getattr(self._thread, "reading", OTHER_READ)
        self._thread.reading = what
        try:
            yield
        finally:
            self._thread.reading = before

    def _send_joint_target(self, q: np.ndarray) -> None:
        self.calls.append("send_joint_target")
        number = len(self.targets) + 1
        if number == self.fail_target:
            self.fail_target = None
            raise OSError("bus off")
        self._real["send_joint_target"](q)
        self.targets.append(np.asarray(q, dtype=float).tolist())
        self.approved.append(tuple(h.approved_commands for h in self._harnesses))
        for hook in list(self.after_target):
            hook(number)

    def _set_gripper(self, pos: float, effort: float = 1.0) -> None:
        self.calls.append("set_gripper")
        self._real["set_gripper"](pos, effort)
        self.jaw_commands.append((float(pos), float(effort)))
        for hook in list(self.after_jaw_command):
            hook(float(pos), float(effort))

    def _get_state(self) -> Any:
        self.calls.append("get_state")
        self.state_reads.append(getattr(self._thread, "reading", OTHER_READ))
        return self._real["get_state"]()

    def _read_positions_for_monitor(self) -> np.ndarray:
        self.calls.append("read_positions_for_monitor")
        if self.fail_reads:
            raise RuntimeError("CAN feedback lost")
        with self._reading(FRESH_READ):
            return self._real["read_positions_for_monitor"]()

    def _disconnect(self) -> None:
        self.calls.append("disconnect")
        self.released_at.append(self.pose())
        self._real["disconnect"]()


class ApaiHarness(AppHarness):
    """The robot app on APAI's mock arm and APAI's checkers, with a client."""

    def __init__(
        self,
        config: RobotAppConfig,
        profile: apai_backend.ApaiProfile,
        clock: FakeClock,
    ) -> None:
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        arm = apai_backend.build_apai_arm(config, profile, clock)
        self.spy = BackendSpy(arm, checkers)
        super().__init__(config, arm, checkers, clock)

    @property
    def harnesses(self) -> list[Any]:
        """The agent's, the human's and the recovery checker's SafetyHarness."""
        return [checker.harness for checker in self.checkers.all()]

    def sent(self) -> list[list[float]]:
        """Every joint target the driver took so far."""
        return list(self.spy.targets)

    def hold_move_after(self, count: int) -> tuple[threading.Event, threading.Event]:
        """Hold the motion worker once the driver took `count` joint targets in all.

        Returns (reached, release), as the simulated arm's harness does.
        """
        reached, release = threading.Event(), threading.Event()
        self._releases.append(release)

        def hold(number: int) -> None:
            if number == count:
                reached.set()
                release.wait(WAIT_S)

        self.spy.after_target.append(hold)
        return reached, release


def check_driver_use(h: ApaiHarness) -> None:
    """What holds after every scenario, whatever it did.

    The driver was connected once, never stopped, never asked for its cached
    state in place of a fresh joint read, and disconnected at measured rest only.
    """
    spy, rest = h.spy, h.config.rest
    assert not set(BackendSpy.FORBIDDEN) & set(spy.calls)
    assert OTHER_READ not in spy.state_reads
    assert spy.calls.count("connect") == 1
    assert len(spy.released_at) == spy.calls.count("disconnect") <= 1
    for pose in spy.released_at:
        off = max(abs(at - goal) for at, goal in zip(pose, rest.rest_q))
        assert off <= rest.rest_tol_rad, f"disconnected {off:.3f} rad from rest"
    assert h.client.unexpected == []


@pytest.fixture(scope="module")
def profile() -> apai_backend.ApaiProfile:
    return apai_backend.load_apai_profile("mock")


@pytest.fixture
def config(profile: apai_backend.ApaiProfile) -> RobotAppConfig:
    return make_config(profile)


@pytest.fixture
def make_app(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> Iterator[Callable[..., ApaiHarness]]:
    """Builds and starts a robot app on APAI's mock arm, at rest.

    make_app(config=..., profile=...). After the test each app is closed and
    check_driver_use() is applied to it.
    """
    made: list[ApaiHarness] = []

    def make(
        config: RobotAppConfig = config,
        profile: apai_backend.ApaiProfile = profile,
    ) -> ApaiHarness:
        harness = ApaiHarness(config, profile, clock)
        made.append(harness)
        harness.start()
        return harness

    yield make
    for harness in made:
        harness.close()
        check_driver_use(harness)


@pytest.fixture
def apai_app(make_app: Callable[..., ApaiHarness]) -> ApaiHarness:
    """A started robot app on APAI's mock arm at rest, with a workstation client."""
    return make_app()


def statuses(h: ApaiHarness, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def final(h: ApaiHarness, command_id: str) -> Result:
    """A command's final RESULT, once the ROBOT_STATE that shows it has arrived.

    From that state the client knows the robot app's clock as it stands after
    the command, so the next permit is stamped with the right time.
    """
    result = h.client.wait_result(command_id, timeout_s=WAIT_S)
    h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == command_id, WAIT_S
    )
    return result


def run(
    h: ApaiHarness,
    setpoints: Sequence[Sequence[float]],
    start: Sequence[float],
    **fields: Any,
) -> Result:
    """An agent MOVE from `start` with permits for all of it, to its final RESULT."""
    result = h.client.run_move(setpoints, WAIT_S, start_q=start, **fields)
    return final(h, result.command_id)


def held_move(h: ApaiHarness, count: int = 6) -> tuple[str, threading.Event]:
    """An agent MOVE along LIFT, held once `count` joint targets were sent."""
    reached, release = h.hold_move_after(count)
    command_id = h.client.move(LIFT, start_q=REST)
    h.client.permit(command_id, len(LIFT) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def give_back(h: ApaiHarness) -> None:
    """The agent gives the arm back: OWNER_REQUEST "nobody", with its ROBOT_STATE."""
    result = final(h, h.client.request_owner("nobody", reason="task done"))
    assert result.status == "completed", result.reason
    number = result.detail["owner_number"]
    h.client.wait_state(lambda s: s.owner == "nobody" and s.owner_number == number)


def raise_arm(h: ApaiHarness) -> None:
    """A task lifts the arm to RAISED and ends; nobody owns the arm afterwards."""
    h.new_task()
    result = run(h, LIFT, REST)
    assert result.status == "completed", result.reason
    give_back(h)


def contact_while_sweeping(h: ApaiHarness) -> Result:
    """A task lifts the arm, then the first joint meets something as it turns.

    The arm ends holding at RAISED. Afterwards the obstacle is taken away.
    """
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"
    h.spy.backend.block_joint(0)
    result = run(h, SWEEP, RAISED)
    assert (result.status, result.reason) == ("failed", "contact")
    h.spy.backend.unblock_joints()
    return result


# Start and WELCOME


def test_start_connects_the_driver_once_and_welcome_carries_the_model_limits(
    apai_app: ApaiHarness, profile: apai_backend.ApaiProfile
) -> None:
    h = apai_app
    # The joint limits of APAI's reBot RS arm model, which its mock arm also uses.
    low, high = (list(side) for side in profile.kinematics.joint_limits)
    assert low == [-2.8, 0.0, 0.0, -1.69, -1.57, -3.14]
    assert high == [2.8, 3.14, 3.14, 1.79, 1.57, 3.14]
    welcome = h.client.welcome
    assert welcome.session_id == h.app.session_id
    assert (welcome.robot_id, welcome.profile_id) == ("apai-mock-arm", "apai-rs-model")
    # WELCOME carries the agent's joint box: narrower on the first joint.
    assert welcome.joint_min == [-2.0, *low[1:]]
    assert welcome.joint_max == [2.0, *high[1:]]
    assert welcome.rest_q == REST
    assert welcome.setpoint_period_s == STEP_S
    assert welcome.limit_set_versions == {
        "agent": "agent-1",
        "human": "human-1",
        "recovery": "recovery-1",
    }
    assert h.spy.calls.count("connect") == 1
    with pytest.raises(RuntimeError, match=r"start\(\) was already called"):
        h.app.start()
    assert h.spy.calls.count("connect") == 1


def test_at_start_the_arm_holds_at_rest_with_referenced_jaws(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    state = h.client.state
    assert (state.owner, state.owner_number) == ("nobody", 0)
    assert (state.stop_number, state.stop_level) == (0, "none")
    assert (state.motion, state.torque_on) == ("rest_hold", True)
    assert [sample.value for sample in state.joints] == REST
    assert all(sample.valid for sample in state.joints)
    jaws = state.gripper
    assert jaws.referenced and jaws.valid and not jaws.may_be_holding
    assert (jaws.opening, jaws.last_target, jaws.load) == (0.0, 0.0, "empty")
    # The reference closed the jaws at full effort; no joint target was sent.
    assert h.spy.jaw_commands == [(JAWS_CLOSED, 1.0)]
    assert h.spy.targets == []
    for harness in h.harnesses:
        assert not harness.estopped and harness.halted is None
        assert harness.approved_commands == 0


# OWNER_REQUEST


def test_a_new_task_is_accepted_at_rest_on_the_joint_stops(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    # The harness alone refuses the rest pose: two joints lie on their stops,
    # inside the margin it keeps to the model's limits.
    assert h.checkers.agent.harness.vet_pose(np.array(REST)) is not None
    result = h.new_task()
    assert result.detail == {"owner": "agent", "owner_number": 1}
    state = h.client.state
    assert (state.owner, state.owner_number) == ("agent", 1)
    assert state.limit_set_version == "agent-1"
    assert h.spy.targets == []


@pytest.mark.parametrize(
    ("pose", "reason"),
    [
        (BELOW, r"type rest first: TCP z=0\.011 below table clearance 0\.020"),
        (moved(HOME, 0, 0.9), r"type rest first: TCP .* outside workspace"),
        (
            moved(HOME, 5, 1.6),
            r"type rest first: the step starts outside the agent envelope, which "
            r"cannot be re-entered: joint 6 at 1\.600 rad",
        ),
    ],
    ids=["below the table clearance", "beside the workspace", "outside the envelope"],
)
def test_a_new_task_is_refused_at_a_pose_the_agent_checker_refuses(
    apai_app: ApaiHarness, pose: list[float], reason: str
) -> None:
    h = apai_app
    h.spy.place(pose)
    result = h.client.start_task()
    assert result.status == "refused"
    assert re.match(reason, result.reason), result.reason
    state = h.app.snapshot()
    assert (state.owner, state.owner_number) == ("nobody", 0)
    assert h.spy.targets == []


# MOVE


def test_a_permitted_move_completes_with_every_setpoint_approved_by_the_harness(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    result = run(h, LIFT, REST)
    assert statuses(h, result.command_id) == ["accepted", "completed"]
    assert result.detail == {"outcome": "completed", "sent": 25, "total": 25}
    assert result.end_q == pytest.approx(RAISED)
    assert h.sent() == LIFT
    assert h.spy.pose() == pytest.approx(RAISED)
    # When each target reached the driver, the agent's harness had approved that
    # many commands; the other two harnesses were never asked.
    assert h.spy.approved == [(number, 0, 0) for number in range(1, 26)]
    agent = h.checkers.agent.harness
    assert agent.motions_begun == 1
    assert not agent.motion_active
    assert h.arm.torque_on()


def test_a_move_out_of_the_agents_envelope_is_refused_before_any_setpoint(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    edge = moved(HOME, 5, 1.5)  # the last joint on the edge of the envelope
    leaving = straight(edge, moved(HOME, 5, 1.6), 20)
    h.spy.place(edge)
    h.new_task()
    result = run(h, leaving, edge)
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason == (
        "the target is outside the agent envelope, which cannot be re-entered: "
        "joint 6 at 1.505 rad, envelope -1.500..1.500"
    )
    assert result.detail == {"outcome": "refused", "sent": 0, "total": 20}
    assert result.end_q == pytest.approx(edge)
    assert h.spy.targets == []
    # The envelope is the agent checker's own rule: the harness has none and was
    # not asked.
    agent = h.checkers.agent.harness
    assert agent.vet_step(np.array(edge), np.array(leaving[0]), STEP_S) is None
    assert agent.approved_commands == 0


def test_a_move_faster_than_the_agents_speed_limit_is_refused_by_the_harness(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    goal = [0.0, 0.3, 0.3, 0.0, 0.0, 0.0]
    fast = straight(REST, goal, 15)  # 1 rad/s
    result = run(h, fast, REST)
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason == "joint 2 velocity 1.00 rad/s exceeds 0.40"
    assert result.detail == {"outcome": "refused", "sent": 0, "total": 15}
    assert h.spy.targets == []
    agent = h.checkers.agent.harness
    assert result.reason in agent.violations  # the harness's own refusal
    assert agent.approved_commands == 0
    assert not agent.motion_active
    # The same path at 0.2 rad/s, half the agent's limit, is taken.
    slow = straight(REST, goal, 75)
    assert run(h, slow, REST).status == "completed"
    assert h.sent() == slow


def test_a_move_along_the_joint_stops_is_refused_by_the_harness_joint_margin(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    # The first joint turns while the second and third stay on their stops.
    turn = straight(REST, moved(REST, 0, 0.1), 25)
    result = run(h, turn, REST)
    assert result.status == "refused"
    assert result.reason == "joint 2 target 0.000 rad outside [0.020, 3.120]"
    assert h.spy.targets == []
    # Off the stops first, the same turn is taken.
    assert run(h, LIFT, REST).status == "completed"
    turn = straight(RAISED, moved(RAISED, 0, 0.1), 25)
    assert run(h, turn, RAISED).status == "completed"


def test_a_grasp_region_opens_the_table_rule_for_its_move_only(
    apai_app: ApaiHarness, profile: apai_backend.ApaiProfile
) -> None:
    h = apai_app
    descent = straight(ABOVE, BELOW, 10)
    h.spy.place(ABOVE)
    h.new_task()
    refused = run(h, descent, ABOVE)
    assert refused.status == "refused"
    assert re.match(
        r"TCP z=0\.01\d below table clearance 0\.020 outside the grasp exemption",
        refused.reason,
    )
    # The arm went down to the clearance and holds at the last target sent.
    sent = refused.detail["sent"]
    assert 0 < sent < 10
    assert h.sent() == descent[:sent]

    x, y, _ = profile.kinematics.fk(np.array(BELOW))[:3, 3]
    region = GraspRegion((float(x), float(y)), 0.05, 0.0, None)
    result = run(h, descent[sent:], descent[sent - 1], grasp_region=region)
    assert result.status == "completed", result.reason
    assert h.spy.pose() == pytest.approx(BELOW)
    # The move is over: the harness applies the table rule again.
    agent = h.checkers.agent.harness
    again = agent.vet_step(np.array(ABOVE), np.array(BELOW), 1.0)
    assert again is not None and "below table clearance" in again


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {"center_xy": (0.3, 5.8), "radius_m": 1e6, "z_min": -1e6},
            "the grasp region's radius is 1e+06 m, the limit is 0.3 m",
        ),
        (
            {"center_xy": (0.3, 0.0), "radius_m": 50.0},
            "the grasp region's radius is 50 m, the limit is 0.3 m",
        ),
        (
            {"z_min": -5.0},
            "the grasp region reaches 5 m below the table, the limit is 0.06 m",
        ),
    ],
    ids=["as wide as the world", "50 m", "5 m deep"],
)
def test_a_move_with_a_grasp_region_beyond_the_bounds_is_refused_and_moves_nothing(
    apai_app: ApaiHarness,
    profile: apai_backend.ApaiProfile,
    change: dict[str, Any],
    reason: str,
) -> None:
    h = apai_app
    h.spy.place(ABOVE)
    h.new_task()
    x, y, _ = profile.kinematics.fk(np.array(BELOW))[:3, 3]
    region = dataclasses.replace(
        GraspRegion((float(x), float(y)), 0.05, 0.0, None), **change
    )
    result = run(h, straight(ABOVE, BELOW, 10), ABOVE, grasp_region=region)
    # Refused before it is accepted: the table rule was never opened.
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason == reason
    assert h.sent() == []
    assert h.spy.pose() == pytest.approx(ABOVE)
    # The region is not the last grasp region, so it blocks no return to rest.
    assert h.app._grasp_region is None


# The perception watchdog


def test_a_move_without_a_permit_is_refused_and_the_watchdog_was_never_fed(
    make_app: Callable[..., ApaiHarness], config: RobotAppConfig
) -> None:
    from apai.types import SafetyViolation

    h = make_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    command_id = h.client.move(LIFT, start_q=REST)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert result.reason == "no permit for this move (waited 0.02 s)"
    assert result.detail == {"outcome": "refused", "sent": 0, "total": 25}
    assert h.spy.targets == []
    agent = h.checkers.agent.harness
    assert agent.motions_begun == 0
    # Asked directly, the harness would have refused the motion as well.
    with pytest.raises(SafetyViolation, match=r"perception watchdog: no observation"):
        agent.check_before_motion()


def test_permits_feed_the_agent_harness_watchdog_and_no_other(
    apai_app: ApaiHarness,
) -> None:
    from apai.types import SafetyViolation

    h = apai_app
    h.new_task()
    agent, human, recovery = h.harnesses
    with pytest.raises(SafetyViolation, match=r"perception watchdog: no observation"):
        agent.check_before_motion()
    command_id = h.client.new_id()
    h.client.permit(command_id, len(LIFT) - 1)
    h.step()  # the dispatcher has taken the permit
    agent.check_before_motion()
    h.client.move(LIFT, start_q=REST, command_id=command_id)
    assert final(h, command_id).status == "completed"
    assert h.sent() == LIFT
    # The human's harness waits for controller samples; the recovery harness
    # has no age limit.
    with pytest.raises(SafetyViolation, match=r"perception watchdog: no observation"):
        human.check_before_motion()
    recovery.check_before_motion()


def test_evidence_older_than_the_harness_watchdog_allows_starts_no_move(
    make_app: Callable[..., ApaiHarness],
) -> None:
    # The harness accepts evidence up to 1 s old, the permit gate up to 3 s.
    strict = apai_backend.load_apai_profile("mock", {"safety": {"watchdog_s": 1.0}})
    h = make_app(profile=strict)
    h.new_task()
    command_id = h.client.new_id()
    _, wall = h.client.thor_now()
    h.client.permit(command_id, len(LIFT) - 1, evidence_wall_s=wall - 2.0)
    h.client.move(LIFT, start_q=REST, command_id=command_id)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert re.fullmatch(
        r"perception watchdog: last observation 2\.\ds old", result.reason
    )
    assert h.spy.targets == []
    assert h.checkers.agent.harness.motions_begun == 0

    # With fresh evidence the same move runs.
    assert run(h, LIFT, REST).status == "completed"


# STOP


def test_a_stop_during_a_move_halts_all_three_harnesses_and_not_the_driver(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    counts = [checker.halt_count() for checker in h.checkers.all()]
    move_id, release = held_move(h, 6)
    h.client.stop("operator pressed stop")
    state = h.client.wait_state(lambda s: s.stop_number == 1)
    assert (state.owner, state.stop_level) == ("nobody", "stop")
    for harness in h.harnesses:
        assert harness.halted == "operator pressed stop"
        assert "HALT: operator pressed stop" in harness.violations
        assert not harness.estopped
    assert [checker.halt_count() for checker in h.checkers.all()] == [
        count + 1 for count in counts
    ]
    release.set()
    result = final(h, move_id)
    assert result.status == "cancelled"
    assert result.reason == "STOP from console: operator pressed stop"
    assert result.detail == {"outcome": "stopped", "sent": 6, "total": 25}
    # The seventh setpoint never reached the driver, which holds the sixth.
    assert h.sent() == LIFT[:6]
    assert h.arm.last_target()[0] == LIFT[5]
    assert h.spy.pose() == pytest.approx(LIFT[5])
    assert h.arm.torque_on()
    assert not {"stop", "hard_estop", "disconnect"} & set(h.spy.calls)
    # The jaws were told to hold lightly where they are; nothing opened them.
    assert h.spy.jaw_commands[-1] == (JAWS_CLOSED, pytest.approx(LIGHT))


def test_after_a_stop_old_commands_are_refused_and_a_new_task_moves_again(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"
    h.client.stop("operator pressed stop")
    h.client.wait_state(lambda s: s.stop_number == 1)
    agent = h.checkers.agent.harness
    assert agent.halted == "operator pressed stop"
    old = final(h, h.client.move(CLIMB, start_q=RAISED, stop_number=0, owner_number=1))
    assert old.status == "refused"
    assert old.reason == "stop number 0 does not match the current 1"
    # The harness forgets its halt when the next admitted move begins.
    h.new_task()
    assert run(h, CLIMB, RAISED).status == "completed"
    assert agent.halted is None
    assert h.sent() == [*LIFT, *CLIMB]


# Fault-stop


def test_a_driver_fault_latches_all_three_harnesses_until_a_reset(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    fault = "the driver did not take a joint target: bus off"
    h.new_task()
    h.spy.fail_target = 7
    result = run(h, LIFT, REST)
    assert (result.status, result.reason) == ("failed", f"fault: {fault}")
    assert result.detail == {"outcome": "failed", "sent": 6, "total": 25}
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == fault
    assert (state.owner, state.stop_number, state.torque_on) == ("nobody", 1, True)
    for harness in h.harnesses:
        assert harness.estopped
        assert f"ESTOP: {fault}" in harness.violations
    assert h.sent() == LIFT[:6]

    # Latched: no task is taken, and a reset must quote the current stop number.
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == (
        "refused",
        f"fault-stop latched: {fault}",
    )
    old = final(h, h.client.reset(stop_number=0))
    assert old.status == "refused"
    assert old.reason == "stop number 0 does not match the current 1"
    assert all(harness.estopped for harness in h.harnesses)

    assert final(h, h.client.reset()).status == "completed"
    assert not any(harness.estopped for harness in h.harnesses)
    state = h.client.wait_state(lambda s: s.stop_level == "none")
    assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
    # The reset moved nothing; a new task then finishes the lift.
    assert h.sent() == LIFT[:6]
    h.new_task()
    assert run(h, LIFT[6:], LIFT[5]).status == "completed"
    assert h.sent() == LIFT


def test_joint_feedback_lost_in_a_hold_is_a_fault_stop(apai_app: ApaiHarness) -> None:
    h = apai_app
    h.spy.fail_reads = True
    h.advance_until(lambda: h.heard("fault"), step_s=STEP_S, limit_s=1.0)
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert re.fullmatch(
        r"joint feedback lost for 0\.3[6-9] s in hold", state.fault_reason
    )
    assert state.torque_on
    assert all(harness.estopped for harness in h.harnesses)
    assert h.spy.targets == []


# Jaws


def test_a_jaw_close_under_way_at_a_stop_ends_at_the_light_holding_effort(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    h.spy.backend.object_stop_frac = 0.6  # an object: the jaws jam at opening 0.4
    light_hold_sent = threading.Event()

    def on_jaw_command(position: float, effort: float) -> None:
        if effort < 0.5:
            light_hold_sent.set()

    h.spy.after_jaw_command.append(on_jaw_command)
    # The close is under way: the driver has its command, and the gripper worker
    # waits at its first look at the jaws.
    looking, release = h.hold_call("gripper_read")
    jaw_id = h.client.gripper(0.0, max_effort=0.8)
    assert looking.wait(WAIT_S)
    assert h.spy.jaw_commands[-1] == (JAWS_CLOSED, 0.8)
    h.client.stop("operator pressed stop")
    # The stop does not wait for the close to return.
    assert light_hold_sent.wait(WAIT_S)
    assert h.client.wait_result(jaw_id, final=False).status == "accepted"
    assert statuses(h, jaw_id) == ["accepted"]
    release.set()
    result = final(h, jaw_id)
    assert statuses(h, jaw_id) == ["accepted", "cancelled"]
    assert result.detail == {"outcome": "cancelled", "opening": pytest.approx(0.4)}
    # Everything the driver was sent after the close is the closed target at the
    # light effort: the jaws were not opened and not left closing.
    after = h.spy.jaw_commands[h.spy.jaw_commands.index((JAWS_CLOSED, 0.8)) + 1 :]
    assert len(after) >= 2
    assert all(command == (JAWS_CLOSED, pytest.approx(LIGHT)) for command in after)
    state = h.client.wait_state(
        lambda s: s.stop_number == 1 and s.gripper.may_be_holding
    )
    assert (state.gripper.opening, state.gripper.load) == (pytest.approx(0.4), "loaded")
    assert "stop" not in h.spy.calls


@pytest.mark.parametrize(
    ("mode", "status"), [("plain", "failed"), ("close_until_contact", "completed")]
)
def test_a_close_that_stalls_on_an_object_is_left_at_the_light_holding_effort(
    apai_app: ApaiHarness, mode: str, status: str
) -> None:
    h = apai_app
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    h.spy.backend.object_stop_frac = 0.6  # an object: the jaws jam at opening 0.4
    result = final(h, h.client.gripper(0.0, max_effort=0.8, mode=mode))
    assert result.status == status
    assert result.detail == {"outcome": "stalled", "opening": pytest.approx(0.4)}
    # The closed target stands, at the light effort; the jaws were not opened.
    assert h.spy.jaw_commands[-2:] == [
        (JAWS_CLOSED, 0.8),
        (JAWS_CLOSED, pytest.approx(LIGHT)),
    ]
    h.advance(5.0)
    assert h.spy.backend._gripper_effort == pytest.approx(LIGHT)
    state = h.client.wait_state(lambda s: s.gripper.may_be_holding)
    assert (state.gripper.opening, state.gripper.load) == (pytest.approx(0.4), "loaded")


def test_jaws_closed_on_an_object_refuse_a_release_and_a_return_to_rest(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    h.spy.backend.object_stop_frac = 0.6
    result = final(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert result.status == "completed"
    assert result.detail == {"outcome": "stalled", "opening": pytest.approx(0.4)}
    assert h.spy.jaw_commands[-1] == (JAWS_CLOSED, pytest.approx(LIGHT))

    loaded = "the jaws are not proven empty (loaded): the jaws may be holding something"
    # The arm is at rest; only the jaws stand in the way of a release.
    release = final(h, h.client.release())
    assert (release.status, release.reason) == ("refused", loaded)
    assert "disconnect" not in h.spy.calls
    # Raised with the object, the arm is not returned to rest either: it holds.
    assert run(h, LIFT, REST).status == "completed"
    give_back(h)
    rest = final(h, h.client.rest())
    assert (rest.status, rest.reason) == ("refused", loaded)
    assert h.sent() == LIFT
    assert h.spy.pose() == pytest.approx(RAISED)
    assert h.arm.torque_on()


# Contact


def test_a_contact_found_by_the_harness_monitor_ends_in_a_hold_at_the_measured_pose(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"

    def meet_something(number: int) -> None:
        if number == len(LIFT) + 20:
            h.spy.backend.block_joint(0)  # the first joint stops where it is

    h.spy.after_target.append(meet_something)
    result = run(h, SWEEP, RAISED)
    assert (result.status, result.reason) == ("failed", "contact")
    assert re.match(
        r"contact detected at joint 1: tracking error 0\.\d+ rad for \d+ ms -- "
        r"something is blocking the arm",
        result.detail["contact"],
    )
    stopped_at = SWEEP[19]
    assert result.end_q == pytest.approx(stopped_at)
    sent = result.detail["sent"]
    assert result.detail["outcome"] == "contact"
    assert 20 < sent < len(SWEEP)
    # The driver's last target is the pose it measured, not the setpoint the
    # arm could not reach.
    assert h.sent()[len(LIFT) :] == [*SWEEP[:sent], pytest.approx(stopped_at)]
    assert h.spy.pose() == pytest.approx(stopped_at)

    # A hold, not a fault-stop: the agent still owns the arm.
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == result.command_id
    )
    assert (state.owner, state.stop_level, state.motion) == ("agent", "none", "hold")
    assert state.torque_on
    assert not any(harness.estopped for harness in h.harnesses)
    # The agent's and the recovery harness keep the spot; a person is not bound.
    assert [len(harness.contact_zones) for harness in h.harnesses] == [1, 0, 1]


def test_after_a_contact_a_new_task_is_accepted_where_the_arm_holds(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    contact_while_sweeping(h)
    # The agent gives up: nobody owns the arm, which holds where the contact left it.
    stuck = final(h, h.client.request_owner("help_needed", reason="blocked"))
    assert stuck.status == "completed", stuck.reason
    number = stuck.detail["owner_number"]
    h.client.wait_state(lambda s: s.owner == "nobody" and s.owner_number == number)
    # A return to rest is refused while the arm holds at a contact.
    rest = final(h, h.client.rest())
    assert (rest.status, rest.reason) == ("refused", "the arm is holding at a contact")
    # That leaves the agent to move the arm: the harness lets it leave the spot.
    task = h.client.start_task()
    assert task.status == "completed", task.reason
    assert run(h, CLIMB, RAISED).status == "completed"
    assert h.spy.pose() == pytest.approx(HIGHER)


def test_a_return_to_rest_that_would_enter_a_contact_spot_is_refused(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    contact_while_sweeping(h)
    # Leaving the spot is the agent's to do; the arm ends 9 cm away from it.
    assert run(h, CLIMB, RAISED).status == "completed"
    give_back(h)
    sent = len(h.sent())
    result = final(h, h.client.rest())
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert re.match(
        r"the path to rest was refused: setpoint \d+: TCP would enter the contact zone",
        result.reason,
    )
    assert len(h.sent()) == sent
    assert h.spy.pose() == pytest.approx(HIGHER)
    assert h.arm.torque_on()
    assert h.app.snapshot().owner == "nobody"


def test_a_contact_spot_that_was_cleared_does_not_block_the_return_to_rest(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    contact_while_sweeping(h)
    # The obstacle is gone and the workstation says so, as APAI's reset_scene does.
    cleared = final(h, h.client.checker_state("clear_contact_spots"))
    assert cleared.status == "completed"
    assert run(h, CLIMB, RAISED).status == "completed"
    give_back(h)
    # The recovery checker keeps the spots the agent checker has, and no others.
    result = final(h, h.client.rest())
    assert result.status == "completed", result.reason
    assert h.spy.pose() == pytest.approx(REST)


# Return to rest, release, shutdown and close


def test_a_return_to_rest_runs_under_the_recovery_harness_and_ends_powered(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    raise_arm(h)
    result = final(h, h.client.rest())
    assert statuses(h, result.command_id) == ["accepted", "completed"]
    assert result.detail == {"outcome": "at_rest"}
    assert result.end_q == pytest.approx(REST)
    # 0.1 rad of travel at the recovery speed: 105 setpoints, the last one rest.
    back = h.sent()[len(LIFT) :]
    assert len(back) == 105
    assert back[-1] == REST
    # The recovery harness approved every one of them, and nobody else did.
    assert h.spy.approved[len(LIFT) :] == [(25, 0, n) for n in range(1, 106)]
    recovery = h.checkers.recovery.harness
    assert recovery.motions_begun == 1
    assert not recovery.motion_active
    state = h.client.wait_state(
        lambda s: s.motion == "rest_hold" and s.owner == "nobody"
    )
    assert state.torque_on
    assert h.arm.torque_on()
    assert "disconnect" not in h.spy.calls


def show_the_state_before_the_hand_back(h: ApaiHarness) -> None:
    """Make the client hold the ROBOT_STATE between rest_hold and the hand-back.

    The return gives the arm back only once the client has motion rest_hold
    with recovery still the owner, and the next ROBOT_STATE leaves 50 ms of
    real time later.
    """
    late = threading.Event()
    give_back, send_latest = h.app._rest._give_back, h.robot_end.send_latest

    def held_give_back(*numbers: Any) -> None:
        h.client.wait_state(lambda s: s.motion == "rest_hold", WAIT_S)
        threading.Timer(0.05, late.set).start()
        h.robot_end.send_latest = lambda *sent: late.wait(WAIT_S) and send_latest(*sent)
        give_back(*numbers)

    h.app._rest._give_back = held_give_back


@pytest.mark.parametrize(
    "in_between", [False, True], ids=["as it comes", "a state before the hand-back"]
)
def test_the_timed_return_after_the_assistance_wait_runs_under_the_recovery_harness(
    apai_app: ApaiHarness, in_between: bool
) -> None:
    h = apai_app
    if in_between:
        show_the_state_before_the_hand_back(h)
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"
    stuck = final(h, h.client.request_owner("help_needed", reason="model error"))
    assert stuck.detail["timed_return"] is True
    number = stuck.detail["owner_number"]
    h.client.wait_state(lambda s: s.owner == "nobody" and s.owner_number == number)
    # No step of the clock after the wait ran out: the return is then moving.
    h.advance_until(lambda: h.heard("expired"))
    h.idle()
    assert h.heard("rest")[-1][1] == "return to rest: the arm is at rest, motors on"
    assert h.spy.pose() == pytest.approx(REST)
    assert [harness.approved_commands for harness in h.harnesses] == [25, 0, 105]
    # A ROBOT_STATE can show rest_hold before the arm is given back.
    state = h.client.wait_state(
        lambda s: s.motion == "rest_hold" and s.owner == "nobody"
    )
    assert state.torque_on
    assert "disconnect" not in h.spy.calls


def test_a_contact_during_a_return_to_rest_ends_it_in_a_powered_hold(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"
    assert run(h, CLIMB, RAISED).status == "completed"
    give_back(h)
    before = len(h.sent())

    def meet_something(number: int) -> None:
        if number == before + 200:
            h.spy.backend.block_joint(1)  # the second joint stops where it is

    h.spy.after_target.append(meet_something)
    result = final(h, h.client.rest())
    assert statuses(h, result.command_id) == ["accepted", "failed"]
    assert re.match(
        r"contact after \d+ of 521 setpoints: contact detected at joint 2: tracking",
        result.reason,
    )
    assert result.detail == {"outcome": "failed"}
    # The driver's last target is the measured pose: the second joint where it
    # stopped, short of the last setpoint of the return.
    *back, held = h.sent()[before:]
    assert 200 < len(back) < 521
    assert held[1] == pytest.approx(back[199][1])
    assert held[1] - back[-1][1] > 0.1
    assert result.end_q == pytest.approx(held)
    assert h.spy.pose() == pytest.approx(held)
    # One attempt: the arm holds with the motors on and nobody owns it.
    state = h.client.state
    assert (state.owner, state.stop_level, state.motion) == ("nobody", "none", "hold")
    assert state.torque_on
    assert "disconnect" not in h.spy.calls
    assert [len(harness.contact_zones) for harness in h.harnesses] == [1, 0, 1]
    recovery = h.checkers.recovery.harness
    assert recovery.motions_begun == 1
    assert recovery.approved_commands == len(back)


def test_a_return_to_rest_is_refused_with_the_gripper_tip_at_the_table(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.spy.place(BELOW)
    result = final(h, h.client.rest())
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason == (
        "the gripper tip is at z=0.011 m, below the table clearance 0.020 m"
    )
    assert h.spy.targets == []
    assert h.arm.torque_on()


def test_a_release_away_from_rest_is_refused_and_the_driver_stays_connected(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    raise_arm(h)
    result = final(h, h.client.release())
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason.startswith(
        "the arm is not at rest: joint index 1 is at 0.1000 rad"
    )
    assert "disconnect" not in h.spy.calls
    assert h.arm.torque_on()
    assert h.client.wait_event("refused").text.startswith("release refused: ")


def test_a_release_at_rest_is_the_only_disconnect_of_the_driver(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    raise_arm(h)
    assert final(h, h.client.rest()).status == "completed"
    assert "disconnect" not in h.spy.calls
    result = final(h, h.client.release())
    assert statuses(h, result.command_id) == ["accepted", "completed"]
    assert h.spy.calls.count("disconnect") == 1
    assert h.spy.released_at == [pytest.approx(REST)]
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on
    # The motors stay off for the rest of the run.
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    assert h.spy.calls.count("disconnect") == 1


@pytest.mark.parametrize(
    "stop",
    [
        lambda app: app.stop("Ctrl+C", "thor_terminal"),
        lambda app: app._fault("motor fault"),
    ],
    ids=["a STOP", "a fault-stop"],
)
def test_a_stop_after_a_release_sends_the_driver_nothing(
    apai_app: ApaiHarness, stop: Callable[[Any], Any]
) -> None:
    h = apai_app
    assert final(h, h.client.release()).status == "completed"
    light_holds: list[bool] = []
    light_hold = h.arm.gripper_light_hold
    h.arm.gripper_light_hold = lambda: light_holds.append(True) or light_hold()
    stop(h.app)
    h.step()
    # The stop asked the arm port for its light jaw hold, as every stop does.
    assert light_holds == [True]
    after = h.spy.calls[h.spy.calls.index("disconnect") + 1 :]
    assert not {"send_joint_target", "set_gripper", "connect"} & set(after)
    assert not h.arm.torque_on()


def test_a_shutdown_returns_to_rest_before_it_disconnects_the_driver(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    raise_arm(h)
    # The app closes after this RESULT, so no ROBOT_STATE is waited for.
    command_id = h.client.shutdown()
    result = h.client.wait_result(command_id, timeout_s=WAIT_S)
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert result.detail == {"outcome": "at_rest"}
    assert h.app.wait_closed(WAIT_S)
    assert h.spy.released_at == [pytest.approx(REST)]
    assert h.sent()[-1] == REST
    assert not h.arm.torque_on()
    # Once disconnected, the driver was sent no joint target and no jaw command.
    after = h.spy.calls[h.spy.calls.index("disconnect") + 1 :]
    assert not {"send_joint_target", "set_gripper"} & set(after)


def test_closing_the_app_away_from_rest_leaves_the_driver_connected_and_holding(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    assert run(h, LIFT, REST).status == "completed"
    h.app.close()
    assert h.app.wait_closed(0.0)
    assert not h.app.running
    assert not {"disconnect", "stop", "hard_estop"} & set(h.spy.calls)
    assert h.arm.torque_on()
    assert h.arm.last_target()[0] == RAISED
    assert h.spy.pose() == pytest.approx(RAISED)


# Joint reads


def test_no_joint_value_comes_from_the_drivers_cached_state(
    apai_app: ApaiHarness,
) -> None:
    h = apai_app
    h.new_task()
    mark = len(h.spy.state_reads)
    assert run(h, LIFT, REST).status == "completed"
    # During the move the driver's state was read inside its fresh joint read
    # only: every joint after every setpoint, and the start and end poses.
    during = h.spy.state_reads[mark:]
    assert set(during) == {FRESH_READ}
    assert len(during) >= len(LIFT) * 6
    give_back(h)
    assert final(h, h.client.rest()).status == "completed"
    assert final(h, h.client.release()).status == "completed"
    # The return and the release also read the jaws, through the arm port.
    assert set(h.spy.state_reads) == {FRESH_READ, JAW_READ}
