# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the APAI checkers and arm port in
isaaccapture_examples.agentic_remote.robot_app.apai_backend.

They need an APAI checkout (AGENTIC_REMOTE_APAI_SRC) and use its mock arm profile.
The reBot RS branch runs on APAI's real driver class over a stand-in motor SDK.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion, Result
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    GraspConfig,
    Limits,
    LimitSet,
    RestConfig,
    RobotAppConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    ArmPort,
    Checker,
    CheckerSet,
    ContactStop,
    ContactWatch,
    FeedbackLost,
    GripperOutcome,
    JointReading,
    MotionState,
    ReadFailed,
    Refusal,
)
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

STEP_S = 0.02
# The joint limits of APAI's reBot RS arm model, which its mock arm also uses.
MODEL_LOW = (-2.8, 0.0, 0.0, -1.69, -1.57, -3.14)
MODEL_HIGH = (2.8, 3.14, 3.14, 1.79, 1.57, 3.14)
# The agent's and the human's joint box is narrower than the model on the first joint.
BOX_LOW = (-2.0, *MODEL_LOW[1:])
BOX_HIGH = (2.0, *MODEL_HIGH[1:])
ENVELOPE_LOW = (-1.0, 0.0, 0.0, -1.0, -1.0, -1.5)
ENVELOPE_HIGH = (1.0, 2.5, 2.5, 1.0, 1.0, 1.5)

REST = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # on the stops of the second and third joint
HOME = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
# The gripper tip about 3 cm above the table, and about 1 cm above it.
ABOVE = (0.0, 1.62, 0.7, 0.0, 0.5, 0.0)
BELOW = (0.0, 1.68, 0.7, 0.0, 0.5, 0.0)


def moved(q: Sequence[float], joint: int, value: float) -> list[float]:
    """q with one joint index set to value."""
    pose = list(q)
    pose[joint] = value
    return pose


NEAR_HOME = moved(HOME, 0, 0.001)


def min_jerk_path(
    start: Sequence[float], goal: Sequence[float], steps: int
) -> list[list[float]]:
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    fractions = (i / steps for i in range(1, steps + 1))
    return [
        (a + (b - a) * (10 * t**3 - 15 * t**4 + 6 * t**5)).tolist() for t in fractions
    ]


def start_motion(checker: Checker) -> None:
    """Begin a motion the way an admitted move does."""
    checker.feed_watchdog()
    checker.begin_motion(checker.halt_count())


def refusal_of(call: Callable[[], Any], match: str) -> Refusal:
    """The refusal a call raises; a contact stop is not accepted in its place."""
    with pytest.raises(Refusal, match=match) as caught:
        call()
    assert type(caught.value) is Refusal
    return caught.value


class History:
    """What the watch needs of JointHistory."""

    def __init__(self) -> None:
        self.readings: list[JointReading] = []

    def add(self, reading: JointReading) -> None:
        self.readings.append(reading)


class Recorder:
    """Passes every call on to a backend and records the name of each method."""

    def __init__(self, backend: Any) -> None:
        self.wrapped = backend
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.wrapped, name)
        if not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return value(*args, **kwargs)

        return call


def make_config(rest_q: Sequence[float] = REST, joints: int = 6) -> RobotAppConfig:
    """A robot app configuration for APAI's arm model, or for its first joints.

    The human joint speed is the reBot RS profile's speed cap, which is below
    the mock profile's 1.2 rad/s: a faster limit set is refused.
    """
    return RobotAppConfig(
        robot_id="apai-test-arm",
        profile_id="apai-rs-model",
        n_joints=joints,
        joint_names=tuple(f"joint{i}" for i in range(1, joints + 1)),
        limits=Limits(
            agent=LimitSet(
                name="agent",
                version="agent-1",
                joint_min=BOX_LOW[:joints],
                joint_max=BOX_HIGH[:joints],
                max_joint_speed=0.4,
                envelope_min=ENVELOPE_LOW[:joints],
                envelope_max=ENVELOPE_HIGH[:joints],
            ),
            human=LimitSet(
                name="human",
                version="human-1",
                joint_min=BOX_LOW[:joints],
                joint_max=BOX_HIGH[:joints],
                max_joint_speed=0.8,
                max_step_rad=0.1,
            ),
            recovery=LimitSet(
                name="recovery",
                version="recovery-1",
                joint_min=MODEL_LOW[:joints],
                joint_max=MODEL_HIGH[:joints],
                max_joint_speed=0.1,
            ),
        ),
        rest=RestConfig(
            rest_q=rest_q[:joints],
            region_min=MODEL_LOW[:joints],
            region_max=MODEL_HIGH[:joints],
        ),
    )


@pytest.fixture(scope="module")
def profile() -> apai_backend.ApaiProfile:
    return apai_backend.load_apai_profile("mock")


@pytest.fixture
def config() -> RobotAppConfig:
    return make_config()


@pytest.fixture
def checkers(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> CheckerSet:
    return apai_backend.build_apai_checkers(config, profile, clock)


@pytest.fixture
def arm(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> apai_backend.ApaiArm:
    """The arm port on APAI's mock arm, at REST and not connected yet."""
    return apai_backend.build_apai_arm(config, profile, clock)


@pytest.fixture
def connected(arm: apai_backend.ApaiArm) -> apai_backend.ApaiArm:
    """The mock arm connected, sent to HOME, with referenced jaws."""
    arm.connect()
    arm.gripper_reference()
    arm.send_target(HOME)  # the mock arm is there at once
    return arm


def tip_of(profile: apai_backend.ApaiProfile, q: Sequence[float]) -> np.ndarray:
    return profile.kinematics.fk(np.array(q, dtype=float))[:3, 3]


# Building the checkers


def test_three_checkers_stand_on_unmodified_harnesses(
    checkers: CheckerSet, config: RobotAppConfig, profile: apai_backend.ApaiProfile
) -> None:
    from apai.safety.harness import SafetyHarness

    limit_sets = (config.limits.agent, config.limits.human, config.limits.recovery)
    assert len({id(checker.harness) for checker in checkers.all()}) == 3
    for checker, limit_set in zip(checkers.all(), limit_sets):
        assert isinstance(checker, Checker)
        assert (checker.name, checker.version) == (limit_set.name, limit_set.version)
        harness = checker.harness
        assert type(harness) is SafetyHarness
        # No method is replaced on the object, as an envelope wrapper would do.
        assert not {"approve", "vet_pose", "vet_step"} & set(vars(harness))
        assert harness.occupancy is None
        assert harness.limits.max_joint_vel == limit_set.max_joint_speed
        assert harness.limits.contact.enforcing
        assert harness.limits.contact.min_reads_per_cycle == 2
    profile_watchdog_s = profile.apai_cfg.safety.watchdog_s
    assert checkers.agent.harness.limits.watchdog_s == profile_watchdog_s
    assert checkers.human.harness.limits.watchdog_s == profile_watchdog_s
    assert checkers.recovery.harness.limits.watchdog_s == math.inf
    assert checkers.for_owner("recovery") is checkers.recovery


def test_build_refuses_a_config_with_another_joint_count(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> None:
    five = make_config(joints=5)
    with pytest.raises(ValueError, match=r"profile has 6 joints but n_joints is 5"):
        apai_backend.build_apai_checkers(five, profile, clock)
    with pytest.raises(ValueError, match=r"limits\.agent has 5 joints"):
        apai_backend.ApaiChecker(
            five.limits.agent,
            checkers.agent.harness,
            clock,
            period_s=STEP_S,
            table_clearance_m=0.02,
        )


def test_checker_needs_the_contact_monitor(
    profile: apai_backend.ApaiProfile, config: RobotAppConfig, clock: FakeClock
) -> None:
    from apai.safety.harness import SafetyHarness, SafetyLimits

    limits = SafetyLimits.from_config(profile.apai_cfg.safety)
    assert not limits.contact.enabled  # the mock profile leaves it off
    harness = SafetyHarness(limits, kinematics=profile.kinematics)
    with pytest.raises(ValueError, match=r"contact monitor is off or only measures"):
        apai_backend.ApaiChecker(
            config.limits.agent, harness, clock, period_s=STEP_S, table_clearance_m=0.02
        )


def test_profile_overrides_reach_the_harness(
    config: RobotAppConfig, clock: FakeClock
) -> None:
    monitor = {"enabled": True, "mode": "measure", "lag_s": 0.3}
    overrides = {"safety": {"table_clearance": 0.05, "contact_monitor": monitor}}
    changed = apai_backend.load_apai_profile("mock", overrides)
    checkers = apai_backend.build_apai_checkers(config, changed, clock)
    limits = checkers.human.harness.limits
    assert limits.table_clearance == 0.05
    assert limits.contact.lag_s == 0.3
    # A monitor that only measures cannot stop a move: the robot app enforces.
    assert limits.contact.mode == "enforce"


def with_speed(config: RobotAppConfig, owner: str, speed: float) -> RobotAppConfig:
    """The configuration with one limit set's joint speed changed."""
    limit_set = dataclasses.replace(
        getattr(config.limits, owner), max_joint_speed=speed
    )
    limits = dataclasses.replace(config.limits, **{owner: limit_set})
    return dataclasses.replace(config, limits=limits)


@pytest.mark.parametrize(
    ("owner", "speed"), [("agent", 3.0), ("human", 1.5), ("recovery", 0.9)]
)
def test_a_limit_set_faster_than_the_profiles_speed_cap_is_refused(
    config: RobotAppConfig,
    rs_profile: apai_backend.ApaiProfile,
    clock: FakeClock,
    owner: str,
    speed: float,
) -> None:
    assert rs_profile.apai_cfg.safety.max_joint_vel == 0.8
    with pytest.raises(
        ValueError,
        match=rf"limits\.{owner}\.max_joint_speed {speed:g} rad/s is above the APAI "
        r"profile's speed cap of 0\.8 rad/s \(safety\.max_joint_vel\)",
    ):
        apai_backend.build_apai_checkers(
            with_speed(config, owner, speed), rs_profile, clock
        )


def test_limit_sets_up_to_the_profiles_speed_cap_keep_their_own_speed(
    config: RobotAppConfig, rs_profile: apai_backend.ApaiProfile, clock: FakeClock
) -> None:
    at_the_cap = with_speed(config, "human", 0.8)
    checkers = apai_backend.build_apai_checkers(at_the_cap, rs_profile, clock)
    speeds = [checker.harness.limits.max_joint_vel for checker in checkers.all()]
    assert speeds == [0.4, 0.8, 0.1]
    # The cap of a rig is given to the profile, and then binds every limit set.
    rig = apai_backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.1}})
    with pytest.raises(ValueError, match=r"limits\.agent\.max_joint_speed 0\.4 rad/s"):
        apai_backend.build_apai_checkers(at_the_cap, rig, clock)


@pytest.mark.parametrize("name", ["APAI_ROBOT", "APAI_BASE"])
def test_load_refuses_an_environment_that_selects_another_configuration(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "something")
    with pytest.raises(ValueError, match=rf"unset {name}"):
        apai_backend.load_apai_profile("mock")


# Watchdog


def test_recovery_starts_unfed_while_the_agent_waits_for_evidence(
    checkers: CheckerSet,
) -> None:
    recovery, agent = checkers.recovery, checkers.agent
    recovery.begin_motion(recovery.halt_count())
    recovery.check_setpoint(HOME, NEAR_HOME, STEP_S)
    recovery.end_motion()

    refusal_of(
        lambda: agent.begin_motion(agent.halt_count()),
        r"perception watchdog: no observation yet",
    )
    refusal_of(
        lambda: agent.check_setpoint(HOME, NEAR_HOME, STEP_S),
        r"perception watchdog: no observation yet",
    )
    agent.feed_watchdog()
    agent.begin_motion(agent.halt_count())
    agent.check_setpoint(HOME, NEAR_HOME, STEP_S)
    agent.end_motion()


def test_watchdog_takes_evidence_times_on_the_app_clock(
    checkers: CheckerSet, clock: FakeClock
) -> None:
    agent = checkers.agent

    def begin() -> None:
        agent.begin_motion(agent.halt_count())

    agent.feed_watchdog(clock.monotonic() + 1.0)  # stamped in the future: ignored
    refusal_of(begin, r"no observation yet")
    agent.feed_watchdog(clock.monotonic() - 60.0)
    refusal_of(begin, r"perception watchdog: last observation 60\.\ds old")
    agent.feed_watchdog(clock.monotonic() - 1.0)
    begin()
    agent.end_motion()


# Setpoints


def test_setpoint_inside_every_limit_passes(checkers: CheckerSet) -> None:
    for checker in checkers.all():
        start_motion(checker)
        checker.check_setpoint(HOME, NEAR_HOME, STEP_S)
        checker.end_motion()


def test_setpoint_outside_the_limit_set_is_refused(checkers: CheckerSet) -> None:
    human = checkers.human
    start_motion(human)
    refusal = refusal_of(
        lambda: human.check_setpoint(
            moved(HOME, 0, 1.999), moved(HOME, 0, 2.004), 0.02
        ),
        r"joint 1 target 2\.004 rad is outside the human joint limits "
        r"-2\.000\.\.2\.000",
    )
    assert refusal.reason == str(refusal)


def test_joint_outside_the_limit_set_may_only_move_back(checkers: CheckerSet) -> None:
    recovery = checkers.recovery
    start_motion(recovery)
    # The second joint a little under its stop, where it can read at rest.
    under, back, further = (moved(REST, 1, v) for v in (-0.002, -0.001, -0.003))
    recovery.check_setpoint(under, back, STEP_S)
    recovery.check_path(under, [back, REST], STEP_S)
    for target in (under, further):
        refusal_of(
            lambda target=target: recovery.check_setpoint(under, target, STEP_S),
            r"joint 2 target -0\.00\d rad is outside the recovery joint limits",
        )
    refusal_of(lambda: recovery.check_pose(under), r"outside the recovery joint limits")


def test_setpoint_inside_the_harness_joint_margin_is_refused(
    checkers: CheckerSet,
) -> None:
    human = checkers.human
    start_motion(human)
    # Inside the limit set's joint box, and inside the 0.02 rad the harness keeps
    # clear of the arm model's limit.
    refusal_of(
        lambda: human.check_setpoint(moved(HOME, 4, 1.556), moved(HOME, 4, 1.56), 0.02),
        r"joint 5 target 1\.560 rad outside \[-1\.550, 1\.550\]",
    )


def test_each_owner_has_its_own_speed_limit(checkers: CheckerSet) -> None:
    for checker in checkers.all():
        start_motion(checker)
    slow, fast = moved(HOME, 0, 0.003), moved(HOME, 0, 0.012)  # 0.15 and 0.6 rad/s

    refusal_of(
        lambda: checkers.recovery.check_setpoint(HOME, slow, STEP_S),
        r"joint 1 velocity 0\.15 rad/s exceeds 0\.10",
    )
    checkers.agent.check_setpoint(HOME, slow, STEP_S)
    refusal_of(
        lambda: checkers.agent.check_setpoint(HOME, fast, STEP_S),
        r"joint 1 velocity 0\.60 rad/s exceeds 0\.40",
    )
    checkers.human.check_setpoint(HOME, fast, STEP_S)
    refusal_of(
        lambda: checkers.human.check_setpoint(HOME, moved(HOME, 0, 0.02), STEP_S),
        r"joint 1 velocity 1\.00 rad/s exceeds 0\.80",
    )


@pytest.mark.parametrize(
    ("q_prev", "q_next", "dt", "match"),
    [
        (HOME, moved(HOME, 0, math.nan), STEP_S, r"not finite"),
        (moved(HOME, 0, math.inf), HOME, STEP_S, r"not finite"),
        (HOME, HOME[:5], STEP_S, r"needs 6 values"),
        (HOME, ["a"] * 6, STEP_S, r"list of numbers"),
        # The harness alone would skip its speed check for these step times.
        (HOME, moved(HOME, 0, 1.0), 0.0, r"step time must be above 0"),
        (HOME, moved(HOME, 0, 1.0), -0.02, r"step time must be above 0"),
        (HOME, NEAR_HOME, math.nan, r"step time must be above 0"),
        (HOME, NEAR_HOME, True, r"step time must be a number"),
    ],
)
def test_malformed_setpoint_is_refused(
    checkers: CheckerSet, q_prev: Any, q_next: Any, dt: Any, match: str
) -> None:
    recovery = checkers.recovery
    start_motion(recovery)
    refusal_of(lambda: recovery.check_setpoint(q_prev, q_next, dt), match)


# Halt and latch


def test_halt_stops_a_move_and_an_old_halt_count_cannot_start_one(
    checkers: CheckerSet,
) -> None:
    checker = checkers.recovery
    count = checker.halt_count()
    checker.begin_motion(count)
    checker.check_setpoint(HOME, NEAR_HOME, STEP_S)

    assert checker.halt("operator STOP") == count + 1
    assert checker.halt_count() == count + 1
    refusal_of(
        lambda: checker.check_setpoint(HOME, NEAR_HOME, STEP_S),
        r"halted: operator STOP",
    )
    checker.end_motion()

    # A move admitted before the halt quotes the old count and never starts.
    refusal_of(lambda: checker.begin_motion(count), r"halt received")
    refusal_of(
        lambda: checker.check_setpoint(HOME, NEAR_HOME, STEP_S),
        r"halted: operator STOP",
    )

    checker.begin_motion(checker.halt_count())
    checker.check_setpoint(HOME, NEAR_HOME, STEP_S)
    checker.end_motion()


def test_halt_on_one_checker_leaves_the_others_alone(checkers: CheckerSet) -> None:
    before = [checker.halt_count() for checker in checkers.all()]
    checkers.agent.halt("agent stop")
    after = [checker.halt_count() for checker in checkers.all()]
    assert after == [before[0] + 1, before[1], before[2]]


def test_latch_refuses_everything_until_reset(checkers: CheckerSet) -> None:
    checker = checkers.recovery
    count = checker.halt_count()
    assert not checker.latched()
    checker.latch("motor fault")
    assert checker.latched()
    refused: list[Callable[[], Any]] = [
        lambda: checker.begin_motion(count),
        lambda: checker.check_setpoint(HOME, NEAR_HOME, STEP_S),
        lambda: checker.check_pose(HOME),
        lambda: checker.check_path(HOME, [NEAR_HOME], STEP_S),
    ]
    for call in refused:
        refusal_of(call, r"e-stop latched")

    checker.reset_latch()
    assert not checker.latched()
    assert checker.halt_count() == count  # the latch is not a halt
    assert not checker.harness.motion_active  # the reset starts no motion
    checker.begin_motion(count)
    checker.check_setpoint(HOME, NEAR_HOME, STEP_S)
    checker.end_motion()


def test_reset_keeps_a_halt_raised_while_latched(checkers: CheckerSet) -> None:
    checker = checkers.recovery
    count = checker.halt_count()
    checker.latch("motor fault")
    checker.halt("operator STOP")
    checker.reset_latch()
    refusal_of(lambda: checker.begin_motion(count), r"halt received")


# Envelope


def test_envelope_has_no_way_back_in(checkers: CheckerSet) -> None:
    agent, recovery = checkers.agent, checkers.recovery
    # The last joint only turns the gripper about its own axis.
    outside, back = moved(HOME, 5, 1.6), moved(HOME, 5, 1.599)
    edge, beyond = moved(HOME, 5, 1.5), moved(HOME, 5, 1.501)
    start_motion(agent)

    refusal_of(
        lambda: agent.check_setpoint(outside, back, STEP_S),
        r"the step starts outside the agent envelope, which cannot be re-entered: "
        r"joint 6 at 1\.600 rad, envelope -1\.500\.\.1\.500",
    )
    refusal_of(
        lambda: agent.check_setpoint(edge, beyond, STEP_S),
        r"the target is outside the agent envelope",
    )
    refusal_of(lambda: agent.check_pose(outside), r"outside the agent envelope")
    refusal_of(
        lambda: agent.check_path(outside, [back], STEP_S),
        r"setpoint 0: the step starts outside the agent envelope",
    )
    agent.check_setpoint(edge, moved(HOME, 5, 1.499), STEP_S)

    # The recovery checker has no envelope: the step back is its to take.
    start_motion(recovery)
    recovery.check_setpoint(outside, back, STEP_S)
    recovery.check_pose(outside)


# Poses and paths


def test_every_owner_accepts_the_rest_pose_and_a_raised_pose(
    checkers: CheckerSet,
) -> None:
    # A strict harness vet refuses REST: two of its joints lie on their stops,
    # inside the joint margin of the agent's and the human's harness.
    assert checkers.agent.harness.vet_pose(np.array(REST)) is not None
    for checker in checkers.all():
        checker.check_pose(REST)
        checker.check_pose(HOME)


def test_pose_outside_the_limits_is_refused(checkers: CheckerSet) -> None:
    refusal_of(
        lambda: checkers.human.check_pose(moved(HOME, 0, 2.5)),
        r"joint 1 target 2\.500 rad is outside the human joint limits",
    )
    # Inside the recovery joint box, with the gripper tip beside the workspace.
    refusal_of(
        lambda: checkers.recovery.check_pose(moved(HOME, 0, 1.5)),
        r"outside workspace",
    )
    refusal_of(lambda: checkers.recovery.check_pose(BELOW), r"below table clearance")
    refusal_of(lambda: checkers.recovery.check_pose(HOME[:3]), r"needs 6 values")


def test_recovery_path_reaches_the_rest_pose_on_the_joint_stops(
    checkers: CheckerSet,
) -> None:
    # 1.2 rad of travel in 25 s peaks at 0.09 rad/s, under the recovery speed limit.
    path = min_jerk_path(HOME, REST, 250)
    checkers.recovery.check_path(HOME, path, 0.1)
    # The agent's harness keeps its joint margin, so the stops are out of its reach.
    refusal_of(
        lambda: checkers.agent.check_path(HOME, path, 0.1),
        r"setpoint \d+: joint 2 target 0\.0\d\d rad outside \[0\.020, 3\.120\]",
    )


def test_path_refusal_names_the_setpoint(checkers: CheckerSet) -> None:
    recovery = checkers.recovery
    path = [moved(HOME, 0, 0.001), moved(HOME, 0, 0.002), moved(HOME, 0, 0.022)]
    refusal_of(
        lambda: recovery.check_path(HOME, path, STEP_S),
        r"setpoint 2: joint 1 velocity 1\.00 rad/s exceeds 0\.10",
    )
    refusal_of(
        lambda: recovery.check_path(HOME, [NEAR_HOME, HOME[:4]], STEP_S),
        r"setpoint 1: a joint pose needs 6 values",
    )
    refusal_of(lambda: recovery.check_path(HOME, [], STEP_S), r"no setpoints")
    refusal_of(lambda: recovery.check_path(HOME, path, 0.0), r"step time")
    # A path check starts no motion and needs no fed watchdog.
    checkers.agent.check_path(HOME, path[:2], STEP_S)
    assert not checkers.agent.harness.motion_active


# Grasp region


@pytest.mark.parametrize("z_max", [None, 0.05])
def test_grasp_region_lets_the_tip_below_the_table_clearance(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile, z_max: float | None
) -> None:
    agent = checkers.agent
    x, y, z = tip_of(profile, BELOW)
    assert tip_of(profile, ABOVE)[2] > 0.02 > z > 0.0
    start_motion(agent)

    def descend() -> None:
        agent.check_setpoint(ABOVE, BELOW, 1.0)

    refusal_of(descend, r"below table clearance 0\.020 outside the grasp exemption")
    agent.set_grasp_region(GraspRegion((x, y), 0.05, 0.0, z_max))
    descend()
    # A region elsewhere in the workspace box opens nothing where the tip is.
    agent.set_grasp_region(GraspRegion((x - 0.2, y), 0.05, 0.0, z_max))
    refusal_of(descend, r"below table clearance")
    agent.set_grasp_region(GraspRegion((x, y), 0.05, 0.0, z_max))
    descend()
    agent.set_grasp_region(None)
    refusal_of(descend, r"below table clearance")


def test_grasp_region_ends_at_its_ceiling(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile
) -> None:
    agent = checkers.agent
    x, y, z = tip_of(profile, BELOW)
    start_motion(agent)
    agent.set_grasp_region(GraspRegion((x, y), 0.05, 0.0, z - 0.005))
    refusal_of(
        lambda: agent.check_setpoint(ABOVE, BELOW, 1.0), r"below table clearance"
    )
    # Closing a region with a ceiling restores the rule, not an older region.
    agent.set_grasp_region(None)
    refusal_of(
        lambda: agent.check_setpoint(ABOVE, BELOW, 1.0), r"below table clearance"
    )


@pytest.mark.parametrize(
    ("region", "match"),
    [
        (GraspRegion((0.3, math.nan), 0.05, 0.0, None), r"not a number"),
        (GraspRegion((0.3, 0.0), 0.05, 0.0, math.inf), r"not a number"),
        (GraspRegion((0.3, 0.0), 0.0, 0.0, None), r"radius above 0"),
    ],
)
def test_malformed_grasp_region_is_refused(
    checkers: CheckerSet, region: GraspRegion, match: str
) -> None:
    refusal_of(lambda: checkers.agent.set_grasp_region(region), match)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"center_xy": (0.0, 0.3), "radius_m": 50.0}, r"radius is 50 m, the limit"),
        ({"z_min": -5.0}, r"reaches 5 m below the table, the limit is 0\.06 m"),
        ({"z_min": 0.05, "z_max": 0.01}, r"ends at z = 0\.01 m, below its lowest"),
        ({"center_xy": (5.8, 0.0)}, r"centre \(5\.8, 0\) m is outside the workspace"),
    ],
    ids=["radius", "depth", "upside down", "centre"],
)
def test_a_grasp_region_beyond_the_thors_bounds_opens_nothing(
    checkers: CheckerSet,
    profile: apai_backend.ApaiProfile,
    change: dict[str, Any],
    match: str,
) -> None:
    agent = checkers.agent
    x, y, _ = tip_of(profile, BELOW)
    region = dataclasses.replace(GraspRegion((x, y), 0.05, 0.0, None), **change)
    start_motion(agent)
    refusal_of(lambda: agent.set_grasp_region(region), match)
    refusal_of(lambda: agent.check_grasp_region(region), match)
    assert agent.harness._grasp_exempt is None
    refusal_of(
        lambda: agent.check_setpoint(ABOVE, BELOW, 1.0), r"below table clearance"
    )


def test_the_grasp_regions_apais_skills_set_are_taken(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile
) -> None:
    agent = checkers.agent
    x, y, _ = tip_of(profile, BELOW)
    table_z = agent.harness.limits.table_z
    start_motion(agent)
    for radius, z_min in ((0.07, table_z - 0.06), (0.07, 0.03), (0.25, 0.0)):
        region = GraspRegion((x, y), radius, z_min, None)
        agent.check_grasp_region(region)
        agent.set_grasp_region(region)
    agent.set_grasp_region(GraspRegion((x, y), 0.07, table_z - 0.06, None))
    agent.check_setpoint(ABOVE, BELOW, 1.0)


def test_the_bounds_of_a_grasp_region_come_from_the_thors_configuration(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> None:
    tight = dataclasses.replace(
        config, grasp=GraspConfig(max_radius_m=0.06, max_depth_m=0.0)
    )
    agent = apai_backend.build_apai_checkers(tight, profile, clock).agent
    x, y, _ = tip_of(profile, BELOW)
    refusal_of(
        lambda: agent.check_grasp_region(GraspRegion((x, y), 0.07, 0.0, None)),
        r"radius is 0\.07 m, the limit is 0\.06 m",
    )
    refusal_of(
        lambda: agent.check_grasp_region(GraspRegion((x, y), 0.05, -0.01, None)),
        r"reaches 0\.01 m below the table, the limit is 0 m",
    )
    agent.check_grasp_region(GraspRegion((x, y), 0.05, 0.0, None))


def test_tip_is_blocked_below_the_clearance_or_inside_the_region(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile
) -> None:
    recovery = checkers.recovery
    x, y, z = tip_of(profile, HOME)
    around = GraspRegion((x, y), 0.05, z - 0.05, None)
    beside = GraspRegion((x + 0.2, y), 0.05, 0.0, None)
    under = GraspRegion((x, y), 0.05, 0.0, z - 0.05)
    over = GraspRegion((x, y), 0.05, z + 0.05, None)

    assert recovery.tip_blocked(HOME, None) is None
    assert recovery.tip_blocked(HOME, around) == (
        "the gripper tip is inside the last grasp region"
    )
    for region in (beside, under, over):
        assert recovery.tip_blocked(HOME, region) is None
    below = recovery.tip_blocked(BELOW, None)
    assert below is not None
    assert re.search(r"z=0\.011 m, below the table clearance 0\.020 m", below)
    assert recovery.tip_blocked(moved(HOME, 0, math.nan), None) is not None


# Contact spots


def test_contact_spot_keeps_the_tip_out_until_it_is_cleared(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile
) -> None:
    recovery = checkers.recovery
    away = moved(HOME, 2, 1.0)
    assert np.linalg.norm(tip_of(profile, away) - tip_of(profile, HOME)) > 0.08
    start_motion(recovery)

    recovery.note_contact(HOME)
    assert len(recovery.harness.contact_zones) == 1
    assert checkers.agent.harness.contact_zones == []  # each owner keeps its own
    recovery.check_pose(HOME)  # the arm may start inside the spot
    recovery.check_setpoint(HOME, away, 10.0)  # leaving the spot is allowed
    # The spot is not blamed on the checker's own owner: any move can have met it.
    refusal_of(
        lambda: recovery.check_setpoint(away, HOME, 10.0),
        r"would enter the contact zone .*: contact detected at joint unknown "
        r"during an earlier move",
    )
    refusal_of(
        lambda: recovery.check_path(away, [HOME], 10.0),
        r"setpoint 0: TCP would enter the contact zone",
    )

    recovery.clear_contact_spots()
    assert recovery.harness.contact_zones == []
    recovery.check_setpoint(away, HOME, 10.0)
    recovery.check_pose(HOME)


def test_a_pose_inside_a_contact_spot_is_still_checked_for_everything_else(
    checkers: CheckerSet,
) -> None:
    for checker in (checkers.agent, checkers.recovery):
        checker.note_contact(BELOW)
        # The pose vet names the table first, so the spot excuses nothing.
        with pytest.raises(Refusal, match=r"below table clearance"):
            checker.check_pose(BELOW)
        checker.clear_contact_spots()
    agent = checkers.agent
    outside = moved(HOME, 0, 1.2)
    agent.note_contact(outside)
    refusal_of(lambda: agent.check_pose(outside), r"outside the agent envelope")
    agent.note_contact(HOME)
    agent.latch("motor fault")
    refusal_of(lambda: agent.check_pose(HOME), r"e-stop latched")


# Poses with the gripper tip above the table clearance and the origin of the
# last link less than 1 cm above the table, for the agent's and the recovery
# checker. The pose vet names a contact spot before it looks at the links.
LOW_LINK = {
    "agent": (-0.862, 2.365, 0.403, -0.979, -0.946, 1.395),
    "recovery": (-0.78, 2.438, 0.802, -1.299, -0.953, 1.494),
}


@pytest.mark.parametrize("owner", ["agent", "recovery"])
def test_a_low_link_is_refused_at_a_pose_inside_a_contact_spot_too(
    checkers: CheckerSet, profile: apai_backend.ApaiProfile, owner: str
) -> None:
    checker, pose = getattr(checkers, owner), LOW_LINK[owner]
    assert tip_of(profile, pose)[2] > 0.02
    low_link = r"link/joint 5 at z=0\.00\d would hit the table"
    refusal_of(lambda: checker.check_pose(pose), low_link)
    checker.note_contact(pose)
    # Now the pose vet stops at the spot; the step vet still finds the link.
    assert checker.harness.vet_pose(pose).startswith("TCP would enter the contact")
    refusal_of(lambda: checker.check_pose(pose), low_link)


def test_contact_spot_at_a_pose_that_is_not_one_is_dropped(
    checkers: CheckerSet,
) -> None:
    checkers.agent.note_contact(HOME[:2])
    checkers.agent.note_contact(moved(HOME, 0, math.nan))
    assert checkers.agent.harness.contact_zones == []


def test_harness_refusal_texts_do_not_pile_up(checkers: CheckerSet) -> None:
    recovery = checkers.recovery
    start_motion(recovery)
    for _ in range(600):
        with pytest.raises(Refusal):
            recovery.check_setpoint(HOME, moved(HOME, 0, 1.0), STEP_S)
    assert len(recovery.harness.violations) <= 256


# APAI's errors


def test_apai_errors_map_to_the_robot_app_errors() -> None:
    from apai.types import ContactDetected, GripperFault, MotionHalted, SafetyViolation

    contact = apai_backend._refusal(ContactDetected("blocked", {"joint": 2}))
    assert type(contact) is ContactStop
    assert contact.reason == "blocked"
    for error in (MotionHalted("halted"), SafetyViolation("no"), GripperFault("jaw")):
        mapped = apai_backend._refusal(error)
        assert type(mapped) is Refusal
        assert mapped.reason == str(error)


# Contact watch


def stream(
    arm: ArmPort,
    watch: ContactWatch,
    clock: FakeClock,
    poses: Sequence[Sequence[float]],
) -> list[list[JointReading]]:
    """Send each pose and run the watch after it, one step apart."""
    steps = []
    for index, pose in enumerate(poses):
        arm.send_target(pose)
        steps.append(watch.after_send(index, pose, clock.monotonic() + STEP_S))
        clock.advance(STEP_S)
    return steps


def ramp(joint: int, per_step: float, steps: int) -> list[list[float]]:
    return [moved(HOME, joint, HOME[joint] + per_step * i) for i in range(1, steps + 1)]


def test_contact_watch_reads_every_joint_into_the_history(
    checkers: CheckerSet, connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    history = History()
    watch = checkers.agent.contact_watch(connected, HOME, history)
    assert isinstance(watch, ContactWatch)
    began = clock.monotonic()
    poses = ramp(0, 0.004, 5)
    steps = stream(connected, watch, clock, poses)
    watch.finish()
    watch.finish()

    for index, (readings, pose) in enumerate(zip(steps, poses)):
        assert [reading.joint for reading in readings] == list(range(6))
        assert [reading.value for reading in readings] == pytest.approx(pose)
        sent_at = pytest.approx(began + index * STEP_S)
        assert all(r.t_start == r.t_end == sent_at and r.valid for r in readings)
    assert history.readings == [reading for step in steps for reading in step]


def test_contact_watch_works_without_a_history(
    checkers: CheckerSet, connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    watch = checkers.agent.contact_watch(connected, HOME, None)
    assert len(stream(connected, watch, clock, ramp(0, 0.004, 2))[1]) == 6


def test_contact_watch_stops_on_a_blocked_joint(
    checkers: CheckerSet, connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    history = History()
    watch = checkers.human.contact_watch(connected, HOME, history)
    stream(connected, watch, clock, ramp(0, 0.01, 5))  # a free arm follows its targets
    connected.backend.block_joint(0)
    with pytest.raises(ContactStop, match=r"contact detected at joint 1") as stopped:
        stream(connected, watch, clock, ramp(0, 0.01, 80)[5:])
    watch.finish()

    assert "stalled" in stopped.value.reason
    blocked_at = HOME[0] + 0.05
    first_joint = [r.value for r in history.readings if r.joint == 0]
    assert first_joint[-1] == pytest.approx(blocked_at)
    # The watch only reports: the hold at the measured pose is the executor's.
    assert connected.last_target()[0][0] > blocked_at + 0.1
    assert checkers.human.harness.contact_zones == []


def test_contact_watch_refuses_when_a_joint_gives_no_feedback(
    checkers: CheckerSet, connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    class Deaf:
        """The arm with one joint that never answers."""

        def read_joint(self, joint: int) -> JointReading:
            if joint == 2:
                raise ReadFailed("no answer")
            return connected.read_joint(joint)

    history = History()
    watch = checkers.recovery.contact_watch(Deaf(), HOME, history)
    steps = []
    with pytest.raises(Refusal, match=r"lost joint 3 feedback for 3\d\d ms") as lost:
        for index in range(40):
            deadline = clock.monotonic() + STEP_S
            steps.append(watch.after_send(index, HOME, deadline))
            clock.advance(STEP_S)
    assert type(lost.value) is FeedbackLost

    # 0.35 s without a good reading of the joint, counted in steps of the move.
    assert len(steps) in (17, 18)
    assert all([r.joint for r in step] == [0, 1, 3, 4, 5] for step in steps)
    # The joints read in the refused step before the deaf one are kept too.
    assert [r.joint for r in history.readings[-2:]] == [0, 1]


def test_reading_the_monitor_cannot_believe_is_kept_as_not_valid(
    checkers: CheckerSet, arm: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    arm.connect()
    history = History()
    # The watch is told the move starts at HOME while the arm is at REST: two
    # joints read further from HOME than a joint can travel in one step.
    watch = checkers.recovery.contact_watch(arm, HOME, history)
    readings = watch.after_send(0, HOME, clock.monotonic() + STEP_S)
    assert [reading.value for reading in readings] == list(REST)
    valid = [reading.valid for reading in readings]
    assert valid == [True, False, False, True, True, True]
    assert history.readings == readings


def test_contact_watch_refuses_a_start_pose_that_is_not_one(
    checkers: CheckerSet, connected: apai_backend.ApaiArm
) -> None:
    refusal_of(
        lambda: checkers.agent.contact_watch(connected, HOME[:4], None),
        r"needs 6 values",
    )


def test_watch_is_paced_unless_the_clock_is_a_fake_one(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile
) -> None:
    from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock

    clock = SystemClock()
    arm = apai_backend.build_apai_arm(config, profile, clock)
    arm.connect()
    checkers = apai_backend.build_apai_checkers(config, profile, clock)
    watch = checkers.recovery.contact_watch(arm, REST, None)
    # Paced: a step whose deadline has passed gets the configured reads only.
    assert len(watch.after_send(0, REST, clock.monotonic())) == 2
    watch.finish()


def test_paced_watch_reads_the_configured_joints_per_step(
    config: RobotAppConfig,
    profile: apai_backend.ApaiProfile,
    clock: FakeClock,
    connected: apai_backend.ApaiArm,
) -> None:
    paced = apai_backend.build_apai_checkers(config, profile, clock, paced=True)
    history = History()
    watch = paced.recovery.contact_watch(connected, HOME, history)
    # A deadline that has passed leaves room for the configured reads only.
    joints = [
        [r.joint for r in watch.after_send(index, HOME, clock.monotonic())]
        for index in range(3)
    ]
    assert joints == [[0, 1], [2, 3], [4, 5]]
    assert [r.joint for r in history.readings] == [0, 1, 2, 3, 4, 5]
    # With time left in the step the watch reads on, as APAI's streams do.
    assert len(watch.after_send(3, HOME, clock.monotonic() + 30.0)) == 6


# The arm port over APAI's mock arm


def test_arm_port_sends_targets_and_reads_joints_with_times(
    arm: apai_backend.ApaiArm, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(arm, ArmPort)
    assert arm.n_joints == 6
    assert arm.joint_names == [f"joint{i}" for i in range(1, 7)]
    assert not arm.torque_on()
    assert arm.last_target() is None
    arm.connect()
    assert arm.torque_on()
    assert arm.faults() == []
    assert arm.temperatures() is None
    # A real arm starts at rest, so the mock arm is placed there too.
    assert [reading.value for reading in arm.read_all()] == list(REST)

    target = moved(HOME, 0, 0.1)
    arm.send_target(target)
    sent_at = clock.monotonic()
    assert [command.tolist() for command in arm.backend.commands] == [target]
    assert arm.last_target() == (target, sent_at)

    clock.advance(0.5)
    reading = arm.read_joint(1)
    assert reading == JointReading(1, 1.2, clock.monotonic(), clock.monotonic())

    # A read that takes time is stamped with its start and its end.
    fresh = arm.backend.read_positions_for_monitor

    def slow_read() -> np.ndarray:
        clock.advance(0.008)
        return fresh()

    monkeypatch.setattr(arm.backend, "read_positions_for_monitor", slow_read)
    before = clock.monotonic()
    reading = arm.read_joint(0)
    assert (reading.joint, reading.value) == (0, 0.1)
    assert reading.t_start == before
    assert reading.t_end == pytest.approx(before + 0.008)

    before = clock.monotonic()
    readings = arm.read_all()
    assert [r.joint for r in readings] == list(range(6))
    assert [r.value for r in readings] == pytest.approx(target)
    assert all(r.t_start == before for r in readings)
    assert all(r.t_end == pytest.approx(before + 0.008) for r in readings)
    assert arm.last_target() == (target, sent_at)  # reading sends nothing


def test_failed_reads_raise_read_failed(
    connected: apai_backend.ApaiArm, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> np.ndarray:
        raise RuntimeError("CAN feedback lost")

    monkeypatch.setattr(connected.backend, "read_positions_for_monitor", broken)
    with pytest.raises(ReadFailed, match=r"joint3: CAN feedback lost"):
        connected.read_joint(2)
    with pytest.raises(ReadFailed, match=r"CAN feedback lost"):
        connected.read_all()
    sent = len(connected.backend.commands)
    with pytest.raises(ReadFailed):
        connected.hold_measured()
    assert len(connected.backend.commands) == sent  # no hold at a pose not measured

    bad = np.array([0.0, math.nan, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(connected.backend, "read_positions_for_monitor", lambda: bad)
    with pytest.raises(ReadFailed, match=r"joint2: read nan"):
        connected.read_joint(1)
    with pytest.raises(ReadFailed, match=r"the joints read as"):
        connected.read_all()
    with pytest.raises(ValueError, match=r"no joint index 6"):
        connected.read_joint(6)


def test_hold_measured_commands_the_freshly_measured_pose(
    connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    connected.send_target(HOME)
    connected.backend.block_joint(0)
    connected.send_target(moved(HOME, 0, 0.3))  # the blocked joint stays behind
    clock.advance(1.0)
    connected.hold_measured()
    assert connected.backend.commands[-1].tolist() == pytest.approx(list(HOME))
    assert connected.last_target() == (pytest.approx(list(HOME)), clock.monotonic())


def test_send_target_refuses_what_is_not_a_setpoint_or_cannot_be_sent(
    arm: apai_backend.ApaiArm, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ArmFault, match=r"not connected or its motors are off"):
        arm.send_target(HOME)
    arm.connect()
    for bad in (HOME[:5], moved(HOME, 2, math.nan)):
        with pytest.raises(ValueError, match=r"needs 6 finite values"):
            arm.send_target(bad)

    def no_bus(_q: np.ndarray) -> None:
        raise OSError("bus off")

    monkeypatch.setattr(arm.backend, "send_joint_target", no_bus)
    with pytest.raises(ArmFault, match=r"did not take a joint target: bus off"):
        arm.send_target(HOME)
    assert arm.last_target() is None
    with pytest.raises(RuntimeError, match=r"connected once"):
        arm.connect()


def test_release_torque_is_the_only_path_to_disconnect(
    config: RobotAppConfig, profile: apai_backend.ApaiProfile, clock: FakeClock
) -> None:
    from apai.control.mock_arm import MockArm

    recorder = Recorder(MockArm(profile.arm_cfg, profile.kinematics))
    arm = apai_backend.ApaiArm(
        recorder, config, clock, gripper_cfg=profile.arm_cfg.get("gripper")
    )
    cancel = threading.Event()

    def jaw_commands() -> None:
        assert arm.gripper_command(1.0, 0.5, "plain", cancel).status == "reached"
        recorder.wrapped.object_stop_frac = 0.5
        closed = arm.gripper_command(0.0, 0.5, "close_until_contact", cancel)
        assert closed.status == "stalled"

    def watched_step() -> None:
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        watch = checkers.recovery.contact_watch(arm, HOME, History())
        watch.after_send(0, HOME, clock.monotonic() + STEP_S)
        watch.finish()

    # Every method of the arm port but release_torque, in an order that works.
    calls: dict[str, Callable[[], Any]] = {
        "connect": arm.connect,
        "torque_on": arm.torque_on,
        "send_target": lambda: arm.send_target(HOME),
        "last_target": arm.last_target,
        "read_joint": lambda: arm.read_joint(0),
        "read_all": arm.read_all,
        "hold_measured": arm.hold_measured,
        "release_confirmed": arm.release_confirmed,
        "faults": arm.faults,
        "temperatures": arm.temperatures,
        "gripper_referenced": arm.gripper_referenced,
        "gripper_reference": arm.gripper_reference,
        "gripper_read": arm.gripper_read,
        "gripper_last_target": arm.gripper_last_target,
        "gripper_command": jaw_commands,
        "gripper_light_hold": arm.gripper_light_hold,
    }
    port_methods = {
        name
        for name, _ in inspect.getmembers(ArmPort, inspect.isfunction)
        if not name.startswith("_")
    }
    assert set(calls) == port_methods - {"release_torque"}
    for call in calls.values():
        call()
    watched_step()

    assert "send_joint_target" in recorder.calls
    assert not {"disconnect", "stop", "hard_estop", "resume"} & set(recorder.calls)
    assert arm.torque_on()

    arm.release_torque()
    assert recorder.calls[-1] == "disconnect"
    assert not arm.torque_on()
    with pytest.raises(ArmFault, match=r"motors are off"):
        arm.send_target(HOME)
    with pytest.raises(ArmFault, match=r"motors are off"):
        arm.gripper_reference()
    arm.gripper_light_hold()
    assert arm.gripper_command(1.0, 0.5, "plain", cancel) == GripperOutcome(
        "refused", None, "the arm is not connected or its motors are off"
    )
    assert recorder.calls.count("disconnect") == 1
    assert not {"stop", "hard_estop", "resume"} & set(recorder.calls)


def test_source_reaches_disconnect_once_and_never_stops_the_driver() -> None:
    source = inspect.getsource(apai_backend)
    assert len(re.findall(r"\.disconnect\(", source)) == 1
    assert not re.findall(r"\.(stop|hard_estop|resume)\(", source)
    assert "import time" not in source  # only SystemClock reads the machine's clocks


def test_build_arm_refuses_another_arm_type_or_rest_pose(
    profile: apai_backend.ApaiProfile, clock: FakeClock
) -> None:
    other = apai_backend.load_apai_profile("mock", {"arm": {"type": "isaac"}})
    with pytest.raises(ValueError, match=r"arm type 'isaac' is not one of mock"):
        apai_backend.build_apai_arm(make_config(), other, clock)
    parked = apai_backend.load_apai_profile("mock", {"arm": {"park_q": list(REST)}})
    apai_backend.build_apai_arm(make_config(), parked, clock)
    raised = make_config(rest_q=moved(REST, 1, 0.1))
    with pytest.raises(ValueError, match=r"differs from the APAI profile's park_q"):
        apai_backend.build_apai_arm(raised, parked, clock)
    with pytest.raises(ValueError, match=r"the backend has 6 joints"):
        apai_backend.build_apai_arm(make_config(joints=5), profile, clock)
    # APAI's mock profile names no park pose and is taken; a profile that
    # drives hardware is not: APAI's driver would measure rest against zeros.
    assert profile.arm_cfg.get("park_q") is None
    unparked = apai_backend.load_apai_profile("rebot_rs", {"arm": {"park_q": None}})
    for rest_q in (REST, moved(REST, 1, 0.4)):
        fitted = apai_backend.config_for_profile(make_config(rest_q), unparked)
        with pytest.raises(ValueError, match=r"names no park pose \(park_q\)"):
            apai_backend.build_apai_arm(fitted, unparked, clock)


def test_connect_lets_apai_claim_no_operators_confirmation(
    arm: apai_backend.ApaiArm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apai.control import arm_hold

    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(tmp_path / "holds"))
    written = arm_hold.confirm("holding the arm")  # fresh, for any arm
    claim = arm_hold.claim_confirmation
    connect, claimed = arm.backend.connect, []

    def connecting() -> None:
        # What APAI's hold gate asks during a connect that cuts torque.
        claimed.append(arm_hold.claim_confirmation("can4"))
        claimed.append(arm_hold.take_confirmation(None))
        connect()

    monkeypatch.setattr(arm.backend, "connect", connecting)
    arm.connect()
    assert claimed == [None, None] and arm.torque_on()
    assert arm_hold.claim_confirmation is claim  # APAI's own again
    assert [path.name for path in written.parent.iterdir()] == [written.name]


def test_a_connect_that_fails_puts_apais_claim_back(
    arm: apai_backend.ApaiArm, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apai.control import arm_hold

    claim = arm_hold.claim_confirmation

    def no_bus() -> None:
        assert arm_hold.claim_confirmation is not claim
        raise OSError("can4: no such device")

    monkeypatch.setattr(arm.backend, "connect", no_bus)
    with pytest.raises(OSError, match="no such device"):
        arm.connect()
    assert arm_hold.claim_confirmation is claim
    assert not arm.torque_on()


# Jaws on APAI's mock arm; its profile counts the jaw position from 0 open to 1 closed


def test_jaws_are_unknown_until_referenced(arm: apai_backend.ApaiArm) -> None:
    cancel = threading.Event()
    assert arm.gripper_command(1.0, 0.5, "plain", cancel) == GripperOutcome(
        "refused", None, "the arm is not connected or its motors are off"
    )
    arm.connect()
    assert not arm.gripper_referenced()
    assert arm.gripper_read() is None
    assert arm.gripper_last_target() is None
    assert arm.gripper_command(1.0, 0.5, "plain", cancel) == GripperOutcome(
        "refused", None, "the jaws are not referenced"
    )
    arm.gripper_light_hold()  # nothing to hold yet

    arm.gripper_reference()
    assert arm.gripper_referenced()
    assert arm.gripper_read() == pytest.approx(0.0)  # the reference closes the jaws
    assert arm.gripper_last_target() == pytest.approx(0.0)
    assert arm.backend.get_state().gripper_pos == 1.0


def test_reference_fails_with_something_between_the_jaws(
    arm: apai_backend.ApaiArm,
) -> None:
    arm.connect()
    arm.backend.object_stop_frac = 0.6
    with pytest.raises(ArmFault, match=r"could not be referenced: the jaws stopped"):
        arm.gripper_reference()
    assert not arm.gripper_referenced()
    assert arm.gripper_read() is None


def test_plain_command_moves_the_jaws_to_the_opening(
    connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    cancel = threading.Event()
    before = clock.monotonic()
    assert connected.gripper_command(1.0, 0.8, "plain", cancel) == GripperOutcome(
        "reached", pytest.approx(1.0)
    )
    assert connected.backend.get_state().gripper_pos == pytest.approx(0.0)
    assert connected.gripper_last_target() == pytest.approx(1.0)
    outcome = connected.gripper_command(0.25, 0.8, "plain", cancel)
    assert outcome == GripperOutcome("reached", pytest.approx(0.25))
    assert connected.backend.get_state().gripper_pos == pytest.approx(0.75)
    assert connected.gripper_read() == pytest.approx(0.25)
    assert clock.monotonic() == before  # a jaw already there needs no waiting


@pytest.mark.parametrize("mode", ["close_until_contact", "plain"])
def test_close_onto_an_object_stalls_and_holds_lightly_in_either_mode(
    connected: apai_backend.ApaiArm, clock: FakeClock, mode: str
) -> None:
    cancel = threading.Event()
    connected.gripper_command(1.0, 0.8, "plain", cancel)
    connected.backend.object_stop_frac = 0.6  # an object at opening 0.4
    before = clock.monotonic()
    outcome = connected.gripper_command(0.0, 0.8, mode, cancel)

    assert outcome == GripperOutcome("stalled", pytest.approx(0.4))
    # Two polls of grace, then one more without movement.
    assert clock.monotonic() - before == pytest.approx(0.45)
    assert connected.gripper_last_target() == pytest.approx(0.0)  # the target stands
    assert connected.backend._gripper_effort == pytest.approx(1.0 / 6.0)
    assert connected.gripper_read() == pytest.approx(0.4)


def test_a_command_that_reaches_or_opens_is_not_followed_by_a_light_hold(
    connected: apai_backend.ApaiArm, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel = threading.Event()
    assert connected.gripper_command(1.0, 0.8, "plain", cancel).status == "reached"
    assert connected.backend._gripper_effort == pytest.approx(0.8)
    assert connected.gripper_command(0.3, 0.7, "plain", cancel).status == "reached"
    assert connected.backend._gripper_effort == pytest.approx(0.7)
    # An opening that ends short of its target: the jaws read 0.3 throughout.
    monkeypatch.setattr(connected, "gripper_read", lambda: 0.3)
    opened = connected.gripper_command(0.9, 0.6, "close_until_contact", cancel)
    assert opened == GripperOutcome("stalled", 0.3)
    assert connected.backend._gripper_effort == pytest.approx(0.6)


def test_cancel_ends_a_jaw_command_in_a_light_hold(
    connected: apai_backend.ApaiArm, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel = threading.Event()
    connected.gripper_command(1.0, 0.8, "plain", cancel)
    connected.backend.object_stop_frac = 0.6
    advance = clock.sleep

    def sleep_then_cancel(seconds: float) -> None:
        advance(seconds)
        cancel.set()

    monkeypatch.setattr(clock, "sleep", sleep_then_cancel)
    outcome = connected.gripper_command(0.0, 0.8, "plain", cancel)
    assert outcome == GripperOutcome("cancelled", pytest.approx(0.4))
    assert connected.backend._gripper_effort == pytest.approx(1.0 / 6.0)

    # A command that finds the cancel already set sends nothing.
    outcome = connected.gripper_command(1.0, 0.8, "plain", cancel)
    assert outcome == GripperOutcome("cancelled", pytest.approx(0.4))
    assert connected.gripper_last_target() == pytest.approx(0.0)


def test_light_hold_that_cannot_be_sent_is_an_arm_fault(
    connected: apai_backend.ApaiArm, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_bus(_pos: float, _effort: float = 1.0) -> None:
        raise OSError("bus off")

    monkeypatch.setattr(connected.backend, "set_gripper", no_bus)
    with pytest.raises(ArmFault, match=r"light jaw hold could not be sent: bus off"):
        connected.gripper_light_hold()
    with pytest.raises(ArmFault, match=r"the jaw command failed: bus off"):
        connected.gripper_command(1.0, 0.5, "plain", threading.Event())


@pytest.mark.parametrize("mode", ["close_until_contact", "plain"])
@pytest.mark.parametrize(("opening", "effort_after"), [(0.5, 1.0 / 6.0), (1.0, 0.8)])
def test_jaws_that_cannot_be_read_time_out(
    connected: apai_backend.ApaiArm,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    opening: float,
    effort_after: float,
) -> None:
    monkeypatch.setattr(connected, "gripper_read", lambda: None)
    before = clock.monotonic()
    outcome = connected.gripper_command(opening, 0.8, mode, threading.Event())
    assert outcome == GripperOutcome("timeout", None)
    assert clock.monotonic() - before == pytest.approx(4.05)
    # Without a reading only a command to full open is known not to be a close.
    assert connected.backend._gripper_effort == pytest.approx(effort_after)


@pytest.mark.parametrize(
    ("opening", "max_effort", "mode", "match"),
    [
        (1.5, 0.5, "plain", r"opening must be between 0 and 1, got 1\.5"),
        (math.nan, 0.5, "plain", r"opening must be between 0 and 1"),
        (0.5, -0.1, "plain", r"max_effort must be between 0 and 1, got -0\.1"),
        (0.5, "full", "plain", r"max_effort must be a number"),
        (True, 0.5, "plain", r"opening must be a number"),
        (0.5, 0.5, "squeeze", r"unknown mode 'squeeze'"),
    ],
)
def test_malformed_jaw_command_is_refused_with_the_reading(
    connected: apai_backend.ApaiArm,
    opening: Any,
    max_effort: Any,
    mode: str,
    match: str,
) -> None:
    outcome = connected.gripper_command(opening, max_effort, mode, threading.Event())
    assert outcome.status == "refused"
    assert outcome.opening == pytest.approx(0.0)
    assert re.search(match, outcome.detail)
    assert connected.gripper_last_target() == pytest.approx(0.0)


# The reBot RS branch: APAI's driver class and gripper guard over a stand-in motor SDK

RS_TRAVEL = 6.2  # rad from closed to open in APAI's rebot_rs profile
RS_HOLD_KP = 0.01  # its light holding stiffness; its closing stiffness is 2.0


class Motor:
    def __init__(self) -> None:
        self.value = 0.0
        self.reads: list[tuple[int, int | None]] = []
        self.missed = 0  # how many of the next reads get no answer
        # Answers to its next reads within a read of all motors, in place of
        # where it is. Such a read passes no timeout.
        self.wrong: list[float] = []

    def robstride_get_param_f32(
        self, param: int, timeout_ms: int | None = None
    ) -> float:
        self.reads.append((param, timeout_ms))
        if self.missed > 0:
            self.missed -= 1
            raise OSError("no answer")
        if timeout_ms is None and self.wrong:
            return self.wrong.pop(0)
        return self.value


class MotorGroup:
    """Motors that reach each position command at once.

    Like the real motor library's send_mit, a command that reaches no motor
    raises nothing: from command number `lost_from` on, each is only counted.
    """

    def __init__(self, names: list[str]) -> None:
        self.joint_names = names
        self._mm = {name: Motor() for name in names}
        self.sent: list[tuple[list[float], list[float]]] = []
        self.caps: dict[str, float] = {}  # a motor cannot pass this value
        self.lost_from: int | None = None
        self.lost: list[tuple[list[float], list[float]]] = []

    def send_mit(self, q: Any, kp: Any = None, kd: Any = None, tau: Any = None) -> None:
        command = (np.array(q).tolist(), np.array(kp).tolist())
        if self.lost_from is not None and len(self.sent) >= self.lost_from:
            self.lost.append(command)
            return
        self.sent.append(command)
        for name, value in zip(self.joint_names, q):
            value = min(float(value), self.caps.get(name, math.inf))
            self._mm[name].value = self.reached(value)

    def reached(self, value: float) -> float:
        return value


class JawGroup(MotorGroup):
    """The jaw motor: it cannot close past its stop, or past an object."""

    stop = 0.0
    blocked_at: float | None = None  # it cannot open past this
    per_command: float | None = None  # the most it travels per command; None: all

    def reached(self, value: float) -> float:
        at = self._mm["jaw"].value
        if self.per_command is not None:
            value = at + max(-self.per_command, min(self.per_command, value - at))
        if self.blocked_at is not None:
            value = min(value, self.blocked_at)
        return max(value, self.stop)

    def at(self) -> float:
        return self._mm["jaw"].value

    def light_after(self, mark: int) -> bool:
        """Whether a command at the profile's hold stiffness came after `mark`."""
        return any(kp == [RS_HOLD_KP] for _, kp in list(self.sent[mark:]))


class MotorSdk:
    has_gripper = True

    def __init__(self) -> None:
        self.arm = MotorGroup([f"motor{i}" for i in range(1, 7)])
        self.gripper = JawGroup(["jaw"])
        self.disconnects = 0
        self.guard_time = GuardTime()  # rs_arm() puts APAI's gripper guard on it

    def disconnect(self) -> None:
        self.disconnects += 1


class GuardTime:
    """The clock APAI's gripper guard runs on; sleeping advances it."""

    def __init__(self) -> None:
        self.now = 50.0
        # Called inside every sleep, on the thread that commands the jaws.
        self.on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


@pytest.fixture(scope="module")
def rs_profile() -> apai_backend.ApaiProfile:
    return apai_backend.load_apai_profile("rebot_rs")


def rs_arm(
    config: RobotAppConfig,
    rs_profile: apai_backend.ApaiProfile,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[apai_backend.ApaiArm, MotorSdk]:
    """The arm port on APAI's RebotRSArm; its connect() takes a stand-in motor SDK.

    Not connected yet. config is one that config_for_profile() gave.
    """
    import apai.control.rs_gripper as rs_gripper

    monkeypatch.setattr(sys, "path", list(sys.path))  # the driver adds its SDK path
    arm = apai_backend.build_apai_arm(config, rs_profile, clock)
    backend, sdk = arm.backend, MotorSdk()
    monkeypatch.setattr(rs_gripper, "time", sdk.guard_time)
    sdk.gripper._mm["jaw"].value = 0.5  # the jaws start a little open

    def connect() -> None:
        # The state RebotRSArm.connect leaves behind, without the bus.
        backend._arm = sdk
        backend._stopped = False
        backend._mit_kp = np.full(6, 30.0)
        backend._mit_kd = np.ones(6)
        backend._read_failures = 0
        backend._last_q = backend._read_positions()
        backend._last_q_t = time.monotonic()
        backend._last_cmd_q = None
        backend._last_cmd_t = None

    monkeypatch.setattr(backend, "connect", connect)
    return arm, sdk


@pytest.fixture
def rs(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[apai_backend.ApaiArm, MotorSdk]:
    """The arm port on APAI's RebotRSArm, connected to a stand-in motor SDK."""
    rig = apai_backend.config_for_profile(config, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    arm.connect()
    return arm, sdk


def test_rs_profile_builds_checkers_that_accept_the_rest_pose(
    rs_profile: apai_backend.ApaiProfile, config: RobotAppConfig, clock: FakeClock
) -> None:
    checkers = apai_backend.build_apai_checkers(config, rs_profile, clock)
    for checker in checkers.all():
        checker.check_pose(REST)
        assert checker.harness.tool_volume is None  # only an obstacle map reads it
    checkers.recovery.check_path(HOME, min_jerk_path(HOME, REST, 250), 0.1)


def test_rs_arm_sends_and_reads_through_the_driver(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm, sdk = rs
    assert type(arm.backend).__name__ == "RebotRSArm"

    def cached() -> None:
        raise AssertionError("get_state can return a cached pose")

    monkeypatch.setattr(arm.backend, "get_state", cached)
    target = [0.0, 0.1, 0.12, 0.0, 0.0, 0.0]
    arm.send_target(target)
    assert sdk.arm.sent == [(target, [30.0] * 6)]
    assert arm.last_target() == (target, clock.monotonic())

    # One motor's position parameter, with the contact monitor's read timeout in ms.
    assert arm.read_joint(2) == JointReading(2, 0.12, 1000.0, 1000.0)
    assert sdk.arm._mm["motor3"].reads[-1] == (0x7019, 50)
    assert [r.value for r in arm.read_all()] == target
    sdk.arm._mm["motor2"].value = 0.15  # the arm sagged
    arm.hold_measured()
    assert sdk.arm.sent[-1][0] == [0.0, 0.15, 0.12, 0.0, 0.0, 0.0]

    sdk.arm._mm["motor4"].value = math.nan
    with pytest.raises(ReadFailed, match=r"joint4: implausible mechPos"):
        arm.read_joint(3)
    with pytest.raises(ReadFailed, match=r"implausible mechPos"):
        arm.read_all()


def test_rs_jaws_go_through_the_gripper_guard(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    cancel = threading.Event()
    jaw = sdk.gripper
    assert not arm.gripper_referenced()
    assert arm.gripper_read() is None
    assert arm.gripper_command(1.0, 0.5, "plain", cancel).status == "refused"
    assert jaw.sent == []  # an unreferenced gripper is not moved by a command

    arm.gripper_reference()
    assert arm.gripper_referenced()
    assert arm.gripper_read() == pytest.approx(0.0)
    assert arm.gripper_last_target() == pytest.approx(0.0)

    # Opening 0..1 is the guard's travel, 0 rad closed to 6.2 rad open; the effort
    # scales the profile's stiffness of 2.0. An opening goes out as a ramp.
    sends = len(jaw.sent)
    assert arm.gripper_command(1.0, 0.5, "plain", cancel) == GripperOutcome(
        "reached", pytest.approx(1.0)
    )
    assert len(jaw.sent) - sends > 10
    assert jaw.sent[-1] == ([pytest.approx(6.2)], [1.0])
    assert arm.gripper_command(0.5, 0.5, "plain", cancel) == GripperOutcome(
        "reached", pytest.approx(0.5)
    )
    assert jaw.sent[-1] == ([pytest.approx(3.1)], [1.0])
    assert arm.gripper_last_target() == pytest.approx(0.5)

    jaw.stop = 2.0  # an object
    outcome = arm.gripper_command(0.0, 0.5, "close_until_contact", cancel)
    assert outcome == GripperOutcome("stalled", pytest.approx(2.0 / 6.2))
    # The closed target stands, at the profile's hold stiffness.
    assert jaw.sent[-1] == ([0.0], [0.01])
    assert arm.gripper_last_target() == pytest.approx(0.0)

    # A plain close that stalls ends the same way: sent at the closing stiffness,
    # left at the hold stiffness.
    sends = len(jaw.sent)
    assert arm.gripper_command(0.0, 0.5, "plain", cancel).status == "stalled"
    assert jaw.sent[sends:] == [([0.0], [1.0]), ([0.0], [0.01])]
    sends = len(jaw.sent)
    arm.gripper_light_hold()
    assert jaw.sent[sends:] == [([0.0], [0.01])]


def test_rs_gripper_fault_is_an_arm_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    sdk.gripper._mm["jaw"].value = 1.5
    sdk.gripper.stop = 1.0  # something between the jaws
    with pytest.raises(ArmFault, match=r"could not be referenced: gripper reference"):
        arm.gripper_reference()
    assert not arm.gripper_referenced()


def test_rs_release_torque_is_the_only_disconnect(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.send_target(REST)
    arm.read_all()
    arm.hold_measured()
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    arm.gripper_light_hold()
    assert sdk.disconnects == 0
    assert arm.backend._stopped is False  # the driver's stop flag was never raised

    arm.release_torque()
    assert sdk.disconnects == 1
    assert not arm.torque_on()
    with pytest.raises(ReadFailed):
        arm.read_all()


# "At rest at start" on the rig: the profile's rest check tolerance

# Three full joint reads of the rig lying folded at rest with the motors off, rad
# (feedback-connected, rest-recovery-readings, operator-hold-reset). The
# second and third joint lie on their stops and read a few mrad below 0.
RIG_REST_READINGS = [
    [-0.019729, -0.007883, 0.000980, 0.007785, -0.059097, 0.056450],
    [-0.054968, -0.006605, 0.000426, 0.007938, -0.008744, 0.077696],
    [-0.020453, -0.007585, 0.001150, 0.006558, -0.009089, 0.077811],
]


def test_the_profile_gives_the_tolerance_for_at_rest_at_start(
    rs_profile: apai_backend.ApaiProfile,
    profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
) -> None:
    assert config.rest.start_rest_tol_rad == config.rest.rest_tol_rad == 0.05
    rig = apai_backend.config_for_profile(config, rs_profile)
    # The rig profile's rest_check_tol_rad; the release keeps its own.
    assert (rig.rest.start_rest_tol_rad, rig.rest.rest_tol_rad) == (0.12, 0.05)
    # Nothing but that and the jaw target tolerance differs.
    assert dataclasses.replace(rig, rest=config.rest, gripper=config.gripper) == config
    # The mock profile names none: APAI's default.
    mock = apai_backend.config_for_profile(config, profile)
    assert mock.rest.start_rest_tol_rad == 0.12
    # A rig measured tighter than a release needs: never below the release's.
    tight = apai_backend.load_apai_profile(
        "rebot_rs", {"arm": {"rest_check_tol_rad": 0.03}}
    )
    assert apai_backend.config_for_profile(config, tight).rest == config.rest
    wide = apai_backend.load_apai_profile(
        "rebot_rs", {"arm": {"rest_check_tol_rad": 0.2}}
    )
    with pytest.raises(ValueError, match=r"rest_check_tol_rad: 0\.2 is not"):
        apai_backend.config_for_profile(config, wide)


def test_build_arm_refuses_a_real_arm_without_the_profiles_start_tolerance(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(
        ValueError,
        match=r"rest\.start_rest_tol_rad 0\.05 differs from the APAI profile's "
        r"rest check tolerance 0\.12: use config_for_profile\(\)",
    ):
        apai_backend.build_apai_arm(config, rs_profile, clock)


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_the_rigs_rest_reading_passes_apais_rest_check_only_at_the_profiles_tolerance(
    rs_profile: apai_backend.ApaiProfile, reading: list[float]
) -> None:
    from apai.control import arm_hold

    park = rs_profile.arm_cfg.park_q
    at_start = arm_hold.rest_tolerance(rs_profile.arm_cfg)
    assert arm_hold.rest_problem(reading, park, at_start) is None
    assert arm_hold.rest_problem(reading, park, 0.05) is not None


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_the_app_started_at_the_rigs_rest_reading_references_the_jaws(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    reading: list[float],
) -> None:
    rig = apai_backend.config_for_profile(config, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = apai_backend.build_apai_checkers(rig, rs_profile, clock)
    h = AppHarness(rig, arm, checkers, clock)
    try:
        h.start()
        state = h.client.state
        assert state.gripper.referenced and state.gripper.load == "empty"
        # Not "rest_hold": a release keeps its 0.05 rad and refuses this pose.
        assert (state.motion, state.torque_on) == ("hold", True)
        assert sdk.arm.sent == []  # no joint was commanded
        refused = h.client.wait_result(h.client.release(), timeout_s=5.0)
        assert refused.status == "refused"
        assert refused.reason.startswith("the arm is not at rest: joint index ")
        assert arm.torque_on() and sdk.disconnects == 0
        # The recovery region of this configuration ends at 0 on the joint stops.
        rest = h.client.wait_result(h.client.rest(), timeout_s=5.0)
        assert rest.status == "refused"
        assert "is outside the tested recovery region 0.0000..3.1400" in rest.reason
    finally:
        h.close()


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_a_typed_rest_from_the_rigs_rest_reading_makes_the_release_possible(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    reading: list[float],
) -> None:
    # The recovery limits and region reach 0.05 rad below the two joint stops.
    low = (MODEL_LOW[0], -0.05, -0.05, *MODEL_LOW[3:])
    recovery = dataclasses.replace(config.limits.recovery, joint_min=low)
    rig = dataclasses.replace(
        config,
        limits=dataclasses.replace(config.limits, recovery=recovery),
        rest=dataclasses.replace(config.rest, region_min=low),
    )
    rig = apai_backend.config_for_profile(rig, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = apai_backend.build_apai_checkers(rig, rs_profile, clock)
    h = AppHarness(rig, arm, checkers, clock)
    try:
        h.start()
        assert h.client.state.motion == "hold"
        rest = h.client.wait_result(h.client.rest(), timeout_s=5.0)
        assert rest.status == "completed", rest.reason
        assert h.app.motion_state is MotionState.REST_HOLD
        assert sdk.arm.sent[-1][0] == list(REST)
        assert sdk.disconnects == 0
        released = h.client.wait_result(h.client.release(), timeout_s=5.0)
        assert released.status == "completed", released.reason
        assert sdk.disconnects == 1 and not arm.torque_on()
    finally:
        h.close()


# The jaws of the reBot RS branch: a cancel during an opening, missed jaw answers

WAIT_S = 5.0  # real seconds a test gives another thread before it fails


def targets_at(jaw: JawGroup, mark: int, kp: float) -> list[float]:
    """The targets of the jaw commands at this stiffness sent after `mark`."""
    return [q[0] for q, stiffness in jaw.sent[mark:] if stiffness == [kp]]


def test_rs_cancel_ends_an_opening_within_one_pass_of_the_guard(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    jaw, cancel, marks = sdk.gripper, threading.Event(), []
    arm.gripper_reference()
    probe = arm.backend._grip_guard._stopped

    def cancel_part_open() -> None:
        if jaw.at() >= 1.05 and not cancel.is_set():
            marks.append(len(jaw.sent))
            cancel.set()

    sdk.guard_time.on_sleep = cancel_part_open
    outcome = arm.gripper_command(1.0, 0.5, "plain", cancel)

    # The opening measured after the cancel, not the one that was asked for.
    assert outcome == GripperOutcome("cancelled", pytest.approx(1.1 / RS_TRAVEL))
    assert jaw.at() == pytest.approx(1.1)
    # Nothing more went out at the opening stiffness; the last is the light hold.
    assert targets_at(jaw, marks[0], 1.0) == []
    assert jaw.sent[marks[0] :] == [([pytest.approx(1.1)], [RS_HOLD_KP])]
    assert arm.gripper_last_target() == pytest.approx(1.1 / RS_TRAVEL)
    # The driver's own stop flag was not touched and the guard asks it again.
    assert arm.backend._stopped is False
    assert arm.backend._grip_guard._stopped is probe

    # The jaws work afterwards: a new command opens them the rest of the way.
    sdk.guard_time.on_sleep = None
    opened = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert opened == GripperOutcome("reached", pytest.approx(1.0))


def test_rs_a_cancelled_opening_is_held_where_the_jaws_are_not_where_the_ramp_was(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    jaw.per_command = 0.06  # the jaws trail the guard's ramp of 0.1 rad per pass

    def cancel_part_open() -> None:
        if jaw.at() >= 1.05:
            cancel.set()

    sdk.guard_time.on_sleep = cancel_part_open
    outcome = arm.gripper_command(1.0, 0.5, "plain", cancel)
    at = jaw.at()
    assert 1.05 <= at < 1.2
    ramp = max(targets_at(jaw, 0, 1.0))
    assert ramp > at + 0.3  # the ramp was well ahead of the jaws
    assert outcome == GripperOutcome("cancelled", pytest.approx(at / RS_TRAVEL))
    assert jaw.sent[-1] == ([pytest.approx(at)], [RS_HOLD_KP])
    assert arm.gripper_last_target() == pytest.approx(at / RS_TRAVEL)


def test_rs_cancel_during_a_close_leaves_the_close_target_at_the_hold_stiffness(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm, sdk = rs
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.5, "plain", cancel)
    jaw.stop = 2.0  # an object
    advance = clock.sleep

    def sleep_then_cancel(seconds: float) -> None:
        advance(seconds)
        cancel.set()

    monkeypatch.setattr(clock, "sleep", sleep_then_cancel)
    sends = len(jaw.sent)
    outcome = arm.gripper_command(0.0, 0.5, "plain", cancel)
    assert outcome == GripperOutcome("cancelled", pytest.approx(2.0 / RS_TRAVEL))
    assert jaw.sent[sends:] == [([0.0], [1.0]), ([0.0], [RS_HOLD_KP])]
    assert arm.gripper_last_target() == pytest.approx(0.0)


def test_rs_one_missed_jaw_answer_before_an_opening_is_not_a_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk], clock: FakeClock
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    before = clock.monotonic()
    sdk.gripper._mm["jaw"].missed = 1
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert outcome == GripperOutcome("reached", pytest.approx(1.0))
    assert arm.gripper_referenced()
    assert clock.monotonic() - before == pytest.approx(0.05)  # one wait, then again


def test_rs_an_opening_from_jaws_that_stay_unread_is_refused_and_sends_nothing(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    sends = len(sdk.gripper.sent)
    sdk.gripper._mm["jaw"].missed = 3
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.opening) == ("refused", None)
    assert outcome.detail.startswith("the jaws could not be read, so no jaw command")
    assert "no gripper command sent" in outcome.detail
    assert sdk.gripper.sent[sends:] == []
    # Three reads in a row without an answer: APAI's guard drops its reference.
    assert not arm.gripper_referenced()
    refused = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert refused == GripperOutcome("refused", None, "the jaws are not referenced")


def test_rs_a_close_makes_no_read_before_it_is_sent(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    sends = len(sdk.gripper.sent)
    sdk.gripper._mm["jaw"].missed = 1
    outcome = arm.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert outcome == GripperOutcome("reached", pytest.approx(0.0))
    assert sdk.gripper.sent[sends:] == [([0.0], [1.0])]


def test_rs_jaws_blocked_while_they_open_are_still_an_arm_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    sdk.gripper.blocked_at = 1.0
    with pytest.raises(ArmFault, match=r"jaw command failed: gripper blocked while"):
        arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert sdk.gripper.at() == pytest.approx(1.0)


def test_rs_feedback_lost_during_an_opening_is_still_an_arm_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    jaw = sdk.gripper
    arm.gripper_reference()

    def lose_the_jaw_motor() -> None:
        if jaw.at() >= 1.05:
            jaw._mm["jaw"].missed = 10

    sdk.guard_time.on_sleep = lose_the_jaw_motor
    with pytest.raises(ArmFault, match=r"gripper feedback lost while opening"):
        arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert len(targets_at(jaw, 0, 1.0)) > 5  # jaw commands had gone out


def test_the_profile_gives_the_jaw_target_tolerance_of_proven_empty(
    rs_profile: apai_backend.ApaiProfile,
    profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
) -> None:
    assert config.gripper.target_tol == 0.03  # 0.186 rad of the rig's 6.2 rad
    rig = apai_backend.config_for_profile(config, rs_profile)
    assert rig.gripper.target_tol == pytest.approx(0.1 / RS_TRAVEL)
    # The other two jaw tolerances stay as configured.
    assert (rig.gripper.reach_tol, rig.gripper.full_open_tol) == (0.03, 0.03)
    assert dataclasses.replace(rig, rest=config.rest, gripper=config.gripper) == config
    # A tolerance already tighter than 0.1 rad stays.
    tight = dataclasses.replace(
        config, gripper=dataclasses.replace(config.gripper, target_tol=0.01)
    )
    assert apai_backend.config_for_profile(tight, rs_profile).gripper == tight.gripper
    # The mock arm's jaw position is not an angle: its tolerance stays.
    assert apai_backend.config_for_profile(config, profile).gripper == config.gripper


def test_build_arm_refuses_a_real_arm_with_a_jaw_target_tolerance_over_a_tenth_rad(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    rig = apai_backend.config_for_profile(config, rs_profile)
    wide = dataclasses.replace(rig, gripper=config.gripper)
    with pytest.raises(
        ValueError,
        match=r"gripper\.target_tol 0\.03 is 0\.186 rad of the APAI profile's jaw "
        r"travel of 6\.2 rad, over the 0\.1 rad of \"jaws proven empty\": use "
        r"config_for_profile\(\)",
    ):
        apai_backend.build_apai_arm(wide, rs_profile, clock)


# The robot app on the reBot RS branch


@pytest.fixture
def rs_app(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[AppHarness, MotorSdk]]:
    """A started robot app on APAI's RebotRSArm at rest, its jaws referenced."""
    rig = apai_backend.config_for_profile(config, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    checkers = apai_backend.build_apai_checkers(rig, rs_profile, clock)
    h = AppHarness(rig, arm, checkers, clock)
    try:
        h.start()
        assert h.client.state.gripper.referenced
        yield h, sdk
    finally:
        h.close()


def final(h: AppHarness, command_id: str) -> Result:
    """A command's final RESULT, once the ROBOT_STATE that shows it has arrived."""
    result = h.client.wait_result(command_id, timeout_s=WAIT_S)
    h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == command_id, WAIT_S
    )
    return result


def hold_an_object(h: AppHarness, sdk: MotorSdk, at_rad: float) -> None:
    """The agent takes the arm, opens the jaws and closes them on an object."""
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    sdk.gripper.stop = at_rad
    closed = final(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert closed.detail == {
        "outcome": "stalled",
        "opening": pytest.approx(at_rad / RS_TRAVEL),
    }
    assert sdk.gripper.sent[-1] == ([0.0], [RS_HOLD_KP])


def break_the_link(h: AppHarness) -> None:
    h.robot_end.break_link()


def raise_a_fault(h: AppHarness) -> None:
    h.arm.faults = lambda: ["jaw motor over temperature"]


JAW_STOPS: dict[str, Callable[[AppHarness], Any]] = {
    "a STOP": lambda h: h.client.stop("operator pressed stop"),
    "a fault-stop": raise_a_fault,
    "help_needed": lambda h: h.client.request_owner("help_needed", reason="stuck"),
    "the agent gives the arm back": lambda h: h.client.request_owner(
        "nobody", reason="task done"
    ),
    "the robot connection lost": break_the_link,
}


@pytest.mark.parametrize("stopped_by", JAW_STOPS)
def test_an_agents_opening_of_a_held_object_ends_where_it_is_stopped(
    rs_app: tuple[AppHarness, MotorSdk], stopped_by: str
) -> None:
    h, sdk = rs_app
    jaw, marks = sdk.gripper, []
    hold_an_object(h, sdk, 2.0)

    def stop_part_open() -> None:
        # Inside the guard's 50 ms sleep, the jaws 0.3 rad into the opening.
        if marks or jaw.at() < 2.25:
            return
        marks.append(len(jaw.sent))
        JAW_STOPS[stopped_by](h)
        deadline = time.monotonic() + WAIT_S
        while not jaw.light_after(marks[0]) and time.monotonic() < deadline:
            time.sleep(0.001)

    sdk.guard_time.on_sleep = stop_part_open
    jaw_id = h.client.gripper(1.0, max_effort=0.5)
    deadline = time.monotonic() + WAIT_S
    while not marks and time.monotonic() < deadline:
        time.sleep(0.001)
    assert marks, "the opening never got 0.3 rad far"
    assert h.app._worker.wait_idle(WAIT_S)

    # Not one more command went out at the opening stiffness, and every command
    # since the stop holds the jaws where they were, at the hold stiffness.
    assert targets_at(jaw, marks[0], 1.0) == []
    after = jaw.sent[marks[0] :]
    assert len(after) >= 2
    assert all(sent == ([pytest.approx(2.3)], [RS_HOLD_KP]) for sent in after)
    assert jaw.at() == pytest.approx(2.3)
    assert h.arm.backend._stopped is False  # the driver was not stopped
    assert h.arm.torque_on() and sdk.disconnects == 0
    if stopped_by != "the robot connection lost":
        result = h.client.wait_result(jaw_id, timeout_s=WAIT_S)
        assert result.status == "cancelled"
        assert result.detail["opening"] == pytest.approx(2.3 / RS_TRAVEL)
        state = h.client.wait_state(lambda s: s.owner == "nobody")
        assert state.gripper.may_be_holding and state.gripper.load == "loaded"
        assert state.gripper.last_target == pytest.approx(2.3 / RS_TRAVEL)


def test_the_operators_open_ends_where_it_is_stopped(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    jaw, marks = sdk.gripper, []

    def stop_part_open() -> None:
        if marks or jaw.at() < 0.75:
            return
        marks.append(len(jaw.sent))
        h.client.stop("operator pressed stop")
        deadline = time.monotonic() + WAIT_S
        while not jaw.light_after(marks[0]) and time.monotonic() < deadline:
            time.sleep(0.001)

    sdk.guard_time.on_sleep = stop_part_open
    open_id = h.client.open_jaws()
    result = final(h, open_id)
    assert result.status == "cancelled"
    # The step under way from 0.62 rad ended at once; no further step was sent.
    assert targets_at(jaw, marks[0], 1.0) == []
    assert jaw.at() == pytest.approx(0.82)
    assert jaw.sent[-1] == ([pytest.approx(0.82)], [RS_HOLD_KP])


@pytest.mark.parametrize(
    ("object_rad", "release", "reason"),
    [
        # Thinner than 0.1 rad: within the design's known limit, it reads as empty.
        (0.08, "completed", ""),
        (
            0.15,
            "refused",
            "the jaws are not proven empty (loaded): the jaws are at 0.024, not at "
            "their target 0.000",
        ),
    ],
)
def test_jaws_more_than_a_tenth_rad_from_their_target_are_not_proven_empty(
    rs_app: tuple[AppHarness, MotorSdk], object_rad: float, release: str, reason: str
) -> None:
    h, sdk = rs_app
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    sdk.gripper.stop = object_rad
    closed = final(h, h.client.gripper(0.0))
    assert closed.status == "completed"  # inside the tolerance of "reached"
    assert closed.detail["opening"] == pytest.approx(object_rad / RS_TRAVEL)
    assert final(h, h.client.request_owner("nobody", reason="done")).status == (
        "completed"
    )
    h.client.wait_state(lambda s: s.owner == "nobody")
    released = final(h, h.client.release())
    assert (released.status, released.reason) == (release, reason)
    assert sdk.disconnects == (1 if release == "completed" else 0)
    assert h.arm.torque_on() is (release != "completed")


def test_one_missed_jaw_answer_before_an_agents_opening_is_no_fault_stop(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    h.new_task()
    sdk.gripper._mm["jaw"].missed = 1
    result = final(h, h.client.gripper(1.0))
    assert (result.status, result.reason) == ("completed", "")
    state = h.client.wait_state(lambda s: s.gripper.opening == pytest.approx(1.0))
    assert (state.stop_level, state.owner) == ("none", "agent")


def test_an_agents_opening_from_jaws_that_stay_unread_is_refused_without_a_fault_stop(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    h.new_task()
    sends = len(sdk.gripper.sent)
    sdk.gripper._mm["jaw"].missed = 3
    result = final(h, h.client.gripper(1.0))
    assert result.status == "refused"
    assert result.reason.startswith("the jaws could not be read, so no jaw command")
    assert sdk.gripper.sent[sends:] == []
    state = h.client.wait_state(lambda s: not s.gripper.referenced)
    assert (state.stop_level, state.owner, state.torque_on) == ("none", "agent", True)


# The hold after a contact, on the reBot RS driver

RS_RAISED = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]  # off the joint stops
RS_TURN = moved(RS_RAISED, 0, 0.42)
RS_STOPPED = moved(RS_RAISED, 0, 0.05)  # where the first joint meets something


class DriverTime:
    """The driver's `time` on the app clock: on hardware both are the machine's."""

    def __init__(self, clock: FakeClock) -> None:
        self.monotonic = clock.monotonic
        self.sleep = clock.sleep


def straight(
    start: Sequence[float], goal: Sequence[float], count: int
) -> list[list[float]]:
    """`count` setpoints at one speed from start to goal; the last is goal."""
    a, b = np.array(start, dtype=float), np.array(goal, dtype=float)
    rows = [(a + (b - a) * (index / count)).tolist() for index in range(1, count)]
    return [*rows, [float(value) for value in goal]]


def rs_move(h: AppHarness, setpoints: list[list[float]], start: Sequence[float]):
    """An agent MOVE with permits for all of it, to its final RESULT."""
    result = h.client.run_move(setpoints, WAIT_S, start_q=list(start))
    return final(h, result.command_id)


def turn_into_a_stop(
    h: AppHarness,
    sdk: MotorSdk,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    wrong: list[float],
) -> Result:
    """An agent MOVE turns the first joint into something that stops it.

    The third joint's motor, at 0.1 rad, then answers the next reads of all
    motors with `wrong`. The driver's clock is the app's.
    """
    import apai.control.rebot_rs_arm as rebot_rs_arm

    monkeypatch.setattr(rebot_rs_arm, "time", DriverTime(clock))
    h.arm.backend._last_q_t = clock.monotonic()
    h.new_task()
    assert rs_move(h, straight(REST, RS_RAISED, 25), REST).status == "completed"
    sdk.arm.caps["motor1"] = RS_STOPPED[0]
    real, armed = h.arm.send_target, []

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if not armed:
            # No read of all motors is made between a move's start and its end.
            armed.append(True)
            sdk.arm._mm["motor3"].wrong = list(wrong)

    h.arm.send_target = send_target
    return rs_move(h, straight(RS_RAISED, RS_TURN, 60), RS_RAISED)


HOLD_FAILED = "contact; the hold at the measured pose failed: last command (measured "
DISAGREES = "pose unavailable: reading disagrees with the monitor's samples)"


@pytest.mark.parametrize(
    ("wrong", "why"),
    [
        # The driver compares with its last read of all motors, made when the
        # move began over 0.2 s before, and lets 0.5 rad pass. It keeps that
        # reading, so it then refuses the true one of the second try.
        ([0.6], "pose unavailable: ReadFailed: the joints could not be read: "),
        ([0.6, 0.6], DISAGREES),
    ],
    ids=["0.5 rad off once", "0.5 rad off twice"],
)
def test_a_wrong_reading_at_a_contact_is_not_sent_to_the_motors(
    rs_app: tuple[AppHarness, MotorSdk],
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    wrong: list[float],
    why: str,
) -> None:
    h, sdk = rs_app
    result = turn_into_a_stop(h, sdk, clock, monkeypatch, wrong)
    assert result.status == "failed"
    assert result.reason.startswith(HOLD_FAILED + why)
    assert result.reason.endswith("; the last setpoint stands")
    sent = result.detail["sent"]
    assert sent > 10
    # Nothing was sent after the last setpoint, which stands.
    turn = straight(RS_RAISED, RS_TURN, 60)
    targets = [q for q, _ in sdk.arm.sent][-sent:]
    assert targets[0] == pytest.approx(turn[0])
    assert targets[-1] == pytest.approx(turn[sent - 1])
    assert [q for q in targets if q[2] != pytest.approx(0.1)] == []
    # Not a fault-stop, and nothing is released.
    state = h.client.state
    assert (state.stop_level, state.owner, state.torque_on) == ("none", "agent", True)
    assert sdk.disconnects == 0


def test_a_reading_the_driver_refuses_at_a_contact_is_read_again_and_held(
    rs_app: tuple[AppHarness, MotorSdk],
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, sdk = rs_app
    # 2.8 rad off is more than the driver believes of the time since its last
    # read of all motors; it keeps nothing of such a reading.
    result = turn_into_a_stop(h, sdk, clock, monkeypatch, [2.9])
    assert (result.status, result.reason) == ("failed", "contact")
    targets = [q for q, _ in sdk.arm.sent][-result.detail["sent"] - 1 :]
    assert targets[-1] == pytest.approx(RS_STOPPED)
    assert [q for q in targets if q[2] != pytest.approx(0.1)] == []
    assert result.end_q == pytest.approx(RS_STOPPED)
    state = h.client.state
    assert (state.stop_level, state.owner, state.torque_on) == ("none", "agent", True)


# Commands that reach no motor. The real motor library raises nothing then:
# these show what the robot app does, and what it cannot know.


def test_a_release_on_the_rs_driver_is_reported_as_sent_not_as_confirmed(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    assert not h.arm.release_confirmed()
    released = final(h, h.client.release())
    assert (released.status, released.reason) == ("completed", "")
    assert sdk.disconnects == 1
    # The driver's disconnect reports nothing back and may not be read after.
    assert not h.arm.release_confirmed()
    _, text, data = h.heard("release")[-1]
    assert text == (
        "torque release sent at rest: the driver does not confirm that the "
        "motors are off"
    )
    assert data["confirmed"] is False
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on


@pytest.mark.parametrize(
    "stop",
    [
        lambda app: app.stop("Ctrl+C", "thor_terminal"),
        lambda app: app._fault("motor fault"),
    ],
    ids=["a STOP", "a fault-stop"],
)
def test_a_stop_after_a_release_on_the_rs_driver_commands_no_motor(
    rs_app: tuple[AppHarness, MotorSdk], stop: Callable[[Any], Any]
) -> None:
    h, sdk = rs_app
    assert final(h, h.client.release()).status == "completed"
    assert sdk.disconnects == 1
    jaw, joints = len(sdk.gripper.sent), len(sdk.arm.sent)
    stop(h.app)
    h.step()
    # The light jaw hold of a stop goes to no motor once the driver is released.
    assert (len(sdk.gripper.sent), len(sdk.arm.sent)) == (jaw, joints)
    assert not h.arm.torque_on()


def test_a_shutdown_on_the_rs_driver_says_the_release_was_sent(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    # The app closes after this RESULT, so no ROBOT_STATE is waited for.
    result = h.client.wait_result(h.client.shutdown(), timeout_s=WAIT_S)
    assert result.status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert h.heard("shutdown")[-1][1] == (
        "shutdown: torque release sent at rest; the robot app closes"
    )
    assert sdk.disconnects == 1


def test_joint_targets_that_reach_no_motor_end_the_move_at_a_contact(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    h.new_task()
    assert rs_move(h, straight(REST, RS_RAISED, 25), REST).status == "completed"
    sdk.arm.lost_from = len(sdk.arm.sent) + 5
    result = rs_move(h, straight(RS_RAISED, RS_TURN, 60), RS_RAISED)
    # The joints still answer, so the arm that no longer follows reads as
    # blocked: no ArmFault, no fault-stop.
    assert (result.status, result.reason) == ("failed", "contact")
    assert "stalled" in result.detail["contact"]
    assert result.detail["sent"] > 5
    # Every target after the fifth, the hold at the measured pose included.
    assert len(sdk.arm.lost) == result.detail["sent"] - 5 + 1
    state = h.client.state
    assert (state.stop_level, state.owner, state.torque_on) == ("none", "agent", True)


def test_a_light_jaw_hold_that_reaches_no_motor_raises_nothing(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    cancel = threading.Event()
    assert arm.gripper_command(1.0, 1.0, "plain", cancel).status == "reached"
    sdk.gripper.stop = 2.0  # an object between the jaws
    sdk.gripper.lost_from = len(sdk.gripper.sent) + 1  # the close itself arrives
    outcome = arm.gripper_command(0.0, 1.0, "plain", cancel)
    assert outcome.status == "stalled"
    arm.gripper_light_hold()
    # Nothing tells the app that the jaws are still at the closing stiffness.
    assert [kp for _, kp in sdk.gripper.lost] == [[RS_HOLD_KP], [RS_HOLD_KP]]
    assert sdk.gripper.sent[-1][1] != [RS_HOLD_KP]
    assert arm.faults() == []


# The jaw reference on the reBot RS driver: a cancel, and the typed command

RS_REF_KP = 2.0  # the stiffness of the guard's referencing close


def test_rs_a_cancel_ends_a_reference_within_one_pass_of_the_guard(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    jaw, cancel, marks = sdk.gripper, threading.Event(), []
    guard = arm.backend._grip_guard
    assert guard.ref_kp == RS_REF_KP
    probe = guard._stopped
    jaw._mm["jaw"].value = 3.0  # the jaws are half open
    jaw.per_command = 0.06  # and trail the guard's closing ramp

    def cancel_part_closed() -> None:
        if jaw.at() <= 2.5 and not cancel.is_set():
            marks.append(len(jaw.sent))
            cancel.set()

    sdk.guard_time.on_sleep = cancel_part_closed
    arm.gripper_reference(cancel)  # ended by the cancel: it raises nothing

    assert not arm.gripper_referenced() and arm.gripper_read() is None
    at = jaw.at()
    assert 2.4 < at <= 2.5
    assert min(targets_at(jaw, 0, RS_REF_KP)) < at - 0.2  # the ramp was ahead
    # Not one more command of the closing ramp went out; the one command since
    # holds the jaws where they are, at the hold stiffness.
    assert targets_at(jaw, marks[0], RS_REF_KP) == []
    assert jaw.sent[marks[0] :] == [([pytest.approx(at)], [RS_HOLD_KP])]
    # The driver's own stop flag was not touched and the guard asks it again.
    assert arm.backend._stopped is False and guard._stopped is probe
    # A stop's light hold sends nothing to jaws without a reference.
    sends = len(jaw.sent)
    arm.gripper_light_hold()
    assert len(jaw.sent) == sends

    # The jaws can be referenced afterwards.
    sdk.guard_time.on_sleep = None
    arm.gripper_reference(threading.Event())
    assert arm.gripper_referenced()
    assert arm.gripper_read() == pytest.approx(0.0)
    assert guard._stopped is probe


def test_rs_a_cancel_set_before_a_reference_sends_nothing(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    cancel = threading.Event()
    cancel.set()
    arm.gripper_reference(cancel)
    assert not arm.gripper_referenced()
    assert sdk.gripper.sent == [] and sdk.gripper._mm["jaw"].reads == []
    assert sdk.gripper.at() == pytest.approx(0.5)


def test_rs_a_reference_that_fails_is_an_arm_fault_with_a_cancel_too(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    sdk.gripper._mm["jaw"].value = 1.5
    sdk.gripper.stop = 1.0  # something between the jaws
    with pytest.raises(ArmFault, match=r"could not be referenced: gripper reference"):
        arm.gripper_reference(threading.Event())
    assert not arm.gripper_referenced()
    assert arm.backend._grip_guard._stopped() is False


@pytest.fixture
def rs_reference_app(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[AppHarness, MotorSdk]]:
    """As rs_app, with the setting jaws.allow_reference on."""
    rig = apai_backend.config_for_profile(config, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    checkers = apai_backend.build_apai_checkers(rig, rs_profile, clock)
    h = AppHarness(rig, arm, checkers, clock, app_options={"allow_reference": True})
    try:
        h.start()
        assert h.client.state.gripper.referenced
        yield h, sdk
    finally:
        h.close()


def lose_the_jaw_reference(h: AppHarness, sdk: MotorSdk) -> None:
    """Three jaw reads in a row get no answer: APAI's guard drops its reference."""
    sdk.gripper._mm["jaw"].missed = 3
    for _ in range(3):
        assert h.arm.gripper_read() is None
    assert not h.arm.gripper_referenced()


def test_a_typed_reference_references_the_jaws_and_rest_is_accepted_after_an_open(
    rs_reference_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_reference_app
    jaw = sdk.gripper
    # The agent closes the jaws on an object, raises the arm and gives it back.
    hold_an_object(h, sdk, 2.0)
    assert rs_move(h, straight(REST, RS_RAISED, 25), REST).status == "completed"
    given = final(h, h.client.request_owner("nobody", reason="task done"))
    assert given.status == "completed"
    lose_the_jaw_reference(h, sdk)

    # The defect: the raised arm can neither rest nor release, nor open its jaws.
    thor = h.thor_terminal()
    told = thor.ask("rest", "path is clear", answer="rest: ")
    assert told.startswith("rest: refused: the jaws are not proven empty (unknown)")
    assert "refused" in thor.ask("release", answer="release: ")
    assert final(h, h.client.open_jaws()).reason == "the jaws are not referenced"
    targets = len(sdk.arm.sent)

    # The way out. The person takes the object away and types the phrase.
    jaw.stop = 0.0
    sends = len(jaw.sent)
    told = thor.ask("reference", "jaws are empty", answer="reference: ")
    assert told == "reference: completed"
    closing = targets_at(jaw, sends, RS_REF_KP)
    # A slow close in small steps, from where the jaws were.
    assert len(closing) > 10 and max(closing) == pytest.approx(2.0)
    assert jaw.at() == pytest.approx(0.0)
    state = h.client.wait_state(
        lambda s: "reference" in (s.last_result or {}).get("command_id", ""), WAIT_S
    )
    assert state.gripper.referenced
    assert state.gripper.opening == pytest.approx(0.0)
    assert (state.owner, state.stop_level, state.torque_on) == ("nobody", "none", True)
    # The reference does not clear "may be holding": rest is still refused.
    assert state.gripper.may_be_holding and state.gripper.load == "loaded"
    told = thor.ask("rest", "path is clear", answer="rest: ")
    assert told.startswith("rest: refused: the jaws are not proven empty (loaded)")

    # The guarded opening from the console clears it at a measured full open.
    opened = final(h, h.client.open_jaws())
    assert opened.status == "completed"
    assert opened.detail["opening"] == pytest.approx(1.0)
    assert len(sdk.arm.sent) == targets  # no joint moved so far
    assert thor.ask("rest", "path is clear", answer="rest: ") == "rest: completed"
    assert h.app.motion_state is MotionState.REST_HOLD
    assert sdk.arm.sent[-1][0] == list(REST)
    assert h.arm.torque_on() and sdk.disconnects == 0
    assert h.arm.backend._stopped is False


def test_a_stop_ends_a_reference_without_a_fault_stop(
    rs_reference_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_reference_app
    jaw, marks = sdk.gripper, []
    assert final(h, h.client.open_jaws()).status == "completed"
    lose_the_jaw_reference(h, sdk)
    jaw._mm["jaw"].value = 3.0
    jaw.per_command = 0.06  # the jaws trail the guard's closing ramp

    def stop_part_closed() -> None:
        # Inside the guard's 50 ms sleep, the jaws 0.5 rad into the close.
        if marks or jaw.at() > 2.5:
            return
        marks.append(len(jaw.sent))
        h.client.stop("operator pressed stop")
        deadline = time.monotonic() + WAIT_S
        while h.app.snapshot().stop_number == 0 and time.monotonic() < deadline:
            time.sleep(0.001)

    sdk.guard_time.on_sleep = stop_part_closed
    done: list[Result] = []
    ended = threading.Event()

    def on_done(result: Result) -> None:
        done.append(result)
        ended.set()

    assert h.app.request_reference(True, on_done) is None
    assert ended.wait(WAIT_S)
    assert h.app._worker.wait_idle(WAIT_S)
    (result,) = done
    assert (result.status, result.reason) == (
        "cancelled",
        "stopped before the jaws were referenced",
    )
    # A STOP, not a fault-stop: nothing is latched.
    state = h.app.snapshot()
    assert (state.stop_level.value, state.fault_reason) == ("stop", "")
    assert h.heard("fault") == []
    # The close ended within one pass; the jaws are held where they are.
    at = jaw.at()
    assert 2.4 < at <= 2.5
    assert targets_at(jaw, marks[0], RS_REF_KP) == []
    assert jaw.sent[marks[0] :] == [([pytest.approx(at)], [RS_HOLD_KP])]
    assert not h.arm.gripper_referenced()
    assert h.arm.backend._stopped is False  # the driver was not stopped
    assert h.arm.torque_on() and sdk.disconnects == 0
    # The reference can be typed again and then runs to the closed stop.
    sdk.guard_time.on_sleep = None
    ended.clear()
    assert h.app.request_reference(True, on_done) is None
    assert ended.wait(WAIT_S)
    assert done[-1].status == "completed" and h.arm.gripper_referenced()


def test_unresolved_hold_reads_the_marker_of_the_profiles_bus(
    rs_profile: apai_backend.ApaiProfile,
    profile: apai_backend.ApaiProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apai.control import arm_hold

    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(tmp_path))
    assert apai_backend.unresolved_hold(rs_profile) is None
    key = arm_hold.hold_key("rebot_rs", rs_profile.arm_cfg)
    assert key
    arm_hold.record("another-bus", arm="rebot_rs", reason="held elsewhere")
    assert apai_backend.unresolved_hold(rs_profile) is None
    written = arm_hold.record(key, arm="rebot_rs", reason="left raised by a run")
    marker = apai_backend.unresolved_hold(rs_profile)
    assert marker == written
    assert (marker["key"], marker["reason"]) == (key, "left raised by a run")
    # A marker that cannot be read counts as a hold.
    arm_hold.marker_path(key).write_text("not json")
    assert apai_backend.unresolved_hold(rs_profile)["unreadable"] is True
    # The mock arm drives no bus: it is never held, whatever the folder has.
    assert arm_hold.hold_key("mock", profile.arm_cfg) is None
    assert apai_backend.unresolved_hold(profile) is None
    # Only read: the folder holds the two markers written here.
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [arm_hold.marker_path(key).name, arm_hold.marker_path("another-bus").name]
    )


def test_hold_bus_and_speed_cap_come_from_the_profile(
    rs_profile: apai_backend.ApaiProfile, profile: apai_backend.ApaiProfile
) -> None:
    from apai.control import arm_hold

    bus = apai_backend.hold_bus(rs_profile)
    assert bus and bus == arm_hold.hold_key("rebot_rs", rs_profile.arm_cfg)
    assert apai_backend.hold_bus(profile) is None  # the mock arm drives no bus
    assert apai_backend.speed_cap(rs_profile) == 0.8
    assert apai_backend.speed_cap(rs_profile) == float(
        rs_profile.apai_cfg.safety.max_joint_vel
    )


@pytest.mark.parametrize(
    "named", [None, "can4", "can4.", " can4", "can4/", "another-bus"]
)
def test_any_confirmation_file_in_the_hold_folder_is_on_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: str | None
) -> None:
    from apai.control import arm_hold

    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(tmp_path / "holds"))
    assert apai_backend.confirmation_on_file() is None  # no folder yet
    marker = arm_hold.record("can4", arm="rebot_rs", reason="a hold is no confirmation")
    assert apai_backend.confirmation_on_file() is None
    written = arm_hold.confirm("holding the arm", named)
    assert apai_backend.confirmation_on_file() == str(written)
    # Only read: nothing was claimed or removed.
    assert sorted(path.name for path in written.parent.iterdir()) == sorted(
        [arm_hold.marker_path(marker["key"]).name, written.name]
    )
    written.unlink()
    assert apai_backend.confirmation_on_file() is None


@pytest.mark.parametrize(
    "content", ["not json", "", '{"key": "can4", "at": 1.0}', '{"key": "can0"}']
)
def test_a_confirmation_file_counts_whatever_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    from apai.control import arm_hold

    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(tmp_path))
    path = tmp_path / "can4.confirm.json"
    path.write_text(content)
    assert arm_hold.pending_confirmations() == []  # APAI lists none of them
    assert apai_backend.confirmation_on_file() == str(path)
    path.chmod(0)  # one that cannot be read
    assert apai_backend.confirmation_on_file() == str(path)


def test_a_hold_folder_that_cannot_be_listed_counts_as_a_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "holds"
    folder.write_text("a file where the folder would be")
    monkeypatch.setenv("APAI_ARM_HOLD_DIR", str(folder))
    assert apai_backend.confirmation_on_file() is None  # nothing can lie in it
    assert apai_backend.hold_folder_unlistable() is None
    folder.unlink()
    assert apai_backend.hold_folder_unlistable() is None  # no folder at all
    folder.mkdir()
    (folder / "_any.confirm.json").write_text("{}")
    assert apai_backend.hold_folder_unlistable() is None
    folder.chmod(0o300)  # searchable, not listable: APAI could still claim the file
    try:
        assert apai_backend.confirmation_on_file() == str(folder)
        assert apai_backend.hold_folder_unlistable() == (
            f"[Errno 13] Permission denied: '{folder}'"
        )
    finally:
        folder.chmod(0o700)
    assert sorted(path.name for path in folder.iterdir()) == ["_any.confirm.json"]
