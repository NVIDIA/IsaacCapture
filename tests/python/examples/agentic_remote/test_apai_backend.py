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
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion, Result
from isaaccapture_examples.agentic_remote.robot_app.app import (
    MOTORS_UNKNOWN,
    NEW_START,
    RELEASE_FAILED,
    TORQUE_UNKNOWN,
)
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import (
    Clock,
    FakeClock,
    SystemClock,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (
    GraspConfig,
    Limits,
    LimitSet,
    RestConfig,
    RobotAppConfig,
)
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
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
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.state import JointHistory
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


def test_on_the_rig_file_only_the_agent_enters_the_joint_margin_at_the_rest_stops(
    clock: FakeClock,
) -> None:
    """What the fold rule lets the Thor's agent checker accept, and nobody else's.

    The joint margin is the 0.02 rad a harness keeps clear of each joint
    stop. The rest pose lies on the lower stops of the second and third
    joint. An agent setpoint may be inside the margin at these two stops
    whatever the other joints do: down to the stop and no further, and at
    the agent's speed limit. A joint that reads below its stop may come back
    up. Every other stop keeps its margin, and so does a person's checker.
    """
    from isaaccapture_examples.agentic_remote.robot_app.config import load_config
    from repo_paths import repo_root

    file = repo_root() / "examples/agentic_remote/configs/robot_app_rebot.yaml"
    profile = apai_backend.load_apai_profile("rebot_rs")
    rig = apai_backend.config_for_profile(load_config(file), profile)
    checkers = apai_backend.build_apai_checkers(rig, profile, clock)
    agent, human, recovery = checkers.agent, checkers.human, checkers.recovery
    for checker in checkers.all():
        start_motion(checker)
    margin = 0.02
    assert agent.harness.limits.joint_margin == margin
    assert human.harness.limits.joint_margin == margin
    assert recovery.harness.limits.joint_margin == 0.0  # it reaches the stops itself
    speed = rig.limits.agent.max_joint_speed
    assert agent.harness.limits.max_joint_vel == speed
    step = 0.5 * speed * STEP_S  # rad a period: half the agent's speed limit
    low, high = (np.asarray(s, float) for s in profile.kinematics.joint_limits)
    assert list(low[1:3]) == list(rig.rest.rest_q[1:3]) == [0.0, 0.0]
    # The arm raised, the second joint half a step above its margin.
    raised = [0.3, margin + step / 2, 0.5, 0.0, 0.2, 0.0]
    inside = margin - step / 2
    on_stop = moved(raised, 1, 0.0)
    refused = rf"joint 2 target {inside:.3f} rad outside \[0\.020, 3\.120\]"

    def own_margin(start: Sequence[float], target: Sequence[float]) -> str | None:
        return agent.harness.vet_step(np.array(start), np.array(target), STEP_S)

    # Inside the margin at the two stops; the harness's own margin refuses each.
    accepted = {
        "into the margin, the arm raised": (raised, moved(raised, 1, inside)),
        "that, while the base turns away from the rest pose": (
            raised,
            moved(moved(raised, 1, inside), 0, 0.3 + step),
        ),
        "down onto the stop": (moved(raised, 1, step), on_stop),
        "a hold on the stop": (on_stop, on_stop),
        "the third joint": (
            [0.3, 0.5, margin + step / 2, 0.0, 0.2, 0.0],
            [0.3 + step, 0.5, inside, 0.0, 0.2, 0.0],
        ),
        "one joint back up from below its stop while the other folds": (
            [0.0, -2 * step, 2 * step, 0.0, 0.0, 0.0],
            [0.0, -step, step, 0.0, 0.0, 0.0],
        ),
    }
    for name, (start, target) in accepted.items():
        assert "outside [0.020" in own_margin(start, target), name
        assert agent._fold_margin(np.array(start), np.array(target)) == 0.0, name
        agent.check_setpoint(start, target, STEP_S)
        # A person's checker and the return to rest have no such rule.
        assert human._fold_margin(np.array(start), np.array(target)) is None, name
        assert recovery._fold_margin(np.array(start), np.array(target)) is None, name
        refusal_of(
            lambda s=start, t=target: human.check_setpoint(s, t, STEP_S),
            r"rad outside \[0\.020, 3\.120\]",
        )

    # Never below the stop, never faster than the agent's speed limit.
    below = moved(raised, 1, -0.001)
    for start in (moved(raised, 1, 0.001), below):
        refusal_of(
            lambda start=start: agent.check_setpoint(start, below, STEP_S),
            r"joint 2 target -0\.001 rad outside \[0\.020, 3\.120\]",
        )
    refusal_of(
        lambda: agent.check_setpoint(moved(raised, 1, 3 * step), on_stop, STEP_S),
        rf"joint 2 velocity {1.5 * speed:.2f} rad/s exceeds {speed:.2f}",
    )
    # A stop the rest pose does not lie on keeps its margin, alone and
    # together with a step into the margin at the second joint.
    edge = high[4] - margin
    kept = {
        r"joint 5 target 1\.551 rad outside \[-1\.550, 1\.550\]": (
            moved(raised, 4, edge - 0.001),
            moved(raised, 4, edge + 0.001),
        ),
        r"joint 4 target -1\.671 rad outside \[-1\.670, 1\.770\]": (
            moved(raised, 3, -1.669),
            moved(raised, 3, -1.671),
        ),
        refused: (
            moved(raised, 4, edge - 0.001),
            moved(moved(raised, 1, inside), 4, edge + 0.001),
        ),
    }
    for text, (start, target) in kept.items():
        assert agent._fold_margin(np.array(start), np.array(target)) is None, text
        refusal_of(lambda s=start, t=target: agent.check_setpoint(s, t, STEP_S), text)

    # Random steps around the stops: where a setpoint the agent's checker
    # accepts lies inside a margin, it is at one of the two stops and not
    # below it, or the joint moves back out of the margin as the harness
    # lets any joint. A person's checker accepts only the latter.
    rng = np.random.default_rng(0)
    entered = 0
    for _ in range(3000):
        stop = np.where(rng.random(6) < 0.5, low, high) + rng.uniform(-0.03, 0.03, 6)
        near = rng.random(6) < 0.3
        near[1:3] |= rng.random(2) < 0.5
        start = np.where(near, stop, np.array(raised))
        target = start + rng.uniform(-1.5 * step, 1.5 * step, 6)
        under, over = target < low + margin - 1e-9, target > high - margin + 1e-9
        leaves = (under & (target > start)) | (over & (target < start))
        at_rest_stop = under & np.isin(np.arange(6), (1, 2)) & (target >= low - 1e-9)
        for checker, allowed in ((agent, leaves | at_rest_stop), (human, leaves)):
            try:
                checker.check_setpoint(start, target, STEP_S)
            except Refusal:
                continue
            assert (allowed | ~(under | over)).all(), (checker.name, start, target)
            if checker is agent and (at_rest_stop & ~leaves).any():
                entered += 1
    assert entered > 100


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


def tip_at(profile: apai_backend.ApaiProfile, axis: int, value: float) -> list[float]:
    """Joint values near HOME whose gripper tip has this coordinate on axis, m."""
    pose = np.array(profile.kinematics.fk(np.array(HOME)), dtype=float)
    pose[axis, 3] = value
    solved = profile.kinematics.ik(pose, q_init=np.array(HOME), retries=0)
    assert solved.success
    return [float(q) for q in solved.q]


@pytest.mark.parametrize("mounted", [False, True], ids=["no base_pose", "base_pose"])
def test_the_tip_box_is_the_box_the_harness_holds_a_persons_tip_in(
    config: RobotAppConfig,
    profile: apai_backend.ApaiProfile,
    clock: FakeClock,
    mounted: bool,
) -> None:
    # What WELCOME announces as human_tip_box: read off the checker that
    # enforces it. With a base_pose, an arm bolted away from the table's
    # origin, the harness still compares the arm model's FK with the same two
    # corners: the box stays in the arm's base frame, the frame the human
    # program's IK works in.
    arm_cfg = profile.arm_cfg.as_dict()
    if mounted:
        arm_cfg["base_pose"] = [0.2, -0.1, 0.0, 0.0, 0.0, 0.0]
    stood = dataclasses.replace(profile, arm_cfg=arm_cfg)
    human = apai_backend.build_apai_checkers(config, stood, clock).human
    assert (human.harness.base_pose is not None) == mounted
    limits = human.harness.limits
    low, high = human.tip_box()
    assert low == [float(value) for value in limits.workspace_min]
    assert high == [float(value) for value in limits.workspace_max]
    assert all(type(value) is float for value in (*low, *high))
    assert all(a < b for a, b in zip(low, high))
    # A step whose tip ends a millimetre inside the front wall is taken, and
    # one a millimetre outside it is refused for the workspace, whatever the
    # base_pose: the announced wall is the enforced one.
    front = high[0]
    at, inside = tip_at(profile, 0, front - 0.02), tip_at(profile, 0, front - 0.001)
    outside = tip_at(profile, 0, front + 0.001)
    human.check_path(at, [inside], 10.0)
    refused = refusal_of(
        lambda: human.check_path(at, [outside], 10.0), r"outside workspace"
    )
    assert refused.reason.startswith("setpoint 0: TCP ")
    # A tip outside may only come back toward the box.
    further = tip_at(profile, 0, front + 0.004)
    human.check_path(further, [outside], 10.0)
    refusal_of(lambda: human.check_path(outside, [further], 10.0), r"outside workspace")


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


def test_the_mock_arms_peek_is_its_jaw_read(arm: apai_backend.ApaiArm) -> None:
    arm.connect()
    assert arm.gripper_peek() is None  # not referenced
    arm.gripper_reference()
    arm.gripper_command(0.4, 0.5, "plain", threading.Event())
    assert arm.gripper_peek() == arm.gripper_read() == pytest.approx(0.4)


def test_holding_is_said_only_of_joint_readings_at_most_a_second_old(
    arm: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    holding = (apai_backend.MOTORS_HOLDING, "every joint reads at the target it holds")
    arm.connect()
    arm.hold_measured()
    clock.advance(arm._hold._after_s + 0.01)
    arm.read_all()
    assert arm.hold_evidence() == holding
    clock.advance(apai_backend.HOLD_READ_AGE_S - 0.1)
    assert arm.hold_evidence() == holding
    # The joints stop answering: nothing reads the hold any more.
    clock.advance(0.2)
    names = ", ".join(arm.joint_names)
    assert arm.hold_evidence() == (
        apai_backend.MOTORS_UNKNOWN,
        f"{names} not read for more than 1 s",
    )
    assert arm.faults() == []  # a missing reading is no joint off its hold
    # Each joint that is read at its target again counts again.
    arm.read_joint(2)
    assert arm.hold_evidence() == (
        apai_backend.MOTORS_UNKNOWN,
        "joint1, joint2, joint4, joint5, joint6 not read for more than 1 s",
    )
    arm.read_all()
    assert arm.hold_evidence() == holding


def test_no_motor_fault_is_judged_before_the_connect_has_its_baseline(
    arm: apai_backend.ApaiArm,
) -> None:
    during: list[Any] = []

    class StandingFault:
        """A motor whose fault word stays set after the driver cleared it."""

        def robstride_get_fault_report(self) -> tuple[int, int]:
            if not during:
                # As the fault poll of another thread, while the baseline is read.
                during.append(None)
                during[0] = (arm.torque_on(), arm.faults())
            return 4, 0

    arm.backend._arm = types.SimpleNamespace(_motor_map={"joint2": StandingFault()})
    arm.connect()
    assert during == [(False, [])]
    assert arm.torque_on() and arm.faults() == []


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
    # A wider opening than the jaw target in force is the driver's opening.
    assert connected.gripper_command(1.0, 0.8, "plain", cancel) == GripperOutcome(
        "reached", pytest.approx(1.0)
    )
    assert connected.backend.get_state().gripper_pos == pytest.approx(0.0)
    assert connected.gripper_last_target() == pytest.approx(1.0)
    # A close returns once it is sent, as the driver's set_gripper does.
    outcome = connected.gripper_command(0.25, 0.8, "plain", cancel)
    assert outcome == GripperOutcome("sent", pytest.approx(0.25))
    assert connected.backend.get_state().gripper_pos == pytest.approx(0.75)
    assert connected.gripper_read() == pytest.approx(0.25)
    assert clock.monotonic() == before  # no plain command waits for the jaws


def test_a_plain_close_onto_an_object_is_sent_and_keeps_its_stiffness(
    connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    cancel = threading.Event()
    connected.gripper_command(1.0, 0.8, "plain", cancel)
    connected.backend.object_stop_frac = 0.6  # an object at opening 0.4
    before = clock.monotonic()
    outcome = connected.gripper_command(0.0, 0.8, "plain", cancel)

    assert outcome == GripperOutcome("sent", pytest.approx(0.4))
    assert clock.monotonic() == before
    assert connected.gripper_last_target() == pytest.approx(0.0)  # the target stands
    assert connected.backend._gripper_effort == pytest.approx(0.8)  # no light hold


def test_a_close_until_contact_waits_for_the_jaws_and_keeps_its_stiffness(
    connected: apai_backend.ApaiArm, clock: FakeClock
) -> None:
    from apai.control.robstride import GRIP_GRACE_S, GRIP_POLL_S

    cancel = threading.Event()
    connected.gripper_command(1.0, 0.8, "plain", cancel)
    connected.backend.object_stop_frac = 0.6  # an object at opening 0.4
    before = clock.monotonic()
    outcome = connected.gripper_command(0.0, 0.8, "close_until_contact", cancel)

    assert outcome == GripperOutcome("stalled", pytest.approx(0.4))
    # APAI's grace after a command, then one more look without movement.
    assert clock.monotonic() - before == pytest.approx(GRIP_GRACE_S + GRIP_POLL_S)
    assert connected.gripper_last_target() == pytest.approx(0.0)  # the target stands
    # The jaws go on pushing at the close's stiffness, as after APAI's own close.
    assert connected.backend._gripper_effort == pytest.approx(0.8)
    assert connected.gripper_read() == pytest.approx(0.4)


def test_the_other_modes_on_the_mock_arm_are_its_one_jaw_routine(
    connected: apai_backend.ApaiArm,
) -> None:
    from apai.grasping.force import select_profile

    cancel = threading.Event()
    # "open": the mock driver has no opening of its own: set_gripper to open.
    assert connected.gripper_command(0.0, 0.6, "open", cancel) == GripperOutcome(
        "reached", pytest.approx(1.0)
    )
    assert connected.backend._gripper_effort == pytest.approx(0.6)
    # The typed close: the seat stage of APAI's rigid grip profile, at its effort.
    rigid = select_profile("", "rigid")
    closed = connected.gripper_command(0.0, 0.5, "close", cancel)
    assert closed == GripperOutcome(
        "reached", pytest.approx(1.0 - rigid.close_frac_stage2)
    )
    assert connected.backend._gripper_effort == pytest.approx(rigid.effort)
    # A two-stage close: its seat stage alone.
    two = connected.gripper_close_two_stage(0.5, 0.05, 0.45, cancel)
    assert two == GripperOutcome("reached", pytest.approx(0.05))
    assert connected.backend._gripper_effort == pytest.approx(0.45)
    # The close before a torque release: to the closed stop.
    assert connected.gripper_command(0.0, 0.6, "close_torque", cancel).status == (
        "reached"
    )
    assert connected.gripper_read() == pytest.approx(0.0)
    assert connected.gripper_close_torque(cancel).status == "reached"


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
    outcome = connected.gripper_command(0.0, 0.8, "close_until_contact", cancel)
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


@pytest.mark.parametrize("opening", [0.5, 1.0])
def test_a_close_until_contact_on_jaws_that_cannot_be_read_ends_stalled(
    connected: apai_backend.ApaiArm,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    opening: float,
) -> None:
    monkeypatch.setattr(connected, "gripper_read", lambda: None)
    before = clock.monotonic()
    outcome = connected.gripper_command(
        opening, 0.8, "close_until_contact", threading.Event()
    )
    # The end of the wait is no fault, and the close keeps its stiffness.
    assert outcome == GripperOutcome("stalled", None)
    assert clock.monotonic() - before == pytest.approx(4.05)
    assert connected.backend._gripper_effort == pytest.approx(0.8)


def test_a_plain_command_on_jaws_that_cannot_be_read_does_not_wait(
    connected: apai_backend.ApaiArm, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connected, "gripper_read", lambda: None)
    before = clock.monotonic()
    outcome = connected.gripper_command(0.0, 0.8, "plain", threading.Event())
    assert outcome == GripperOutcome("sent", None)
    assert clock.monotonic() == before


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
    assert jaw.sent == []

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
    # A close is one command of the driver, which returns once it is sent.
    assert arm.gripper_command(0.5, 0.5, "plain", cancel) == GripperOutcome(
        "sent", pytest.approx(0.5)
    )
    assert jaw.sent[-1] == ([pytest.approx(3.1)], [1.0])
    assert arm.gripper_last_target() == pytest.approx(0.5)

    jaw.stop = 2.0  # an object
    sends = len(jaw.sent)
    outcome = arm.gripper_command(0.0, 0.5, "close_until_contact", cancel)
    assert outcome == GripperOutcome("stalled", pytest.approx(2.0 / 6.2))
    # One command: the jaws go on pushing at the closed target with the
    # close's stiffness, as after the driver's own close.
    assert jaw.sent[sends:] == [([0.0], [1.0])]
    assert arm.gripper_last_target() == pytest.approx(0.0)

    # A plain close onto the object: sent, and nothing after it.
    sends = len(jaw.sent)
    assert arm.gripper_command(0.0, 0.5, "plain", cancel) == GripperOutcome(
        "sent", pytest.approx(2.0 / 6.2)
    )
    assert jaw.sent[sends:] == [([0.0], [1.0])]
    # The light hold is its own call: a STOP's.
    sends = len(jaw.sent)
    arm.gripper_light_hold()
    assert jaw.sent[sends:] == [([0.0], [0.01])]


def test_rs_the_peek_reads_the_jaw_motor_with_the_idle_reads_timeout(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    jaw = sdk.gripper._mm["jaw"]
    reads = len(jaw.reads)
    assert arm.gripper_peek() is None  # not referenced: nothing is read
    assert len(jaw.reads) == reads
    arm.gripper_reference()
    arm.gripper_command(0.5, 0.5, "plain", threading.Event())
    sends, reads = len(sdk.gripper.sent), len(jaw.reads)
    assert arm.gripper_peek() == pytest.approx(0.5)
    # One read of the motor's position, with an idle joint read's timeout in ms.
    assert jaw.reads[reads:] == [(0x7019, 50)]
    assert arm.read_joint(2).valid and sdk.arm._mm["motor3"].reads[-1] == (0x7019, 50)
    # The driver's own read passes none: the motor library waits 1 s.
    assert arm.gripper_read() == pytest.approx(0.5)
    assert jaw.reads[reads + 1 :] == [(0x7019, None)]
    assert sdk.gripper.sent[sends:] == []  # neither moves the jaws


@pytest.mark.parametrize("answer", [None, 2.3e18, math.nan, 9.0, -1.0])
def test_rs_a_peek_without_a_jaw_position_is_unknown_and_keeps_the_reference(
    rs: tuple[apai_backend.ApaiArm, MotorSdk], answer: float | None
) -> None:
    """None: no answer. 2.3e18: read on the rig. 9.0, -1.0: outside the travel."""
    arm, sdk = rs
    arm.gripper_reference()
    arm.gripper_command(0.5, 0.5, "plain", threading.Event())
    jaw, guard = sdk.gripper._mm["jaw"], arm.backend._grip_guard
    at = jaw.value
    for _ in range(5):  # the guard gives up its reference at the third
        if answer is None:
            jaw.missed = 1
        else:
            jaw.value = answer
        assert arm.gripper_peek() is None
    assert arm.gripper_referenced() and guard._read_failures == 0
    assert arm.gripper_last_target() == pytest.approx(0.5)
    jaw.value = at
    assert arm.gripper_peek() == pytest.approx(0.5)
    # The driver's own read, a jaw command's, counts them as before.
    jaw.missed = 3
    assert [arm.gripper_read() for _ in range(3)] == [None] * 3
    assert not arm.gripper_referenced()
    assert arm.gripper_peek() is None


def test_rs_no_peek_reaches_a_driver_that_was_released(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    arm.release_torque()
    reads = len(sdk.gripper._mm["jaw"].reads)
    assert arm.gripper_peek() is None
    assert len(sdk.gripper._mm["jaw"].reads) == reads


def test_rs_the_driver_references_the_jaws_before_their_first_command(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    # As at the arm: APAI's guard closes unreferenced jaws slowly to their
    # stop, then runs the command.
    arm, sdk = rs
    jaw = sdk.gripper
    assert not arm.gripper_referenced() and jaw.at() == pytest.approx(0.5)
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert outcome == GripperOutcome("reached", pytest.approx(1.0))
    assert arm.gripper_referenced()
    positions = [q[0] for q, _ in jaw.sent]
    closed = positions.index(min(positions))
    assert positions[closed] <= 0.0  # the reference pressed the closed stop
    assert positions[closed:] == sorted(positions[closed:])  # then one opening


def test_rs_a_reference_before_a_command_that_finds_an_object_is_no_arm_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    sdk.gripper._mm["jaw"].value = 1.5
    sdk.gripper.stop = 1.0  # something between the jaws
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    # The driver's own text; the jaws stay unreferenced and where they stopped.
    assert outcome.status == "gripper_fault"
    assert outcome.detail.startswith("gripper reference failed: the jaws stopped")
    assert not arm.gripper_referenced()
    assert sdk.gripper.at() == pytest.approx(1.0)


def test_rs_the_two_stage_close_is_the_drivers(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apai.control.rebot_rs_arm as rebot_rs_arm
    from apai.control.robstride import GRIP_REACH_RAD

    arm, sdk = rs
    monkeypatch.setattr(rebot_rs_arm, "time", sdk.guard_time)
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.8, "plain", cancel)
    calls: list[dict[str, float]] = []
    close = arm.backend.close_gripper_two_stage
    monkeypatch.setattr(
        arm.backend,
        "close_gripper_two_stage",
        lambda **kwargs: calls.append(kwargs) or close(**kwargs),
    )

    # On nothing: both stages, the scout at 0.7 of the effort; reached.
    sends = len(jaw.sent)
    outcome = arm.gripper_close_two_stage(0.5, 0.15, 0.7, cancel)
    assert calls == [
        {
            "width_frac_stage1": pytest.approx(0.5),
            "width_frac_stage2": pytest.approx(0.85),
            "effort": 0.7,
        }
    ]
    assert jaw.sent[sends:] == [
        ([pytest.approx(3.1)], [pytest.approx(0.98)]),
        ([pytest.approx(0.93)], [pytest.approx(1.4)]),
    ]
    assert (outcome.status, outcome.stages_skipped) == ("reached", 0)
    assert outcome.opening == pytest.approx(0.15)

    # On an object: "stalled" once it is more than APAI's reach from the seat
    # stage, and the jaws keep pushing at the seat stage with the close's
    # stiffness: no light hold follows.
    arm.gripper_command(1.0, 0.8, "plain", cancel)
    jaw.stop = 0.93 + 2 * GRIP_REACH_RAD
    sends = len(jaw.sent)
    outcome = arm.gripper_close_two_stage(0.5, 0.15, 0.7, cancel)
    assert (outcome.status, outcome.stages_skipped) == ("stalled", 0)
    assert outcome.opening == pytest.approx(jaw.stop / RS_TRAVEL)
    assert jaw.sent[sends:][-1] == ([pytest.approx(0.93)], [pytest.approx(1.4)])
    assert not jaw.light_after(sends)
    # A second close on the seated jaws: the driver skips the scout stage.
    outcome = arm.gripper_close_two_stage(0.5, 0.15, 0.7, cancel)
    assert outcome.stages_skipped == 1


def test_rs_a_stop_during_the_two_stage_close_keeps_the_seat_stage_back(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apai.control.rebot_rs_arm as rebot_rs_arm

    arm, sdk = rs
    monkeypatch.setattr(rebot_rs_arm, "time", sdk.guard_time)
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.8, "plain", cancel)
    sends = len(jaw.sent)

    def stop_after_the_scout_stage() -> None:
        if len(jaw.sent) > sends:
            cancel.set()

    sdk.guard_time.on_sleep = stop_after_the_scout_stage
    outcome = arm.gripper_close_two_stage(0.5, 0.15, 0.7, cancel)
    assert outcome.status == "cancelled"
    # The scout stage, then only the light hold at it: never the seat stage.
    assert jaw.sent[sends:] == [
        ([pytest.approx(3.1)], [pytest.approx(0.98)]),
        ([pytest.approx(3.1)], [RS_HOLD_KP]),
    ]
    assert arm.backend._stopped is False  # the driver's own stop flag stays down


def test_rs_the_close_before_a_torque_release_is_the_drivers(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apai.control.rebot_rs_arm as rebot_rs_arm

    arm, sdk = rs
    monkeypatch.setattr(rebot_rs_arm, "time", sdk.guard_time)
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.8, "plain", cancel)
    calls: list[dict[str, float]] = []
    close = arm.backend.close_gripper_torque
    monkeypatch.setattr(
        arm.backend,
        "close_gripper_torque",
        lambda **kwargs: calls.append(kwargs) or close(**kwargs),
    )
    jaw.stop = 0.5  # the jaws meet something
    sends = len(jaw.sent)
    outcome = arm.gripper_close_torque(cancel)
    assert calls == [{}]  # the driver's own effort and limits
    assert (outcome.status, outcome.closed) == ("stalled", True)
    assert outcome.opening == pytest.approx(0.5 / RS_TRAVEL)
    # Closed at the driver's effort, then held lightly on what the jaws met.
    assert jaw.sent[sends:] == [([0.0], [pytest.approx(1.2)]), ([0.0], [RS_HOLD_KP])]
    # The same routine as a GRIPPER's mode, at the effort the GRIPPER names.
    jaw.stop = 0.0
    arm.gripper_command(1.0, 0.8, "plain", cancel)
    outcome = arm.gripper_command(0.0, 0.4, "close_torque", cancel)
    assert calls[-1] == {"effort": 0.4}
    assert (outcome.status, outcome.closed) == ("reached", True)


def test_rs_the_typed_open_and_close_are_the_drivers_routines(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apai.control.rebot_rs_arm as rebot_rs_arm
    from apai.grasping.force import select_profile

    arm, sdk = rs
    monkeypatch.setattr(rebot_rs_arm, "time", sdk.guard_time)
    jaw, cancel = sdk.gripper, threading.Event()
    arm.gripper_reference()
    opened: list[int] = []
    open_gripper = arm.backend.open_gripper
    monkeypatch.setattr(
        arm.backend, "open_gripper", lambda: opened.append(1) or open_gripper()
    )
    # "open": the driver's open_gripper, whatever effort the command names.
    outcome = arm.gripper_command(0.0, 0.1, "open", cancel)
    assert opened == [1]
    assert outcome == GripperOutcome("reached", pytest.approx(1.0))
    assert jaw.sent[-1] == ([pytest.approx(6.2)], [pytest.approx(1.6)])
    # "close": the driver's two-stage close with APAI's rigid grip profile.
    rigid = select_profile("", "rigid")
    sends = len(jaw.sent)
    outcome = arm.gripper_command(0.0, 0.1, "close", cancel)
    assert outcome.opening == pytest.approx(1.0 - rigid.close_frac_stage2)
    assert [kp for _, kp in jaw.sent[sends:]] == [
        [pytest.approx(2.0 * 0.7 * rigid.effort)],
        [pytest.approx(2.0 * rigid.effort)],
    ]


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
    # Nothing but that, the jaw tolerances and the driver's settle rule differs.
    assert (
        dataclasses.replace(
            rig, rest=config.rest, gripper=config.gripper, executor=config.executor
        )
        == config
    )
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
    outcome = arm.gripper_command(0.0, 0.5, "close_until_contact", cancel)
    assert outcome == GripperOutcome("cancelled", pytest.approx(2.0 / RS_TRAVEL))
    assert jaw.sent[sends:] == [([0.0], [1.0]), ([0.0], [RS_HOLD_KP])]
    assert arm.gripper_last_target() == pytest.approx(0.0)


def test_rs_a_missed_jaw_answer_before_an_opening_ends_it_as_the_driver_does(
    rs: tuple[apai_backend.ApaiArm, MotorSdk], clock: FakeClock
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    sends = len(sdk.gripper.sent)
    before = clock.monotonic()
    sdk.gripper._mm["jaw"].missed = 1
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    # The driver's own fault and text; nothing was sent, and nothing is tried
    # again: at the arm the skill fails here too.
    assert outcome.status == "gripper_fault"
    assert outcome.detail.startswith("gripper position unknown")
    assert outcome.detail.endswith("no gripper command sent")
    assert sdk.gripper.sent[sends:] == []
    assert clock.monotonic() == before
    # The arm stays usable: the next command opens the jaws.
    assert arm.gripper_referenced()
    opened = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert opened == GripperOutcome("reached", pytest.approx(1.0))


def test_rs_a_close_makes_no_read_before_it_is_sent(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    sends = len(sdk.gripper.sent)
    sdk.gripper._mm["jaw"].missed = 1
    outcome = arm.gripper_command(0.0, 0.5, "plain", threading.Event())
    # The one read that missed is the one after the command was sent.
    assert outcome == GripperOutcome("sent", None)
    assert sdk.gripper.sent[sends:] == [([0.0], [1.0])]


def test_rs_jaws_blocked_while_they_open_are_the_drivers_fault_not_an_arm_fault(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    sdk.gripper.blocked_at = 1.0
    outcome = arm.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert outcome.status == "gripper_fault"
    assert outcome.opening == pytest.approx(1.0 / RS_TRAVEL)
    assert outcome.detail.startswith(
        "gripper blocked while opening at 1.00 rad (target 6.20): holding there"
    )
    # The guard holds the jaws where they stopped, at the opening's stiffness.
    assert sdk.gripper.at() == pytest.approx(1.0)
    assert sdk.gripper.sent[-1] == ([pytest.approx(1.0)], [1.0])


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
    from apai.control import rs_gripper
    from apai.control.robstride import GRIP_REACH_RAD

    assert config.gripper.target_tol == 0.03  # 0.186 rad of the rig's 6.2 rad
    rig = apai_backend.config_for_profile(config, rs_profile)
    # How near its target APAI's driver takes a close as reached.
    assert rig.gripper.target_tol == pytest.approx(GRIP_REACH_RAD / RS_TRAVEL)
    # How far short of its target APAI's guarded opening may end.
    assert rig.gripper.full_open_tol == pytest.approx(
        rs_gripper.OPEN_SHORT_RAD / RS_TRAVEL
    )
    # Below this APAI takes the jaws as closed on nothing.
    assert rig.gripper.closed_frac == rs_profile.apai_cfg.grasp.air_grasp_frac
    # The tolerance of the jaws under a person's trigger stays as configured.
    assert rig.gripper.reach_tol == config.gripper.reach_tol
    # A tolerance already tighter than APAI's reach stays.
    tight = dataclasses.replace(
        config, gripper=dataclasses.replace(config.gripper, target_tol=0.01)
    )
    assert apai_backend.config_for_profile(tight, rs_profile).gripper.target_tol == 0.01
    # The mock arm's jaw position is not an angle: its tolerances stay.
    mock = apai_backend.config_for_profile(config, profile).gripper
    assert (mock.target_tol, mock.full_open_tol) == (0.03, 0.03)
    assert mock.closed_frac == profile.apai_cfg.grasp.air_grasp_frac


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
    # "Open" and "closed" must be APAI's too.
    copied = dataclasses.replace(
        rig, gripper=dataclasses.replace(rig.gripper, full_open_tol=0.03)
    )
    with pytest.raises(
        ValueError,
        match=r"gripper\.full_open_tol 0\.03 and gripper\.closed_frac 0\.04 differ "
        r"from APAI's 0\.0645161 and 0\.04: use config_for_profile\(\)",
    ):
        apai_backend.build_apai_arm(copied, rs_profile, clock)


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
    # The jaws grip at the close's stiffness: effort 0.5 of the profile's 2.0.
    assert sdk.gripper.sent[-1] == ([0.0], [1.0])


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
    mark = len(jaw.sent)
    open_id = h.client.open_jaws()
    result = final(h, open_id)
    assert result.status == "cancelled"
    # One ramp of the driver's own opening, 0.1 rad per pass: it ended at the
    # pass after the stop, and nothing but the light hold followed.
    ramp = [q[0] for q, _ in jaw.sent[mark : marks[0]]]
    assert ramp == pytest.approx([0.1 * step for step in range(len(ramp))])
    assert jaw.at() == pytest.approx(0.8)
    after = jaw.sent[marks[0] :]
    assert after and all(sent == ([pytest.approx(0.8)], [RS_HOLD_KP]) for sent in after)


@pytest.mark.parametrize(
    ("object_rad", "release", "reason"),
    [
        # At or below the closed line (0.04 of the travel, 0.248 rad): jaws told
        # to close count as closed on nothing.
        (0.08, "completed", ""),
        (0.15, "completed", ""),
        (
            0.3,
            "refused",
            "the jaws are not proven empty (loaded): the jaws are at 0.048, not at "
            "their target 0.000",
        ),
    ],
)
def test_jaws_closed_above_the_closed_line_are_not_proven_empty(
    rs_app: tuple[AppHarness, MotorSdk], object_rad: float, release: str, reason: str
) -> None:
    h, sdk = rs_app
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    sdk.gripper.stop = object_rad
    closed = final(h, h.client.gripper(0.0))
    assert closed.status == "completed"  # a plain close is done once it is sent
    assert closed.detail["opening"] == pytest.approx(object_rad / RS_TRAVEL)
    assert final(h, h.client.request_owner("nobody", reason="done")).status == (
        "completed"
    )
    h.client.wait_state(lambda s: s.owner == "nobody")
    released = final(h, h.client.release())
    assert (released.status, released.reason) == (release, reason)
    assert sdk.disconnects == (1 if release == "completed" else 0)
    assert h.arm.torque_on() is (release != "completed")


def test_an_agents_opening_from_jaws_that_give_no_answer_is_no_fault_stop(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    h.new_task()
    sends = len(sdk.gripper.sent)
    sdk.gripper._mm["jaw"].missed = 1
    result = final(h, h.client.gripper(1.0))
    # The driver's own fault, with its text: the agent's skill fails with it,
    # as at the arm, and the arm stays the agent's.
    assert result.status == "failed"
    assert result.detail["outcome"] == "gripper_fault"
    assert result.reason.startswith("gripper position unknown")
    assert result.reason.endswith("no gripper command sent")
    assert sdk.gripper.sent[sends:] == []
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == result.command_id
    )
    assert (state.stop_level, state.owner, state.torque_on) == ("none", "agent", True)
    sdk.gripper._mm["jaw"].missed = 0
    assert final(h, h.client.gripper(1.0)).status == "completed"


def test_an_agents_opening_that_is_blocked_is_no_fault_stop(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    h.new_task()
    sdk.gripper.blocked_at = 1.0
    result = final(h, h.client.gripper(1.0))
    assert (result.status, result.detail["outcome"]) == ("failed", "gripper_fault")
    assert result.reason.startswith("gripper blocked while opening at 1.00 rad")
    assert result.detail["opening"] == pytest.approx(1.0 / RS_TRAVEL)
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == result.command_id
    )
    assert (state.stop_level, state.owner) == ("none", "agent")
    # What blocked them is taken away: the next opening runs.
    sdk.gripper.blocked_at = None
    assert final(h, h.client.gripper(1.0)).status == "completed"


def test_an_agents_two_stage_close_reports_what_the_driver_skipped(
    rs_app: tuple[AppHarness, MotorSdk], monkeypatch: pytest.MonkeyPatch
) -> None:
    import apai.control.rebot_rs_arm as rebot_rs_arm

    h, sdk = rs_app
    monkeypatch.setattr(rebot_rs_arm, "time", sdk.guard_time)
    h.new_task()
    assert final(h, h.client.gripper(1.0)).status == "completed"
    sdk.gripper.stop = 2.0  # an object
    sends = len(sdk.gripper.sent)
    closed = final(
        h,
        h.client.gripper(
            0.15, max_effort=0.7, mode="close_until_contact", scout_opening=0.5
        ),
    )
    assert closed.status == "completed"
    assert closed.detail == {
        "outcome": "stalled",
        "opening": pytest.approx(2.0 / RS_TRAVEL),
        "close_stages_skipped": 0,
    }
    # The scout stage, then the seat stage; the grip keeps the seat's stiffness.
    assert [q[0] for q, _ in sdk.gripper.sent[sends:]] == pytest.approx([3.1, 0.93])
    assert not sdk.gripper.light_after(sends)
    state = h.client.wait_state(lambda s: s.gripper.may_be_holding)
    assert state.gripper.load == "loaded"
    # The agent gives the arm back and a STOP follows: the grip is not relaxed.
    assert final(h, h.client.request_owner("nobody", reason="done")).status == (
        "completed"
    )
    h.client.stop("operator pressed stop")
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert h.app._worker.wait_idle(WAIT_S)
    assert not sdk.gripper.light_after(sends)


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
    assert outcome.status == "sent"
    arm.gripper_light_hold()
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


# The end read of a move on the reBot RS driver: the read of every joint that
# decides whether the move has settled

RS_READ_S = 0.008  # what one motor read costs on the arm's motor bus
RS_LIBRARY_WAIT_S = 1.0  # the motor library's wait for a read given no timeout
RS_NUDGE = straight(REST, moved(REST, 0, 0.01), 10)  # 0.05 rad/s for 0.2 s


def run_rs_nudge(
    arm: apai_backend.ApaiArm,
    rig: RobotAppConfig,
    rs_profile: apai_backend.ApaiProfile,
    clock: Clock,
    in_watch: list[bool],
) -> MoveOutcome:
    """Run RS_NUDGE from rest on a move executor, as a return to rest runs.

    in_watch[0] is True while the move's contact watch makes the reads of
    one pass. On a clock that is not a fake one the watch paces itself.
    """
    recovery = apai_backend.build_apai_checkers(rig, rs_profile, clock).recovery
    made = recovery.contact_watch

    def contact_watch(port: Any, q_start: Any, history: Any) -> Any:
        watch = made(port, q_start, history)
        real = watch.after_send

        def after_send(index: int, q: Any, deadline_mono: float) -> list:
            in_watch[0] = True
            try:
                return real(index, q, deadline_mono)
            finally:
                in_watch[0] = False

        watch.after_send = after_send
        return watch

    recovery.contact_watch = contact_watch
    authority = Authority()
    authority.set_owner(Owner.RECOVERY, "return to rest")
    history = JointHistory(rig.n_joints, rig.state.history_s)
    executor = MoveExecutor(arm, authority, history, rig.executor, clock)
    return executor.run(
        RS_NUDGE,
        REST,
        0.02,
        STEP_S,
        Owner.RECOVERY,
        recovery,
        authority.snapshot(),
        "nudge",
    )


def test_the_end_read_of_a_move_gives_every_motor_read_the_short_timeout(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    clock: FakeClock,
) -> None:
    arm, sdk = rs
    motors = list(sdk.arm._mm.values())
    marks: list[int] = []
    send = arm.send_target

    def send_target(q: Sequence[float]) -> None:
        send(q)
        if len(sdk.arm.sent) == len(RS_NUDGE):
            marks.extend(len(motor.reads) for motor in motors)

    arm.send_target = send_target
    rig = apai_backend.config_for_profile(config, rs_profile)
    outcome = run_rs_nudge(arm, rig, rs_profile, clock, [False])
    assert (outcome.status, outcome.reason) == ("completed", "")
    assert outcome.end_q == pytest.approx(RS_NUDGE[-1])
    # Once the last setpoint is out no motor is asked without a timeout: such
    # a read waits the motor library's own time for a motor that is silent.
    after = [motor.reads[mark:] for motor, mark in zip(motors, marks, strict=True)]
    assert all(reads for reads in after)
    assert {read for reads in after for read in reads} == {(0x7019, 50)}


def test_a_motor_that_misses_one_answer_in_the_end_read_does_not_end_the_move(
    rs_profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In real time, each motor read costing what it costs on the arm's bus.

    With motors that answer at once the watch reads every joint in its next
    pass, and a read that waited too long would not show.
    """
    clock = SystemClock()
    rig = apai_backend.config_for_profile(config, rs_profile)
    arm, sdk = rs_arm(rig, rs_profile, clock, monkeypatch)
    arm.connect()
    in_watch = [False]
    missed: list[int | None] = []  # the timeout of the read that got no answer
    silent = sdk.arm._mm["motor1"]

    def on_the_bus(motor: Motor) -> None:
        answer = motor.robstride_get_param_f32

        def read(param: int, timeout_ms: int | None = None) -> float:
            own = len(sdk.arm.sent) == len(RS_NUDGE) and not in_watch[0]
            if motor is silent and own and not missed:
                # The executor's own first read of it once the move is sent.
                missed.append(timeout_ms)
                wait_ms = (
                    1000.0 * RS_LIBRARY_WAIT_S if timeout_ms is None else timeout_ms
                )
                time.sleep(wait_ms / 1000.0)
                raise OSError("no answer")
            time.sleep(RS_READ_S)
            return answer(param, timeout_ms)

        motor.robstride_get_param_f32 = read

    for motor in sdk.arm._mm.values():
        on_the_bus(motor)
    outcome = run_rs_nudge(arm, rig, rs_profile, clock, in_watch)
    assert (outcome.status, outcome.reason) == ("completed", "")
    # The silent motor cost the contact monitor's 50 ms, under the 0.35 s it
    # allows a joint without a reading.
    assert missed == [50]
    assert outcome.end_q == pytest.approx(RS_NUDGE[-1])
    assert len(sdk.arm.sent) == len(RS_NUDGE)
    assert arm.torque_on() and sdk.disconnects == 0


# A torque release on the reBot RS driver: while the driver's disconnect runs,
# and after one that raised

RS_RELEASE_FAULT = f"{RELEASE_FAILED}: the driver's disconnect failed: bus off"
UNDER_WAY = "a torque release is under way"


def hold_disconnect(
    sdk: MotorSdk, error: Exception | None = None
) -> tuple[threading.Event, threading.Event]:
    """Hold the motor SDK's disconnect inside its call until `go` is set.

    Returns (inside, go); inside is set once the call waits. With an error
    the call then raises it.
    """
    inside, go = threading.Event(), threading.Event()

    def disconnect() -> None:
        sdk.disconnects += 1
        inside.set()
        go.wait(WAIT_S)
        if error is not None:
            raise error

    sdk.disconnect = disconnect
    return inside, go


def stop_over_the_link(h: AppHarness) -> None:
    """A STOP link message, handled by the robot app before this returns."""
    heard = len(h.heard("stop"))
    h.client.stop("operator pressed stop")
    h.wait_local_events("stop", heard + 1)


RELEASE_STOPS: dict[str, Callable[[AppHarness], Any]] = {
    "a STOP": stop_over_the_link,
    "the robot connection lost": lambda h: h.lose_link(),
}


@pytest.mark.parametrize("stopped_by", RELEASE_STOPS)
def test_while_the_drivers_disconnect_runs_a_stop_sends_the_driver_nothing(
    rs_app: tuple[AppHarness, MotorSdk], stopped_by: str
) -> None:
    h, sdk = rs_app
    inside, go = hold_disconnect(sdk)
    try:
        h.client.release()
        assert inside.wait(WAIT_S), "the release never reached the disconnect"
        jaw_mark, arm_mark = len(sdk.gripper.sent), len(sdk.arm.sent)
        RELEASE_STOPS[stopped_by](h)
        # The stop's light jaw hold did not go to a driver that is switching
        # off, and the motors are not reported off before the call returned.
        assert sdk.gripper.sent[jaw_mark:] == [] and sdk.arm.sent[arm_mark:] == []
        assert h.arm.torque_on()
    finally:
        go.set()
    h.idle()
    assert not h.arm.torque_on() and sdk.disconnects == 1
    assert h.app.motion_state is MotionState.REST_UNPOWERED
    assert h.heard("fault") == []
    assert sdk.gripper.sent[jaw_mark:] == []


def test_a_stop_during_a_disconnect_that_then_raises_sends_the_driver_nothing(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    inside, go = hold_disconnect(sdk, OSError("bus off"))
    try:
        release_id = h.client.release()
        assert inside.wait(WAIT_S), "the release never reached the disconnect"
        jaw_mark = len(sdk.gripper.sent)
        stop_over_the_link(h)
        assert sdk.gripper.sent[jaw_mark:] == []
    finally:
        go.set()
    result = final(h, release_id)
    assert (result.status, result.reason) == ("failed", f"fault: {RS_RELEASE_FAULT}")
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert (state.motion, state.torque_on) == ("release_failed", True)
    assert state.fault_reason == RS_RELEASE_FAULT
    assert sdk.gripper.sent[jaw_mark:] == []


def test_a_stop_between_the_drivers_disconnect_and_the_arm_ports_note_is_no_fault(
    rs_app: tuple[AppHarness, MotorSdk],
) -> None:
    h, sdk = rs_app
    backend = h.arm.backend
    real = backend.disconnect
    stopped: list[int] = []

    def disconnect() -> None:
        real()
        # The driver has dropped its motor SDK; the arm port's call goes on.
        assert backend._arm is None
        stop = threading.Thread(
            target=lambda: stopped.append(h.app.stop("operator pressed stop"))
        )
        stop.start()
        stop.join(WAIT_S)

    backend.disconnect = disconnect
    result = final(h, h.client.release())
    assert (result.status, result.reason) == ("completed", "")
    assert len(stopped) == 1, "the STOP did not return"
    state = h.client.wait_state(lambda s: not s.torque_on)
    # No light jaw hold was tried on a driver without its motor SDK.
    assert (state.motion, state.stop_level) == ("rest_unpowered", "stop")
    assert h.heard("fault") == []
    assert sdk.disconnects == 1


def one_idle_read_fails_before_the_pause(
    h: AppHarness, sdk: MotorSdk, clock: FakeClock
) -> threading.Event:
    """The idle joint read under way when a release pauses the reads gets no answer.

    Returns the event that is set once that read began.
    """
    armed, in_flight = threading.Event(), threading.Event()

    def unanswered_once(read: Callable[..., float]) -> Callable[..., float]:
        def get(param: int, timeout_ms: int | None = None) -> float:
            # Only a read of one joint passes a timeout.
            if timeout_ms is not None and armed.is_set() and not in_flight.is_set():
                in_flight.set()
                time.sleep(0.03)
                raise OSError("no answer")
            return read(param, timeout_ms)

        return get

    for motor in sdk.arm._mm.values():
        motor.robstride_get_param_f32 = unanswered_once(motor.robstride_get_param_f32)
    paused = h.app._rest._reads_paused

    def paused_after_that_read() -> Any:
        armed.set()
        clock.advance(0.05)  # an idle read is due
        assert in_flight.wait(WAIT_S), "no idle read came"
        return paused()

    h.app._rest._reads_paused = paused_after_that_read
    return in_flight


RELEASE_ENDS: dict[str, tuple[Exception | None, bool]] = {
    "the disconnect returns": (None, False),
    "a STOP during the disconnect": (None, True),
    "the disconnect raises": (OSError("bus off"), False),
}


@pytest.mark.parametrize("how", RELEASE_ENDS)
def test_a_joint_read_that_failed_just_before_a_release_is_no_lost_feedback(
    rs_app: tuple[AppHarness, MotorSdk], clock: FakeClock, how: str
) -> None:
    h, sdk = rs_app
    error, stop = RELEASE_ENDS[how]
    idle = h.app._idle
    missed = one_idle_read_fails_before_the_pause(h, sdk, clock)
    inside, go = hold_disconnect(sdk, error)
    try:
        release_id = h.client.release()
        assert inside.wait(WAIT_S), "the release never reached the disconnect"
        assert missed.is_set() and sum(idle.failures()) >= 1
        if stop:
            stop_over_the_link(h)
        # The driver's disconnect takes half a second, as on the arm.
        clock.advance(0.5)
        h.step()
        h.step()
        assert idle.consecutive_failures_s(clock.monotonic()) == 0.0
        assert h.heard("fault") == []
    finally:
        go.set()
    result = final(h, release_id)
    assert idle._held == 0  # the pause is lifted, however the release ended
    if error is None:
        assert (result.status, result.reason) == ("completed", "")
        state = h.client.wait_state(lambda s: not s.torque_on)
        level = "stop" if stop else "none"
        assert (state.motion, state.stop_level) == ("rest_unpowered", level)
        h.advance(1.0, 0.1)
        assert h.heard("fault") == []
    else:
        # A release that failed is a fault-stop of its own; the motors count
        # as on, so the idle reads and their count of lost feedback go on.
        assert result.status == "failed"
        state = h.client.wait_state(lambda s: s.stop_level == "fault")
        assert (state.motion, state.torque_on) == ("release_failed", True)
        assert idle.due_in(clock.monotonic()) is not None


def test_no_jaw_reading_is_taken_while_a_joints_idle_read_fails(
    rs_app: tuple[AppHarness, MotorSdk], clock: FakeClock
) -> None:
    h, sdk = rs_app
    jaw, idle = sdk.gripper._mm["jaw"], h.app._idle
    mark = len(jaw.reads)
    h.advance(0.3, 0.02)
    # In a hold the jaw reading of each ROBOT_STATE is the bounded one.
    periodic = jaw.reads[mark:]
    assert len(periodic) >= 3 and set(periodic) == {(0x7019, 50)}
    sdk.arm._mm["motor3"].missed = 10**9  # one joint stops answering
    h.advance_until(
        lambda: idle.consecutive_failures_s(clock.monotonic()) > 0.0, 0.02, 1.0
    )
    mark, began = len(jaw.reads), clock.monotonic()
    h.advance(0.1, 0.02)
    assert jaw.reads[mark:] == []  # the bus is the joints'
    assert h.heard("fault") == []
    # The reading the agent had keeps its own time: none is made up.
    shown = h.client.wait_state(lambda s: s.clock_mono_s >= began + 0.05)
    assert shown.gripper.valid and shown.gripper.t <= began
    sdk.arm._mm["motor3"].missed = 0
    h.advance(0.3, 0.02)
    assert (0x7019, 50) in jaw.reads[mark:]
    assert h.heard("fault") == []


def test_rs_arm_port_sends_the_driver_nothing_while_its_disconnect_runs(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    opened = arm.gripper_command(1.0, 1.0, "plain", threading.Event())
    assert opened.status == "reached"
    inside, go = hold_disconnect(sdk)
    release = threading.Thread(target=arm.release_torque)
    release.start()
    try:
        assert inside.wait(WAIT_S), "the release never reached the disconnect"
        jaw_mark, arm_mark = len(sdk.gripper.sent), len(sdk.arm.sent)
        arm.gripper_light_hold()
        with pytest.raises(ArmFault, match=UNDER_WAY):
            arm.send_target(REST)
        with pytest.raises(ArmFault, match=UNDER_WAY):
            arm.gripper_reference()
        refused = arm.gripper_command(0.0, 0.5, "plain", threading.Event())
        assert (refused.status, refused.detail) == ("refused", UNDER_WAY)
        assert sdk.gripper.sent[jaw_mark:] == [] and sdk.arm.sent[arm_mark:] == []
        # Not off before the driver's call has returned.
        assert arm.torque_on()
    finally:
        go.set()
    release.join(WAIT_S)
    assert not release.is_alive() and not arm.torque_on()
    arm.gripper_light_hold()
    assert sdk.gripper.sent[jaw_mark:] == []


class WhenItFalls:
    """In place of ApaiArm._releasing, the arm port's note that its driver's
    disconnect is under way: calls `then` each time the note is taken back."""

    def __init__(self, then: Callable[[], None]) -> None:
        self._then = then

    def __get__(self, arm: Any, owner: Any = None) -> bool:
        return False if arm is None else arm.__dict__.get("under way", False)

    def __set__(self, arm: Any, value: bool) -> None:
        arm.__dict__["under way"] = value
        if not value:
            self._then()


def test_rs_arm_port_counts_a_release_before_it_stops_refusing_commands_for_it(
    rs: tuple[apai_backend.ApaiArm, MotorSdk], monkeypatch: pytest.MonkeyPatch
) -> None:
    arm, sdk = rs
    arm.gripper_reference()
    opened = arm.gripper_command(1.0, 1.0, "plain", threading.Event())
    assert opened.status == "reached"
    seen: list[tuple[bool, list]] = []

    def a_stop_right_then() -> None:
        # What a STOP that lands at this moment does: a light jaw hold.
        mark = len(sdk.gripper.sent)
        arm.gripper_light_hold()
        seen.append((arm.torque_on(), sdk.gripper.sent[mark:]))

    port = apai_backend.ApaiArm
    monkeypatch.setattr(port, "_releasing", WhenItFalls(a_stop_right_then), False)
    arm.release_torque()
    # No gap in which the driver, without its motor SDK by then, is sent one.
    assert seen == [(False, [])]
    assert sdk.disconnects == 1


def fail_first_disconnect(sdk: MotorSdk) -> None:
    """The motor SDK's first disconnect raises; a later one would work."""
    real = sdk.disconnect

    def disconnect() -> None:
        if sdk.disconnects:
            real()
            return
        sdk.disconnects += 1
        raise OSError("bus off")

    sdk.disconnect = disconnect


def test_rs_arm_port_takes_no_second_release_after_a_disconnect_that_raised(
    rs: tuple[apai_backend.ApaiArm, MotorSdk],
) -> None:
    arm, sdk = rs
    fail_first_disconnect(sdk)
    with pytest.raises(ArmFault, match=r"the driver's disconnect failed: bus off"):
        arm.release_torque()
    assert arm.torque_on() and not arm.release_confirmed()
    # APAI's driver dropped its motor SDK when its disconnect raised: its next
    # disconnect returns without reaching a motor, which proves nothing.
    assert arm.backend._arm is None
    with pytest.raises(
        ArmFault,
        match=r"an earlier disconnect failed: the motors may be on or off, and "
        r"this driver takes no second release",
    ):
        arm.release_torque()
    assert sdk.disconnects == 1
    assert arm.torque_on() and not arm.release_confirmed()
    with pytest.raises(ArmFault, match=r"a torque release failed: the motors may"):
        arm.send_target(REST)


def test_after_a_failed_release_on_the_rs_driver_only_a_new_start_is_left(
    rs_app: tuple[AppHarness, MotorSdk], clock: FakeClock
) -> None:
    h, sdk = rs_app
    fail_first_disconnect(sdk)
    port_calls: list[int] = []
    release_torque = h.arm.release_torque

    def counted() -> None:
        port_calls.append(1)
        release_torque()

    h.arm.release_torque = counted
    result = final(h, h.client.release())
    assert (result.status, result.reason) == ("failed", f"fault: {RS_RELEASE_FAULT}")
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert (state.motion, state.torque_on) == ("release_failed", True)
    # The fault-stop's line names the new start, not a reset.
    assert h.heard("fault")[-1][1] == (
        f"FAULT-STOP: {RS_RELEASE_FAULT}; end the robot app with SIGTERM and "
        f"start it again with the arm at rest, stop number {state.stop_number}"
    )
    # APAI's driver dropped its motor SDK: no joint and no jaw reads from here.
    with pytest.raises(ReadFailed):
        h.arm.read_all()
    assert h.arm.gripper_read() is None

    # A reset clears the fault-stop; a release sent at once is refused.
    assert final(h, h.client.reset()).status == "completed"
    h.client.wait_state(lambda s: s.stop_level == "none")
    refused = final(h, h.client.release())
    # The release that failed had judged the jaws: they are not judged again,
    # and the joints, which must be read at rest, cannot be read.
    assert refused.status == "refused"
    assert refused.reason.startswith(
        "a joint read failed: the joints could not be read: "
    )
    assert h.heard("refused")[-1][1] == (
        f"release refused: {refused.reason}; {TORQUE_UNKNOWN}; {NEW_START}"
    )
    # Without joint reads the fault-stop latches again.
    for _ in range(25):
        clock.advance(0.02)
        h.step()
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost for ")
    assert h.heard("fault")[-1][1] == (
        f"FAULT-STOP: {state.fault_reason}; {MOTORS_UNKNOWN}, stop number "
        f"{state.stop_number}"
    )
    refused = final(h, h.client.release())
    latched = f"fault-stop latched: {state.fault_reason}"
    assert (refused.status, refused.reason) == ("refused", latched)
    # A quit tries the release again: refused too, and the app runs on.
    reason = f"a torque release failed and {TORQUE_UNKNOWN}; release refused: {latched}"
    typed = h.thor_terminal().ask("quit", answer="quit:")
    assert typed == f"quit: failed: {reason}; the robot app keeps running"
    assert h.heard("shutdown")[-1][1] == (
        f"shutdown failed: {reason}; the robot app keeps running; {NEW_START}"
    )
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == (
        "refused",
        "a torque release failed and the motors may be on or off; end the robot "
        "app with SIGTERM and start it again with the arm at rest",
    )
    # Nothing reached the driver a second time, and nothing said "motors off".
    assert port_calls == [1] and sdk.disconnects == 1
    assert not [text for _, text, _ in h.local_events if "motors are off" in text]
    state = h.client.state
    assert (state.motion, state.torque_on) == ("release_failed", True)
    assert h.app.running and h.arm.torque_on()


# A wanted motors-off at rest on the reBot RS driver


def test_no_joint_read_meets_the_driver_while_its_disconnect_runs(
    rs_app: tuple[AppHarness, MotorSdk], clock: FakeClock
) -> None:
    """Seen on the arm: "joint feedback lost for 0.58 s in hold" after a release."""
    h, sdk = rs_app
    inside, go = hold_disconnect(sdk)
    motors = list(sdk.arm._mm.values())
    try:
        release_id = h.client.release()
        assert inside.wait(WAIT_S), "the release never reached the disconnect"
        # A motor that is being switched off answers no read.
        for motor in motors:
            motor.missed = 10**6
        # Nor is a joint that leaves its hold target a fault meanwhile.
        watch = h.arm._hold
        watch._off[0], watch._last[0] = watch._samples, (0.3, 0.3)
        assert watch.problem() is not None and h.arm.faults() == []
        reads = [len(motor.reads) for motor in motors]
        # 0.8 s of the app's clock pass, an idle joint read due on every pass.
        for _ in range(40):
            clock.advance(0.02)
            h.step()
        assert [len(motor.reads) for motor in motors] == reads
        assert h.heard("fault") == []
    finally:
        go.set()
    assert final(h, release_id).status == "completed"
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert (state.stop_level, state.fault_reason) == ("none", "")
    assert not state.torque_on
    # No fault-stop follows either: nothing is read with the motors off.
    for _ in range(40):
        clock.advance(0.02)
        h.step()
    assert [len(motor.reads) for motor in motors] == reads
    assert h.heard("fault") == []
    assert h.app.snapshot().stop_level is StopLevel.NONE


def test_a_release_leaves_the_jaws_in_the_light_hold_of_the_drivers_close(
    rs_app: tuple[AppHarness, MotorSdk], monkeypatch: pytest.MonkeyPatch
) -> None:
    h, sdk = rs_app
    jaw = sdk.gripper
    calls: list[dict[str, float]] = []
    close = h.arm.backend.close_gripper_torque
    monkeypatch.setattr(
        h.arm.backend,
        "close_gripper_torque",
        lambda **kwargs: calls.append(kwargs) or close(**kwargs),
    )
    # The jaws are open, proven empty, and meet something 0.2 rad short of
    # their closed stop.
    opened = h.thor_terminal().ask("open", "yes", answer="open: ")
    assert opened == "open: completed"
    jaw.stop = 0.2
    mark = len(jaw.sent)
    released = final(h, h.client.release())
    assert (released.status, released.reason) == ("completed", "")
    # The close of APAI's rest, at the driver's own effort: to the closed
    # stop, then the light hold where the jaws met something. Not the typed
    # close, which leaves them pushing at 0.15 of the travel.
    assert calls == [{}]
    assert jaw.sent[mark:] == [([0.0], [pytest.approx(1.2)]), ([0.0], [RS_HOLD_KP])]
    assert jaw.at() == pytest.approx(0.2)
    assert sdk.disconnects == 1 and not h.arm.torque_on()
    assert [text for _, text, _ in h.heard("jaws")] == [
        "at rest: closing the jaws with the driver's close",
        "at rest: the jaws are closed",
    ]


# The joint speed caps of the arm profile the experiment's launcher runs

# The launcher's joint speed in the workstation's copy of the experiment:
# overrides.safety.max_joint_vel of its configs/rebot_workstation.yaml.
LAUNCHER_SPEED = 0.4


def thor_arm_profile(root: Path, speed: float = LAUNCHER_SPEED, **arm: Any) -> str:
    """An arm profile laid out as the experiment's on the Thor; its path.

    The experiment is the on-device agent this example was split from. Its
    launcher's settings name the arm profile it runs, and a second arm
    profile holds the joint speed it runs it at. Here the first extends
    APAI's rebot_rs profile; `arm` are further keys of it. The robot app
    loads such a profile as on the Thor: with the launcher's speed for its
    speed cap.
    """
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    files = {
        apai_backend.LAUNCHER_SETTINGS: {"arm_profile": "configs/thor_operation.yaml"},
        f"{apai_backend.SPEED_SOURCE}.yaml": {
            "extends": apai_backend.SPEED_CEILING,
            "overrides": {"safety": {"max_joint_vel": speed}},
        },
        "thor_operation.yaml": {"extends": apai_backend.SPEED_CEILING, **arm},
    }
    for name, content in files.items():
        (configs / name).write_text(yaml.safe_dump(content), encoding="utf-8")
    return str(configs / "thor_operation")


def test_on_the_launchers_profile_only_a_person_may_be_faster_than_the_launcher(
    tmp_path: Path,
    config: RobotAppConfig,
    rs_profile: apai_backend.ApaiProfile,
    clock: FakeClock,
) -> None:
    thor = apai_backend.load_apai_profile(thor_arm_profile(tmp_path))
    assert thor.launcher_dir == tmp_path.resolve()
    # The profile's cap is the launcher's speed; a person's bound is the cap
    # of APAI's rebot_rs profile, which the launcher holds its own speed to.
    assert apai_backend.speed_cap(thor) == LAUNCHER_SPEED
    ceiling = apai_backend.speed_cap(rs_profile)
    assert apai_backend.human_speed_cap(thor) == ceiling == 0.8
    fits = with_speed(with_speed(config, "agent", LAUNCHER_SPEED), "human", ceiling)
    checkers = apai_backend.build_apai_checkers(fits, thor, clock)
    speeds = [checker.harness.limits.max_joint_vel for checker in checkers.all()]
    assert speeds == [LAUNCHER_SPEED, ceiling, 0.1]
    for owner in ("agent", "recovery"):
        with pytest.raises(
            ValueError,
            match=rf"limits\.{owner}\.max_joint_speed 0\.41 rad/s is above the APAI "
            r"profile's speed cap of 0\.4 rad/s \(safety\.max_joint_vel\)",
        ):
            apai_backend.build_apai_checkers(with_speed(fits, owner, 0.41), thor, clock)
    with pytest.raises(
        ValueError,
        match=r"limits\.human\.max_joint_speed 0\.81 rad/s is above the speed cap "
        r"of APAI's rebot_rs profile of 0\.8 rad/s \(safety\.max_joint_vel\)",
    ):
        apai_backend.build_apai_checkers(with_speed(fits, "human", 0.81), thor, clock)
    # A profile no launcher runs binds a person by its own cap.
    capped = {"safety": {"max_joint_vel": LAUNCHER_SPEED}}
    plain = apai_backend.load_apai_profile("rebot_rs", capped)
    assert plain.launcher_dir is None
    assert apai_backend.human_speed_cap(plain) == LAUNCHER_SPEED
    with pytest.raises(
        ValueError,
        match=r"limits\.human\.max_joint_speed 0\.8 rad/s is above the APAI "
        r"profile's speed cap of 0\.4 rad/s",
    ):
        apai_backend.build_apai_checkers(fits, plain, clock)


# The motion bounds of the experiment's launcher

# Two modules beside the launcher's bounds module, which imports them by name
# as the Thor's copy imports thor_motion: the first when it is loaded, the
# second only when a move is checked.
AT_LOAD, AT_CHECK = "rig_at_load", "rig_at_check"
BOUNDS_SOURCE = f"""\
from {AT_LOAD} import joint_limit


def restrict_motion(harness, config, *, label="calibration"):
    from apai.types import SafetyViolation

    limit = joint_limit(config)
    approve = harness.approve

    def bounded(q_prev, q_next, *args, **kwargs):
        from {AT_CHECK} import reason

        why = reason(label, q_next, limit)
        if why:
            raise SafetyViolation(why)
        return approve(q_prev, q_next, *args, **kwargs)

    harness.approve = bounded
    print(f"[{{label}}] Motion bounds: the first joint within {{limit:g}} rad")
"""
SIBLING_SOURCES = {
    AT_LOAD: 'def joint_limit(config):\n    return float(config["first_joint_rad"])\n',
    AT_CHECK: (
        "def reason(label, q, limit):\n"
        "    if abs(q[0]) > limit:\n"
        '        return f"{label} motion bounds: the first joint is beyond {limit:g} rad"\n'
        "    return None\n"
    ),
}


@pytest.fixture
def import_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """After the test the import path and the imported modules are as before."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in (AT_LOAD, AT_CHECK):
        sys.modules.pop(name, None)


def launcher_with_bounds(
    root: Path, siblings: Sequence[str] = (AT_LOAD, AT_CHECK)
) -> apai_backend.ApaiProfile:
    """The launcher's arm profile with motion bounds, in a folder that holds
    its bounds module and these of the two modules that one imports."""
    arm = thor_arm_profile(root, motion_bounds={"first_joint_rad": 0.5})
    files = {apai_backend.BOUNDS_MODULE: BOUNDS_SOURCE}
    files.update({name: SIBLING_SOURCES[name] for name in siblings})
    for name, source in files.items():
        (root / f"{name}.py").write_text(source, encoding="utf-8")
    return apai_backend.load_apai_profile(arm)


def test_the_launchers_bounds_module_imports_the_modules_beside_it(
    tmp_path: Path, config: RobotAppConfig, clock: FakeClock, import_state: None
) -> None:
    """As in the launcher, which is started as a script from its folder."""
    thor = launcher_with_bounds(tmp_path)
    checkers = apai_backend.build_apai_checkers(config, thor, clock)
    assert AT_LOAD in sys.modules and AT_CHECK not in sys.modules
    agent = checkers.agent
    start_motion(agent)
    agent.check_setpoint(HOME, NEAR_HOME, STEP_S)
    assert AT_CHECK in sys.modules  # imported by the first setpoint check
    inside, beyond = moved(HOME, 0, 0.499), moved(HOME, 0, 0.501)
    refusal_of(
        lambda: agent.check_setpoint(inside, beyond, STEP_S),
        r"operation motion bounds: the first joint is beyond 0\.5 rad",
    )
    agent.end_motion()
    # The human's checker keeps its own limits.
    start_motion(checkers.human)
    checkers.human.check_setpoint(inside, beyond, STEP_S)


def test_the_launcher_folder_goes_last_on_the_import_path_and_only_once(
    tmp_path: Path, config: RobotAppConfig, clock: FakeClock, import_state: None
) -> None:
    # A module of the same name that is on the path already.
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / f"{AT_LOAD}.py").write_text(
        "def joint_limit(config):\n    return 0.25\n", encoding="utf-8"
    )
    sys.path.insert(0, str(earlier))
    before = list(sys.path)
    thor = launcher_with_bounds(tmp_path / "experiment")
    for _ in range(2):
        apai_backend.build_apai_checkers(config, thor, clock)
        assert sys.path == [*before, str(thor.launcher_dir)]
    # The launcher folder's file of that name shadows nothing.
    assert Path(sys.modules[AT_LOAD].__file__).parent == earlier


def test_a_module_missing_beside_the_launchers_bounds_module_is_refused(
    tmp_path: Path, config: RobotAppConfig, clock: FakeClock, import_state: None
) -> None:
    thor = launcher_with_bounds(tmp_path, siblings=(AT_CHECK,))
    with pytest.raises(
        ValueError,
        match=r"the experiment's motion bounds \(.*thor_calibration_rig\.py\) could "
        rf"not be applied: ModuleNotFoundError: No module named '{AT_LOAD}'",
    ):
        apai_backend.build_apai_checkers(config, thor, clock)


# A new task from where the arm was left

START_POSES = {
    "at rest": REST,
    "at home": HOME,
    "the tip above the table clearance": ABOVE,
    "the tip under the table clearance": BELOW,
    "the tip beside the workspace box": moved(HOME, 0, 0.9),
    "a link on the table": (0.0, 2.0, 0.2, 0.0, 0.0, 0.0),
}


@pytest.mark.parametrize("where", START_POSES)
def test_a_task_may_start_where_apais_step_check_takes_a_hold(
    profile: apai_backend.ApaiProfile,
    config: RobotAppConfig,
    checkers: CheckerSet,
    where: str,
) -> None:
    """APAI checks no pose when a task starts.

    Its stream checks a move's first setpoint from where the arm is, and that
    check lets a low gripper tip rise and one outside the workspace box come
    back. A hold where the arm is is the least such step.
    """
    from apai.safety.harness import SafetyHarness, SafetyLimits

    pose = np.array(START_POSES[where], dtype=float)
    # The checker as the experiment builds it for its agent, without a map.
    limits = dataclasses.replace(
        SafetyLimits.from_config(profile.apai_cfg.safety),
        max_joint_vel=config.limits.agent.max_joint_speed,
    )
    functional = SafetyHarness(limits, kinematics=profile.kinematics)
    # On a joint stop, as at rest, APAI holds at the nearest pose its joint
    # margin takes.
    inside = profile.kinematics.clamp(pose, limits.joint_margin)
    theirs = functional.vet_step(inside, inside, STEP_S)
    try:
        checkers.agent.check_start_pose(pose)
        ours = None
    except Refusal as refusal:
        ours = refusal.reason
    assert ours == theirs
    assert (theirs is None) == (where != "a link on the table")
    # The pose check takes a pose for a target, without the escape rules.
    low = where in (
        "the tip under the table clearance",
        "the tip beside the workspace box",
    )
    if low:
        with pytest.raises(Refusal, match=r"below table clearance|outside workspace"):
            checkers.agent.check_pose(pose)


def test_from_a_low_tip_the_agent_checker_takes_a_lift_and_no_lowering(
    checkers: CheckerSet,
) -> None:
    agent = checkers.agent
    agent.check_start_pose(BELOW)
    start_motion(agent)
    # Every setpoint is checked as APAI's stream checks it: the tip may rise.
    lift = min_jerk_path(BELOW, ABOVE, 40)
    prev: Sequence[float] = BELOW
    for row in lift:
        agent.check_setpoint(prev, row, STEP_S)
        prev = row
    lower = moved(BELOW, 1, BELOW[1] + 0.002)
    refusal_of(
        lambda: agent.check_setpoint(BELOW, lower, STEP_S), r"below table clearance"
    )
    agent.end_motion()


def test_a_latched_checker_and_a_pose_outside_the_envelope_start_no_task(
    checkers: CheckerSet,
) -> None:
    agent = checkers.agent
    outside = moved(HOME, 5, 1.6)
    refusal_of(lambda: agent.check_start_pose(outside), r"outside the agent envelope")
    refusal_of(lambda: agent.check_start_pose(HOME[:3]), r"needs 6 values")
    agent.latch("motor fault")
    refusal_of(lambda: agent.check_start_pose(HOME), r"e-stop latched")
