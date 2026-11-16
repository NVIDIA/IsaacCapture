# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rig files in examples/agentic_remote/configs: robot_app_rebot.yaml,
agent_rebot.yaml and link_service_rebot.yaml.

A rig file holds the rig's own values, as opposed to the shipped demo values.
The tests marked "apai" hold robot_app_rebot.yaml against APAI's rebot_rs arm
profile and run the robot app with it on a stand-in motor SDK.
"""

from __future__ import annotations

import dataclasses
import re
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.human.ik import Solver
from isaaccapture_examples.agentic_remote.human.types import IkSettings
from isaaccapture_examples.agentic_remote.link_service import __main__ as service_main
from isaaccapture_examples.agentic_remote.link_service import config as service_config
from isaaccapture_examples.agentic_remote.protocol.messages import HumanSetpoint
from isaaccapture_examples.agentic_remote.robot_app.app import (
    LOW_TIP,
    OUTSIDE_HUMAN_LIMITS,
    TAKEN_OVER,
    TAKEOVER_OPEN,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    ConfigError,
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.human import LIFT, SPEED_MARGIN
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
    Refusal,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from repo_paths import repo_root

import test_apai_backend as rs
from conftest import AppHarness, basic_checkers
from test_takeover import (
    ask_takeover,
    drives,
    feed,
    final,
    keep_human_fed,
    setpoint,
    take_over,
)

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
ROBOT_RIG = CONFIGS / "robot_app_rebot.yaml"
AGENT_RIG = CONFIGS / "agent_rebot.yaml"
SERVICE_RIG = CONFIGS / "link_service_rebot.yaml"
RIG_FILES = (ROBOT_RIG, AGENT_RIG, SERVICE_RIG)
CALIBRATION_ID = "stereo-c50bf5b2b899+mount-41d3ce701628"
PROFILE_ID = "rebot-b601-rs-thor"
# The joint limits of APAI's arm model of the rebot_rs profile, rad.
MODEL_LOW = (-2.8, 0.0, 0.0, -1.69, -1.57, -3.14)
MODEL_HIGH = (2.8, 3.14, 3.14, 1.79, 1.57, 3.14)
# The rig file's joint boxes: the second and third joint begin below their stop.
BOX_LOW = (-2.8, -0.05, -0.05, -1.69, -1.57, -3.14)
# The agent's working box: 120 degrees around 0, cut to the joint box.
ENVELOPE_LOW = (-2.0944, -0.05, -0.05, -1.69, -1.57, -2.0944)
ENVELOPE_HIGH = (2.0944, 2.0944, 2.0944, 1.79, 1.57, 2.0944)
# How far inside the recovery region (rest.region_*, the start poses a return
# to rest is taken from) each bound of the human joint box lies, rad. A bound
# is one side of one joint: (joint index, "min" or "max").
HUMAN_BOX_MARGIN_RAD = 0.02
# The two bounds without the margin: the second and third joint lie on their
# stops at rest, and their lower bound is the region's.
ON_STOPS = ((1, "min"), (2, "min"))
HUMAN_LOW = (-2.0744, -0.05, -0.05, -1.67, -1.55, -2.0744)
HUMAN_HIGH = (2.0744, 2.0744, 2.0744, 1.77, 1.55, 2.0744)

apai = pytest.mark.apai


@pytest.fixture(scope="module")
def robot_rig() -> RobotAppConfig:
    return load_config(ROBOT_RIG)


@pytest.fixture
def path_variables(monkeypatch, tmp_path) -> dict[str, str]:
    """The three variables the rig's agent file names in its paths."""
    folders = {
        name: str(tmp_path / name.lower()) for name in agent_config.PATH_VARIABLES
    }
    for name, folder in folders.items():
        monkeypatch.setenv(name, folder)
    return folders


def shipped(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


# robot_app_rebot.yaml


def test_the_rig_robot_file_holds_the_rigs_values(robot_rig):
    assert (robot_rig.robot_id, robot_rig.profile_id) == ("rebot-thor-can4", PROFILE_ID)
    assert robot_rig.n_joints == 6
    assert robot_rig.joint_names == tuple(f"joint{i}" for i in range(1, 7))
    limits = robot_rig.limits
    for limit_set in (limits.agent, limits.human, limits.recovery):
        assert limit_set.max_joint_speed == 0.1
    for limit_set in (limits.agent, limits.recovery):
        assert limit_set.version == "rebot-p2-1"
        assert (limit_set.joint_min, limit_set.joint_max) == (BOX_LOW, MODEL_HIGH)
    # The human joint box lies inside the recovery region, which is the agent's
    # working box: by HUMAN_BOX_MARGIN_RAD on every bound but the two ON_STOPS.
    assert limits.human.version == "rebot-p3-1"
    assert (limits.human.joint_min, limits.human.joint_max) == (
        HUMAN_LOW,
        HUMAN_HIGH,
    )
    assert limits.human.max_step_rad == 0.1
    assert (limits.agent.envelope_min, limits.agent.envelope_max) == (
        ENVELOPE_LOW,
        ENVELOPE_HIGH,
    )
    rest = robot_rig.rest
    assert rest.rest_q == (0.0,) * 6
    assert (rest.rest_tol_rad, rest.start_rest_tol_rad) == (0.05, 0.05)
    assert rest.recovery_speed == 0.1
    # The recovery region is the agent's working box; no timed return.
    assert (rest.region_min, rest.region_max) == (ENVELOPE_LOW, ENVELOPE_HIGH)
    assert rest.auto_return is False
    assert robot_rig.camera_calibration_id == CALIBRATION_ID


def test_the_rig_robot_file_keeps_the_shipped_values_elsewhere(robot_rig):
    default = RobotAppConfig.default()
    for block in (
        "executor",
        "permits",
        "waits",
        "gripper",
        "grasp",
        "state",
        "human",
        "connection",
    ):
        assert getattr(robot_rig, block) == getattr(default, block), block
    rig, demo = shipped("robot_app_rebot.yaml"), shipped("robot_app.yaml")
    assert set(rig) == set(demo)
    assert {
        name: set(block) for name, block in rig.items() if isinstance(block, dict)
    } == {name: set(block) for name, block in demo.items() if isinstance(block, dict)}
    changed = {"auto_return", "region_min", "region_max"}
    for key, value in demo["rest"].items():
        assert (rig["rest"][key] == value) == (key not in changed), key


def test_the_human_settings_ship_in_the_safe_position(robot_rig):
    # Human driving off on the rig, and no person driving without a live view.
    assert (robot_rig.human.enabled, robot_rig.human.require_live_view) == (
        False,
        True,
    )
    for name in ("robot_app.yaml", "robot_app_rebot.yaml"):
        human = shipped(name)["human"]
        assert human["enabled"] is False and human["require_live_view"] is True, name
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert "\nhuman:\n  enabled: false" in text, name


def test_the_rig_file_keeps_a_person_inside_the_recovery_region(robot_rig):
    human, rest = robot_rig.limits.human, robot_rig.rest
    # A joint driven to the edge of the human joint box can read a few
    # thousandths of a rad outside it. The margin keeps such a reading inside
    # the region, where a typed rest is taken.
    for joint in range(robot_rig.n_joints):
        if (joint, "min") in ON_STOPS:
            # A joint cannot be driven below the stop it lies on at rest.
            assert human.joint_min[joint] == rest.region_min[joint] == -0.05
        else:
            below = human.joint_min[joint] - rest.region_min[joint]
            assert below == pytest.approx(HUMAN_BOX_MARGIN_RAD, abs=1e-9), joint
        above = rest.region_max[joint] - human.joint_max[joint]
        assert above == pytest.approx(HUMAN_BOX_MARGIN_RAD, abs=1e-9), joint
    # So the rig file passes the check that human.enabled brings.
    on = dataclasses.replace(robot_rig.human, enabled=True)
    assert dataclasses.replace(robot_rig, human=on).human.enabled
    # With the arm model's full joint box for the person it would not.
    wide = dataclasses.replace(human, joint_min=BOX_LOW, joint_max=MODEL_HIGH)
    limits = dataclasses.replace(robot_rig.limits, human=wide)
    full_box = dataclasses.replace(robot_rig, limits=limits)
    with pytest.raises(
        ConfigError,
        match=r"human\.enabled needs limits\.human inside rest\.region_\*: joint "
        r"index 0 has human -2\.8\.\.2\.8 and region -2\.0944\.\.2\.0944",
    ):
        dataclasses.replace(full_box, human=on)
    # The recovery joint box still contains the human one.
    recovery = robot_rig.limits.recovery
    for joint in range(robot_rig.n_joints):
        assert recovery.joint_min[joint] <= human.joint_min[joint]
        assert recovery.joint_max[joint] >= human.joint_max[joint]


def test_the_typed_jaw_reference_ships_off_in_every_robot_file(robot_rig):
    assert robot_rig.jaws.allow_reference is False
    assert load_config(CONFIGS / "robot_app.yaml").jaws.allow_reference is False
    for name in ("robot_app.yaml", "robot_app_rebot.yaml"):
        assert shipped(name)["jaws"] == {"allow_reference": False}
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert "jaws:\n  allow_reference: false\n" in text


def test_the_rig_files_name_no_machines_own_path():
    for path in RIG_FILES:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text and "~" not in text, path.name
        assert "SPDX-License-Identifier: Apache-2.0" in text
        named = set(re.findall(r"\$\{([^}]*)\}", text))
        assert named <= set(agent_config.PATH_VARIABLES), path.name
    agent = AGENT_RIG.read_text(encoding="utf-8")
    assert set(re.findall(r"\$\{([^}]*)\}", agent)) == set(agent_config.PATH_VARIABLES)


@apai
def test_the_rig_file_loads_and_fits_the_rebot_profile(robot_rig, monkeypatch):
    backend = rs.apai_backend
    monkeypatch.setattr(rs.sys, "path", list(rs.sys.path))  # the driver's SDK path
    clock = FakeClock()
    # APAI's own profile (cap 0.8 rad/s) and the rig's two caps, 0.1 and 0.2.
    for cap in (None, 0.1, 0.2):
        overrides = None if cap is None else {"safety": {"max_joint_vel": cap}}
        profile = backend.load_apai_profile("rebot_rs", overrides)
        config = backend.config_for_profile(robot_rig, profile)
        # At rest at start: the profile's rest check tolerance; a release keeps 0.05.
        assert (config.rest.start_rest_tol_rad, config.rest.rest_tol_rad) == (
            0.12,
            0.05,
        )
        assert config.gripper.target_tol == pytest.approx(0.1 / 6.2)
        checkers = backend.build_apai_checkers(config, profile, clock)
        assert [checker.name for checker in checkers.all()] == [
            "agent",
            "human",
            "recovery",
        ]
        arm = backend.build_apai_arm(config, profile, clock)
        assert type(arm.backend).__name__ == "RebotRSArm"
        assert not arm.torque_on()
    # A profile capped below the file's 0.1 rad/s refuses it.
    slow = backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.05}})
    with pytest.raises(
        ValueError, match=r"above the APAI profile's speed cap of 0\.05"
    ):
        backend.build_apai_checkers(robot_rig, slow, clock)
    # The shipped demo file does not fit: its human limit set is too fast.
    demo = load_config(CONFIGS / "robot_app.yaml")
    stock = backend.load_apai_profile("rebot_rs")
    with pytest.raises(ValueError, match=r"limits\.human\.max_joint_speed 1\.5 rad/s"):
        backend.build_apai_checkers(demo, stock, clock)
    assert list(profile.arm_cfg.park_q) == list(robot_rig.rest.rest_q)


@apai
def test_the_rig_joint_boxes_lie_in_the_arm_model_and_hold_the_rest_readings(
    robot_rig,
):
    profile = rs.apai_backend.load_apai_profile("rebot_rs")
    low, high = (
        tuple(float(v) for v in side) for side in profile.kinematics.joint_limits
    )
    assert (low, high) == (MODEL_LOW, MODEL_HIGH)
    limits, rest = robot_rig.limits, robot_rig.rest
    boxes = {
        "agent": (limits.agent.joint_min, limits.agent.joint_max),
        "human": (limits.human.joint_min, limits.human.joint_max),
        "recovery": (limits.recovery.joint_min, limits.recovery.joint_max),
        "envelope": (limits.agent.envelope_min, limits.agent.envelope_max),
        "region": (rest.region_min, rest.region_max),
    }
    for name, (box_low, box_high) in boxes.items():
        for joint in range(6):
            # Only the two joints that rest on their stops reach below the model.
            below = 0.05 if joint in (1, 2) else 0.0
            assert box_low[joint] >= low[joint] - below, (name, joint)
            assert box_high[joint] <= high[joint], (name, joint)
            assert box_low[joint] >= limits.recovery.joint_min[joint], (name, joint)
            assert box_high[joint] <= limits.recovery.joint_max[joint], (name, joint)
        for reading in rs.RIG_REST_READINGS:
            inside = [lo < q < hi for q, lo, hi in zip(reading, box_low, box_high)]
            assert all(inside), (name, reading)
    # The second and third joint of every rest reading lie below the model's 0.
    assert all(reading[1] < 0.0 for reading in rs.RIG_REST_READINGS)
    assert boxes["envelope"][0][1] == boxes["envelope"][0][2] == -0.05


@apai
@pytest.mark.parametrize("reading", rs.RIG_REST_READINGS)
def test_a_typed_rest_from_each_rig_reading_is_accepted_with_the_rig_file(
    robot_rig, clock, monkeypatch, reading
):
    backend = rs.apai_backend
    profile = backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.1}})
    config = backend.config_for_profile(robot_rig, profile)
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, arm, checkers, clock)
    try:
        h.start()
        state = h.client.state
        assert state.gripper.referenced and state.gripper.load == "empty"
        assert (state.motion, state.torque_on) == ("hold", True)
        assert sdk.arm.sent == []  # no joint was commanded by the start
        refused = h.client.wait_result(h.client.release(), timeout_s=5.0)
        assert refused.status == "refused" and sdk.disconnects == 0
        thor = h.thor_terminal()
        # Only the typed phrase confirms the return.
        assert "rest: refused" in thor.ask("rest", "yes", answer="rest: ")
        assert sdk.arm.sent == []
        assert thor.ask("rest", "path is clear", answer="rest: ") == "rest: completed"
        assert h.app.motion_state is MotionState.REST_HOLD
        assert sdk.arm.sent[-1][0] == [0.0] * 6
        # The return moved no joint farther than the reading lay from rest.
        travel = max(abs(value) for value in reading)
        assert all(
            max(abs(q) for q in sent) <= travel + 1e-9 for sent, _ in sdk.arm.sent
        )
        assert sdk.disconnects == 0 and arm.torque_on()
        released = h.client.wait_result(h.client.release(), timeout_s=5.0)
        assert released.status == "completed", released.reason
        assert sdk.disconnects == 1 and not arm.torque_on()
    finally:
        h.close()


@apai
@pytest.mark.parametrize("reading", rs.RIG_REST_READINGS)
def test_a_new_task_starts_from_each_rig_reading_with_the_rig_file(
    robot_rig, clock, monkeypatch, reading
):
    backend = rs.apai_backend
    profile = backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.1}})
    config = backend.config_for_profile(robot_rig, profile)
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, arm, checkers, clock)
    try:
        h.start()
        h.new_task()
        # Off the two joint stops, at half the agent's 0.1 rad/s.
        raised = [reading[0], 0.1, 0.1, *reading[3:]]
        moved = rs.rs_move(h, rs.straight(reading, raised, 110), reading)
        assert moved.status == "completed", moved.reason
        assert sdk.arm.sent[-1][0] == pytest.approx(raised)
        # A move at twice the agent's speed limit is refused whole.
        back = rs.rs_move(h, rs.straight(raised, list(reading), 25), raised)
        assert back.status == "refused", back.reason
        assert sdk.disconnects == 0 and arm.torque_on()
    finally:
        h.close()


# A person on the rig file. A takeover makes a person the control owner; the
# hand-back ends it at rest. A human drive is one stretch of human motion: the
# arm follows the joint targets of HUMAN_SETPOINT link messages. The human
# program makes them on the workstation by inverse kinematics (IK): from a
# pose of the gripper tip to joint values.

# The rig's limit on the hand's speed, m/s: retarget.max_linear_velocity of
# configs/human_rebot.yaml; a test below holds the two together. The hand below
# keeps its orientation, so that file's limit on turning is not met.
HAND_SPEED = 0.01
LIFT_M = 0.03  # the first human motion planned on the rig: 3 cm straight up


class LiftingHand:
    """A hand that squeezes and rises LIFT_M: keep_human_fed's setpoint.

    Its target, a pose of the gripper tip, rises by HAND_SPEED times the period
    after each setpoint the arm was sent, and the human program's IK solves it
    from the solution before. The grip is released once the arm is there.
    """

    def __init__(self, h: AppHarness, sdk: rs.MotorSdk, solver: Solver) -> None:
        self.h, self.sdk, self.solver = h, sdk, solver
        self.rise = HAND_SPEED * h.config.executor.period_s
        self.tip: np.ndarray | None = None  # None: the grip is released
        self.top = 0.0  # the height the hand rises to, m
        self.q: tuple[float, ...] = ()  # the newest IK solution
        self.problem = ""  # why a solve gave no joint values
        self.ahead = 0.0  # the most a solution led the last setpoint sent, rad

    def squeeze(self, measured: Sequence[float]) -> HumanSetpoint:
        """The first engaged target: the measured pose inside the IK's margin."""
        self.q = self.solver.inside(measured)
        self.tip = self.solver.fk(self.q)
        self.top = float(self.tip[2, 3]) + LIFT_M
        return setpoint(self.h, self.q)

    def __call__(self) -> HumanSetpoint:
        if self.tip is None:
            return setpoint(self.h)
        if self.tip[2, 3] < self.top:
            self.tip[2, 3] = min(self.top, self.tip[2, 3] + self.rise)
            step = self.solver.solve(self.tip, self.q)
            if step.q is None:
                self.problem, self.tip = step.problem, None
                return setpoint(self.h)
            self.q = step.q
        last = self.sdk.arm.sent[-1][0]
        lead = max(abs(to - at) for at, to in zip(last, self.q))
        self.ahead = max(self.ahead, lead)
        if lead < 1e-6 and self.tip[2, 3] >= self.top:
            self.tip = None  # the arm is at the top
            return setpoint(self.h)
        return setpoint(self.h, self.q)


@apai
@pytest.mark.parametrize("reading", rs.RIG_REST_READINGS)
def test_a_takeover_and_a_lift_run_from_each_rig_reading_with_the_rig_file(
    robot_rig, clock, monkeypatch, reading
):
    backend = rs.apai_backend
    # Human driving ships switched off in the rig file: on here, in memory only.
    on = dataclasses.replace(robot_rig.human, enabled=True)
    profile = backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.1}})
    config = backend.config_for_profile(
        dataclasses.replace(robot_rig, human=on), profile
    )
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, arm, checkers, clock)
    solver = Solver(profile.kinematics)
    try:
        h.start()
        # The takeover at measured rest sends the motors nothing.
        take_over(h)
        assert h.app.snapshot().owner is Owner.HUMAN
        assert sdk.arm.sent == []

        # A squeeze, then 3 cm straight up, ended by the released grip.
        ended = drives(h)
        hand = LiftingHand(h, sdk, solver)
        keep_human_fed(h, hand)
        feed(h)  # fresh controller data: the grip is released
        feed(h, hand.squeeze(reading))
        h.idle()
        assert hand.problem == ""
        assert h.heard("human") == []
        sent = [q for q, _ in sdk.arm.sent]
        assert [(end.status, end.sent) for end in ended] == [("released", len(sent))]
        # APAI's human checker of the rig's profile approved every setpoint.
        assert checkers.human.harness.approved_commands == len(sent)
        # No step is above the speed clamp of the rig's 0.1 rad/s.
        human, period = config.limits.human, config.executor.period_s
        steps = np.abs(np.diff(np.array([reading, *sent]), axis=0))
        assert steps.max() <= SPEED_MARGIN * human.max_joint_speed * period + 1e-12
        # No solution lay as far ahead of the last setpoint sent as makes the
        # human program end the squeeze: ik.max_ahead_rad, here the default of
        # human/types.py (0.08), which configs/human_rebot.yaml repeats.
        assert hand.ahead < IkSettings().max_ahead_rad
        start_tip = solver.fk(solver.inside(reading))[:3, 3]
        rise = solver.fk(sent[-1])[:3, 3] - start_tip
        assert rise == pytest.approx([0.0, 0.0, LIFT_M], abs=1e-3)
        assert sdk.disconnects == 0 and arm.torque_on()

        # The typed rest, then the hand-back.
        rest = h.client.wait_result(h.client.rest(), timeout_s=5.0)
        assert rest.status == "completed", rest.reason
        assert h.app.motion_state is MotionState.REST_HOLD
        assert sdk.arm.sent[-1][0] == [0.0] * 6
        count = len(sdk.arm.sent), len(sdk.gripper.sent)
        handback = h.client.request_owner("handback", reason="right A held 1.0 s")
        back = h.client.wait_result(handback, timeout_s=5.0)
        assert back.status == "completed", back.reason
        assert not h.app.takeover_open
        # The hand-back sends the motors nothing, the jaw motor included.
        assert (len(sdk.arm.sent), len(sdk.gripper.sent)) == count
        assert sdk.disconnects == 0 and arm.torque_on()
    finally:
        h.close()


def test_the_rig_human_file_names_the_hand_speed_and_the_ahead_limit_of_the_lift():
    # The lift above stands for the rig's human file only while that file
    # names the speed its hand rises with and the ahead limit it is held to.
    rig = shipped("human_rebot.yaml")
    assert rig["retarget"]["max_linear_velocity"] == HAND_SPEED
    assert rig["ik"]["max_ahead_rad"] == IkSettings().max_ahead_rad


# The edge of the rig file's human joint box. A STOP makes nobody the control
# owner and leaves the takeover open. A new takeover is refused where the
# human checker refuses the measured pose, and a typed rest, the operator's
# return to rest confirmed with "path is clear", is taken only from inside
# rest.region_*. The outer band is the part of the region outside the box.

OVER_RAD = 0.005  # a joint driven to a bound reads this far outside it
BAND_RAD = 0.01  # how far outside the box the pose in the outer band lies

# Per bound that has the margin, a pose there; the bound's own joint is filled
# in by beside(). With that joint 0.01 rad inside its bound APAI's human
# checker of the rebot_rs profile takes the pose, and with it up to BAND_RAD
# outside, APAI's recovery checker takes every setpoint of the return to rest.
BESIDE = {
    (0, "min"): (0.0, 0.3, 0.3, 1.5, -1.0, 0.0),
    (0, "max"): (0.0, 0.3, 0.3, 1.5, 1.0, 0.0),
    (1, "max"): (0.0, 0.0, 0.3, 1.5, 0.0, 0.0),
    (2, "max"): (0.0, 0.9, 0.0, -1.5, 0.0, 0.0),
    (3, "min"): (0.0, 0.6, 1.2, 0.0, 0.0, 0.0),
    (3, "max"): (0.0, 0.6, 0.1, 0.0, 0.0, 0.0),
    (4, "min"): (0.0, 0.3, 0.3, 0.0, 0.0, 0.0),
    (4, "max"): (0.0, 0.3, 0.3, 0.0, 0.0, 0.0),
    (5, "min"): (0.0, 0.3, 0.3, 0.5, 1.0, 0.0),
    (5, "max"): (0.0, 0.3, 0.3, 0.5, -1.0, 0.0),
}
REST_REFUSED = "rest: refused: joint index {} at {:.4f} rad is outside the tested "


def named(bound: tuple[int, str]) -> str:
    return f"joint{bound[0]}_{bound[1]}"


def beside(config: RobotAppConfig, bound: tuple[int, str], off: float) -> list[float]:
    """A pose whose joint lies `off` rad outside this bound of the human joint box.

    A negative `off` lies inside the box. The other joints are BESIDE's, or 0.
    """
    joint, side = bound
    human = config.limits.human
    pose = list(BESIDE.get(bound, (0.0,) * config.n_joints))
    if side == "min":
        pose[joint] = human.joint_min[joint] - off
    else:
        pose[joint] = human.joint_max[joint] + off
    return pose


class PersonRig(NamedTuple):
    """The rig file's robot app with human driving switched on, and its arm."""

    h: AppHarness
    place: Callable[[Sequence[float]], None]  # the arm reads this pose and holds
    sent: Callable[[], list[list[float]]]  # every joint target the motors got


@pytest.fixture(params=["sim", pytest.param("apai", marks=apai)])
def person_rig(request, robot_rig, clock, monkeypatch) -> Iterator[PersonRig]:
    """The rig file's robot app at rest, started, with human driving switched on.

    Once on the simulated arm with its checkers, which read exactly what was
    sent and know no kinematics, and once on APAI's driver of the rebot_rs
    profile over the stand-in motor SDK, with APAI's checkers.
    """
    # Human driving ships switched off in the rig file: on here, in memory only.
    on = dataclasses.replace(robot_rig.human, enabled=True)
    config = dataclasses.replace(robot_rig, human=on)
    if request.param == "sim":
        rest_q = list(config.rest.rest_q)
        arm = SimArm(n_joints=config.n_joints, clock=clock, q0=rest_q)
        h = AppHarness(config, arm, basic_checkers(config, clock), clock)
        place, sent = h.hold_at, h.sent
    else:
        backend = rs.apai_backend
        cap = {"safety": {"max_joint_vel": 0.1}}
        profile = backend.load_apai_profile("rebot_rs", cap)
        config = backend.config_for_profile(config, profile)
        arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
        checkers = backend.build_apai_checkers(config, profile, clock)
        h = AppHarness(config, arm, checkers, clock)
        # APAI's driver refuses a full read that lies farther from its last
        # good one than the arm can travel in the time between, which it takes
        # from the machine's clock. A return to rest of 40 s on the fake clock
        # lasts a fraction of a second there: the driver reads the app's clock.
        driver = arm.backend
        app_time = types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
        monkeypatch.setattr(sys.modules[type(driver).__module__], "time", app_time)

        def place(pose: Sequence[float]) -> None:
            for motor, value in zip(sdk.arm.joint_names, pose):
                sdk.arm._mm[motor].value = value
            # The arm is here without having travelled: the driver's last good
            # full read is this pose.
            driver._last_q = driver._read_positions()
            driver._last_q_t = clock.monotonic()
            h.app._motion.set(MotionState.HOLD)

        def sent() -> list[list[float]]:
            return [q for q, _ in sdk.arm.sent]

    try:
        h.start()
        yield PersonRig(h, place, sent)
    finally:
        h.close()


def refused_for_the_human_joint_box(h: AppHarness) -> None:
    """A takeover asked now is refused by the human joint box; it names the rest."""
    before = h.app.snapshot()
    feed(h)  # fresh controller data and a live view: neither is the reason
    refused = ask_takeover(h)
    assert refused.status == "refused"
    assert refused.reason.startswith(OUTSIDE_HUMAN_LIMITS), refused.reason
    assert "is outside the human joint limits" in refused.reason
    assert refused.reason.endswith("; type rest")
    assert h.app.snapshot() == before


def typed_rest(h: AppHarness) -> str:
    """Type rest and "path is clear" at the Thor terminal; the terminal's answer."""
    return h.thor_terminal().ask("rest", "path is clear", answer="rest: ")


def at_rest_after_the_return(rig: PersonRig) -> None:
    """The return to rest ran to its end, and a takeover is taken there."""
    h = rig.h
    assert h.app.motion_state is MotionState.REST_HOLD
    assert rig.sent()[-1] == list(h.config.rest.rest_q)
    assert h.arm.torque_on()
    assert take_over(h).status == "completed"


@pytest.mark.parametrize("bound", list(BESIDE), ids=named)
def test_a_typed_rest_is_taken_after_a_stop_just_outside_the_rig_human_box(
    person_rig, bound
):
    h, place, sent = person_rig
    # A person owns the arm, one joint 0.01 rad inside a bound of the box.
    place(beside(h.config, bound, -0.01))
    take_over(h)
    assert h.app.snapshot().owner is Owner.HUMAN
    # The joint then reads a few thousandths of a rad outside, and a STOP comes.
    place(beside(h.config, bound, OVER_RAD))
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.owner == "nobody" and s.stop_number == 1)
    assert h.app.takeover_open
    # No squeeze brings the joint back: a new takeover is refused there, and
    # the agent gets no task while the takeover is open.
    refused_for_the_human_joint_box(h)
    assert final(h, h.client.request_owner("agent")).reason == TAKEOVER_OPEN
    assert sent() == []
    # The reading lies inside rest.region_*: the typed rest is the way out.
    assert typed_rest(h) == "rest: completed"
    at_rest_after_the_return(person_rig)


@pytest.mark.parametrize("bound", list(BESIDE), ids=named)
def test_the_outer_band_of_the_rig_file_refuses_a_takeover_and_takes_a_typed_rest(
    person_rig, bound
):
    h, place, sent = person_rig
    joint, _ = bound
    # What the margin costs. The arm holds where an agent's move may end:
    # inside the region, outside the human joint box. Nobody owns it.
    band = beside(h.config, bound, BAND_RAD)
    rest = h.config.rest
    assert rest.region_min[joint] < band[joint] < rest.region_max[joint]
    place(band)
    assert h.app.snapshot().owner is Owner.NOBODY
    refused_for_the_human_joint_box(h)
    assert not h.app.takeover_open and sent() == []
    assert typed_rest(h) == "rest: completed"
    at_rest_after_the_return(person_rig)


@pytest.mark.parametrize("bound", ON_STOPS, ids=named)
def test_below_the_two_bounds_without_the_margin_a_typed_rest_is_refused(
    person_rig, bound
):
    # A known limit, left as it is. The second and third joint lie on their
    # stops at rest, and a joint target cannot drive a joint below its stop:
    # a reading below the region's -0.05 there is no tracking error. A margin
    # at these two bounds would take its width from a takeover at rest.
    h, place, sent = person_rig
    joint, _ = bound
    place(beside(h.config, bound, -0.01))
    take_over(h)
    under = beside(h.config, bound, OVER_RAD)
    place(under)
    h.client.stop(reason="right B", sender="controller")
    h.client.wait_state(lambda s: s.owner == "nobody" and s.stop_number == 1)
    refused_for_the_human_joint_box(h)
    refused = typed_rest(h)
    assert refused.startswith(REST_REFUSED.format(joint, under[joint])), refused
    assert refused.endswith("recovery region -0.0500..2.0944")
    assert sent() == [] and h.arm.torque_on()


# A low gripper tip on the rig file. The table clearance is the height over the
# table the gripper tip keeps outside a grasp: rest.table_clearance_m of the
# file and safety.table_clearance of APAI's profile. A tip under it is low: a
# typed rest is refused there, a takeover is granted, and until the measured
# tip is clear a human target is followed only when it does not lower the tip.

# Poses of the rebot_rs arm model inside the rig's human joint box: the gripper
# tip 1 cm over the table, the same with the tip 4 mm lower, and the tip 3 cm
# over the table.
TIP_LOW = [0.0, 1.68, 0.7, 0.0, 0.5, 0.0]
TIP_LOWER = [0.0, 1.69, 0.7, 0.0, 0.5, 0.0]
TIP_CLEAR = [0.0, 1.62, 0.7, 0.0, 0.5, 0.0]
STRIDE = 0.05  # rad a hand's target runs ahead of the last joint target sent


class JointHand:
    """A hand that squeezes toward a pose: keep_human_fed's setpoint.

    Its target runs at most STRIDE per joint ahead of the last joint target
    the motors got. The grip is released once they got the goal.
    """

    def __init__(self, rig: PersonRig) -> None:
        self.rig = rig
        self.goal: list[float] | None = None  # None: the grip is released
        keep_human_fed(rig.h, self)

    def __call__(self) -> HumanSetpoint:
        if self.goal is None:
            return setpoint(self.rig.h)
        last = self.rig.sent()[-1]
        if max(abs(to - at) for at, to in zip(last, self.goal)) < 1e-9:
            self.goal = None
            return setpoint(self.rig.h)
        return setpoint(self.rig.h, self.ahead(last))

    def ahead(self, last: Sequence[float]) -> list[float]:
        """The target that leads `last` toward the goal."""
        return [
            at + max(-STRIDE, min(STRIDE, to - at)) for at, to in zip(last, self.goal)
        ]

    def squeeze(self, measured: Sequence[float], goal: Sequence[float]) -> None:
        """Squeeze at the measured pose toward goal; the human drive has ended."""
        h = self.rig.h
        self.goal = None
        feed(h)  # fresh controller data: the grip is released
        self.goal = [float(value) for value in goal]
        feed(h, setpoint(h, self.ahead(measured)))
        h.idle()


@pytest.mark.parametrize(
    "person_rig", [pytest.param("apai", marks=apai)], indirect=True
)
def test_a_person_lifts_a_low_gripper_tip_out_with_the_rig_file(person_rig):
    # Only APAI's checkers know a table: the simulated arm's have no such rule.
    h, place, sent = person_rig
    human = h.checkers.human
    clearance = h.config.rest.table_clearance_m
    place(TIP_LOW)
    low = human.tip_state(TIP_LOW, None)
    assert 0.0 < low.z < clearance and low.low is not None
    # The human checker refuses the pose, and for the low tip alone.
    with pytest.raises(Refusal, match=r"^TCP z=0\.011 below table clearance 0\.020$"):
        human.check_pose(TIP_LOW)
    human.check_low_pose(TIP_LOW)
    # No typed command moves the arm from there.
    assert typed_rest(h) == f"rest: refused: {low.low}"
    # A takeover is granted, and it says what a person can do there.
    number = take_over(h).detail["owner_number"]
    assert h.heard("owner")[-1][1] == TAKEN_OVER.format(number=number) + LOW_TIP
    assert h.app.snapshot().owner is Owner.HUMAN and sent() == []
    ended = drives(h)
    hand = JointHand(person_rig)
    # A target that lowers the tip moves nothing: the arm holds.
    hand.squeeze(TIP_LOW, TIP_LOWER)
    assert [(end.status, end.sent) for end in ended] == [("refused", 0)]
    assert ended[0].reason.startswith(f"{LIFT}{low.low}, and the target would lower")
    assert sent() == [] and h.arm.torque_on()
    # The lift: every setpoint raises the tip, up to above the table clearance.
    hand.squeeze(TIP_LOW, TIP_CLEAR)
    lift = sent()
    assert [(end.status, end.sent) for end in ended[1:]] == [("released", len(lift))]
    assert lift[-1] == pytest.approx(TIP_CLEAR, abs=1e-9)
    heights = [human.tip_state(q, None).z for q in lift]
    assert all(b > a for a, b in zip([low.z, *heights], heights))
    assert heights[-1] > clearance and human.tip_state(lift[-1], None).low is None
    assert len(h.heard("human")) == 1  # the refused squeeze's, none since
    # With the tip clear the typed rest is taken; the takeover stays open.
    assert typed_rest(h) == "rest: completed"
    assert h.app.motion_state is MotionState.REST_HOLD
    assert sent()[-1] == list(h.config.rest.rest_q)
    assert h.arm.torque_on() and h.app.takeover_open


# agent_rebot.yaml


def test_the_two_rig_files_agree_on_speed_waits_profile_and_calibration(
    robot_rig, path_variables
):
    settings = agent_config.load_settings(AGENT_RIG)
    thor = settings.thor
    assert thor.first_wait_s == robot_rig.permits.first_wait_s == 0.5
    assert thor.settle_timeout_s == robot_rig.executor.settle_timeout_s == 2.0
    assert thor.agent_max_joint_speed == robot_rig.limits.agent.max_joint_speed == 0.1
    assert settings.task.max_joint_speed == 0.1
    assert settings.task.timed_return is False and robot_rig.rest.auto_return is False
    link = settings.link
    assert link.expected_profile_id == robot_rig.profile_id == PROFILE_ID
    # The name is made from the two files' hashes; a robot app must name one.
    assert link.calibration_id == "" and link.allow_unnamed_calibration is False
    assert robot_rig.camera_calibration_id
    assert settings.rig.park_q == robot_rig.rest.rest_q
    # The agent's bound on the joints is not wider than the Thor's working box.
    bound = settings.rig.motion_bounds["max_joint_deviation_rad"]
    agent = robot_rig.limits.agent
    assert all(bound <= high or high < 2.0944 for high in agent.envelope_max)
    assert bound <= 2.0944


def test_the_rig_agent_file_is_the_real_run(path_variables):
    settings = agent_config.load_settings(AGENT_RIG)
    camera, experiment = settings.camera, settings.experiment
    folder = path_variables[agent_config.EXPERIMENT_DIR_ENV]
    assert camera.calibration_dir == f"{folder}/.state/thor/calibration"
    assert camera.hand_eye_json == (
        f"{folder}/.state/thor/mounting/20261107T005443Z/handeye.json"
    )
    camera_pr = path_variables["AGENTIC_REMOTE_CAMERA_PR"]
    assert camera.bridge_python == f"{camera_pr}/cuda_ipc_consumer/.venv/bin/python"
    assert (camera.depth_backend, camera.fps) == ("ffs", 7.0)
    assert camera.allow_invented_times is False
    assert experiment.stack == "full" and not settings.minimal
    env_dir = path_variables["AGENTIC_REMOTE_ENV_DIR"]
    assert experiment.codex_bin == f"{env_dir}/codex/node_modules/.bin/codex"
    # It runs against the real driver, and against nothing that declares none.
    agent_config.check_robot_driver(settings, "real", invented=False)
    agent_config.check_robot_driver(settings, "", invented=False)
    with pytest.raises(agent_config.ConfigError, match="allow_invented_times is true"):
        agent_config.check_robot_driver(settings, "real", invented=True)


def test_the_rig_agent_file_differs_from_the_shipped_one_where_the_rig_does(
    path_variables,
):
    rig, demo = shipped("agent_rebot.yaml"), shipped("agent.yaml")
    differs = {
        (block, key)
        for block in demo
        if isinstance(demo[block], dict)
        for key in demo[block]
        if rig[block][key] != demo[block][key]
    }
    assert differs == {
        ("link", "expected_profile_id"),
        ("camera", "bridge_python"),
        ("camera", "calibration_dir"),
        ("camera", "hand_eye_json"),
        ("experiment", "codex_bin"),
    }
    assert set(rig) == set(demo)
    for block in demo:
        if isinstance(demo[block], dict):
            assert set(rig[block]) == set(demo[block]), block
        else:
            assert rig[block] == demo[block], block
    # The permits, the move and the jaws keep the shipped first values.
    settings = agent_config.load_settings(AGENT_RIG)
    assert settings.permits.reach_setpoints == 25
    assert (settings.permits.renew_s, settings.permits.life_s) == (0.05, 1.0)


def test_the_rig_agent_file_is_refused_without_its_variables(monkeypatch):
    for name in agent_config.PATH_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(agent_config.ConfigError, match="which is not set"):
        agent_config.load_settings(AGENT_RIG)


@pytest.mark.experiment
def test_the_two_rig_files_name_the_calibration_of_the_experiments_files(
    robot_rig, experiment_dir: Path, monkeypatch, tmp_path
):
    monkeypatch.setenv(agent_config.EXPERIMENT_DIR_ENV, str(experiment_dir))
    for name in ("AGENTIC_REMOTE_CAMERA_PR", "AGENTIC_REMOTE_ENV_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    settings = agent_config.load_settings(AGENT_RIG)
    assert Path(settings.camera.calibration_dir, "manifest.json").is_file()
    assert Path(settings.camera.hand_eye_json).is_file()
    named = agent_config.calibration_id(settings, "link")
    assert named == robot_rig.camera_calibration_id == CALIBRATION_ID
    # The rig block is the experiment's arm profile, as in the shipped file.
    theirs = yaml.safe_load(
        (experiment_dir / "configs/thor_operation.yaml").read_text()
    )
    assert list(settings.rig.home_q) == theirs["home_q"]
    assert dict(settings.rig.motion_bounds) == theirs["motion_bounds"]
    # The motor file names the six joints the robot file names.
    motors = yaml.safe_load(
        (experiment_dir / ".state/thor/robot/sdk-can4.yaml").read_text()
    )
    assert tuple(motors["groups"]["arm"]["joints"]) == robot_rig.joint_names
    assert motors["channel"] == "can4" and robot_rig.robot_id.endswith("can4")


# link_service_rebot.yaml


def test_the_shipped_rig_service_file_is_refused_until_the_address_is_written(
    tmp_path, capsys
):
    with pytest.raises(
        service_config.ConfigError, match="allowed_peers must be a non-empty list"
    ):
        service_config.load_link_service_config(SERVICE_RIG)
    # The program ends before it listens anywhere.
    assert service_main.main(["--config", str(SERVICE_RIG)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "robot link service: the configuration cannot be used: allowed_peers must "
        "be a non-empty list"
    )
    # With the Thor's address written in, as the file's comment shows.
    text = SERVICE_RIG.read_text(encoding="utf-8")
    assert text.count("\nallowed_peers: []\n") == 1
    assert "#   allowed_peers: [192.0.2.10]\n" in text
    written = tmp_path / "link_service_rebot.yaml"
    written.write_text(
        text.replace("\nallowed_peers: []\n", "\nallowed_peers: [192.0.2.10]\n"),
        encoding="utf-8",
    )
    config = service_config.load_link_service_config(written)
    assert config.listen_endpoint == "0.0.0.0:47610"
    assert config.allowed_peers == ("192.0.2.10",)
    # Everything else is the shipped service file.
    demo = service_config.load_link_service_config(CONFIGS / "link_service.yaml")
    assert config.service_options() == {
        **demo.service_options(),
        "allowed_peers": ("192.0.2.10",),
    }
    assert (config.socket_path, config.log_level) == (demo.socket_path, demo.log_level)
