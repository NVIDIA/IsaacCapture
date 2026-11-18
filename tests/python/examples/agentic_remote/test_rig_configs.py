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
import itertools
import math
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
from isaaccapture_examples.agentic_remote.human import logic as human_logic
from isaaccapture_examples.agentic_remote.human.ik import Solver
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
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.rest import (
    planned_duration_s,
    setpoint_count,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from repo_paths import repo_root

import test_apai_backend as rs
from conftest import AppHarness, basic_checkers, import_experiment
from test_takeover import (
    ask_takeover,
    drives,
    feed,
    final,
    human_facts,
    keep_human_fed,
    setpoint,
    state_now,
    take_over,
)

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
ROBOT_RIG = CONFIGS / "robot_app_rebot.yaml"
AGENT_RIG = CONFIGS / "agent_rebot.yaml"
SERVICE_RIG = CONFIGS / "link_service_rebot.yaml"
RIG_FILES = (ROBOT_RIG, AGENT_RIG, SERVICE_RIG)
CALIBRATION_ID = "stereo-c50bf5b2b899+mount-6ba1cda8fdc3"
PROFILE_ID = "rebot-b601-rs-thor"
# The owner's values of 2026-11-16, rad/s: the agent's moves and the return to
# rest, and the bound on a person's motion.
OWNER_SPEED = 0.2
HUMAN_SPEED = 0.8
# The farthest a human target may lie from the arm at a squeeze, and from the
# last joint target sent after it, rad: limits.human.max_step_rad of the rig's
# robot file. Below it, the farthest the human program lets its own joint
# target run ahead: ik.max_ahead_rad of configs/human_rebot.yaml.
HUMAN_STEP = 0.05
HUMAN_AHEAD = 0.045
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


def rig_profile(folder: Path) -> Any:
    """The rig's arm profile as the robot app loads it on the Thor.

    Laid out in `folder` as the experiment's files are there
    (rs.thor_arm_profile): APAI's rebot_rs profile, run by the experiment's
    launcher, whose joint speed is the profile's speed cap.
    """
    return rs.apai_backend.load_apai_profile(rs.thor_arm_profile(folder))


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
    # The owner's speeds. test_motion_conform.py holds the agent's and the
    # recovery speed against the experiment launcher's, which bounds them.
    assert limits.human.max_joint_speed == HUMAN_SPEED
    for limit_set in (limits.agent, limits.recovery):
        assert limit_set.max_joint_speed == OWNER_SPEED
        assert limit_set.version == "rebot-p4-1"
        assert (limit_set.joint_min, limit_set.joint_max) == (BOX_LOW, MODEL_HIGH)
    # The human joint box lies inside the recovery region, which is the agent's
    # working box: by HUMAN_BOX_MARGIN_RAD on every bound but the two ON_STOPS.
    assert limits.human.version == "rebot-p3-1"
    assert (limits.human.joint_min, limits.human.joint_max) == (
        HUMAN_LOW,
        HUMAN_HIGH,
    )
    assert limits.human.max_step_rad == HUMAN_STEP
    assert (limits.agent.envelope_min, limits.agent.envelope_max) == (
        ENVELOPE_LOW,
        ENVELOPE_HIGH,
    )
    rest = robot_rig.rest
    assert rest.rest_q == (0.0,) * 6
    assert (rest.rest_tol_rad, rest.start_rest_tol_rad) == (0.05, 0.05)
    assert rest.recovery_speed == OWNER_SPEED
    # The recovery region is the agent's working box; no timed return.
    assert (rest.region_min, rest.region_max) == (ENVELOPE_LOW, ENVELOPE_HIGH)
    assert rest.auto_return is False
    assert robot_rig.camera_calibration_id == CALIBRATION_ID


def test_the_rig_robot_file_keeps_the_shipped_values_elsewhere(robot_rig):
    default = RobotAppConfig.default()
    # The settle tolerance of APAI's reBot driver, which --arm apai reads from
    # the driver itself (apai_backend.config_for_profile).
    assert robot_rig.executor == dataclasses.replace(
        default.executor, settle_tol_rad=0.05
    )
    for block in (
        "permits",
        "waits",
        "gripper",
        "grasp",
        "connection",
    ):
        assert getattr(robot_rig, block) == getattr(default, block), block
    # The human block is the shipped default's: human driving off, and the
    # live view required.
    assert robot_rig.human == default.human
    # An idle joint is read as often as the experiment reads it.
    assert robot_rig.state == dataclasses.replace(
        default.state, idle_read_period_s=0.01
    )
    rig, demo = shipped("robot_app_rebot.yaml"), shipped("robot_app.yaml")
    assert set(rig) == set(demo)
    assert {
        name: set(block) for name, block in rig.items() if isinstance(block, dict)
    } == {name: set(block) for name, block in demo.items() if isinstance(block, dict)}
    changed = {"auto_return", "region_min", "region_max", "recovery_speed"}
    for key, value in demo["rest"].items():
        assert (rig["rest"][key] == value) == (key not in changed), key


def test_the_owners_speeds_are_marked_as_the_owners_in_the_rig_file():
    lines = ROBOT_RIG.read_text(encoding="utf-8").splitlines()
    marked = [line.strip() for line in lines if "the owner's value of" in line]
    assert marked == [
        f"max_joint_speed: {OWNER_SPEED}  # the owner's value of 2026-11-16",
        f"max_joint_speed: {HUMAN_SPEED}  # the owner's value of 2026-11-16",
        f"max_joint_speed: {OWNER_SPEED}  # the owner's value of 2026-11-16",
        f"recovery_speed: {OWNER_SPEED}  # the owner's value of 2026-11-16",
    ]


@pytest.mark.experiment
def test_the_rig_reads_an_idle_joint_as_often_as_the_functional_sampler(robot_rig):
    sampler = import_experiment("thor_self_mask").IdleJointSampler
    assert robot_rig.state.idle_read_period_s == sampler.PERIOD_S
    # One ROBOT_STATE carries the readings of several such reads.
    assert robot_rig.state.publish_period_s >= 2 * sampler.PERIOD_S


def test_what_scales_with_the_speeds_still_holds_at_the_owners(robot_rig):
    rest, executor = robot_rig.rest, robot_rig.executor
    # The longest leg of a return to rest, from the far end of the recovery
    # region, fits a move: the loader checks it, and it is the count the
    # file's comment names.
    travel = max(
        max(abs(low - q), abs(high - q))
        for low, high, q in zip(rest.region_min, rest.region_max, rest.rest_q)
    )
    duration = planned_duration_s(travel, rest.recovery_speed, rest.min_duration_s)
    count = setpoint_count(duration, executor.period_s)
    assert count == 1091 <= executor.max_setpoints
    assert f"{count} setpoints for {travel:g} rad at {OWNER_SPEED} rad/s" in " ".join(
        ROBOT_RIG.read_text(encoding="utf-8").replace("#", " ").split()
    )
    assert rest.recovery_speed <= robot_rig.limits.recovery.max_joint_speed
    # A person: the Thor clamps one step of a human drive inside the farthest
    # a human target may lie from the last joint target sent, so the clamp is
    # met before a target is refused for its distance. The farthest target
    # that is taken is reached with the fourth setpoint, after 0.08 s.
    human = robot_rig.limits.human
    step = SPEED_MARGIN * human.max_joint_speed * executor.period_s
    assert step == pytest.approx(0.01568)
    assert 3 * step < human.max_step_rad < 4 * step
    # The human program moves the joint target it sends at 0.9 of that step,
    # whatever the hand does, so the Thor's own joint target is never left
    # behind and the step limit is not met while the arm follows: a target
    # lies one such step from the Thor's last one, and three of them fit
    # inside the limit, which is what a pause of the Thor's loop may cost.
    own = JOINT_SHARE * step
    assert own == pytest.approx(0.014112) and 3 * own < human.max_step_rad
    # The bounds of the tip's target in the rig's human file ask the joints
    # for more than that: on the arm model 1 m/s at the gripper tip asks a
    # joint for 10 rad/s in the middle of the poses, 1 rad/s of the tip's
    # turning for 3. The joints, not those bounds, set the arm's pace.
    retarget = shipped("human_rebot.yaml")["retarget"]
    assert (retarget["max_linear_velocity"], retarget["max_angular_velocity"]) == (
        TIP_SPEED,
        TIP_TURN,
    )
    asked = min(10.0 * TIP_SPEED, 3.0 * TIP_TURN)
    assert asked == pytest.approx(2.5) and asked > human.max_joint_speed


def test_the_human_settings_ship_in_the_safe_position(robot_rig):
    # Human driving off on the rig, as in the shipped default.
    assert robot_rig.human.enabled is False
    for name in ("robot_app.yaml", "robot_app_rebot.yaml"):
        assert shipped(name)["human"]["enabled"] is False, name
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert "\nhuman:\n  enabled: false" in text, name


def test_the_rig_files_ask_for_the_live_view_as_the_shipped_default_does(robot_rig):
    # The live view is the viewer's page with its live picture: the rig's robot
    # file and the file of the headset checks before power require it, and say
    # in one line what it is.
    said = (
        "  # The live view is the viewer's page with its live picture "
        "(run_viewer.sh).\n  require_live_view: true\n"
    )
    assert robot_rig.human.require_live_view is True
    for name in ("robot_app_rebot.yaml", "robot_app_human_check.yaml"):
        assert shipped(name)["human"]["require_live_view"] is True, name
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert said in text, name
        # Neither says any more that the person needs no live view, nor names
        # the option that started the page of before.
        for gone in (
            "sees it",
            "see the arm",
            "--live-view",
            "require_live_view: false",
        ):
            assert gone not in text, (name, gone)
    assert shipped("robot_app_human_check.yaml")["human"]["enabled"] is True
    assert robot_rig.human.view_age_limit_s == 0.5
    # As the shipped default.
    assert shipped("robot_app.yaml")["human"]["require_live_view"] is True
    assert RobotAppConfig.default().human.require_live_view is True
    # No robot file says any more that false needs a simulated driver.
    for path in sorted(CONFIGS.glob("robot_app*.yaml")):
        text = path.read_text(encoding="utf-8").lower()
        assert "only with a simulated" not in text, path.name


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
def test_the_rig_file_loads_and_fits_the_rebot_profile(
    robot_rig, monkeypatch, tmp_path
):
    backend = rs.apai_backend
    monkeypatch.setattr(rs.sys, "path", list(rs.sys.path))  # the driver's SDK path
    clock = FakeClock()
    # APAI's own profile (cap 0.8 rad/s), and the rig's profile as the robot
    # app loads it on the Thor: capped at the experiment launcher's speed.
    for profile in (backend.load_apai_profile("rebot_rs"), rig_profile(tmp_path)):
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
        # Each owner's checker has its own speed of the file, whatever the cap.
        speeds = [checker.harness.limits.max_joint_vel for checker in checkers.all()]
        assert speeds == [OWNER_SPEED, HUMAN_SPEED, OWNER_SPEED]
    # On the Thor the launcher's speed bounds the agent and the return to
    # rest, and the cap of APAI's rebot_rs profile bounds a person.
    assert backend.speed_cap(profile) == rs.LAUNCHER_SPEED >= OWNER_SPEED
    assert backend.human_speed_cap(profile) == HUMAN_SPEED
    # A launcher at the owner's speed takes the file, a slower one refuses it:
    # the Thor's copy of the launcher's file cannot be below the file's speeds.
    same = backend.load_apai_profile(rs.thor_arm_profile(tmp_path / "a", OWNER_SPEED))
    backend.build_apai_checkers(robot_rig, same, clock)
    slow = backend.load_apai_profile(rs.thor_arm_profile(tmp_path / "b", 0.1))
    with pytest.raises(
        ValueError,
        match=r"limits\.agent\.max_joint_speed 0\.2 rad/s is above the APAI "
        r"profile's speed cap of 0\.1 rad/s",
    ):
        backend.build_apai_checkers(robot_rig, slow, clock)
    # A profile no launcher runs binds a person by its own cap: the rig's own
    # fallback cap, 0.2, is below the file's human speed.
    capped = backend.load_apai_profile("rebot_rs", {"safety": {"max_joint_vel": 0.2}})
    with pytest.raises(ValueError, match=r"limits\.human\.max_joint_speed 0\.8"):
        backend.build_apai_checkers(robot_rig, capped, clock)
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
    robot_rig, clock, monkeypatch, reading, tmp_path
):
    backend = rs.apai_backend
    profile = rig_profile(tmp_path)
    config = backend.config_for_profile(robot_rig, profile)
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    # As the robot app's program builds it: with a new arm port for the motors
    # to be switched on again, a typed rest ends with the motors off.
    built: list[tuple[Any, rs.MotorSdk]] = []

    def rebuild_arm() -> Any:
        built.append(rs.rs_arm(config, profile, clock, monkeypatch))
        return built[-1][0]

    options = {"rebuild_arm": rebuild_arm}
    h = AppHarness(config, arm, checkers, clock, app_options=options)
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
        jaw_mark = len(sdk.gripper.sent)
        assert thor.ask("rest", "path is clear", answer="rest: ") == "rest: completed"
        # It ends as the functional rest: at rest, the jaws closed with the
        # driver's close, the motors off.
        assert h.app.motion_state is MotionState.REST_UNPOWERED
        assert sdk.arm.sent[-1][0] == [0.0] * 6
        # The return moved no joint farther than the reading lay from rest.
        travel = max(abs(value) for value in reading)
        assert all(
            max(abs(q) for q in sent) <= travel + 1e-9 for sent, _ in sdk.arm.sent
        )
        closes = sdk.gripper.sent[jaw_mark:]
        assert closes and all(q == [0.0] for q, _ in closes)
        assert sdk.disconnects == 1 and not arm.torque_on()
        # The wanted motors-off brings no fault-stop.
        h.advance(1.0, step_s=0.02)
        assert h.heard("fault") == []
        assert h.app.snapshot().stop_level is StopLevel.NONE
        again = h.client.wait_result(h.client.release(), timeout_s=5.0)
        assert (again.status, again.reason) == ("refused", "the motors are already off")
        # A typed on switches them on again, on a new arm port, at measured rest.
        assert built == []
        assert thor.ask("on", "yes", answer="on: ") == "on: completed"
        assert h.app.motion_state is MotionState.REST_HOLD
        (new_arm, new_sdk) = built[0]
        assert new_arm.torque_on() and new_sdk.disconnects == 0
        assert new_sdk.arm.sent == [] and sdk.disconnects == 1
    finally:
        h.close()


@apai
@pytest.mark.parametrize("reading", rs.RIG_REST_READINGS)
def test_a_new_task_starts_from_each_rig_reading_with_the_rig_file(
    robot_rig, clock, monkeypatch, reading, tmp_path
):
    backend = rs.apai_backend
    profile = rig_profile(tmp_path)
    config = backend.config_for_profile(robot_rig, profile)
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, arm, checkers, clock)
    try:
        h.start()
        h.new_task()
        # Off the two joint stops, at half the agent's speed limit.
        raised = [reading[0], 0.1, 0.1, *reading[3:]]
        travel = max(abs(to - at) for at, to in zip(reading, raised))
        per_step = config.limits.agent.max_joint_speed * config.executor.period_s
        assert per_step == pytest.approx(OWNER_SPEED * 0.02)
        slow = math.ceil(travel / (0.5 * per_step))
        moved = rs.rs_move(h, rs.straight(reading, raised, slow), reading)
        assert moved.status == "completed", moved.reason
        assert sdk.arm.sent[-1][0] == pytest.approx(raised)
        # A move at twice the agent's speed limit is refused whole: the
        # launcher's speed, twice the owner's, does not reach the arm.
        sent = len(sdk.arm.sent)
        fast = int(travel / (2.0 * per_step))
        back = rs.rs_move(h, rs.straight(raised, list(reading), fast), raised)
        assert back.status == "refused" and "velocity" in back.reason, back.reason
        assert f"exceeds {OWNER_SPEED:.2f}" in back.reason
        assert len(sdk.arm.sent) == sent
        assert sdk.disconnects == 0 and arm.torque_on()
    finally:
        h.close()


# A person on the rig file. A takeover makes a person the control owner; the
# hand-back ends it at rest. A human drive is one stretch of human motion: the
# arm follows the joint targets of HUMAN_SETPOINT link messages. The human
# program makes them on the workstation by inverse kinematics (IK): from a
# pose of the gripper tip to joint values.

# The hand of the tests below is the human program itself: its retargeting
# graph, its solver and its logic on configs/human_rebot.yaml, run pass by
# pass on a script of the synthetic controller and on the ROBOT_STATE the
# robot app would send at that moment. The tests are skipped without Isaac
# Capture, whose nodes the graph is made of.

# m/s: a hand faster than the joints let the arm follow it.
HAND_SPEED = 0.2
LIFT_M = 0.10  # straight up
# The bounds of the tip's target and the farthest the hand may be ahead in
# configs/human_rebot.yaml; a test below holds them to the file.
TIP_SPEED, TIP_TURN, HAND_AHEAD_M = 0.25, 1.0, 0.5
# The share of the Thor's speed clamp at which the human program moves the
# joint target it sends (human/logic.py, JOINT_SPEED_SHARE).
JOINT_SHARE = 0.9
# Hand moves in the stage space, as unit vectors: the gripper tip then moves
# along +z, -x and +y of the robot's base axes.
HAND_UP, HAND_BACK, HAND_LEFT = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)
HAND_AT = (0.0, 1.2, -0.3)  # where the hand is at the squeeze, m


def hand_rows(
    way: Sequence[float], metres: float, speed: float, hold_s: float | None = None
) -> list[dict[str, Any]]:
    """A script: a squeeze, the hand moves metres at speed, a release hold_s later.

    The hand moves from the pass after the squeeze on. hold_s None: no release.
    """
    period = 0.02
    goal = [at + metres * step for at, step in zip(HAND_AT, way)]
    rows: list[dict[str, Any]] = [
        {"t": 0.0, "pos": list(HAND_AT), "quat": [0, 0, 0, 1]},
        {"t": 2 * period, "squeeze": 1.0},
        {"t": 2 * period + metres / speed, "pos": goal},
    ]
    if hold_s is not None:
        rows.append({"t": 2 * period + metres / speed + hold_s, "squeeze": 0.0})
    return rows


class ProgramHand:
    """The human program's graph, solver and logic: keep_human_fed's setpoint.

    Each call is one pass on the app's clock. Before the squeeze run() makes
    the passes, one period apart; from the squeeze on the hand is asked after
    every setpoint the arm is sent, in step with the human drive.
    """

    def __init__(self, h: AppHarness, rows: Sequence[dict[str, Any]]) -> None:
        pytest.importorskip("isaaccapture.retargeters")
        from isaaccapture_examples.agentic_remote.human import __main__ as program
        from isaaccapture_examples.agentic_remote.human.feed import ScriptFeed
        from isaaccapture_examples.agentic_remote.human.logic import HumanLogic
        from isaaccapture_examples.agentic_remote.human.retarget import HumanGraph

        self.h, self._as_sent = h, program.as_sent
        self.settings = program.load_human_settings(CONFIGS / "human_rebot.yaml")
        ik = self.settings.ik
        self.solver = Solver.from_apai(
            ik.apai_arm, max_iter=ik.max_iter, tol=ik.tol, max_jump_rad=ik.max_jump_rad
        )
        self.graph = HumanGraph(self.settings.retarget)
        self.feed = ScriptFeed(self.graph, rows)
        # The headset path's word: a real arm takes no synthetic controller.
        self.logic = HumanLogic(self.graph, self.solver, self.settings, "teleop")
        self.sent: list[HumanSetpoint] = []  # the HUMAN_SETPOINTs it made
        self.on_pass: Callable[[], None] | None = None  # called before each pass
        keep_human_fed(h, self)

    def __call__(self) -> HumanSetpoint | None:
        """One pass: its HUMAN_SETPOINT; None on a pass that makes none."""
        if self.on_pass is not None:
            self.on_pass()
        h, logic, graph = self.h, self.logic, self.graph
        now = h.clock.monotonic()
        events = graph.events(stopped=not logic.armed)
        fed = self.feed.step(graph.external_inputs(logic.armed), events, now)
        state = h.app._publisher.build(now)
        period = h.config.executor.period_s
        messages = logic.step(now, fed, h.client.welcome, state, period, ())
        logic.log_pass()
        made = None
        for message in messages:
            if isinstance(message, HumanSetpoint):
                made = self._as_sent(message, now, now, now)
                self.sent.append(made)
            else:
                h.client.send(message)  # right B's STOP, ahead of the setpoint
        return made

    def _pass(self) -> None:
        """One pass a period after the last; the app has taken what it made."""
        self.h.clock.advance(self.h.config.executor.period_s)
        made = self()
        if made is not None:
            feed(self.h, made)
        self.h.step()

    def run(self, after: int = 3) -> None:
        """The script up to the end of the human drive its squeeze starts.

        Then `after` more passes, on the ROBOT_STATEs that show the drive's end.
        """
        for _ in range(200):
            self._pass()
            if self.sent[-1].engaged:
                break
        assert self.sent[-1].engaged, self.sent[-1].note
        self.h.idle()
        for _ in range(after):
            self._pass()

    def targets(self) -> np.ndarray:
        """The joint targets the program sent, in their order."""
        return np.array([sent.q for sent in self.sent if sent.engaged])

    def ended(self) -> dict[str, int]:
        """How many engagements the human program itself ended, by cause."""
        counts = self.logic.counts()
        own = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
        return {key: counts[key] for key in own if counts[key]}


@apai
@pytest.mark.parametrize("reading", rs.RIG_REST_READINGS)
def test_a_takeover_and_a_lift_run_from_each_rig_reading_with_the_rig_file(
    robot_rig, clock, monkeypatch, reading, tmp_path
):
    backend = rs.apai_backend
    # Human driving ships switched off in the rig file: on here, in memory only.
    on = dataclasses.replace(robot_rig.human, enabled=True)
    profile = rig_profile(tmp_path)
    config = backend.config_for_profile(
        dataclasses.replace(robot_rig, human=on), profile
    )
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    for motor, value in zip(sdk.arm.joint_names, reading):
        sdk.arm._mm[motor].value = value
    checkers = backend.build_apai_checkers(config, profile, clock)
    # As on the rig: the robot app declares the real driver, and the rig file
    # requires the live view. A live VIEW_STATUS, the viewer's word that its
    # page shows the live picture, goes with every HUMAN_SETPOINT below.
    assert config.human.require_live_view is True
    h = AppHarness(config, arm, checkers, clock, app_options={"driver": "real"})
    solver = Solver(profile.kinematics)
    try:
        h.start()
        assert h.client.welcome.driver == "real"
        # The takeover at measured rest sends the motors nothing.
        feed(h)  # fresh controller data, the grip released, and a live view
        assert human_facts(h)["takeover"] == ""
        taken = ask_takeover(h)
        assert taken.status == "completed", taken.reason
        h.client.wait_state(lambda s: s.owner_number == taken.detail["owner_number"])
        assert h.app.snapshot().owner is Owner.HUMAN
        assert sdk.arm.sent == []

        # A squeeze with the hand rising 10 cm at 20 cm a second from the
        # squeeze on, ended by the released grip: the human program's own
        # joint targets, as it makes them on the rig's human file.
        ended = drives(h)
        hand = ProgramHand(h, hand_rows(HAND_UP, LIFT_M, HAND_SPEED, hold_s=1.5))
        hand.run()
        assert hand.ended() == {} and hand.logic.counts()["engagements"] == 1
        assert hand.logic.counts()["needs_release"] == 0
        assert h.heard("human") == []
        assert state_now(h).view_age_s is not None
        sent = [q for q, _ in sdk.arm.sent]
        assert [(end.status, end.sent) for end in ended] == [("released", len(sent))]
        # APAI's human checker of the rig's profile approved every setpoint.
        assert checkers.human.harness.approved_commands == len(sent)
        # No step is above the Thor's speed clamp.
        human, period = config.limits.human, config.executor.period_s
        clamp = SPEED_MARGIN * human.max_joint_speed * period
        steps = np.abs(np.diff(np.array([reading, *sent]), axis=0)).max(axis=1)
        assert steps.max() <= clamp + 1e-12
        # The squeeze begins inside the IK's joint margin, off the two joint
        # stops the arm rests on: the arm takes that distance at the clamp,
        # in a few setpoints, while the human program's target stands. From
        # there the fastest joint runs at 0.9 of the clamp, the human
        # program's own bound, until the arm is where the hand stopped.
        first = np.array(solver.inside(reading))
        off_the_stops = np.abs(first - np.array(reading)).max()
        catch_up = math.ceil(off_the_stops / clamp)
        assert 1 <= catch_up <= 3
        assert steps[: catch_up - 1] == pytest.approx(clamp, abs=1e-12)
        own = JOINT_SHARE * clamp
        at_the_clamp = int(np.sum(steps > own + 1e-9))
        assert catch_up - 1 <= at_the_clamp <= catch_up + 12
        assert 0 < state_now(h).health["clamped"] <= at_the_clamp
        assert int(np.sum(np.abs(steps - own) < 1e-9)) > 20
        # The squeeze's first target is the farthest of all from what the arm
        # was sent: about 0.03 rad, on the second joint. It lies inside what
        # the human program lets a joint target lie from the arm
        # (ik.max_ahead_rad of configs/human_rebot.yaml), and that inside the
        # Thor's step limit. No later target left the Thor's own behind.
        assert 0.031 < off_the_stops < 0.033 < HUMAN_AHEAD < human.max_step_rad
        targets = hand.targets()
        np.testing.assert_allclose(targets[0], first, atol=1e-12)
        assert np.abs(np.diff(targets, axis=0)).max() <= own + 1e-9
        # The arm stands where the hand stopped: 10 cm up, less what the hand
        # had risen when the squeeze was read.
        start_tip = solver.fk(solver.inside(reading))[:3, 3]
        rise = solver.fk(sent[-1])[:3, 3] - start_tip
        assert rise[:2] == pytest.approx([0.0, 0.0], abs=1e-3)
        assert LIFT_M - 2 * HAND_SPEED * period <= rise[2] <= LIFT_M + 1e-3
        np.testing.assert_allclose(sent[-1], targets[-1], atol=1e-12)
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


@apai
def test_the_same_takeover_is_refused_on_the_rig_without_the_viewers_live_picture(
    robot_rig, clock, monkeypatch, tmp_path
):
    # The rig file as it ships, with human driving switched on, on APAI's
    # driver: without a VIEW_STATUS of the viewer no takeover is taken.
    backend = rs.apai_backend
    on = dataclasses.replace(robot_rig.human, enabled=True)
    profile = rig_profile(tmp_path)
    config = backend.config_for_profile(
        dataclasses.replace(robot_rig, human=on), profile
    )
    arm, sdk = rs.rs_arm(config, profile, clock, monkeypatch)
    checkers = backend.build_apai_checkers(config, profile, clock)
    h = AppHarness(config, arm, checkers, clock, app_options={"driver": "real"})
    try:
        h.start()
        before = h.app.snapshot()
        feed(h, view=False)
        none = (
            "the live view is not fresh: no VIEW_STATUS has arrived; show the "
            "viewer's live picture in a visible browser tab"
        )
        assert human_facts(h)["takeover"] == none
        refused = ask_takeover(h)
        assert (refused.status, refused.reason) == ("refused", none)
        assert h.app.snapshot() == before and not h.app.takeover_open
        assert sdk.arm.sent == [] and sdk.disconnects == 0 and arm.torque_on()
        # With a fresh VIEW_STATUS the same request is taken.
        assert take_over(h).status == "completed"
        assert sdk.arm.sent == []
    finally:
        h.close()


def test_the_rig_human_file_names_the_bounds_the_tests_of_a_person_drive_with():
    # The hands of this file run on the rig's human file itself; these are
    # the values the tests name beside it.
    rig = shipped("human_rebot.yaml")
    assert rig["retarget"]["max_linear_velocity"] == TIP_SPEED > HAND_SPEED
    assert rig["retarget"]["max_angular_velocity"] == TIP_TURN
    assert rig["ik"]["max_ahead_m"] == HAND_AHEAD_M
    assert rig["ik"]["max_ahead_rad"] == HUMAN_AHEAD
    # The share of the Thor's clamp is the human program's own.
    assert human_logic.JOINT_SPEED_SHARE == JOINT_SHARE
    assert human_logic.THOR_SPEED_MARGIN == SPEED_MARGIN
    # A squeeze's first target the Thor would refuse ends the engagement in
    # the human program first: its limit lies below the Thor's, in the rig's
    # robot file and in the file of the headset checks before power, which
    # runs with the same human file.
    for name in ("robot_app_rebot.yaml", "robot_app_human_check.yaml"):
        step = shipped(name)["limits"]["human"]["max_step_rad"]
        assert HUMAN_AHEAD < step == HUMAN_STEP, name
    # The one-machine demos keep their own pair.
    demo = shipped("robot_app.yaml")["limits"]["human"]["max_step_rad"]
    assert shipped("human.yaml")["ik"]["max_ahead_rad"] == 0.08 < demo == 0.1


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
def person_rig(request, robot_rig, clock, monkeypatch, tmp_path) -> Iterator[PersonRig]:
    """The rig file's robot app at rest, started, with human driving switched on.

    Once on the simulated arm with its checkers, which read exactly what was
    sent and know no kinematics, and once on APAI's driver of the rebot_rs
    profile over the stand-in motor SDK, with APAI's checkers. The app is
    built without a new arm port for "motors on again", so a typed rest ends
    at rest with the motors on, as it does on the rig while a takeover is
    open; with nobody owning the arm the rig's typed rest also closes the
    jaws and switches the motors off.
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
        profile = rig_profile(tmp_path)
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
# rad a hand's target runs ahead of the last joint target sent: inside the
# Thor's step limit.
STRIDE = 0.04


class JointHand:
    """A hand that squeezes toward a pose: keep_human_fed's setpoint.

    Its target runs at most `stride` per joint ahead of the last joint target
    the motors got. The grip is released once they got the goal.
    """

    def __init__(self, rig: PersonRig, stride: float = STRIDE) -> None:
        self.rig = rig
        self.stride = stride
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
        stride = self.stride
        return [
            at + max(-stride, min(stride, to - at)) for at, to in zip(last, self.goal)
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
    # This hand runs ahead of the arm: the Thor clamps each step at 0.98 of
    # the owner's 0.8 rad/s for a person.
    steps = np.abs(np.diff(np.array([TIP_LOW, *lift]), axis=0))
    clamp = SPEED_MARGIN * HUMAN_SPEED * h.config.executor.period_s
    assert steps.max() == pytest.approx(clamp, abs=1e-12)
    heights = [human.tip_state(q, None).z for q in lift]
    assert all(b > a for a, b in zip([low.z, *heights], heights))
    assert heights[-1] > clearance and human.tip_state(lift[-1], None).low is None
    assert len(h.heard("human")) == 1  # the refused squeeze's, none since
    # With the tip clear the typed rest is taken; the takeover stays open.
    assert typed_rest(h) == "rest: completed"
    assert h.app.motion_state is MotionState.REST_HOLD
    assert sent()[-1] == list(h.config.rest.rest_q)
    assert h.arm.torque_on() and h.app.takeover_open


# The step limit of the rig file: limits.human.max_step_rad. A squeeze starts a
# human drive. Its first target is held against the measured pose and every
# later one against the last joint target sent; one beyond the limit ends the
# drive, refused, and the arm holds.

# The home pose of APAI's rebot_rs profile: the arm raised over the table.
READY = [0.0, 1.2, 1.2, 0.0, 0.0, 0.0]
ONE_JOINT, EVERY_JOINT = (0,), (0, 1, 2, 3, 4, 5)
# Two more poses of the arm model: stretched forward, and folded near rest.
STRETCHED = [0.0, 1.5, 0.6, 0.0, 0.3, 0.0]
NEAR_REST = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]
by_joints = pytest.mark.parametrize(
    "joints", [ONE_JOINT, EVERY_JOINT], ids=["one joint", "every joint"]
)


def away(joints: Sequence[int], by: float) -> list[float]:
    """READY with these joints moved on by `by` rad."""
    return [q + (by if joint in joints else 0.0) for joint, q in enumerate(READY)]


def squeezing_at_ready(
    rig: PersonRig, goal: Sequence[float], stride: float
) -> tuple[JointHand, list[Any]]:
    """A person takes the arm over at READY and squeezes toward goal.

    The hand's target runs `stride` ahead of the last joint target sent.
    Returns the hand and how each human drive ended.
    """
    rig.place(READY)
    take_over(rig.h)
    ended = drives(rig.h)
    hand = JointHand(rig, stride)
    hand.squeeze(READY, goal)
    return hand, ended


@by_joints
def test_the_first_target_of_a_squeeze_is_taken_just_inside_the_step_limit(
    person_rig, joints
):
    # The largest jump a person can ask of the arm at a squeeze: 0.05 rad, 2.9
    # degrees, on every joint at once. On APAI's rebot_rs model that moves the
    # gripper tip up to 7.3 cm from READY, 6.6 cm with the arm stretched
    # forward and 6.1 cm near rest: the test of the tip's travel below.
    h, _, sent = person_rig
    goal = away(joints, 0.049)
    _, ended = squeezing_at_ready(person_rig, goal, stride=1.0)
    targets = sent()
    # Taken in four setpoints, 0.08 s: three at the speed clamp and the rest.
    assert [(end.status, end.sent) for end in ended] == [("released", 4)]
    assert targets[-1] == pytest.approx(goal, abs=1e-12)
    clamp = SPEED_MARGIN * HUMAN_SPEED * h.config.executor.period_s
    steps = np.abs(np.diff(np.array([READY, *targets]), axis=0))
    assert steps.max() == pytest.approx(clamp, abs=1e-12)
    assert steps[:3, list(joints)] == pytest.approx(clamp, abs=1e-12)
    assert h.heard("human") == [] and h.arm.torque_on()


@by_joints
def test_a_first_target_beyond_the_step_limit_is_refused_and_the_arm_holds(
    person_rig, joints
):
    h, _, sent = person_rig
    hand, ended = squeezing_at_ready(person_rig, away(joints, 0.051), stride=1.0)
    assert [(end.status, end.sent) for end in ended] == [("refused", 0)]
    assert re.fullmatch(
        r"the target is 0\.051 rad from the arm on joint index \d, "
        r"the limit is 0\.05 rad",
        ended[0].reason,
    )
    # Refused, not clipped: nothing was sent, and the arm holds with its
    # motors on under the person.
    assert sent() == [] and h.arm.torque_on()
    assert h.app.snapshot().owner is Owner.HUMAN
    assert h.app.motion_state is MotionState.HOLD
    assert len(h.heard("human")) == 1
    # A new squeeze with a nearer target is taken.
    near = away(joints, 0.03)
    hand.squeeze(READY, near)
    assert [(end.status, end.sent) for end in ended[1:]] == [("released", 2)]
    assert sent()[-1] == pytest.approx(near, abs=1e-12)


def test_a_target_that_stays_inside_the_step_limit_is_followed_at_the_speed_clamp(
    person_rig,
):
    # The step limit bounds the jump, not the travel. A target that keeps
    # 0.049 rad ahead of the last joint target sent is followed at the Thor's
    # clamp, 0.98 of the owner's 0.8 rad/s: 0.7 rad in 0.9 s. The human
    # program asks for a little less: its joint target moves at 0.9 of that
    # clamp and stays one such step ahead, as the tests below hold.
    h, _, sent = person_rig
    goal = away(ONE_JOINT, 0.7)
    _, ended = squeezing_at_ready(person_rig, goal, stride=0.049)
    targets = sent()
    assert [(end.status, end.sent) for end in ended] == [("released", 45)]
    assert targets[-1] == pytest.approx(goal, abs=1e-9)
    clamp = SPEED_MARGIN * HUMAN_SPEED * h.config.executor.period_s
    steps = np.abs(np.diff(np.array([READY, *targets]), axis=0))[:, 0]
    assert steps[:-1] == pytest.approx(clamp, abs=1e-12) and steps[-1] <= clamp
    assert h.heard("human") == []


# The human program on the rig's robot app: the human program's own graph,
# solver and logic on configs/human_rebot.yaml make the joint targets, from
# READY, on the fake clock. The hand stays inside the workspace box of APAI's
# rebot_rs profile, which APAI's human checker holds the gripper tip to: from
# READY that is 4 cm forward, 11 cm up and 30 cm to each side. The Thor reads
# the arm for 0.06 s before a drive's first setpoint, and this hand makes no
# pass meanwhile, so it is no faster than 50 cm a second: the rate limiter
# would take a longer step in one pass for a hand it rejects.

by_way = pytest.mark.parametrize(
    "way", [HAND_UP, HAND_BACK, HAND_LEFT], ids=["up", "back", "left"]
)


def tip_way(way: Sequence[float]) -> np.ndarray:
    """The way the gripper tip goes, in the robot's base axes, for a hand move."""
    x, y, z = way
    return np.array([-z, -x, y])


def program_at_ready(
    rig: PersonRig, rows: Sequence[dict[str, Any]], run: bool = True
) -> tuple[ProgramHand, list[Any]]:
    """A person takes the arm over at READY; the human program runs a script.

    Returns its hand and how each human drive ended.
    """
    rig.place(READY)
    take_over(rig.h)
    ended = drives(rig.h)
    hand = ProgramHand(rig.h, rows)
    if run:
        hand.run()
    return hand, ended


def clamp_of(h: AppHarness) -> float:
    """The farthest the Thor moves a person's joint with one setpoint, rad."""
    return SPEED_MARGIN * HUMAN_SPEED * h.config.executor.period_s


@by_way
@pytest.mark.parametrize("speed", [0.05, 0.2, 0.5])
def test_the_human_program_drives_the_rig_robot_app_at_any_hand_speed(
    person_rig, way, speed
):
    h, _, sent = person_rig
    metres, period = 0.10, h.config.executor.period_s
    rows = hand_rows(way, metres, speed, hold_s=1.5)
    hand, ended = program_at_ready(person_rig, rows)
    targets = sent()
    # One human drive, ended by the released grip: the Thor refused no target
    # and ended no drive, and the human program ended no engagement.
    assert [(end.status, end.sent) for end in ended] == [("released", len(targets))]
    assert h.heard("human") == [] and hand.ended() == {}
    counts = hand.logic.counts()
    assert (counts["engagements"], counts["released"]) == (1, 1)
    assert counts["needs_release"] == 0
    # No joint of the arm's joint targets was faster than 0.9 of the Thor's
    # clamp, the human program's own bound; the Thor's clamp cut no step.
    clamp = clamp_of(h)
    steps = np.abs(np.diff(np.array([READY, *targets]), axis=0)).max(axis=1)
    assert steps.max() <= JOINT_SHARE * clamp + 1e-9
    assert state_now(h).health["clamped"] == 0
    if speed >= 0.2:
        # A hand faster than the joints allow: the fastest joint ran at it.
        assert steps.max() == pytest.approx(JOINT_SHARE * clamp, abs=1e-9)
        assert int(np.sum(steps > JOINT_SHARE * clamp - 1e-9)) > 15
    # The arm stands where the hand stopped: its way, less what the hand had
    # gone when the squeeze was read.
    solver = hand.solver
    moved = solver.fk(targets[-1])[:3, 3] - solver.fk(solver.inside(READY))[:3, 3]
    along = float(moved @ tip_way(way))
    assert metres - 2 * speed * period - 1e-3 <= along <= metres + 1e-3
    assert np.linalg.norm(moved - along * tip_way(way)) < 1e-3
    np.testing.assert_allclose(targets[-1], hand.targets()[-1], atol=1e-12)
    assert h.arm.torque_on() and h.app.snapshot().owner is Owner.HUMAN


@pytest.mark.parametrize("stop", ["a released grip", "right B"])
def test_the_arm_gets_no_setpoint_after_a_release_or_right_b_at_full_speed(
    person_rig, stop
):
    h, _, sent = person_rig
    period = h.config.executor.period_s
    # The hand goes 25 cm to the left in 0.5 s; 0.4 s after the squeeze it
    # lets go of the grip, or presses right B, with the arm on its way at
    # full speed and the hand ahead of it.
    rows = hand_rows(HAND_LEFT, 0.25, 0.5)
    change = {"squeeze": 0.0} if stop == "a released grip" else {"b": True}
    rows.insert(2, {"t": 0.44, "pos": [-0.20, *HAND_AT[1:]], **change})
    hand, ended = program_at_ready(person_rig, rows)
    targets = sent()
    clamp = clamp_of(h)
    steps = np.abs(np.diff(np.array([READY, *targets]), axis=0)).max(axis=1)
    assert steps[-1] == pytest.approx(JOINT_SHARE * clamp, abs=1e-9)
    # The pass that read it made the last HUMAN_SETPOINT with a joint target
    # before it; the arm was sent that target, or the one before, and none
    # after: it stands within one step of the Thor, 0.9 degrees a joint.
    last = max(index for index, made in enumerate(hand.sent) if made.engaged)
    assert not hand.sent[last + 1].engaged
    asked = hand.targets()
    behind = np.abs(np.array(targets[-1]) - asked[-1]).max()
    assert behind <= JOINT_SHARE * clamp + 1e-9 < math.radians(0.9)
    assert h.arm.torque_on() and h.app.motion_state is MotionState.HOLD
    # It stands short of the hand's 20 cm: it did not go on.
    solver = hand.solver
    travelled = solver.fk(targets[-1])[:3, 3] - solver.fk(READY)[:3, 3]
    assert 0.02 < float(np.linalg.norm(travelled)) < 0.15
    if stop == "a released grip":
        assert [(end.status, end.sent) for end in ended] == [("released", len(targets))]
        assert h.heard("human") == [] and h.app.snapshot().owner is Owner.HUMAN
        assert hand.logic.counts()["released"] == 1
    else:
        assert [(end.status, end.sent) for end in ended] == [("stopped", len(targets))]
        now = h.app.snapshot()
        assert now.owner is Owner.NOBODY and now.stop_level is StopLevel.STOP
        assert [event[2]["sender"] for event in h.heard("stop")] == ["controller"]
    assert len(sent()) == len(targets) and period == 0.02


@pytest.mark.parametrize(
    "person_rig", [pytest.param("apai", marks=apai)], indirect=True
)
def test_a_joint_that_is_blocked_ends_the_drive_as_a_contact_within_a_fifth_of_a_second(
    person_rig,
):
    # Only APAI's contact watch reads whether the arm follows: the simulated
    # arm's checkers have none. The first joint stops where it is after 20
    # setpoints of a move to the right, which turns it at full speed.
    h, _, sent = person_rig
    sdk = h.arm.backend._arm
    hand_right = tuple(-step for step in HAND_LEFT)
    rows = hand_rows(hand_right, 0.25, 0.5, hold_s=2.0)
    hand, ended = program_at_ready(person_rig, rows, run=False)
    blocked: list[int] = []

    def block() -> None:
        if len(sent()) >= 20 and not blocked:
            sdk.arm.caps["motor1"] = sdk.arm._mm["motor1"].value
            blocked.append(len(sent()))

    hand.on_pass = block
    hand.run()
    targets = sent()
    # The Thor ended the drive at the contact and holds at the measured pose:
    # APAI's contact watch saw the joint stand for 0.12 s while it was told
    # to move. That is 6 setpoints, and one more for the hold.
    [end] = ended
    assert end.status == "contact" and "something is blocking the arm" in end.reason
    after = len(targets) - blocked[0]
    assert after == 7
    held = sdk.arm.caps["motor1"]
    assert targets[-1][0] == pytest.approx(held, abs=1e-9)
    # The hold put the Thor's own joint target back to where the arm is. The
    # human program's lead saw its joint target that far ahead of an arm that
    # does not follow, and ended the engagement on its next pass.
    assert hand.ended() == {"solution_ahead": 1}
    assert hand.logic.counts()["needs_release"] == 0
    ahead = float(abs(hand.targets()[-1, 0] - held))
    assert 0.07 < ahead < 0.11  # 7 passes at 0.71 rad/s
    last = max(index for index, made in enumerate(hand.sent) if made.engaged)
    said = hand.sent[last + 1].note
    assert said.startswith("release the grip and squeeze again: the solution is 0.0")
    assert said.endswith(" rad ahead of the arm")
    assert not any(made.engaged for made in hand.sent[last + 1 :])
    assert len(h.heard("human")) == 1 and h.arm.torque_on()
    assert h.app.snapshot().owner is Owner.HUMAN
    print(f"\nblocked: {after} setpoints to the hold; target {ahead:.3f} rad ahead")


@apai
def test_the_largest_jump_at_a_squeeze_moves_the_gripper_tip_by_centimetres(
    robot_rig,
):
    kinematics = rs.apai_backend.load_apai_profile("rebot_rs").kinematics
    step = robot_rig.limits.human.max_step_rad

    def travel(pose: Sequence[float]) -> float:
        """The farthest the tip goes with every joint `step` away, either way."""
        at = kinematics.fk(np.array(pose))[:3, 3]
        return max(
            float(np.linalg.norm(kinematics.fk(np.array(pose) + by)[:3, 3] - at))
            for by in itertools.product((-step, step), repeat=len(pose))
        )

    reached = [travel(pose) for pose in (READY, STRETCHED, NEAR_REST)]
    assert reached == pytest.approx([0.0728, 0.0656, 0.0608], abs=5e-4)
    # The rig file's comment names the largest jump.
    assert math.degrees(step) == pytest.approx(2.9, abs=0.05)
    comment = " ".join(ROBOT_RIG.read_text(encoding="utf-8").replace("#", " ").split())
    assert (
        "the largest jump at a squeeze is 2.9 degrees a joint: 7.3 cm at the "
        "gripper tip in the ready pose, up to 9 cm with the arm reaching out, "
        "taken in 0.08 s"
    ) in comment


# agent_rebot.yaml


def test_the_two_rig_files_agree_on_speed_waits_profile_and_calibration(
    robot_rig, path_variables
):
    settings = agent_config.load_settings(AGENT_RIG)
    thor = settings.thor
    assert thor.first_wait_s == robot_rig.permits.first_wait_s == 0.5
    assert thor.settle_timeout_s == robot_rig.executor.settle_timeout_s == 2.0
    # The agent's copy of the Thor's speed limit: its tasks are planned at the
    # experiment launcher's speed and at most at this one, the owner's.
    agent_speed = robot_rig.limits.agent.max_joint_speed
    assert thor.agent_max_joint_speed == agent_speed == OWNER_SPEED
    assert settings.task.timed_return is False and robot_rig.rest.auto_return is False
    link = settings.link
    assert link.expected_profile_id == robot_rig.profile_id == PROFILE_ID
    # The name is made from the two files' hashes; a robot app must name one.
    assert link.calibration_id == "" and link.allow_unnamed_calibration is False
    assert robot_rig.camera_calibration_id
    # The arm profile is the experiment's own: the file holds no copy of it.
    assert settings.rig == agent_config.RigSettings()
    assert "rig" not in shipped("agent_rebot.yaml")


@pytest.mark.experiment
def test_the_rig_agent_file_takes_its_arm_profile_from_the_experiment(
    robot_rig, experiment_dir: Path
):
    arm = agent_config.experiment_arm_profile(experiment_dir)
    theirs = yaml.safe_load(
        (experiment_dir / "configs/thor_operation.yaml").read_text()
    )
    assert arm["home_q"] == theirs["home_q"]
    assert arm["motion_bounds"] == theirs["motion_bounds"]
    assert tuple(arm["park_q"]) == robot_rig.rest.rest_q
    # The agent's bound on the joints is not wider than the Thor's working box.
    bound = arm["motion_bounds"]["max_joint_deviation_rad"]
    agent = robot_rig.limits.agent
    assert all(bound <= high or high < 2.0944 for high in agent.envelope_max)
    assert bound <= 2.0944


def test_the_rig_agent_file_is_the_real_run(path_variables):
    settings = agent_config.load_settings(AGENT_RIG)
    camera, experiment = settings.camera, settings.experiment
    folder = path_variables[agent_config.EXPERIMENT_DIR_ENV]
    assert camera.calibration_dir == f"{folder}/.state/thor/calibration"
    # The fit of 2026-11-17.
    assert camera.hand_eye_json == (
        f"{folder}/.state/thor/mounting/20261118T015157Z-live-1rrfshus/handeye.json"
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
        # The copy of the rig robot file's agent speed limit.
        ("thor", "agent_max_joint_speed"),
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
    # The file ships with an empty list. On a rig machine the Thor's address
    # is written in, as the file's comment says: the file as shipped is then
    # that file with the list emptied again.
    text = SERVICE_RIG.read_text(encoding="utf-8")
    lists = re.findall(r"^allowed_peers: \[(.*)\]$", text, re.M)
    assert len(lists) == 1 and "#   allowed_peers: [192.0.2.10]\n" in text
    as_shipped = tmp_path / "as_shipped.yaml"
    as_shipped.write_text(
        text.replace(f"\nallowed_peers: [{lists[0]}]\n", "\nallowed_peers: []\n"),
        encoding="utf-8",
    )
    text = as_shipped.read_text(encoding="utf-8")
    assert text.count("\nallowed_peers: []\n") == 1
    with pytest.raises(
        service_config.ConfigError, match="allowed_peers must be a non-empty list"
    ):
        service_config.load_link_service_config(as_shipped)
    # The program ends before it listens anywhere.
    assert service_main.main(["--config", str(as_shipped)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "robot link service: the configuration cannot be used: allowed_peers must "
        "be a non-empty list"
    )
    # With the Thor's address written in, as the file's comment shows.
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
