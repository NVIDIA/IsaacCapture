# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rig files in examples/agentic_remote/configs: robot_app_rebot.yaml,
agent_rebot.yaml and link_service_rebot.yaml.

A rig file holds the rig's own values, as opposed to the shipped demo values.
The tests marked "apai" hold robot_app_rebot.yaml against APAI's rebot_rs arm
profile and run the robot app with it on a stand-in motor SDK.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.link_service import __main__ as service_main
from isaaccapture_examples.agentic_remote.link_service import config as service_config
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import (
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState
from repo_paths import repo_root

import test_apai_backend as rs
from conftest import AppHarness

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
        assert limit_set.version == "rebot-p2-1"
        assert limit_set.max_joint_speed == 0.1
        assert (limit_set.joint_min, limit_set.joint_max) == (BOX_LOW, MODEL_HIGH)
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
