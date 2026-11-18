# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the robot app settings in
isaaccapture_examples.agentic_remote.robot_app.config."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.robot_app.config import (
    JAW_RELEASE_OPEN_FULLY,
    JAW_RELEASE_STEPS,
    JAW_RELEASES,
    MIN_ASSISTANCE_S,
    ConfigError,
    ExecutorConfig,
    GraspConfig,
    GripperConfig,
    HumanConfig,
    JawsConfig,
    Limits,
    LimitSet,
    PermitsConfig,
    RestConfig,
    RobotAppConfig,
    StateConfig,
    WaitsConfig,
    load_config,
    longest_return_setpoints,
)
from isaaccapture_examples.agentic_remote.robot_app.rest import plan_return
from repo_paths import repo_root

CONFIG_YAML = repo_root() / "examples" / "agentic_remote" / "configs" / "robot_app.yaml"

Change = Callable[[dict], None]


def _put(dotted: str, value: object) -> Change:
    """A change that sets the setting at a dotted path."""

    def change(data: dict) -> None:
        *parents, leaf = dotted.split(".")
        for key in parents:
            data = data[key]
        data[leaf] = value

    return change


def _drop(dotted: str) -> Change:
    """A change that removes the setting at a dotted path."""

    def change(data: dict) -> None:
        *parents, leaf = dotted.split(".")
        for key in parents:
            data = data[key]
        del data[leaf]

    return change


def _put_item(dotted: str, index: int, value: object) -> Change:
    """A change that sets one item of the list at a dotted path."""

    def change(data: dict) -> None:
        for key in dotted.split("."):
            data = data[key]
        data[index] = value

    return change


def _load_changed(tmp_path, *changes: Change) -> RobotAppConfig:
    """Load the shipped YAML with changes applied to its settings."""
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    for change in changes:
        change(data)
    path = tmp_path / "robot_app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def _every_joint(value: float, count: int = 6) -> list[float]:
    return [value] * count


def test_executor_defaults():
    assert ExecutorConfig() == ExecutorConfig(
        period_s=0.02,
        max_setpoints=3000,
        small_delay_s=0.005,
        large_delay_s=0.1,
        settle_tol_rad=0.03,
        settle_timeout_s=2.0,
        reads_per_step=2,
        max_start_tol_rad=0.05,
    )


def test_permits_defaults():
    assert PermitsConfig() == PermitsConfig(
        evidence_age_limit_s=3.0, required_for_agent=True, first_wait_s=0.5
    )


def test_waits_defaults():
    waits = WaitsConfig()
    assert waits.assistance_s == 120.0
    assert waits.bell_at_s == (30.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0)


def test_gripper_defaults():
    assert GripperConfig() == GripperConfig(
        closed_frac=0.04,
        target_tol=0.03,
        reach_tol=0.03,
        full_open_tol=0.03,
        agree_tol=0.02,
        reads=3,
        read_gap_s=0.05,
        open_step=0.05,
        follow_open_effort=0.8,
        reference_at_start=True,
    )


def test_grasp_defaults():
    assert GraspConfig() == GraspConfig(max_radius_m=0.3, max_depth_m=0.06)
    assert RobotAppConfig.default().grasp == GraspConfig()


class Region:
    """What GraspConfig.problem reads of a grasp region."""

    def __init__(self, center_xy, radius_m, z_min, z_max=None) -> None:
        self.center_xy, self.radius_m = center_xy, radius_m
        self.z_min, self.z_max = z_min, z_max


@pytest.mark.parametrize(
    ("region", "problem"),
    [
        (Region((0.3, 0.0), 0.3, 0.94), None),  # table_z is 1.0 below
        (Region((0.5, -0.3), 0.05, 1.0, 1.0), None),
        (Region((0.3, 0.0), 0.301, 0.0), "radius is 0.301 m, the limit is 0.3 m"),
        (Region((0.3, 0.0), 0.05, 0.93), "reaches 0.07 m below the table, the limit"),
        (Region((0.3, 0.0), 0.05, 1.0, 0.99), "ends at z = 0.99 m, below its lowest"),
        (Region((0.3, 0.0), 0.0, 1.0), "needs a radius above 0"),
        (Region((0.3, 0.0), -1.0, 1.0), "needs a radius above 0"),
        (Region((0.3, math.nan), 0.05, 1.0), "holds a value that is not a number"),
        (Region((0.3, 0.0), math.inf, 1.0), "holds a value that is not a number"),
        (Region((0.3, 0.0), 0.05, 1.0, math.inf), "holds a value that is not a number"),
        (Region((0.3,), 0.05, 1.0), "holds a value that is not a number"),
        (Region((0.3, 0.0), "wide", 1.0), "holds a value that is not a number"),
        (Region(None, 0.05, 1.0), "holds a value that is not a number"),
        (Region((0.6, 0.0), 0.05, 1.0), "centre (0.6, 0) m is outside the workspace"),
        (Region((0.3, -0.4), 0.05, 1.0), "centre (0.3, -0.4) m is outside the work"),
    ],
)
def test_a_grasp_region_is_held_to_the_grasp_settings(region, problem):
    box = ((0.1, -0.3, -0.01), (0.5, 0.3, 0.55))
    found = GraspConfig().problem(region, table_z=1.0, box=box)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


def test_without_a_workspace_box_the_centre_of_a_grasp_region_is_free():
    assert GraspConfig().problem(Region((50.0, 50.0), 0.05, 0.0)) is None


def test_state_defaults():
    assert StateConfig() == StateConfig(
        publish_period_s=0.05,
        recent_window_s=0.5,
        history_s=6.0,
        idle_read_period_s=0.02,
    )


def test_human_defaults():
    assert HumanConfig() == HumanConfig(
        enabled=False,
        require_live_view=True,
        view_age_limit_s=0.5,
        sample_age_limit_s=0.25,
        stage_keep_s=4.0,
    )


def test_the_human_settings_ship_in_the_safe_position():
    # Human driving off, and the live view required: the shipped default.
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))["human"]
    assert data["enabled"] is False and data["require_live_view"] is True
    shipped = load_config(CONFIG_YAML).human
    assert (shipped.enabled, shipped.require_live_view) == (False, True)
    # A file that names neither gets the same.
    assert (HumanConfig().enabled, HumanConfig().require_live_view) == (False, True)


def test_a_file_without_jaw_release_opens_the_jaws_fully_when_the_trigger_is_let_go(
    tmp_path,
):
    # human.jaw_release: what the jaws do when a person lets the trigger go.
    assert JAW_RELEASES == (JAW_RELEASE_OPEN_FULLY, JAW_RELEASE_STEPS)
    assert HumanConfig().jaw_release == JAW_RELEASE_OPEN_FULLY
    loaded = _load_changed(tmp_path, _drop("human.jaw_release"))
    assert loaded.human.jaw_release == JAW_RELEASE_OPEN_FULLY


@pytest.mark.parametrize("value", JAW_RELEASES)
def test_each_named_jaw_release_is_loaded(tmp_path, value):
    loaded = _load_changed(tmp_path, _put("human.jaw_release", value))
    assert loaded.human.jaw_release == value
    assert HumanConfig(jaw_release=value).jaw_release == value


@pytest.mark.parametrize(
    "value", ["fully", "Steps", "open fully", "", True, 1, None, ["steps"]]
)
def test_a_jaw_release_that_is_not_one_of_its_names_is_refused(tmp_path, value):
    allowed = ", ".join(JAW_RELEASES)
    refusal = rf"human\.jaw_release must be one of {allowed}, got "
    with pytest.raises(ConfigError, match=refusal):
        _load_changed(tmp_path, _put("human.jaw_release", value))
    with pytest.raises(ConfigError, match=refusal):
        HumanConfig(jaw_release=value)


@pytest.mark.parametrize("name", ["robot_app.yaml", "robot_app_rebot.yaml"])
def test_the_robot_app_files_name_jaw_release_and_say_what_each_value_does(name):
    path = CONFIG_YAML.with_name(name)
    assert load_config(path).human.jaw_release in JAW_RELEASES
    lines = path.read_text(encoding="utf-8").splitlines()
    key = next(i for i, line in enumerate(lines) if line.startswith("  jaw_release:"))
    first = key
    while lines[first - 1].lstrip().startswith("#"):
        first -= 1
    comment = lines[first:key]
    for value in JAW_RELEASES:
        assert any(line.startswith(f"  #   {value} ") for line in comment), value


@pytest.mark.parametrize("value", [0.05, 0.6, 1, 1.0])
def test_an_effort_of_the_opening_under_the_trigger_is_loaded(tmp_path, value):
    # gripper.follow_open_effort: read with human.jaw_release "open_fully".
    loaded = _load_changed(tmp_path, _put("gripper.follow_open_effort", value))
    effort = loaded.gripper.follow_open_effort
    assert effort == value and isinstance(effort, float)
    without = _load_changed(tmp_path, _drop("gripper.follow_open_effort"))
    assert without.gripper.follow_open_effort == GripperConfig().follow_open_effort


@pytest.mark.parametrize(
    ("value", "refusal"),
    [
        (0, "must be above 0 and at most 1, got 0.0"),
        (-0.5, "must be above 0 and at most 1, got -0.5"),
        (1.01, "must be above 0 and at most 1, got 1.01"),
        ("0.8", "must be a number, got '0.8'"),
        (True, "must be a number, got True"),
        (None, "must be a number, got None"),
    ],
)
def test_an_effort_that_is_no_share_above_0_is_refused(tmp_path, value, refusal):
    told = rf"gripper\.follow_open_effort {refusal}"
    with pytest.raises(ConfigError, match=told):
        _load_changed(tmp_path, _put("gripper.follow_open_effort", value))
    with pytest.raises(ConfigError, match=told):
        GripperConfig(follow_open_effort=value)


@pytest.mark.parametrize("name", ["robot_app.yaml", "robot_app_rebot.yaml"])
def test_the_robot_app_files_say_which_jaw_release_reads_each_opening_setting(name):
    path = CONFIG_YAML.with_name(name)
    assert load_config(path).gripper.follow_open_effort > 0.0
    lines = path.read_text(encoding="utf-8").splitlines()
    read_with = {
        "open_step": JAW_RELEASE_STEPS,
        "follow_open_effort": JAW_RELEASE_OPEN_FULLY,
    }
    for key, value in read_with.items():
        at = next(i for i, line in enumerate(lines) if line.startswith(f"  {key}:"))
        first = at
        while lines[first - 1].lstrip().startswith("#"):
            first -= 1
        comment = " ".join(line.strip("# ") for line in lines[first:at])
        assert f'Read only with human.jaw_release "{value}"' in comment, key


def _human_box(config: RobotAppConfig, low, high) -> RobotAppConfig:
    """The configuration with this human joint box."""
    human = dataclasses.replace(
        config.limits.human, joint_min=tuple(low), joint_max=tuple(high)
    )
    return dataclasses.replace(
        config, limits=dataclasses.replace(config.limits, human=human)
    )


def test_human_driving_needs_the_human_box_inside_the_recovery_region(config, tmp_path):
    # A return to rest starts only from inside rest.region_*: a person who
    # could drive beyond it could leave the arm where a typed rest is refused.
    rest = config.rest
    assert config.limits.human.joint_max[0] > rest.region_max[0]
    assert not config.human.enabled  # accepted while human driving is off
    on = HumanConfig(enabled=True)
    with pytest.raises(ConfigError) as refused:
        dataclasses.replace(config, human=on)
    assert str(refused.value) == (
        "human.enabled needs limits.human inside rest.region_*: joint index 0 has "
        "human -3.0..3.0 and region -2.5..2.5; a person could leave the arm where "
        "a typed rest is refused"
    )
    # The loader refuses the same file.
    with pytest.raises(ConfigError, match=r"human\.enabled needs limits\.human"):
        _load_changed(tmp_path, _put("human.enabled", True))
    # With the human box equal to the region it is taken, from a file too.
    inside = _human_box(config, rest.region_min, rest.region_max)
    assert dataclasses.replace(inside, human=on).human.enabled
    loaded = _load_changed(
        tmp_path,
        _put("human.enabled", True),
        _put("limits.human.joint_min", list(rest.region_min)),
        _put("limits.human.joint_max", list(rest.region_max)),
    )
    assert loaded.human.enabled
    # One bound of one joint outside is refused and names that joint.
    low = list(rest.region_min)
    low[4] -= 0.01
    below = _human_box(config, low, rest.region_max)
    assert not below.human.enabled
    with pytest.raises(ConfigError, match=r"joint index 4 has human -2\.51\.\.2\.5 "):
        dataclasses.replace(below, human=on)
    high = list(rest.region_max)
    high[5] += 0.5
    above = _human_box(config, rest.region_min, high)
    with pytest.raises(ConfigError, match=r"joint index 5 has human -2\.5\.\.3\.0 "):
        dataclasses.replace(above, human=on)


def test_the_typed_jaw_reference_is_off_by_default(config, tmp_path):
    assert JawsConfig() == JawsConfig(allow_reference=False)
    assert config.jaws == JawsConfig()
    assert _load_changed(tmp_path, _drop("jaws")).jaws.allow_reference is False
    shipped = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    assert shipped["jaws"] == {"allow_reference": False}


def test_the_typed_jaw_reference_setting_is_loaded(tmp_path):
    config = _load_changed(tmp_path, _put("jaws.allow_reference", True))
    assert config.jaws.allow_reference is True
    assert dataclasses.replace(config, jaws=JawsConfig()) == RobotAppConfig.default()


@pytest.mark.parametrize("value", [1, 0, "true", "yes", None, [True]])
def test_a_jaw_reference_setting_that_is_not_a_flag_is_refused(tmp_path, value):
    with pytest.raises(
        ConfigError, match=r"jaws\.allow_reference must be true or false"
    ):
        _load_changed(tmp_path, _put("jaws.allow_reference", value))


def test_the_jaws_block_takes_no_other_key(tmp_path):
    with pytest.raises(ConfigError, match=r"jaws has unknown key\(s\) reference"):
        _load_changed(tmp_path, _put("jaws.reference", True))
    with pytest.raises(ConfigError, match=r"jaws must be a mapping of settings"):
        _load_changed(tmp_path, _put("jaws", True))
    with pytest.raises(ConfigError, match=r"jaws must be a JawsConfig"):
        dataclasses.replace(RobotAppConfig.default(), jaws={"allow_reference": True})


def test_rest_defaults(config):
    assert config.rest == RestConfig(
        rest_q=_every_joint(0.0),
        rest_tol_rad=0.05,
        start_rest_tol_rad=0.05,
        agree_tol_rad=0.02,
        read_gap_s=0.2,
        recovery_speed=0.1,
        min_duration_s=2.0,
        region_min=_every_joint(-2.5),
        region_max=_every_joint(2.5),
        auto_return=True,
        table_clearance_m=0.02,
    )


def test_default_robot(config):
    assert config.robot_id == "sim-arm"
    assert config.profile_id == "sim-6dof"
    assert config.n_joints == 6
    assert config.joint_names == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )


def test_the_camera_calibration_is_unnamed_by_default(config, tmp_path):
    assert config.camera_calibration_id == ""
    left_out = _load_changed(tmp_path, _drop("camera_calibration_id"))
    assert left_out.camera_calibration_id == ""


def test_the_camera_calibration_id_is_loaded(tmp_path):
    config = _load_changed(tmp_path, _put("camera_calibration_id", "rig-2026-11-14"))
    assert config.camera_calibration_id == "rig-2026-11-14"


@pytest.mark.parametrize("value", [7, None, " ", " rig-1", "rig-1\n"])
def test_a_camera_calibration_id_that_is_not_a_plain_name_is_refused(tmp_path, value):
    with pytest.raises(ConfigError, match=r"camera_calibration_id must be"):
        _load_changed(tmp_path, _put("camera_calibration_id", value))


def test_default_limit_sets(config):
    assert config.limits.agent == LimitSet(
        name="agent",
        version="sim-1",
        joint_min=_every_joint(-2.0),
        joint_max=_every_joint(2.0),
        max_joint_speed=0.4,
        max_step_rad=None,
        envelope_min=_every_joint(-1.5),
        envelope_max=_every_joint(1.5),
    )
    assert config.limits.human == LimitSet(
        name="human",
        version="sim-1",
        joint_min=_every_joint(-3.0),
        joint_max=_every_joint(3.0),
        max_joint_speed=1.5,
        max_step_rad=0.1,
        envelope_min=None,
        envelope_max=None,
    )
    assert config.limits.recovery == LimitSet(
        name="recovery",
        version="sim-1",
        joint_min=_every_joint(-3.0),
        joint_max=_every_joint(3.0),
        max_joint_speed=0.1,
        max_step_rad=None,
        envelope_min=None,
        envelope_max=None,
    )


def test_default_blocks_are_the_block_defaults(config):
    assert config.executor == ExecutorConfig()
    assert config.permits == PermitsConfig()
    assert config.waits == WaitsConfig()
    assert config.gripper == GripperConfig()
    assert config.state == StateConfig()
    assert config.human == HumanConfig()
    assert config.jaws == JawsConfig()


def test_default_recovery_accepts_every_pose_the_human_set_accepts(config):
    human, recovery = config.limits.human, config.limits.recovery
    for joint in range(config.n_joints):
        assert recovery.joint_min[joint] <= human.joint_min[joint]
        assert recovery.joint_max[joint] >= human.joint_max[joint]


def test_default_calls_build_equal_separate_objects():
    assert RobotAppConfig.default() == RobotAppConfig.default()
    assert RobotAppConfig.default() is not RobotAppConfig.default()


def test_shipped_yaml_equals_the_defaults():
    assert load_config(CONFIG_YAML) == RobotAppConfig.default()


def test_shipped_yaml_names_every_setting():
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    config = RobotAppConfig.default()
    assert set(data) == {field.name for field in dataclasses.fields(config)}
    for field in dataclasses.fields(config):
        block = getattr(config, field.name)
        if dataclasses.is_dataclass(block) and not isinstance(block, Limits):
            assert set(data[field.name]) == {f.name for f in dataclasses.fields(block)}


def test_load_config_accepts_a_string_path():
    assert load_config(str(CONFIG_YAML)) == RobotAppConfig.default()


@pytest.mark.parametrize(
    ("name", "differs"),
    [
        (
            "robot_app_demo2.yaml",
            {"human.enabled": True, "human.require_live_view": False},
        ),
        (
            "robot_app_human_check.yaml",
            # The speed limit of the rig's file for a person, whose step
            # limit is the shipped file's; the live view stays required, as
            # on the rig.
            {"human.enabled": True, "limits.human.max_joint_speed": 1.6},
        ),
    ],
)
def test_the_demo_2_robot_files_are_the_shipped_file_with_the_keys_they_list(
    name, differs, tmp_path
):
    # Demo 2: a person takes the arm over. Both files keep the human joint box
    # inside rest.region_*, as human.enabled needs.
    box = {
        "limits.human.joint_min": _every_joint(-2.5),
        "limits.human.joint_max": _every_joint(2.5),
    }
    changes = [_put(key, value) for key, value in {**differs, **box}.items()]
    path = CONFIG_YAML.with_name(name)
    # The file names every setting the shipped file names, with these values.
    settings = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    for change in changes:
        change(settings)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == settings
    loaded = load_config(path)
    assert loaded == _load_changed(tmp_path, *changes)
    assert loaded.human.enabled is True
    # Only the one-machine recipe's file asks for no live view.
    asked = differs.get("human.require_live_view", True)
    assert loaded.human.require_live_view is asked
    assert asked is (name == "robot_app_human_check.yaml")
    if "limits.human.max_joint_speed" in differs:
        rig = load_config(CONFIG_YAML.with_name("robot_app_rebot.yaml"))
        speed = loaded.limits.human.max_joint_speed
        assert speed == rig.limits.human.max_joint_speed
        assert loaded.limits.human.max_step_rad == rig.limits.human.max_step_rad


def test_settings_left_out_take_their_defaults(tmp_path):
    optional_blocks = (
        "executor",
        "permits",
        "waits",
        "gripper",
        "grasp",
        "state",
        "human",
        "jaws",
    )
    optional_rest = (
        "rest_tol_rad",
        "start_rest_tol_rad",
        "agree_tol_rad",
        "read_gap_s",
        "recovery_speed",
        "min_duration_s",
        "auto_return",
        "table_clearance_m",
    )
    changes = [_drop(name) for name in optional_blocks]
    changes += [_drop(f"rest.{name}") for name in optional_rest]
    changes += [_drop("limits.agent.envelope_min"), _drop("limits.agent.envelope_max")]
    config = _load_changed(tmp_path, *changes)
    default = RobotAppConfig.default()
    assert config.limits.agent.envelope_min is None
    assert config.limits.agent.envelope_max is None
    assert dataclasses.replace(config, limits=default.limits) == default


def test_whole_numbers_load_as_floats(tmp_path):
    config = _load_changed(
        tmp_path, _put("waits.assistance_s", 150), _put("rest.read_gap_s", 1)
    )
    assert config.waits.assistance_s == 150.0
    assert isinstance(config.waits.assistance_s, float)
    assert isinstance(config.rest.read_gap_s, float)


def test_a_start_rest_tolerance_wider_than_the_release_tolerance_loads(tmp_path):
    config = _load_changed(tmp_path, _put("rest.start_rest_tol_rad", 0.12))
    assert (config.rest.start_rest_tol_rad, config.rest.rest_tol_rad) == (0.12, 0.05)


def test_the_shortest_assistance_wait_loads(tmp_path):
    config = _load_changed(tmp_path, _put("waits.assistance_s", MIN_ASSISTANCE_S))
    assert config.waits.assistance_s == 90.0


def test_config_is_frozen(config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_joints = 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.waits.assistance_s = 10.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.limits.agent.max_joint_speed = 9.0
    with pytest.raises(TypeError):
        config.rest.rest_q[0] = 1.0


def test_lists_become_tuples_of_floats():
    limit_set = LimitSet(
        name="agent",
        version="v1",
        joint_min=[-1, -1],
        joint_max=np.array([1.0, 1.0]),
        max_joint_speed=1,
    )
    assert limit_set.joint_min == (-1.0, -1.0)
    assert limit_set.joint_max == (1.0, 1.0)
    assert all(isinstance(value, float) for value in limit_set.joint_min)
    assert isinstance(limit_set.max_joint_speed, float)
    assert limit_set == LimitSet(
        name="agent",
        version="v1",
        joint_min=(-1.0, -1.0),
        joint_max=(1.0, 1.0),
        max_joint_speed=1.0,
    )


def test_the_longest_return_to_rest_fits_in_one_move(config):
    # 2.5 rad at a peak of 0.09 rad/s: 52.1 s of 20 ms setpoints.
    assert longest_return_setpoints(config) == 2605
    assert longest_return_setpoints(config) <= config.executor.max_setpoints


@pytest.mark.parametrize(
    ("rest_q", "low", "high", "speed"),
    [
        (0.0, -2.5, 2.5, 0.1),
        (0.4, -1.0, 2.0, 0.1),
        (-0.3, -3.0, 0.5, 0.05),
        (0.0, -0.01, 0.01, 0.1),  # the shortest duration decides
    ],
)
def test_the_longest_return_is_counted_as_rest_plans_it(
    config, rest_q, low, high, speed
):
    rest = dataclasses.replace(
        config.rest,
        rest_q=_every_joint(rest_q),
        region_min=_every_joint(low),
        region_max=_every_joint(high),
        recovery_speed=speed,
    )
    changed = dataclasses.replace(config, rest=rest)
    far = low if rest_q - low > high - rest_q else high
    for corner in (_every_joint(low), _every_joint(high)):
        setpoints, _ = plan_return(
            corner, rest.rest_q, 0.02, speed, rest.min_duration_s
        )
        assert len(setpoints) <= longest_return_setpoints(changed)
    longest, _ = plan_return(
        _every_joint(far), rest.rest_q, 0.02, speed, rest.min_duration_s
    )
    assert len(longest) == longest_return_setpoints(changed)


def test_a_file_whose_longest_return_just_fits_loads(tmp_path):
    loaded = _load_changed(tmp_path, _put("executor.max_setpoints", 2605))
    assert loaded.executor.max_setpoints == 2605


def test_only_a_loaded_file_is_held_to_the_longest_return(config):
    # Tests build short move limits in code; the return is then refused when asked.
    short = dataclasses.replace(config.executor, max_setpoints=10)
    assert dataclasses.replace(config, executor=short).executor.max_setpoints == 10


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_changed_copies_are_checked_too(config):
    with pytest.raises(ConfigError, match=r"waits\.assistance_s must be at least 90 s"):
        dataclasses.replace(config.waits, assistance_s=60.0)
    with pytest.raises(
        ConfigError, match=r"joint_names has 6 values but n_joints is 5"
    ):
        dataclasses.replace(config, n_joints=5)
    with pytest.raises(ConfigError, match=r"rest must be a RestConfig"):
        dataclasses.replace(config, rest={"rest_q": [0.0] * 6})


def test_a_person_keeps_the_profiles_table_and_tip_box_unless_the_file_names_them(
    config, tmp_path
):
    # limits.human.table_clearance_m is the height over the table plane under
    # which a person's gripper tip may only rise; workspace_min and
    # workspace_max are the corners of the box that tip is held in.
    for limit_set in (config.limits.agent, config.limits.human, config.limits.recovery):
        assert limit_set.table_clearance_m is None
        assert limit_set.workspace_min is None and limit_set.workspace_max is None
    shipped = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    for key in ("table_clearance_m", "workspace_min", "workspace_max"):
        assert key not in shipped["limits"]["human"]
    named = _load_changed(
        tmp_path,
        _put("limits.human.table_clearance_m", 0),
        _put("limits.human.workspace_min", [0.1, -0.5, -0.01]),
        _put("limits.human.workspace_max", [0.55, 0.5, 1]),
    )
    human = named.limits.human
    # 0 is a clearance: the tip may go down to the table plane.
    assert human.table_clearance_m == 0.0
    assert isinstance(human.table_clearance_m, float)
    assert human.workspace_min == (0.1, -0.5, -0.01)
    assert human.workspace_max == (0.55, 0.5, 1.0)
    assert all(isinstance(value, float) for value in human.workspace_max)
    # Nothing else of the configuration follows them.
    plain = dataclasses.replace(
        human, table_clearance_m=None, workspace_min=None, workspace_max=None
    )
    limits = dataclasses.replace(named.limits, human=plain)
    assert dataclasses.replace(named, limits=limits) == config
    assert named.rest.table_clearance_m == 0.02


def test_the_rig_file_gives_a_person_a_table_and_a_tip_box_of_their_own():
    rig = load_config(CONFIG_YAML.with_name("robot_app_rebot.yaml"))
    human = rig.limits.human
    # The owner's values of 2026-11-18 for a person on the rig.
    assert human.max_joint_speed == 1.6 and human.max_step_rad == 0.1
    assert human.table_clearance_m == 0.0
    assert human.workspace_min == (0.10, -0.50, -0.01)
    assert human.workspace_max == (0.55, 0.50, 0.70)
    # The agent and a return to rest keep theirs: the arm profile's box and
    # clearance, and 2 cm above the table before a return to rest.
    for limit_set in (rig.limits.agent, rig.limits.recovery):
        assert limit_set.max_joint_speed == 0.2
        assert limit_set.table_clearance_m is None
        assert limit_set.workspace_min is None and limit_set.workspace_max is None
    assert rig.rest.table_clearance_m == 0.02


def test_limits_reject_a_limit_set_in_the_wrong_slot(config):
    limits = config.limits
    with pytest.raises(
        ConfigError, match=r"limits\.agent holds the limit set named 'recovery'"
    ):
        Limits(agent=limits.recovery, human=limits.human, recovery=limits.recovery)


INVALID = [
    # Unknown keys.
    pytest.param(
        [_put("colour", "red")],
        r"the top level has unknown key\(s\) colour; known keys: robot_id",
        id="unknown-top-level-key",
    ),
    pytest.param(
        [_put("executor.perid_s", 0.02)],
        r"executor has unknown key\(s\) perid_s; known keys: period_s",
        id="unknown-block-key",
    ),
    pytest.param(
        [_put("limits.agent.name", "agent")],
        r"limits\.agent has unknown key\(s\) name",
        id="limit-set-name-is-its-key",
    ),
    pytest.param(
        [_put("limits.visitor", {"version": "v"})],
        r"limits has unknown key\(s\) visitor; known keys: agent, human, recovery",
        id="unknown-limit-set",
    ),
    pytest.param(
        [_put("rest.park_q", [0.0]), _put("rest.home_q", [0.0])],
        r"rest has unknown key\(s\) home_q, park_q",
        id="several-unknown-keys",
    ),
    # Required keys.
    pytest.param(
        [_drop("n_joints")],
        r"the top level lacks required key\(s\) n_joints",
        id="missing-n-joints",
    ),
    pytest.param(
        [_drop("limits")],
        r"the top level lacks required key\(s\) limits",
        id="missing-limits",
    ),
    pytest.param(
        [_drop("limits.recovery")],
        r"limits lacks required key\(s\) recovery",
        id="missing-limit-set",
    ),
    pytest.param(
        [_drop("limits.agent.joint_max")],
        r"limits\.agent lacks required key\(s\) joint_max",
        id="missing-joint-max",
    ),
    pytest.param(
        [_drop("rest.rest_q"), _drop("rest.region_min")],
        r"rest lacks required key\(s\) rest_q, region_min",
        id="missing-rest-pose",
    ),
    # List lengths equal n_joints.
    pytest.param(
        [_put("joint_names", ["a", "b"])],
        r"joint_names has 2 values but n_joints is 6",
        id="joint-names-length",
    ),
    pytest.param(
        [_put("n_joints", 5)],
        r"joint_names has 6 values but n_joints is 5",
        id="n-joints-below-lists",
    ),
    pytest.param(
        [_put("n_joints", 7), _put("joint_names", list("abcdefg"))],
        r"limits\.agent\.joint_min has 6 values but n_joints is 7",
        id="limits-length",
    ),
    pytest.param(
        [_put("limits.agent.joint_min", _every_joint(-2.0, 5))],
        r"limits\.agent\.joint_min has 5 values but limits\.agent\.joint_max has 6",
        id="joint-min-length",
    ),
    pytest.param(
        [
            _put("limits.human.joint_min", _every_joint(-3.0, 5)),
            _put("limits.human.joint_max", _every_joint(3.0, 5)),
        ],
        r"limits\.human has 5 joints but limits\.agent has 6",
        id="one-limit-set-shorter",
    ),
    pytest.param(
        [_put("limits.agent.envelope_max", _every_joint(1.5, 7))],
        r"limits\.agent\.envelope_min has 6 values but "
        r"limits\.agent\.envelope_max has 7",
        id="envelope-max-length",
    ),
    pytest.param(
        [
            _put("limits.agent.envelope_min", _every_joint(-1.5, 5)),
            _put("limits.agent.envelope_max", _every_joint(1.5, 5)),
        ],
        r"limits\.agent\.envelope_min has 5 values but limits\.agent\.joint_min has 6",
        id="envelope-length",
    ),
    pytest.param(
        [_put("rest.rest_q", _every_joint(0.0, 5))],
        r"rest\.rest_q has 5 values but rest\.region_min has 6",
        id="rest-q-length",
    ),
    pytest.param(
        [_put("rest.region_max", _every_joint(2.5, 5))],
        r"rest\.region_min has 6 values but rest\.region_max has 5",
        id="region-length",
    ),
    pytest.param(
        [
            _put("rest.rest_q", _every_joint(0.0, 5)),
            _put("rest.region_min", _every_joint(-2.5, 5)),
            _put("rest.region_max", _every_joint(2.5, 5)),
        ],
        r"rest\.rest_q has 5 values but n_joints is 6",
        id="rest-length",
    ),
    # Min below max.
    pytest.param(
        [_put_item("limits.agent.joint_min", 0, 2.0)],
        r"limits\.agent\.joint_min\[0\] must be below limits\.agent\.joint_max\[0\], "
        r"got 2\.0 and 2\.0",
        id="joint-min-equals-max",
    ),
    pytest.param(
        [_put_item("limits.human.joint_max", 5, -3.5)],
        r"limits\.human\.joint_min\[5\] must be below limits\.human\.joint_max\[5\]",
        id="joint-max-below-min",
    ),
    pytest.param(
        [_put_item("limits.agent.envelope_min", 2, 1.6)],
        r"limits\.agent\.envelope_min\[2\] must be below "
        r"limits\.agent\.envelope_max\[2\]",
        id="envelope-min-above-max",
    ),
    pytest.param(
        [_put_item("rest.region_max", 3, -2.5)],
        r"rest\.region_min\[3\] must be below rest\.region_max\[3\]",
        id="region-min-equals-max",
    ),
    # The assistance wait.
    pytest.param(
        [_put("waits.assistance_s", 89.9)],
        r"waits\.assistance_s must be at least 90 s, got 89\.9",
        id="assistance-wait-too-short",
    ),
    pytest.param(
        [_put("waits.assistance_s", 0)],
        r"waits\.assistance_s must be at least 90 s, got 0\.0",
        id="assistance-wait-zero",
    ),
    # The recovery joint box contains the human one.
    pytest.param(
        [_put_item("limits.recovery.joint_max", 2, 2.9)],
        r"limits\.recovery joint box must contain the limits\.human joint box: "
        r"joint index 2 has human -3\.0\.\.3\.0 and recovery -3\.0\.\.2\.9",
        id="recovery-max-inside-human",
    ),
    pytest.param(
        [_put_item("limits.recovery.joint_min", 0, -2.9)],
        r"limits\.recovery joint box must contain the limits\.human joint box: "
        r"joint index 0",
        id="recovery-min-inside-human",
    ),
    # Speeds are positive.
    pytest.param(
        [_put("limits.agent.max_joint_speed", 0)],
        r"limits\.agent\.max_joint_speed must be above 0, got 0\.0",
        id="agent-speed-zero",
    ),
    pytest.param(
        [_put("limits.human.max_joint_speed", -1.5)],
        r"limits\.human\.max_joint_speed must be above 0, got -1\.5",
        id="human-speed-negative",
    ),
    pytest.param(
        [_put("limits.recovery.max_joint_speed", 0.0)],
        r"limits\.recovery\.max_joint_speed must be above 0",
        id="recovery-speed-zero",
    ),
    pytest.param(
        [_put("rest.recovery_speed", 0.0)],
        r"rest\.recovery_speed must be above 0",
        id="rest-speed-zero",
    ),
    pytest.param(
        [_put("limits.human.max_step_rad", 0)],
        r"limits\.human\.max_step_rad must be above 0",
        id="human-step-zero",
    ),
    pytest.param(
        [_put("rest.recovery_speed", 0.2)],
        r"rest\.recovery_speed must not exceed limits\.recovery\.max_joint_speed, "
        r"got 0\.2 and 0\.1",
        id="rest-speed-above-recovery-limit",
    ),
    # What each control owner's limit set may hold.
    pytest.param(
        [_drop("limits.human.max_step_rad")],
        r"limits\.human\.max_step_rad is required",
        id="human-without-step-clamp",
    ),
    pytest.param(
        [_put("limits.agent.max_step_rad", 0.1)],
        r"limits\.agent\.max_step_rad must not be set",
        id="agent-with-step-clamp",
    ),
    pytest.param(
        [_put("limits.recovery.max_step_rad", 0.1)],
        r"limits\.recovery\.max_step_rad must not be set",
        id="recovery-with-step-clamp",
    ),
    # The table clearance and the tip box of a person's checker.
    pytest.param(
        [_put("limits.human.table_clearance_m", -0.001)],
        r"limits\.human\.table_clearance_m must not be negative, got -0\.001",
        id="human-table-clearance-negative",
    ),
    pytest.param(
        [_put("limits.human.table_clearance_m", "low")],
        r"limits\.human\.table_clearance_m must be a number",
        id="human-table-clearance-not-a-number",
    ),
    pytest.param(
        [_put("limits.agent.table_clearance_m", 0.0)],
        r"limits\.agent\.table_clearance_m must not be set: only the human "
        r"limit set has a table clearance and a tip box of its own",
        id="agent-with-table-clearance",
    ),
    pytest.param(
        [_put("limits.recovery.table_clearance_m", 0.0)],
        r"limits\.recovery\.table_clearance_m must not be set",
        id="recovery-with-table-clearance",
    ),
    pytest.param(
        [_put("limits.human.workspace_min", [0.1, -0.5, -0.01])],
        r"limits\.human\.workspace_min and limits\.human\.workspace_max must be "
        r"given together",
        id="human-tip-box-without-its-highest-corner",
    ),
    pytest.param(
        [_put("limits.human.workspace_max", [0.55, 0.5, 0.7])],
        r"limits\.human\.workspace_min and limits\.human\.workspace_max must be "
        r"given together",
        id="human-tip-box-without-its-lowest-corner",
    ),
    pytest.param(
        [
            _put("limits.human.workspace_min", [0.1, -0.5]),
            _put("limits.human.workspace_max", [0.55, 0.5]),
        ],
        r"limits\.human\.workspace_min must hold 3 values, x, y and z, "
        r"got \[0\.1, -0\.5\]",
        id="human-tip-box-with-two-values",
    ),
    pytest.param(
        [
            _put("limits.human.workspace_min", [0.1, -0.5, -0.01]),
            _put("limits.human.workspace_max", [0.55, 0.5, 0.7, 1.0]),
        ],
        r"limits\.human\.workspace_max must hold 3 values",
        id="human-tip-box-with-four-values",
    ),
    pytest.param(
        [
            _put("limits.human.workspace_min", [0.1, -0.5, 0.7]),
            _put("limits.human.workspace_max", [0.55, 0.5, 0.7]),
        ],
        r"limits\.human\.workspace_min\[2\] must be below "
        r"limits\.human\.workspace_max\[2\], got 0\.7 and 0\.7",
        id="human-tip-box-with-no-height",
    ),
    pytest.param(
        [
            _put("limits.human.workspace_min", [0.1, float("nan"), -0.01]),
            _put("limits.human.workspace_max", [0.55, 0.5, 0.7]),
        ],
        r"limits\.human\.workspace_min\[1\] must be finite",
        id="human-tip-box-not-finite",
    ),
    pytest.param(
        [
            _put("limits.agent.workspace_min", [0.1, -0.5, -0.01]),
            _put("limits.agent.workspace_max", [0.55, 0.5, 0.7]),
        ],
        r"limits\.agent\.workspace_min must not be set",
        id="agent-with-tip-box",
    ),
    pytest.param(
        [
            _put("limits.recovery.workspace_min", [0.1, -0.5, -0.01]),
            _put("limits.recovery.workspace_max", [0.55, 0.5, 0.7]),
        ],
        r"limits\.recovery\.workspace_min must not be set",
        id="recovery-with-tip-box",
    ),
    pytest.param(
        [
            _put("limits.recovery.envelope_min", _every_joint(-1.0)),
            _put("limits.recovery.envelope_max", _every_joint(1.0)),
        ],
        r"limits\.recovery must not have an envelope",
        id="recovery-with-envelope",
    ),
    pytest.param(
        [
            _put("limits.human.envelope_min", _every_joint(-1.0)),
            _put("limits.human.envelope_max", _every_joint(1.0)),
        ],
        r"limits\.human must not have an envelope",
        id="human-with-envelope",
    ),
    pytest.param(
        [_drop("limits.agent.envelope_min")],
        r"limits\.agent\.envelope_min and limits\.agent\.envelope_max must be given "
        r"together",
        id="half-an-envelope",
    ),
    # Kinds of value.
    pytest.param(
        [_put("executor.period_s", "fast")],
        r"executor\.period_s must be a number, got 'fast'",
        id="text-for-a-number",
    ),
    pytest.param(
        [_put("executor.period_s", True)],
        r"executor\.period_s must be a number, got True",
        id="flag-for-a-number",
    ),
    pytest.param(
        [_put("executor.period_s", math.nan)],
        r"executor\.period_s must be finite, got nan",
        id="nan",
    ),
    pytest.param(
        [_put("state.history_s", math.inf)],
        r"state\.history_s must be finite, got inf",
        id="infinity",
    ),
    pytest.param(
        [_put("executor.max_setpoints", 10.5)],
        r"executor\.max_setpoints must be a whole number, got 10\.5",
        id="fraction-for-a-count",
    ),
    pytest.param(
        [_put("gripper.reads", True)],
        r"gripper\.reads must be a whole number, got True",
        id="flag-for-a-count",
    ),
    pytest.param(
        [_put("permits.required_for_agent", "yes")],
        r"permits\.required_for_agent must be true or false, got 'yes'",
        id="text-for-a-flag",
    ),
    pytest.param(
        [_put("rest.auto_return", 1)],
        r"rest\.auto_return must be true or false, got 1",
        id="number-for-a-flag",
    ),
    pytest.param(
        [_put("limits.agent.version", 1)],
        r"limits\.agent\.version must be a non-empty string, got 1",
        id="number-for-a-version",
    ),
    pytest.param(
        [_put("robot_id", " ")],
        r"robot_id must be a non-empty string",
        id="blank-robot-id",
    ),
    pytest.param(
        [_put("profile_id", None)],
        r"profile_id must be a non-empty string, got None",
        id="no-profile-id",
    ),
    pytest.param(
        [_put("n_joints", 0), _put("joint_names", [])],
        r"n_joints must be at least 1, got 0",
        id="no-joints",
    ),
    pytest.param(
        [_put_item("joint_names", 1, "joint1")],
        r"joint_names must not repeat a name",
        id="repeated-joint-name",
    ),
    pytest.param(
        [_put_item("joint_names", 4, 5)],
        r"joint_names\[4\] must be a non-empty string, got 5",
        id="number-for-a-joint-name",
    ),
    pytest.param(
        [_put("rest.rest_q", 0.0)],
        r"rest\.rest_q must be a non-empty list, got 0\.0",
        id="number-for-a-list",
    ),
    pytest.param(
        [_put("joint_names", "joint1")],
        r"joint_names must be a non-empty list, got 'joint1'",
        id="text-for-a-list",
    ),
    pytest.param(
        [_put_item("limits.agent.joint_min", 1, "low")],
        r"limits\.agent\.joint_min\[1\] must be a number, got 'low'",
        id="text-in-a-list",
    ),
    pytest.param(
        [_put("limits", [])],
        r"limits must be a mapping of settings, got \[\]",
        id="list-for-limits",
    ),
    pytest.param(
        [_put("limits.human", "wide")],
        r"limits\.human must be a mapping of settings, got 'wide'",
        id="text-for-a-limit-set",
    ),
    pytest.param(
        [_put("executor", None)],
        r"executor must be a mapping of settings, got None",
        id="empty-block",
    ),
    # Ranges of the other settings.
    pytest.param(
        [_put("executor.period_s", 0.0)],
        r"executor\.period_s must be above 0",
        id="period-zero",
    ),
    pytest.param(
        [_put("executor.max_setpoints", 0)],
        r"executor\.max_setpoints must be at least 1, got 0",
        id="no-setpoints-allowed",
    ),
    pytest.param(
        [_put("executor.reads_per_step", 0)],
        r"executor\.reads_per_step must be at least 1",
        id="no-reads-per-step",
    ),
    pytest.param(
        [_put("executor.max_start_tol_rad", 0.0)],
        r"executor\.max_start_tol_rad must be above 0",
        id="no-start-tolerance-allowed",
    ),
    pytest.param(
        [_put("executor.small_delay_s", -0.001)],
        r"executor\.small_delay_s must not be negative",
        id="negative-small-delay",
    ),
    pytest.param(
        [_put("executor.large_delay_s", 0.005)],
        r"executor\.large_delay_s must be above executor\.small_delay_s",
        id="large-delay-not-above-small",
    ),
    pytest.param(
        [_put("permits.evidence_age_limit_s", 0)],
        r"permits\.evidence_age_limit_s must be above 0",
        id="evidence-age-zero",
    ),
    pytest.param(
        [_put("permits.first_wait_s", -0.1)],
        r"permits\.first_wait_s must not be negative, got -0\.1",
        id="negative-first-permit-wait",
    ),
    pytest.param(
        [_put("waits.bell_at_s", [10.0, 30.0])],
        r"waits\.bell_at_s must run from the largest mark down",
        id="bell-marks-rising",
    ),
    pytest.param(
        [_put("waits.bell_at_s", [10.0, 10.0])],
        r"waits\.bell_at_s must run from the largest mark down",
        id="bell-mark-repeated",
    ),
    pytest.param(
        [_put("waits.bell_at_s", [150.0, 10.0])],
        r"waits\.bell_at_s must lie above 0 and within waits\.assistance_s",
        id="bell-before-the-wait",
    ),
    pytest.param(
        [_put("waits.bell_at_s", [5.0, 0.0])],
        r"waits\.bell_at_s must lie above 0 and within waits\.assistance_s",
        id="bell-at-zero",
    ),
    pytest.param(
        [_put("waits.bell_at_s", [])],
        r"waits\.bell_at_s must be a non-empty list",
        id="no-bell",
    ),
    pytest.param(
        [_put_item("rest.rest_q", 0, 2.7)],
        r"rest\.rest_q\[0\] must lie inside rest\.region_min\.\.region_max, got 2\.7",
        id="rest-outside-region",
    ),
    pytest.param(
        [_put_item("rest.region_max", 0, 3.5), _put_item("rest.rest_q", 0, 3.2)],
        r"rest\.rest_q\[0\] must lie inside the limits\.recovery joint box, got 3\.2",
        id="rest-outside-recovery-limits",
    ),
    pytest.param(
        [_put("rest.rest_tol_rad", 0)],
        r"rest\.rest_tol_rad must be above 0",
        id="rest-tolerance-zero",
    ),
    pytest.param(
        [_put("rest.start_rest_tol_rad", 0)],
        r"rest\.start_rest_tol_rad must be above 0",
        id="start-rest-tolerance-zero",
    ),
    pytest.param(
        [_put("rest.start_rest_tol_rad", 0.04)],
        r"rest\.start_rest_tol_rad must not be below rest\.rest_tol_rad, "
        r"got 0\.04 and 0\.05",
        id="start-rest-tolerance-below-the-release-tolerance",
    ),
    pytest.param(
        [_put("rest.rest_tol_rad", 0.08)],
        r"rest\.start_rest_tol_rad must not be below rest\.rest_tol_rad, "
        r"got 0\.05 and 0\.08",
        id="release-tolerance-above-the-start-rest-tolerance",
    ),
    pytest.param(
        [_put("rest.table_clearance_m", -0.01)],
        r"rest\.table_clearance_m must not be negative",
        id="negative-table-clearance",
    ),
    pytest.param(
        [_put("gripper.closed_frac", 1.5)],
        r"gripper\.closed_frac must be between 0 and 1, got 1\.5",
        id="closed-fraction-above-one",
    ),
    pytest.param(
        [_put("gripper.reach_tol", -0.01)],
        r"gripper\.reach_tol must be between 0 and 1, got -0\.01",
        id="negative-jaw-reach-tolerance",
    ),
    pytest.param(
        [_put("gripper.full_open_tol", 1.5)],
        r"gripper\.full_open_tol must be between 0 and 1, got 1\.5",
        id="full-open-tolerance-above-one",
    ),
    pytest.param(
        [_put("grasp.max_radius_m", 0)],
        r"grasp\.max_radius_m must be above 0, got 0\.0",
        id="grasp-radius-zero",
    ),
    pytest.param(
        [_put("grasp.max_depth_m", -0.01)],
        r"grasp\.max_depth_m must not be negative, got -0\.01",
        id="grasp-depth-negative",
    ),
    pytest.param(
        [_put("grasp.max_height_m", 1.0)],
        r"grasp has unknown key\(s\) max_height_m",
        id="grasp-unknown-key",
    ),
    pytest.param(
        [_put("gripper.open_step", 0)],
        r"gripper\.open_step must be between 0 and 1",
        id="open-step-zero",
    ),
    pytest.param(
        [_put("gripper.reads", 1)],
        r"gripper\.reads must be at least 2, got 1",
        id="one-jaw-read",
    ),
    pytest.param(
        [_put("state.recent_window_s", 7.0)],
        r"state\.recent_window_s must not exceed state\.history_s, got 7\.0 and 6\.0",
        id="recent-window-above-history",
    ),
    pytest.param(
        [_put("state.publish_period_s", -0.05)],
        r"state\.publish_period_s must be above 0",
        id="negative-publish-period",
    ),
    pytest.param(
        [_put("human.view_age_limit_s", 0)],
        r"human\.view_age_limit_s must be above 0",
        id="view-age-zero",
    ),
    # The longest return to rest must fit in one move.
    pytest.param(
        [_put("executor.max_setpoints", 2604)],
        r"executor\.max_setpoints must be at least 2605, got 2604: a return to rest "
        r"from the far end of rest\.region_min\.\.region_max \(largest joint travel "
        r"2\.5 rad at rest\.recovery_speed 0\.1 rad/s\) would be refused",
        id="longest-return-over-the-move-limit",
    ),
    pytest.param(
        [_put_item("rest.region_max", 4, 2.9)],
        r"executor\.max_setpoints must be at least 3021, got 3000",
        id="rest-region-too-wide-for-the-move-limit",
    ),
    pytest.param(
        [_put("rest.recovery_speed", 0.08)],
        r"executor\.max_setpoints must be at least 3256, got 3000",
        id="recovery-speed-too-low-for-the-move-limit",
    ),
    pytest.param(
        [_put("executor.period_s", 0.01)],
        r"executor\.max_setpoints must be at least 5209, got 3000",
        id="period-too-short-for-the-move-limit",
    ),
]


@pytest.mark.parametrize(("changes", "message"), INVALID)
def test_invalid_setting_is_rejected(tmp_path, changes, message):
    with pytest.raises(ConfigError, match=message):
        _load_changed(tmp_path, *changes)


def test_every_block_setting_has_a_rejected_value(tmp_path):
    """No setting of the flat blocks is loaded unchecked."""
    blocks = {
        "executor": ExecutorConfig,
        "permits": PermitsConfig,
        "waits": WaitsConfig,
        "rest": RestConfig,
        "gripper": GripperConfig,
        "state": StateConfig,
        "human": HumanConfig,
        "jaws": JawsConfig,
    }
    for block, kind in blocks.items():
        for field in dataclasses.fields(kind):
            setting = f"{block}.{field.name}"
            with pytest.raises(ConfigError, match=rf"{block}\.{field.name} must be"):
                _load_changed(tmp_path, _put(setting, "wrong"))


def test_error_names_the_file(tmp_path):
    with pytest.raises(ConfigError) as caught:
        _load_changed(tmp_path, _put("waits.assistance_s", 60))
    assert str(caught.value).startswith(f"{tmp_path / 'robot_app.yaml'}: waits.")


@pytest.mark.parametrize("text", ["", "- 1\n- 2\n", "just text\n"])
def test_top_level_must_be_a_mapping(tmp_path, text):
    path = tmp_path / "robot_app.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"the top level must be a mapping"):
        load_config(path)


def test_broken_yaml_is_a_config_error(tmp_path):
    path = tmp_path / "robot_app.yaml"
    path.write_text("limits: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"robot_app\.yaml: not valid YAML"):
        load_config(path)


def test_repeated_key_is_rejected(tmp_path):
    text = CONFIG_YAML.read_text(encoding="utf-8")
    path = tmp_path / "robot_app.yaml"
    path.write_text(text + "\nwaits:\n  assistance_s: 300.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"line \d+: key 'waits' is given twice"):
        load_config(path)


def test_repeated_nested_key_is_rejected(tmp_path):
    path = tmp_path / "robot_app.yaml"
    path.write_text("waits:\n  assistance_s: 300.0\n  assistance_s: 90.0\n", "utf-8")
    with pytest.raises(ConfigError, match=r"line 3: key 'assistance_s' is given twice"):
        load_config(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")
