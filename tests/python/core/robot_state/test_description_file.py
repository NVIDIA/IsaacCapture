# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the robot description file: both example files load, and every mistake is refused in plain words."""

import copy
import pickle

import pytest
from repo_paths import repo_root

from isaaccapture.robot_state import (
    GripperDescription,
    RobotDescription,
    RobotDescriptionError,
    SeriesDescription,
    load_robot_description,
    parse_robot_description,
)
from isaaccapture.robot_state.description import DEFAULT_STALE_AFTER_S

DESCRIPTIONS = (
    repo_root()
    / "examples"
    / "robot_state"
    / "python"
    / "isaaccapture_examples"
    / "robot_state"
    / "descriptions"
)

ARM = """
robot = "arm"

[series.arm]
shape = "joints"
joints = [
  { name = "j1", kind = "revolute", min = -1.0, max = 1.0 },
  { name = "jaw", kind = "gripper", min = 0.0, max = 1.0 },
]

[series.status]
shape = "status"
readings = ["motors_on"]
"""


def refused(text: str) -> str:
    """The reason `text` is refused for, without the source."""
    with pytest.raises(RobotDescriptionError) as caught:
        parse_robot_description(text)
    message = str(caught.value)
    assert message.startswith("<string>: ")
    return message.removeprefix("<string>: ")


def joints_series(joints: str) -> str:
    """A description with one joints series whose joint entries are `joints`."""
    return f'robot = "r"\n[series.arm]\nshape = "joints"\njoints = [{joints}]\n'


class TestSixJointArm:
    @pytest.fixture
    def description(self) -> RobotDescription:
        return load_robot_description(DESCRIPTIONS / "six_joint_arm.toml")

    def test_robot_and_series(self, description):
        assert description.robot == "rebot_b601"
        assert list(description.series) == ["arm", "status"]
        assert description.attributes == {}

    def test_arm_joints_in_file_order(self, description):
        arm = description.series["arm"]

        assert isinstance(arm, SeriesDescription)
        assert arm.shape == "joints"
        assert arm.shape_name == "joints"
        assert arm.joint_names == [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        assert arm.rate_hz == 20.0
        assert arm.stale_after_s == DEFAULT_STALE_AFTER_S == 0.5
        assert arm.has_velocity is False
        assert arm.has_effort is False
        assert arm.values == ()
        assert arm.value_names == []

    def test_arm_limits_and_units(self, description):
        arm = description.series["arm"]

        assert arm.limits["joint2"] == (-0.05, 3.14)
        assert arm.limits["gripper"] == (0.0, 1.0)
        assert len(arm.limits) == 7
        assert [joint.kind for joint in arm.joints] == ["revolute"] * 6 + ["gripper"]
        assert [joint.unit for joint in arm.joints] == ["rad"] * 6 + [""]
        assert arm.joints[0].max_velocity is None
        assert arm.joints[0].max_effort is None

    def test_gripper_comes_from_the_gripper_joint(self, description):
        assert description.grippers == {
            "gripper": GripperDescription(
                name="gripper", series="arm", joints=("gripper",), open=1.0, closed=0.0
            )
        }

    def test_status_readings(self, description):
        status = description.series["status"]

        assert status.shape == "status"
        assert status.value_names == ["motors_on", "fault"]
        assert [value.kind for value in status.values] == ["any", "any"]
        assert status.rate_hz is None
        assert status.joints == ()
        assert status.limits == {}


class TestTwoArmRobot:
    @pytest.fixture
    def description(self) -> RobotDescription:
        return load_robot_description(str(DESCRIPTIONS / "two_arm_robot.toml"))

    def test_robot_and_series(self, description):
        assert description.robot == "two_arm_example"
        assert list(description.series) == [
            "left_arm",
            "right_arm",
            "waist",
            "left_wrist_force",
            "right_wrist_force",
            "status",
        ]

    def test_arms(self, description):
        left = description.series["left_arm"]
        right = description.series["right_arm"]

        assert left.joint_names == [
            "l_shoulder_pan",
            "l_shoulder_lift",
            "l_elbow",
            "l_wrist_roll",
            "l_gripper",
        ]
        assert right.joint_names == [
            name.replace("l_", "r_", 1) for name in left.joint_names
        ]
        assert left.has_effort is True
        assert left.has_velocity is False
        assert left.joints[0].max_velocity == 2.0
        assert left.joints[0].max_effort == 40.0
        assert left.limits["l_elbow"] == (-2.5, 2.5)

    def test_a_continuous_joint_has_no_limits(self, description):
        left = description.series["left_arm"]
        roll = left.joints[3]

        assert roll.kind == "continuous"
        assert roll.min is None
        assert roll.max is None
        assert roll.unit == "rad"
        assert "l_wrist_roll" not in left.limits

    def test_waist(self, description):
        waist = description.series["waist"]

        assert waist.joint_names == ["waist_yaw"]
        assert waist.has_effort is False
        assert waist.limits == {"waist_yaw": (-1.5, 1.5)}

    def test_a_gripper_in_each_arm(self, description):
        assert list(description.grippers) == ["l_gripper", "r_gripper"]
        left = description.grippers["l_gripper"]
        assert left.series == "left_arm"
        assert left.joints == ("l_gripper",)
        assert (left.closed, left.open) == (0.0, 0.08)
        assert description.series["left_arm"].joints[4].unit == "m"
        assert description.grippers["r_gripper"].series == "right_arm"

    def test_a_force_reading_in_each_wrist(self, description):
        for name, wrist in (
            ("left_wrist_force", "l_wrist"),
            ("right_wrist_force", "r_wrist"),
        ):
            force = description.series[name]
            assert force.shape == "custom"
            assert force.shape_name == "force"
            assert force.value_names == ["fx", "fy", "fz", "tx", "ty", "tz"]
            assert [value.kind for value in force.values] == ["number"] * 6
            assert [value.unit for value in force.values] == ["N"] * 3 + ["N*m"] * 3
            assert force.rate_hz == 50.0
            assert force.attributes == {"at": wrist}
            assert force.joints == ()

    def test_status(self, description):
        status = description.series["status"]

        assert status.stale_after_s == 2.0
        assert status.value_names == ["motors_on", "fault", "battery"]
        assert status.values[2].kind == "number"
        assert status.values[2].unit == "V"


class TestLoading:
    def test_a_missing_file_raises_its_os_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_robot_description(tmp_path / "none.toml")

    def test_a_directory_raises_its_os_error(self, tmp_path):
        with pytest.raises(OSError):
            load_robot_description(tmp_path)

    def test_the_text_names_the_file(self, tmp_path):
        path = tmp_path / "robot.toml"
        path.write_text('robot = "r"\n')

        with pytest.raises(RobotDescriptionError) as caught:
            load_robot_description(path)

        assert (
            str(caught.value) == f"{path}: series must declare at least one data series"
        )

    def test_a_file_that_is_not_text(self, tmp_path):
        path = tmp_path / "robot.toml"
        path.write_bytes(b"\xff\xfe\x00")

        with pytest.raises(RobotDescriptionError, match="not valid TOML"):
            load_robot_description(path)

    def test_the_error_is_a_value_error(self):
        assert issubclass(RobotDescriptionError, ValueError)

    def test_the_source_is_named(self):
        with pytest.raises(RobotDescriptionError) as caught:
            parse_robot_description("robot = 5", source="rig.toml")

        assert str(caught.value).startswith("rig.toml: robot must be a name")

    def test_a_description_can_be_copied_and_pickled(self):
        description = parse_robot_description(ARM)

        assert copy.deepcopy(description) == description
        assert pickle.loads(pickle.dumps(description)) == description

    def test_a_description_is_frozen(self):
        description = parse_robot_description(ARM)

        with pytest.raises(AttributeError):
            description.robot = "other"


class TestTopLevel:
    def test_toml_syntax(self):
        assert refused("robot = ").startswith("not valid TOML: ")

    def test_unknown_key(self):
        assert (
            refused(ARM.replace('robot = "arm"', 'robot = "arm"\nrobots = 2'))
            == "unknown key 'robots' (expected robot, series, shapes, grippers, attributes)"
        )

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "robot = 5",
            'robot = ""',
            'robot = "two words"',
            'robot = "-x"',
            f'robot = "{"a" * 65}"',
        ],
    )
    def test_robot_name(self, line):
        assert (
            refused(ARM.replace('robot = "arm"', line))
            == "robot must be a name of 1 to 64 letters, digits, '_', '.' or '-'"
        )

    @pytest.mark.parametrize("name", ["a", "rebot_b601", "9.arm-2", "a" * 64])
    def test_robot_names_that_pass(self, name):
        assert (
            parse_robot_description(ARM.replace('"arm"', f'"{name}"', 1)).robot == name
        )

    @pytest.mark.parametrize("text", ['robot = "r"', 'robot = "r"\n[series]'])
    def test_no_series(self, text):
        assert refused(text) == "series must declare at least one data series"

    def test_series_is_not_a_table(self):
        assert refused('robot = "r"\nseries = ["arm"]') == "series must be a table"

    def test_a_series_is_not_a_table(self):
        assert (
            refused('robot = "r"\n[series]\narm = 5') == "series 'arm' must be a table"
        )

    def test_too_many_series(self):
        series = "".join(
            f'[series.s{i}]\nshape = "status"\nreadings = ["a"]\n' for i in range(65)
        )

        assert refused('robot = "r"\n' + series) == "more than 64 series"

    def test_attributes(self):
        description = parse_robot_description(
            'robot = "r"\nattributes = { maker = "nobody", serial = "7" }\n'
            '[series.s]\nshape = "status"\nreadings = ["a"]\nattributes = { bus = "can0" }\n'
        )

        assert description.attributes == {"maker": "nobody", "serial": "7"}
        assert description.series["s"].attributes == {"bus": "can0"}

    def test_attributes_must_be_a_table_of_text(self):
        assert (
            refused(ARM.replace('robot = "arm"', 'robot = "arm"\nattributes = 5'))
            == "attributes must be a table"
        )
        assert (
            refused(
                ARM.replace(
                    'robot = "arm"', 'robot = "arm"\nattributes = { serial = 7 }'
                )
            )
            == "attributes.serial must be text"
        )
        assert (
            refused(
                ARM.replace(
                    'shape = "status"', 'shape = "status"\nattributes = { bus = true }'
                )
            )
            == "series 'status': attributes.bus must be text"
        )

    def test_an_attribute_value_that_is_too_long(self):
        long = "x" * 257
        assert (
            refused(
                ARM.replace(
                    'robot = "arm"',
                    f'robot = "arm"\nattributes = {{ note = "{long}" }}',
                )
            )
            == "more than 256 characters in attribute 'note'"
        )
        assert (
            refused(
                ARM.replace(
                    'shape = "status"',
                    f'shape = "status"\nattributes = {{ note = "{long}" }}',
                )
            )
            == "series 'status': more than 256 characters in attribute 'note'"
        )


NAME_RULE = "must be 1 to 64 letters, digits or '_', starting with a letter"


class TestNames:
    @pytest.mark.parametrize("name", ["1arm", "my arm", "arm-1", "_arm", "a" * 65])
    def test_series_name(self, name):
        text = f'robot = "r"\n[series."{name}"]\nshape = "status"\nreadings = ["a"]\n'

        assert refused(text) == f"series name '{name}' {NAME_RULE}"

    def test_shape_name(self):
        text = (
            'robot = "r"\n[shapes."my-shape"]\nfields = ["a"]\n'
            + '[series.s]\nshape = "my-shape"\n'
        )

        assert refused(text) == f"shape name 'my-shape' {NAME_RULE}"

    def test_joint_name(self):
        assert (
            refused(joints_series('{ name = "j 1", min = 0, max = 1 }'))
            == f"joint name 'j 1' {NAME_RULE}"
        )

    def test_reading_name(self):
        text = 'robot = "r"\n[series.s]\nshape = "status"\nreadings = ["motors on"]\n'

        assert refused(text) == f"reading name 'motors on' {NAME_RULE}"

    def test_field_name(self):
        text = 'robot = "r"\n[shapes.force]\nfields = [{ name = "9x" }]\n[series.s]\nshape = "force"\n'

        assert refused(text) == f"field name '9x' {NAME_RULE}"

    def test_gripper_name(self):
        text = (
            ARM
            + '[grippers."2nd"]\nseries = "arm"\njoints = ["j1"]\nopen = 1\nclosed = 0\n'
        )

        assert refused(text) == f"gripper name '2nd' {NAME_RULE}"

    def test_reserved_series_name(self):
        text = 'robot = "r"\n[series.robot_description]\nshape = "status"\nreadings = ["a"]\n'

        assert refused(text) == "series name 'robot_description' is reserved"

    @pytest.mark.parametrize("name", ["joints", "status"])
    def test_reserved_shape_name(self, name):
        text = f'robot = "r"\n[shapes.{name}]\nfields = ["a"]\n[series.s]\nshape = "{name}"\n'

        assert refused(text) == f"shape name '{name}' is reserved"


class TestSeries:
    def test_unknown_key(self):
        assert (
            refused(ARM.replace('shape = "joints"', 'shape = "joints"\nrate = 5'))
            == "series 'arm': unknown key 'rate'"
        )

    def test_shape_missing(self):
        assert (
            refused('robot = "r"\n[series.s]\nreadings = ["a"]\n')
            == "series 's': shape is required (joints, status, or a shape under [shapes])"
        )

    def test_shape_unknown(self):
        assert (
            refused('robot = "r"\n[series.s]\nshape = "force"\n')
            == "series 's': unknown shape 'force'"
        )

    def test_shape_is_not_text(self):
        assert (
            refused('robot = "r"\n[series.s]\nshape = 3\n')
            == "series 's': shape must be text"
        )

    @pytest.mark.parametrize(
        ("shape", "line", "key"),
        [
            ("joints", 'readings = ["a"]', "readings"),
            ("status", "joints = []", "joints"),
            ("status", "has_velocity = true", "has_velocity"),
            ("status", "has_effort = true", "has_effort"),
            ("force", 'readings = ["a"]', "readings"),
            ("force", "has_effort = false", "has_effort"),
        ],
    )
    def test_key_of_another_shape(self, shape, line, key):
        text = f'robot = "r"\n[shapes.force]\nfields = ["fx"]\n[series.s]\nshape = "{shape}"\n{line}\n'

        assert refused(text) == f"series 's': '{key}' does not apply to shape '{shape}'"

    @pytest.mark.parametrize(
        "line", ["", "joints = []", 'joints = "j1"', "joints = { name = 'j1' }"]
    )
    def test_joints_missing_or_empty(self, line):
        text = f'robot = "r"\n[series.arm]\nshape = "joints"\n{line}\n'

        assert refused(text) == "series 'arm': joints must be a non-empty list"

    @pytest.mark.parametrize("line", ["", "readings = []", "readings = 4"])
    def test_readings_missing_or_empty(self, line):
        text = f'robot = "r"\n[series.s]\nshape = "status"\n{line}\n'

        assert refused(text) == "series 's': readings must be a non-empty list"

    @pytest.mark.parametrize("value", ['"fast"', "true", "inf", "nan"])
    def test_rate_is_not_a_finite_number(self, value):
        text = ARM.replace('shape = "joints"', f'shape = "joints"\nrate_hz = {value}')

        assert refused(text) == "series 'arm': rate_hz must be a finite number"

    @pytest.mark.parametrize("value", ["0", "-20.0"])
    def test_rate_must_be_above_zero(self, value):
        text = ARM.replace('shape = "joints"', f'shape = "joints"\nrate_hz = {value}')

        assert refused(text) == "series 'arm': rate_hz must be above zero"

    def test_stale_after(self):
        def with_stale(value: str) -> str:
            return ARM.replace(
                'shape = "joints"', f'shape = "joints"\nstale_after_s = {value}'
            )

        assert (
            refused(with_stale("-0.1"))
            == "series 'arm': stale_after_s must be zero or above"
        )
        assert (
            refused(with_stale('"1s"'))
            == "series 'arm': stale_after_s must be a finite number"
        )
        assert (
            parse_robot_description(with_stale("0")).series["arm"].stale_after_s == 0.0
        )
        assert (
            parse_robot_description(with_stale("3")).series["arm"].stale_after_s == 3.0
        )
        assert parse_robot_description(ARM).series["arm"].stale_after_s == 0.5

    @pytest.mark.parametrize("key", ["has_velocity", "has_effort"])
    def test_flags_must_be_true_or_false(self, key):
        text = ARM.replace('shape = "joints"', f'shape = "joints"\n{key} = 1')

        assert refused(text) == f"series 'arm': {key} must be true or false"

    def test_flags(self):
        text = ARM.replace('shape = "joints"', 'shape = "joints"\nhas_velocity = true')
        arm = parse_robot_description(text).series["arm"]

        assert arm.has_velocity is True
        assert arm.has_effort is False

    def test_attributes_must_be_a_table(self):
        text = ARM.replace('shape = "joints"', 'shape = "joints"\nattributes = "x"')

        assert refused(text) == "series 'arm': attributes must be a table"


class TestJoints:
    @pytest.mark.parametrize(
        "entry", ['"j1"', "5", "{ kind = 'revolute' }", "{ name = 5 }"]
    )
    def test_entry_without_a_name(self, entry):
        assert (
            refused(joints_series(entry))
            == "series 'arm': joints[0] must be a table with a name"
        )

    def test_the_index_of_the_entry_is_named(self):
        text = joints_series('{ name = "a", min = 0, max = 1 }, 7')

        assert refused(text) == "series 'arm': joints[1] must be a table with a name"

    def test_listed_twice(self):
        text = joints_series(
            '{ name = "a", min = 0, max = 1 }, { name = "a", min = 0, max = 1 }'
        )

        assert refused(text) == "series 'arm': joint 'a' is listed twice"

    def test_unknown_key(self):
        text = joints_series('{ name = "a", min = 0, max = 1, limit = 3 }')

        assert refused(text) == "series 'arm': joint 'a': unknown key 'limit'"

    @pytest.mark.parametrize("kind", ['"hinge"', '"Revolute"', "3"])
    def test_kind(self, kind):
        text = joints_series(f'{{ name = "a", kind = {kind}, min = 0, max = 1 }}')
        word = kind.strip('"')

        assert (
            refused(text)
            == f"series 'arm': joint 'a': kind '{word}' is not one of revolute, prismatic, continuous, gripper"
        )

    def test_a_joint_without_a_kind_is_revolute(self):
        joint = (
            parse_robot_description(joints_series('{ name = "a", min = 0, max = 1 }'))
            .series["arm"]
            .joints[0]
        )

        assert joint.kind == "revolute"
        assert joint.unit == "rad"

    @pytest.mark.parametrize("kind", ["revolute", "prismatic", "gripper"])
    @pytest.mark.parametrize("limits", ["", ", min = 0", ", max = 1"])
    def test_limits_missing(self, kind, limits):
        text = joints_series(f'{{ name = "a", kind = "{kind}"{limits} }}')

        assert (
            refused(text)
            == f"series 'arm': joint 'a': min and max are required for kind '{kind}'"
        )

    @pytest.mark.parametrize("limits", ["min = 0", "max = 1", "min = 0, max = 1"])
    def test_limits_on_a_continuous_joint(self, limits):
        text = joints_series(f'{{ name = "a", kind = "continuous", {limits} }}')

        assert (
            refused(text)
            == "series 'arm': joint 'a': min and max do not apply to kind 'continuous'"
        )

    @pytest.mark.parametrize("key", ["min", "max", "max_velocity", "max_effort"])
    @pytest.mark.parametrize("value", ['"1"', "true", "inf", "-inf", "nan", "[1]"])
    def test_not_a_finite_number(self, key, value):
        fields = {"min": "0", "max": "1"} | {key: value}
        entry = ", ".join(f"{name} = {text}" for name, text in fields.items())

        assert (
            refused(joints_series(f'{{ name = "a", {entry} }}'))
            == f"series 'arm': joint 'a': {key} must be a finite number"
        )

    @pytest.mark.parametrize(("low", "high"), [("1", "1"), ("2", "1"), ("0.5", "-0.5")])
    def test_min_must_be_below_max(self, low, high):
        text = joints_series(f'{{ name = "a", min = {low}, max = {high} }}')

        assert (
            refused(text)
            == f"series 'arm': joint 'a': min {low} must be below max {high}"
        )

    @pytest.mark.parametrize("key", ["max_velocity", "max_effort"])
    @pytest.mark.parametrize("value", ["0", "-2.0"])
    def test_largest_speed_and_effort_must_be_above_zero(self, key, value):
        text = joints_series(f'{{ name = "a", min = 0, max = 1, {key} = {value} }}')

        assert refused(text) == f"series 'arm': joint 'a': {key} must be above zero"

    def test_unit(self):
        prismatic = '{ name = "a", kind = "prismatic", min = 0, max = 1 }'
        stated = '{ name = "a", kind = "prismatic", min = 0, max = 100, unit = "mm" }'
        wrong = '{ name = "a", min = 0, max = 1, unit = 3 }'

        assert (
            parse_robot_description(joints_series(prismatic))
            .series["arm"]
            .joints[0]
            .unit
            == "m"
        )
        assert (
            parse_robot_description(joints_series(stated)).series["arm"].joints[0].unit
            == "mm"
        )
        assert (
            refused(joints_series(wrong))
            == "series 'arm': joint 'a': unit must be text"
        )

    def test_whole_numbers_are_numbers(self):
        joint = (
            parse_robot_description(joints_series('{ name = "a", min = -1, max = 2 }'))
            .series["arm"]
            .joints[0]
        )

        assert (joint.min, joint.max) == (-1.0, 2.0)
        assert isinstance(joint.min, float)

    def test_too_many_joints(self):
        joints = ", ".join(f'{{ name = "j{i}", min = 0, max = 1 }}' for i in range(257))

        assert refused(joints_series(joints)) == "series 'arm': more than 256 joints"

    def test_as_many_joints_as_allowed(self):
        joints = ", ".join(f'{{ name = "j{i}", min = 0, max = 1 }}' for i in range(256))

        assert (
            len(parse_robot_description(joints_series(joints)).series["arm"].joints)
            == 256
        )


class TestGripperJoints:
    @pytest.mark.parametrize("key", ["open", "closed"])
    @pytest.mark.parametrize("kind", ["revolute", "prismatic"])
    def test_open_and_closed_on_another_kind(self, key, kind):
        text = joints_series(
            f'{{ name = "a", kind = "{kind}", min = 0, max = 1, {key} = 0.5 }}'
        )

        assert (
            refused(text)
            == "series 'arm': joint 'a': open and closed apply to kind 'gripper' only"
        )

    def test_open_and_closed_default_to_the_limits(self):
        text = joints_series(
            '{ name = "jaw", kind = "gripper", min = 0.01, max = 0.09 }'
        )
        gripper = parse_robot_description(text).grippers["jaw"]

        assert (gripper.closed, gripper.open) == (0.01, 0.09)
        assert gripper.series == "arm"
        assert gripper.joints == ("jaw",)

    def test_open_and_closed_as_stated(self):
        text = joints_series(
            '{ name = "jaw", kind = "gripper", min = 0, max = 1, open = 0.1, closed = 0.9 }'
        )
        gripper = parse_robot_description(text).grippers["jaw"]

        assert (gripper.open, gripper.closed) == (0.1, 0.9)

    def test_only_one_of_them_stated(self):
        text = joints_series(
            '{ name = "jaw", kind = "gripper", min = 0, max = 1, open = 0.8 }'
        )
        gripper = parse_robot_description(text).grippers["jaw"]

        assert (gripper.open, gripper.closed) == (0.8, 0.0)

    @pytest.mark.parametrize(
        "stated", ["open = 0.5, closed = 0.5", "open = 0", "closed = 1"]
    )
    def test_open_and_closed_must_differ(self, stated):
        text = joints_series(
            f'{{ name = "jaw", kind = "gripper", min = 0, max = 1, {stated} }}'
        )

        assert refused(text) == "series 'arm': joint 'jaw': open and closed must differ"

    def test_open_is_not_a_number(self):
        text = joints_series(
            '{ name = "jaw", kind = "gripper", min = 0, max = 1, open = "wide" }'
        )

        assert (
            refused(text) == "series 'arm': joint 'jaw': open must be a finite number"
        )

    def test_the_same_gripper_joint_in_two_series(self):
        jaw = 'joints = [{ name = "jaw", kind = "gripper", min = 0, max = 1 }]\n'
        text = f'robot = "r"\n[series.left]\nshape = "joints"\n{jaw}[series.right]\nshape = "joints"\n{jaw}'

        assert refused(text) == (
            "series 'right': joint 'jaw': a gripper named 'jaw' is already declared by joint "
            "'jaw' of series 'left'; gripper names are unique across the robot: rename the "
            "joint, or give it another kind and declare its gripper under [grippers]"
        )

    def test_two_identical_arms_are_told_what_to_change(self):
        def robot(right_joint: str, more: str = "") -> str:
            shoulder = '{ name = "shoulder", min = -1, max = 1 }'
            left = '{ name = "gripper", kind = "gripper", min = 0, max = 1 }'
            return (
                'robot = "r"\n'
                f'[series.left_arm]\nshape = "joints"\njoints = [{shoulder}, {left}]\n'
                f'[series.right_arm]\nshape = "joints"\njoints = [{shoulder}, {right_joint}]\n'
                f"{more}"
            )

        same = '{ name = "gripper", kind = "gripper", min = 0, max = 1 }'
        with pytest.raises(RobotDescriptionError) as caught:
            parse_robot_description(robot(same), source="two_arms.toml")
        message = str(caught.value)
        assert message.startswith("two_arms.toml: series 'right_arm': joint 'gripper':")
        assert "series 'left_arm'" in message
        assert "rename the joint" in message

        # Either remedy the text names is accepted.
        renamed = '{ name = "r_gripper", kind = "gripper", min = 0, max = 1 }'
        assert list(parse_robot_description(robot(renamed)).grippers) == [
            "gripper",
            "r_gripper",
        ]
        other_kind = '{ name = "gripper", kind = "prismatic", min = 0, max = 1 }'
        table = '[grippers.right_gripper]\nseries = "right_arm"\njoints = ["gripper"]\nopen = 1\nclosed = 0\n'
        grippers = parse_robot_description(robot(other_kind, table)).grippers
        assert [(gripper.name, gripper.series) for gripper in grippers.values()] == [
            ("gripper", "left_arm"),
            ("right_gripper", "right_arm"),
        ]


class TestGripperTables:
    GRIPPER = '[grippers.hand]\nseries = "arm"\njoints = ["j1", "jaw"]\nopen = 0.0\nclosed = 2.0\n'

    def test_a_gripper_of_several_joints(self):
        description = parse_robot_description(ARM + self.GRIPPER)

        assert list(description.grippers) == ["jaw", "hand"]
        assert description.grippers["hand"] == GripperDescription(
            name="hand", series="arm", joints=("j1", "jaw"), open=0.0, closed=2.0
        )

    def test_unknown_key(self):
        assert (
            refused(ARM + self.GRIPPER + "force = 3\n")
            == "gripper 'hand': unknown key 'force'"
        )

    def test_not_a_table(self):
        assert (
            refused(ARM + '[grippers]\nhand = "arm"\n')
            == "gripper 'hand' must be a table"
        )
        assert (
            refused(ARM.replace('robot = "arm"', 'robot = "arm"\ngrippers = 3'))
            == "grippers must be a table"
        )

    @pytest.mark.parametrize("series", ["status", "none"])
    def test_series_is_not_a_joints_series(self, series):
        text = ARM + self.GRIPPER.replace('series = "arm"', f'series = "{series}"')

        assert (
            refused(text) == f"gripper 'hand': series '{series}' is not a joints series"
        )

    def test_series_missing(self):
        assert (
            refused(ARM + self.GRIPPER.replace('series = "arm"\n', ""))
            == "gripper 'hand': series must be text"
        )

    def test_joint_is_not_in_the_series(self):
        text = ARM + self.GRIPPER.replace('"j1", "jaw"', '"j1", "j9"')

        assert refused(text) == "gripper 'hand': joint 'j9' is not in series 'arm'"

    def test_joint_listed_twice(self):
        text = ARM + self.GRIPPER.replace('"j1", "jaw"', '"j1", "j1"')

        assert refused(text) == "gripper 'hand': joint 'j1' is listed twice"

    @pytest.mark.parametrize("line", ["", "joints = []\n", 'joints = "j1"\n'])
    def test_joints_missing_or_empty(self, line):
        text = ARM + self.GRIPPER.replace('joints = ["j1", "jaw"]\n', line)

        assert refused(text) == "gripper 'hand': joints must be a non-empty list"

    def test_joint_entry_is_not_a_name(self):
        text = ARM + self.GRIPPER.replace('"j1", "jaw"', '"j1", 2')

        assert refused(text) == "gripper 'hand': joints[1] must be a name"

    @pytest.mark.parametrize("key", ["open", "closed"])
    def test_open_and_closed_are_required_numbers(self, key):
        missing = (
            "\n".join(
                line for line in self.GRIPPER.splitlines() if not line.startswith(key)
            )
            + "\n"
        )
        wrong = "\n".join(
            f"{key} = true" if line.startswith(key) else line
            for line in self.GRIPPER.splitlines()
        )

        assert (
            refused(ARM + missing) == f"gripper 'hand': {key} must be a finite number"
        )
        assert (
            refused(ARM + wrong + "\n")
            == f"gripper 'hand': {key} must be a finite number"
        )

    def test_open_and_closed_must_differ(self):
        text = ARM + self.GRIPPER.replace("closed = 2.0", "closed = 0.0")

        assert refused(text) == "gripper 'hand': open and closed must differ"

    def test_a_name_a_gripper_joint_has_taken(self):
        text = ARM + self.GRIPPER.replace("grippers.hand", "grippers.jaw")

        assert refused(text) == (
            "gripper 'jaw': a gripper named 'jaw' is already declared by joint 'jaw' of series "
            "'arm'; gripper names are unique across the robot: give this gripper another name"
        )


class TestReadingsAndFields:
    STATUS = 'robot = "r"\n[series.s]\nshape = "status"\nreadings = [{}]\n'
    FORCE = 'robot = "r"\n[shapes.force]\nfields = [{}]\n[series.s]\nshape = "force"\n'

    @pytest.mark.parametrize(
        "entry", ["5", "{ kind = 'number' }", "{ name = 5 }", "['a']"]
    )
    def test_entry_without_a_name(self, entry):
        assert (
            refused(self.STATUS.format(f'"a", {entry}'))
            == "series 's': readings[1] must be a name or a table with a name"
        )
        assert (
            refused(self.FORCE.format(f'"a", {entry}'))
            == "shape 'force': fields[1] must be a name or a table with a name"
        )

    def test_listed_twice(self):
        assert (
            refused(self.STATUS.format('"a", { name = "a" }'))
            == "series 's': reading 'a' is listed twice"
        )
        assert (
            refused(self.FORCE.format('"a", "a"'))
            == "shape 'force': field 'a' is listed twice"
        )

    def test_unknown_key(self):
        assert (
            refused(self.STATUS.format('{ name = "a", scale = 2 }'))
            == "series 's': reading 'a': unknown key 'scale'"
        )
        assert (
            refused(self.FORCE.format('{ name = "a", scale = 2 }'))
            == "shape 'force': field 'a': unknown key 'scale'"
        )
        assert refused(
            self.FORCE.format('"a"').replace("fields", "size = 2\nfields")
        ) == ("shape 'force': unknown key 'size'")

    def test_reading_kinds(self):
        text = self.STATUS.format(
            '"a", { name = "b", kind = "number" }, { name = "c", kind = "flag" }, { name = "d", kind = "text" }'
        )
        values = parse_robot_description(text).series["s"].values

        assert [value.kind for value in values] == ["any", "number", "flag", "text"]

    @pytest.mark.parametrize("kind", ['"numbers"', '"bool"', "7"])
    def test_reading_kind_refused(self, kind):
        word = kind.strip('"')

        assert (
            refused(self.STATUS.format(f'{{ name = "a", kind = {kind} }}'))
            == f"series 's': reading 'a': kind '{word}' is not one of any, number, flag, text"
        )

    def test_field_kinds(self):
        kinds = ["number", "flag", "text", "numbers", "any"]
        fields = ", ".join(
            f'{{ name = "f{i}", kind = "{kind}" }}' for i, kind in enumerate(kinds)
        )
        series = parse_robot_description(
            self.FORCE.format('"plain", ' + fields)
        ).series["s"]

        assert [value.kind for value in series.values] == ["number", *kinds]
        assert series.shape == "custom"
        assert series.shape_name == "force"

    def test_field_kind_refused(self):
        assert (
            refused(self.FORCE.format('{ name = "a", kind = "vector" }'))
            == "shape 'force': field 'a': kind 'vector' is not one of number, flag, text, numbers, any"
        )

    def test_unit_must_be_text(self):
        assert (
            refused(self.STATUS.format('{ name = "a", unit = 5 }'))
            == "series 's': reading 'a': unit must be text"
        )
        assert (
            refused(self.FORCE.format('{ name = "a", unit = 5 }'))
            == "shape 'force': field 'a': unit must be text"
        )

    @pytest.mark.parametrize("fields", ["", "fields = []", 'fields = "fx"'])
    def test_fields_missing_or_empty(self, fields):
        text = f'robot = "r"\n[shapes.force]\n{fields}\n[series.s]\nshape = "force"\n'

        assert refused(text) == "shape 'force': fields must be a non-empty list"

    def test_a_shape_is_not_a_table(self):
        assert (
            refused('robot = "r"\n[shapes]\nforce = ["fx"]\n')
            == "shape 'force' must be a table"
        )
        assert refused('robot = "r"\nshapes = ["force"]\n') == "shapes must be a table"

    def test_too_many_readings_and_fields(self):
        names = ", ".join(f'"v{i}"' for i in range(257))

        assert (
            refused(self.STATUS.format(names)) == "series 's': more than 256 readings"
        )
        assert (
            refused(self.FORCE.format(names)) == "shape 'force': more than 256 fields"
        )

    def test_a_shape_no_series_uses(self):
        text = ARM + '[shapes.force]\nfields = ["fx"]\n'

        assert list(parse_robot_description(text).series) == ["arm", "status"]

    def test_two_series_of_one_shape_share_its_fields(self):
        text = self.FORCE.format('"fx", "fy"') + '[series.t]\nshape = "force"\n'
        series = parse_robot_description(text).series

        assert series["s"].values == series["t"].values
        assert series["s"].value_names == ["fx", "fy"]
