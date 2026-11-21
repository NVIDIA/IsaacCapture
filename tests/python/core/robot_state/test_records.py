# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot-state records: a description into its record and back, readings into records and back."""

import json
import math

import numpy as np
import pytest
from repo_paths import repo_root

from isaaccapture import schema
from isaaccapture.robot_state import (
    Fault,
    RobotDescription,
    load_robot_description,
    parse_robot_description,
    records,
)

DESCRIPTIONS = (
    repo_root()
    / "examples"
    / "robot_state"
    / "python"
    / "isaaccapture_examples"
    / "robot_state"
    / "descriptions"
)

# Every kind of joint, reading and field, with attributes and a gripper of two joints.
FULL = """
robot = "full.robot-1"
attributes = { maker = "nobody" }

[shapes.probe]
fields = [
  { name = "depth", kind = "number", unit = "mm" },
  { name = "touching", kind = "flag" },
  { name = "label", kind = "text" },
  { name = "trace", kind = "numbers" },
  { name = "extra", kind = "any" },
]

[series.arm]
shape = "joints"
rate_hz = 100.0
stale_after_s = 0.25
has_velocity = true
has_effort = true
attributes = { bus = "can0" }
joints = [
  { name = "lift", kind = "prismatic", min = 0.0, max = 0.4, max_velocity = 0.5, max_effort = 80.0 },
  { name = "turn", kind = "continuous" },
  { name = "bend", kind = "revolute", min = -1.5, max = 1.5, unit = "deg" },
  { name = "left_jaw", kind = "gripper", min = 0.0, max = 0.04, unit = "m", open = 0.035, closed = 0.002 },
  { name = "right_jaw", kind = "revolute", min = 0.0, max = 0.04 },
]

[series.probe]
shape = "probe"
stale_after_s = 0.0

[series.status]
shape = "status"
readings = [
  "anything",
  { name = "battery", kind = "number", unit = "V" },
  { name = "motors_on", kind = "flag" },
  { name = "mode", kind = "text" },
]

[grippers.hand]
series = "arm"
joints = ["left_jaw", "right_jaw"]
open = 0.08
closed = 0.0
"""


@pytest.fixture
def six_joint_arm() -> RobotDescription:
    return load_robot_description(DESCRIPTIONS / "six_joint_arm.toml")


@pytest.fixture
def two_arm_robot() -> RobotDescription:
    return load_robot_description(DESCRIPTIONS / "two_arm_robot.toml")


@pytest.fixture
def full() -> RobotDescription:
    return parse_robot_description(FULL)


def refused(call) -> str:
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


class TestDescriptionRecord:
    @pytest.mark.parametrize("name", ["six_joint_arm", "two_arm_robot"])
    def test_an_example_file_into_its_record_and_back(self, name):
        description = load_robot_description(DESCRIPTIONS / f"{name}.toml")
        data = records.encode_description(description, 1_000)

        assert isinstance(data, bytes)
        assert records.decode_description(data) == description

    def test_every_kind_into_the_record_and_back(self, full):
        assert records.decode_description(records.encode_description(full, 5)) == full

    def test_series_keep_file_order(self, two_arm_robot):
        back = records.decode_description(records.encode_description(two_arm_robot, 5))

        assert list(back.series) == list(two_arm_robot.series)
        assert (
            back.series["left_arm"].joint_names
            == two_arm_robot.series["left_arm"].joint_names
        )
        assert list(back.grippers) == ["l_gripper", "r_gripper"]

    def test_the_record_as_the_schema_reads_it(self, six_joint_arm):
        record = schema.RobotDescriptionRecord.from_bytes(
            records.encode_description(six_joint_arm, 77)
        )

        assert record.timestamp.sample_time_local_common_clock == 77
        assert record.timestamp.available_time_local_common_clock == 77
        assert record.timestamp.sample_time_raw_device_clock == 0
        assert record.data.robot == "rebot_b601"
        arm, status = record.data.series
        assert arm.shape == schema.RobotSeriesShape.JOINTS
        assert arm.shape_name == "joints"
        assert arm.rate_hz == 20.0
        assert arm.stale_after_s == 0.5
        assert [joint.name for joint in arm.joints][-1] == "gripper"
        assert arm.joints[1].has_limits is True
        assert (arm.joints[1].min_position, arm.joints[1].max_position) == (-0.05, 3.14)
        assert arm.joints[6].kind == schema.RobotJointKind.GRIPPER
        assert status.shape == schema.RobotSeriesShape.STATUS
        assert status.rate_hz == 0.0
        assert [value.name for value in status.values] == ["motors_on", "fault"]

    def test_grippers_are_written_once_and_read_from_the_list(
        self, six_joint_arm, full
    ):
        written = schema.RobotDescriptionRecord.from_bytes(
            records.encode_description(six_joint_arm, 1)
        ).data
        assert [gripper.name for gripper in written.grippers] == ["gripper"]
        assert written.grippers[0].joints == ["gripper"]

        # A record whose list has no gripper decodes to none, whatever its joints' kinds are.
        series = schema.RobotSeriesDescription(
            name="arm",
            shape=schema.RobotSeriesShape.JOINTS,
            shape_name="joints",
            joints=[
                schema.RobotJointDescription(
                    name="jaw",
                    kind=schema.RobotJointKind.GRIPPER,
                    has_limits=True,
                    max_position=1.0,
                )
            ],
        )
        bare = schema.RobotDescriptionRecord(
            schema.RobotDescription(robot="r", series=[series]),
            schema.DeviceDataTimestamp(1, 1, 0),
        )
        decoded = records.decode_description(bare.to_bytes())
        assert decoded.grippers == {}
        assert decoded.series["arm"].joints[0].kind == "gripper"

        assert list(
            records.decode_description(records.encode_description(full, 1)).grippers
        ) == ["left_jaw", "hand"]

    def test_what_a_record_leaves_out_reads_as_not_stated(self):
        series = schema.RobotSeriesDescription(
            name="arm",
            shape_name="joints",
            joints=[
                schema.RobotJointDescription(name="a"),
                schema.RobotJointDescription(name="b", unit="deg"),
            ],
        )
        record = schema.RobotDescriptionRecord(
            schema.RobotDescription(robot="r", series=[series]),
            schema.DeviceDataTimestamp(1, 1, 0),
        )
        arm = records.decode_description(record.to_bytes()).series["arm"]

        assert arm.rate_hz is None
        assert arm.stale_after_s == 0.0
        assert arm.joints[0].min is None
        assert arm.joints[0].max is None
        assert arm.joints[0].max_velocity is None
        assert arm.joints[0].max_effort is None
        assert arm.joints[0].unit == "rad"
        assert arm.joints[1].unit == "deg"
        assert arm.limits == {}

    @pytest.mark.parametrize("junk", [b"", b"junk", b"\x00" * 32])
    def test_a_bad_buffer_is_refused(self, junk):
        assert (
            refused(lambda: records.decode_description(junk))
            == "not a valid RobotDescriptionRecord buffer"
        )

    def test_a_record_without_a_description_is_refused(self):
        empty = schema.RobotDescriptionRecord(None, schema.DeviceDataTimestamp(1, 1, 0))

        assert refused(lambda: records.decode_description(empty.to_bytes())) == (
            "a RobotDescriptionRecord without a description"
        )


class TestDeclarations:
    def test_the_description_comes_first(self, six_joint_arm):
        declared = records.declarations(six_joint_arm)

        assert [declaration.name for declaration in declared] == [
            "robot_description",
            "arm",
            "status",
        ]
        first = declared[0]
        assert first.name == records.DESCRIPTION_TOPIC
        assert first.shape == "description"
        assert first.schema_name == "core.RobotDescriptionRecord"
        assert first.schema_encoding == "flatbuffer"
        assert first.schema_data == schema.robot_state_record_schema("description")[1]
        assert first.message_encoding == "flatbuffer"
        assert first.stale_after_ns == 0
        assert first.in_order is True
        assert first.rate_hz == 0.0

    def test_joints_and_status(self, six_joint_arm):
        _, arm, status = records.declarations(six_joint_arm)

        assert arm.shape == "joints"
        assert arm.schema_name == "core.JointStateOutputRecord"
        assert arm.schema_encoding == "flatbuffer"
        assert arm.schema_data == schema.robot_state_record_schema("joints")[1]
        assert arm.message_encoding == "flatbuffer"
        assert arm.stale_after_ns == 500_000_000
        assert arm.in_order is False
        assert arm.rate_hz == 20.0

        assert status.shape == "status"
        assert status.schema_name == "core.RobotStatusRecord"
        assert status.schema_data == schema.robot_state_record_schema("status")[1]
        assert status.stale_after_ns == records.DEFAULT_STALE_AFTER_S * 1e9
        assert status.rate_hz == 0.0

    def test_the_two_arm_robot_in_file_order(self, two_arm_robot):
        declared = records.declarations(two_arm_robot)

        assert [(declaration.name, declaration.shape) for declaration in declared] == [
            ("robot_description", "description"),
            ("left_arm", "joints"),
            ("right_arm", "joints"),
            ("waist", "joints"),
            ("left_wrist_force", "force"),
            ("right_wrist_force", "force"),
            ("status", "status"),
        ]
        assert declared[-1].stale_after_ns == 2_000_000_000
        assert [declaration.in_order for declaration in declared] == [True] + [
            False
        ] * 6

    def test_a_shape_the_robot_defines_brings_a_json_schema(self, two_arm_robot):
        force = records.declarations(two_arm_robot)[4]

        assert force.schema_name == "two_arm_example.force"
        assert force.schema_encoding == "jsonschema"
        assert force.message_encoding == "json"
        assert force.rate_hz == 50.0
        newton = {"type": "number", "unit": "N"}
        newton_metre = {"type": "number", "unit": "N*m"}
        assert json.loads(force.schema_data) == {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "two_arm_example.force",
            "type": "object",
            "properties": {
                "fx": newton,
                "fy": newton,
                "fz": newton,
                "tx": newton_metre,
                "ty": newton_metre,
                "tz": newton_metre,
            },
            "required": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "additionalProperties": False,
        }

    def test_the_json_schema_is_compact_and_ordered(self, full):
        probe = records.declarations(full)[2]

        assert probe.name == "probe"
        assert probe.stale_after_ns == 0
        assert probe.schema_data.decode() == (
            '{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"full.robot-1.probe",'
            '"type":"object","properties":{"depth":{"type":"number","unit":"mm"},"touching":{"type":"boolean"},'
            '"label":{"type":"string"},"trace":{"type":"array","items":{"type":"number"}},"extra":{}},'
            '"required":["depth","touching","label","trace","extra"],"additionalProperties":false}'
        )

    def test_a_declaration_is_frozen(self, six_joint_arm):
        with pytest.raises(AttributeError):
            records.declarations(six_joint_arm)[0].name = "other"


class TestJoints:
    VALUES = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.96)

    def test_round_trip(self, six_joint_arm):
        data = records.encode_reading(six_joint_arm, "arm", self.VALUES, time_ns=7)
        joints = records.decode_joints(data)

        assert [joint.name for joint in joints] == six_joint_arm.series[
            "arm"
        ].joint_names
        assert [joint.position for joint in joints] == pytest.approx(self.VALUES)
        assert all(joint.valid for joint in joints)
        assert all(joint.velocity is None and joint.effort is None for joint in joints)

    def test_the_record_is_the_existing_joint_state(self, six_joint_arm):
        data = records.encode_reading(
            six_joint_arm, "arm", self.VALUES, time_ns=7, publish_time_ns=9
        )
        record = schema.JointStateOutputRecord.from_bytes(data)

        assert record.data.device_id == "rebot_b601"
        assert record.data.has_velocity is False
        assert record.data.has_effort is False
        assert record.data.ee_pose is None
        assert [joint.name for joint in record.data.joints] == six_joint_arm.series[
            "arm"
        ].joint_names
        assert record.timestamp.sample_time_local_common_clock == 7
        assert record.timestamp.available_time_local_common_clock == 9
        assert record.timestamp.sample_time_raw_device_clock == 0

    def test_publish_time_defaults_to_the_time_of_the_reading(self, six_joint_arm):
        data = records.encode_reading(six_joint_arm, "arm", self.VALUES, time_ns=7)

        assert (
            schema.JointStateOutputRecord.from_bytes(
                data
            ).timestamp.available_time_local_common_clock
            == 7
        )

    def test_velocities_and_efforts(self, full):
        data = records.encode_reading(
            full,
            "arm",
            [0.1, 0.2, 0.3, 0.4, 0.5],
            velocities=[1.0, 2.0, 3.0, 4.0, 5.0],
            efforts=[10.0, 20.0, 30.0, 40.0, 50.0],
            time_ns=1,
        )
        joints = records.decode_joints(data)

        assert [joint.velocity for joint in joints] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert [joint.effort for joint in joints] == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_efforts_only(self, two_arm_robot):
        data = records.encode_reading(
            two_arm_robot,
            "left_arm",
            [0, 0, 0, 0, 0.04],
            efforts=[1, 2, 3, 4, 5],
            time_ns=1,
        )
        joints = records.decode_joints(data)

        assert [joint.effort for joint in joints] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert all(joint.velocity is None for joint in joints)
        assert joints[4].position == pytest.approx(0.04)

    def test_a_joint_that_could_not_be_read(self, six_joint_arm):
        values = [0.1, float("nan"), None, 0.4, 0.5, 0.6, 1.0]
        valid = [True, False, False, True, True, True, True]
        joints = records.decode_joints(
            records.encode_reading(six_joint_arm, "arm", values, valid=valid, time_ns=1)
        )

        assert [joint.valid for joint in joints] == valid
        assert joints[1].position == 0.0
        assert joints[2].position == 0.0
        assert joints[3].position == pytest.approx(0.4)

    def test_values_by_name(self, six_joint_arm):
        names = six_joint_arm.series["arm"].joint_names
        by_name = dict(zip(reversed(names), reversed(self.VALUES)))
        joints = records.decode_joints(
            records.encode_reading(six_joint_arm, "arm", by_name, time_ns=1)
        )

        assert [joint.name for joint in joints] == names
        assert [joint.position for joint in joints] == pytest.approx(self.VALUES)

    def test_any_sequence_of_numbers(self, six_joint_arm):
        data = records.encode_reading(
            six_joint_arm, "arm", np.array(self.VALUES, dtype=np.float32), time_ns=1
        )

        assert [
            joint.position for joint in records.decode_joints(data)
        ] == pytest.approx(self.VALUES)
        assert records.decode_joints(
            records.encode_reading(six_joint_arm, "arm", iter([1] * 7), time_ns=1)
        )[0] == (
            "joint1",
            1.0,
            None,
            None,
            True,
        )

    @pytest.mark.parametrize("value", [0.3, 1, np.float32(0.3), np.array(0.3)])
    def test_a_single_number_for_a_series_of_one_joint(self, two_arm_robot, value):
        joints = records.decode_joints(
            records.encode_reading(two_arm_robot, "waist", value, time_ns=1)
        )

        assert [joint.name for joint in joints] == ["waist_yaw"]
        assert joints[0].position == pytest.approx(float(value))
        assert joints[0].valid is True

    def test_single_values_beside_a_series_of_one_joint(self):
        one = parse_robot_description(
            'robot = "r"\n[series.lift]\nshape = "joints"\nhas_velocity = true\n'
            'has_effort = true\njoints = [{ name = "lift", kind = "continuous" }]\n'
        )
        joints = records.decode_joints(
            records.encode_reading(
                one, "lift", 0.5, velocities=0.25, efforts=2, valid=False, time_ns=1
            )
        )

        assert joints == [("lift", 0.0, 0.0, 0.0, False)]

    def test_valid_from_a_numpy_comparison(self, six_joint_arm):
        valid = np.arange(7) != 2
        joints = records.decode_joints(
            records.encode_reading(
                six_joint_arm, "arm", self.VALUES, valid=valid, time_ns=1
            )
        )

        assert [joint.valid for joint in joints] == [
            True,
            True,
            False,
            True,
            True,
            True,
            True,
        ]

    def test_names_come_from_the_record(self):
        joints = [
            schema.JointState(name="b", position=2.0),
            schema.JointState(name="a", position=1.0, valid=False),
        ]
        record = schema.JointStateOutputRecord(
            schema.JointStateOutput(joints=joints, has_velocity=True),
            schema.DeviceDataTimestamp(1, 1, 0),
        )
        decoded = records.decode_joints(record.to_bytes())

        assert [(joint.name, joint.position, joint.valid) for joint in decoded] == [
            ("b", 2.0, True),
            ("a", 1.0, False),
        ]
        assert decoded[0].velocity == 0.0
        assert decoded[0].effort is None

    def test_a_record_without_joints(self):
        record = schema.JointStateOutputRecord(
            None, schema.DeviceDataTimestamp(1, 1, 0)
        )

        assert records.decode_joints(record.to_bytes()) == []

    def test_a_bad_buffer_is_refused(self):
        assert (
            refused(lambda: records.decode_joints(b"junk"))
            == "not a valid JointStateOutputRecord buffer"
        )


class TestStatus:
    def test_round_trip(self, six_joint_arm):
        data = records.encode_reading(six_joint_arm, "status", [True, ""], time_ns=3)

        assert records.decode_status(data) == ({"motors_on": True, "fault": ""}, [])

    def test_every_kind_of_value(self, full):
        data = records.encode_reading(
            full, "status", [7, 24.5, False, "hold"], time_ns=3
        )
        readings, faults = records.decode_status(data)

        assert readings == {
            "anything": 7.0,
            "battery": 24.5,
            "motors_on": False,
            "mode": "hold",
        }
        assert faults == []
        kinds = [
            reading.kind
            for reading in schema.RobotStatusRecord.from_bytes(data).data.readings
        ]
        assert kinds == [
            schema.RobotReadingKind.NUMBER,
            schema.RobotReadingKind.NUMBER,
            schema.RobotReadingKind.FLAG,
            schema.RobotReadingKind.TEXT,
        ]

    @pytest.mark.parametrize("value", [True, 3, 2.5, "text"])
    def test_a_reading_of_any_kind_takes_each_kind(self, full, value):
        data = records.encode_reading(full, "status", [value, 1.0, True, ""], time_ns=3)

        assert records.decode_status(data)[0]["anything"] == value

    def test_numpy_values(self, full):
        temperatures = np.array([70.0, 85.0])
        data = records.encode_reading(
            full,
            "status",
            [np.bool_(True), np.float32(24.5), (temperatures > 80).any(), "hold"],
            time_ns=3,
        )
        readings, _ = records.decode_status(data)

        assert readings == {
            "anything": True,
            "battery": 24.5,
            "motors_on": True,
            "mode": "hold",
        }
        assert type(readings["anything"]) is bool
        assert type(readings["motors_on"]) is bool
        kinds = [
            reading.kind
            for reading in schema.RobotStatusRecord.from_bytes(data).data.readings
        ]
        assert kinds[0] == schema.RobotReadingKind.FLAG
        assert kinds[2] == schema.RobotReadingKind.FLAG

    def test_a_single_value_for_a_status_of_one_reading(self):
        one = parse_robot_description(
            'robot = "r"\n[series.status]\nshape = "status"\nreadings = ["motors_on"]\n'
        )

        for value in (True, 1.5, "idle", None, np.bool_(False)):
            data = records.encode_reading(one, "status", value, time_ns=3)

            assert records.decode_status(data)[0] == {"motors_on": value}

    def test_a_reading_that_could_not_be_read(self, full):
        data = records.encode_reading(
            full, "status", [None, None, None, None], time_ns=3
        )
        readings, _ = records.decode_status(data)

        assert readings == {
            "anything": None,
            "battery": None,
            "motors_on": None,
            "mode": None,
        }
        encoded = schema.RobotStatusRecord.from_bytes(data).data.readings
        assert [reading.valid for reading in encoded] == [False] * 4
        assert encoded[2].kind == schema.RobotReadingKind.FLAG
        assert encoded[3].kind == schema.RobotReadingKind.TEXT

    def test_values_by_name(self, six_joint_arm):
        data = records.encode_reading(
            six_joint_arm, "status", {"fault": "stall", "motors_on": False}, time_ns=3
        )

        assert records.decode_status(data)[0] == {"motors_on": False, "fault": "stall"}

    def test_the_timestamp(self, six_joint_arm):
        data = records.encode_reading(
            six_joint_arm, "status", [True, ""], time_ns=3, publish_time_ns=4
        )
        timestamp = schema.RobotStatusRecord.from_bytes(data).timestamp

        assert timestamp.sample_time_local_common_clock == 3
        assert timestamp.available_time_local_common_clock == 4
        assert timestamp.sample_time_raw_device_clock == 0

    def test_faults_in_every_form(self, six_joint_arm):
        faults = [
            Fault("overtemperature", "joint3 is too hot", "joint3", "fatal"),
            ("stall", "motor stalled", "joint2", "warning"),
            ("bus", "bus errors", "can0"),
            ("estop", "stop button pressed"),
            Fault("low_battery", "battery is low"),
        ]
        data = records.encode_reading(
            six_joint_arm,
            "status",
            [False, "overtemperature"],
            faults=faults,
            time_ns=3,
        )

        assert records.decode_status(data)[1] == [
            Fault("overtemperature", "joint3 is too hot", "joint3", "fatal"),
            Fault("stall", "motor stalled", "joint2", "warning"),
            Fault("bus", "bus errors", "can0", "error"),
            Fault("estop", "stop button pressed", "", "error"),
            Fault("low_battery", "battery is low", "", "error"),
        ]

    def test_a_fault_that_ended_is_absent_from_the_next_record(self, six_joint_arm):
        first = records.encode_reading(
            six_joint_arm, "status", [True, "x"], faults=[("x", "y")], time_ns=3
        )
        second = records.encode_reading(
            six_joint_arm, "status", [True, ""], faults=(), time_ns=4
        )

        assert len(records.decode_status(first)[1]) == 1
        assert records.decode_status(second)[1] == []

    def test_a_record_without_a_status(self):
        record = schema.RobotStatusRecord(None, schema.DeviceDataTimestamp(1, 1, 0))

        assert records.decode_status(record.to_bytes()) == ({}, [])

    def test_a_bad_buffer_is_refused(self):
        assert (
            refused(lambda: records.decode_status(b"junk"))
            == "not a valid RobotStatusRecord buffer"
        )


class TestCustom:
    FORCE = (1.0, -2.0, 3.5, 0.1, 0.2, -0.3)

    def test_round_trip_in_file_order(self, two_arm_robot):
        data = records.encode_reading(
            two_arm_robot, "left_wrist_force", self.FORCE, time_ns=3
        )

        assert data == b'{"fx":1.0,"fy":-2.0,"fz":3.5,"tx":0.1,"ty":0.2,"tz":-0.3}'
        assert records.decode_custom(data) == dict(
            zip(("fx", "fy", "fz", "tx", "ty", "tz"), self.FORCE)
        )

    def test_round_trip_by_name(self, two_arm_robot):
        by_name = {"tz": 6, "ty": 5, "tx": 4, "fz": 3, "fy": 2, "fx": 1}
        data = records.encode_reading(
            two_arm_robot, "right_wrist_force", by_name, time_ns=3
        )

        assert list(records.decode_custom(data).items()) == [
            ("fx", 1.0),
            ("fy", 2.0),
            ("fz", 3.0),
            ("tx", 4.0),
            ("ty", 5.0),
            ("tz", 6.0),
        ]

    def test_every_kind_of_field(self, full):
        record = {
            "depth": 2,
            "touching": True,
            "label": "probe 1",
            "trace": (1, 2.5),
            "extra": {"raw": [1, "a", None]},
        }
        data = records.encode_reading(full, "probe", record, time_ns=3)

        assert records.decode_custom(data) == {
            "depth": 2.0,
            "touching": True,
            "label": "probe 1",
            "trace": [1.0, 2.5],
            "extra": {"raw": [1, "a", None]},
        }

    def test_numpy_flags(self, full):
        record = {
            "depth": np.float64(2.0),
            "touching": np.bool_(True),
            "label": "probe 1",
            "trace": np.array([1.0, 2.5]),
            "extra": np.float64(3.0) > 2.0,
        }
        data = records.encode_reading(full, "probe", record, time_ns=3)

        assert records.decode_custom(data) == {
            "depth": 2.0,
            "touching": True,
            "label": "probe 1",
            "trace": [1.0, 2.5],
            "extra": True,
        }

    def test_the_record_fits_the_schema_it_declares(self, full):
        declared = json.loads(records.declarations(full)[2].schema_data)
        record = records.decode_custom(
            records.encode_reading(full, "probe", [1.5, False, "", [], None], time_ns=3)
        )

        assert list(record) == declared["required"] == list(declared["properties"])
        python_types = {"number": float, "boolean": bool, "string": str, "array": list}
        for name, rule in declared["properties"].items():
            if "type" in rule:
                assert type(record[name]) is python_types[rule["type"]]

    @pytest.mark.parametrize(
        "data", [b"", b"{", b"[1, 2]", b'"text"', b"\xff\xfe", b"3"]
    )
    def test_what_is_not_a_json_object_is_refused(self, data):
        assert refused(lambda: records.decode_custom(data)).startswith(
            "not a JSON record: "
        )


class TestRefusedReadings:
    def test_unknown_series(self, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(two_arm_robot, "arm", [1.0], time_ns=1)
        ) == (
            "robot 'two_arm_example' has no series 'arm' (it has: left_arm, right_arm, waist, "
            "left_wrist_force, right_wrist_force, status)"
        )

    def test_the_description_is_not_a_series_to_publish(self, six_joint_arm):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "robot_description", [], time_ns=1
            )
        ) == (
            "robot 'rebot_b601' has no series 'robot_description' (it has: arm, status)"
        )

    @pytest.mark.parametrize("count", [0, 6, 8])
    def test_count_of_joints(self, six_joint_arm, count):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "arm", [0.0] * count, time_ns=1
            )
        ) == (
            f"series 'arm' takes 7 values (joint1, joint2, joint3, joint4, joint5, joint6, gripper), got {count}"
        )

    def test_count_of_readings_and_fields(self, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "status", [True, ""], time_ns=1
            )
        ) == ("series 'status' takes 3 values (motors_on, fault, battery), got 2")
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "left_wrist_force", [1, 2, 3], time_ns=1
            )
        ) == (
            "series 'left_wrist_force' takes 6 values (fx, fy, fz, tx, ty, tz), got 3"
        )
        assert refused(
            lambda: records.encode_reading(two_arm_robot, "status", "ok", time_ns=1)
        ) == ("series 'status' takes 3 values (motors_on, fault, battery), got 1")

    @pytest.mark.parametrize("value", [0.5, 1, None, True, np.float32(0.5)])
    def test_a_single_value_for_a_series_of_several(self, six_joint_arm, value):
        assert refused(
            lambda: records.encode_reading(six_joint_arm, "arm", value, time_ns=1)
        ) == (
            "series 'arm' takes 7 values (joint1, joint2, joint3, joint4, joint5, joint6, gripper), got 1"
        )
        assert refused(
            lambda: records.encode_reading(six_joint_arm, "status", value, time_ns=1)
        ) == ("series 'status' takes 2 values (motors_on, fault), got 1")

    def test_a_single_value_beside_a_series_of_several(self, full):
        assert refused(
            lambda: records.encode_reading(
                full, "arm", [0.0] * 5, velocities=0.0, efforts=[0.0] * 5, time_ns=1
            )
        ) == ("series 'arm': velocities takes 5 values, got 1")
        assert refused(
            lambda: records.encode_reading(
                full, "arm", [0.0] * 5, velocities=[0.0] * 5, efforts=1.5, time_ns=1
            )
        ) == ("series 'arm': efforts takes 5 values, got 1")

    @pytest.mark.parametrize(
        ("argument", "flag"),
        [("velocities", "has_velocity"), ("efforts", "has_effort")],
    )
    def test_length_of_velocities_and_efforts(self, full, argument, flag):
        given = {"velocities": [0.0] * 5, "efforts": [0.0] * 5} | {argument: [0.0] * 4}

        assert refused(
            lambda: records.encode_reading(full, "arm", [0.0] * 5, time_ns=1, **given)
        ) == (f"series 'arm': {argument} takes 5 values, got 4")

    def test_length_of_valid(self, six_joint_arm):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "arm", [0.0] * 7, valid=[True], time_ns=1
            )
        ) == ("series 'arm': valid takes 7 values, got 1")

    def test_velocities_or_efforts_not_declared(self, six_joint_arm, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "arm", [0.0] * 7, velocities=[0.0] * 7, time_ns=1
            )
        ) == ("series 'arm' does not declare has_velocity")
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "arm", [0.0] * 7, efforts=[0.0] * 7, time_ns=1
            )
        ) == ("series 'arm' does not declare has_effort")
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot,
                "left_arm",
                [0.0] * 5,
                velocities=[0.0] * 5,
                efforts=[0.0] * 5,
                time_ns=1,
            )
        ) == ("series 'left_arm' does not declare has_velocity")

    def test_declared_but_not_given(self, full, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "left_arm", [0.0] * 5, time_ns=1
            )
        ) == ("series 'left_arm' declares has_effort: efforts are required")
        assert refused(
            lambda: records.encode_reading(
                full, "arm", [0.0] * 5, efforts=[0.0] * 5, time_ns=1
            )
        ) == ("series 'arm' declares has_velocity: velocities are required")

    @pytest.mark.parametrize(
        "value",
        [math.nan, math.inf, -math.inf, None, "0.5", True, np.bool_(True), 1e39, [1.0]],
    )
    def test_a_position_that_is_not_a_finite_number(self, six_joint_arm, value):
        values = [0.0, 0.0, value, 0.0, 0.0, 0.0, 0.0]

        assert refused(
            lambda: records.encode_reading(six_joint_arm, "arm", values, time_ns=1)
        ) == ("series 'arm': 'joint3' is not a finite number")

    def test_a_velocity_or_effort_that_is_not_a_finite_number(self, full):
        good = [0.0] * 5
        bad = [0.0, math.nan, 0.0, 0.0, 0.0]

        for given in (
            {"velocities": bad, "efforts": good},
            {"velocities": good, "efforts": bad},
        ):
            assert refused(
                lambda given=given: records.encode_reading(
                    full, "arm", good, time_ns=1, **given
                )
            ) == ("series 'arm': 'turn' is not a finite number")

    def test_a_status_number_that_is_not_finite(self, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "status", [True, "", math.nan], time_ns=1
            )
        ) == ("series 'status': 'battery' is not a finite number")
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "status", [math.inf, "", 24.0], time_ns=1
            )
        ) == ("series 'status': 'motors_on' is not a finite number")

    @pytest.mark.parametrize(
        ("values", "text"),
        [
            ([1, "24.5", True, ""], "reading 'battery' is declared number, got str"),
            ([1, True, True, ""], "reading 'battery' is declared number, got bool"),
            ([1, 24.5, 1, ""], "reading 'motors_on' is declared flag, got int"),
            ([1, 24.5, "on", ""], "reading 'motors_on' is declared flag, got str"),
            ([1, 24.5, True, 3], "reading 'mode' is declared text, got int"),
            ([1, 24.5, True, False], "reading 'mode' is declared text, got bool"),
            (
                [1, np.bool_(True), True, ""],
                "reading 'battery' is declared number, got bool",
            ),
            (
                [1, 24.5, np.int64(1), ""],
                "reading 'motors_on' is declared flag, got int64",
            ),
            ([[1, 2], 24.5, True, ""], "reading 'anything' is declared any, got list"),
            (
                [{"a": 1}, 24.5, True, ""],
                "reading 'anything' is declared any, got dict",
            ),
            ([b"raw", 24.5, True, ""], "reading 'anything' is declared any, got bytes"),
        ],
    )
    def test_a_status_value_against_its_declared_kind(self, full, values, text):
        assert (
            refused(lambda: records.encode_reading(full, "status", values, time_ns=1))
            == f"series 'status': {text}"
        )

    @pytest.mark.parametrize(
        ("series", "values", "argument", "shape"),
        [
            ("arm", [0.0] * 7, "faults", "joints"),
            ("status", [True, ""], "velocities", "status"),
            ("status", [True, ""], "efforts", "status"),
            ("status", [True, ""], "valid", "status"),
        ],
    )
    def test_an_argument_of_another_shape(
        self, six_joint_arm, series, values, argument, shape
    ):
        given = {argument: [("a", "b")] if argument == "faults" else [1.0, 2.0]}

        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, series, values, time_ns=1, **given
            )
        ) == (f"series '{series}' has shape '{shape}': {argument} does not apply")

    @pytest.mark.parametrize("argument", ["velocities", "efforts", "valid", "faults"])
    def test_an_argument_on_a_shape_the_robot_defines(self, two_arm_robot, argument):
        assert (
            refused(
                lambda: records.encode_reading(
                    two_arm_robot,
                    "left_wrist_force",
                    [0.0] * 6,
                    time_ns=1,
                    **{argument: []},
                )
            )
            == f"series 'left_wrist_force' has shape 'force': {argument} does not apply"
        )

    def test_a_custom_record_that_lacks_fields(self, two_arm_robot):
        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "left_wrist_force", {"fx": 1.0, "tz": 2.0}, time_ns=1
            )
        ) == ("series 'left_wrist_force': record lacks fy, fz, tx, ty")

    def test_a_custom_record_with_unknown_fields(self, two_arm_robot):
        record = dict.fromkeys(("fx", "fy", "fz", "tx", "ty", "tz"), 0.0) | {
            "mx": 1.0,
            "my": 2.0,
        }

        assert refused(
            lambda: records.encode_reading(
                two_arm_robot, "left_wrist_force", record, time_ns=1
            )
        ) == ("series 'left_wrist_force': record has unknown fields mx, my")

    def test_joints_and_status_by_name_are_checked_the_same_way(self, six_joint_arm):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "status", {"fault": ""}, time_ns=1
            )
        ) == ("series 'status': record lacks motors_on")
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm,
                "status",
                {"motors_on": True, "fault": "", "battery": 1},
                time_ns=1,
            )
        ) == ("series 'status': record has unknown fields battery")

    @pytest.mark.parametrize(
        ("field", "value", "kind", "got"),
        [
            ("depth", "deep", "number", "str"),
            ("depth", True, "number", "bool"),
            ("depth", np.bool_(True), "number", "bool"),
            ("depth", None, "number", "NoneType"),
            ("touching", 1, "flag", "int"),
            ("label", 5, "text", "int"),
            ("trace", 5.0, "numbers", "float"),
            ("trace", "12", "numbers", "str"),
        ],
    )
    def test_a_custom_field_against_its_declared_kind(
        self, full, field, value, kind, got
    ):
        record = {
            "depth": 1.0,
            "touching": True,
            "label": "",
            "trace": [],
            "extra": None,
        } | {field: value}

        assert refused(
            lambda: records.encode_reading(full, "probe", record, time_ns=1)
        ) == (f"series 'probe': field '{field}' is declared {kind}, got {got}")

    @pytest.mark.parametrize(
        ("field", "value"),
        [("depth", math.nan), ("trace", [1.0, math.inf]), ("trace", ["a"])],
    )
    def test_a_custom_number_that_is_not_finite(self, full, field, value):
        record = {
            "depth": 1.0,
            "touching": True,
            "label": "",
            "trace": [],
            "extra": None,
        } | {field: value}

        assert refused(
            lambda: records.encode_reading(full, "probe", record, time_ns=1)
        ) == (f"series 'probe': '{field}' is not a finite number")

    @pytest.mark.parametrize("value", [math.nan, {1, 2}, b"raw", object()])
    def test_a_custom_value_json_cannot_carry(self, full, value):
        record = {
            "depth": 1.0,
            "touching": True,
            "label": "",
            "trace": [],
            "extra": value,
        }

        assert refused(
            lambda: records.encode_reading(full, "probe", record, time_ns=1)
        ).startswith("series 'probe': record is not JSON: ")

    @pytest.mark.parametrize(
        "fault",
        ["overtemperature", ("code",), ("a", "b", "c", "d", "e"), ("a", 5), 7, None],
    )
    def test_a_fault_of_another_form(self, six_joint_arm, fault):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm, "status", [True, ""], faults=[fault], time_ns=1
            )
        ) == (
            "series 'status': a fault is (code, text), (code, text, source) or (code, text, source, severity)"
        )

    def test_a_fault_severity_that_is_not_one(self, six_joint_arm):
        assert refused(
            lambda: records.encode_reading(
                six_joint_arm,
                "status",
                [True, ""],
                faults=[("a", "b", "", "critical")],
                time_ns=1,
            )
        ) == (
            "series 'status': fault severity 'critical' is not one of warning, error, fatal"
        )
