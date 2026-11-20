# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RobotDescription types in isaaccapture.schema.

- RobotJointDescription, RobotGripperDescription, RobotValueDescription, RobotTextPair
- RobotSeriesDescription: one data series (joints, status, or a shape the robot defines)
- RobotDescription and its record wrapper RobotDescriptionRecord
- robot_state_record_schema: the record schema of each standard shape
"""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    RobotDescription,
    RobotDescriptionRecord,
    RobotGripperDescription,
    RobotJointDescription,
    RobotJointKind,
    RobotSeriesDescription,
    RobotSeriesShape,
    RobotTextPair,
    RobotValueDescription,
    RobotValueKind,
    robot_state_record_schema,
)


def _description() -> RobotDescription:
    arm = RobotSeriesDescription(
        name="arm",
        shape=RobotSeriesShape.JOINTS,
        shape_name="joints",
        rate_hz=20.0,
        stale_after_s=0.5,
        joints=[
            RobotJointDescription(
                name="shoulder",
                kind=RobotJointKind.REVOLUTE,
                has_limits=True,
                min_position=-2.8,
                max_position=2.8,
                max_velocity=2.0,
                max_effort=40.0,
            ),
            RobotJointDescription(name="roll", kind=RobotJointKind.CONTINUOUS),
            RobotJointDescription(
                name="jaw",
                kind=RobotJointKind.GRIPPER,
                has_limits=True,
                max_position=0.08,
                unit="m",
            ),
        ],
        has_velocity=True,
        has_effort=True,
    )
    status = RobotSeriesDescription(
        name="status",
        shape=RobotSeriesShape.STATUS,
        shape_name="status",
        stale_after_s=2.0,
        values=[
            RobotValueDescription(name="motors_on"),
            RobotValueDescription(name="battery", kind=RobotValueKind.NUMBER, unit="V"),
        ],
    )
    force = RobotSeriesDescription(
        name="wrist_force",
        shape=RobotSeriesShape.CUSTOM,
        shape_name="force",
        values=[RobotValueDescription("fz", RobotValueKind.NUMBER, "N")],
        attributes=[RobotTextPair("at", "wrist")],
    )
    return RobotDescription(
        robot="test_arm",
        series=[arm, status, force],
        grippers=[
            RobotGripperDescription(
                name="jaw",
                series="arm",
                joints=["jaw", "roll"],
                open_position=0.08,
                closed_position=0.0,
            )
        ],
        attributes=[RobotTextPair(key="maker", value="nobody")],
    )


class TestRobotJointDescription:
    def test_defaults(self):
        joint = RobotJointDescription()

        assert joint.name == ""
        assert joint.kind == RobotJointKind.REVOLUTE
        assert joint.has_limits is False
        assert joint.min_position == 0.0
        assert joint.max_position == 0.0
        assert joint.max_velocity == 0.0
        assert joint.max_effort == 0.0
        assert joint.unit == ""

    def test_every_field_reads_back(self):
        joint = RobotJointDescription(
            "slide", RobotJointKind.PRISMATIC, True, -0.1, 0.4, 0.5, 80.0, "m"
        )

        assert joint.name == "slide"
        assert joint.kind == RobotJointKind.PRISMATIC
        assert joint.has_limits is True
        assert joint.min_position == -0.1
        assert joint.max_position == 0.4
        assert joint.max_velocity == 0.5
        assert joint.max_effort == 80.0
        assert joint.unit == "m"
        assert "slide" in repr(joint)

    def test_is_read_only(self):
        joint = RobotJointDescription(name="slide")

        with pytest.raises(AttributeError):
            joint.name = "other"

    def test_kind_values(self):
        assert int(RobotJointKind.REVOLUTE) == 0
        assert int(RobotJointKind.PRISMATIC) == 1
        assert int(RobotJointKind.CONTINUOUS) == 2
        assert int(RobotJointKind.GRIPPER) == 3


class TestRobotGripperDescription:
    def test_defaults(self):
        gripper = RobotGripperDescription()

        assert gripper.name == ""
        assert gripper.series == ""
        assert gripper.joints == []
        assert gripper.open_position == 0.0
        assert gripper.closed_position == 0.0

    def test_every_field_reads_back(self):
        gripper = RobotGripperDescription("jaw", "arm", ["left", "right"], 0.08, 0.01)

        assert gripper.name == "jaw"
        assert gripper.series == "arm"
        assert gripper.joints == ["left", "right"]
        assert gripper.open_position == 0.08
        assert gripper.closed_position == 0.01
        assert "jaw" in repr(gripper)


class TestRobotValueDescriptionAndTextPair:
    def test_value_defaults(self):
        value = RobotValueDescription()

        assert value.name == ""
        assert value.kind == RobotValueKind.ANY
        assert value.unit == ""

    def test_value_reads_back(self):
        value = RobotValueDescription("raw", RobotValueKind.NUMBERS, "N")

        assert value.name == "raw"
        assert value.kind == RobotValueKind.NUMBERS
        assert value.unit == "N"
        assert "raw" in repr(value)

    def test_value_kind_values(self):
        assert int(RobotValueKind.ANY) == 0
        assert int(RobotValueKind.NUMBER) == 1
        assert int(RobotValueKind.FLAG) == 2
        assert int(RobotValueKind.TEXT) == 3
        assert int(RobotValueKind.NUMBERS) == 4

    def test_text_pair(self):
        pair = RobotTextPair("at", "wrist")

        assert pair.key == "at"
        assert pair.value == "wrist"
        assert RobotTextPair().key == ""
        assert RobotTextPair().value == ""
        assert "at" in repr(pair)


class TestRobotSeriesDescription:
    def test_defaults(self):
        series = RobotSeriesDescription()

        assert series.name == ""
        assert series.shape == RobotSeriesShape.JOINTS
        assert series.shape_name == ""
        assert series.rate_hz == 0.0
        assert series.stale_after_s == 0.0
        assert series.joints == []
        assert series.has_velocity is False
        assert series.has_effort is False
        assert series.values == []
        assert series.attributes == []

    def test_joints_series_reads_back_in_order(self):
        arm = _description().series[0]

        assert arm.name == "arm"
        assert arm.shape == RobotSeriesShape.JOINTS
        assert arm.shape_name == "joints"
        assert arm.rate_hz == 20.0
        assert arm.stale_after_s == 0.5
        assert arm.has_velocity is True
        assert arm.has_effort is True
        assert [joint.name for joint in arm.joints] == ["shoulder", "roll", "jaw"]
        assert arm.joints[0].min_position == -2.8
        assert arm.joints[1].has_limits is False
        assert arm.joints[2].kind == RobotJointKind.GRIPPER
        assert arm.joints[2].unit == "m"
        assert "arm" in repr(arm)

    def test_status_and_custom_series_read_back(self):
        _, status, force = _description().series

        assert status.shape == RobotSeriesShape.STATUS
        assert [value.name for value in status.values] == ["motors_on", "battery"]
        assert status.values[1].kind == RobotValueKind.NUMBER
        assert status.values[1].unit == "V"
        assert force.shape == RobotSeriesShape.CUSTOM
        assert force.shape_name == "force"
        assert force.attributes[0].key == "at"
        assert force.attributes[0].value == "wrist"

    def test_shape_values(self):
        assert int(RobotSeriesShape.JOINTS) == 0
        assert int(RobotSeriesShape.STATUS) == 1
        assert int(RobotSeriesShape.CUSTOM) == 2


class TestRobotDescription:
    def test_defaults(self):
        description = RobotDescription()

        assert description.robot == ""
        assert description.series == []
        assert description.grippers == []
        assert description.attributes == []

    def test_every_field_reads_back(self):
        description = _description()

        assert description.robot == "test_arm"
        assert [series.name for series in description.series] == [
            "arm",
            "status",
            "wrist_force",
        ]
        assert description.grippers[0].name == "jaw"
        assert description.grippers[0].joints == ["jaw", "roll"]
        assert description.attributes[0].key == "maker"
        assert description.attributes[0].value == "nobody"
        assert repr(description) == "RobotDescription(robot=test_arm, series=3)"


class TestRobotDescriptionRecord:
    def test_a_record_without_a_description(self):
        record = RobotDescriptionRecord(None, DeviceDataTimestamp(1, 1, 0))

        assert record.data is None
        assert RobotDescriptionRecord.from_bytes(record.to_bytes()).data is None

    def test_record_carries_description_and_timestamp(self):
        record = RobotDescriptionRecord(_description(), DeviceDataTimestamp(11, 10, 0))

        assert record.data.robot == "test_arm"
        assert record.timestamp.available_time_local_common_clock == 11
        assert record.timestamp.sample_time_local_common_clock == 10

    def test_bytes_round_trip(self):
        data = RobotDescriptionRecord(
            _description(), DeviceDataTimestamp(11, 10, 0)
        ).to_bytes()
        record = RobotDescriptionRecord.from_bytes(data)

        assert record.to_bytes() == data
        assert record.data.robot == "test_arm"
        assert record.data.series[0].joints[2].name == "jaw"
        assert record.data.grippers[0].open_position == 0.08
        assert record.timestamp.sample_time_local_common_clock == 10

    def test_nested_table_round_trips_on_its_own(self):
        record = RobotDescriptionRecord(_description(), DeviceDataTimestamp(11, 10, 0))
        description = RobotDescription.from_bytes(record.data.to_bytes())

        assert description.robot == "test_arm"
        assert len(description.series) == 3

    @pytest.mark.parametrize(
        "junk", [b"", b"\x00", b"not a flatbuffer", b"\xff" * 64, b"\x00" * 64]
    )
    def test_from_bytes_refuses_junk(self, junk):
        with pytest.raises(
            ValueError, match="not a valid RobotDescriptionRecord buffer"
        ):
            RobotDescriptionRecord.from_bytes(junk)

    def test_from_bytes_refuses_a_truncated_record(self):
        data = RobotDescriptionRecord(
            _description(), DeviceDataTimestamp(11, 10, 0)
        ).to_bytes()

        with pytest.raises(
            ValueError, match="not a valid RobotDescriptionRecord buffer"
        ):
            RobotDescriptionRecord.from_bytes(data[:40])


class TestRobotStateRecordSchema:
    @pytest.mark.parametrize(
        ("shape", "name"),
        [
            ("joints", "core.JointStateOutputRecord"),
            ("status", "core.RobotStatusRecord"),
            ("description", "core.RobotDescriptionRecord"),
        ],
    )
    def test_standard_shapes(self, shape, name):
        schema_name, data = robot_state_record_schema(shape)

        assert schema_name == name
        assert isinstance(data, bytes)
        # A binary FlatBuffers schema carries the file identifier "BFBS" after its root offset.
        assert data[4:8] == b"BFBS"

    def test_the_three_schemas_differ(self):
        schemas = {
            robot_state_record_schema(shape)[1]
            for shape in ("joints", "status", "description")
        }

        assert len(schemas) == 3

    @pytest.mark.parametrize("shape", ["", "force", "JOINTS", "custom"])
    def test_another_shape_is_refused(self, shape):
        with pytest.raises(ValueError, match="joints, status or description"):
            robot_state_record_schema(shape)
