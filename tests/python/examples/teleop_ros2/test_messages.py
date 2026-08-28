# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for teleoperation publication-output builders."""

import msgpack
import numpy as np
from builtin_interfaces.msg import Time
from std_msgs.msg import ByteMultiArray

from isaaccapture.retargeting_engine.interface import (
    OptionalTensorGroup,
    TensorGroup,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    NDArrayType,
    NUM_BODY_JOINTS,
    NUM_HAND_JOINTS,
    ControllerInput,
    ControllerInputIndex,
    FullBodyInput,
    FullBodyInputIndex,
    HandInput,
    HandInputIndex,
    HandJointIndex,
    HeadInput,
    HeadInputIndex,
    RobotHandJoints,
)

from isaaccapture_examples.teleop_ros2.constants import (
    BODY_JOINT_NAMES,
    HAND_POSE_NAMES,
    EePoseFrame,
)
from isaaccapture_examples.teleop_ros2.messages import (
    build_controller_msg,
    build_ee_output_from_controllers,
    build_ee_output_from_hands,
    build_finger_joints_msg,
    build_full_body_msg,
    build_hand_msg,
    build_head_output,
    build_root_command_output,
)


def _active_controller() -> TensorGroup:
    controller = TensorGroup(ControllerInput())
    controller[ControllerInputIndex.GRIP_POSITION] = np.array(
        [1.0, 2.0, 3.0], dtype=np.float32
    )
    controller[ControllerInputIndex.GRIP_ORIENTATION] = np.array(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    controller[ControllerInputIndex.GRIP_IS_VALID] = True
    controller[ControllerInputIndex.AIM_POSITION] = np.array(
        [4.0, 5.0, 6.0], dtype=np.float32
    )
    controller[ControllerInputIndex.AIM_ORIENTATION] = np.array(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    controller[ControllerInputIndex.AIM_IS_VALID] = True
    controller[ControllerInputIndex.PRIMARY_CLICK] = 1.0
    controller[ControllerInputIndex.SECONDARY_CLICK] = 0.0
    controller[ControllerInputIndex.THUMBSTICK_CLICK] = 1.0
    controller[ControllerInputIndex.MENU_CLICK] = 0.0
    controller[ControllerInputIndex.THUMBSTICK_X] = 0.25
    controller[ControllerInputIndex.THUMBSTICK_Y] = -0.5
    controller[ControllerInputIndex.SQUEEZE_VALUE] = 0.75
    controller[ControllerInputIndex.TRIGGER_VALUE] = 0.5
    return controller


def _active_full_body() -> TensorGroup:
    full_body = TensorGroup(FullBodyInput())
    positions = np.zeros((NUM_BODY_JOINTS, 3), dtype=np.float32)
    positions[0] = [1.0, 2.0, 3.0]
    orientations = np.zeros((NUM_BODY_JOINTS, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    valid = np.zeros(NUM_BODY_JOINTS, dtype=np.uint8)
    valid[0] = 1

    full_body[FullBodyInputIndex.JOINT_POSITIONS] = positions
    full_body[FullBodyInputIndex.JOINT_ORIENTATIONS] = orientations
    full_body[FullBodyInputIndex.JOINT_VALID] = valid
    return full_body


def _active_hand() -> TensorGroup:
    hand = TensorGroup(HandInput())
    positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    positions[:, 0] = np.arange(NUM_HAND_JOINTS, dtype=np.float32)
    orientations = np.zeros((NUM_HAND_JOINTS, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    valid = np.ones(NUM_HAND_JOINTS, dtype=np.uint8)

    hand[HandInputIndex.JOINT_POSITIONS] = positions
    hand[HandInputIndex.JOINT_ORIENTATIONS] = orientations
    hand[HandInputIndex.JOINT_RADII] = np.ones(NUM_HAND_JOINTS, dtype=np.float32)
    hand[HandInputIndex.JOINT_VALID] = valid
    return hand


def _active_head() -> TensorGroup:
    head = TensorGroup(HeadInput())
    head[HeadInputIndex.POSITION] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    head[HeadInputIndex.ORIENTATION] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    head[HeadInputIndex.IS_VALID] = True
    head[HeadInputIndex.IS_TRACKED] = True
    return head


def _decode_payload(msg: ByteMultiArray) -> dict:
    return msgpack.unpackb(b"".join(msg.data), raw=False)


def _position(pose) -> list[float]:
    return [pose.position.x, pose.position.y, pose.position.z]


def _root_command(values: list[float] | None) -> OptionalTensorGroup:
    group_type = TensorGroupType(
        "root_command",
        [
            NDArrayType(
                "command",
                shape=(4,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            )
        ],
    )
    root_command = OptionalTensorGroup(group_type)
    if values is not None:
        root_command[0] = np.asarray(values, dtype=np.float32)
    return root_command


def test_build_controller_msg_encodes_schema_and_absent_side_defaults() -> None:
    msg = build_controller_msg(
        _active_controller(), OptionalTensorGroup(ControllerInput())
    )

    assert msg is not None
    payload = _decode_payload(msg)
    assert set(payload) == {
        "timestamp",
        "left_thumbstick",
        "right_thumbstick",
        "left_trigger_value",
        "right_trigger_value",
        "left_squeeze_value",
        "right_squeeze_value",
        "left_aim_position",
        "right_aim_position",
        "left_grip_position",
        "right_grip_position",
        "left_aim_orientation",
        "right_aim_orientation",
        "left_grip_orientation",
        "right_grip_orientation",
        "left_primary_click",
        "right_primary_click",
        "left_secondary_click",
        "right_secondary_click",
        "left_thumbstick_click",
        "right_thumbstick_click",
        "left_menu_click",
        "right_menu_click",
        "left_is_active",
        "right_is_active",
    }
    assert isinstance(payload["timestamp"], int)
    assert payload["left_is_active"] is True
    assert payload["right_is_active"] is False
    assert payload["left_thumbstick"] == [0.25, -0.5]
    assert payload["right_aim_position"] == [0.0, 0.0, 0.0]
    assert payload["right_aim_orientation"] == [0.0, 0.0, 0.0, 1.0]


def test_build_controller_msg_returns_none_when_both_sides_absent() -> None:
    assert (
        build_controller_msg(
            OptionalTensorGroup(ControllerInput()),
            OptionalTensorGroup(ControllerInput()),
        )
        is None
    )


def test_build_ee_output_from_controllers_keeps_message_and_tfs_consistent() -> None:
    msg, transforms = build_ee_output_from_controllers(
        _active_controller(),
        OptionalTensorGroup(ControllerInput()),
        Time(sec=12, nanosec=34),
        "world",
        "left_wrist",
        "right_wrist",
    )

    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 34
    assert msg.header.frame_id == "world"
    assert list(msg.name) == ["left", "right"]
    assert list(msg.is_valid) == [True, False]
    assert msg.pose[0].position.x == 4.0
    assert msg.pose[1].position.x == 0.0
    assert msg.pose[1].orientation.w == 1.0
    assert len(transforms) == 1
    transform = transforms[0]
    assert transform.header.stamp.sec == 12
    assert transform.header.stamp.nanosec == 34
    assert transform.header.frame_id == "world"
    assert transform.child_frame_id == "left_wrist"
    assert transform.transform.translation.x == 4.0
    assert transform.transform.translation.y == 5.0
    assert transform.transform.translation.z == 6.0
    assert transform.transform.rotation.w == 1.0


def test_build_ee_output_from_controllers_preserves_manus_calibration() -> None:
    msg, transforms = build_ee_output_from_controllers(
        _active_controller(),
        OptionalTensorGroup(ControllerInput()),
        Time(),
        "world",
        "left_wrist",
        "right_wrist",
        apply_manus_controller_mount_offset=True,
    )

    expected_position = [4.0103315472, 5.055536544, 5.9433523928]
    expected_orientation = [
        -0.1315856570103413,
        -0.3586609382547703,
        0.9111816820492541,
        -0.15425786377769118,
    ]
    actual_pose = msg.pose[0]
    actual_transform = transforms[0].transform

    np.testing.assert_allclose(
        [
            actual_pose.position.x,
            actual_pose.position.y,
            actual_pose.position.z,
        ],
        expected_position,
    )
    np.testing.assert_allclose(
        [
            actual_pose.orientation.x,
            actual_pose.orientation.y,
            actual_pose.orientation.z,
            actual_pose.orientation.w,
        ],
        expected_orientation,
    )
    np.testing.assert_allclose(
        [
            actual_transform.translation.x,
            actual_transform.translation.y,
            actual_transform.translation.z,
        ],
        expected_position,
    )


def test_build_ee_output_from_controllers_in_the_head_frame() -> None:
    right = _active_controller()
    right[ControllerInputIndex.AIM_POSITION] = np.array(
        [2.0, 4.0, 6.0], dtype=np.float32
    )
    msg, transforms = build_ee_output_from_controllers(
        _active_controller(),
        right,
        Time(sec=12, nanosec=34),
        "world",
        "left_wrist",
        "right_wrist",
        head=_active_head(),
        head_frame="headset",
        ee_poses_frame=EePoseFrame.HEAD,
    )

    assert msg.header.frame_id == "headset"
    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 34
    assert list(msg.name) == ["left", "right"]
    assert list(msg.is_valid) == [True, True]
    np.testing.assert_allclose(_position(msg.pose[0]), [3.0, 3.0, 3.0])
    np.testing.assert_allclose(_position(msg.pose[1]), [1.0, 2.0, 3.0])
    assert [transform.header.frame_id for transform in transforms] == [
        "headset",
        "headset",
    ]
    assert [transform.child_frame_id for transform in transforms] == [
        "left_wrist",
        "right_wrist",
    ]


def test_build_ee_output_from_hands_in_head_frame_keeps_untracked_sides_invalid() -> (
    None
):
    msg, transforms = build_ee_output_from_hands(
        _active_hand(),
        OptionalTensorGroup(HandInput()),
        Time(),
        "world",
        "left_wrist",
        "right_wrist",
        head=_active_head(),
        head_frame="headset",
        ee_poses_frame=EePoseFrame.HEAD,
    )

    assert msg.header.frame_id == "headset"
    assert list(msg.is_valid) == [True, False]
    np.testing.assert_allclose(_position(msg.pose[1]), [0.0, 0.0, 0.0])
    assert msg.pose[1].orientation.w == 1.0
    assert len(transforms) == 1
    assert transforms[0].header.frame_id == "headset"
    assert transforms[0].child_frame_id == "left_wrist"


def test_build_ee_output_from_hands_uses_valid_wrist_entries() -> None:
    msg, transforms = build_ee_output_from_hands(
        _active_hand(),
        OptionalTensorGroup(HandInput()),
        Time(),
        "world",
        "left_wrist",
        "right_wrist",
    )

    wrist_x = float(HandJointIndex.WRIST)
    assert list(msg.name) == ["left", "right"]
    assert list(msg.is_valid) == [True, False]
    assert msg.pose[0].position.x == wrist_x
    assert len(transforms) == 1
    assert transforms[0].child_frame_id == "left_wrist"
    assert transforms[0].transform.translation.x == wrist_x


def test_build_finger_joints_msg_combines_present_sides() -> None:
    left = TensorGroup(
        RobotHandJoints("left_finger_joints", ["left_index", "left_thumb"])
    )
    left[0] = 1.0
    left[1] = 2.0
    right_type = RobotHandJoints("right_finger_joints", ["right_index"])

    msg = build_finger_joints_msg(
        left,
        OptionalTensorGroup(right_type),
        Time(sec=12, nanosec=34),
        "world",
    )

    assert msg is not None
    assert msg.header.stamp.sec == 12
    assert msg.header.stamp.nanosec == 34
    assert msg.header.frame_id == "world"
    assert list(msg.name) == ["left_index", "left_thumb"]
    assert list(msg.position) == [1.0, 2.0]


def test_build_finger_joints_msg_returns_none_when_both_sides_are_absent() -> None:
    left_type = RobotHandJoints("left_finger_joints", ["left_index"])
    right_type = RobotHandJoints("right_finger_joints", ["right_index"])

    assert (
        build_finger_joints_msg(
            OptionalTensorGroup(left_type),
            OptionalTensorGroup(right_type),
            Time(),
            "world",
        )
        is None
    )


def test_build_full_body_msg_encodes_schema() -> None:
    msg = build_full_body_msg(_active_full_body())

    assert msg is not None
    payload = _decode_payload(msg)
    assert set(payload) == {
        "timestamp",
        "joint_names",
        "joint_positions",
        "joint_orientations",
        "joint_valid",
    }
    assert isinstance(payload["timestamp"], int)
    assert payload["joint_names"] == BODY_JOINT_NAMES
    assert payload["joint_positions"][0] == [1.0, 2.0, 3.0]
    assert payload["joint_orientations"][0] == [0.0, 0.0, 0.0, 1.0]
    assert payload["joint_valid"][0] is True
    assert not any(payload["joint_valid"][1:])


def test_build_full_body_msg_returns_none_when_input_is_absent() -> None:
    assert build_full_body_msg(OptionalTensorGroup(FullBodyInput())) is None


def test_build_hand_msg_keeps_stable_names_and_absent_side_placeholders() -> None:
    msg = build_hand_msg(
        _active_hand(),
        OptionalTensorGroup(HandInput()),
        Time(),
        "world",
    )

    expected_names = [
        f"{side}_{name}" for side in ("left", "right") for name in HAND_POSE_NAMES
    ]
    assert list(msg.name) == expected_names
    assert len(msg.pose) == len(expected_names)
    assert all(msg.is_valid[: len(HAND_POSE_NAMES)])
    assert not any(msg.is_valid[len(HAND_POSE_NAMES) :])
    assert "left_PALM" not in msg.name


def test_build_head_output_keeps_pose_and_tf_consistent() -> None:
    output = build_head_output(
        _active_head(),
        Time(sec=12, nanosec=34),
        "world",
        "head",
    )

    assert output is not None
    head_msg, transform = output
    assert head_msg.header.stamp.sec == 12
    assert head_msg.header.stamp.nanosec == 34
    assert head_msg.header.frame_id == "world"
    assert head_msg.pose.position.x == 1.0
    assert head_msg.pose.position.y == 2.0
    assert head_msg.pose.position.z == 3.0
    assert head_msg.pose.orientation.w == 1.0
    assert transform.header.stamp.sec == 12
    assert transform.header.stamp.nanosec == 34
    assert transform.header.frame_id == "world"
    assert transform.child_frame_id == "head"
    assert transform.transform.translation.x == 1.0
    assert transform.transform.translation.y == 2.0
    assert transform.transform.translation.z == 3.0
    assert transform.transform.rotation.w == 1.0


def test_build_head_output_returns_none_when_head_is_absent() -> None:
    assert (
        build_head_output(
            OptionalTensorGroup(HeadInput()),
            Time(),
            "world",
            "head",
        )
        is None
    )


def test_build_root_command_output_maps_twist_and_height() -> None:
    output = build_root_command_output(
        _root_command([1.0, -2.0, 3.0, 0.75]),
        Time(sec=12, nanosec=34),
        "world",
    )

    assert output is not None
    twist_msg, pose_msg = output
    assert twist_msg.header.stamp.sec == 12
    assert twist_msg.header.stamp.nanosec == 34
    assert twist_msg.header.frame_id == "world"
    assert twist_msg.twist.linear.x == 1.0
    assert twist_msg.twist.linear.y == -2.0
    assert twist_msg.twist.linear.z == 0.0
    assert twist_msg.twist.angular.z == 3.0
    assert pose_msg.header.stamp.sec == 12
    assert pose_msg.header.stamp.nanosec == 34
    assert pose_msg.header.frame_id == "world"
    assert pose_msg.pose.position.z == 0.75
    assert pose_msg.pose.orientation.w == 1.0


def test_build_root_command_output_returns_none_when_input_is_absent() -> None:
    assert build_root_command_output(_root_command(None), Time(), "world") is None
