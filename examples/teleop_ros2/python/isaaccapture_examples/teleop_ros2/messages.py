# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure publication-output builders for teleop_ros2_node."""

import time
from typing import Dict, Sequence

import msgpack
import msgpack_numpy as mnp
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import ByteMultiArray
from teleop_ros2_interfaces.msg import NamedPoseArray

from isaaccapture.retargeting_engine.interface import OptionalTensorGroup
from isaaccapture.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    FullBodyInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)

from .constants import (
    BODY_JOINT_NAMES,
    HAND_POSE_JOINT_INDICES,
    HAND_POSE_NAMES,
    EePoseFrame,
)
from .geometry import (
    apply_manus_controller_to_hand_pose,
    apply_relative_pose,
    apply_transform_to_pose,
    make_transform,
    to_pose,
)
from .tensor_group_helpers import (
    controller_aim_is_valid,
    hand_wrist_is_valid,
    head_is_valid,
    joint_names_from_group_type,
)


def _compose_ee_msg(
    left_pose: Pose | None,
    right_pose: Pose | None,
    now,
    frame_id: str,
) -> NamedPoseArray:
    msg = NamedPoseArray()
    msg.header.stamp = now
    msg.header.frame_id = frame_id
    for side, pose in (("left", left_pose), ("right", right_pose)):
        # Set the pose and validity together so the parallel fields cannot diverge.
        if pose is None:
            pose = to_pose([0.0, 0.0, 0.0])
            pose_is_valid = False
        else:
            pose_is_valid = True
        msg.name.append(side)
        msg.pose.append(pose)
        msg.is_valid.append(pose_is_valid)
    return msg


def _compute_ee_pose_from_controller(
    ctrl: OptionalTensorGroup,
    side: str,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
    apply_manus_controller_mount_offset: bool = False,
) -> Pose | None:
    if not controller_aim_is_valid(ctrl):
        return None

    pos = [float(x) for x in ctrl[ControllerInputIndex.AIM_POSITION]]
    ori = [float(x) for x in ctrl[ControllerInputIndex.AIM_ORIENTATION]]
    pose = to_pose(pos, ori)

    if transform_rot is not None or transform_trans is not None:
        pose = apply_transform_to_pose(pose, transform_rot, transform_trans)

    if apply_manus_controller_mount_offset:
        pose = apply_manus_controller_to_hand_pose(pose, side)

    return pose


def _compute_ee_pose_from_hand(
    hand: OptionalTensorGroup,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
) -> Pose | None:
    if not hand_wrist_is_valid(hand):
        return None

    positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS])
    orientations = np.asarray(hand[HandInputIndex.JOINT_ORIENTATIONS])
    pose = to_pose(
        positions[HandJointIndex.WRIST],
        orientations[HandJointIndex.WRIST],
    )
    if transform_rot is not None or transform_trans is not None:
        pose = apply_transform_to_pose(pose, transform_rot, transform_trans)
    return pose


def _compute_head_pose(
    head: OptionalTensorGroup | None,
    transform_rot: Rotation | None,
    transform_trans: Sequence[float] | None,
) -> Pose | None:
    if head is None or not head_is_valid(head):
        return None

    position = [float(x) for x in head[HeadInputIndex.POSITION]]
    orientation = [float(x) for x in head[HeadInputIndex.ORIENTATION]]
    pose = to_pose(position, orientation)
    if transform_rot is not None or transform_trans is not None:
        pose = apply_transform_to_pose(pose, transform_rot, transform_trans)
    return pose


def _finalize_ee_output(
    left_pose: Pose | None,
    right_pose: Pose | None,
    now,
    frame_id: str,
    left_wrist_frame: str,
    right_wrist_frame: str,
    *,
    head: OptionalTensorGroup | None,
    head_frame: str,
    ee_poses_frame: EePoseFrame,
    transform_rot: Rotation | None,
    transform_trans: Sequence[float] | None,
) -> tuple[NamedPoseArray, list[TransformStamped]]:
    if ee_poses_frame == EePoseFrame.HEAD:
        frame_id = head_frame
        head_pose = _compute_head_pose(head, transform_rot, transform_trans)
        if head_pose is None:
            # Tracking loss must not change coordinates or reparent wrist TFs.
            left_pose = right_pose = None
        else:
            if left_pose is not None:
                left_pose = apply_relative_pose(head_pose, left_pose)
            if right_pose is not None:
                right_pose = apply_relative_pose(head_pose, right_pose)

    ee_msg = _compose_ee_msg(left_pose, right_pose, now, frame_id)
    return ee_msg, _wrist_tfs_from_ee_msg(ee_msg, left_wrist_frame, right_wrist_frame)


def _to_msgpack_byte_multi_array(payload: Dict) -> ByteMultiArray:
    encoded_payload = msgpack.packb(payload, default=mnp.encode)
    msg = ByteMultiArray()
    msg.data = tuple(bytes([value]) for value in encoded_payload)
    return msg


def _wrist_tfs_from_ee_msg(
    ee_msg: NamedPoseArray,
    left_wrist_frame: str,
    right_wrist_frame: str,
) -> list[TransformStamped]:
    wrist_frames = {
        "left": left_wrist_frame,
        "right": right_wrist_frame,
    }
    transforms = []
    for side, pose, is_valid in zip(ee_msg.name, ee_msg.pose, ee_msg.is_valid):
        if not is_valid:
            continue
        transforms.append(
            make_transform(
                ee_msg.header.stamp,
                ee_msg.header.frame_id,
                wrist_frames[side],
                [pose.position.x, pose.position.y, pose.position.z],
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            )
        )
    return transforms


def build_controller_msg(
    left_ctrl: OptionalTensorGroup,
    right_ctrl: OptionalTensorGroup,
) -> ByteMultiArray | None:
    """Build a msgpack controller message, or None when both inputs are absent."""
    if left_ctrl.is_none and right_ctrl.is_none:
        return None

    def _as_list(ctrl, index):
        if ctrl.is_none:
            return [0.0, 0.0, 0.0]
        return [float(x) for x in ctrl[index]]

    def _as_quat(ctrl, index):
        if ctrl.is_none:
            return [0.0, 0.0, 0.0, 1.0]
        return [float(x) for x in ctrl[index]]

    def _as_float(ctrl, index):
        if ctrl.is_none:
            return 0.0
        return float(ctrl[index])

    return _to_msgpack_byte_multi_array(
        {
            "timestamp": time.time_ns(),
            "left_thumbstick": [
                _as_float(left_ctrl, ControllerInputIndex.THUMBSTICK_X),
                _as_float(left_ctrl, ControllerInputIndex.THUMBSTICK_Y),
            ],
            "right_thumbstick": [
                _as_float(right_ctrl, ControllerInputIndex.THUMBSTICK_X),
                _as_float(right_ctrl, ControllerInputIndex.THUMBSTICK_Y),
            ],
            "left_trigger_value": _as_float(
                left_ctrl, ControllerInputIndex.TRIGGER_VALUE
            ),
            "right_trigger_value": _as_float(
                right_ctrl, ControllerInputIndex.TRIGGER_VALUE
            ),
            "left_squeeze_value": _as_float(
                left_ctrl, ControllerInputIndex.SQUEEZE_VALUE
            ),
            "right_squeeze_value": _as_float(
                right_ctrl, ControllerInputIndex.SQUEEZE_VALUE
            ),
            "left_aim_position": _as_list(left_ctrl, ControllerInputIndex.AIM_POSITION),
            "right_aim_position": _as_list(
                right_ctrl, ControllerInputIndex.AIM_POSITION
            ),
            "left_grip_position": _as_list(
                left_ctrl, ControllerInputIndex.GRIP_POSITION
            ),
            "right_grip_position": _as_list(
                right_ctrl, ControllerInputIndex.GRIP_POSITION
            ),
            "left_aim_orientation": _as_quat(
                left_ctrl, ControllerInputIndex.AIM_ORIENTATION
            ),
            "right_aim_orientation": _as_quat(
                right_ctrl, ControllerInputIndex.AIM_ORIENTATION
            ),
            "left_grip_orientation": _as_quat(
                left_ctrl, ControllerInputIndex.GRIP_ORIENTATION
            ),
            "right_grip_orientation": _as_quat(
                right_ctrl, ControllerInputIndex.GRIP_ORIENTATION
            ),
            "left_primary_click": _as_float(
                left_ctrl, ControllerInputIndex.PRIMARY_CLICK
            ),
            "right_primary_click": _as_float(
                right_ctrl, ControllerInputIndex.PRIMARY_CLICK
            ),
            "left_secondary_click": _as_float(
                left_ctrl, ControllerInputIndex.SECONDARY_CLICK
            ),
            "right_secondary_click": _as_float(
                right_ctrl, ControllerInputIndex.SECONDARY_CLICK
            ),
            "left_thumbstick_click": _as_float(
                left_ctrl, ControllerInputIndex.THUMBSTICK_CLICK
            ),
            "right_thumbstick_click": _as_float(
                right_ctrl, ControllerInputIndex.THUMBSTICK_CLICK
            ),
            "left_menu_click": _as_float(left_ctrl, ControllerInputIndex.MENU_CLICK),
            "right_menu_click": _as_float(right_ctrl, ControllerInputIndex.MENU_CLICK),
            "left_is_active": not left_ctrl.is_none,
            "right_is_active": not right_ctrl.is_none,
        }
    )


def build_ee_output_from_controllers(
    left_ctrl: OptionalTensorGroup,
    right_ctrl: OptionalTensorGroup,
    now,
    frame_id: str,
    left_wrist_frame: str,
    right_wrist_frame: str,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
    apply_manus_controller_mount_offset: bool = False,
    *,
    head: OptionalTensorGroup | None = None,
    head_frame: str = "head",
    ee_poses_frame: EePoseFrame = EePoseFrame.WORLD,
) -> tuple[NamedPoseArray, list[TransformStamped]]:
    """Build the controller-derived EE message and its valid wrist TFs."""
    left_pose = _compute_ee_pose_from_controller(
        left_ctrl,
        "left",
        transform_rot,
        transform_trans,
        apply_manus_controller_mount_offset,
    )
    right_pose = _compute_ee_pose_from_controller(
        right_ctrl,
        "right",
        transform_rot,
        transform_trans,
        apply_manus_controller_mount_offset,
    )
    return _finalize_ee_output(
        left_pose,
        right_pose,
        now,
        frame_id,
        left_wrist_frame,
        right_wrist_frame,
        head=head,
        head_frame=head_frame,
        ee_poses_frame=ee_poses_frame,
        transform_rot=transform_rot,
        transform_trans=transform_trans,
    )


def build_ee_output_from_hands(
    left_hand: OptionalTensorGroup,
    right_hand: OptionalTensorGroup,
    now,
    frame_id: str,
    left_wrist_frame: str,
    right_wrist_frame: str,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
    *,
    head: OptionalTensorGroup | None = None,
    head_frame: str = "head",
    ee_poses_frame: EePoseFrame = EePoseFrame.WORLD,
) -> tuple[NamedPoseArray, list[TransformStamped]]:
    """Build the hand-derived EE message and its valid wrist TFs."""
    left_pose = _compute_ee_pose_from_hand(
        left_hand,
        transform_rot,
        transform_trans,
    )
    right_pose = _compute_ee_pose_from_hand(
        right_hand,
        transform_rot,
        transform_trans,
    )
    return _finalize_ee_output(
        left_pose,
        right_pose,
        now,
        frame_id,
        left_wrist_frame,
        right_wrist_frame,
        head=head,
        head_frame=head_frame,
        ee_poses_frame=ee_poses_frame,
        transform_rot=transform_rot,
        transform_trans=transform_trans,
    )


def build_finger_joints_msg(
    left_joints: OptionalTensorGroup,
    right_joints: OptionalTensorGroup,
    now,
    frame_id: str,
) -> JointState | None:
    if left_joints.is_none and right_joints.is_none:
        return None

    finger_joints_msg = JointState()
    finger_joints_msg.header.stamp = now
    finger_joints_msg.header.frame_id = frame_id
    left_arr = (
        np.asarray(list(left_joints), dtype=np.float32)
        if not left_joints.is_none
        else np.array([], dtype=np.float32)
    )
    right_arr = (
        np.asarray(list(right_joints), dtype=np.float32)
        if not right_joints.is_none
        else np.array([], dtype=np.float32)
    )
    finger_joints_msg.name = (
        joint_names_from_group_type(left_joints.group_type)
        if not left_joints.is_none
        else []
    ) + (
        joint_names_from_group_type(right_joints.group_type)
        if not right_joints.is_none
        else []
    )
    finger_joints_msg.position = np.concatenate([left_arr, right_arr]).tolist()
    return finger_joints_msg


def build_full_body_msg(
    full_body: OptionalTensorGroup,
) -> ByteMultiArray | None:
    """Build a msgpack full-body message, or None when the input is absent."""
    if full_body.is_none:
        return None

    positions = np.asarray(full_body[FullBodyInputIndex.JOINT_POSITIONS])
    orientations = np.asarray(full_body[FullBodyInputIndex.JOINT_ORIENTATIONS])
    valid = np.asarray(full_body[FullBodyInputIndex.JOINT_VALID])
    return _to_msgpack_byte_multi_array(
        {
            "timestamp": time.time_ns(),
            "joint_names": BODY_JOINT_NAMES,
            "joint_positions": [[float(v) for v in pos] for pos in positions],
            "joint_orientations": [[float(v) for v in ori] for ori in orientations],
            "joint_valid": [bool(v) for v in valid],
        }
    )


def build_hand_msg(
    left_hand: OptionalTensorGroup,
    right_hand: OptionalTensorGroup,
    now,
    frame_id: str,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
) -> NamedPoseArray:
    """Build fixed left/right hand joint entries with invalid placeholders."""
    msg = NamedPoseArray()
    msg.header.stamp = now
    msg.header.frame_id = frame_id

    for side, hand in (("left", left_hand), ("right", right_hand)):
        if hand.is_none:
            for joint_name in HAND_POSE_NAMES:
                msg.name.append(f"{side}_{joint_name}")
                msg.pose.append(to_pose([0.0, 0.0, 0.0]))
                msg.is_valid.append(False)
            continue
        positions = np.asarray(hand[HandInputIndex.JOINT_POSITIONS])
        orientations = np.asarray(hand[HandInputIndex.JOINT_ORIENTATIONS])
        joint_valid = np.asarray(hand[HandInputIndex.JOINT_VALID])
        for joint_idx, joint_name in zip(HAND_POSE_JOINT_INDICES, HAND_POSE_NAMES):
            joint_is_valid = bool(joint_valid[joint_idx])
            if joint_is_valid:
                pose = to_pose(positions[joint_idx], orientations[joint_idx])
                if transform_rot is not None or transform_trans is not None:
                    pose = apply_transform_to_pose(pose, transform_rot, transform_trans)
            else:
                pose = to_pose([0.0, 0.0, 0.0])
            msg.name.append(f"{side}_{joint_name}")
            msg.pose.append(pose)
            msg.is_valid.append(joint_is_valid)
    return msg


def build_head_output(
    head: OptionalTensorGroup,
    now,
    frame_id: str,
    child_frame: str,
    transform_rot: Rotation | None = None,
    transform_trans: Sequence[float] | None = None,
) -> tuple[PoseStamped, TransformStamped] | None:
    """Build the head pose message and matching TF, or None when invalid."""
    pose = _compute_head_pose(head, transform_rot, transform_trans)
    if pose is None:
        return None

    head_msg = PoseStamped()
    head_msg.header.stamp = now
    head_msg.header.frame_id = frame_id
    head_msg.pose = pose
    head_tf = make_transform(
        now,
        frame_id,
        child_frame,
        [pose.position.x, pose.position.y, pose.position.z],
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    )
    return head_msg, head_tf


def build_root_command_output(
    root_command: OptionalTensorGroup,
    now,
    frame_id: str,
) -> tuple[TwistStamped, PoseStamped] | None:
    """Build root twist and height-pose messages, or None for an absent command."""
    if root_command.is_none:
        return None

    command = np.asarray(root_command[0])
    twist_msg = TwistStamped()
    twist_msg.header.stamp = now
    twist_msg.header.frame_id = frame_id
    twist_msg.twist.linear.x = float(command[0])
    twist_msg.twist.linear.y = float(command[1])
    twist_msg.twist.linear.z = 0.0
    twist_msg.twist.angular.z = float(command[2])

    pose_msg = PoseStamped()
    pose_msg.header.stamp = now
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.position.z = float(command[3])
    pose_msg.pose.orientation.w = 1.0
    return twist_msg, pose_msg
