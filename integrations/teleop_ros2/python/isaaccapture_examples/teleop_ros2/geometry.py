# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry helpers for ROS message construction."""

from typing import Sequence, Union

import numpy as np
from geometry_msgs.msg import Pose, TransformStamped
from scipy.spatial.transform import Rotation


def apply_relative_pose(reference: Pose, pose: Pose) -> Pose:
    """
    Transform ``pose`` in ``reference``'s frame, both poses needs to be
    initially in the same frame.

    This is equivalent to:

        T_reference_pose = inv(T_world_reference) @ T_world_pose
    """
    reference_pos = np.array(
        [reference.position.x, reference.position.y, reference.position.z],
        dtype=float,
    )
    reference_rot = Rotation.from_quat(
        [
            reference.orientation.x,
            reference.orientation.y,
            reference.orientation.z,
            reference.orientation.w,
        ]
    )
    pos = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=float,
    )
    rot = Rotation.from_quat(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )

    reference_rot_inv = reference_rot.inv()
    return to_pose(
        reference_rot_inv.apply(pos - reference_pos),
        (reference_rot_inv * rot).as_quat(),
    )


def apply_transform_to_pose(
    pose: Pose,
    rotation: Rotation | None = None,
    translation: Sequence[float] | None = None,
) -> Pose:
    """
    Return a new Pose with world-frame position transform and orientation
    basis change applied.
    """
    p = [pose.position.x, pose.position.y, pose.position.z]
    orientation = Rotation.from_quat(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )

    if rotation is not None:
        p = rotation.apply(p)
        # Conjugation keeps the same physical orientation while expressing it
        # in the rotated basis used for published EE and hand poses.
        orientation = rotation * orientation * rotation.inv()

    q = orientation.as_quat()

    result = Pose()
    if translation is not None:
        result.position.x = float(p[0]) + translation[0]
        result.position.y = float(p[1]) + translation[1]
        result.position.z = float(p[2]) + translation[2]
    else:
        result.position.x = float(p[0])
        result.position.y = float(p[1])
        result.position.z = float(p[2])

    result.orientation.x = float(q[0])
    result.orientation.y = float(q[1])
    result.orientation.z = float(q[2])
    result.orientation.w = float(q[3])
    return result


def make_transform(
    stamp,
    parent_frame: str,
    child_frame: str,
    position: Union[np.ndarray, Sequence[float]],
    orientation: Union[np.ndarray, Sequence[float]],
) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent_frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = float(position[0])
    tf.transform.translation.y = float(position[1])
    tf.transform.translation.z = float(position[2])
    tf.transform.rotation.x = float(orientation[0])
    tf.transform.rotation.y = float(orientation[1])
    tf.transform.rotation.z = float(orientation[2])
    tf.transform.rotation.w = float(orientation[3])
    return tf


def to_pose(position, orientation=None) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    if orientation is None:
        pose.orientation.w = 1.0
    else:
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
    return pose
