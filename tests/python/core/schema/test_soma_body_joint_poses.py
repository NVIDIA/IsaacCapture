# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SomaBodyJointPoseArray,
    SomaBodyJointPoses,
    SomaBodyJointPosesRecord,
    SomaBodyJoint,
)


def test_soma_body_joint_pose_views_alias_storage():
    joints = SomaBodyJointPoseArray()
    positions = np.arange(231, dtype=np.float32).reshape(77, 3)
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (77, 1))
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[int(SomaBodyJoint.HEAD)] = 1

    np.testing.assert_array_equal(joints.positions, positions)
    np.testing.assert_array_equal(joints.orientations, orientations)
    head = joints.values(int(SomaBodyJoint.HEAD))
    assert head.pose.position.x == positions[int(SomaBodyJoint.HEAD), 0]
    assert head.pose.orientation.w == 1.0
    assert head.is_valid


def test_soma_body_joint_pose_index_check():
    with pytest.raises(IndexError):
        SomaBodyJointPoseArray().values(77)


def test_soma_body_joint_poses_record_round_trip():
    joints = SomaBodyJointPoseArray()
    joints.positions[int(SomaBodyJoint.HIPS)] = [1.0, 2.0, 3.0]
    joints.orientations[:, 3] = 1.0
    joints.is_valid[:] = 1
    pose = SomaBodyJointPoses(joints)
    record = SomaBodyJointPosesRecord(pose, DeviceDataTimestamp(10, 20, 30))

    np.testing.assert_array_equal(record.data.joint_poses.positions, joints.positions)
    np.testing.assert_array_equal(
        record.data.joint_poses.orientations, joints.orientations
    )
    np.testing.assert_array_equal(record.data.joint_poses.is_valid, joints.is_valid)
    assert record.data.to_bytes() == pose.to_bytes()
    assert record.timestamp.available_time_local_common_clock == 10
