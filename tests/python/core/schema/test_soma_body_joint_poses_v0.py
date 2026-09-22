# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SomaBodyJointPoseArrayV0,
    SomaBodyJointPosesV0,
    SomaBodyJointPosesV0Record,
    SomaBodyJointV0,
)


def test_soma_body_joint_pose_views_alias_storage():
    joints = SomaBodyJointPoseArrayV0()
    positions = np.arange(231, dtype=np.float32).reshape(77, 3)
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (77, 1))
    joints.positions[:] = positions
    joints.orientations[:] = orientations
    joints.is_valid[int(SomaBodyJointV0.HEAD)] = 1

    np.testing.assert_array_equal(joints.positions, positions)
    np.testing.assert_array_equal(joints.orientations, orientations)
    head = joints.values(int(SomaBodyJointV0.HEAD))
    assert head.pose.position.x == positions[int(SomaBodyJointV0.HEAD), 0]
    assert head.pose.orientation.w == 1.0
    assert head.is_valid


def test_soma_body_joint_pose_index_check():
    with pytest.raises(IndexError):
        SomaBodyJointPoseArrayV0().values(77)


def test_soma_body_joint_poses_record_round_trip():
    joints = SomaBodyJointPoseArrayV0()
    joints.positions[int(SomaBodyJointV0.HIPS)] = [1.0, 2.0, 3.0]
    joints.orientations[:, 3] = 1.0
    joints.is_valid[:] = 1
    pose = SomaBodyJointPosesV0(joints)
    record = SomaBodyJointPosesV0Record(pose, DeviceDataTimestamp(10, 20, 30))

    np.testing.assert_array_equal(record.data.joint_poses.positions, joints.positions)
    np.testing.assert_array_equal(
        record.data.joint_poses.orientations, joints.orientations
    )
    np.testing.assert_array_equal(record.data.joint_poses.is_valid, joints.is_valid)
    assert record.data.to_bytes() == pose.to_bytes()
    assert record.timestamp.available_time_local_common_clock == 10
