# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SomaHandedness,
    SomaHandJointPoseArray,
    SomaHandJointPoses,
    SomaHandJointPosesRecord,
    SomaHandJoint,
)


def test_soma_hand_joint_pose_views_alias_storage():
    poses = SomaHandJointPoseArray()

    assert poses.positions.shape == (25, 3)
    assert poses.orientations.shape == (25, 4)
    assert poses.is_valid.shape == (25,)
    assert poses.positions.dtype == np.float32
    assert poses.orientations.dtype == np.float32
    assert poses.is_valid.dtype == np.uint8

    poses.positions[SomaHandJoint.INDEX_END] = [1.0, 2.0, 3.0]
    poses.orientations[SomaHandJoint.INDEX_END] = [0.0, 0.0, 0.6, 0.8]
    poses.is_valid[SomaHandJoint.INDEX_END] = 1

    index_end = poses.values(int(SomaHandJoint.INDEX_END))
    assert index_end.pose.position.y == pytest.approx(2.0)
    assert index_end.pose.orientation.w == pytest.approx(0.8)
    assert index_end.is_valid is True


def test_soma_hand_joint_poses_record_lifetime():
    poses = SomaHandJointPoseArray()
    poses.positions[:] = np.arange(75, dtype=np.float32).reshape(25, 3)
    poses.orientations[:, 3] = 1.0
    poses.is_valid[:] = 1
    payload = SomaHandJointPoses(poses, SomaHandedness.RIGHT)
    record = SomaHandJointPosesRecord(payload, DeviceDataTimestamp(100, 200, 300))

    poses.positions[:] = 0.0
    payload.joint_poses.positions[:] = 0.0
    del poses, payload
    gc.collect()

    assert record.data.handedness == SomaHandedness.RIGHT
    assert record.data.joint_poses.positions[24, 2] == pytest.approx(74.0)
    assert record.timestamp.sample_time_raw_device_clock == 300
