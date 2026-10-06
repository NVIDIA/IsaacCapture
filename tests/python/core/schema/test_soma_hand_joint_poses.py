# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Pose,
    Quaternion,
    SomaHandedness,
    SomaHandJoint,
    SomaHandJointPose,
    SomaHandJointPoses,
    SomaHandJointPosesRecord,
)


def test_soma_hand_joint_poses_are_keyed_and_sparse():
    payload = SomaHandJointPoses(
        [
            SomaHandJointPose(
                SomaHandJoint.INDEX_END,
                Pose(Point(1.0, 2.0, 3.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            )
        ],
        SomaHandedness.LEFT,
    )

    assert len(payload.joint_poses) == 1
    assert payload.lookup(SomaHandJoint.INDEX_END).pose.position.y == pytest.approx(2.0)
    assert payload.lookup(SomaHandJoint.WRIST) is None
    assert payload.handedness == SomaHandedness.LEFT


def test_soma_hand_joint_poses_sort_and_reject_duplicates():
    wrist = SomaHandJointPose(SomaHandJoint.WRIST, Pose())
    pinky = SomaHandJointPose(SomaHandJoint.PINKY_END, Pose())
    payload = SomaHandJointPoses([pinky, wrist])

    assert [entry.joint for entry in payload.joint_poses] == [
        SomaHandJoint.WRIST,
        SomaHandJoint.PINKY_END,
    ]
    with pytest.raises(ValueError, match="duplicate SomaHandJoint WRIST"):
        SomaHandJointPoses([wrist, wrist])


def test_soma_hand_joint_poses_record_round_trip():
    payload = SomaHandJointPoses(
        [
            SomaHandJointPose(
                SomaHandJoint.PINKY_END,
                Pose(Point(72.0, 73.0, 74.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            )
        ],
        SomaHandedness.RIGHT,
    )
    record = SomaHandJointPosesRecord(payload, DeviceDataTimestamp(10, 20, 30))

    assert record.data.lookup(SomaHandJoint.PINKY_END).pose.position.z == pytest.approx(
        74.0
    )
    assert record.data.handedness == SomaHandedness.RIGHT
    assert record.timestamp.available_time_local_common_clock == 10
