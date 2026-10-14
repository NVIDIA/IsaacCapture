# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Pose,
    Quaternion,
    SomaBodyJoint,
    SomaBodyJointPose,
    SomaBodyJointPoses,
    SomaBodyJointPosesRecord,
)


def test_soma_body_joint_poses_are_keyed_and_sparse():
    payload = SomaBodyJointPoses(
        [
            SomaBodyJointPose(
                SomaBodyJoint.HEAD,
                Pose(Point(1.0, 2.0, 3.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            )
        ]
    )

    assert len(payload.joint_poses) == 1
    assert payload.joint_poses[0].joint == SomaBodyJoint.HEAD
    assert payload.lookup(SomaBodyJoint.HEAD).pose.position.y == pytest.approx(2.0)
    assert payload.lookup(SomaBodyJoint.HIPS) is None


def test_soma_body_joint_poses_sort_and_reject_duplicates():
    hips = SomaBodyJointPose(SomaBodyJoint.HIPS, Pose())
    head = SomaBodyJointPose(SomaBodyJoint.HEAD, Pose())
    payload = SomaBodyJointPoses([head, hips])

    assert [entry.joint for entry in payload.joint_poses] == [
        SomaBodyJoint.HIPS,
        SomaBodyJoint.HEAD,
    ]
    with pytest.raises(ValueError, match="duplicate SomaBodyJoint HEAD"):
        SomaBodyJointPoses([head, head])


def test_soma_body_joint_poses_record_round_trip():
    payload = SomaBodyJointPoses(
        [
            SomaBodyJointPose(
                SomaBodyJoint.RIGHT_TOE_END,
                Pose(Point(1.0, 2.0, 3.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
            )
        ]
    )
    record = SomaBodyJointPosesRecord(payload, DeviceDataTimestamp(10, 20, 30))

    assert record.data.lookup(SomaBodyJoint.RIGHT_TOE_END).pose.position.z == 3.0
    assert record.data.to_bytes() == payload.to_bytes()
    assert record.timestamp.available_time_local_common_clock == 10
