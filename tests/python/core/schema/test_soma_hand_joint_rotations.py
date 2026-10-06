# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Quaternion,
    SomaHandedness,
    SomaHandJoint,
    SomaHandJointRotation,
    SomaHandJointRotations,
    SomaHandJointRotationsRecord,
)


def test_soma_hand_joint_order():
    assert int(SomaHandJoint.WRIST) == 0
    assert int(SomaHandJoint.INDEX1) == 5
    assert int(SomaHandJoint.MIDDLE1) == 10
    assert int(SomaHandJoint.RING1) == 15
    assert int(SomaHandJoint.PINKY_END) == 24
    assert int(SomaHandJoint.NUM_JOINTS) == 25


def test_soma_hand_rotations_sort_lookup_and_omit_unavailable_joints():
    payload = SomaHandJointRotations(
        [
            SomaHandJointRotation(
                SomaHandJoint.PINKY_END, Quaternion(0.0, 0.0, 0.6, 0.8)
            ),
            SomaHandJointRotation(SomaHandJoint.WRIST, Quaternion(0.0, 0.0, 0.0, 1.0)),
        ],
        Point(1.0, 2.0, 3.0),
        True,
        SomaHandedness.LEFT,
    )

    assert [entry.joint for entry in payload.joint_rotations] == [
        SomaHandJoint.WRIST,
        SomaHandJoint.PINKY_END,
    ]
    assert payload.lookup(SomaHandJoint.PINKY_END).rotation.w == pytest.approx(0.8)
    assert payload.lookup(SomaHandJoint.INDEX1) is None
    assert payload.handedness == SomaHandedness.LEFT


def test_soma_hand_rotations_reject_duplicate_joint_identifiers():
    entry = SomaHandJointRotation(SomaHandJoint.WRIST, Quaternion(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="duplicate SomaHandJoint WRIST"):
        SomaHandJointRotations([entry, entry])


def test_soma_hand_rotation_record_preserves_quaternion_layout():
    quaternion = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    payload = SomaHandJointRotations(
        [SomaHandJointRotation(SomaHandJoint.WRIST, Quaternion(*quaternion))],
        handedness=SomaHandedness.RIGHT,
    )
    record = SomaHandJointRotationsRecord(payload, DeviceDataTimestamp(100, 200, 300))

    wrist = record.data.lookup(SomaHandJoint.WRIST)
    np.testing.assert_array_equal(
        [wrist.rotation.x, wrist.rotation.y, wrist.rotation.z, wrist.rotation.w],
        quaternion,
    )
    assert record.data.handedness == SomaHandedness.RIGHT
    assert record.timestamp.sample_time_local_common_clock == 200
