# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Quaternion,
    SomaBodyJoint,
    SomaBodyJointRotation,
    SomaBodyJointRotations,
    SomaBodyJointRotationsRecord,
)


def test_soma_body_joint_order():
    assert int(SomaBodyJoint.HIPS) == 0
    assert int(SomaBodyJoint.LEFT_SHOULDER) == 11
    assert int(SomaBodyJoint.RIGHT_SHOULDER) == 39
    assert int(SomaBodyJoint.LEFT_LEG) == 67
    assert int(SomaBodyJoint.RIGHT_TOE_END) == 76
    assert int(SomaBodyJoint.NUM_JOINTS) == 77


def test_soma_body_rotation_has_explicit_joint_identifier():
    entry = SomaBodyJointRotation(SomaBodyJoint.HEAD, Quaternion(0.0, 0.0, 0.6, 0.8))

    assert entry.joint == SomaBodyJoint.HEAD
    assert entry.rotation.z == pytest.approx(0.6)
    assert entry.rotation.w == pytest.approx(0.8)
    assert "HEAD" in repr(entry)


def test_soma_body_rotations_sort_lookup_and_omit_unavailable_joints():
    payload = SomaBodyJointRotations(
        [
            SomaBodyJointRotation(
                SomaBodyJoint.RIGHT_TOE_END, Quaternion(0.0, 0.0, 0.6, 0.8)
            ),
            SomaBodyJointRotation(SomaBodyJoint.HIPS, Quaternion(0.0, 0.0, 0.0, 1.0)),
        ],
        Point(1.0, 2.0, 3.0),
        True,
    )

    assert [entry.joint for entry in payload.joint_rotations] == [
        SomaBodyJoint.HIPS,
        SomaBodyJoint.RIGHT_TOE_END,
    ]
    assert payload.lookup(SomaBodyJoint.RIGHT_TOE_END).rotation.w == pytest.approx(0.8)
    assert payload.lookup(SomaBodyJoint.HEAD) is None
    assert payload.global_translation.y == pytest.approx(2.0)
    assert payload.global_translation_is_valid


def test_soma_body_rotations_reject_duplicate_joint_identifiers():
    entry = SomaBodyJointRotation(SomaBodyJoint.HEAD, Quaternion(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="duplicate SomaBodyJoint HEAD"):
        SomaBodyJointRotations([entry, entry])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_soma_body_rotation_record_preserves_quaternion_layout(sign):
    quaternion = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    quaternion *= sign / np.linalg.norm(quaternion)
    payload = SomaBodyJointRotations(
        [SomaBodyJointRotation(SomaBodyJoint.HIPS, Quaternion(*quaternion))]
    )
    record = SomaBodyJointRotationsRecord(payload, DeviceDataTimestamp(100, 200, 300))

    hips = record.data.lookup(SomaBodyJoint.HIPS)
    np.testing.assert_array_equal(
        [hips.rotation.x, hips.rotation.y, hips.rotation.z, hips.rotation.w],
        quaternion,
    )
    assert record.timestamp.sample_time_raw_device_clock == 300
    assert record.data.to_bytes() == payload.to_bytes()


def test_soma_body_rotations_default_to_an_empty_snapshot():
    payload = SomaBodyJointRotations()

    assert payload.joint_rotations == []
    assert payload.lookup(SomaBodyJoint.HIPS) is None
    assert payload.global_translation.x == 0.0
    assert not payload.global_translation_is_valid
