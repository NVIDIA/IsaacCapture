# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Quaternion,
    SomaBodyJoint,
    SomaBodyJointRotationArray,
    SomaBodyJointRotations,
    SomaBodyJointRotationsRecord,
    SomaJointRotation,
)


def test_soma_body_joint_rotations_joint_order():
    expected = [
        "HIPS",
        "SPINE1",
        "SPINE2",
        "CHEST",
        "NECK1",
        "NECK2",
        "HEAD",
        "HEAD_END",
        "JAW",
        "LEFT_EYE",
        "RIGHT_EYE",
        "LEFT_SHOULDER",
        "LEFT_ARM",
        "LEFT_FORE_ARM",
        "LEFT_HAND",
        "LEFT_HAND_THUMB1",
        "LEFT_HAND_THUMB2",
        "LEFT_HAND_THUMB3",
        "LEFT_HAND_THUMB_END",
        "LEFT_HAND_INDEX1",
        "LEFT_HAND_INDEX2",
        "LEFT_HAND_INDEX3",
        "LEFT_HAND_INDEX4",
        "LEFT_HAND_INDEX_END",
        "LEFT_HAND_MIDDLE1",
        "LEFT_HAND_MIDDLE2",
        "LEFT_HAND_MIDDLE3",
        "LEFT_HAND_MIDDLE4",
        "LEFT_HAND_MIDDLE_END",
        "LEFT_HAND_RING1",
        "LEFT_HAND_RING2",
        "LEFT_HAND_RING3",
        "LEFT_HAND_RING4",
        "LEFT_HAND_RING_END",
        "LEFT_HAND_PINKY1",
        "LEFT_HAND_PINKY2",
        "LEFT_HAND_PINKY3",
        "LEFT_HAND_PINKY4",
        "LEFT_HAND_PINKY_END",
        "RIGHT_SHOULDER",
        "RIGHT_ARM",
        "RIGHT_FORE_ARM",
        "RIGHT_HAND",
        "RIGHT_HAND_THUMB1",
        "RIGHT_HAND_THUMB2",
        "RIGHT_HAND_THUMB3",
        "RIGHT_HAND_THUMB_END",
        "RIGHT_HAND_INDEX1",
        "RIGHT_HAND_INDEX2",
        "RIGHT_HAND_INDEX3",
        "RIGHT_HAND_INDEX4",
        "RIGHT_HAND_INDEX_END",
        "RIGHT_HAND_MIDDLE1",
        "RIGHT_HAND_MIDDLE2",
        "RIGHT_HAND_MIDDLE3",
        "RIGHT_HAND_MIDDLE4",
        "RIGHT_HAND_MIDDLE_END",
        "RIGHT_HAND_RING1",
        "RIGHT_HAND_RING2",
        "RIGHT_HAND_RING3",
        "RIGHT_HAND_RING4",
        "RIGHT_HAND_RING_END",
        "RIGHT_HAND_PINKY1",
        "RIGHT_HAND_PINKY2",
        "RIGHT_HAND_PINKY3",
        "RIGHT_HAND_PINKY4",
        "RIGHT_HAND_PINKY_END",
        "LEFT_LEG",
        "LEFT_SHIN",
        "LEFT_FOOT",
        "LEFT_TOE_BASE",
        "LEFT_TOE_END",
        "RIGHT_LEG",
        "RIGHT_SHIN",
        "RIGHT_FOOT",
        "RIGHT_TOE_BASE",
        "RIGHT_TOE_END",
    ]

    assert len(expected) == int(SomaBodyJoint.NUM_JOINTS) == 77
    for index, name in enumerate(expected):
        assert int(getattr(SomaBodyJoint, name)) == index


def test_soma_body_joint_rotations_joint_rotation():
    rotation = SomaJointRotation(Quaternion(0.0, 0.0, 0.6, 0.8), True)

    assert rotation.rotation.x == 0.0
    assert rotation.rotation.y == 0.0
    assert rotation.rotation.z == pytest.approx(0.6)
    assert rotation.rotation.w == pytest.approx(0.8)
    assert rotation.is_valid is True


def test_soma_body_joint_rotations_rotation_views_alias_storage():
    rotations = SomaBodyJointRotationArray()

    assert rotations.rotations.shape == (77, 4)
    assert rotations.rotations.dtype == np.float32
    assert rotations.is_valid.shape == (77,)
    assert rotations.is_valid.dtype == np.uint8
    assert not rotations.rotations.flags.owndata

    rotations.rotations[SomaBodyJoint.HEAD] = [0.0, 0.0, 0.6, 0.8]
    rotations.is_valid[SomaBodyJoint.HEAD] = 1

    head = rotations.values(int(SomaBodyJoint.HEAD))
    assert head.rotation.z == pytest.approx(0.6)
    assert head.rotation.w == pytest.approx(0.8)
    assert head.is_valid is True


def test_soma_body_joint_rotations_rotation_index_check():
    with pytest.raises(IndexError):
        SomaBodyJointRotationArray().values(77)


def test_soma_body_joint_rotations_pose_construction_and_lifetime():
    rotations = SomaBodyJointRotationArray()
    rotations.rotations[:] = [0.0, 0.0, 0.0, 1.0]
    rotations.rotations[SomaBodyJoint.RIGHT_TOE_END] = [0.0, 0.0, 0.6, 0.8]
    rotations.is_valid[:] = 1

    pose = SomaBodyJointRotations(rotations, Point(1.0, 2.0, 3.0), True)
    quaternion_rotations = pose.joint_rotations.rotations
    del pose
    gc.collect()

    assert quaternion_rotations[76, 3] == pytest.approx(0.8)


def test_soma_body_joint_rotations_pose_defaults_and_global_translation():
    pose = SomaBodyJointRotations()

    assert pose.joint_rotations is not None
    assert pose.global_translation.x == 0.0
    assert pose.global_translation_is_valid is False

    translated = SomaBodyJointRotations(
        global_translation=Point(1.0, 2.0, 3.0),
        global_translation_is_valid=True,
    )
    assert translated.global_translation.y == pytest.approx(2.0)
    assert translated.global_translation_is_valid is True


def test_soma_body_joint_rotations_pose_record():
    record = SomaBodyJointRotationsRecord(
        SomaBodyJointRotations(), DeviceDataTimestamp(100, 200, 300)
    )

    assert record.data is not None
    assert record.timestamp.sample_time_local_common_clock == 200


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("translation_is_valid", [False, True])
def test_soma_body_joint_rotations_record_preserves_quaternion_layout(
    sign, translation_is_valid
):
    quaternion = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    quaternion *= sign / np.linalg.norm(quaternion)
    rotations = SomaBodyJointRotationArray()
    rotations.rotations[SomaBodyJoint.HIPS] = quaternion
    rotations.rotations[SomaBodyJoint.RIGHT_TOE_END] = -quaternion
    rotations.is_valid[SomaBodyJoint.HIPS] = 1
    rotations.is_valid[SomaBodyJoint.RIGHT_TOE_END] = 1
    expected_rotations = rotations.rotations.copy()
    expected_validity = rotations.is_valid.copy()
    pose = SomaBodyJointRotations(
        rotations, Point(1.0, -2.0, 3.0), translation_is_valid
    )
    record = SomaBodyJointRotationsRecord(pose, DeviceDataTimestamp(100, 200, 300))

    rotations.rotations[:] = 0.0
    rotations.is_valid[:] = 0
    pose.joint_rotations.rotations[:] = 0.0
    pose.joint_rotations.is_valid[:] = 0
    del pose, rotations
    gc.collect()

    data = record.data
    np.testing.assert_array_equal(data.joint_rotations.rotations, expected_rotations)
    np.testing.assert_array_equal(data.joint_rotations.is_valid, expected_validity)
    hips = data.joint_rotations.values(int(SomaBodyJoint.HIPS)).rotation
    np.testing.assert_array_equal([hips.x, hips.y, hips.z, hips.w], quaternion)
    assert data.global_translation.x == pytest.approx(1.0)
    assert data.global_translation.y == pytest.approx(-2.0)
    assert data.global_translation.z == pytest.approx(3.0)
    assert data.global_translation_is_valid is translation_is_valid
    assert record.timestamp.available_time_local_common_clock == 100
    assert record.timestamp.sample_time_local_common_clock == 200
    assert record.timestamp.sample_time_raw_device_clock == 300
