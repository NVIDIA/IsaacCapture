# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    Point,
    Quaternion,
    SomaHandednessV0,
    SomaHandJointRotationArrayV0,
    SomaHandJointRotationsV0,
    SomaHandJointRotationsV0Record,
    SomaHandJointV0,
    SomaJointRotationV0,
)


def test_soma_hand_v0_joint_order():
    expected = [
        "WRIST",
        "THUMB1",
        "THUMB2",
        "THUMB3",
        "THUMB_END",
        "INDEX1",
        "INDEX2",
        "INDEX3",
        "INDEX4",
        "INDEX_END",
        "MIDDLE1",
        "MIDDLE2",
        "MIDDLE3",
        "MIDDLE4",
        "MIDDLE_END",
        "RING1",
        "RING2",
        "RING3",
        "RING4",
        "RING_END",
        "PINKY1",
        "PINKY2",
        "PINKY3",
        "PINKY4",
        "PINKY_END",
    ]

    assert len(expected) == int(SomaHandJointV0.NUM_JOINTS) == 25
    for index, name in enumerate(expected):
        assert int(getattr(SomaHandJointV0, name)) == index


def test_soma_hand_v0_joint_rotation():
    rotation = SomaJointRotationV0(Quaternion(0.0, 0.0, 0.6, 0.8), True)

    assert rotation.rotation.x == 0.0
    assert rotation.rotation.y == 0.0
    assert rotation.rotation.z == pytest.approx(0.6)
    assert rotation.rotation.w == pytest.approx(0.8)
    assert rotation.is_valid is True


def test_soma_hand_v0_rotation_views_alias_storage():
    rotations = SomaHandJointRotationArrayV0()

    assert rotations.rotations.shape == (25, 4)
    assert rotations.rotations.dtype == np.float32
    assert rotations.is_valid.shape == (25,)
    assert rotations.is_valid.dtype == np.uint8
    assert not rotations.rotations.flags.owndata

    rotations.rotations[SomaHandJointV0.INDEX_END] = [0.0, 0.0, 0.6, 0.8]
    rotations.is_valid[SomaHandJointV0.INDEX_END] = 1

    index_end = rotations.values(int(SomaHandJointV0.INDEX_END))
    assert index_end.rotation.z == pytest.approx(0.6)
    assert index_end.rotation.w == pytest.approx(0.8)
    assert index_end.is_valid is True


def test_soma_hand_v0_rotation_index_check():
    with pytest.raises(IndexError):
        SomaHandJointRotationArrayV0().values(25)


def test_soma_hand_v0_pose_construction_and_lifetime():
    rotations = SomaHandJointRotationArrayV0()
    rotations.rotations[:] = [0.0, 0.0, 0.0, 1.0]
    rotations.rotations[SomaHandJointV0.PINKY_END] = [0.0, 0.0, 0.6, 0.8]
    rotations.is_valid[:] = 1

    pose = SomaHandJointRotationsV0(
        rotations, Point(1.0, 2.0, 3.0), True, SomaHandednessV0.RIGHT
    )
    quaternion_rotations = pose.joint_rotations.rotations
    del pose
    gc.collect()

    assert quaternion_rotations[24, 3] == pytest.approx(0.8)


def test_soma_hand_v0_pose_defaults_and_global_translation():
    pose = SomaHandJointRotationsV0()

    assert pose.joint_rotations is not None
    assert pose.global_translation.x == 0.0
    assert pose.global_translation_is_valid is False
    assert pose.handedness == SomaHandednessV0.UNSPECIFIED

    right = SomaHandJointRotationsV0(
        global_translation=Point(1.0, 2.0, 3.0),
        global_translation_is_valid=True,
        handedness=SomaHandednessV0.RIGHT,
    )
    assert right.global_translation.y == pytest.approx(2.0)
    assert right.global_translation_is_valid is True
    assert right.handedness == SomaHandednessV0.RIGHT


def test_soma_hand_v0_pose_record():
    record = SomaHandJointRotationsV0Record(
        SomaHandJointRotationsV0(handedness=SomaHandednessV0.LEFT),
        DeviceDataTimestamp(100, 200, 300),
    )

    assert record.data is not None
    assert record.data.handedness == SomaHandednessV0.LEFT
    assert record.timestamp.sample_time_local_common_clock == 200


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("translation_is_valid", [False, True])
@pytest.mark.parametrize("handedness", [SomaHandednessV0.LEFT, SomaHandednessV0.RIGHT])
def test_soma_hand_v0_record_preserves_quaternion_layout(
    sign, translation_is_valid, handedness
):
    quaternion = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    quaternion *= sign / np.linalg.norm(quaternion)
    rotations = SomaHandJointRotationArrayV0()
    rotations.rotations[SomaHandJointV0.WRIST] = quaternion
    rotations.rotations[SomaHandJointV0.PINKY_END] = -quaternion
    rotations.is_valid[SomaHandJointV0.WRIST] = 1
    rotations.is_valid[SomaHandJointV0.PINKY_END] = 1
    expected_rotations = rotations.rotations.copy()
    expected_validity = rotations.is_valid.copy()
    pose = SomaHandJointRotationsV0(
        rotations, Point(1.0, -2.0, 3.0), translation_is_valid, handedness
    )
    record = SomaHandJointRotationsV0Record(pose, DeviceDataTimestamp(100, 200, 300))

    rotations.rotations[:] = 0.0
    rotations.is_valid[:] = 0
    pose.joint_rotations.rotations[:] = 0.0
    pose.joint_rotations.is_valid[:] = 0
    del pose, rotations
    gc.collect()

    data = record.data
    np.testing.assert_array_equal(data.joint_rotations.rotations, expected_rotations)
    np.testing.assert_array_equal(data.joint_rotations.is_valid, expected_validity)
    wrist = data.joint_rotations.values(int(SomaHandJointV0.WRIST)).rotation
    np.testing.assert_array_equal([wrist.x, wrist.y, wrist.z, wrist.w], quaternion)
    assert data.global_translation.x == pytest.approx(1.0)
    assert data.global_translation.y == pytest.approx(-2.0)
    assert data.global_translation.z == pytest.approx(3.0)
    assert data.global_translation_is_valid is translation_is_valid
    assert data.handedness == handedness
    assert record.timestamp.available_time_local_common_clock == 100
    assert record.timestamp.sample_time_local_common_clock == 200
    assert record.timestamp.sample_time_raw_device_clock == 300
