# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.deviceio_trackers import (
    SomaHandJointPosesTracker,
    SomaHandJointRotationsTracker,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    DeviceIOSomaHandJointPosesTracked,
    DeviceIOSomaHandJointRotationsTracked,
    SomaHandRepresentation,
    SomaHandSource,
)
from isaaccapture.retargeting_engine.interface.tensor_group import TensorGroup
from isaaccapture.retargeting_engine.tensor_types import SomaHandInputIndex
from isaaccapture.schema import (
    Point,
    SomaHandedness,
    SomaHandJointPoseArray,
    SomaHandJointPoses,
    SomaHandJointRotationArray,
    SomaHandJointRotations,
)


def fake_layer():
    layer = MagicMock()
    layer.rig_data = {"joint_names": [f"Joint{index}" for index in range(25)]}
    layer.joint_parent_ids = np.array([0, 0, 1, 2, 3, *range(20)])[:25]
    return layer


def rotations(handedness=SomaHandedness.LEFT):
    joints = SomaHandJointRotationArray()
    joints.rotations[:] = (0, 0, 0, 1)
    joints.is_valid[:] = 1
    return SomaHandJointRotations(joints, Point(1, 2, 3), True, handedness)


def poses(handedness=SomaHandedness.LEFT):
    joints = SomaHandJointPoseArray()
    joints.positions[:] = np.arange(75, dtype=np.float32).reshape(25, 3)
    joints.orientations[:] = np.tile([0, 0, 0, 1], (25, 1))
    joints.is_valid[:] = 1
    return SomaHandJointPoses(joints, handedness)


def payload_group(group_type, data):
    group = TensorGroup(group_type)
    group[0] = data
    return group


def test_rotation_source_evaluates_received_payload():
    source = SomaHandSource("left", "vendor.left", SomaHandedness.LEFT, fake_layer())
    assert isinstance(source.get_tracker(), SomaHandJointRotationsTracker)
    raw = rotations()
    expected = (
        np.ones((25, 3), dtype=np.float32),
        np.tile([0, 0, 0, 1], (25, 1)).astype(np.float32),
        np.ones(25, dtype=np.uint8),
    )
    source._evaluator.evaluate = MagicMock(return_value=expected)
    inputs = payload_group(source.input_spec()["deviceio_soma_hand"], raw)

    evaluated = source({"deviceio_soma_hand": inputs})[SomaHandSource.HAND]

    source._evaluator.evaluate.assert_called_once_with(raw)
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_POSITIONS], expected[0]
    )
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_ORIENTATIONS], expected[1]
    )
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_VALID], expected[2]
    )


def test_joint_pose_source_maps_without_fk():
    source = SomaHandSource(
        "left",
        "vendor.left",
        SomaHandedness.LEFT,
        representation=SomaHandRepresentation.JOINT_POSES,
    )
    assert isinstance(source.get_tracker(), SomaHandJointPosesTracker)
    raw = poses()
    inputs = payload_group(source.input_spec()["deviceio_soma_hand"], raw)

    evaluated = source({"deviceio_soma_hand": inputs})[SomaHandSource.HAND]

    assert source._evaluator is None
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_POSITIONS], raw.joint_poses.positions
    )
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_VALID], raw.joint_poses.is_valid
    )


def test_payload_handedness_must_match_collection():
    source = SomaHandSource(
        "left",
        "vendor.left",
        SomaHandedness.LEFT,
        representation=SomaHandRepresentation.JOINT_POSES,
    )
    inputs = payload_group(
        source.input_spec()["deviceio_soma_hand"], poses(SomaHandedness.RIGHT)
    )
    with pytest.raises(ValueError, match="handedness"):
        source({"deviceio_soma_hand": inputs})


def test_hand_transport_profiles_are_distinct():
    rotation_type = DeviceIOSomaHandJointRotationsTracked()
    pose_type = DeviceIOSomaHandJointPosesTracked()
    with pytest.raises(ValueError, match="type mismatch"):
        rotation_type.check_compatibility(pose_type)
    with pytest.raises(TypeError, match="SomaHandJointPoses"):
        payload_group(pose_type, rotations())


def test_rotation_source_requires_layer_and_valid_side():
    with pytest.raises(ValueError, match="prepared SOMA hand layer"):
        SomaHandSource("left", "vendor.left", SomaHandedness.LEFT)
    with pytest.raises(ValueError, match="LEFT or RIGHT"):
        SomaHandSource(
            "left",
            "vendor.left",
            SomaHandedness.UNSPECIFIED,
            fake_layer(),
        )
