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
from isaaccapture.retargeting_engine.utilities.soma_hand_evaluator import (
    _SomaHandEvaluator,
)
from isaaccapture.schema import (
    Point,
    Pose,
    Quaternion,
    SomaHandedness,
    SomaHandJoint,
    SomaHandJointPose,
    SomaHandJointPoses,
    SomaHandJointRotation,
    SomaHandJointRotations,
)


def hand_joint_names():
    return tuple(
        name
        for name, joint in sorted(
            SomaHandJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    )


def fake_layer(hand_type="left"):
    layer = MagicMock()
    prefix = f"{hand_type.capitalize()}Hand"
    layer.rig_data = {
        "joint_names": [
            prefix if index == 0 else f"{prefix}{name.replace('_', '')}"
            for index, name in enumerate(hand_joint_names())
        ]
    }
    layer.joint_parent_ids = np.array(
        [
            0,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            8,
            0,
            10,
            11,
            12,
            13,
            0,
            15,
            16,
            17,
            18,
            0,
            20,
            21,
            22,
            23,
        ]
    )
    layer.hand_type = hand_type
    layer.output_unit = MagicMock(meters_per_unit=1.0)
    layer.get_reference_pose.return_value = np.tile(np.eye(3), (25, 1, 1))
    return layer


def rotations(handedness=SomaHandedness.LEFT):
    joints = [
        SomaHandJointRotation(SomaHandJoint(index), Quaternion(0, 0, 0, 1))
        for index in range(25)
    ]
    return SomaHandJointRotations(joints, Point(1, 2, 3), True, handedness)


def poses(handedness=SomaHandedness.LEFT):
    positions = np.arange(75, dtype=np.float32).reshape(25, 3)
    joints = [
        SomaHandJointPose(
            SomaHandJoint(index),
            Pose(Point(*position), Quaternion(0, 0, 0, 1)),
        )
        for index, position in enumerate(positions)
    ]
    return SomaHandJointPoses(joints, handedness)


def payload_group(group_type, data):
    group = TensorGroup(group_type)
    group[0] = data
    return group


def test_rotation_source_evaluates_received_payload():
    layer = fake_layer()
    source = SomaHandSource("left", "vendor.left", SomaHandedness.LEFT, layer)
    assert isinstance(source.get_tracker(), SomaHandJointRotationsTracker)
    layer.get_reference_pose.assert_called_once_with(version="v0.3.1")
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
    expected_positions = np.array(
        [
            [entry.pose.position.x, entry.pose.position.y, entry.pose.position.z]
            for entry in raw.joint_poses
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_POSITIONS], expected_positions
    )
    assert evaluated[SomaHandInputIndex.JOINT_VALID].all()


def test_joint_pose_source_invalidates_bad_numeric_entries():
    source = SomaHandSource(
        "left",
        "vendor.left",
        SomaHandedness.LEFT,
        representation=SomaHandRepresentation.JOINT_POSES,
    )
    raw = SomaHandJointPoses(
        [
            SomaHandJointPose(
                SomaHandJoint.WRIST,
                Pose(Point(1, 2, 3), Quaternion(0, 0, 0, -2)),
            ),
            SomaHandJointPose(
                SomaHandJoint.THUMB1,
                Pose(Point(np.nan, 0, 0), Quaternion(0, 0, 0, 1)),
            ),
            SomaHandJointPose(
                SomaHandJoint.THUMB2,
                Pose(Point(0, 0, 0), Quaternion()),
            ),
        ],
        SomaHandedness.LEFT,
    )
    inputs = payload_group(source.input_spec()["deviceio_soma_hand"], raw)

    evaluated = source({"deviceio_soma_hand": inputs})[SomaHandSource.HAND]

    np.testing.assert_array_equal(
        evaluated[SomaHandInputIndex.JOINT_VALID][:3], [1, 0, 0]
    )
    np.testing.assert_allclose(
        evaluated[SomaHandInputIndex.JOINT_ORIENTATIONS][0], [0, 0, 0, -1]
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


def test_rotation_source_rejects_layer_for_other_hand():
    with pytest.raises(ValueError, match="does not match"):
        SomaHandSource(
            "left",
            "vendor.left",
            SomaHandedness.LEFT,
            fake_layer("right"),
        )


def test_rotation_source_rejects_wrong_hand_hierarchy():
    layer = fake_layer()
    layer.joint_parent_ids[2] = 0

    with pytest.raises(ValueError, match="hierarchy"):
        SomaHandSource(
            "left",
            "vendor.left",
            SomaHandedness.LEFT,
            layer,
        )


@pytest.mark.parametrize(
    "change,error",
    [
        (
            lambda layer: layer.rig_data["joint_names"].reverse(),
            "joint order",
        ),
        (
            lambda layer: setattr(layer.output_unit, "meters_per_unit", 0.01),
            "meters",
        ),
        (
            lambda layer: setattr(
                layer.get_reference_pose, "side_effect", KeyError("missing")
            ),
            "reference",
        ),
    ],
)
def test_rotation_source_rejects_incompatible_hand_contract(change, error):
    layer = fake_layer()
    change(layer)

    with pytest.raises(ValueError, match=error):
        SomaHandSource(
            "left",
            "vendor.left",
            SomaHandedness.LEFT,
            layer,
        )


def test_hand_rotation_inputs_normalize_and_invalidate_numeric_values():
    payload = SomaHandJointRotations(
        [
            SomaHandJointRotation(SomaHandJoint.WRIST, Quaternion()),
            SomaHandJointRotation(SomaHandJoint.THUMB1, Quaternion(0, 0, 0, 2)),
            SomaHandJointRotation(SomaHandJoint.THUMB2, Quaternion(0, 0, 0, -2)),
        ],
        Point(np.inf, 0, 0),
        True,
        SomaHandedness.LEFT,
    )

    rotations, valid, translation, translation_valid = _SomaHandEvaluator._pose_inputs(
        payload
    )

    np.testing.assert_array_equal(valid[:3], [False, True, True])
    np.testing.assert_allclose(rotations[1], [0, 0, 0, 1])
    np.testing.assert_allclose(rotations[2], [0, 0, 0, -1])
    assert not translation_valid
    np.testing.assert_array_equal(translation, np.zeros(3, dtype=np.float32))
