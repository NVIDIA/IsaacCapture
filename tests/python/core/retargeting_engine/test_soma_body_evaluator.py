# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

from isaaccapture.retargeting_engine.utilities.soma_body_evaluator import (
    _dense_joint_poses,
    _dense_joint_rotations,
    _SomaBodyEvaluator,
)
from isaaccapture.retargeting_engine.utilities.soma_contract import (
    SOMA_REFERENCE_VERSION,
)
from isaaccapture.schema import (
    Point,
    Pose,
    Quaternion,
    SomaBodyJoint,
    SomaBodyJointPose,
    SomaBodyJointRotation,
)


def body_joint_names():
    return tuple(
        name
        for name, joint in sorted(
            SomaBodyJoint.__members__.items(), key=lambda item: int(item[1])
        )
        if name != "NUM_JOINTS"
    )


def compatible_layer():
    layer = MagicMock()
    layer.public_joint_names = ("Root", *body_joint_names())
    layer.output_joint_parent_ids = np.array([0, 0, 1, 1, *range(3, 77)])
    layer.output_unit = MagicMock(meters_per_unit=1.0)
    layer.get_reference_pose.return_value = np.tile(np.eye(3), (78, 1, 1))
    return layer


def evaluator_with_two_branches() -> _SomaBodyEvaluator:
    return _SomaBodyEvaluator(compatible_layer())


def test_evaluator_exposes_native_joint_order_and_topology():
    evaluator = evaluator_with_two_branches()
    assert evaluator.joint_names[:3] == ("HIPS", "SPINE1", "SPINE2")
    assert evaluator.bones[:3] == ((0, 1), (0, 2), (2, 3))
    assert len(evaluator.joint_names) == 77
    assert len(evaluator.bones) == 76


def test_evaluator_resolves_declared_reference_once():
    layer = compatible_layer()
    evaluator = _SomaBodyEvaluator(layer)

    layer.get_reference_pose.assert_called_once_with(version=SOMA_REFERENCE_VERSION)
    assert evaluator.reference_pose is layer.get_reference_pose.return_value


@pytest.mark.parametrize(
    "change,error",
    [
        (
            lambda layer: setattr(
                layer,
                "public_joint_names",
                ("Root", *reversed(body_joint_names())),
            ),
            "joint order",
        ),
        (
            lambda layer: setattr(layer.output_unit, "meters_per_unit", 0.01),
            "meters",
        ),
        (
            lambda layer: setattr(
                layer,
                "output_joint_parent_ids",
                np.array([0, 0, 2, *range(2, 77)]),
            ),
            "hierarchy",
        ),
    ],
)
def test_evaluator_rejects_incompatible_model_contract(change, error):
    layer = compatible_layer()
    change(layer)

    with pytest.raises(ValueError, match=error):
        _SomaBodyEvaluator(layer)


def test_evaluator_rejects_unavailable_declared_reference():
    layer = compatible_layer()
    layer.get_reference_pose.side_effect = KeyError("missing")

    with pytest.raises(ValueError, match="reference"):
        _SomaBodyEvaluator(layer)


def test_evaluator_propagates_validity_through_ancestors():
    evaluator = evaluator_with_two_branches()
    controls = np.ones(77, dtype=bool)
    controls[1] = False
    valid = evaluator._joint_validity(controls, translation_valid=True)
    assert valid[0]
    assert not valid[1]
    assert valid[2]
    assert valid[3]
    assert not evaluator._joint_validity(controls, translation_valid=False).any()


def test_evaluator_defaults_omitted_rotations_to_invalid_identity():
    data = MagicMock(
        joint_rotations=None,
        global_translation=MagicMock(x=1.0, y=2.0, z=3.0),
        global_translation_is_valid=True,
    )

    rotations, control_valid, translation, translation_valid = (
        _SomaBodyEvaluator._pose_inputs(data)
    )

    np.testing.assert_array_equal(rotations, np.tile([0.0, 0.0, 0.0, 1.0], (77, 1)))
    assert not control_valid.any()
    np.testing.assert_array_equal(translation, [1.0, 2.0, 3.0])
    assert translation_valid
    assert (
        not evaluator_with_two_branches()
        ._joint_validity(control_valid, translation_valid)
        .any()
    )


def test_evaluator_marks_omitted_translation_invalid():
    rotations = np.tile([0.0, 0.0, 0.0, 1.0], (77, 1))
    data = MagicMock(
        joint_rotations=[
            SomaBodyJointRotation(SomaBodyJoint(index), Quaternion(*rotation))
            for index, rotation in enumerate(rotations)
        ],
        global_translation=None,
        global_translation_is_valid=True,
    )

    actual_rotations, control_valid, translation, translation_valid = (
        _SomaBodyEvaluator._pose_inputs(data)
    )

    np.testing.assert_array_equal(actual_rotations, rotations)
    assert control_valid.all()
    np.testing.assert_array_equal(translation, np.zeros(3, dtype=np.float32))
    assert not translation_valid
    assert (
        not evaluator_with_two_branches()
        ._joint_validity(control_valid, translation_valid)
        .any()
    )


def test_dense_adapter_rejects_unsorted_or_duplicate_joint_identifiers():
    hips = SomaBodyJointRotation(SomaBodyJoint.HIPS, Quaternion(0.0, 0.0, 0.0, 1.0))
    head = SomaBodyJointRotation(SomaBodyJoint.HEAD, Quaternion(0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="sorted and unique"):
        _dense_joint_rotations([head, hips], 77)
    with pytest.raises(ValueError, match="sorted and unique"):
        _dense_joint_rotations([head, head], 77)


def test_dense_rotations_normalize_usable_values_and_invalidate_bad_values():
    entries = [
        SomaBodyJointRotation(SomaBodyJoint.HIPS, Quaternion()),
        SomaBodyJointRotation(SomaBodyJoint.SPINE1, Quaternion(np.nan, 0.0, 0.0, 1.0)),
        SomaBodyJointRotation(SomaBodyJoint.SPINE2, Quaternion(0.0, 0.0, 0.0, 2.0)),
        SomaBodyJointRotation(SomaBodyJoint.CHEST, Quaternion(0.0, 0.0, 0.0, -2.0)),
        SomaBodyJointRotation(SomaBodyJoint.NECK1, Quaternion(0.0, 0.0, 0.0, 1e-11)),
        SomaBodyJointRotation(SomaBodyJoint.NECK2, Quaternion(0.0, 0.0, 0.0, 1e-12)),
    ]

    rotations, valid = _dense_joint_rotations(entries, 77)

    np.testing.assert_array_equal(valid[:6], [False, False, True, True, True, False])
    np.testing.assert_array_equal(rotations[:2], np.tile([0, 0, 0, 1], (2, 1)))
    np.testing.assert_allclose(rotations[2], [0, 0, 0, 1])
    np.testing.assert_allclose(rotations[3], [0, 0, 0, -1])
    np.testing.assert_allclose(rotations[4], [0, 0, 0, 1])


def test_dense_poses_normalize_or_invalidate_each_entry():
    entries = [
        SomaBodyJointPose(
            SomaBodyJoint.HIPS,
            Pose(Point(1.0, 2.0, 3.0), Quaternion(0.0, 0.0, 0.0, 2.0)),
        ),
        SomaBodyJointPose(
            SomaBodyJoint.SPINE1,
            Pose(Point(np.inf, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
        ),
        SomaBodyJointPose(
            SomaBodyJoint.SPINE2,
            Pose(Point(0.0, 0.0, 0.0), Quaternion()),
        ),
    ]

    positions, orientations, valid = _dense_joint_poses(entries, 77)

    np.testing.assert_array_equal(valid[:3], [1, 0, 0])
    np.testing.assert_array_equal(positions[0], [1, 2, 3])
    np.testing.assert_allclose(orientations[0], [0, 0, 0, 1])
    assert np.isfinite(positions).all()
    assert np.isfinite(orientations).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_translation_is_unavailable(bad_value):
    data = MagicMock(
        joint_rotations=None,
        global_translation=MagicMock(x=bad_value, y=0.0, z=0.0),
        global_translation_is_valid=True,
    )

    _, _, translation, translation_valid = _SomaBodyEvaluator._pose_inputs(data)

    assert not translation_valid
    np.testing.assert_array_equal(translation, np.zeros(3, dtype=np.float32))
