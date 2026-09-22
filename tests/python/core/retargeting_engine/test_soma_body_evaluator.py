# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np

from isaaccapture.retargeting_engine.utilities.soma_body_evaluator import (
    _SomaBodyEvaluator,
)


def evaluator_with_two_branches() -> _SomaBodyEvaluator:
    layer = MagicMock()
    layer.public_joint_names = ("Root", *(f"Joint{index}" for index in range(77)))
    layer.output_joint_parent_ids = np.array([0, 0, 1, 1, *range(3, 77)])
    return _SomaBodyEvaluator(layer)


def test_evaluator_exposes_native_joint_order_and_topology():
    evaluator = evaluator_with_two_branches()
    assert evaluator.joint_names[:3] == ("Joint0", "Joint1", "Joint2")
    assert evaluator.bones[:3] == ((0, 1), (0, 2), (2, 3))
    assert len(evaluator.joint_names) == 77
    assert len(evaluator.bones) == 76


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
        joint_rotations=MagicMock(
            rotations=rotations,
            is_valid=np.ones(77, dtype=np.uint8),
        ),
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
