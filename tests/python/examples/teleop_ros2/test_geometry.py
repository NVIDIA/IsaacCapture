# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ROS pose geometry helpers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaaccapture_examples.teleop_ros2.geometry import (
    apply_manus_controller_to_hand_pose,
    apply_relative_pose,
    apply_transform_to_pose,
    to_pose,
)


def _orientation(pose) -> np.ndarray:
    return np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )


def _position(pose) -> np.ndarray:
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def test_apply_transform_rotates_translates_and_changes_orientation_basis() -> None:
    pose = to_pose(
        [1.0, 0.0, 0.0],
        Rotation.from_euler("x", 30.0, degrees=True).as_quat(),
    )
    basis_rotation = Rotation.from_euler("z", 90.0, degrees=True)

    transformed = apply_transform_to_pose(
        pose,
        rotation=basis_rotation,
        translation=[1.0, 2.0, 3.0],
    )

    np.testing.assert_allclose(_position(transformed), [1.0, 3.0, 3.0], atol=1e-7)
    expected_orientation = (
        basis_rotation * Rotation.from_quat(_orientation(pose)) * basis_rotation.inv()
    )
    np.testing.assert_allclose(
        Rotation.from_quat(_orientation(transformed)).as_matrix(),
        expected_orientation.as_matrix(),
        atol=1e-7,
    )


def test_apply_transform_returns_a_new_pose_without_mutating_input() -> None:
    pose = to_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])

    transformed = apply_transform_to_pose(pose, translation=[4.0, 5.0, 6.0])

    assert transformed is not pose
    np.testing.assert_allclose(_position(pose), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(_position(transformed), [5.0, 7.0, 9.0])


@pytest.mark.parametrize(
    ("side", "expected_position", "expected_orientation"),
    (
        (
            "left",
            [0.0103315472, 0.055536544, -0.0566476072],
            [
                -0.1315856570103413,
                -0.3586609382547703,
                0.9111816820492541,
                -0.15425786377769118,
            ],
        ),
        (
            "right",
            [0.0103315472, -0.055536544, -0.0566476072],
            [
                -0.1315856570103413,
                0.3586609382547703,
                0.9111816820492541,
                0.15425786377769118,
            ],
        ),
    ),
)
def test_manus_controller_calibration_matches_known_good_transform(
    side: str,
    expected_position: list[float],
    expected_orientation: list[float],
) -> None:
    controller_pose = to_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    calibrated_pose = apply_manus_controller_to_hand_pose(controller_pose, side)

    np.testing.assert_allclose(_position(calibrated_pose), expected_position)
    np.testing.assert_allclose(_orientation(calibrated_pose), expected_orientation)


def test_manus_controller_calibration_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        apply_manus_controller_to_hand_pose(
            to_pose([0.0, 0.0, 0.0]),
            "center",
        )


def test_relative_pose_with_unrotated_reference_subtracts_position() -> None:
    reference = to_pose([0.0, 0.10, 1.60])
    pose = to_pose([-0.20, 0.15, 1.20])

    relative = apply_relative_pose(reference, pose)

    np.testing.assert_allclose(_position(relative), [-0.20, 0.05, -0.40], atol=1e-7)


def test_relative_pose_rotates_the_offset_into_the_reference_basis() -> None:
    # Subtracting positions alone would leave the offset unchanged, so a yawed
    # reference is what distinguishes a correct implementation from that one.
    reference = to_pose(
        [0.0, 0.10, 1.60],
        Rotation.from_euler("z", 90.0, degrees=True).as_quat(),
    )
    pose = to_pose([-0.20, 0.15, 1.20])

    relative = apply_relative_pose(reference, pose)

    np.testing.assert_allclose(_position(relative), [0.05, 0.20, -0.40], atol=1e-7)


def test_relative_pose_composes_orientation_instead_of_changing_its_basis() -> None:
    # Conjugation would return the pose orientation unchanged here; composition
    # returns the reference's inverse.
    reference_rotation = Rotation.from_euler("z", 40.0, degrees=True)
    reference = to_pose([0.0, 0.0, 0.0], reference_rotation.as_quat())
    pose = to_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

    relative = apply_relative_pose(reference, pose)

    np.testing.assert_allclose(
        Rotation.from_quat(_orientation(relative)).as_matrix(),
        reference_rotation.inv().as_matrix(),
        atol=1e-7,
    )


def test_relative_pose_returns_a_new_pose_without_mutating_inputs() -> None:
    reference = to_pose([1.0, 0.0, 0.0])
    pose = to_pose([3.0, 0.0, 0.0])

    relative = apply_relative_pose(reference, pose)

    assert relative is not pose
    assert relative is not reference
    np.testing.assert_allclose(_position(reference), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(_position(pose), [3.0, 0.0, 0.0])
    np.testing.assert_allclose(_position(relative), [2.0, 0.0, 0.0])
