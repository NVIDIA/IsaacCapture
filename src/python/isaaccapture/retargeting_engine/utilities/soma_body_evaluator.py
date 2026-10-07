# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal evaluation of transported SOMA controls into global joint poses."""

from importlib import import_module
from typing import Any

import numpy as np

from .soma_contract import resolve_soma_body_contract


_QUATERNION_EPS = 1e-12


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_quaternion(value: Any) -> tuple[np.ndarray, bool]:
    quaternion = np.asarray(value, dtype=np.float32)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), False
    norm = float(np.linalg.norm(quaternion.astype(np.float64)))
    if norm <= _QUATERNION_EPS:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), False
    return quaternion / np.float32(norm), True


def _dense_joint_rotations(
    entries: Any, joint_count: int
) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.zeros((joint_count, 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    provided = np.zeros(joint_count, dtype=bool)
    previous = -1
    for entry in entries or ():
        joint = int(entry.joint)
        if joint < 0 or joint >= joint_count:
            raise ValueError(f"SOMA joint index {joint} is out of range")
        if joint <= previous:
            raise ValueError("SOMA joint entries must be sorted and unique")
        rotation = entry.rotation
        rotations[joint], provided[joint] = _normalize_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)
        )
        previous = joint
    return rotations, provided


def _dense_joint_poses(
    entries: Any, joint_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((joint_count, 3), dtype=np.float32)
    orientations = np.zeros((joint_count, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    provided = np.zeros(joint_count, dtype=np.uint8)
    previous = -1
    for entry in entries or ():
        joint = int(entry.joint)
        if joint < 0 or joint >= joint_count:
            raise ValueError(f"SOMA joint index {joint} is out of range")
        if joint <= previous:
            raise ValueError("SOMA joint entries must be sorted and unique")
        pose = entry.pose
        position = np.asarray(
            (pose.position.x, pose.position.y, pose.position.z), dtype=np.float32
        )
        orientation, orientation_valid = _normalize_quaternion(
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        )
        if np.isfinite(position).all() and orientation_valid:
            positions[joint] = position
            orientations[joint] = orientation
            provided[joint] = 1
        previous = joint
    return positions, orientations, provided


def _translation_input(data: Any) -> tuple[np.ndarray, bool]:
    translation = data.global_translation
    valid = bool(data.global_translation_is_valid) and translation is not None
    value = np.asarray(
        (translation.x, translation.y, translation.z) if valid else (0.0, 0.0, 0.0),
        dtype=np.float32,
    )
    if not np.isfinite(value).all():
        return np.zeros(3, dtype=np.float32), False
    return value, valid


class _SomaBodyEvaluator:
    """Run upstream SOMA FK and expose all public transported joints."""

    def __init__(self, layer: Any) -> None:
        reference_pose = resolve_soma_body_contract(layer)
        joint_names = tuple(str(name) for name in layer.public_joint_names)
        parent_ids = _numpy(layer.output_joint_parent_ids).astype(np.int64)

        self.layer = layer
        self.reference_pose = reference_pose
        self.joint_names = joint_names[1:]
        self.parent_ids = tuple(int(parent) for parent in parent_ids)
        self.bones = tuple(
            (int(parent) - 1, child - 1)
            for child, parent in enumerate(parent_ids[1:], 1)
            if parent != 0
        )

    def _joint_validity(
        self, control_valid: np.ndarray, translation_valid: bool
    ) -> np.ndarray:
        valid = np.zeros(78, dtype=np.uint8)
        valid[0] = translation_valid
        for child in range(1, 78):
            valid[child] = valid[self.parent_ids[child]] and control_valid[child - 1]
        return valid[1:]

    @staticmethod
    def _pose_inputs(
        data: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        rotations, control_valid = _dense_joint_rotations(data.joint_rotations, 77)

        translation_xyz, translation_valid = _translation_input(data)
        return rotations, control_valid, translation_xyz, translation_valid

    def evaluate(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = import_module("torch")
        transforms = import_module("soma.geometry.transforms")

        rotations, control_valid, translation_xyz, translation_valid = (
            self._pose_inputs(data)
        )

        with torch.no_grad():
            output = self.layer.pose(
                transforms.quaternion_xyzw_to_matrix(
                    torch.from_numpy(rotations).unsqueeze(0)
                ),
                transl=torch.from_numpy(translation_xyz).unsqueeze(0),
                pose2rot=False,
                fk_only=True,
                apply_correctives=False,
                reference_pose=self.reference_pose,
            )
            orientations = transforms.matrix_to_quaternion_xyzw(
                output["transforms"][0, 1:, :3, :3]
            )
        return (
            _numpy(output["joints"][0]).astype(np.float32, copy=False),
            _numpy(orientations).astype(np.float32, copy=False),
            self._joint_validity(control_valid, translation_valid),
        )
