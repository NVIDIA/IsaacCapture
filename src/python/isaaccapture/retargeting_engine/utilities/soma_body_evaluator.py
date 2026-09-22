# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal evaluation of transported SOMA controls into global joint poses."""

from importlib import import_module
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class _SomaBodyEvaluator:
    """Run upstream SOMA FK and expose all public transported joints."""

    def __init__(self, layer: Any) -> None:
        joint_names = tuple(str(name) for name in layer.public_joint_names)
        parent_ids = _numpy(layer.output_joint_parent_ids).astype(np.int64)
        if len(joint_names) != 78 or joint_names[0].upper() != "ROOT":
            raise ValueError("SOMA evaluator requires Root plus 77 public joints")
        if parent_ids.shape != (78,):
            raise ValueError("SOMA evaluator requires 78 public parent IDs")
        if any(
            parent < 0 or parent >= child
            for child, parent in enumerate(parent_ids[1:], 1)
        ):
            raise ValueError("SOMA public joints must follow parent-before-child order")

        self.layer = layer
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
        # Nested FlatBuffer structs are optional on the wire even though the
        # Python constructor supplies defaults. Keep vendor omissions safe.
        joint_rotations = data.joint_rotations
        if joint_rotations is None:
            control_valid = np.zeros(77, dtype=bool)
            rotations = np.zeros((77, 4), dtype=np.float32)
            rotations[:, 3] = 1.0
        else:
            control_valid = np.asarray(joint_rotations.is_valid, dtype=bool)
            rotations = np.asarray(joint_rotations.rotations, dtype=np.float32).copy()
            rotations[~control_valid] = (0.0, 0.0, 0.0, 1.0)

        translation = data.global_translation
        translation_valid = (
            bool(data.global_translation_is_valid) and translation is not None
        )
        translation_xyz = np.asarray(
            [translation.x, translation.y, translation.z]
            if translation_valid
            else [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
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
            )
            orientations = transforms.matrix_to_quaternion_xyzw(
                output["transforms"][0, 1:, :3, :3]
            )
        return (
            _numpy(output["joints"][0]).astype(np.float32, copy=False),
            _numpy(orientations).astype(np.float32, copy=False),
            self._joint_validity(control_valid, translation_valid),
        )
