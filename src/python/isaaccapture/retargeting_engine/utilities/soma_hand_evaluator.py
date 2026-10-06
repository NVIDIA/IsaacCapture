# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal evaluation of transported SOMA hand controls."""

from importlib import import_module
from typing import Any

import numpy as np

from .soma_body_evaluator import _numpy


class _SomaHandEvaluator:
    """Run upstream SOMA hand FK and expose its 25 evaluated joints."""

    def __init__(self, layer: Any) -> None:
        joint_names = tuple(str(name) for name in layer.rig_data["joint_names"])
        parent_ids = _numpy(layer.joint_parent_ids).astype(np.int64)
        if len(joint_names) != 25 or parent_ids.shape != (25,):
            raise ValueError("SOMA hand evaluator requires 25 joints")
        if parent_ids[0] != 0 or any(
            parent < 0 or parent >= child
            for child, parent in enumerate(parent_ids[1:], 1)
        ):
            raise ValueError("SOMA hand joints must follow parent-before-child order")

        self.layer = layer
        self.joint_names = joint_names
        self.parent_ids = tuple(int(parent) for parent in parent_ids)
        self.bones = tuple(
            (int(parent), child) for child, parent in enumerate(parent_ids[1:], 1)
        )

    def _joint_validity(
        self, control_valid: np.ndarray, translation_valid: bool
    ) -> np.ndarray:
        valid = np.zeros(25, dtype=np.uint8)
        valid[0] = translation_valid and control_valid[0]
        for child in range(1, 25):
            valid[child] = valid[self.parent_ids[child]] and control_valid[child]
        return valid

    @staticmethod
    def _pose_inputs(
        data: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        joint_rotations = data.joint_rotations
        if joint_rotations is None:
            control_valid = np.zeros(25, dtype=bool)
            rotations = np.zeros((25, 4), dtype=np.float32)
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
                global_translation=torch.from_numpy(translation_xyz).unsqueeze(0),
                pose2rot=False,
                fk_only=True,
                apply_correctives=False,
            )
            orientations = transforms.matrix_to_quaternion_xyzw(
                output["transforms"][0, :, :3, :3]
            )
        return (
            _numpy(output["joints"][0]).astype(np.float32, copy=False),
            _numpy(orientations).astype(np.float32, copy=False),
            self._joint_validity(control_valid, translation_valid),
        )
