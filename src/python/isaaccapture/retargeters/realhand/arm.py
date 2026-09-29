# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P7 absolute controller and wrist pose mapping.

The nodes emit the standard seven-element ``ee_pose`` expected by an Isaac Lab task-space
controller. Robot inverse kinematics remains in the environment; these nodes only transform an
OpenXR pose into the calibrated P7 workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    DLDataType,
    HandInput,
    HandInputIndex,
    HandJointIndex,
    NDArrayType,
)


def _as_np(value) -> np.ndarray:
    return np.from_dlpack(value)


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def _step_towards(
    current: np.ndarray, target: np.ndarray, max_step: float
) -> np.ndarray:
    delta = target - current
    norm = np.linalg.norm(delta)
    if max_step <= 0.0 or norm <= max_step:
        return target
    return current + delta * (max_step / norm)


def _rotation_towards(
    current: Rotation, target: Rotation, alpha: float, max_step_deg: float
) -> Rotation:
    relative = target * current.inv()
    angle = relative.magnitude()
    if angle < 1.0e-8:
        return target
    max_step = np.deg2rad(max_step_deg)
    fraction = min(
        float(np.clip(alpha, 0.0, 1.0)),
        max_step / angle if max_step > 0.0 else 1.0,
    )
    return Rotation.from_rotvec(relative.as_rotvec() * fraction) * current


@dataclass
class P7WorkspacePoseConfig:
    """Shared absolute workspace mapping parameters for one P7 arm."""

    input_device: str
    fallback_position: tuple[float, float, float]
    fallback_rotation: tuple[float, float, float, float]
    input_center: tuple[float, float, float]
    workspace_center: tuple[float, float, float]
    position_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_delta: tuple[float, float, float] = (0.65, 0.65, 0.75)
    rotation_offset_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_position_step_m: float = 1.0
    position_smoothing_alpha: float = 1.0
    orientation_smoothing_alpha: float = 1.0
    max_orientation_step_deg: float = 180.0


class _P7WorkspacePoseRetargeter(BaseRetargeter):
    def __init__(self, config: P7WorkspacePoseConfig, name: str) -> None:
        self._config = config
        self._fallback_position = np.asarray(config.fallback_position, dtype=np.float64)
        self._fallback_rotation = Rotation.from_quat(
            _normalized_quaternion(np.asarray(config.fallback_rotation))
        )
        self._input_center = np.asarray(config.input_center, dtype=np.float64)
        self._workspace_center = np.asarray(config.workspace_center, dtype=np.float64)
        self._position_scale = np.asarray(config.position_scale, dtype=np.float64)
        self._max_delta = np.asarray(config.max_delta, dtype=np.float64)
        self._rotation_offset = Rotation.from_euler(
            "XYZ", config.rotation_offset_rpy_deg, degrees=True
        )
        self._position_offset_local = np.asarray(
            config.position_offset_local, dtype=np.float64
        )
        self._smoothed_position = self._fallback_position.copy()
        self._smoothed_rotation = self._fallback_rotation
        self._has_valid_pose = False
        self._last_pose = np.concatenate(
            [self._fallback_position, self._fallback_rotation.as_quat()]
        ).astype(np.float32)
        super().__init__(name=name)

    def output_spec(self) -> RetargeterIOType:
        return {
            "ee_pose": TensorGroupType(
                "ee_pose",
                [
                    NDArrayType(
                        "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def _update_pose(
        self,
        input_position: np.ndarray,
        input_rotation: Rotation,
        output,
        reset: bool,
    ) -> None:
        delta = (input_position - self._input_center) * self._position_scale
        delta = np.clip(delta, -self._max_delta, self._max_delta)
        target_rotation = input_rotation * self._rotation_offset
        target_position = (
            self._workspace_center
            + delta
            + target_rotation.apply(self._position_offset_local)
        )

        if reset or not self._has_valid_pose:
            self._smoothed_position = target_position.copy()
            self._smoothed_rotation = target_rotation
            self._has_valid_pose = True
        else:
            alpha = float(np.clip(self._config.position_smoothing_alpha, 0.0, 1.0))
            filtered = alpha * target_position + (1.0 - alpha) * self._smoothed_position
            self._smoothed_position = _step_towards(
                self._smoothed_position,
                filtered,
                self._config.max_position_step_m,
            )

            target_quaternion = target_rotation.as_quat()
            if np.dot(target_quaternion, self._smoothed_rotation.as_quat()) < 0.0:
                target_quaternion = -target_quaternion
            self._smoothed_rotation = _rotation_towards(
                self._smoothed_rotation,
                Rotation.from_quat(_normalized_quaternion(target_quaternion)),
                self._config.orientation_smoothing_alpha,
                self._config.max_orientation_step_deg,
            )

        self._last_pose = np.concatenate(
            [self._smoothed_position, self._smoothed_rotation.as_quat()]
        ).astype(np.float32)
        output[0] = self._last_pose


class P7ControllerPoseRetargeter(_P7WorkspacePoseRetargeter):
    """Map an absolute OpenXR controller grip pose into one P7 arm workspace."""

    def input_spec(self) -> RetargeterIOType:
        return {self._config.input_device: OptionalType(ControllerInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output = outputs["ee_pose"]
        controller = inputs[self._config.input_device]
        if controller.is_none or not bool(
            controller[ControllerInputIndex.GRIP_IS_VALID]
        ):
            output[0] = self._last_pose
            return
        self._update_pose(
            np.asarray(
                _as_np(controller[ControllerInputIndex.GRIP_POSITION]),
                dtype=np.float64,
            ),
            Rotation.from_quat(
                _normalized_quaternion(
                    _as_np(controller[ControllerInputIndex.GRIP_ORIENTATION])
                )
            ),
            output,
            context.execution_events.reset,
        )


class P7HandPoseRetargeter(_P7WorkspacePoseRetargeter):
    """Map an absolute OpenXR wrist pose into one P7 arm workspace."""

    def input_spec(self) -> RetargeterIOType:
        return {self._config.input_device: OptionalType(HandInput())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output = outputs["ee_pose"]
        hand = inputs[self._config.input_device]
        if hand.is_none:
            output[0] = self._last_pose
            return
        valid = _as_np(hand[HandInputIndex.JOINT_VALID])
        if valid[HandJointIndex.WRIST] == 0:
            output[0] = self._last_pose
            return
        positions = _as_np(hand[HandInputIndex.JOINT_POSITIONS])
        orientations = _as_np(hand[HandInputIndex.JOINT_ORIENTATIONS])
        self._update_pose(
            np.asarray(positions[HandJointIndex.WRIST], dtype=np.float64),
            Rotation.from_quat(
                _normalized_quaternion(orientations[HandJointIndex.WRIST])
            ),
            output,
            context.execution_events.reset,
        )
