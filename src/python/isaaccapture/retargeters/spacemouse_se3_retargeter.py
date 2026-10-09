# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse SE3 Retargeter Module.

Maps SpaceMouse translation and rotation axes to end-effector delta commands.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SpaceMouseRotationType,
    SpaceMouseTranslationType,
)
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import DLDataType, NDArrayType


@dataclass
class SpaceMouseToSe3RelRetargeterConfig:
    """Configuration for the SpaceMouse-to-SE3-relative retargeter."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8


class SpaceMouseToSe3RelRetargeter(BaseRetargeter):
    """
    Maps SpaceMouse translation and rotation axes to a 6D end-effector delta command.

    Axis bindings (device axes are [x, y, z]):
        Translation y -> +X, translation x -> +Y, translation z -> -Z (position)
        Rotation y -> +roll, rotation x -> +pitch, rotation z -> -yaw

    Output is the instantaneous command implied by the current deflection (scaled by
    sensitivity), not an integrated delta -- matching a continuous-axis input device.
    """

    def __init__(self, config: SpaceMouseToSe3RelRetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "spacemouse_translation": OptionalType(SpaceMouseTranslationType()),
            "spacemouse_rotation": OptionalType(SpaceMouseRotationType()),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "ee_delta": TensorGroupType(
                "ee_delta",
                [
                    NDArrayType(
                        "delta", shape=(6,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        ee_delta = outputs["ee_delta"]
        translation_in = inputs["spacemouse_translation"]
        rotation_in = inputs["spacemouse_rotation"]
        if translation_in.is_none or rotation_in.is_none:
            ee_delta[0] = np.zeros(6, dtype=np.float32)
            return

        tx, ty, tz = np.asarray(translation_in[0], dtype=np.float64)
        rx, ry, rz = np.asarray(rotation_in[0], dtype=np.float64)
        delta_pos = self._config.pos_sensitivity * np.array([ty, tx, -tz])
        delta_euler = self._config.rot_sensitivity * np.array([ry, rx, -rz])
        delta_rot = Rotation.from_euler("XYZ", delta_euler).as_rotvec()

        ee_delta[0] = np.concatenate([delta_pos, delta_rot]).astype(np.float32)
