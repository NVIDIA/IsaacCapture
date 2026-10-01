# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse SE2 Retargeter Module.

Maps raw SpaceMouse translation/rotation state to a base velocity command
(v_x, v_y, omega_z).
"""

from dataclasses import dataclass

import numpy as np

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
class SpaceMouseToSe2RetargeterConfig:
    """Configuration for the spacemouse-to-SE2 base-velocity retargeter."""

    v_x_sensitivity: float = 1.0
    v_y_sensitivity: float = 1.0
    omega_z_sensitivity: float = 1.0


class SpaceMouseToSe2Retargeter(BaseRetargeter):
    """
    Maps raw SpaceMouse translation/rotation axes to a 3D base velocity command
    (v_x, v_y, omega_z).

    Axis bindings:
        Move mouse laterally: base v_x / v_y
        Twist mouse about the z-axis: base omega_z

    Output is the instantaneous command implied by the current axis deflection
    (scaled by sensitivity), not an integrated velocity -- matching a continuous-axis
    input device.
    """

    def __init__(self, config: SpaceMouseToSe2RetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "spacemouse_translation": OptionalType(SpaceMouseTranslationType()),
            "spacemouse_rotation": OptionalType(SpaceMouseRotationType()),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "base_command": TensorGroupType(
                "base_command",
                [
                    NDArrayType(
                        "velocity", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        base_command = outputs["base_command"]
        translation_in = inputs["spacemouse_translation"]
        rotation_in = inputs["spacemouse_rotation"]
        if translation_in.is_none or rotation_in.is_none:
            base_command[0] = np.zeros(3, dtype=np.float32)
            return

        translation = np.asarray(translation_in[0])
        rotation = np.asarray(rotation_in[0])

        velocity = np.zeros(3)
        velocity[1] = self._config.v_y_sensitivity * translation[0]
        velocity[0] = self._config.v_x_sensitivity * translation[1]
        # Unlike Se3's yaw, omega_z is not sign-inverted here -- a faithful port of the
        # legacy Se2SpaceMouse's (likely unintentional) asymmetry with Se3's rotation mapping.
        velocity[2] = self._config.omega_z_sensitivity * rotation[1]

        base_command[0] = velocity.astype(np.float32)
