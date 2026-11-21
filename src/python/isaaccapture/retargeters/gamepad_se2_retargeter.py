# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad SE2 Retargeter Module.

Maps raw gamepad axis state to a base velocity command (v_x, v_y, omega_z).
"""

from dataclasses import dataclass

import numpy as np

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    GamepadInput,
    GamepadInputIndex,
    NDArrayType,
)


@dataclass
class GamepadToSe2RetargeterConfig:
    """Configuration for the gamepad-to-SE2 base-velocity retargeter."""

    v_x_sensitivity: float = 1.0
    v_y_sensitivity: float = 1.0
    omega_z_sensitivity: float = 1.0
    # An idle Xbox 360 pad's stick drift reaches 0.14.
    dead_zone: float = 0.15


class GamepadToSe2Retargeter(BaseRetargeter):
    """
    Maps gamepad stick state to a 3D base velocity command (v_x, v_y, omega_z).

    Stick bindings:
        Left stick up/down: +/-v_x      Left stick right/left: +/-v_y
        Right stick right/left: +/-omega_z

    Output is the instantaneous command implied by the current stick deflection
    (scaled by sensitivity), not an integrated velocity -- matching a continuous-axis
    input device.
    """

    def __init__(self, config: GamepadToSe2RetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {"gamepad": OptionalType(GamepadInput())}

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
        gamepad = inputs["gamepad"]
        if gamepad.is_none:
            base_command[0] = np.zeros(3, dtype=np.float32)
            return

        dead_zone = self._config.dead_zone

        def deadzoned(value: float) -> float:
            return 0.0 if abs(value) < dead_zone else value

        v_x = (
            deadzoned(float(gamepad[GamepadInputIndex.THUMBSTICK_LEFT_Y]))
            * self._config.v_x_sensitivity
        )
        v_y = (
            deadzoned(float(gamepad[GamepadInputIndex.THUMBSTICK_LEFT_X]))
            * self._config.v_y_sensitivity
        )
        omega_z = (
            deadzoned(float(gamepad[GamepadInputIndex.THUMBSTICK_RIGHT_X]))
            * self._config.omega_z_sensitivity
        )

        base_command[0] = np.array([v_x, v_y, omega_z], dtype=np.float32)
