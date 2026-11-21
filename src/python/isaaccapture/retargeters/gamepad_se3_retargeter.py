# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad SE3 Retargeter Module.

Maps raw gamepad axis state to end-effector delta commands.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

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
class GamepadToSe3RelRetargeterConfig:
    """Configuration for the gamepad-to-SE3-relative retargeter."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    # An idle Xbox 360 pad's stick drift reaches 0.14.
    dead_zone: float = 0.15


class GamepadToSe3RelRetargeter(BaseRetargeter):
    """
    Maps gamepad stick/dpad state to a 6D end-effector delta command.

    Stick/D-pad bindings:
        Left stick up/down: +/-X, Left stick right/left: +/-Y,
        Right stick up/down: +/-Z (position)
        D-pad left/right: +/-roll, D-pad down/up: +/-pitch,
        Right stick right/left: +/-yaw (rotation)

    Output is the instantaneous command implied by the current stick/dpad deflection
    (scaled by sensitivity), not an integrated delta -- matching a continuous-axis
    input device.
    """

    def __init__(self, config: GamepadToSe3RelRetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {"gamepad": OptionalType(GamepadInput())}

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
        gamepad = inputs["gamepad"]
        if gamepad.is_none:
            ee_delta[0] = np.zeros(6, dtype=np.float32)
            return

        def value(field) -> float:
            return float(gamepad[field])

        i = GamepadInputIndex
        pos_sens = self._config.pos_sensitivity
        rot_sens = self._config.rot_sensitivity
        dead_zone = self._config.dead_zone

        def deadzoned(value: float) -> float:
            return 0.0 if abs(value) < dead_zone else value

        delta_pos = np.zeros(3)
        delta_pos[0] = deadzoned(value(i.THUMBSTICK_LEFT_Y)) * pos_sens
        delta_pos[1] = deadzoned(value(i.THUMBSTICK_LEFT_X)) * pos_sens
        delta_pos[2] = deadzoned(value(i.THUMBSTICK_RIGHT_Y)) * pos_sens

        delta_euler = np.zeros(3)
        delta_euler[0] = (
            (value(i.DPAD_LEFT_CLICK) - value(i.DPAD_RIGHT_CLICK)) * rot_sens * 0.8
        )
        delta_euler[1] = (
            (value(i.DPAD_DOWN_CLICK) - value(i.DPAD_UP_CLICK)) * rot_sens * 0.8
        )
        delta_euler[2] = deadzoned(value(i.THUMBSTICK_RIGHT_X)) * rot_sens

        delta_rot = Rotation.from_euler("XYZ", delta_euler).as_rotvec()

        ee_delta[0] = np.concatenate([delta_pos, delta_rot]).astype(np.float32)
