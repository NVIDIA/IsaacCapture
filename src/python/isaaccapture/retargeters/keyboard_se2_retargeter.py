# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard SE2 Retargeter Module.

Maps raw keyboard press state to a base velocity command (v_x, v_y, omega_z).
"""

import numpy as np
from dataclasses import dataclass

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    EvdevKeyCode,
    KeyboardAllKeysType,
)
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    TensorGroupType,
    OptionalType,
)
from isaaccapture.retargeting_engine.tensor_types import NDArrayType, DLDataType


@dataclass
class KeyboardToSe2RetargeterConfig:
    """Configuration for the keyboard-to-SE2 base-velocity retargeter."""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0


class KeyboardToSe2Retargeter(BaseRetargeter):
    """
    Maps keyboard press state to a 3D base velocity command (v_x, v_y, omega_z).

    Key bindings (matching Isaac Lab's legacy Se2Keyboard):
        Numpad 8 / Arrow Up: +v_x        Numpad 2 / Arrow Down: -v_x
        Numpad 4 / Arrow Left: +v_y      Numpad 6 / Arrow Right: -v_y
        Numpad 7 / Z: +omega_z           Numpad 9 / X: -omega_z

    Consumes the "keyboard_all_keys" bitmap (rather than the fixed 13-key SE3
    subset) since numpad and arrow keys fall outside it.

    Output is the instantaneous command implied by the currently held keys (scaled
    by sensitivity), not an integrated velocity -- matching a continuous-axis input
    device.
    """

    def __init__(self, config: KeyboardToSe2RetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {"keyboard_all_keys": OptionalType(KeyboardAllKeysType())}

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
        all_keys = inputs["keyboard_all_keys"]
        if all_keys.is_none:
            base_command[0] = np.zeros(3, dtype=np.float32)
            return

        bitmap = np.asarray(all_keys[0])
        v_x_sens = self._config.v_x_sensitivity
        v_y_sens = self._config.v_y_sensitivity
        omega_z_sens = self._config.omega_z_sensitivity

        velocity = np.zeros(3)
        velocity[0] += (
            v_x_sens
            if (bitmap[EvdevKeyCode.KEY_KP8] or bitmap[EvdevKeyCode.KEY_UP])
            else 0.0
        )
        velocity[0] -= (
            v_x_sens
            if (bitmap[EvdevKeyCode.KEY_KP2] or bitmap[EvdevKeyCode.KEY_DOWN])
            else 0.0
        )
        velocity[1] += (
            v_y_sens
            if (bitmap[EvdevKeyCode.KEY_KP4] or bitmap[EvdevKeyCode.KEY_LEFT])
            else 0.0
        )
        velocity[1] -= (
            v_y_sens
            if (bitmap[EvdevKeyCode.KEY_KP6] or bitmap[EvdevKeyCode.KEY_RIGHT])
            else 0.0
        )
        velocity[2] += (
            omega_z_sens
            if (bitmap[EvdevKeyCode.KEY_KP7] or bitmap[EvdevKeyCode.KEY_Z])
            else 0.0
        )
        velocity[2] -= (
            omega_z_sens
            if (bitmap[EvdevKeyCode.KEY_KP9] or bitmap[EvdevKeyCode.KEY_X])
            else 0.0
        )

        base_command[0] = velocity.astype(np.float32)
