# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard SE3 Retargeter Module.

Maps raw keyboard press state to end-effector delta commands.
"""

import numpy as np
from dataclasses import dataclass

from isaaccapture.deviceio_trackers import EvdevKeyCode
from isaaccapture.retargeting_engine.deviceio_source_nodes import KeyboardHeldType
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

from scipy.spatial.transform import Rotation


# (axis, sign, key) for position (W/S, A/D, Q/E) and rotation (Z/X, T/G, C/V): holding the key
# adds sign * sensitivity to that axis.
_POSITION_BINDINGS = (
    (0, +1, EvdevKeyCode.KeyW),
    (0, -1, EvdevKeyCode.KeyS),
    (1, +1, EvdevKeyCode.KeyA),
    (1, -1, EvdevKeyCode.KeyD),
    (2, +1, EvdevKeyCode.KeyQ),
    (2, -1, EvdevKeyCode.KeyE),
)
_ROTATION_BINDINGS = (
    (0, +1, EvdevKeyCode.KeyZ),
    (0, -1, EvdevKeyCode.KeyX),
    (1, +1, EvdevKeyCode.KeyT),
    (1, -1, EvdevKeyCode.KeyG),
    (2, +1, EvdevKeyCode.KeyC),
    (2, -1, EvdevKeyCode.KeyV),
)


def _axes(bitmap: np.ndarray, bindings, sensitivity: float) -> np.ndarray:
    out = np.zeros(3)
    for axis, sign, key in bindings:
        if bitmap[key]:
            out[axis] += sign * sensitivity
    return out


@dataclass
class KeyboardToSe3RelRetargeterConfig:
    """Configuration for the keyboard-to-SE3-relative retargeter."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8


class KeyboardToSe3RelRetargeter(BaseRetargeter):
    """
    Maps keyboard press state to a 6D end-effector delta command.

    Key bindings:
        W/S: +/-X, A/D: +/-Y, Q/E: +/-Z (position)
        Z/X: +/-roll, T/G: +/-pitch, C/V: +/-yaw (rotation)

    Output is the instantaneous command implied by the currently held keys (scaled by
    sensitivity), not an integrated delta -- matching a continuous-axis input device.

    Z/X are also yaw in :class:`KeyboardToSe2Retargeter`: the two are alternative mappings for
    one keyboard (arm or base), not meant to share one.
    """

    def __init__(self, config: KeyboardToSe3RelRetargeterConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {"keyboard_held": OptionalType(KeyboardHeldType())}

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
        held_keys = inputs["keyboard_held"]
        if held_keys.is_none:
            ee_delta[0] = np.zeros(6, dtype=np.float32)
            return

        bitmap = np.asarray(held_keys[0])
        delta_pos = _axes(bitmap, _POSITION_BINDINGS, self._config.pos_sensitivity)
        delta_euler = _axes(bitmap, _ROTATION_BINDINGS, self._config.rot_sensitivity)
        delta_rot = Rotation.from_euler("XYZ", delta_euler).as_rotvec()

        ee_delta[0] = np.concatenate([delta_pos, delta_rot]).astype(np.float32)
