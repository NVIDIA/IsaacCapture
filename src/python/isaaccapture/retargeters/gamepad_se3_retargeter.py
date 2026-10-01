# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gamepad SE3 Retargeter Module.

Maps raw gamepad button/axis state to end-effector delta commands and a gripper toggle.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    GamepadAxesType,
    GamepadButtonsType,
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
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    FloatType,
    NDArrayType,
)

# Linux joystick-API axis indices for a typical Xbox-style pad under the xpad driver.
# Axis convention: pushing a stick left/up reports a negative value, right/down positive
# (standard HID convention). The D-pad is reported as a hat switch (axes 6/7) on most
# xpad-driver controllers rather than as buttons.
AXIS_LEFT_X, AXIS_LEFT_Y = 0, 1
AXIS_RIGHT_X, AXIS_RIGHT_Y = 3, 4
AXIS_DPAD_X, AXIS_DPAD_Y = 6, 7

# Typical xpad button ordering: A=0, B=1, X=2, Y=3, LB=4, RB=5, ...
BUTTON_X = 2


@dataclass
class GamepadToSe3RelRetargeterConfig:
    """Configuration for the gamepad-to-SE3-relative retargeter."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    dead_zone: float = 0.01


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
        return {"gamepad_axes": OptionalType(GamepadAxesType())}

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
        axes_in = inputs["gamepad_axes"]
        if axes_in.is_none:
            ee_delta[0] = np.zeros(6, dtype=np.float32)
            return

        axes = np.asarray(axes_in[0])
        pos_sens = self._config.pos_sensitivity
        rot_sens = self._config.rot_sensitivity
        dead_zone = self._config.dead_zone

        def deadzoned(value: float) -> float:
            return 0.0 if abs(value) < dead_zone else value

        delta_pos = np.zeros(3)
        delta_pos[0] = -deadzoned(axes[AXIS_LEFT_Y]) * pos_sens
        delta_pos[1] = deadzoned(axes[AXIS_LEFT_X]) * pos_sens
        delta_pos[2] = -deadzoned(axes[AXIS_RIGHT_Y]) * pos_sens

        delta_euler = np.zeros(3)
        delta_euler[0] = -deadzoned(axes[AXIS_DPAD_X]) * rot_sens * 0.8
        delta_euler[1] = deadzoned(axes[AXIS_DPAD_Y]) * rot_sens * 0.8
        delta_euler[2] = deadzoned(axes[AXIS_RIGHT_X]) * rot_sens

        delta_rot = Rotation.from_euler("XYZ", delta_euler).as_rotvec()

        ee_delta[0] = np.concatenate([delta_pos, delta_rot]).astype(np.float32)


class GamepadGripperRetargeter(BaseRetargeter):
    """
    Toggles a gripper open/closed state on each rising edge of the X button.

    Output matches GripperRetargeter's convention: -1.0 when closed, 1.0 when open.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self._closed = False
        self._prev_x_pressed = False

    def input_spec(self) -> RetargeterIOType:
        return {"gamepad_buttons": OptionalType(GamepadButtonsType())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "gripper_command": TensorGroupType(
                "gripper_command", [FloatType("command")]
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        gripper_out = outputs["gripper_command"]
        buttons_in = inputs["gamepad_buttons"]
        x_pressed = (
            False if buttons_in.is_none else bool(np.asarray(buttons_in[0])[BUTTON_X])
        )

        if context.execution_events.reset:
            self._closed = False
            # Sync to the current button state without toggling -- X may already be
            # held on a reset frame, and that isn't a rising edge. Leave
            # _prev_x_pressed alone when the device is inactive this frame;
            # overwriting it to False would misread a still-held button as a fresh
            # rising edge once data resumes.
            if not buttons_in.is_none:
                self._prev_x_pressed = x_pressed
            gripper_out[0] = -1.0 if self._closed else 1.0
            return

        if buttons_in.is_none:
            gripper_out[0] = -1.0 if self._closed else 1.0
            return

        if x_pressed and not self._prev_x_pressed:
            self._closed = not self._closed
        self._prev_x_pressed = x_pressed

        gripper_out[0] = -1.0 if self._closed else 1.0
