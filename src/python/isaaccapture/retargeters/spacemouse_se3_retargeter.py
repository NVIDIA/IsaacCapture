# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SpaceMouse SE3 Retargeter Module.

Maps raw SpaceMouse translation/rotation/button state to end-effector delta commands
and a gripper toggle.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    SpaceMouseButtonsType,
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
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    FloatType,
    NDArrayType,
)

# Button bit position: the left button toggles the gripper. (Bit 1, the right button, requests a
# reset -- a session-level concern handled outside this retargeter, the same way it is for the
# keyboard's R key.)
BUTTON_LEFT = 0


@dataclass
class SpaceMouseToSe3RelRetargeterConfig:
    """Configuration for the spacemouse-to-SE3-relative retargeter."""

    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8


class SpaceMouseToSe3RelRetargeter(BaseRetargeter):
    """
    Maps raw SpaceMouse translation/rotation axes to a 6D end-effector delta command.

    Axis bindings:
        Move mouse laterally: x-y plane position delta
        Move mouse vertically: z position delta (inverted)
        Twist mouse about an axis: rotation delta about the corresponding axis

    Output is the instantaneous command implied by the current axis deflection
    (scaled by sensitivity), not an integrated delta -- matching a continuous-axis
    input device.
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

        translation = np.asarray(translation_in[0])
        rotation = np.asarray(rotation_in[0])
        pos_sens = self._config.pos_sensitivity
        rot_sens = self._config.rot_sensitivity

        delta_pos = np.zeros(3)
        delta_pos[1] = pos_sens * translation[0]
        delta_pos[0] = pos_sens * translation[1]
        delta_pos[2] = -pos_sens * translation[2]

        delta_euler = np.zeros(3)
        delta_euler[1] = rot_sens * rotation[0]
        delta_euler[0] = rot_sens * rotation[1]
        delta_euler[2] = -rot_sens * rotation[2]

        delta_rot = Rotation.from_euler("XYZ", delta_euler).as_rotvec()

        ee_delta[0] = np.concatenate([delta_pos, delta_rot]).astype(np.float32)


class SpaceMouseGripperRetargeter(BaseRetargeter):
    """
    Toggles a gripper open/closed state on each rising edge of the left button.

    Output matches GripperRetargeter's convention: -1.0 when closed, 1.0 when open.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self._closed = False
        self._prev_left_pressed = False

    def input_spec(self) -> RetargeterIOType:
        return {"spacemouse_buttons": OptionalType(SpaceMouseButtonsType())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "gripper_command": TensorGroupType(
                "gripper_command", [FloatType("command")]
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        gripper_out = outputs["gripper_command"]
        buttons_in = inputs["spacemouse_buttons"]
        left_pressed = (
            False
            if buttons_in.is_none
            else bool(np.asarray(buttons_in[0])[BUTTON_LEFT])
        )

        if context.execution_events.reset:
            self._closed = False
            # Sync to the current button state without toggling -- the left button
            # may already be held on a reset frame, and that isn't a rising edge.
            # Leave _prev_left_pressed alone when the device is inactive this
            # frame; overwriting it to False would misread a still-held button as
            # a fresh rising edge once data resumes.
            if not buttons_in.is_none:
                self._prev_left_pressed = left_pressed
            gripper_out[0] = -1.0 if self._closed else 1.0
            return

        if buttons_in.is_none:
            gripper_out[0] = -1.0 if self._closed else 1.0
            return

        if left_pressed and not self._prev_left_pressed:
            self._closed = not self._closed
        self._prev_left_pressed = left_pressed

        gripper_out[0] = -1.0 if self._closed else 1.0
