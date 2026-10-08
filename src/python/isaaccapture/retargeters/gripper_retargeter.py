# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gripper Retargeter Module.

Gripper control from hand tracking or controller input (GripperRetargeter) and from a keyboard
toggle (KeyboardGripperRetargeter).
"""

import numpy as np
from dataclasses import dataclass

from isaaccapture.deviceio_trackers import EvdevKeyCode
from isaaccapture.retargeting_engine.deviceio_source_nodes import KeyboardPressedType
from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    TensorGroupType,
    OptionalType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    HandInput,
    ControllerInput,
    FloatType,
    HandInputIndex,
    HandJointIndex,
    ControllerInputIndex,
)


@dataclass
class GripperRetargeterConfig:
    """Configuration for gripper retargeter."""

    hand_side: str = "right"
    gripper_close_meters: float = 0.03
    gripper_open_meters: float = 0.05
    controller_threshold: float = 0.5


class GripperRetargeter(BaseRetargeter):
    """
    Retargeter specifically for gripper control based on hand tracking data.

    This retargeter analyzes the distance between thumb and index finger tips to determine
    whether the gripper should be open or closed. It includes hysteresis to prevent rapid
    toggling between states when the finger distance is near the threshold.
    """

    def __init__(self, config: GripperRetargeterConfig, name: str) -> None:
        self._config = config
        if self._config.hand_side not in ["left", "right"]:
            raise ValueError(
                f"hand_side must be 'left' or 'right', got: {self._config.hand_side}"
            )

        super().__init__(name=name)

        self._previous_gripper_command = False  # False = open, True = closed

    def input_spec(self) -> RetargeterIOType:
        """Requires hand tracking input and controller input (both Optional)."""
        spec: RetargeterIOType = {}

        if self._config.hand_side == "left":
            spec["hand_left"] = OptionalType(HandInput())
        else:
            spec["hand_right"] = OptionalType(HandInput())

        spec[f"controller_{self._config.hand_side}"] = OptionalType(ControllerInput())

        return spec

    def output_spec(self) -> RetargeterIOType:
        """Outputs a single float value (-1.0 for closed, 1.0 for open)."""
        return {
            "gripper_command": TensorGroupType(
                "gripper_command", [FloatType("command")]
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Computes gripper command based on controller trigger (priority) or pinch distance (fallback)."""
        if context.execution_events.reset:
            self._previous_gripper_command = False

        gripper_out = outputs["gripper_command"]

        used_controller = False
        controller_key = f"controller_{self._config.hand_side}"
        controller_group = inputs[controller_key]

        if not controller_group.is_none:
            used_controller = True
            trigger_value = controller_group[ControllerInputIndex.TRIGGER_VALUE]

            if trigger_value > self._config.controller_threshold:
                self._previous_gripper_command = True  # Close
            else:
                self._previous_gripper_command = False  # Open

        if not used_controller:
            hand_key = f"hand_{self._config.hand_side}"
            hand_group = inputs[hand_key]

            if hand_group.is_none:
                gripper_out[0] = 1.0  # Default open
                return

            # Get joint positions
            # Shape (26, 3)
            joint_positions = np.from_dlpack(hand_group[HandInputIndex.JOINT_POSITIONS])
            joint_valid = np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID])

            # OpenXR indices: Thumb Tip = 5, Index Tip = 10
            # 0: palm, 1: wrist
            # 2: thumb_metacarpal, 3: thumb_proximal, 4: thumb_distal, 5: thumb_tip
            # 6: index_metacarpal, 7: index_proximal, 8: index_intermediate, 9: index_distal, 10: index_tip

            if (
                joint_valid[HandJointIndex.THUMB_TIP]
                and joint_valid[HandJointIndex.INDEX_TIP]
            ):
                thumb_pos = joint_positions[HandJointIndex.THUMB_TIP]
                index_pos = joint_positions[HandJointIndex.INDEX_TIP]

                distance = np.linalg.norm(thumb_pos - index_pos)

                if distance > self._config.gripper_open_meters:
                    self._previous_gripper_command = False  # Open
                elif distance < self._config.gripper_close_meters:
                    self._previous_gripper_command = True  # Close

        # Output: -1.0 if closed, 1.0 if open (matching IsaacLab implementation)
        gripper_out[0] = -1.0 if self._previous_gripper_command else 1.0


class KeyboardGripperRetargeter(BaseRetargeter):
    """
    Toggles a gripper open/closed state on each press of the K key.

    Consumes ``keyboard_pressed`` (keys with a press event this frame), so a tap shorter
    than a frame still toggles and a key held across frames or across a reset never
    re-toggles -- no edge state to keep. Toggles at most once per frame: two taps within one
    frame count as one.

    Output matches GripperRetargeter's convention: -1.0 when closed, 1.0 when open.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self._closed = False

    def input_spec(self) -> RetargeterIOType:
        return {"keyboard_pressed": OptionalType(KeyboardPressedType())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "gripper_command": TensorGroupType(
                "gripper_command", [FloatType("command")]
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        gripper_out = outputs["gripper_command"]
        pressed = inputs["keyboard_pressed"]

        if context.execution_events.reset:
            # A press landing on the reset frame is consumed by the reset.
            self._closed = False
        elif not pressed.is_none and np.asarray(pressed[0])[EvdevKeyCode.KeyK]:
            self._closed = not self._closed

        gripper_out[0] = -1.0 if self._closed else 1.0
