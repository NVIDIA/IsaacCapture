# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TriHand Motion Controller Retargeter Module.

Retargeter that maps motion controller inputs to G1 robot hand joint angles.
This class implements simple logic to map button presses and trigger/joystick inputs
to finger joint angles, specifically designed for dexterous hand control.

Based on IsaacLab's DexMotionController, adapted for Isaac Teleop's retargeting framework.
"""

import numpy as np
from typing import List
from dataclasses import dataclass

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group import (
    TensorGroup,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import OptionalType
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    RobotHandJoints,
)


@dataclass
class TriHandMotionControllerConfig:
    """Configuration for the dexterous motion controller retargeter.

    Attributes:
        hand_joint_names: List of joint names in the robot hand (in order)
        controller_side: Which controller to use ("left" or "right")
    """

    hand_joint_names: List[str]
    controller_side: str = "left"


class TriHandMotionControllerRetargeter(BaseRetargeter):
    """
    Retargeter that maps motion controller inputs to G1 robot hand joint angles.

    This retargeter implements simple logic to map controller button presses
    and analog inputs (trigger, squeeze) to finger joint angles. It's designed
    for quick testing and simple dexterous hand control without hand tracking.

    Inputs:
        - "controller_{side}": Motion controller data (trigger, squeeze, buttons)

    Outputs:
        - "hand_joints": Joint angles for the robot hand (7 DOFs)

    The mapping is:
    - Trigger: Controls index finger
    - Squeeze: Controls middle finger
    - Thumb curl: Proximal/distal angles follow max(trigger, squeeze)
    - Thumb rotation: Follows 0.5 * (trigger - squeeze), with sign per hand;
      equal inputs cancel rotation while keeping the thumb curled

    Output DOF order:
    0. Thumb Rotation
    1. Thumb Middle/Proximal
    2. Thumb Distal
    3. Index Proximal
    4. Index Distal
    5. Middle Proximal
    6. Middle Distal
    """

    # Mapping constants
    THUMB_PROXIMAL_SCALE = 0.4
    THUMB_DISTAL_SCALE = 0.7
    THUMB_ROTATION_TRIGGER_SCALE = 0.5
    THUMB_ROTATION_SQUEEZE_SCALE = 0.5

    def __init__(self, config: TriHandMotionControllerConfig, name: str) -> None:
        """
        Initialize the TriHand motion controller retargeter.

        Args:
            config: Configuration object for the retargeter
            name: Name identifier for this retargeter (must be unique)
        """
        self._config = config
        self._hand_joint_names = config.hand_joint_names
        self._controller_side = config.controller_side.lower()

        if self._controller_side not in ["left", "right"]:
            raise ValueError(
                f"controller_side must be 'left' or 'right', got: {self._controller_side}"
            )

        self._is_left = self._controller_side == "left"

        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Define input collections for motion controller (Optional)."""
        if self._controller_side == "left":
            return {"controller_left": OptionalType(ControllerInput())}
        else:
            return {"controller_right": OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        """Define output collections for robot hand joint angles."""
        return {
            "hand_joints": RobotHandJoints(
                f"hand_joints_{self._controller_side}", self._hand_joint_names
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Execute the motion controller to hand joint mapping.

        Args:
            inputs: Dict with motion controller data
            outputs: Dict with "hand_joints" TensorGroup for robot joint angles
            context: ComputeContext (unused by this retargeter).
        """
        output_group = outputs["hand_joints"]

        controller_input_key = f"controller_{self._controller_side}"
        controller_group = inputs[controller_input_key]

        if controller_group.is_none:
            for i in range(len(self._hand_joint_names)):
                output_group[i] = 0.0
            return

        trigger_value = float(controller_group[ControllerInputIndex.TRIGGER_VALUE])
        squeeze_value = float(controller_group[ControllerInputIndex.SQUEEZE_VALUE])

        # Map controller inputs to hand joints (7 DOFs)
        hand_joints = self._map_to_hand_joints(trigger_value, squeeze_value)

        # For left hand, negate joint angles for proper mirroring
        if self._is_left:
            hand_joints = -hand_joints

        # If we have exactly 7 joint names, use direct mapping
        if len(self._hand_joint_names) == 7:
            for i in range(7):
                output_group[i] = float(hand_joints[i])
        else:
            # If different number of joints, need to map by name
            # For now, just fill what we can
            for i in range(min(len(self._hand_joint_names), 7)):
                output_group[i] = float(hand_joints[i])

    def _map_to_hand_joints(self, trigger: float, squeeze: float) -> np.ndarray:
        """
        Map controller inputs to hand joint angles.

        Args:
            trigger: Trigger value (0.0 to 1.0)
            squeeze: Squeeze/grip value (0.0 to 1.0)

        Returns:
            Array of 7 joint angles
        """
        hand_joints = np.zeros(7, dtype=np.float32)

        # Thumb button: max of trigger and squeeze
        thumb_button = max(trigger, squeeze)
        thumb_angle = -thumb_button

        # Thumb rotation: combination of trigger and squeeze
        # This creates different thumb poses depending on input
        thumb_rotation = (
            self.THUMB_ROTATION_TRIGGER_SCALE * trigger
            - self.THUMB_ROTATION_SQUEEZE_SCALE * squeeze
        )
        if not self._is_left:
            thumb_rotation = -thumb_rotation

        # Set thumb joints
        hand_joints[0] = thumb_rotation  # Thumb rotation
        hand_joints[1] = thumb_angle * self.THUMB_PROXIMAL_SCALE  # Thumb proximal
        hand_joints[2] = thumb_angle * self.THUMB_DISTAL_SCALE  # Thumb distal

        # Index finger: controlled by trigger
        index_angle = trigger * 1.0
        hand_joints[3] = index_angle  # Index proximal
        hand_joints[4] = index_angle  # Index distal

        # Middle finger: controlled by squeeze
        middle_angle = squeeze * 1.0
        hand_joints[5] = middle_angle  # Middle proximal
        hand_joints[6] = middle_angle  # Middle distal

        return hand_joints


class TriHandBiManualMotionControllerRetargeter(BaseRetargeter):
    """
    Wrapper around two TriHandMotionControllerRetargeter instances for bimanual control.

    This retargeter instantiates two TriHandMotionControllerRetargeter instances (one for each hand)
    and combines their outputs into a single vector.

    Inputs:
        - "controller_left": Left motion controller data
        - "controller_right": Right motion controller data

    Outputs:
        - "hand_joints": Combined joint angles for both hands
    """

    def __init__(
        self,
        left_config: TriHandMotionControllerConfig,
        right_config: TriHandMotionControllerConfig,
        target_joint_names: List[str],
        name: str,
    ) -> None:
        """
        Initialize the bimanual motion controller retargeter.

        Args:
            left_config: Configuration for left hand controller
            right_config: Configuration for right hand controller
            target_joint_names: Ordered list of joint names for combined output
            name: Name identifier for this retargeter (must be unique)
        """
        self._target_joint_names = target_joint_names

        super().__init__(name=name)

        # Ensure configs have correct controller sides
        left_config.controller_side = "left"
        right_config.controller_side = "right"

        # Create individual controllers
        self._left_controller = TriHandMotionControllerRetargeter(
            left_config, name=f"{name}_left"
        )
        self._right_controller = TriHandMotionControllerRetargeter(
            right_config, name=f"{name}_right"
        )

        # Prepare index mapping
        self._left_indices = []
        self._right_indices = []
        self._output_indices_left = []
        self._output_indices_right = []

        left_joints = left_config.hand_joint_names
        right_joints = right_config.hand_joint_names

        for i, name in enumerate(target_joint_names):
            if name in left_joints:
                self._output_indices_left.append(i)
                self._left_indices.append(left_joints.index(name))
            elif name in right_joints:
                self._output_indices_right.append(i)
                self._right_indices.append(right_joints.index(name))

    def input_spec(self) -> RetargeterIOType:
        """Define input collections for both controllers (Optional)."""
        return {
            "controller_left": OptionalType(ControllerInput()),
            "controller_right": OptionalType(ControllerInput()),
        }

    def output_spec(self) -> RetargeterIOType:
        """Define output collections for combined hand joints."""
        return {
            "hand_joints": RobotHandJoints(
                "hand_joints_bimanual", self._target_joint_names
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Execute bimanual motion controller retargeting.

        Args:
            inputs: Dict with "controller_left" and "controller_right" data
            outputs: Dict with "hand_joints" combined output
            context: ComputeContext, propagated to child controllers.
        """
        # Create temporary output groups for individual controllers
        left_outputs: RetargeterIO = {
            "hand_joints": TensorGroup(
                self._left_controller.output_spec()["hand_joints"]
            )
        }
        right_outputs: RetargeterIO = {
            "hand_joints": TensorGroup(
                self._right_controller.output_spec()["hand_joints"]
            )
        }

        # Run individual controllers, propagating context
        left_inputs = {"controller_left": inputs["controller_left"]}
        right_inputs = {"controller_right": inputs["controller_right"]}

        self._left_controller.compute(left_inputs, left_outputs, context=context)
        self._right_controller.compute(right_inputs, right_outputs, context=context)

        # Combine outputs
        combined_output = outputs["hand_joints"]

        # Map left hand joints
        for src_idx, dst_idx in zip(self._left_indices, self._output_indices_left):
            combined_output[dst_idx] = float(left_outputs["hand_joints"][src_idx])

        # Map right hand joints
        for src_idx, dst_idx in zip(self._right_indices, self._output_indices_right):
            combined_output[dst_idx] = float(right_outputs["hand_joints"][src_idx])
