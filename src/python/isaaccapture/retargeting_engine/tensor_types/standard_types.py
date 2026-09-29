# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standard TensorGroupType definitions for common teleoperation data structures.

These definitions match the FlatBuffers schemas in IsaacTeleop/src/core/schema/fbs/
and provide type-safe specifications for hand tracking, head tracking, and controller data.
"""

import warnings

from ..interface.tensor_group_type import TensorGroupType
from .scalar_types import FloatType, BoolType
from .ndarray_types import NDArrayType, DLDataType


# Constants
NUM_HAND_JOINTS = 26  # XR_HAND_JOINT_COUNT_EXT from OpenXR
NUM_BODY_JOINTS = 24  # XR_BODY_JOINT_COUNT_BD from XR_BD_body_tracking

# ============================================================================
# Hand Tracking Types
# ============================================================================


def HandInput() -> TensorGroupType:
    """
    Standard TensorGroupType for hand tracking data.

    Matches the HandPose schema from hand.fbs with 26 joints (XR_HAND_JOINT_COUNT_EXT).

    The actual left/right distinction comes from the RetargeterIO dictionary key
    (e.g., "hand_left" vs "hand_right"), not from the type itself. Both left and
    right hands use the same type structure.

    Fields:
        - joint_positions: (26, 3) float32 array - XYZ positions for each joint
        - joint_orientations: (26, 4) float32 array - XYZW quaternions for each joint
        - joint_radii: (26,) float32 array - Radius of each joint
        - joint_valid: (26,) bool array - Validity flag for each joint

    Returns:
        TensorGroupType for hand tracking data

    Schema reference: IsaacTeleop/src/core/schema/fbs/hand.fbs

    Example:
        # Left and right hands use the same type, distinguished by dict key.
        # Wrap in OptionalType because HandsSource outputs are optional.
        def input_spec(self) -> RetargeterIO:
            return {
                "hand_left": OptionalType(HandInput()),
                "hand_right": OptionalType(HandInput()),
            }
    """
    return TensorGroupType(
        "hand",
        [
            NDArrayType(
                "hand_joint_positions",
                shape=(NUM_HAND_JOINTS, 3),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "hand_joint_orientations",
                shape=(NUM_HAND_JOINTS, 4),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "hand_joint_radii",
                shape=(NUM_HAND_JOINTS,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "hand_joint_valid",
                shape=(NUM_HAND_JOINTS,),
                dtype=DLDataType.UINT,
                dtype_bits=8,  # bool represented as uint8
            ),
        ],
    )


# ============================================================================
# Head Tracking Types
# ============================================================================


def HeadInput() -> TensorGroupType:
    """
    Standard TensorGroupType for head tracking data.

    Matches the HeadPose schema from head.fbs.

    Fields:
        - head_position: (3,) float32 array - XYZ position
        - head_orientation: (4,) float32 array - XYZW quaternion
        - head_is_valid: bool - Whether the pose value is valid to read (OpenXR VALID)
        - head_is_tracked: bool - Whether the pose is actively tracked (OpenXR TRACKED)

    Returns:
        TensorGroupType for head tracking data

    Schema reference: IsaacTeleop/src/core/schema/fbs/head.fbs
    """
    return TensorGroupType(
        "head",
        [
            NDArrayType(
                "head_position", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32
            ),
            NDArrayType(
                "head_orientation", shape=(4,), dtype=DLDataType.FLOAT, dtype_bits=32
            ),
            BoolType("head_is_valid"),
            BoolType("head_is_tracked"),
        ],
    )


# ============================================================================
# Controller Types
# ============================================================================


def ControllerInput() -> TensorGroupType:
    """
    Standard TensorGroupType for VR controller data.

    Provides grip pose, aim pose, and all controller inputs (buttons, triggers, thumbstick).

    The actual left/right distinction comes from the RetargeterIO dictionary key
    (e.g., "controller_left" vs "controller_right"), not from the type itself. Both
    left and right controllers use the same type structure.

    Fields:
        - grip_position: (3,) float32 array - XYZ position of grip pose
        - grip_orientation: (4,) float32 array - XYZW quaternion of grip pose
        - grip_is_valid: bool - Whether grip pose data is valid
        - aim_position: (3,) float32 array - XYZ position of aim pose
        - aim_orientation: (4,) float32 array - XYZW quaternion of aim pose
        - aim_is_valid: bool - Whether aim pose data is valid
        - primary_click: float - Primary button (e.g., A/X button) [0.0-1.0]
        - secondary_click: float - Secondary button (e.g., B/Y button) [0.0-1.0]
        - thumbstick_x: float - Thumbstick X axis [-1.0 to 1.0]
        - thumbstick_y: float - Thumbstick Y axis [-1.0 to 1.0]
        - thumbstick_click: float - Thumbstick button press [0.0-1.0]
        - menu_click: float - Menu (hamburger) button; left hand only on Oculus Touch [0.0-1.0]
        - squeeze_value: float - Grip/squeeze trigger [0.0-1.0]
        - trigger_value: float - Index finger trigger [0.0-1.0]

    Returns:
        TensorGroupType for controller data

    Example:
        # Left and right controllers use the same type, distinguished by dict key.
        # Wrap in OptionalType because ControllersSource outputs are optional.
        def input_spec(self) -> RetargeterIO:
            return {
                "controller_left": OptionalType(ControllerInput()),
                "controller_right": OptionalType(ControllerInput()),
            }
    """
    return TensorGroupType(
        "controller",
        [
            NDArrayType(
                "controller_grip_position",
                shape=(3,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "controller_grip_orientation",
                shape=(4,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            BoolType("controller_grip_is_valid"),
            NDArrayType(
                "controller_aim_position",
                shape=(3,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "controller_aim_orientation",
                shape=(4,),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            BoolType("controller_aim_is_valid"),
            FloatType("controller_primary_click"),
            FloatType("controller_secondary_click"),
            FloatType("controller_thumbstick_x"),
            FloatType("controller_thumbstick_y"),
            FloatType("controller_thumbstick_click"),
            FloatType("controller_menu_click"),
            FloatType("controller_squeeze_value"),
            FloatType("controller_trigger_value"),
        ],
    )


# ============================================================================
# Full Body Tracking Types
# ============================================================================


def FullBodyInput() -> TensorGroupType:
    """
    Standard TensorGroupType for full body tracking data.

    Matches the FullBodyPose schema from full_body.fbs with 24 joints
    (XR_BD_body_tracking extension layout).

    Fields:
        - joint_positions: (24, 3) float32 array - XYZ positions for each joint
        - joint_orientations: (24, 4) float32 array - XYZW quaternions for each joint
        - joint_valid: (24,) uint8 array - Validity flag for each joint

    Returns:
        TensorGroupType for full body tracking data

    Schema reference: IsaacTeleop/src/core/schema/fbs/full_body.fbs
    """
    return TensorGroupType(
        "body",
        [
            NDArrayType(
                "body_joint_positions",
                shape=(NUM_BODY_JOINTS, 3),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "body_joint_orientations",
                shape=(NUM_BODY_JOINTS, 4),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
            NDArrayType(
                "body_joint_valid",
                shape=(NUM_BODY_JOINTS,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            ),
        ],
    )


# ============================================================================
# Transform Types
# ============================================================================


def TransformMatrix() -> TensorGroupType:
    """
    Standard TensorGroupType for a 4x4 homogeneous transformation matrix.

    Fields:
        - matrix: (4, 4) float32 array - Homogeneous transformation matrix

    Returns:
        TensorGroupType for a 4x4 transform matrix
    """
    return TensorGroupType(
        "transform",
        [
            NDArrayType(
                "transform_matrix", shape=(4, 4), dtype=DLDataType.FLOAT, dtype_bits=32
            ),
        ],
    )


# ============================================================================
# Robot Types
# ============================================================================


def RobotHandJoints(name: str, joint_names: list[str]) -> TensorGroupType:
    """
    Standard TensorGroupType for robot hand joint angles.

    Creates a TensorGroupType with FloatType fields for each joint name.

    Args:
        name: Name of the TensorGroupType (e.g., "hand_joints_left")
        joint_names: List of joint names

    Returns:
        TensorGroupType for robot hand joints
    """
    return TensorGroupType(name, [FloatType(joint_name) for joint_name in joint_names])


# ============================================================================
# Pedal Types
# ============================================================================


def Generic3AxisPedalInput() -> TensorGroupType:
    """
    Standard TensorGroupType for generic 3-axis foot pedal data.

    Matches the Generic3AxisPedalOutput schema from pedals.fbs. Used as input
    to foot-pedal retargeters (e.g., FootPedalRootCmdRetargeter) for lower-body
    control. Axis values are typically in [-1, 1].

    Fields:
        - left_pedal: float - Left pedal axis [-1.0 to 1.0]
        - right_pedal: float - Right pedal axis [-1.0 to 1.0]
        - rudder: float - Rudder axis [-1.0 to 1.0]

    Returns:
        TensorGroupType for pedal data

    Schema reference: TeleopCore/src/core/schema/fbs/pedals.fbs
    """
    return TensorGroupType(
        "generic_3axis_pedal",
        [
            FloatType("pedal_left_pedal"),
            FloatType("pedal_right_pedal"),
            FloatType("pedal_rudder"),
        ],
    )


# Deprecated aliases resolved lazily via __getattr__ so accessing them emits a
# DeprecationWarning.
_DEPRECATED_ALIASES = {
    # Renamed for consistency with its siblings (HandInput, ControllerInput,
    # FullBodyInput, Generic3AxisPedalInput); the group's fields are unchanged.
    "HeadPose": "HeadInput",
}


def __getattr__(name: str):
    new_name = _DEPRECATED_ALIASES.get(name)
    if new_name is not None:
        warnings.warn(
            f"{name} is deprecated; use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
