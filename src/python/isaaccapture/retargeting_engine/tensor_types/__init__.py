# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic tensor types for the retargeting engine."""

import warnings

from .scalar_types import FloatType, IntType, BoolType
from .ndarray_types import NDArrayType, DLDeviceType, DLDataType
from .standard_types import (
    HandInput,
    HeadInput,
    ControllerInput,
    FullBodyInput,
    TransformMatrix,
    Generic3AxisPedalInput,
    NUM_HAND_JOINTS,
    NUM_BODY_JOINTS,
    RobotHandJoints,
)
from .tactile_types import (
    TactileVector,
    TactileHeatmap,
    FingerPowerVector,
    ControllerHapticPulse,
    EndEffectorForce,
    NUM_HAPTIC_FINGERS,
    NUM_CONTROLLER_HAPTIC_FIELDS,
    NUM_END_EFFECTOR_FORCE_AXES,
)
from .indices import (
    HandInputIndex,
    HeadInputIndex,
    ControllerInputIndex,
    Generic3AxisPedalInputIndex,
    FullBodyInputIndex,
    HandJointIndex,
    BodyJointIndex,
    FingerIndex,
    ControllerHapticPulseField,
    EndEffectorForceAxis,
)

__all__ = [
    "FloatType",
    "IntType",
    "BoolType",
    "NDArrayType",
    "DLDeviceType",
    "DLDataType",
    # Standard types
    "HandInput",
    "HeadInput",
    "ControllerInput",
    "FullBodyInput",
    "TransformMatrix",
    "Generic3AxisPedalInput",
    "NUM_HAND_JOINTS",
    "NUM_BODY_JOINTS",
    "RobotHandJoints",
    # Tactile / haptic types
    "TactileVector",
    "TactileHeatmap",
    "FingerPowerVector",
    "ControllerHapticPulse",
    "EndEffectorForce",
    "NUM_HAPTIC_FINGERS",
    "NUM_CONTROLLER_HAPTIC_FIELDS",
    "NUM_END_EFFECTOR_FORCE_AXES",
    # Indices
    "HandInputIndex",
    "HeadInputIndex",
    "ControllerInputIndex",
    "Generic3AxisPedalInputIndex",
    "FullBodyInputIndex",
    "HandJointIndex",
    "BodyJointIndex",
    "FingerIndex",
    "ControllerHapticPulseField",
    "EndEffectorForceAxis",
]

# Deprecated re-exports resolved lazily so access emits a DeprecationWarning; kept out
# of __all__ and the eager imports above so importing this module stays quiet.
_DEPRECATED_ALIASES = {
    "BodyJointPicoIndex": "BodyJointIndex",
    "NUM_BODY_JOINTS_PICO": "NUM_BODY_JOINTS",
    # Renamed for consistency with its siblings (HandInput, ControllerInput,
    # FullBodyInput, Generic3AxisPedalInput); the group's fields are unchanged.
    "HeadPose": "HeadInput",
    "HeadPoseIndex": "HeadInputIndex",
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
