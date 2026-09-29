# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamically generated indices for standard TensorGroupTypes.

This module provides IntEnum classes for indexing into standard tensor groups
(HandInput, HeadInput, ControllerInput, Generic3AxisPedalInput, FullBodyInput) and standard joint arrays
(HandJointIndex, BodyJointIndex).

The indices for TensorGroupTypes are generated automatically from the type definitions
to ensure they always match the schema.
"""

import warnings
from typing import Any
from enum import IntEnum
from .standard_types import (
    HandInput,
    HeadInput,
    ControllerInput,
    Generic3AxisPedalInput,
    FullBodyInput,
)


def _create_index_enum(name: str, group_type, prefix: str = "") -> IntEnum:
    """Helper to create an IntEnum from a TensorGroupType."""
    members = {}
    for i, t in enumerate(group_type.types):
        key = t.name
        if prefix and key.startswith(prefix):
            key = key[len(prefix) :]
        members[key.upper()] = i
    return IntEnum(name, members)


# Generate indices dynamically
HandInputIndex: Any = _create_index_enum("HandInputIndex", HandInput(), "hand_")
HeadInputIndex: Any = _create_index_enum("HeadInputIndex", HeadInput(), "head_")
ControllerInputIndex: Any = _create_index_enum(
    "ControllerInputIndex", ControllerInput(), "controller_"
)
Generic3AxisPedalInputIndex: Any = _create_index_enum(
    "Generic3AxisPedalInputIndex", Generic3AxisPedalInput(), "pedal_"
)
FullBodyInputIndex: Any = _create_index_enum(
    "FullBodyInputIndex", FullBodyInput(), "body_"
)


class HandJointIndex(IntEnum):
    """Indices for OpenXR hand joints (XR_HAND_JOINT_COUNT_EXT = 26)."""

    PALM = 0
    WRIST = 1
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    LITTLE_METACARPAL = 21
    LITTLE_PROXIMAL = 22
    LITTLE_INTERMEDIATE = 23
    LITTLE_DISTAL = 24
    LITTLE_TIP = 25


class BodyJointIndex(IntEnum):
    """Indices for body joints (XR_BD_body_tracking layout, 24 joints)."""

    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HAND = 22
    RIGHT_HAND = 23


# Deprecated aliases resolved lazily via __getattr__ so accessing them emits a
# DeprecationWarning.
_DEPRECATED_ALIASES = {
    "BodyJointPicoIndex": "BodyJointIndex",
    # Renamed for consistency with its siblings (HandInputIndex, ControllerInputIndex,
    # FullBodyInputIndex); the underlying fields are unchanged.
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


class FingerIndex(IntEnum):
    """Channel indices into a :func:`FingerPowerVector`, standard glove order."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


class ControllerHapticPulseField(IntEnum):
    """Field indices into a :func:`ControllerHapticPulse` ``[amplitude, frequency_hz, duration_s]`` vector."""

    AMPLITUDE = 0
    FREQUENCY_HZ = 1
    DURATION_S = 2


class EndEffectorForceAxis(IntEnum):
    """Component indices into an :func:`EndEffectorForce` ``[fx, fy, fz]`` vector."""

    X = 0
    Y = 1
    Z = 2
