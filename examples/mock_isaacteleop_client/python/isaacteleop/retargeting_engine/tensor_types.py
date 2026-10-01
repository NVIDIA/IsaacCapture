# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Index enums matching the real ``isaacteleop.retargeting_engine.tensor_types`` field layout.

Member names/values are fixed by ``GumanIsaacTeleopUsage.md`` sections 5.1-5.3 (itself read from
the real package) -- not generated here, since this shim has no real tensor-type schema to
generate them from.
"""

from enum import IntEnum

NUM_HAPTIC_FINGERS = 5


class HeadPoseIndex(IntEnum):
    POSITION = 0
    ORIENTATION = 1
    IS_VALID = 2


class ControllerInputIndex(IntEnum):
    GRIP_POSITION = 0
    GRIP_ORIENTATION = 1
    GRIP_IS_VALID = 2
    AIM_POSITION = 3
    AIM_ORIENTATION = 4
    AIM_IS_VALID = 5
    PRIMARY_CLICK = 6
    SECONDARY_CLICK = 7
    THUMBSTICK_X = 8
    THUMBSTICK_Y = 9
    THUMBSTICK_CLICK = 10
    MENU_CLICK = 11
    SQUEEZE_VALUE = 12
    TRIGGER_VALUE = 13


class HandInputIndex(IntEnum):
    JOINT_POSITIONS = 0
    JOINT_ORIENTATIONS = 1
    JOINT_RADII = 2
    JOINT_VALID = 3


class HandJointIndex(IntEnum):
    """OpenXR ``XR_HAND_JOINT_*_EXT`` layout (``GumanIsaacTeleopUsage.md`` §5.3)."""

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


class FullBodyInputIndex(IntEnum):
    """Not used by Guman's ``isaac_teleop_backend.py`` today; included for completeness."""

    JOINT_POSITIONS = 0
    JOINT_ORIENTATIONS = 1
    JOINT_VALID = 2
