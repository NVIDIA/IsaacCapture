# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Real, SDK-pinned tensor-group index constants, shared verbatim by the
production :mod:`capture_server.acquisition.isaac_capture` adapter and
:mod:`capture_server.mock.mock_isaac_capture` -- not mock behavior (these
are fixed numeric indices the real IC SDK defines, reproduced from
``retargeting_engine/tensor_types/indices.py`` at the pinned commit
``b418850e9ec47067b7c0c63f35944bfa0cc5ce37``) and not adapter logic
either, so they live here once rather than in either of those.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["ControllerInputIndex", "HandInputIndex", "HandJointIndex", "HeadInputIndex"]


class HeadInputIndex(IntEnum):
    """HeadInput's declared field order (head.fbs)."""

    POSITION = 0
    ORIENTATION = 1
    IS_VALID = 2
    IS_TRACKED = 3


class ControllerInputIndex(IntEnum):
    """ControllerInput's declared field order."""

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
    """HandInput's declared field order (hand.fbs)."""

    JOINT_POSITIONS = 0
    JOINT_ORIENTATIONS = 1
    JOINT_RADII = 2
    JOINT_VALID = 3


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
