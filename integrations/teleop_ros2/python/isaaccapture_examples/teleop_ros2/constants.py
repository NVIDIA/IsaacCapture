# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants and enum values shared by the Teleop ROS 2 node."""

from enum import StrEnum

from isaaccapture.retargeting_engine.tensor_types.indices import (
    BodyJointIndex,
    HandJointIndex,
)


class HandRetargeter(StrEnum):
    MODE_DEFAULT = "mode_default"
    TRIHAND = "trihand"
    PINK_IK = "pink_ik"
    DEXPILOT = "dexpilot"
    WUJI = "wuji"


class HandTrackingProvider(StrEnum):
    NATIVE = "native"
    MANUS = "manus"
    WUJI = "wuji"


class TeleopMode(StrEnum):
    CONTROLLER_TELEOP = "controller_teleop"
    HAND_TELEOP = "hand_teleop"
    CONTROLLER_RAW = "controller_raw"
    FULL_BODY = "full_body"


class EePoseFrame(StrEnum):
    WORLD = "world"
    HEAD = "head"


BODY_JOINT_NAMES = [e.name for e in BodyJointIndex]
HAND_POSE_JOINT_INDICES = tuple(
    HandJointIndex(i)
    for i in range(HandJointIndex.WRIST, HandJointIndex.LITTLE_TIP + 1)
)
HAND_POSE_NAMES = [joint.name for joint in HAND_POSE_JOINT_INDICES]
HAND_RETARGETERS = tuple(retargeter.value for retargeter in HandRetargeter)
HAND_TRACKING_PROVIDERS = tuple(provider.value for provider in HandTrackingProvider)
SHARPA_HAND_RETARGETERS = (HandRetargeter.PINK_IK, HandRetargeter.DEXPILOT)
TRACKED_HAND_RETARGETERS = (*SHARPA_HAND_RETARGETERS, HandRetargeter.WUJI)
TELEOP_MODES = tuple(mode.value for mode in TeleopMode)
WUJI_HAND_MODELS = ("wuji_hand", "wuji_hand_2")
EE_POSE_FRAMES = tuple(frame.value for frame in EePoseFrame)

TRIHAND_JOINT_NAMES = [
    "thumb_rotation",
    "thumb_proximal",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
]
LEFT_FINGER_JOINT_NAMES = [f"left_{n}" for n in TRIHAND_JOINT_NAMES]
RIGHT_FINGER_JOINT_NAMES = [f"right_{n}" for n in TRIHAND_JOINT_NAMES]

SHARPA_WAVE_JOINT_NAMES = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]
SHARPA_FINGER_JOINT_COUNT = len(SHARPA_WAVE_JOINT_NAMES)
LEFT_SHARPA_WAVE_JOINT_NAMES = [f"left_{n}" for n in SHARPA_WAVE_JOINT_NAMES]
RIGHT_SHARPA_WAVE_JOINT_NAMES = [f"right_{n}" for n in SHARPA_WAVE_JOINT_NAMES]

WUJI_HAND_JOINT_NAMES = [
    f"{finger}_j{joint_index}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint_index in range(4)
]
WUJI_HAND_JOINT_COUNT = len(WUJI_HAND_JOINT_NAMES)
LEFT_WUJI_HAND_JOINT_NAMES = [f"left_{n}" for n in WUJI_HAND_JOINT_NAMES]
RIGHT_WUJI_HAND_JOINT_NAMES = [f"right_{n}" for n in WUJI_HAND_JOINT_NAMES]

DEX_HANDTRACKING_TO_BASELINK_FRAME_TRANSFORM = (0, -1, 0, -1, 0, 0, 0, 0, -1)


def resolve_hand_retargeter(
    mode: TeleopMode, hand_retargeter: HandRetargeter
) -> HandRetargeter:
    if hand_retargeter == HandRetargeter.MODE_DEFAULT:
        if mode == TeleopMode.CONTROLLER_TELEOP:
            return HandRetargeter.TRIHAND
        if mode == TeleopMode.HAND_TELEOP:
            return HandRetargeter.DEXPILOT
        return hand_retargeter

    if mode == TeleopMode.HAND_TELEOP and hand_retargeter == HandRetargeter.TRIHAND:
        raise ValueError(
            "Parameter 'hand_retargeter:=trihand' is only valid with "
            "mode:=controller_teleop"
        )

    return hand_retargeter
