# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading retargeting-engine tensor groups and group types."""

from isaaccapture.retargeting_engine.interface import OptionalTensorGroup
from isaaccapture.retargeting_engine.tensor_types.indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)


def _flag_is_valid(group: OptionalTensorGroup, *indices: int) -> bool:
    if group.is_none:
        return False
    value = group[indices[0]]
    for index in indices[1:]:
        value = value[index]
    return bool(value)


def controller_aim_is_valid(ctrl: OptionalTensorGroup) -> bool:
    return _flag_is_valid(ctrl, ControllerInputIndex.AIM_IS_VALID)


def hand_wrist_is_valid(hand: OptionalTensorGroup) -> bool:
    return _flag_is_valid(hand, HandInputIndex.JOINT_VALID, HandJointIndex.WRIST)


def head_is_valid(head: OptionalTensorGroup) -> bool:
    return _flag_is_valid(head, HeadInputIndex.IS_VALID)


def joint_names_from_group_type(group_type) -> list[str]:
    return [tensor_type.name for tensor_type in group_type.types]
