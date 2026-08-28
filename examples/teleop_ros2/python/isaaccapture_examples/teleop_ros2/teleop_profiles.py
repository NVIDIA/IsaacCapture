# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved runtime profiles for teleoperation session results and publishing."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypedDict, cast

from .constants import (
    TRACKED_HAND_RETARGETERS,
    HandRetargeter,
    HandTrackingProvider,
    StrEnum,
    TeleopMode,
)
from isaaccapture.retargeting_engine.interface import OptionalTensorGroup


class PublishType(StrEnum):
    CONTROLLER_PAYLOAD = "controller_payload"
    EE_FROM_CONTROLLERS = "ee_from_controllers"
    EE_FROM_HANDS = "ee_from_hands"
    FINGER_JOINTS = "finger_joints"
    FULL_BODY_PAYLOAD = "full_body_payload"
    HAND_POSES = "hand_poses"
    HEAD = "head"
    ROOT_COMMAND = "root_command"


class TeleopProfile(StrEnum):
    CONTROLLER_TELEOP = "controller_teleop"
    CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE = (
        "controller_teleop_with_hand_controller_ee"
    )
    CONTROLLER_TELEOP_WITH_HAND_MANUS_EE = "controller_teleop_with_hand_manus_ee"
    CONTROLLER_TELEOP_WITH_HAND_WRIST_EE = "controller_teleop_with_hand_wrist_ee"
    HAND_TELEOP = "hand_teleop"
    CONTROLLER_RAW = "controller_raw"
    FULL_BODY = "full_body"


class SessionResult(TypedDict, total=False):
    controller_left: OptionalTensorGroup
    controller_right: OptionalTensorGroup
    finger_joints_left: OptionalTensorGroup
    finger_joints_right: OptionalTensorGroup
    full_body: OptionalTensorGroup
    hand_left: OptionalTensorGroup
    hand_right: OptionalTensorGroup
    head: OptionalTensorGroup
    root_command: OptionalTensorGroup


@dataclass(frozen=True)
class TeleopProfileSpec:
    """Describe the frame inputs and publish paths for one resolved profile."""

    mode: TeleopMode
    required_result_keys: frozenset[str]
    publish_types: frozenset[PublishType]
    apply_manus_controller_mount_offset: bool = False


_CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE_SPEC = TeleopProfileSpec(
    mode=TeleopMode.CONTROLLER_TELEOP,
    required_result_keys=frozenset(
        {
            "controller_left",
            "controller_right",
            "finger_joints_left",
            "finger_joints_right",
            "hand_left",
            "hand_right",
            "head",
            "root_command",
        }
    ),
    publish_types=frozenset(
        {
            PublishType.CONTROLLER_PAYLOAD,
            PublishType.EE_FROM_CONTROLLERS,
            PublishType.FINGER_JOINTS,
            PublishType.HAND_POSES,
            PublishType.HEAD,
            PublishType.ROOT_COMMAND,
        }
    ),
)


TELEOP_PROFILE_SPECS = {
    TeleopProfile.CONTROLLER_TELEOP: TeleopProfileSpec(
        mode=TeleopMode.CONTROLLER_TELEOP,
        required_result_keys=frozenset(
            {
                "controller_left",
                "controller_right",
                "finger_joints_left",
                "finger_joints_right",
                "head",
                "root_command",
            }
        ),
        publish_types=frozenset(
            {
                PublishType.CONTROLLER_PAYLOAD,
                PublishType.EE_FROM_CONTROLLERS,
                PublishType.FINGER_JOINTS,
                PublishType.HEAD,
                PublishType.ROOT_COMMAND,
            }
        ),
    ),
    TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE: (
        _CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE_SPEC
    ),
    TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_MANUS_EE: replace(
        _CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE_SPEC,
        apply_manus_controller_mount_offset=True,
    ),
    TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_WRIST_EE: TeleopProfileSpec(
        mode=TeleopMode.CONTROLLER_TELEOP,
        required_result_keys=frozenset(
            {
                "controller_left",
                "controller_right",
                "finger_joints_left",
                "finger_joints_right",
                "hand_left",
                "hand_right",
                "head",
                "root_command",
            }
        ),
        publish_types=frozenset(
            {
                PublishType.CONTROLLER_PAYLOAD,
                PublishType.EE_FROM_HANDS,
                PublishType.FINGER_JOINTS,
                PublishType.HAND_POSES,
                PublishType.HEAD,
                PublishType.ROOT_COMMAND,
            }
        ),
    ),
    TeleopProfile.HAND_TELEOP: TeleopProfileSpec(
        mode=TeleopMode.HAND_TELEOP,
        required_result_keys=frozenset(
            {
                "finger_joints_left",
                "finger_joints_right",
                "hand_left",
                "hand_right",
                "head",
                "root_command",
            }
        ),
        publish_types=frozenset(
            {
                PublishType.EE_FROM_HANDS,
                PublishType.FINGER_JOINTS,
                PublishType.HAND_POSES,
                PublishType.HEAD,
                PublishType.ROOT_COMMAND,
            }
        ),
    ),
    TeleopProfile.CONTROLLER_RAW: TeleopProfileSpec(
        mode=TeleopMode.CONTROLLER_RAW,
        required_result_keys=frozenset({"controller_left", "controller_right"}),
        publish_types=frozenset({PublishType.CONTROLLER_PAYLOAD}),
    ),
    TeleopProfile.FULL_BODY: TeleopProfileSpec(
        mode=TeleopMode.FULL_BODY,
        required_result_keys=frozenset(
            {"controller_left", "controller_right", "full_body"}
        ),
        publish_types=frozenset(
            {
                PublishType.CONTROLLER_PAYLOAD,
                PublishType.FULL_BODY_PAYLOAD,
            }
        ),
    ),
}


def resolve_teleop_profile_spec(
    mode: TeleopMode,
    resolved_hand_retargeter: HandRetargeter,
    hand_tracking_provider: HandTrackingProvider = HandTrackingProvider.NATIVE,
) -> TeleopProfileSpec:
    """Resolve user-facing settings to one complete immutable runtime profile."""
    if (
        mode == TeleopMode.CONTROLLER_TELEOP
        and resolved_hand_retargeter in TRACKED_HAND_RETARGETERS
    ):
        if hand_tracking_provider == HandTrackingProvider.MANUS:
            profile = TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_MANUS_EE
        elif hand_tracking_provider == HandTrackingProvider.WUJI:
            profile = TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_WRIST_EE
        else:
            profile = TeleopProfile.CONTROLLER_TELEOP_WITH_HAND_CONTROLLER_EE
    else:
        profile = TeleopProfile(mode.value)
    return TELEOP_PROFILE_SPECS[profile]


def validate_session_result(
    result: Mapping[str, OptionalTensorGroup],
    profile_spec: TeleopProfileSpec,
) -> SessionResult:
    """Validate and type-narrow a session frame against its profile contract."""
    expected_keys = profile_spec.required_result_keys
    actual_keys = frozenset(result)
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys: {sorted(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected keys: {sorted(unexpected_keys)}")
        raise ValueError(
            f"Invalid session result for mode {profile_spec.mode.value!r}: "
            f"{'; '.join(details)}"
        )
    return cast(SessionResult, result)
