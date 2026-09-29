# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeviceIO Tensor Types - payload handles from DeviceIO trackers.

These tensor types represent the encoded payloads returned by DeviceIO trackers.
Each carries a handle over the encoded payload, read directly through the schema
accessors. An inactive device arrives as None rather than as an empty handle.
"""

import warnings
from enum import IntEnum
from typing import Any
from ..interface.tensor_type import TensorType
from ..interface.tensor_group_type import TensorGroupType
from isaaccapture.schema import (
    HeadPose,
    HandPose,
    ControllerSnapshot,
    Generic3AxisPedalOutput,
    JointStateOutput,
    FullBodyPose,
    MessageChannelMessagesTracked,
)


class _PayloadTensorType(TensorType):
    """Carries one DeviceIO payload handle.

    Every payload validates the same way, so a subclass only names the schema class it
    accepts. The class itself is what distinguishes one payload from another --
    ``TensorType.is_compatible_with`` rejects a mismatch before it gets here.
    """

    #: Schema view class this tensor type accepts. Set by each subclass.
    _payload_cls: type

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Expected {type(self).__name__}, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        # None is how an inactive device arrives; only a wrong type is an error.
        if value is not None and not isinstance(value, self._payload_cls):
            raise TypeError(
                f"Expected {self._payload_cls.__name__} for '{self.name}', got {type(value).__name__}"
            )


class _RequiredPayloadTensorType(_PayloadTensorType):
    """Payload with no inactive state: a value is produced every frame, so None is an error.

    Subclass this rather than ``_PayloadTensorType`` when the source always has something
    to report -- a constant stand-in for "nothing happened" counts. Consumers of such a
    slot are written without a None branch, so letting one through moves the failure to
    whichever of them dereferences it first.
    """

    def validate_value(self, value: Any) -> None:
        if value is None:
            raise TypeError(
                f"Expected {self._payload_cls.__name__} for '{self.name}', got None"
            )
        super().validate_value(value)


class HeadPoseTrackedType(_PayloadTensorType):
    """HeadPose payload from DeviceIO HeadTracker."""

    _payload_cls = HeadPose


class HandPoseTrackedType(_PayloadTensorType):
    """HandPose payload from DeviceIO HandTracker."""

    _payload_cls = HandPose


class ControllerSnapshotTrackedType(_PayloadTensorType):
    """ControllerSnapshot payload from DeviceIO ControllerTracker."""

    _payload_cls = ControllerSnapshot


class Generic3AxisPedalOutputTrackedType(_PayloadTensorType):
    """Generic3AxisPedalOutput payload from DeviceIO Generic3AxisPedalTracker."""

    _payload_cls = Generic3AxisPedalOutput


class JointStateOutputTrackedType(_PayloadTensorType):
    """JointStateOutput payload from DeviceIO JointStateTracker."""

    _payload_cls = JointStateOutput


class FullBodyPoseTrackedType(_PayloadTensorType):
    """FullBodyPose payload from DeviceIO FullBodyTracker.

    Vendor-agnostic: the full-body tracker produces the same FullBodyPose
    payload regardless of the live vendor (native XR, pushed tensor, ...).
    """

    _payload_cls = FullBodyPose


class MessageChannelMessagesTrackedType(_RequiredPayloadTensorType):
    """MessageChannelMessagesTracked batch from DeviceIO MessageChannelTracker."""

    _payload_cls = MessageChannelMessagesTracked


class MessageChannelConnectionStatus(IntEnum):
    """Message channel connection states exposed by MessageChannelSource."""

    CONNECTING = 0
    CONNECTED = 1
    SHUTTING = 2
    DISCONNECTED = 3
    UNKNOWN = -1


class MessageChannelStatusType(_RequiredPayloadTensorType):
    """Enum status for message channel connectivity."""

    _payload_cls = MessageChannelConnectionStatus


def DeviceIOHeadPoseTracked() -> TensorGroupType:
    """Tracked head pose from DeviceIO HeadTracker.

    Contains:
        head_tracked: HeadPose handle, or None when inactive
    """
    return TensorGroupType("deviceio_head_pose", [HeadPoseTrackedType("head_tracked")])


def DeviceIOHandPoseTracked() -> TensorGroupType:
    """Tracked hand pose from DeviceIO HandTracker.

    Contains:
        hand_tracked: HandPose handle, or None when inactive
    """
    return TensorGroupType("deviceio_hand_pose", [HandPoseTrackedType("hand_tracked")])


def DeviceIOControllerSnapshotTracked() -> TensorGroupType:
    """Tracked controller snapshot from DeviceIO ControllerTracker.

    Contains:
        controller_tracked: ControllerSnapshot handle, or None when inactive
    """
    return TensorGroupType(
        "deviceio_controller_snapshot",
        [ControllerSnapshotTrackedType("controller_tracked")],
    )


def DeviceIOGeneric3AxisPedalOutputTracked() -> TensorGroupType:
    """Tracked pedal data from DeviceIO Generic3AxisPedalTracker.

    Contains:
        pedal_tracked: Generic3AxisPedalOutput handle, or None when inactive
    """
    return TensorGroupType(
        "deviceio_generic_3axis_pedal_output",
        [Generic3AxisPedalOutputTrackedType("pedal_tracked")],
    )


def DeviceIOJointStateOutputTracked() -> TensorGroupType:
    """Tracked joint-state data from DeviceIO JointStateTracker.

    Contains:
        joint_state_tracked: JointStateOutput handle, or None when inactive
    """
    return TensorGroupType(
        "deviceio_joint_state_output",
        [JointStateOutputTrackedType("joint_state_tracked")],
    )


def DeviceIOFullBodyPoseTracked() -> TensorGroupType:
    """Tracked full body pose data from DeviceIO FullBodyTracker.

    Contains:
        full_body_tracked: FullBodyPose handle, or None when inactive
    """
    return TensorGroupType(
        "deviceio_full_body_pose",
        [FullBodyPoseTrackedType("full_body_tracked")],
    )


def DeviceIOMessageChannelMessagesTracked() -> TensorGroupType:
    """Message batch from DeviceIO MessageChannelTracker."""
    return TensorGroupType(
        "deviceio_message_channel_messages_tracked",
        [MessageChannelMessagesTrackedType("messages_tracked")],
    )


def MessageChannelMessagesTrackedGroup() -> TensorGroupType:
    """Tracked batch of messages drained in one update."""
    return TensorGroupType(
        "message_channel_messages_tracked",
        [MessageChannelMessagesTrackedType("messages_tracked")],
    )


def MessageChannelStatusGroup() -> TensorGroupType:
    """Message channel connection status enum."""
    return TensorGroupType(
        "message_channel_status",
        [MessageChannelStatusType("status")],
    )


# Deprecated aliases resolved lazily via __getattr__ so accessing them emits a
# DeprecationWarning.
_DEPRECATED_ALIASES = {
    "FullBodyPosePicoTrackedType": "FullBodyPoseTrackedType",
    "DeviceIOFullBodyPosePicoTracked": "DeviceIOFullBodyPoseTracked",
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
