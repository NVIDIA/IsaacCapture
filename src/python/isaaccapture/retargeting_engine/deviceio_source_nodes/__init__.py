# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeviceIO Source Nodes - Stateless converters from DeviceIO to retargeting engine formats."""

import warnings

from .interface import IDeviceIOSource
from .sink_interface import IDeviceIOSink
from .head_source import HeadSource
from .hands_source import HandsSource
from .controllers_source import ControllersSource
from .pedals_source import Generic3AxisPedalSource
from .spacemouse_source import (
    SpaceMouseButtonsType,
    SpaceMouseRotationType,
    SpaceMouseSource,
    SpaceMouseTranslationType,
)

from .keyboard_source import (
    KeyboardHeldType,
    KeyboardPressedType,
    KeyboardAttachment,
    KeyboardSource,
    KeyEventHandle,
    KeyEventSource,
    KeyListeners,
)
from .source_lookup import find_sources
from .key_event_testing import FakeKeyEventSource
from .joint_state_source import JointStateSource
from .full_body_source import FullBodySource
from .soma_body_source import SomaBodyRepresentation, SomaBodySource
from .soma_hand_source import SomaHandRepresentation, SomaHandSource
from .message_channel_source import MessageChannelSource
from .message_channel_sink import MessageChannelSink
from .message_channel_config import (
    MessageChannelConfig,
    message_channel_config,
    messageChannelConfig,
)
from .haptic_sink import HapticSink
from .deviceio_tensor_types import (
    HeadPoseTrackedType,
    HandPoseTrackedType,
    ControllerSnapshotTrackedType,
    Generic3AxisPedalOutputTrackedType,
    KeyboardOutputTrackedType,
    SpaceMouseOutputTrackedType,
    JointStateOutputTrackedType,
    FullBodyPoseTrackedType,
    SomaBodyJointRotationsTrackedType,
    SomaBodyJointPosesTrackedType,
    SomaHandJointRotationsTrackedType,
    SomaHandJointPosesTrackedType,
    DeviceIOHeadPoseTracked,
    DeviceIOHandPoseTracked,
    DeviceIOControllerSnapshotTracked,
    DeviceIOGeneric3AxisPedalOutputTracked,
    DeviceIOKeyboardOutputTracked,
    DeviceIOSpaceMouseOutputTracked,
    DeviceIOJointStateOutputTracked,
    DeviceIOFullBodyPoseTracked,
    DeviceIOSomaBodyJointRotationsTracked,
    DeviceIOSomaBodyJointPosesTracked,
    DeviceIOSomaHandJointRotationsTracked,
    DeviceIOSomaHandJointPosesTracked,
    MessageChannelMessagesTrackedType,
    MessageChannelConnectionStatus,
    MessageChannelStatusType,
    DeviceIOMessageChannelMessagesTracked,
    MessageChannelMessagesTrackedGroup,
    MessageChannelStatusGroup,
)

__all__ = [
    "IDeviceIOSource",
    "IDeviceIOSink",
    "HeadSource",
    "HandsSource",
    "ControllersSource",
    "Generic3AxisPedalSource",
    "SpaceMouseButtonsType",
    "SpaceMouseRotationType",
    "SpaceMouseSource",
    "SpaceMouseTranslationType",
    "KeyboardHeldType",
    "KeyboardPressedType",
    "KeyboardAttachment",
    "KeyboardSource",
    "KeyEventHandle",
    "KeyEventSource",
    "KeyListeners",
    "find_sources",
    "FakeKeyEventSource",
    "JointStateSource",
    "FullBodySource",
    "SomaBodySource",
    "SomaBodyRepresentation",
    "SomaHandSource",
    "SomaHandRepresentation",
    "MessageChannelSource",
    "MessageChannelSink",
    "MessageChannelConfig",
    "message_channel_config",
    "messageChannelConfig",
    "HapticSink",
    "HeadPoseTrackedType",
    "HandPoseTrackedType",
    "ControllerSnapshotTrackedType",
    "Generic3AxisPedalOutputTrackedType",
    "KeyboardOutputTrackedType",
    "SpaceMouseOutputTrackedType",
    "JointStateOutputTrackedType",
    "FullBodyPoseTrackedType",
    "SomaBodyJointRotationsTrackedType",
    "SomaBodyJointPosesTrackedType",
    "SomaHandJointRotationsTrackedType",
    "SomaHandJointPosesTrackedType",
    "MessageChannelMessagesTrackedType",
    "MessageChannelConnectionStatus",
    "MessageChannelStatusType",
    "DeviceIOHeadPoseTracked",
    "DeviceIOHandPoseTracked",
    "DeviceIOControllerSnapshotTracked",
    "DeviceIOGeneric3AxisPedalOutputTracked",
    "DeviceIOKeyboardOutputTracked",
    "DeviceIOSpaceMouseOutputTracked",
    "DeviceIOJointStateOutputTracked",
    "DeviceIOFullBodyPoseTracked",
    "DeviceIOSomaBodyJointRotationsTracked",
    "DeviceIOSomaBodyJointPosesTracked",
    "DeviceIOSomaHandJointRotationsTracked",
    "DeviceIOSomaHandJointPosesTracked",
    "DeviceIOMessageChannelMessagesTracked",
    "MessageChannelMessagesTrackedGroup",
    "MessageChannelStatusGroup",
]

# Deprecated re-exports resolved lazily so access emits a DeprecationWarning; kept out
# of __all__ and the eager imports above so importing this module stays quiet.
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
