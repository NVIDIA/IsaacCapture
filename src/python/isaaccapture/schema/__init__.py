# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop Schema - FlatBuffer message types for teleoperation.

This module provides Python bindings for FlatBuffer-based message types used in
teleoperation, including poses and controller data.

Each table is a read-only view over its encoded bytes: attribute reads go straight
into the buffer, and joint arrays come back as zero-copy NumPy views. To produce one,
call its constructor -- that encodes the arguments and hands back the view, which is
the only way to build these from Python.
"""

import warnings

from ._schema import (
    # Timestamp types.
    DeviceDataTimestamp,
    # Pose-related types (structs).
    Point,
    Quaternion,
    Pose,
    # Head-related types.
    HeadPose,
    HeadPoseRecord,
    # Hand-related types.
    HandJoint,
    HandJointPose,
    HandJoints,
    HandPose,
    HandPoseRecord,
    # Controller-related types.
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    ControllerSnapshotRecord,
    # Pedals-related types.
    Generic3AxisPedalOutput,
    Generic3AxisPedalOutputRecord,
    # Keyboard types (held keys plus ordered key events).
    KeyAction,
    KeyEvent,
    KeyboardOutput,
    KeyboardOutputRecord,
    # OGLO tactile glove types.
    OgloGloveSample,
    OgloGloveSampleRecord,
    # Joint-state types (generic joint-space devices: leader arms, exoskeletons, ...).
    JointState,
    JointStateOutput,
    JointStateOutputRecord,
    # Joint SE(3) pose types (sparse tracked joints: raw glove tips, skeletons, ...).
    JointType,
    JointName,
    JointSe3Pose,
    JointSe3PoseOutput,
    JointSe3PoseOutputRecord,
    # SE3 tracker types (generic 6-DoF pose sources: tracker pucks, mocap rigid bodies, ...).
    Se3TrackerPose,
    Se3TrackerPoseRecord,
    # Message channel types.
    MessageChannelMessages,
    MessageChannelMessagesTracked,
    MessageChannelMessagesRecord,
    # Plugin device monitoring types.
    PluginDeviceReason,
    PluginDeviceState,
    PluginDeviceStatus,
    PluginDeviceStatusSnapshot,
    PluginDeviceStatusSnapshotRecord,
    # Haptic command types (vendor-neutral cross-process device output).
    HapticCommand,
    # Camera-related types.
    StreamType,
    FrameMetadataOak,
    FrameMetadataOakRecord,
    # Full body-related types.
    BodyJoint,
    BodyJointPose,
    BodyJoints,
    FullBodyPose,
    FullBodyPoseRecord,
)

# Deprecated aliases, resolved lazily via __getattr__ so accessing them emits a
# DeprecationWarning. Omitted from __all__.
#
# The `...T` spellings name the FlatBuffers object-API types, which Python does not see;
# each alias resolves to the encoded view instead. Reads are identical, but the view is
# immutable and is built by passing every field to the constructor rather than by
# assigning attributes afterwards.
#
_DEPRECATED_ALIASES = {
    "BodyJointPico": "BodyJoint",
    "BodyJointsPico": "BodyJoints",
    "FullBodyPosePicoT": "FullBodyPose",
    "FullBodyPosePicoRecord": "FullBodyPoseRecord",
    "HeadPoseT": "HeadPose",
    "HandPoseT": "HandPose",
    "Se3TrackerPoseT": "Se3TrackerPose",
    "MessageChannelMessagesTrackedT": "MessageChannelMessagesTracked",
    "FullBodyPoseT": "FullBodyPose",
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


__all__ = [
    # Timestamp types.
    "DeviceDataTimestamp",
    # Pose types (structs).
    "Point",
    "Quaternion",
    "Pose",
    # Head types.
    "HeadPose",
    "HeadPoseRecord",
    # Hand types.
    "HandJoint",
    "HandJointPose",
    "HandJoints",
    "HandPose",
    "HandPoseRecord",
    # Controller types.
    "ControllerInputState",
    "ControllerPose",
    "ControllerSnapshot",
    "ControllerSnapshotRecord",
    # Pedals types.
    "Generic3AxisPedalOutput",
    "Generic3AxisPedalOutputRecord",
    # Keyboard types (held keys plus ordered key events).
    "KeyAction",
    "KeyEvent",
    "KeyboardOutput",
    "KeyboardOutputRecord",
    # OGLO tactile glove types.
    "OgloGloveSample",
    "OgloGloveSampleRecord",
    # Joint-state types (generic joint-space devices).
    "JointState",
    "JointStateOutput",
    "JointStateOutputRecord",
    # Joint SE(3) pose types (sparse tracked joints).
    "JointType",
    "JointName",
    "JointSe3Pose",
    "JointSe3PoseOutput",
    "JointSe3PoseOutputRecord",
    # SE3 tracker types (generic 6-DoF pose sources).
    "Se3TrackerPose",
    "Se3TrackerPoseRecord",
    # Message channel types.
    "MessageChannelMessages",
    "MessageChannelMessagesTracked",
    "MessageChannelMessagesRecord",
    # Plugin device monitoring types.
    "PluginDeviceReason",
    "PluginDeviceState",
    "PluginDeviceStatus",
    "PluginDeviceStatusSnapshot",
    "PluginDeviceStatusSnapshotRecord",
    # Haptic command types.
    "HapticCommand",
    # Camera types.
    "StreamType",
    "FrameMetadataOak",
    "FrameMetadataOakRecord",
    # Full body types.
    "BodyJointPose",
    "BodyJoint",
    "BodyJoints",
    "FullBodyPose",
    "FullBodyPoseRecord",
]
