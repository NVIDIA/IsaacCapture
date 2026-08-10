# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop DEVICEIO - Device I/O Module (backward-compatible re-exports)

Prefer importing directly:
    from isaaccapture.deviceio_trackers import HeadTracker, HandTracker
    from isaaccapture.deviceio_session import DeviceIOSession, McapRecordingConfig
"""

import warnings

from isaaccapture.deviceio_trackers import (
    ITracker,
    HandTracker,
    HeadTracker,
    ControllerTracker,
    MessageChannelStatus,
    MessageChannelTracker,
    FrameMetadataTrackerOak,
    Generic3AxisPedalTracker,
    OgloTactileTracker,
    TensorPushTracker,
    JointStateTracker,
    FullBodyTracker,
    NUM_JOINTS,
    JOINT_PALM,
    JOINT_WRIST,
    JOINT_THUMB_TIP,
    JOINT_INDEX_TIP,
)

from isaaccapture.deviceio_session import (
    DeviceIOSession,
    McapRecordingConfig,
    McapReplayConfig,
    ReplaySession,
    TrackerVendor,
    VendorConfig,
)

from ..oxr import OpenXRSessionHandles

from ..schema import (
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    DeviceDataTimestamp,
    StreamType,
    FrameMetadataOak,
    Generic3AxisPedalOutput,
    OgloGloveSample,
)

__all__ = [
    "ControllerInputState",
    "ControllerPose",
    "ControllerSnapshot",
    "DeviceDataTimestamp",
    "StreamType",
    "FrameMetadataOak",
    "Generic3AxisPedalOutput",
    "OgloGloveSample",
    "ITracker",
    "HandTracker",
    "HeadTracker",
    "ControllerTracker",
    "MessageChannelStatus",
    "MessageChannelTracker",
    "FrameMetadataTrackerOak",
    "Generic3AxisPedalTracker",
    "OgloTactileTracker",
    "TensorPushTracker",
    "JointStateTracker",
    "FullBodyTracker",
    "OpenXRSessionHandles",
    "DeviceIOSession",
    "McapRecordingConfig",
    "McapReplayConfig",
    "ReplaySession",
    "TrackerVendor",
    "VendorConfig",
    "NUM_JOINTS",
    "JOINT_PALM",
    "JOINT_WRIST",
    "JOINT_THUMB_TIP",
    "JOINT_INDEX_TIP",
]

# Deprecated re-exports resolved lazily so access emits a DeprecationWarning; kept out
# of __all__ and out of the eager imports above so importing this module stays quiet.
_DEPRECATED_ALIASES = {"FullBodyTrackerPico": "FullBodyTracker"}


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
