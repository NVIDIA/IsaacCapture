# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop DeviceIO Trackers — tracker classes for device I/O."""

import warnings

from ._deviceio_trackers import (
    ITracker,
    HandTracker,
    HeadTracker,
    GamepadTracker,
    discover_gamepad_device,
    ControllerTracker,
    MessageChannelStatus,
    MessageChannelTracker,
    HapticCommandReaderTracker,
    TensorPushTracker,
    FullBodyTracker,
    ITrackerSession,
    NUM_JOINTS,
    JOINT_PALM,
    JOINT_WRIST,
    JOINT_THUMB_TIP,
    JOINT_INDEX_TIP,
)

# Tracker classes declared in trackers.toml. Star-imported (and appended to __all__ below)
# so a new manifest entry needs no edit here.
from ._generated_tracker_exports import *  # noqa: F403
from ._generated_tracker_exports import __all__ as _GENERATED_TRACKERS

# Deprecated aliases resolved lazily via __getattr__ so that accessing them emits a
# DeprecationWarning. Intentionally omitted from __all__ so `import *` no longer pulls
# the old names.
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


# PLE0604: _GENERATED_TRACKERS is _generated_tracker_exports.__all__, a list of str.
# ruff cannot see across the module boundary to prove that.
__all__ = [  # noqa: PLE0604
    "ControllerTracker",
    "MessageChannelStatus",
    "MessageChannelTracker",
    "FullBodyTracker",
    "HapticCommandReaderTracker",
    "TensorPushTracker",
    "HandTracker",
    "HeadTracker",
    "GamepadTracker",
    "discover_gamepad_device",
    "ITracker",
    "JOINT_INDEX_TIP",
    "JOINT_PALM",
    "JOINT_THUMB_TIP",
    "JOINT_WRIST",
    "NUM_JOINTS",
    "ITrackerSession",
    *_GENERATED_TRACKERS,
]
