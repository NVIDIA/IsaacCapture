# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EGO recording, clock calibration, and offline sample association."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .recording import configure_recording, embedded_recording

if TYPE_CHECKING:
    from typing import Any

_SYNC_EXPORTS = (
    "RecordReference",
    "MediaReference",
    "VideoFrame",
    "ImuSample",
    "AudioSlice",
    "SyncWindow",
    "iter_sync_windows",
    "analyze_recording_sync",
    "recording_fingerprint",
)
_CALIBRATION_EXPORTS = (
    "PairedTimeEvent",
    "TimeAlignmentFit",
    "fit_time_alignment",
    "make_recording_alignment",
    "evidence_sha256",
)

__all__ = [
    "configure_recording",
    "embedded_recording",
    "RecordReference",
    "MediaReference",
    "VideoFrame",
    "ImuSample",
    "AudioSlice",
    "SyncWindow",
    "iter_sync_windows",
    "analyze_recording_sync",
    "recording_fingerprint",
    "PairedTimeEvent",
    "TimeAlignmentFit",
    "fit_time_alignment",
    "make_recording_alignment",
    "evidence_sha256",
]


def __getattr__(name: str) -> Any:
    if name in _SYNC_EXPORTS:
        module = import_module(".sync", __name__)
    elif name in _CALIBRATION_EXPORTS:
        module = import_module(".calibration", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
