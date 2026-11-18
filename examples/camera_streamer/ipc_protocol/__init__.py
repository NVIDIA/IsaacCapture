# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write-attempt markers shared by camera_receiver and its CUDA IPC readers."""

from __future__ import annotations

try:
    from ._slot_markers import (  # type: ignore[import-not-found]
        LAYOUT_VERSION,
        MAX_SLOTS,
        MAX_STREAMS,
        REGION_SIZE,
        SlotMarkers,
    )
except ImportError as e:
    raise ImportError(
        "camera_streamer slot marker module not built. Run "
        "`ipc_protocol/build.sh` from the camera_streamer directory."
    ) from e

__all__ = [
    "LAYOUT_VERSION",
    "MAX_SLOTS",
    "MAX_STREAMS",
    "REGION_SIZE",
    "SlotMarkers",
]
