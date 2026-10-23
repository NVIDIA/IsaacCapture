# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Public EgoHand video API, shared with the independently packaged detector."""

from inference.media.video_io import (
    FPS_FALLBACK,
    find_videos,
    get_video_meta,
    read_video_frames_batched,
)

__all__ = ["FPS_FALLBACK", "find_videos", "get_video_meta", "read_video_frames_batched"]
