# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK capture binding for the EGO live-preview source.

Capture owns one stereo SDK pipeline and returns paired host payloads.
sources.ego handles GPU upload/decoding and the FrameSource lifecycle;
sensor recording and MCAP output belong to the separate EGO plugin.
See README.md here for build instructions and payload semantics.
"""

try:
    from ._ego_capture import Capture
except ImportError as exc:
    raise ImportError(
        "EGO camera_viz binding is not built. Run "
        "`ego_preview/build.sh --sdk-root PATH`."
    ) from exc

__all__ = ["Capture"]
