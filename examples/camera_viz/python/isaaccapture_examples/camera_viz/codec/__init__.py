# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVENC/NVDEC codec — pybind11 module re-exports.

Build the .so with ``codec/build.sh`` (also invoked by
``camera_viz.sh setup``). Requires CUDA toolkit + NVIDIA driver.
"""

from __future__ import annotations

try:
    from ._camera_viz_codec import (  # type: ignore[import-not-found]
        DecoderConfig,
        EncoderConfig,
        H264Decoder,
        H264Encoder,
        PixelFormat,
    )
except ImportError as e:
    raise ImportError(
        "camera_viz native codec not built. Run `examples/camera_viz/codec/build.sh`."
    ) from e

__all__ = [
    "DecoderConfig",
    "EncoderConfig",
    "H264Decoder",
    "H264Encoder",
    "PixelFormat",
]
