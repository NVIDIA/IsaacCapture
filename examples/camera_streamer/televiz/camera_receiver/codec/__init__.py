# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native NVDEC video decoder for camera_receiver."""

from __future__ import annotations

try:
    from ._camera_receiver_codec import (  # type: ignore[import-not-found]
        Codec,
        DecoderConfig,
        NvVideoDecoder,
    )
except ImportError as e:
    raise ImportError(
        "camera_receiver native codec not built. Run "
        "`python launch.py build-receiver` from examples/camera_streamer with the host venv."
    ) from e

__all__ = [
    "Codec",
    "DecoderConfig",
    "NvVideoDecoder",
]
