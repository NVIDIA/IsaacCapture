# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JPEG encoding for SENSING frames: owned RGB host pixels in, one complete JPEG out.

The wire protocol carries only independently decodable JPEG images with a hard 2 MiB payload ceiling.
A frame that does not fit is a configuration/source error -- never silently resized, re-quantized or
replaced, so the diagnostic names the dimensions and quality that produced it.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .errors import SourceError

__all__ = ["MAX_JPEG_BYTES", "encode_rgb_jpeg"]

MAX_JPEG_BYTES = 2 * 1024 * 1024


def encode_rgb_jpeg(rgb: np.ndarray, quality: int, *, max_bytes: int = MAX_JPEG_BYTES) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB frame as a complete JPEG at ``quality``; returns owned bytes."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB pixels, got {rgb.shape} {rgb.dtype}")
    height, width = rgb.shape[:2]
    out = io.BytesIO()
    # Encoding finishes before this returns, so the caller may reuse its buffer immediately afterwards;
    # nothing here keeps a reference to it.
    Image.fromarray(rgb, "RGB").save(out, format="JPEG", quality=quality)
    jpeg = out.getvalue()
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise SourceError(f"encoder produced an incomplete JPEG for {width}x{height} at quality {quality}")
    if len(jpeg) > max_bytes:
        raise SourceError(
            f"{width}x{height} JPEG at quality {quality} is {len(jpeg)} B, over the {max_bytes} B payload ceiling"
        )
    return jpeg
