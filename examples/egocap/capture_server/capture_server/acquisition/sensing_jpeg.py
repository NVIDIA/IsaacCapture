# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JPEG encoding for SENSING frames: owned RGB host pixels in, one complete JPEG out.

The wire protocol carries only independently decodable JPEG images with a hard 2 MiB payload ceiling.
A frame that does not fit is a configuration/source error -- never silently resized, re-quantized or
replaced, so the diagnostic names the dimensions and quality that produced it.

OpenCV encodes here rather than Pillow: at 5 MP it is about 40% faster, and it releases the GIL for the whole
encode, which keeps the other streams' decoder and file-reading threads running while one frame is compressed.
"""

from __future__ import annotations

import numpy as np

from .errors import SourceError

__all__ = ["MAX_JPEG_BYTES", "encode_rgb_jpeg"]

MAX_JPEG_BYTES = 2 * 1024 * 1024


def encode_rgb_jpeg(rgb: np.ndarray, quality: int, *, max_bytes: int = MAX_JPEG_BYTES) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB frame as a complete JPEG at ``quality``; returns owned bytes."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 RGB pixels, got {rgb.shape} {rgb.dtype}")
    import cv2  # noqa: PLC0415 -- lazy: runs without a SENSING stream never load OpenCV

    height, width = rgb.shape[:2]
    # OpenCV's encoder takes BGR; the conversion makes a new array, so nothing here keeps a reference to the
    # caller's buffer and the caller may reuse it as soon as this returns.
    encoded, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not encoded:
        raise SourceError(f"encoder failed for {width}x{height} at quality {quality}")
    jpeg = buffer.tobytes()
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise SourceError(f"encoder produced an incomplete JPEG for {width}x{height} at quality {quality}")
    if len(jpeg) > max_bytes:
        raise SourceError(
            f"{width}x{height} JPEG at quality {quality} is {len(jpeg)} B, over the {max_bytes} B payload ceiling"
        )
    return jpeg
