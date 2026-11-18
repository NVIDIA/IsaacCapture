# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live picture's encoder: one eye's camera frame as a small JPEG."""

from __future__ import annotations

import numpy as np


def encode_picture(nv12: np.ndarray, width_px: int, quality: int) -> bytes:
    """One eye's NV12 camera frame as a JPEG at most width_px wide."""
    import cv2

    rows, width = nv12.shape
    if rows % 3 or width % 2:
        raise ValueError(f"not an NV12 image: shape {nv12.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(nv12), cv2.COLOR_YUV2BGR_NV12)
    height = rows * 2 // 3
    if width > width_px:
        size = (width_px, max(1, round(height * width_px / width)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    done, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not done:
        raise ValueError("OpenCV could not pack the JPEG")
    return jpeg.tobytes()
