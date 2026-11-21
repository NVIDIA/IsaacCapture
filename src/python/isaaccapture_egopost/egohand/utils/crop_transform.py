# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Crop-transform utilities for EgoHand."""

import numpy as np


def get_transform(center, scale, res, rot=0):
    """Generate affine transformation matrix.

    Args:
        center: (2,) crop center
        scale: crop scale (pixels)
        res: (h, w) output resolution
        rot: rotation angle in degrees
    Returns:
        t: (2, 3) affine transform
    """
    h, w = res
    t = np.zeros((2, 3), dtype=np.float32)

    # Scale
    t[0, 0] = w / scale
    t[1, 1] = h / scale

    # Translation (center to origin then to output center)
    t[0, 2] = w / 2.0 - center[0] * w / scale
    t[1, 2] = h / 2.0 - center[1] * h / scale

    if rot != 0:
        # Rotation around output center
        rot_rad = np.deg2rad(rot)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        # Apply rotation to the scale+translate part
        center_out = np.array([w / 2.0, h / 2.0])
        t_rot = np.zeros_like(t)
        t_rot[:2, :2] = rot_mat @ t[:2, :2]
        t_rot[:2, 2] = rot_mat @ (t[:2, 2] - center_out) + center_out
        t = t_rot

    return t
