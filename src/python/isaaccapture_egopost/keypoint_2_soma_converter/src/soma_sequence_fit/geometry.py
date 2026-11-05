# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Camera geometry, robust rigid initialization, and rotation utilities."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# This is an approximate semantic mapping, not a MANO-derived regressor.
DEFAULT_JOINT_MAP = [
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
]
FINGERS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]


def project_np(points, K):
    """Project OpenCV camera-space points; invalid/behind-camera points become NaN."""
    p = np.asarray(points)
    h = p @ np.asarray(K).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = h[..., :2] / h[..., 2:3]
    return np.where(
        (np.isfinite(p).all(-1) & (p[..., 2] > 1e-5))[..., None], uv, np.nan
    )


def project(points, K):
    h = points @ K.transpose(-1, -2)
    return h[..., :2] / h[..., 2:3].clamp_min(1e-5)


def rotvec_matrix(v):
    """Stable Rodrigues formula with finite derivatives at zero rotation."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *v.shape[:-1], 3, 3
    )
    theta = torch.linalg.vector_norm(v, dim=-1)[..., None, None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(skew)
    return (
        eye
        + torch.sinc(theta / torch.pi) * skew
        + 0.5 * torch.sinc(theta / (2 * torch.pi)) ** 2 * (skew @ skew)
    )


def align_points(source, target, weights, estimate_scale=True):
    """Weighted proper similarity alignment, using row-vector point arrays."""
    valid = np.isfinite(source).all(1) & np.isfinite(target).all(1) & (weights > 0)
    if valid.sum() < 3:
        raise ValueError(
            "At least three reliable non-collinear correspondences are required."
        )
    x, y, w = source[valid], target[valid], weights[valid].astype(float)
    w /= w.sum()
    mx, my = (x * w[:, None]).sum(0), (y * w[:, None]).sum(0)
    xc, yc = x - mx, y - my
    if (
        np.linalg.matrix_rank(xc, tol=1e-6) < 2
        or np.linalg.matrix_rank(yc, tol=1e-6) < 2
    ):
        raise ValueError("Degenerate landmark geometry.")
    u, _, vt = np.linalg.svd((xc * w[:, None]).T @ yc)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ fix @ u.T
    scale = (
        float(
            ((xc @ R.T) * yc * w[:, None]).sum()
            / max((xc * xc * w[:, None]).sum(), 1e-12)
        )
        if estimate_scale
        else 1.0
    )
    scale = float(np.clip(scale, 0.4, 2.5))
    return R, my - scale * (R @ mx), scale


def matrix_rotvec(R):
    return Rotation.from_matrix(np.asarray(R)).as_rotvec().astype(np.float32)


def crop_camera(K, box, size):
    x0, y0, x1, y1 = box
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    # Pixel-center convention matches OpenCV resize/warp sampling.
    A = np.array(
        [[sx, 0, sx * (0.5 - x0) - 0.5], [0, sy, sy * (0.5 - y0) - 0.5], [0, 0, 1]],
        np.float32,
    )
    return A @ K, A
