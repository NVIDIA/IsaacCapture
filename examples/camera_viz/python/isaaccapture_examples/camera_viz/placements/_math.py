# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin scipy.spatial.transform.Rotation adapters + a couple of stdlib helpers.

Quaternion convention matches OpenXR / GLM: ``(w, x, y, z)``. scipy's
``Rotation`` uses ``(x, y, z, w)``, so the adapters here convert at the
boundary and the rest of the code stays on the wxyz tuple it already
hands to ``viz.Pose3D``.

The non-rotation helpers (``angle_between_xz_deg``, ``normalize_angle``,
``smoothstep``) are stdlib one-liners — no point pulling a lib just for
those.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)


def _rot_from_wxyz(q: Quat) -> Rotation:
    """Build a scipy Rotation from a wxyz quaternion (our convention)."""
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w])


def _rot_to_wxyz(r: Rotation) -> Quat:
    """Convert a scipy Rotation back to a wxyz quaternion tuple."""
    x, y, z, w = r.as_quat()
    return (float(w), float(x), float(y), float(z))


def rotate_vec(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector ``v`` by unit quaternion ``q``."""
    rotated = _rot_from_wxyz(q).apply(np.asarray(v, dtype=np.float64))
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product ``a * b``."""
    return _rot_to_wxyz(_rot_from_wxyz(a) * _rot_from_wxyz(b))


def yaw_quat(yaw: float) -> Quat:
    """Quaternion for a rotation of ``yaw`` radians about +Y."""
    return _rot_to_wxyz(Rotation.from_euler("y", yaw))


def heading_deg(q: Quat) -> float:
    """Which way ``q`` faces about +Y, in degrees.

    0 looks down -Z (the reference space's forward), +90 down +X's opposite,
    i.e. positive is a turn to the left. Inverse of :func:`yaw_quat` up to
    the projection, so ``heading_deg(yaw_quat(t)) == degrees(t)``.
    """
    fx, _fy, fz = project_forward_xz(q)
    return math.degrees(math.atan2(-fx, -fz))


def project_forward_xz(q: Quat) -> Vec3:
    """OpenXR head forward (-Z in head local) projected onto XZ, normalized.

    Falls back to ``(0, 0, -1)`` if the head is looking near-vertical."""
    fx, _fy, fz = rotate_vec(q, (0.0, 0.0, -1.0))
    length = math.hypot(fx, fz)
    if length < 1e-3:
        return (0.0, 0.0, -1.0)
    return (fx / length, 0.0, fz / length)


def angle_between_xz_deg(a: Vec3, b: Vec3) -> float:
    """Angle between two XZ-plane vectors, in degrees. 0 if either is ~0."""
    la = math.hypot(a[0], a[2])
    lb = math.hypot(b[0], b[2])
    if la < 1e-6 or lb < 1e-6:
        return 0.0
    d = max(-1.0, min(1.0, (a[0] * b[0] + a[2] * b[2]) / (la * lb)))
    return math.degrees(math.acos(d))


def normalize_angle(angle: float) -> float:
    """Wrap ``angle`` into (-pi, pi]."""
    two_pi = 2.0 * math.pi
    return ((angle + math.pi) % two_pi) - math.pi


def smoothstep(t: float) -> float:
    """C¹-continuous interpolation curve, t clamped to [0, 1]."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)
