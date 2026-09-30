# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector and quaternion arithmetic on plain tuples.

Checks run over whole recordings one frame at a time, so these are written to allocate
nothing beyond their result and to accept the tuples ``frames`` already hands out.
Quaternions are x,y,z,w, matching the wire schema.
"""

from __future__ import annotations

import math

Vector = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def sub(a: Vector, b: Vector) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalised(a: Vector) -> Vector | None:
    length = norm(a)
    if length < 1e-9 or not math.isfinite(length):
        return None
    return (a[0] / length, a[1] / length, a[2] / length)


def is_finite(a: tuple[float, ...]) -> bool:
    return all(math.isfinite(component) for component in a)


def rotate(q: Quaternion, v: Vector) -> Vector:
    """Rotates world-frame vector ``v`` by orientation ``q``."""
    x, y, z, w = q
    first = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    second = (
        y * first[2] - z * first[1],
        z * first[0] - x * first[2],
        x * first[1] - y * first[0],
    )
    return (
        v[0] + 2.0 * w * first[0] + 2.0 * second[0],
        v[1] + 2.0 * w * first[1] + 2.0 * second[1],
        v[2] + 2.0 * w * first[2] + 2.0 * second[2],
    )


def rotate_by_inverse(q: Quaternion, v: Vector) -> Vector:
    """Expresses world-frame vector ``v`` in the local frame of orientation ``q``."""
    x, y, z, w = q
    # The inverse of a unit quaternion is its conjugate.
    return rotate((-x, -y, -z, w), v)


def as_wxyz(stored: Quaternion) -> Quaternion:
    """Reinterprets fields written w,x,y,z into the x,y,z,w slots."""
    a, b, c, d = stored
    return (b, c, d, a)


def conjugate(q: Quaternion) -> Quaternion:
    return (-q[0], -q[1], -q[2], q[3])


def multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def relative(parent: Quaternion, child: Quaternion) -> Quaternion:
    """The child's rotation expressed in the parent's frame."""
    return multiply(conjugate(parent), child)


def signed_angle_about(q: Quaternion, axis: Vector) -> float:
    """Radians of ``q`` about ``axis``, signed by the right-hand rule.

    A scripted pose built as a single rotation about one axis is recovered in its own
    units by projecting onto that axis, instead of as the magnitude that ``2*acos(w)``
    would give.
    """
    unit = normalised(axis)
    if unit is None:
        return 0.0
    along = dot((q[0], q[1], q[2]), unit)
    return 2.0 * math.atan2(along, q[3])
