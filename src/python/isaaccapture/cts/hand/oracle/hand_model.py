# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The oracle's own hand: geometry, forward kinematics and a thumb IK; independent of the checker.

Built as a right hand in the OpenXR wrist frame (+Y dorsal, +Z toward the forearm,
fingers along -Z, X = Y x Z, so the little finger is on +X) and mirrored for the left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # x, y, z, w

NUM_JOINTS = 26
PALM, WRIST = 0, 1
THUMB = (2, 3, 4, 5)  # METACARPAL, PROXIMAL, DISTAL, TIP
FINGER_JOINTS = {  # METACARPAL, PROXIMAL, INTERMEDIATE, DISTAL, TIP
    "index": (6, 7, 8, 9, 10),
    "middle": (11, 12, 13, 14, 15),
    "ring": (16, 17, 18, 19, 20),
    "little": (21, 22, 23, 24, 25),
}
FINGERS = tuple(FINGER_JOINTS)

# Right hand, metres, wrist frame. Knuckle = PROXIMAL joint; phalanges proximal,
# intermediate, distal. Splay turns a finger about +Y; positive fans toward -X.
KNUCKLE: dict[str, Vec] = {
    "index": (-0.024, 0.000, -0.094),
    "middle": (-0.004, 0.002, -0.097),
    "ring": (0.015, 0.000, -0.091),
    "little": (0.032, -0.004, -0.082),
}
PHALANGES: dict[str, tuple[float, float, float]] = {
    "index": (0.040, 0.024, 0.020),
    "middle": (0.045, 0.028, 0.022),
    "ring": (0.042, 0.026, 0.021),
    "little": (0.032, 0.019, 0.018),
}
SPLAY: dict[str, float] = {"index": 0.08, "middle": 0.0, "ring": -0.06, "little": -0.13}
# Finger METACARPAL joints sit near the wrist end of their metacarpal bone.
METACARPAL_FRACTION = 0.22

THUMB_BASE: Vec = (-0.022, -0.012, -0.028)
THUMB_BONES = (0.046, 0.033, 0.027)
THUMB_OPEN_DIR: Vec = (-0.55, -0.18, -0.82)
# Roughly where the thumbnail faces; only used to fix the thumb joints' roll.
THUMB_DORSAL_HINT: Vec = (-0.6, 0.8, 0.0)

RADIUS = {"tip": 0.008, "joint": 0.010, "wrist": 0.020, "palm": 0.025}


# ---- vector and quaternion arithmetic -------------------------------------------------


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec, s: float) -> Vec:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec) -> Vec:
    n = length(a)
    return (a[0] / n, a[1] / n, a[2] / n)


def quat_from_axes(x: Vec, y: Vec, z: Vec) -> Quat:
    """Quaternion whose rotation matrix has columns x, y, z (Shepperd's method)."""
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        q = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        q = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        q = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[3] >= 0 else tuple(-c for c in q)


def quat_axis_angle(axis: Vec, angle: float) -> Quat:
    a = unit(axis)
    s = math.sin(angle / 2)
    return (a[0] * s, a[1] * s, a[2] * s, math.cos(angle / 2))


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def quat_nlerp(a: Quat, b: Quat, t: float) -> Quat:
    if sum(x * y for x, y in zip(a, b)) < 0:
        b = (-b[0], -b[1], -b[2], -b[3])
    q = tuple(x + (y - x) * t for x, y in zip(a, b))
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)  # type: ignore[return-value]


def quat_rotate(q: Quat, v: Vec) -> Vec:
    qv = (v[0], v[1], v[2], 0.0)
    conj = (-q[0], -q[1], -q[2], q[3])
    r = quat_mul(quat_mul(q, qv), conj)
    return (r[0], r[1], r[2])


def frame_along(bone: Vec, dorsal_hint: Vec) -> Quat:
    """OpenXR joint frame: -Z along ``bone``, +Y as close to ``dorsal_hint`` as allowed."""
    z = unit(scale(bone, -1.0))
    y = unit(sub(dorsal_hint, scale(z, dot(dorsal_hint, z))))
    return quat_from_axes(cross(y, z), y, z)


# ---- poses ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """Flexion in radians per finger joint (MCP, PIP, DIP) and a thumb tip target."""

    flex: dict[str, tuple[float, float, float]]
    thumb_target: Vec | None = None


def _deg(*values: float) -> tuple[float, ...]:
    return tuple(math.radians(v) for v in values)


OPEN = Pose({f: _deg(2, 3, 2) for f in FINGERS})
# Pressed flat by the table.
FLAT = Pose({f: _deg(0, 0, 0) for f in FINGERS})
FIST = Pose(
    {f: _deg(85, 100, 65) for f in FINGERS},
    thumb_target=(-0.004, -0.048, -0.080),
)
SHALLOW_FIST = Pose(
    {f: _deg(60, 70, 40) for f in FINGERS},
    thumb_target=(-0.030, -0.040, -0.110),
)
PINCH_FLEX = {
    "index": _deg(38, 35, 18),
    "middle": _deg(42, 38, 18),
    "ring": _deg(50, 45, 22),
    "little": _deg(58, 50, 25),
}


def pinch(finger: str) -> Pose:
    flex = {f: _deg(8, 10, 5) for f in FINGERS}
    flex[finger] = PINCH_FLEX[finger]
    return Pose(flex, thumb_target=("pinch", finger))  # resolved in solve()


def blend(a: Pose, b: Pose, t: float) -> Pose:
    flex = {
        f: tuple(x + (y - x) * t for x, y in zip(a.flex[f], b.flex[f])) for f in FINGERS
    }
    return Pose(flex, thumb_target=(a, b, t))


# ---- kinematics -----------------------------------------------------------------------


def _finger(finger: str, flex: tuple[float, float, float]) -> list[tuple[Vec, Quat]]:
    """PROXIMAL, INTERMEDIATE, DISTAL, TIP poses of one finger."""
    psi = SPLAY[finger]
    point = KNUCKLE[finger]
    phi = 0.0
    out: list[tuple[Vec, Quat]] = []
    for bone, bend in zip(PHALANGES[finger], flex):
        phi += bend
        direction = (
            -math.cos(phi) * math.sin(psi),
            -math.sin(phi),
            -math.cos(phi) * math.cos(psi),
        )
        dorsal = (
            -math.sin(phi) * math.sin(psi),
            math.cos(phi),
            -math.sin(phi) * math.cos(psi),
        )
        out.append((point, frame_along(direction, dorsal)))
        point = add(point, scale(direction, bone))
    out.append((point, out[-1][1]))
    return out


def fabrik(
    base: Vec, bones: tuple[float, ...], start: list[Vec], target: Vec
) -> list[Vec]:
    """FABRIK on a chain rooted at ``base``; returns joint positions including the tip."""
    points = list(start)
    reach = sum(bones)
    if length(sub(target, base)) >= reach:
        direction = unit(sub(target, base))
        points = [base]
        for bone in bones:
            points.append(add(points[-1], scale(direction, bone)))
        return points
    for _ in range(40):
        points[-1] = target
        for i in range(len(bones) - 1, -1, -1):
            d = unit(sub(points[i], points[i + 1]))
            points[i] = add(points[i + 1], scale(d, bones[i]))
        points[0] = base
        for i in range(len(bones)):
            d = unit(sub(points[i + 1], points[i]))
            points[i + 1] = add(points[i], scale(d, bones[i]))
        if length(sub(points[-1], target)) < 1e-5:
            break
    return points


def _thumb_open() -> list[Vec]:
    direction = unit(THUMB_OPEN_DIR)
    points = [THUMB_BASE]
    for bone in THUMB_BONES:
        points.append(add(points[-1], scale(direction, bone)))
    return points


def _thumb_target(pose: Pose, fingers: dict[str, list[tuple[Vec, Quat]]]) -> Vec | None:
    target = pose.thumb_target
    if target is None:
        return None
    if isinstance(target[0], str):
        tip, tip_frame = fingers[target[1]][-1]
        # Pad to pad: rest the thumb tip just palmar of the fingertip.
        return add(tip, scale(quat_rotate(tip_frame, (0.0, -1.0, 0.0)), 0.012))
    if isinstance(target[0], Pose):
        a, b, t = target
        ta, tb = _thumb_target(a, fingers), _thumb_target(b, fingers)
        open_tip = _thumb_open()[-1]
        ta = ta or open_tip
        tb = tb or open_tip
        return add(ta, scale(sub(tb, ta), t))
    return target


def solve(pose: Pose) -> list[tuple[Vec, Quat]]:
    """All 26 joint poses of a right hand in its wrist frame."""
    joints: list[tuple[Vec, Quat] | None] = [None] * NUM_JOINTS
    identity: Quat = (0.0, 0.0, 0.0, 1.0)
    joints[WRIST] = ((0.0, 0.0, 0.0), identity)

    fingers = {f: _finger(f, pose.flex[f]) for f in FINGERS}
    for finger, chain in FINGER_JOINTS.items():
        knuckle = fingers[finger][0][0]
        base = scale(knuckle, METACARPAL_FRACTION)
        joints[chain[0]] = (base, frame_along(sub(knuckle, base), (0.0, 1.0, 0.0)))
        for index, placed in zip(chain[1:], fingers[finger]):
            joints[index] = placed

    middle_mc, middle_knuckle = joints[11][0], joints[12][0]
    palm = scale(add(middle_mc, middle_knuckle), 0.5)
    joints[PALM] = (palm, frame_along(sub(middle_knuckle, middle_mc), (0.0, 1.0, 0.0)))

    points = _thumb_open()
    target = _thumb_target(pose, fingers)
    if target is not None:
        points = fabrik(THUMB_BASE, THUMB_BONES, points, target)
    for i, index in enumerate(THUMB):
        bone = sub(points[i + 1], points[i]) if i < 3 else sub(points[3], points[2])
        joints[index] = (points[i], frame_along(bone, THUMB_DORSAL_HINT))
    return joints  # type: ignore[return-value]


def mirror(joints: list[tuple[Vec, Quat]]) -> list[tuple[Vec, Quat]]:
    """Right hand to left: x negated, rotations conjugated by the mirror, R -> M R M."""
    return [((-p[0], p[1], p[2]), (q[0], -q[1], -q[2], q[3])) for p, q in joints]


def radius(index: int) -> float:
    if index == WRIST:
        return RADIUS["wrist"]
    if index == PALM:
        return RADIUS["palm"]
    if index in (5, 10, 15, 20, 25):
        return RADIUS["tip"]
    return RADIUS["joint"]
