# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A minimal analytic hand for unit tests: four fingers curling in one plane each.

Deliberately cruder than ``../oracle/hand_model.py``, and unrelated to it, so a unit
test pins one measurement without depending on the oracle.
"""

from __future__ import annotations

import math

from acceptance_common.vectors import multiply, rotate

from hand_acceptance.frames import Frame, JointPose

KNUCKLES = {
    7: (-0.024, 0.0, -0.094),
    12: (-0.004, 0.0, -0.097),
    17: (0.015, 0.0, -0.091),
    22: (0.032, -0.004, -0.082),
}
BONES = (0.040, 0.025, 0.020)
THUMB = ((-0.022, -0.012, -0.028), (-0.6, -0.2, -0.77), (0.045, 0.032, 0.026))
PERIOD_NS = 16_666_667
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def axis_angle(axis, angle):
    n = math.sqrt(sum(c * c for c in axis))
    s = math.sin(angle / 2) / n
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def right_local(curl: float = 0.0) -> list[tuple[tuple, tuple]]:
    """26 (position, xyzw) pairs of a right hand in its wrist frame."""
    joints: list = [((0.0, 0.0, 0.0), IDENTITY)] * 26
    for knuckle, base in KNUCKLES.items():
        point, phi = base, 0.0
        joints[knuckle - 1] = (tuple(0.2 * c for c in base), IDENTITY)
        for k, bone in enumerate(BONES):
            phi += curl
            q = axis_angle((1.0, 0.0, 0.0), -phi)
            joints[knuckle + k] = (point, q)
            d = (0.0, -math.sin(phi), -math.cos(phi))
            point = tuple(p + bone * c for p, c in zip(point, d))
        joints[knuckle + 3] = (point, joints[knuckle + 2][1])
    base, direction, bones = THUMB
    n = math.sqrt(sum(c * c for c in direction))
    direction = tuple(c / n for c in direction)
    point = base
    for k, bone in enumerate(bones):
        joints[2 + k] = (point, IDENTITY)
        point = tuple(p + bone * c for p, c in zip(point, direction))
    joints[5] = (point, IDENTITY)
    middle_mc, middle_prox = joints[11][0], joints[12][0]
    joints[0] = (tuple((a + b) / 2 for a, b in zip(middle_mc, middle_prox)), IDENTITY)
    return joints


def mirrored(joints):
    return [((-p[0], p[1], p[2]), (q[0], -q[1], -q[2], q[3])) for p, q in joints]


def frame(
    tick: int,
    side: str = "right",
    local=None,
    wrist=(0.2, 1.1, -0.3),
    orientation=IDENTITY,
    valid=None,
    radius: float = 0.01,
) -> Frame:
    local = local if local is not None else right_local()
    if side == "left" and local is not None:
        local = mirrored(local)
    t = 1_000_000_000 + tick * PERIOD_NS
    joints = tuple(
        JointPose(
            position=tuple(w + c for w, c in zip(wrist, rotate(orientation, p))),
            orientation=multiply(orientation, q),
            is_valid=True if valid is None else i in valid,
            radius=radius,
        )
        for i, (p, q) in enumerate(local)
    )
    return Frame(
        sequence=tick,
        log_time_ns=t,
        publish_time_ns=t,
        has_payload=True,
        side=side,
        available_time_ns=t,
        sample_time_ns=t,
        device_time_ns=t + 5,
        joints=joints,
    )


def take(count: int, side: str = "right", transform=None, **kwargs) -> list[Frame]:
    """``count`` frames curling open to closed and back, so extremes exist."""
    frames = []
    for tick in range(count):
        curl = math.radians(70) * (1 - math.cos(2 * math.pi * tick / count)) / 2
        local = right_local(curl)
        if transform is not None:
            local = transform(local)
        frames.append(frame(tick, side, local=local, **kwargs))
    return frames
