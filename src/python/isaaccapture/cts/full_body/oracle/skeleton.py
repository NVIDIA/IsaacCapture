# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skeleton model, quaternion math and the scripted clean animation.

Right-handed, Y-up, metres. The subject faces -Z, so RIGHT_* joints sit at +X.
Quaternions are (x, y, z, w), matching ``core.Quaternion``.

Joint indices and parents come from ``src/core/schema/fbs/full_body.fbs`` (BodyJoint
enum) and ``docs/source/device/body_tracking.rst``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

# --------------------------------------------------------------------------------------
# Joint layout (ground truth: full_body.fbs BodyJoint enum)
# --------------------------------------------------------------------------------------

JOINT_NAMES = [
    "PELVIS",  # 0
    "LEFT_HIP",  # 1
    "RIGHT_HIP",  # 2
    "SPINE1",  # 3
    "LEFT_KNEE",  # 4
    "RIGHT_KNEE",  # 5
    "SPINE2",  # 6
    "LEFT_ANKLE",  # 7
    "RIGHT_ANKLE",  # 8
    "SPINE3",  # 9
    "LEFT_FOOT",  # 10
    "RIGHT_FOOT",  # 11
    "NECK",  # 12
    "LEFT_COLLAR",  # 13
    "RIGHT_COLLAR",  # 14
    "HEAD",  # 15
    "LEFT_SHOULDER",  # 16
    "RIGHT_SHOULDER",  # 17
    "LEFT_ELBOW",  # 18
    "RIGHT_ELBOW",  # 19
    "LEFT_WRIST",  # 20
    "RIGHT_WRIST",  # 21
    "LEFT_HAND",  # 22
    "RIGHT_HAND",  # 23
]
NUM_JOINTS = 24
J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Parent of each joint, from docs/source/device/body_tracking.rst. -1 = root.
PARENT = [
    -1,  # PELVIS
    J["PELVIS"],  # LEFT_HIP
    J["PELVIS"],  # RIGHT_HIP
    J["PELVIS"],  # SPINE1
    J["LEFT_HIP"],  # LEFT_KNEE
    J["RIGHT_HIP"],  # RIGHT_KNEE
    J["SPINE1"],  # SPINE2
    J["LEFT_KNEE"],  # LEFT_ANKLE
    J["RIGHT_KNEE"],  # RIGHT_ANKLE
    J["SPINE2"],  # SPINE3
    J["LEFT_ANKLE"],  # LEFT_FOOT
    J["RIGHT_ANKLE"],  # RIGHT_FOOT
    J["SPINE3"],  # NECK
    J["SPINE3"],  # LEFT_COLLAR
    J["SPINE3"],  # RIGHT_COLLAR
    J["NECK"],  # HEAD
    J["LEFT_COLLAR"],  # LEFT_SHOULDER
    J["RIGHT_COLLAR"],  # RIGHT_SHOULDER
    J["LEFT_SHOULDER"],  # LEFT_ELBOW
    J["RIGHT_SHOULDER"],  # RIGHT_ELBOW
    J["LEFT_ELBOW"],  # LEFT_WRIST
    J["RIGHT_ELBOW"],  # RIGHT_WRIST
    J["LEFT_WRIST"],  # LEFT_HAND
    J["RIGHT_WRIST"],  # RIGHT_HAND
]

# Mirror partner of each joint (left <-> right); self for the spine chain.
MIRROR = list(range(NUM_JOINTS))
for _l, _r in [
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_KNEE", "RIGHT_KNEE"),
    ("LEFT_ANKLE", "RIGHT_ANKLE"),
    ("LEFT_FOOT", "RIGHT_FOOT"),
    ("LEFT_COLLAR", "RIGHT_COLLAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_HAND", "RIGHT_HAND"),
]:
    MIRROR[J[_l]] = J[_r]
    MIRROR[J[_r]] = J[_l]

# --------------------------------------------------------------------------------------
# Rest offsets: child position in the parent joint's frame, metres. PELVIS holds the root world position.
# --------------------------------------------------------------------------------------

REST_OFFSETS = {
    "PELVIS": (0.000, 0.980, 0.000),  # world root
    "LEFT_HIP": (-0.090, -0.020, 0.000),
    "RIGHT_HIP": (0.090, -0.020, 0.000),
    "SPINE1": (0.000, 0.100, 0.000),
    "LEFT_KNEE": (0.000, -0.420, 0.000),
    "RIGHT_KNEE": (0.000, -0.420, 0.000),
    "SPINE2": (0.000, 0.120, 0.000),
    "LEFT_ANKLE": (0.000, -0.410, 0.000),
    "RIGHT_ANKLE": (0.000, -0.410, 0.000),
    "SPINE3": (0.000, 0.130, 0.000),
    "LEFT_FOOT": (0.000, -0.070, -0.120),  # forward is -Z
    "RIGHT_FOOT": (0.000, -0.070, -0.120),
    "NECK": (0.000, 0.180, 0.000),
    "LEFT_COLLAR": (-0.040, 0.140, 0.000),
    "RIGHT_COLLAR": (0.040, 0.140, 0.000),
    "HEAD": (0.000, 0.120, 0.000),
    "LEFT_SHOULDER": (-0.130, 0.020, 0.000),
    "RIGHT_SHOULDER": (0.130, 0.020, 0.000),
    "LEFT_ELBOW": (-0.285, 0.000, 0.000),  # upper arm
    "RIGHT_ELBOW": (0.285, 0.000, 0.000),
    "LEFT_WRIST": (-0.255, 0.000, 0.000),  # forearm
    "RIGHT_WRIST": (0.255, 0.000, 0.000),
    "LEFT_HAND": (-0.090, 0.000, 0.000),
    "RIGHT_HAND": (0.090, 0.000, 0.000),
}
REST = [REST_OFFSETS[n] for n in JOINT_NAMES]

# --------------------------------------------------------------------------------------
# Quaternion / vector helpers.  Quaternions are (x, y, z, w).
# --------------------------------------------------------------------------------------

QID = (0.0, 0.0, 0.0, 1.0)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def qaxis(axis, angle_rad):
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    s = math.sin(angle_rad / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle_rad / 2.0))


def qrot(q, v):
    """Rotate vector v by quaternion q."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ramp(t, t0, t1):
    """Smooth 0->1 over [t0, t1]."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return smoothstep((t - t0) / (t1 - t0))


def pulse(t, t0, t1, t2, t3):
    """Smooth 0 -> 1 over [t0,t1], hold, 1 -> 0 over [t2,t3]."""
    return ramp(t, t0, t1) * (1.0 - ramp(t, t2, t3))


# --------------------------------------------------------------------------------------
# Recording data model.
# --------------------------------------------------------------------------------------


@dataclass
class Joint:
    pos: tuple  # (x, y, z) metres
    quat: tuple  # (x, y, z, w)
    valid: bool = True


@dataclass
class Frame:
    # ``None`` means the FlatBuffer ``joints`` struct field is absent from FullBodyPose.
    joints: Optional[List[Joint]]
    all_tracked: bool
    available_ns: int
    sample_ns: int
    device_ns: int
    # ``False`` means the record carries a timestamp but no ``data`` table.
    has_data: bool = True


@dataclass
class Recording:
    frames: List[Frame] = field(default_factory=list)


def zero_joint() -> Joint:
    """A zeroed joint with is_valid false, as the Noitom plugin's make_invalid_body_joint_pose() produces."""
    return Joint(pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 0.0), valid=False)


# --------------------------------------------------------------------------------------
# The scripted animation.
# --------------------------------------------------------------------------------------

# Scripted timeline, seconds.
T_APOSE_END = 2.5  # A-pose still window
T_TPOSE_IN = 4.0  # A -> T transition done
T_TPOSE1_END = 6.5  # T-pose still window #1
T_ARM_UP = 7.5  # right arm fully raised
T_ARM_DOWN = 8.5  # right arm back to T
T_LEG_UP = 9.5  # left leg fully raised
T_LEG_DOWN = 10.5  # left leg back down
T_SETTLE = 11.5  # settled back into T
DURATION = 14.0  # T-pose still window #2 runs T_SETTLE -> DURATION

STILL_WINDOWS = {
    "a_pose": (0.2, T_APOSE_END),
    "t_pose_open": (T_TPOSE_IN + 0.2, T_TPOSE1_END),
    "t_pose_close": (T_SETTLE + 0.2, DURATION),
}


def local_rotations(t: float, rng: random.Random) -> List[tuple]:
    """Per-joint parent-local rotation at time ``t``; the rest pose is the T-pose."""
    q = [QID] * NUM_JOINTS

    # A-pose: arms swung down; +Z rotation lowers the left arm, so the right takes the opposite sign.
    a_pose = 1.0 - ramp(t, T_APOSE_END, T_TPOSE_IN)
    arm_down = math.radians(52.0) * a_pose
    q[J["LEFT_SHOULDER"]] = qaxis((0, 0, 1), arm_down)
    q[J["RIGHT_SHOULDER"]] = qaxis((0, 0, 1), -arm_down)

    # Single-arm raise: right arm only.
    raise_amt = pulse(t, T_TPOSE1_END, T_ARM_UP, T_ARM_UP, T_ARM_DOWN)
    if raise_amt > 0.0:
        q[J["RIGHT_SHOULDER"]] = qmul(
            q[J["RIGHT_SHOULDER"]], qaxis((0, 0, 1), math.radians(98.0) * raise_amt)
        )
        q[J["RIGHT_ELBOW"]] = qaxis((0, 1, 0), math.radians(-18.0) * raise_amt)

    # Single-leg raise: left knee lift only.
    leg_amt = pulse(t, T_ARM_DOWN, T_LEG_UP, T_LEG_UP, T_LEG_DOWN)
    if leg_amt > 0.0:
        q[J["LEFT_HIP"]] = qaxis((1, 0, 0), math.radians(72.0) * leg_amt)
        q[J["LEFT_KNEE"]] = qaxis((1, 0, 0), math.radians(-82.0) * leg_amt)
        q[J["LEFT_ANKLE"]] = qaxis((1, 0, 0), math.radians(14.0) * leg_amt)
        # weight shifts onto the right leg
        q[J["PELVIS"]] = qaxis((0, 0, 1), math.radians(-5.0) * leg_amt)

    # Breathing / postural sway, present throughout.
    breath = math.sin(2.0 * math.pi * 0.26 * t)
    sway = math.sin(2.0 * math.pi * 0.13 * t + 0.7)
    q[J["SPINE2"]] = qmul(q[J["SPINE2"]], qaxis((1, 0, 0), math.radians(0.9) * breath))
    q[J["SPINE1"]] = qmul(q[J["SPINE1"]], qaxis((0, 0, 1), math.radians(0.6) * sway))
    q[J["NECK"]] = qmul(q[J["NECK"]], qaxis((0, 1, 0), math.radians(1.4) * sway))
    return q


def forward_kinematics(
    local_q: Sequence[tuple],
    root_pos: tuple,
    offsets: Sequence[tuple],
) -> List[Joint]:
    world_q: List[tuple] = [QID] * NUM_JOINTS
    world_p: List[tuple] = [(0.0, 0.0, 0.0)] * NUM_JOINTS
    for i in range(NUM_JOINTS):
        p = PARENT[i]
        if p < 0:
            world_q[i] = qnorm(local_q[i])
            world_p[i] = root_pos
        else:
            world_q[i] = qnorm(qmul(world_q[p], local_q[i]))
            world_p[i] = vadd(world_p[p], qrot(world_q[p], offsets[i]))
    return [
        Joint(pos=world_p[i], quat=world_q[i], valid=True) for i in range(NUM_JOINTS)
    ]


# Timing model.  Numbers are nanoseconds in the "local common clock" (system monotonic)
# domain, matching DeviceDataTimestamp in timestamp.fbs.
CLOCK_BASE_NS = 1_842_000_000_000  # ~30 min of uptime
FRAME_PERIOD_NS = 16_666_667  # 60 Hz
PIPELINE_LATENCY_NS = 3_400_000  # 3.4 ms sample -> available
# The raw device clock is a *different* domain: arbitrary epoch plus a small rate error.
DEVICE_EPOCH_NS = 41_337_000_000_000
DEVICE_RATE_ERROR = 31e-6  # 31 ppm fast


def build_clean(
    duration: float = DURATION,
    rate_hz: float = 60.0,
    seed: int = 20260913,
    offsets_fn: Optional[Callable[[float], Sequence[tuple]]] = None,
    pos_noise_m: float = 0.0003,
) -> Recording:
    """Build the one clean recording every fixture is derived from.

    ``offsets_fn(t)`` overrides the bone offsets over time.
    """
    rng = random.Random(seed)
    n = round(duration * rate_hz)
    period_ns = round(1e9 / rate_hz)
    rec = Recording()

    for i in range(n):
        t = i / rate_hz
        offsets = offsets_fn(t) if offsets_fn is not None else REST

        root = REST_OFFSETS["PELVIS"]
        # small vertical bob + lateral sway so the root is not perfectly static
        root = (
            root[0] + 0.004 * math.sin(2.0 * math.pi * 0.13 * t + 0.7),
            root[1] + 0.005 * math.sin(2.0 * math.pi * 0.26 * t),
            root[2],
        )
        joints = forward_kinematics(local_rotations(t, rng), root, offsets)

        if pos_noise_m > 0.0:
            for jt in joints:
                jt.pos = (
                    jt.pos[0] + rng.gauss(0.0, pos_noise_m),
                    jt.pos[1] + rng.gauss(0.0, pos_noise_m),
                    jt.pos[2] + rng.gauss(0.0, pos_noise_m),
                )

        sample_ns = CLOCK_BASE_NS + i * period_ns + int(rng.gauss(0.0, 90_000))
        available_ns = (
            sample_ns + PIPELINE_LATENCY_NS + int(abs(rng.gauss(0.0, 180_000)))
        )
        device_ns = DEVICE_EPOCH_NS + int(
            (sample_ns - CLOCK_BASE_NS) * (1.0 + DEVICE_RATE_ERROR)
        )

        rec.frames.append(
            Frame(
                joints=joints,
                all_tracked=True,
                available_ns=available_ns,
                sample_ns=sample_ns,
                device_ns=device_ns,
                has_data=True,
            )
        )
    return rec
