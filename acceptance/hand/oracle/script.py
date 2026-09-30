# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scripted two-hand take, synthesised frame by frame.

Each window opens with the performer already in the pose: the move into it happens in
the cue-and-countdown gap before, as the timed capture paces it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import hand_model as hm

RATE_HZ = 60.0
T0_NS = 1_000_000_000_000
DEVICE_CLOCK_OFFSET_NS = 7_300_000_000_000
PREROLL_S = 1.5
GAP_S = 2.0
MOVE_S = 0.7
TAIL_S = 1.0

# (label, hold seconds), the order the capture plays them.
STEPS: tuple[tuple[str, float], ...] = (
    ("open_hand_open", 4.0),
    ("fist", 4.0),
    ("right_grasps_left_fist", 4.0),
    ("left_grasps_right_fist", 4.0),
    ("pinch_index", 3.0),
    ("pinch_middle", 3.0),
    ("pinch_ring", 3.0),
    ("pinch_little", 3.0),
    ("wrist_rotate", 6.0),
    ("open_hand_close", 4.0),
)

SIDES = ("left", "right")
WRIST_HOME = {"left": (-0.18, 1.05, -0.35), "right": (0.18, 1.05, -0.35)}


@dataclass
class Options:
    """What a fixture changes about the performance or the glove's model of the hand.
    Faults in what the plugin publishes are applied afterwards, in ``generate_fixtures``."""

    seed: int = 1
    fist: hm.Pose = hm.FIST
    flex_sign: float = 1.0
    arrival_delay_s: float = 0.0
    move_s: float = MOVE_S
    # Build this side from the other side's geometry.
    mirrored: frozenset[str] = frozenset()
    steps: tuple[tuple[str, float], ...] = STEPS


@dataclass(frozen=True)
class Window:
    index: int
    label: str
    start_ns: int
    end_ns: int


@dataclass
class Joint:
    position: hm.Vec
    orientation: hm.Quat
    valid: bool = True
    radius: float = 0.01


@dataclass
class HandSample:
    side: str
    sample_ns: int
    t: float
    label: str | None
    joints: list[Joint] | None
    wrist_roll: float = 0.0


def pose_for(label: str, side: str, options: Options) -> hm.Pose:
    if label in ("open_hand_open", "open_hand_close", "wrist_rotate"):
        return hm.OPEN
    if label == "fist":
        return options.fist
    if label.endswith("_fist"):
        grasper = label.split("_", 1)[0]
        return hm.GRASP if side == grasper else options.fist
    if label.startswith("pinch_"):
        return hm.pinch(label.removeprefix("pinch_"))
    raise KeyError(label)


def schedule(options: Options) -> tuple[list[Window], float]:
    windows, t = [], PREROLL_S
    for index, (label, hold) in enumerate(options.steps):
        t += GAP_S
        windows.append(
            Window(index, label, T0_NS + int(t * 1e9), T0_NS + int((t + hold) * 1e9))
        )
        t += hold
    return windows, t + TAIL_S


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def _signed(pose: hm.Pose, sign: float) -> hm.Pose:
    if sign == 1.0:
        return pose
    return replace(
        pose, flex={f: tuple(sign * a for a in v) for f, v in pose.flex.items()}
    )


def _tremor(pose: hm.Pose, t: float, rng_phase: dict[str, float]) -> hm.Pose:
    wobble = math.radians(1.0)
    flex = {
        f: tuple(
            a + wobble * math.sin(2 * math.pi * (0.8 + 0.1 * k) * t + rng_phase[f])
            for k, a in enumerate(v)
        )
        for f, v in pose.flex.items()
    }
    return replace(pose, flex=flex)


def synthesise(
    options: Options,
) -> tuple[list[tuple[int, HandSample, HandSample]], list[Window]]:
    """Every tick as (sample_ns, left, right), plus the label windows."""
    windows, total_s = schedule(options)
    rng = random.Random(options.seed)
    phases = {
        side: {f: rng.uniform(0, 2 * math.pi) for f in hm.FINGERS} for side in SIDES
    }
    period_ns = 1e9 / RATE_HZ
    ticks = int(total_s * RATE_HZ)

    def pose_at(side: str, t_ns: int) -> tuple[hm.Pose, str | None, float]:
        current, label, roll = _signed(hm.OPEN, options.flex_sign), None, 0.0
        for w in windows:
            target = _signed(pose_for(w.label, side, options), options.flex_sign)
            move_start = w.start_ns - int((GAP_S - 0.3 - options.arrival_delay_s) * 1e9)
            if t_ns < move_start:
                break
            u = (t_ns - move_start) / (options.move_s * 1e9)
            current = hm.blend(current, target, _smooth(u)) if u < 1 else target
            if w.start_ns <= t_ns < w.end_ns:
                label = w.label
                if w.label == "wrist_rotate":
                    u = (t_ns - w.start_ns) / (w.end_ns - w.start_ns)
                    roll = math.pi * (1 - math.cos(2 * math.pi * u)) / 2
        return current, label, roll

    out = []
    for tick in range(ticks):
        jitter = int(rng.uniform(-0.4e6, 0.4e6))
        sample_ns = T0_NS + int(tick * period_ns) + jitter
        t = tick / RATE_HZ
        pair = []
        for side in SIDES:
            pose, label, roll = pose_at(side, sample_ns)
            local = hm.solve(_tremor(pose, t, phases[side]))
            if (side == "left") != (side in options.mirrored):
                local = hm.mirror(local)
            wrist_q = _wrist_orientation(side, t, roll, rng)
            wrist_p = _wrist_position(side, t, rng)
            joints = []
            for index, (p, q) in enumerate(local):
                joints.append(
                    Joint(
                        position=hm.add(wrist_p, hm.quat_rotate(wrist_q, p)),
                        orientation=hm.quat_mul(wrist_q, q),
                        radius=hm.radius(index),
                    )
                )
            pair.append(HandSample(side, sample_ns, t, label, joints, wrist_roll=roll))
        out.append((sample_ns, pair[0], pair[1]))
    return out, windows


def _wrist_position(side: str, t: float, rng: random.Random) -> hm.Vec:
    home = WRIST_HOME[side]
    sway = (
        0.004 * math.sin(2 * math.pi * 0.21 * t),
        0.003 * math.sin(2 * math.pi * 0.17 * t + 1.0),
        0.004 * math.sin(2 * math.pi * 0.13 * t + 2.0),
    )
    noise = tuple(rng.gauss(0.0, 0.0003) for _ in range(3))
    return hm.add(hm.add(home, sway), noise)


def _wrist_orientation(side: str, t: float, roll: float, rng: random.Random) -> hm.Quat:
    """Palm down, fingers along -Z, turned about the forearm by ``roll``."""
    sway = hm.quat_axis_angle(
        (0.3, 1.0, 0.2), math.radians(3.0) * math.sin(2 * math.pi * 0.11 * t)
    )
    sense = 1.0 if side == "right" else -1.0
    turn = hm.quat_axis_angle((0.0, 0.0, 1.0), sense * roll)
    axis = hm.unit((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    noise = hm.quat_axis_angle(axis, math.radians(0.1) * rng.gauss(0, 1))
    return hm.quat_mul(noise, hm.quat_mul(sway, turn))
