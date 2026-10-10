# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scripted two-hand take, synthesised frame by frame.

Each window opens with the performer already in the pose; the move into it happens in
the gap before. Every placement is written for the right hand and mirrored for the left;
both hands share one body sway.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

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
    ("flat_on_table_open", 4.0),
    ("fist", 4.0),
    ("pinch_index", 3.0),
    ("pinch_middle", 3.0),
    ("pinch_ring", 3.0),
    ("pinch_little", 3.0),
    ("palms_together", 4.0),
    ("tips_together", 3.0),
    ("right_tip_roll", 6.0),
    ("left_tip_roll", 6.0),
    ("flat_on_table_close", 4.0),
)
FLAT = ("flat_on_table_open", "flat_on_table_close")
ROLLS = {"right_tip_roll": "right", "left_tip_roll": "left"}

SIDES = ("left", "right")
IDENTITY: hm.Quat = (0.0, 0.0, 0.0, 1.0)

# Right-hand placements in the stage (+Y up, performer facing -Z).
TABLE_Y = 0.75
# Joint centres sit this far above a surface the finger lies on, and this far from a
# pad pressed against another.
FINGER_HALF_M = 0.008
FLAT_WRIST: hm.Vec = (0.18, TABLE_Y + FINGER_HALF_M, -0.35)
HOME_WRIST: hm.Vec = (0.18, 1.05, -0.35)
# Fingers up, palm toward -X (the left hand); finger joints FINGER_HALF_M off x = 0.
PALMS_WRIST: hm.Vec = (FINGER_HALF_M + 0.001, 1.05, -0.30)
PALMS_Q = hm.quat_from_axes((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
# Fingers along -X toward the left hand, palm toward the performer, thumb up.
TIPS_Q = hm.quat_from_axes((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))
TIPS_AT: hm.Vec = (0.004, 1.25, -0.30)
MIDDLE_TIP = hm.FINGER_JOINTS["middle"][-1]
SWAY_PIVOT: hm.Vec = (0.0, 1.0, -0.35)


@dataclass
class Options:
    """What a fixture changes about the performance or the glove's model of the hand."""

    seed: int = 1
    fist: hm.Pose = hm.FIST
    flex_sign: float = 1.0
    # Radians added to one finger's MCP flexion in every pose.
    finger_bias: dict[str, float] = field(default_factory=dict)
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


def pose_for(label: str, options: Options) -> hm.Pose:
    if label == "fist":
        return options.fist
    if label.startswith("pinch_"):
        return hm.pinch(label.removeprefix("pinch_"))
    if label in FLAT:
        return hm.FLAT
    return hm.OPEN


def _tips_wrist() -> hm.Vec:
    tip_local = hm.solve(hm.OPEN)[MIDDLE_TIP][0]
    return hm.sub(TIPS_AT, hm.quat_rotate(TIPS_Q, tip_local))


TIPS_WRIST = _tips_wrist()


def placement_for(label: str) -> tuple[hm.Vec, hm.Quat]:
    """The right wrist's pose held through ``label``, before any roll."""
    if label in ("fist",) or label.startswith("pinch_"):
        return HOME_WRIST, IDENTITY
    if label == "palms_together":
        return PALMS_WRIST, PALMS_Q
    if label == "tips_together" or label in ROLLS:
        return TIPS_WRIST, TIPS_Q
    return FLAT_WRIST, IDENTITY


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


def _adjusted(pose: hm.Pose, options: Options) -> hm.Pose:
    flex = {}
    for f, v in pose.flex.items():
        bias = options.finger_bias.get(f, 0.0)
        v = (v[0] + bias, v[1], v[2])
        flex[f] = tuple(options.flex_sign * a for a in v)
    return replace(pose, flex=flex)


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


def _mirror_pose(p: hm.Vec, q: hm.Quat) -> tuple[hm.Vec, hm.Quat]:
    return (-p[0], p[1], p[2]), (q[0], -q[1], -q[2], q[3])


def _sway(t: float) -> tuple[hm.Quat, hm.Vec]:
    """The body's drift, shared by both hands: a small turn about the chest and a shift."""
    turn = hm.quat_axis_angle(
        (0.2, 1.0, 0.1), math.radians(1.5) * math.sin(2 * math.pi * 0.11 * t)
    )
    shift = (
        0.004 * math.sin(2 * math.pi * 0.21 * t),
        0.003 * math.sin(2 * math.pi * 0.17 * t + 1.0),
        0.004 * math.sin(2 * math.pi * 0.13 * t + 2.0),
    )
    return turn, shift


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
    tip_local = hm.solve(hm.OPEN)[MIDDLE_TIP][0]

    def pose_at(side: str, t_ns: int):
        """(finger pose, label, right-convention wrist pose, roll) at ``t_ns``."""
        current = _adjusted(hm.OPEN, options)
        place = (FLAT_WRIST, IDENTITY)
        label, roll = None, 0.0
        for w in windows:
            target = _adjusted(pose_for(w.label, options), options)
            target_place = placement_for(w.label)
            move_start = w.start_ns - int((GAP_S - 0.3 - options.arrival_delay_s) * 1e9)
            if t_ns < move_start:
                break
            u = (t_ns - move_start) / (options.move_s * 1e9)
            if u < 1:
                s = _smooth(u)
                current = hm.blend(current, target, s)
                place = (
                    hm.add(place[0], hm.scale(hm.sub(target_place[0], place[0]), s)),
                    hm.quat_nlerp(place[1], target_place[1], s),
                )
            else:
                current, place = target, target_place
            if w.start_ns <= t_ns < w.end_ns:
                label = w.label
                if ROLLS.get(w.label) == side:
                    u = (t_ns - w.start_ns) / (w.end_ns - w.start_ns)
                    roll = math.pi * (1 - math.cos(2 * math.pi * u)) / 2
        if roll:
            # Turn about the middle finger's own axis through its tip, which stays put.
            p, q = place
            tip = hm.add(p, hm.quat_rotate(q, tip_local))
            q = hm.quat_mul(q, hm.quat_axis_angle((0.0, 0.0, 1.0), roll))
            place = (hm.sub(tip, hm.quat_rotate(q, tip_local)), q)
        return current, label, place, roll

    out = []
    for tick in range(ticks):
        jitter = int(rng.uniform(-0.4e6, 0.4e6))
        sample_ns = T0_NS + int(tick * period_ns) + jitter
        t = tick / RATE_HZ
        turn, shift = _sway(t)
        pair = []
        for side in SIDES:
            pose, label, (p, q), roll = pose_at(side, sample_ns)
            if side == "left":
                p, q = _mirror_pose(p, q)
            local = hm.solve(_tremor(pose, t, phases[side]))
            if (side == "left") != (side in options.mirrored):
                local = hm.mirror(local)
            wrist_q = hm.quat_mul(_noise(rng), hm.quat_mul(turn, q))
            wrist_p = hm.add(
                hm.add(SWAY_PIVOT, hm.quat_rotate(turn, hm.sub(p, SWAY_PIVOT))),
                hm.add(shift, tuple(rng.gauss(0.0, 0.0003) for _ in range(3))),
            )
            joints = []
            for index, (lp, lq) in enumerate(local):
                joints.append(
                    Joint(
                        position=hm.add(wrist_p, hm.quat_rotate(wrist_q, lp)),
                        orientation=hm.quat_mul(wrist_q, lq),
                        radius=hm.radius(index),
                    )
                )
            pair.append(HandSample(side, sample_ns, t, label, joints, wrist_roll=roll))
        out.append((sample_ns, pair[0], pair[1]))
    return out, windows


def _noise(rng: random.Random) -> hm.Quat:
    axis = hm.unit((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
    return hm.quat_axis_angle(axis, math.radians(0.1) * rng.gauss(0, 1))
