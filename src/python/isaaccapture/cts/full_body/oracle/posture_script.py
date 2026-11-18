# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Posture motion script (11 steps, ~41 s) and its injectable parameters.

Every departure from a correct performance is a named field on :class:`PostureParams`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from skeleton import (
    CLOCK_BASE_NS,
    DEVICE_EPOCH_NS,
    DEVICE_RATE_ERROR,
    J,
    NUM_JOINTS,
    PIPELINE_LATENCY_NS,
    QID,
    REST,
    REST_OFFSETS,
    Frame,
    Recording,
    forward_kinematics,
    qaxis,
    qmul,
    smoothstep,
)

RATE_HZ = 50.0  # 60 Hz would exceed the repo's 2000 KiB file limit

# (label, duration seconds) -- the 11-step script.
STEPS: List[Tuple[str, float]] = [
    ("clap", 1.0),  # 0  oracle only
    ("a_pose_still", 3.0),  # 1
    ("t_pose_hold_open", 3.0),  # 2
    ("neutral_stance", 2.0),  # 3
    ("left_arm_raise", 4.0),  # 4
    ("right_arm_raise", 4.0),  # 5
    ("left_leg_raise", 4.0),  # 6
    ("right_leg_raise", 4.0),  # 7
    ("squat_x2", 8.0),  # 8
    ("march_in_place", 5.0),  # 9
    ("t_pose_hold_close", 3.0),  # 10
]
DURATION = sum(d for _, d in STEPS)  # 41.0 s

# Labels of the still windows.
STILL_STEPS = {
    "a_pose_still",
    "t_pose_hold_open",
    "neutral_stance",
    "t_pose_hold_close",
}

# Shoulder "down angle" held during each step, degrees (0 = arms horizontal = T-pose).
SD_NEUTRAL = 75.0
SD_APOSE = 52.0
SD_CLAP = 58.0

ARM_OVERHEAD_DEG = -95.0  # negative = above horizontal
LEG_RAISE_HIP_DEG = 72.0
LEG_RAISE_KNEE_DEG = -82.0
SQUAT_HIP_DEG = 55.0
MARCH_HIP_DEG = 45.0
MARCH_CADENCE_HZ = 0.9  # full left+right cycles per second (1.8 steps/s)


@dataclass
class PostureParams:
    """Injected magnitudes; all default to a clean performance."""

    # --- graded series -----------------------------------------------------------
    drift_rad: float = 0.0  # closing T-pose shoulder offset vs opening
    crosstalk_deg: float = 0.0  # opposite hip motion during a single-leg raise
    tpose_droop_deg: float = 0.0  # both arms below horizontal in both T-poses
    asymmetry_deg: float = 0.0  # LEFT arm lower than RIGHT in both T-poses

    # --- device faults ------------------------------------------------------------
    march_in_phase: bool = False  # ankles move together instead of antiphase
    squat_rep2_delta: float = 0.0  # rep 2 depth as a fraction of rep 1 (0 = equal)
    squat_knee_asym_deg: float = 0.0  # extra RIGHT knee flexion at full depth
    arm_raise_saturate_deg: Optional[float] = None  # shoulder angle stops here

    # --- performance faults -------------------------------------------------------
    squat_depth_scale: float = 1.0  # < 1 = too shallow to measure
    march_irregular: bool = False  # wandering cadence and amplitude
    lean_deg: float = 0.0  # torso lean while raising an arm
    skip_steps: Sequence[str] = ()  # labelled step never performed
    motion_order: Optional[Sequence[str]] = None  # actual order of the performed steps

    seed: int = 424242


# --------------------------------------------------------------------------------------
# Timeline helpers
# --------------------------------------------------------------------------------------


def step_windows() -> List[Dict]:
    """Nominal label windows: what the operator intended, in seconds from t0."""
    out, t = [], 0.0
    for i, (label, dur) in enumerate(STEPS):
        out.append({"index": i, "label": label, "start_s": t, "end_s": t + dur})
        t += dur
    return out


def _motion_windows(p: PostureParams) -> List[Dict]:
    """What is actually performed: the nominal windows unless ``motion_order`` reorders the steps."""
    win = step_windows()
    if p.motion_order is None:
        return win
    order = list(p.motion_order)
    assert sorted(order) == sorted(label for label, _ in STEPS), (
        "motion_order must be a permutation"
    )
    durations = dict(STEPS)
    out, t = [], 0.0
    for label in order:
        d = durations[label]
        out.append({"index": len(out), "label": label, "start_s": t, "end_s": t + d})
        t += d
    return out


def _bump(u: float) -> float:
    """Smooth 0 -> 1 -> 0 over u in [0, 1]."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * u))


def _hold_profile(u: float, rise: float = 0.40, hold: float = 0.20) -> float:
    """0 -> 1 over the first ``rise`` of the window, hold, then back to 0."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    if u < rise:
        return smoothstep(u / rise)
    if u < rise + hold:
        return 1.0
    return 1.0 - smoothstep((u - rise - hold) / (1.0 - rise - hold))


def _piecewise(t: float, segs: List[Tuple[float, float, float]], blend: float) -> float:
    """Value from ``segs`` [(start, end, value)], blended smoothly into each segment."""
    for i, (s, e, v) in enumerate(segs):
        if t < e or i == len(segs) - 1:
            if i == 0 or blend <= 0.0:
                return v
            prev = segs[i - 1][2]
            return prev + (v - prev) * smoothstep((t - s) / blend)
    return segs[-1][2]


def _shoulder_down_segments(p: PostureParams) -> List[Tuple[float, float, float]]:
    droop = p.tpose_droop_deg
    drift = math.degrees(p.drift_rad)
    per_label = {
        "clap": SD_CLAP,
        "a_pose_still": SD_APOSE,
        "t_pose_hold_open": droop,
        "neutral_stance": SD_NEUTRAL,
        "left_arm_raise": SD_NEUTRAL,
        "right_arm_raise": SD_NEUTRAL,
        "left_leg_raise": SD_NEUTRAL,
        "right_leg_raise": SD_NEUTRAL,
        "squat_x2": SD_NEUTRAL,
        "march_in_place": SD_NEUTRAL,
        "t_pose_hold_close": droop + drift,
    }
    return [
        (w["start_s"], w["end_s"], per_label[w["label"]]) for w in _motion_windows(p)
    ]


# --------------------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------------------


def posture_local_rotations(
    t: float, p: PostureParams, rng: random.Random, march_plan: Dict
) -> List[tuple]:
    q = [QID] * NUM_JOINTS
    windows = _motion_windows(p)

    # Baseline arm carriage, smoothly blended across step boundaries.
    sd = _piecewise(t, _shoulder_down_segments(p), blend=0.55)
    sd_left, sd_right = sd, sd

    # T-pose asymmetry: left arm systematically lower than right, in both T-poses.
    if p.asymmetry_deg:
        tpose = [w for w in windows if w["label"].startswith("t_pose")]
        for w in tpose:
            if w["start_s"] - 0.55 <= t < w["end_s"]:
                sd_left += p.asymmetry_deg * smoothstep(
                    (t - (w["start_s"] - 0.55)) / 0.55
                )

    # Active step.
    cur = None
    for w in windows:
        if w["start_s"] <= t < w["end_s"]:
            cur = w
            break
    if cur is None:
        cur = windows[-1]
    label = cur["label"]
    u = (t - cur["start_s"]) / (cur["end_s"] - cur["start_s"])
    skipped = label in p.skip_steps

    hipL = hipR = kneeL = kneeR = ankL = ankR = 0.0
    spine_lean = 0.0
    pelvis_z = 0.0
    elbowL = elbowR = 0.0

    if label == "clap" and not skipped:
        # Two sharp bilateral claps: forearms swing to the midline and back.
        base, amp = 62.0, 34.0
        phase = 2.0 * math.pi * 2.0 * u  # two claps in the 1 s window
        s = 0.5 * (1.0 - math.cos(phase))
        elbowL = -(base + amp * s)
        elbowR = base + amp * s

    elif label in ("left_arm_raise", "right_arm_raise") and not skipped:
        prof = _hold_profile(u)
        target = ARM_OVERHEAD_DEG
        if p.arm_raise_saturate_deg is not None:
            target = max(target, p.arm_raise_saturate_deg)
        if label == "left_arm_raise":
            sd_left += (target - sd_left) * prof
        else:
            sd_right += (target - sd_right) * prof
        pelvis_z = p.lean_deg * prof

    elif label in ("left_leg_raise", "right_leg_raise") and not skipped:
        prof = _hold_profile(u)
        if label == "left_leg_raise":
            hipL, kneeL = LEG_RAISE_HIP_DEG * prof, LEG_RAISE_KNEE_DEG * prof
            ankL = 14.0 * prof
            hipR = p.crosstalk_deg * prof  # contralateral cross-talk
        else:
            hipR, kneeR = LEG_RAISE_HIP_DEG * prof, LEG_RAISE_KNEE_DEG * prof
            ankR = 14.0 * prof
            hipL = p.crosstalk_deg * prof

    elif label == "squat_x2" and not skipped:
        depth_max = SQUAT_HIP_DEG * p.squat_depth_scale
        rep1 = _bump((u - 0.06) / 0.44) * depth_max
        rep2 = _bump((u - 0.52) / 0.44) * depth_max * (1.0 + p.squat_rep2_delta)
        phi = rep1 + rep2
        hipL = hipR = phi
        kneeL = -2.0 * phi
        kneeR = -2.0 * phi - p.squat_knee_asym_deg * (phi / max(1e-6, SQUAT_HIP_DEG))
        ankL = ankR = phi * 0.95
        spine_lean = -0.42 * phi

    elif label == "march_in_place" and not skipped:
        f = march_plan["cadence"](u)
        amp = march_plan["amplitude"](u)
        phase_l = 2.0 * math.pi * f * (t - cur["start_s"])
        phase_r = phase_l if p.march_in_phase else phase_l + math.pi
        lift_l = amp * max(0.0, math.sin(phase_l))
        lift_r = amp * max(0.0, math.sin(phase_r))
        hipL, kneeL, ankL = lift_l, -1.6 * lift_l, 0.5 * lift_l
        hipR, kneeR, ankR = lift_r, -1.6 * lift_r, 0.5 * lift_r

    # Assemble; the right shoulder takes the opposite sign.
    q[J["LEFT_SHOULDER"]] = qaxis((0, 0, 1), math.radians(sd_left))
    q[J["RIGHT_SHOULDER"]] = qaxis((0, 0, 1), -math.radians(sd_right))
    if elbowL or elbowR:
        q[J["LEFT_ELBOW"]] = qaxis((0, 1, 0), math.radians(elbowL))
        q[J["RIGHT_ELBOW"]] = qaxis((0, 1, 0), math.radians(elbowR))

    q[J["LEFT_HIP"]] = qaxis((1, 0, 0), math.radians(hipL))
    q[J["RIGHT_HIP"]] = qaxis((1, 0, 0), math.radians(hipR))
    q[J["LEFT_KNEE"]] = qaxis((1, 0, 0), math.radians(kneeL))
    q[J["RIGHT_KNEE"]] = qaxis((1, 0, 0), math.radians(kneeR))
    q[J["LEFT_ANKLE"]] = qaxis((1, 0, 0), math.radians(ankL))
    q[J["RIGHT_ANKLE"]] = qaxis((1, 0, 0), math.radians(ankR))

    # Breathing / sway.
    breath = math.sin(2.0 * math.pi * 0.26 * t)
    sway = math.sin(2.0 * math.pi * 0.13 * t + 0.7)
    q[J["SPINE1"]] = qmul(
        qaxis((1, 0, 0), math.radians(spine_lean)),
        qaxis((0, 0, 1), math.radians(0.6 * sway)),
    )
    q[J["SPINE2"]] = qaxis((1, 0, 0), math.radians(0.9 * breath))
    q[J["NECK"]] = qaxis((0, 1, 0), math.radians(1.4 * sway))
    q[J["PELVIS"]] = qaxis((0, 0, 1), math.radians(pelvis_z))
    return q


def _march_plan(p: PostureParams) -> Dict:
    if not p.march_irregular:
        return {
            "cadence": lambda u: MARCH_CADENCE_HZ,
            "amplitude": lambda u: MARCH_HIP_DEG,
        }
    rng = random.Random(p.seed ^ 0x9E37)
    knots = [
        (i / 8.0, rng.uniform(0.35, 0.75), rng.uniform(14.0, 38.0)) for i in range(9)
    ]

    def interp(u, idx):
        u = max(0.0, min(0.999, u))
        k = int(u * 8)
        a, b = knots[k], knots[min(8, k + 1)]
        w = (u - a[0]) * 8.0
        return a[idx] + (b[idx] - a[idx]) * w

    return {"cadence": lambda u: interp(u, 1), "amplitude": lambda u: interp(u, 2)}


REST_ANKLE_Y = (
    REST_OFFSETS["PELVIS"][1]
    + REST_OFFSETS["LEFT_HIP"][1]
    + REST_OFFSETS["LEFT_KNEE"][1]
    + REST_OFFSETS["LEFT_ANKLE"][1]
)


DEFAULT_POSTURE = PostureParams()


def build_posture(
    p: PostureParams = DEFAULT_POSTURE, rate_hz: float = RATE_HZ
) -> Recording:
    rng = random.Random(p.seed)
    ground_ref = replace(p, march_in_phase=False)
    plan = _march_plan(p)
    n = round(DURATION * rate_hz)
    period_ns = round(1e9 / rate_hz)
    rec = Recording()

    for i in range(n):
        t = i / rate_hz
        root = (
            0.004 * math.sin(2.0 * math.pi * 0.13 * t + 0.7),
            REST_OFFSETS["PELVIS"][1] + 0.005 * math.sin(2.0 * math.pi * 0.26 * t),
            0.0,
        )
        joints = forward_kinematics(
            posture_local_rotations(t, p, rng, plan), root, REST
        )

        # Ground the support foot: shift everything so the lower ankle is back at its standing height.
        # An in-phase march is grounded from the correctly-phased pose.
        ground_src = joints
        if p.march_in_phase:
            ground_src = forward_kinematics(
                posture_local_rotations(t, ground_ref, rng, plan), root, REST
            )
        low = min(
            ground_src[J["LEFT_ANKLE"]].pos[1], ground_src[J["RIGHT_ANKLE"]].pos[1]
        )
        dy = REST_ANKLE_Y - low
        for jt in joints:
            jt.pos = (jt.pos[0], jt.pos[1] + dy, jt.pos[2])
            jt.pos = tuple(c + rng.gauss(0.0, 0.0003) for c in jt.pos)

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


def sidecar_labels(
    rec: Recording, offset_s: float = 0.0, corrupt_windows: bool = False
) -> Dict:
    """Step labels in the record timestamp clock domain, as a JSON sidecar to the MCAP."""
    t0 = (
        rec.frames[0]["sample_ns"]
        if isinstance(rec.frames[0], dict)
        else rec.frames[0].sample_ns
    )
    steps = []
    wins = step_windows()
    for k, w in enumerate(wins):
        s, e = w["start_s"] + offset_s, w["end_s"] + offset_s
        if corrupt_windows:
            # Alternate overlap and gap on different boundaries; both on one would cancel.
            if k % 4 == 0:
                e += 0.6  # overlaps the next window
            elif k % 4 == 2:
                s += 0.8  # leaves a gap after the previous window
        steps.append(
            {
                "index": w["index"],
                "label": w["label"],
                "start_ns": t0 + int(s * 1e9),
                "end_ns": t0 + int(e * 1e9),
                "start_s_from_first_sample": round(s, 4),
                "end_s_from_first_sample": round(e, 4),
                "is_still_window": w["label"] in STILL_STEPS,
            }
        )
    return {
        "note": "Step labels for the recording beside this file.",
        "clock_domain": "sample_time_local_common_clock (system monotonic, nanoseconds)",
        "nominal_rate_hz": RATE_HZ,
        "steps": steps,
    }
