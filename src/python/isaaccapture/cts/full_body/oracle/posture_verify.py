# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measures the posture fixtures back out of the written MCAPs and checks each against the injected magnitude.

Run:  ./generate.sh
"""

from __future__ import annotations

import json
import math
import os
import sys

import toolchain

toolchain.ensure()

HERE = os.path.dirname(os.path.abspath(__file__))

import posture_script as posture  # noqa: E402
from skeleton import J  # noqa: E402
from verify_fixtures import read  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "posture")
failures, notes = [], []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def times(fr):
    t0 = fr[0]["sample_ns"]
    return [(f["sample_ns"] - t0) / 1e9 for f in fr]


def window(fr, label, trim=0.5):
    """Frames inside a nominal label window, trimmed at both ends."""
    w = next(x for x in posture.step_windows() if x["label"] == label)
    t = times(fr)
    return [f for f, tt in zip(fr, t) if w["start_s"] + trim <= tt < w["end_s"] - trim]


def arm_down_angle(f, side):
    """Degrees the arm hangs below horizontal (0 = T-pose, negative = raised)."""
    sh = f["joints"][J[f"{side}_SHOULDER"]]["pos"]
    wr = f["joints"][J[f"{side}_WRIST"]]["pos"]
    v = tuple(a - b for a, b in zip(wr, sh))
    lateral = v[0] if side == "RIGHT" else -v[0]
    return math.degrees(math.atan2(-v[1], lateral))


def hip_flexion(f, side):
    """Degrees the thigh is swung forward from straight-down."""
    hp = f["joints"][J[f"{side}_HIP"]]["pos"]
    kn = f["joints"][J[f"{side}_KNEE"]]["pos"]
    v = tuple(a - b for a, b in zip(kn, hp))
    return math.degrees(math.atan2(-v[2], -v[1]))


def knee_flexion(f, side):
    """Degrees between the thigh and the shin (0 = straight leg)."""
    hp = f["joints"][J[f"{side}_HIP"]]["pos"]
    kn = f["joints"][J[f"{side}_KNEE"]]["pos"]
    an = f["joints"][J[f"{side}_ANKLE"]]["pos"]
    a = tuple(x - y for x, y in zip(kn, hp))
    b = tuple(x - y for x, y in zip(an, kn))
    na = math.sqrt(sum(c * c for c in a))
    nb = math.sqrt(sum(c * c for c in b))
    d = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def mean(xs):
    return sum(xs) / len(xs)


def corr(a, b):
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


# --------------------------------------------------------------------------------------
# Measurements: re-measure the injected quantities, sharing no code with the checker
# --------------------------------------------------------------------------------------


def m_drift_rad(fr):
    o = mean(
        [
            mean([arm_down_angle(f, "LEFT"), arm_down_angle(f, "RIGHT")])
            for f in window(fr, "t_pose_hold_open")
        ]
    )
    c = mean(
        [
            mean([arm_down_angle(f, "LEFT"), arm_down_angle(f, "RIGHT")])
            for f in window(fr, "t_pose_hold_close")
        ]
    )
    return math.radians(c - o)


def m_droop_deg(fr):
    fs = window(fr, "t_pose_hold_open") + window(fr, "t_pose_hold_close")
    return mean(
        [mean([arm_down_angle(f, "LEFT"), arm_down_angle(f, "RIGHT")]) for f in fs]
    )


def m_asym_deg(fr):
    fs = window(fr, "t_pose_hold_open") + window(fr, "t_pose_hold_close")
    return mean([arm_down_angle(f, "LEFT") - arm_down_angle(f, "RIGHT") for f in fs])


def m_crosstalk_deg(fr):
    left = max(hip_flexion(f, "RIGHT") for f in window(fr, "left_leg_raise", trim=0.1))
    right = max(hip_flexion(f, "LEFT") for f in window(fr, "right_leg_raise", trim=0.1))
    return max(left, right)


def m_arm_peak_deg(fr):
    fs = window(fr, "left_arm_raise", trim=0.1)
    return min(arm_down_angle(f, "LEFT") for f in fs)


def m_squat_reps(fr):
    fs = window(fr, "squat_x2", trim=0.0)
    y = [f["joints"][J["PELVIS"]]["pos"][1] for f in fs]
    top = max(y)
    half = len(fs) // 2
    return top - min(y[:half]), top - min(y[half:])


def m_knee_asym_deg(fr):
    fs = window(fr, "squat_x2", trim=0.0)
    k = max(range(len(fs)), key=lambda i: knee_flexion(fs[i], "LEFT"))
    return knee_flexion(fs[k], "RIGHT") - knee_flexion(fs[k], "LEFT")


def m_march_ankle_corr(fr):
    fs = window(fr, "march_in_place", trim=0.3)
    a = [f["joints"][J["LEFT_ANKLE"]]["pos"][1] for f in fs]
    b = [f["joints"][J["RIGHT_ANKLE"]]["pos"][1] for f in fs]
    return corr(a, b)


def m_march_step_intervals(fr):
    fs = window(fr, "march_in_place", trim=0.2)
    y = [f["joints"][J["LEFT_ANKLE"]]["pos"][1] for f in fs]
    ts = [(f["sample_ns"] - fs[0]["sample_ns"]) / 1e9 for f in fs]
    peaks = [
        i
        for i in range(1, len(y) - 1)
        if y[i] > y[i - 1]
        and y[i] >= y[i + 1]
        and y[i] - min(y) > 0.35 * (max(y) - min(y))
    ]
    return [ts[b] - ts[a] for a, b in zip(peaks, peaks[1:])]


LEG_JOINTS = {
    "LEFT": (J["LEFT_KNEE"], J["LEFT_ANKLE"], J["LEFT_FOOT"]),
    "RIGHT": (J["RIGHT_KNEE"], J["RIGHT_ANKLE"], J["RIGHT_FOOT"]),
}


def m_leg_motion(fr, label, side):
    """Path length travelled by one leg inside a labelled window."""
    fs = window(fr, label, trim=0.35)
    tot = 0.0
    for a, b in zip(fs, fs[1:]):
        for i in LEG_JOINTS[side]:
            tot += math.dist(a["joints"][i]["pos"], b["joints"][i]["pos"])
    return tot


def m_lean_deg(fr, label):
    fs = window(fr, label, trim=0.2)
    out = []
    for f in fs:
        p = f["joints"][J["PELVIS"]]["pos"]
        s = f["joints"][J["SPINE3"]]["pos"]
        v = tuple(a - b for a, b in zip(s, p))
        out.append(math.degrees(math.atan2(v[0], v[1])))
    return max(abs(x) for x in out)


# --------------------------------------------------------------------------------------


def main():
    index = json.load(open(os.path.join(HERE, "fixtures_index.json")))
    posture_entries = [e for e in index["fixtures"] if e.get("batch") == "posture"]
    check(
        len(posture_entries) == 28,
        f"expected 28 posture fixtures, index has {len(posture_entries)}",
    )
    data = {}
    for e in posture_entries:
        name = os.path.basename(e["filename"])[:-5]
        env, fr = read(os.path.join(HERE, e["filename"]))
        check(env["schema_name"] == "core.FullBodyPoseRecord", f"{name}: schema name")
        check(env["topic"] == "full_body/full_body", f"{name}: topic")
        check(env["message_encoding"] == "flatbuffer", f"{name}: message encoding")
        check(len(fr) == e["frames"], f"{name}: frame count")
        data[name] = (e, fr)
        print(f"  read {name} ({len(fr)} frames)")

    # ---- golden -------------------------------------------------------------------
    _, gf = data["posture_golden_full_script"]
    check(abs(m_droop_deg(gf)) < 1.0, f"golden: T-pose droop {m_droop_deg(gf):.2f} deg")
    check(abs(m_asym_deg(gf)) < 1.0, f"golden: asymmetry {m_asym_deg(gf):.2f} deg")
    check(abs(math.degrees(m_drift_rad(gf))) < 1.0, "golden: drift between T-poses")
    check(m_crosstalk_deg(gf) < 2.0, f"golden: crosstalk {m_crosstalk_deg(gf):.2f} deg")
    check(
        m_arm_peak_deg(gf) < -90.0,
        f"golden: arm peak only {m_arm_peak_deg(gf):.1f} deg",
    )
    r1, r2 = m_squat_reps(gf)
    check(
        abs(r2 - r1) / r1 < 0.08,
        f"golden: squat reps differ {100 * abs(r2 - r1) / r1:.1f}%",
    )
    check(
        abs(m_knee_asym_deg(gf)) < 3.0,
        f"golden: knee asym {m_knee_asym_deg(gf):.2f} deg",
    )
    check(
        m_march_ankle_corr(gf) < -0.35,
        f"golden: ankles not antiphase ({m_march_ankle_corr(gf):.2f})",
    )
    check(
        m_lean_deg(gf, "left_arm_raise") < 3.0, "golden: torso leans during arm raise"
    )
    notes.append(
        f"golden: squat depth {r1 * 100:.1f}/{r2 * 100:.1f} cm, arm peak "
        f"{m_arm_peak_deg(gf):.1f} deg, ankle corr {m_march_ankle_corr(gf):+.2f}, "
        f"march step interval {mean(m_march_step_intervals(gf)) * 1000:.0f} ms"
    )

    # ---- graded series: measured vs injected, and monotonic -------------------------
    series = {
        "cumulative_drift": (m_drift_rad, 1.0, 0.004),
        "tpose_arm_droop": (m_droop_deg, 1.0, 0.6),
        "tpose_left_right_asymmetry": (m_asym_deg, 1.0, 0.6),
        "contralateral_crosstalk": (m_crosstalk_deg, 1.0, 1.2),
    }
    for qty, (fn, scale, tol) in series.items():
        pts = []
        for e, fr in data.values():
            if (
                e.get("injected", {}).get("quantity") == qty
                and e["category"] == "graded"
            ):
                truth = e["injected"]["value"]
                got = fn(fr) * scale
                pts.append((truth, got))
                check(
                    abs(got - truth) <= tol,
                    f"{qty}: injected {truth}, measured {got:.4f} (tol {tol})",
                )
        pts.sort()
        check(all(b[1] > a[1] for a, b in zip(pts, pts[1:])), f"{qty}: not monotonic")
        notes.append(f"{qty}: " + ", ".join(f"{t}->{m:.3f}" for t, m in pts))

    # ---- device faults --------------------------------------------------------------
    _, d = data["posture_device_march_ankles_in_phase"]
    c = m_march_ankle_corr(d)
    check(
        c > 0.8,
        f"march_in_phase: ankle correlation {c:+.2f}, expected strongly positive",
    )
    notes.append(
        f"device march in-phase: ankle corr {c:+.2f} (golden {m_march_ankle_corr(gf):+.2f})"
    )

    _, d = data["posture_device_squat_reps_mismatched"]
    a, b = m_squat_reps(d)
    check(abs(b - a) / a > 0.2, f"squat reps mismatch only {100 * (b - a) / a:.1f}%")
    notes.append(
        f"device squat rep mismatch: {a * 100:.1f} cm vs {b * 100:.1f} cm ({100 * (b - a) / a:+.0f}%)"
    )

    _, d = data["posture_device_squat_knee_asymmetry"]
    k = m_knee_asym_deg(d)
    check(k > 15.0, f"knee asymmetry measured {k:.1f} deg, injected 22")
    notes.append(
        f"device knee asymmetry: {k:.1f} deg (injected 22, golden {m_knee_asym_deg(gf):+.2f})"
    )

    _, d = data["posture_device_arm_raise_saturates"]
    pk = m_arm_peak_deg(d)
    check(-32.0 < pk < -18.0, f"arm raise saturation peak {pk:.1f} deg, expected ~-25")
    notes.append(
        f"device arm saturation: peak {pk:.1f} deg (golden {m_arm_peak_deg(gf):.1f})"
    )

    # ---- performance faults ---------------------------------------------------------
    _, d = data["posture_perf_squat_too_shallow"]
    a, b = m_squat_reps(d)
    check(
        a < 0.3 * m_squat_reps(gf)[0],
        f"shallow squat depth {a * 100:.1f} cm not shallow enough",
    )
    ka = m_knee_asym_deg(d)
    notes.append(
        f"perf shallow squat: depth {a * 100:.1f} cm (golden {m_squat_reps(gf)[0] * 100:.1f}), "
        f"knee asym {ka:+.2f} deg -- symmetric, so it does not look like the device fault"
    )

    _, d = data["posture_perf_march_slow_irregular"]
    iv = m_march_step_intervals(d)
    gi = m_march_step_intervals(gf)
    spread = (max(iv) - min(iv)) if len(iv) > 1 else 0.0
    gspread = (max(gi) - min(gi)) if len(gi) > 1 else 0.0
    check(
        spread > 3 * max(gspread, 0.01),
        f"irregular march spread {spread:.3f}s vs golden {gspread:.3f}s",
    )
    c2 = m_march_ankle_corr(d)
    check(
        c2 < 0.0,
        "irregular march lost its antiphase, which would mimic the device fault",
    )
    notes.append(
        f"perf irregular march: interval spread {spread * 1000:.0f} ms (golden {gspread * 1000:.0f}), "
        f"ankle corr {c2:+.2f} -- still antiphase, so distinguishable from the device fault"
    )

    _, d = data["posture_perf_lean_during_arm_raise"]
    lean = m_lean_deg(d, "left_arm_raise")
    check(lean > 8.0, f"lean measured {lean:.1f} deg, injected 12")
    # The world-frame arm angle changes with the lean; correct for the torso tilt.
    corrected = m_arm_peak_deg(d) - lean
    check(
        abs(corrected - m_arm_peak_deg(gf)) < 4.0,
        f"lean fixture changed the torso-relative arm ROM ({corrected:.1f} vs {m_arm_peak_deg(gf):.1f})",
    )
    notes.append(
        f"perf lean: torso {lean:.1f} deg (golden {m_lean_deg(gf, 'left_arm_raise'):.1f}); "
        f"arm peak {m_arm_peak_deg(d):.1f} deg world / {corrected:.1f} deg torso-relative "
        f"(golden {m_arm_peak_deg(gf):.1f}) -- world-frame ROM alone would misread this "
        f"as a device fault"
    )

    _, d = data["posture_perf_left_leg_raise_skipped"]
    e_skip = m_leg_motion(d, "left_leg_raise", "LEFT")
    e_gold = m_leg_motion(gf, "left_leg_raise", "LEFT")
    peak_hip = max(
        hip_flexion(f, "LEFT") for f in window(d, "left_leg_raise", trim=0.35)
    )
    gold_hip = max(
        hip_flexion(f, "LEFT") for f in window(gf, "left_leg_raise", trim=0.35)
    )
    check(
        peak_hip < 5.0, f"skipped step still flexes the left hip to {peak_hip:.1f} deg"
    )
    check(gold_hip > 60.0, f"golden left leg raise only reaches {gold_hip:.1f} deg")
    check(
        e_skip < 0.2 * e_gold,
        f"skipped step still has {e_skip:.3f} m of leg motion (golden {e_gold:.3f})",
    )
    check(
        m_leg_motion(d, "right_leg_raise", "RIGHT")
        > 0.5 * m_leg_motion(gf, "right_leg_raise", "RIGHT"),
        "skip fixture also removed the right leg raise",
    )
    notes.append(
        f"perf skipped step: peak left hip flexion {peak_hip:.1f} deg vs {gold_hip:.1f} deg "
        f"in the golden; leg path length {e_skip:.3f} m vs {e_gold:.3f} m (the residual "
        f"is the 0.3 mm/frame position-noise floor); the right leg raise is untouched"
    )

    _, d = data["posture_perf_steps_out_of_order"]
    # the window labelled left_leg_raise actually contains an arm raise
    arm_in_leg_window = min(
        arm_down_angle(f, "RIGHT") for f in window(d, "left_leg_raise", trim=0.1)
    )
    leg_in_leg_window = max(
        hip_flexion(f, "LEFT") for f in window(d, "left_leg_raise", trim=0.1)
    )
    check(
        arm_in_leg_window < -60.0,
        "out-of-order: no arm raise inside the left_leg_raise window",
    )
    check(
        leg_in_leg_window < 20.0,
        "out-of-order: a leg raise still happens in its labelled window",
    )
    notes.append(
        f"perf out of order: left_leg_raise window contains a right-arm raise "
        f"(peak {arm_in_leg_window:.0f} deg) and no hip flexion ({leg_in_leg_window:.1f} deg)"
    )

    # ---- segmentation -----------------------------------------------------------------
    check(
        not os.path.exists(os.path.join(FIX, "posture_seg_labels_absent.labels.json")),
        "labels_absent fixture still has a sidecar",
    )
    lab = json.load(open(os.path.join(FIX, "posture_seg_labels_offset.labels.json")))
    base = json.load(open(os.path.join(FIX, "posture_golden_full_script.labels.json")))
    off = [
        (a["start_ns"] - b["start_ns"]) / 1e9
        for a, b in zip(lab["steps"], base["steps"])
    ]
    check(
        all(abs(o - 0.5) < 1e-6 for o in off),
        f"label offset is {set(off)}, expected 0.5 s",
    )

    bad = json.load(
        open(os.path.join(FIX, "posture_seg_labels_overlap_and_gaps.labels.json"))
    )
    ov = sum(
        1 for a, b in zip(bad["steps"], bad["steps"][1:]) if b["start_ns"] < a["end_ns"]
    )
    gp = sum(
        1 for a, b in zip(bad["steps"], bad["steps"][1:]) if b["start_ns"] > a["end_ns"]
    )
    check(ov >= 3 and gp >= 3, f"corrupt labels: {ov} overlaps, {gp} gaps")
    clean_ov = sum(
        1
        for a, b in zip(base["steps"], base["steps"][1:])
        if b["start_ns"] != a["end_ns"]
    )
    check(clean_ov == 0, "golden labels are not a clean partition")
    notes.append(
        f"segmentation: corrupt sidecar has {ov} overlapping and {gp} gapped boundaries; "
        f"golden sidecar partitions the run exactly"
    )

    print("\nmeasurements:")
    for n in notes:
        print("  " + n)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  ! " + f)
        sys.exit(1)
    print("\nALL POSTURE FIXTURES VERIFIED")


if __name__ == "__main__":
    main()
