# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads every generated fixture back with the `mcap` library and asserts it matches the index.

Run:  ./generate.sh
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import toolchain

toolchain.ensure()

HERE = os.path.dirname(os.path.abspath(__file__))

from mcap.reader import NonSeekingReader, make_reader  # noqa: E402

from core.FullBodyPoseRecord import FullBodyPoseRecord  # noqa: E402
from core.Point import Point  # noqa: E402
from core.Pose import Pose  # noqa: E402
from core.Quaternion import Quaternion  # noqa: E402
from skeleton import J, NUM_JOINTS, PARENT  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
with open(toolchain.BFBS_PATH, "rb") as _fh:
    GOLDEN_BFBS = _fh.read()

failures = []
notes = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


def read(path):
    """-> (envelope dict, list of decoded frame dicts)"""
    frames = []
    env = {}
    with open(path, "rb") as fh:
        env["profile"] = make_reader(fh).get_header().profile
    with open(path, "rb") as fh:
        # Read in file order; make_reader() re-sorts by log time and hides out-of-order timestamps.
        reader = NonSeekingReader(fh)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            env.setdefault("schema_name", schema.name)
            env.setdefault("schema_encoding", schema.encoding)
            env.setdefault("schema_data", schema.data)
            env.setdefault("topic", channel.topic)
            env.setdefault("message_encoding", channel.message_encoding)
            rec = FullBodyPoseRecord.GetRootAs(bytearray(message.data), 0)
            ts = rec.Timestamp()
            data = rec.Data()
            f = {
                "log_time": message.log_time,
                "publish_time": message.publish_time,
                "sequence": message.sequence,
                "available_ns": ts.AvailableTimeLocalCommonClock(),
                "sample_ns": ts.SampleTimeLocalCommonClock(),
                "device_ns": ts.SampleTimeRawDeviceClock(),
                "has_data": data is not None,
                "joints": None,
                "all_tracked": False,
            }
            if data is not None:
                f["all_tracked"] = data.AllJointPosesTracked()
                bj = data.Joints()
                if bj is not None:
                    js = []
                    for i in range(NUM_JOINTS):
                        jp = bj.Joints(i)
                        pose = jp.Pose(Pose())
                        p = pose.Position(Point())
                        o = pose.Orientation(Quaternion())
                        js.append(
                            {
                                "pos": (p.X(), p.Y(), p.Z()),
                                "quat": (o.X(), o.Y(), o.Z(), o.W()),
                                "valid": jp.IsValid(),
                            }
                        )
                    f["joints"] = js
            frames.append(f)
    return env, frames


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def qn(q):
    return math.sqrt(sum(c * c for c in q))


def bone_lengths(frame):
    out = {}
    for i in range(NUM_JOINTS):
        p = PARENT[i]
        if p >= 0:
            out[i] = dist(frame["joints"][i]["pos"], frame["joints"][p]["pos"])
    return out


# --------------------------------------------------------------------------------------
# Per-fixture assertions: each returns None, or appends to `failures`.
# --------------------------------------------------------------------------------------


def v_golden(env, fr, name):
    js = [f for f in fr if f["joints"]]
    check(len(js) == len(fr), f"{name}: some frames lack joints")
    check(
        all(all(j["valid"] for j in f["joints"]) for f in js),
        f"{name}: invalid joints present",
    )
    check(all(f["all_tracked"] for f in js), f"{name}: all_joint_poses_tracked not set")
    # unit quaternions
    worst_q = max(abs(qn(j["quat"]) - 1.0) for f in js for j in f["joints"])
    check(worst_q < 1e-5, f"{name}: quaternion norm off by {worst_q}")
    # Y-up: head above pelvis, ankles below, in Y
    f0 = js[0]
    check(
        f0["joints"][J["HEAD"]]["pos"][1]
        > f0["joints"][J["PELVIS"]]["pos"][1]
        > f0["joints"][J["LEFT_ANKLE"]]["pos"][1],
        f"{name}: not Y-up",
    )
    # right is +X
    check(
        f0["joints"][J["RIGHT_SHOULDER"]]["pos"][0]
        > 0
        > f0["joints"][J["LEFT_SHOULDER"]]["pos"][0],
        f"{name}: +X is not the subject's right",
    )
    # metres / stature
    stature = f0["joints"][J["HEAD"]]["pos"][1] - min(j["pos"][1] for j in f0["joints"])
    check(1.2 < stature < 1.9, f"{name}: implausible head-above-floor {stature:.3f} m")
    # bone lengths constant
    base = bone_lengths(js[0])
    worst = 0.0
    for f in js:
        bl = bone_lengths(f)
        for i, L in bl.items():
            worst = max(worst, abs(L - base[i]))
    check(worst < 0.004, f"{name}: bone length varies by {worst * 1000:.2f} mm")
    # forearm shorter than upper arm (human)
    check(
        base[J["LEFT_WRIST"]] < base[J["LEFT_ELBOW"]],
        f"{name}: forearm longer than upper arm",
    )
    # timestamps
    check(
        all(f["sample_ns"] < f["available_ns"] for f in fr),
        f"{name}: available before sample",
    )
    check(
        all(b["sample_ns"] > a["sample_ns"] for a, b in zip(fr, fr[1:])),
        f"{name}: non-monotonic",
    )
    check(
        all(f["device_ns"] != f["sample_ns"] for f in fr),
        f"{name}: device clock copies common clock",
    )
    dts = [b["sample_ns"] - a["sample_ns"] for a, b in zip(fr, fr[1:])]
    spread = max(dts) - min(dts)
    check(
        spread < 2_000_000,
        f"{name}: interval spread {spread / 1e6:.2f} ms too large for a clean file",
    )
    check(
        all(math.isfinite(c) for f in js for j in f["joints"] for c in j["pos"]),
        f"{name}: non-finite position",
    )
    notes.append(
        f"  {name}: stature {stature:.3f} m, "
        f"upper arm {base[J['LEFT_ELBOW']] * 100:.1f} cm, forearm {base[J['LEFT_WRIST']] * 100:.1f} cm, "
        f"bone-length drift {worst * 1000:.2f} mm, median dt {sorted(dts)[len(dts) // 2] / 1e6:.3f} ms"
    )


def v_benign_invalid_zero(env, fr, name):
    tgt = [J["LEFT_COLLAR"], J["RIGHT_COLLAR"], J["LEFT_FOOT"], J["RIGHT_FOOT"]]
    ok = all(
        not f["joints"][i]["valid"]
        and f["joints"][i]["pos"] == (0.0, 0.0, 0.0)
        and f["joints"][i]["quat"] == (0.0, 0.0, 0.0, 0.0)
        for f in fr
        for i in tgt
    )
    check(ok, f"{name}: expected 4 permanently-invalid all-zero joints")
    others = [i for i in range(NUM_JOINTS) if i not in tgt]
    check(
        all(f["joints"][i]["valid"] for f in fr for i in others),
        f"{name}: other joints not valid",
    )
    check(
        all(abs(qn(f["joints"][i]["quat"]) - 1) < 1e-5 for f in fr for i in others),
        f"{name}: non-unit quats on the valid joints",
    )


def v_benign_backfill(env, fr, name):
    pairs = [
        (J["LEFT_HAND"], J["LEFT_WRIST"]),
        (J["RIGHT_HAND"], J["RIGHT_WRIST"]),
        (J["LEFT_FOOT"], J["LEFT_ANKLE"]),
        (J["RIGHT_FOOT"], J["RIGHT_ANKLE"]),
    ]
    ok = all(
        f["joints"][c]["pos"] == f["joints"][s]["pos"]
        and f["joints"][c]["quat"] == f["joints"][s]["quat"]
        and f["joints"][c]["valid"]
        for f in fr
        for c, s in pairs
    )
    check(ok, f"{name}: endpoints are not exact copies of their parents")
    zero_bones = [i for i, L in bone_lengths(fr[0]).items() if L == 0.0]
    check(
        sorted(zero_bones) == sorted(c for c, _ in pairs),
        f"{name}: expected exactly the 4 back-filled bones to be zero length, got {zero_bones}",
    )


def v_benign_dropout(env, fr, name):
    chain = [J["LEFT_SHOULDER"], J["LEFT_ELBOW"], J["LEFT_WRIST"], J["LEFT_HAND"]]
    bad = [i for i, f in enumerate(fr) if not f["joints"][chain[0]]["valid"]]
    check(bad, f"{name}: no dropout found")
    check(
        bad == list(range(bad[0], bad[-1] + 1)),
        f"{name}: dropout is not one contiguous window",
    )
    dur = (fr[bad[-1]]["sample_ns"] - fr[bad[0]]["sample_ns"]) / 1e9
    check(0.8 < dur < 1.2, f"{name}: dropout lasts {dur:.2f} s, expected ~1 s")
    check(
        fr[0]["joints"][chain[0]]["valid"] and fr[-1]["joints"][chain[0]]["valid"],
        f"{name}: no recovery",
    )
    notes.append(
        f"  {name}: dropout frames {bad[0]}..{bad[-1]} ({dur:.2f} s), recovers"
    )


def v_z_up(env, fr, name):
    f0 = fr[0]
    check(
        f0["joints"][J["HEAD"]]["pos"][2] > f0["joints"][J["PELVIS"]]["pos"][2],
        f"{name}: head is not above pelvis in Z",
    )
    check(
        abs(f0["joints"][J["HEAD"]]["pos"][1] - f0["joints"][J["PELVIS"]]["pos"][1])
        < 0.05,
        f"{name}: Y still carries the vertical",
    )
    base = bone_lengths(f0)
    check(
        abs(base[J["LEFT_ELBOW"]] - 0.285) < 0.01, f"{name}: bone lengths changed too"
    )


def v_cm(env, fr, name):
    f0 = fr[0]
    stature = f0["joints"][J["HEAD"]]["pos"][1] - min(j["pos"][1] for j in f0["joints"])
    check(
        150 < stature < 200, f"{name}: stature reads {stature:.1f}, expected ~175 (cm)"
    )
    check(
        abs(qn(f0["joints"][0]["quat"]) - 1) < 1e-5,
        f"{name}: quaternions were scaled too",
    )


def v_lr_swap(env, fr, name):
    f0 = fr[0]
    check(
        f0["joints"][J["RIGHT_SHOULDER"]]["pos"][0]
        < 0
        < f0["joints"][J["LEFT_SHOULDER"]]["pos"][0],
        f"{name}: left/right not swapped",
    )
    # Detectable only dynamically: during the RIGHT-arm raise the joint labelled LEFT_WRIST goes overhead.
    t = [(f["sample_ns"] - fr[0]["sample_ns"]) / 1e9 for f in fr]
    k = min(range(len(fr)), key=lambda i: abs(t[i] - 7.5))
    g = fr[k]["joints"]
    check(
        g[J["LEFT_WRIST"]]["pos"][1]
        > g[J["HEAD"]]["pos"][1]
        > g[J["RIGHT_WRIST"]]["pos"][1],
        f"{name}: the raised limb is not mislabelled at t=7.5 s",
    )
    check(
        abs(bone_lengths(f0)[J["LEFT_ELBOW"]] - 0.285) < 0.01,
        f"{name}: bone lengths should be untouched by a pure relabel",
    )


def v_wxyz(env, fr, name):
    # still unit norm ...
    check(
        all(abs(qn(j["quat"]) - 1) < 1e-5 for j in fr[0]["joints"]),
        f"{name}: norms broken",
    )
    # ... but w is no longer ~1 at rest for the pelvis (it moved into x)
    q = fr[0]["joints"][J["PELVIS"]]["quat"]
    check(
        abs(q[0]) > 0.9 and abs(q[3]) < 0.2,
        f"{name}: pelvis quat {q} does not look like w,x,y,z written into x,y,z,w",
    )


def v_mirror(env, fr, name):
    f0 = fr[0]
    check(f0["joints"][J["RIGHT_SHOULDER"]]["pos"][0] < 0, f"{name}: X not negated")
    base = bone_lengths(f0)
    check(
        abs(base[J["LEFT_ELBOW"]] - 0.285) < 0.01, f"{name}: bone lengths not preserved"
    )
    # handedness: the triad (hips x spine) flips sign
    p = f0["joints"]

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    right = sub(p[J["RIGHT_HIP"]]["pos"], p[J["LEFT_HIP"]]["pos"])
    up = sub(p[J["SPINE3"]]["pos"], p[J["PELVIS"]]["pos"])
    fwd_from_pos = cross(right, up)
    # forward implied by the pelvis orientation, rotating local -Z
    x, y, z, w = p[J["PELVIS"]]["quat"]

    def qrot(q, v):
        qx, qy, qz, qw = q
        tx = 2 * (qy * v[2] - qz * v[1])
        ty = 2 * (qz * v[0] - qx * v[2])
        tz = 2 * (qx * v[1] - qy * v[0])
        return (
            v[0] + qw * tx + (qy * tz - qz * ty),
            v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx),
        )

    fwd_from_quat = qrot((x, y, z, w), (0, 0, -1))
    # On clean data cross(right, up) = +Z, opposite the orientation-derived forward; a mirror flips it.
    check(
        dot(fwd_from_pos, fwd_from_quat) > 0,
        f"{name}: chirality not inverted (triple product {dot(fwd_from_pos, fwd_from_quat):.4f})",
    )


def v_half_rotated(env, fr, name):
    f0 = fr[0]
    # positions rotated 90 deg about Y: the shoulder line now runs along Z
    d = tuple(
        a - b
        for a, b in zip(
            f0["joints"][J["RIGHT_SHOULDER"]]["pos"],
            f0["joints"][J["LEFT_SHOULDER"]]["pos"],
        )
    )
    check(
        abs(d[2]) > 0.3 and abs(d[0]) < 0.05,
        f"{name}: positions not rotated about Y, delta {d}",
    )
    # orientations untouched: pelvis is still near identity
    q = f0["joints"][J["PELVIS"]]["quat"]
    check(abs(q[3]) > 0.99, f"{name}: orientations were rotated too (pelvis {q})")


def v_permute(env, fr, name):
    f0 = fr[0]
    check(
        f0["joints"][J["NECK"]]["pos"][1] > f0["joints"][J["HEAD"]]["pos"][1],
        f"{name}: NECK/HEAD not swapped",
    )
    check(
        f0["joints"][J["LEFT_KNEE"]]["pos"][1]
        < f0["joints"][J["LEFT_ANKLE"]]["pos"][1],
        f"{name}: LEFT_KNEE/LEFT_ANKLE not swapped",
    )


def v_rubber(env, fr, name):
    a = bone_lengths(fr[0])[J["LEFT_ELBOW"]]
    b = bone_lengths(fr[-1])[J["LEFT_ELBOW"]]
    check(b / a > 1.18, f"{name}: upper arm grew only {100 * (b / a - 1):.1f}%")
    notes.append(
        f"  {name}: upper arm {a * 100:.1f} cm -> {b * 100:.1f} cm ({100 * (b / a - 1):.1f}%)"
    )


def v_giant(env, fr, name):
    f0 = fr[0]
    stature = f0["joints"][J["HEAD"]]["pos"][1] - min(j["pos"][1] for j in f0["joints"])
    base = bone_lengths(f0)
    ratio = base[J["LEFT_WRIST"]] / base[J["LEFT_ELBOW"]]
    check(stature > 2.6, f"{name}: stature only {stature:.2f} m")
    check(ratio > 2.0, f"{name}: forearm/upper-arm ratio only {ratio:.2f}")
    notes.append(f"  {name}: stature {stature:.2f} m, forearm/upper-arm {ratio:.2f}")


def v_teleport(env, fr, name):
    worst = 0.0
    for a, b in zip(fr, fr[1:]):
        dt = (b["sample_ns"] - a["sample_ns"]) / 1e9
        d = dist(a["joints"][J["PELVIS"]]["pos"], b["joints"][J["PELVIS"]]["pos"])
        worst = max(worst, d / dt)
    check(worst > 100.0, f"{name}: peak speed only {worst:.1f} m/s")
    notes.append(f"  {name}: peak pelvis speed {worst:.0f} m/s")


def v_nan_inf(env, fr, name):
    nan = sum(1 for f in fr for j in f["joints"] for c in j["pos"] if math.isnan(c))
    inf = sum(1 for f in fr for j in f["joints"] for c in j["pos"] if math.isinf(c))
    check(nan > 0 and inf > 0, f"{name}: nan={nan} inf={inf}, expected both")
    notes.append(f"  {name}: {nan} NaN and {inf} Inf components")


def v_non_unit(env, fr, name):
    norms = defaultdict(set)
    for f in fr:
        for i, j in enumerate(f["joints"]):
            if j["valid"]:
                norms[i].add(round(qn(j["quat"]), 3))
    off = {i for i, s in norms.items() if any(abs(n - 1) > 0.05 for n in s)}
    check(
        off == {J["LEFT_ELBOW"], J["RIGHT_KNEE"], J["SPINE2"]},
        f"{name}: non-unit joints are {off}",
    )


def v_no_joints(env, fr, name):
    missing = [f for f in fr if f["has_data"] and f["joints"] is None]
    check(
        0.05 < len(missing) / len(fr) < 0.14,
        f"{name}: {len(missing)}/{len(fr)} records missing `joints`",
    )
    check(
        all(f["has_data"] for f in fr),
        f"{name}: some records lost the whole payload instead",
    )
    notes.append(f"  {name}: {len(missing)}/{len(fr)} records have data but no joints")


def v_zero_valid(env, fr, name):
    tgt = [
        J["LEFT_FOOT"],
        J["RIGHT_FOOT"],
        J["LEFT_HAND"],
        J["RIGHT_HAND"],
        J["LEFT_COLLAR"],
        J["RIGHT_COLLAR"],
    ]
    ok = all(
        f["joints"][i]["valid"] and f["joints"][i]["pos"] == (0.0, 0.0, 0.0)
        for f in fr
        for i in tgt
    )
    check(ok, f"{name}: expected 6 valid joints at the origin")


def v_non_monotonic(env, fr, name):
    back = [
        i for i, (a, b) in enumerate(zip(fr, fr[1:])) if b["sample_ns"] < a["sample_ns"]
    ]
    check(len(back) >= 10, f"{name}: only {len(back)} backward steps")
    notes.append(f"  {name}: {len(back)} backward sample-time steps")


def v_device_copy(env, fr, name):
    check(
        all(f["device_ns"] == f["sample_ns"] for f in fr),
        f"{name}: device clock still distinct",
    )


def v_avail_before(env, fr, name):
    check(
        all(f["available_ns"] < f["sample_ns"] for f in fr),
        f"{name}: available not before sample",
    )
    check(
        all(f["log_time"] == f["available_ns"] for f in fr),
        f"{name}: logTime is not the available time",
    )


def v_null_payload(env, fr, name):
    nulls = [f for f in fr if not f["has_data"]]
    frac = len(nulls) / len(fr)
    check(0.6 < frac < 0.75, f"{name}: null payload fraction {frac:.2f}")
    notes.append(f"  {name}: {frac * 100:.0f}% of records carry a timestamp only")


def v_flag_lies(env, fr, name):
    bad = [J["LEFT_FOOT"], J["RIGHT_FOOT"], J["LEFT_HAND"], J["RIGHT_HAND"]]
    ok = all(
        f["all_tracked"] and not any(f["joints"][i]["valid"] for i in bad) for f in fr
    )
    check(ok, f"{name}: flag/per-joint contradiction not present on every frame")


def v_jitter(env, fr, name):
    dts = [(b["sample_ns"] - a["sample_ns"]) / 1e6 for a, b in zip(fr, fr[1:])]
    spread = max(dts) - min(dts)
    check(spread > 10.0, f"{name}: interval spread only {spread:.2f} ms")
    notes.append(f"  {name}: intervals {min(dts):.1f}-{max(dts):.1f} ms (nominal 16.7)")


def v_drops(env, fr, name):
    dts = [(b["sample_ns"] - a["sample_ns"]) / 1e6 for a, b in zip(fr, fr[1:])]
    gaps = [d for d in dts if d > 200.0]
    check(len(gaps) == 4, f"{name}: found {len(gaps)} gaps > 200 ms, expected 4")
    notes.append(f"  {name}: gaps {['%.0f ms' % g for g in gaps]}")


def v_degrade(env, fr, name):
    def cov(f):
        return sum(1 for j in f["joints"] if j["valid"]) / NUM_JOINTS

    head = sum(cov(f) for f in fr[:30]) / 30
    tail = sum(cov(f) for f in fr[-30:]) / 30
    check(head > 0.98 and tail < 0.45, f"{name}: coverage {head:.2f} -> {tail:.2f}")
    notes.append(f"  {name}: validity coverage {head * 100:.0f}% -> {tail * 100:.0f}%")


VERIFIERS = {
    "golden_scripted_sequence": v_golden,
    "golden_tpose_hold": v_golden,
    "benign_invalid_joints_zero_pose": v_benign_invalid_zero,
    "benign_backfilled_endpoints": v_benign_backfill,
    "benign_dropout_recovery": v_benign_dropout,
    "defect_z_up": v_z_up,
    "defect_centimetre_units": v_cm,
    "defect_left_right_swapped": v_lr_swap,
    "defect_quaternion_wxyz_order": v_wxyz,
    "defect_mirrored_handedness": v_mirror,
    "defect_positions_rotated_orientations_not": v_half_rotated,
    "defect_joint_index_permutation": v_permute,
    "defect_bone_length_drift": v_rubber,
    "defect_implausible_proportions": v_giant,
    "defect_pose_teleport": v_teleport,
    "defect_nan_inf_positions": v_nan_inf,
    "defect_non_unit_quaternions": v_non_unit,
    "defect_joints_field_absent": v_no_joints,
    "defect_zero_positions_on_valid_joints": v_zero_valid,
    "defect_non_monotonic_timestamps": v_non_monotonic,
    "defect_device_clock_copies_common": v_device_copy,
    "defect_available_before_sample": v_avail_before,
    "defect_null_payload_majority": v_null_payload,
    "defect_all_tracked_flag_inconsistent": v_flag_lies,
    "defect_irregular_frame_intervals": v_jitter,
    "defect_dropped_frames": v_drops,
    "defect_validity_degradation": v_degrade,
}


def main():
    index = json.load(open(os.path.join(HERE, "fixtures_index.json")))
    # Envelope batch only.
    entries = {
        os.path.basename(e["filename"])[:-5]: e
        for e in index["fixtures"]
        if e.get("batch") == "envelope"
    }
    check(set(entries) == set(VERIFIERS), "index and verifier set disagree")

    for name in sorted(entries):
        path = os.path.join(FIX, name + ".mcap")
        env, fr = read(path)
        # Envelope.
        check(
            env["schema_name"] == "core.FullBodyPoseRecord",
            f"{name}: schema name {env['schema_name']}",
        )
        check(
            env["schema_encoding"] == "flatbuffer",
            f"{name}: schema encoding {env['schema_encoding']}",
        )
        check(
            env["schema_data"] == GOLDEN_BFBS,
            f"{name}: embedded bfbs differs from the repo golden",
        )
        check(env["topic"] == "full_body/full_body", f"{name}: topic {env['topic']}")
        check(
            env["message_encoding"] == "flatbuffer",
            f"{name}: message encoding {env['message_encoding']}",
        )
        check(env["profile"] == "teleop", f"{name}: profile {env['profile']}")
        check(
            all(f["log_time"] == f["available_ns"] for f in fr),
            f"{name}: logTime != available time",
        )
        check(
            all(f["publish_time"] == f["log_time"] for f in fr),
            f"{name}: publishTime != logTime",
        )
        check(
            [f["sequence"] for f in fr] == list(range(len(fr))),
            f"{name}: sequence not 0..n-1",
        )
        check(len(fr) == entries[name]["frames"], f"{name}: frame count != index")
        VERIFIERS[name](env, fr, name)
        print(f"  checked {name} ({len(fr)} frames)")

    print("\nmeasurements:")
    for n in notes:
        print(n)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  ! " + f)
        sys.exit(1)
    print("\nALL FIXTURES VERIFIED")


if __name__ == "__main__":
    main()
