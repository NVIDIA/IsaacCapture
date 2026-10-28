# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the synthetic MCAP fixture set for the full-body CTS checker.

Each fixture is a named transformation applied to a copy of the clean recording
(``skeleton.build_clean``), added with the ``fixture`` decorator:

    @fixture("defect_my_new_fault", DEFECT, "one line", verdict="fail",
             check="name_of_check_expected_to_catch_it")
    def _my_new_fault(rec):
        ...mutate rec in place...

Run:  ./.venv/bin/python generate_fixtures.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import skeleton as sk
from mcap_io import load_bfbs, write_mcap
from skeleton import (
    DURATION,
    J,
    MIRROR,
    NUM_JOINTS,
    REST,
    REST_OFFSETS,
    Joint,
    Recording,
    qmul,
    qaxis,
    qrot,
    zero_joint,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "fixtures")

GOLDEN, BENIGN, DEFECT = "golden", "benign", "defect"


@dataclass
class Fixture:
    name: str
    category: str
    description: str
    verdict: str
    check: Optional[str]
    build: Callable[[], Recording]


REGISTRY: List[Fixture] = []


def fixture(name, category, description, verdict, check=None, source=None):
    """Register a fixture.  ``source`` defaults to a fresh copy of the clean recording;
    the decorated function mutates it in place (or returns a replacement)."""

    def deco(fn):
        def build():
            rec = source() if source is not None else clean()
            out = fn(rec)
            return rec if out is None else out

        REGISTRY.append(Fixture(name, category, description, verdict, check, build))
        return fn

    return deco


_CLEAN_CACHE: Dict[str, Recording] = {}


def clean() -> Recording:
    if "clean" not in _CLEAN_CACHE:
        _CLEAN_CACHE["clean"] = sk.build_clean()
    return copy.deepcopy(_CLEAN_CACHE["clean"])


def each_joint(rec: Recording):
    """(frame, joint_index, joint) for every real joint in the recording."""
    for f in rec.frames:
        if f.joints is None or not f.has_data:
            continue
        for i, j in enumerate(f.joints):
            yield f, i, j


def frame_time(rec: Recording, idx: int) -> float:
    return (rec.frames[idx].sample_ns - rec.frames[0].sample_ns) / 1e9


# ======================================================================================
# A. GOLDEN
# ======================================================================================


@fixture(
    "golden_scripted_sequence",
    GOLDEN,
    "14 s @ 60 Hz: A-pose still, T-pose hold, single right-arm raise, single left-leg "
    "raise, return to T-pose. Y-up right-handed metres, all joints valid.",
    verdict="pass",
)
def _golden_scripted(rec):
    return None


@fixture(
    "golden_tpose_hold",
    GOLDEN,
    "6 s @ 60 Hz static T-pose hold with breathing sway only -- a minimal baseline for "
    "still-window statistics and noise-floor estimation.",
    verdict="pass",
    source=lambda: sk.build_clean(duration=6.0, seed=771),
)
def _golden_tpose(rec):
    # Hold the T-pose for the whole file.
    held = sk.build_clean(duration=6.0, seed=771)
    # Re-run the animation with time clamped into the open T-pose window.
    rng = random.Random(9001)
    for i, f in enumerate(held.frames):
        t = (
            sk.T_TPOSE_IN + 0.2 + (i / 60.0) * 0.35
        )  # slow drift through the hold window
        joints = sk.forward_kinematics(
            sk.local_rotations(t, rng), sk.REST_OFFSETS["PELVIS"], REST
        )
        for jt in joints:
            jt.pos = tuple(c + rng.gauss(0.0, 0.0003) for c in jt.pos)
        f.joints = joints
    return held


# ======================================================================================
# B. BENIGN -- unusual but correct
# ======================================================================================


@fixture(
    "benign_invalid_joints_zero_pose",
    BENIGN,
    "Four joints the vendor does not provide (collars, feet) are permanently "
    "is_valid=false carrying an all-zero Point and all-zero Quaternion -- byte-for-byte "
    "what make_invalid_body_joint_pose() emits in the Noitom plugin. A quaternion-norm "
    "check applied unconditionally would wrongly fail this valid output.",
    verdict="pass",
)
def _benign_invalid_zero(rec):
    absent = [J["LEFT_COLLAR"], J["RIGHT_COLLAR"], J["LEFT_FOOT"], J["RIGHT_FOOT"]]
    for f in rec.frames:
        for i in absent:
            f.joints[i] = zero_joint()
        f.all_tracked = False


@fixture(
    "benign_backfilled_endpoints",
    BENIGN,
    "Hands and feet are back-filled from the wrists and ankles and marked valid, as the "
    "Noitom plugin does for a skeleton with no hand/foot joints. Consequence: those four "
    "bones have a constantly zero length. Legitimate -- reportable as 'derived, not "
    "measured', but not a fault.",
    verdict="pass",
)
def _benign_backfill(rec):
    pairs = [
        (J["LEFT_HAND"], J["LEFT_WRIST"]),
        (J["RIGHT_HAND"], J["RIGHT_WRIST"]),
        (J["LEFT_FOOT"], J["LEFT_ANKLE"]),
        (J["RIGHT_FOOT"], J["RIGHT_ANKLE"]),
    ]
    for f in rec.frames:
        for child, src in pairs:
            f.joints[child] = Joint(
                pos=f.joints[src].pos, quat=f.joints[src].quat, valid=True
            )


@fixture(
    "benign_dropout_recovery",
    BENIGN,
    "The whole left-arm chain goes is_valid=false (zero pose) for ~1 s mid-run and then "
    "recovers cleanly; all_joint_poses_tracked follows. A transient occlusion, not a "
    "broken integration.",
    verdict="pass",
)
def _benign_dropout(rec):
    chain = [J["LEFT_SHOULDER"], J["LEFT_ELBOW"], J["LEFT_WRIST"], J["LEFT_HAND"]]
    for idx, f in enumerate(rec.frames):
        if 6.0 <= frame_time(rec, idx) < 7.0:
            for i in chain:
                f.joints[i] = zero_joint()
            f.all_tracked = False


# ======================================================================================
# C. DEFECTS -- geometry / frame
# ======================================================================================


@fixture(
    "defect_z_up",
    DEFECT,
    "Whole rig expressed Z-up instead of Y-up (right-handed throughout): positions and "
    "orientations both rotated +90 deg about X.",
    verdict="fail",
    check="coordinate_frame.up_axis",
)
def _z_up(rec):
    q = qaxis((1, 0, 0), math.pi / 2)
    for _f, _i, j in each_joint(rec):
        j.pos = qrot(q, j.pos)
        j.quat = qmul(q, j.quat)


@fixture(
    "defect_centimetre_units",
    DEFECT,
    "Positions written in centimetres; orientations untouched. Stature reads ~175 units.",
    verdict="fail",
    check="units.position_scale_metres",
)
def _cm(rec):
    for _f, _i, j in each_joint(rec):
        j.pos = tuple(c * 100.0 for c in j.pos)


@fixture(
    "defect_left_right_swapped",
    DEFECT,
    "Every left/right joint pair has its data written into the other's index. The "
    "skeleton is geometrically intact but the labelling is inverted.",
    verdict="fail",
    check="skeleton.left_right_labelling",
)
def _lr_swap(rec):
    for f in rec.frames:
        src = list(f.joints)
        for i in range(NUM_JOINTS):
            f.joints[i] = copy.deepcopy(src[MIRROR[i]])


@fixture(
    "defect_quaternion_wxyz_order",
    DEFECT,
    "Quaternion components serialised w,x,y,z into the x,y,z,w fields of core.Quaternion. "
    "Norm still 1, so only an orientation-vs-position consistency test catches it.",
    verdict="fail",
    check="quaternion.component_order",
)
def _wxyz(rec):
    for _f, _i, j in each_joint(rec):
        x, y, z, w = j.quat
        j.quat = (w, x, y, z)


@fixture(
    "defect_mirrored_handedness",
    DEFECT,
    "X negated on positions with the matching mirror conjugation on orientations: a "
    "left-handed coordinate frame. Bone lengths and the joint tree survive, chirality "
    "does not.",
    verdict="fail",
    check="coordinate_frame.handedness",
)
def _mirror(rec):
    for _f, _i, j in each_joint(rec):
        j.pos = (-j.pos[0], j.pos[1], j.pos[2])
        x, y, z, w = j.quat
        j.quat = (x, -y, -z, w)


@fixture(
    "defect_positions_rotated_orientations_not",
    DEFECT,
    "Positions rotated 90 deg about Y into another frame while orientations were left in "
    "the original one -- the classic half-finished frame conversion.",
    verdict="fail",
    check="consistency.position_orientation_same_frame",
)
def _half_rotated(rec):
    q = qaxis((0, 1, 0), math.pi / 2)
    for _f, _i, j in each_joint(rec):
        j.pos = qrot(q, j.pos)


@fixture(
    "defect_joint_index_permutation",
    DEFECT,
    "Three joint pairs written to each other's indices (SPINE1<->SPINE2, NECK<->HEAD, "
    "LEFT_KNEE<->LEFT_ANKLE): the parent/child tree no longer matches the XR_BD_body_tracking layout.",
    verdict="fail",
    check="skeleton.joint_index_assignment",
)
def _permute(rec):
    perm = list(range(NUM_JOINTS))
    for a, b in [("SPINE1", "SPINE2"), ("NECK", "HEAD"), ("LEFT_KNEE", "LEFT_ANKLE")]:
        perm[J[a]], perm[J[b]] = perm[J[b]], perm[J[a]]
    for f in rec.frames:
        src = list(f.joints)
        for i in range(NUM_JOINTS):
            f.joints[i] = copy.deepcopy(src[perm[i]])


@fixture(
    "defect_bone_length_drift",
    DEFECT,
    "Every bone grows smoothly by 22% over the run -- a 'rubber' skeleton whose segment "
    "lengths are not invariant.",
    verdict="fail",
    check="skeleton.bone_length_constancy",
    source=lambda: sk.build_clean(
        offsets_fn=lambda t: [
            tuple(c * (1.0 + 0.22 * (t / DURATION)) for c in off) for off in REST
        ]
    ),
)
def _rubber(rec):
    return None


@fixture(
    "defect_implausible_proportions",
    DEFECT,
    "Anthropometrically impossible skeleton: forearms 2.3x the upper arm and a ~3.0 m "
    "stature. Internally consistent, just not a human.",
    verdict="fail",
    check="skeleton.anthropometric_plausibility",
    source=lambda: sk.build_clean(offsets_fn=lambda t: _giant_offsets()),
)
def _giant(rec):
    return None


def _giant_offsets():
    off = dict(REST_OFFSETS)
    # ~3.0 m stature
    off["PELVIS"] = (0.0, 1.70, 0.0)
    for k in ("LEFT_KNEE", "RIGHT_KNEE"):
        off[k] = (0.0, -0.78, 0.0)
    for k in ("LEFT_ANKLE", "RIGHT_ANKLE"):
        off[k] = (0.0, -0.76, 0.0)
    for k in ("SPINE1", "SPINE2", "SPINE3", "NECK", "HEAD"):
        x, y, z = off[k]
        off[k] = (x, y * 1.9, z)
    # forearm >> upper arm
    off["LEFT_ELBOW"], off["RIGHT_ELBOW"] = (-0.20, 0.0, 0.0), (0.20, 0.0, 0.0)
    off["LEFT_WRIST"], off["RIGHT_WRIST"] = (-0.46, 0.0, 0.0), (0.46, 0.0, 0.0)
    return [off[n] for n in sk.JOINT_NAMES]


@fixture(
    "defect_pose_teleport",
    DEFECT,
    "At t=5.5 s (inside a still window) the whole skeleton jumps 2.6 m in one 16.7 ms "
    "frame and jumps back three frames later: ~155 m/s, far past any human movement.",
    verdict="fail",
    check="continuity.max_joint_velocity",
)
def _teleport(rec):
    for idx, f in enumerate(rec.frames):
        if 5.5 <= frame_time(rec, idx) < 5.55:
            for j in f.joints:
                j.pos = (j.pos[0] + 2.6, j.pos[1] + 0.4, j.pos[2] - 1.1)


# ======================================================================================
# C. DEFECTS -- record / envelope
# ======================================================================================


@fixture(
    "defect_nan_inf_positions",
    DEFECT,
    "NaN and +/-infinity injected into the positions of a handful of valid joints on "
    "about 4% of frames.",
    verdict="fail",
    check="values.finite",
)
def _nan_inf(rec):
    rng = random.Random(4242)
    bad = [J["RIGHT_WRIST"], J["LEFT_ANKLE"], J["HEAD"]]
    for idx, f in enumerate(rec.frames):
        if rng.random() < 0.04:
            j = f.joints[bad[idx % len(bad)]]
            choice = idx % 3
            if choice == 0:
                j.pos = (float("nan"), j.pos[1], j.pos[2])
            elif choice == 1:
                j.pos = (j.pos[0], float("inf"), j.pos[2])
            else:
                j.pos = (j.pos[0], j.pos[1], float("-inf"))


@fixture(
    "defect_non_unit_quaternions",
    DEFECT,
    "Orientations on several is_valid=true joints are scaled off the unit sphere "
    "(norms ~1.8 and ~0.35) -- unnormalised output, not the benign all-zero invalid case.",
    verdict="fail",
    check="quaternion.unit_norm_on_valid_joints",
)
def _non_unit(rec):
    scaled = {J["LEFT_ELBOW"]: 1.8, J["RIGHT_KNEE"]: 0.35, J["SPINE2"]: 1.27}
    for _f, i, j in each_joint(rec):
        s = scaled.get(i)
        if s is not None:
            j.quat = tuple(c * s for c in j.quat)


@fixture(
    "defect_joints_field_absent",
    DEFECT,
    "About 9% of records carry a FullBodyPose table whose `joints` struct field is "
    "absent, though the schema comment states all fields are present whenever the table "
    "is. Distinct from a null payload: the data table itself is there.",
    verdict="fail",
    check="schema.required_field_present.joints",
)
def _no_joints(rec):
    rng = random.Random(7)
    for f in rec.frames:
        if rng.random() < 0.09:
            f.joints = None
            f.all_tracked = False


@fixture(
    "defect_zero_positions_on_valid_joints",
    DEFECT,
    "Six joints report position (0,0,0) with an identity quaternion while is_valid stays "
    "true -- an uninitialised slot reported as good data.",
    verdict="fail",
    check="values.zero_pose_on_valid_joint",
)
def _zero_valid(rec):
    targets = [
        J["LEFT_FOOT"],
        J["RIGHT_FOOT"],
        J["LEFT_HAND"],
        J["RIGHT_HAND"],
        J["LEFT_COLLAR"],
        J["RIGHT_COLLAR"],
    ]
    for f in rec.frames:
        for i in targets:
            f.joints[i] = Joint(
                pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0), valid=True
            )


@fixture(
    "defect_non_monotonic_timestamps",
    DEFECT,
    "Twelve frames have sample/available timestamps (and therefore MCAP logTime) that "
    "step ~40 ms backwards relative to their predecessor.",
    verdict="fail",
    check="timestamps.monotonic",
)
def _non_monotonic(rec):
    rng = random.Random(31)
    picks = sorted(rng.sample(range(30, len(rec.frames) - 5), 12))
    for i in picks:
        f = rec.frames[i]
        back = 40_000_000
        f.sample_ns -= back
        f.available_ns -= back
        f.device_ns -= back


@fixture(
    "defect_device_clock_copies_common",
    DEFECT,
    "sample_time_raw_device_clock is a verbatim copy of sample_time_local_common_clock: "
    "the vendor exposes no real device clock, so cross-device sync has nothing to use.",
    verdict="fail",
    check="timestamps.device_clock_distinct",
)
def _device_copy(rec):
    for f in rec.frames:
        f.device_ns = f.sample_ns


@fixture(
    "defect_available_before_sample",
    DEFECT,
    "available_time_local_common_clock is 5 ms *earlier* than "
    "sample_time_local_common_clock on every record: the sample became available before "
    "it was taken.",
    verdict="fail",
    check="timestamps.available_not_before_sample",
)
def _avail_before(rec):
    for f in rec.frames:
        f.available_ns = f.sample_ns - 5_000_000


@fixture(
    "defect_null_payload_majority",
    DEFECT,
    "About 68% of records carry a timestamp and no `data` table at all. Each such record "
    "is legal (pack_record emits exactly this for an inactive device) but in bulk it "
    "means the integration is producing almost no body data.",
    verdict="fail",
    check="coverage.payload_presence_rate",
)
def _null_payloads(rec):
    rng = random.Random(1234)
    for f in rec.frames:
        if rng.random() < 0.68:
            f.has_data = False
            f.joints = None
            f.all_tracked = False


@fixture(
    "defect_all_tracked_flag_inconsistent",
    DEFECT,
    "all_joint_poses_tracked stays true on every record while four joints report "
    "is_valid=false throughout -- the quality flag contradicts the per-joint flags.",
    verdict="fail",
    check="consistency.all_joint_poses_tracked",
)
def _flag_lies(rec):
    bad = [J["LEFT_FOOT"], J["RIGHT_FOOT"], J["LEFT_HAND"], J["RIGHT_HAND"]]
    for f in rec.frames:
        for i in bad:
            f.joints[i].valid = False
        f.all_tracked = True


# ======================================================================================
# C. DEFECTS -- rate / continuity
# ======================================================================================


@fixture(
    "defect_irregular_frame_intervals",
    DEFECT,
    "Frame intervals jitter by up to +/-9 ms around the 16.7 ms nominal period, well "
    "beyond the sub-millisecond jitter of the clean recording.",
    verdict="fail",
    check="rate.interval_regularity",
)
def _jitter(rec):
    rng = random.Random(55)
    for f in rec.frames:
        d = int(rng.uniform(-9e6, 9e6))
        f.sample_ns += d
        f.available_ns += d
        f.device_ns += d
    rec.frames.sort(key=lambda f: f.sample_ns)


@fixture(
    "defect_dropped_frames",
    DEFECT,
    "Four blocks of consecutive frames are missing, leaving 250-450 ms gaps in an "
    "otherwise 60 Hz stream.",
    verdict="fail",
    check="rate.frame_gaps",
)
def _drops(rec):
    gaps = [(2.0, 2.30), (5.0, 5.45), (8.2, 8.50), (12.0, 12.35)]
    keep = []
    for idx, f in enumerate(rec.frames):
        t = frame_time(rec, idx)
        if any(a <= t < b for a, b in gaps):
            continue
        keep.append(f)
    rec.frames = keep


@fixture(
    "defect_validity_degradation",
    DEFECT,
    "Per-joint validity coverage falls steadily from 100% at the start to ~30% at the "
    "end -- trackers dropping out progressively rather than a single transient.",
    verdict="fail",
    check="coverage.validity_trend",
)
def _degrade(rec):
    rng = random.Random(808)
    n = len(rec.frames)
    order = list(range(NUM_JOINTS))
    rng.shuffle(order)
    for idx, f in enumerate(rec.frames):
        frac_invalid = 0.70 * (idx / max(1, n - 1))
        n_invalid = round(frac_invalid * NUM_JOINTS)
        for i in order[:n_invalid]:
            f.joints[i] = zero_joint()
        f.all_tracked = n_invalid == 0


# ======================================================================================
# Driver
# ======================================================================================


def main() -> None:
    bfbs = load_bfbs()
    os.makedirs(OUT_DIR, exist_ok=True)

    index = {
        "schema": {
            "mcap_schema_name": "core.FullBodyPoseRecord",
            "schema_encoding": "flatbuffer",
            "schema_data": "binary FlatBuffers schema (.bfbs) for full_body.fbs",
            "message_encoding": "flatbuffer",
            "topic": "full_body/full_body",
            "mcap_profile": "teleop",
            "log_time": "available_time_local_common_clock",
            "publish_time": "available_time_local_common_clock",
        },
        "conventions": {
            "up_axis": "+Y",
            "handedness": "right",
            "facing": "-Z",
            "lateral": "+X is the subject's right, -X the subject's left",
            "position_units": "metres",
            "quaternion_order": "x, y, z, w",
            "nominal_rate_hz": 60.0,
        },
        "categories": {
            "golden": "clean and correct; the checker must report an overall pass",
            "benign": "unusual but correct; the checker must also pass these -- they are "
            "the false-positive guards, modelled on the Noitom plugin's output",
            "defect": "exactly one injected fault; the checker must fail the named check",
        },
        "fixtures": [],
    }

    for fx in REGISTRY:
        rec = fx.build()
        path = os.path.join(OUT_DIR, fx.name + ".mcap")
        write_mcap(path, rec, bfbs)
        size = os.path.getsize(path)
        entry = {
            "filename": os.path.relpath(path, HERE),
            "batch": "envelope",
            "category": fx.category,
            "description": fx.description,
            "expected_verdict": fx.verdict,
            "frames": len(rec.frames),
            "size_bytes": size,
        }
        if fx.check:
            entry["expected_failing_check"] = fx.check
        index["fixtures"].append(entry)
        print(
            f"{fx.category:7s} {fx.name:46s} {len(rec.frames):5d} frames  {size / 1024:8.1f} KiB"
        )

    import posture_fixtures

    print()
    index["fixtures"] += posture_fixtures.build_all(
        os.path.join(OUT_DIR, "posture"), bfbs, HERE
    )

    index["batches"] = {
        "envelope": "Schema/encoding/geometry/rate correctness of the recording itself.",
        "posture": "Posture semantics over the 11-step motion script.",
    }
    index["verdicts"] = {
        "pass": "checker must accept",
        "fail": "CTS failure attributable to the device/integration",
        "retake": "capture unusable because of human performance; device not implicated",
        "graded": "a known magnitude is injected; the checker reports the measured value and "
        "does not judge it",
    }
    index["motion_script"] = [
        {"index": i, "label": label, "duration_s": d}
        for i, (label, d) in enumerate(__import__("posture_script").STEPS)
    ]

    with open(os.path.join(HERE, "fixtures_index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")
    print(f"\n{len(index['fixtures'])} fixtures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
