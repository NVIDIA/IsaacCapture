# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Posture fixture batch: the 11-step script built with different ``PostureParams``.

Categories: graded series (a known magnitude at several levels, verdict "graded"),
device faults (CTS failures), performance faults (retake) and segmentation fixtures
that attack the label windows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from posture_script import STEPS, PostureParams, build_posture, sidecar_labels
from mcap_io import write_mcap

GOLDEN, GRADED, DEVICE, PERF, SEG = (
    "golden",
    "graded",
    "device_fault",
    "performance_fault",
    "segmentation",
)

# Verdict vocabulary used by this batch.
V_PASS = "pass"  # checker must report the run acceptable
V_FAIL = "fail"  # CTS failure attributable to the device/integration
V_RETAKE = "retake"  # capture is unusable; device not implicated
V_GRADED = "graded"  # reported as a measurement, not judged


@dataclass
class PostureFixture:
    name: str
    category: str
    kind: Optional[str]  # "device_fault" | "performance_fault" | None
    description: str
    verdict: str
    check: Optional[str] = None
    injected: Optional[dict] = None
    params: PostureParams = None
    label_offset_s: float = 0.0
    corrupt_labels: bool = False
    write_labels: bool = True
    notes: Optional[str] = None


NOMINAL_ORDER = [label for label, _ in STEPS]
SWAPPED_ORDER = list(NOMINAL_ORDER)
_i, _j = SWAPPED_ORDER.index("right_arm_raise"), SWAPPED_ORDER.index("left_leg_raise")
SWAPPED_ORDER[_i], SWAPPED_ORDER[_j] = SWAPPED_ORDER[_j], SWAPPED_ORDER[_i]


def _fixtures() -> List[PostureFixture]:
    F: List[PostureFixture] = []

    F.append(
        PostureFixture(
            "posture_golden_full_script",
            GOLDEN,
            None,
            "Clean run of the full 11-step script (clap, A-pose, T-pose, neutral, L/R arm "
            "raise, L/R leg raise, 2x squat, march, closing T-pose) at 50 Hz. Correct device, "
            "competent performance. Every posture assertion should pass.",
            V_PASS,
            params=PostureParams(),
        )
    )

    # ---- graded series -------------------------------------------------------------
    for v in (0.0, 0.02, 0.05, 0.15):
        F.append(
            PostureFixture(
                f"posture_graded_drift_{round(v * 1000):03d}mrad",
                GRADED,
                None,
                f"Closing T-pose shoulder angle offset from the opening T-pose by exactly "
                f"{v} rad ({v * 57.2958:.2f} deg), applied to both arms.",
                V_PASS if v == 0.0 else V_GRADED,
                check="posture.cumulative_drift_between_tpose_windows",
                injected={"quantity": "cumulative_drift", "value": v, "unit": "rad"},
                params=PostureParams(drift_rad=v),
            )
        )

    for v in (0.0, 2.0, 10.0, 25.0):
        F.append(
            PostureFixture(
                f"posture_graded_crosstalk_{int(v):02d}deg",
                GRADED,
                None,
                f"During each single-leg raise the *opposite* hip also flexes by exactly "
                f"{v} deg at peak -- contralateral cross-talk.",
                V_PASS if v == 0.0 else V_GRADED,
                check="posture.contralateral_crosstalk_single_leg_raise",
                injected={
                    "quantity": "contralateral_crosstalk",
                    "value": v,
                    "unit": "deg",
                },
                params=PostureParams(crosstalk_deg=v),
            )
        )

    for v in (0.0, 5.0, 15.0, 30.0):
        F.append(
            PostureFixture(
                f"posture_graded_droop_{int(v):02d}deg",
                GRADED,
                None,
                f"Both arms held exactly {v} deg below horizontal during both T-pose windows.",
                V_PASS if v == 0.0 else V_GRADED,
                check="posture.tpose_arm_droop",
                injected={"quantity": "tpose_arm_droop", "value": v, "unit": "deg"},
                params=PostureParams(tpose_droop_deg=v),
            )
        )

    for v in (0.0, 3.0, 10.0):
        F.append(
            PostureFixture(
                f"posture_graded_asymmetry_{int(v):02d}deg",
                GRADED,
                None,
                f"LEFT arm held exactly {v} deg lower than the RIGHT in both T-pose windows.",
                V_PASS if v == 0.0 else V_GRADED,
                check="posture.tpose_left_right_asymmetry",
                injected={
                    "quantity": "tpose_left_right_asymmetry",
                    "value": v,
                    "unit": "deg",
                },
                params=PostureParams(asymmetry_deg=v),
            )
        )

    # ---- device faults --------------------------------------------------------------
    F.append(
        PostureFixture(
            "posture_device_march_ankles_in_phase",
            DEVICE,
            DEVICE,
            "March performed normally but the two ankles move in phase instead of antiphase: "
            "a left/right confusion that only shows up dynamically, invisible in any static pose.",
            V_FAIL,
            check="posture.march_ankle_antiphase",
            injected={
                "quantity": "march_phase_offset",
                "value": 0.0,
                "unit": "rad",
                "expected_clean_value": 3.14159,
            },
            params=PostureParams(march_in_phase=True),
        )
    )
    F.append(
        PostureFixture(
            "posture_device_squat_reps_mismatched",
            DEVICE,
            DEVICE,
            "Two squat reps of the same real depth are reported 35% apart -- error "
            "accumulating between reps rather than a performance difference.",
            V_FAIL,
            check="posture.squat_rep_repeatability",
            injected={
                "quantity": "squat_rep2_hip_angle_delta",
                "value": 0.35,
                "unit": "fraction of rep-1 peak hip flexion angle",
                "note": "pelvis-height depth is a nonlinear function of the hip angle, "
                "so the measured depth ratio is larger (~+71%) than 0.35",
            },
            params=PostureParams(squat_rep2_delta=0.35),
        )
    )
    F.append(
        PostureFixture(
            "posture_device_squat_knee_asymmetry",
            DEVICE,
            DEVICE,
            "Knee flexion differs by 22 deg between left and right at full squat depth while "
            "hips, ankles and pelvis stay symmetric -- the person squatted evenly.",
            V_FAIL,
            check="posture.squat_knee_symmetry",
            injected={"quantity": "squat_knee_asymmetry", "value": 22.0, "unit": "deg"},
            params=PostureParams(squat_knee_asym_deg=22.0),
        )
    )
    F.append(
        PostureFixture(
            "posture_device_arm_raise_saturates",
            DEVICE,
            DEVICE,
            "Both arm raises stop tracking at 25 deg above horizontal and flat-line there "
            "instead of reaching overhead: the shoulder angle saturates mid-raise.",
            V_FAIL,
            check="posture.arm_raise_range_of_motion",
            injected={
                "quantity": "arm_raise_saturation_angle",
                "value": -25.0,
                "unit": "deg",
                "expected_clean_value": -95.0,
            },
            params=PostureParams(arm_raise_saturate_deg=-25.0),
        )
    )

    # ---- performance faults ---------------------------------------------------------
    F.append(
        PostureFixture(
            "posture_perf_squat_too_shallow",
            PERF,
            PERF,
            "Squats only 22% of nominal depth -- too shallow for knee symmetry or pelvis "
            "height to be measurable. Device is fine; the capture needs redoing.",
            V_RETAKE,
            check="performance.squat_depth_sufficient",
            injected={
                "quantity": "squat_depth_scale",
                "value": 0.22,
                "unit": "fraction",
            },
            params=PostureParams(squat_depth_scale=0.22),
            notes="Reduced depth also shrinks any knee-symmetry signal, so a checker that "
            "reports knee symmetry without a depth precondition will look clean here.",
        )
    )
    F.append(
        PostureFixture(
            "posture_perf_march_slow_irregular",
            PERF,
            PERF,
            "March performed slowly and irregularly (cadence wandering 0.35-0.75 Hz, "
            "amplitude 14-38 deg) instead of at a steady cadence. Still antiphase.",
            V_RETAKE,
            check="performance.march_cadence_steady",
            injected={
                "quantity": "march_cadence_range",
                "value": [0.35, 0.75],
                "unit": "Hz",
                "expected_clean_value": 0.9,
            },
            params=PostureParams(march_irregular=True),
        )
    )
    F.append(
        PostureFixture(
            "posture_perf_lean_during_arm_raise",
            PERF,
            PERF,
            "Torso leans 12 deg away while each arm is raised. The arm itself tracks "
            "correctly; the isolation assumption does not hold.",
            V_RETAKE,
            check="performance.arm_raise_torso_stability",
            injected={
                "quantity": "torso_lean_during_arm_raise",
                "value": 12.0,
                "unit": "deg",
            },
            params=PostureParams(lean_deg=12.0),
        )
    )
    F.append(
        PostureFixture(
            "posture_perf_left_leg_raise_skipped",
            PERF,
            PERF,
            "The left-leg raise never happens -- the subject stands in neutral through the "
            "whole labelled window. Labels still claim the step was performed.",
            V_RETAKE,
            check="segmentation.labelled_step_actually_performed",
            injected={"quantity": "skipped_steps", "value": ["left_leg_raise"]},
            params=PostureParams(skip_steps=("left_leg_raise",)),
        )
    )
    F.append(
        PostureFixture(
            "posture_perf_steps_out_of_order",
            PERF,
            PERF,
            "right_arm_raise and left_leg_raise are performed in each other's time slots; "
            "the sidecar labels still describe the nominal order.",
            V_RETAKE,
            check="segmentation.step_order_matches_labels",
            injected={
                "quantity": "swapped_steps",
                "value": ["right_arm_raise", "left_leg_raise"],
            },
            params=PostureParams(motion_order=SWAPPED_ORDER),
        )
    )

    # ---- segmentation ----------------------------------------------------------------
    F.append(
        PostureFixture(
            "posture_seg_labels_absent",
            SEG,
            None,
            "Clean performance, no sidecar at all. The checker must say that the per-step "
            "posture measurements are skipped.",
            V_PASS,
            check="segmentation.fallback_without_labels",
            params=PostureParams(),
            write_labels=False,
        )
    )
    F.append(
        PostureFixture(
            "posture_seg_labels_offset",
            SEG,
            None,
            "Clean performance, but every sidecar window is 0.5 s later than the motion it "
            "names -- each window straddles a step boundary.",
            V_FAIL,
            check="segmentation.label_alignment",
            injected={"quantity": "label_offset", "value": 0.5, "unit": "s"},
            params=PostureParams(),
            label_offset_s=0.5,
        )
    )
    F.append(
        PostureFixture(
            "posture_seg_labels_overlap_and_gaps",
            SEG,
            None,
            "Clean performance, but the sidecar windows overlap by 0.6 s in places and leave "
            "0.8 s unlabelled in others: the label set is not a partition of the run.",
            V_FAIL,
            check="segmentation.label_windows_wellformed",
            params=PostureParams(),
            corrupt_labels=True,
        )
    )
    return F


def build_all(out_dir: str, bfbs: bytes, rel_to: str) -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for fx in _fixtures():
        rec = build_posture(fx.params)
        path = os.path.join(out_dir, fx.name + ".mcap")
        write_mcap(path, rec, bfbs)

        label_path = None
        if fx.write_labels:
            labels = sidecar_labels(
                rec, offset_s=fx.label_offset_s, corrupt_windows=fx.corrupt_labels
            )
            labels["mcap"] = fx.name + ".mcap"
            label_path = os.path.join(out_dir, fx.name + ".labels.json")
            with open(label_path, "w") as fh:
                json.dump(labels, fh, indent=2)
                fh.write("\n")

        entry = {
            "filename": os.path.relpath(path, rel_to),
            "batch": "posture",
            "category": fx.category,
            "kind": fx.kind,
            "description": fx.description,
            "expected_verdict": fx.verdict,
            "frames": len(rec.frames),
            "size_bytes": os.path.getsize(path),
            "labels_sidecar": os.path.relpath(label_path, rel_to)
            if label_path
            else None,
        }
        if fx.check:
            entry[
                "expected_failing_check"
                if fx.verdict in (V_FAIL, V_RETAKE)
                else "measured_by_check"
            ] = fx.check
        if fx.injected:
            entry["injected"] = fx.injected
        if fx.notes:
            entry["notes"] = fx.notes
        entries.append(entry)
        print(
            f"{fx.category:18s} {fx.name:38s} {len(rec.frames):5d} frames  "
            f"{entry['size_bytes'] / 1024:8.1f} KiB"
        )
    return entries
