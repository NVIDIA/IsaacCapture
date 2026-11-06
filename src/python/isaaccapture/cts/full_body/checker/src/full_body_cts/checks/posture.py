# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Posture measurements over the labelled motion windows.

Each is a local joint angle, ``conj(q_parent) * q_child`` projected onto the body axis
the motion turns about.
"""

from __future__ import annotations

import math
import statistics
from typing import ClassVar

from ..frames import Frame
from ..labels import Step, StepTimeline
from ..profile import FULL_BODY, SkeletonProfile
from cts_common.checks.base import (
    Attribution,
    Check,
    Outcome,
    Severity,
    Status,
)
from cts_common.vectors import relative, signed_angle_about

# Body axes the posture script turns each joint about.
SAGITTAL = (1.0, 0.0, 0.0)  # hip, knee and ankle flexion
FRONTAL = (0.0, 0.0, 1.0)  # shoulder elevation, torso lean

# Leading fraction of a held window skipped; measurement uses the trailing part.
SETTLE_FRACTION = 0.40


class _PostureCheck(Check):
    """Accumulates per-window angle samples, keyed by label."""

    group = "posture"
    needs_timeline = True
    required = False

    depends_on = (
        "segmentation.label_windows_wellformed",
        "segmentation.label_alignment",
    )

    # Labels whose frames this check needs; empty means every labelled frame.
    wanted: frozenset[str] = frozenset()

    def __init__(
        self,
        timeline: StepTimeline | None = None,
        profile: SkeletonProfile = FULL_BODY,
    ) -> None:
        super().__init__()
        self.timeline = timeline
        self.profile = profile

    def _settled(self, step: Step, sample_ns: int) -> bool:
        span = step.end_ns - step.start_ns
        return sample_ns >= step.start_ns + int(SETTLE_FRACTION * span)

    def _angle(self, frame: Frame, joint: str, axis, sign: float = 1.0) -> float | None:
        """The joint's rotation relative to its parent, about ``axis``, in degrees."""
        index = self.profile.index(joint)
        parent = self.profile.parents[index]
        joints = frame.joints
        if joints is None or parent < 0:
            return None
        child, base = joints[index], joints[parent]
        if not (child.is_valid and base.is_valid):
            return None
        radians = signed_angle_about(
            relative(base.orientation, child.orientation), axis
        )
        if not math.isfinite(radians):
            return None
        return sign * math.degrees(radians)

    def _update(self, frame: Frame) -> None:
        if self.timeline is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or frame.joints is None:
            return
        if self.wanted and step.label not in self.wanted:
            return
        self._observe(step, frame)

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "no motion labels beside the recording, so no window can be measured",
            )
        return self._measure()

    def _observe(self, step: Step, frame: Frame) -> None:
        raise NotImplementedError

    def _measure(self) -> Outcome:
        raise NotImplementedError


class _ShoulderCheck(_PostureCheck):
    """Shoulder elevation in both T-pose windows; the right is negated so positive means below horizontal."""

    wanted = frozenset({"t_pose_hold_open", "t_pose_hold_close"})

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.held: dict[str, list[tuple[float, float]]] = {}

    def _observe(self, step: Step, frame: Frame) -> None:
        if frame.sample_time_ns is None or not self._settled(
            step, frame.sample_time_ns
        ):
            return
        left = self._angle(frame, "LEFT_SHOULDER", FRONTAL)
        right = self._angle(frame, "RIGHT_SHOULDER", FRONTAL, sign=-1.0)
        if left is None or right is None:
            return
        self.held.setdefault(step.label, []).append((left, right))

    def _arms(self, label: str) -> tuple[float, float] | None:
        samples = self.held.get(label)
        if not samples:
            return None
        return (
            statistics.median(left for left, _ in samples),
            statistics.median(right for _, right in samples),
        )


class TposeArmDroop(_ShoulderCheck):
    name = "posture.tpose_arm_droop"
    severity = Severity.ADVISORY
    judged = False
    summary = "How far below horizontal the arms sit in the opening T-pose"

    def _measure(self) -> Outcome:
        arms = self._arms("t_pose_hold_open")
        if arms is None:
            return Outcome(
                Status.INSUFFICIENT_DATA, "the opening T-pose window has no valid arms"
            )
        left, right = arms
        # Droop of the arm that sits higher.
        droop = min(left, right)
        return Outcome(
            Status.PASS,
            f"arms {droop:.1f} deg below horizontal in the opening T-pose",
            {"droop_deg": droop, "left_deg": left, "right_deg": right},
        )


class TposeLeftRightAsymmetry(_ShoulderCheck):
    name = "posture.tpose_left_right_asymmetry"
    severity = Severity.ADVISORY
    judged = False
    summary = "How differently the two arms are held in the T-pose"

    def _measure(self) -> Outcome:
        arms = self._arms("t_pose_hold_open")
        if arms is None:
            return Outcome(
                Status.INSUFFICIENT_DATA, "the opening T-pose window has no valid arms"
            )
        left, right = arms
        return Outcome(
            Status.PASS,
            f"left arm sits {left - right:+.1f} deg from the right in the T-pose",
            {"asymmetry_deg": left - right, "left_deg": left, "right_deg": right},
        )


class CumulativeDriftBetweenTposeWindows(_ShoulderCheck):
    name = "posture.cumulative_drift_between_tpose_windows"
    severity = Severity.ADVISORY
    judged = False
    summary = "How far the same held pose has moved by the end of the session"

    def _measure(self) -> Outcome:
        opening, closing = (
            self._arms("t_pose_hold_open"),
            self._arms("t_pose_hold_close"),
        )
        if opening is None or closing is None:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "the session needs both T-pose windows to read drift",
            )
        before = sum(opening) / 2.0
        after = sum(closing) / 2.0
        drift_deg = after - before
        return Outcome(
            Status.PASS,
            f"the closing T-pose sits {drift_deg:+.2f} deg "
            f"({math.radians(drift_deg):+.4f} rad) from the opening one",
            {
                "drift_rad": math.radians(drift_deg),
                "drift_deg": drift_deg,
                "opening_deg": before,
                "closing_deg": after,
            },
        )


class ContralateralCrosstalkSingleLegRaise(_PostureCheck):
    name = "posture.contralateral_crosstalk_single_leg_raise"
    severity = Severity.ADVISORY
    judged = False
    summary = "How much the still leg moves while the other one is raised"

    # Window read for each hip's resting angle, not for crosstalk.
    BASELINE_WINDOW = "neutral_stance"
    wanted = frozenset({"left_leg_raise", "right_leg_raise", BASELINE_WINDOW})

    depends_on = _PostureCheck.depends_on + (
        "segmentation.labelled_step_actually_performed",
        "segmentation.step_order_matches_labels",
    )

    OTHER_HIP: ClassVar[dict[str, str]] = {
        "left_leg_raise": "RIGHT_HIP",
        "right_leg_raise": "LEFT_HIP",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extremes: dict[str, tuple[float, float]] = {}
        self.resting: dict[str, list[float]] = {}

    def _observe(self, step: Step, frame: Frame) -> None:
        if step.label == self.BASELINE_WINDOW:
            if frame.sample_time_ns is None or not self._settled(
                step, frame.sample_time_ns
            ):
                return
            for hip in self.OTHER_HIP.values():
                angle = self._angle(frame, hip, SAGITTAL)
                if angle is not None:
                    self.resting.setdefault(hip, []).append(angle)
            return
        angle = self._angle(frame, self.OTHER_HIP[step.label], SAGITTAL)
        if angle is None:
            return
        low, high = self.extremes.get(step.label, (angle, angle))
        self.extremes[step.label] = (min(low, angle), max(high, angle))

    def _baseline(self, hip: str) -> float:
        samples = self.resting.get(hip)
        return statistics.median(samples) if samples else 0.0

    def _measure(self) -> Outcome:
        if not self.extremes:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "neither single-leg raise window was measurable",
            )
        peak = {}
        for label, (low, high) in self.extremes.items():
            rest = self._baseline(self.OTHER_HIP[label])
            peak[label] = max(abs(high - rest), abs(low - rest))
        worst_label = max(peak, key=peak.__getitem__)
        worst = peak[worst_label]
        return Outcome(
            Status.PASS,
            f"the still hip moves up to {worst:.1f} deg from its resting angle "
            f"during {worst_label}",
            {
                "crosstalk_deg": worst,
                "worst_window": worst_label,
                "per_window_deg": dict(peak),
                "resting_deg": {
                    hip: self._baseline(hip) for hip in self.OTHER_HIP.values()
                },
            },
        )


class ArmRaiseRangeOfMotion(_PostureCheck):
    name = "posture.arm_raise_range_of_motion"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "A raised arm reaches overhead rather than stopping short"

    depends_on = _PostureCheck.depends_on + (
        "segmentation.labelled_step_actually_performed",
        "segmentation.step_order_matches_labels",
    )

    wanted = frozenset({"left_arm_raise", "right_arm_raise"})

    # Minimum peak elevation of the shoulder-to-hand line above horizontal, degrees.
    MIN_ELEVATION_DEG = 60.0

    SIDE: ClassVar[dict[str, str]] = {
        "left_arm_raise": "LEFT",
        "right_arm_raise": "RIGHT",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reached: dict[str, float] = {}

    def _observe(self, step: Step, frame: Frame) -> None:
        """Elevation of the shoulder-to-hand line, never the shoulder joint's angle."""
        side = self.SIDE[step.label]
        joints = frame.joints
        if joints is None:
            return
        shoulder = joints[self.profile.index(f"{side}_SHOULDER")]
        hand = joints[self.profile.index(f"{side}_HAND")]
        if not (shoulder.is_valid and hand.is_valid):
            return
        reach = [hand.position[i] - shoulder.position[i] for i in range(3)]
        length = math.sqrt(sum(component * component for component in reach))
        if length <= 0.0:
            return
        elevation = math.degrees(math.asin(max(-1.0, min(1.0, reach[1] / length))))
        self.reached[step.label] = max(self.reached.get(step.label, -180.0), elevation)

    def _measure(self) -> Outcome:
        if not self.reached:
            return Outcome(
                Status.INSUFFICIENT_DATA, "neither arm-raise window was measurable"
            )
        worst_label = min(self.reached, key=self.reached.__getitem__)
        worst = self.reached[worst_label]
        measurements = {
            "elevation_deg": worst,
            "worst_window": worst_label,
            "per_window_deg": dict(self.reached),
        }
        if worst >= self.MIN_ELEVATION_DEG:
            return Outcome(
                Status.PASS,
                f"both hands reach {worst:.0f} deg above the shoulder",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"the hand in {worst_label} reaches {worst:.0f} deg above the "
            f"shoulder, short of "
            f"{self.MIN_ELEVATION_DEG:.0f}; record again reaching fully overhead, and "
            f"if it still stops short the device is clipping the arm",
            measurements,
        )


class MarchAnkleAntiphase(_PostureCheck):
    name = "posture.march_ankle_antiphase"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "The legs alternate while marching rather than moving together"

    depends_on = _PostureCheck.depends_on + (
        "segmentation.labelled_step_actually_performed",
        "segmentation.step_order_matches_labels",
    )

    wanted = frozenset({"march_in_place"})

    # Maximum left/right hip correlation for the legs to count as alternating.
    MAX_CORRELATION = 0.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left: list[float] = []
        self.right: list[float] = []

    def _observe(self, step: Step, frame: Frame) -> None:
        left = self._angle(frame, "LEFT_HIP", SAGITTAL)
        right = self._angle(frame, "RIGHT_HIP", SAGITTAL)
        if left is None or right is None:
            return
        self.left.append(left)
        self.right.append(right)

    def _measure(self) -> Outcome:
        if len(self.left) < 30:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the march window yielded {len(self.left)} usable frames",
            )
        try:
            correlation = statistics.correlation(self.left, self.right)
        except statistics.StatisticsError:
            return Outcome(
                Status.INSUFFICIENT_DATA, "neither leg moves during the march window"
            )
        measurements = {"hip_correlation": correlation, "samples": len(self.left)}
        if correlation <= self.MAX_CORRELATION:
            return Outcome(
                Status.PASS,
                f"the legs alternate (hip correlation {correlation:+.2f})",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"both hips rise together (correlation {correlation:+.2f}), so the two legs "
            f"are being driven from one signal",
            measurements,
        )


class _SquatCheck(_PostureCheck):
    """Per-rep peak hip flexion, and the knee angles at the deepest frame of each rep."""

    wanted = frozenset({"squat_x2"})

    # Minimum hip flexion for a rep to count, degrees.
    REP_ONSET_DEG = 5.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[tuple[float, float, float]] = []

    def _observe(self, step: Step, frame: Frame) -> None:
        hip_left = self._angle(frame, "LEFT_HIP", SAGITTAL)
        hip_right = self._angle(frame, "RIGHT_HIP", SAGITTAL)
        knee_left = self._angle(frame, "LEFT_KNEE", SAGITTAL)
        knee_right = self._angle(frame, "RIGHT_KNEE", SAGITTAL)
        if None in (hip_left, hip_right, knee_left, knee_right):
            return
        # Ankles are optional: an invalid ankle does not discard the depth and knee readings.
        ankle_left = self._angle(frame, "LEFT_ANKLE", SAGITTAL)
        ankle_right = self._angle(frame, "RIGHT_ANKLE", SAGITTAL)
        self.samples.append(
            (
                (hip_left + hip_right) / 2.0,
                knee_left,
                knee_right,
                hip_left,
                hip_right,
                ankle_left,
                ankle_right,
            )
        )

    def _reps(self) -> list[list[tuple[float, float, float]]]:
        """Splits the window into runs of frames that are below standing."""
        reps: list[list[tuple[float, float, float]]] = []
        current: list[tuple[float, float, float]] = []
        for sample in self.samples:
            if sample[0] >= self.REP_ONSET_DEG:
                current.append(sample)
            elif current:
                reps.append(current)
                current = []
        if current:
            reps.append(current)
        return reps


class SquatKneeSymmetry(_SquatCheck):
    name = "posture.squat_knee_symmetry"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "Both knees flex by the same amount at the bottom of a squat"

    depends_on = _PostureCheck.depends_on + (
        "segmentation.labelled_step_actually_performed",
        "segmentation.step_order_matches_labels",
    )

    MAX_ASYMMETRY_DEG = 10.0

    # Minimum hip flexion at which knee symmetry is judged, degrees.
    MIN_DEPTH_DEG = 20.0

    # Hip or ankle asymmetry above which the failure is attributed to the performer, degrees.
    CHAIN_ASYMMETRY_DEG = 5.0

    def _measure(self) -> Outcome:
        reps = self._reps()
        if not reps:
            return Outcome(Status.INSUFFICIENT_DATA, "no squat rep found in the window")
        depth = max(sample[0] for rep in reps for sample in rep)
        if depth < self.MIN_DEPTH_DEG:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the deepest squat reaches {depth:.0f} deg of hip flexion, too shallow "
                f"to read knee symmetry",
                {"depth_deg": depth},
            )
        worst = hips = ankles = 0.0
        for rep in reps:
            bottom = max(rep, key=lambda sample: sample[0])
            worst = max(worst, abs(abs(bottom[2]) - abs(bottom[1])))
            hips = max(hips, abs(abs(bottom[4]) - abs(bottom[3])))
            if bottom[5] is not None and bottom[6] is not None:
                ankles = max(ankles, abs(abs(bottom[6]) - abs(bottom[5])))
        chain = max(hips, ankles)
        measurements = {
            "knee_asymmetry_deg": worst,
            "hip_asymmetry_deg": hips,
            "ankle_asymmetry_deg": ankles,
            "depth_deg": depth,
            "reps": len(reps),
        }
        if worst <= self.MAX_ASYMMETRY_DEG:
            return Outcome(
                Status.PASS, f"knees differ by {worst:.1f} deg at depth", measurements
            )
        if chain > self.CHAIN_ASYMMETRY_DEG:
            return Outcome(
                Status.FAIL,
                f"the knees differ by {worst:.1f} deg at the bottom of a squat, but so "
                f"do the hips ({hips:.1f} deg) and ankles ({ankles:.1f} deg), so the "
                f"squat itself was uneven; record again loading both legs equally",
                measurements,
                attribution=Attribution.PERFORMANCE,
            )
        return Outcome(
            Status.FAIL,
            f"the knees differ by {worst:.1f} deg at the bottom of a squat, more than "
            f"{self.MAX_ASYMMETRY_DEG:.0f}, while the hips ({hips:.1f} deg) and ankles "
            f"({ankles:.1f} deg) stay symmetric, so the knee estimate is at fault",
            measurements,
        )


class SquatRepRepeatability(_SquatCheck):
    name = "posture.squat_rep_repeatability"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "Two squats of the same depth are reported as the same depth"

    depends_on = _PostureCheck.depends_on + (
        "segmentation.labelled_step_actually_performed",
        "segmentation.step_order_matches_labels",
    )

    MAX_REP_DELTA = 0.20

    def _measure(self) -> Outcome:
        reps = self._reps()
        peaks = [max(sample[0] for sample in rep) for rep in reps]
        if len(peaks) < 2:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"found {len(peaks)} squat reps, needs two to compare",
                {"rep_peaks_deg": peaks},
            )
        first, second = peaks[0], peaks[1]
        delta = (second - first) / first if first else float("inf")
        measurements = {
            "rep_delta": delta,
            "rep_peaks_deg": peaks,
            "reps": len(peaks),
        }
        if abs(delta) <= self.MAX_REP_DELTA:
            return Outcome(
                Status.PASS,
                f"the two reps agree to {abs(delta):.0%} of hip flexion",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"rep 2 reaches {delta:+.0%} of rep 1's hip flexion ({first:.0f} then "
            f"{second:.0f} deg) on two squats performed alike",
            measurements,
        )


class SquatDepthSufficient(_SquatCheck):
    name = "performance.squat_depth_sufficient"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "The squat is deep enough to measure anything from"

    MIN_DEPTH_DEG = 20.0

    def _measure(self) -> Outcome:
        reps = self._reps()
        if not reps:
            return Outcome(Status.INSUFFICIENT_DATA, "no squat rep found in the window")
        depth = max(sample[0] for rep in reps for sample in rep)
        measurements = {"depth_deg": depth, "reps": len(reps)}
        if depth >= self.MIN_DEPTH_DEG:
            return Outcome(
                Status.PASS,
                f"the squat reaches {depth:.0f} deg of hip flexion",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"the deepest squat reaches only {depth:.0f} deg of hip flexion, below "
            f"{self.MIN_DEPTH_DEG:.0f}; ask for a deeper squat rather than blaming the "
            f"device",
            measurements,
        )


class MarchCadenceSteady(_PostureCheck):
    name = "performance.march_cadence_steady"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "The march holds a steady cadence"

    wanted = frozenset({"march_in_place"})

    LIFT_ONSET_DEG = 5.0
    MAX_INTERVAL_CV = 0.25

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lift_times: list[float] = []
        self._airborne = False

    def _observe(self, step: Step, frame: Frame) -> None:
        angle = self._angle(frame, "LEFT_HIP", SAGITTAL)
        if angle is None or frame.sample_time_ns is None:
            return
        lifting = angle >= self.LIFT_ONSET_DEG
        if lifting and not self._airborne:
            self.lift_times.append(frame.sample_time_ns / 1e9)
        self._airborne = lifting

    def _measure(self) -> Outcome:
        intervals = [
            later - earlier
            for earlier, later in zip(self.lift_times, self.lift_times[1:])
        ]
        if len(intervals) < 2:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"counted {len(self.lift_times)} left-leg lifts, needs three to judge "
                f"steadiness",
                {"lifts": len(self.lift_times)},
            )
        mean = statistics.fmean(intervals)
        spread = statistics.stdev(intervals) / mean if mean else float("inf")
        measurements = {
            "cadence_hz": 1.0 / mean if mean else 0.0,
            "interval_cv": spread,
            "lifts": len(self.lift_times),
        }
        if spread <= self.MAX_INTERVAL_CV:
            return Outcome(
                Status.PASS,
                f"cadence holds at {1.0 / mean:.2f} Hz, step intervals varying "
                f"{spread:.0%}",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"step intervals vary {spread:.0%} about {1.0 / mean:.2f} Hz; ask for an "
            f"even march rather than blaming the device",
            measurements,
        )


class ArmRaiseTorsoStability(_PostureCheck):
    name = "performance.arm_raise_torso_stability"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "The torso stays upright while an arm is raised"

    wanted = frozenset({"left_arm_raise", "right_arm_raise"})

    MAX_LEAN_DEG = 6.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.peak_lean = 0.0

    def _observe(self, step: Step, frame: Frame) -> None:
        joints = frame.joints
        pelvis = self.profile.index("PELVIS")
        if joints is None or not joints[pelvis].is_valid:
            return
        # The pelvis is the root, so its orientation is world-relative.
        radians = signed_angle_about(joints[pelvis].orientation, FRONTAL)
        if math.isfinite(radians):
            self.peak_lean = max(self.peak_lean, abs(math.degrees(radians)))

    def _measure(self) -> Outcome:
        measurements = {"lean_deg": self.peak_lean}
        if self.peak_lean <= self.MAX_LEAN_DEG:
            return Outcome(
                Status.PASS,
                f"the torso stays within {self.peak_lean:.1f} deg of upright",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"the torso leans {self.peak_lean:.1f} deg while raising an arm, over "
            f"{self.MAX_LEAN_DEG:.0f}; ask for the raise without the lean",
            measurements,
        )
