# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks read through the label windows. Held poses are measured on settled frames only.

Thresholds marked "placeholder" are loose until real glove recordings exist.
"""

from __future__ import annotations

import statistics
from typing import Any, ClassVar

from acceptance_common.checks import segmentation as _shared
from acceptance_common.checks.base import (
    Attribution,
    Check,
    Outcome,
    Severity,
    Status,
)
from acceptance_common.vectors import norm, sub

from ..frames import Frame, Vec3
from ..labels import GRASPS, PINCHES, SCRIPT, StepTimeline, settled
from ..profile import FINGERS, REQUIRED, TIP
from .base import HandCheck, angle_deg, hand_axes, hand_straightness, median, per_hand

GROUP = "windows"

WELLFORMED = "segmentation.label_windows_wellformed"
ALIGNMENT = "segmentation.label_alignment"
ORDER = "segmentation.step_order_matches_labels"
OPEN_WINDOWS = ("open_hand_open", "open_hand_close")


class LabelWindowsWellformed(_shared.LabelWindowsWellformed):
    gate = GROUP


class LabelAlignment(_shared.LabelAlignment):
    gate = GROUP


def _straightness(frame: Frame) -> float | None:
    return hand_straightness([frame.position(i) for i in range(len(frame.joints))])


def _pinch_distances(frame: Frame) -> dict[str, float] | None:
    thumb = frame.position(TIP["thumb"])
    tips = {finger: frame.position(TIP[finger]) for finger in FINGERS}
    if thumb is None or any(tip is None for tip in tips.values()):
        return None
    return {finger: norm(sub(tip, thumb)) for finger, tip in tips.items()}


def _rotation_range(axes: list[tuple[Vec3, Vec3, Vec3]]) -> float:
    if not axes:
        return 0.0
    first = axes[0]
    return max(
        (max(angle_deg(a, b) for a, b in zip(frame, first)) for frame in axes[1:]),
        default=0.0,
    )


class _WindowCheck(HandCheck):
    """Collects one value per settled frame of each window this check reads."""

    needs_timeline = True
    required = False
    labels: ClassVar[tuple[str, ...]] = SCRIPT

    def __init__(self, side: str, timeline: StepTimeline | None = None) -> None:
        super().__init__(side)
        self.timeline = timeline
        self.values: dict[str, list[Any]] = {}

    def measure(self, frame: Frame, label: str) -> Any: ...

    def judge(self) -> Outcome: ...

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label not in self.labels:
            return
        if not settled(step, frame.sample_time_ns):
            return
        value = self.measure(frame, step.label)
        if value is not None:
            self.values.setdefault(step.label, []).append(value)

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels")
        return self.judge()


class _Features(_WindowCheck):
    def measure(self, frame: Frame, label: str) -> Any:
        straight = _straightness(frame)
        return (straight, _pinch_distances(frame), hand_axes(frame))

    def straightness(self, label: str) -> float | None:
        values = [v[0] for v in self.values.get(label, ()) if v[0] is not None]
        return median(values) if values else None

    def rotation(self, label: str) -> float | None:
        axes = [v[2] for v in self.values.get(label, ()) if v[2] is not None]
        return _rotation_range(axes) if axes else None

    def nearest_pinch(self, label: str) -> float | None:
        distances = [v[1] for v in self.values.get(label, ()) if v[1] is not None]
        if not distances:
            return None
        return min(median([d[f] for d in distances]) for f in FINGERS)

    def fist_labels(self) -> tuple[str, ...]:
        """Windows where this hand holds a fist."""
        return ("fist",) + tuple(
            label for label, (_, fist) in GRASPS.items() if fist == self.side
        )


class _StepActuallyPerformed(_Features):
    name = "segmentation.labelled_step_actually_performed"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Each labelled window shows at least the gross motion its label names"

    # Gross presence only; the graded thresholds are the posture checks'.
    OPEN_MIN = 0.6
    FIST_DROP = 0.15
    FIST_MAX = 0.75
    PINCH_MAX_M = 0.04
    ROTATE_MIN_DEG = 60.0

    depends_on = (WELLFORMED, ALIGNMENT)

    def judge(self) -> Outcome:
        reference = self.straightness("open_hand_open")
        fist_limit = (
            self.FIST_MAX
            if reference is None
            else min(self.FIST_MAX, reference - self.FIST_DROP)
        )
        observed: dict[str, float] = {}
        missed: list[str] = []
        for step in self.timeline.steps:
            label = step.label
            if label in OPEN_WINDOWS or label in self.fist_labels():
                value = self.straightness(label)
                ok = value is not None and (
                    value >= self.OPEN_MIN
                    if label in OPEN_WINDOWS
                    else value <= fist_limit
                )
            elif label in PINCHES:
                value = self.nearest_pinch(label)
                ok = value is not None and value <= self.PINCH_MAX_M
            elif label == "wrist_rotate":
                value = self.rotation(label)
                ok = value is not None and value >= self.ROTATE_MIN_DEG
            else:
                continue
            if value is not None:
                observed[label] = value
            if not ok:
                missed.append(label)
        measurements = {"observed": observed, "not_performed": missed}
        if not observed:
            return Outcome(
                Status.INSUFFICIENT_DATA, "no scripted window measurable", measurements
            )
        if not missed:
            return Outcome(
                Status.PASS,
                f"all {len(observed)} windows show their motion",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"not performed: {', '.join(missed)}",
            measurements,
        )


class _StepOrderMatchesLabels(_Features):
    name = "segmentation.step_order_matches_labels"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "The fist is the most curled window and wrist_rotate the most turned"

    depends_on = (WELLFORMED, ALIGNMENT)

    def judge(self) -> Outcome:
        fist = self.straightness("fist")
        opens = {
            label: self.straightness(label)
            for label in OPEN_WINDOWS
            if self.straightness(label) is not None
        }
        rotations = {
            step.label: self.rotation(step.label)
            for step in self.timeline.steps
            if self.rotation(step.label) is not None
        }
        measurements = {"fist": fist, "open": opens, "rotation_deg": rotations}
        if fist is None or not opens or "wrist_rotate" not in rotations:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "fist, open-hand or wrist_rotate windows not measurable",
                measurements,
            )
        problems = [
            f"{label} is more curled than fist"
            for label, value in opens.items()
            if value <= fist
        ]
        turned = max(rotations, key=rotations.get)
        if turned != "wrist_rotate":
            problems.append(f"{turned} turns more than wrist_rotate")
        if not problems:
            return Outcome(Status.PASS, "windows hold their own motion", measurements)
        return Outcome(
            Status.FAIL,
            f"the script was performed out of order: {'; '.join(problems)}",
            measurements,
        )


class _OpenHandExtension(_Features):
    name = "posture.open_hand_extension"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Fingers are straight in both open-hand windows"
    labels = OPEN_WINDOWS

    # Placeholder.
    MIN_STRAIGHTNESS = 0.80

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def judge(self) -> Outcome:
        values = {label: self.straightness(label) for label in OPEN_WINDOWS}
        measured = {k: v for k, v in values.items() if v is not None}
        if not measured:
            return Outcome(Status.INSUFFICIENT_DATA, "no open-hand window measurable")
        bent = [
            f"{k} {v:.2f}" for k, v in measured.items() if v < self.MIN_STRAIGHTNESS
        ]
        measurements = {"straightness": measured}
        if not bent:
            shown = ", ".join(f"{k} {v:.2f}" for k, v in measured.items())
            return Outcome(Status.PASS, f"straightness {shown}", measurements)
        return Outcome(
            Status.FAIL,
            f"fingers not extended: {', '.join(bent)} (needs {self.MIN_STRAIGHTNESS})",
            measurements,
        )


class _FistClosure(_Features):
    name = "posture.fist_closure"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Fingers are fully curled in the fist window"
    labels = ("fist",)

    # Placeholder. A shallow fist and a stuck sensor look the same in one take, so a
    # miss asks for a retake rather than blaming the device.
    MAX_STRAIGHTNESS = 0.60

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def judge(self) -> Outcome:
        value = self.straightness("fist")
        if value is None:
            return Outcome(Status.INSUFFICIENT_DATA, "fist window not measurable")
        measurements = {"straightness": value}
        if value <= self.MAX_STRAIGHTNESS:
            return Outcome(Status.PASS, f"straightness {value:.2f}", measurements)
        return Outcome(
            Status.FAIL,
            f"fist straightness {value:.2f} (needs at most {self.MAX_STRAIGHTNESS})",
            measurements,
        )


class _PinchFingerIdentity(_WindowCheck):
    name = "pinch.finger_identity"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "In each pinch, the fingertip nearest the thumb tip is the named finger"
    labels = tuple(PINCHES)

    MIN_RATE = 0.6
    MIN_FRAMES = 5
    # Beyond this nothing is being pinched and the nearest tip means nothing.
    CONTACT_M = 0.04

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def measure(self, frame: Frame, label: str) -> Any:
        return _pinch_distances(frame)

    def judge(self) -> Outcome:
        rates, wrong, unmeasured = {}, {}, []
        for label, finger in PINCHES.items():
            frames = self.values.get(label, [])
            touching = [d for d in frames if min(d.values()) <= self.CONTACT_M]
            if len(touching) < self.MIN_FRAMES:
                unmeasured.append(label)
                continue
            nearest = [min(d, key=d.get) for d in touching]
            rates[label] = nearest.count(finger) / len(nearest)
            if rates[label] < self.MIN_RATE:
                wrong[label] = statistics.mode(nearest)
        measurements = {"match_rate": rates, "nearest_instead": wrong}
        if not rates:
            return Outcome(Status.INSUFFICIENT_DATA, "no pinch reached contact")
        if wrong:
            shown = ", ".join(f"{k} reads {v}" for k, v in wrong.items())
            return Outcome(
                Status.FAIL,
                f"{shown}: fingertip indices are assigned to the wrong fingers",
                measurements,
            )
        detail = f"{len(rates)} pinches land on the named finger"
        if unmeasured:
            detail += f" ({', '.join(unmeasured)} never reached contact)"
        return Outcome(Status.PASS, detail, measurements)


class _PinchContactDistance(_WindowCheck):
    name = "pinch.contact_distance"
    severity = Severity.SOFT
    judged = False
    summary = "Thumb-tip to named-fingertip distance while pinching (measured only)"
    labels = tuple(PINCHES)

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def measure(self, frame: Frame, label: str) -> Any:
        distances = _pinch_distances(frame)
        return None if distances is None else distances[PINCHES[label]]

    def judge(self) -> Outcome:
        distances = {label: median(v) for label, v in self.values.items()}
        if not distances:
            return Outcome(Status.INSUFFICIENT_DATA, "no pinch window measurable")
        shown = ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in distances.items())
        return Outcome(Status.PASS, shown, {"median_distance_m": distances})


class _OpenHandDrift(_WindowCheck):
    name = "drift.open_hand_between_windows"
    severity = Severity.SOFT
    judged = False
    summary = "Hand-shape change between the first and last open hand (measured only)"
    labels = OPEN_WINDOWS

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def measure(self, frame: Frame, label: str) -> Any:
        local = frame.local
        if local is None or any(local[i] is None for i in REQUIRED):
            return None
        return [local[i] for i in REQUIRED]

    def judge(self) -> Outcome:
        means = {}
        for label in OPEN_WINDOWS:
            frames = self.values.get(label)
            if frames:
                means[label] = [
                    tuple(statistics.fmean(f[j][k] for f in frames) for k in range(3))
                    for j in range(len(REQUIRED))
                ]
        if len(means) < 2:
            return Outcome(Status.INSUFFICIENT_DATA, "needs both open-hand windows")
        first, last = means[OPEN_WINDOWS[0]], means[OPEN_WINDOWS[1]]
        shifts = [norm(sub(a, b)) for a, b in zip(first, last)]
        drift = statistics.fmean(shifts)
        return Outcome(
            Status.PASS,
            f"mean joint shift {drift * 1000:.1f} mm, worst {max(shifts) * 1000:.1f} mm",
            {"mean_shift_m": drift, "max_shift_m": max(shifts)},
        )


class GraspFistApertureMatch(Check):
    name = "grasp.fist_aperture_match"
    gate = GROUP
    severity = Severity.SOFT
    judged = False
    required = False
    needs_timeline = True
    summary = "Grasping hand's opening against the grasped fist's thickness (measured)"

    depends_on = (WELLFORMED, ALIGNMENT, ORDER)

    def __init__(self, timeline: StepTimeline | None = None) -> None:
        super().__init__()
        self.timeline = timeline
        self.aperture: dict[str, list[float]] = {}
        self.thickness: dict[str, list[float]] = {}

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label not in GRASPS:
            return
        if not settled(step, frame.sample_time_ns):
            return
        grasper, fist = GRASPS[step.label]
        if frame.side == grasper:
            thumb, middle = frame.position(TIP["thumb"]), frame.position(TIP["middle"])
            if thumb is not None and middle is not None:
                self.aperture.setdefault(step.label, []).append(
                    norm(sub(middle, thumb))
                )
        elif frame.side == fist:
            local = frame.local
            if local is not None and all(local[i] is not None for i in REQUIRED):
                heights = [local[i][1] for i in REQUIRED]
                self.thickness.setdefault(step.label, []).append(
                    max(heights) - min(heights)
                )

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels")
        rows = {}
        for label in GRASPS:
            if self.aperture.get(label) and self.thickness.get(label):
                aperture = median(self.aperture[label])
                thickness = median(self.thickness[label])
                rows[label] = {
                    "aperture_m": aperture,
                    "fist_thickness_m": thickness,
                    "ratio": aperture / thickness if thickness > 1e-6 else None,
                }
        if not rows:
            return Outcome(Status.INSUFFICIENT_DATA, "no grasp window measurable")
        shown = ", ".join(
            f"{k} {v['aperture_m'] * 100:.1f}/{v['fist_thickness_m'] * 100:.1f} cm"
            for k, v in rows.items()
        )
        return Outcome(Status.PASS, shown, {"grasps": rows})


StepActuallyPerformed = per_hand(_StepActuallyPerformed, GROUP)
StepOrderMatchesLabels = per_hand(_StepOrderMatchesLabels, GROUP)
OpenHandExtension = per_hand(_OpenHandExtension, GROUP)
FistClosure = per_hand(_FistClosure, GROUP)
PinchFingerIdentity = per_hand(_PinchFingerIdentity, GROUP)
PinchContactDistance = per_hand(_PinchContactDistance, GROUP)
OpenHandDrift = per_hand(_OpenHandDrift, GROUP)

CHECKS = (
    LabelWindowsWellformed,
    LabelAlignment,
    StepActuallyPerformed,
    StepOrderMatchesLabels,
    OpenHandExtension,
    FistClosure,
    PinchFingerIdentity,
    PinchContactDistance,
    OpenHandDrift,
    GraspFistApertureMatch,
)
