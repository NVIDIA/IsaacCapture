# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks read through the label windows.

Held poses are measured on settled frames only; the roll windows on every frame.
Thresholds marked provisional are loose.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any, ClassVar

from cts_common.checks import segmentation as _shared
from cts_common.checks.base import (
    Attribution,
    Check,
    Outcome,
    Severity,
    Status,
)
from cts_common.vectors import norm, sub

from ..frames import Frame, Vec3
from ..labels import (
    FLAT_WINDOWS,
    PINCHES,
    ROLLS,
    SCRIPT,
    Step,
    StepTimeline,
    settled,
)
from ..profile import CHAINS, FINGERS, JOINT_NAMES, REQUIRED, TIP, WRIST
from .base import (
    HandCheck,
    angle_deg,
    fit_plane,
    hand_axes,
    hand_straightness,
    median,
    palm_centre,
    palmar_normal,
    per_hand,
    percentile,
)

GROUP = "windows"

WELLFORMED = "segmentation.label_windows_wellformed"
ALIGNMENT = "segmentation.label_alignment"
ORDER = "segmentation.step_order_matches_labels"
WINDOWED = (WELLFORMED, ALIGNMENT, ORDER)

UP: Vec3 = (0.0, 1.0, 0.0)
# PROXIMAL..TIP of the four fingers: what lies on the table. The thumb rests on its side.
FLAT_JOINTS = tuple(i for finger in FINGERS for i in CHAINS[finger])
# Cosine of 45 deg: "pointing that way" for the gross is-it-performed gates.
COS_45 = 0.7071


class LabelWindowsWellformed(_shared.LabelWindowsWellformed):
    group = GROUP


class LabelAlignment(_shared.LabelAlignment):
    group = GROUP


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


def _flat_heights(frame: Frame) -> list[Vec3] | None:
    points = [frame.position(i) for i in FLAT_JOINTS]
    return None if any(p is None for p in points) else points  # type: ignore[return-value]


# Finger joint centres sit about half a finger's thickness above the surface they rest on.
JOINT_ABOVE_TABLE_M = 0.008


def flat_height(frame: Frame) -> float | None:
    """Mean height of the four fingers' joints while the palm faces down, else None."""
    points, normal = _flat_heights(frame), palmar_normal(frame)
    if points is None or normal is None or normal[1] > -COS_45:
        return None
    return _mean_height(points)


def table_floor(heights: list[float]) -> float:
    """The table surface under a flat window's palm-down finger heights."""
    return percentile(heights, 0.1) - JOINT_ABOVE_TABLE_M


# Frames within this height of a window's lowest are on the table, metres.
TABLE_BAND_M = 0.05
# Smallest share of a flat window's frames that must be on the table.
ON_TABLE_MIN = 0.5


# Wrist speed at or below which a hand counts as at rest, m/s.
REST_SPEED_M_S = 0.05


# Span over which wrist speed is measured, nanoseconds.
REST_SPAN_NS = 100_000_000


class _Rest:
    """Whether each side's wrist moved slower than ``REST_SPEED_M_S`` over the last
    ``REST_SPAN_NS``."""

    def __init__(self) -> None:
        self._history: dict[str | None, deque[tuple[int, Vec3]]] = {}

    def still(self, frame: Frame) -> bool:
        wrist, now = frame.position(WRIST), frame.sample_time_ns
        if wrist is None or now is None:
            return False
        history = self._history.setdefault(frame.side, deque())
        history.append((now, wrist))
        while len(history) > 1 and now - history[1][0] >= REST_SPAN_NS:
            history.popleft()
        then, where = history[0]
        if now - then < REST_SPAN_NS:
            return False
        return norm(sub(wrist, where)) / ((now - then) / 1e9) <= REST_SPEED_M_S


def _on_table(heights: list[float]) -> list[bool] | None:
    """Which frames of one flat window rest on the table, or None if most did not."""
    floor = percentile(heights, 0.1)
    mask = [h <= floor + TABLE_BAND_M for h in heights]
    return mask if sum(mask) >= ON_TABLE_MIN * len(heights) else None


def _mean_height(points: list[Vec3]) -> float:
    return statistics.fmean(p[1] for p in points)


def _measured(step: Step, sample_ns: int | None) -> bool:
    """Held windows count from their settled part; motion windows from the start."""
    return sample_ns is not None and (
        not step.is_still_window or settled(step, sample_ns)
    )


class _WindowCheck(HandCheck):
    """Collects one value per measured frame of each window this check reads."""

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
        if not _measured(step, frame.sample_time_ns):
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
        return (
            _straightness(frame),
            _pinch_distances(frame),
            hand_axes(frame),
            palmar_normal(frame),
        )

    def straightness(self, label: str) -> float | None:
        values = [v[0] for v in self.values.get(label, ()) if v[0] is not None]
        return median(values) if values else None

    def rotation(self, label: str) -> float | None:
        axes = [v[2] for v in self.values.get(label, ()) if v[2] is not None]
        return _rotation_range(axes) if axes else None

    def pointing_up(self, label: str) -> float | None:
        """Median upward component of the hand's wrist-to-knuckle direction."""
        values = [v[2][0][1] for v in self.values.get(label, ()) if v[2] is not None]
        return median(values) if values else None

    def palm_up(self, label: str) -> float | None:
        """Median upward component of the palmar normal; -1 is palm down."""
        values = [v[3][1] for v in self.values.get(label, ()) if v[3] is not None]
        return median(values) if values else None

    def nearest_pinch(self, label: str) -> float | None:
        distances = [v[1] for v in self.values.get(label, ()) if v[1] is not None]
        if not distances:
            return None
        return min(median([d[f] for d in distances]) for f in FINGERS)


class _StepActuallyPerformed(_Features):
    name = "segmentation.labelled_step_actually_performed"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Each labelled window shows at least the gross motion its label names"

    # Limits for the gross motion each label names.
    OPEN_MIN = 0.6
    FIST_DROP = 0.15
    FIST_MAX = 0.75
    PINCH_MAX_M = 0.04
    ROTATE_MIN_DEG = 60.0

    depends_on = (WELLFORMED, ALIGNMENT)

    def _ok(self, label: str, fist_limit: float) -> tuple[float | None, bool]:
        straight = self.straightness(label)
        if label in FLAT_WINDOWS:
            palm = self.palm_up(label)
            return straight, (
                straight is not None
                and straight >= self.OPEN_MIN
                and palm is not None
                and palm <= -COS_45
            )
        if label == "fist":
            return straight, straight is not None and straight <= fist_limit
        if label in PINCHES:
            value = self.nearest_pinch(label)
            return value, value is not None and value <= self.PINCH_MAX_M
        if label == "palms_together":
            up = self.pointing_up(label)
            return straight, (
                straight is not None
                and straight >= self.OPEN_MIN
                and up is not None
                and up >= COS_45
            )
        if label == "tips_together":
            up = self.pointing_up(label)
            return straight, (
                straight is not None
                and straight >= self.OPEN_MIN
                and up is not None
                and abs(up) <= COS_45
            )
        value = self.rotation(label)
        return value, value is not None and value >= self.ROTATE_MIN_DEG

    def judge(self) -> Outcome:
        reference = self.straightness(FLAT_WINDOWS[0])
        fist_limit = (
            self.FIST_MAX
            if reference is None
            else min(self.FIST_MAX, reference - self.FIST_DROP)
        )
        observed: dict[str, float] = {}
        missed: list[str] = []
        for step in self.timeline.steps:
            label = step.label
            if label not in SCRIPT or ROLLS.get(label, self.side) != self.side:
                continue
            value, ok = self._ok(label, fist_limit)
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
    summary = (
        "The fist is more curled than the flat windows; this hand's roll turns most"
    )

    depends_on = (WELLFORMED, ALIGNMENT)

    def judge(self) -> Outcome:
        own_roll = next(label for label, side in ROLLS.items() if side == self.side)
        other_roll = next(label for label, side in ROLLS.items() if side != self.side)
        fist = self.straightness("fist")
        flats = {
            label: self.straightness(label)
            for label in FLAT_WINDOWS
            if self.straightness(label) is not None
        }
        rotations = {
            label: self.rotation(label)
            for label in (own_roll, other_roll)
            if self.rotation(label) is not None
        }
        measurements = {"fist": fist, "flat": flats, "rotation_deg": rotations}
        if fist is None or not flats or own_roll not in rotations:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"fist, flat-hand or {own_roll} windows not measurable",
                measurements,
            )
        problems = [
            f"{label} is more curled than fist"
            for label, value in flats.items()
            if value <= fist
        ]
        if rotations.get(other_roll, 0.0) > rotations[own_roll]:
            problems.append(f"{other_roll} turns more than {own_roll}")
        if not problems:
            return Outcome(Status.PASS, "windows hold their own motion", measurements)
        return Outcome(
            Status.FAIL,
            f"the script was performed out of order: {'; '.join(problems)}",
            measurements,
        )


class _FlatHandExtension(_Features):
    name = "posture.flat_hand_extension"
    severity = Severity.HARD
    attribution = Attribution.PERFORMANCE
    summary = "Fingers are straight in both flat-on-the-table windows"
    labels = FLAT_WINDOWS

    # Provisional.
    MIN_STRAIGHTNESS = 0.80

    depends_on = WINDOWED

    def judge(self) -> Outcome:
        values = {label: self.straightness(label) for label in FLAT_WINDOWS}
        measured = {k: v for k, v in values.items() if v is not None}
        if not measured:
            return Outcome(Status.INSUFFICIENT_DATA, "no flat-hand window measurable")
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

    # Provisional.
    MAX_STRAIGHTNESS = 0.60

    depends_on = WINDOWED

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
    # Thumb-tip to fingertip distance within which a frame counts as pinching, metres.
    CONTACT_M = 0.04

    depends_on = WINDOWED

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

    depends_on = WINDOWED

    def measure(self, frame: Frame, label: str) -> Any:
        distances = _pinch_distances(frame)
        return None if distances is None else distances[PINCHES[label]]

    def judge(self) -> Outcome:
        distances = {label: median(v) for label, v in self.values.items()}
        if not distances:
            return Outcome(Status.INSUFFICIENT_DATA, "no pinch window measurable")
        shown = ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in distances.items())
        return Outcome(Status.PASS, shown, {"median_distance_m": distances})


class _FlatOnTable(_WindowCheck):
    """Per flat window, the mean position of each finger joint over frames lying flat.

    A frame counts once the palm faces down, the fingers are straight, and the hand is
    still at the table's height.
    """

    labels = FLAT_WINDOWS
    depends_on = WINDOWED

    def __init__(self, side: str, timeline: StepTimeline | None = None) -> None:
        super().__init__(side, timeline)
        self.rest = _Rest()

    def measure(self, frame: Frame, label: str) -> Any:
        points, normal = _flat_heights(frame), palmar_normal(frame)
        straight = _straightness(frame)
        if not self.rest.still(frame):
            return None
        if points is None or normal is None or straight is None:
            return None
        if normal[1] > -COS_45 or straight < _StepActuallyPerformed.OPEN_MIN:
            return None
        return points

    def planes(self) -> dict[str, tuple[float, float, str]]:
        """Per window held on the table: median over its resting frames of the plane's
        tilt from up, its largest residual, and the joint that most often has it.

        Fitted frame by frame; never from a joint-wise mean over the window.
        """
        out = {}
        for label, frames in self.values.items():
            mask = _on_table([_mean_height(f) for f in frames])
            if mask is None:
                continue
            tilts, worst, names = [], [], []
            for points, keep in zip(frames, mask):
                if not keep:
                    continue
                _, normal, residuals = fit_plane(points, UP)
                index = max(range(len(residuals)), key=lambda i: abs(residuals[i]))
                tilts.append(angle_deg(normal, UP))
                worst.append(abs(residuals[index]))
                names.append(JOINT_NAMES[FLAT_JOINTS[index]])
            out[label] = (median(tilts), median(worst), statistics.mode(names))
        return out


class _FingersCoplanar(_FlatOnTable):
    name = "table.fingers_coplanar"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "Finger joints of a hand flat on the table lie in one plane"

    # Provisional. Largest joint distance from the fitted plane, metres.
    MAX_RESIDUAL_M = 0.015

    def judge(self) -> Outcome:
        planes = self.planes()
        if not planes:
            return Outcome(
                Status.INSUFFICIENT_DATA, "no window was held flat on the table"
            )
        worst = {label: (v[1], v[2]) for label, v in planes.items()}
        measurements = {
            "max_residual_m": {k: v[0] for k, v in worst.items()},
            "worst_joint": {k: v[1] for k, v in worst.items()},
        }
        shown = ", ".join(
            f"{k} {v[0] * 1000:.1f} mm ({v[1]})" for k, v in worst.items()
        )
        bent = [k for k, v in worst.items() if v[0] > self.MAX_RESIDUAL_M]
        if not bent:
            return Outcome(
                Status.PASS, f"worst joint off the plane: {shown}", measurements
            )
        return Outcome(
            Status.FAIL,
            f"finger joints leave the table plane: {shown} "
            f"(allowed {self.MAX_RESIDUAL_M * 1000:.0f} mm)",
            measurements,
        )


class _TableLevel(_FlatOnTable):
    name = "table.level"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "The plane of a hand flat on the table is level in the world"

    # Provisional. Largest tilt of the plane from up, degrees.
    MAX_TILT_DEG = 10.0

    def judge(self) -> Outcome:
        planes = self.planes()
        if not planes:
            return Outcome(
                Status.INSUFFICIENT_DATA, "no window was held flat on the table"
            )
        tilts = {label: v[0] for label, v in planes.items()}
        shown = ", ".join(f"{k} {v:.1f} deg" for k, v in tilts.items())
        measurements = {"tilt_deg": tilts}
        if all(v <= self.MAX_TILT_DEG for v in tilts.values()):
            return Outcome(Status.PASS, f"tilt {shown}", measurements)
        return Outcome(
            Status.FAIL,
            f"the table reads tilted: {shown} (allowed {self.MAX_TILT_DEG:.0f} deg); "
            f"the world up axis or the wrist orientation is off",
            measurements,
        )


class _FlatHandDrift(_FlatOnTable):
    name = "drift.flat_hand_between_windows"
    severity = Severity.SOFT
    judged = False
    summary = (
        "Hand shape and table height, first flat window against the last (measured)"
    )

    def measure(self, frame: Frame, label: str) -> Any:
        points, local = super().measure(frame, label), frame.local
        if points is None or local is None or any(local[i] is None for i in REQUIRED):
            return None
        return ([local[i] for i in REQUIRED], _mean_height(points))

    def judge(self) -> Outcome:
        shapes, heights = {}, {}
        for label in FLAT_WINDOWS:
            frames = self.values.get(label)
            mask = _on_table([f[1] for f in frames]) if frames else None
            if mask is not None:
                frames = [f for f, keep in zip(frames, mask) if keep]
                shapes[label] = [
                    tuple(
                        statistics.fmean(f[0][j][k] for f in frames) for k in range(3)
                    )
                    for j in range(len(REQUIRED))
                ]
                heights[label] = statistics.fmean(f[1] for f in frames)
        if len(shapes) < 2:
            return Outcome(
                Status.INSUFFICIENT_DATA, "needs both flat windows held on the table"
            )
        first, last = FLAT_WINDOWS
        shifts = [norm(sub(a, b)) for a, b in zip(shapes[first], shapes[last])]
        drift = statistics.fmean(shifts)
        height = heights[last] - heights[first]
        return Outcome(
            Status.PASS,
            f"mean joint shift {drift * 1000:.1f} mm, worst {max(shifts) * 1000:.1f} "
            f"mm; table height moved {height * 1000:+.1f} mm",
            {
                "mean_shift_m": drift,
                "max_shift_m": max(shifts),
                "height_shift_m": height,
            },
        )


# Largest sample-time difference between a left and a right record that still pair,
# nanoseconds: half a 60 Hz period.
PAIR_TOLERANCE_NS = 8_000_000

NO_PAIRS = "no left/right records close enough in time to pair"


class _PairCheck(Check):
    """Both hands at once, fed left/right record pairs within ``PAIR_TOLERANCE_NS``."""

    group = GROUP
    required = False
    needs_timeline = True
    labels: ClassVar[tuple[str, ...]] = ()

    depends_on = WINDOWED

    def __init__(self, timeline: StepTimeline | None = None) -> None:
        super().__init__()
        self.timeline = timeline
        self._pending: Frame | None = None
        self.pairs = 0

    def pair(self, label: str, left: Frame, right: Frame) -> None: ...

    def judge(self) -> Outcome: ...

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.side not in ("left", "right"):
            return
        pending, self._pending = self._pending, frame
        if (
            pending is None
            or pending.side == frame.side
            or pending.sample_time_ns is None
            or frame.sample_time_ns is None
            or abs(pending.sample_time_ns - frame.sample_time_ns) > PAIR_TOLERANCE_NS
        ):
            return
        self._pending = None
        self.pairs += 1
        if pending.joints is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label not in self.labels:
            return
        if not _measured(step, frame.sample_time_ns):
            return
        left, right = (pending, frame) if pending.side == "left" else (frame, pending)
        self.pair(step.label, left, right)

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels")
        if self.pairs == 0:
            return Outcome(Status.INSUFFICIENT_DATA, NO_PAIRS)
        return self.judge()


class HandsSameHeight(_PairCheck):
    name = "table.hands_same_height"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "Both hands flat on one table sit at one height"
    labels = FLAT_WINDOWS

    # Provisional.
    MAX_DIFF_M = 0.03

    def __init__(self, timeline: StepTimeline | None = None) -> None:
        super().__init__(timeline)
        self.heights: dict[str, list[tuple[float, float]]] = {}
        self.rest = _Rest()

    def pair(self, label: str, left: Frame, right: Frame) -> None:
        heights = []
        still = [self.rest.still(left), self.rest.still(right)]
        if not all(still):
            return
        for frame in (left, right):
            height = flat_height(frame)
            if height is None:
                return
            heights.append(height)
        self.heights.setdefault(label, []).append((heights[0], heights[1]))

    def judge(self) -> Outcome:
        diffs, surfaces = {}, []
        for label, pairs in self.heights.items():
            left = _on_table([h for h, _ in pairs])
            right = _on_table([h for _, h in pairs])
            if left is None or right is None:
                continue
            kept = [
                r_h - l_h
                for (l_h, r_h), on_l, on_r in zip(pairs, left, right)
                if on_l and on_r
            ]
            if kept:
                diffs[label] = median(kept)
                surfaces.append(
                    table_floor(
                        [
                            h
                            for pair, a, b in zip(pairs, left, right)
                            if a and b
                            for h in pair
                        ]
                    )
                )
        if not diffs:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "the hands were never held flat on the table together",
            )
        shown = ", ".join(f"{k} {v * 1000:+.1f} mm" for k, v in diffs.items())
        measurements = {"right_minus_left_m": diffs, "table_height_m": median(surfaces)}
        if all(abs(v) <= self.MAX_DIFF_M for v in diffs.values()):
            return Outcome(Status.PASS, f"right above left by {shown}", measurements)
        return Outcome(
            Status.FAIL,
            f"right above left by {shown} on one table "
            f"(allowed {self.MAX_DIFF_M * 1000:.0f} mm): the hands are placed apart",
            measurements,
        )


class _PalmsTogether(_PairCheck):
    """Frames of ``palms_together`` in which the palms face each other, close."""

    labels = ("palms_together",)

    # A frame counts as palms together within this centre distance (metres) and above
    # this angle between the palm normals (degrees).
    CENTRE_MAX_M = 0.08
    FACING_MIN_DEG = 120.0
    MIN_FRAMES = 10

    def __init__(self, timeline: StepTimeline | None = None) -> None:
        super().__init__(timeline)
        self.frames: list[tuple[dict[str, Vec3], dict[str, Vec3]]] = []
        self.seen = 0

    def pair(self, label: str, left: Frame, right: Frame) -> None:
        self.seen += 1
        centres = palm_centre(left), palm_centre(right)
        normals = palmar_normal(left), palmar_normal(right)
        if None in centres or None in normals:
            return
        if norm(sub(centres[0], centres[1])) > self.CENTRE_MAX_M:
            return
        if angle_deg(normals[0], normals[1]) < self.FACING_MIN_DEG:
            return
        tips = []
        for frame in (left, right):
            found = {f: frame.position(TIP[f]) for f in FINGERS}
            if any(p is None for p in found.values()):
                return
            tips.append(found)
        self.frames.append((tips[0], tips[1]))

    def together(self) -> bool:
        return len(self.frames) >= self.MIN_FRAMES


class PalmsFingertipGap(_PalmsTogether):
    name = "palms.fingertip_gap"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = "With palms pressed together, same-named fingertips coincide"

    # Provisional. Largest median gap between same-named fingertips, metres.
    MAX_GAP_M = 0.05

    def judge(self) -> Outcome:
        if not self.together():
            return Outcome(
                Status.FAIL,
                f"the palms never came together ({len(self.frames)} of {self.seen} "
                f"frames facing and close)",
                {"frames_together": len(self.frames)},
                Attribution.PERFORMANCE,
            )
        gaps = {
            finger: median(
                [norm(sub(left[finger], right[finger])) for left, right in self.frames]
            )
            for finger in FINGERS
        }
        shown = ", ".join(f"{k} {v * 1000:.0f} mm" for k, v in gaps.items())
        measurements = {"median_gap_m": gaps}
        if all(v <= self.MAX_GAP_M for v in gaps.values()):
            return Outcome(Status.PASS, shown, measurements)
        return Outcome(
            Status.FAIL,
            f"touching fingertips read apart: {shown} "
            f"(allowed {self.MAX_GAP_M * 1000:.0f} mm)",
            measurements,
        )


class TipsRollContact(_PairCheck):
    name = "tips.roll_contact"
    severity = Severity.HARD
    attribution = Attribution.DEVICE
    summary = (
        "While one hand turns about its middle finger, its tip stays on the other's"
    )
    labels = ("tips_together", *ROLLS)

    # Provisional. Largest 95th-percentile excursion of the turning tip from the
    # tips_together reference, metres.
    MAX_EXCURSION_M = 0.03
    TOUCH_MAX_M = 0.08
    ROTATE_MIN_DEG = 90.0
    MIN_FRAMES = 30

    def __init__(self, timeline: StepTimeline | None = None) -> None:
        super().__init__(timeline)
        self.rel: dict[str, list[Vec3]] = {}
        self.axes: dict[str, list[tuple[Vec3, Vec3, Vec3]]] = {}

    def pair(self, label: str, left: Frame, right: Frame) -> None:
        moving = right if ROLLS.get(label, "right") == "right" else left
        a, b = right.position(TIP["middle"]), left.position(TIP["middle"])
        axes = hand_axes(moving)
        if a is None or b is None or axes is None:
            return
        # Always right minus left.
        self.rel.setdefault(label, []).append(sub(a, b))
        self.axes.setdefault(label, []).append(axes)

    def judge(self) -> Outcome:
        # The held tips_together pose is the reference.
        baseline = self.rel.get("tips_together", [])
        if len(baseline) < self.MIN_FRAMES:
            return Outcome(Status.INSUFFICIENT_DATA, "tips_together not measurable")
        ref = tuple(median([r[k] for r in baseline]) for k in range(3))
        measurements: dict[str, Any] = {"tips_together_gap_m": norm(ref)}
        if norm(ref) > self.TOUCH_MAX_M:
            return Outcome(
                Status.FAIL,
                f"tips_together held the middle tips {norm(ref) * 100:.0f} cm apart",
                measurements,
                Attribution.PERFORMANCE,
            )
        rows: dict[str, dict[str, float]] = {}
        device, performance = [], []
        for label in ROLLS:
            rel = self.rel.get(label, [])
            if len(rel) < self.MIN_FRAMES:
                continue
            excursion = percentile([norm(sub(r, ref)) for r in rel], 0.95)
            rotation = _rotation_range(self.axes[label])
            rows[label] = {"excursion_p95_m": excursion, "rotation_deg": rotation}
            if rotation < self.ROTATE_MIN_DEG:
                performance.append(f"{label} turned only {rotation:.0f} deg")
            elif excursion > self.MAX_EXCURSION_M:
                device.append(label)
        measurements["rolls"] = rows
        if not rows:
            return Outcome(
                Status.INSUFFICIENT_DATA, "no roll window measurable", measurements
            )
        shown = ", ".join(
            f"{k} {v['excursion_p95_m'] * 1000:.0f} mm over {v['rotation_deg']:.0f} deg"
            for k, v in rows.items()
        )
        if device:
            return Outcome(
                Status.FAIL,
                f"the turning fingertip leaves the still one: {shown} "
                f"(allowed {self.MAX_EXCURSION_M * 1000:.0f} mm); "
                f"the turning hand's rotation axis is misplaced",
                measurements,
            )
        if performance:
            return Outcome(
                Status.FAIL,
                "; ".join(performance),
                measurements,
                Attribution.PERFORMANCE,
            )
        return Outcome(Status.PASS, f"tip excursion {shown}", measurements)


StepActuallyPerformed = per_hand(_StepActuallyPerformed, GROUP)
StepOrderMatchesLabels = per_hand(_StepOrderMatchesLabels, GROUP)
FlatHandExtension = per_hand(_FlatHandExtension, GROUP)
FistClosure = per_hand(_FistClosure, GROUP)
PinchFingerIdentity = per_hand(_PinchFingerIdentity, GROUP)
PinchContactDistance = per_hand(_PinchContactDistance, GROUP)
FingersCoplanar = per_hand(_FingersCoplanar, GROUP)
TableLevel = per_hand(_TableLevel, GROUP)
FlatHandDrift = per_hand(_FlatHandDrift, GROUP)

CHECKS = (
    LabelWindowsWellformed,
    LabelAlignment,
    StepActuallyPerformed,
    StepOrderMatchesLabels,
    FlatHandExtension,
    FistClosure,
    PinchFingerIdentity,
    PinchContactDistance,
    FingersCoplanar,
    TableLevel,
    HandsSameHeight,
    FlatHandDrift,
    PalmsFingertipGap,
    TipsRollContact,
)
