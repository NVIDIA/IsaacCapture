# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-shape checks: what the glove itself reports, independent of wrist placement.

Every check here reads one hand. Thresholds marked provisional are loose.
"""

from __future__ import annotations

import math
import statistics
from typing import ClassVar

from cts_common.checks.base import Outcome, Severity, Status
from cts_common.vectors import cross, dot, norm, sub

from ..frames import NUM_JOINTS, Frame
from ..profile import (
    CHAINS,
    FINGERS,
    INDEX,
    JOINT_NAMES,
    METACARPALS,
    OPTIONAL,
    PALM,
    PROXIMAL,
    REQUIRED,
    TIP,
    WRIST,
    judged_bones,
)
from .base import HandCheck, bone_length, hand_straightness, median, per_hand

GROUP = "shape"

COMPONENT_ORDER = "quaternion.component_order"


# Fraction of a take taken as the curled set; the script holds one fist, a few percent.
CURLED_FRACTION = 0.03


def _extremes(
    samples: list[tuple[float, float]], fraction: float = 0.2
) -> tuple[list[float], list[float], float]:
    """Values from the most-curled and most-extended ``fraction`` of frames.

    ``samples`` pairs a straightness with the value to split; the third result is the
    straightness range between the two groups, which says whether the take curled at all.
    """
    ordered = sorted(samples)
    count = max(1, int(fraction * len(ordered)))
    curled, extended = ordered[:count], ordered[-count:]
    spread = statistics.mean(s for s, _ in extended) - statistics.mean(
        s for s, _ in curled
    )
    return [v for _, v in curled], [v for _, v in extended], spread


class _RequiredJointsValid(HandCheck):
    name = "coverage.required_joints_valid"
    severity = Severity.HARD
    summary = "All 21 judged joints are valid whenever the hand is active"

    # Provisional.
    MIN_RATE = 0.95

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.active = 0
        self.complete = 0
        self.missing: dict[int, int] = {}

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        self.active += 1
        absent = [i for i in REQUIRED if not frame.joints[i].is_valid]
        if not absent:
            self.complete += 1
        for i in absent:
            self.missing[i] = self.missing.get(i, 0) + 1

    def _result(self) -> Outcome:
        if self.active == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "the hand was never active")
        rate = self.complete / self.active
        worst = sorted(self.missing.items(), key=lambda kv: -kv[1])[:5]
        measurements = {
            "active_frames": self.active,
            "complete_rate": rate,
            "most_often_invalid": {JOINT_NAMES[i]: n / self.active for i, n in worst},
        }
        if rate >= self.MIN_RATE:
            return Outcome(Status.PASS, f"{rate:.1%} of active frames", measurements)
        names = ", ".join(JOINT_NAMES[i] for i, _ in worst)
        return Outcome(
            Status.FAIL,
            f"only {rate:.1%} of active frames have all 21 (most often missing: "
            f"{names}); a retargeter consuming these drops or zero-fills the rest",
            measurements,
        )


class _All26ValidRate(HandCheck):
    name = "coverage.all_26_valid_rate"
    severity = Severity.SOFT
    judged = False
    required = False
    summary = "How often each of the 26 OpenXR joints is valid (measured only)"

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.active = 0
        self.all_valid = 0
        self.valid = [0] * NUM_JOINTS

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        self.active += 1
        flags = [joint.is_valid for joint in frame.joints]
        self.all_valid += all(flags)
        for i, flag in enumerate(flags):
            self.valid[i] += flag

    def _result(self) -> Outcome:
        if self.active == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "the hand was never active")
        per_joint = {JOINT_NAMES[i]: n / self.active for i, n in enumerate(self.valid)}
        never = [name for name, rate in per_joint.items() if rate == 0.0]
        detail = f"all 26 valid in {self.all_valid / self.active:.1%} of active frames"
        if never:
            detail += f"; never valid: {', '.join(never)}"
        return Outcome(
            Status.PASS,
            detail,
            {"all_26_rate": self.all_valid / self.active, "per_joint": per_joint},
        )


class _ValidSetStable(HandCheck):
    name = "consistency.valid_set_stable"
    severity = Severity.HARD
    summary = "The set of valid joints does not change mid-take (no source switch)"

    # A different valid set must persist this long to count, seconds.
    MIN_RUN_S = 0.5

    def __init__(self, side: str) -> None:
        super().__init__(side)
        # [valid set, first sample ns, last sample ns, frames]
        self.runs: list[list] = []

    def _update(self, frame: Frame) -> None:
        if frame.joints is None or frame.sample_time_ns is None:
            return
        valid = frozenset(i for i, _ in frame.valid_joints())
        if not valid:
            return
        t = frame.sample_time_ns
        if self.runs and self.runs[-1][0] == valid:
            self.runs[-1][2] = t
            self.runs[-1][3] += 1
        else:
            self.runs.append([valid, t, t, 1])

    def _result(self) -> Outcome:
        if not self.runs:
            return Outcome(Status.INSUFFICIENT_DATA, "no frame with a valid joint")
        sustained = [
            run for run in self.runs if (run[2] - run[1]) / 1e9 >= self.MIN_RUN_S
        ]
        switches = []
        for earlier, later in zip(sustained, sustained[1:]):
            if earlier[0] != later[0]:
                gained = sorted(later[0] - earlier[0])
                lost = sorted(earlier[0] - later[0])
                switches.append(
                    {
                        "at_s": (later[1] - self.runs[0][1]) / 1e9,
                        "gained": [JOINT_NAMES[i] for i in gained],
                        "lost": [JOINT_NAMES[i] for i in lost],
                    }
                )
        distinct = {run[0] for run in self.runs}
        measurements = {
            "distinct_sets": len(distinct),
            "sustained_runs": len(sustained),
            "switches": switches,
        }
        if not switches:
            return Outcome(
                Status.PASS,
                f"one sustained valid set of {len(sustained[0][0]) if sustained else 0}"
                f" joints ({len(distinct)} distinct sets including brief flickers)",
                measurements,
            )
        first = switches[0]
        return Outcome(
            Status.FAIL,
            f"the valid set changes {len(switches)} time(s) and holds, first at "
            f"{first['at_s']:.1f} s (gained {first['gained'] or '-'}, lost "
            f"{first['lost'] or '-'}): the data source switched mid-take",
            measurements,
        )


class _RadiusPlausible(HandCheck):
    name = "joints.radius_plausible"
    severity = Severity.SOFT
    summary = "Valid joints carry a finite, positive radius of plausible size"

    MIN_M = 0.001
    MAX_M = 0.05
    MAX_BAD_RATE = 0.01

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.samples = 0
        self.bad = 0
        self.low = math.inf
        self.high = -math.inf

    def _update(self, frame: Frame) -> None:
        for _, joint in frame.valid_joints():
            self.samples += 1
            r = joint.radius
            if not math.isfinite(r) or not self.MIN_M <= r <= self.MAX_M:
                self.bad += 1
            if math.isfinite(r):
                self.low, self.high = min(self.low, r), max(self.high, r)

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no valid joints")
        rate = self.bad / self.samples
        measurements = {"bad_rate": rate, "min_m": self.low, "max_m": self.high}
        span = f"{self.low * 1000:.1f}-{self.high * 1000:.1f} mm"
        if rate <= self.MAX_BAD_RATE:
            return Outcome(Status.PASS, f"radii {span}", measurements)
        return Outcome(
            Status.FAIL,
            f"{rate:.1%} of valid joints have a radius outside "
            f"{self.MIN_M * 1000:.0f}-{self.MAX_M * 1000:.0f} mm (seen {span})",
            measurements,
        )


def _hand_length(frame: Frame) -> float | None:
    chain = (WRIST, *CHAINS["middle"])
    positions = [frame.position(i) for i in chain]
    if any(p is None for p in positions):
        return None
    return sum(norm(sub(b, a)) for a, b in zip(positions, positions[1:]))


class _PositionScaleMetres(HandCheck):
    name = "units.position_scale_metres"
    severity = Severity.HARD
    summary = "Wrist-to-middle-tip length is an adult hand's, in metres"

    # Accepted hand length, metres, summed along the joints. Provisional.
    MIN_M = 0.10
    MAX_M = 0.30

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.lengths: list[float] = []

    def _update(self, frame: Frame) -> None:
        length = _hand_length(frame)
        if length is not None and math.isfinite(length):
            self.lengths.append(length)

    def _result(self) -> Outcome:
        if not self.lengths:
            return Outcome(
                Status.INSUFFICIENT_DATA, "wrist-to-middle chain never valid"
            )
        length = median(self.lengths)
        measurements = {"median_hand_length_m": length}
        if self.MIN_M <= length <= self.MAX_M:
            return Outcome(Status.PASS, f"{length * 100:.1f} cm", measurements)
        return Outcome(
            Status.FAIL,
            f"hand length reads {length:.4g} (expected {self.MIN_M}-{self.MAX_M} m); "
            f"positions are not in metres",
            measurements,
        )


class _BoneLengthConstancy(HandCheck):
    name = "skeleton.bone_length_constancy"
    severity = Severity.SOFT
    summary = "Each bone keeps its length through the take"

    # Provisional.
    MAX_CV = 0.20

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.lengths: dict[tuple[int, int], list[float]] = {}

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        positions = [frame.position(i) for i in range(NUM_JOINTS)]
        for bone in judged_bones():
            length = bone_length(positions, *bone)
            if length is not None and math.isfinite(length):
                self.lengths.setdefault(bone, []).append(length)

    def _result(self) -> Outcome:
        cvs = {
            bone: statistics.pstdev(values) / statistics.mean(values)
            for bone, values in self.lengths.items()
            if len(values) >= 2 and statistics.mean(values) > 1e-6
        }
        if not cvs:
            return Outcome(Status.INSUFFICIENT_DATA, "no bone seen twice")
        worst_bone, worst = max(cvs.items(), key=lambda kv: kv[1])
        label = f"{JOINT_NAMES[worst_bone[0]]}->{JOINT_NAMES[worst_bone[1]]}"
        measurements = {
            "max_cv": worst,
            "worst_bone": label,
            "per_bone_cv": {
                f"{JOINT_NAMES[a]}->{JOINT_NAMES[b]}": cv for (a, b), cv in cvs.items()
            },
        }
        if worst <= self.MAX_CV:
            return Outcome(Status.PASS, f"worst CV {worst:.3f} ({label})", measurements)
        return Outcome(
            Status.FAIL,
            f"{label} varies by CV {worst:.2f} (limit {self.MAX_CV}); the skeleton "
            f"is not rigid",
            measurements,
        )


def _digit_length(frame: Frame, digit: str) -> float | None:
    positions = [frame.position(i) for i in CHAINS[digit]]
    if any(p is None for p in positions):
        return None
    return sum(norm(sub(b, a)) for a, b in zip(positions, positions[1:]))


class _AnthropometricPlausibility(HandCheck):
    name = "skeleton.anthropometric_plausibility"
    severity = Severity.SOFT
    summary = "Finger proportions fall within human priors"

    # Provisional priors on medians. Digit lengths run root joint to tip; the palm is
    # WRIST to MIDDLE_PROXIMAL.
    RATIOS: ClassVar[dict[str, tuple[str, str, float, float]]] = {
        "middle_over_palm": ("middle", "palm", 0.6, 1.5),
        "little_over_middle": ("little", "middle", 0.55, 1.0),
        "thumb_over_middle": ("thumb", "middle", 0.6, 1.5),
    }

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.samples: dict[str, list[float]] = {}

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        values = {digit: _digit_length(frame, digit) for digit in CHAINS}
        wrist, knuckle = frame.position(WRIST), frame.position(PROXIMAL["middle"])
        values["palm"] = (
            None if wrist is None or knuckle is None else norm(sub(knuckle, wrist))
        )
        for key, value in values.items():
            if value is not None and math.isfinite(value):
                self.samples.setdefault(key, []).append(value)

    def _result(self) -> Outcome:
        lengths = {key: median(values) for key, values in self.samples.items()}
        ratios, out = {}, []
        for ratio, (top, bottom, low, high) in self.RATIOS.items():
            if lengths.get(top) and lengths.get(bottom):
                value = lengths[top] / lengths[bottom]
                ratios[ratio] = value
                if not low <= value <= high:
                    out.append(f"{ratio} {value:.2f} (expected {low}-{high})")
        if not ratios:
            return Outcome(Status.INSUFFICIENT_DATA, "no complete digit chain")
        measurements = {"lengths_m": lengths, "ratios": ratios}
        if not out:
            shown = ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
            return Outcome(Status.PASS, shown, measurements)
        return Outcome(Status.FAIL, "; ".join(out), measurements)


class _JointIndexAssignment(HandCheck):
    name = "skeleton.joint_index_assignment"
    severity = Severity.HARD
    summary = "Fingers sit in index-to-little order and each chain runs root to tip"

    # Across the palm the knuckles must project in index, middle, ring, little order along
    # index->little, with THUMB_METACARPAL on the index half. Along each digit, distance
    # from the wrist must grow root to tip, judged on the most extended frames.
    MIN_RATE = 0.9
    THUMB_SIDE_MAX_T = 0.5
    EXTENDED_FRACTION = 0.25

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.palm_frames = 0
        self.order_ok = 0
        self.thumb_ok = 0
        self.chain_samples: list[tuple[float, float]] = []
        self.first_bad_chain: dict[str, int] = {}

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        p = [frame.position(i) for i in range(NUM_JOINTS)]
        if any(p[i] is None for i in REQUIRED):
            return
        index, little = p[PROXIMAL["index"]], p[PROXIMAL["little"]]
        across = sub(little, index)
        span = dot(across, across)
        if span > 1e-9:
            self.palm_frames += 1
            t_middle = dot(sub(p[PROXIMAL["middle"]], index), across) / span
            t_ring = dot(sub(p[PROXIMAL["ring"]], index), across) / span
            t_thumb = dot(sub(p[CHAINS["thumb"][0]], index), across) / span
            self.order_ok += 0.0 < t_middle < t_ring < 1.0
            self.thumb_ok += t_thumb < self.THUMB_SIDE_MAX_T

        straight = hand_straightness(p)
        if straight is None:
            return
        wrist = p[WRIST]
        ok = True
        for digit, chain in CHAINS.items():
            reach = [norm(sub(p[i], wrist)) for i in chain]
            if not all(a < b for a, b in zip(reach, reach[1:])):
                ok = False
                self.first_bad_chain[digit] = self.first_bad_chain.get(digit, 0) + 1
        self.chain_samples.append((straight, 1.0 if ok else 0.0))

    def _result(self) -> Outcome:
        if self.palm_frames == 0 or not self.chain_samples:
            return Outcome(Status.INSUFFICIENT_DATA, "no frame with all 21 joints")
        order_rate = self.order_ok / self.palm_frames
        thumb_rate = self.thumb_ok / self.palm_frames
        _, extended, _ = _extremes(self.chain_samples, self.EXTENDED_FRACTION)
        chain_rate = statistics.mean(extended)
        measurements = {
            "across_palm_order_rate": order_rate,
            "thumb_side_rate": thumb_rate,
            "chain_order_rate_extended": chain_rate,
            "chains_out_of_order_frames": self.first_bad_chain,
        }
        problems = []
        if order_rate < self.MIN_RATE:
            problems.append(
                f"knuckles are in index-middle-ring-little order in only "
                f"{order_rate:.0%} of frames"
            )
        if thumb_rate < self.MIN_RATE:
            problems.append(
                f"the thumb base is on the index half of the palm in only "
                f"{thumb_rate:.0%} of frames"
            )
        if chain_rate < self.MIN_RATE:
            digits = ", ".join(sorted(self.first_bad_chain))
            problems.append(
                f"root-to-tip order holds in only {chain_rate:.0%} of extended frames "
                f"({digits})"
            )
        if not problems:
            return Outcome(
                Status.PASS,
                f"order holds in {order_rate:.0%} of frames across the palm and "
                f"{chain_rate:.0%} of extended frames along the digits",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            "; ".join(problems) + ": joint indices are assigned to the wrong fingers",
            measurements,
        )


def _position_chirality(frame: Frame) -> tuple[float, float] | None:
    """(straightness, s): s > 0 when the fingertips sit on the right-hand palm side.

    Finger direction x across-palm is a right hand's palmar normal.
    """
    p = [frame.position(i) for i in range(NUM_JOINTS)]
    if any(p[i] is None for i in REQUIRED):
        return None
    straight = hand_straightness(p)
    if straight is None:
        return None
    forward = sub(p[PROXIMAL["middle"]], p[WRIST])
    across = sub(p[PROXIMAL["little"]], p[PROXIMAL["index"]])
    normal = cross(forward, across)
    size = norm(normal)
    if size < 1e-9:
        return None
    offsets = [sub(p[TIP[f]], p[PROXIMAL[f]]) for f in FINGERS]
    return straight, sum(dot(o, normal) for o in offsets) / (len(offsets) * size)


class _Handedness(HandCheck):
    name = "coordinate_frame.handedness"
    severity = Severity.HARD
    summary = "The hand on each channel has that side's chirality"

    # In the OpenXR joint frame X = Y x Z with +Y dorsal and +Z toward the wrist, so
    # INDEX_PROXIMAL->LITTLE_PROXIMAL in wrist coordinates has x > 0 on a right hand.
    MIN_COMPONENT = 0.3

    depends_on = (COMPONENT_ORDER,)

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.x: list[float] = []
        self.chirality: list[tuple[float, float]] = []

    def _update(self, frame: Frame) -> None:
        local = frame.local
        if local is None:
            return
        index, little = local[PROXIMAL["index"]], local[PROXIMAL["little"]]
        if index is not None and little is not None:
            across = sub(little, index)
            size = norm(across)
            if size > 1e-9:
                self.x.append(across[0] / size)
        sample = _position_chirality(frame)
        if sample is not None:
            self.chirality.append(sample)

    def _result(self) -> Outcome:
        if not self.x:
            return Outcome(Status.INSUFFICIENT_DATA, "no frame with a valid wrist")
        x = median(self.x)
        expected = 1.0 if self.side == "right" else -1.0
        measurements: dict = {"across_palm_x": x}
        cross_check = None
        if len(self.chirality) >= 10:
            curled, _, spread = _extremes(self.chirality, CURLED_FRACTION)
            if spread >= 0.25:
                cross_check = "right" if statistics.mean(curled) > 0 else "left"
        measurements["position_chirality"] = cross_check

        if abs(x) < self.MIN_COMPONENT:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the across-palm direction has x {x:+.2f} in the wrist frame, too "
                f"small to call; the wrist orientation does not follow the OpenXR "
                f"joint convention",
                measurements,
            )
        seen = "right" if x > 0 else "left"
        note = (
            ""
            if cross_check in (None, seen)
            else f" (fingertip curl reads {cross_check}; see shape.flex_direction)"
        )
        if x * expected > 0:
            return Outcome(Status.PASS, f"reads {seen}{note}", measurements)
        return Outcome(
            Status.FAIL,
            f"the {self.side} channel carries a {seen} hand (across-palm x "
            f"{x:+.2f}){note}",
            measurements,
        )


def _dorsal_tip_offset(local, finger: str) -> float | None:
    """Tip offset from the proximal phalanx's line, metres; + toward the back of the hand.

    The dorsal side is wrist-frame X crossed with the bone, so +Y on a straight finger.
    """
    root, middle, _, tip = CHAINS[finger]
    dorsal = cross((1.0, 0.0, 0.0), sub(local[middle], local[root]))
    size = norm(dorsal)
    if size < 1e-9:
        return None
    return dot(sub(local[tip], local[root]), dorsal) / size


class _FlexDirection(HandCheck):
    name = "shape.flex_direction"
    severity = Severity.HARD
    required = False
    summary = "Curling moves the fingertips toward the palm, not the back of the hand"

    # The tip's offset from its proximal phalanx's line, dorsal side positive, must fall
    # as the hand curls. A bend at the knuckle alone leaves it at zero.
    MIN_SPREAD = 0.25
    MIN_SHIFT_M = 0.01

    depends_on = (COMPONENT_ORDER,)

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.samples: list[tuple[float, float]] = []

    def _update(self, frame: Frame) -> None:
        local = frame.local
        if local is None or any(local[i] is None for i in REQUIRED):
            return
        straight = hand_straightness(local)
        if straight is None:
            return
        offsets = [_dorsal_tip_offset(local, f) for f in FINGERS]
        if any(offset is None for offset in offsets):
            return
        self.samples.append((straight, statistics.mean(offsets)))

    def _result(self) -> Outcome:
        if len(self.samples) < 10:
            return Outcome(Status.INSUFFICIENT_DATA, "too few complete frames")
        curled, extended, spread = _extremes(self.samples, CURLED_FRACTION)
        shift = statistics.mean(curled) - statistics.mean(extended)
        measurements = {"straightness_spread": spread, "tip_shift_m": shift}
        if spread < self.MIN_SPREAD:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the hand never curled (straightness spread {spread:.2f})",
                measurements,
            )
        if shift <= -self.MIN_SHIFT_M:
            return Outcome(
                Status.PASS,
                f"tips move {-shift * 100:.1f} cm palmward when curled",
                measurements,
            )
        if shift >= self.MIN_SHIFT_M:
            return Outcome(
                Status.FAIL,
                f"tips move {shift * 100:.1f} cm toward the back of the hand when "
                f"curled: a flexion axis is reversed",
                measurements,
            )
        return Outcome(
            Status.INSUFFICIENT_DATA,
            f"tips move {shift * 100:+.1f} cm off the finger's line when curled, too "
            f"little to call",
            measurements,
        )


def _segment_distance(point, a, b) -> tuple[float, float]:
    """Distance from ``point`` to segment ab, and the projection parameter along it."""
    ab = sub(b, a)
    span = dot(ab, ab)
    t = 0.0 if span < 1e-12 else dot(sub(point, a), ab) / span
    clamped = max(0.0, min(1.0, t))
    nearest = tuple(a[i] + clamped * ab[i] for i in range(3))
    return norm(sub(point, nearest)), t


class _OptionalJointsGeometry(HandCheck):
    name = "optional_joints.geometry_consistent"
    severity = Severity.ADVISORY
    required = False
    summary = "PALM and finger METACARPALs, where valid, sit where OpenXR puts them"

    # PALM at the middle of MIDDLE_METACARPAL->MIDDLE_PROXIMAL, or on
    # WRIST->MIDDLE_PROXIMAL when that metacarpal is invalid; each METACARPAL on its
    # WRIST->PROXIMAL segment; none on top of another joint. Tolerances are a fraction
    # of the palm length.
    TOLERANCE = 0.25
    COINCIDENT_M = 0.003
    MAX_BAD_RATE = 0.05

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.seen: dict[int, int] = {}
        self.bad: dict[int, int] = {}
        self.coincident: dict[int, str] = {}

    def _judge(self, p, joint: int, palm: float) -> bool:
        wrist = p[WRIST]
        if joint == PALM:
            middle_mc, middle_prox = p[METACARPALS["middle"]], p[PROXIMAL["middle"]]
            if middle_mc is not None:
                centre = tuple((middle_mc[i] + middle_prox[i]) / 2 for i in range(3))
                return norm(sub(p[PALM], centre)) <= self.TOLERANCE * palm
            distance, t = _segment_distance(p[PALM], wrist, middle_prox)
            return distance <= self.TOLERANCE * palm and 0.0 <= t <= 1.0
        finger = next(f for f, j in METACARPALS.items() if j == joint)
        distance, t = _segment_distance(p[joint], wrist, p[PROXIMAL[finger]])
        return distance <= self.TOLERANCE * palm and -0.1 <= t <= 1.0

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        p = [frame.position(i) for i in range(NUM_JOINTS)]
        if p[WRIST] is None or any(p[PROXIMAL[f]] is None for f in FINGERS):
            return
        palm = norm(sub(p[PROXIMAL["middle"]], p[WRIST]))
        if palm < 1e-6:
            return
        for joint in OPTIONAL:
            if p[joint] is None:
                continue
            self.seen[joint] = self.seen.get(joint, 0) + 1
            placed = self._judge(p, joint, palm)
            for other, q in enumerate(p):
                if other != joint and q is not None:
                    if norm(sub(p[joint], q)) < self.COINCIDENT_M:
                        placed = False
                        self.coincident.setdefault(joint, JOINT_NAMES[other])
                        break
            if not placed:
                self.bad[joint] = self.bad.get(joint, 0) + 1

    def _result(self) -> Outcome:
        if not self.seen:
            return Outcome(Status.INSUFFICIENT_DATA, "none of the 5 is ever valid")
        rates = {
            JOINT_NAMES[j]: self.bad.get(j, 0) / n for j, n in sorted(self.seen.items())
        }
        measurements = {
            "misplaced_rate": rates,
            "coincides_with": {JOINT_NAMES[j]: o for j, o in self.coincident.items()},
        }
        off = [name for name, rate in rates.items() if rate > self.MAX_BAD_RATE]
        if not off:
            return Outcome(
                Status.PASS,
                f"{', '.join(rates)} placed as OpenXR expects",
                measurements,
            )
        shown = ", ".join(
            f"{name} ({rates[name]:.0%}"
            + (
                f", on {self.coincident[INDEX[name]]}"
                if INDEX[name] in self.coincident
                else ""
            )
            + ")"
            for name in off
        )
        return Outcome(Status.FAIL, f"misplaced: {shown}", measurements)


RequiredJointsValid = per_hand(_RequiredJointsValid, GROUP)
All26ValidRate = per_hand(_All26ValidRate, GROUP)
ValidSetStable = per_hand(_ValidSetStable, GROUP)
RadiusPlausible = per_hand(_RadiusPlausible, GROUP)
PositionScaleMetres = per_hand(_PositionScaleMetres, GROUP)
BoneLengthConstancy = per_hand(_BoneLengthConstancy, GROUP)
AnthropometricPlausibility = per_hand(_AnthropometricPlausibility, GROUP)
JointIndexAssignment = per_hand(_JointIndexAssignment, GROUP)
Handedness = per_hand(_Handedness, GROUP)
FlexDirection = per_hand(_FlexDirection, GROUP)
OptionalJointsGeometry = per_hand(_OptionalJointsGeometry, GROUP)

CHECKS = (
    RequiredJointsValid,
    All26ValidRate,
    ValidSetStable,
    RadiusPlausible,
    PositionScaleMetres,
    BoneLengthConstancy,
    AnthropometricPlausibility,
    JointIndexAssignment,
    Handedness,
    FlexDirection,
    OptionalJointsGeometry,
)
