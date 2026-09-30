# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks that need a skeleton profile.

Bone lengths carry most of the weight here because they do not depend on posture, so a
size or topology judgement made from them cannot be confounded by what the operator was
doing at the time.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..frames import Frame
from ..profile import FULL_BODY, SkeletonProfile
from acceptance_common.checks.base import Check, Outcome, Severity, Status
from acceptance_common.vectors import (
    cross,
    dot,
    is_finite,
    norm,
    normalised,
    rotate,
    sub,
)


class _Welford:
    __slots__ = ("count", "mean", "m2", "minimum", "maximum")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def stdev(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    @property
    def cv(self) -> float:
        return self.stdev / self.mean if self.mean > 0 else 0.0


class _ProfileCheck(Check):
    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__()
        self.profile = profile


class _BoneLengthCheck(_ProfileCheck):
    """Accumulates one length distribution per bone, valid endpoints only."""

    min_frames = 30

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self.bones: dict[tuple[int, int], _Welford] = {
            bone: _Welford() for bone in profile.bones()
        }

    def _update(self, frame: Frame) -> None:
        joints = frame.joints
        if joints is None:
            return
        for bone, stats in self.bones.items():
            parent, child = bone
            if not (joints[parent].is_valid and joints[child].is_valid):
                continue
            length = math.dist(joints[parent].position, joints[child].position)
            if math.isfinite(length):
                stats.add(length)

    def measured(self) -> dict[tuple[int, int], _Welford]:
        return {bone: s for bone, s in self.bones.items() if s.count > 0}

    def derived_bones(self) -> list[str]:
        """Constantly-zero bones are back-filled endpoints, not broken ones.

        Noitom derives hands from wrists and feet from ankles and marks them valid, so
        those four bones have zero length for the whole session. Reportable as derived
        rather than measured; never a fault.
        """
        return [
            self.profile.bone_name(*bone)
            for bone, stats in self.bones.items()
            if stats.count > 0 and stats.maximum == 0.0
        ]

    def substantive(self) -> dict[tuple[int, int], _Welford]:
        return {
            bone: stats
            for bone, stats in self.measured().items()
            if stats.maximum > 0.0
        }

    # A bone this steady is fixed; what is left is float32 storage in the payload.
    RIGID_CV = 0.002

    # Below this share of fixed bones the skeleton is not a rig with solved endpoints,
    # it is simply not rigid, and nothing here should be excused.
    MIN_RIGID_FRACTION = 0.6

    def solved_bones(self) -> dict[tuple[int, int], _Welford]:
        """Bones that vary on a rig whose other bones are fixed.

        An IK solver reaching a tracked endpoint puts the reach error in the last
        segment, so the forearm stretches while everything else holds. This is the
        mirror of the back-filled case above: derived, not broken. A genuinely rubber
        skeleton has no fixed bones to contrast against -- defect_bone_length_drift
        varies all 23 of them, where a three-tracker PICO 4 Ultra varies 2 and holds
        21 to within float32.
        """
        bones = self.substantive()
        if not bones:
            return {}
        rigid = sum(1 for stats in bones.values() if stats.cv <= self.RIGID_CV)
        if rigid / len(bones) < self.MIN_RIGID_FRACTION:
            return {}
        return {
            bone: stats
            for bone, stats in bones.items()
            if stats.cv > self.profile.max_bone_length_cv
        }


class BoneLengthConstancy(_BoneLengthCheck):
    name = "skeleton.bone_length_constancy"
    gate = "G2"
    severity = Severity.SOFT
    summary = "Bone lengths are invariant over the session"

    def _result(self) -> Outcome:
        bones = self.substantive()
        if not bones:
            return Outcome(Status.INSUFFICIENT_DATA, "no bone had two valid endpoints")

        solved = self.solved_bones()
        judged = {b: s for b, s in bones.items() if b not in solved} or bones
        solved_names = sorted(self.profile.bone_name(*b) for b in solved)

        worst_bone, worst = max(judged.items(), key=lambda item: item[1].cv)
        measurements = {
            "bones_measured": len(bones),
            "worst_bone": self.profile.bone_name(*worst_bone),
            "worst_cv": worst.cv,
            "worst_range_cm": [
                round(worst.minimum * 100, 2),
                round(worst.maximum * 100, 2),
            ],
            "derived_bones": self.derived_bones(),
            "solved_bones": solved_names,
        }
        if worst.cv <= self.profile.max_bone_length_cv:
            detail = f"worst fixed bone varies {worst.cv:.2%} of its length"
            if solved_names:
                detail += (
                    f"; {', '.join(solved_names)} vary on an otherwise fixed "
                    f"skeleton, so they are solved to a tracked endpoint and their "
                    f"lengths are derived rather than measured"
                )
            return Outcome(Status.PASS, detail, measurements)
        return Outcome(
            Status.FAIL,
            f"{self.profile.bone_name(*worst_bone)} varies {worst.cv:.1%} of its "
            f"length ({worst.minimum * 100:.1f} to {worst.maximum * 100:.1f} cm), so "
            f"segment lengths are not invariant",
            measurements,
        )


class _StatureCheck(_BoneLengthCheck):
    def chain_length(self) -> float | None:
        """Summed bone lengths from ankle to head; independent of posture."""
        total = 0.0
        chain = self.profile.stature_chain
        for lower, upper in zip(chain, chain[1:]):
            a, b = self.profile.index(lower), self.profile.index(upper)
            stats = self.bones.get((a, b)) or self.bones.get((b, a))
            if stats is None or stats.count == 0:
                return None
            total += stats.mean
        return total


class PositionScaleMetres(_StatureCheck):
    name = "units.position_scale_metres"
    gate = "G2"
    severity = Severity.HARD
    summary = "Positions are in metres"

    def _result(self) -> Outcome:
        stature = self.chain_length()
        if stature is None:
            return Outcome(Status.INSUFFICIENT_DATA, "stature chain incomplete")
        low, high = self.profile.plausible_scale_m
        measurements = {
            "skeletal_height": stature,
            "plausible_range": [low, high],
        }
        if low <= stature <= high:
            return Outcome(Status.PASS, f"skeleton spans {stature:.2f} m", measurements)
        factor = stature / 1.7
        return Outcome(
            Status.FAIL,
            f"the skeleton measures {stature:.1f} units ankle to head, about "
            f"{factor:.0f}x a human in metres, so the positions are not in metres",
            measurements,
        )


class AnthropometricPlausibility(_StatureCheck):
    name = "skeleton.anthropometric_plausibility"
    gate = "G2"
    severity = Severity.SOFT
    summary = "Segment lengths and their ratios are human"

    def _result(self) -> Outcome:
        stature = self.chain_length()
        if stature is None:
            return Outcome(Status.INSUFFICIENT_DATA, "stature chain incomplete")

        # A solved forearm has no length of its own -- the solver sets it to whatever
        # reaches the controller -- so its ratio to the upper arm says nothing about
        # the subject's proportions and must not be judged as if it did.
        solved = set(self.solved_bones())
        ratios: dict[str, float] = {}
        derived: list[str] = []
        for side in ("LEFT", "RIGHT"):
            arm = (
                self.profile.index(f"{side}_SHOULDER"),
                self.profile.index(f"{side}_ELBOW"),
            )
            forearm = (
                self.profile.index(f"{side}_ELBOW"),
                self.profile.index(f"{side}_WRIST"),
            )
            upper, fore = self.bones.get(arm), self.bones.get(forearm)
            if not (upper and fore and upper.count and fore.count and upper.mean > 0):
                continue
            if arm in solved or forearm in solved:
                derived.append(side.lower())
                continue
            ratios[side.lower()] = fore.mean / upper.mean

        low, high = self.profile.stature_range_m
        ratio_low, ratio_high = self.profile.forearm_over_upper_arm
        measurements = {
            "skeletal_height": stature,
            "stature_range": [low, high],
            "forearm_over_upper_arm": ratios,
            "derived_arms": derived,
            "ratio_range": [ratio_low, ratio_high],
        }

        problems = []
        if not low <= stature <= high:
            problems.append(
                f"summed ankle-to-head length {stature:.2f} m is outside "
                f"{low:.2f}-{high:.2f} m"
            )
        for side, ratio in ratios.items():
            if not ratio_low <= ratio <= ratio_high:
                problems.append(
                    f"the {side} forearm is {ratio:.2f} of its upper arm, outside "
                    f"{ratio_low:.2f}-{ratio_high:.2f}"
                )

        if problems:
            return Outcome(Status.FAIL, "; ".join(problems), measurements)

        shape = ", ".join(f"{side} {ratio:.2f}" for side, ratio in ratios.items())
        detail = f"{stature:.2f} m" + (f", forearm/upper-arm {shape}" if shape else "")
        if derived:
            detail += (
                f"; the {' and '.join(derived)} forearm"
                f"{'s are' if len(derived) > 1 else ' is'} solved to a tracked "
                f"endpoint, so that proportion is not the subject's"
            )
        return Outcome(Status.PASS, detail, measurements)


class UpAxis(_ProfileCheck):
    name = "coordinate_frame.up_axis"
    gate = "G2"
    severity = Severity.HARD
    summary = "The up axis is +Y"
    min_frames = 30

    # Measured from head to pelvis rather than from a bounding box, so a T-pose arm span
    # cannot outvote the torso.
    AXIS_NAMES = ("X", "Y", "Z")
    EXPECTED_AXIS = 1
    EXPECTED_SIGN = 1.0
    DOMINANCE = 2.0

    # Below this the torso vector carries no direction at all, which happens when the
    # subject is lying down or the feed is degenerate. Neither is evidence about the up
    # axis, so the check declines rather than reporting the wrong axis.
    MIN_TORSO_M = 0.10

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self.total = [0.0, 0.0, 0.0]
        self.samples = 0
        self._head = profile.index("HEAD")
        self._pelvis = profile.index("PELVIS")

    def _update(self, frame: Frame) -> None:
        joints = frame.joints
        if joints is None:
            return
        head, pelvis = joints[self._head], joints[self._pelvis]
        if not (head.is_valid and pelvis.is_valid):
            return
        delta = [h - p for h, p in zip(head.position, pelvis.position)]
        if not all(math.isfinite(component) for component in delta):
            return
        self.samples += 1
        for axis in range(3):
            self.total[axis] += delta[axis]

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "head or pelvis never valid")
        mean = [component / self.samples for component in self.total]
        magnitudes = [abs(component) for component in mean]
        dominant = max(range(3), key=lambda axis: magnitudes[axis])
        others = max(magnitudes[axis] for axis in range(3) if axis != dominant)
        length = math.sqrt(sum(component * component for component in mean))
        measurements = {
            "mean_pelvis_to_head": mean,
            "torso_length_m": length,
            "dominant_axis": self.AXIS_NAMES[dominant],
            "dominance": magnitudes[dominant] / others if others > 0 else math.inf,
        }

        if length < self.MIN_TORSO_M:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"pelvis to head spans only {length:.3f} m, which says nothing about "
                f"the up axis",
                measurements,
            )
        if others > 0 and magnitudes[dominant] / others < self.DOMINANCE:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"no axis dominates pelvis-to-head ({mean[0]:.2f}, {mean[1]:.2f}, "
                f"{mean[2]:.2f}); the subject may not be upright",
                measurements,
            )
        if dominant == self.EXPECTED_AXIS and (mean[dominant] * self.EXPECTED_SIGN > 0):
            return Outcome(
                Status.PASS,
                f"+{self.AXIS_NAMES[dominant]} up",
                measurements,
            )
        sign = "+" if mean[dominant] > 0 else "-"
        return Outcome(
            Status.FAIL,
            f"pelvis to head points along {sign}{self.AXIS_NAMES[dominant]}, not "
            f"+Y, so the recording is not Y-up",
            measurements,
        )


class BodyAxes(NamedTuple):
    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    # Which reference produced ``forward``; see _BodyAxesCheck for why it matters.
    forward_from: str


class _BodyAxesCheck(_ProfileCheck):
    """Per-frame anatomical right, up and forward vectors.

    Forward prefers the mean of the two ankle-to-foot vectors. Averaging the feet makes
    it blind to the very faults these checks look for: swapping left and right
    exchanges two terms of a mean, and mirroring negates an axis the mean does not
    depend on. It is also position-only, so a wrong quaternion order cannot corrupt it.

    Feet are not always usable. A vendor that back-fills them from the ankles leaves a
    zero-length foot bone, and Pico does not report them at all, so forward then falls
    back to the pelvis orientation. That fallback is still independent of the left/right
    labels, but it does inherit any orientation fault, which is why callers record
    ``forward_from`` in their measurements.
    """

    min_frames = 30

    # A foot bone shorter than this carries no direction: back-filling an endpoint from
    # its parent produces exactly this case.
    MIN_FOOT_BONE_M = 0.02

    # An ankle-to-toe bone is around a quarter of the torso; anything approaching the
    # torso's own length is not a foot.
    MAX_FOOT_BONE_OVER_TORSO = 0.5

    # Both toes must point within 60 degrees of each other for the pair to be believed.
    # Length alone does not establish that: a foot zeroed to the world origin sits a
    # plausible 12 cm from its ankle, because the ankle is itself near the origin, and
    # only the disagreement between the two feet reveals it.
    MIN_FOOT_AGREEMENT = 0.5

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self._pelvis = profile.index("PELVIS")
        self._head = profile.index("HEAD")
        self._left_hip = profile.index("LEFT_HIP")
        self._right_hip = profile.index("RIGHT_HIP")
        self._feet = (
            (profile.index("LEFT_ANKLE"), profile.index("LEFT_FOOT")),
            (profile.index("RIGHT_ANKLE"), profile.index("RIGHT_FOOT")),
        )

    def _forward_from_feet(self, joints, torso: float):
        max_bone = self.MAX_FOOT_BONE_OVER_TORSO * torso
        toes = []
        for ankle, foot in self._feet:
            if not (joints[ankle].is_valid and joints[foot].is_valid):
                continue
            bone = sub(joints[foot].position, joints[ankle].position)
            if not is_finite(bone):
                continue
            length = norm(bone)
            if not self.MIN_FOOT_BONE_M <= length <= max_bone:
                continue
            direction = normalised(bone)
            if direction is not None:
                toes.append(direction)
        if len(toes) != 2 or dot(*toes) < self.MIN_FOOT_AGREEMENT:
            return None
        return normalised(tuple(sum(axis) / len(toes) for axis in zip(*toes)))

    def _axes(self, frame: Frame) -> BodyAxes | None:
        joints = frame.joints
        if joints is None:
            return None
        pelvis, head = joints[self._pelvis], joints[self._head]
        left_hip, right_hip = joints[self._left_hip], joints[self._right_hip]
        if not (
            pelvis.is_valid
            and head.is_valid
            and left_hip.is_valid
            and right_hip.is_valid
        ):
            return None

        right = sub(right_hip.position, left_hip.position)
        up = sub(head.position, pelvis.position)
        if not is_finite(up):
            return None
        forward = self._forward_from_feet(joints, norm(up))
        forward_from = "feet"
        if forward is None:
            forward = rotate(pelvis.orientation, self.profile.forward_axis)
            forward_from = "pelvis_orientation"
        for vector in (right, up, forward):
            if not is_finite(vector) or norm(vector) < 1e-6:
                return None
        return BodyAxes(pelvis.position, right, up, forward, forward_from)


class Handedness(_BodyAxesCheck):
    name = "coordinate_frame.handedness"
    gate = "G2"
    severity = Severity.HARD
    summary = "The coordinate frame is right-handed"

    DECISIVE = 0.2

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self.total = 0.0
        self.samples = 0
        self.forward_sources: set[str] = set()

    def _update(self, frame: Frame) -> None:
        axes = self._axes(frame)
        if axes is None:
            return
        triple = dot(cross(axes.right, axes.up), axes.forward) / (
            norm(axes.right) * norm(axes.up) * norm(axes.forward)
        )
        self.total += triple
        self.samples += 1
        self.forward_sources.add(axes.forward_from)

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "body axes never resolvable")
        chirality = self.total / self.samples
        expected = self.profile.expected_chirality
        measurements = {
            "chirality": chirality,
            "expected_sign": expected,
            "samples": self.samples,
            "forward_from": sorted(self.forward_sources),
        }
        if abs(chirality) < self.DECISIVE:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"body axes are too close to coplanar to judge ({chirality:+.2f})",
                measurements,
            )
        if chirality * expected > 0:
            return Outcome(
                Status.PASS, f"right-handed ({chirality:+.2f})", measurements
            )
        return Outcome(
            Status.FAIL,
            f"right x up points along forward rather than against it "
            f"({chirality:+.2f}), so the frame is left-handed",
            measurements,
        )


class LeftRightLabelling(_BodyAxesCheck):
    name = "skeleton.left_right_labelling"
    gate = "G2"
    severity = Severity.HARD
    summary = "LEFT and RIGHT joints are on the sides their names claim"

    MAX_VIOLATION_RATE = 0.05

    # A joint sitting this close to the midline has no side to be on. Positions zeroed
    # out on a joint the device still marks valid land there, and that fault belongs to
    # values.zero_pose_on_valid_joint, not here.
    MIN_LATERAL_OFFSET_M = 0.02

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self.pairs = profile.lateral_pairs()
        self.checked = 0
        self.violations = 0
        self.offenders: set[str] = set()
        self.forward_sources: set[str] = set()

    def _update(self, frame: Frame) -> None:
        axes = self._axes(frame)
        if axes is None:
            return
        # up x forward points to the subject's own left, derived without reference to
        # any joint's label, which is what lets it judge the labels.
        lateral = normalised(cross(axes.up, axes.forward))
        if lateral is None:
            return
        self.forward_sources.add(axes.forward_from)

        joints = frame.joints
        assert joints is not None
        for left_index, right_index in self.pairs:
            left, right = joints[left_index], joints[right_index]
            if not (left.is_valid and right.is_valid):
                continue
            left_side = dot(sub(left.position, axes.origin), lateral)
            right_side = dot(sub(right.position, axes.origin), lateral)
            if min(abs(left_side), abs(right_side)) < self.MIN_LATERAL_OFFSET_M:
                continue
            self.checked += 1
            if left_side > 0.0 > right_side:
                continue
            self.violations += 1
            self.offenders.add(
                f"{self.profile.joint_names[left_index]}/"
                f"{self.profile.joint_names[right_index]}"
            )

    def _result(self) -> Outcome:
        if self.checked == 0:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                "no lateral pair was ever valid and clear of the midline",
            )
        rate = self.violations / self.checked
        measurements = {
            "pairs_checked": self.checked,
            "violations": self.violations,
            "rate": rate,
            "pairs": sorted(self.offenders),
            "forward_from": sorted(self.forward_sources),
        }
        if rate <= self.MAX_VIOLATION_RATE:
            return Outcome(
                Status.PASS, "left and right are on their own sides", measurements
            )
        return Outcome(
            Status.FAIL,
            f"{rate:.0%} of samples put a LEFT joint on the body's right or the "
            f"reverse, across {len(self.offenders)} pairs",
            measurements,
        )


class JointIndexAssignment(_ProfileCheck):
    name = "skeleton.joint_index_assignment"
    gate = "G2"
    severity = Severity.HARD
    summary = "Joints occupy the indices the layout assigns them"
    min_frames = 120

    # Measures how much of a bone's own length is spent travelling back toward the
    # root. Put two adjacent joints in each other's slots and the bone between them
    # reverses, pointing straight at the root for a ratio of +1. A folded limb also
    # brings the child closer to the root, but it does so sideways: across the whole
    # fixture corpus no correctly indexed bone exceeds -0.47, so 0.8 is not a tuned
    # threshold but the midpoint of an empty gap.
    REVERSED = 0.8

    # One transient reversal is noise; a relabelled joint reverses on every frame,
    # because the relabelling is fixed and the skeleton is rigid.
    MAX_PERSISTENT_RATE = 0.60

    # Below this a bone has not been seen in enough postures to tell a reversal from a
    # pose held throughout a short window of validity.
    MIN_SAMPLES = 120

    MIN_BONE_M = 0.01

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__(profile)
        self._root = profile.index("PELVIS")
        self.reversed_count: dict[tuple[int, int], int] = {}
        self.compared: dict[tuple[int, int], int] = {}
        self.worst_ratio: dict[tuple[int, int], float] = {}

    def _update(self, frame: Frame) -> None:
        joints = frame.joints
        if joints is None:
            return
        root = joints[self._root]
        if not root.is_valid:
            return
        for child, parent in enumerate(self.profile.parents):
            # Bones hanging straight off the root have no meaningful radial direction.
            if parent < 0 or self.profile.parents[parent] < 0:
                continue
            if not (joints[child].is_valid and joints[parent].is_valid):
                continue
            length = math.dist(joints[child].position, joints[parent].position)
            to_parent = math.dist(joints[parent].position, root.position)
            to_child = math.dist(joints[child].position, root.position)
            if not is_finite((length, to_parent, to_child)):
                continue
            # A zero-length bone has no direction to judge, and back-filled endpoints
            # legitimately produce them.
            if length < self.MIN_BONE_M:
                continue
            bone = (parent, child)
            ratio = (to_parent - to_child) / length
            self.compared[bone] = self.compared.get(bone, 0) + 1
            self.worst_ratio[bone] = max(self.worst_ratio.get(bone, -math.inf), ratio)
            if ratio > self.REVERSED:
                self.reversed_count[bone] = self.reversed_count.get(bone, 0) + 1

    def _result(self) -> Outcome:
        judged = {
            bone: count
            for bone, count in self.compared.items()
            if count >= self.MIN_SAMPLES
        }
        if not judged:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"no bone held valid endpoints for {self.MIN_SAMPLES} frames",
                {"bones_seen": len(self.compared)},
            )
        rates = {
            bone: self.reversed_count.get(bone, 0) / count
            for bone, count in judged.items()
        }
        worst_bone, worst_rate = max(rates.items(), key=lambda item: item[1])
        reversed_bones = {
            self.profile.bone_name(*bone): rate
            for bone, rate in rates.items()
            if rate > self.MAX_PERSISTENT_RATE
        }
        measurements = {
            "bones_judged": len(judged),
            "worst_bone": self.profile.bone_name(*worst_bone),
            "worst_reversal_rate": worst_rate,
            "worst_inward_ratio": self.worst_ratio[worst_bone],
            "reversed_bones": reversed_bones,
        }
        if not reversed_bones:
            return Outcome(
                Status.PASS,
                f"every bone points away from the pelvis; worst is "
                f"{self.profile.bone_name(*worst_bone)} at "
                f"{self.worst_ratio[worst_bone]:+.2f} of its length",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"{', '.join(sorted(reversed_bones))} points back at the pelvis on "
            f"{worst_rate:.0%} of frames, so those joints are in each other's indices",
            measurements,
        )
