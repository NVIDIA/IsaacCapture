# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks on the relationship between orientations and positions.

All of these rest on one invariant that needs no priors at all: a child's offset
expressed in its parent's local frame is a rigid property of the skeleton, so it must be
constant over the session no matter what the subject does. Rotate the positions without
rotating the orientations, or serialise the quaternion components in the wrong order,
and that offset starts moving.

Two faults deliberately do *not* disturb it. Expressing the whole rig Z-up rotates
positions and orientations together, and writing positions in centimetres scales the
offset without varying it; both are self-consistent recordings and are caught elsewhere.
"""

from __future__ import annotations

import math

from ..frames import Frame
from ..profile import FULL_BODY, SkeletonProfile
from acceptance_common.checks.base import Check, Outcome, Severity, Status
from acceptance_common.vectors import Quaternion, Vector, as_wxyz, rotate_by_inverse


class _OffsetSpread:
    """Tracks how much a bone's parent-local offset moves about its own mean."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = [0.0, 0.0, 0.0]
        self.m2 = [0.0, 0.0, 0.0]

    def add(self, offset: Vector) -> None:
        self.count += 1
        for axis in range(3):
            delta = offset[axis] - self.mean[axis]
            self.mean[axis] += delta / self.count
            self.m2[axis] += delta * (offset[axis] - self.mean[axis])

    @property
    def length(self) -> float:
        return math.sqrt(sum(component * component for component in self.mean))

    @property
    def spread(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(sum(self.m2[axis] / (self.count - 1) for axis in range(3)))

    @property
    def relative_spread(self) -> float:
        return self.spread / self.length if self.length > 1e-6 else 0.0


class _LocalOffsetCheck(Check):
    """Accumulates parent-local child offsets under both quaternion readings.

    Only bones whose parent actually turns are counted. A joint that holds still cannot
    expose a frame mismatch at all, and averaging it in buries the ones that can: across
    every bone the scripted sequence separates 0.005 from 0.030, while across the seven
    bones that do rotate it separates 0.003 from 0.390.
    """

    min_frames = 30
    required = False

    MIN_PARENT_ROTATION_DEG = 20.0
    MIN_ROTATING_BONES = 3

    def __init__(self, profile: SkeletonProfile = FULL_BODY) -> None:
        super().__init__()
        self.profile = profile
        self.bones = profile.bones()
        self.as_stored = {bone: _OffsetSpread() for bone in self.bones}
        self.as_reordered = {bone: _OffsetSpread() for bone in self.bones}
        self._reference: dict[tuple[int, int], Quaternion] = {}
        self.rotation_deg = {bone: 0.0 for bone in self.bones}

    def _update(self, frame: Frame) -> None:
        joints = frame.joints
        if joints is None:
            return
        for bone in self.bones:
            parent_index, child_index = bone
            parent, child = joints[parent_index], joints[child_index]
            if not (parent.is_valid and child.is_valid):
                continue
            delta = tuple(c - p for c, p in zip(child.position, parent.position))
            if not all(math.isfinite(component) for component in delta):
                continue
            if not all(math.isfinite(component) for component in parent.orientation):
                continue
            self.as_stored[bone].add(rotate_by_inverse(parent.orientation, delta))
            self.as_reordered[bone].add(
                rotate_by_inverse(as_wxyz(parent.orientation), delta)
            )
            reference = self._reference.setdefault(bone, parent.orientation)
            dot = abs(sum(a * b for a, b in zip(reference, parent.orientation)))
            self.rotation_deg[bone] = max(
                self.rotation_deg[bone],
                2.0 * math.degrees(math.acos(min(1.0, dot))),
            )

    def _rotating_bones(self) -> list[tuple[int, int]]:
        return [
            bone
            for bone in self.bones
            if self.as_stored[bone].count >= 2
            and self.as_stored[bone].length > 1e-3
            and self.rotation_deg[bone] >= self.MIN_PARENT_ROTATION_DEG
        ]

    def _median_spread(self, table: dict) -> float | None:
        bones = self._rotating_bones()
        if len(bones) < self.MIN_ROTATING_BONES:
            return None
        values = sorted(table[bone].relative_spread for bone in bones)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return 0.5 * (values[middle - 1] + values[middle])

    def _too_still(self) -> Outcome:
        rotating = len(self._rotating_bones())
        return Outcome(
            Status.INSUFFICIENT_DATA,
            f"only {rotating} bones turn more than "
            f"{self.MIN_PARENT_ROTATION_DEG:.0f} degrees; a held pose cannot show "
            f"whether the frames agree",
            {"rotating_bones": rotating},
        )


class PositionOrientationSameFrame(_LocalOffsetCheck):
    name = "consistency.position_orientation_same_frame"
    gate = "G2"
    severity = Severity.HARD
    summary = "Positions and orientations are expressed in the same frame"

    MAX_RELATIVE_SPREAD = 0.10

    def _result(self) -> Outcome:
        stored = self._median_spread(self.as_stored)
        if stored is None:
            return self._too_still()
        reordered = self._median_spread(self.as_reordered)
        measurements = {
            "median_relative_spread": stored,
            "median_relative_spread_if_wxyz": reordered,
            "rotating_bones": len(self._rotating_bones()),
        }
        if stored <= self.MAX_RELATIVE_SPREAD:
            return Outcome(
                Status.PASS,
                f"parent-local offsets hold to {stored:.1%} of their length",
                measurements,
            )
        # A component-order mistake produces the same symptom, and
        # quaternion.component_order owns it; saying so there and staying quiet here
        # keeps one injected fault from being reported as two.
        if reordered is not None and reordered < stored * 0.5:
            return Outcome(
                Status.PASS,
                f"offsets drift {stored:.0%}, but the component order explains it",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"parent-local child offsets move {stored:.0%} of their own length, so "
            f"orientations do not describe the frame the positions are in",
            measurements,
        )


class ComponentOrder(_LocalOffsetCheck):
    name = "quaternion.component_order"
    gate = "G2"
    severity = Severity.HARD
    summary = "Quaternion components are stored x, y, z, w"

    MAX_RELATIVE_SPREAD = 0.10
    IMPROVEMENT = 2.0

    def _result(self) -> Outcome:
        stored = self._median_spread(self.as_stored)
        reordered = self._median_spread(self.as_reordered)
        if stored is None or reordered is None:
            return self._too_still()
        measurements = {
            "median_relative_spread": stored,
            "median_relative_spread_if_wxyz": reordered,
        }
        # Only a decisive win counts. Reading a correct recording the wrong way round
        # scatters the offsets, so the comparison is safely one-sided.
        if stored <= self.MAX_RELATIVE_SPREAD:
            return Outcome(Status.PASS, f"x,y,z,w fits to {stored:.1%}", measurements)
        if reordered * self.IMPROVEMENT < stored:
            return Outcome(
                Status.FAIL,
                f"reading the components as w,x,y,z fits the skeleton "
                f"{stored / max(reordered, 1e-9):.0f}x better ({reordered:.1%} against "
                f"{stored:.1%}), so they are serialised in the wrong order",
                measurements,
            )
        return Outcome(
            Status.PASS,
            f"offsets drift {stored:.0%}, but not because of the component order",
            measurements,
        )
