# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement and frame checks, read in STAGE coordinates."""

from __future__ import annotations

import math
import statistics
from typing import Any

from cts_common.checks.base import Outcome, Severity, Status
from cts_common.vectors import as_wxyz, dot, norm, normalised, rotate, sub
from cts_common.vectors import rotate_by_inverse

from ..frames import Frame, Vec3
from ..labels import ROLLS, StepTimeline
from ..profile import CHAINS, FINGERS, TIP, WRIST
from .base import HandCheck, angle_deg, hand_axes, per_hand, percentile

GROUP = "placement"

_MINUS_Z = (0.0, 0.0, -1.0)


class _NotAtOrigin(HandCheck):
    name = "placement.not_at_origin"
    severity = Severity.HARD
    summary = "The wrist is placed in the room, not left at the STAGE origin"

    # Wrist distance from the STAGE origin under which a frame counts as near it, metres.
    RADIUS_M = 0.25
    MAX_RATE = 0.5

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.frames = 0
        self.near = 0
        self.distances: list[float] = []

    def _update(self, frame: Frame) -> None:
        wrist = frame.position(WRIST)
        if wrist is None:
            return
        distance = norm(wrist)
        self.frames += 1
        self.near += distance < self.RADIUS_M
        self.distances.append(distance)

    def _result(self) -> Outcome:
        if self.frames == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "the wrist is never valid")
        rate = self.near / self.frames
        typical = statistics.median(self.distances)
        measurements = {"near_origin_rate": rate, "median_distance_m": typical}
        if rate <= self.MAX_RATE:
            return Outcome(
                Status.PASS, f"median {typical:.2f} m from the origin", measurements
            )
        return Outcome(
            Status.FAIL,
            f"the wrist is within {self.RADIUS_M} m of the STAGE origin in {rate:.0%} "
            f"of frames: no wrist source is applied",
            measurements,
        )


class _WristNotFrozen(HandCheck):
    name = "placement.wrist_not_frozen"
    severity = Severity.SOFT
    summary = "The wrist pose does not stay bit-identical while the fingers move"

    # Fingertip movement between frames that counts as the fingers moving, metres.
    FINGER_MOTION_M = 0.001
    MAX_RATE = 0.2
    MIN_MOVING = 30

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.previous: tuple[Any, tuple] | None = None
        self.moving = 0
        self.frozen = 0

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        wrist = frame.joints[WRIST]
        if not wrist.is_valid:
            self.previous = None
            return
        pose = (wrist.position, wrist.orientation)
        tips = tuple(frame.position(TIP[f]) for f in FINGERS)
        previous, self.previous = self.previous, (pose, tips)
        if previous is None or any(t is None for t in tips + previous[1]):
            return
        moved = max(norm(sub(a, b)) for a, b in zip(tips, previous[1]))
        if moved < self.FINGER_MOTION_M:
            return
        self.moving += 1
        self.frozen += pose == previous[0]

    def _result(self) -> Outcome:
        if self.moving < self.MIN_MOVING:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the fingers moved in only {self.moving} frames",
            )
        rate = self.frozen / self.moving
        measurements = {"frozen_rate": rate, "finger_motion_frames": self.moving}
        if rate <= self.MAX_RATE:
            return Outcome(
                Status.PASS, f"wrist repeats in {rate:.1%} of frames", measurements
            )
        return Outcome(
            Status.FAIL,
            f"the wrist pose is bit-identical to the previous frame in {rate:.0%} of "
            f"frames where the fingers moved: the plugin is holding a stale wrist",
            measurements,
        )


def _bones() -> tuple[tuple[int, int], ...]:
    return tuple(pair for chain in CHAINS.values() for pair in zip(chain, chain[1:]))


class _ComponentOrder(HandCheck):
    name = "quaternion.component_order"
    severity = Severity.HARD
    summary = "Orientations are xyzw and follow the OpenXR joint axes"

    # Minimum mean alignment of joint -Z with its bone; OpenXR points -Z toward the child.
    MIN_ALIGNMENT = 0.7

    def __init__(self, side: str) -> None:
        super().__init__(side)
        self.samples = 0
        self.xyzw = 0.0
        self.wxyz = 0.0

    def _update(self, frame: Frame) -> None:
        if frame.joints is None:
            return
        for parent, child in _bones():
            a, b = frame.joints[parent], frame.joints[child]
            if not (a.is_valid and b.is_valid):
                continue
            direction = normalised(sub(b.position, a.position))
            swapped = as_wxyz(a.orientation)
            size = math.sqrt(sum(c * c for c in swapped))
            if direction is None or size < 1e-9:
                continue
            swapped = tuple(c / size for c in swapped)
            self.samples += 1
            self.xyzw += dot(rotate(a.orientation, _MINUS_Z), direction)
            self.wxyz += dot(rotate(swapped, _MINUS_Z), direction)

    def _result(self) -> Outcome:
        if self.samples == 0:
            return Outcome(Status.INSUFFICIENT_DATA, "no bone with both ends valid")
        xyzw, wxyz = self.xyzw / self.samples, self.wxyz / self.samples
        measurements = {"alignment_xyzw": xyzw, "alignment_wxyz": wxyz}
        if xyzw >= self.MIN_ALIGNMENT and xyzw >= wxyz:
            return Outcome(
                Status.PASS, f"-Z runs down the bones ({xyzw:.2f})", measurements
            )
        if wxyz >= self.MIN_ALIGNMENT:
            return Outcome(
                Status.FAIL,
                f"orientations only line up with the bones when read as wxyz "
                f"({wxyz:.2f} against {xyzw:.2f} as xyzw)",
                measurements,
            )
        hint = (
            " (+Z runs down the bones instead)" if xyzw <= -self.MIN_ALIGNMENT else ""
        )
        return Outcome(
            Status.FAIL,
            f"joint -Z does not follow the bones either way (xyzw {xyzw:.2f}, "
            f"wxyz {wxyz:.2f}){hint}: orientations are not in OpenXR joint axes",
            measurements,
        )


class _PositionOrientationSameFrame(HandCheck):
    name = "consistency.position_orientation_same_frame"
    severity = Severity.HARD
    required = False
    needs_timeline = True
    summary = (
        "While the wrist turns, the wrist quaternion turns with the joint positions"
    )

    # Wrist rotation needed to judge, and largest allowed palm-axis deviation, degrees.
    MIN_ROTATION_DEG = 90.0
    MAX_DEVIATION_DEG = 30.0

    depends_on = (
        "segmentation.label_windows_wellformed",
        "segmentation.label_alignment",
        "quaternion.component_order",
    )

    def __init__(self, side: str, timeline: StepTimeline | None = None) -> None:
        super().__init__(side)
        self.timeline = timeline
        self.label = next(label for label, mover in ROLLS.items() if mover == side)
        self.reference: tuple[tuple[Vec3, ...], tuple[Vec3, ...]] | None = None
        self.rotation = 0.0
        self.deviations: list[float] = []

    def _update(self, frame: Frame) -> None:
        if self.timeline is None or frame.joints is None:
            return
        step = self.timeline.step_at(frame.sample_time_ns)
        if step is None or step.label != self.label:
            return
        axes = hand_axes(frame)
        wrist = frame.joints[WRIST]
        if axes is None or not wrist.is_valid:
            return
        local = tuple(rotate_by_inverse(wrist.orientation, v) for v in axes)
        if self.reference is None:
            self.reference = (axes, local)
            return
        world_ref, local_ref = self.reference
        self.rotation = max(
            self.rotation, *(angle_deg(a, b) for a, b in zip(axes, world_ref))
        )
        self.deviations.append(max(angle_deg(a, b) for a, b in zip(local, local_ref)))

    def _result(self) -> Outcome:
        if self.timeline is None:
            return Outcome(Status.INSUFFICIENT_DATA, "no motion labels")
        if not self.deviations:
            return Outcome(Status.INSUFFICIENT_DATA, f"no frames in {self.label}")
        deviation = percentile(self.deviations, 0.95)
        measurements = {"rotation_deg": self.rotation, "deviation_p95_deg": deviation}
        if self.rotation < self.MIN_ROTATION_DEG:
            return Outcome(
                Status.INSUFFICIENT_DATA,
                f"the hand turned only {self.rotation:.0f} deg in {self.label}",
                measurements,
            )
        if deviation <= self.MAX_DEVIATION_DEG:
            return Outcome(
                Status.PASS,
                f"{deviation:.1f} deg apart over {self.rotation:.0f} deg of rotation",
                measurements,
            )
        return Outcome(
            Status.FAIL,
            f"over {self.rotation:.0f} deg of rotation the wrist quaternion drifts "
            f"{deviation:.0f} deg from the palm its own positions describe: position "
            f"and orientation are in different frames",
            measurements,
        )


NotAtOrigin = per_hand(_NotAtOrigin, GROUP)
WristNotFrozen = per_hand(_WristNotFrozen, GROUP)
ComponentOrder = per_hand(_ComponentOrder, GROUP)
PositionOrientationSameFrame = per_hand(_PositionOrientationSameFrame, GROUP)

CHECKS = (NotAtOrigin, WristNotFrozen, ComponentOrder, PositionOrientationSameFrame)
