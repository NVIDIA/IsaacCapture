# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a single-stream check once per hand, and the hand geometry the checks share."""

from __future__ import annotations

import math
import statistics
from typing import Any, ClassVar, Sequence

from acceptance_common.checks.base import Check, Outcome, Status
from acceptance_common.vectors import cross, dot, norm, normalised, sub

from ..frames import SIDES, Frame, Vec3
from ..profile import CHAINS, FINGERS, PROXIMAL, WRIST, phalanges

__all__ = [
    "HandCheck",
    "PerHand",
    "angle_deg",
    "bone_length",
    "hand_axes",
    "hand_straightness",
    "median",
    "per_hand",
    "percentile",
    "straightness",
]


class HandCheck(Check):
    """A check that sees one hand's frames and knows which hand it is."""

    def __init__(self, side: str) -> None:
        super().__init__()
        self.side = side


class PerHand(Check):
    """Feeds each side's frames to its own ``inner`` and merges the two outcomes.

    Either hand failing fails the check. A hand that never appears leaves it unanswered:
    the script is two-handed, so one silent channel is a finding, not a pass.
    """

    inner: ClassVar[type[Check]]
    min_frames = 0

    def __init__(self, timeline: Any = None) -> None:
        super().__init__()
        self.parts = {side: self._make(side, timeline) for side in SIDES}

    def _make(self, side: str, timeline: Any) -> Check:
        inner = self.inner
        timed = getattr(inner, "needs_timeline", False)
        if issubclass(inner, HandCheck):
            return inner(side, timeline) if timed else inner(side)
        return inner(timeline) if timed else inner()

    def _update(self, frame: Frame) -> None:
        part = self.parts.get(frame.side)
        if part is not None:
            part.update(frame)

    def _result(self) -> Outcome:
        outcomes = {side: part.result() for side, part in self.parts.items()}
        measurements = {side: dict(o.measurements) for side, o in outcomes.items()}
        failed = [s for s, o in outcomes.items() if o.status is Status.FAIL]
        unanswered = [
            s for s, o in outcomes.items() if o.status is Status.INSUFFICIENT_DATA
        ]
        if failed:
            status = Status.FAIL
            shown = failed
        elif unanswered:
            status = Status.INSUFFICIENT_DATA
            shown = unanswered
        else:
            status = Status.PASS
            shown = list(outcomes)
        detail = "; ".join(f"{side}: {outcomes[side].detail}" for side in shown)
        attribution = next(
            (outcomes[s].attribution for s in failed if outcomes[s].attribution), None
        )
        return Outcome(status, detail, measurements, attribution)


_POLICY = (
    "name",
    "severity",
    "attribution",
    "summary",
    "required",
    "judged",
    "depends_on",
    "needs_timeline",
)


def per_hand(inner: type[Check], gate: str) -> type[PerHand]:
    fields = {key: getattr(inner, key) for key in _POLICY if hasattr(inner, key)}
    return type(inner.__name__, (PerHand,), {"inner": inner, "gate": gate, **fields})


def bone_length(positions: Sequence[Vec3 | None], a: int, b: int) -> float | None:
    pa, pb = positions[a], positions[b]
    if pa is None or pb is None:
        return None
    return norm(sub(pb, pa))


def straightness(positions: Sequence[Vec3 | None], finger: str) -> float | None:
    """Root-to-tip distance over the summed phalanx lengths: 1 straight, lower curled."""
    lengths = [bone_length(positions, a, b) for a, b in phalanges(finger)]
    if any(length is None for length in lengths):
        return None
    total = sum(lengths)
    reach = bone_length(positions, CHAINS[finger][0], CHAINS[finger][-1])
    if total <= 1e-9 or reach is None:
        return None
    return reach / total


def hand_straightness(positions: Sequence[Vec3 | None]) -> float | None:
    """Mean over the four fingers; the thumb curls into a fist differently."""
    values = [straightness(positions, finger) for finger in FINGERS]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def hand_axes(frame: Frame) -> tuple[Vec3, Vec3, Vec3] | None:
    """Unit forward, across-palm and their cross product, from STAGE positions only."""
    wrist, knuckle = frame.position(WRIST), frame.position(PROXIMAL["middle"])
    index, little = (
        frame.position(PROXIMAL["index"]),
        frame.position(PROXIMAL["little"]),
    )
    if wrist is None or knuckle is None or index is None or little is None:
        return None
    forward = normalised(sub(knuckle, wrist))
    across = normalised(sub(little, index))
    if forward is None or across is None:
        return None
    normal = normalised(cross(forward, across))
    return None if normal is None else (forward, across, normal)


def angle_deg(a: Vec3, b: Vec3) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan
