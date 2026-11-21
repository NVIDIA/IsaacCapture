# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running a single-stream check once per hand, and the hand geometry the checks share."""

from __future__ import annotations

import math
import statistics
from typing import Any, ClassVar, Sequence

from cts_common.checks.base import Check, Outcome, Status
from cts_common.vectors import cross, dot, norm, normalised, sub

from ..frames import SIDES, Frame, Vec3
from ..profile import CHAINS, FINGERS, PROXIMAL, WRIST, phalanges

__all__ = [
    "HandCheck",
    "PerHand",
    "angle_deg",
    "bone_length",
    "fit_plane",
    "hand_axes",
    "hand_straightness",
    "median",
    "palm_centre",
    "palmar_normal",
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

    Either hand failing fails the check; a hand that never appears leaves it unanswered.
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


def per_hand(inner: type[Check], group: str) -> type[PerHand]:
    fields = {key: getattr(inner, key) for key in _POLICY if hasattr(inner, key)}
    return type(inner.__name__, (PerHand,), {"inner": inner, "group": group, **fields})


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
    """Mean over the four fingers, thumb excluded."""
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


def palmar_normal(frame: Frame) -> Vec3 | None:
    """Unit normal out of the palm. ``hand_axes`` gives it on a right hand, the back on a left."""
    axes = hand_axes(frame)
    if axes is None:
        return None
    normal = axes[2]
    return normal if frame.side == "right" else (-normal[0], -normal[1], -normal[2])


def palm_centre(frame: Frame) -> Vec3 | None:
    """Mean of the four finger PROXIMAL joints."""
    knuckles = [frame.position(PROXIMAL[finger]) for finger in FINGERS]
    if any(k is None for k in knuckles):
        return None
    return tuple(statistics.fmean(k[axis] for k in knuckles) for axis in range(3))  # type: ignore[return-value]


def fit_plane(
    points: Sequence[Vec3], guess: Vec3 = (0.0, 1.0, 0.0)
) -> tuple[Vec3, Vec3, list[float]]:
    """Least-squares plane: (centroid, unit normal, signed distance of each point).

    The normal is the covariance's smallest eigenvector, by power iteration on
    ``trace * I - C`` seeded with ``guess``; the normal is returned on ``guess``'s side.
    """
    count = len(points)
    centroid = tuple(statistics.fmean(p[axis] for p in points) for axis in range(3))
    centred = [sub(p, centroid) for p in points]
    cov = [
        [sum(c[i] * c[j] for c in centred) / count for j in range(3)] for i in range(3)
    ]
    trace = cov[0][0] + cov[1][1] + cov[2][2]
    shifted = [
        [(trace if i == j else 0.0) - cov[i][j] for j in range(3)] for i in range(3)
    ]
    normal = normalised(guess) or (0.0, 1.0, 0.0)
    for _ in range(200):
        step = tuple(sum(shifted[i][j] * normal[j] for j in range(3)) for i in range(3))
        unit = normalised(step)
        if unit is None:
            break
        normal = unit
    if dot(normal, guess) < 0:
        normal = (-normal[0], -normal[1], -normal[2])
    return centroid, normal, [dot(c, normal) for c in centred]  # type: ignore[return-value]


def angle_deg(a: Vec3, b: Vec3) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)))))


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else math.nan
