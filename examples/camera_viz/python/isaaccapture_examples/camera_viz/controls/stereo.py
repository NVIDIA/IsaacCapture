# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry for the stereo plane gap.

A stereo layer draws each eye's image on its own surface. Separating the two
by ``gap`` moves where zero-disparity content appears::

    D = Z * ipd / (ipd - gap)          for a surface Z away

which inverts to ``gap = ipd * (1 - Z / D)``. The stick ramps the gap rather
than the distance because the gap is linear in vergence angle, which is what
the eyes actually track; the distance is what gets displayed, because that is
what the operator means.

The camera's own baseline is deliberately absent from all of this. It is
baked into the captured pixels and sets the scene's depth *scale*, which no
viewer-side knob can change; the gap only sets where that scene sits.

Pure functions of (surface distance, IPD, limits) — no session, no layers, so
the maths is testable on its own.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Fallback until the runtime reports eye poses. Adult mean.
DEFAULT_IPD_MM = 63.0

# At a gap equal to the IPD the eyes' rays are parallel and the feed sits at
# infinity; past it they would have to diverge, which eyes cannot do and which
# is the classic cause of stereo eye strain. Stay clear of the cliff.
MAX_FRACTION_OF_IPD = 0.9

# Where the suggestion aims the far end of the scene. Far enough that the
# scene spreads out, near enough that the suggestion keeps clear of the
# ceiling instead of pinning to it -- advice that is always "the maximum" is
# not advice.
FAR_TARGET_M = 6.0

# The gap is quantised to this, so held-stick values land on clean numbers
# and match what the readout prints.
STEP_CM = 0.1


def step(value: float) -> float:
    """Snap to :data:`STEP_CM`.

    Apply on the way out, never to the running total: one frame at 60 Hz
    moves less than half a step, so rounding the total every frame would
    round it straight back and the stick would never move at all.
    """
    return round(value / STEP_CM) * STEP_CM


def max_gap_cm(ipd_mm: float, configured_max_cm: float) -> float:
    """Ceiling: the configured one, or the IPD-derived one if tighter."""
    return min(configured_max_cm, ipd_mm / 10.0 * MAX_FRACTION_OF_IPD)


def clamp_gap_cm(value: float, ipd_mm: float, limits: Tuple[float, float]) -> float:
    minimum, maximum = limits
    return min(max(value, minimum), max_gap_cm(ipd_mm, maximum))


def suggested_gap_cm(
    surface_m: Optional[float], ipd_mm: float, limits: Tuple[float, float]
) -> Optional[float]:
    """A starting gap for a surface ``surface_m`` away, or None when the shape
    has no meaningful distance (an infinite sphere).

    Stepped as well as clamped: a suggestion you cannot dial in exactly is no
    use.
    """
    if surface_m is None:
        return None
    ideal = ipd_mm / 10.0 * (1.0 - surface_m / FAR_TARGET_M)
    return step(clamp_gap_cm(ideal, ipd_mm, limits))


def perceived_distance_cm(
    surface_m: Optional[float], ipd_mm: float, gap_cm: float
) -> Optional[float]:
    """Where zero-disparity content currently appears, in cm.

    None when the geometry does not define it: an infinite sphere, or a gap
    at the IPD where the rays are parallel and the answer is infinity.
    """
    if surface_m is None:
        return None
    denominator = ipd_mm - gap_cm * 10.0
    if denominator <= 1e-3:
        return None
    return surface_m * ipd_mm / denominator * 100.0
