# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The setpoints of one move, sampled as APAI's own driver samples them.

A move is a list of setpoints one period apart from a start pose to a goal
pose. The start pose is not among them; the last one is the goal. Nothing here
imports APAI.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def min_jerk(s: float) -> float:
    """APAI's time law (apai.control.motion_profile.min_jerk): 0 at 0, 1 at 1."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def setpoint_count(duration_s: float, rate_hz: float) -> int:
    """How many setpoints a move of this length has: at least two."""
    if not (math.isfinite(duration_s) and math.isfinite(rate_hz)):
        raise ValueError("duration_s and rate_hz must be finite")
    if duration_s < 0 or rate_hz <= 0:
        raise ValueError("duration_s must be 0 or more and rate_hz positive")
    return max(2, int(duration_s * rate_hz))


def sample_move(
    start: Sequence[float],
    target: Sequence[float],
    duration_s: float,
    rate_hz: float,
    time_law: Callable[[float], float] = min_jerk,
) -> list[np.ndarray]:
    """The setpoints ArmBase.stream_to sends for this move, value for value."""
    q_start = np.asarray(start, dtype=float).reshape(-1)
    q_target = np.asarray(target, dtype=float).reshape(-1)
    if q_start.shape != q_target.shape:
        raise ValueError(
            f"the start pose has {q_start.size} joints, the goal {q_target.size}"
        )
    steps = setpoint_count(duration_s, rate_hz)
    return [
        q_start + (q_target - q_start) * time_law(i / steps)
        for i in range(1, steps + 1)
    ]


def largest_step(start: Sequence[float], setpoint: Sequence[float]) -> float:
    """The largest change of one joint from one pose to the next, in rad."""
    return float(
        np.max(np.abs(np.asarray(setpoint, dtype=float) - np.asarray(start, float)))
    )
