# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One frame reduced to what a viewer draws. Pure Python, no renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import nan
from typing import Protocol

Vec3 = tuple[float, float, float]


class Topology(Protocol):
    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def bones(self) -> tuple[tuple[int, int], ...]: ...


@dataclass(frozen=True, slots=True)
class Sample:
    """One frame reduced to what a viewer draws.

    ``positions`` holds a joint at its last valid position for as long as it stays
    invalid, and ``None`` for one that has never been valid. The recorded position of
    an invalid joint is deliberately dropped here: invalid joints carry arbitrary
    values — a quaternion component of −16363.96 was measured on real hardware — and
    one of those in a point cloud moves the camera so far that nothing else is visible.
    ``valid`` is what says whether a drawn joint is live or held.
    """

    sequence: int
    t_s: float
    positions: tuple[Vec3 | None, ...]
    valid: tuple[bool, ...]
    interval_ms: float | None
    step: str | None

    @property
    def valid_count(self) -> int:
        return sum(self.valid)

    @property
    def rate_hz(self) -> float:
        """NaN for the first frame and for any interval that did not move forward.

        A non-monotonic timestamp is evidence, not something to smooth over, and the
        plot draws a NaN as a break in the line.
        """
        if self.interval_ms is None or self.interval_ms <= 0.0:
            return nan
        return 1000.0 / self.interval_ms
