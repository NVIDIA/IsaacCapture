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

    ``positions`` holds each joint's last valid position, ``None`` if never valid;
    ``valid`` says whether a drawn joint is live or held.
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
        """NaN for the first frame and for any interval that did not move forward."""
        if self.interval_ms is None or self.interval_ms <= 0.0:
            return nan
        return 1000.0 / self.interval_ms
