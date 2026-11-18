# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimates the table height from the first flat window, for drawing the grid on it.

Display only.
"""

from __future__ import annotations

from hand_cts.checks.windows import flat_height, table_floor
from hand_cts.frames import Frame
from hand_cts.labels import FLAT_WINDOWS, SETTLE_FRACTION

from steps import Phase, Take

MIN_FRAMES = 30


class TableEstimate:
    def __init__(self) -> None:
        self._heights: list[float] = []
        self.height_m: float | None = None

    def update(self, take: Take, now_ns: int, frames: dict[str, Frame | None]) -> bool:
        """Feed one tick; True on the tick the estimate is first made."""
        if self.height_m is not None:
            return False
        if take.windows and take.windows[0].label == FLAT_WINDOWS[0]:
            if len(self._heights) < MIN_FRAMES:
                return False
            self.height_m = table_floor(self._heights)
            return True
        if take.phase is not Phase.HOLDING or take.label != FLAT_WINDOWS[0]:
            return False
        remaining = take.remaining_s(now_ns) or 0.0
        if remaining > (1.0 - SETTLE_FRACTION) * take.duration_s:
            return False
        for frame in frames.values():
            if frame is not None:
                height = flat_height(frame)
                if height is not None:
                    self._heights.append(height)
        return False
