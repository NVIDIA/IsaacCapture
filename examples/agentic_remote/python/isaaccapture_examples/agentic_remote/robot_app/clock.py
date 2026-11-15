# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clocks: the machine's, and a fake one that moves only when a test tells it to."""

from __future__ import annotations

import math
import threading
import time
from typing import Protocol, runtime_checkable


def _duration(s: float) -> float:
    if not math.isfinite(s) or s < 0.0:
        raise ValueError(f"a duration must be finite and not negative, got {s!r}")
    return float(s)


@runtime_checkable
class Clock(Protocol):
    """The time source every robot app module takes instead of calling time.*."""

    def monotonic(self) -> float:
        """Seconds on a clock that never goes back."""

    def wall(self) -> float:
        """Seconds since the Unix epoch."""

    def sleep(self, s: float) -> None:
        """Wait s seconds; a negative or non-finite s raises ValueError."""


class SystemClock:
    """The machine's clocks."""

    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, s: float) -> None:
        time.sleep(_duration(s))


class FakeClock:
    """A clock for tests: time moves only through advance() and sleep()."""

    def __init__(self, start: float = 1000.0, wall_offset: float = 1.7e9) -> None:
        if not (math.isfinite(start) and math.isfinite(wall_offset)):
            raise ValueError("start and wall_offset must be finite")
        self._lock = threading.Lock()
        self._now = float(start)
        self._wall_offset = float(wall_offset)

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def wall(self) -> float:
        with self._lock:
            return self._now + self._wall_offset

    def advance(self, s: float) -> None:
        """Move both clocks forward by s seconds."""
        s = _duration(s)
        with self._lock:
            self._now += s

    def sleep(self, s: float) -> None:
        """Advance by s seconds and return at once."""
        self.advance(s)
