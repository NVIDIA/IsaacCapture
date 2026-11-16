# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The check ahead: vets setpoints the Thor has not sent yet on APAI's obstacle grid.

One CheckAhead stands on the workstation checker (APAI's SafetyHarness with its
obstacle grid). A MoveCheck does one pass of the permit loop for one move: it
clears the next setpoints with the checker callback and says how far a PERMIT
may reach and which camera frame its grid came from.

CheckRefused subclasses APAI's SafetyViolation, so the class is made on first
use: read it as `checkahead.CheckRefused`, or import it inside a function.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_log = logging.getLogger("isaaccapture.examples.agentic_remote.checkahead")

NO_FRAME_TIME = "the grid has no known camera-frame time"
FRAME_TIMES_KEPT = 256
# How often vet_whole starts when the grid changes under it.
WHOLE_TRIES = 3
_FRESH_POLL_S = 0.01

_refused_lock = threading.Lock()
_refused_type: type | None = None


def _check_refused_type() -> type:
    global _refused_type
    with _refused_lock:
        if _refused_type is None:
            from apai.types import SafetyViolation

            class CheckRefused(SafetyViolation):
                """The check ahead refused setpoint `index` of a move.

                count is the number of setpoints of the move, when known.
                """

                def __init__(
                    self, index: int, reason: str, count: int | None = None
                ) -> None:
                    self.index = int(index)
                    self.reason = str(reason)
                    self.count = None if count is None else int(count)
                    of = "" if self.count is None else f" of {self.count}"
                    super().__init__(
                        f"check ahead refused setpoint {self.index}{of}: {self.reason}"
                    )

            CheckRefused.__module__ = __name__
            _refused_type = CheckRefused
        return _refused_type


def __getattr__(name: str) -> Any:
    if name == "CheckRefused":
        return _check_refused_type()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class GridBusy(Exception):
    """The grid lock was not free within lock_wait_s; nothing was checked."""


@dataclass(frozen=True)
class Evidence:
    """Which grid a check used: its camera-frame time and its stamp."""

    wall_s: float  # capture time of the grid's camera frame, the Thor's wall clock
    grid_stamp: tuple[int, int]  # (refreshes, cache_clears) of the grid


@dataclass(frozen=True)
class Grant:
    """What one pass allows a PERMIT to say."""

    through_index: int
    evidence_wall_s: float


class FrameTimes:
    """Capture times of the newest camera frames, by their Frame.t. Thread-safe.

    frame_t is the workstation's monotonic time APAI stamped the camera frame
    with; evidence_wall_s is its capture time on the Thor's wall clock.
    """

    def __init__(self, keep: int = FRAME_TIMES_KEPT) -> None:
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self._keep = int(keep)
        self._lock = threading.Lock()
        self._times: OrderedDict[float, float] = OrderedDict()

    def note(self, frame_t: float, evidence_wall_s: float) -> None:
        key, wall = float(frame_t), float(evidence_wall_s)
        if not (math.isfinite(key) and math.isfinite(wall)):
            raise ValueError("camera-frame times must be finite")
        with self._lock:
            self._times.pop(key, None)
            self._times[key] = wall
            while len(self._times) > self._keep:
                self._times.popitem(last=False)

    def lookup(self, frame_t: float) -> float | None:
        with self._lock:
            return self._times.get(float(frame_t))

    def clear(self) -> None:
        with self._lock:
            self._times.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._times)


def _pose(q: Sequence[float]) -> np.ndarray:
    return np.asarray(q, dtype=float)


class CheckAhead:
    """Vets setpoints on the workstation checker's grid, under the grid's lock.

    The vets run on the calling thread, which must be the skill thread: the
    checker keeps per-thread state. grid_stamp and wait_fresh_grid take no lock.
    """

    def __init__(
        self,
        harness: Any,
        frame_times: FrameTimes,
        *,
        period_s: float,
        lock_wait_s: float = 0.05,
        hold_max: int = 64,
        whole_lock_wait_s: float = 2.0,
    ) -> None:
        if getattr(harness, "occupancy", None) is None:
            raise ValueError("the check ahead needs a checker with an obstacle grid")
        if not (math.isfinite(period_s) and period_s > 0):
            raise ValueError("period_s must be positive")
        if hold_max < 1:
            raise ValueError("hold_max must be at least 1")
        self._harness = harness
        self._grid = harness.occupancy
        self._lock = self._grid._refresh_lock
        self._frame_times = frame_times
        self.period_s = float(period_s)
        self.lock_wait_s = float(lock_wait_s)
        self.hold_max = int(hold_max)
        self.whole_lock_wait_s = float(whole_lock_wait_s)

    @property
    def harness(self) -> Any:
        return self._harness

    def grid_stamp(self) -> tuple[int, int]:
        return (int(self._grid.refreshes), int(self._grid.cache_clears))

    def wait_fresh_grid(self, max_age_s: float, timeout_s: float) -> None:
        """Return once the grid's camera frame is at most max_age_s old."""
        deadline = time.monotonic() + timeout_s
        while True:
            now = time.monotonic()
            frame_t = self._grid._last_refresh
            if frame_t is not None and 0.0 <= now - frame_t <= max_age_s:
                return
            if now >= deadline:
                from apai.types import SafetyViolation

                raise SafetyViolation(
                    f"no camera frame newer than {max_age_s:g} s in the obstacle "
                    "grid; no motion sent"
                )
            time.sleep(min(_FRESH_POLL_S, deadline - now))

    def vet(
        self,
        start_q: Sequence[float],
        setpoints: Sequence[Sequence[float]],
        first: int,
        last: int,
    ) -> Evidence:
        """vet_step for setpoints first..last under one hold of the grid lock."""
        return self._vet_hold(start_q, setpoints, first, last, self.lock_wait_s)

    def vet_whole(
        self, start_q: Sequence[float], setpoints: Sequence[Sequence[float]]
    ) -> Evidence:
        """vet_step for every setpoint, at most hold_max per hold of the lock.

        The evidence is of the grid every setpoint was vetted on. A grid
        installed between two holds starts the vet over, WHOLE_TRIES times at
        most; the last try goes on and returns the oldest evidence it used.
        """
        count = len(setpoints)
        if count == 0:
            raise ValueError("the move has no setpoints")
        for attempt in range(1, WHOLE_TRIES + 1):
            oldest: Evidence | None = None
            for first in range(0, count, self.hold_max):
                last = min(count, first + self.hold_max) - 1
                try:
                    evidence = self._vet_hold(
                        start_q, setpoints, first, last, self.whole_lock_wait_s
                    )
                except GridBusy:
                    raise self._refused(
                        first,
                        f"the obstacle grid was busy for {self.whole_lock_wait_s:g} s",
                        count,
                    ) from None
                if oldest is None:
                    oldest = evidence
                elif evidence.grid_stamp != oldest.grid_stamp:
                    if attempt < WHOLE_TRIES:
                        oldest = None
                        break
                    if evidence.wall_s < oldest.wall_s:
                        oldest = evidence
            if oldest is not None:
                return oldest
        raise AssertionError("unreachable")

    def begin(
        self,
        approve: Callable[[Any, Any, float], None],
        start_q: Sequence[float],
        setpoints: Sequence[Sequence[float]],
    ) -> MoveCheck:
        return MoveCheck(self, approve, start_q, setpoints)

    def _vet_hold(
        self,
        start_q: Sequence[float],
        setpoints: Sequence[Sequence[float]],
        first: int,
        last: int,
        wait_s: float,
    ) -> Evidence:
        count = len(setpoints)
        if not 0 <= first <= last < count:
            raise ValueError(f"setpoints {first}..{last} are not within 0..{count - 1}")
        if not self._lock.acquire(timeout=wait_s):
            raise GridBusy(f"the obstacle grid lock was not free in {wait_s:g} s")
        try:
            stamp, wall_s = self._grid_now()
            prev = _pose(start_q if first == 0 else setpoints[first - 1])
            for index in range(first, last + 1):
                target = _pose(setpoints[index])
                reason = self._harness.vet_step(prev, target, self.period_s)
                if reason is not None:
                    raise self._refused(index, reason, count)
                prev = target
            if wall_s is None:
                raise self._refused(first, NO_FRAME_TIME, count)
            return Evidence(wall_s=wall_s, grid_stamp=stamp)
        finally:
            self._lock.release()

    def _grid_now(self) -> tuple[tuple[int, int], float | None]:
        """The grid's stamp and its camera frame's capture time (lock held)."""
        frame_t = self._grid._last_refresh
        wall_s = None if frame_t is None else self._frame_times.lookup(frame_t)
        return self.grid_stamp(), wall_s

    def _refused(self, index: int, reason: str, count: int) -> BaseException:
        """A CheckRefused, filed in the checker's violations as its own are."""
        self._harness.violations.append(reason)
        refusal = _check_refused_type()(index, reason, count)
        _log.info("%s", refusal)
        return refusal


class MoveCheck:
    """The check ahead of one move. Only the thread that made it may step it."""

    def __init__(
        self,
        check: CheckAhead,
        approve: Callable[[Any, Any, float], None],
        start_q: Sequence[float],
        setpoints: Sequence[Sequence[float]],
    ) -> None:
        if len(setpoints) == 0:
            raise ValueError("the move has no setpoints")
        self._check = check
        self._approve = approve
        self._start = _pose(start_q)
        self._q = [_pose(row) for row in setpoints]
        self._thread = threading.get_ident()
        self._stamp: tuple[int, int] | None = None
        self.approved_through = -1
        self.refusal: BaseException | None = None

    @property
    def count(self) -> int:
        return len(self._q)

    def step(self, sent: int, reach: int) -> Grant | None:
        """One pass: how far a PERMIT may reach now, or None for no permit.

        sent is the number of setpoints the Thor reports as sent; the PERMIT
        reaches at most `reach` setpoints beyond the last of them and never
        past the move's last one. None means the grid lock was busy, nothing
        is cleared yet, or a refusal is set.
        """
        if threading.get_ident() != self._thread:
            raise RuntimeError("a MoveCheck is stepped only on the skill thread")
        if self.refusal is not None:
            return None
        check = self._check
        if not check._lock.acquire(timeout=check.lock_wait_s):
            return None
        try:
            return self._pass(max(0, min(int(sent), self.count)), int(reach))
        finally:
            check._lock.release()

    def _prev(self, index: int) -> np.ndarray:
        return self._start if index == 0 else self._q[index - 1]

    def _vet(self, index: int) -> bool:
        check = self._check
        reason = check._harness.vet_step(
            self._prev(index), self._q[index], check.period_s
        )
        if reason is not None:
            self.refusal = check._refused(index, reason, self.count)
        return reason is None

    def _pass(self, sent: int, reach: int) -> Grant | None:
        check = self._check
        stamp, wall_s = check._grid_now()
        checked = 0
        if stamp != self._stamp:
            # A new grid: what was cleared and may not be sent yet is vetted again.
            for index in range(sent, self.approved_through + 1):
                checked += 1
                if not self._vet(index):
                    return None
            self._stamp = stamp
        # sent is a count: the last setpoint sent has index sent - 1.
        last = min(
            self.count - 1, sent + reach - 1, self.approved_through + check.hold_max
        )
        for index in range(self.approved_through + 1, last + 1):
            checked += 1
            try:
                self._approve(self._prev(index), self._q[index], check.period_s)
            except Exception as exc:
                self.refusal = exc
                return None
            self.approved_through = index
        if checked == 0 and sent < self.count and not self._vet(sent):
            return None
        if wall_s is None:
            self.refusal = check._refused(sent, NO_FRAME_TIME, self.count)
            return None
        if self.approved_through < 0:
            return None
        return Grant(through_index=self.approved_through, evidence_wall_s=wall_s)
