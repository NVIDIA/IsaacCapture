# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The timed step loop: cue, 3-2-1, hold, end tone, next. Nobody presses anything.

A trigger pull moves a finger and the controller may be strapped to the wrist as the
wrist source, so no input opens a window. No viser, isaacteleop, audio or clock here:
``advance()`` returns what to play and the caller plays it, so
``tests/test_steps.py`` can drive a whole take in memory.

Windows are stamped on the caller's clock, which must be the session's system
monotonic clock -- the one ``sample_time_local_common_clock`` is in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from session import BEAT_S, COUNTDOWN_BEATS, STEPS, STILL_LABELS

TIMER = "timer"


class Phase(Enum):
    IDLE = "idle"
    CUEING = "cueing"
    WAITING = "waiting"  # the 3-2-1 countdown; ends on its own
    HOLDING = "holding"
    CLOSING = "closing"
    DONE = "done"


class Sound(Enum):
    CUE = "cue"
    COUNT = "count"
    START = "start"
    END = "end"


@dataclass(frozen=True, slots=True)
class Window:
    """One held pose as the half-open span ``[start_ns, end_ns)`` on the session clock."""

    index: int
    label: str
    start_ns: int
    end_ns: int
    is_still_window: bool
    source: str = TIMER


class Take:
    def __init__(
        self,
        cue_seconds: Mapping[str, float],
        end_tone_s: float,
        steps: Sequence[tuple[str, float, str]] = tuple(STEPS),
        beats: int = COUNTDOWN_BEATS,
        beat_s: float = BEAT_S,
    ) -> None:
        self._steps = tuple(steps)
        missing = [label for label, _, _ in self._steps if label not in cue_seconds]
        if missing:
            raise ValueError(f"no spoken length for {missing}")
        self._cue_seconds = dict(cue_seconds)
        self._end_tone_s = end_tone_s
        self._beats = beats
        self._beat_s = beat_s
        self._phase = Phase.IDLE
        self._index = 0
        self._ends_ns: int | None = None
        self._beats_left = 0
        self._opened_ns: int | None = None
        self._windows: list[Window] = []

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def index(self) -> int:
        return self._index

    @property
    def steps(self) -> tuple[tuple[str, float, str], ...]:
        return self._steps

    @property
    def count(self) -> int:
        return len(self._steps)

    @property
    def label(self) -> str:
        return self._steps[self._index][0]

    @property
    def duration_s(self) -> float:
        return self._steps[self._index][1]

    @property
    def cue_text(self) -> str:
        return self._steps[self._index][2]

    @property
    def next_cue_text(self) -> str | None:
        following = self._index + 1
        return self._steps[following][2] if following < self.count else None

    @property
    def beats_left(self) -> int:
        return self._beats_left

    @property
    def windows(self) -> tuple[Window, ...]:
        return tuple(self._windows)

    def remaining_s(self, now_ns: int) -> float | None:
        if self._phase is not Phase.HOLDING or self._ends_ns is None:
            return None
        return max(0.0, (self._ends_ns - now_ns) / 1e9)

    def start(self, now_ns: int) -> Sound | None:
        """Speak the first cue; called once the device has produced a usable frame."""
        if self._phase is not Phase.IDLE:
            return None
        self._enter_cueing(now_ns)
        return Sound.CUE

    def advance(self, now_ns: int) -> Sound | None:
        """At most one transition per call; call it every tick."""
        if self._ends_ns is None or now_ns < self._ends_ns:
            return None
        if self._phase is Phase.CUEING:
            self._phase = Phase.WAITING
            self._beats_left = self._beats
            return self._beat(now_ns)
        if self._phase is Phase.WAITING:
            if self._beats_left > 0:
                return self._beat(now_ns)
            self._opened_ns = now_ns
            self._phase = Phase.HOLDING
            self._ends_ns = now_ns + int(self.duration_s * 1e9)
            return Sound.START
        if self._phase is Phase.HOLDING:
            self._close(now_ns)
            if self._index + 1 >= self.count:
                self._phase, self._ends_ns = Phase.DONE, None
            else:
                self._phase = Phase.CLOSING
                self._ends_ns = now_ns + int(self._end_tone_s * 1e9)
            return Sound.END
        if self._phase is Phase.CLOSING:
            self._index += 1
            self._enter_cueing(now_ns)
            return Sound.CUE
        return None

    def _beat(self, now_ns: int) -> Sound:
        self._beats_left -= 1
        self._ends_ns = now_ns + int(self._beat_s * 1e9)
        return Sound.COUNT

    def _enter_cueing(self, now_ns: int) -> None:
        self._phase = Phase.CUEING
        self._ends_ns = now_ns + int(self._cue_seconds[self.label] * 1e9)

    def _close(self, now_ns: int) -> None:
        assert self._opened_ns is not None, "HOLDING without an opened window"
        self._windows.append(
            Window(
                index=self._index,
                label=self.label,
                start_ns=self._opened_ns,
                end_ns=now_ns,
                is_still_window=self.label in STILL_LABELS,
            )
        )
        self._opened_ns = None
