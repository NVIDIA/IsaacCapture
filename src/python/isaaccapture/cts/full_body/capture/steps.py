# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step loop: cue the pose, wait for the performer, hold for the step's own length.

``advance()`` and ``press()`` return what to play; the caller plays it.
Boundaries are record numbers, not times.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from session import STEPS, STILL_LABELS

TRIGGER = "trigger"
KEYBOARD = "keyboard"


class Phase(Enum):
    """Where in one step the take is.

    A press opens a window only in ``WAITING``; in ``CUEING`` it cuts the cue short.
    In ``HOLDING`` and ``CLOSING`` it is swallowed with no tone, window or visible reaction.
    """

    IDLE = "idle"
    CUEING = "cueing"
    WAITING = "waiting"
    HOLDING = "holding"
    CLOSING = "closing"
    DONE = "done"


class Sound(Enum):
    CUE = "cue"
    START = "start"
    END = "end"


class Press(Enum):
    IGNORED = "ignored"
    CUT_CUE = "cut_cue"
    ACCEPTED = "accepted"


@dataclass(frozen=True, slots=True)
class Window:
    """One measured pose, as the half-open record range ``[start_frame, end_frame)``.

    ``source`` is the input that opened it.
    """

    index: int
    label: str
    start_frame: int
    end_frame: int
    is_still_window: bool
    source: str


class Take:
    """One performance of the script, paced by the performer."""

    def __init__(
        self,
        cue_seconds: Mapping[str, float],
        end_tone_s: float,
        steps: Sequence[tuple[str, float, str]] = tuple(STEPS),
    ) -> None:
        self._steps = tuple(steps)
        missing = [label for label, _, _ in self._steps if label not in cue_seconds]
        if missing:
            raise ValueError(f"no spoken length for {missing}")
        self._cue_seconds = dict(cue_seconds)
        self._end_tone_s = end_tone_s
        self._phase = Phase.IDLE
        self._index = 0
        # When the current phase runs out; None while it ends on a press.
        self._ends_s: float | None = None
        self._opened: tuple[int, str] | None = None
        self._windows: list[Window] = []

    # -- what the sidebar reads -------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def index(self) -> int:
        return self._index

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
    def done_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _, _ in self._steps[: self._index])

    @property
    def todo_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _, _ in self._steps[self._index + 1 :])

    @property
    def windows(self) -> tuple[Window, ...]:
        return tuple(self._windows)

    def remaining_s(self, now_s: float) -> float | None:
        """Seconds left in this pose, or None when nothing is being timed."""
        if self._phase is not Phase.HOLDING or self._ends_s is None:
            return None
        return max(0.0, self._ends_s - now_s)

    def elapsed_fraction(self, now_s: float) -> float:
        remaining = self.remaining_s(now_s)
        if remaining is None or self.duration_s <= 0.0:
            return 0.0
        return min(1.0, max(0.0, 1.0 - remaining / self.duration_s))

    # -- what drives it ---------------------------------------------------------

    def start(self, now_s: float) -> Sound | None:
        """Speak the first cue. Called once the device has produced a usable frame."""
        if self._phase is not Phase.IDLE:
            return None
        self._enter_cueing(now_s)
        return Sound.CUE

    def advance(self, now_s: float, frame: int) -> Sound | None:
        """Run the transitions time drives.

        ``frame`` is the number of records written so far, the first one after the window.
        """
        if self._ends_s is None or now_s < self._ends_s:
            return None
        if self._phase is Phase.CUEING:
            self._phase, self._ends_s = Phase.WAITING, None
            return None
        if self._phase is Phase.HOLDING:
            self._close(frame)
            if self._index + 1 >= self.count:
                self._phase, self._ends_s = Phase.DONE, None
            else:
                self._phase, self._ends_s = Phase.CLOSING, now_s + self._end_tone_s
            return Sound.END
        if self._phase is Phase.CLOSING:
            self._index += 1
            self._enter_cueing(now_s)
            return Sound.CUE
        return None

    def press(self, now_s: float, frame: int, source: str) -> Press:
        """Offer a trigger pull or a key press.

        ``frame`` is the record the press was observed on, which becomes the window's first.
        """
        if self._phase is Phase.CUEING:
            self._phase, self._ends_s = Phase.WAITING, None
            return Press.CUT_CUE
        if self._phase is Phase.WAITING:
            self._opened = (frame, source)
            self._phase, self._ends_s = Phase.HOLDING, now_s + self.duration_s
            return Press.ACCEPTED
        return Press.IGNORED

    # -- internals --------------------------------------------------------------

    def _enter_cueing(self, now_s: float) -> None:
        self._phase = Phase.CUEING
        self._ends_s = now_s + self._cue_seconds[self.label]

    def _close(self, frame: int) -> None:
        assert self._opened is not None, "HOLDING without an accepted press"
        start_frame, source = self._opened
        self._opened = None
        self._windows.append(
            Window(
                index=self._index,
                label=self.label,
                start_frame=start_frame,
                end_frame=frame,
                is_still_window=self.label in STILL_LABELS,
                source=source,
            )
        )
