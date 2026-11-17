# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text for a reason, an operator line or the log: printable, cut, not repeated."""

from __future__ import annotations

import threading
from collections.abc import Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .clock import Clock

# The most characters kept of a text that a link message carries.
MAX_ID_CHARS = 128  # command ids and move ids
MAX_WORD_CHARS = 64  # sender, request, action, mode
MAX_REASON_CHARS = 256  # reasons
# The longest reason the robot app puts into a RESULT, an EVENT or ROBOT_STATE.
MAX_OUT_CHARS = 1024
CUT_MARK = "..."

# What an arm port's hold_evidence() says of the motors, and ROBOT_STATE's
# health["motors"]["state"] carries: every joint reads at the target it holds;
# a joint has left it or a motor reports a fault; or it cannot be said.
MOTORS_HOLDING = "holding"
MOTORS_NOT_HOLDING = "not holding"
MOTORS_UNKNOWN = "unknown"
# What an operator line says of the motors: torque_text() gives one of these.
TORQUE_OFF = "torque off"
TORQUE_ON = "torque on"  # the hold is not reported, as by a simulated arm
TORQUE_HOLDING = "torque on (holding)"
TORQUE_NOT_HOLDING = "torque NOT HOLDING"
TORQUE_HOLD_UNKNOWN = "torque on, hold unknown"
TORQUE_UNKNOWN = "torque unknown"  # a torque release failed
_TORQUE_BY_HOLD = {
    MOTORS_HOLDING: TORQUE_HOLDING,
    MOTORS_NOT_HOLDING: TORQUE_NOT_HOLDING,
    MOTORS_UNKNOWN: TORQUE_HOLD_UNKNOWN,
}

# A repeated line is written once per this many seconds of the app clock.
LINE_WINDOW_S = 1.0
# How many kinds of line are told apart in one such window.
LINE_KINDS = 16


def printable(text: object, limit: int) -> str:
    """str(text) for one line: at most `limit` characters, all of them printable.

    A text that was cut ends in CUT_MARK, where the limit has room for it. A
    character that cannot be printed becomes a space.
    """
    text = str(text)
    if len(text) > limit > len(CUT_MARK):
        text = text[: limit - len(CUT_MARK)] + CUT_MARK
    elif len(text) > limit:
        text = text[: max(0, limit)]  # no room for the mark
    if text.isprintable():
        return text
    return "".join(char if char.isprintable() else " " for char in text)


def torque_text(torque_on: object, hold: object, release_failed: bool = False) -> str:
    """What an operator line says of the motors: one of the TORQUE_* texts.

    torque_on says that the robot app switched the motors on and has not
    released them; it is no reading of them. hold is what the joint readings
    say, one of the MOTORS_* words; anything else: not reported.
    release_failed: a torque release failed, so the motors may be on or off.
    """
    if release_failed:
        return TORQUE_UNKNOWN
    if torque_on is not True:
        return TORQUE_OFF
    return _TORQUE_BY_HOLD.get(hold if isinstance(hold, str) else "", TORQUE_ON)


class LineLimit:
    """Lets a line of each kind through once per window and counts its repeats.

    The first LINE_KINDS kinds of a window are told apart; lines of further
    kinds are counted together under the kind None. Thread-safe.
    """

    def __init__(
        self, clock: Clock, window_s: float = LINE_WINDOW_S, kinds: int = LINE_KINDS
    ) -> None:
        self._clock = clock
        self._window_s = float(window_s)
        self._kinds = int(kinds)
        self._lock = threading.Lock()
        self._began: float | None = None  # when the window under way began
        self._repeats: dict[Hashable, int] = {}  # of that window, by kind
        self._others = 0  # lines of that window whose kind was one too many
        self._ended: dict[Hashable, int] = {}  # of windows that ended, not yet told

    def allow(self, kind: Hashable) -> bool:
        """True if a line of this kind is to be written now; else it was counted."""
        with self._lock:
            self._roll()
            if self._began is None:
                self._began = self._clock.monotonic()
            if kind in self._repeats:
                self._repeats[kind] += 1
                return False
            if len(self._repeats) < self._kinds:
                self._repeats[kind] = 0
                return True
            self._others += 1
            return False

    def ended(self) -> list[tuple[Hashable, int]]:
        """(kind, lines not written) for the windows that have ended; told once."""
        with self._lock:
            self._roll()
            told = list(self._ended.items())
            self._ended = {}
            return told

    def _roll(self) -> None:
        """Close the window under way once its time is up; the lock is held."""
        began = self._began
        if began is None or self._clock.monotonic() - began < self._window_s:
            return
        for kind, count in self._repeats.items():
            if count:
                self._count_ended(kind, count)
        if self._others:
            self._count_ended(None, self._others)
        self._repeats, self._others, self._began = {}, 0, None

    def _count_ended(self, kind: Hashable, count: int) -> None:
        # Bounded also when nobody asks ended(): further kinds count as None.
        if kind not in self._ended and len(self._ended) > self._kinds:
            kind = None
        self._ended[kind] = self._ended.get(kind, 0) + count
