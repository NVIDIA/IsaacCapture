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
