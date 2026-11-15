# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The assistance wait: how long the arm holds for a person after the agent failed.

See section 10 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable

from .clock import Clock
from .config import WaitsConfig

_log = logging.getLogger("isaaccapture.examples.agentic_remote.waits")

# Event kinds returned by AssistanceWait.step().
COUNTDOWN = "countdown"
BELL = "bell"
EXPIRED = "expired"

BELL_TEXT_REST = "ARM WILL MOVE TO REST"
BELL_TEXT_HOLD = "ARM WILL KEEP HOLDING"

# The most time one step() call counts: a stalled app or a clock that jumps
# cannot run the wait out without the warning bells.
MAX_STEP_S = 1.0
COUNTDOWN_EVERY_S = 10

WaitEvent = tuple[str, str]  # (kind, text for the operator)


class AssistanceWait:
    """Counts the assistance wait down and says what to tell the operator.

    Only step() moves the count. Thread-safe; on_expire(timed_return) runs on the
    thread that calls step(), outside the lock, so it may call back into the wait.
    """

    def __init__(
        self, config: WaitsConfig, clock: Clock, on_expire: Callable[[bool], None]
    ) -> None:
        self._config = config
        self._clock = clock
        self._on_expire = on_expire
        self._lock = threading.Lock()
        self._left: float | None = None  # seconds left; None when no wait runs
        self._last = 0.0  # app-clock time the count has reached
        self._reason = ""
        self._timed_return = True
        self._announce = False  # the next step() reports the start
        self._next_countdown = 0  # seconds left at the next countdown line
        self._bells_rung = 0  # how many of config.bell_at_s have rung

    def start(self, reason: str, timed_return: bool = True) -> None:
        """Start the wait at its full length; a running wait starts over.

        timed_return tells whether the caller returns the arm to rest at the end.
        """
        full = self._config.assistance_s
        with self._lock:
            restarted = self._left is not None
            self._left = full
            self._last = self._clock.monotonic()
            self._reason = reason
            self._timed_return = bool(timed_return)
            self._announce = True
            self._next_countdown = COUNTDOWN_EVERY_S * (
                math.ceil(full / COUNTDOWN_EVERY_S) - 1
            )
            self._bells_rung = 0
        _log.info(
            "assistance wait %s: %g s, timed return %s: %s",
            "restarted" if restarted else "started",
            full,
            "on" if timed_return else "off",
            reason,
        )

    def cancel(self, reason: str) -> bool:
        """End a running wait without an expiry; True if one was running."""
        with self._lock:
            left = self._left
            self._left = None
        if left is None:
            return False
        _log.info("assistance wait cancelled with %.0f s left: %s", left, reason)
        return True

    def active(self) -> bool:
        with self._lock:
            return self._left is not None

    def remaining(self, now: float) -> float | None:
        """Seconds left as step(now) would count them; None when no wait runs."""
        with self._lock:
            if self._left is None:
                return None
            return max(0.0, self._left - self._elapsed(now))

    def step(self, now: float) -> list[WaitEvent]:
        """Count the time up to `now` and return the events that fell due.

        When the wait runs out it ends, then on_expire(timed_return) is called once.
        """
        events: list[WaitEvent] = []
        with self._lock:
            if self._left is None:
                return events
            left = max(0.0, self._left - self._elapsed(now))
            if self._last < now < math.inf:
                self._last = now
            if self._announce:
                self._announce = False
                events.append((COUNTDOWN, self._start_text(left)))
            if 0 < self._next_countdown and left <= self._next_countdown:
                text = f"assistance wait: {self._next_countdown} s left"
                events.append((COUNTDOWN, text))
                self._next_countdown -= COUNTDOWN_EVERY_S
            marks = self._config.bell_at_s
            while self._bells_rung < len(marks) and left <= marks[self._bells_rung]:
                events.append((BELL, self._bell_text(marks[self._bells_rung])))
                self._bells_rung += 1
            expired = left <= 0.0
            timed_return = self._timed_return
            if expired:
                events.append((EXPIRED, self._expired_text()))
                self._left = None
            else:
                self._left = left
        if expired:
            _log.info(
                "assistance wait ran out, timed return %s",
                "on" if timed_return else "off",
            )
            self._on_expire(timed_return)
        return events

    def _elapsed(self, now: float) -> float:
        """Time a step at `now` counts: none for an older or non-finite `now`."""
        dt = now - self._last
        if not 0.0 < dt < math.inf:
            return 0.0
        return min(dt, MAX_STEP_S)

    def _start_text(self, left: float) -> str:
        then = "moves to rest" if self._timed_return else "keeps holding"
        why = f" ({self._reason})" if self._reason else ""
        return f"assistance wait: {left:.0f} s left, then the arm {then}{why}"

    def _bell_text(self, mark: float) -> str:
        if self._timed_return:
            return f"{BELL_TEXT_REST} in {mark:g} s"
        return f"{BELL_TEXT_HOLD}, assistance wait ends in {mark:g} s"

    def _expired_text(self) -> str:
        if self._timed_return:
            return "assistance wait over: timed return to rest requested"
        return "assistance wait over: the arm keeps holding"
