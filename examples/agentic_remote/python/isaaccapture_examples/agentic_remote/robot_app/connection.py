# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's side of the robot connection: it dials, and dials again.

See sections 5 and 14 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from .config import MIN_RETRY_S

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from .clock import Clock

_log = logging.getLogger("isaaccapture.examples.agentic_remote.connection")

# Longest close() waits, in real time, for the dialer's thread to end.
_JOIN_S = 5.0
# How many distinct error texts are remembered as logged.
_TEXTS_KEPT = 64
_NOT_TAKEN = "the robot app did not take the new robot connection"


class DialedApp(Protocol):
    """What a RobotDialer asks of the robot app."""

    @property
    def wants_carrier(self) -> bool:
        """Whether attach_carrier() would take a live carrier now."""

    def attach_carrier(self, carrier: Carrier) -> bool:
        """Take the carrier; False when it was not taken."""

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Whether the app has closed, waiting up to timeout."""


class RobotDialer:
    """Keeps the robot app connected: one thread that dials the workstation.

    While the app wants a carrier the thread calls connect(), which returns a
    live Carrier or raises, and attaches the result. When that link is dead it
    closes it and dials again once the app has handled the loss, not sooner than
    retry_s after the link died or the last dial failed. It never touches the
    arm; a lost or a new connection starts no motion in the app.
    """

    def __init__(
        self,
        app: DialedApp,
        connect: Callable[[], Carrier],
        clock: Clock,
        *,
        retry_s: float = 1.0,
        poll_s: float = 0.02,
    ) -> None:
        """retry_s is counted on `clock`; poll_s is counted in real time.

        poll_s is how often the thread looks at the app and the link. Raises
        ValueError for a retry_s below MIN_RETRY_S, which would allow a burst
        of dials.
        """
        if not (math.isfinite(retry_s) and retry_s >= MIN_RETRY_S):
            raise ValueError(
                f"retry_s must be finite and at least {MIN_RETRY_S:g}, got {retry_s!r}"
            )
        if not (math.isfinite(poll_s) and poll_s > 0.0):
            raise ValueError(f"poll_s must be finite and above 0, got {poll_s!r}")
        self._app = app
        self._connect = connect
        self._clock = clock
        self._retry_s = float(retry_s)
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._carrier: Carrier | None = None  # the one this dialer attached
        self._attempts = 0
        self._last_error = ""
        self._logged: set[str] = set()  # error texts logged since the last connection
        self._next_dial_t: float | None = None  # on the clock; None: at once
        self._ending = threading.Event()
        self._ended = threading.Event()
        self._started = False
        self._pass_cond = threading.Condition()
        self._passes_started = 0
        self._passes_finished = 0
        self._thread = threading.Thread(
            target=self._run, name="robot-dialer", daemon=True
        )

    @property
    def running(self) -> bool:
        """True while the dialer's thread runs."""
        return self._thread.is_alive()

    @property
    def attempts(self) -> int:
        """How many times connect() was called."""
        with self._lock:
            return self._attempts

    @property
    def carrier(self) -> Carrier | None:
        """The carrier this dialer attached, until it is found dead."""
        with self._lock:
            return self._carrier

    @property
    def last_error(self) -> str:
        """Why the last dial gave the app no connection; "" after one that did."""
        with self._lock:
            return self._last_error

    def start(self) -> None:
        """Start the thread; it waits until the app runs and wants a carrier."""
        with self._lock:
            if self._ending.is_set():
                raise RuntimeError("the dialer was closed")
            if self._started:
                raise RuntimeError("the dialer was already started")
            self._started = True
        self._thread.start()

    def close(self) -> None:
        """End the thread and close the link this dialer attached.

        The app then handles the loss of its robot connection: it holds. A
        connect() under way is waited for, for at most a few seconds; what it
        returns later is closed unused. Idempotent.
        """
        with self._lock:
            self._ending.set()
            started = self._started
        if started and self._thread is not threading.current_thread():
            self._thread.join(_JOIN_S)
            if self._thread.is_alive():
                _log.warning("the dialer's thread has not ended: a dial is under way")

    def wait_ended(self, timeout_s: float | None = None) -> bool:
        """Wait in real time until the thread has ended; False if it has not."""
        return self._ended.wait(timeout_s)

    def wait_pass(self, timeout_s: float = 2.0) -> bool:
        """Wait in real time for one pass of the thread that began after this call."""
        with self._pass_cond:
            target = self._passes_started + 1
            return self._pass_cond.wait_for(
                lambda: self._passes_finished >= target, timeout_s
            )

    # The dialer's thread

    def _run(self) -> None:
        try:
            while not self._ending.is_set() and not self._app.wait_closed(0.0):
                with self._pass_cond:
                    self._passes_started += 1
                try:
                    self._pass()
                except Exception:
                    _log.exception("the dialer failed; it goes on")
                    self._retry_later()
                with self._pass_cond:
                    self._passes_finished = self._passes_started
                    self._pass_cond.notify_all()
                self._ending.wait(self._poll_s)
        finally:
            self._drop_carrier()
            self._ended.set()
            with self._pass_cond:
                self._pass_cond.notify_all()

    def _pass(self) -> None:
        """Look at the link once; dial when the app wants a carrier and one is due."""
        carrier = self.carrier
        if carrier is not None:
            if carrier.alive:
                return
            reason = getattr(carrier, "close_reason", "")
            self._drop_carrier()
            self._retry_later()
            _log.warning(
                "robot connection lost%s; dialing again in %g s",
                f" ({reason})" if reason else "",
                self._retry_s,
            )
            return
        due = self._next_dial_t
        if due is not None and self._clock.monotonic() < due:
            return
        if not self._app.wants_carrier:
            return
        self._dial()

    def _dial(self) -> None:
        with self._lock:
            self._attempts += 1
        try:
            carrier = self._connect()
        except Exception as error:
            self._failed(f"{type(error).__name__}: {error}", str(error))
            return
        taken = False
        try:
            taken = not self._ending.is_set() and self._app.attach_carrier(carrier)
        finally:
            if not taken:
                carrier.close()
        if not taken:
            if not self._ending.is_set():
                self._failed(_NOT_TAKEN, _NOT_TAKEN)
            return
        with self._lock:
            self._carrier = carrier
            self._last_error = ""
            self._logged.clear()
            self._next_dial_t = None
        _log.info("robot connection made after %d dial(s)", self.attempts)

    def _failed(self, line: str, text: str) -> None:
        """Note a dial that gave no connection; each distinct text is logged once."""
        self._retry_later()
        with self._lock:
            self._last_error = text
            news = line not in self._logged
            if news:
                if len(self._logged) >= _TEXTS_KEPT:
                    self._logged.clear()
                self._logged.add(line)
        if news:
            _log.warning(
                "no robot connection: %s; dialing again every %g s",
                line,
                self._retry_s,
            )

    def _retry_later(self) -> None:
        with self._lock:
            self._next_dial_t = self._clock.monotonic() + self._retry_s

    def _drop_carrier(self) -> None:
        """Close the link this dialer attached, if it has one."""
        with self._lock:
            carrier, self._carrier = self._carrier, None
        if carrier is not None:
            carrier.close()
