# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The jaws: what they may hold, and the one thread that runs their commands.

Also how they follow the trigger of a person who owns the arm. See sections 9,
11 and 12 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

from ..protocol.messages import GripperState
from .interfaces import (
    ArmFault,
    GripperOutcome,
    JawLoad,
    Owner,
    StopLevel,
)

if TYPE_CHECKING:
    from .authority import Authority
    from .clock import Clock
    from .config import GripperConfig
    from .interfaces import ArmPort, AuthorityState

_log = logging.getLogger("isaaccapture.examples.agentic_remote.gripper")

PLAIN = "plain"
CLOSE_UNTIL_CONTACT = "close_until_contact"
JAW_MODES = (PLAIN, CLOSE_UNTIL_CONTACT)
# What a queued jaw reference carries in place of a jaw mode; no link message
# can ask for it.
REFERENCE = "reference"
# What a queued step of the jaws toward a person's trigger carries in place of
# a jaw mode; no link message can ask for it.
FOLLOW = "follow"
ALREADY_REFERENCED = "the jaws are referenced already"
JAWS_BUSY = "busy: a jaw command is running or waiting"
DROPPED_BY_STOP = "dropped by a stop"

# GripperOutcome statuses. FAILED is the worker's own: the driver raised.
REACHED = "reached"
STALLED = "stalled"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
REFUSED = "refused"
FAILED = "failed"

# Effort of each guarded opening step.
OPEN_EFFORT = 0.5
# The most requests that wait behind the running one; a further one is refused.
MAX_WAITING = 4
# Longest stop_now() waits, in real time, for a cancelled command to return.
STOP_WAIT_S = 0.25
# Longest close() waits, in real time, for the worker thread to end.
_CLOSE_WAIT_S = 2.0

_EPS = 1e-9

# Why the trigger of a person who owns the arm moves no jaw (JawFollower). The
# grip level is the jaw opening in force when the person took the arm over.
NOT_MATCHED = "the trigger has not reached the grip level"
NO_OPENING = "the human target carries no jaw opening"
NO_GRIP_LEVEL = "the grip level is not known: "
STEP_OUT = "a jaw step that was cut short has not returned"
# Why the jaws stay wider than the trigger asks.
NO_TIGHTER = "no closing beyond the grip level until the jaws were measured fully open"
NO_CLOSING = "no closing: "

DoneFn = Callable[[GripperOutcome], None]


def _finite(value: float | None) -> float | None:
    """value as a float; None when it is missing or not a finite number."""
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class JawState:
    """What measurements say about the jaws' load; it never moves the jaws.

    Thread-safe. It remembers the newest jaw reading: the end reading of the
    last command, or the readings of the last classify() call.
    """

    def __init__(self, config: GripperConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._may_be_holding = False
        # Lowest and highest value of the newest reading or set of readings; None
        # before the first one and after a failed read.
        self._span: tuple[float, float] | None = None
        self._opening: float | None = None  # the newest reading
        self._t: float | None = None  # its app-clock time
        self._reason = ""

    @property
    def config(self) -> GripperConfig:
        return self._config

    @property
    def may_be_holding(self) -> bool:
        """Set by a close that ended short; cleared only by a measured full open."""
        with self._lock:
            return self._may_be_holding

    @property
    def load_reason(self) -> str:
        """Why the last classify() did not answer EMPTY; empty when it did."""
        with self._lock:
            return self._reason

    def note_command_result(
        self, opening_target: float, mode: str, outcome: GripperOutcome
    ) -> None:
        """Take in how one jaw command given to the arm ended.

        The "may be holding" flag is set when a close ended short of its target
        or without an end reading, and cleared only when a command to full open
        ended at a measured full open. A refused command moved nothing.
        """
        measured = _finite(outcome.opening)
        now = self._clock.monotonic()
        if outcome.status == REFUSED:
            if measured is not None:
                with self._lock:
                    self._remember([measured], now)
            return
        tol = self._config.reach_tol
        full_open = 1.0 - self._config.full_open_tol - _EPS
        holding: bool | None = None  # None: the flag stays as it is
        if opening_target >= full_open:
            if measured is not None and measured >= full_open:
                holding = False
        elif measured is None:
            # Without an end reading, any command but a full open counts as a close.
            holding = True
        elif measured > opening_target + _EPS:
            # The jaws ended more open than asked: a close.
            short = measured > opening_target + tol
            if short or outcome.status in (STALLED, TIMEOUT):
                holding = True
        with self._lock:
            self._remember([] if measured is None else [measured], now)
            changed = holding is not None and holding != self._may_be_holding
            if holding is not None:
                self._may_be_holding = holding
        if changed:
            _log.info(
                "jaws may be holding: %s (%s to %.3f ended %s at %s)",
                "set" if holding else "cleared",
                mode,
                opening_target,
                outcome.status,
                "an unknown opening" if measured is None else f"{measured:.3f}",
            )

    def classify(self, arm: ArmPort) -> JawLoad:
        """Whether fresh readings prove the jaws empty; the jaws are not moved.

        UNKNOWN when the jaws are not referenced or a read fails, EMPTY only
        when every "jaws proven empty" condition of the design's section 11
        holds, LOADED otherwise. Takes config.reads readings read_gap_s apart.
        """
        referenced = bool(arm.gripper_referenced())
        readings: list[float] = []
        if referenced:
            for index in range(self._config.reads):
                if index:
                    self._clock.sleep(self._config.read_gap_s)
                value = _finite(arm.gripper_read())
                if value is None:
                    readings.clear()
                    break
                readings.append(value)
        target = _finite(arm.gripper_last_target())
        now = self._clock.monotonic()
        with self._lock:
            if referenced:
                self._remember(readings, now)
            load, reason = self._judge(referenced, self._span, target)
            if referenced and not readings:
                reason = "a jaw read failed"
            self._reason = reason
        _log.debug("jaws classified %s%s", load, f": {reason}" if reason else "")
        return load

    def gripper_state(self, arm: ArmPort) -> GripperState:
        """The jaw part of ROBOT_STATE from the remembered reading; reads nothing.

        load applies the classify rule to that reading; only classify() takes
        the fresh readings that prove it.
        """
        referenced = bool(arm.gripper_referenced())
        target = _finite(arm.gripper_last_target())
        with self._lock:
            # Without the reference a jaw reading counts as unknown.
            opening, t = (self._opening, self._t) if referenced else (None, None)
            holding = self._may_be_holding
            load, _ = self._judge(referenced, self._span, target)
        return GripperState(
            opening=opening,
            t=t,
            valid=opening is not None,
            referenced=referenced,
            last_target=target,
            may_be_holding=holding,
            load=load.value,
        )

    def _remember(self, readings: list[float], now: float) -> None:
        """Keep the newest readings; none records a failed read. Lock held."""
        if readings:
            self._span = (min(readings), max(readings))
            self._opening, self._t = readings[-1], now
        else:
            self._span = self._opening = self._t = None

    def _judge(
        self,
        referenced: bool,
        span: tuple[float, float] | None,
        target: float | None,
    ) -> tuple[JawLoad, str]:
        """The load the readings in span prove, and why not EMPTY. Lock held."""
        cfg = self._config
        if not referenced:
            return JawLoad.UNKNOWN, "the jaws are not referenced"
        if span is None:
            return JawLoad.UNKNOWN, "there is no valid jaw reading"
        low, high = span
        if self._may_be_holding:
            return JawLoad.LOADED, "the jaws may be holding something"
        if high - low > cfg.agree_tol + _EPS:
            return (
                JawLoad.LOADED,
                f"the jaw readings disagree: {low:.3f} to {high:.3f}",
            )
        if target is None:
            return JawLoad.LOADED, "no jaw target was sent yet"
        worst = max(low, high, key=lambda reading: abs(reading - target))
        if abs(worst - target) > cfg.target_tol + _EPS:
            return (
                JawLoad.LOADED,
                f"the jaws are at {worst:.3f}, not at their target {target:.3f}",
            )
        if target >= 1.0 - _EPS or high <= cfg.closed_frac + _EPS:
            return JawLoad.EMPTY, ""
        return JawLoad.LOADED, f"the jaws are part open at {high:.3f}"


@dataclass
class _Job:
    """One queued request: a single command, or a guarded opening in steps."""

    opening: float
    max_effort: float
    mode: str
    guarded: bool
    on_done: DoneFn | None
    owner_number: int  # the authority's numbers when the request was admitted
    stop_number: int
    # App-clock time at or after which the request is not started; None: none.
    expires_mono_s: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    returned: threading.Event = field(default_factory=threading.Event)
    light_hold_owed: bool = False  # a stop gave up waiting for the arm call
    measured: float | None = None  # the newest opening measured for this request


class GripperWorker:
    """Runs jaw commands one at a time on its own thread.

    on_done gets one GripperOutcome per request: on the caller's thread when it
    is refused at once or dropped by a stop, on the worker thread otherwise. A
    stop cancels, drops and then holds at the light stiffness; it never opens
    the jaws. Every arm jaw command's end goes to the JawState.
    """

    def __init__(
        self,
        arm: ArmPort,
        jaw_state: JawState,
        authority: Authority,
        clock: Clock,
        *,
        on_fault: Callable[[str], None] | None = None,
        stop_wait_s: float = STOP_WAIT_S,
    ) -> None:
        # Told the reason when the driver raises; may be set after construction.
        self.on_fault = on_fault
        self._arm = arm
        self._jaw_state = jaw_state
        self._authority = authority
        self._clock = clock
        self._stop_wait_s = stop_wait_s
        self._cond = threading.Condition()
        self._queue: deque[_Job] = deque()
        self._running: _Job | None = None  # its arm calls are under way
        # Its arm calls have returned; an owed light hold or a fault report runs.
        self._finishing: _Job | None = None
        self._delivering = False  # a finished request's on_done is running
        self._reporting = False  # on_fault is running
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="gripper-worker", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        opening: float,
        max_effort: float,
        mode: str,
        admitted_owner: Owner,
        on_done: DoneFn | None = None,
        expires_mono_s: float | None = None,
    ) -> str | None:
        """Queue one jaw command; returns why it was refused at once, or None.

        Refused when a fault-stop is latched, when the control owner is not
        admitted_owner, when a value is out of range, when MAX_WAITING
        requests wait already, or when expires_mono_s, a time of the app
        clock, has passed. A command whose expiry passes while it waits ends
        "refused" with no arm command sent; None never expires.
        """
        reason = _value_refusal(opening, max_effort, mode)
        reason = reason or self._expired(expires_mono_s)
        return self._enqueue(
            opening,
            max_effort,
            mode,
            False,
            Owner(admitted_owner),
            on_done,
            reason,
            expires_mono_s=expires_mono_s,
        )

    def open_guarded(
        self,
        on_done: DoneFn | None = None,
        admitted_owner: Owner | None = None,
        max_effort: float = OPEN_EFFORT,
    ) -> str | None:
        """Queue an opening to full open in steps of open_step; None if queued.

        Each step is its own arm command and none starts after a stop. Refused
        when a fault-stop is latched, or when admitted_owner is given and is
        not the control owner.
        """
        owner = None if admitted_owner is None else Owner(admitted_owner)
        reason = _fraction_refusal("max_effort", max_effort)
        return self._enqueue(1.0, max_effort, PLAIN, True, owner, on_done, reason)

    def reference(self, on_done: DoneFn | None = None) -> str | None:
        """Queue a jaw reference: the jaws close slowly to their stop; None if queued.

        Refused when a fault-stop is latched, when the control owner is not
        nobody, when the jaws are referenced, or when any jaw request runs or
        waits. A stop ends it "cancelled" with the jaws unreferenced. It ends
        "reached" with the opening measured after it, and leaves the "may be
        holding" flag as it was.
        """
        reason = ALREADY_REFERENCED if self._arm.gripper_referenced() else None
        return self._enqueue(
            0.0, 0.0, REFERENCE, False, Owner.NOBODY, on_done, reason, alone=True
        )

    def follow(
        self,
        opening: float,
        owner_number: int,
        stop_number: int,
        on_done: DoneFn | None = None,
        max_effort: float = OPEN_EFFORT,
    ) -> str | None:
        """Queue one step of the jaws toward `opening`; None if queued.

        For the trigger of a person who owns the arm. The step is at most one
        arm command: one step of open_step toward a wider opening, or a close
        to `opening`, which ends in the light hold when the jaws stall. None
        is sent when the jaws are there, or hold something with a jaw target
        no wider than `opening`. Refused when a fault-stop is latched, when
        the control owner is not a person, when the two numbers are not the
        current ones, or when any jaw request runs or waits.
        """
        reason = _value_refusal(opening, max_effort, PLAIN)
        return self._enqueue(
            opening,
            max_effort,
            FOLLOW,
            False,
            Owner.HUMAN,
            on_done,
            reason,
            alone=True,
            numbers=(owner_number, stop_number),
        )

    def stop_now(self, keep_reference: bool = False) -> bool:
        """Cancel the running command, drop the queued ones and hold lightly.

        A running command gets the light hold at once, so a close does not go on
        at the closing stiffness, and again once it has returned, so the light
        hold is the last thing the jaws are sent. Waits up to stop_wait_s of real
        time for that return; True when the cancelled command has not returned
        by then, so the jaws stay busy.

        keep_reference: a jaw reference, which is typed at the Thor terminal
        only, goes on, and while one runs the jaws are sent no light hold.
        """
        with self._cond:
            kept: list[_Job] = []
            dropped: list[_Job] = []
            for queued in self._queue:
                keep = keep_reference and queued.mode == REFERENCE
                (kept if keep else dropped).append(queued)
            self._queue = deque(kept)
            job = self._running
            spared = keep_reference and job is not None and job.mode == REFERENCE
            if spared:
                job = None
            if job is not None:
                job.cancel.set()
            self._cond.notify_all()
        if spared:
            for queued in dropped:
                _call(queued.on_done, GripperOutcome(CANCELLED, None, DROPPED_BY_STOP))
            _log.info(
                "jaw stop: the reference goes on, %d queued dropped", len(dropped)
            )
            return False
        held = True
        if job is not None:
            held = self._light_hold()
        for queued in dropped:
            _call(queued.on_done, GripperOutcome(CANCELLED, None, DROPPED_BY_STOP))
        late = False
        if job is not None:
            on_worker = threading.current_thread() is self._thread
            if on_worker or not job.returned.wait(self._stop_wait_s):
                with self._cond:
                    late = self._running is job
                    job.light_hold_owed = late
        if not late:
            # A light hold that already failed in this stop is not reported twice.
            self._light_hold(report=held)
        if job is not None or dropped:
            _log.info(
                "jaw stop: %s, %d queued dropped, light hold sent",
                "nothing was running" if job is None else "running command cancelled",
                len(dropped),
            )
        if late:
            _log.warning(
                "the cancelled jaw command has not returned; "
                "the light hold is sent again when it does"
            )
        return late

    def busy(self) -> bool:
        """True while a request is queued or its work on the jaws is not over.

        False once a request's on_done is called, so the command that follows
        its outcome is not refused for jaws that are busy.
        """
        with self._cond:
            return self._busy()

    def referencing(self) -> bool:
        """True while a jaw reference waits, runs or is being finished."""
        with self._cond:
            return any(job.mode == REFERENCE for job in self._unfinished())

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait in real time until no request is busy and no on_done runs."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not (self._busy() or self._delivering), timeout_s
            )

    def close(self) -> None:
        """End the worker thread; a request still queued or running is stopped."""
        with self._cond:
            self._closed = True
            pending = self._running is not None or bool(self._queue)
            self._cond.notify_all()
        if pending:
            self.stop_now()
        if threading.current_thread() is not self._thread:
            self._thread.join(_CLOSE_WAIT_S)
            if self._thread.is_alive():
                _log.warning("the gripper worker thread has not ended")

    def _busy(self) -> bool:
        return bool(self._unfinished())

    def _unfinished(self) -> list[_Job]:
        """The requests that wait, run or are being finished."""
        ahead = [job for job in (self._running, self._finishing) if job is not None]
        return [*ahead, *self._queue]

    def _enqueue(
        self,
        opening: float,
        max_effort: float,
        mode: str,
        guarded: bool,
        owner: Owner | None,
        on_done: DoneFn | None,
        reason: str | None,
        alone: bool = False,
        numbers: tuple[int, int] | None = None,
        expires_mono_s: float | None = None,
    ) -> str | None:
        """Queue a request unless `reason` or the authority refuses it.

        alone: it is taken only while no other jaw request runs or waits.
        numbers: the owner number and the stop number the request quotes; it
        is taken only while they are the current ones.
        expires_mono_s: the request is not started at or after this time.
        """
        state = self._authority.snapshot()
        reason = _authority_refusal(state, owner) or reason
        if reason is None and numbers is not None:
            reason = _numbers_refusal(state, numbers)
        if reason is None:
            job = _Job(
                float(opening),
                float(max_effort),
                mode,
                guarded,
                on_done,
                state.owner_number,
                state.stop_number,
                expires_mono_s,
            )
            with self._cond:
                if self._closed:
                    reason = "the gripper worker is closed"
                elif alone and self._busy():
                    reason = JAWS_BUSY
                elif len(self._queue) >= MAX_WAITING:
                    reason = f"busy: {len(self._queue)} jaw commands are waiting"
                else:
                    self._queue.append(job)
                    self._cond.notify_all()
        if reason is not None:
            _log.info("jaw command refused: %s", reason)
            _call(on_done, GripperOutcome(REFUSED, None, reason))
        return reason

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                job = self._queue.popleft()
                self._running = job
            try:
                outcome, fault = self._execute(job)
            except Exception as exc:  # the worker thread outlives any error
                _log.exception("jaw command error")
                fault = f"jaw command error: {exc!r}"
                outcome = GripperOutcome(FAILED, job.measured, fault)
            with self._cond:
                self._running = None
                self._finishing = job
                owed = job.light_hold_owed
            job.returned.set()
            if owed:
                # The arm call may have re-sent the closing stiffness since the stop.
                self._light_hold()
            if fault is not None:
                self._report_fault(fault)
            with self._cond:
                # No longer busy when the outcome goes out, so the command that
                # follows it is not refused; wait_idle() still waits for on_done.
                self._finishing = None
                self._delivering = True
                self._cond.notify_all()
            _call(job.on_done, outcome)
            with self._cond:
                self._delivering = False
                self._cond.notify_all()

    def _execute(self, job: _Job) -> tuple[GripperOutcome, str | None]:
        """Run a request; returns its outcome and the fault reason, if any."""
        expired = self._expired(job.expires_mono_s)
        if expired is not None:
            _log.info("jaw command refused: %s", expired)
            return GripperOutcome(REFUSED, None, expired), None
        if job.mode == REFERENCE:
            return self._reference(job)
        if job.mode == FOLLOW:
            return self._follow(job)
        if not job.guarded:
            return self._command(job, job.opening, job.mode)
        if not self._arm.gripper_referenced():
            return GripperOutcome(REFUSED, None, "the jaws are not referenced"), None
        start = _finite(self._arm.gripper_read())
        if start is None:
            # A step from an unknown opening could close the jaws on an object.
            detail = "the jaw opening could not be read, so no opening step was sent"
            return GripperOutcome(REFUSED, None, detail), None
        job.measured = start
        step = self._jaw_state.config.open_step
        steps = max(1, math.ceil((1.0 - start) / step - _EPS))
        index = 0
        while True:
            index += 1
            target = 1.0 if index == steps else start + index * step
            outcome, fault = self._command(job, target, PLAIN)
            if index == steps or fault is not None or outcome.status != REACHED:
                return outcome, fault

    def _reference(self, job: _Job) -> tuple[GripperOutcome, str | None]:
        """Reference the jaws, unless the request was stopped first.

        A reference the driver fails is a fault; one a stop ended is not.
        """
        detail = self._ended_early(job)
        if detail is not None:
            return GripperOutcome(CANCELLED, None, detail), None
        if self._arm.gripper_referenced():
            return GripperOutcome(REFUSED, None, ALREADY_REFERENCED), None
        try:
            self._arm.gripper_reference(job.cancel)
        except ArmFault as exc:
            fault = f"the jaw reference failed: {exc}"
            return GripperOutcome(FAILED, None, fault), fault
        if not self._arm.gripper_referenced():
            if job.cancel.is_set():
                detail = "stopped before the jaws were referenced"
                return GripperOutcome(CANCELLED, None, detail), None
            detail = "the jaws are not referenced after the reference"
            return GripperOutcome(FAILED, None, detail), None
        # One classification, so ROBOT_STATE has a jaw reading again.
        self._jaw_state.classify(self._arm)
        job.measured = self._jaw_state.gripper_state(self._arm).opening
        _log.info("jaws referenced; opening %s", job.measured)
        return GripperOutcome(REACHED, job.measured), None

    def _follow(self, job: _Job) -> tuple[GripperOutcome, str | None]:
        """One step toward the opening a trigger asks for: at most one arm command.

        It ends "reached" with nothing sent when the jaws are within reach_tol
        of the opening, or hold something with a jaw target no wider than it:
        that light hold stands, so easing the trigger squeezes nothing anew.
        """
        detail = self._ended_early(job)
        if detail is not None:
            return GripperOutcome(CANCELLED, None, detail), None
        if not self._arm.gripper_referenced():
            return GripperOutcome(REFUSED, None, "the jaws are not referenced"), None
        start = _finite(self._arm.gripper_read())
        if start is None:
            # A step from an unknown opening could drop or squeeze an object.
            detail = "the jaw opening could not be read, so no jaw command was sent"
            return GripperOutcome(REFUSED, None, detail), None
        job.measured = start
        config = self._jaw_state.config
        wanted, tol = job.opening, config.reach_tol
        to_full_open = wanted >= 1.0 - _EPS and start < 1.0 - _EPS
        if wanted > start + tol or to_full_open:
            step = min(wanted, start + config.open_step)
            return self._command(job, step, PLAIN)
        if wanted < start - tol:
            target = _finite(self._arm.gripper_last_target())
            held = self._jaw_state.may_be_holding and target is not None
            if not (held and wanted >= target - tol):
                return self._command(job, wanted, CLOSE_UNTIL_CONTACT)
        return GripperOutcome(REACHED, start), None

    def _command(
        self, job: _Job, opening: float, mode: str
    ) -> tuple[GripperOutcome, str | None]:
        """Give the arm one jaw command, unless the request was stopped first."""
        detail = self._ended_early(job)
        if detail is not None:
            return GripperOutcome(CANCELLED, job.measured, detail), None
        fault: str | None = None
        t_start = self._clock.monotonic()
        try:
            outcome = self._arm.gripper_command(
                opening, job.max_effort, mode, job.cancel
            )
        except ArmFault as exc:
            fault = f"gripper fault: {exc}"
        except Exception as exc:  # a driver error must not end the worker thread
            _log.exception("the jaw command raised")
            fault = f"jaw command error: {exc!r}"
        if fault is not None:
            outcome = GripperOutcome(FAILED, None, fault)
        self._jaw_state.note_command_result(opening, mode, outcome)
        if outcome.opening is not None:
            job.measured = outcome.opening
        _log.debug(
            "jaw command %s to %.3f: %s at %s after %.2f s",
            mode,
            opening,
            outcome.status,
            outcome.opening,
            self._clock.monotonic() - t_start,
        )
        return outcome, fault

    def _expired(self, expires_mono_s: float | None) -> str | None:
        """Why a request with this expiry is not started now, or None."""
        if expires_mono_s is None:
            return None
        late = self._clock.monotonic() - expires_mono_s
        return None if late < 0.0 else f"the jaw command expired {late:.3f} s ago"

    def _ended_early(self, job: _Job) -> str | None:
        """Why the request's next arm command must not be sent, or None."""
        state = self._authority.snapshot()
        if state.stop_level is StopLevel.FAULT:
            return _fault_text(state)
        if state.stop_number != job.stop_number:
            return "a STOP was raised after the command was admitted"
        if state.owner_number != job.owner_number:
            return "the control owner changed after the command was admitted"
        # Checked last, so a stop that came during the checks above is seen.
        if job.cancel.is_set():
            return "stopped before the command was sent"
        return None

    def _light_hold(self, report: bool = True) -> bool:
        """Send the light hold; False when the driver raised.

        The failure goes to on_fault, or only to the log when report is False.
        """
        try:
            self._arm.gripper_light_hold()
        except ArmFault as exc:
            reason = f"gripper fault: the light hold failed: {exc}"
        except Exception as exc:  # a stop must run to its end
            _log.exception("the light jaw hold raised")
            reason = f"jaw command error: the light hold failed: {exc!r}"
        else:
            return True
        if report:
            self._report_fault(reason)
        else:
            _log.error("%s", reason)
        return False

    def _report_fault(self, reason: str) -> None:
        """Tell on_fault; a fault raised while it runs is only logged."""
        handler = self.on_fault
        with self._cond:
            skip = handler is None or self._reporting
            if not skip:
                self._reporting = True
        if skip:
            _log.error("%s", reason)
            return
        try:
            handler(reason)
        except Exception:
            _log.exception("the gripper fault handler raised")
        finally:
            with self._cond:
                self._reporting = False


class JawFollower:
    """Lets the jaws follow the trigger of a person who owns the arm.

    The grip level is the jaw opening in force when the person took the arm
    over. The trigger is ignored until the opening it asks for is within
    match_tol of the grip level or has crossed it: the trigger has matched.
    From then on the jaws follow it, one jaw step at a time and toward the
    newest opening only. Until the jaws were measured fully open no opening
    below the grip level is asked of them. Thread-safe.
    """

    def __init__(
        self,
        arm: ArmPort,
        jaw_state: JawState,
        worker: GripperWorker,
        authority: Authority,
        match_tol: float,
        say: Callable[[str], None] | None = None,
    ) -> None:
        """say(line) logs when the jaws begin and end to follow the trigger."""
        self._arm = arm
        self._jaw_state = jaw_state
        self._worker = worker
        self._authority = authority
        self._match_tol = float(match_tol)
        self._say = say if say is not None else _log.info
        config = jaw_state.config
        # A new opening is asked of the jaws only when it differs by more.
        self._band = config.reach_tol
        self._full_open = 1.0 - config.full_open_tol - _EPS
        # Reentrant: a step that is refused at once ends on the caller's thread.
        self._lock = threading.RLock()
        # The owner number and the stop number the grip level was noted for.
        self._numbers: tuple[int, int] | None = None
        self._level: float | None = None  # the grip level; None: not known
        self._matched = False
        self._opened = False  # measured fully open since the takeover
        self._asked: float | None = None  # the opening the trigger asked for last
        self._wanted: float | None = None  # the opening the jaws are to follow
        # The opening the jaws have settled for; None while an opening goes on.
        self._settled: float | None = None
        self._flying: float | None = None  # the opening of the step under way
        self._measured: float | None = None  # the end reading of the last step
        self._step_end: float | None = None  # where an opening's last step ended
        self._refusal = ""  # why the trigger moves nothing, or holds the jaws wider

    @property
    def following(self) -> bool:
        """Whether the trigger has matched, so the jaws follow it."""
        with self._lock:
            return self._matched

    def note(self, owner_number: int, stop_number: int) -> None:
        """Note the grip level of a takeover with these numbers; sends nothing."""
        with self._lock:
            self._opened = False
            self._note((owner_number, stop_number))

    def offer(
        self,
        opening: object,
        owner_number: int,
        stop_number: int,
        no_close: str | None = None,
    ) -> None:
        """Take the opening the trigger of one admitted human target asks for.

        owner_number and stop_number are those the target quotes, which are
        the current ones with a person the control owner. no_close says why
        the jaws may only open now; None: they may close too. An opening of
        None leaves the jaws as they are, and one that is not a number from 0
        to 1 ends their motion; after either the trigger must match again.
        """
        numbers = (owner_number, stop_number)
        with self._lock:
            if self._flying is not None and not self._matched:
                self._refusal = STEP_OUT
                return
            bad = None if opening is None else _fraction_refusal("jaw_opening", opening)
            cut = bad is not None and self._matched
            if bad is not None:
                self._drop(f"refused: {bad}")
            elif opening is not None:
                self._take(float(opening), numbers, no_close)
            elif self._matched:
                self._drop(NO_OPENING)
            else:
                if numbers != self._numbers:
                    self._note(numbers)
                self._asked = None
                if self._level is not None:
                    self._refusal = NO_OPENING
        if cut:
            self._cut(numbers)

    def end(self, reason: str, hold: bool = True) -> None:
        """End the jaw motion as a STOP ends one; the trigger must match again.

        Jaws that followed the trigger are held lightly where they are, never
        opened; a STOP or an owner change has done that itself, and so has
        the caller that passes hold False. Jaws that did not follow the
        trigger are sent nothing.
        """
        with self._lock:
            cut, numbers = self._matched, self._numbers
            if cut:
                self._say(f"the jaws no longer follow the trigger: {reason}")
            self._drop(reason)
        if cut and hold and numbers is not None:
            self._cut(numbers)

    def facts(self) -> dict[str, Any]:
        """For ROBOT_STATE: whether the trigger has matched, and why not."""
        with self._lock:
            return {
                "matched": self._matched,
                "grip_level": self._level,
                "opened": self._opened,
                "asked": self._asked,
                "refusal": self._refusal,
            }

    def _note(self, numbers: tuple[int, int]) -> None:
        """Note the grip level from what is known of the jaws; the lock is held.

        The last jaw target; fully open when the jaws are proven empty and
        open; without a target the newest reading. Nothing is read or sent.
        """
        state = self._jaw_state.gripper_state(self._arm)
        reading = _finite(state.opening)
        is_open = reading is not None and reading >= self._full_open
        level: float | None = None
        why = ""
        if not state.referenced:
            why = "the jaws are not referenced"
        elif is_open and state.load == JawLoad.EMPTY.value:
            level = 1.0
        elif state.last_target is not None:
            level = min(1.0, max(0.0, state.last_target))
        elif reading is not None:
            level = min(1.0, max(0.0, reading))
        else:
            why = "the jaws have no target and no reading"
        self._drop(NOT_MATCHED if level is not None else f"{NO_GRIP_LEVEL}{why}")
        self._numbers, self._level = numbers, level
        self._opened = self._opened or is_open
        _log.debug("grip level %s for owner number %d (%s)", level, numbers[0], why)

    def _take(
        self, asked: float, numbers: tuple[int, int], no_close: str | None
    ) -> None:
        """Follow one opening the trigger asks for, once it has matched; lock held."""
        if numbers != self._numbers:
            self._note(numbers)
        before, self._asked = self._asked, asked
        level = self._level
        if level is None:
            return  # _note() said why
        if not self._matched:
            near = abs(asked - level) <= self._match_tol + _EPS
            crossed = before is not None and (before - level) * (asked - level) < 0.0
            if not (near or crossed):
                self._refusal = NOT_MATCHED
                return
            self._matched = True
            self._settled = level  # the jaws are there, or held short of it
            self._say(f"the jaws follow the trigger from the grip level {level:.2f}")
        wanted = 1.0 if asked >= self._full_open else asked
        refusal = ""
        if not self._opened and wanted < level:
            wanted, refusal = level, NO_TIGHTER
        # Where the jaws are, or are going: without the leave to close, no further.
        floor = self._settled
        if floor is None:
            floor = self._measured if self._flying is None else self._flying
        if no_close is not None and floor is not None and wanted < floor:
            wanted, refusal = floor, f"{NO_CLOSING}{no_close}"
        self._wanted, self._refusal = wanted, refusal
        self._pump()

    def _pump(self) -> None:
        """Ask the next jaw step, unless one is under way; the lock is held."""
        wanted, numbers = self._wanted, self._numbers
        if self._flying is not None or not self._matched:
            return
        if wanted is None or numbers is None:
            return
        settled = self._settled
        if settled is not None:
            to_full_open = wanted >= 1.0 - _EPS and settled < 1.0 - _EPS
            if abs(wanted - settled) <= self._band and not to_full_open:
                return
        self._flying = wanted
        # A refusal has called _done() before this returns.
        self._worker.follow(wanted, numbers[0], numbers[1], self._done)

    def _done(self, outcome: GripperOutcome) -> None:
        """The end of one jaw step: on the gripper worker's thread, or a caller's."""
        with self._lock:
            wanted, self._flying = self._flying, None
            measured = _finite(outcome.opening)
            if measured is not None:
                self._measured = measured
                if measured >= self._full_open:
                    self._opened = True
            if wanted is None or not self._matched:
                return  # ended meanwhile: the grip level is noted anew
            if outcome.status not in (REACHED, STALLED, TIMEOUT):
                why = outcome.detail or "no reason given"
                reason = f"a jaw step ended {outcome.status}: {why}"
                self._say(f"the jaws no longer follow the trigger: {reason}")
                self._drop(reason)
                return
            # An opening goes one step at a time: more of it may be left.
            short = measured is not None and measured < wanted - self._band
            further = self._step_end is None or (
                measured is not None and measured > self._step_end + _EPS
            )
            if outcome.status == REACHED and short and further:
                self._step_end, self._settled = measured, None
            else:
                self._step_end, self._settled = None, wanted
            self._pump()

    def _drop(self, reason: str) -> None:
        """Forget the match and the grip level; the lock is held."""
        self._matched = False
        self._numbers = self._level = None
        self._asked = self._wanted = self._settled = self._step_end = None
        self._refusal = reason

    def _cut(self, numbers: tuple[int, int]) -> None:
        """Hold the jaws lightly where they are, while these numbers stand."""
        state = self._authority.snapshot()
        current = (state.owner_number, state.stop_number)
        if state.owner is Owner.HUMAN and current == numbers:
            self._worker.stop_now()


def _call(on_done: DoneFn | None, outcome: GripperOutcome) -> None:
    if on_done is None:
        return
    try:
        on_done(outcome)
    except Exception:
        _log.exception("a jaw command's on_done raised")


def _fraction_refusal(name: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return f"{name} must be a number, got {value!r}"
    if not 0.0 <= value <= 1.0:  # false for NaN
        return f"{name} must be between 0 and 1, got {value!r}"
    return None


def _value_refusal(opening: object, max_effort: object, mode: object) -> str | None:
    reason = _fraction_refusal("opening", opening)
    reason = reason or _fraction_refusal("max_effort", max_effort)
    if reason is None and mode not in JAW_MODES:
        reason = f"unknown mode {mode!r}"
    return reason


def _fault_text(state: AuthorityState) -> str:
    if state.fault_reason:
        return f"fault-stop latched: {state.fault_reason}"
    return "fault-stop latched"


def _authority_refusal(state: AuthorityState, owner: Owner | None) -> str | None:
    if state.stop_level is StopLevel.FAULT:
        return _fault_text(state)
    if owner is not None and state.owner != owner:
        return f"control owner is {state.owner}, not {owner}"
    return None


def _numbers_refusal(state: AuthorityState, numbers: tuple[int, int]) -> str | None:
    """Why a request that quotes (owner number, stop number) is not taken."""
    current = (state.owner_number, state.stop_number)
    if numbers == current:
        return None
    return (
        f"the request quotes owner number {numbers[0]} and stop number "
        f"{numbers[1]}, the current ones are {current[0]} and {current[1]}"
    )
