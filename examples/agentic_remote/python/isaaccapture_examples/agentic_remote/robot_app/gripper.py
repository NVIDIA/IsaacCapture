# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The jaws: what they may hold, and the one thread that runs their commands.

See sections 9 and 11 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING

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
    ) -> str | None:
        """Queue one jaw command; returns why it was refused at once, or None.

        Refused when a fault-stop is latched, when the control owner is not
        admitted_owner, when a value is out of range, or when MAX_WAITING
        requests wait already.
        """
        reason = _value_refusal(opening, max_effort, mode)
        return self._enqueue(
            opening, max_effort, mode, False, Owner(admitted_owner), on_done, reason
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
    ) -> str | None:
        """Queue a request unless `reason` or the authority refuses it.

        alone: it is taken only while no other jaw request runs or waits.
        """
        state = self._authority.snapshot()
        reason = _authority_refusal(state, owner) or reason
        if reason is None:
            job = _Job(
                float(opening),
                float(max_effort),
                mode,
                guarded,
                on_done,
                state.owner_number,
                state.stop_number,
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
        if job.mode == REFERENCE:
            return self._reference(job)
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
