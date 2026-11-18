# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The jaws: what they may hold, and the one thread that runs their commands.

Each jaw command is one routine of the arm's motor driver, named by its mode;
when it ends and how is the driver's verdict. Also how the jaws follow the
trigger of a person who owns the arm. See sections 9, 11 and 12 of
docs/design/claude_agentic_remote_execution.md.
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

from ..protocol.messages import (
    GRIPPER_MODES,
    JAW_CLOSE_TORQUE,
    JAW_CLOSE_UNTIL_CONTACT,
    JAW_OPEN,
    JAW_PLAIN,
    GripperState,
)
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

# The jaw modes a GRIPPER may name (GRIPPER_MODES of protocol/messages.py).
PLAIN = JAW_PLAIN
CLOSE_UNTIL_CONTACT = JAW_CLOSE_UNTIL_CONTACT
OPEN = JAW_OPEN
CLOSE_TORQUE = JAW_CLOSE_TORQUE
JAW_MODES = GRIPPER_MODES
# The modes every arm port takes. A port that runs the others with routines of
# its motor driver names them all in its attribute jaw_modes; for a port that
# does not, the worker gives OPEN as a plain command to full open and the
# closes as one close until contact.
BASIC_MODES = (PLAIN, CLOSE_UNTIL_CONTACT)
# What the typed close carries: the close the driver's side knows as its own.
# No GRIPPER can ask for it.
CLOSE = "close"
# What a queued jaw reference carries in place of a jaw mode; no link message
# can ask for it.
REFERENCE = "reference"
# What a queued step of the jaws toward a person's trigger carries in place of
# a jaw mode; no link message can ask for it.
FOLLOW = "follow"
ALREADY_REFERENCED = "the jaws are referenced already"
NOT_REFERENCED = "the jaws are not referenced"
JAWS_BUSY = "busy: a jaw command is running or waiting"
DROPPED_BY_STOP = "dropped by a stop"

# GripperOutcome statuses. REACHED: the driver's routine ended at its target,
# or its opening ended where the driver takes it as done. SENT: the routine
# returned once the command was sent, as the driver's close and hold do.
# STALLED: a close ended short of its target. GRIPPER_FAULT: the driver ended
# or refused the command with a fault of the jaws that leaves the arm usable;
# its text is the outcome's detail. FAILED is the worker's own: the driver
# raised anything else, and a fault-stop follows.
REACHED = "reached"
SENT = "sent"
STALLED = "stalled"
TIMEOUT = "timeout"
CANCELLED = "cancelled"
REFUSED = "refused"
GRIPPER_FAULT = "gripper_fault"
FAILED = "failed"

# Effort of a typed opening or close on a driver without a routine of its own
# for it, and of each step of the jaws toward a person's trigger.
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
# Why the jaws stay narrower than the trigger asks; the opening they are at
# follows, as a share of the jaw travel.
OPENING_STALLED = (
    "the jaws stalled while opening: check what is against them; they are at "
)

DoneFn = Callable[[GripperOutcome], None]


@dataclass(frozen=True)
class DriverOutcome(GripperOutcome):
    """A jaw command's end with what the driver's own routine reported."""

    # How many stages the driver's two-stage close skipped; None: not said.
    stages_skipped: int | None = None
    # What the driver's close to the closed stop returned; None: not that.
    closed: bool | None = None


def _finite(value: float | None) -> float | None:
    """value as a float; None when it is missing or not a finite number."""
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class JawState:
    """What measurements say about the jaws' load; it never moves the jaws.

    Thread-safe. It remembers the newest jaw reading: the end reading of the
    last command, the readings of the last classify() call, or the one of
    the last refresh() call. The operator's word is what the operator typed
    in place of the proof: that the jaws are empty.
    """

    def __init__(self, config: GripperConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._may_be_holding = False
        # The operator's word stands until the jaws are next sent a command.
        self._word = False
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

    @property
    def word_given(self) -> bool:
        """Whether the operator's word that the jaws are empty stands."""
        with self._lock:
            return self._word

    def take_word(self) -> None:
        """The operator typed that the jaws are empty: taken until the next command.

        classify() answers EMPTY while it stands, whatever the jaws read, and
        "may be holding" is cleared.
        """
        with self._lock:
            self._word = True
            self._may_be_holding = False
        _log.info("jaws: the operator typed that they are empty")

    def clear_word(self) -> None:
        """Forget the operator's word: the jaws may hold something again."""
        with self._lock:
            self._word = False

    def note_command_result(
        self, opening_target: float, mode: str, outcome: GripperOutcome
    ) -> None:
        """Take in how one jaw command given to the arm ended.

        The "may be holding" flag is set when a close stalled short of its
        target above the closed line (closed_frac), when a stop cut a close
        short, or when a close ended without an end reading. It is cleared
        only when a command to full open ended at a measured full open. A
        command the driver only sent (SENT) leaves it as it is: what the jaws
        read afterwards says where they went. A refused command moved nothing.
        """
        measured = _finite(outcome.opening)
        now = self._clock.monotonic()
        status = outcome.status
        if status == REFUSED:
            if measured is not None:
                with self._lock:
                    self._remember([measured], now)
            return
        cfg = self._config
        full_open = 1.0 - cfg.full_open_tol - _EPS
        holding: bool | None = None  # None: the flag stays as it is
        if opening_target >= full_open:
            if measured is not None and measured >= full_open:
                holding = False
        elif status in (SENT, GRIPPER_FAULT):
            pass
        elif measured is None:
            # Without an end reading, any command but a full open counts as a close.
            holding = True
        elif measured > max(opening_target, cfg.closed_frac) + _EPS:
            # The jaws ended more open than asked, and not closed: a close.
            cut = status == CANCELLED and measured > opening_target + cfg.target_tol
            if cut or status in (STALLED, TIMEOUT):
                holding = True
        with self._lock:
            self._word = False  # given for the jaws as they were
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
        holds, LOADED otherwise. Jaws told to close that read at or below
        closed_frac count as at their target. Takes config.reads readings
        read_gap_s apart. While the operator's word stands the answer is EMPTY.
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
            measured, word = (load, reason), self._word
            if word:
                load, reason = JawLoad.EMPTY, ""
            self._reason = reason
        if word and measured[0] is not JawLoad.EMPTY:
            _log.info(
                "jaws taken as empty on the operator's word; measured %s: %s",
                *measured,
            )
        _log.debug("jaws classified %s%s", load, f": {reason}" if reason else "")
        return load

    def refresh(self, arm: ArmPort) -> None:
        """Take one jaw reading for ROBOT_STATE; the jaws are not moved.

        The agent decides from it what the jaws hold. Nothing is read while
        the jaws are not referenced. The reading is the arm port's
        gripper_peek() where it has one: a read that cannot wait long and
        cannot cost the jaws their reference. A failed one is unknown.
        """
        if not arm.gripper_referenced():
            return
        value = _finite(getattr(arm, "gripper_peek", arm.gripper_read)())
        now = self._clock.monotonic()
        with self._lock:
            self._remember([] if value is None else [value], now)

    def not_proven(self, arm: ArmPort) -> str | None:
        """Why the remembered reading does not show the jaws empty; reads nothing.

        None when it does, or the operator's word stands. What the terminal
        asks the operator about before a rest, a release or a typed close.
        """
        state = self.gripper_state(arm)
        if state.load == JawLoad.EMPTY.value:
            return None
        referenced = bool(arm.gripper_referenced())
        target = _finite(arm.gripper_last_target())
        with self._lock:
            _, reason = self._judge(referenced, self._span, target)
        return f"{state.load}: {reason}" if reason else state.load

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
            if self._word:
                load = JawLoad.EMPTY
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
        closed = cfg.closed_frac + _EPS
        if target <= closed and high <= closed:
            # Told to close and at or below the closed line: closed on nothing,
            # however far short of the target the driver's close ended.
            return JawLoad.EMPTY, ""
        worst = max(low, high, key=lambda reading: abs(reading - target))
        tol = cfg.target_tol
        if target >= 1.0 - _EPS:
            # A full open ends where the driver's guarded opening lets it end.
            tol = max(tol, cfg.full_open_tol)
        if abs(worst - target) > tol + _EPS:
            return (
                JawLoad.LOADED,
                f"the jaws are at {worst:.3f}, not at their target {target:.3f}",
            )
        if target >= 1.0 - _EPS or high <= closed:
            return JawLoad.EMPTY, ""
        return JawLoad.LOADED, f"the jaws are part open at {high:.3f}"


@dataclass
class _Job:
    """One queued request: one routine of the driver, or one of the worker's own."""

    opening: float
    max_effort: float
    mode: str
    on_done: DoneFn | None
    owner_number: int  # the authority's numbers when the request was admitted
    stop_number: int
    # App-clock time at or after which the request is not started; None: none.
    expires_mono_s: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    returned: threading.Event = field(default_factory=threading.Event)
    light_hold_owed: bool = False  # a stop gave up waiting for the arm call
    measured: float | None = None  # the newest opening measured for this request
    may_close: bool = True  # False: a FOLLOW step sends no close
    closing: bool = False  # a FOLLOW step's close is with the arm
    cut: bool = False  # cut_follow() ended that close
    # The scout stage's opening of a two-stage close; None: one close.
    scout: float | None = None
    own_effort: bool = False  # a CLOSE_TORQUE at the driver's own effort


class GripperWorker:
    """Runs jaw commands one at a time on its own thread.

    on_done gets one GripperOutcome per request: on the caller's thread when it
    is refused at once or dropped by a stop, on the worker thread otherwise. A
    stop cancels a running command, which then ends at the light stiffness,
    and drops the queued ones; it never opens the jaws, and jaws no command is
    moving stay as they are. Every arm jaw command's end goes to the JawState.
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
        # cut_follow() ended the last close, and the jaws were sent nothing
        # but the light hold at that close's target since.
        self._after_cut = False
        # The last jaw command given to the arm was a step toward a trigger.
        self._by_trigger = False
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
        scout_opening: float | None = None,
    ) -> str | None:
        """Queue one jaw command; returns why it was refused at once, or None.

        scout_opening makes a close until contact the driver's two-stage
        close: the scout stage to it, then the seat stage to `opening`.
        Refused when a fault-stop is latched, when the control owner is not
        admitted_owner, when a value is out of range, when MAX_WAITING
        requests wait already, or when expires_mono_s, a time of the app
        clock, has passed. A command whose expiry passes while it waits ends
        "refused" with no arm command sent; None never expires.
        """
        reason = _value_refusal(opening, max_effort, mode)
        reason = reason or _scout_refusal(scout_opening, mode)
        reason = reason or self._expired(expires_mono_s)
        return self._enqueue(
            opening,
            max_effort,
            mode,
            Owner(admitted_owner),
            on_done,
            reason,
            expires_mono_s=expires_mono_s,
            scout=scout_opening,
        )

    def open_guarded(
        self,
        on_done: DoneFn | None = None,
        admitted_owner: Owner | None = None,
        max_effort: float = OPEN_EFFORT,
    ) -> str | None:
        """Queue the typed opening to full open; None if queued.

        It is the driver's own opening: one ramp that watches the jaws and
        ends at a stop. max_effort counts only on a driver without one.
        Refused when a fault-stop is latched, or when admitted_owner is given
        and is not the control owner.
        """
        owner = None if admitted_owner is None else Owner(admitted_owner)
        reason = _fraction_refusal("max_effort", max_effort)
        return self._enqueue(1.0, max_effort, OPEN, owner, on_done, reason)

    def close_jaws(
        self,
        on_done: DoneFn | None = None,
        admitted_owner: Owner | None = None,
        max_effort: float = OPEN_EFFORT,
    ) -> str | None:
        """Queue the typed close; None if queued.

        It is the close the driver's side has for a typed "close the
        gripper"; max_effort counts only on a driver without one. Refused as
        open_guarded() is.
        """
        owner = None if admitted_owner is None else Owner(admitted_owner)
        reason = _fraction_refusal("max_effort", max_effort)
        return self._enqueue(0.0, max_effort, CLOSE, owner, on_done, reason)

    def close_at_rest(
        self,
        on_done: DoneFn | None = None,
        admitted_owner: Owner | None = None,
        max_effort: float = OPEN_EFFORT,
    ) -> str | None:
        """Queue the close that comes before a torque release; None if queued.

        It is the driver's close to the closed stop at the driver's own
        effort, which leaves the jaws in the light hold where they met the
        stop or what they hold; max_effort counts only on a driver without
        one. Refused as open_guarded() is.
        """
        owner = None if admitted_owner is None else Owner(admitted_owner)
        reason = _fraction_refusal("max_effort", max_effort)
        return self._enqueue(
            0.0, max_effort, CLOSE_TORQUE, owner, on_done, reason, own_effort=True
        )

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
            0.0, 0.0, REFERENCE, Owner.NOBODY, on_done, reason, alone=True
        )

    def follow(
        self,
        opening: float,
        owner_number: int,
        stop_number: int,
        on_done: DoneFn | None = None,
        max_effort: float = OPEN_EFFORT,
        may_close: bool = True,
    ) -> str | None:
        """Queue one step of the jaws toward `opening`; None if queued.

        For the trigger of a person who owns the arm. The step is at most one
        arm command: one step of open_step toward a wider opening, or a close
        to `opening`; the light hold follows either when the jaws stall. None
        is sent when the jaws are there, or hold something with a jaw target
        no wider than `opening`, or would close with may_close False. The
        first step after cut_follow() ended a close always sends the jaws its
        opening. Refused when a fault-stop is latched, when the control owner
        is not a person, when the two numbers are not the current ones, or
        when any jaw request runs or waits.
        """
        reason = _value_refusal(opening, max_effort, PLAIN)
        return self._enqueue(
            opening,
            max_effort,
            FOLLOW,
            Owner.HUMAN,
            on_done,
            reason,
            alone=True,
            numbers=(owner_number, stop_number),
            may_close=may_close,
        )

    def cut_follow(self, opening: float) -> bool:
        """End a close of follow() that is under way; True when one was ended.

        For a trigger that now asks for `opening`: only a close to an opening
        narrower than it by more than reach_tol is ended. It ends "cancelled"
        and the jaws get the light hold, as at a STOP; the next step of
        follow() then sends them its opening.
        """
        tol = self._jaw_state.config.reach_tol
        with self._cond:
            job = self._running
            if job is None or not job.closing or job.cut:
                return False
            if job.opening >= opening - tol:
                return False
            job.cut = True
            job.cancel.set()
        _log.debug("the close to %.3f is cut: %.3f is asked", job.opening, opening)
        return True

    def stop_now(self, keep_reference: bool = False, hold_idle: bool = False) -> bool:
        """Cancel the running command and drop the queued ones.

        A running command gets the light hold at once, so a close does not go on
        at the closing stiffness, and again once it has returned, so the light
        hold is the last thing the jaws are sent. Waits up to stop_wait_s of real
        time for that return; True when the cancelled command has not returned
        by then, so the jaws stay busy. With no command running the jaws are
        sent nothing: they keep their target and stiffness, and what they grip.

        keep_reference: a jaw reference, which is typed at the Thor terminal
        only, goes on, and while one runs the jaws are sent no light hold.
        hold_idle: the light hold is sent with no command running too, for
        jaws a person's trigger moved.
        """
        with self._cond:
            kept: list[_Job] = []
            dropped: list[_Job] = []
            for queued in self._queue:
                keep = keep_reference and queued.mode == REFERENCE
                (kept if keep else dropped).append(queued)
            self._queue = deque(kept)
            job = self._running
            # A command whose arm calls just ended, while its fault is told:
            # the jaws are where that command left them. So are jaws a
            # person's trigger moved last.
            hold_idle = hold_idle or self._finishing is not None or self._by_trigger
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
        if (job is not None or hold_idle) and not late:
            # A light hold that already failed in this stop is not reported twice.
            self._light_hold(report=held)
        if job is not None or dropped:
            _log.info(
                "jaw stop: %s, %d queued dropped",
                "nothing was running, the jaws stay as they are"
                if job is None
                else "running command cancelled, light hold sent",
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
        owner: Owner | None,
        on_done: DoneFn | None,
        reason: str | None,
        alone: bool = False,
        numbers: tuple[int, int] | None = None,
        expires_mono_s: float | None = None,
        may_close: bool = True,
        scout: float | None = None,
        own_effort: bool = False,
    ) -> str | None:
        """Queue a request unless `reason` or the authority refuses it.

        alone: it is taken only while no other jaw request runs or waits.
        numbers: the owner number and the stop number the request quotes; it
        is taken only while they are the current ones.
        expires_mono_s: the request is not started at or after this time.
        may_close: False lets a FOLLOW step send no close.
        scout: the scout stage's opening of a two-stage close.
        own_effort: a CLOSE_TORQUE runs at the driver's own effort.
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
                on_done,
                state.owner_number,
                state.stop_number,
                expires_mono_s,
                may_close=bool(may_close),
                scout=None if scout is None else float(scout),
                own_effort=bool(own_effort),
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
        return self._command(job, job.opening, job.mode)

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
        # One classification, so ROBOT_STATE has a jaw reading again. The
        # operator's word was for the jaws as they were before they closed.
        self._jaw_state.clear_word()
        self._jaw_state.classify(self._arm)
        job.measured = self._jaw_state.gripper_state(self._arm).opening
        _log.info("jaws referenced; opening %s", job.measured)
        return GripperOutcome(REACHED, job.measured), None

    def _follow(self, job: _Job) -> tuple[GripperOutcome, str | None]:
        """One step toward the opening a trigger asks for: at most one arm command.

        It ends "reached" with nothing sent when the jaws are within reach_tol
        of the opening, or hold something with a jaw target no wider than it:
        that light hold stands, so easing the trigger squeezes nothing anew.
        Without may_close it sends no close. A step that stalls or times out
        short of its target is left in the light hold. After
        cut_follow() ended a close the jaws are still aimed at that close's
        target: the step then always sends them the opening, or with no close
        allowed where they are.
        """
        detail = self._ended_early(job)
        if detail is not None:
            return GripperOutcome(CANCELLED, None, detail), None
        if not self._arm.gripper_referenced():
            return GripperOutcome(REFUSED, None, NOT_REFERENCED), None
        start = _finite(self._arm.gripper_read())
        if start is None:
            # A step from an unknown opening could drop or squeeze an object.
            detail = "the jaw opening could not be read, so no jaw command was sent"
            return GripperOutcome(REFUSED, None, detail), None
        job.measured = start
        config = self._jaw_state.config
        wanted, tol = job.opening, config.reach_tol
        with self._cond:
            after_cut = self._after_cut
        to_full_open = wanted >= 1.0 - _EPS and start < 1.0 - _EPS
        if wanted > start + tol or to_full_open:
            step = min(wanted, start + config.open_step)
            outcome, fault = self._command(job, step, PLAIN)
            ended = _finite(outcome.opening)
            short = ended is None or ended < step - tol
            ran = outcome.status in (REACHED, SENT, STALLED, TIMEOUT, GRIPPER_FAULT)
            if fault is None and ran and short:
                # Blocked: the jaws are not left pushing at the opening stiffness.
                self._light_hold()
                if outcome.status != TIMEOUT:
                    # Also where the driver took the opening as done, or held
                    # the jaws where they were blocked: for a trigger, a stall.
                    outcome = GripperOutcome(STALLED, ended, outcome.detail)
            return _as_step(outcome), fault
        if after_cut:
            aim = wanted if job.may_close else max(wanted, start)
            if aim < start:
                return self._close_toward(job, aim)
            outcome, fault = self._command(job, aim, PLAIN)
            return _as_step(outcome), fault
        if wanted < start - tol and job.may_close:
            target = _finite(self._arm.gripper_last_target())
            held = self._jaw_state.may_be_holding and target is not None
            if not (held and wanted >= target - tol):
                return self._close_toward(job, wanted)
        return GripperOutcome(REACHED, start), None

    def _close_toward(
        self, job: _Job, opening: float
    ) -> tuple[GripperOutcome, str | None]:
        """A close toward a trigger: one arm command that cut_follow() may end."""
        with self._cond:
            job.closing = True
        outcome, fault = self._command(job, opening, CLOSE_UNTIL_CONTACT)
        with self._cond:
            job.closing = False
            cut = job.cut
        ended = _finite(outcome.opening)
        if fault is None and outcome.status in (STALLED, TIMEOUT):
            if ended is None or ended > opening + _EPS:
                # On what the jaws met: held lightly, not at the closing stiffness.
                self._light_hold()
        if cut and fault is None and outcome.status == CANCELLED:
            target = _finite(self._arm.gripper_last_target())
            if target is not None and abs(target - opening) <= _EPS:
                # The close had begun: the jaws are aimed at its target.
                self._light_hold()
                with self._cond:
                    self._after_cut = True
        return outcome, fault

    def _command(
        self, job: _Job, opening: float, mode: str
    ) -> tuple[GripperOutcome, str | None]:
        """Give the arm one jaw command, unless the request was stopped first.

        Jaws without their reference are referenced by the driver's own
        routines, which close them first: not while they may hold something.
        """
        detail = self._ended_early(job)
        if detail is not None:
            return GripperOutcome(CANCELLED, job.measured, detail), None
        if self._jaw_state.may_be_holding and not self._arm.gripper_referenced():
            return GripperOutcome(REFUSED, None, NOT_REFERENCED), None
        fault: str | None = None
        t_start = self._clock.monotonic()
        with self._cond:
            self._after_cut = False  # this command aims the jaws anew
            self._by_trigger = job.mode == FOLLOW
        try:
            outcome = self._run_on_arm(job, opening, mode)
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

    def _run_on_arm(self, job: _Job, opening: float, mode: str) -> GripperOutcome:
        """The arm port's call for one jaw command of this mode."""
        arm, effort, cancel = self._arm, job.max_effort, job.cancel
        two_stage = getattr(arm, "gripper_close_two_stage", None)
        if mode == CLOSE_UNTIL_CONTACT and job.scout is not None and two_stage:
            return two_stage(job.scout, opening, effort, cancel)
        at_rest = getattr(arm, "gripper_close_torque", None)
        if mode == CLOSE_TORQUE and job.own_effort and at_rest:
            return at_rest(cancel)
        if mode in getattr(arm, "jaw_modes", BASIC_MODES):
            return arm.gripper_command(opening, effort, mode, cancel)
        if mode == OPEN:
            if arm.gripper_referenced() and _finite(arm.gripper_read()) is None:
                # An opening from an unknown opening could drop an object.
                detail = "the jaw opening could not be read, so no opening was sent"
                return GripperOutcome(REFUSED, None, detail)
            return arm.gripper_command(1.0, effort, PLAIN, cancel)
        # CLOSE and CLOSE_TORQUE on a port without the driver's routines.
        outcome = arm.gripper_command(0.0, effort, CLOSE_UNTIL_CONTACT, cancel)
        if mode == CLOSE_TORQUE and outcome.status in (REACHED, STALLED, TIMEOUT):
            # The close before a torque release ends in the light hold.
            arm.gripper_light_hold()
        return outcome

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
        with self._cond:
            # After a stop's light hold no step aims the jaws anew.
            self._after_cut = False
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
    newest opening only: a close under way is cut when the trigger asks for
    a wider opening. Jaws that an opening left short of the grip level open
    to what the trigger asks as it matches. Until the jaws were measured
    fully open no opening below the grip level is asked of them, and without
    the leave to close none below where they are. Thread-safe.
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
        # The opening the jaws have settled for; None while an opening goes on
        # and after this follower cut a close.
        self._settled: float | None = None
        self._flying: float | None = None  # the opening of the step under way
        self._cutting = False  # that step is a close this follower has cut
        self._step_end: float | None = None  # where an opening's last step ended
        # Why the jaws may only open, as the newest human target had it.
        self._no_close: str | None = None
        self._stalled: float | None = None  # where an opening ended short
        # Why the trigger moves nothing, or the jaws are not where it asks.
        self._refusal = ""

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
            # Jaws at the grip level, or held wider by an object, are sent
            # nothing. Jaws that an opening left short of it open to what the
            # trigger asks: they start from where they were read last.
            reading = _finite(self._jaw_state.gripper_state(self._arm).opening)
            short = reading is not None and reading < self._full_open
            short = short and reading < level - self._band - _EPS
            self._settled = reading if short else level
            self._say(f"the jaws follow the trigger from the grip level {level:.2f}")
        wanted = 1.0 if asked >= self._full_open else asked
        refusal = ""
        if not self._opened and wanted < level:
            wanted, refusal = level, NO_TIGHTER
        # Without the leave to close the jaws go no further than they have
        # settled; _done() says where a step under way has left them.
        self._no_close = no_close
        floor = self._settled if self._flying is None else None
        if no_close is not None and floor is not None and wanted < floor:
            wanted, refusal = floor, f"{NO_CLOSING}{no_close}"
        stalled = self._stalled
        if not refusal and stalled is not None and wanted > stalled + self._band:
            refusal = f"{OPENING_STALLED}{stalled:.2f}"
        self._wanted, self._refusal = wanted, refusal
        flying = self._flying
        if flying is not None and not self._cutting and wanted > flying + self._band:
            # A close under way goes no further once the trigger asks for more.
            self._cutting = self._worker.cut_follow(wanted)
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
        self._worker.follow(
            wanted,
            numbers[0],
            numbers[1],
            self._done,
            may_close=self._no_close is None,
        )

    def _done(self, outcome: GripperOutcome) -> None:
        """The end of one jaw step: on the gripper worker's thread, or a caller's."""
        with self._lock:
            flying, self._flying = self._flying, None
            cut, self._cutting = self._cutting, False
            measured = _finite(outcome.opening)
            if measured is not None and measured >= self._full_open:
                self._opened = True
            if flying is None or not self._matched:
                return  # ended meanwhile: the grip level is noted anew
            own_cut = cut and outcome.status == CANCELLED
            if not own_cut and outcome.status not in (REACHED, STALLED, TIMEOUT):
                why = outcome.detail or "no reason given"
                reason = f"a jaw step ended {outcome.status}: {why}"
                self._say(f"the jaws no longer follow the trigger: {reason}")
                self._drop(reason)
                return
            band = self._band
            # An opening goes one step at a time: more of it may be left, and
            # a full open ends at 1.0, not within reach of it.
            short = measured is not None and measured < flying - band
            to_end = flying >= 1.0 - _EPS
            more = short or (to_end and measured is not None and measured < 1.0 - _EPS)
            further = self._step_end is None or (
                measured is not None and measured > self._step_end + _EPS
            )
            self._stalled = None
            if own_cut:
                # The next step aims the jaws anew, wherever the cut left them.
                self._step_end = self._settled = None
            elif outcome.status == REACHED and more and further:
                self._step_end, self._settled = measured, None
            else:
                self._step_end, self._settled = None, flying
                if short:
                    self._stalled = measured
                    self._refusal = f"{OPENING_STALLED}{measured:.2f}"
                    self._say(f"the jaws stalled while opening, at {measured:.2f}")
            self._hold_without_leave(measured, flying, keep_step=own_cut)
            self._pump()

    def _hold_without_leave(
        self, measured: float | None, flying: float, keep_step: bool
    ) -> None:
        """Without the leave to close, stay where the ended step left the jaws.

        The lock is held. Without an end reading that is the step's own
        opening. keep_step leaves the next step to be asked.
        """
        wanted = self._wanted
        if self._no_close is None or wanted is None:
            return
        here = flying if measured is None else measured
        if wanted >= here - self._band:
            return
        self._wanted, self._refusal = here, f"{NO_CLOSING}{self._no_close}"
        if not keep_step:
            self._step_end, self._settled = None, here

    def _drop(self, reason: str) -> None:
        """Forget the match and the grip level; the lock is held."""
        self._matched = False
        self._numbers = self._level = None
        self._asked = self._wanted = self._settled = self._step_end = None
        self._no_close = self._stalled = None
        self._refusal = reason

    def _cut(self, numbers: tuple[int, int]) -> None:
        """Hold the jaws lightly where they are, while these numbers stand."""
        state = self._authority.snapshot()
        current = (state.owner_number, state.stop_number)
        if state.owner is Owner.HUMAN and current == numbers:
            self._worker.stop_now(hold_idle=True)


def _as_step(outcome: GripperOutcome) -> GripperOutcome:
    """A plain command's end as a step toward a trigger: one only sent has ended."""
    if outcome.status != SENT:
        return outcome
    return GripperOutcome(REACHED, outcome.opening, outcome.detail)


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


def _scout_refusal(scout_opening: object, mode: object) -> str | None:
    if scout_opening is None:
        return None
    if mode != CLOSE_UNTIL_CONTACT:
        return f"scout_opening is only for mode {CLOSE_UNTIL_CONTACT!r}"
    return _fraction_refusal("scout_opening", scout_opening)


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
