# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The move executor: sends a move's setpoints on time, each checked just before."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

from ..protocol.messages import MOVE_ARMED, MOVE_SENDING
from .interfaces import (
    ArmFault,
    ContactStop,
    FeedbackLost,
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
)

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .authority import Authority
    from .clock import Clock
    from .config import ExecutorConfig
    from .interfaces import (
        ArmPort,
        AuthorityState,
        Checker,
        ContactWatch,
        JointReading,
    )
    from .permits import PermitGate
    from .state import JointHistory

    # How a move ended: (status, reason).
    Ending = tuple[str, str]
    # The setpoints of a move, planned from its measured start pose; may raise
    # Refusal.
    Replan = Callable[[list[float]], Sequence[Sequence[float]]]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.executor")

# One wait for the first permit; the gate and the stop conditions are asked again
# after each.
_PERMIT_POLL_S = 0.001
# A hold at the measured pose needs a joint read, and one failed read is tried again.
_HOLD_ATTEMPTS = 2
# In a move's reason, between the contact and why no hold followed it.
HOLD_FAILED = "; the hold at the measured pose failed: "

_IDLE: tuple[str, int, int] = ("", 0, 0)


def _never_stop() -> str | None:
    return None


def _is_number(value: object) -> bool:
    """True for a finite real number that is not a bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _as_pose(row: object, n: int) -> list[float] | None:
    """row as n finite floats, None when it is anything else."""
    if hasattr(row, "tolist"):  # a numpy array
        row = row.tolist()
    if not isinstance(row, (list, tuple)) or len(row) != n:
        return None
    if not all(_is_number(value) for value in row):
        return None
    return [float(value) for value in row]


def _with_permit_refusal(reason: str, gate: PermitGate, move_id: str) -> str:
    """Why no permit covers a setpoint, with why the move's last permit was refused."""
    why = gate.last_refusal(move_id)
    if why is None:
        return reason
    return f"{reason}; its last permit was refused: {why}"


def _pose_of(readings: Sequence[JointReading], n: int) -> list[float] | None:
    """The pose of one full read, None unless every joint has a valid finite value."""
    pose: list[float | None] = [None] * n
    for reading in readings:
        if 0 <= reading.joint < n and reading.valid and math.isfinite(reading.value):
            pose[reading.joint] = float(reading.value)
    return None if any(value is None for value in pose) else pose


@dataclass
class _Move:
    """What one run() call carries from step to step."""

    move_id: str
    owner: Owner
    checker: Checker
    admitted: AuthorityState
    should_stop: Callable[[], str | None]
    expires_mono_s: float | None = None  # no first setpoint at or after this time
    halt_count: int = 0  # the checker's halt count the move was admitted at
    watch: ContactWatch | None = None
    total: int = 0
    sent: int = 0  # setpoints sent so far
    missed: int = 0  # deadlines missed by more than the small delay
    settled_q: list[float] | None = None  # the full read the move settled on


class MoveExecutor:
    """Runs one move at a time: every setpoint once, in order, one per period.

    Before each setpoint it asks the stop conditions, the permit gate and the
    checker, the first one from the measured start pose. It never changes the
    control owner, raises no stop and leaves the jaws alone. MoveOutcome.index
    and progress() count the setpoints sent.
    """

    def __init__(
        self,
        arm: ArmPort,
        authority: Authority,
        history: JointHistory | None,
        config: ExecutorConfig,
        clock: Clock,
        first_wait_s: float = 0.5,
        *,
        on_missed_deadline: Callable[[], None] | None = None,
        reads_paused: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        """first_wait_s is the longest a move waits for its first covering permit.

        on_missed_deadline is called for every setpoint that goes out later than
        the small delay allows, and for a move that ends "late". A move's start
        read runs inside reads_paused() (IdleReader.paused), and so does the
        move from its first setpoint on; the wait for the first permit that
        lies between the two does not.
        """
        if not (_is_number(first_wait_s) and first_wait_s >= 0.0):
            raise ValueError(f"first_wait_s must be 0 or more, got {first_wait_s!r}")
        self._arm = arm
        self._authority = authority
        self._history = history
        self._config = config
        self._clock = clock
        self._first_wait_s = float(first_wait_s)
        self._on_missed_deadline = on_missed_deadline
        self._reads_paused = reads_paused or contextlib.nullcontext
        self._busy = threading.Lock()  # held while a move runs
        self._lock = threading.Lock()  # guards _progress and _stage
        self._progress = _IDLE
        self._stage = ""

    def progress(self) -> tuple[str, int, int]:
        """(move id, setpoints sent, setpoints in the move); ("", 0, 0) when idle."""
        with self._lock:
            return self._progress

    def stage(self) -> str:
        """How far the running move has got: "", MOVE_ARMED or MOVE_SENDING.

        "" when idle and until the move's start checks have passed. Armed: they
        passed, nothing was sent and the move waits for its first permit.
        Sending: from the first covering permit on, or at once without a gate.
        """
        with self._lock:
            return self._stage

    @contextlib.contextmanager
    def claim_idle(self) -> Iterator[bool]:
        """Keep every move from starting while the block runs.

        Yields False, and claims nothing, when a move is running. The guarded
        torque release uses it, so no move can begin between its checks and the cut.
        """
        claimed = self._busy.acquire(blocking=False)
        try:
            yield claimed
        finally:
            if claimed:
                self._busy.release()

    def run(
        self,
        setpoints: Sequence[Sequence[float]],
        start_q: Sequence[float],
        start_tol: float,
        period_s: float,
        owner: Owner,
        checker: Checker,
        admitted: AuthorityState,
        move_id: str,
        permit_gate: PermitGate | None = None,
        grasp_region: GraspRegion | None = None,
        should_stop: Callable[[], str | None] = _never_stop,
        *,
        permit_arrived: threading.Event | None = None,
        halt_count: int | None = None,
        expires_mono_s: float | None = None,
        replan: Replan | None = None,
    ) -> MoveOutcome:
        """Run one move to its end and say how it ended.

        It stops once the authority's numbers differ from `admitted`. With a gate
        nothing is sent until a permit covers the first setpoint (permit_arrived
        wakes that wait), and no first setpoint once expires_mono_s passed.
        halt_count None reads the checker's. replan(measured pose) plans the rows.
        """
        move = _Move(
            str(move_id), Owner(owner), checker, admitted, should_stop, expires_mono_s
        )
        if not self._busy.acquire(blocking=False):
            # Also while a torque release holds claim_idle().
            return MoveOutcome("refused", "another move is running")
        try:
            if halt_count is None:
                halt_count = checker.halt_count()
            move.halt_count = halt_count
            try:
                outcome = self._execute(
                    move,
                    setpoints,
                    start_q,
                    start_tol,
                    period_s,
                    permit_gate,
                    grasp_region,
                    permit_arrived,
                    replan,
                )
            except ArmFault as fault:
                outcome = MoveOutcome("failed", f"fault: {fault}", move.sent)
        finally:
            try:
                self._leave(checker, move.watch)
            finally:
                self._set_progress(_IDLE)
                self._busy.release()
        _log.info(
            "move %s for %s: %s, sent %d of %d setpoints, missed %d deadlines (%s)",
            move.move_id,
            move.owner,
            outcome.status,
            move.sent,
            move.total,
            move.missed,
            outcome.reason or "no reason given",
        )
        return outcome

    def _execute(
        self,
        move: _Move,
        setpoints: Sequence[Sequence[float]],
        start_q: Sequence[float],
        start_tol: float,
        period_s: float,
        gate: PermitGate | None,
        region: GraspRegion | None,
        arrived: threading.Event | None,
        replan: Replan | None,
    ) -> MoveOutcome:
        try:
            rows, start = self._checked(move, setpoints, start_q, start_tol, period_s)
        except Refusal as refusal:
            return MoveOutcome("refused", refusal.reason)
        move.total = len(rows)
        self._set_progress((move.move_id, 0, move.total))
        stop = self._stop_reason(move)
        if stop is not None:
            return MoveOutcome("stopped", stop)
        if not self._arm.torque_on():
            return MoveOutcome("refused", "the motors are off")
        expired = self._expired(move)
        if expired is not None:
            return MoveOutcome("refused", expired)
        # No idle read shares the bus, or the clock, with the start read.
        with self._reads_paused():
            begun = self._read_start(move, rows, start, start_tol, replan)
            if isinstance(begun, MoveOutcome):
                return begun
            rows, measured = begun
            if gate is None:
                self._set_stage(MOVE_SENDING)
                return self._from_start(move, rows, measured, gate, region)
        # Armed: every check that needs no permit has passed and nothing was
        # sent. The idle reads go on during the wait, so the joint history and
        # the lost-feedback rule of a hold stay live.
        self._set_stage(MOVE_ARMED)
        # The first permit also feeds the watchdog that begin_motion checks.
        ended = self._await_first_permit(move, gate, arrived)
        if ended is not None:
            return MoveOutcome(*ended)
        self._set_stage(MOVE_SENDING)
        # Nor with the move from here on. The arm held since the start read, so
        # the first setpoint is checked from the pose that read measured.
        with self._reads_paused():
            return self._from_start(move, rows, measured, gate, region)

    def _read_start(
        self,
        move: _Move,
        rows: list[list[float]],
        start: list[float],
        start_tol: float,
        replan: Replan | None,
    ) -> tuple[list[list[float]], list[float]] | MoveOutcome:
        """Read the start pose and compare it with the move's.

        Returns the setpoints, replanned when asked, and the measured pose; or
        the refusal. Nothing is sent.
        """
        try:
            readings = self._arm.read_all()
        except ReadFailed as failed:
            return MoveOutcome("refused", f"the start pose could not be read: {failed}")
        self._keep(readings)
        measured = _pose_of(readings, self._arm.n_joints)
        if measured is None:
            return MoveOutcome("refused", "the start pose read left a joint out")
        for joint, (at, expected) in enumerate(zip(measured, start)):
            if abs(at - expected) > start_tol:
                reason = (
                    f"start pose: joint index {joint} is at {at:.4f} rad, the move "
                    f"starts at {expected:.4f} rad (tolerance {start_tol:g} rad)"
                )
                return MoveOutcome("refused", reason, 0, measured)
        if replan is not None:
            try:
                rows = self._rows(replan(measured))
            except Refusal as refusal:
                return MoveOutcome("refused", refusal.reason, 0, measured)
            move.total = len(rows)
            self._set_progress((move.move_id, 0, move.total))
        return rows, measured

    def _from_start(
        self,
        move: _Move,
        rows: list[list[float]],
        measured: list[float],
        gate: PermitGate | None,
        region: GraspRegion | None,
    ) -> MoveOutcome:
        """Send the move from its measured start pose; how it ended."""
        halt_count = move.halt_count
        try:
            move.checker.set_grasp_region(region)
            move.checker.begin_motion(halt_count)
            move.watch = move.checker.contact_watch(self._arm, measured, self._history)
            ended = (
                self._send_setpoints(move, rows, measured, gate)
                or self._settle(move, rows[-1])
                or ("completed", "")
            )
        except ContactStop as contact:
            ended = ("contact", contact.reason + self._hold_measured(move.watch))
        except FeedbackLost as lost:
            # Like a driver fault: the caller raises a fault-stop.
            ended = ("failed", f"fault: {lost.reason}")
        except Refusal as refusal:
            ended = self._refused(move, refusal.reason)
        except ReadFailed as failed:
            ended = ("refused", f"a joint read failed: {failed}")
        # A completed move reports the pose of the full read it settled on.
        completed = ended[0] == "completed"
        end_q = move.settled_q if completed else self._read_pose()
        return MoveOutcome(*ended, move.sent, end_q)

    def _checked(
        self,
        move: _Move,
        setpoints: object,
        start_q: object,
        start_tol: float,
        period_s: float,
    ) -> tuple[list[list[float]], list[float]]:
        """The setpoints and the start pose as lists of floats; raises Refusal."""
        n = self._arm.n_joints
        config = self._config
        if move.owner is Owner.NOBODY:
            raise Refusal("control owner nobody cannot move the arm")
        rows = self._rows(setpoints)
        if not (
            _is_number(period_s)
            and math.isclose(period_s, config.period_s, rel_tol=1e-9)
        ):
            raise Refusal(
                f"period_s is {period_s!r}, the configured period is "
                f"{config.period_s:g} s"
            )
        if not (_is_number(start_tol) and start_tol >= 0.0):
            raise Refusal(f"start_tol must be 0 or more, got {start_tol!r}")
        if start_tol > config.max_start_tol_rad:
            raise Refusal(
                f"start_tol is {start_tol:g} rad, the limit is "
                f"{config.max_start_tol_rad:g} rad"
            )
        expires = move.expires_mono_s
        if expires is not None and not _is_number(expires):
            raise Refusal(f"expires_mono_s must be a finite time, got {expires!r}")
        start = _as_pose(start_q, n)
        if start is None:
            raise Refusal(f"start_q needs {n} finite joint values")
        return rows, start

    def _rows(self, setpoints: object) -> list[list[float]]:
        """The setpoints as lists of floats; raises Refusal."""
        n = self._arm.n_joints
        if hasattr(setpoints, "tolist"):  # a numpy array
            setpoints = setpoints.tolist()
        if not isinstance(setpoints, (list, tuple)) or not setpoints:
            raise Refusal("the move has no setpoints")
        if len(setpoints) > self._config.max_setpoints:
            raise Refusal(
                f"the move has {len(setpoints)} setpoints, the limit is "
                f"{self._config.max_setpoints}"
            )
        rows: list[list[float]] = []
        for index, row in enumerate(setpoints):
            q = _as_pose(row, n)
            if q is None:
                raise Refusal(f"setpoint {index} needs {n} finite joint values")
            rows.append(q)
        return rows

    def _stop_reason(self, move: _Move) -> str | None:
        """Why the move must stop now, None while it may go on."""
        reason = move.should_stop()
        if reason is not None:
            return reason or "stop requested"
        now = self._authority.snapshot()
        admitted = move.admitted
        if now.stop_number != admitted.stop_number:
            return (
                f"stop number is now {now.stop_number}, the move was admitted "
                f"at {admitted.stop_number}"
            )
        if now.owner_number != admitted.owner_number:
            return (
                f"owner number is now {now.owner_number}, the move was admitted "
                f"at {admitted.owner_number}"
            )
        if now.owner is not move.owner:
            return f"control owner is {now.owner}, not {move.owner}"
        return None

    def _expired(self, move: _Move) -> str | None:
        """Why the move may no longer begin, None while its expiry lies ahead."""
        if move.expires_mono_s is None:
            return None
        late = self._clock.monotonic() - move.expires_mono_s
        return None if late < 0.0 else f"the move expired {late:.3f} s ago"

    def _refused(self, move: _Move, reason: str) -> Ending:
        """A refusal, reported as a stop when the move was stopped meanwhile."""
        stop = self._stop_reason(move)
        return ("refused", reason) if stop is None else ("stopped", stop)

    def _await_first_permit(
        self, move: _Move, gate: PermitGate, arrived: threading.Event | None
    ) -> Ending | None:
        """Wait for a permit that covers the first setpoint; None once one does.

        With `arrived` the wait is in real time, so a fake clock stands still and
        another thread can deliver the permit; the wait then also ends after
        first_wait_s of its own waiting. Without it the wait sleeps on the clock.
        """
        started = self._clock.monotonic()
        waited = 0.0  # seconds of waits on `arrived` that ran out
        while True:
            stop = self._stop_reason(move)
            if stop is not None:
                return "stopped", stop
            now = self._clock.monotonic()
            reason = gate.covers(move.move_id, 0, now)
            if reason is None:
                return None
            expired = self._expired(move)
            if expired is not None:
                return self._refused(move, expired)
            if max(now - started, waited) >= self._first_wait_s:
                reason = _with_permit_refusal(reason, gate, move.move_id)
                return self._refused(
                    move, f"{reason} (waited {self._first_wait_s:g} s)"
                )
            if arrived is None:
                self._clock.sleep(_PERMIT_POLL_S)
            elif arrived.wait(_PERMIT_POLL_S):
                arrived.clear()
            else:
                waited += _PERMIT_POLL_S

    def _send_setpoints(
        self,
        move: _Move,
        rows: list[list[float]],
        measured: list[float],
        gate: PermitGate | None,
    ) -> Ending | None:
        """Send every setpoint in turn; how the move ended early, None if all went.

        Setpoint i is due at t0 + i * period. A late wake-up moves t0 back, and
        so does a check and send that took longer than the small delay; what
        they usually cost does not, so it cannot stretch the move.
        """
        config = self._config
        period = config.period_s
        q_prev = measured  # the first step starts where the arm is
        t0 = self._clock.monotonic()
        for index, q in enumerate(rows):
            stop = self._stop_reason(move)
            if stop is not None:
                return "stopped", stop
            now = self._clock.monotonic()
            ended = self._permit_problem(move, gate, index, now)
            if ended is not None:
                return ended
            if index == 0:
                expired = self._expired(move)
                if expired is not None:
                    return self._refused(move, expired)
            late = now - (t0 + index * period)
            if late > config.large_delay_s:
                return self._late(move, index, late, "is")
            if late > 0.0:
                # Later setpoints move back with this one: none is skipped and
                # none is sent sooner to catch up.
                t0 += late
                if late > config.small_delay_s:
                    self._note_missed(move)
            due = t0 + index * period
            move.checker.check_setpoint(q_prev, q, period)
            # The check took time: what held before it is asked again.
            stop = self._stop_reason(move)
            if stop is not None:
                return "stopped", stop
            now = self._clock.monotonic()
            ended = self._permit_problem(move, gate, index, now)
            if ended is not None:
                return ended
            if index == 0:
                expired = self._expired(move)
                if expired is not None:
                    return self._refused(move, expired)
            if now - due > config.large_delay_s:
                return self._late(move, index, now - due, "is")
            self._arm.send_target(q)
            behind = self._clock.monotonic() - due
            move.sent = index + 1
            self._set_progress((move.move_id, move.sent, move.total))
            if behind > config.large_delay_s:
                return self._late(move, index, behind, "went out")
            if behind > config.small_delay_s:
                # A delay, not the usual cost: the next setpoint keeps its distance.
                t0 += behind
                self._note_missed(move)
            deadline = t0 + (index + 1) * period
            move.watch.after_send(index, q, deadline)
            q_prev = q
            self._clock.sleep(max(0.0, deadline - self._clock.monotonic()))
        return None

    def _permit_problem(
        self, move: _Move, gate: PermitGate | None, index: int, now: float
    ) -> Ending | None:
        """How the move ends when no permit covers setpoint `index` at `now`."""
        reason = None if gate is None else gate.covers(move.move_id, index, now)
        if reason is None:
            return None
        return self._refused(move, _with_permit_refusal(reason, gate, move.move_id))

    def _late(self, move: _Move, index: int, by_s: float, what: str) -> Ending:
        """The end of a move whose setpoint is, or went out, over the large delay."""
        self._note_missed(move)
        limit = self._config.large_delay_s
        return "late", (
            f"setpoint {index} {what} {by_s:.3f} s late, over the limit of {limit:g} s"
        )

    def _settle(self, move: _Move, q_last: list[float]) -> Ending | None:
        """Read until one full read has every joint at the last setpoint; None then.

        The watch reads a few joints a pass. Once each joint's newest reading
        is valid and near, one read of every joint decides.
        """
        config = self._config
        near = [False] * len(q_last)
        started = self._clock.monotonic()
        while True:
            stop = self._stop_reason(move)
            if stop is not None:
                return "stopped", stop
            deadline = self._clock.monotonic() + config.period_s
            # Nothing is sent here: the watch reads against the standing setpoint,
            # so a contact while settling still ends in a hold.
            readings = move.watch.after_send(move.sent - 1, q_last, deadline)
            self._note_near(near, readings, q_last)
            if all(near):
                near = self._read_near(move, q_last)
                if all(near):
                    return None
            now = self._clock.monotonic()
            if now - started >= config.settle_timeout_s:
                return "not_settled", (
                    f"joint index {near.index(False)} is not within "
                    f"{config.settle_tol_rad:g} rad of the last setpoint after "
                    f"{config.settle_timeout_s:g} s"
                )
            self._clock.sleep(max(0.0, deadline - now))

    def _note_near(
        self, near: list[bool], readings: Sequence[JointReading], q_last: list[float]
    ) -> None:
        """Set near[joint] from each reading: valid and inside the settle tolerance.

        A reading that is not valid takes back what an earlier one showed.
        """
        tol = self._config.settle_tol_rad
        for reading in readings:
            joint = reading.joint
            if 0 <= joint < len(near):
                valid = reading.valid and math.isfinite(reading.value)
                near[joint] = valid and abs(reading.value - q_last[joint]) <= tol

    def _read_near(self, move: _Move, q_last: list[float]) -> list[bool]:
        """Per joint, whether one fresh full read has it valid and near q_last.

        Keeps that read's pose in move.settled_q when every joint is.
        """
        near = [False] * len(q_last)
        try:
            readings = self._arm.read_all()
        except ReadFailed as failed:
            _log.debug("no settled pose yet: the full read failed: %s", failed)
            return near
        self._keep(readings)
        self._note_near(near, readings, q_last)
        if all(near):
            move.settled_q = _pose_of(readings, len(near))
        return near

    def _hold_measured(self, watch: ContactWatch | None) -> str:
        """Hold at a freshly measured pose; text for the reason when that failed.

        The move's watch does it, so the reading is one that watch trusts. When
        it fails nothing is sent and the last setpoint stands.
        """
        if watch is None:
            return ""  # the move had sent nothing yet: the target before it stands
        error: ReadFailed | None = None
        for _ in range(_HOLD_ATTEMPTS):
            try:
                watch.hold()
            except ReadFailed as failed:
                error = failed
            else:
                return ""
        _log.error("no hold at the measured pose after a contact: %s", error)
        return f"{HOLD_FAILED}{error}; the last setpoint stands"

    def _read_pose(self) -> list[float] | None:
        """The pose from one fresh full read, None when the read failed."""
        try:
            readings = self._arm.read_all()
        except ReadFailed as failed:
            _log.warning("the end pose could not be read: %s", failed)
            return None
        self._keep(readings)
        return _pose_of(readings, self._arm.n_joints)

    def _keep(self, readings: Sequence[JointReading]) -> None:
        if self._history is not None:
            for reading in readings:
                self._history.add(reading)

    def _note_missed(self, move: _Move) -> None:
        move.missed += 1
        if self._on_missed_deadline is not None:
            self._on_missed_deadline()

    def _set_progress(self, progress: tuple[str, int, int]) -> None:
        with self._lock:
            self._progress = progress
            if progress == _IDLE:
                self._stage = ""

    def _set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    @staticmethod
    def _leave(checker: Checker, watch: ContactWatch | None) -> None:
        """What every exit path owes the checker and the watch."""
        try:
            checker.set_grasp_region(None)
        except Refusal as refusal:
            _log.error("the grasp region could not be cleared: %s", refusal.reason)
        finally:
            try:
                checker.end_motion()
            finally:
                if watch is not None:
                    watch.finish()
