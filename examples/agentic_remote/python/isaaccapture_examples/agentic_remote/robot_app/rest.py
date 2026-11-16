# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return to rest and the guarded torque release.

See section 11 of docs/design/claude_agentic_remote_execution.md. No other module
may switch the motors off.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from .interfaces import (
    JawLoad,
    MotionState,
    Owner,
    ReadFailed,
    Refusal,
    RestOutcome,
    StopLevel,
)

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .authority import Authority
    from .clock import Clock
    from .config import RobotAppConfig
    from .executor import MoveExecutor
    from .gripper import JawState
    from .interfaces import (
        ArmPort,
        AuthorityState,
        CheckerSet,
        MoveOutcome,
    )
    from .state import JointHistory

    # The planned move: (setpoints, duration in seconds).
    Plan = tuple[list[list[float]], float]
    # Sends one operator event: (kind, text, data).
    EmitEvent = Callable[[str, str, dict[str, Any]], None]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.rest")

# Who asked for the return.
TYPED = "typed"  # the operator typed `rest`
WAIT = "wait"  # the assistance wait ran out: the timed return
SHUTDOWN = "shutdown"
TAKEOVER_END = "takeover_end"
SOURCES = (TYPED, WAIT, SHUTDOWN, TAKEOVER_END)

# RestOutcome statuses.
AT_REST = "at_rest"  # measured at rest, motors on
REFUSED = "refused"  # a condition failed before recovery took the arm: no change
STOPPED = "stopped"  # ended by a STOP, a fault-stop or an owner change
# Recovery had the arm: its move did not start or ended early, or rest is unproven.
FAILED = "failed"

# Kinds of the operator events.
EVENT_REST = "rest"
EVENT_RELEASE = "release"
# What a release tells the operator: with a driver that knows its motors are
# off, and with one that only sent the command.
RELEASED = "torque released at rest: the motors are off"
RELEASE_SENT = (
    "torque release sent at rest: the driver does not confirm that the motors are off"
)

# Peak speed of the minimum-jerk time law over its mean speed.
_PEAK_OVER_MEAN = 1.875
# The planned peak joint speed as a share of the recovery speed.
_SPEED_SHARE = 0.9
# Added to agree_tol_rad for the start pose the executor reads once more.
_START_MARGIN_RAD = 0.005
_EPS = 1e-9

# Why a typed rest, or a shutdown that has to move the arm, is refused unconfirmed.
CONFIRM_PATH = "confirm the path is clear"
_BUSY = "a return to rest or a release is already running"
_OWNER_CHANGED = "the control owner changed while the return to rest was checked"


class MotionStateHolder(Protocol):
    """Where the robot app keeps the arm's motion state."""

    def get(self) -> MotionState:
        """The motion state now."""

    def set(self, state: MotionState) -> None:
        """Replace the motion state."""


def planned_duration_s(
    travel_rad: float, recovery_speed: float, min_duration_s: float
) -> float:
    """Time of a minimum-jerk move whose peak speed is 0.9 of recovery_speed."""
    by_speed = _PEAK_OVER_MEAN * travel_rad / (_SPEED_SHARE * recovery_speed)
    return max(min_duration_s, by_speed)


def setpoint_count(duration_s: float, period_s: float) -> int:
    """Setpoints of a move of duration_s: one per period, rounded up."""
    return max(1, math.ceil(duration_s / period_s - _EPS))


def plan_return(
    q_start: Sequence[float],
    rest_q: Sequence[float],
    period_s: float,
    recovery_speed: float,
    min_duration_s: float,
) -> tuple[list[list[float]], float]:
    """A straight joint-space move from q_start to rest_q, one setpoint per period.

    The path fraction follows s(t) = 10 t^3 - 15 t^4 + 6 t^5. Returns the
    setpoints, the last of which is rest_q, and their duration: planned_duration_s
    of the largest joint travel, rounded up to whole periods.
    """
    start = [float(value) for value in q_start]
    goal = [float(value) for value in rest_q]
    travel = max(abs(b - a) for a, b in zip(start, goal))
    duration = planned_duration_s(travel, recovery_speed, min_duration_s)
    count = setpoint_count(duration, period_s)
    rows: list[list[float]] = []
    for index in range(1, count):
        t = index / count
        s = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
        rows.append([a + (b - a) * s for a, b in zip(start, goal)])
    rows.append(goal)
    return rows, count * period_s


def _never_stop() -> str | None:
    return None


def _fault_text(state: AuthorityState) -> str:
    if state.fault_reason:
        return f"fault-stop latched: {state.fault_reason}"
    return "fault-stop latched"


def _stop_number_text(seen: int, current: int) -> str:
    return f"stop number {seen} does not match the current {current}"


class ReturnToRest:
    """The checked move to the rest pose, and the only torque release.

    A request checks every condition before it changes anything, runs at most
    one move and retries nothing. The motors stay on however a return ends.
    One request or release runs at a time; a second one is refused.
    """

    def __init__(
        self,
        arm: ArmPort,
        authority: Authority,
        checkers: CheckerSet,
        executor: MoveExecutor,
        jaw_state: JawState,
        history: JointHistory | None,
        config: RobotAppConfig,
        clock: Clock,
        emit_event: EmitEvent,
        motion_state: MotionStateHolder,
        last_grasp_region: Callable[[], GraspRegion | None],
        contact_hold_active: Callable[[], bool],
        move_running: Callable[[], bool],
        *,
        jaws_busy: Callable[[], bool] | None = None,
    ) -> None:
        """emit_event(kind, text, data) sends one operator event.

        last_grasp_region() is the grasp region of the agent's last grasp,
        contact_hold_active() whether the arm holds at a contact, move_running()
        whether a move is being sent to the arm, and jaws_busy() whether a jaw
        command is queued or running (GripperWorker.busy).
        """
        self._arm = arm
        self._authority = authority
        self._recovery = checkers.recovery
        self._executor = executor
        self._jaw_state = jaw_state
        self._history = history
        self._rest = config.rest
        self._period_s = config.executor.period_s
        self._max_setpoints = config.executor.max_setpoints
        # The executor reads the start pose once more and refuses a wider tolerance.
        self._start_tol_rad = min(
            config.rest.agree_tol_rad + _START_MARGIN_RAD,
            config.executor.max_start_tol_rad,
        )
        self._clock = clock
        self._emit_event = emit_event
        self._motion_state = motion_state
        self._last_grasp_region = last_grasp_region
        self._contact_hold_active = contact_hold_active
        self._move_running = move_running
        self._jaws_busy = jaws_busy
        self._busy = threading.Lock()  # held while a request or a release runs
        self._moves = 0  # moves started, for their ids
        self._last_move: MoveOutcome | None = None

    @property
    def last_move(self) -> MoveOutcome | None:
        """How the newest request's move ended; None if that request sent no move."""
        return self._last_move

    def request(
        self,
        source: str,
        confirmed: bool = False,
        *,
        stop_number_seen: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
    ) -> RestOutcome:
        """Return the arm to rest if every condition holds; says how it ended.

        confirmed says the operator typed that the path is clear: "typed" needs
        it, "shutdown" when the arm is measured away from rest. "wait" needs
        rest.auto_return. stop_number_seen, when given, must still be the stop
        number. should_stop is asked before every setpoint; a reason ends the move.
        """
        if source not in SOURCES:
            raise ValueError(f"unknown return-to-rest source {source!r}")
        if not self._busy.acquire(blocking=False):
            outcome = RestOutcome(REFUSED, _BUSY)
        else:
            try:
                outcome = self._request(
                    source, confirmed, stop_number_seen, should_stop or _never_stop
                )
            finally:
                self._busy.release()
        self._report(source, outcome)
        return outcome

    def release(
        self,
        stop_number_seen: int,
        should_stop: Callable[[], str | None] | None = None,
    ) -> str | None:
        """Switch the motors off at measured rest; why not, or None when done.

        Needs the current stop number, no fault-stop, no move running, the jaws
        proven empty, and two fresh full reads read_gap_s apart within
        rest_tol_rad of rest and of each other. A reason from should_stop, asked
        first and again just before the cut, keeps the motors on.
        """
        if not self._busy.acquire(blocking=False):
            reason: str | None = _BUSY
        else:
            try:
                reason = self._release(stop_number_seen, should_stop or _never_stop)
            finally:
                self._busy.release()
        if reason is None:
            confirmed = bool(self._arm.release_confirmed())
            if confirmed:
                _log.info("torque released at measured rest")
            else:
                _log.info("torque release sent at measured rest, not confirmed")
            text = RELEASED if confirmed else RELEASE_SENT
            data = {"stop_number": stop_number_seen, "confirmed": confirmed}
            self._tell(EVENT_RELEASE, text, data)
        else:
            _log.warning("torque release refused: %s", reason)
        return reason

    def proven_at_rest(self) -> str | None:
        """Why the arm is not proven at rest with empty jaws; None when it is.

        The jaws must be proven empty, and two fresh full reads read_gap_s
        apart must lie within rest_tol_rad of rest and of each other. It reads
        the jaws and the joints and sends nothing.
        """
        return self._jaws_problem() or self._rest_problem()

    # A request, step by step

    def _request(
        self,
        source: str,
        confirmed: bool,
        stop_number_seen: int | None,
        should_stop: Callable[[], str | None],
    ) -> RestOutcome:
        self._last_move = None
        before = self._authority.snapshot()
        # Read before the checks, so a halt raised during them refuses the move.
        halt_count = self._recovery.halt_count()
        try:
            self._check_request(source, confirmed, stop_number_seen, before)
            start = self._check_arm()
            moves = self._off_rest(start) is not None
            if moves and source == SHUTDOWN and not confirmed:
                # Decided on the pose just measured, not on what was last known.
                raise Refusal(CONFIRM_PATH)
            plan = self._plan(start) if moves else None
        except Refusal as refusal:
            return RestOutcome(REFUSED, refusal.reason)
        # Recovery takes the arm only if no STOP and no owner change came since
        # the checks began.
        number = self._take_arm(source, before)
        if number is None:
            now = self._authority.snapshot()
            if self._only_ownership_ended(before, now):
                # No check depends on who owned the arm: once more, from nobody.
                before = now
                number = self._take_arm(source, now)
        if number is None:
            return RestOutcome(REFUSED, self._changed_since(before) or _OWNER_CHANGED)
        try:
            admitted = self._authority.snapshot()
            if (admitted.stop_number, admitted.owner_number) != (
                before.stop_number,
                number,
            ):
                return RestOutcome(
                    STOPPED,
                    "a STOP or an owner change arrived as the return to rest began",
                )
            if plan is not None:
                ended = self._move(
                    source, plan, start, admitted, halt_count, should_stop
                )
                if ended is not None:
                    return ended
            problem = self._rest_problem()
            if problem is not None:
                self._motion_state.set(MotionState.HOLD)
                return RestOutcome(FAILED, problem)
            self._motion_state.set(MotionState.REST_HOLD)
            return RestOutcome(AT_REST)
        finally:
            self._give_back(before.owner is Owner.HUMAN, before.stop_number, number)

    def _check_request(
        self,
        source: str,
        confirmed: bool,
        stop_number_seen: int | None,
        before: AuthorityState,
    ) -> None:
        """The conditions that need no reading of the arm; raises Refusal."""
        if source == TYPED and not confirmed:
            raise Refusal(CONFIRM_PATH)
        if source == WAIT and not self._rest.auto_return:
            raise Refusal("the timed return to rest is switched off")
        if stop_number_seen is not None and stop_number_seen != before.stop_number:
            raise Refusal(_stop_number_text(stop_number_seen, before.stop_number))
        if before.stop_level is StopLevel.FAULT:
            raise Refusal(_fault_text(before))
        if not self._arm.torque_on():
            raise Refusal("the motors are off")
        if self._moving():
            raise Refusal("a move is running")

    def _check_arm(self) -> list[float]:
        """The jaw, contact, reading, region and tip conditions; the start pose.

        Raises Refusal. The start pose is the second of the two reads.
        """
        problem = self._jaws_problem()
        if problem is not None:
            raise Refusal(problem)
        if self._contact_hold_active():
            raise Refusal("the arm is holding at a contact")
        _, start = self._two_reads(self._rest.agree_tol_rad)
        low, high = self._rest.region_min, self._rest.region_max
        for joint, at in enumerate(start):
            if not low[joint] <= at <= high[joint]:
                raise Refusal(
                    f"joint index {joint} at {at:.4f} rad is outside the tested "
                    f"recovery region {low[joint]:.4f}..{high[joint]:.4f}"
                )
        blocked = self._recovery.tip_blocked(start, self._last_grasp_region())
        if blocked is not None:
            raise Refusal(blocked or "the gripper tip is blocked")
        return start

    def _plan(self, start: list[float]) -> Plan:
        """The move from start to rest, checked whole; raises Refusal."""
        rest = self._rest
        travel = max(abs(goal - at) for at, goal in zip(start, rest.rest_q))
        duration = planned_duration_s(travel, rest.recovery_speed, rest.min_duration_s)
        count = setpoint_count(duration, self._period_s)
        if count > self._max_setpoints:
            raise Refusal(
                f"the return to rest needs {count} setpoints, the limit is "
                f"{self._max_setpoints}"
            )
        plan = plan_return(
            start, rest.rest_q, self._period_s, rest.recovery_speed, rest.min_duration_s
        )
        try:
            self._recovery.check_path(start, plan[0], self._period_s)
        except Refusal as refusal:
            raise Refusal(f"the path to rest was refused: {refusal.reason}") from None
        return plan

    def _from_measured(self, measured: list[float]) -> list[list[float]]:
        """The same move from the pose the executor itself has just read.

        The executor checks the first setpoint from that pose, and one step at
        the recovery speed is smaller than two reads of a still arm can differ.
        Every setpoint is checked as it is sent, so the path is not checked here.
        """
        rest = self._rest
        return plan_return(
            measured,
            rest.rest_q,
            self._period_s,
            rest.recovery_speed,
            rest.min_duration_s,
        )[0]

    def _take_arm(self, source: str, seen: AuthorityState) -> int | None:
        """Make recovery the owner while the numbers of `seen` stand; its number."""
        return self._authority.set_owner_if(
            Owner.RECOVERY,
            f"return to rest: {source}",
            stop_number=seen.stop_number,
            owner_number=seen.owner_number,
        )

    @staticmethod
    def _only_ownership_ended(before: AuthorityState, now: AuthorityState) -> bool:
        """Whether nothing came since `before` but the end of an ownership.

        So when the agent's ownership ended: on a help request, or with the
        robot connection. A STOP, a fault-stop or a new owner is not that.
        """
        return (
            now.stop_number == before.stop_number
            and now.stop_level is not StopLevel.FAULT
            and now.owner is Owner.NOBODY
        )

    def _changed_since(self, before: AuthorityState) -> str | None:
        """What a STOP or an owner change did since the checks began, else None."""
        now = self._authority.snapshot()
        if now.stop_level is StopLevel.FAULT:
            return _fault_text(now)
        if now.stop_number != before.stop_number:
            return "a STOP was raised while the return to rest was checked"
        if now.owner_number != before.owner_number:
            return _OWNER_CHANGED
        return None

    def _move(
        self,
        source: str,
        plan: Plan,
        start: list[float],
        admitted: AuthorityState,
        halt_count: int,
        should_stop: Callable[[], str | None],
    ) -> RestOutcome | None:
        """Run the planned move once; None when it completed, else how it ended."""
        setpoints, duration = plan
        stopped = should_stop()
        if stopped is not None:
            # Told to stop while it was checked: no move is announced or sent.
            where = f"after 0 of {len(setpoints)} setpoints"
            return RestOutcome(STOPPED, f"{stopped or 'stop requested'} ({where})")
        travel = max(abs(goal - at) for at, goal in zip(start, self._rest.rest_q))
        text = (
            f"return to rest: the arm moves for {duration:.1f} s "
            f"(largest joint travel {travel:.2f} rad)"
        )
        data = {
            "source": source,
            "duration_s": duration,
            "travel_rad": travel,
            "setpoints": len(setpoints),
        }
        self._tell(EVENT_REST, text, data)
        self._moves += 1
        self._motion_state.set(MotionState.MOVING)
        try:
            move = self._executor.run(
                setpoints,
                start,
                self._start_tol_rad,
                self._period_s,
                Owner.RECOVERY,
                self._recovery,
                admitted,
                f"rest-{self._moves}",
                None,
                None,
                should_stop,
                halt_count=halt_count,
                replan=self._from_measured,
            )
        finally:
            # However the move ended, setpoints are no longer being sent.
            self._motion_state.set(MotionState.HOLD)
        self._last_move = move
        if move.status == "completed":
            return None
        where = f"after {move.index} of {len(setpoints)} setpoints"
        if move.status == "stopped":
            return RestOutcome(STOPPED, f"{move.reason} ({where})")
        # Also a move refused before its first setpoint: recovery has taken the
        # arm, so this is no refusal that changed nothing.
        return RestOutcome(FAILED, f"{move.status} {where}: {move.reason}")

    def _give_back(
        self, person_owned: bool, stop_number: int, owner_number: int
    ) -> None:
        """Set the control owner back, unless a STOP or someone else changed it.

        A STOP always leaves nobody the owner, and a takeover keeps its owner.
        """
        self._authority.set_owner_if(
            Owner.HUMAN if person_owned else Owner.NOBODY,
            "the return to rest ended",
            stop_number=stop_number,
            owner_number=owner_number,
        )

    def _report(self, source: str, outcome: RestOutcome) -> None:
        """Log how a request ended and tell the operator."""
        data = {"source": source, "status": outcome.status, "reason": outcome.reason}
        if outcome.status == AT_REST:
            _log.info("return to rest (%s): at rest, motors on", source)
            text = "return to rest: the arm is at rest, motors on"
            self._tell(EVENT_REST, text, data)
            return
        level = logging.INFO if outcome.status == REFUSED else logging.WARNING
        _log.log(
            level, "return to rest (%s) %s: %s", source, outcome.status, outcome.reason
        )
        text = f"return to rest {outcome.status}: {outcome.reason}"
        if outcome.status != REFUSED:
            text += "; the arm holds"
        self._tell(EVENT_REST, text, data)

    # The release, step by step

    def _release(
        self, stop_number_seen: int, should_stop: Callable[[], str | None]
    ) -> str | None:
        state = self._authority.snapshot()
        if stop_number_seen != state.stop_number:
            return _stop_number_text(stop_number_seen, state.stop_number)
        if state.stop_level is StopLevel.FAULT:
            return _fault_text(state)
        stopped = should_stop()
        if stopped is not None:
            return stopped or "stop requested"
        if not self._arm.torque_on():
            return "the motors are already off"
        # The executor starts no move from here until the release is decided.
        with self._executor.claim_idle() as idle:
            if not idle or self._move_running():
                return "a move is running"
            problem = self._jaws_problem() or self._rest_problem()
            if problem is not None:
                return problem
            # The joint reads above are the newest evidence. A STOP that came
            # while they were taken keeps the motors on, and so does whatever
            # told this work to stop, such as the loss of the robot connection.
            if self._authority.stop_number != stop_number_seen:
                return "a STOP was raised while the release was checked"
            stopped = should_stop()
            if stopped is not None:
                return stopped or "stop requested"
            if self._move_running():
                return "a move began while the release was checked"
            try:
                self._arm.release_torque()
            finally:
                if not self._arm.torque_on():
                    self._motion_state.set(MotionState.REST_UNPOWERED)
        return None

    # Shared checks

    def _moving(self) -> bool:
        """Whether a move is running, by the app's account or the executor's."""
        return bool(self._move_running()) or self._executor.progress()[2] > 0

    def _jaws_problem(self) -> str | None:
        """Why the jaws are not proven empty, None when they are."""
        if self._jaws_busy is not None and self._jaws_busy():
            return "a jaw command is running"
        load = self._jaw_state.classify(self._arm)
        if load is JawLoad.EMPTY:
            return None
        why = self._jaw_state.load_reason
        return f"the jaws are not proven empty ({load})" + (f": {why}" if why else "")

    def _read_pose(self) -> list[float]:
        """The pose from one fresh full read; raises ReadFailed."""
        readings = self._arm.read_all()
        if self._history is not None:
            for reading in readings:
                self._history.add(reading)
        pose: list[float | None] = [None] * self._arm.n_joints
        for reading in readings:
            usable = reading.valid and math.isfinite(reading.value)
            if usable and 0 <= reading.joint < len(pose):
                pose[reading.joint] = float(reading.value)
        for joint, value in enumerate(pose):
            if value is None:
                raise ReadFailed(f"joint index {joint} has no valid reading")
        return pose

    def _two_reads(self, agree_tol_rad: float) -> tuple[list[float], list[float]]:
        """Two full reads read_gap_s apart that agree; raises Refusal."""
        gap_s = self._rest.read_gap_s
        try:
            first = self._read_pose()
            self._clock.sleep(gap_s)
            second = self._read_pose()
        except ReadFailed as failed:
            raise Refusal(f"a joint read failed: {failed}") from None
        for joint, (a, b) in enumerate(zip(first, second)):
            if abs(a - b) > agree_tol_rad:
                raise Refusal(
                    f"two joint reads {gap_s:g} s apart disagree: joint index {joint} "
                    f"read {a:.4f} then {b:.4f} rad (tolerance {agree_tol_rad:g} rad)"
                )
        return first, second

    def _off_rest(self, pose: Sequence[float]) -> str | None:
        """Which joint is outside rest_tol_rad of rest, None when none is."""
        tol = self._rest.rest_tol_rad
        for joint, (at, goal) in enumerate(zip(pose, self._rest.rest_q)):
            if abs(at - goal) > tol:
                return (
                    f"joint index {joint} is at {at:.4f} rad, rest is {goal:.4f} rad "
                    f"(tolerance {tol:g} rad)"
                )
        return None

    def _rest_problem(self) -> str | None:
        """Why two fresh reads do not prove the arm still and at rest, else None."""
        try:
            poses = self._two_reads(self._rest.rest_tol_rad)
        except Refusal as refusal:
            return refusal.reason
        for pose in poses:
            problem = self._off_rest(pose)
            if problem is not None:
                return f"the arm is not at rest: {problem}"
        return None

    def _tell(self, kind: str, text: str, data: dict[str, Any]) -> None:
        """Send one operator event; an error in the sender changes nothing else."""
        try:
            self._emit_event(kind, text, data)
        except Exception:
            _log.exception("the event sender raised")
