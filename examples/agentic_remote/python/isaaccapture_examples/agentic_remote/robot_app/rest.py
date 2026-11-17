# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return to rest, the guarded torque release, and the motors on again.

See section 11 of docs/design/claude_agentic_remote_execution.md. No other module
may switch the motors off.

The functional rest is APAI's rest tool (apai.skills.rest_pose.move_to_rest):
up to the ready pose unless the arm is there, the fold onto the rest pose in
measured steps, the proof of rest by measurement, the jaws closed with the
driver's close, the motors off. With a RestRoute the return to rest drives
that route; power_down() is its end.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
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

    # The planned return: per leg, (the leg, its setpoints, its duration in s).
    Plan = list[tuple["Leg", list[list[float]], float]]
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
MOTORS_ALREADY_OFF = "the motors are already off"
_STOP_WHILE_CHECKED = "a STOP was raised while the release was checked"
# The one leg of a return to rest without a RestRoute, and the legs of one.
LEG_STRAIGHT = "to the rest pose"
LEG_UP = "up to the ready pose"
LEG_FOLD = "fold step {step} of {steps}"
# What power_down() says of the jaws before the motors go off, and its kind.
EVENT_JAWS = "jaws"
CLOSING_JAWS = "at rest: closing the jaws with the driver's close"
JAWS_CLOSED = "at rest: the jaws are closed"
JAWS_LEFT = "at rest: the jaws are left as they were: {why}"
# What switches the motors on again, and why it did not.
MOTORS_ON_AGAIN = "the motors are on again at rest"
NOT_ON_AGAIN = "the motors were not switched on again: {why}"
MOTORS_ARE_ON = "the motors are on"


class MotionStateHolder(Protocol):
    """Where the robot app keeps the arm's motion state."""

    def get(self) -> MotionState:
        """The motion state now."""

    def set(self, state: MotionState) -> None:
        """Replace the motion state."""


@dataclass(frozen=True)
class RestRoute:
    """The route of the functional rest and its numbers.

    Every value but home_q is read from apai.skills.rest_pose by
    functional_route(). home_q is the arm profile's ready pose, raised and
    ready for work; None: the arm folds from where it is.
    """

    home_q: tuple[float, ...] | None
    track_tol_rad: float  # measured against a leg's goal after the leg
    home_s: float  # the shortest the leg up to the ready pose takes
    segment_s: float  # the shortest one fold step takes
    # fold_waypoints(start, rest pose): the goals of the fold steps, in order.
    waypoints: Callable[[Sequence[float], Sequence[float]], Sequence[Any]]


def functional_route(home_q: Sequence[float] | None = None) -> RestRoute:
    """The route APAI's rest tool drives, with its own numbers; needs APAI."""
    from apai.skills import rest_pose as functional

    return RestRoute(
        home_q=None if home_q is None else tuple(float(value) for value in home_q),
        track_tol_rad=float(functional.TRACK_TOL_RAD),
        home_s=float(functional.HOME_S),
        segment_s=float(functional.SEGMENT_S),
        waypoints=functional.fold_waypoints,
    )


@dataclass(frozen=True)
class Leg:
    """One measured stretch of a return to rest."""

    name: str
    goal: tuple[float, ...]
    min_duration_s: float
    # The farthest the measured pose may lie from the goal afterwards; None:
    # the last leg, which the proof of rest judges.
    track_tol_rad: float | None = None


class SwitchedArm:
    """An arm port whose motors can be switched on again.

    Every call goes to the driver's port in use. connect() after a torque
    release makes a new port with `build` and connects that one, as APAI's
    LazyArm makes a new backend each time it wakes.
    """

    def __init__(self, port: ArmPort, build: Callable[[], ArmPort]) -> None:
        self._port = port
        self._build = build
        self._connects = 0

    def connect(self) -> None:
        if self._connects:
            if self._port.torque_on():
                raise RuntimeError(MOTORS_ARE_ON)
            self._port = self._build()
        self._connects += 1
        self._port.connect()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._port, name)


def _far(pose: Sequence[float], goal: Sequence[float]) -> float:
    """The largest joint distance between two poses, rad."""
    return max(abs(float(at) - float(to)) for at, to in zip(pose, goal))


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
        route: RestRoute | None = None,
        close_jaws: Callable[[Callable[[], str | None]], str | None] | None = None,
        reads_paused: Callable[[], AbstractContextManager[None]] | None = None,
    ) -> None:
        """emit_event(kind, text, data) sends one operator event.

        last_grasp_region() is the grasp region of the agent's last grasp,
        contact_hold_active() whether the arm holds at a contact, move_running()
        whether a move is being sent to the arm, and jaws_busy() whether a jaw
        command is queued or running (GripperWorker.busy). route is the
        functional rest's route; None: one straight move to the rest pose.
        close_jaws(should_stop) closes the jaws with the driver's close and
        returns None, or why they were left as they were; None: no close.
        reads_paused() is a block inside which no idle joint read begins
        (IdleReader.paused): the torque release runs inside it, so no read
        meets a driver that is switching its motors off.
        """
        self._route = route
        self._close_jaws = close_jaws
        self._reads_paused = reads_paused or contextlib.nullcontext
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
        # The (stop number, owner number) at which power_down() judged the
        # jaws empty; None before, and once the arm is next moved from here.
        self._jaws_judged: tuple[int, int] | None = None
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

    def power_down(
        self,
        stop_number_seen: int,
        should_stop: Callable[[], str | None] | None = None,
        *,
        jaws_empty: bool = False,
        close: bool = True,
    ) -> str | None:
        """The end of the functional rest; why the motors stay on, or None.

        At measured rest the jaws are closed with the driver's close and the
        motors are switched off, in that order. jaws_empty: the jaws were
        found empty before the arm was folded, so they are not judged again;
        else they must be proven empty first, by measurement or the operator's
        word. close False sends the jaws nothing. A close that fails is told
        and the release follows: at rest the jaws hold nothing up. Away from
        measured rest nothing is sent and the motors stay on.
        """
        stop = should_stop or _never_stop
        now = self._authority.snapshot()
        # What refuses the release itself is said before anything is read.
        if stop_number_seen != now.stop_number:
            return _stop_number_text(stop_number_seen, now.stop_number)
        if now.stop_level is StopLevel.FAULT:
            return _fault_text(now)
        stopped = stop()
        if stopped is not None:
            return stopped or "stop requested"
        if not self._arm.torque_on():
            return MOTORS_ALREADY_OFF
        # Jaws this rest has judged and closed are not judged again while no
        # STOP and no owner change came: what its close left is not a load.
        numbers = (now.stop_number, now.owner_number)
        judged = jaws_empty or self._jaws_judged == numbers
        problem = None if judged else self._jaws_problem()
        problem = problem or self._rest_problem()
        if problem is not None:
            _log.warning("the motors stay on: %s", problem)
            return problem
        self._jaws_judged = numbers
        if self._authority.stop_number != stop_number_seen:
            return _STOP_WHILE_CHECKED
        if self._jaws_busy is not None and self._jaws_busy():
            # One that began while the arm was read: the jaws are not as judged.
            self._jaws_judged = None
            return "a jaw command is running"
        if close and self._close_jaws is not None:
            self._tell(EVENT_JAWS, CLOSING_JAWS, {})
            why = self._close_jaws(stop)
            text = JAWS_CLOSED if why is None else JAWS_LEFT.format(why=why)
            self._tell(EVENT_JAWS, text, {"problem": why or ""})
            if self._authority.stop_number != stop_number_seen:
                return _STOP_WHILE_CHECKED
        return self.release(stop_number_seen, stop, jaws_checked=True)

    def release(
        self,
        stop_number_seen: int,
        should_stop: Callable[[], str | None] | None = None,
        *,
        jaws_checked: bool = False,
    ) -> str | None:
        """Switch the motors off at measured rest; why not, or None when done.

        Needs the current stop number, no fault-stop, no move running, the jaws
        proven empty, and two fresh full reads read_gap_s apart within
        rest_tol_rad of rest and of each other. A reason from should_stop, asked
        first and again just before the cut, keeps the motors on. jaws_checked:
        power_down() has judged the jaws before it closed them.
        """
        if not self._busy.acquire(blocking=False):
            reason: str | None = _BUSY
        else:
            try:
                reason = self._release(
                    stop_number_seen, should_stop or _never_stop, jaws_checked
                )
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

    def legs(self, start: Sequence[float]) -> list[Leg]:
        """The legs of a return to rest from a pose away from rest, in order.

        Without a route: one straight move. With one, the functional rest's:
        up to the ready pose unless the arm is within the route's tracking
        tolerance of it, then the fold in the route's steps.
        """
        rest, route = self._rest, self._route
        if route is None:
            park = tuple(float(value) for value in rest.rest_q)
            return [Leg(LEG_STRAIGHT, park, rest.min_duration_s)]
        home = route.home_q
        if home is not None and _far(start, home) > route.track_tol_rad:
            up = Leg(LEG_UP, home, route.home_s, route.track_tol_rad)
            return [up, *self._fold(home)]
        return self._fold(start)

    def _fold(self, start: Sequence[float]) -> list[Leg]:
        """The fold's legs from a pose: the route's steps, the last onto rest."""
        route = self._route
        assert route is not None
        park = tuple(float(value) for value in self._rest.rest_q)
        goals = [
            tuple(float(value) for value in goal)
            for goal in route.waypoints(start, park)
        ]
        found: list[Leg] = []
        for step, goal in enumerate(goals, start=1):
            name = LEG_FOLD.format(step=step, steps=len(goals))
            if step == len(goals):
                found.append(Leg(name, park, route.segment_s))
            else:
                found.append(Leg(name, goal, route.segment_s, route.track_tol_rad))
        return found

    def _plan(self, start: list[float], legs: Sequence[Leg] | None = None) -> Plan:
        """Every leg from start to rest, checked whole; raises Refusal.

        legs None plans legs(start).
        """
        rest = self._rest
        plan: Plan = []
        at = list(start)
        for leg in self.legs(start) if legs is None else legs:
            travel = _far(at, leg.goal)
            duration = planned_duration_s(
                travel, rest.recovery_speed, leg.min_duration_s
            )
            count = setpoint_count(duration, self._period_s)
            if count > self._max_setpoints:
                raise Refusal(
                    f"the return to rest needs {count} setpoints, the limit is "
                    f"{self._max_setpoints}"
                )
            rows, duration = self._leg_rows(at, leg)
            try:
                self._recovery.check_path(at, rows, self._period_s)
            except Refusal as refusal:
                raise Refusal(
                    f"the path to rest was refused: {refusal.reason}"
                ) from None
            plan.append((leg, rows, duration))
            at = list(leg.goal)
        return plan

    def _leg_rows(
        self, start: Sequence[float], leg: Leg
    ) -> tuple[list[list[float]], float]:
        """One leg's setpoints from a pose, and their duration."""
        return plan_return(
            start,
            leg.goal,
            self._period_s,
            self._rest.recovery_speed,
            leg.min_duration_s,
        )

    def _from_measured(self, leg: Leg) -> Callable[[list[float]], list[list[float]]]:
        """The same leg from the pose the executor itself has just read.

        The executor checks the first setpoint from that pose, and one step at
        the recovery speed is smaller than two reads of a still arm can differ.
        Every setpoint is checked as it is sent, so the path is not checked here.
        """
        return lambda measured: self._leg_rows(measured, leg)[0]

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
        """Run the planned legs once; None when all completed, else how it ended.

        Each leg starts from the pose measured after the one before, and that
        pose must lie within the leg's tracking tolerance of its goal. After
        the leg up to the ready pose the fold is planned and checked anew from
        the measured pose, as the functional rest does.
        """
        at = list(start)
        legs = list(plan)
        while legs:
            leg, setpoints, duration = legs.pop(0)
            named = "" if self._route is None else f"{leg.name}: "
            stopped = should_stop()
            if stopped is not None:
                # Told to stop while it was checked: no move is announced or sent.
                where = f"{named}after 0 of {len(setpoints)} setpoints"
                return RestOutcome(STOPPED, f"{stopped or 'stop requested'} ({where})")
            travel = _far(at, leg.goal)
            text = (
                f"return to rest: {named}the arm moves for {duration:.1f} s "
                f"(largest joint travel {travel:.2f} rad)"
            )
            data = {
                "source": source,
                "duration_s": duration,
                "travel_rad": travel,
                "setpoints": len(setpoints),
            }
            if self._route is not None:
                data["leg"] = leg.name
            self._tell(EVENT_REST, text, data)
            self._moves += 1
            self._jaws_judged = None
            self._motion_state.set(MotionState.MOVING)
            try:
                move = self._executor.run(
                    setpoints,
                    at,
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
                    replan=self._from_measured(leg),
                )
            finally:
                # However the move ended, setpoints are no longer being sent.
                self._motion_state.set(MotionState.HOLD)
            self._last_move = move
            where = f"{named}after {move.index} of {len(setpoints)} setpoints"
            if move.status == "stopped":
                return RestOutcome(STOPPED, f"{move.reason} ({where})")
            if move.status != "completed":
                # Also a move refused before its first setpoint: recovery has
                # taken the arm, so this is no refusal that changed nothing.
                return RestOutcome(FAILED, f"{move.status} {where}: {move.reason}")
            if not legs:
                return None
            try:
                at = list(move.end_q) if move.end_q is not None else self._read_pose()
            except ReadFailed as failed:
                return RestOutcome(FAILED, f"{leg.name}: a joint read failed: {failed}")
            off = _far(at, leg.goal)
            if leg.track_tol_rad is not None and off > leg.track_tol_rad:
                return RestOutcome(
                    FAILED,
                    f"{leg.name} did not track: the arm is {off:.3f} rad from its "
                    f"goal (tolerance {leg.track_tol_rad:g} rad)",
                )
            if leg.name == LEG_UP:
                try:
                    legs = self._plan(at, self._fold(at))
                except Refusal as refusal:
                    return RestOutcome(
                        FAILED,
                        f"the fold was refused before it started: {refusal.reason}",
                    )
        return None

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
        self,
        stop_number_seen: int,
        should_stop: Callable[[], str | None],
        jaws_checked: bool = False,
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
            return MOTORS_ALREADY_OFF
        # The executor starts no move from here until the release is decided.
        with self._executor.claim_idle() as idle:
            if not idle or self._move_running():
                return "a move is running"
            problem = None if jaws_checked else self._jaws_problem()
            problem = problem or self._rest_problem()
            if problem is not None:
                return problem
            # The joint reads above are the newest evidence. A STOP that came
            # while they were taken keeps the motors on, and so does whatever
            # told this work to stop, such as the loss of the robot connection.
            if self._authority.stop_number != stop_number_seen:
                return _STOP_WHILE_CHECKED
            stopped = should_stop()
            if stopped is not None:
                return stopped or "stop requested"
            if self._move_running():
                return "a move began while the release was checked"
            # An idle joint read that met the driver while it switches its
            # motors off would count as lost joint feedback: a fault-stop.
            with self._reads_paused():
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
