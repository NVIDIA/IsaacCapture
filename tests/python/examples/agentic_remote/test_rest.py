# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the return to rest and the torque release in robot_app.rest."""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import Event, GraspRegion
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
from isaaccapture_examples.agentic_remote.robot_app.gripper import JawState
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    CheckerSet,
    GripperOutcome,
    JointReading,
    MotionState,
    MoveOutcome,
    Owner,
    RestOutcome,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.rest import (
    AT_REST,
    FAILED,
    REFUSED,
    STOPPED,
    ReturnToRest,
    plan_return,
    planned_duration_s,
    setpoint_count,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm
from isaaccapture_examples.agentic_remote.robot_app.state import JointHistory

N = 6
PERIOD = 0.02
READ_ALL_S = N * 0.008  # the simulated arm's time for one read of every joint
REST = [0.0] * N
RAISED = [0.5, -0.3, 0.2, 0.0, 0.1, -0.4]  # largest joint travel to rest: 0.5 rad
RAISED_SETPOINTS = 521  # 1.875 * 0.5 / (0.9 * 0.1) s, one setpoint per 20 ms
REGION = GraspRegion((0.3, 0.0), 0.05, 0.01, None)
BUSY = "a return to rest or a release is already running"
MOVE_EVENT = "return to rest: the arm moves for"


def raised(joint: int, value: float) -> list[float]:
    """The rest pose with one joint elsewhere."""
    q = list(REST)
    q[joint] = value
    return q


def off_rest(pose: Sequence[float]) -> float:
    return max(abs(at - goal) for at, goal in zip(pose, REST))


class Holder:
    """The arm's motion state, as the robot app keeps it."""

    def __init__(self, state: MotionState) -> None:
        self.states = [state]

    def get(self) -> MotionState:
        return self.states[-1]

    def set(self, state: MotionState) -> None:
        self.states.append(state)


class Rig:
    """A ReturnToRest on the simulated arm, with the real executor and jaw state."""

    def __init__(
        self,
        arm: SimArm,
        checkers: CheckerSet,
        config: RobotAppConfig,
        clock: FakeClock,
        patch: pytest.MonkeyPatch,
        *,
        connect: bool = True,
        reference: bool = True,
    ) -> None:
        if connect:
            arm.connect()
        if reference:
            arm.gripper_reference()
        self.arm = arm
        self.checkers = checkers
        self.recovery = checkers.recovery
        self.config = config
        self.clock = clock
        self.patch = patch
        self.authority = Authority()
        self.history = JointHistory(config.n_joints, config.state.history_s)
        self.executor = MoveExecutor(
            arm,
            self.authority,
            self.history,
            config.executor,
            clock,
            first_wait_s=config.permits.first_wait_s,
        )
        self.jaw_state = JawState(config.gripper, clock)
        self.motion = Holder(MotionState.HOLD)
        self.events: list[tuple[str, str]] = []  # (kind, text)
        self.event_data: list[dict] = []
        self.on_event: Callable[[str, str], None] | None = None
        self.region: GraspRegion | None = None
        self.contact_hold = False
        self.moving = False
        self.jaws_busy = False
        self.rest = ReturnToRest(
            arm,
            self.authority,
            checkers,
            self.executor,
            self.jaw_state,
            self.history,
            config,
            clock,
            self.emit,
            self.motion,
            lambda: self.region,
            lambda: self.contact_hold,
            lambda: self.moving,
            jaws_busy=lambda: self.jaws_busy,
        )

    def emit(self, kind: str, text: str, data: dict) -> None:
        self.events.append((kind, text))
        self.event_data.append(data)
        if self.on_event is not None:
            self.on_event(kind, text)

    def request(
        self, source: str = "typed", confirmed: bool = True, **more: object
    ) -> RestOutcome:
        return self.rest.request(source, confirmed, **more)

    def release(self) -> str | None:
        """Ask for the release, quoting the current stop number."""
        return self.rest.release(self.authority.stop_number)

    def sent(self) -> list[list[float]]:
        return [q for q, _ in self.arm.sent]

    def state(self) -> dict[str, object]:
        """Everything a refused request or release must leave as it was."""
        return {
            "authority": self.authority.snapshot(),
            "motion": list(self.motion.states),
            "sent": len(self.arm.sent),
            "holds": len(self.arm.holds),
            "released": len(self.arm.released_at),
            "torque": self.arm.torque_on(),
            "jaw commands": len(self.arm.jaw_commands),
            "light holds": len(self.arm.light_holds),
            "pose": self.arm.pose(),
            "in motion": self.recovery.in_motion,
            "last move": self.rest.last_move,
        }

    def on_read_all(self, call: int, action: Callable[[], object]) -> None:
        """Run action() just before the arm's call-th full read from now on."""
        real = self.arm.read_all
        calls: list[int] = []

        def read_all() -> list[JointReading]:
            calls.append(1)
            if len(calls) == call:
                action()
            return real()

        self.patch.setattr(self.arm, "read_all", read_all)

    def after_sends(self, count: int, action: Callable[[], object]) -> None:
        """Run action() once the arm was sent its count-th setpoint."""
        real = self.arm.send_target

        def send(q: Sequence[float]) -> None:
            real(q)
            if len(self.arm.sent) == count:
                action()

        self.patch.setattr(self.arm, "send_target", send)

    def spy_run(self, after: Callable[[], object] | None = None) -> list[tuple]:
        """The executor's run() calls as (args, kwargs); after() follows each."""
        real = self.executor.run
        calls: list[tuple] = []

        def run(*args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            outcome = real(*args, **kwargs)
            if after is not None:
                after()
            return outcome

        self.patch.setattr(self.executor, "run", run)
        return calls

    def stop(self) -> None:
        self.authority.raise_stop("operator", "console")

    def close_on_object(self) -> None:
        """Close the jaws onto an object: they stall at half open."""
        self.arm.jaw_object(0.5)
        self.arm.gripper_command(0.0, 0.5, "close_until_contact", threading.Event())


MakeRig = Callable[..., Rig]


@pytest.fixture
def make_rig(sim_arm, checkers, config, clock, monkeypatch) -> MakeRig:
    def make(**options: object) -> Rig:
        chosen = options.pop("checkers", checkers)
        settings = options.pop("config", config)
        return Rig(sim_arm, chosen, settings, clock, monkeypatch, **options)

    return make


@pytest.fixture
def rig(make_rig) -> Rig:
    """The arm at rest, motors on, jaws referenced and open, nobody the owner."""
    return make_rig()


@pytest.fixture
def up(rig) -> Rig:
    """The same with the arm holding at RAISED."""
    rig.arm.set_pose(RAISED)
    return rig


def with_recovery(
    checkers: CheckerSet, clock: FakeClock, **limits: object
) -> CheckerSet:
    """The checkers with a recovery checker whose limit set is changed."""
    limit_set = dataclasses.replace(RobotAppConfig.default().limits.recovery, **limits)
    recovery = BasicChecker(limit_set, clock, watchdog_s=None)
    return CheckerSet(agent=checkers.agent, human=checkers.human, recovery=recovery)


def refused(
    rig: Rig,
    match: str,
    *,
    ignore: Sequence[str] = (),
    reads: bool = True,
    **request: object,
) -> RestOutcome:
    """Request a return; it must be refused for `match` and change nothing.

    reads=False: it must be refused before the jaws or the joints are read.
    """
    before = rig.state()
    events = len(rig.events)
    t_before = rig.clock.monotonic()
    outcome = rig.request(**request)
    assert outcome.status == REFUSED
    assert match in outcome.reason
    if not reads:
        # Every joint reading costs clock time on the simulated arm.
        assert rig.clock.monotonic() == t_before
    after = rig.state()
    for key in ignore:
        before.pop(key)
        after.pop(key)
    assert after == before
    assert rig.events[events:] == [
        ("rest", f"return to rest refused: {outcome.reason}")
    ]
    return outcome


# The planned move


def test_planned_duration_follows_the_formula():
    assert planned_duration_s(0.5, 0.1, 2.0) == pytest.approx(1.875 * 0.5 / 0.09)
    # The design's "about 21 s per radian" at the recovery speed of 0.1 rad/s.
    assert planned_duration_s(1.0, 0.1, 2.0) == pytest.approx(20.8333, abs=1e-4)
    assert planned_duration_s(1.0, 0.2, 2.0) == pytest.approx(10.4167, abs=1e-4)


def test_planned_duration_is_never_below_the_minimum():
    assert planned_duration_s(0.06, 0.1, 2.0) == 2.0
    assert planned_duration_s(0.0, 0.1, 2.0) == 2.0
    assert planned_duration_s(0.06, 0.1, 0.5) == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("duration", "count"),
    [(2.0, 100), (10.0, 500), (10.001, 501), (10.4167, 521), (0.001, 1)],
)
def test_setpoint_count_is_one_per_period_rounded_up(duration, count):
    assert setpoint_count(duration, PERIOD) == count


def test_plan_is_a_straight_line_on_the_minimum_jerk_law():
    start = [0.5, -0.25, 0.0]
    goal = [0.0, 0.25, 0.0]
    rows, duration = plan_return(start, goal, PERIOD, 0.1, 2.0)
    count = len(rows)
    assert count == RAISED_SETPOINTS
    assert duration == pytest.approx(count * PERIOD)
    assert rows[-1] == goal
    for index, row in enumerate(rows, start=1):
        t = index / count
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        expected = [a + (b - a) * s for a, b in zip(start, goal)]
        assert row == pytest.approx(expected, abs=1e-12)


def test_plan_peaks_at_nine_tenths_of_the_recovery_speed():
    rows, _ = plan_return([1.0, 0.0], [0.0, 0.0], PERIOD, 0.1, 2.0)
    path = [1.0] + [row[0] for row in rows]
    speeds = [(a - b) / PERIOD for a, b in zip(path, path[1:])]
    assert all(speed > 0.0 for speed in speeds)
    assert max(speeds) <= 0.09 + 1e-9
    assert max(speeds) > 0.0899
    # The move starts and ends slowly.
    assert speeds[0] < 0.001 and speeds[-1] < 0.001


def test_plan_of_a_short_move_takes_the_minimum_duration():
    rows, duration = plan_return([0.06], [0.0], PERIOD, 0.1, 2.0)
    assert len(rows) == 100
    assert duration == pytest.approx(2.0)
    assert rows[-1] == [0.0]


def test_plan_accepts_tuples_and_returns_lists_of_floats():
    rows, _ = plan_return((1, 0), (0, 0), PERIOD, 0.1, 2.0)
    assert all(isinstance(row, list) for row in rows)
    assert all(isinstance(value, float) for row in rows for value in row)


# A return that reaches rest


def test_returns_to_rest_from_a_raised_pose(up):
    before = up.authority.snapshot()
    outcome = up.request()
    assert outcome == RestOutcome(AT_REST)
    assert off_rest(up.arm.pose()) < 0.005
    assert up.sent()[-1] == REST
    assert len(up.sent()) == RAISED_SETPOINTS
    assert up.motion.states == [
        MotionState.HOLD,
        MotionState.MOVING,
        MotionState.HOLD,
        MotionState.REST_HOLD,
    ]
    # Powered at rest, nothing released, the jaws left alone.
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    assert up.arm.holds == []
    assert up.arm.jaw_commands == []
    assert up.arm.light_holds == []
    # Nobody owns the arm; the two owner changes raised the owner number.
    after = up.authority.snapshot()
    assert after.owner is Owner.NOBODY
    assert after.owner_number == before.owner_number + 2
    assert after.stop_number == before.stop_number
    assert after.stop_level is StopLevel.NONE
    assert not up.recovery.in_motion
    assert up.executor.progress() == ("", 0, 0)


def test_the_setpoints_sent_are_the_planned_ones(up):
    rows, _ = plan_return(RAISED, REST, PERIOD, 0.1, 2.0)
    up.request()
    assert up.sent() == rows


def test_the_operator_is_told_the_planned_duration_and_the_end(up):
    up.request()
    assert up.events == [
        ("rest", f"{MOVE_EVENT} 10.4 s (largest joint travel 0.50 rad)"),
        ("rest", "return to rest: the arm is at rest, motors on"),
    ]


def test_the_events_carry_their_facts_as_data(up):
    up.request("wait")
    moves, ended = up.event_data
    assert moves == {
        "source": "wait",
        "duration_s": pytest.approx(RAISED_SETPOINTS * PERIOD),
        "travel_rad": pytest.approx(0.5),
        "setpoints": RAISED_SETPOINTS,
    }
    assert ended == {"source": "wait", "status": "at_rest", "reason": ""}
    up.close_on_object()
    outcome = up.request("shutdown")
    assert up.event_data[-1] == {
        "source": "shutdown",
        "status": "refused",
        "reason": outcome.reason,
    }
    # Every event can go out as a link message.
    for kind, text in up.events:
        Event(seq=1, kind=kind, text=text, data=up.event_data[0]).to_payload()
    for data in up.event_data:
        Event(seq=1, kind="rest", text="", data=data).to_payload()


def test_the_move_is_given_to_the_executor_for_the_recovery_owner(up):
    calls = up.spy_run()
    up.recovery.halt("an earlier stop")
    up.region = REGION
    up.request()
    assert len(calls) == 1
    args, kwargs = calls[0]
    setpoints, start_q, start_tol, period_s, owner, checker, admitted = args[:7]
    assert len(setpoints) == RAISED_SETPOINTS
    assert start_q == pytest.approx(RAISED)
    assert start_tol == pytest.approx(up.config.rest.agree_tol_rad + 0.005)
    assert period_s == PERIOD
    assert owner is Owner.RECOVERY
    assert checker is up.recovery
    assert admitted.owner is Owner.RECOVERY
    assert admitted.owner_number == 1
    # No permit gate and no grasp region: neither applies to the recovery owner.
    assert args[8:10] == (None, None)
    # The executor is given the way to plan the move from its own start read.
    replan = kwargs.pop("replan")
    assert kwargs == {"halt_count": 1}
    assert replan(list(RAISED)) == setpoints


def test_the_start_tolerance_never_exceeds_the_executors_limit(make_rig, config):
    narrow = dataclasses.replace(config.executor, max_start_tol_rad=0.01)
    rig = make_rig(config=dataclasses.replace(config, executor=narrow))
    rig.arm.set_pose(RAISED)
    calls = rig.spy_run()
    assert rig.request().status == AT_REST
    assert calls[0][0][2] == 0.01


def test_recovery_owns_the_arm_while_it_returns(up):
    up.authority.set_owner(Owner.AGENT, "new task")
    agent = up.authority.snapshot()
    seen: list[object] = []

    def look() -> None:
        now = up.authority.snapshot()
        seen.append(now.owner)
        seen.append(up.motion.get())
        seen.append(up.recovery.in_motion)
        # A command the agent sends now quotes an owner number that is over.
        seen.append(
            up.authority.admit(
                agent.session_id, agent.owner_number, agent.stop_number, Owner.AGENT
            )
        )

    up.after_sends(5, look)
    assert up.request().status == AT_REST
    assert seen == [
        Owner.RECOVERY,
        MotionState.MOVING,
        True,
        "control owner is recovery, not agent",
    ]


@pytest.mark.parametrize(
    ("owner", "afterwards"),
    [
        (Owner.NOBODY, Owner.NOBODY),
        (Owner.AGENT, Owner.NOBODY),
        (Owner.HUMAN, Owner.HUMAN),
    ],
)
def test_the_owner_afterwards_is_the_person_or_nobody(up, owner, afterwards):
    if owner is not Owner.NOBODY:
        up.authority.set_owner(owner, "test")
    number = up.authority.owner_number
    assert up.request().status == AT_REST
    assert up.authority.owner is afterwards
    assert up.authority.owner_number == number + 2
    assert up.arm.torque_on()


@pytest.mark.parametrize("source", ["wait", "takeover_end"])
def test_a_request_no_operator_typed_needs_no_confirmation(up, source):
    assert up.request(source, False).status == AT_REST
    assert up.arm.released_at == []


def test_a_confirmed_shutdown_moves_the_arm_to_rest(up):
    assert up.request("shutdown", True).status == AT_REST
    assert len(up.arm.sent) == RAISED_SETPOINTS
    assert up.arm.released_at == []


def test_an_unknown_source_is_an_error_and_changes_nothing(up):
    before = up.state()
    with pytest.raises(ValueError, match=r"unknown return-to-rest source 'home'"):
        up.request("home")
    assert up.state() == before
    assert up.events == []


def test_the_start_pose_is_the_second_of_the_two_reads(up):
    calls = up.spy_run()
    drifted = list(RAISED)
    drifted[0] += 0.015  # inside the agreement tolerance
    up.on_read_all(2, lambda: up.arm.set_pose(drifted))
    assert up.request().status == AT_REST
    assert calls[0][0][1] == pytest.approx(drifted)


def test_the_move_is_planned_from_the_pose_the_executor_reads(up):
    # Between the reads of the checks and the executor's own the arm is read
    # 0.004 rad elsewhere: two steps at the recovery speed.
    read = list(RAISED)
    read[0] -= 0.004
    up.on_read_all(3, lambda: up.arm.set_pose(read))
    assert up.request().status == AT_REST
    rows, _ = plan_return(read, REST, PERIOD, 0.1, 2.0)
    assert up.sent() == rows
    assert abs(up.sent()[0][0] - read[0]) < 0.002


def test_a_pose_the_executor_reads_too_far_from_the_checked_one_is_refused(up):
    read = list(RAISED)
    read[0] -= 0.03  # outside the start tolerance of 0.025 rad
    up.on_read_all(3, lambda: up.arm.set_pose(read))
    outcome = up.request()
    assert outcome.status == FAILED
    assert outcome.reason.startswith("refused after 0 of 521 setpoints: start pose: ")
    assert up.arm.sent == []


def test_the_two_condition_reads_are_the_read_gap_apart(up):
    times: list[float] = []
    up.on_read_all(1, lambda: times.append(up.clock.monotonic()))
    up.on_read_all(2, lambda: times.append(up.clock.monotonic()))
    up.request()
    gap = times[1] - (times[0] + READ_ALL_S)
    assert gap == pytest.approx(up.config.rest.read_gap_s)


def test_the_joint_readings_go_to_the_history(rig):
    rig.request()
    newest = rig.history.latest()
    assert all(reading is not None for reading in newest)
    # The newest reading of each joint comes from the last of the four reads.
    assert all(
        rig.clock.monotonic() - reading.t_end <= READ_ALL_S for reading in newest
    )


@pytest.mark.parametrize(
    ("start", "count"),
    [
        (RAISED, 521),
        ([-1.2, 0.7, 0.0, 0.0, 0.0, 0.3], 1250),
        (raised(2, 2.4), 2500),
        (raised(4, -0.06), 100),  # the minimum duration of 2 s
    ],
)
def test_duration_follows_the_formula_and_speed_stays_under_the_limit(
    rig, start, count
):
    rest = rig.config.rest
    travel = off_rest(start)
    formula = max(rest.min_duration_s, 1.875 * travel / (0.9 * rest.recovery_speed))
    assert count == math.ceil(formula / PERIOD - 1e-9)
    rig.arm.set_pose(start)
    assert rig.request().status == AT_REST
    sent = rig.arm.sent
    assert len(sent) == count
    assert rig.events[0][1].startswith(f"{MOVE_EVENT} {count * PERIOD:.1f} s ")
    # One setpoint per period, so the move lasts the planned time.
    times = [t for _, t in sent]
    assert all(
        later - earlier == pytest.approx(PERIOD, abs=1e-6)
        for earlier, later in zip(times, times[1:])
    )
    assert times[-1] - times[0] == pytest.approx((count - 1) * PERIOD, abs=1e-6)
    # No joint is ever asked to move faster than 0.9 of the recovery speed.
    path = [list(start)] + [q for q, _ in sent]
    fastest = max(
        abs(b - a) / PERIOD
        for q_prev, q_next in zip(path, path[1:])
        for a, b in zip(q_prev, q_next)
    )
    assert fastest <= 0.9 * rest.recovery_speed + 1e-9
    assert fastest < rest.recovery_speed
    if formula > rest.min_duration_s:
        assert fastest > 0.89 * rest.recovery_speed


# Already at rest


def test_already_at_rest_needs_no_move(rig):
    calls = rig.spy_run()
    outcome = rig.request()
    assert outcome == RestOutcome(AT_REST)
    assert calls == []
    assert rig.arm.sent == []
    assert rig.motion.states == [MotionState.HOLD, MotionState.REST_HOLD]
    assert rig.authority.owner is Owner.NOBODY
    assert rig.authority.owner_number == 2
    assert rig.arm.torque_on()
    assert rig.arm.released_at == []
    assert rig.rest.last_move is None
    assert rig.events == [("rest", "return to rest: the arm is at rest, motors on")]


def test_at_rest_a_person_stays_the_owner(rig):
    rig.authority.set_owner(Owner.HUMAN, "takeover")
    assert rig.request("takeover_end").status == AT_REST
    assert rig.authority.owner is Owner.HUMAN
    assert rig.arm.sent == []


def test_inside_the_rest_tolerance_counts_as_at_rest(rig):
    rig.arm.set_pose(raised(1, 0.04))
    assert rig.request().status == AT_REST
    assert rig.arm.sent == []
    assert rig.motion.get() is MotionState.REST_HOLD


def test_just_outside_the_rest_tolerance_the_arm_is_moved(rig):
    rig.arm.set_pose(raised(1, 0.06))
    assert rig.request().status == AT_REST
    assert len(rig.arm.sent) == 100
    assert off_rest(rig.arm.pose()) < 0.005


def test_at_rest_the_conditions_still_apply(rig):
    rig.close_on_object()
    refused(rig, "(loaded)")


def test_at_rest_is_claimed_only_from_two_reads_that_agree(rig):
    # Read 3 is the first of the two that prove rest; the arm moves before read 4.
    rig.on_read_all(4, lambda: rig.arm.set_pose(raised(0, 0.2)))
    outcome = rig.request()
    assert outcome.status == FAILED
    assert "disagree: joint index 0" in outcome.reason
    assert rig.motion.get() is MotionState.HOLD
    assert rig.authority.owner is Owner.NOBODY
    assert rig.arm.torque_on()


# Refused: nothing changes and nothing is sent


def test_refused_typed_request_without_the_confirmation(up):
    refused(
        up, "confirm the path is clear", reads=False, source="typed", confirmed=False
    )


def test_refused_shutdown_that_has_to_move_the_arm_without_the_confirmation(up):
    # Decided on the measured pose: the refusal comes after the joint reads.
    refused(up, "confirm the path is clear", source="shutdown", confirmed=False)
    assert up.arm.sent == []


@pytest.mark.parametrize("off", [0.0, 0.04])
def test_a_shutdown_measured_at_rest_needs_no_confirmation(rig, off):
    rig.arm.set_pose(raised(1, off))
    assert rig.request("shutdown", False).status == AT_REST
    assert rig.arm.sent == []


def test_just_outside_the_rest_tolerance_a_shutdown_needs_the_confirmation(rig):
    rig.arm.set_pose(raised(1, 0.06))
    refused(rig, "confirm the path is clear", source="shutdown", confirmed=False)
    assert rig.request("shutdown", True).status == AT_REST
    assert len(rig.arm.sent) == 100


def test_refused_timed_return_when_it_is_switched_off(make_rig, config):
    off = dataclasses.replace(config.rest, auto_return=False)
    rig = make_rig(config=dataclasses.replace(config, rest=off))
    rig.arm.set_pose(RAISED)
    refused(rig, "the timed return to rest is switched off", reads=False, source="wait")
    # The switch is for the timed return only.
    assert rig.request("typed", True).status == AT_REST


def test_refused_with_a_stop_number_that_is_over(up):
    seen = up.authority.stop_number
    up.stop()
    stale = "stop number 0 does not match the current 1"
    refused(up, stale, reads=False, stop_number_seen=seen)
    ahead = "stop number 2 does not match the current 1"
    refused(up, ahead, reads=False, stop_number_seen=2)


def test_the_current_stop_number_is_accepted(up):
    up.stop()
    outcome = up.request("wait", stop_number_seen=up.authority.stop_number)
    assert outcome.status == AT_REST


def test_refused_with_a_fault_latched(up):
    up.authority.raise_fault("motor 3 reports a fault")
    refused(up, "fault-stop latched: motor 3 reports a fault", reads=False)


def test_refused_with_the_motors_off(make_rig):
    rig = make_rig(connect=False, reference=False)
    rig.arm.set_pose(RAISED)
    refused(rig, "the motors are off", reads=False)


def test_refused_while_a_move_is_running(up):
    up.moving = True
    refused(up, "a move is running", reads=False)


def test_refused_while_the_executor_runs_another_move(rig):
    rig.authority.set_owner(Owner.AGENT, "new task")
    rig.checkers.agent.feed_watchdog()
    answers: list[RestOutcome] = []
    rig.after_sends(3, lambda: answers.append(rig.request()))
    path = [raised(0, 0.005 * step) for step in range(1, 11)]
    admitted = rig.authority.snapshot()
    move = rig.executor.run(
        path, REST, 0.02, PERIOD, Owner.AGENT, rig.checkers.agent, admitted, "move-1"
    )
    assert move.status == "completed"
    assert answers == [RestOutcome(REFUSED, "a move is running")]
    assert rig.authority.snapshot() == admitted


def test_refused_with_loaded_jaws(up):
    up.close_on_object()
    outcome = refused(up, "the jaws are not proven empty (loaded)")
    assert "the jaws are at 0.500, not at their target 0.000" in outcome.reason


def test_refused_while_the_jaws_may_be_holding(up):
    up.jaw_state.note_command_result(0.0, "plain", GripperOutcome("stalled", 0.5))
    outcome = refused(up, "the jaws are not proven empty (loaded)")
    assert "may be holding" in outcome.reason


def test_refused_with_unreferenced_jaws(make_rig):
    rig = make_rig(reference=False)
    rig.arm.set_pose(RAISED)
    outcome = refused(rig, "the jaws are not proven empty (unknown)")
    assert "the jaws are not referenced" in outcome.reason


def test_refused_when_a_jaw_read_fails(up):
    up.arm.fail_gripper_reads(1)
    outcome = refused(up, "the jaws are not proven empty (unknown)")
    assert "a jaw read failed" in outcome.reason


def test_refused_while_a_jaw_command_runs(up):
    up.jaws_busy = True
    refused(up, "a jaw command is running", reads=False)
    up.jaws_busy = False
    assert up.request().status == AT_REST


def test_without_a_jaw_worker_the_jaw_readings_alone_decide(rig):
    rest = ReturnToRest(
        rig.arm,
        rig.authority,
        rig.checkers,
        rig.executor,
        rig.jaw_state,
        rig.history,
        rig.config,
        rig.clock,
        rig.emit,
        rig.motion,
        lambda: None,
        lambda: False,
        lambda: False,
    )
    assert rest.request("wait").status == AT_REST
    assert rest.release(rig.authority.stop_number) is None


def test_the_jaws_are_judged_and_never_moved(up):
    up.close_on_object()
    commands = len(up.arm.jaw_commands)
    up.request()
    assert len(up.arm.jaw_commands) == commands
    assert up.arm.gripper_read() == 0.5


def test_refused_while_the_arm_holds_at_a_contact(up):
    reads: list[int] = []
    up.on_read_all(1, lambda: reads.append(1))
    up.contact_hold = True
    refused(up, "the arm is holding at a contact")
    assert reads == []
    up.contact_hold = False
    assert up.request().status == AT_REST


@pytest.mark.parametrize("read", [1, 2])
def test_refused_when_a_joint_read_fails(up, read):
    up.on_read_all(read, lambda: up.arm.fail_reads(2, 1))
    refused(up, "a joint read failed: joint index 2")


def test_refused_when_a_reading_is_not_valid(up):
    real = up.arm.read_all

    def read_all() -> list[JointReading]:
        readings = real()
        readings[3] = dataclasses.replace(readings[3], valid=False)
        return readings

    up.patch.setattr(up.arm, "read_all", read_all)
    refused(up, "a joint read failed: joint index 3 has no valid reading")


def test_refused_when_a_reading_is_not_finite(up):
    real = up.arm.read_all

    def read_all() -> list[JointReading]:
        readings = real()
        readings[5] = dataclasses.replace(readings[5], value=math.nan)
        return readings

    up.patch.setattr(up.arm, "read_all", read_all)
    refused(up, "a joint read failed: joint index 5 has no valid reading")


def test_refused_when_the_two_reads_disagree(up):
    moved = list(RAISED)
    moved[1] -= 0.03  # more than the 0.02 rad the two reads may differ by
    up.on_read_all(2, lambda: up.arm.set_pose(moved))
    outcome = refused(up, "disagree: joint index 1", ignore=["pose"])
    assert "read -0.3000 then -0.3300 rad (tolerance 0.02 rad)" in outcome.reason


@pytest.mark.parametrize("value", [2.8, -2.8])
def test_refused_outside_the_recovery_region(rig, value):
    # Inside the recovery joint limits, outside the tested region.
    rig.arm.set_pose(raised(3, value))
    outcome = refused(rig, "is outside the tested recovery region -2.5000..2.5000")
    assert f"joint index 3 at {value:.4f} rad" in outcome.reason


@pytest.mark.parametrize("value", [0.5, -0.5])
def test_the_edge_of_the_recovery_region_is_inside(make_rig, config, value):
    box = dataclasses.replace(
        config.rest, region_min=(-0.5,) * N, region_max=(0.5,) * N
    )
    rig = make_rig(config=dataclasses.replace(config, rest=box))
    rig.arm.set_pose(raised(3, value))
    assert rig.request().status == AT_REST
    rig.arm.set_pose(raised(3, value * 1.01))
    outside = "is outside the tested recovery region -0.5000..0.5000"
    refused(rig, outside, ignore=["last move"])


def test_refused_when_the_gripper_tip_is_blocked(up):
    asked: list[tuple] = []

    def hook(q: list[float], region: GraspRegion | None) -> str | None:
        asked.append((q, region))
        return "the gripper tip is inside the last grasp region"

    up.region = REGION
    up.recovery.set_tip_hook(hook)
    refused(up, "the gripper tip is inside the last grasp region")
    # Asked once, for the measured start pose and the last grasp region.
    assert len(asked) == 1
    assert asked[0][0] == pytest.approx(RAISED)
    assert asked[0][1] is REGION


def test_a_blocked_tip_without_a_reason_is_still_refused(up):
    up.recovery.set_tip_hook(lambda q, region: "")
    refused(up, "the gripper tip is blocked")


def test_refused_when_the_path_is_over_the_recovery_speed_limit(
    make_rig, checkers, clock
):
    rig = make_rig(checkers=with_recovery(checkers, clock, max_joint_speed=0.05))
    rig.arm.set_pose(RAISED)
    outcome = refused(rig, "the path to rest was refused: setpoint ")
    assert "over the recovery speed limit 0.050 rad/s" in outcome.reason


def test_refused_when_the_path_starts_outside_the_recovery_joint_limits(
    make_rig, checkers, clock
):
    narrow = with_recovery(checkers, clock, joint_min=[-0.4] * N, joint_max=[0.4] * N)
    rig = make_rig(checkers=narrow)
    rig.arm.set_pose(RAISED)
    outcome = refused(rig, "the path to rest was refused: setpoint 0: joint index 0")
    assert "outside the recovery joint limits" in outcome.reason


def test_refused_when_the_recovery_checker_is_latched(up):
    up.recovery.latch("lost joint feedback")
    refused(up, "the path to rest was refused: fault-stop latched")


def test_refused_when_the_move_is_longer_than_the_executor_allows(make_rig, config):
    short = dataclasses.replace(config.executor, max_setpoints=520)
    rig = make_rig(config=dataclasses.replace(config, executor=short))
    rig.arm.set_pose(RAISED)
    refused(rig, "the return to rest needs 521 setpoints, the limit is 520")


def test_a_move_of_exactly_the_longest_length_is_accepted(make_rig, config):
    limit = dataclasses.replace(config.executor, max_setpoints=RAISED_SETPOINTS)
    rig = make_rig(config=dataclasses.replace(config, executor=limit))
    rig.arm.set_pose(RAISED)
    assert rig.request().status == AT_REST
    assert len(rig.arm.sent) == RAISED_SETPOINTS


def test_a_stop_while_the_conditions_are_checked_wins(up):
    calls = up.spy_run()
    up.on_read_all(2, up.stop)
    outcome = up.request()
    assert outcome.status == REFUSED
    assert outcome.reason == "a STOP was raised while the return to rest was checked"
    assert calls == []
    assert up.arm.sent == []
    # Only the STOP changed the owner number: recovery never became the owner.
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.owner_number == 1
    assert up.authority.stop_level is StopLevel.STOP


def test_a_fault_while_the_conditions_are_checked_wins(up):
    up.on_read_all(2, lambda: up.authority.raise_fault("motor fault"))
    outcome = up.request()
    assert outcome == RestOutcome(REFUSED, "fault-stop latched: motor fault")
    assert up.arm.sent == []
    assert up.authority.owner_number == 1


def test_an_owner_change_while_the_conditions_are_checked_wins(up):
    up.on_read_all(2, lambda: up.authority.set_owner(Owner.HUMAN, "takeover"))
    outcome = up.request()
    assert outcome.status == REFUSED
    assert "the control owner changed" in outcome.reason
    assert up.arm.sent == []
    assert up.authority.owner is Owner.HUMAN
    assert up.authority.owner_number == 1


@pytest.mark.parametrize("read", [1, 2])
def test_the_end_of_the_agents_ownership_while_the_conditions_are_checked_refuses_nothing(
    up, read
):
    up.authority.set_owner(Owner.AGENT, "new task")
    up.on_read_all(read, lambda: up.authority.set_owner(Owner.NOBODY, "help needed"))
    calls = up.spy_run()
    assert up.request() == RestOutcome(AT_REST)
    assert len(up.arm.sent) == RAISED_SETPOINTS
    # Recovery took the arm from nobody, with the numbers of that moment.
    admitted = calls[0][0][6]
    assert (admitted.owner, admitted.owner_number) == (Owner.RECOVERY, 3)
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.owner_number == 4


def test_the_arm_is_taken_once_more_and_no_further(up):
    up.authority.set_owner(Owner.AGENT, "new task")
    real = up.authority.set_owner_if
    tried: list[int] = []

    def set_owner_if(owner: Owner, reason: str, **numbers: int) -> int | None:
        if owner is Owner.RECOVERY:
            tried.append(numbers["owner_number"])
            # Each time someone else's ownership has just ended.
            up.authority.set_owner(Owner.AGENT, "new task")
            up.authority.set_owner(Owner.NOBODY, "help needed")
        return real(owner, reason, **numbers)

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    outcome = up.request()
    assert outcome == RestOutcome(
        REFUSED, "the control owner changed while the return to rest was checked"
    )
    assert tried == [1, 3]
    assert up.arm.sent == []
    assert up.authority.owner is Owner.NOBODY


@pytest.mark.parametrize("change", ["a STOP", "a fault-stop", "the agent as owner"])
def test_the_arm_is_not_taken_once_more_after_anything_but_an_ownership_ending(
    up, change
):
    def changed() -> None:
        if change == "a STOP":
            up.stop()
        elif change == "a fault-stop":
            up.authority.raise_fault("motor 3 overheated")
        else:
            up.authority.set_owner(Owner.AGENT, "new task")

    real = up.authority.set_owner_if
    tried: list[Owner] = []

    def set_owner_if(owner: Owner, reason: str, **numbers: int) -> int | None:
        if not tried:
            changed()
        tried.append(owner)
        return real(owner, reason, **numbers)

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    outcome = up.request()
    assert outcome.status == REFUSED
    assert tried == [Owner.RECOVERY]
    assert up.arm.sent == []
    assert up.authority.owner is not Owner.RECOVERY


def test_a_request_told_to_stop_while_it_was_checked_announces_no_move(up):
    told: list[str] = []
    up.on_read_all(2, lambda: told.append("robot connection lost"))
    outcome = up.request(should_stop=lambda: told[0] if told else None)
    assert outcome == RestOutcome(
        STOPPED, "robot connection lost (after 0 of 521 setpoints)"
    )
    assert up.arm.sent == []
    assert up.events == [
        ("rest", f"return to rest stopped: {outcome.reason}; the arm holds")
    ]
    assert up.authority.owner is Owner.NOBODY
    assert up.rest.last_move is None


def test_a_second_request_during_a_return_is_refused(up):
    answers: list[RestOutcome] = []
    up.after_sends(5, lambda: answers.append(up.request()))
    assert up.request().status == AT_REST
    assert answers == [RestOutcome(REFUSED, BUSY)]
    assert len(up.arm.sent) == RAISED_SETPOINTS


def test_an_arm_fault_while_the_conditions_are_checked_is_passed_on(up):
    def fault() -> None:
        raise ArmFault("the bus is down")

    before = up.state()
    up.on_read_all(1, fault)
    with pytest.raises(ArmFault, match=r"the bus is down"):
        up.request()
    assert up.state() == before
    # The request is over: a later one is not refused as "already running".
    up.patch.undo()
    assert up.request().status == AT_REST


# STOP wins at any moment


def test_a_stop_just_before_the_first_setpoint_sends_none(up):
    def on_event(kind: str, text: str) -> None:
        if text.startswith(MOVE_EVENT):
            up.stop()

    up.on_event = on_event
    outcome = up.request("wait")
    assert outcome.status == STOPPED
    assert "stop number is now 1" in outcome.reason
    assert "after 0 of 521 setpoints" in outcome.reason
    assert up.arm.sent == []
    assert up.arm.holds == []
    assert up.arm.pose() == RAISED
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_level is StopLevel.STOP
    assert up.motion.get() is MotionState.HOLD
    assert up.rest.last_move.status == "stopped"
    assert up.events[-1] == (
        "rest",
        f"return to rest stopped: {outcome.reason}; the arm holds",
    )


def test_a_stop_during_the_start_read_of_the_move_sends_none(up):
    # Reads 1 and 2 are the conditions; read 3 is the executor's start pose.
    up.on_read_all(3, up.stop)
    outcome = up.request()
    assert outcome.status == STOPPED
    assert up.arm.sent == []
    assert up.arm.torque_on()
    assert up.authority.owner is Owner.NOBODY


def test_a_stop_as_recovery_becomes_the_owner_sends_none(up):
    calls = up.spy_run()
    real = up.authority.set_owner_if

    def set_owner_if(owner: Owner, reason: str, **numbers: int) -> int | None:
        number = real(owner, reason, **numbers)
        if owner is Owner.RECOVERY:
            up.stop()
        return number

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    outcome = up.request()
    assert outcome.status == STOPPED
    assert "arrived as the return to rest began" in outcome.reason
    assert calls == []
    assert up.arm.sent == []
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_level is StopLevel.STOP
    assert all(not text.startswith(MOVE_EVENT) for _, text in up.events)


@pytest.mark.parametrize("owner", [Owner.NOBODY, Owner.HUMAN])
def test_a_stop_just_before_recovery_becomes_the_owner_sends_none(up, owner):
    if owner is Owner.HUMAN:
        up.authority.set_owner(owner, "takeover")
    calls = up.spy_run()
    real = up.authority.set_owner_if
    answers: list[int | None] = []

    def set_owner_if(new_owner: Owner, reason: str, **numbers: int) -> int | None:
        if new_owner is Owner.RECOVERY:
            up.stop()  # lands after the last check and before the change
        answers.append(real(new_owner, reason, **numbers))
        return answers[-1]

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    owner_number = up.authority.owner_number
    outcome = up.request()
    assert outcome == RestOutcome(
        REFUSED, "a STOP was raised while the return to rest was checked"
    )
    assert calls == []
    assert up.arm.sent == []
    assert up.arm.torque_on()
    # Recovery never had the arm, and the person does not get it back: only the
    # STOP changed the owner, and its STOP level stands.
    assert answers == [None]
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.owner_number == owner_number + 1
    assert up.authority.stop_number == 1
    assert up.authority.stop_level is StopLevel.STOP
    assert up.motion.states == [MotionState.HOLD]


def test_a_halt_before_the_first_setpoint_sends_none(up):
    up.on_read_all(3, lambda: up.recovery.halt("the robot connection is lost"))
    outcome = up.request()
    # Recovery had taken the arm: failed, not a refusal that changed nothing.
    assert outcome.status == FAILED
    assert outcome.reason.startswith("refused after 0 of 521 setpoints: ")
    assert "halted since this motion was admitted" in outcome.reason
    assert up.arm.sent == []
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.owner_number == 2
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()


def test_a_halt_while_the_conditions_are_checked_sends_none(up):
    up.on_read_all(2, lambda: up.recovery.halt("stop"))
    outcome = up.request()
    assert outcome.status == FAILED
    assert "halt count is 1, the motion quotes 0" in outcome.reason
    assert up.arm.sent == []


@pytest.mark.parametrize("owner", [Owner.NOBODY, Owner.HUMAN])
def test_a_stop_during_the_return_ends_it_in_a_hold_with_motors_on(up, owner):
    if owner is Owner.HUMAN:
        up.authority.set_owner(owner, "takeover")
    calls = up.spy_run()
    up.after_sends(40, up.stop)
    outcome = up.request()
    assert outcome.status == STOPPED
    assert "after 40 of 521 setpoints" in outcome.reason
    # The setpoint after the STOP was not sent, and nothing was tried again.
    assert len(up.arm.sent) == 40
    assert len(calls) == 1
    # The arm holds at the last setpoint with the motors on.
    up.clock.advance(1.0)
    assert up.arm.pose() == pytest.approx(up.sent()[-1], abs=1e-6)
    assert off_rest(up.arm.pose()) > 0.4
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    assert up.arm.holds == []
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.jaw_commands == []
    # A STOP always leaves nobody the owner, also when a person owned the arm.
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_level is StopLevel.STOP
    assert up.rest.last_move.status == "stopped"
    assert not up.recovery.in_motion


def test_a_fault_stop_during_the_return_ends_it_in_a_hold(up):
    up.after_sends(40, lambda: up.authority.raise_fault("motor fault"))
    outcome = up.request()
    assert outcome.status == STOPPED
    assert len(up.arm.sent) == 40
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_level is StopLevel.FAULT


def test_a_takeover_during_the_return_ends_it_and_keeps_the_new_owner(up):
    up.after_sends(40, lambda: up.authority.set_owner(Owner.HUMAN, "takeover"))
    outcome = up.request("wait")
    assert outcome.status == STOPPED
    assert "owner number is now 2" in outcome.reason
    assert len(up.arm.sent) == 40
    assert up.authority.owner is Owner.HUMAN
    assert up.authority.owner_number == 2
    assert up.arm.torque_on()


def test_a_lost_robot_connection_during_the_return_ends_it_in_a_hold(up):
    up.after_sends(40, lambda: up.authority.set_owner(Owner.NOBODY, "link lost"))
    outcome = up.request("wait")
    assert outcome.status == STOPPED
    assert len(up.arm.sent) == 40
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.owner_number == 2
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()


@pytest.mark.parametrize(
    ("owner", "afterwards"),
    [(Owner.AGENT, Owner.NOBODY), (Owner.HUMAN, Owner.HUMAN)],
)
def test_should_stop_ends_the_return_and_the_owner_is_set_back(up, owner, afterwards):
    up.authority.set_owner(owner, "test")

    def should_stop() -> str | None:
        return "the operator closed the app" if len(up.arm.sent) >= 25 else None

    outcome = up.request(should_stop=should_stop)
    assert outcome.status == STOPPED
    assert outcome.reason == "the operator closed the app (after 25 of 521 setpoints)"
    assert len(up.arm.sent) == 25
    assert up.authority.owner is afterwards
    assert up.authority.owner_number == 3
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()


def test_a_stop_as_the_owner_is_set_back_leaves_nobody_the_owner(up):
    up.authority.set_owner(Owner.HUMAN, "takeover")
    real = up.authority.set_owner_if

    def set_owner_if(owner: Owner, reason: str, **numbers: int) -> int | None:
        if owner is Owner.HUMAN:
            up.stop()  # lands just before the owner is set back
        return real(owner, reason, **numbers)

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    assert up.request().status == AT_REST
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_number == 1
    assert up.authority.stop_level is StopLevel.STOP


def test_a_takeover_as_the_owner_is_set_back_keeps_the_person(up):
    real = up.authority.set_owner_if

    def set_owner_if(owner: Owner, reason: str, **numbers: int) -> int | None:
        if owner is Owner.NOBODY:
            up.authority.set_owner(Owner.HUMAN, "takeover")
        return real(owner, reason, **numbers)

    up.patch.setattr(up.authority, "set_owner_if", set_owner_if)
    assert up.request().status == AT_REST
    assert up.authority.owner is Owner.HUMAN


# A return that ends early or away from rest


def test_a_contact_ends_the_return_in_a_hold_at_the_measured_pose(up):
    calls = up.spy_run()
    up.arm.block_joint(0, 0.3)
    outcome = up.request("wait")
    assert outcome.status == FAILED
    assert outcome.reason.startswith("contact after ")
    assert "joint index 0 trails its setpoints" in outcome.reason
    # One hold at the measured pose, no second try, motors on.
    assert len(calls) == 1
    assert len(up.arm.holds) == 1
    assert up.arm.holds[0][0][0] == pytest.approx(0.3, abs=1e-6)
    assert len(up.arm.sent) < RAISED_SETPOINTS
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    assert up.motion.get() is MotionState.HOLD
    assert up.authority.owner is Owner.NOBODY
    assert up.authority.stop_number == 0
    assert up.rest.last_move.status == "contact"
    assert up.rest.last_move.end_q[0] == pytest.approx(0.3, abs=1e-6)
    assert up.events[-1] == (
        "rest",
        f"return to rest failed: {outcome.reason}; the arm holds",
    )


def test_after_a_failed_return_a_new_request_starts_from_scratch(up):
    up.arm.block_joint(0, 0.3)
    assert up.request().status == FAILED
    up.arm.block_joint(0, None)
    events = len(up.events)
    assert up.request().status == AT_REST
    assert off_rest(up.arm.pose()) < 0.005
    assert up.rest.last_move.status == "completed"
    # The second move starts where the arm was held: 0.3 rad from rest.
    assert up.events[events][1].startswith(f"{MOVE_EVENT} 6.3 s ")


def test_after_a_contact_a_person_is_the_owner_again(up):
    up.authority.set_owner(Owner.HUMAN, "takeover")
    up.arm.block_joint(0, 0.3)
    assert up.request("takeover_end").status == FAILED
    assert up.authority.owner is Owner.HUMAN


def test_a_return_that_does_not_settle_fails_and_holds(up):
    up.arm.block_joint(0, 0.04)  # short of rest by more than the settle tolerance
    outcome = up.request()
    assert outcome.status == FAILED
    assert outcome.reason.startswith("not_settled after 521 of 521 setpoints")
    assert up.motion.get() is MotionState.HOLD
    assert up.authority.owner is Owner.NOBODY
    assert up.arm.torque_on()
    assert up.arm.released_at == []


def test_a_completed_move_that_is_not_at_rest_fails_and_holds(up):
    calls = up.spy_run(after=lambda: up.arm.set_pose(raised(0, 0.2)))
    outcome = up.request()
    assert outcome.status == FAILED
    assert "the arm is not at rest: joint index 0 is at 0.2000 rad" in outcome.reason
    assert len(calls) == 1
    assert up.motion.states[-2:] == [MotionState.HOLD, MotionState.HOLD]
    assert up.authority.owner is Owner.NOBODY
    assert up.arm.torque_on()
    assert up.rest.last_move.status == "completed"


def test_a_failed_read_after_the_move_fails_and_holds(up):
    up.spy_run(after=lambda: up.arm.fail_reads(4, 1))
    outcome = up.request()
    assert outcome.status == FAILED
    assert "a joint read failed: joint index 4" in outcome.reason
    assert up.motion.get() is MotionState.HOLD
    assert up.authority.owner is Owner.NOBODY


@pytest.mark.parametrize("owner", [Owner.NOBODY, Owner.AGENT, Owner.HUMAN])
def test_a_refusal_by_the_executor_before_any_setpoint_is_a_failure(up, owner):
    if owner is not Owner.NOBODY:
        up.authority.set_owner(owner, "test")
    number = up.authority.owner_number
    # The arm moves after the conditions were read: the start pose is no longer it.
    up.on_read_all(3, lambda: up.arm.set_pose(raised(0, 0.9)))
    outcome = up.request()
    # Recovery had taken the arm by then, so the request did change something.
    assert outcome.status == FAILED
    assert outcome.reason.startswith(
        "refused after 0 of 521 setpoints: start pose: joint index 0 is at 0.9000 rad"
    )
    assert up.arm.sent == []
    assert up.authority.owner is (owner if owner is Owner.HUMAN else Owner.NOBODY)
    assert up.authority.owner_number == number + 2
    assert up.motion.get() is MotionState.HOLD
    assert up.rest.last_move.status == "refused"
    assert up.events[-1] == (
        "rest",
        f"return to rest failed: {outcome.reason}; the arm holds",
    )


def test_a_refusal_in_the_middle_of_the_move_is_a_failure(up):
    up.after_sends(30, lambda: up.arm.fail_reads(0, 1000))
    outcome = up.request()
    assert outcome.status == FAILED
    assert outcome.reason.startswith("failed after ")
    assert "fault: joint feedback lost: joint index 0" in outcome.reason
    # The reason the app raises a fault-stop for.
    assert up.rest.last_move.reason.startswith("fault: ")
    assert 30 < len(up.arm.sent) < RAISED_SETPOINTS
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()


def test_a_motor_fault_during_the_move_fails_and_is_shown_in_last_move(up):
    def fault() -> None:
        raise ArmFault("motor 2 overheated")

    up.after_sends(30, fault)
    outcome = up.request()
    assert outcome.status == FAILED
    assert "fault: motor 2 overheated" in outcome.reason
    assert up.rest.last_move.status == "failed"
    assert up.rest.last_move.reason == "fault: motor 2 overheated"
    assert up.authority.owner is Owner.NOBODY
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()
    assert up.arm.released_at == []


def test_an_arm_fault_after_the_move_is_passed_on_with_the_owner_set_back(up):
    def fault() -> None:
        raise ArmFault("the bus is down")

    up.authority.set_owner(Owner.HUMAN, "takeover")
    # Reads 1 and 2 are the conditions, 3 and 4 the executor's, 5 proves rest.
    up.on_read_all(5, fault)
    with pytest.raises(ArmFault, match=r"the bus is down"):
        up.request()
    assert up.authority.owner is Owner.HUMAN
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on()
    assert up.arm.released_at == []
    up.patch.undo()
    assert up.request().status == AT_REST


def test_last_move_is_cleared_by_a_request_that_sends_no_move(up):
    up.request()
    assert up.rest.last_move.status == "completed"
    assert up.rest.last_move.index == RAISED_SETPOINTS
    up.request()
    assert up.rest.last_move is None


def test_an_error_in_the_event_sender_does_not_end_the_return(up):
    def on_event(kind: str, text: str) -> None:
        raise RuntimeError("the link is gone")

    up.on_event = on_event
    assert up.request().status == AT_REST
    assert len(up.events) == 2
    assert off_rest(up.arm.pose()) < 0.005
    assert up.authority.owner is Owner.NOBODY


# The torque release


def test_release_at_rest_cuts_torque_once_at_the_rest_pose(rig):
    before = rig.authority.snapshot()
    assert rig.release() is None
    assert rig.arm.released_at == [REST]
    assert not rig.arm.torque_on()
    assert rig.motion.get() is MotionState.REST_UNPOWERED
    assert rig.authority.snapshot() == before
    assert rig.arm.sent == []
    assert rig.arm.jaw_commands == []
    assert rig.events == [("release", "torque released at rest: the motors are off")]
    assert rig.event_data == [{"stop_number": before.stop_number, "confirmed": True}]


def test_a_release_the_driver_only_sent_is_not_reported_as_motors_off(rig):
    rig.patch.setattr(rig.arm, "release_confirmed", lambda: False)
    assert rig.release() is None
    assert rig.events == [
        (
            "release",
            "torque release sent at rest: the driver does not confirm that the "
            "motors are off",
        )
    ]
    assert rig.event_data == [{"stop_number": 0, "confirmed": False}]
    assert rig.motion.get() is MotionState.REST_UNPOWERED


def test_release_after_a_return_to_rest(up):
    assert up.request("shutdown", True).status == AT_REST
    assert up.arm.released_at == []
    assert up.release() is None
    assert len(up.arm.released_at) == 1
    assert off_rest(up.arm.released_at[0]) < 0.005
    assert up.motion.get() is MotionState.REST_UNPOWERED


def test_release_takes_two_fresh_reads_the_read_gap_apart(rig):
    times: list[float] = []
    rig.on_read_all(1, lambda: times.append(rig.clock.monotonic()))
    rig.on_read_all(2, lambda: times.append(rig.clock.monotonic()))
    rig.on_read_all(3, lambda: times.append(rig.clock.monotonic()))
    rig.motion.set(MotionState.REST_HOLD)
    assert rig.release() is None
    assert len(times) == 2
    assert times[1] - (times[0] + READ_ALL_S) == pytest.approx(
        rig.config.rest.read_gap_s
    )


def test_release_is_decided_by_the_reads_and_not_by_the_motion_state(rig):
    # The motion state says "hold", the reads say rest: released.
    assert rig.motion.get() is MotionState.HOLD
    assert rig.release() is None
    assert rig.arm.released_at == [REST]


def release_refused(rig: Rig, match: str, *, ignore: Sequence[str] = ()) -> str:
    """Ask for the release; it must be refused for `match` and change nothing."""
    before = rig.state()
    events = list(rig.events)
    reason = rig.release()
    assert reason is not None
    assert match in reason
    after = rig.state()
    for key in ignore:
        before.pop(key)
        after.pop(key)
    assert after == before
    assert rig.arm.torque_on()
    assert rig.arm.released_at == []
    assert rig.events == events
    return reason


def test_release_refused_away_from_rest(up):
    reason = release_refused(up, "the arm is not at rest: joint index 0")
    assert "is at 0.5000 rad, rest is 0.0000 rad (tolerance 0.05 rad)" in reason


def test_release_refused_just_outside_the_rest_tolerance(rig):
    rig.arm.set_pose(raised(5, -0.06))
    release_refused(rig, "the arm is not at rest: joint index 5")


def test_release_inside_the_rest_tolerance(rig):
    rig.arm.set_pose(raised(5, -0.04))
    assert rig.release() is None
    assert rig.arm.released_at == [raised(5, -0.04)]


def test_release_refused_away_from_rest_whatever_the_motion_state_says(up):
    up.motion.set(MotionState.REST_HOLD)
    release_refused(up, "the arm is not at rest")


def test_release_refused_when_the_arm_leaves_rest_between_the_reads(rig):
    rig.on_read_all(2, lambda: rig.arm.set_pose(raised(0, 0.3)))
    release_refused(rig, "disagree: joint index 0", ignore=["pose"])


def test_release_refused_when_the_two_reads_are_not_within_tolerance_of_each_other(
    rig,
):
    # Each read is within 0.05 rad of rest, but they are 0.08 rad apart.
    rig.arm.set_pose(raised(2, 0.04))
    rig.on_read_all(2, lambda: rig.arm.set_pose(raised(2, -0.04)))
    reason = release_refused(rig, "disagree: joint index 2", ignore=["pose"])
    assert "(tolerance 0.05 rad)" in reason


@pytest.mark.parametrize(("first", "second"), [(0.07, 0.03), (0.03, 0.07)])
def test_release_needs_both_reads_at_rest(rig, first, second):
    # The two reads agree within 0.05 rad, but one is 0.07 rad from rest.
    rig.arm.set_pose(raised(2, first))
    rig.on_read_all(2, lambda: rig.arm.set_pose(raised(2, second)))
    reason = release_refused(rig, "the arm is not at rest", ignore=["pose"])
    assert "joint index 2 is at 0.0700 rad" in reason


def test_release_reads_the_joints_after_the_jaws(rig):
    order: list[str] = []
    jaw_read = rig.arm.gripper_read

    def gripper_read() -> float | None:
        order.append("jaws")
        return jaw_read()

    rig.patch.setattr(rig.arm, "gripper_read", gripper_read)
    rig.on_read_all(1, lambda: order.append("joints"))
    rig.on_read_all(2, lambda: order.append("joints"))
    assert rig.release() is None
    # The joint reads are the last evidence taken before the torque is cut.
    assert order == ["jaws"] * rig.config.gripper.reads + ["joints"] * 2


@pytest.mark.parametrize("read", [1, 2])
def test_release_refused_when_a_joint_read_fails(rig, read):
    rig.on_read_all(read, lambda: rig.arm.fail_reads(1, 1))
    release_refused(rig, "a joint read failed: joint index 1")


def test_release_refused_with_loaded_jaws(rig):
    rig.close_on_object()
    release_refused(rig, "the jaws are not proven empty (loaded)")


def test_release_refused_while_a_jaw_command_runs(rig):
    rig.jaws_busy = True
    release_refused(rig, "a jaw command is running")


def test_release_refused_with_unknown_jaws(make_rig):
    rig = make_rig(reference=False)
    reason = release_refused(rig, "the jaws are not proven empty (unknown)")
    assert "not referenced" in reason


def test_release_refused_with_a_stop_number_that_is_over(rig):
    seen = rig.authority.stop_number
    rig.stop()
    before = rig.state()
    reason = rig.rest.release(seen)
    assert reason == "stop number 0 does not match the current 1"
    assert rig.rest.release(2) == "stop number 2 does not match the current 1"
    assert rig.state() == before
    assert rig.arm.released_at == []
    # The current number releases.
    assert rig.release() is None


def test_release_refused_with_a_fault_latched(rig):
    rig.authority.raise_fault("gripper fault")
    release_refused(rig, "fault-stop latched: gripper fault")


def test_release_refused_while_a_move_is_running(rig):
    rig.moving = True
    release_refused(rig, "a move is running")


def test_release_refused_while_the_executor_runs_a_move(rig):
    rig.authority.set_owner(Owner.AGENT, "new task")
    rig.checkers.agent.feed_watchdog()
    answers: list[str | None] = []
    rig.after_sends(3, lambda: answers.append(rig.release()))
    path = [raised(0, 0.005 * step) for step in range(1, 11)]
    move = rig.executor.run(
        path,
        REST,
        0.02,
        PERIOD,
        Owner.AGENT,
        rig.checkers.agent,
        rig.authority.snapshot(),
        "move-1",
    )
    assert move.status == "completed"
    assert answers == ["a move is running"]
    assert rig.arm.released_at == []
    assert rig.arm.torque_on()


def test_release_refused_during_a_return_to_rest(up):
    answers: list[str | None] = []
    up.after_sends(5, lambda: answers.append(up.release()))
    assert up.request().status == AT_REST
    assert answers == [BUSY]
    assert up.arm.released_at == []
    assert up.arm.torque_on()


def test_release_refused_from_inside_a_request(up):
    answers: list[object] = []

    def on_event(kind: str, text: str) -> None:
        if text.startswith(MOVE_EVENT):
            answers.append(up.release())
            answers.append(up.request())

    up.on_event = on_event
    assert up.request().status == AT_REST
    assert answers == [BUSY, RestOutcome(REFUSED, BUSY)]
    assert up.arm.released_at == []


def test_a_stop_while_the_release_is_checked_keeps_the_motors_on(rig):
    rig.on_read_all(2, rig.stop)
    seen = rig.authority.stop_number
    reason = rig.rest.release(seen)
    assert reason == "a STOP was raised while the release was checked"
    assert rig.arm.released_at == []
    assert rig.arm.torque_on()
    assert rig.motion.get() is MotionState.HOLD


def test_a_release_told_to_stop_before_it_begins_reads_nothing(rig):
    before = rig.state()
    t_before = rig.clock.monotonic()
    reason = rig.rest.release(0, lambda: "robot connection lost")
    assert reason == "robot connection lost"
    assert rig.clock.monotonic() == t_before  # every joint read costs clock time
    assert rig.state() == before
    assert rig.events == []


@pytest.mark.parametrize("read", [1, 2])
def test_a_release_told_to_stop_while_it_is_checked_keeps_the_motors_on(rig, read):
    told: list[str] = []
    rig.on_read_all(read, lambda: told.append("robot connection lost"))
    reason = rig.rest.release(0, lambda: told[0] if told else None)
    assert reason == "robot connection lost"
    assert rig.arm.released_at == []
    assert rig.arm.torque_on()
    assert rig.motion.get() is MotionState.HOLD
    # Not told to stop, the same release goes through.
    assert rig.rest.release(0, lambda: None) is None
    assert rig.arm.released_at == [REST]


def test_a_stop_is_named_before_what_else_told_the_release_to_stop(rig):
    told: list[str] = []

    def stop() -> None:
        rig.stop()
        told.append("STOP from console: operator")

    rig.on_read_all(2, stop)
    reason = rig.rest.release(0, lambda: told[0] if told else None)
    assert reason == "a STOP was raised while the release was checked"
    reason = rig.rest.release(0, lambda: told[0])
    assert reason == "stop number 0 does not match the current 1"
    assert rig.arm.released_at == [] and rig.arm.torque_on()


def test_a_move_that_begins_while_the_release_is_checked_keeps_the_motors_on(rig):
    def begin() -> None:
        rig.moving = True

    rig.on_read_all(2, begin)
    reason = rig.release()
    assert reason == "a move began while the release was checked"
    assert rig.arm.released_at == []
    assert rig.arm.torque_on()


def test_no_move_can_begin_on_the_executor_while_the_release_is_checked(rig):
    rig.authority.set_owner(Owner.AGENT, "new task")
    rig.checkers.agent.feed_watchdog()
    moves: list[MoveOutcome] = []

    def begin() -> None:
        path = [raised(0, 0.005 * step) for step in range(1, 11)]
        admitted = rig.authority.snapshot()
        agent = rig.checkers.agent
        moves.append(
            rig.executor.run(
                path, REST, 0.02, PERIOD, Owner.AGENT, agent, admitted, "move-1"
            )
        )

    rig.on_read_all(2, begin)
    assert rig.release() is None
    assert moves == [MoveOutcome("refused", "another move is running")]
    assert rig.arm.sent == []
    assert rig.arm.released_at == [REST]


def test_a_release_that_is_refused_gives_the_executor_back(up):
    release_refused(up, "the arm is not at rest")
    assert up.request().status == AT_REST


def test_a_second_release_is_refused(rig):
    assert rig.release() is None
    assert rig.release() == "the motors are already off"
    assert rig.arm.released_at == [REST]
    assert rig.motion.get() is MotionState.REST_UNPOWERED


def test_after_a_release_a_return_to_rest_is_refused(rig):
    assert rig.release() is None
    outcome = rig.request()
    assert outcome == RestOutcome(REFUSED, "the motors are off")
    assert rig.motion.get() is MotionState.REST_UNPOWERED
    assert rig.arm.sent == []
    assert rig.authority.owner_number == 0


def test_a_request_never_releases(rig):
    for source in ("typed", "wait", "shutdown", "takeover_end"):
        rig.arm.set_pose(raised(1, 0.1))
        assert rig.request(source, True).status == AT_REST
        assert len(rig.arm.sent) % 105 == 0  # each one moved the arm
    assert rig.arm.released_at == []
    assert rig.arm.torque_on()


def test_release_passes_a_driver_fault_on_and_shows_the_motor_state(rig):
    def failing() -> None:
        raise ArmFault("the driver's disconnect failed")

    rig.patch.setattr(rig.arm, "release_torque", failing)
    with pytest.raises(ArmFault, match=r"disconnect failed"):
        rig.release()
    # The motors are still on, so the motion state does not claim otherwise.
    assert rig.arm.torque_on()
    assert rig.motion.get() is MotionState.HOLD
    assert rig.events == []
    rig.patch.undo()
    assert rig.release() is None


def test_a_release_whose_driver_call_fails_after_the_cut_shows_motors_off(rig):
    cut = rig.arm.release_torque

    def cut_then_fail() -> None:
        cut()
        raise ArmFault("the driver's disconnect failed")

    rig.patch.setattr(rig.arm, "release_torque", cut_then_fail)
    with pytest.raises(ArmFault, match=r"disconnect failed"):
        rig.release()
    assert not rig.arm.torque_on()
    assert rig.motion.get() is MotionState.REST_UNPOWERED
    assert rig.arm.released_at == [REST]


def test_the_torque_release_has_one_call_site_in_rest_py():
    source = Path(rest_module.__file__).read_text(encoding="utf-8")
    assert source.count("release_torque" + "(") == 1
    assert source.count(".release_torque" + "()") == 1
