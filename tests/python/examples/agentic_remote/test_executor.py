# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the move executor in
isaaccapture_examples.agentic_remote.robot_app.executor."""

from __future__ import annotations

import dataclasses
import math
import threading
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion, Permit
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    CheckerSet,
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
)
from isaaccapture_examples.agentic_remote.robot_app.permits import PermitGate
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.state import JointHistory

N = 6
PERIOD = 0.02
READ_S = 0.008  # the simulated arm's time for one joint read
START_READ_S = N * READ_S  # a move begins with one read of every joint
REST = [0.0] * N
START_TOL = 0.02
MOVE_ID = "move-1"
IDLE = ("", 0, 0)
REGION = GraspRegion((0.3, 0.0), 0.05, 0.01, None)
WAIT_S = 5.0  # longest a test waits for another thread


def ramp(
    to: float, count: int, *, joint: int = 0, start: Sequence[float] = REST
) -> list[list[float]]:
    """count setpoints that take one joint from its start value to `to` evenly."""
    rows = []
    for i in range(1, count + 1):
        q = list(start)
        q[joint] = start[joint] + (to - start[joint]) * i / count
        rows.append(q)
    return rows


# Joint 0 at 0.32 rad/s, under the agent's 0.4 rad/s.
PATH = ramp(0.16, 25)
LONG_PATH = ramp(0.32, 50)


def path_with(row: object, index: int = 3) -> list:
    """PATH with one setpoint replaced."""
    rows: list = [list(q) for q in PATH]
    rows[index] = row
    return rows


class Rig:
    """An executor on a connected simulated arm whose control owner is the agent."""

    def __init__(
        self,
        arm: SimArm,
        checkers: CheckerSet,
        config: RobotAppConfig,
        clock: FakeClock,
        patch: pytest.MonkeyPatch,
    ) -> None:
        arm.connect()
        self.arm = arm
        self.checkers = checkers
        self.agent = checkers.agent
        self.config = config
        self.clock = clock
        self.patch = patch
        self.authority = Authority()
        self.authority.set_owner(Owner.AGENT, "new task")
        self.history = JointHistory(config.n_joints, config.state.history_s)
        self.gate = PermitGate(self.authority, self.agent, config.permits, clock)
        self.missed: list[int] = []  # one entry per on_missed_deadline call
        self.executor = self.new_executor()

    def new_executor(
        self, *, first_wait_s: float | None = None, **settings: object
    ) -> MoveExecutor:
        if first_wait_s is None:
            first_wait_s = self.config.permits.first_wait_s
        return MoveExecutor(
            self.arm,
            self.authority,
            self.history,
            dataclasses.replace(self.config.executor, **settings),
            self.clock,
            first_wait_s=first_wait_s,
            on_missed_deadline=lambda: self.missed.append(1),
        )

    def run(
        self,
        setpoints: object = PATH,
        *,
        executor: MoveExecutor | None = None,
        start_q: object = REST,
        start_tol: float = START_TOL,
        period_s: float = PERIOD,
        owner: Owner = Owner.AGENT,
        checker: object = None,
        admitted: object = None,
        **more: object,
    ) -> MoveOutcome:
        """Run PATH for the agent, with whatever a test changes."""
        return (executor or self.executor).run(
            setpoints,
            start_q,
            start_tol,
            period_s,
            owner,
            self.agent if checker is None else checker,
            self.authority.snapshot() if admitted is None else admitted,
            MOVE_ID,
            **more,
        )

    def sent(self) -> list[list[float]]:
        return [q for q, _ in self.arm.sent]

    def times(self) -> list[float]:
        return [t for _, t in self.arm.sent]

    def gaps(self) -> list[float]:
        times = self.times()
        return [later - earlier for earlier, later in zip(times, times[1:])]

    def on_send(
        self,
        action: Callable[[], object],
        *,
        index: int | None = None,
        before: bool = False,
    ) -> None:
        """Call action() around the send of setpoint `index`, or of every one."""
        real = self.arm.send_target

        def send(q: Sequence[float]) -> None:
            due = index is None or len(self.arm.sent) == index
            if due and before:
                action()
            real(q)
            if due and not before:
                action()

        self.patch.setattr(self.arm, "send_target", send)

    def stall_in_reads(self, index: int, seconds: float) -> None:
        """Lose `seconds` once, in the joint reads that follow setpoint `index`."""
        real = self.arm.read_joint
        done: list[bool] = []

        def read(joint: int) -> object:
            if not done and len(self.arm.sent) == index + 1:
                done.append(True)
                self.clock.advance(seconds)
            return real(joint)

        self.patch.setattr(self.arm, "read_joint", read)

    def permit(
        self,
        through_index: int,
        *,
        expires_in_s: float = 5.0,
        evidence_age_s: float = 0.0,
        move_id: str = MOVE_ID,
    ) -> Permit:
        """Give the gate a permit that quotes the authority's current values."""
        state = self.authority.snapshot()
        permit = Permit(
            move_id=move_id,
            session_id=state.session_id,
            owner_number=state.owner_number,
            stop_number=state.stop_number,
            through_index=through_index,
            evidence_wall_s=self.clock.wall() - evidence_age_s,
            expires_mono_s=self.clock.monotonic() + expires_in_s,
        )
        assert self.gate.update(permit) is None
        return permit


@pytest.fixture
def bare(sim_arm, checkers, config, clock, monkeypatch) -> Rig:
    """A rig whose agent checker's watchdog was never fed: permit tests use it."""
    return Rig(sim_arm, checkers, config, clock, monkeypatch)


@pytest.fixture
def rig(bare) -> Rig:
    """A rig whose agent checker's watchdog is fed, as an accepted permit leaves it."""
    bare.agent.feed_watchdog()
    return bare


@pytest.fixture
def leaving(checkers, monkeypatch) -> list[str]:
    """The agent checker's set_grasp_region and end_motion calls, in order."""
    calls: list[str] = []
    agent = checkers.agent
    real_region, real_end = agent.set_grasp_region, agent.end_motion

    def set_grasp_region(region: GraspRegion | None) -> None:
        calls.append("region off" if region is None else "region on")
        real_region(region)

    def end_motion() -> None:
        calls.append("end")
        real_end()

    monkeypatch.setattr(agent, "set_grasp_region", set_grasp_region)
    monkeypatch.setattr(agent, "end_motion", end_motion)
    return calls


@pytest.fixture
def watches(checkers, monkeypatch) -> list:
    """Every contact watch the agent checker hands out."""
    made: list = []
    real = checkers.agent.contact_watch

    def contact_watch(arm: object, q_start: object, history: object) -> object:
        made.append(real(arm, q_start, history))
        return made[-1]

    monkeypatch.setattr(checkers.agent, "contact_watch", contact_watch)
    return made


class StopProbe:
    """A should_stop that tells a test once the wait for a permit is under way.

    The executor asks once on entry and once per pass of its wait, so the third
    question means one wait has already run out.
    """

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason  # returned from the third question on
        self.calls = 0
        self.waiting = threading.Event()

    def __call__(self) -> str | None:
        self.calls += 1
        if self.calls < 3:
            return None
        self.waiting.set()
        return self.reason


def in_thread(call: Callable[[], MoveOutcome]) -> tuple[threading.Thread, list]:
    outcomes: list[MoveOutcome] = []
    thread = threading.Thread(target=lambda: outcomes.append(call()), daemon=True)
    thread.start()
    return thread, outcomes


# A completed move


def test_a_completed_move_sends_every_setpoint_once_in_order_on_time(rig):
    began = rig.clock.monotonic()
    outcome = rig.run()
    assert (outcome.status, outcome.reason, outcome.index) == ("completed", "", 25)
    assert rig.sent() == PATH
    first = began + START_READ_S
    on_time = [first + i * PERIOD for i in range(len(PATH))]
    assert rig.times() == pytest.approx(on_time, abs=1e-9)
    assert rig.arm.holds == []
    assert rig.missed == []


def test_a_completed_move_ends_settled_with_a_freshly_measured_pose(rig):
    outcome = rig.run()
    tol = rig.config.executor.settle_tol_rad
    assert outcome.end_q == pytest.approx(PATH[-1], abs=tol)
    assert outcome.end_q == pytest.approx(rig.arm.pose(), abs=0.005)
    # The end pose comes from a full read made after the last setpoint.
    assert rig.history.age(rig.clock.monotonic()) <= START_READ_S + 1e-9
    newest = rig.history.latest()
    assert [reading.value for reading in newest] == outcome.end_q
    assert all(reading.t_start > rig.times()[-1] for reading in newest)


def test_each_setpoint_is_checked_against_the_one_before_it_over_one_period(rig):
    checked = []
    real = rig.agent.check_setpoint

    def check(q_prev: Sequence[float], q_next: Sequence[float], dt: float) -> None:
        # Checked just before it is sent: the setpoints before it have gone out.
        checked.append((list(q_prev), list(q_next), dt, len(rig.arm.sent)))
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.agent, "check_setpoint", check)
    rig.run()
    steps = zip([REST] + PATH, PATH)
    assert checked == [(a, b, PERIOD, i) for i, (a, b) in enumerate(steps)]


def test_numpy_setpoints_are_sent_as_they_are(rig):
    outcome = rig.run(np.array(PATH), start_q=np.zeros(N))
    assert outcome.status == "completed"
    assert rig.sent() == PATH


def test_a_start_pose_within_the_tolerance_is_accepted(rig):
    # Near enough to the claimed start that the first step is within the speed.
    rig.arm.set_pose([0.001, -0.005, 0.0, 0.0, 0.0, 0.0])
    assert rig.run().status == "completed"


def test_a_second_move_can_follow_the_first(rig):
    assert rig.run().status == "completed"
    back = ramp(0.0, 25, start=PATH[-1])
    outcome = rig.run(back, start_q=PATH[-1])
    assert outcome.status == "completed"
    assert rig.sent() == PATH + back


def test_the_recovery_owner_runs_on_its_checker_without_a_gate(rig):
    rig.authority.set_owner(Owner.RECOVERY, "return to rest")
    slow = ramp(0.04, 25)  # 0.08 rad/s, under the recovery 0.1 rad/s
    outcome = rig.run(slow, owner=Owner.RECOVERY, checker=rig.checkers.recovery)
    assert outcome.status == "completed"
    assert rig.sent() == slow


def test_the_owner_may_be_given_as_its_string(rig):
    assert rig.run(owner="agent").status == "completed"


def test_an_unknown_owner_is_a_caller_error(rig):
    with pytest.raises(ValueError, match=r"robot"):
        rig.run(owner="robot")
    assert rig.run().status == "completed"  # the executor is not left busy


def test_the_grasp_region_is_set_for_the_move_and_cleared_after_it(
    rig, leaving, watches
):
    during = []
    rig.on_send(lambda: during.append((rig.agent.grasp_region, rig.agent.in_motion)))
    outcome = rig.run(grasp_region=REGION)
    assert outcome.status == "completed"
    assert during == [(REGION, True)] * len(PATH)
    assert leaving == ["region on", "region off", "end"]
    assert rig.agent.grasp_region is None
    assert not rig.agent.in_motion
    assert [watch.finished for watch in watches] == [True]


def test_a_grasp_region_that_cannot_be_cleared_does_not_change_the_outcome(
    rig, watches
):
    real = rig.agent.set_grasp_region

    def set_grasp_region(region: GraspRegion | None) -> None:
        if region is None:
            raise Refusal("the region is still in use")
        real(region)

    rig.patch.setattr(rig.agent, "set_grasp_region", set_grasp_region)
    outcome = rig.run(grasp_region=REGION)
    assert outcome.status == "completed"
    assert not rig.agent.in_motion
    assert [watch.finished for watch in watches] == [True]


# progress()


def test_progress_counts_the_setpoints_sent_and_is_idle_outside_a_move(rig):
    before, after = [], []
    rig.on_send(lambda: before.append(rig.executor.progress()), before=True)
    rig.on_send(lambda: after.append(rig.executor.progress()))
    assert rig.executor.progress() == IDLE
    rig.run()
    assert before == [(MOVE_ID, i, 25) for i in range(25)]
    # The count goes up once send_target has returned.
    assert after == [(MOVE_ID, i, 25) for i in range(25)]
    assert rig.executor.progress() == IDLE


def test_progress_shows_the_whole_move_sent_while_it_settles(rig):
    seen = set()
    real = rig.arm.read_joint

    def read(joint: int) -> object:
        if len(rig.arm.sent) == len(PATH):
            seen.add(rig.executor.progress())
        return real(joint)

    rig.patch.setattr(rig.arm, "read_joint", read)
    rig.run()
    assert seen == {(MOVE_ID, 25, 25)}


def test_progress_can_be_read_from_another_thread_while_a_move_runs(rig):
    seen = set()
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            seen.add(rig.executor.progress())
        seen.add(rig.executor.progress())  # after the move, whenever the thread ran

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    outcome = rig.run()
    done.set()
    reader.join(WAIT_S)
    assert not reader.is_alive()
    assert outcome.status == "completed"
    assert seen <= {IDLE} | {(MOVE_ID, i, 25) for i in range(26)}
    assert IDLE in seen


# Refused before anything is sent


@pytest.mark.parametrize(
    ("setpoints", "reason"),
    [
        pytest.param([], "no setpoints", id="empty"),
        pytest.param(None, "no setpoints", id="none"),
        pytest.param("stop", "no setpoints", id="text for the list"),
        pytest.param(path_with([0.0] * 5), "setpoint 3 needs 6", id="short row"),
        pytest.param(path_with([0.0] * 7), "setpoint 3 needs 6", id="long row"),
        pytest.param(path_with(0.0), "setpoint 3 needs 6", id="number for a row"),
        pytest.param(
            path_with([math.nan] + [0.0] * 5, 0), "setpoint 0 needs 6", id="nan"
        ),
        pytest.param(
            path_with([0.0] * 5 + [math.inf], 24), "setpoint 24 needs 6", id="inf"
        ),
        pytest.param(path_with([0.0] * 5 + ["0.0"]), "setpoint 3 needs 6", id="text"),
        pytest.param(path_with([0.0] * 5 + [None]), "setpoint 3 needs 6", id="null"),
        pytest.param(path_with([0.0] * 5 + [True]), "setpoint 3 needs 6", id="bool"),
        pytest.param(path_with([0.0] * 5 + [10**400]), "setpoint 3 needs 6", id="huge"),
    ],
)
def test_bad_setpoints_are_refused_with_nothing_sent_or_read(
    rig, leaving, setpoints, reason
):
    began = rig.clock.monotonic()
    outcome = rig.run(setpoints)
    assert outcome.status == "refused"
    assert reason in outcome.reason
    assert (outcome.index, outcome.end_q) == (0, None)
    assert rig.arm.sent == []
    assert rig.clock.monotonic() == began  # a joint read would have cost time
    assert leaving == ["region off", "end"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        pytest.param({"period_s": 0.01}, "configured period", id="shorter period"),
        pytest.param({"period_s": 0.021}, "configured period", id="longer period"),
        pytest.param(
            {"period_s": PERIOD * (1 + 1e-6)}, "configured period", id="near period"
        ),
        pytest.param({"period_s": math.nan}, "configured period", id="nan period"),
        pytest.param({"period_s": "0.02"}, "configured period", id="text period"),
        pytest.param({"start_q": [0.0] * 5}, "start_q needs 6", id="short start"),
        pytest.param(
            {"start_q": [math.nan] + [0.0] * 5}, "start_q needs 6", id="nan start"
        ),
        pytest.param({"start_tol": -0.01}, "start_tol", id="negative tolerance"),
        pytest.param({"start_tol": math.nan}, "start_tol", id="nan tolerance"),
        pytest.param(
            {"start_tol": 0.051},
            "start_tol is 0.051 rad, the limit is 0.05 rad",
            id="wide tolerance",
        ),
        pytest.param(
            {"expires_mono_s": math.inf}, "expires_mono_s", id="endless expiry"
        ),
        pytest.param({"expires_mono_s": "soon"}, "expires_mono_s", id="text expiry"),
        pytest.param({"owner": Owner.NOBODY}, "nobody", id="owner nobody"),
    ],
)
def test_bad_arguments_are_refused_with_nothing_sent_or_read(rig, changes, reason):
    began = rig.clock.monotonic()
    outcome = rig.run(**changes)
    assert outcome.status == "refused"
    assert reason in outcome.reason
    assert rig.arm.sent == []
    assert rig.clock.monotonic() == began


def test_the_widest_start_tolerance_allowed_is_accepted(rig):
    limit = rig.config.executor.max_start_tol_rad
    rig.arm.set_pose([0.0, 0.0, 0.0, 0.0, 0.0, limit - 0.001])
    # The setpoints leave the last joint where it is measured: the first one
    # is checked from the measured pose, not from the claimed start.
    rows = [[*q[:5], limit - 0.001] for q in PATH]
    assert rig.run(rows, start_tol=limit).status == "completed"


def test_the_start_tolerance_limit_is_a_setting(rig):
    wide = rig.new_executor(max_start_tol_rad=0.2)
    assert rig.run(executor=wide, start_tol=0.2).status == "completed"


# The move's own expiry


def test_an_expired_move_is_refused_with_nothing_sent_or_read(rig, leaving):
    began = rig.clock.monotonic()
    outcome = rig.run(expires_mono_s=began)
    assert outcome.status == "refused"
    assert outcome.reason == "the move expired 0.000 s ago"
    assert (outcome.index, outcome.end_q) == (0, None)
    assert rig.arm.sent == []
    assert rig.clock.monotonic() == began
    assert leaving == ["region off", "end"]


def test_a_move_whose_expiry_lies_ahead_runs_to_its_end(rig):
    # The start read takes 0.048 s; the expiry falls during the move itself.
    outcome = rig.run(expires_mono_s=rig.clock.monotonic() + 0.1)
    assert outcome.status == "completed"
    assert rig.sent() == PATH


def test_a_move_that_expires_during_its_start_read_sends_nothing(rig):
    outcome = rig.run(expires_mono_s=rig.clock.monotonic() + 0.03)
    assert outcome.status == "refused"
    assert "the move expired" in outcome.reason
    assert rig.arm.sent == []
    assert outcome.end_q == REST
    assert not rig.agent.in_motion


def test_a_move_that_expires_while_it_waits_for_a_permit_is_refused(bare):
    began = bare.clock.monotonic()
    outcome = bare.run(permit_gate=bare.gate, expires_mono_s=began + 0.1)
    assert outcome.status == "refused"
    assert "the move expired" in outcome.reason
    assert bare.arm.sent == []
    # The wait ended at the expiry, not at the end of the first wait.
    assert bare.clock.monotonic() - began == pytest.approx(0.1, abs=0.002)


def test_a_permit_does_not_start_a_move_that_has_expired(bare):
    bare.permit(len(PATH) - 1)
    outcome = bare.run(permit_gate=bare.gate, expires_mono_s=bare.clock.monotonic())
    assert outcome.status == "refused"
    assert bare.arm.sent == []


def test_a_period_that_differs_only_by_rounding_is_accepted(rig):
    assert rig.run(period_s=PERIOD * (1 + 1e-12)).status == "completed"


def test_more_setpoints_than_the_configured_limit_are_refused(rig):
    small = rig.new_executor(max_setpoints=10)
    outcome = rig.run(ramp(0.0704, 11), executor=small)
    assert outcome.status == "refused"
    assert "11 setpoints, the limit is 10" in outcome.reason
    assert rig.arm.sent == []
    assert rig.run(ramp(0.064, 10), executor=small).status == "completed"


def test_a_wrong_start_pose_is_refused_with_nothing_sent(rig):
    outcome = rig.run(start_q=[0.1] + [0.0] * 5)
    assert outcome.status == "refused"
    assert "start pose: joint index 0 is at 0.0000 rad" in outcome.reason
    assert rig.arm.sent == []
    assert outcome.end_q == REST  # the pose that was measured
    assert [reading.value for reading in rig.history.latest()] == REST
    assert not rig.agent.in_motion


def test_a_start_pose_just_outside_the_tolerance_is_refused(rig):
    rig.arm.set_pose([0.0, 0.0, 0.0, 0.0, 0.0, START_TOL + 0.001])
    outcome = rig.run()
    assert outcome.status == "refused"
    assert "joint index 5" in outcome.reason
    assert rig.arm.sent == []


def test_a_failed_start_read_is_refused_with_nothing_sent(rig):
    rig.arm.fail_reads(2, 1)
    outcome = rig.run()
    assert outcome.status == "refused"
    assert "the start pose could not be read" in outcome.reason
    assert (rig.arm.sent, outcome.end_q) == ([], None)


def test_a_start_read_without_a_valid_value_for_a_joint_is_refused(rig):
    real = rig.arm.read_all

    def read_all() -> list:
        readings = real()
        readings[4] = dataclasses.replace(readings[4], valid=False)
        return readings

    rig.patch.setattr(rig.arm, "read_all", read_all)
    outcome = rig.run()
    assert outcome.status == "refused"
    assert outcome.reason == "the start pose read left a joint out"
    assert rig.arm.sent == []


def test_with_the_motors_off_nothing_is_sent(rig):
    unpowered = SimArm(n_joints=N, clock=rig.clock, q0=REST)  # never connected
    executor = MoveExecutor(
        unpowered, rig.authority, None, rig.config.executor, rig.clock
    )
    outcome = rig.run(executor=executor)
    assert (outcome.status, outcome.reason) == ("refused", "the motors are off")
    assert unpowered.sent == []


def test_a_latched_checker_refuses_the_move_with_nothing_sent(rig):
    rig.agent.latch("motor fault")
    outcome = rig.run()
    assert outcome.status == "refused"
    assert "fault-stop latched: motor fault" in outcome.reason
    assert rig.arm.sent == []


def test_an_unfed_agent_watchdog_refuses_the_move_with_nothing_sent(bare):
    outcome = bare.run()
    assert outcome.status == "refused"
    assert "perception watchdog: never fed" in outcome.reason
    assert bare.arm.sent == []


def test_a_second_run_during_a_move_is_refused_and_disturbs_nothing(rig, leaving):
    nested = []

    def run_again() -> None:
        nested.append(rig.run(grasp_region=None))
        nested.append((rig.agent.in_motion, rig.agent.grasp_region, list(leaving)))

    rig.on_send(run_again, index=3)
    outcome = rig.run(grasp_region=REGION)
    assert nested[0] == MoveOutcome("refused", "another move is running")
    assert nested[1] == (True, REGION, ["region on"])
    assert outcome.status == "completed"
    assert rig.sent() == PATH


def test_no_move_starts_while_the_executor_is_claimed_idle(rig, leaving):
    began = rig.clock.monotonic()
    with rig.executor.claim_idle() as idle:
        assert idle is True
        outcome = rig.run()
        assert outcome == MoveOutcome("refused", "another move is running")
        assert rig.executor.progress() == ("", 0, 0)
    assert rig.arm.sent == []
    assert rig.clock.monotonic() == began  # nothing was read either
    assert leaving == []  # the checker was not touched
    assert rig.run().status == "completed"


def test_a_running_move_cannot_be_claimed_idle_and_goes_on(rig):
    claims: list[bool] = []

    def claim() -> None:
        with rig.executor.claim_idle() as idle:
            claims.append(idle)

    rig.on_send(claim, index=3)
    assert rig.run().status == "completed"
    assert claims == [False]
    assert rig.sent() == PATH
    # The failed claim took nothing: the next one succeeds.
    with rig.executor.claim_idle() as idle:
        assert idle is True


def test_the_idle_claim_is_given_up_when_its_block_raises(rig):
    with pytest.raises(RuntimeError, match=r"driver"):
        with rig.executor.claim_idle():
            raise RuntimeError("driver")
    assert rig.run().status == "completed"


@pytest.mark.parametrize("first_wait_s", [-0.1, math.nan, math.inf, None])
def test_a_first_wait_that_is_not_a_time_is_rejected(rig, first_wait_s):
    with pytest.raises(ValueError, match=r"first_wait_s"):
        MoveExecutor(
            rig.arm,
            rig.authority,
            rig.history,
            rig.config.executor,
            rig.clock,
            first_wait_s=first_wait_s,
        )


# Stop conditions


def test_a_move_admitted_before_a_stop_sends_nothing(rig):
    admitted = rig.authority.snapshot()
    rig.authority.raise_stop("operator", "console")
    began = rig.clock.monotonic()
    outcome = rig.run(admitted=admitted)
    assert outcome.status == "stopped"
    assert "stop number is now 1, the move was admitted at 0" in outcome.reason
    assert rig.arm.sent == []
    assert rig.clock.monotonic() == began


def test_a_new_stop_number_mid_move_stops_it_at_the_next_step(rig):
    rig.on_send(lambda: rig.authority.raise_stop("operator", "console"), index=4)
    outcome = rig.run()
    assert outcome.status == "stopped"
    assert "stop number is now 1" in outcome.reason
    assert rig.sent() == PATH[:5]
    assert outcome.index == 5
    # The last setpoint stands: the arm holds there with nothing more sent.
    assert rig.arm.last_target()[0] == PATH[4]
    assert rig.arm.holds == []
    assert outcome.end_q == pytest.approx(PATH[4], abs=0.03)


def test_a_new_owner_number_mid_move_stops_it_at_the_next_step(rig):
    rig.on_send(lambda: rig.authority.set_owner(Owner.AGENT, "new task"), index=4)
    outcome = rig.run()
    assert outcome.status == "stopped"
    assert "owner number is now 2, the move was admitted at 1" in outcome.reason
    assert rig.sent() == PATH[:5]


def test_a_move_for_an_owner_who_is_not_the_control_owner_sends_nothing(rig):
    outcome = rig.run(
        ramp(0.04, 25), owner=Owner.RECOVERY, checker=rig.checkers.recovery
    )
    assert outcome.status == "stopped"
    assert outcome.reason == "control owner is agent, not recovery"
    assert rig.arm.sent == []


def test_should_stop_ends_the_move_with_its_reason(rig):
    def should_stop() -> str | None:
        return "operator stop" if len(rig.arm.sent) == 5 else None

    outcome = rig.run(should_stop=should_stop)
    assert (outcome.status, outcome.reason) == ("stopped", "operator stop")
    assert rig.sent() == PATH[:5]
    assert outcome.index == 5


def test_should_stop_with_an_empty_reason_still_stops(rig):
    outcome = rig.run(should_stop=lambda: "")
    assert (outcome.status, outcome.reason) == ("stopped", "stop requested")
    assert rig.arm.sent == []


def test_a_stop_while_settling_ends_the_move_as_stopped(rig):
    def should_stop() -> str | None:
        return "operator stop" if len(rig.arm.sent) == len(PATH) else None

    outcome = rig.run(should_stop=should_stop)
    assert (outcome.status, outcome.index) == ("stopped", 25)
    assert rig.sent() == PATH


def test_a_halt_mid_move_refuses_the_next_setpoint(rig):
    rig.on_send(lambda: rig.agent.halt("look out"), index=4)
    outcome = rig.run()
    assert (outcome.status, outcome.reason) == ("refused", "halted: look out")
    assert rig.sent() == PATH[:5]


def test_a_refusal_that_comes_with_a_stop_is_reported_as_stopped(rig):
    real = rig.agent.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        if len(rig.arm.sent) == 4:
            rig.authority.raise_stop("operator", "console")
            rig.agent.halt("operator")
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.agent, "check_setpoint", check)
    outcome = rig.run()
    assert outcome.status == "stopped"
    assert "stop number" in outcome.reason
    assert rig.sent() == PATH[:4]


def test_a_halt_before_the_move_is_cleared_by_it(rig):
    rig.agent.halt("an earlier move")
    assert rig.run().status == "completed"


def test_a_halt_since_the_quoted_halt_count_refuses_the_move(rig):
    quoted = rig.agent.halt_count()
    rig.agent.halt("stop")
    outcome = rig.run(halt_count=quoted)
    assert outcome.status == "refused"
    assert "halted since this motion was admitted" in outcome.reason
    assert rig.arm.sent == []


# Permits


def test_without_a_permit_nothing_is_sent(bare):
    began = bare.clock.monotonic()
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert outcome.reason == "no permit for this move (waited 0.5 s)"
    assert (bare.arm.sent, outcome.end_q) == ([], None)
    # The wait ran on the clock, and no joint was read after it.
    waited = bare.clock.monotonic() - began
    assert waited == pytest.approx(bare.config.permits.first_wait_s, abs=0.002)
    assert not bare.agent.in_motion


def test_with_no_first_wait_a_missing_permit_is_refused_at_once(bare):
    began = bare.clock.monotonic()
    outcome = bare.run(
        executor=bare.new_executor(first_wait_s=0.0), permit_gate=bare.gate
    )
    assert outcome.status == "refused"
    assert bare.arm.sent == []
    assert bare.clock.monotonic() == began


def test_a_held_permit_starts_the_move_at_once_and_feeds_the_watchdog(bare):
    began = bare.clock.monotonic()
    bare.permit(len(PATH) - 1)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "completed"
    assert bare.sent() == PATH
    assert bare.times()[0] == pytest.approx(began + START_READ_S, abs=1e-9)


def test_a_permit_for_another_move_does_not_start_this_one(bare):
    bare.permit(len(PATH) - 1, move_id="move-0")
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert "no permit for this move" in outcome.reason
    assert bare.arm.sent == []


def test_the_move_holds_at_the_last_setpoint_its_permit_covers(bare):
    bare.permit(9)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert outcome.reason == "permit reaches setpoint 9, not 10"
    assert bare.sent() == PATH[:10]
    assert outcome.index == 10
    assert bare.arm.last_target()[0] == PATH[9]
    assert bare.arm.holds == []
    assert outcome.end_q == pytest.approx(PATH[9], abs=0.03)


def test_a_further_permit_mid_move_lets_it_go_on(bare):
    bare.permit(9)
    bare.on_send(lambda: bare.permit(len(PATH) - 1), index=5)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "completed"
    assert bare.sent() == PATH


def test_no_setpoint_is_sent_once_the_permit_has_expired(bare):
    # Setpoints go out at 0.048, 0.068, ... s; the permit runs out between two.
    permit = bare.permit(len(PATH) - 1, expires_in_s=START_READ_S + 0.09)
    outcome = bare.run(permit_gate=bare.gate)
    assert (outcome.status, outcome.reason) == ("refused", "permit expired")
    assert bare.sent() == PATH[:5]
    assert all(t < permit.expires_mono_s for t in bare.times())
    assert bare.times()[-1] + PERIOD > permit.expires_mono_s


def test_a_permit_that_expires_during_the_start_read_sends_nothing(bare):
    bare.permit(len(PATH) - 1, expires_in_s=START_READ_S / 2)
    outcome = bare.run(permit_gate=bare.gate)
    assert (outcome.status, outcome.reason) == ("refused", "permit expired")
    assert bare.arm.sent == []


def refused_permit(rig: Rig, move_id: str = MOVE_ID, **fields: object) -> Permit:
    """Give the gate a permit it refuses: its evidence is 4 s old."""
    state = rig.authority.snapshot()
    permit = Permit(
        move_id=move_id,
        session_id=state.session_id,
        owner_number=state.owner_number,
        stop_number=state.stop_number,
        through_index=len(PATH) - 1,
        evidence_wall_s=rig.clock.wall() - 4.0,
        expires_mono_s=rig.clock.monotonic() + 5.0,
    )
    assert rig.gate.update(permit) == "permit evidence is too old"
    return permit


def test_a_move_without_a_permit_names_why_its_last_permit_was_refused(bare):
    refused_permit(bare)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert outcome.reason == (
        "no permit for this move; its last permit was refused: permit evidence "
        "is too old (waited 0.5 s)"
    )
    assert bare.arm.sent == []


def test_a_refused_permit_of_another_move_is_not_named(bare):
    refused_permit(bare, move_id="move-0")
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.reason == "no permit for this move (waited 0.5 s)"


def test_a_move_that_runs_out_of_permit_names_the_refused_renewal(bare):
    bare.permit(9)
    bare.on_send(lambda: refused_permit(bare), index=5)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert outcome.reason == (
        "permit reaches setpoint 9, not 10; its last permit was refused: permit "
        "evidence is too old"
    )
    assert bare.sent() == PATH[:10]
    assert not bare.agent.in_motion


def test_evidence_that_grows_too_old_stops_the_move(bare):
    bare.permit(len(PATH) - 1, evidence_age_s=2.9)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "refused"
    assert outcome.reason == "permit evidence is too old"
    assert bare.sent() == PATH[:3]


def test_a_stop_after_the_permit_wins_over_the_permit(bare):
    bare.permit(len(PATH) - 1)
    bare.on_send(lambda: bare.authority.raise_stop("operator", "console"), index=4)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "stopped"
    assert bare.sent() == PATH[:5]


def stop_inside_covers(rig: Rig, index: int) -> None:
    """Raise a STOP just as the gate is asked about setpoint `index`."""
    real = rig.gate.covers

    def covers(move_id: str, asked: int, now: float) -> str | None:
        if asked == index:
            rig.authority.raise_stop("operator", "console")
        return real(move_id, asked, now)

    rig.patch.setattr(rig.gate, "covers", covers)


def test_a_stop_that_the_permit_gate_sees_first_is_reported_as_stopped(bare):
    bare.permit(len(PATH) - 1)
    stop_inside_covers(bare, 5)
    outcome = bare.run(permit_gate=bare.gate)
    assert outcome.status == "stopped"
    assert "stop number is now 1" in outcome.reason
    assert bare.sent() == PATH[:5]


def test_a_stop_that_lands_as_the_first_wait_runs_out_is_reported_as_stopped(bare):
    stop_inside_covers(bare, 0)
    outcome = bare.run(
        executor=bare.new_executor(first_wait_s=0.0), permit_gate=bare.gate
    )
    assert outcome.status == "stopped"
    assert bare.arm.sent == []


def test_a_permit_that_arrives_during_the_wait_starts_the_move(rig):
    # The watchdog is fed beforehand: the gate holds a permit a moment before it
    # feeds the watchdog, and the waiting move may start in between.
    executor = rig.new_executor(first_wait_s=30.0)
    arrived = threading.Event()
    probe = StopProbe()
    thread, outcomes = in_thread(
        lambda: rig.run(
            executor=executor,
            permit_gate=rig.gate,
            permit_arrived=arrived,
            should_stop=probe,
        )
    )
    assert probe.waiting.wait(WAIT_S)
    assert rig.arm.sent == []
    rig.permit(len(PATH) - 1)
    arrived.set()
    thread.join(WAIT_S)
    assert not thread.is_alive()
    assert outcomes[0].status == "completed"
    assert rig.sent() == PATH


def test_the_wait_ends_once_the_clock_passes_the_first_wait(bare):
    executor = bare.new_executor(first_wait_s=30.0)
    arrived = threading.Event()
    probe = StopProbe()
    thread, outcomes = in_thread(
        lambda: bare.run(
            executor=executor,
            permit_gate=bare.gate,
            permit_arrived=arrived,
            should_stop=probe,
        )
    )
    assert probe.waiting.wait(WAIT_S)
    bare.clock.advance(30.0)
    thread.join(WAIT_S)
    assert not thread.is_alive()
    assert outcomes[0].status == "refused"
    assert "no permit for this move" in outcomes[0].reason
    assert bare.arm.sent == []


def test_with_the_event_the_wait_runs_out_by_itself_and_leaves_the_clock_alone(bare):
    began = bare.clock.monotonic()
    outcome = bare.run(
        executor=bare.new_executor(first_wait_s=0.003),
        permit_gate=bare.gate,
        permit_arrived=threading.Event(),
    )
    assert outcome.status == "refused"
    assert "no permit for this move" in outcome.reason
    assert bare.arm.sent == []
    assert bare.clock.monotonic() == began


def test_a_set_event_without_a_covering_permit_starts_nothing(bare):
    arrived = threading.Event()
    arrived.set()
    bare.permit(len(PATH) - 1, move_id="move-0")
    outcome = bare.run(
        executor=bare.new_executor(first_wait_s=0.003),
        permit_gate=bare.gate,
        permit_arrived=arrived,
    )
    assert outcome.status == "refused"
    assert bare.arm.sent == []
    assert not arrived.is_set()


@pytest.mark.parametrize("with_event", [False, True])
def test_a_stop_during_the_wait_for_the_first_permit_sends_nothing(bare, with_event):
    probe = StopProbe("operator stop")
    outcome = bare.run(
        executor=bare.new_executor(first_wait_s=30.0),
        permit_gate=bare.gate,
        permit_arrived=threading.Event() if with_event else None,
        should_stop=probe,
    )
    assert (outcome.status, outcome.reason) == ("stopped", "operator stop")
    assert bare.arm.sent == []
    assert probe.calls == 3


def test_a_stop_number_raised_during_the_wait_is_reported_as_stopped(bare):
    def should_stop() -> str | None:
        bare.authority.raise_stop("operator", "console")
        return None

    outcome = bare.run(permit_gate=bare.gate, should_stop=should_stop)
    assert outcome.status == "stopped"
    assert "stop number" in outcome.reason
    assert bare.arm.sent == []


# Late setpoints


def test_a_delay_under_the_small_limit_shifts_the_rest_and_is_not_counted(rig):
    rig.stall_in_reads(5, 0.006)  # the step overruns its deadline by 2 ms
    outcome = rig.run()
    assert outcome.status == "completed"
    assert rig.sent() == PATH
    first = rig.times()[0]
    shifted = [first + i * PERIOD + (0.002 if i > 5 else 0.0) for i in range(25)]
    assert rig.times() == pytest.approx(shifted, abs=1e-9)
    assert rig.missed == []  # too small to count as a missed deadline


def test_a_delay_over_the_small_limit_shifts_the_rest_of_the_move(rig):
    rig.stall_in_reads(5, 0.03)  # the step overruns its deadline by 18 ms
    outcome = rig.run()
    assert outcome.status == "completed"
    assert rig.sent() == PATH  # none skipped, none twice
    first = rig.times()[0]
    shifted = [first + i * PERIOD + (0.018 if i > 5 else 0.0) for i in range(25)]
    assert rig.times() == pytest.approx(shifted, abs=1e-9)
    assert rig.missed == [1]


def test_a_delay_over_the_large_limit_ends_the_move_in_a_hold(rig):
    rig.stall_in_reads(5, 0.2)  # the step overruns its deadline by 188 ms
    outcome = rig.run()
    assert outcome.status == "late"
    assert outcome.reason == "setpoint 6 is 0.188 s late, over the limit of 0.1 s"
    assert rig.sent() == PATH[:6]
    assert outcome.index == 6
    assert rig.arm.last_target()[0] == PATH[5]
    assert rig.arm.holds == []
    assert outcome.end_q == pytest.approx(PATH[5], abs=0.03)
    assert rig.missed == [1]


def test_a_delay_just_under_the_large_limit_still_shifts(rig):
    rig.stall_in_reads(5, 0.1)  # 88 ms over the deadline
    outcome = rig.run()
    assert outcome.status == "completed"
    assert rig.gaps()[5] == pytest.approx(PERIOD + 0.088, abs=1e-9)


def test_a_setpoint_that_goes_out_late_takes_the_later_ones_with_it(rig):
    rig.on_send(lambda: rig.clock.advance(0.012), index=5, before=True)
    outcome = rig.run()
    assert outcome.status == "completed"
    first = rig.times()[0]
    shifted = [first + i * PERIOD + (0.012 if i >= 5 else 0.0) for i in range(25)]
    assert rig.times() == pytest.approx(shifted, abs=1e-9)
    assert rig.missed == [1]


@pytest.mark.parametrize("where", ["reads", "send"])
@pytest.mark.parametrize("seconds", [0.003, 0.006, 0.009, 0.012, 0.03, 0.09])
def test_setpoints_never_follow_closer_than_a_period(rig, where, seconds):
    for index in (2, 3, 11):
        if where == "reads":
            rig.stall_in_reads(index, seconds)
        else:
            rig.on_send(lambda: rig.clock.advance(seconds), index=index, before=True)
    outcome = rig.run()
    assert outcome.status == "completed"
    assert rig.sent() == PATH
    # A send that took no longer than the small delay moves nothing back, so
    # the setpoint after it follows that much sooner.
    small = rig.config.executor.small_delay_s
    sooner = seconds if where == "send" and seconds <= small else 0.0
    assert min(rig.gaps()) == pytest.approx(PERIOD - sooner, abs=1e-9)


def test_a_slow_check_does_not_bring_setpoints_closer_than_allowed(rig):
    real = rig.agent.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        if len(rig.arm.sent) in (5, 6):
            rig.clock.advance(0.009)
        real(q_prev, q_next, dt)

    rig.patch.setattr(rig.agent, "check_setpoint", check)
    outcome = rig.run()
    assert outcome.status == "completed"
    assert min(rig.gaps()) >= PERIOD - 1e-9


# Contact, refusals while moving, settling


def test_a_blocked_joint_ends_the_move_in_contact_with_a_hold_at_the_measured_pose(
    rig,
):
    rig.arm.block_joint(0, 0.05)
    outcome = rig.run(LONG_PATH)
    assert outcome.status == "contact"
    assert outcome.reason.startswith("contact: joint index 0")
    assert 0 < outcome.index < len(LONG_PATH)
    assert rig.sent() == LONG_PATH[: outcome.index]
    assert len(rig.arm.holds) == 1
    held, _ = rig.arm.holds[0]
    assert held[0] == pytest.approx(0.05, abs=1e-6)
    assert rig.arm.last_target()[0] == held
    assert outcome.end_q == pytest.approx(held, abs=1e-6)


def test_a_contact_while_settling_also_ends_in_a_hold(rig):
    rig.arm.block_joint(0, 0.05)  # 0.11 rad short of the goal
    outcome = rig.run()
    assert outcome.status == "contact"
    assert rig.sent() == PATH
    assert outcome.index == 25
    assert len(rig.arm.holds) == 1


def test_a_hold_that_fails_once_is_tried_again(rig):
    rig.arm.block_joint(0, 0.05)
    real = rig.arm.hold_measured
    calls = []

    def hold() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise ReadFailed("joint index 2: the read failed")
        real()

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    outcome = rig.run(LONG_PATH)
    assert outcome.status == "contact"
    assert "hold" not in outcome.reason
    assert (len(calls), len(rig.arm.holds)) == (2, 1)


def test_a_hold_that_keeps_failing_is_named_in_the_reason(rig):
    rig.arm.block_joint(0, 0.05)
    calls = []

    def hold() -> None:
        calls.append(1)
        raise ReadFailed("joint index 2: the read failed")

    rig.patch.setattr(rig.arm, "hold_measured", hold)
    outcome = rig.run(LONG_PATH)
    assert outcome.status == "contact"
    assert outcome.reason.startswith("contact: joint index 0")
    assert "the hold at the measured pose failed: joint index 2" in outcome.reason
    assert len(calls) == 2


def test_a_setpoint_outside_the_limits_is_refused_and_nothing_further_is_sent(rig):
    start = [1.45, 0.0, 0.0, 0.0, 0.0, 0.0]
    rig.arm.set_pose(start)
    path = ramp(1.61, 25, start=start)  # leaves the agent envelope of 1.5 rad
    outcome = rig.run(path, start_q=start)
    assert outcome.status == "refused"
    assert "outside the agent envelope" in outcome.reason
    assert rig.sent() == path[:7]
    assert all(q[0] <= 1.5 for q in rig.sent())
    assert outcome.index == 7
    assert rig.arm.last_target()[0] == path[6]
    assert rig.arm.holds == []


def test_a_setpoint_that_is_too_fast_is_refused_and_nothing_further_is_sent(rig):
    path = [list(q) for q in PATH]
    path[10][3] = 0.05  # 2.5 rad/s
    outcome = rig.run(path)
    assert outcome.status == "refused"
    assert "speed limit" in outcome.reason
    assert rig.sent() == path[:10]


def test_lost_joint_feedback_ends_the_move_as_a_fault(rig):
    rig.on_send(lambda: rig.arm.fail_reads(1, 10_000), index=0)
    outcome = rig.run(LONG_PATH)
    assert outcome.status == "failed"
    assert outcome.reason.startswith("fault: joint feedback lost: joint index 1")
    assert rig.sent() == LONG_PATH[: outcome.index]
    assert 0 < outcome.index < len(LONG_PATH)
    assert outcome.end_q is None  # the end pose read failed as well
    assert rig.arm.holds == []


def test_a_read_failure_that_escapes_the_watch_ends_the_move_refused(rig):
    class Watch:
        finished = False

        def after_send(self, index: int, q: object, deadline_mono: float) -> list:
            if index == 3:
                raise ReadFailed("joint index 4: no answer")
            return []

        def finish(self) -> None:
            self.finished = True

    watch = Watch()
    rig.patch.setattr(rig.agent, "contact_watch", lambda arm, q, history: watch)
    outcome = rig.run()
    assert outcome.status == "refused"
    assert outcome.reason == "a joint read failed: joint index 4: no answer"
    assert rig.sent() == PATH[:4]
    assert watch.finished


def test_a_joint_that_stays_short_of_the_last_setpoint_is_not_settled(rig):
    rig.arm.block_joint(0, 0.11)  # 0.05 rad short: over the tolerance, no contact
    outcome = rig.run()
    assert outcome.status == "not_settled"
    assert outcome.reason == (
        "joint index 0 is not within 0.03 rad of the last setpoint after 2 s"
    )
    assert rig.sent() == PATH
    assert outcome.index == 25
    assert rig.arm.holds == []
    assert outcome.end_q[0] == pytest.approx(0.11, abs=1e-6)
    timeout = rig.config.executor.settle_timeout_s
    settling = rig.clock.monotonic() - rig.times()[-1] - PERIOD - START_READ_S
    assert timeout <= settling < timeout + 2 * PERIOD


def test_settling_reads_the_joints_and_never_sends(rig):
    rig.arm.block_joint(0, 0.11)
    rig.run()
    after_last = [
        reading
        for reading in rig.history.since(rig.times()[-1] + PERIOD)
        if reading.joint == 0
    ]
    # Joint 0 is read every third step for the two seconds of settling.
    assert len(after_last) > 30
    assert len(rig.arm.sent) == len(PATH)


# Arm faults


def raise_fault() -> None:
    raise ArmFault("motor 3 over temperature")


def test_an_arm_fault_while_sending_ends_the_move_failed(rig, leaving, watches):
    rig.on_send(raise_fault, index=4, before=True)
    outcome = rig.run(grasp_region=REGION)
    assert outcome.status == "failed"
    assert outcome.reason == "fault: motor 3 over temperature"
    assert rig.sent() == PATH[:4]
    assert (outcome.index, outcome.end_q) == (4, None)
    assert rig.arm.holds == []
    assert leaving[-2:] == ["region off", "end"]
    assert [watch.finished for watch in watches] == [True]


def test_an_arm_fault_in_the_start_read_ends_the_move_failed(rig):
    rig.patch.setattr(rig.arm, "read_all", raise_fault)
    outcome = rig.run()
    assert outcome.status == "failed"
    assert outcome.reason.startswith("fault:")
    assert rig.arm.sent == []


def test_an_arm_fault_in_a_joint_read_ends_the_move_failed(rig):
    real = rig.arm.read_joint

    def read(joint: int) -> object:
        if len(rig.arm.sent) == 7:
            raise_fault()
        return real(joint)

    rig.patch.setattr(rig.arm, "read_joint", read)
    outcome = rig.run()
    assert outcome.status == "failed"
    assert outcome.reason.startswith("fault:")
    assert rig.sent() == PATH[:7]


def test_an_arm_fault_in_the_hold_after_a_contact_ends_the_move_failed(rig):
    rig.arm.block_joint(0, 0.05)
    rig.patch.setattr(rig.arm, "hold_measured", raise_fault)
    outcome = rig.run(LONG_PATH)
    assert outcome.status == "failed"
    assert outcome.reason.startswith("fault:")


# Every way a move ends


def completes(rig: Rig) -> dict:
    return {}


def has_no_setpoints(rig: Rig) -> dict:
    return {"setpoints": []}


def quotes_an_old_stop_number(rig: Rig) -> dict:
    admitted = rig.authority.snapshot()
    rig.authority.raise_stop("operator", "console")
    return {"admitted": admitted}


def gets_no_permit(rig: Rig) -> dict:
    return {"permit_gate": rig.gate}


def starts_elsewhere(rig: Rig) -> dict:
    return {"start_q": [0.1] + [0.0] * 5}


def cannot_read_its_start(rig: Rig) -> dict:
    rig.arm.fail_reads(2, 1)
    return {}


def meets_a_latch(rig: Rig) -> dict:
    rig.agent.latch("motor fault")
    return {}


def is_stopped(rig: Rig) -> dict:
    return {"should_stop": lambda: "stop" if len(rig.arm.sent) == 5 else None}


def is_halted(rig: Rig) -> dict:
    rig.on_send(lambda: rig.agent.halt("look out"), index=4)
    return {}


def outruns_its_permit(rig: Rig) -> dict:
    rig.permit(9)
    return {"permit_gate": rig.gate}


def touches_something(rig: Rig) -> dict:
    rig.arm.block_joint(0, 0.05)
    return {"setpoints": LONG_PATH}


def falls_behind(rig: Rig) -> dict:
    rig.stall_in_reads(5, 0.2)
    return {}


def stays_short(rig: Rig) -> dict:
    rig.arm.block_joint(0, 0.11)
    return {}


def meets_a_fault(rig: Rig) -> dict:
    rig.on_send(raise_fault, index=4, before=True)
    return {}


ENDINGS = [
    (completes, "completed"),
    (has_no_setpoints, "refused"),
    (quotes_an_old_stop_number, "stopped"),
    (gets_no_permit, "refused"),
    (starts_elsewhere, "refused"),
    (cannot_read_its_start, "refused"),
    (meets_a_latch, "refused"),
    (is_stopped, "stopped"),
    (is_halted, "refused"),
    (outruns_its_permit, "refused"),
    (touches_something, "contact"),
    (falls_behind, "late"),
    (stays_short, "not_settled"),
    (meets_a_fault, "failed"),
]


@pytest.mark.parametrize(
    ("arrange", "status"), ENDINGS, ids=[arrange.__name__ for arrange, _ in ENDINGS]
)
def test_however_a_move_ends_it_leaves_everything_in_order(
    rig, leaving, watches, arrange, status
):
    changes = arrange(rig)
    setpoints = changes.get("setpoints", PATH)
    owner_before = rig.authority.snapshot()
    outcome = rig.run(grasp_region=REGION, **changes)
    assert outcome.status == status
    # Setpoints went out once each, in order, and the outcome counts them.
    assert outcome.index == len(rig.arm.sent)
    assert rig.sent() == setpoints[: outcome.index]
    # The checker and the watch are left as before the move.
    assert leaving[-2:] == ["region off", "end"]
    assert rig.agent.grasp_region is None
    assert not rig.agent.in_motion
    assert all(watch.finished for watch in watches)
    assert rig.executor.progress() == IDLE
    # The executor moves the arm and nothing else.
    assert rig.authority.snapshot() == owner_before
    assert rig.arm.torque_on()
    assert rig.arm.released_at == []
    assert (rig.arm.jaw_commands, rig.arm.light_holds) == ([], [])
    # It is free for the next move.
    assert rig.run([]).reason == "the move has no setpoints"


# The first setpoint is checked from the measured pose


def test_the_first_setpoint_is_checked_from_the_measured_pose(rig):
    # The arm is at 0.0; the move claims a start 0.045 rad away, inside the
    # widest start tolerance, and steps on from there at the agent's speed.
    claimed = [0.045] * N
    rows = [[0.045 + 0.008 * i] * N for i in range(1, 11)]
    outcome = rig.run(rows, start_q=claimed, start_tol=0.05)
    assert outcome.status == "refused"
    assert outcome.reason == (
        "setpoint: joint index 0 would move at 2.650 rad/s, over the agent speed "
        "limit 0.400 rad/s"
    )
    assert rig.arm.sent == []


def test_the_setpoints_can_be_planned_from_the_measured_start_pose(rig):
    rig.arm.set_pose([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    seen: list[list[float]] = []

    def replan(measured: list[float]) -> list[list[float]]:
        seen.append(list(measured))
        return ramp(0.16, 25, start=measured)

    # 0.004 rad in one step is 0.2 rad/s: PATH itself would pass, a recovery
    # move at 0.1 rad/s would not.
    outcome = rig.run(replan=replan)
    assert outcome.status == "completed"
    assert seen == [[0.004, 0.0, 0.0, 0.0, 0.0, 0.0]]
    assert rig.sent() == ramp(0.16, 25, start=seen[0])
    assert outcome.index == 25


@pytest.mark.parametrize(
    ("planned", "reason"),
    [
        (Refusal("the path to rest was refused"), "the path to rest was refused"),
        ([], "the move has no setpoints"),
        ([[0.0] * 5], "setpoint 0 needs 6 finite joint values"),
    ],
    ids=["refusal", "no setpoints", "short row"],
)
def test_a_plan_from_the_measured_pose_that_cannot_be_used_is_refused(
    rig, planned, reason
):
    def replan(measured: list[float]) -> object:
        if isinstance(planned, Refusal):
            raise planned
        return planned

    outcome = rig.run(replan=replan)
    assert (outcome.status, outcome.reason) == ("refused", reason)
    assert rig.arm.sent == []
    assert outcome.end_q == REST


# A stall between the checks and the send


def stall_in_check(rig: Rig, index: int, action: Callable[[], object]) -> None:
    """Call action() inside the check of setpoint `index`, after the check ran."""
    real = rig.agent.check_setpoint

    def check(q_prev: object, q_next: object, dt: float) -> None:
        real(q_prev, q_next, dt)
        if len(rig.arm.sent) == index:
            action()

    rig.patch.setattr(rig.agent, "check_setpoint", check)


def test_a_stall_over_the_large_limit_inside_the_check_ends_the_move_late(rig):
    stall_in_check(rig, 10, lambda: rig.clock.advance(0.15))
    outcome = rig.run()
    assert outcome.status == "late"
    assert outcome.reason == "setpoint 10 is 0.150 s late, over the limit of 0.1 s"
    assert rig.sent() == PATH[:10]
    assert rig.missed == [1]


def test_no_setpoint_goes_out_once_its_permit_expired_during_the_check(bare):
    # Setpoint 5 is due at 0.148 s; its permit runs out 6 ms later.
    permit = bare.permit(len(PATH) - 1, expires_in_s=START_READ_S + 0.106)
    stall_in_check(bare, 5, lambda: bare.clock.advance(0.05))
    outcome = bare.run(permit_gate=bare.gate)
    assert (outcome.status, outcome.reason) == ("refused", "permit expired")
    assert bare.sent() == PATH[:5]
    assert all(t < permit.expires_mono_s for t in bare.times())


def test_a_stop_raised_during_the_check_keeps_the_setpoint_back(rig):
    stall_in_check(rig, 10, lambda: rig.authority.raise_stop("operator", "console"))
    outcome = rig.run()
    assert outcome.status == "stopped"
    assert rig.sent() == PATH[:10]


def test_a_send_that_takes_longer_than_the_large_limit_ends_the_move_late(rig):
    rig.on_send(lambda: rig.clock.advance(0.15), index=10, before=True)
    outcome = rig.run()
    assert outcome.status == "late"
    assert outcome.reason == (
        "setpoint 10 went out 0.150 s late, over the limit of 0.1 s"
    )
    assert rig.sent() == PATH[:11]
    assert outcome.index == 11
    assert rig.missed == [1]


# The cost of the check and the send


@pytest.mark.parametrize("cost_s", [0.0002, 0.0012, 0.003, 0.0035])
def test_what_the_check_and_the_send_cost_does_not_stretch_the_move(rig, cost_s):
    # The step's two joint reads take 16 ms, so the cost leaves the period room.
    rig.on_send(lambda: rig.clock.advance(cost_s), before=True)
    outcome = rig.run()
    assert outcome.status == "completed"
    assert rig.sent() == PATH
    assert rig.gaps() == pytest.approx([PERIOD] * 24, abs=1e-9)
    assert rig.missed == []


def test_a_cost_that_changes_brings_setpoints_closer_by_less_than_the_small_delay(
    rig,
):
    costs = iter([0.004, 0.0] * 13)
    rig.on_send(lambda: rig.clock.advance(next(costs)), before=True)
    outcome = rig.run()
    assert outcome.status == "completed"
    assert min(rig.gaps()) == pytest.approx(PERIOD - 0.004, abs=1e-9)
    assert max(rig.gaps()) == pytest.approx(PERIOD + 0.004, abs=1e-9)
    # The move as a whole is not stretched.
    assert rig.times()[-1] - rig.times()[0] == pytest.approx(24 * PERIOD, abs=1e-9)
