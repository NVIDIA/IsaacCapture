# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A move among the robot app's other work, on the simulated arm.

The idle joint reads around a move and their period in a hold, and the grasp
region a MOVE may carry.
"""

from __future__ import annotations

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion
from isaaccapture_examples.agentic_remote.robot_app.app import RobotApp
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    JointReading,
    ReadFailed,
    StopLevel,
)
from conftest import APP_WAIT_S

REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
READ_S = 0.008  # the simulated arm's time for one joint read
NO_ANSWER_S = 0.05  # how long a joint read waits for an answer that never comes
IDLE_PERIOD_S = 0.02  # state.idle_read_period_s


def state_after(h, command_id: str):
    """The ROBOT_STATE that shows this command's final RESULT."""
    return h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == command_id, APP_WAIT_S
    )


# The first setpoint


def test_a_move_that_claims_a_start_away_from_the_measured_pose_sends_nothing(
    robot_app,
):
    h = robot_app
    h.new_task()
    # The arm is at 0.0. The claimed start is inside the widest start tolerance
    # and the setpoints go on from it at the agent's speed.
    rows = [[0.045 + 0.008 * i] * 6 for i in range(1, 11)]
    result = h.client.run_move(
        rows, APP_WAIT_S, start_q=[0.045] * 6, start_tol_rad=0.05
    )
    assert result.status == "refused"
    assert result.reason == (
        "setpoint: joint index 0 would move at 2.650 rad/s, over the agent speed "
        "limit 0.400 rad/s"
    )
    assert result.detail == {"outcome": "refused", "sent": 0, "total": 10}
    assert h.arm.sent == []


# Idle joint reads and a move


def test_no_idle_read_lands_on_the_first_setpoint_of_a_move(robot_app):
    h = robot_app
    h.new_task()
    real = h.checkers.agent.check_setpoint
    tried: list[JointReading | None] = []

    def check(q_prev, q_next, dt) -> None:
        if not h.arm.sent and not tried:
            # The dispatcher's idle read at the worst moment: the motion worker
            # stands in front of the send of the first setpoint.
            tried.append(h.app._idle.step(h.clock.monotonic()))
        real(q_prev, q_next, dt)

    h.checkers.agent.check_setpoint = check
    result = h.client.run_move(PATH, APP_WAIT_S, start_q=REST)
    assert result.status == "completed", result.reason
    assert tried == [None]
    times = [t for _, t in h.arm.sent]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert gaps == pytest.approx([0.02] * 24, abs=1e-9)
    assert state_after(h, result.command_id).health["missed_deadlines"] == 0


def test_the_idle_reads_go_on_while_a_move_waits_for_its_first_permit(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.move(PATH, start_q=REST)  # no permit follows
    result = h.client.wait_result(command_id, timeout_s=APP_WAIT_S)
    assert (result.status, result.detail["sent"]) == ("refused", 0)
    # One failed idle read afterwards is no lost feedback: the wait was no pause
    # that the reader did not know of.
    h.arm.fail_reads(2, 1)
    h.advance(0.4, step_s=0.02)
    assert h.app._idle.failures()[2] == 1
    assert h.app.snapshot().stop_level is StopLevel.NONE


class ClockedCarrier:
    """A quiet link whose wait for a link message takes time on the app clock.

    On the machine clock that wait is part of every dispatcher pass; on a fake
    clock a loopback link's wait takes none.
    """

    alive = True

    def __init__(self, clock) -> None:
        self._clock = clock

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        pass

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        pass

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        if timeout_s > 0.0:
            self._clock.sleep(timeout_s)
        return None

    def close(self) -> None:
        self.alive = False


class HoldingApp:
    """A robot app that holds at rest on a quiet ClockedCarrier.

    Its dispatcher runs by itself and is the only thing that moves the clock.
    The read number `fail_read` of joint index 3 gets no answer.
    """

    def __init__(self, config, arm, checkers, clock, fail_read: int | None) -> None:
        self.clock = clock
        self.reads: list[tuple[int, float]] = []  # (joint index, when it began)
        real = arm.read_joint

        def read_joint(joint: int) -> JointReading:
            self.reads.append((joint, clock.monotonic()))
            if joint == 3 and [j for j, _ in self.reads].count(3) == fail_read:
                clock.sleep(NO_ANSWER_S)
                raise ReadFailed("joint index 3: no answer")
            return real(joint)

        arm.read_joint = read_joint
        self.app = RobotApp(config, arm, checkers, ClockedCarrier(clock), clock)

    def run_for(self, seconds: float) -> None:
        """Start the app and let its dispatcher run this long on the app clock."""
        self.app.start()
        until = self.clock.monotonic() + seconds
        while self.clock.monotonic() < until:
            assert self.app.wait_step(APP_WAIT_S), "the dispatcher made no pass"

    def idle_reads(self) -> list[tuple[int, float]]:
        """The idle reads: every read after the six of the app's start."""
        return self.reads[6:]


@pytest.fixture
def holding(sim_arm, checkers, config, clock):
    made: list[HoldingApp] = []

    def make(fail_read: int | None = None) -> HoldingApp:
        made.append(HoldingApp(config, sim_arm, checkers, clock, fail_read))
        return made[-1]

    yield make
    for held in made:
        held.app.close()


def test_in_a_hold_one_joint_is_read_every_idle_read_period(holding):
    held = holding()
    held.run_for(2.0)
    reads = held.idle_reads()
    assert [joint for joint, _ in reads[:12]] == [0, 1, 2, 3, 4, 5] * 2
    began = [t for _, t in reads]
    gaps = [later - earlier for earlier, later in zip(began, began[1:])]
    # Whatever the wait for a link message and the read itself took.
    assert gaps == pytest.approx([IDLE_PERIOD_S] * len(gaps), abs=1e-6)
    assert len(reads) >= 95


@pytest.mark.parametrize("fail_read", [3, 4, 5, 6])
def test_one_joint_read_without_an_answer_in_a_hold_is_no_fault_stop(
    holding, fail_read
):
    held = holding(fail_read)
    held.run_for(2.0)
    assert held.app._idle.failures() == [0, 0, 0, 1, 0, 0]
    state = held.app.snapshot()
    assert (state.stop_level, state.fault_reason) == (StopLevel.NONE, "")
    # The joint was read again at once, when the failed read was over.
    reads = held.idle_reads()
    failed = [i for i, (joint, _) in enumerate(reads) if joint == 3][fail_read - 2]
    assert reads[failed + 1][0] == 3
    assert reads[failed + 1][1] - reads[failed][1] == pytest.approx(NO_ANSWER_S)


def test_a_joint_that_stays_silent_in_a_hold_is_still_a_fault_stop(holding, sim_arm):
    held = holding()
    sim_arm.fail_reads(3, 1000)
    held.run_for(1.0)
    state = held.app.snapshot()
    assert state.stop_level is StopLevel.FAULT
    assert state.fault_reason.startswith("joint feedback lost for 0.3")
    assert sim_arm.torque_on() and sim_arm.released_at == []


# The grasp region of a MOVE

OBJECT = (0.3, 0.0)


def refused_before_it_is_accepted(h, region: GraspRegion) -> str:
    """Send PATH with this grasp region; the reason of its one RESULT."""
    command_id = h.client.move(PATH, start_q=REST, grasp_region=region)
    result = h.client.wait_result(command_id, timeout_s=APP_WAIT_S)
    assert [r.status for r in h.client.results(command_id)] == ["refused"]
    assert h.arm.sent == []
    assert h.app._grasp_region is None  # a return to rest is not blocked by it
    return result.reason


@pytest.mark.parametrize(
    ("region", "reason"),
    [
        (
            GraspRegion((0.0, 0.0), 50.0, 0.0, None),
            "the grasp region's radius is 50 m, the limit is 0.3 m",
        ),
        (
            GraspRegion(OBJECT, 0.05, -5.0, None),
            "the grasp region reaches 5 m below the table, the limit is 0.06 m",
        ),
        (
            GraspRegion(OBJECT, 0.05, 0.05, 0.01),
            "the grasp region ends at z = 0.01 m, below its lowest z = 0.05 m",
        ),
        (
            GraspRegion(OBJECT, 0.0, 0.0, None),
            "the grasp region needs a radius above 0",
        ),
    ],
    ids=["radius", "depth", "upside down", "no radius"],
)
def test_a_move_with_a_grasp_region_beyond_the_thors_settings_is_refused(
    robot_app, region, reason
):
    h = robot_app
    h.new_task()
    assert refused_before_it_is_accepted(h, region) == reason


@pytest.mark.parametrize(
    "region",
    [
        GraspRegion(OBJECT, 0.07, -0.06, None),  # APAI's grasp
        GraspRegion(OBJECT, 0.07, 0.03, None),  # its place, 2 cm under the release
        GraspRegion(OBJECT, 0.25, -0.005, None),  # its push over 0.15 m
        GraspRegion(OBJECT, 0.05, 0.0, 0.08),  # a column with a ceiling
    ],
    ids=["grasp", "place", "push", "column"],
)
def test_a_move_with_a_grasp_region_of_the_size_apai_uses_is_taken(robot_app, region):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, APP_WAIT_S, start_q=REST, grasp_region=region)
    assert result.status == "completed", result.reason
    assert h.app._grasp_region == region
