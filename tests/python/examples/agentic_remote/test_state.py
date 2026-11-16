# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the joint history, the idle reader and the state publisher."""

from __future__ import annotations

import dataclasses
import logging
import math
import threading

import pytest
from isaaccapture_examples.agentic_remote.protocol.carrier import receive_message
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MAX_PAYLOAD_BYTES,
    GripperState,
    JointSample,
    RobotState,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    AuthorityState,
    JawLoad,
    JointReading,
    MotionState,
    Owner,
    ReadFailed,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.state import (
    IDLE_AFTER_TARGET_S,
    NOT_SENT_LOG_S,
    IdleReader,
    JointHistory,
    StatePublisher,
)

LOGGER = "isaaccapture.examples.agentic_remote.state"
PERIOD_S = 0.02  # state.idle_read_period_s of the default configuration
JOIN_S = 10.0


def reading(
    joint: int, t_end: float, value: float = 0.5, valid: bool = True
) -> JointReading:
    """A reading whose read took 8 ms and ended at t_end."""
    return JointReading(joint, value, t_end - 0.008, t_end, valid)


class FakeArm:
    """The part of an arm port that the idle reader and the state publisher use."""

    def __init__(self, clock, n_joints: int = 3, reads_allowed: bool = True) -> None:
        self.n_joints = n_joints
        self.joint_names = [f"joint{i}" for i in range(1, n_joints + 1)]
        self._clock = clock
        self._reads_allowed = reads_allowed
        self.read_time_s = 0.0
        self.values = [0.1 * (joint + 1) for joint in range(n_joints)]
        self.failing: set[int] = set()  # joints whose read raises ReadFailed
        self.invalid: set[int] = set()  # joints whose reading comes back not valid
        self.raises: Exception | None = None  # raised by the next read instead
        self.asked: list[int] = []  # every joint a read was asked for, in order
        self.target: tuple[list[float], float] | None = None
        self.fault_list: list[str] = []
        self.temps: list[float] | None = None
        self.powered = True

    def last_target(self) -> tuple[list[float], float] | None:
        return self.target

    def torque_on(self) -> bool:
        return self.powered

    def faults(self) -> list[str]:
        return list(self.fault_list)

    def temperatures(self) -> list[float] | None:
        return self.temps

    def read_joint(self, joint: int) -> JointReading:
        if not self._reads_allowed:
            raise AssertionError("read_joint() must not be called here")
        self.asked.append(joint)
        t_start = self._clock.monotonic()
        self._clock.advance(self.read_time_s)
        if self.raises is not None:
            error, self.raises = self.raises, None
            raise error
        if joint in self.failing:
            raise ReadFailed(f"joint {joint} did not answer")
        return JointReading(
            joint,
            self.values[joint],
            t_start,
            self._clock.monotonic(),
            valid=joint not in self.invalid,
        )

    def read_all(self) -> list[JointReading]:
        raise AssertionError("read_all() must not be called here")


class RecordingCarrier:
    """A carrier that records which send call each link message used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bytes]] = []
        self.alive = True

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("reliable", msg_type, payload))

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        self.calls.append(("latest", msg_type, payload))

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        return None

    def close(self) -> None:
        self.alive = False


# --- JointHistory ---------------------------------------------------------------


def test_history_has_no_reading_at_first():
    history = JointHistory(n_joints=3, keep_s=6.0)
    assert history.n_joints == 3
    assert history.latest() == [None, None, None]
    assert history.since(0.0) == []
    assert history.age(1000.0) is None


@pytest.mark.parametrize("n_joints", [0, -1])
def test_history_needs_at_least_one_joint(n_joints):
    with pytest.raises(ValueError, match=r"n_joints"):
        JointHistory(n_joints=n_joints, keep_s=6.0)


@pytest.mark.parametrize("keep_s", [0.0, -1.0, math.nan, math.inf])
def test_history_needs_a_positive_finite_keep_time(keep_s):
    with pytest.raises(ValueError, match=r"keep_s"):
        JointHistory(n_joints=3, keep_s=keep_s)


def test_latest_is_the_newest_valid_reading_of_each_joint():
    history = JointHistory(n_joints=3, keep_s=6.0)
    older, newer = reading(0, 10.0, value=0.1), reading(0, 10.2, value=0.2)
    other = reading(2, 10.1, value=0.7)
    for item in (older, other, newer):
        history.add(item)
    assert history.latest() == [newer, None, other]


def test_latest_keeps_the_newest_reading_when_an_older_one_arrives_late():
    history = JointHistory(n_joints=1, keep_s=6.0)
    newer, older = reading(0, 10.2), reading(0, 10.0)
    history.add(newer)
    history.add(older)
    assert history.latest() == [newer]


def test_of_two_readings_that_ended_together_the_one_added_last_is_the_latest():
    history = JointHistory(n_joints=1, keep_s=6.0)
    first, second = reading(0, 10.0, value=0.1), reading(0, 10.0, value=0.2)
    history.add(first)
    history.add(second)
    assert history.latest() == [second]
    assert history.since(0.0) == [first, second]


def test_latest_returns_a_copy():
    history = JointHistory(n_joints=2, keep_s=6.0)
    history.latest()[0] = reading(0, 10.0)
    assert history.latest() == [None, None]


def test_an_invalid_reading_is_stored_but_is_never_the_latest():
    history = JointHistory(n_joints=2, keep_s=6.0)
    good = reading(0, 10.0, value=0.1)
    bad = reading(0, 10.2, value=0.9, valid=False)
    only_bad = reading(1, 10.1, valid=False)
    for item in (good, only_bad, bad):
        history.add(item)
    assert history.latest() == [good, None]
    assert history.since(0.0) == [good, only_bad, bad]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_value_is_stored_as_not_valid(value):
    history = JointHistory(n_joints=1, keep_s=6.0)
    good = reading(0, 10.0)
    history.add(good)
    history.add(reading(0, 10.2, value=value))
    assert history.latest() == [good]
    stored = history.since(10.1)
    assert len(stored) == 1
    assert stored[0].valid is False
    assert stored[0].t_end == 10.2


def test_since_returns_the_readings_in_time_order():
    history = JointHistory(n_joints=3, keep_s=6.0)
    for joint, t_end in ((0, 10.3), (1, 10.1), (2, 10.2), (0, 10.0), (1, 10.4)):
        history.add(reading(joint, t_end))
    assert [item.t_end for item in history.since(0.0)] == [10.0, 10.1, 10.2, 10.3, 10.4]


def test_readings_that_ended_together_keep_the_order_they_were_added_in():
    history = JointHistory(n_joints=3, keep_s=6.0)
    history.add(reading(0, 10.5))
    for joint in (2, 0, 1):
        history.add(reading(joint, 10.0))
    assert [item.joint for item in history.since(0.0)] == [2, 0, 1, 0]


def test_since_includes_a_reading_that_ended_exactly_then():
    history = JointHistory(n_joints=1, keep_s=6.0)
    for t_end in (10.0, 10.25, 10.5):
        history.add(reading(0, t_end))
    assert [item.t_end for item in history.since(10.25)] == [10.25, 10.5]
    assert history.since(10.75) == []


def test_readings_older_than_the_keep_time_are_dropped():
    history = JointHistory(n_joints=1, keep_s=1.0)
    for step in range(30):
        history.add(reading(0, 10.0 + 0.25 * step))
    kept = [item.t_end for item in history.since(0.0)]
    assert kept == [16.25, 16.5, 16.75, 17.0, 17.25]


def test_a_late_reading_older_than_the_keep_time_is_not_kept():
    history = JointHistory(n_joints=2, keep_s=1.0)
    history.add(reading(0, 20.0))
    history.add(reading(1, 18.0, valid=False))
    assert [item.t_end for item in history.since(0.0)] == [20.0]


def test_the_newest_valid_reading_of_a_joint_outlives_the_keep_time():
    history = JointHistory(n_joints=2, keep_s=1.0)
    stale = reading(0, 10.0, value=0.3)
    history.add(stale)
    for step in range(1, 21):
        history.add(reading(1, 10.0 + 0.5 * step))
    assert all(item.joint == 1 for item in history.since(0.0))
    assert history.latest()[0] == stale
    assert history.age(20.5) == pytest.approx(10.5)


def test_age_is_unknown_until_every_joint_has_a_valid_reading():
    history = JointHistory(n_joints=2, keep_s=6.0)
    history.add(reading(0, 10.3))
    assert history.age(10.5) is None
    history.add(reading(1, 10.1, valid=False))
    assert history.age(10.5) is None
    history.add(reading(1, 10.2))
    assert history.age(10.5) == pytest.approx(0.3)


def test_age_is_that_of_the_stalest_joint():
    history = JointHistory(n_joints=3, keep_s=6.0)
    for joint, t_end in ((0, 10.3), (1, 10.0), (2, 10.4)):
        history.add(reading(joint, t_end))
    assert history.age(10.5) == pytest.approx(0.5)
    history.add(reading(1, 10.45))
    assert history.age(10.5) == pytest.approx(0.2)


def test_an_invalid_reading_does_not_make_a_joint_younger():
    history = JointHistory(n_joints=1, keep_s=6.0)
    history.add(reading(0, 10.0))
    history.add(reading(0, 10.4, valid=False))
    history.add(reading(0, 10.45, value=math.nan))
    assert history.age(10.5) == pytest.approx(0.5)


def test_age_is_never_negative():
    history = JointHistory(n_joints=1, keep_s=6.0)
    history.add(reading(0, 10.0))
    assert history.age(9.0) == 0.0


@pytest.mark.parametrize("joint", [-1, 3])
def test_add_refuses_a_joint_the_arm_does_not_have(joint):
    history = JointHistory(n_joints=3, keep_s=6.0)
    with pytest.raises(ValueError, match=r"joint index -?\d is outside 0\.\.2"):
        history.add(reading(joint, 10.0))
    assert history.since(0.0) == []


@pytest.mark.parametrize(
    "times", [(math.nan, 10.0), (9.9, math.nan), (9.9, math.inf), (-math.inf, 10.0)]
)
def test_add_refuses_a_non_finite_time(times):
    history = JointHistory(n_joints=1, keep_s=6.0)
    with pytest.raises(ValueError, match=r"finite times"):
        history.add(JointReading(0, 0.5, *times))
    assert history.since(-1e9) == []


def test_history_takes_readings_from_several_threads_at_once():
    n_joints, per_thread = 4, 400
    history = JointHistory(n_joints=n_joints, keep_s=1e6)
    start = threading.Barrier(n_joints)

    def feed(joint: int) -> None:
        start.wait(JOIN_S)
        for step in range(per_thread):
            history.add(reading(joint, 10.0 + 0.001 * step, value=float(step)))
            history.latest()

    threads = [
        threading.Thread(target=feed, args=(joint,)) for joint in range(n_joints)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert not any(thread.is_alive() for thread in threads)

    stored = history.since(0.0)
    assert len(stored) == n_joints * per_thread
    ends = [item.t_end for item in stored]
    assert ends == sorted(ends)
    assert [item.value for item in history.latest()] == [per_thread - 1.0] * n_joints


# --- IdleReader -----------------------------------------------------------------


@pytest.fixture
def arm(clock) -> FakeArm:
    return FakeArm(clock)


@pytest.fixture
def history(arm) -> JointHistory:
    return JointHistory(arm.n_joints, keep_s=6.0)


@pytest.fixture
def reader(arm, history, config, clock) -> IdleReader:
    return IdleReader(arm, history, config.state, clock)


def idle_steps(reader: IdleReader, clock, count: int) -> list[JointReading | None]:
    """Call step() once per idle read period; the clock ends one period later."""
    results = []
    for _ in range(count):
        results.append(reader.step(clock.monotonic()))
        clock.advance(PERIOD_S)
    return results


def test_default_idle_read_period_is_the_one_these_tests_assume(config):
    assert config.state.idle_read_period_s == PERIOD_S


def test_idle_reader_needs_an_arm_and_a_history_of_the_same_size(clock, config):
    with pytest.raises(ValueError, match=r"3 joints but the history 2"):
        IdleReader(
            FakeArm(clock, n_joints=3), JointHistory(2, 6.0), config.state, clock
        )


def test_idle_reader_reads_the_joints_in_turn(arm, history, reader, clock):
    results = idle_steps(reader, clock, 7)
    assert arm.asked == [0, 1, 2, 0, 1, 2, 0]
    assert [item.joint for item in results] == arm.asked
    assert history.since(0.0) == results
    assert [item.value for item in history.latest()] == arm.values


def test_idle_reader_stamps_the_start_and_the_end_of_each_read(arm, reader, clock):
    arm.read_time_s = 0.008
    start = clock.monotonic()
    got = reader.step(start)
    assert (got.t_start, got.t_end) == (start, clock.monotonic())
    assert got.t_end - got.t_start == pytest.approx(0.008)


def test_idle_reader_reads_at_most_once_per_period(arm, reader, clock):
    start = clock.monotonic()
    assert reader.step(start) is not None
    assert reader.step(start) is None
    assert reader.step(start + 0.01) is None
    assert reader.step(start + 0.019) is None
    assert arm.asked == [0]
    assert reader.step(start + PERIOD_S) is not None
    assert arm.asked == [0, 1]


def test_idle_reader_pauses_after_a_target(arm, reader, clock):
    start = clock.monotonic()
    reader.step(start)
    arm.target = ([0.0, 0.0, 0.0], start + 0.02)
    for later in (0.02, 0.05, 0.119):
        assert reader.step(start + later) is None
    assert arm.asked == [0]
    # The turn goes on where it stopped once the target is old enough.
    got = reader.step(start + 0.02 + IDLE_AFTER_TARGET_S)
    assert got is not None and got.joint == 1
    assert arm.asked == [0, 1]


def test_idle_reader_stays_paused_while_targets_keep_coming(arm, reader, clock):
    for _ in range(50):
        arm.target = ([0.0, 0.0, 0.0], clock.monotonic())
        assert reader.step(clock.monotonic()) is None
        clock.advance(0.02)
    assert arm.asked == []
    clock.advance(IDLE_AFTER_TARGET_S)
    assert reader.step(clock.monotonic()) is not None


def test_a_failed_read_is_counted_and_never_becomes_a_value(
    arm, history, reader, clock
):
    arm.failing = {1}
    results = idle_steps(reader, clock, 8)
    # A joint whose read failed is read a second time before the turn goes on.
    assert arm.asked == [0, 1, 1, 2, 0, 1, 1, 2]
    assert [item is None for item in results] == [False, True, True, False] * 2
    assert reader.failures() == [0, 4, 0]
    assert history.latest()[1] is None
    assert all(item.joint != 1 for item in history.since(0.0))
    assert history.age(clock.monotonic()) is None


def test_failures_returns_a_copy(reader):
    reader.failures()[0] = 9
    assert reader.failures() == [0, 0, 0]


def test_a_reading_that_is_not_valid_counts_as_a_failure(arm, history, reader, clock):
    arm.invalid = {2}
    results = idle_steps(reader, clock, 3)
    assert results[2] is None
    assert reader.failures() == [0, 0, 1]
    assert history.latest()[2] is None
    stored = [item for item in history.since(0.0) if item.joint == 2]
    assert len(stored) == 1 and stored[0].valid is False


def test_a_reading_with_a_non_finite_value_counts_as_a_failure(
    arm, history, reader, clock
):
    arm.values[0] = math.nan
    assert reader.step(clock.monotonic()) is None
    assert reader.failures() == [1, 0, 0]
    assert history.latest()[0] is None
    assert history.since(0.0)[0].valid is False


def test_an_arm_fault_is_not_swallowed_and_the_turn_goes_on(arm, reader, clock):
    arm.raises = ArmFault("motor 1 overheated")
    with pytest.raises(ArmFault):
        reader.step(clock.monotonic())
    assert reader.failures() == [0, 0, 0]
    clock.advance(PERIOD_S)
    assert reader.step(clock.monotonic()).joint == 1


def test_no_time_is_counted_while_the_reads_succeed(reader, clock):
    idle_steps(reader, clock, 9)
    assert reader.consecutive_failures_s(clock.monotonic()) == 0.0


def test_lost_feedback_is_counted_from_the_last_good_reading(
    arm, history, reader, clock
):
    idle_steps(reader, clock, 3)
    last_good = history.latest()[1].t_end
    arm.failing = {1}
    idle_steps(reader, clock, 2)  # joints 0 and 1
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - last_good)
    idle_steps(reader, clock, 3)  # joint 1 fails again
    now = clock.monotonic()
    assert reader.failures() == [0, 2, 0]
    assert reader.consecutive_failures_s(now) == pytest.approx(now - last_good)
    assert now - last_good == pytest.approx(7 * PERIOD_S)


def test_lost_feedback_grows_with_time_between_reads(arm, history, reader, clock):
    idle_steps(reader, clock, 3)
    last_good = history.latest()[0].t_end
    arm.failing = {0}
    reader.step(clock.monotonic())
    clock.advance(0.3)
    assert reader.consecutive_failures_s(clock.monotonic()) == pytest.approx(
        clock.monotonic() - last_good
    )


def test_a_good_idle_read_ends_the_count(arm, reader, clock):
    idle_steps(reader, clock, 3)
    arm.failing = {0, 1}
    idle_steps(reader, clock, 5)  # joints 0 and 1 twice each, then joint 2
    assert arm.asked[3:] == [0, 0, 1, 1, 2]
    assert reader.consecutive_failures_s(clock.monotonic()) > 0.0
    arm.failing = set()
    idle_steps(reader, clock, 1)  # joint 0 answers, joint 1 is still lost
    assert reader.consecutive_failures_s(clock.monotonic()) > 0.0
    idle_steps(reader, clock, 1)
    assert reader.consecutive_failures_s(clock.monotonic()) == 0.0
    assert reader.failures() == [2, 2, 0]


def test_a_joint_that_never_answered_is_counted_from_the_first_idle_read(
    arm, reader, clock
):
    start = clock.monotonic()
    arm.failing = {2}
    idle_steps(reader, clock, 3)
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - start)
    clock.advance(0.5)
    assert reader.consecutive_failures_s(clock.monotonic()) == pytest.approx(
        now - start + 0.5
    )


def test_the_worst_joint_sets_the_lost_time(arm, history, reader, clock):
    idle_steps(reader, clock, 3)
    good = [item.t_end for item in history.latest()]
    arm.failing = {0, 2}
    idle_steps(reader, clock, 3)
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - good[0])


def test_a_good_reading_from_another_reader_ends_the_count(arm, history, reader, clock):
    idle_steps(reader, clock, 3)
    arm.failing = {1}
    idle_steps(reader, clock, 2)
    assert reader.consecutive_failures_s(clock.monotonic()) > 0.0
    # A move's own reads store a good reading of the joint.
    seen = clock.monotonic()
    history.add(reading(1, seen))
    clock.advance(1.0)
    assert reader.consecutive_failures_s(clock.monotonic()) == 0.0
    # The next failed idle read is counted from that reading.
    idle_steps(reader, clock, 3)
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - seen)


def test_a_good_reading_that_ended_before_the_failed_read_does_not_end_the_count(
    arm, history, reader, clock
):
    arm.read_time_s = 0.008
    arm.failing = {0}
    start = clock.monotonic()
    assert reader.step(start) is None  # the failed read ends 8 ms after `start`
    history.add(reading(0, start + 0.004))
    clock.advance(0.3)
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - (start + 0.004))
    history.add(reading(0, start + 0.009))
    assert reader.consecutive_failures_s(now) == 0.0


def test_a_failure_right_after_a_pause_is_counted_from_the_resume(arm, reader, clock):
    idle_steps(reader, clock, 3)
    # A long move during which nothing stores a reading.
    clock.advance(5.0)
    arm.target = ([0.0, 0.0, 0.0], clock.monotonic())
    assert reader.step(clock.monotonic()) is None
    clock.advance(IDLE_AFTER_TARGET_S)
    resumed = clock.monotonic()
    arm.failing = {0}
    assert reader.step(resumed) is None
    assert reader.failures() == [1, 0, 0]
    assert reader.consecutive_failures_s(resumed) == 0.0
    clock.advance(0.2)
    assert reader.consecutive_failures_s(clock.monotonic()) == pytest.approx(0.2)


def test_a_joint_that_was_failing_before_a_pause_keeps_its_count(
    arm, history, reader, clock
):
    idle_steps(reader, clock, 3)
    last_good = history.latest()[0].t_end
    arm.failing = {0}
    idle_steps(reader, clock, 2)  # its turn and its second read
    arm.target = ([0.0, 0.0, 0.0], clock.monotonic())
    assert reader.step(clock.monotonic()) is None
    clock.advance(IDLE_AFTER_TARGET_S)
    assert reader.step(clock.monotonic()).joint == 1  # joint 0 is not read again yet
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - last_good)
    clock.advance(PERIOD_S)
    idle_steps(reader, clock, 2)  # joint 2, then joint 0 fails again
    now = clock.monotonic()
    assert reader.failures() == [3, 0, 0]
    assert reader.consecutive_failures_s(now) == pytest.approx(now - last_good)


def test_idle_reader_reads_nothing_while_the_motors_are_off(arm, reader, clock):
    arm.powered = False
    arm.failing = {0, 1, 2}  # a released driver cannot be read
    assert idle_steps(reader, clock, 6) == [None] * 6
    assert arm.asked == []
    assert reader.failures() == [0, 0, 0]
    assert reader.consecutive_failures_s(clock.monotonic()) == 0.0


def test_motors_going_off_end_the_count_of_lost_feedback(arm, reader, clock):
    idle_steps(reader, clock, 3)
    arm.failing = {0}
    idle_steps(reader, clock, 3)
    assert reader.consecutive_failures_s(clock.monotonic()) > 0.0
    arm.powered = False
    assert reader.step(clock.monotonic()) is None
    clock.advance(1.0)
    assert reader.consecutive_failures_s(clock.monotonic()) == 0.0
    assert reader.failures() == [2, 0, 0]  # the total is kept


def test_idle_reads_resume_when_the_motors_come_on(arm, reader, clock):
    idle_steps(reader, clock, 3)
    arm.powered = False
    idle_steps(reader, clock, 50)
    arm.powered = True
    arm.failing = {0}
    start = clock.monotonic()
    idle_steps(reader, clock, 1)
    # The time with the motors off does not count as lost feedback.
    now = clock.monotonic()
    assert reader.consecutive_failures_s(now) == pytest.approx(now - start)
    assert arm.asked == [0, 1, 2, 0]


def test_idle_reader_never_asks_for_a_full_read(reader, clock):
    # FakeArm.read_all() raises, so any call would fail the steps.
    assert len(idle_steps(reader, clock, 12)) == 12


def test_a_joint_whose_read_failed_is_read_again_on_the_next_step(arm, reader, clock):
    arm.read_time_s = 0.008
    idle_steps(reader, clock, 1)
    arm.failing = {1}
    assert reader.step(clock.monotonic()) is None
    arm.failing = set()
    # At once, not one period later; then the turn goes on in its own time.
    again = reader.step(clock.monotonic())
    assert again is not None and again.joint == 1
    assert reader.step(clock.monotonic()) is None
    assert arm.asked == [0, 1, 1]
    clock.advance(PERIOD_S)
    assert reader.step(clock.monotonic()).joint == 2
    assert reader.failures() == [0, 1, 0]


def test_a_joint_that_fails_twice_in_a_row_waits_for_its_turn(arm, reader, clock):
    arm.failing = {1}
    idle_steps(reader, clock, 9)
    # The other joints are still read: one that stays silent takes two reads
    # of every four.
    assert arm.asked == [0, 1, 1, 2, 0, 1, 1, 2, 0]
    assert reader.failures() == [0, 4, 0]


def test_due_in_tells_how_long_until_the_next_idle_read(arm, reader, clock):
    start = clock.monotonic()
    assert reader.due_in(start) == 0.0  # nothing was read yet
    reader.step(start)
    assert reader.due_in(start) == pytest.approx(PERIOD_S)
    assert reader.due_in(start + 0.012) == pytest.approx(0.008)
    assert reader.due_in(start + 0.5) == 0.0
    # After a target the reads wait until it is old enough.
    arm.target = ([0.0, 0.0, 0.0], start + 0.5)
    assert reader.due_in(start + 0.52) == pytest.approx(IDLE_AFTER_TARGET_S - 0.02)
    arm.failing = {1}
    reader.step(start + 0.6)
    assert reader.due_in(start + 0.6) == 0.0  # the failed joint is read again
    arm.powered = False
    assert reader.due_in(start + 0.6) is None


def test_no_idle_read_begins_while_the_reads_are_paused(arm, reader, clock):
    idle_steps(reader, clock, 2)
    with reader.paused():
        assert reader.due_in(clock.monotonic()) is None
        assert idle_steps(reader, clock, 10) == [None] * 10
        with reader.paused():
            pass
        assert reader.step(clock.monotonic()) is None  # the outer block still holds
    assert arm.asked == [0, 1]
    assert reader.step(clock.monotonic()).joint == 2


def test_the_time_the_reads_were_paused_is_not_lost_feedback(arm, reader, clock):
    idle_steps(reader, clock, 3)
    with reader.paused():
        clock.advance(5.0)  # a long move during which nothing stores a reading
    resumed = clock.monotonic()
    arm.failing = {0}
    assert reader.step(resumed) is None
    assert reader.consecutive_failures_s(resumed) == 0.0
    clock.advance(0.2)
    assert reader.consecutive_failures_s(clock.monotonic()) == pytest.approx(0.2)


def test_pausing_waits_for_the_idle_read_under_way(arm, history, config, clock):
    reader = IdleReader(arm, history, config.state, clock)
    reading_now, go_on, paused = (threading.Event() for _ in range(3))
    real = arm.read_joint

    def read_joint(joint: int) -> JointReading:
        reading_now.set()
        assert go_on.wait(JOIN_S)
        assert not paused.is_set()  # the pause has not begun during the read
        return real(joint)

    arm.read_joint = read_joint
    stepper = threading.Thread(target=lambda: reader.step(clock.monotonic()))
    stepper.start()
    assert reading_now.wait(JOIN_S)

    def pause() -> None:
        with reader.paused():
            paused.set()

    pauser = threading.Thread(target=pause)
    pauser.start()
    assert not paused.wait(0.05)
    go_on.set()
    stepper.join(JOIN_S)
    pauser.join(JOIN_S)
    assert paused.is_set()
    assert arm.asked == [0]


# --- StatePublisher -------------------------------------------------------------


class World:
    """One state publisher and its sources, which a test can change."""

    def __init__(self, clock, config) -> None:
        self.clock = clock
        self.config = config
        self.arm = FakeArm(clock, n_joints=config.n_joints, reads_allowed=False)
        self.history = JointHistory(config.n_joints, config.state.history_s)
        self.authority = AuthorityState("session-1", Owner.AGENT, 3, 2)
        self.progress: tuple[str, int, int] = ("", 0, 0)
        self.motion = MotionState.HOLD
        self.gripper = GripperState(
            opening=1.0,
            t=999.5,
            valid=True,
            referenced=True,
            last_target=1.0,
            may_be_holding=False,
            load=JawLoad.EMPTY,
        )
        self.wait_s: float | None = None
        self.result: dict | None = None
        self.failures = [0] * config.n_joints
        self.stage = ""
        self.publisher = StatePublisher(
            arm=self.arm,
            history=self.history,
            authority_state=lambda: self.authority,
            move_progress=lambda: self.progress,
            motion_state=lambda: self.motion,
            gripper_state=lambda: self.gripper,
            assistance_wait_s=lambda: self.wait_s,
            last_result=lambda: self.result,
            session_id="session-1",
            config=config,
            clock=clock,
            read_failures=lambda: self.failures,
            move_stage=lambda: self.stage,
        )

    def read_every_joint(self) -> list[JointReading]:
        """Store one good reading per joint, 8 ms apart, ending now."""
        readings = []
        for joint in range(self.config.n_joints):
            self.clock.advance(0.008)
            readings.append(reading(joint, self.clock.monotonic(), value=0.1 * joint))
            self.history.add(readings[-1])
        return readings

    def build(self) -> RobotState:
        return self.publisher.build(self.clock.monotonic())


@pytest.fixture
def world(clock, config) -> World:
    return World(clock, config)


def test_build_takes_every_field_from_its_source(world, clock, config):
    readings = world.read_every_joint()
    world.authority = AuthorityState(
        "session-1", Owner.NOBODY, 7, 4, StopLevel.FAULT, "motor 3 overheated"
    )
    world.progress = ("move-9", 12, 250)
    world.stage = "sending"
    world.motion = MotionState.MOVING
    world.gripper = GripperState(0.4, 999.0, True, True, 0.35, True, JawLoad.LOADED)
    world.wait_s = 87.5
    world.result = {"command_id": "move-8", "status": "completed", "end_q": [0.1] * 6}
    world.arm.target = ([0.2] * 6, 1000.01)
    world.arm.powered = False
    now = clock.monotonic()

    state = world.publisher.build(now)

    assert isinstance(state, RobotState)
    assert state.seq == 1
    assert state.session_id == "session-1"
    assert state.clock_mono_s == now
    assert state.clock_wall_s == clock.wall()
    assert state.joints == [JointSample(**dataclasses.asdict(r)) for r in readings]
    assert state.recent == state.joints
    assert (state.target_q, state.target_t) == ([0.2] * 6, 1000.01)
    assert state.gripper == world.gripper
    assert (state.owner, state.owner_number, state.stop_number) == ("nobody", 7, 4)
    assert (state.stop_level, state.fault_reason) == ("fault", "motor 3 overheated")
    assert state.limit_set_version == ""
    assert state.motion == "moving"
    assert (state.move_id, state.move_index, state.move_total) == ("move-9", 12, 250)
    assert state.move_stage == "sending"
    assert state.last_result == world.result
    assert state.assistance_wait_s == 87.5
    assert state.torque_on is False
    assert (state.view_age_s, state.teleop_stage, state.controller_age_s) == (
        None,
        "",
        None,
    )
    assert state.health == {
        "reading_age_s": pytest.approx(0.04),
        "read_failures": [0] * config.n_joints,
        "faults": [],
        "temperatures": None,
        "missed_deadlines": 0,
        "clamped": 0,
        "not_finite": [],
        "permit": None,
        "permit_refusal": "",
    }


def test_view_age_is_none_until_a_live_view_status_arrives(world, clock):
    assert world.build().view_age_s is None  # no source of the view's age
    asked: list[float] = []
    world.publisher._view_age = lambda now: asked.append(now)
    assert world.build().view_age_s is None  # no live VIEW_STATUS yet
    world.publisher._view_age = lambda now: 0.25
    assert world.build().view_age_s == 0.25
    # The source is asked for the age at the state's own time.
    assert asked == [clock.monotonic()]


def test_view_age_grows_with_the_time_since_arrival(world, clock):
    arrived = clock.monotonic()
    world.publisher._view_age = lambda now: 0.2 + (now - arrived)
    assert world.build().view_age_s == pytest.approx(0.2)
    clock.advance(1.5)
    assert world.build().view_age_s == pytest.approx(1.7)
    assert world.publisher.build(arrived + 4.0).view_age_s == pytest.approx(4.2)


def test_a_view_age_that_is_not_finite_is_sent_as_unknown(world):
    world.publisher._view_age = lambda now: math.inf
    state = world.build()
    assert state.view_age_s is None
    assert state.health["not_finite"] == ["view_age_s"]
    state.to_payload()


def test_robot_state_shows_the_held_permit_and_its_evidence_age(world, clock):
    facts = {
        "move_id": "move-9",
        "through_index": 49,
        "evidence_age_s": 0.75,
        "expires_in_s": 0.5,
    }
    asked: list[float] = []
    world.publisher._permit_facts = lambda now: asked.append(now) or dict(facts)
    state = world.build()
    assert state.health["permit"] == facts
    assert asked == [clock.monotonic()]
    assert decode(RobotState.TYPE, state.to_payload()).health["permit"] == facts
    # Without a permit the key is there and says so.
    world.publisher._permit_facts = lambda now: None
    assert world.build().health["permit"] is None


def test_robot_state_names_the_last_permit_refusal(world):
    asked: list[str] = []

    def refusal(move_id: str) -> str | None:
        asked.append(move_id)
        return "permit evidence is too old" if move_id == "move-9" else None

    world.publisher._permit_refusal = refusal
    assert world.build().health["permit_refusal"] == ""  # no move runs
    world.progress = ("move-9", 12, 250)
    assert world.build().health["permit_refusal"] == "permit evidence is too old"
    world.progress = ("move-10", 0, 25)
    assert world.build().health["permit_refusal"] == ""
    # Asked for the move the state itself names.
    assert asked == ["", "move-9", "move-10"]


def test_enum_sources_are_sent_as_plain_strings(world):
    state = world.build()
    for value in (state.owner, state.stop_level, state.motion):
        assert type(value) is str
    assert (state.owner, state.stop_level, state.motion) == ("agent", "none", "hold")


def test_quiet_sources_give_the_empty_values(world):
    state = world.build()
    assert (state.target_q, state.target_t) == (None, None)
    assert (state.move_id, state.move_index, state.move_total) == ("", 0, 0)
    assert state.last_result is None
    assert state.assistance_wait_s is None
    assert state.torque_on is True


def test_no_running_move_may_be_reported_as_none(world):
    world.progress = None
    state = world.build()
    assert (state.move_id, state.move_index, state.move_total) == ("", 0, 0)


def test_the_move_stage_is_sent_beside_the_move_it_belongs_to(world):
    world.progress = ("move-9", 0, 250)
    assert world.build().move_stage == ""  # its start checks are not done
    world.stage = "armed"
    state = world.build()
    assert (state.move_id, state.move_index, state.move_stage) == ("move-9", 0, "armed")


def test_no_stage_is_sent_without_a_running_move(world):
    # The stage is read after the progress; a move may have begun in between.
    world.stage = "armed"
    assert world.build().move_stage == ""


def test_the_move_stage_is_read_after_the_progress(world):
    order: list[str] = []
    world.publisher._move_progress = lambda: order.append("progress") or ("m", 0, 5)
    world.publisher._move_stage = lambda: order.append("stage") or "armed"
    assert world.build().move_stage == "armed"
    assert order == ["progress", "stage"]


def test_without_a_stage_source_the_stage_is_empty(clock, config):
    arm = FakeArm(clock, n_joints=config.n_joints, reads_allowed=False)
    publisher = StatePublisher(
        arm,
        JointHistory(config.n_joints, config.state.history_s),
        lambda: AuthorityState("s", Owner.AGENT, 1, 0),
        lambda: ("move-9", 3, 25),
        lambda: MotionState.MOVING,
        lambda: GripperState(None, None, False, False, None, False, "unknown"),
        lambda: None,
        lambda: None,
        "s",
        config,
        clock,
    )
    assert publisher.build(clock.monotonic()).move_stage == ""


def test_a_joint_without_a_reading_is_sent_not_valid_and_zero(world, clock):
    good = reading(1, clock.monotonic(), value=0.4)
    world.history.add(good)
    world.history.add(reading(2, clock.monotonic(), value=0.9, valid=False))
    state = world.build()
    assert len(state.joints) == world.config.n_joints
    assert state.joints[1] == JointSample(1, 0.4, good.t_start, good.t_end, True)
    for joint in (0, 2, 3, 4, 5):
        assert state.joints[joint] == JointSample(joint, 0.0, 0.0, 0.0, False)
    assert state.health["reading_age_s"] is None


def test_joints_carry_the_newest_valid_reading_of_each_joint(world, clock):
    world.read_every_joint()
    clock.advance(0.1)
    newer = reading(3, clock.monotonic(), value=1.25)
    world.history.add(newer)
    clock.advance(0.01)
    world.history.add(reading(3, clock.monotonic(), value=7.0, valid=False))
    state = world.build()
    assert [sample.joint for sample in state.joints] == list(range(6))
    assert state.joints[3] == JointSample(3, 1.25, newer.t_start, newer.t_end, True)
    assert state.joints[0].value == 0.0 and state.joints[0].valid is True


def test_recent_holds_every_reading_of_the_recent_window(world, clock, config):
    assert config.state.recent_window_s == 0.5
    world.read_every_joint()  # ends at +0.048 s
    clock.advance(0.3)
    inside = reading(0, clock.monotonic(), value=0.2)
    not_valid = reading(1, clock.monotonic(), value=0.3, valid=False)
    world.history.add(inside)
    world.history.add(not_valid)
    clock.advance(0.4)  # the first six readings are now over 0.5 s old
    state = world.build()
    assert state.recent == [
        JointSample(0, 0.2, inside.t_start, inside.t_end, True),
        JointSample(1, 0.3, not_valid.t_start, not_valid.t_end, False),
    ]
    assert len(state.joints) == 6 and all(sample.valid for sample in state.joints)


def test_seq_goes_up_by_one_with_each_build(world):
    assert [world.build().seq for _ in range(5)] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("reads_before_the_end", range(7))
def test_a_command_that_ends_during_a_build_is_never_shown_half_ended(
    world, clock, config, reads_before_the_end
):
    world.authority = AuthorityState("session-1", Owner.RECOVERY, 3, 2)
    world.motion = MotionState.MOVING
    ended = {"command_id": "rest-1", "status": "completed", "reason": ""}
    reads: list[str] = []

    def source(name: str):
        def read():
            if len(reads) == reads_before_the_end:
                # The return to rest ends: what it changed first, its outcome last.
                world.authority = AuthorityState("session-1", Owner.NOBODY, 4, 2)
                world.motion = MotionState.REST_HOLD
                world.result = ended
            reads.append(name)
            return getattr(world, name)

        return read

    publisher = StatePublisher(
        arm=world.arm,
        history=world.history,
        authority_state=source("authority"),
        move_progress=source("progress"),
        motion_state=source("motion"),
        gripper_state=source("gripper"),
        assistance_wait_s=source("wait_s"),
        last_result=source("result"),
        session_id="session-1",
        config=config,
        clock=clock,
    )
    state = publisher.build(clock.monotonic())
    assert len(reads) == 6
    # The outcome is read first, so everything else is at least as new as it.
    assert reads[0] == "result"
    assert (state.last_result == ended) == (reads_before_the_end == 0)
    if state.last_result == ended:
        assert (state.owner, state.owner_number) == ("nobody", 4)
        assert state.motion == "rest_hold"


@pytest.mark.parametrize(
    ("owner", "version"),
    [
        (Owner.AGENT, "agent-v2"),
        (Owner.HUMAN, "human-v3"),
        (Owner.RECOVERY, "recovery-v4"),
        (Owner.NOBODY, ""),
        ("human", "human-v3"),
    ],
)
def test_limit_set_version_is_the_one_of_the_control_owner(
    clock, config, owner, version
):
    limits = config.limits
    config = dataclasses.replace(
        config,
        limits=dataclasses.replace(
            limits,
            agent=dataclasses.replace(limits.agent, version="agent-v2"),
            human=dataclasses.replace(limits.human, version="human-v3"),
            recovery=dataclasses.replace(limits.recovery, version="recovery-v4"),
        ),
    )
    world = World(clock, config)
    world.authority = AuthorityState("session-1", owner, 1, 0)
    assert world.build().limit_set_version == version


def test_the_clock_pair_belongs_to_the_time_given(world, clock):
    now = clock.monotonic()
    wall = clock.wall()
    clock.advance(0.004)  # the caller took `now` a little earlier
    state = world.publisher.build(now)
    assert state.clock_mono_s == now
    assert state.clock_wall_s == pytest.approx(wall, abs=1e-6)


def test_build_refuses_a_time_that_is_not_finite(world):
    with pytest.raises(ValueError, match=r"now must be finite"):
        world.publisher.build(math.nan)
    assert world.build().seq == 1


def test_health_reports_the_arm_and_the_counters(world, clock):
    world.read_every_joint()
    clock.advance(0.25)
    world.arm.fault_list = ["motor 2: overtemperature", "gripper: stall"]
    world.arm.temps = [31.5, 32.0, 40.25, 30.0, 29.5, 33.0]
    world.failures = [0, 3, 0, 0, 1, 0]
    world.publisher.note_missed_deadline()
    world.publisher.note_missed_deadline()
    world.publisher.note_clamped()
    health = world.build().health
    assert health == {
        "reading_age_s": pytest.approx(0.25 + 5 * 0.008),
        "read_failures": [0, 3, 0, 0, 1, 0],
        "faults": ["motor 2: overtemperature", "gripper: stall"],
        "temperatures": [31.5, 32.0, 40.25, 30.0, 29.5, 33.0],
        "missed_deadlines": 2,
        "clamped": 1,
        "not_finite": [],
        "permit": None,
        "permit_refusal": "",
    }


def test_read_failures_are_unknown_without_a_source(clock, config):
    arm = FakeArm(clock, n_joints=config.n_joints, reads_allowed=False)
    publisher = StatePublisher(
        arm,
        JointHistory(config.n_joints, config.state.history_s),
        lambda: AuthorityState("s", Owner.NOBODY, 0, 0),
        lambda: ("", 0, 0),
        lambda: MotionState.IDLE,
        lambda: GripperState(None, None, False, False, None, False, "unknown"),
        lambda: None,
        lambda: None,
        "s",
        config,
        clock,
    )
    state = publisher.build(clock.monotonic())
    assert state.health["read_failures"] is None
    assert state.motion == "idle"


def test_read_failures_can_come_from_the_idle_reader(clock, config):
    arm = FakeArm(clock, n_joints=config.n_joints)
    arm.failing = {4}
    history = JointHistory(config.n_joints, config.state.history_s)
    idle = IdleReader(arm, history, config.state, clock)
    publisher = StatePublisher(
        arm,
        history,
        lambda: AuthorityState("s", Owner.NOBODY, 0, 0),
        lambda: ("", 0, 0),
        lambda: MotionState.HOLD,
        lambda: GripperState(None, None, False, False, None, False, "unknown"),
        lambda: None,
        lambda: None,
        "s",
        config,
        clock,
        read_failures=idle.failures,
    )
    idle_steps(idle, clock, 12)
    state = publisher.build(clock.monotonic())
    # The joint's turn, its second read right after it, and its next turn.
    assert state.health["read_failures"] == [0, 0, 0, 0, 3, 0]
    assert state.joints[4] == JointSample(4, 0.0, 0.0, 0.0, False)
    assert all(sample.valid for sample in state.joints if sample.joint != 4)
    assert state.health["reading_age_s"] is None


def test_the_publisher_never_reads_the_arm(world, clock, carrier_pair):
    # World's arm raises on read_joint() and on read_all().
    world.read_every_joint()
    world.build()
    assert world.publisher.maybe_publish(carrier_pair[0], clock.monotonic()) is True
    assert world.arm.asked == []


def test_maybe_publish_sends_at_most_once_per_period(world, clock, config):
    assert config.state.publish_period_s == 0.05
    carrier = RecordingCarrier()
    sent = []
    for _ in range(21):  # 0.2 s in supervisor steps of 10 ms
        sent.append(world.publisher.maybe_publish(carrier, clock.monotonic()))
        clock.advance(0.01)
    assert sent == [True, False, False, False, False] * 4 + [True]
    assert len(carrier.calls) == 5


def test_maybe_publish_sends_a_newest_wins_robot_state(world, clock):
    carrier = RecordingCarrier()
    now = clock.monotonic()
    assert world.publisher.maybe_publish(carrier, now) is True
    ((call, msg_type, payload),) = carrier.calls
    assert (call, msg_type) == ("latest", RobotState.TYPE)
    state = decode(msg_type, payload)
    assert (state.seq, state.clock_mono_s) == (1, now)


def test_an_unreceived_state_is_replaced_by_the_next(world, clock, carrier_pair):
    robot_end, workstation_end = carrier_pair
    for _ in range(3):
        assert world.publisher.maybe_publish(robot_end, clock.monotonic()) is True
        clock.advance(0.05)
    newest = receive_message(workstation_end, 0)
    assert isinstance(newest, RobotState) and newest.seq == 3
    assert receive_message(workstation_end, 0) is None


def test_maybe_publish_on_a_dead_link_sends_and_builds_nothing(
    world, clock, carrier_pair
):
    robot_end, workstation_end = carrier_pair
    robot_end.break_link()
    assert world.publisher.maybe_publish(robot_end, clock.monotonic()) is False
    assert receive_message(workstation_end, 0) is None
    assert world.build().seq == 1


def test_a_built_state_survives_encoding_and_decoding(world, clock):
    world.read_every_joint()
    world.history.add(reading(2, clock.monotonic(), value=0.3, valid=False))
    world.authority = AuthorityState(
        "session-1", Owner.RECOVERY, 9, 5, StopLevel.STOP, ""
    )
    world.progress = ("rest-1", 40, 1050)
    world.motion = MotionState.MOVING
    world.wait_s = 12.25
    world.result = {
        "command_id": "move-3",
        "status": "failed",
        "reason": "contact",
        "end_q": [0.1, -0.2, 0.3, 0.0, 1.5, -1.0],
        "detail": {"index": 17},
    }
    world.arm.target = ([0.1, -0.2, 0.3, 0.0, 1.5, -1.0], clock.monotonic() - 0.01)
    world.arm.temps = [30.5] * 6
    world.arm.fault_list = ["gripper: stall"]
    world.failures = (1, 0, 0, 2, 0, 0)
    world.publisher.note_missed_deadline()
    world.publisher.note_clamped()

    state = world.build()
    decoded = decode(RobotState.TYPE, state.to_payload())

    assert decoded == state
    assert len(decoded.recent) == 7
    assert decoded.recent[-1].valid is False
    assert decoded.health["read_failures"] == [1, 0, 0, 2, 0, 0]


def test_a_published_state_arrives_as_it_was_built(world, clock, carrier_pair):
    robot_end, workstation_end = carrier_pair
    readings = world.read_every_joint()
    world.arm.target = ([0.5] * 6, clock.monotonic())
    now = clock.monotonic()
    assert world.publisher.maybe_publish(robot_end, now) is True
    state = receive_message(workstation_end, 0)
    assert isinstance(state, RobotState)
    assert state.seq == 1
    assert state.clock_mono_s == now
    assert state.joints == [JointSample(**dataclasses.asdict(r)) for r in readings]
    assert state.target_q == [0.5] * 6
    assert state.gripper == world.gripper
    assert (state.owner, state.owner_number, state.stop_number) == ("agent", 3, 2)
    assert state.limit_set_version == world.config.limits.agent.version


def _numbers(value) -> list[float]:
    """Every float inside nested dataclasses, dicts and lists."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    return [value] if isinstance(value, float) else []


def test_non_finite_values_never_reach_the_link_message(world, clock, caplog):
    now = clock.monotonic()
    world.history.add(reading(0, now, value=math.nan))
    world.history.add(reading(1, now, value=math.inf, valid=False))
    world.history.add(reading(2, now, value=0.25))
    world.arm.target = ([0.0, math.nan, 0.0, 0.0, 0.0, 0.0], now)
    world.arm.temps = [30.0, math.inf, 31.0, -math.inf, math.nan, 29.0]
    world.gripper = GripperState(math.nan, math.inf, True, True, -math.inf, False, "x")
    world.wait_s = math.inf
    world.result = {"end_q": [0.1, math.nan], "detail": {"error": -math.inf}}

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        state = world.build()
        world.build()

    assert all(math.isfinite(number) for number in _numbers(state))
    decoded = decode(RobotState.TYPE, state.to_payload())
    assert decoded == state
    assert state.joints[0] == JointSample(0, 0.0, 0.0, 0.0, False)
    assert state.joints[2].value == 0.25 and state.joints[2].valid is True
    assert [(sample.value, sample.valid) for sample in state.recent] == [
        (0.0, False),
        (0.0, False),
        (0.25, True),
    ]
    assert (state.target_q, state.target_t) == (None, None)
    assert state.gripper == GripperState(None, None, False, True, None, False, "x")
    assert state.assistance_wait_s is None
    assert state.last_result == {"end_q": [0.1, None], "detail": {"error": None}}
    assert state.health["temperatures"] == [30.0, None, 31.0, None, None, 29.0]
    assert state.health["not_finite"] == [
        "target",
        "gripper",
        "assistance_wait_s",
        "last_result",
        "temperatures",
    ]
    # Each source is reported once, not once per build.
    assert len([record for record in caplog.records if record.name == LOGGER]) == 5


@pytest.mark.parametrize(
    "target",
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5, math.inf], 1000.0),
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], math.nan),
    ],
)
def test_a_target_with_a_non_finite_number_is_left_out(world, target):
    world.arm.target = target
    state = world.build()
    assert (state.target_q, state.target_t) == (None, None)
    assert state.health["not_finite"] == ["target"]
    state.to_payload()


def test_a_bad_jaw_time_alone_leaves_the_jaw_reading_valid(world):
    world.gripper = dataclasses.replace(world.gripper, t=math.nan)
    state = world.build()
    assert state.gripper == dataclasses.replace(world.gripper, t=None)
    assert state.health["not_finite"] == ["gripper"]


def test_numpy_values_are_sent_as_plain_numbers(world, clock):
    np = pytest.importorskip("numpy")
    world.arm.temps = np.array([30.0, np.nan, 31.0])
    world.arm.target = (np.array([0.5] * 6), np.float64(clock.monotonic()))
    world.result = {"end_q": np.array([0.1, 0.2]), "index": np.int64(3)}
    state = world.build()
    assert state.health["temperatures"] == [30.0, None, 31.0]
    assert state.last_result == {"end_q": [0.1, 0.2], "index": 3}
    assert decode(RobotState.TYPE, state.to_payload()) == state


def test_a_state_that_cannot_be_encoded_is_logged_once_and_not_sent(
    world, clock, caplog
):
    carrier = RecordingCarrier()
    world.result = {"detail": object()}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            assert world.publisher.maybe_publish(carrier, clock.monotonic()) is False
            clock.advance(0.05)
    assert carrier.calls == []
    logged = [record for record in caplog.records if record.name == LOGGER]
    assert len(logged) == 1
    assert "ROBOT_STATE was not sent" in logged[0].getMessage()
    # The stream goes on once the source is sound again.
    world.result = None
    assert world.publisher.maybe_publish(carrier, clock.monotonic()) is True
    assert decode(*carrier.calls[0][1:]).seq == 4


def test_a_state_that_stays_too_long_is_logged_once_and_then_counted(
    world, clock, caplog
):
    assert NOT_SENT_LOG_S == 5.0
    carrier = RecordingCarrier()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        # The byte count in the problem differs each time: it is still one problem.
        for extra in range(40):
            world.result = {"reason": "y" * (MAX_PAYLOAD_BYTES + extra)}
            assert world.publisher.maybe_publish(carrier, clock.monotonic()) is False
            clock.advance(0.05)
        # A problem of another kind is logged when it first shows.
        world.result = {"detail": object()}
        assert world.publisher.maybe_publish(carrier, clock.monotonic()) is False
        logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        assert len(logged) == 2
        assert logged[0].startswith("ROBOT_STATE was not sent: RobotState: 4")
        assert logged[0].endswith(" bytes is over the limit")
        assert "cannot encode" in logged[1]
        # After five seconds the repeats are told, with how many they were.
        world.result = None
        clock.advance(3.5)  # 5.5 s after the first one
        assert world.publisher.maybe_publish(carrier, clock.monotonic()) is True
    logged = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert logged[2:] == [
        "ROBOT_STATE was not sent 39 more times: RobotState: # bytes is over the limit"
    ]


def test_counters_can_be_bumped_from_other_threads(world):
    start = threading.Barrier(4)

    def bump() -> None:
        start.wait(JOIN_S)
        for _ in range(250):
            world.publisher.note_missed_deadline()
            world.publisher.note_clamped()
            world.build()

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(JOIN_S)
    assert not any(thread.is_alive() for thread in threads)
    state = world.build()
    assert state.seq == 1001
    assert state.health["missed_deadlines"] == 1000
    assert state.health["clamped"] == 1000
