# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simulated arm and the basic checker in
isaaccapture_examples.agentic_remote.robot_app.sim."""

from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import GraspRegion
from isaaccapture_examples.agentic_remote.robot_app import sim as sim_module
from isaaccapture_examples.agentic_remote.robot_app.config import (
    GraspConfig,
    LimitSet,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    ArmPort,
    Checker,
    ContactStop,
    ContactWatch,
    FeedbackLost,
    GripperOutcome,
    JointReading,
    ReadFailed,
    Refusal,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import (
    BasicChecker,
    BasicContactWatch,
    SimArm,
)

PERIOD = 0.02
READ_S = 0.008
TAU_S = 0.05
REST = [0.0] * 6
SIM_LOGGER = "isaaccapture.examples.agentic_remote.sim"


@pytest.fixture
def arm(sim_arm):
    """The simulated arm, connected."""
    sim_arm.connect()
    return sim_arm


@pytest.fixture
def jaws(arm):
    """The connected arm with its jaws referenced."""
    arm.gripper_reference()
    return arm


class History:
    """What a contact watch needs of the joint history."""

    def __init__(self):
        self.readings = []

    def add(self, reading):
        self.readings.append(reading)


def drive(arm, watch, clock, start, per_step, steps):
    """Send setpoints and run the watch one period apart, as the executor does."""
    q = list(start)
    t0 = clock.monotonic()
    for index in range(steps):
        q = [value + step for value, step in zip(q, per_step)]
        deadline = t0 + (index + 1) * PERIOD
        arm.send_target(q)
        watch.after_send(index, q, deadline)
        clock.sleep(max(0.0, deadline - clock.monotonic()))
    return q


# SimArm: construction


def test_fixture_arm_starts_at_rest_and_is_an_arm_port(sim_arm, config):
    assert isinstance(sim_arm, ArmPort)
    assert sim_arm.n_joints == config.n_joints
    assert sim_arm.joint_names == list(config.joint_names)
    assert sim_arm.pose() == list(config.rest.rest_q)
    assert sim_arm.sent == []
    assert sim_arm.released_at == []
    assert sim_arm.jaw_commands == []


def test_constructor_arguments(clock):
    by_position = SimArm(3, clock, [0.1, 0.2, 0.3])
    assert by_position.pose() == [0.1, 0.2, 0.3]
    assert by_position.joint_names == ["joint1", "joint2", "joint3"]
    assert SimArm(clock=clock).pose() == REST
    named = SimArm(n_joints=2, clock=clock, joint_names=("pan", "tilt"))
    assert named.joint_names == ["pan", "tilt"]


def test_constructor_refuses_bad_arguments(clock):
    with pytest.raises(TypeError, match=r"needs a clock"):
        SimArm()
    with pytest.raises(ValueError, match=r"q0 needs 6 joint values"):
        SimArm(n_joints=6, clock=clock, q0=[0.0] * 5)
    with pytest.raises(ValueError, match=r"q0 must be finite"):
        SimArm(n_joints=1, clock=clock, q0=[math.nan])
    with pytest.raises(ValueError, match=r"joint_names needs 6 names"):
        SimArm(n_joints=6, clock=clock, joint_names=["only"])
    with pytest.raises(ValueError, match=r"n_joints"):
        SimArm(n_joints=0, clock=clock)
    with pytest.raises(ValueError, match=r"jaw_step_s"):
        SimArm(clock=clock, jaw_step_s=0.0)


def test_sim_does_not_import_apai():
    source = Path(sim_module.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+apai\b", source, re.MULTILINE)


# SimArm: motors


def test_connect_runs_once(sim_arm):
    assert not sim_arm.torque_on()
    sim_arm.connect()
    assert sim_arm.torque_on()
    with pytest.raises(RuntimeError, match=r"runs once"):
        sim_arm.connect()
    assert sim_arm.torque_on()


def test_measured_pose_approaches_the_target_with_a_first_order_lag(arm, clock):
    arm.send_target([1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    assert arm.pose() == REST
    clock.advance(TAU_S)
    one_tau = 1.0 - math.exp(-1.0)
    assert arm.pose() == pytest.approx([one_tau, -one_tau, 0.5 * one_tau, 0, 0, 0])
    clock.advance(TAU_S)
    two_tau = 1.0 - math.exp(-2.0)
    assert arm.pose() == pytest.approx([two_tau, -two_tau, 0.5 * two_tau, 0, 0, 0])
    clock.advance(2.0)
    assert arm.pose() == pytest.approx([1.0, -1.0, 0.5, 0, 0, 0], abs=1e-9)


def test_time_constant_is_a_constructor_argument(clock):
    slow = SimArm(n_joints=1, clock=clock, time_constant_s=0.5)
    slow.connect()
    slow.send_target([1.0])
    clock.advance(0.5)
    assert slow.pose() == pytest.approx([1.0 - math.exp(-1.0)])


def test_read_joint_costs_read_time_and_gives_start_and_end(arm, clock):
    arm.send_target([1.0] * 6)
    t0 = clock.monotonic()
    reading = arm.read_joint(3)
    assert isinstance(reading, JointReading)
    assert reading.joint == 3
    assert reading.valid
    assert reading.t_start == t0
    assert reading.t_end == pytest.approx(t0 + READ_S)
    assert clock.monotonic() == reading.t_end
    # The value is the pose at the end of the read.
    assert reading.value == pytest.approx(1.0 - math.exp(-READ_S / TAU_S))


def test_read_time_is_a_constructor_argument(clock):
    quick = SimArm(clock=clock, read_time_s=0.001)
    t0 = clock.monotonic()
    quick.read_joint(0)
    assert clock.monotonic() == pytest.approx(t0 + 0.001)


def test_read_all_reads_every_joint_afresh(arm, clock):
    arm.send_target([1.0] * 6)
    t0 = clock.monotonic()
    readings = arm.read_all()
    assert [reading.joint for reading in readings] == list(range(6))
    assert readings[0].t_start == t0
    assert clock.monotonic() == pytest.approx(t0 + 6 * READ_S)
    for earlier, later in zip(readings, readings[1:]):
        assert later.t_start == earlier.t_end
        assert later.value > earlier.value  # the arm moved on during the reads
    again = arm.read_all()
    assert again[0].t_start == readings[-1].t_end
    assert again[0].value > readings[-1].value


def test_read_joint_refuses_an_unknown_joint(arm):
    with pytest.raises(IndexError):
        arm.read_joint(6)
    with pytest.raises(IndexError):
        arm.read_joint(-1)


def test_failed_reads_raise_and_run_out(arm, clock):
    arm.fail_reads(2, 2)
    t0 = clock.monotonic()
    with pytest.raises(ReadFailed, match=r"joint index 2"):
        arm.read_joint(2)
    assert clock.monotonic() == pytest.approx(t0 + READ_S)  # a failed read costs time
    assert arm.read_joint(1).joint == 1  # other joints still read
    with pytest.raises(ReadFailed):
        arm.read_joint(2)
    assert arm.read_joint(2).valid


def test_read_all_fails_when_one_joint_fails(arm):
    arm.fail_reads(4, 1)
    with pytest.raises(ReadFailed, match=r"joint index 4"):
        arm.read_all()
    assert len(arm.read_all()) == 6


def test_blocked_joint_stops_there_and_can_leave(arm, clock):
    arm.block_joint(0, 0.2)
    arm.block_joint(1, -0.3)
    arm.send_target([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
    clock.advance(1.0)
    pose = arm.pose()
    assert pose[0] == 0.2  # the tracking error to the target of 1.0 stays
    assert pose[1] == -0.3
    assert pose[2] == pytest.approx(1.0)
    assert arm.read_joint(0).value == 0.2
    arm.send_target(REST)
    clock.advance(1.0)
    assert arm.pose() == pytest.approx(REST, abs=1e-6)
    arm.block_joint(0, None)
    arm.send_target([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    clock.advance(1.0)
    pose = arm.pose()
    assert pose[0] == pytest.approx(1.0)
    assert pose[1] == -0.3  # still blocked


def test_every_target_is_recorded_with_its_send_time(arm, clock):
    assert arm.last_target() is None
    t0 = clock.monotonic()
    arm.send_target((0.1,) * 6)
    clock.advance(PERIOD)
    arm.send_target([0.2] * 6)
    assert arm.sent == [([0.1] * 6, t0), ([0.2] * 6, pytest.approx(t0 + PERIOD))]
    assert arm.last_target() == ([0.2] * 6, pytest.approx(t0 + PERIOD))
    arm.last_target()[0][0] = 9.0
    assert arm.last_target()[0][0] == 0.2


def test_send_target_refuses_a_malformed_setpoint(arm):
    with pytest.raises(ValueError, match=r"6 joint values"):
        arm.send_target([0.0] * 5)
    with pytest.raises(ValueError, match=r"finite"):
        arm.send_target([math.inf] + [0.0] * 5)
    assert arm.sent == []


def test_targets_are_ignored_while_torque_is_off(sim_arm, clock, caplog):
    with caplog.at_level(logging.WARNING, logger=SIM_LOGGER):
        sim_arm.send_target([1.0] * 6)
        sim_arm.send_target([1.0] * 6)
    assert [record.name for record in caplog.records] == [SIM_LOGGER]  # said once
    assert "torque is off" in caplog.text
    sim_arm.hold_measured()
    clock.advance(1.0)
    assert sim_arm.pose() == REST
    assert sim_arm.last_target() is None
    assert len(sim_arm.sent) == 2  # recorded all the same
    assert sim_arm.holds == []
    sim_arm.connect()
    clock.advance(1.0)
    assert sim_arm.pose() == REST  # the ignored target does not come back


def test_hold_measured_holds_the_pose_it_measures(arm, clock):
    arm.send_target([1.0] * 6)
    clock.advance(TAU_S)
    arm.hold_measured()
    held = arm.pose()
    now = clock.monotonic()
    assert held == pytest.approx([1.0 - math.exp(-1.0)] * 6)
    assert arm.last_target() == (held, now)
    assert arm.holds == [(held, now)]
    assert len(arm.sent) == 1  # a hold is not a setpoint
    clock.advance(1.0)
    assert arm.pose() == pytest.approx(held)


def test_hold_measured_ends_the_push_on_a_blocked_joint(arm, clock):
    arm.block_joint(0, 0.2)
    arm.send_target([1.0] * 6)
    clock.advance(1.0)
    arm.hold_measured()
    assert arm.last_target()[0][0] == 0.2


@pytest.mark.releases_away_from_rest
def test_every_release_is_recorded_with_the_measured_pose(arm, clock):
    arm.send_target([0.5] * 6)
    clock.advance(TAU_S)
    away_from_rest = arm.pose()
    assert not arm.release_confirmed()
    arm.release_torque()
    assert arm.release_confirmed()  # the simulated motors are known to be off
    assert arm.released_at == [away_from_rest]
    assert away_from_rest == pytest.approx([0.5 * (1.0 - math.exp(-1.0))] * 6)
    assert not arm.torque_on()
    assert arm.last_target() is None
    clock.advance(1.0)
    assert arm.pose() == away_from_rest  # nothing pulls it on
    arm.send_target([1.0] * 6)
    clock.advance(1.0)
    assert arm.pose() == away_from_rest
    arm.release_torque()
    assert arm.released_at == [away_from_rest, away_from_rest]
    with pytest.raises(RuntimeError):
        arm.connect()  # torque does not come back
    assert not arm.torque_on()


def test_faults_are_set_and_cleared(arm):
    assert arm.faults() == []
    arm.set_fault("motor 2 over temperature")
    arm.set_fault("gripper stall")
    assert arm.faults() == ["motor 2 over temperature", "gripper stall"]
    arm.faults().clear()
    assert len(arm.faults()) == 2
    arm.clear_faults()
    assert arm.faults() == []


def test_temperatures_are_unknown_unless_set(arm):
    assert arm.temperatures() is None
    arm.set_temperatures([30, 31, 32, 33, 34, 35])
    assert arm.temperatures() == [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
    arm.set_temperatures(None)
    assert arm.temperatures() is None


def test_set_pose_puts_the_arm_somewhere_with_no_target(arm, clock):
    arm.send_target([1.0] * 6)
    away = [2.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    arm.set_pose(away)
    clock.advance(1.0)
    assert arm.pose() == away
    assert arm.last_target() is None
    assert [reading.value for reading in arm.read_all()] == away
    with pytest.raises(ValueError):
        arm.set_pose([0.0])


# SimArm: jaws


def test_jaws_are_unknown_until_referenced(arm):
    assert not arm.gripper_referenced()
    assert arm.gripper_read() is None
    assert arm.gripper_last_target() is None
    outcome = arm.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert outcome == GripperOutcome("refused", None, "the jaws are not referenced")
    assert arm.gripper_last_target() is None
    arm.gripper_reference()
    assert arm.gripper_referenced()
    assert arm.gripper_read() == 1.0
    # Like a driver's reference, it leaves a jaw target: where the jaws are.
    assert arm.gripper_last_target() == 1.0
    assert arm.jaw_commands[1:] == []  # referencing is not a command
    assert arm.jaw_stiffness == "light"


def test_the_reference_target_is_the_measured_opening(clock):
    part_open = SimArm(clock=clock, jaw_opening=0.4)
    part_open.connect()
    part_open.gripper_reference()
    assert part_open.gripper_last_target() == 0.4
    part_open.gripper_command(0.7, 0.5, "plain", threading.Event())
    part_open.gripper_reference()
    assert part_open.gripper_last_target() == 0.7  # a command's target stays


def test_reference_fails_with_torque_off_or_a_fault(sim_arm):
    with pytest.raises(ArmFault, match=r"torque off"):
        sim_arm.gripper_reference()
    sim_arm.connect()
    sim_arm.set_fault("gripper motor fault")
    with pytest.raises(ArmFault, match=r"gripper motor fault"):
        sim_arm.gripper_reference()
    assert not sim_arm.gripper_referenced()
    sim_arm.clear_faults()
    sim_arm.gripper_reference()
    assert sim_arm.gripper_referenced()


def test_a_reference_ends_early_once_its_cancel_is_set(sim_arm):
    sim_arm.connect()
    cancel = threading.Event()
    cancel.set()
    sim_arm.gripper_reference(cancel)  # raises nothing
    assert not sim_arm.gripper_referenced()
    assert sim_arm.gripper_last_target() is None and sim_arm.jaw_commands == []
    # A fault still counts for more than the cancel.
    sim_arm.set_fault("gripper motor fault")
    with pytest.raises(ArmFault, match=r"gripper motor fault"):
        sim_arm.gripper_reference(cancel)
    sim_arm.clear_faults()
    sim_arm.gripper_reference(threading.Event())
    assert sim_arm.gripper_referenced()


def test_jaws_travel_at_jaw_speed(jaws, clock):
    t0 = clock.monotonic()
    outcome = jaws.gripper_command(0.3, 0.5, "plain", threading.Event())
    assert outcome.status == "reached"
    assert outcome.opening == 0.3
    assert outcome.detail == ""
    # 0.7 of the travel at 1.0 per second, in steps of 10 ms.
    assert clock.monotonic() - t0 == pytest.approx(0.7, abs=0.011)
    assert jaws.gripper_read() == 0.3
    assert jaws.gripper_last_target() == 0.3
    t1 = clock.monotonic()
    assert (
        jaws.gripper_command(1.0, 0.5, "plain", threading.Event()).status == "reached"
    )
    assert clock.monotonic() - t1 == pytest.approx(0.7, abs=0.011)
    assert jaws.gripper_read() == 1.0


def test_jaw_speed_is_a_constructor_argument(clock):
    fast = SimArm(clock=clock, jaw_speed=4.0)
    fast.connect()
    fast.gripper_reference()
    t0 = clock.monotonic()
    assert fast.gripper_command(0.0, 1.0, "plain", threading.Event()).opening == 0.0
    assert clock.monotonic() - t0 == pytest.approx(0.25, abs=0.011)


def test_jaw_steps_are_reported_to_the_hook(jaws):
    seen = []
    jaws.on_jaw_step = seen.append
    jaws.gripper_command(0.95, 0.5, "plain", threading.Event())
    assert seen == pytest.approx([0.99, 0.98, 0.97, 0.96, 0.95])


def test_every_jaw_command_is_recorded(jaws, clock):
    t0 = clock.monotonic()
    jaws.gripper_command(0.5, 0.25, "plain", threading.Event())
    t1 = clock.monotonic()
    jaws.gripper_command(0.9, 0.75, "close_until_contact", threading.Event())
    first, second = jaws.jaw_commands
    assert (first.opening, first.max_effort, first.mode, first.t) == (
        0.5,
        0.25,
        "plain",
        t0,
    )
    assert (second.opening, second.max_effort, second.mode, second.t) == (
        0.9,
        0.75,
        "close_until_contact",
        t1,
    )
    assert first.status == second.status == "reached"
    assert first.stiffness == second.stiffness == "closing"


def test_close_stalls_on_an_object_and_drops_to_the_light_stiffness(jaws, clock):
    jaws.jaw_object(0.4)
    assert jaws.jaw_stiffness == "light"
    t0 = clock.monotonic()
    outcome = jaws.gripper_command(0.0, 0.5, "close_until_contact", threading.Event())
    assert outcome.status == "stalled"
    assert outcome.opening == pytest.approx(0.4)
    assert clock.monotonic() - t0 == pytest.approx(0.6, abs=0.011)
    assert jaws.gripper_read() == pytest.approx(0.4)
    assert jaws.gripper_last_target() == 0.0
    assert jaws.jaw_stiffness == "light"
    assert jaws.jaw_commands[-1].stiffness == "light"
    assert jaws.jaw_commands[-1].status == "stalled"
    assert jaws.light_holds == []  # the driver did it by itself


def test_jaws_stop_at_the_object_between_two_steps(jaws):
    jaws.jaw_object(0.405)
    outcome = jaws.gripper_command(0.0, 0.5, "close_until_contact", threading.Event())
    assert (outcome.status, outcome.opening) == ("stalled", 0.405)
    assert jaws.gripper_read() == 0.405


def test_plain_close_also_stalls_on_an_object(jaws):
    jaws.jaw_object(0.4)
    outcome = jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.opening) == ("stalled", pytest.approx(0.4))


def test_close_onto_a_held_object_stalls_at_once(jaws, clock):
    jaws.jaw_object(0.4)
    jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    t0 = clock.monotonic()
    outcome = jaws.gripper_command(0.1, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.opening) == ("stalled", pytest.approx(0.4))
    assert clock.monotonic() == t0


def test_object_does_not_stop_the_jaws_short_of_it_or_opening(jaws):
    jaws.jaw_object(0.4)
    short = jaws.gripper_command(0.6, 0.5, "plain", threading.Event())
    assert (short.status, short.opening) == ("reached", 0.6)
    jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    opened = jaws.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert (opened.status, opened.opening) == ("reached", 1.0)
    jaws.jaw_object(None)
    closed = jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert (closed.status, closed.opening) == ("reached", 0.0)
    assert jaws.jaw_stiffness == "closing"


def test_cancel_leaves_the_jaws_where_they_are_at_the_closing_stiffness(jaws, clock):
    cancel = threading.Event()

    def cancel_half_way(opening):
        if opening <= 0.5 + 1e-9:
            cancel.set()

    jaws.on_jaw_step = cancel_half_way
    outcome = jaws.gripper_command(0.0, 0.5, "plain", cancel)
    assert outcome.status == "cancelled"
    assert outcome.opening == pytest.approx(0.5)
    clock.advance(5.0)
    assert jaws.gripper_read() == pytest.approx(0.5)  # they do not creep on
    assert jaws.gripper_last_target() == 0.0
    assert jaws.jaw_stiffness == "closing"
    assert jaws.jaw_commands[-1].stiffness == "closing"
    assert jaws.jaw_commands[-1].status == "cancelled"

    jaws.gripper_light_hold()
    assert jaws.jaw_stiffness == "light"
    assert jaws.jaw_commands[-1].stiffness == "light"
    assert jaws.light_holds == [clock.monotonic()]
    assert jaws.gripper_read() == pytest.approx(0.5)
    assert len(jaws.jaw_commands) == 1  # a light hold is not a command


def test_command_with_cancel_already_set_moves_nothing(jaws, clock):
    cancel = threading.Event()
    cancel.set()
    t0 = clock.monotonic()
    outcome = jaws.gripper_command(0.0, 0.5, "plain", cancel)
    assert (outcome.status, outcome.opening) == ("cancelled", 1.0)
    assert clock.monotonic() == t0
    # Not even the target or the stiffness changed.
    assert jaws.gripper_last_target() == 1.0
    assert jaws.jaw_stiffness == "light"
    assert jaws.jaw_commands[-1].status == "cancelled"


def test_light_hold_from_another_thread_outlasts_the_cancelled_close(jaws):
    cancel = threading.Event()
    under_way = threading.Event()
    go_on = threading.Event()

    def pause_after_the_first_step(opening):
        if not under_way.is_set():
            under_way.set()
            go_on.wait(5.0)

    jaws.on_jaw_step = pause_after_the_first_step
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(jaws.gripper_command(0.0, 0.5, "plain", cancel))
    )
    worker.start()
    assert under_way.wait(5.0)
    assert jaws.jaw_stiffness == "closing"
    cancel.set()
    jaws.gripper_light_hold()
    go_on.set()
    worker.join(5.0)
    assert not worker.is_alive()
    assert outcomes[0].status == "cancelled"
    assert outcomes[0].opening == pytest.approx(0.99)
    assert jaws.jaw_stiffness == "light"
    assert jaws.jaw_commands[-1].stiffness == "light"


def test_jaw_command_times_out(clock):
    slow = SimArm(clock=clock, jaw_speed=0.1, jaw_timeout_s=0.5)
    slow.connect()
    slow.gripper_reference()
    t0 = clock.monotonic()
    outcome = slow.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert outcome.status == "timeout"
    assert outcome.opening == pytest.approx(0.95)
    assert clock.monotonic() - t0 == pytest.approx(0.5)
    # A close that ended short of its target is left at the light stiffness.
    assert slow.gripper_last_target() == 0.0
    assert slow.jaw_stiffness == "light"
    assert slow.jaw_commands[-1].stiffness == "light"

    # An opening that runs out of time is not: only a close can press an object.
    shut = SimArm(clock=clock, jaw_speed=0.1, jaw_timeout_s=0.5, jaw_opening=0.0)
    shut.connect()
    shut.gripper_reference()
    outcome = shut.gripper_command(1.0, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.opening) == ("timeout", pytest.approx(0.05))
    assert shut.jaw_stiffness == "closing"


def test_a_cancelled_opening_leaves_the_jaw_target_where_the_jaws_are(jaws, clock):
    jaws.gripper_command(0.2, 0.5, "plain", threading.Event())
    cancel = threading.Event()

    def cancel_part_open(opening):
        if opening >= 0.5 - 1e-9:
            cancel.set()

    jaws.on_jaw_step = cancel_part_open
    outcome = jaws.gripper_command(1.0, 0.5, "plain", cancel)
    assert (outcome.status, outcome.opening) == ("cancelled", pytest.approx(0.5))
    # The target in force is the opening reached, not the one that was asked for.
    assert jaws.gripper_last_target() == pytest.approx(0.5)
    jaws.gripper_light_hold()
    clock.advance(5.0)
    assert jaws.gripper_read() == pytest.approx(0.5)
    assert jaws.gripper_last_target() == pytest.approx(0.5)


def test_jaw_timeout_counts_the_commands_own_waiting(jaws, clock):
    # Another thread that advances a fake clock must not time the command out.
    jaws.on_jaw_step = lambda opening: clock.advance(10.0)
    outcome = jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.opening) == ("reached", 0.0)


@pytest.mark.parametrize(
    ("opening", "max_effort", "mode", "detail"),
    [
        (1.2, 0.5, "plain", r"opening must be between 0 and 1"),
        (-0.1, 0.5, "plain", r"opening must be between 0 and 1"),
        (math.nan, 0.5, "plain", r"opening must be between 0 and 1"),
        (None, 0.5, "plain", r"opening must be between 0 and 1"),
        (0.5, math.inf, "plain", r"max_effort must be between 0 and 1"),
        (0.5, 1.5, "plain", r"max_effort must be between 0 and 1"),
        (0.5, 0.5, "squeeze", r"unknown mode 'squeeze'"),
    ],
)
def test_malformed_jaw_command_is_refused(jaws, opening, max_effort, mode, detail):
    outcome = jaws.gripper_command(opening, max_effort, mode, threading.Event())
    assert outcome.status == "refused"
    assert outcome.opening == 1.0
    assert re.search(detail, outcome.detail)
    assert jaws.gripper_read() == 1.0
    assert jaws.gripper_last_target() == 1.0  # still the reference's
    assert jaws.jaw_commands[-1].status == "refused"
    assert jaws.jaw_stiffness == "light"


def test_jaw_command_is_refused_with_torque_off(jaws):
    jaws.release_torque()
    outcome = jaws.gripper_command(0.0, 0.5, "plain", threading.Event())
    assert (outcome.status, outcome.detail) == ("refused", "torque is off")
    assert jaws.gripper_read() == 1.0


def test_injected_jaw_read_failures_give_none(jaws):
    jaws.fail_gripper_reads(2)
    assert jaws.gripper_read() is None
    assert jaws.gripper_read() is None
    assert jaws.gripper_read() == 1.0


# BasicChecker: the contact watch


def test_free_motion_is_read_in_turn_and_never_called_contact(arm, clock, checkers):
    history = History()
    watch = checkers.human.contact_watch(arm, arm.pose(), history)
    assert isinstance(watch, ContactWatch)
    assert isinstance(watch, BasicContactWatch)
    # The human speed limit, out and then back.
    q = drive(arm, watch, clock, REST, [0.03] * 6, 40)
    drive(arm, watch, clock, q, [-0.03] * 6, 40)
    assert len(history.readings) == 160  # two joints per step
    assert [r.joint for r in history.readings[:8]] == [0, 1, 2, 3, 4, 5, 0, 1]
    assert all(r.valid for r in history.readings)
    assert not watch.finished
    watch.finish()
    assert watch.finished


def test_after_send_returns_the_readings_of_the_step(arm, clock, checkers):
    history = History()
    watch = checkers.agent.contact_watch(arm, REST, history)
    t0 = clock.monotonic()
    first = watch.after_send(0, REST, t0 + PERIOD)
    assert [reading.joint for reading in first] == [0, 1]
    assert first[0].t_start == t0
    assert first[1].t_end == pytest.approx(t0 + 2 * READ_S)
    second = watch.after_send(1, REST, t0 + 2 * PERIOD)
    assert [reading.joint for reading in second] == [2, 3]
    assert history.readings == first + second


def test_reads_per_step_is_a_constructor_argument(arm, clock, config):
    checker = BasicChecker(config.limits.agent, clock, reads_per_step=3)
    watch = checker.contact_watch(arm, REST, None)
    readings = watch.after_send(0, REST, clock.monotonic() + 1.0)
    assert [reading.joint for reading in readings] == [0, 1, 2]


def test_watch_works_without_a_history(arm, clock, checkers):
    watch = checkers.agent.contact_watch(arm, REST, None)
    drive(arm, watch, clock, REST, [0.008] * 6, 20)


def test_the_watchs_hold_is_the_arms_hold_at_the_measured_pose(arm, clock, checkers):
    watch = checkers.agent.contact_watch(arm, REST, None)
    arm.block_joint(0, 0.05)
    arm.send_target([0.3] * 6)
    clock.advance(1.0)
    watch.hold()
    assert len(arm.holds) == 1
    assert arm.last_target()[0] == arm.holds[0][0]
    assert arm.last_target()[0][0] == pytest.approx(0.05)


def test_blocked_joint_raises_contact_stop_through_the_watch(arm, clock, checkers):
    history = History()
    watch = checkers.agent.contact_watch(arm, REST, history)
    arm.block_joint(1, 0.05)
    step = [0.0, 0.008, 0.0, 0.0, 0.0, 0.0]  # joint index 1 at the agent speed limit
    with pytest.raises(ContactStop, match=r"contact: joint index 1 trails") as caught:
        drive(arm, watch, clock, REST, step, 200)
    assert isinstance(caught.value, Refusal)
    assert caught.value.reason == str(caught.value)
    # Contact is called once the setpoint sent 10 steps earlier is 0.10 rad past
    # the stop, on the second such reading of the joint.
    last_index = len(arm.sent) - 1
    lagged = 0.008 * (last_index + 1 - 10)
    assert lagged - 0.05 > 0.10
    assert 28 <= last_index <= 36
    assert arm.pose()[1] == 0.05
    assert history.readings[-1].joint == 1
    assert history.readings[-1].value == 0.05


def test_contact_needs_two_readings_in_a_row(clock):
    one_joint = SimArm(n_joints=1, clock=clock)
    limits = LimitSet(
        name="agent",
        version="t",
        joint_min=[-3.0],
        joint_max=[3.0],
        max_joint_speed=1.0,
    )
    checker = BasicChecker(limits, clock, reads_per_step=1)
    watch = checker.contact_watch(one_joint, [0.0], None)

    def step_with_the_joint_at(value):
        one_joint.set_pose([value])
        return watch.after_send(0, [0.0], clock.monotonic() + PERIOD)

    step_with_the_joint_at(0.5)
    step_with_the_joint_at(0.1)  # at the tolerance: not off
    step_with_the_joint_at(-0.5)
    with pytest.raises(ContactStop, match=r"trails its setpoints by 0\.500 rad"):
        step_with_the_joint_at(0.5)


def test_failed_read_between_two_off_readings_does_not_clear_them(clock):
    one_joint = SimArm(n_joints=1, clock=clock)
    limits = LimitSet(
        name="agent",
        version="t",
        joint_min=[-3.0],
        joint_max=[3.0],
        max_joint_speed=1.0,
    )
    watch = BasicChecker(limits, clock, reads_per_step=1).contact_watch(
        one_joint, [0.0], None
    )
    one_joint.set_pose([0.5])
    watch.after_send(0, [0.0], clock.monotonic() + PERIOD)
    one_joint.fail_reads(0, 1)
    assert watch.after_send(1, [0.0], clock.monotonic() + PERIOD) == []
    with pytest.raises(ContactStop):
        watch.after_send(2, [0.0], clock.monotonic() + PERIOD)


def test_contact_tolerance_and_lag_are_constructor_arguments(arm, clock, config):
    checker = BasicChecker(
        config.limits.agent, clock, contact_tol_rad=0.02, contact_lag_steps=2
    )
    watch = checker.contact_watch(arm, REST, None)
    arm.block_joint(1, 0.05)
    with pytest.raises(ContactStop):
        drive(arm, watch, clock, REST, [0.0, 0.008, 0.0, 0.0, 0.0, 0.0], 200)
    assert len(arm.sent) - 1 <= 20


def test_lost_feedback_is_a_refusal_and_not_a_contact(arm, clock, checkers):
    watch = checkers.agent.contact_watch(arm, REST, History())
    arm.fail_reads(2, 10**6)
    t0 = clock.monotonic()
    with pytest.raises(Refusal, match=r"joint feedback lost: joint index 2") as caught:
        drive(arm, watch, clock, REST, [0.008] * 6, 200)
    assert type(caught.value) is FeedbackLost
    assert 0.35 < clock.monotonic() - t0 < 0.35 + 2 * PERIOD


def test_short_loss_of_feedback_is_tolerated(arm, clock, checkers):
    history = History()
    watch = checkers.agent.contact_watch(arm, REST, history)
    arm.fail_reads(2, 3)
    drive(arm, watch, clock, REST, [0.008] * 6, 40)
    assert len(history.readings) == 80 - 3  # a failed read adds nothing


def test_wait_before_the_first_step_is_not_lost_feedback(arm, clock, checkers):
    watch = checkers.agent.contact_watch(arm, REST, None)
    clock.advance(1.0)  # the move waits, for example for its first permit
    drive(arm, watch, clock, REST, [0.008] * 6, 20)


@pytest.mark.parametrize(
    ("value", "valid"), [(0.0, False), (math.nan, True), (math.inf, True)]
)
def test_reading_that_is_not_good_counts_as_no_feedback(clock, config, value, valid):
    class BadFeedbackArm:
        n_joints = 6

        def read_joint(self, joint):
            t_start = clock.monotonic()
            clock.sleep(READ_S)
            return JointReading(joint, value, t_start, clock.monotonic(), valid)

    history = History()
    watch = BasicChecker(config.limits.agent, clock).contact_watch(
        BadFeedbackArm(), REST, history
    )
    t0 = clock.monotonic()
    with pytest.raises(Refusal, match=r"joint feedback lost: joint index 0") as caught:
        for index in range(100):
            deadline = t0 + (index + 1) * PERIOD
            assert len(watch.after_send(index, REST, deadline)) == 2
            clock.sleep(max(0.0, deadline - clock.monotonic()))
    assert not isinstance(caught.value, ContactStop)
    assert 0.35 < clock.monotonic() - t0 < 0.35 + 2 * PERIOD
    assert len(history.readings) > 30  # still handed to the history


def test_feedback_timeout_is_a_constructor_argument(arm, clock, config):
    checker = BasicChecker(config.limits.agent, clock, feedback_timeout_s=0.1)
    watch = checker.contact_watch(arm, REST, None)
    arm.fail_reads(0, 10**6)
    t0 = clock.monotonic()
    with pytest.raises(Refusal, match=r"feedback lost"):
        drive(arm, watch, clock, REST, [0.0] * 6, 200)
    assert clock.monotonic() - t0 < 0.1 + 2 * PERIOD


def test_no_read_begins_after_the_deadline_of_its_step(clock, config):
    slow = SimArm(clock=clock, read_time_s=0.03)
    watch = BasicChecker(config.limits.agent, clock).contact_watch(slow, REST, None)
    late = watch.after_send(0, REST, clock.monotonic() + PERIOD)
    assert [reading.joint for reading in late] == [0]
    in_time = watch.after_send(1, REST, clock.monotonic() + 1.0)
    assert [reading.joint for reading in in_time] == [1, 2]


# BasicChecker: checks


def test_fixture_checkers_are_checkers_named_after_their_limit_sets(checkers, config):
    for owner in ("agent", "human", "recovery"):
        checker = getattr(checkers, owner)
        assert isinstance(checker, Checker)
        assert isinstance(checker, BasicChecker)
        assert checker.name == owner
        assert checker.version == getattr(config.limits, owner).version
        assert not checker.latched()
        assert checker.halt_count() == 0
    assert checkers.agent.watchdog_s == 5.0
    assert checkers.recovery.watchdog_s is None


def test_joint_limits(checkers):
    human = checkers.human
    human.check_pose([3.0] * 6)
    human.check_pose((-3.0,) * 6)
    with pytest.raises(
        Refusal,
        match=r"pose: joint index 2 at 3\.0100 rad is outside the human joint "
        r"limits -3\.0000\.\.3\.0000",
    ):
        human.check_pose([0.0, 0.0, 3.01, 0.0, 0.0, 0.0])
    inside = [-2.99] + [0.0] * 5
    outside = [-3.01] + [0.0] * 5
    human.check_setpoint(inside, [-3.0] + [0.0] * 5, 1.0)
    with pytest.raises(Refusal, match=r"setpoint: joint index 0 at -3\.0100 rad"):
        human.check_setpoint(inside, outside, 1.0)
    # Only the envelope is closed to a step that starts outside.
    human.check_setpoint(outside, inside, 1.0)


def test_speed_limit(checkers):
    agent = checkers.agent  # 0.4 rad/s
    agent.check_setpoint(REST, [0.008] * 6, PERIOD)
    agent.check_setpoint(REST, [-0.008] * 6, PERIOD)
    fast = [0.0, 0.0, 0.0, 0.0, 0.0081, 0.0]
    with pytest.raises(
        Refusal,
        match=r"setpoint: joint index 4 would move at 0\.405 rad/s, over the agent "
        r"speed limit 0\.400 rad/s",
    ):
        agent.check_setpoint(REST, fast, PERIOD)
    with pytest.raises(Refusal, match=r"joint index 4 would move at 0\.405 rad/s"):
        agent.check_setpoint(REST, [-v for v in fast], PERIOD)
    agent.check_setpoint(REST, fast, 2 * PERIOD)  # the same step in twice the time


def test_each_limit_set_has_its_own_speed_limit(checkers):
    step = [0.02] * 6  # 1.0 rad/s
    checkers.human.check_setpoint(REST, step, PERIOD)
    with pytest.raises(Refusal, match=r"over the agent speed limit"):
        checkers.agent.check_setpoint(REST, step, PERIOD)
    with pytest.raises(Refusal, match=r"over the recovery speed limit 0\.100"):
        checkers.recovery.check_setpoint(REST, step, PERIOD)


def test_envelope_has_no_way_back_in(checkers):
    agent = checkers.agent  # envelope -1.5..1.5 inside joint limits -2..2
    edge = [1.5] + [0.0] * 5
    outside = [1.6] + [0.0] * 5
    agent.check_pose(edge)
    with pytest.raises(
        Refusal,
        match=r"pose: joint index 0 at 1\.6000 rad is outside the agent envelope "
        r"-1\.5000\.\.1\.5000",
    ):
        agent.check_pose(outside)
    with pytest.raises(Refusal, match=r"setpoint: joint index 0 at 1\.5050 rad"):
        agent.check_setpoint(edge, [1.505] + [0.0] * 5, PERIOD)
    with pytest.raises(
        Refusal,
        match=r"setpoint: its start: joint index 0 at 1\.6000 rad is outside the "
        r"agent envelope",
    ):
        agent.check_setpoint(outside, edge, 2.0)
    with pytest.raises(Refusal, match=r"setpoint 0: its start: joint index 0"):
        agent.check_path(outside, [edge], 2.0)
    # The recovery set has no envelope: the same step is its way back.
    checkers.recovery.check_pose(outside)
    checkers.recovery.check_setpoint(outside, edge, 2.0)


@pytest.mark.parametrize(
    "q",
    [
        [0.0] * 5,
        [0.0] * 7,
        [math.nan] + [0.0] * 5,
        [math.inf] + [0.0] * 5,
        ["a"] * 6,
        [[0.0] * 6],
        None,
    ],
)
def test_checker_refuses_what_is_not_one_finite_value_per_joint(checkers, q):
    human = checkers.human
    with pytest.raises(Refusal):
        human.check_pose(q)
    with pytest.raises(Refusal):
        human.check_setpoint(REST, q, PERIOD)
    with pytest.raises(Refusal):
        human.check_setpoint(q, REST, PERIOD)
    with pytest.raises(Refusal):
        human.check_path(REST, [REST, q], PERIOD)


@pytest.mark.parametrize("dt", [0.0, -PERIOD, math.nan, math.inf])
def test_checker_refuses_a_step_time_that_is_not_above_zero(checkers, dt):
    with pytest.raises(Refusal, match=r"step time"):
        checkers.human.check_setpoint(REST, REST, dt)
    with pytest.raises(Refusal, match=r"step time"):
        checkers.human.check_path(REST, [REST], dt)


def test_agent_watchdog_refuses_until_fed(checkers, clock):
    agent = checkers.agent
    count = agent.halt_count()
    with pytest.raises(Refusal, match=r"perception watchdog: never fed"):
        agent.begin_motion(count)
    assert not agent.in_motion
    agent.feed_watchdog()
    agent.begin_motion(count)
    assert agent.in_motion
    agent.check_setpoint(REST, REST, PERIOD)
    agent.end_motion()
    assert not agent.in_motion


def test_agent_watchdog_runs_out_and_counts_from_the_evidence_time(checkers, clock):
    agent = checkers.agent
    count = agent.halt_count()
    agent.feed_watchdog()
    clock.advance(5.0)
    agent.begin_motion(count)
    clock.advance(0.1)
    with pytest.raises(
        Refusal, match=r"perception watchdog: last fed 5\.1 s ago, limit 5 s"
    ):
        agent.begin_motion(count)
    agent.feed_watchdog(clock.monotonic() - 6.0)  # evidence that is already too old
    with pytest.raises(Refusal, match=r"last fed 5\.1 s ago"):  # the newer one stays
        agent.begin_motion(count)
    agent.feed_watchdog(clock.monotonic() - 4.0)
    agent.begin_motion(count)
    clock.advance(1.1)
    with pytest.raises(Refusal, match=r"last fed 5\.1 s ago"):
        agent.begin_motion(count)


def test_watchdog_ignores_evidence_from_the_future_or_not_finite(checkers, clock):
    agent = checkers.agent
    agent.feed_watchdog(math.nan)
    agent.feed_watchdog(-math.inf)
    with pytest.raises(Refusal, match=r"never fed"):
        agent.begin_motion(0)
    agent.feed_watchdog(clock.monotonic() + 100.0)  # counts as now
    agent.begin_motion(0)
    clock.advance(5.1)
    with pytest.raises(Refusal, match=r"last fed 5\.1 s ago"):
        agent.begin_motion(0)


def test_watchdog_none_disables_it(checkers, clock):
    recovery = checkers.recovery
    recovery.begin_motion(recovery.halt_count())
    recovery.end_motion()
    clock.advance(1000.0)
    recovery.begin_motion(recovery.halt_count())
    recovery.feed_watchdog()  # accepted, changes nothing
    recovery.begin_motion(recovery.halt_count())


def test_watchdog_without_an_age_limit_still_needs_one_feed(clock, config):
    checker = BasicChecker(config.limits.recovery, clock, watchdog_s=math.inf)
    with pytest.raises(Refusal, match=r"never fed"):
        checker.begin_motion(0)
    checker.feed_watchdog()
    clock.advance(1e6)
    checker.begin_motion(0)


def test_halt_stops_the_move_and_the_next_begin_motion_clears_it(checkers):
    checker = checkers.recovery
    admitted = checker.halt_count()
    checker.begin_motion(admitted)
    checker.check_setpoint(REST, REST, PERIOD)
    assert checker.halt("operator stop") == admitted + 1
    assert checker.halt_count() == admitted + 1
    with pytest.raises(Refusal, match=r"halted: operator stop"):
        checker.check_setpoint(REST, REST, PERIOD)
    checker.end_motion()
    with pytest.raises(Refusal, match=r"halted: operator stop"):
        checker.check_setpoint(REST, REST, PERIOD)
    # A halt does not make a pose or a path wrong.
    checker.check_pose(REST)
    checker.check_path(REST, [REST], PERIOD)
    checker.begin_motion(checker.halt_count())
    checker.check_setpoint(REST, REST, PERIOD)


def test_begin_motion_with_an_old_halt_count_is_refused(checkers):
    checker = checkers.recovery
    admitted = checker.halt_count()
    checker.halt("stop while the move waited")
    with pytest.raises(
        Refusal,
        match=r"halted since this motion was admitted: halt count is 1, the "
        r"motion quotes 0",
    ):
        checker.begin_motion(admitted)
    assert not checker.in_motion
    with pytest.raises(Refusal, match=r"halted"):  # the refused begin cleared nothing
        checker.check_setpoint(REST, REST, PERIOD)
    with pytest.raises(Refusal, match=r"halt count is 1, the motion quotes 2"):
        checker.begin_motion(admitted + 2)
    checker.begin_motion(admitted + 1)
    checker.check_setpoint(REST, REST, PERIOD)


def test_latch_refuses_everything_until_reset(checkers):
    checker = checkers.recovery
    count = checker.halt_count()
    checker.begin_motion(count)
    checker.latch("motor 3 fault")
    assert checker.latched()
    refused = r"fault-stop latched: motor 3 fault"
    with pytest.raises(Refusal, match=refused):
        checker.check_setpoint(REST, REST, PERIOD)
    with pytest.raises(Refusal, match=refused):
        checker.begin_motion(count)
    with pytest.raises(Refusal, match=refused):
        checker.check_pose(REST)
    with pytest.raises(Refusal, match=refused):
        checker.check_path(REST, [REST], PERIOD)
    checker.end_motion()

    checker.reset_latch()
    assert not checker.latched()
    assert checker.halt_count() == count  # a latch is not a halt
    checker.check_pose(REST)
    checker.check_path(REST, [REST], PERIOD)
    checker.begin_motion(count)
    checker.check_setpoint(REST, REST, PERIOD)


def test_latch_on_one_checker_leaves_the_others_alone(checkers):
    checkers.agent.latch("fault")
    assert not checkers.human.latched()
    assert not checkers.recovery.latched()
    checkers.recovery.check_pose(REST)


def test_check_path_accepts_a_whole_move_and_names_the_bad_setpoint(checkers):
    agent = checkers.agent
    path = [[0.008 * (index + 1)] * 6 for index in range(50)]
    agent.check_path(REST, path, PERIOD)
    agent.check_path(tuple(REST), tuple(tuple(q) for q in path), PERIOD)

    jump = [list(q) for q in path]
    jump[20][3] += 0.01
    with pytest.raises(
        Refusal, match=r"setpoint 20: joint index 3 would move at 0\.900 rad/s"
    ):
        agent.check_path(REST, jump, PERIOD)

    long_way = [[0.008 * (index + 1)] + [0.0] * 5 for index in range(200)]
    with pytest.raises(
        Refusal,
        match=r"setpoint 187: joint index 0 at 1\.5040 rad is outside the agent "
        r"envelope",
    ):
        agent.check_path(REST, long_way, PERIOD)

    with pytest.raises(Refusal, match=r"setpoint 0: joint index 0 would move"):
        agent.check_path([0.1] + [0.0] * 5, path, PERIOD)  # a jump from the start
    with pytest.raises(Refusal, match=r"no setpoints"):
        agent.check_path(REST, [], PERIOD)


def test_check_path_needs_no_motion_and_no_watchdog(checkers):
    agent = checkers.agent  # never fed
    agent.check_path(REST, [[0.008] * 6], PERIOD)
    agent.check_pose(REST)
    assert not agent.in_motion


def test_recovery_accepts_every_start_pose_the_human_set_accepts(checkers, config):
    human = config.limits.human
    for corner in (human.joint_min, human.joint_max):
        checkers.human.check_pose(corner)
        checkers.recovery.check_pose(corner)


def test_grasp_region_is_stored(checkers):
    agent = checkers.agent
    region = GraspRegion((0.3, 0.1), 0.05, 0.0, None)
    assert agent.grasp_region is None
    agent.set_grasp_region(region)
    assert agent.grasp_region is region
    agent.set_grasp_region(None)
    assert agent.grasp_region is None


def test_a_grasp_region_beyond_the_grasp_settings_is_refused_and_not_stored(
    clock, config
):
    grasp = GraspConfig(max_radius_m=0.06, max_depth_m=0.0)
    agent = BasicChecker(config.limits.agent, clock, grasp=grasp)
    kept = GraspRegion((0.3, 0.1), 0.05, 0.0, None)
    agent.check_grasp_region(kept)
    agent.set_grasp_region(kept)
    for region, reason in (
        (GraspRegion((0.3, 0.1), 0.07, 0.0, None), "radius is 0.07 m, the limit is"),
        (GraspRegion((0.3, 0.1), 0.05, -0.01, None), "reaches 0.01 m below the table"),
        (GraspRegion((0.3, 0.1), 0.05, 0.02, 0.01), "ends at z = 0.01 m, below its"),
    ):
        for call in (agent.check_grasp_region, agent.set_grasp_region):
            with pytest.raises(Refusal) as refused:
                call(region)
            assert reason in refused.value.reason
    assert agent.grasp_region is kept
    # Without settings of its own a checker applies the defaults.
    default = BasicChecker(config.limits.agent, clock)
    default.check_grasp_region(GraspRegion((0.3, 0.1), 0.3, -0.06, None))
    with pytest.raises(Refusal, match=r"radius is 50 m, the limit is 0\.3 m"):
        default.check_grasp_region(GraspRegion((0.0, 0.0), 50.0, 0.0, None))


def test_contact_spots_are_stored_and_cleared(checkers):
    recovery = checkers.recovery
    assert recovery.contact_spots == []
    recovery.note_contact((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    recovery.note_contact([0.0] * 6)
    assert recovery.contact_spots == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0] * 6]
    recovery.contact_spots.clear()
    assert len(recovery.contact_spots) == 2
    assert checkers.agent.contact_spots == []
    recovery.check_pose([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # stored only
    recovery.clear_contact_spots()
    assert recovery.contact_spots == []


def test_tip_blocked_answers_through_a_hook(checkers):
    recovery = checkers.recovery
    region = GraspRegion((0.3, 0.1), 0.05, 0.0, 0.1)
    assert recovery.tip_blocked(REST, region) is None
    asked = []

    def below_the_table(q, grasp_region):
        asked.append((q, grasp_region))
        return "gripper tip below the table clearance"

    recovery.set_tip_hook(below_the_table)
    assert recovery.tip_blocked(tuple(REST), region) == (
        "gripper tip below the table clearance"
    )
    assert asked == [(REST, region)]
    recovery.set_tip_hook(lambda q, grasp_region: None)
    assert recovery.tip_blocked(REST, None) is None
    recovery.set_tip_hook(None)
    assert recovery.tip_blocked(REST, region) is None
