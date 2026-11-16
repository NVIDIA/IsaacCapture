# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JawState and GripperWorker in
isaaccapture_examples.agentic_remote.robot_app.gripper."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator

import pytest
from isaaccapture_examples.agentic_remote.protocol import messages
from isaaccapture_examples.agentic_remote.protocol.messages import (
    GripperState,
    RobotState,
)
from isaaccapture_examples.agentic_remote.robot_app import sim as sim_module
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.config import GripperConfig
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    ALREADY_REFERENCED,
    CLOSE_UNTIL_CONTACT,
    JAW_MODES,
    JAWS_BUSY,
    MAX_WAITING,
    OPEN_EFFORT,
    PLAIN,
    REFERENCE,
    GripperWorker,
    JawState,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    GripperOutcome,
    JawLoad,
    Owner,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm

WAIT_S = 5.0  # real seconds a test gives the worker thread before it fails
DROPPED = GripperOutcome("cancelled", None, "dropped by a stop")


class ScriptedJaws:
    """The three calls JawState makes on an arm, answered from a script."""

    def __init__(self, readings=(), target=None, referenced=True, clock=None):
        self.readings = list(readings)
        self.target = target
        self.referenced = referenced
        self.read_times: list[float] = []
        self._clock = clock

    def gripper_referenced(self) -> bool:
        return self.referenced

    def gripper_read(self) -> float | None:
        if self._clock is not None:
            self.read_times.append(self._clock.monotonic())
        return self.readings.pop(0)

    def gripper_last_target(self) -> float | None:
        return self.target


class Done:
    """An on_done callback that keeps what it was given and can be waited for."""

    def __init__(self) -> None:
        self.outcomes: list[GripperOutcome] = []
        self.threads: list[threading.Thread] = []
        self._called = threading.Event()

    def __call__(self, outcome: GripperOutcome) -> None:
        self.outcomes.append(outcome)
        self.threads.append(threading.current_thread())
        self._called.set()

    def wait(self) -> GripperOutcome:
        assert self._called.wait(WAIT_S), "on_done was not called"
        return self.outcomes[0]


class JawPause:
    """Stops the simulated jaws inside a command, once, until the test lets go.

    It hooks sim.on_jaw_step, which runs on the thread that commands the jaws.
    """

    def __init__(
        self,
        arm: SimArm,
        when: Callable[[float], bool],
        until: Callable[[], threading.Event] | None = None,
    ) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        self.opening: float | None = None
        self._when = when
        self._until = until or (lambda: self.release)
        arm.on_jaw_step = self._step

    def _step(self, opening: float) -> None:
        if self.reached.is_set() or not self._when(opening):
            return
        self.opening = opening
        self.reached.set()
        self._until().wait(WAIT_S)

    def wait(self) -> float:
        assert self.reached.wait(WAIT_S), "the jaws did not get there"
        return self.opening


def spy_on_cancel(arm: SimArm) -> list[threading.Event]:
    """The cancel event of every gripper_command call the arm gets, in order."""
    cancels: list[threading.Event] = []
    inner = arm.gripper_command

    def gripper_command(opening, max_effort, mode, cancel):
        cancels.append(cancel)
        return inner(opening, max_effort, mode, cancel)

    arm.gripper_command = gripper_command
    return cancels


def forbid_reads(arm: SimArm) -> None:
    def gripper_read():
        raise AssertionError("the jaws were read")

    arm.gripper_read = gripper_read


@pytest.fixture
def arm(sim_arm) -> SimArm:
    """The simulated arm, connected, its jaws referenced and fully open."""
    sim_arm.connect()
    sim_arm.gripper_reference()
    return sim_arm


@pytest.fixture
def jaw_state(config, clock) -> JawState:
    return JawState(config.gripper, clock)


@pytest.fixture
def authority() -> Authority:
    """An authority whose control owner is the agent."""
    authority = Authority()
    authority.set_owner(Owner.AGENT, "new task")
    return authority


@pytest.fixture
def make_worker(jaw_state, authority, clock) -> Iterator[Callable[..., GripperWorker]]:
    made: list[GripperWorker] = []

    def make(arm, **kwargs) -> GripperWorker:
        made.append(GripperWorker(arm, jaw_state, authority, clock, **kwargs))
        return made[-1]

    yield make
    for worker in made:
        worker.close()


@pytest.fixture
def worker(arm, make_worker) -> GripperWorker:
    return make_worker(arm)


def run(worker, opening, max_effort=0.5, mode=PLAIN, owner=Owner.AGENT):
    """Submit one command and wait for its outcome."""
    done = Done()
    assert worker.submit(opening, max_effort, mode, owner, done) is None
    return done.wait()


def open_fully(worker, owner=None) -> GripperOutcome:
    done = Done()
    assert worker.open_guarded(done, owner) is None
    return done.wait()


def outcome(status: str, opening: float | None, detail: str = "") -> GripperOutcome:
    return GripperOutcome(status, opening, detail)


def test_the_default_jaw_settings(config):
    # The numbers in the tests below are chosen around these values.
    cfg = config.gripper
    assert (cfg.closed_frac, cfg.target_tol, cfg.agree_tol) == (0.04, 0.03, 0.02)
    assert (cfg.reads, cfg.read_gap_s, cfg.open_step) == (3, 0.05, 0.05)
    assert (cfg.reach_tol, cfg.full_open_tol) == (0.03, 0.03)


def test_the_three_jaw_tolerances_are_three_settings(clock):
    cfg = GripperConfig(target_tol=0.016, reach_tol=0.03, full_open_tol=0.05)
    state = JawState(cfg, clock)
    # Proven empty: the readings against the last jaw target, target_tol.
    assert state.classify(ScriptedJaws([0.015] * 3, 0.0)) is JawLoad.EMPTY
    assert state.classify(ScriptedJaws([0.02] * 3, 0.0)) is JawLoad.LOADED
    assert state.load_reason == "the jaws are at 0.020, not at their target 0.000"
    assert state.classify(ScriptedJaws([0.98] * 3, 1.0)) is JawLoad.LOADED
    # A close ended short of its target: reach_tol.
    state.note_command_result(0.0, PLAIN, outcome("reached", 0.03))
    assert state.may_be_holding is False
    state.note_command_result(0.0, PLAIN, outcome("reached", 0.035))
    assert state.may_be_holding is True
    # A measured full open: full_open_tol.
    state.note_command_result(1.0, PLAIN, outcome("reached", 0.94))
    assert state.may_be_holding is True
    state.note_command_result(1.0, PLAIN, outcome("reached", 0.95))
    assert state.may_be_holding is False


def test_the_jaw_modes_are_the_simulated_arms():
    assert JAW_MODES == sim_module.JAW_MODES == (PLAIN, CLOSE_UNTIL_CONTACT)


# JawState: the "may be holding" flag


def test_the_flag_starts_clear(jaw_state):
    assert jaw_state.may_be_holding is False
    assert jaw_state.load_reason == ""


@pytest.mark.parametrize(
    ("target", "ended"),
    [
        (0.0, outcome("stalled", 0.4)),
        (0.0, outcome("timeout", 0.9)),
        (0.2, outcome("stalled", 0.21)),  # on an object just short of the target
        (0.2, outcome("timeout", 0.22)),
        (0.0, outcome("timeout", None)),  # the jaw read failed
        (0.5, outcome("cancelled", None)),
        (0.0, outcome("cancelled", 0.5)),  # stopped short of its target
        (0.0, outcome("failed", None, "gripper fault: no answer")),
        (0.96, outcome("stalled", None)),  # just under full open
    ],
)
def test_a_close_that_ends_short_or_unread_sets_the_flag(jaw_state, target, ended):
    jaw_state.note_command_result(target, PLAIN, ended)
    assert jaw_state.may_be_holding is True


@pytest.mark.parametrize(
    ("target", "ended"),
    [
        (0.0, outcome("reached", 0.0)),
        (0.3, outcome("reached", 0.3)),
        (0.3, outcome("reached", 0.32)),  # within the target tolerance
        (0.0, outcome("cancelled", 0.02)),  # stopped as it arrived
        (0.6, outcome("stalled", 0.5)),  # an opening that stopped short
        (0.6, outcome("timeout", 0.3)),
        (0.6, outcome("cancelled", 0.45)),
        (1.0, outcome("stalled", 0.7)),
        (1.0, outcome("timeout", None)),  # a full open is never a close
        (1.0, outcome("failed", None, "gripper fault: no answer")),
        (0.0, outcome("refused", None, "the jaws are not referenced")),
        (0.0, outcome("refused", 0.4, "torque is off")),
    ],
)
def test_other_command_ends_leave_a_clear_flag_clear(jaw_state, target, ended):
    jaw_state.note_command_result(target, PLAIN, ended)
    assert jaw_state.may_be_holding is False


@pytest.mark.parametrize(
    ("target", "ended"),
    [
        (1.0, outcome("reached", 1.0)),
        (1.0, outcome("reached", 0.97)),  # 1 - target_tol
        (0.98, outcome("reached", 0.98)),
        (1.0, outcome("stalled", 0.99)),  # on the open stop
        (1.0, outcome("cancelled", 0.98)),
    ],
)
def test_a_measured_full_open_clears_the_flag(jaw_state, target, ended):
    jaw_state.note_command_result(0.0, CLOSE_UNTIL_CONTACT, outcome("stalled", 0.4))
    jaw_state.note_command_result(target, PLAIN, ended)
    assert jaw_state.may_be_holding is False


@pytest.mark.parametrize(
    ("target", "ended"),
    [
        (0.0, outcome("reached", 0.0)),  # a close that reached
        (0.6, outcome("reached", 0.6)),  # an opening that is not full
        (0.96, outcome("reached", 0.96)),  # just under 1 - target_tol
        (1.0, outcome("stalled", 0.96)),
        (1.0, outcome("timeout", None)),  # full open asked, nothing measured
        (1.0, outcome("refused", 1.0, "torque is off")),  # nothing was run
        (0.0, outcome("refused", None, "the jaws are not referenced")),
    ],
)
def test_nothing_but_a_measured_full_open_clears_the_flag(jaw_state, target, ended):
    jaw_state.note_command_result(0.0, CLOSE_UNTIL_CONTACT, outcome("stalled", 0.4))
    jaw_state.note_command_result(target, PLAIN, ended)
    assert jaw_state.may_be_holding is True


def test_the_mode_does_not_decide_what_a_close_is(jaw_state):
    jaw_state.note_command_result(0.0, PLAIN, outcome("stalled", 0.4))
    assert jaw_state.may_be_holding is True
    jaw_state.note_command_result(1.0, CLOSE_UNTIL_CONTACT, outcome("reached", 1.0))
    assert jaw_state.may_be_holding is False


# JawState: classify


@pytest.mark.parametrize(
    ("readings", "target"),
    [
        ([1.0, 1.0, 1.0], 1.0),  # full open
        ([0.97, 0.98, 0.99], 1.0),  # at the edge of both tolerances
        ([0.0, 0.0, 0.0], 0.0),  # closed on nothing
        ([0.03, 0.02, 0.03], 0.0),
        ([0.04, 0.04, 0.04], 0.02),  # at closed_frac
    ],
)
def test_classify_proves_the_jaws_empty(jaw_state, readings, target):
    jaws = ScriptedJaws(readings, target)
    assert jaw_state.classify(jaws) is JawLoad.EMPTY
    assert jaw_state.load_reason == ""
    assert jaws.readings == []  # every reading was taken


@pytest.mark.parametrize(
    ("readings", "target", "reason"),
    [
        ([0.5, 0.5, 0.5], 0.5, "the jaws are part open at 0.500"),
        ([0.05, 0.05, 0.05], 0.03, "the jaws are part open at 0.050"),
        ([0.96, 0.96, 0.96], 0.96, "the jaws are part open at 0.960"),
        ([0.4, 0.4, 0.4], 0.0, "the jaws are at 0.400, not at their target 0.000"),
        ([0.96, 0.96, 0.96], 1.0, "the jaws are at 0.960, not at their target 1.000"),
        ([0.5, 0.5, 0.5], 1.0, "the jaws are at 0.500, not at their target 1.000"),
        # Every reading counts, not only the last one.
        ([0.035, 0.02, 0.02], 0.0, "the jaws are at 0.035, not at their target 0.000"),
        ([0.96, 0.975, 0.975], 1.0, "the jaws are at 0.960, not at their target 1.000"),
        ([0.035, 0.045, 0.045], 0.03, "the jaws are part open at 0.045"),
        ([1.0, 1.0, 0.97], 1.0, "the jaw readings disagree: 0.970 to 1.000"),
        ([0.0, 0.03, 0.0], 0.0, "the jaw readings disagree: 0.000 to 0.030"),
        ([1.0, 1.0, 1.0], None, "no jaw target was sent yet"),
        ([0.0, 0.0, 0.0], None, "no jaw target was sent yet"),
    ],
)
def test_classify_counts_anything_else_as_loaded(jaw_state, readings, target, reason):
    assert jaw_state.classify(ScriptedJaws(readings, target)) is JawLoad.LOADED
    assert jaw_state.load_reason == reason


def test_classify_is_loaded_while_the_flag_is_set(jaw_state):
    jaw_state.note_command_result(0.0, CLOSE_UNTIL_CONTACT, outcome("stalled", 0.4))
    # The readings alone would prove the jaws empty.
    assert jaw_state.classify(ScriptedJaws([0.0, 0.0, 0.0], 0.0)) is JawLoad.LOADED
    assert jaw_state.load_reason == "the jaws may be holding something"
    assert jaw_state.classify(ScriptedJaws([1.0, 1.0, 1.0], 1.0)) is JawLoad.LOADED


def test_classify_is_unknown_without_the_reference_and_reads_nothing(jaw_state):
    jaws = ScriptedJaws([1.0, 1.0, 1.0], 1.0, referenced=False)
    assert jaw_state.classify(jaws) is JawLoad.UNKNOWN
    assert jaw_state.load_reason == "the jaws are not referenced"
    assert jaws.readings == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("failed", [0, 1, 2])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_classify_is_unknown_when_any_read_fails(jaw_state, failed, bad):
    readings = [1.0, 1.0, 1.0]
    readings[failed] = bad
    jaws = ScriptedJaws(readings, 1.0)
    assert jaw_state.classify(jaws) is JawLoad.UNKNOWN
    assert jaw_state.load_reason == "a jaw read failed"
    assert len(jaws.readings) == 2 - failed  # no read follows the failed one


def test_a_failed_read_is_unknown_even_with_the_flag_set(jaw_state):
    jaw_state.note_command_result(0.0, CLOSE_UNTIL_CONTACT, outcome("stalled", 0.4))
    assert jaw_state.classify(ScriptedJaws([0.4, None], 0.0)) is JawLoad.UNKNOWN


def test_classify_takes_its_readings_read_gap_apart(jaw_state, clock, config):
    jaws = ScriptedJaws([1.0, 1.0, 1.0], 1.0, clock=clock)
    start = clock.monotonic()
    jaw_state.classify(jaws)
    gap = config.gripper.read_gap_s
    assert jaws.read_times == pytest.approx([start, start + gap, start + 2 * gap])
    assert clock.monotonic() == pytest.approx(start + 2 * gap)


def test_the_number_of_readings_is_a_setting(config, clock):
    cfg = dataclasses.replace(config.gripper, reads=5, read_gap_s=0.25)
    jaws = ScriptedJaws([0.0] * 5, 0.0, clock=clock)
    start = clock.monotonic()
    assert JawState(cfg, clock).classify(jaws) is JawLoad.EMPTY
    assert jaws.readings == []
    assert clock.monotonic() == start + 1.0


def test_classify_reflects_the_newest_call(jaw_state):
    assert jaw_state.classify(ScriptedJaws([0.5, 0.5, 0.5], 0.5)) is JawLoad.LOADED
    assert jaw_state.classify(ScriptedJaws([1.0, 1.0, 1.0], 1.0)) is JawLoad.EMPTY
    assert jaw_state.load_reason == ""
    assert jaw_state.classify(ScriptedJaws([None], 1.0)) is JawLoad.UNKNOWN


# JawState: classify on the simulated arm


def test_fresh_jaws_at_full_open_are_proven_empty(arm, jaw_state):
    assert arm.gripper_read() == 1.0
    assert arm.gripper_last_target() == 1.0  # left by the reference
    assert jaw_state.classify(arm) is JawLoad.EMPTY
    assert jaw_state.load_reason == ""


def test_fresh_jaws_that_are_part_open_are_not_proven_empty(clock, jaw_state):
    part_open = SimArm(clock=clock, jaw_opening=0.5)
    part_open.connect()
    part_open.gripper_reference()
    assert jaw_state.classify(part_open) is JawLoad.LOADED
    assert jaw_state.load_reason == "the jaws are part open at 0.500"


def test_unreferenced_jaws_are_unknown(sim_arm, jaw_state):
    sim_arm.connect()
    forbid_reads(sim_arm)
    assert jaw_state.classify(sim_arm) is JawLoad.UNKNOWN


def test_classify_never_moves_the_jaws(arm, jaw_state):
    jaw_state.classify(arm)
    assert arm.jaw_commands == []
    assert arm.light_holds == []
    assert arm.gripper_read() == 1.0


def test_jaws_opened_fully_are_empty(arm, worker, jaw_state):
    assert run(worker, 1.0) == outcome("reached", 1.0)
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_jaws_closed_on_nothing_are_empty(arm, worker, jaw_state):
    assert run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT) == outcome("reached", 0.0)
    assert jaw_state.may_be_holding is False
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_jaws_at_a_part_open_target_are_loaded(arm, worker, jaw_state):
    assert run(worker, 0.5).status == "reached"
    assert jaw_state.may_be_holding is False
    assert jaw_state.classify(arm) is JawLoad.LOADED
    assert jaw_state.load_reason == "the jaws are part open at 0.500"


def test_a_failed_jaw_read_on_the_arm_is_unknown(arm, worker, jaw_state):
    run(worker, 1.0)
    arm.fail_gripper_reads(1)
    assert jaw_state.classify(arm) is JawLoad.UNKNOWN
    assert jaw_state.classify(arm) is JawLoad.EMPTY  # the reads work again


def test_jaws_still_moving_are_loaded(arm, worker, jaw_state):
    run(worker, 1.0)
    readings = iter([1.0, 0.99, 0.97])
    arm.gripper_read = lambda: next(readings)
    assert jaw_state.classify(arm) is JawLoad.LOADED
    assert jaw_state.load_reason == "the jaw readings disagree: 0.970 to 1.000"


def test_the_flag_outlives_a_close_that_reaches_closed(arm, worker, jaw_state):
    arm.jaw_object(0.4)
    assert run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT).status == "stalled"
    arm.jaw_object(None)  # the object slipped out
    assert run(worker, 0.0) == outcome("reached", 0.0)
    assert jaw_state.may_be_holding is True
    assert jaw_state.classify(arm) is JawLoad.LOADED
    assert jaw_state.load_reason == "the jaws may be holding something"


# JawState: gripper_state


def test_gripper_state_before_anything_is_known(sim_arm, jaw_state):
    assert jaw_state.gripper_state(sim_arm) == GripperState(
        opening=None,
        t=None,
        valid=False,
        referenced=False,
        last_target=None,
        may_be_holding=False,
        load="unknown",
    )


def test_gripper_state_has_no_reading_before_a_command_or_classify(arm, jaw_state):
    forbid_reads(arm)
    state = jaw_state.gripper_state(arm)
    assert (state.opening, state.t, state.valid) == (None, None, False)
    assert (state.referenced, state.load) == (True, "unknown")


def test_gripper_state_after_a_command_reads_nothing(arm, worker, jaw_state, clock):
    arm.jaw_object(0.4)
    ended = run(worker, 0.0, 0.7, CLOSE_UNTIL_CONTACT)
    forbid_reads(arm)
    assert jaw_state.gripper_state(arm) == GripperState(
        opening=ended.opening,
        t=clock.monotonic(),  # nothing has moved the clock since the command
        valid=True,
        referenced=True,
        last_target=0.0,
        may_be_holding=True,
        load="loaded",
    )


def test_gripper_state_keeps_the_time_of_its_reading(arm, worker, jaw_state, clock):
    run(worker, 1.0)
    read_at = clock.monotonic()
    clock.advance(2.0)
    state = jaw_state.gripper_state(arm)
    assert (state.opening, state.t, state.load) == (1.0, read_at, "empty")


def test_gripper_state_follows_classify(arm, worker, jaw_state, clock):
    run(worker, 1.0)
    clock.advance(1.0)
    assert jaw_state.classify(arm) is JawLoad.EMPTY
    state = jaw_state.gripper_state(arm)
    assert (state.opening, state.t, state.valid) == (1.0, clock.monotonic(), True)
    assert state.load == "empty"
    arm.fail_gripper_reads(1)
    assert jaw_state.classify(arm) is JawLoad.UNKNOWN
    state = jaw_state.gripper_state(arm)
    assert (state.opening, state.t, state.valid) == (None, None, False)
    assert (state.last_target, state.load) == (1.0, "unknown")


def test_gripper_state_load_is_what_classify_last_proved(jaw_state):
    jaws = ScriptedJaws([1.0, 1.0, 0.97], 1.0)
    assert jaw_state.classify(jaws) is JawLoad.LOADED  # the readings disagree
    state = jaw_state.gripper_state(jaws)
    assert (state.opening, state.load) == (0.97, "loaded")


def test_gripper_state_hides_a_reading_once_the_reference_is_lost(jaw_state):
    jaws = ScriptedJaws([1.0, 1.0, 1.0], 1.0)
    jaw_state.classify(jaws)
    jaws.referenced = False
    state = jaw_state.gripper_state(jaws)
    assert (state.opening, state.t, state.valid) == (None, None, False)
    assert (state.referenced, state.load) == (False, "unknown")


def test_a_refused_command_still_gives_its_reading(jaw_state, clock):
    jaws = ScriptedJaws(target=1.0)
    jaw_state.note_command_result(0.5, PLAIN, outcome("refused", 1.0, "torque off"))
    state = jaw_state.gripper_state(jaws)
    assert (state.opening, state.t, state.load) == (1.0, clock.monotonic(), "empty")
    # A refusal without a reading keeps the one before it.
    jaw_state.note_command_result(0.5, PLAIN, outcome("refused", None, "torque off"))
    assert jaw_state.gripper_state(jaws).opening == 1.0


def test_gripper_state_travels_in_a_robot_state(arm, worker, jaw_state):
    run(worker, 0.5)
    gripper = jaw_state.gripper_state(arm)
    assert type(gripper.load) is str
    state = RobotState(
        seq=1,
        session_id="s",
        clock_mono_s=0.0,
        clock_wall_s=0.0,
        joints=[],
        recent=[],
        target_q=None,
        target_t=None,
        gripper=gripper,
        owner="agent",
        owner_number=1,
        stop_number=0,
        stop_level="none",
        fault_reason="",
        limit_set_version="v",
        motion="hold",
        move_id="",
        move_index=0,
        move_total=0,
        last_result=None,
        assistance_wait_s=None,
        torque_on=True,
        view_age_s=None,
        teleop_stage="",
        controller_age_s=None,
        health={},
    )
    decoded = messages.decode(RobotState.TYPE, state.to_payload())
    assert decoded.gripper == gripper


def test_jaw_state_is_safe_to_use_from_several_threads(jaw_state):
    start = threading.Barrier(4)
    errors: list[Exception] = []

    def work(kind: int) -> None:
        try:
            start.wait(WAIT_S)
            for _ in range(200):
                if kind == 0:
                    jaw_state.note_command_result(0.0, PLAIN, outcome("stalled", 0.4))
                elif kind == 1:
                    jaw_state.note_command_result(1.0, PLAIN, outcome("reached", 1.0))
                elif kind == 2:
                    jaw_state.classify(ScriptedJaws([1.0, 1.0, 1.0], 1.0))
                else:
                    jaw_state.gripper_state(ScriptedJaws(target=1.0))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=work, args=(kind,)) for kind in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S)
    assert errors == []
    jaw_state.note_command_result(1.0, PLAIN, outcome("reached", 1.0))
    assert jaw_state.classify(ScriptedJaws([1.0, 1.0, 1.0], 1.0)) is JawLoad.EMPTY


# GripperWorker: commands


def test_a_plain_command_runs_on_the_worker_thread_and_reaches(arm, worker, jaw_state):
    done = Done()
    assert worker.submit(0.3, 0.5, PLAIN, Owner.AGENT, done) is None
    assert done.wait() == outcome("reached", 0.3)
    assert done.threads[0].name == "gripper-worker"
    assert done.threads[0] is not threading.current_thread()
    [command] = arm.jaw_commands
    assert (command.opening, command.max_effort) == (0.3, 0.5)
    assert (command.mode, command.status) == (PLAIN, "reached")
    assert arm.gripper_read() == 0.3
    assert jaw_state.may_be_holding is False
    assert worker.wait_idle(WAIT_S)
    assert done.outcomes == [outcome("reached", 0.3)]


def test_the_control_owner_may_be_given_as_text(arm, worker):
    done = Done()
    assert worker.submit(0.3, 0.5, PLAIN, "agent", done) is None
    assert done.wait().status == "reached"


def test_a_close_onto_an_object_stalls_and_sets_the_flag(arm, worker, jaw_state):
    arm.jaw_object(0.4)
    ended = run(worker, 0.0, 0.6, CLOSE_UNTIL_CONTACT)
    assert ended.status == "stalled"
    assert ended.opening == pytest.approx(0.4)
    assert jaw_state.may_be_holding is True
    assert arm.jaw_stiffness == "light"
    assert jaw_state.classify(arm) is JawLoad.LOADED


def test_a_measured_full_open_releases_the_object(arm, worker, jaw_state):
    arm.jaw_object(0.4)
    run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT)
    assert run(worker, 0.7).status == "reached"
    assert jaw_state.may_be_holding is True  # opened, but not fully
    assert run(worker, 1.0) == outcome("reached", 1.0)
    assert jaw_state.may_be_holding is False
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_a_close_that_times_out_sets_the_flag(clock, make_worker, jaw_state):
    slow = SimArm(clock=clock, jaw_speed=0.1, jaw_timeout_s=0.05)
    slow.connect()
    slow.gripper_reference()
    ended = run(make_worker(slow), 0.0)
    assert ended.status == "timeout"
    assert ended.opening == pytest.approx(0.995)
    assert jaw_state.may_be_holding is True


def test_commands_run_one_at_a_time_in_the_order_given(arm, worker):
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    done = [Done(), Done(), Done()]
    worker.submit(0.8, 0.5, PLAIN, Owner.AGENT, done[0])
    pause.wait()
    worker.submit(0.6, 0.5, CLOSE_UNTIL_CONTACT, Owner.AGENT, done[1])
    worker.submit(1.0, 0.5, PLAIN, Owner.AGENT, done[2])
    assert len(arm.jaw_commands) == 1  # the others wait their turn
    pause.release.set()
    assert [d.wait() for d in done] == [
        outcome("reached", 0.8),
        outcome("reached", 0.6),
        outcome("reached", 1.0),
    ]
    assert [command.opening for command in arm.jaw_commands] == [0.8, 0.6, 1.0]
    assert [command.mode for command in arm.jaw_commands][1] == CLOSE_UNTIL_CONTACT
    times = [command.t for command in arm.jaw_commands]
    assert times == sorted(times)


def test_busy_ends_before_on_done_and_wait_idle_after_it(arm, worker):
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    done = Done()
    in_on_done: list[tuple[bool, bool]] = []

    def on_done(ended: GripperOutcome) -> None:
        in_on_done.append((worker.busy(), worker.wait_idle(0.0)))
        done(ended)

    assert not worker.busy()
    assert worker.wait_idle(0.0)
    worker.submit(0.5, 0.5, PLAIN, Owner.AGENT, on_done)
    assert worker.busy()
    pause.wait()
    assert worker.busy()
    assert not worker.wait_idle(0.0)
    pause.release.set()
    done.wait()
    # The command that follows the outcome is not told the jaws are busy; a
    # caller of wait_idle() still finds everything on_done did.
    assert in_on_done == [(False, False)]
    assert worker.wait_idle(WAIT_S)
    assert not worker.busy()


def test_a_queued_request_keeps_the_worker_busy_through_on_done(arm, worker):
    busy_in_on_done: list[bool] = []
    worker.submit(
        0.8, 0.5, PLAIN, Owner.AGENT, lambda _: busy_in_on_done.append(worker.busy())
    )
    last = Done()
    worker.submit(0.6, 0.5, PLAIN, Owner.AGENT, last)
    last.wait()
    assert busy_in_on_done == [True]


def test_busy_lasts_through_the_fault_report_and_the_owed_light_hold(arm, make_worker):
    busy: dict[str, bool] = {}
    worker = make_worker(arm, stop_wait_s=0.01)
    worker.on_fault = lambda reason: busy.setdefault("fault report", worker.busy())
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    real_command, real_hold = arm.gripper_command, arm.gripper_light_hold

    def command_then_fault(opening, max_effort, mode, cancel):
        real_command(opening, max_effort, mode, cancel)
        raise ArmFault("jaw motor over current")

    def light_hold() -> None:
        if threading.current_thread().name == "gripper-worker":
            busy.setdefault("owed light hold", worker.busy())
        real_hold()

    arm.gripper_command, arm.gripper_light_hold = command_then_fault, light_hold
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    pause.wait()
    worker.stop_now()  # gives up waiting: the light hold is owed
    pause.release.set()
    assert done.wait().status == "failed"
    assert busy == {"owed light hold": True, "fault report": True}


def test_a_driver_refusal_comes_back_through_on_done(sim_arm, make_worker, jaw_state):
    sim_arm.connect()  # the jaws are not referenced
    ended = run(make_worker(sim_arm), 0.0)
    assert ended == outcome("refused", None, "the jaws are not referenced")
    assert jaw_state.may_be_holding is False


def test_an_error_in_on_done_does_not_end_the_worker(arm, worker):
    def broken(ended: GripperOutcome) -> None:
        raise RuntimeError("the callback failed")

    assert worker.submit(0.8, 0.5, PLAIN, Owner.AGENT, broken) is None
    assert run(worker, 0.6) == outcome("reached", 0.6)
    # Also for a refusal at once, on the caller's thread.
    assert worker.submit(2.0, 0.5, PLAIN, Owner.AGENT, broken) is not None


def test_on_done_is_optional(arm, worker):
    assert worker.submit(0.8, 0.5, PLAIN, Owner.AGENT) is None
    assert worker.submit(2.0, 0.5, PLAIN, Owner.AGENT) is not None
    assert worker.wait_idle(WAIT_S)
    assert arm.gripper_read() == 0.8


# GripperWorker: refusals at once


def refused_at_once(worker, *args) -> str:
    """Submit, expecting a refusal reported on this thread before submit returns."""
    done = Done()
    reason = worker.submit(*args, done)
    assert reason is not None
    assert done.outcomes == [outcome("refused", None, reason)]
    assert done.threads == [threading.current_thread()]
    return reason


def test_refused_while_a_fault_stop_is_latched(arm, worker, authority):
    authority.raise_fault("motor 3 over temperature")
    reason = refused_at_once(worker, 0.5, 0.5, PLAIN, Owner.AGENT)
    assert reason == "fault-stop latched: motor 3 over temperature"
    # Whoever asks: the control owner is nobody under a fault-stop.
    reason = refused_at_once(worker, 0.5, 0.5, PLAIN, Owner.NOBODY)
    assert reason == "fault-stop latched: motor 3 over temperature"
    assert arm.jaw_commands == []
    assert not worker.busy()


def test_accepted_again_after_the_fault_stop_is_reset(arm, worker, authority):
    stop_number = authority.raise_fault("motor 3 over temperature")
    assert authority.reset(stop_number) is None
    authority.set_owner(Owner.AGENT, "new task")
    assert run(worker, 0.5).status == "reached"


def test_refused_when_the_control_owner_differs(arm, worker, authority):
    reason = refused_at_once(worker, 0.5, 0.5, PLAIN, Owner.HUMAN)
    assert reason == "control owner is agent, not human"
    authority.set_owner(Owner.RECOVERY, "return to rest")
    reason = refused_at_once(worker, 0.5, 0.5, PLAIN, Owner.AGENT)
    assert reason == "control owner is recovery, not agent"
    assert arm.jaw_commands == []


def test_refused_after_a_stop_took_the_arm_from_the_agent(arm, worker, authority):
    authority.raise_stop("operator", "console")
    reason = refused_at_once(worker, 0.5, 0.5, PLAIN, Owner.AGENT)
    assert reason == "control owner is nobody, not agent"


@pytest.mark.parametrize(
    ("opening", "max_effort", "mode", "reason"),
    [
        (-0.01, 0.5, PLAIN, "opening must be between 0 and 1, got -0.01"),
        (1.01, 0.5, PLAIN, "opening must be between 0 and 1, got 1.01"),
        (float("nan"), 0.5, PLAIN, "opening must be between 0 and 1, got nan"),
        (float("inf"), 0.5, PLAIN, "opening must be between 0 and 1, got inf"),
        ("0.5", 0.5, PLAIN, "opening must be a number, got '0.5'"),
        (None, 0.5, PLAIN, "opening must be a number, got None"),
        (True, 0.5, PLAIN, "opening must be a number, got True"),
        (0.5, -0.1, PLAIN, "max_effort must be between 0 and 1, got -0.1"),
        (0.5, 1.5, PLAIN, "max_effort must be between 0 and 1, got 1.5"),
        (0.5, float("nan"), PLAIN, "max_effort must be between 0 and 1, got nan"),
        (0.5, None, PLAIN, "max_effort must be a number, got None"),
        (0.5, 0.5, "squeeze", "unknown mode 'squeeze'"),
        (0.5, 0.5, None, "unknown mode None"),
    ],
)
def test_refused_for_a_value_out_of_range(
    arm, worker, opening, max_effort, mode, reason
):
    assert refused_at_once(worker, opening, max_effort, mode, Owner.AGENT) == reason
    assert arm.jaw_commands == []


@pytest.mark.parametrize(("opening", "max_effort"), [(0.0, 0.0), (1.0, 1.0), (0, 1)])
def test_the_ends_of_the_ranges_are_accepted(arm, worker, opening, max_effort):
    assert run(worker, opening, max_effort).status == "reached"


def test_the_refusals_are_checked_in_a_fixed_order(arm, worker, authority):
    # The control owner comes before the values, the fault-stop before both.
    reason = refused_at_once(worker, 2.0, 0.5, "squeeze", Owner.HUMAN)
    assert reason == "control owner is agent, not human"
    reason = refused_at_once(worker, 2.0, 0.5, "squeeze", Owner.AGENT)
    assert reason == "opening must be between 0 and 1, got 2.0"
    authority.raise_fault("motor fault")
    reason = refused_at_once(worker, 2.0, 0.5, "squeeze", Owner.HUMAN)
    assert reason == "fault-stop latched: motor fault"


def test_an_unknown_control_owner_is_a_caller_error(worker):
    with pytest.raises(ValueError, match=r"'robot' is not a valid Owner"):
        worker.submit(0.5, 0.5, PLAIN, "robot", Done())


# GripperWorker: commands overtaken by the authority


def queue_behind_a_paused_command(arm, worker) -> tuple[JawPause, Done, Done]:
    """One command stopped part-way and a second one queued behind it."""
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    first, second = Done(), Done()
    worker.submit(0.8, 0.5, PLAIN, Owner.AGENT, first)
    pause.wait()
    worker.submit(0.2, 0.5, PLAIN, Owner.AGENT, second)
    return pause, first, second


def test_a_queued_command_is_not_sent_after_a_stop_number_change(
    arm, worker, authority
):
    pause, first, second = queue_behind_a_paused_command(arm, worker)
    authority.raise_stop("operator", "console")  # stop_now() was not called
    pause.release.set()
    assert first.wait() == outcome("reached", 0.8)
    assert second.wait() == outcome(
        "cancelled", None, "a STOP was raised after the command was admitted"
    )
    assert [command.opening for command in arm.jaw_commands] == [0.8]


def test_a_queued_command_is_not_sent_after_an_owner_change(arm, worker, authority):
    pause, _, second = queue_behind_a_paused_command(arm, worker)
    authority.set_owner(Owner.NOBODY, "help needed")
    authority.set_owner(Owner.AGENT, "new task")  # the same owner, a new number
    pause.release.set()
    assert second.wait() == outcome(
        "cancelled", None, "the control owner changed after the command was admitted"
    )
    assert [command.opening for command in arm.jaw_commands] == [0.8]


def test_a_queued_command_is_not_sent_under_a_fault_stop(arm, worker, authority):
    pause, _, second = queue_behind_a_paused_command(arm, worker)
    authority.raise_fault("motor 3 over temperature")
    pause.release.set()
    assert second.wait() == outcome(
        "cancelled", None, "fault-stop latched: motor 3 over temperature"
    )
    assert [command.opening for command in arm.jaw_commands] == [0.8]


def test_a_command_overtaken_by_the_authority_leaves_the_flag_alone(
    arm, worker, authority, jaw_state
):
    pause, _, second = queue_behind_a_paused_command(arm, worker)
    authority.raise_stop("operator", "console")
    pause.release.set()
    second.wait()
    # It was a close with no end reading, but it was never sent.
    assert jaw_state.may_be_holding is False


# GripperWorker: the queue is bounded


def test_no_more_than_four_commands_wait_behind_the_running_one(arm, worker):
    assert MAX_WAITING == 4
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    running = Done()
    assert worker.submit(0.8, 0.5, PLAIN, Owner.AGENT, running) is None
    pause.wait()
    waiting = [Done() for _ in range(MAX_WAITING)]
    for index, done in enumerate(waiting):
        target = 0.5 - 0.1 * index
        assert worker.submit(target, 0.5, PLAIN, Owner.AGENT, done) is None
    # A further one is refused at once, on the caller's thread, and never runs.
    over = Done()
    busy = "busy: 4 jaw commands are waiting"
    assert worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, over) == busy
    assert over.outcomes == [outcome("refused", None, busy)]
    assert over.threads == [threading.current_thread()]
    assert worker.open_guarded(Done(), Owner.AGENT) == busy
    pause.release.set()
    assert running.wait() == outcome("reached", 0.8)
    for index, done in enumerate(waiting):
        assert done.wait().status == "reached", index
    assert worker.wait_idle(WAIT_S)
    assert [command.opening for command in arm.jaw_commands] == pytest.approx(
        [0.8, 0.5, 0.4, 0.3, 0.2]
    )
    # With the queue worked off there is room again.
    assert run(worker, 0.0).status == "reached"


def test_a_stop_makes_room_in_the_queue(arm, worker):
    pause = JawPause(arm, lambda opening: opening <= 0.905)
    worker.submit(0.8, 0.5, PLAIN, Owner.AGENT, Done())
    pause.wait()
    waiting = [Done() for _ in range(MAX_WAITING)]
    for done in waiting:
        assert worker.submit(0.5, 0.5, PLAIN, Owner.AGENT, done) is None
    worker.stop_now()
    pause.release.set()
    assert [done.outcomes for done in waiting] == [[DROPPED]] * MAX_WAITING
    assert worker.wait_idle(WAIT_S)
    assert run(worker, 0.5).status == "reached"


# GripperWorker: stop_now


def test_stop_during_a_close_ends_at_the_light_hold_where_the_jaws_were(
    arm, worker, jaw_state
):
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    closing, queued_close, queued_open = Done(), Done(), Done()
    worker.submit(0.0, 0.8, CLOSE_UNTIL_CONTACT, Owner.AGENT, closing)
    where = pause.wait()
    worker.submit(0.2, 0.5, PLAIN, Owner.AGENT, queued_close)
    worker.submit(1.0, 0.5, PLAIN, Owner.AGENT, queued_open)
    assert where == pytest.approx(0.6)
    assert arm.jaw_stiffness == "closing"
    assert arm.light_holds == []

    worker.stop_now()

    # The queued commands were dropped before stop_now() returned.
    assert queued_close.outcomes == [DROPPED]
    assert queued_open.outcomes == [DROPPED]
    assert closing.wait() == outcome("cancelled", where)
    assert arm.gripper_read() == where
    assert arm.jaw_stiffness == "light"
    assert len(arm.light_holds) == 2  # at once, and again after the return
    # Nothing but the close ever reached the arm: the jaws were not opened.
    assert [command.opening for command in arm.jaw_commands] == [0.0]
    assert arm.jaw_commands[0].status == "cancelled"
    assert arm.jaw_commands[0].stiffness == "light"
    assert arm.released_at == []
    assert jaw_state.may_be_holding is True  # the close ended short of its target
    assert worker.wait_idle(WAIT_S)
    assert [len(d.outcomes) for d in (closing, queued_close, queued_open)] == [1, 1, 1]


def test_the_light_hold_is_sent_at_once_and_again_after_the_command_returned(
    arm, worker
):
    order: list[str] = []
    held = threading.Event()
    inner_command, inner_hold = arm.gripper_command, arm.gripper_light_hold

    def gripper_command(*args):
        try:
            return inner_command(*args)
        finally:
            order.append("command returned")

    def gripper_light_hold():
        order.append(f"light hold at the {arm.jaw_stiffness} stiffness")
        inner_hold()
        held.set()

    arm.gripper_command, arm.gripper_light_hold = gripper_command, gripper_light_hold
    # The close stays under way until the first light hold has been sent.
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: held)
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    pause.wait()
    worker.stop_now()
    done.wait()
    assert order == [
        "light hold at the closing stiffness",
        "command returned",
        "light hold at the light stiffness",
    ]


def test_a_stop_with_nothing_running_still_holds_lightly(arm, worker):
    run(worker, 0.5)
    assert arm.jaw_stiffness == "closing"
    worker.stop_now()
    assert arm.jaw_stiffness == "light"
    assert len(arm.light_holds) == 1
    assert arm.gripper_read() == 0.5
    assert [command.opening for command in arm.jaw_commands] == [0.5]


def test_a_stop_never_opens_jaws_that_hold_an_object(arm, worker, jaw_state):
    arm.jaw_object(0.4)
    held = run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT).opening
    worker.stop_now()
    worker.stop_now()
    assert arm.gripper_read() == held
    assert arm.gripper_last_target() == 0.0
    assert arm.jaw_stiffness == "light"
    assert [command.opening for command in arm.jaw_commands] == [0.0]
    assert jaw_state.may_be_holding is True


def test_a_stop_during_an_opening_leaves_the_jaws_where_they_were(arm, worker):
    arm.jaw_object(0.4)
    run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT)
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening >= 0.695, until=lambda: cancels[-1])
    done = Done()
    worker.submit(1.0, 0.5, PLAIN, Owner.AGENT, done)
    where = pause.wait()
    worker.stop_now()
    assert done.wait() == outcome("cancelled", where)
    assert arm.gripper_read() == where == pytest.approx(0.7)
    assert arm.jaw_stiffness == "light"
    # The light hold is at the opening reached, not at the full open asked for.
    assert arm.gripper_last_target() == where


def test_a_stop_reaches_a_command_taken_from_the_queue_but_not_yet_sent(
    arm, worker, authority
):
    # The stop lands while the worker checks the authority, before the arm call.
    entered, go_on = threading.Event(), threading.Event()
    inner = authority.snapshot
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:  # the worker's check; the first call was submit's
            entered.set()
            go_on.wait(WAIT_S)
        return inner()

    authority.snapshot = snapshot
    taken, queued = Done(), Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, taken)
    assert entered.wait(WAIT_S)
    worker.submit(0.5, 0.5, PLAIN, Owner.AGENT, queued)
    stopper = threading.Thread(target=worker.stop_now)
    stopper.start()
    assert queued.wait() == DROPPED  # by now the stop has cancelled the first too
    go_on.set()
    stopper.join(WAIT_S)
    assert taken.wait() == outcome(
        "cancelled", None, "stopped before the command was sent"
    )
    assert arm.jaw_commands == []
    assert arm.gripper_read() == 1.0
    assert len(arm.light_holds) == 2


def test_a_stop_goes_on_as_soon_as_the_cancelled_command_has_returned(arm, make_worker):
    # With this wait a stop that sat out its whole wait would outlast the test.
    worker = make_worker(arm, stop_wait_s=10 * WAIT_S)
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    pause.wait()
    stopper = threading.Thread(target=worker.stop_now)
    stopper.start()
    stopper.join(WAIT_S)
    assert not stopper.is_alive()
    assert done.wait().status == "cancelled"
    assert len(arm.light_holds) == 2


def test_a_stop_from_inside_a_command_does_not_wait_for_it(arm, make_worker):
    # The wait would outlast the test if stop_now() waited on its own thread.
    worker = make_worker(arm, stop_wait_s=10 * WAIT_S)
    stopped_at: list[float] = []

    def stop_at_once(opening: float) -> None:
        if not stopped_at:
            stopped_at.append(opening)
            worker.stop_now()

    arm.on_jaw_step = stop_at_once
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    assert done.wait() == outcome("cancelled", stopped_at[0])
    assert len(arm.light_holds) == 2  # at once, and again after the return
    assert arm.jaw_stiffness == "light"


def test_a_command_that_does_not_return_in_time_gets_the_light_hold_twice(
    arm, make_worker
):
    worker = make_worker(arm, stop_wait_s=0.01)
    pause = JawPause(arm, lambda opening: opening <= 0.605)
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    where = pause.wait()

    worker.stop_now()  # gives up waiting: the jaws are stuck inside the command

    assert done.outcomes == []
    assert len(arm.light_holds) == 1
    assert arm.jaw_stiffness == "light"
    pause.release.set()
    assert done.wait() == outcome("cancelled", where)
    assert len(arm.light_holds) == 2  # sent again once the command had returned
    assert arm.jaw_stiffness == "light"
    assert arm.gripper_read() == where


def test_stop_now_may_be_called_from_on_done(arm, worker):
    done = Done()

    def on_done(ended: GripperOutcome) -> None:
        worker.stop_now()
        done(ended)

    worker.submit(0.5, 0.5, PLAIN, Owner.AGENT, on_done)
    assert done.wait() == outcome("reached", 0.5)
    assert len(arm.light_holds) == 1
    assert arm.jaw_stiffness == "light"


def test_commands_are_accepted_again_after_a_stop(arm, worker):
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    pause.wait()
    worker.stop_now()
    assert done.wait().status == "cancelled"
    # stop_now() alone changes no number; the authority decides who may command.
    assert run(worker, 1.0) == outcome("reached", 1.0)


# GripperWorker: open_guarded


def hold_an_object(arm, worker, at: float = 0.4) -> None:
    arm.jaw_object(at)
    assert run(worker, 0.0, mode=CLOSE_UNTIL_CONTACT).status == "stalled"


def test_open_guarded_opens_in_steps_of_open_step(arm, worker, jaw_state, config):
    hold_an_object(arm, worker)
    assert open_fully(worker) == outcome("reached", 1.0)
    steps = arm.jaw_commands[1:]
    targets = [command.opening for command in steps]
    assert targets == pytest.approx([0.45 + 0.05 * index for index in range(12)])
    assert targets[-1] == 1.0
    assert {command.mode for command in steps} == {PLAIN}
    assert {command.max_effort for command in steps} == {OPEN_EFFORT}
    assert {command.status for command in steps} == {"reached"}
    previous = [0.4] + targets[:-1]
    step = config.gripper.open_step
    assert all(0 < b - a <= step + 1e-9 for a, b in zip(previous, targets))
    assert arm.gripper_read() == 1.0
    assert jaw_state.may_be_holding is False
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_open_guarded_ends_with_a_short_last_step(arm, worker):
    run(worker, 0.87)
    open_fully(worker)
    targets = [command.opening for command in arm.jaw_commands[1:]]
    assert targets == pytest.approx([0.92, 0.97, 1.0])


def test_open_guarded_on_open_jaws_still_sends_the_full_open_target(
    arm, worker, jaw_state
):
    assert open_fully(worker) == outcome("reached", 1.0)
    assert [command.opening for command in arm.jaw_commands] == [1.0]
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_open_guarded_takes_the_effort_it_is_given(arm, worker):
    run(worker, 0.9)
    done = Done()
    assert worker.open_guarded(done, max_effort=0.25) is None
    done.wait()
    assert [command.max_effort for command in arm.jaw_commands[1:]] == [0.25, 0.25]
    done = Done()
    reason = worker.open_guarded(done, max_effort=1.5)
    assert reason == "max_effort must be between 0 and 1, got 1.5"
    assert done.outcomes == [outcome("refused", None, reason)]


def test_a_stop_ends_open_guarded_between_its_steps(arm, worker, jaw_state):
    hold_an_object(arm, worker)
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening >= 0.615, until=lambda: cancels[-1])
    done = Done()
    worker.open_guarded(done)
    where = pause.wait()
    worker.stop_now()
    assert done.wait() == outcome("cancelled", where)
    targets = [command.opening for command in arm.jaw_commands[1:]]
    assert targets == pytest.approx([0.45, 0.5, 0.55, 0.6, 0.65])  # no later step
    assert arm.jaw_commands[-1].status == "cancelled"
    assert arm.gripper_read() == where == pytest.approx(0.62)
    assert arm.jaw_stiffness == "light"
    assert len(arm.light_holds) == 2
    assert jaw_state.may_be_holding is True  # not measured fully open


def test_open_guarded_sends_no_step_after_a_stop_number_change(arm, worker, authority):
    hold_an_object(arm, worker)
    raised = False

    def raise_stop_once(opening: float) -> None:
        nonlocal raised
        if not raised and opening >= 0.515:
            raised = True
            authority.raise_stop("operator", "console")  # stop_now() is not called

    arm.on_jaw_step = raise_stop_once
    ended = open_fully(worker)
    assert ended.status == "cancelled"
    assert ended.detail == "a STOP was raised after the command was admitted"
    assert ended.opening == pytest.approx(0.55)  # the step under way was finished
    targets = [command.opening for command in arm.jaw_commands[1:]]
    assert targets == pytest.approx([0.45, 0.5, 0.55])


def test_open_guarded_stopped_before_its_first_step_reports_what_it_read(
    arm, worker, authority
):
    hold_an_object(arm, worker)
    inner = arm.gripper_read

    def read_then_stop() -> float | None:
        value = inner()
        authority.raise_stop("operator", "console")
        return value

    arm.gripper_read = read_then_stop
    ended = open_fully(worker)
    assert ended.status == "cancelled"
    assert ended.opening == pytest.approx(0.4)
    assert len(arm.jaw_commands) == 1  # the close only


def test_open_guarded_stops_at_a_step_that_does_not_reach(arm, worker, jaw_state):
    hold_an_object(arm, worker)
    inner = arm.gripper_command

    def jammed_at_half(opening, max_effort, mode, cancel):
        if opening > 0.5 + 1e-9:
            return outcome("timeout", 0.5)
        return inner(opening, max_effort, mode, cancel)

    arm.gripper_command = jammed_at_half
    assert open_fully(worker) == outcome("timeout", 0.5)
    targets = [command.opening for command in arm.jaw_commands[1:]]
    assert targets == pytest.approx([0.45, 0.5])
    assert jaw_state.may_be_holding is True


def test_open_guarded_is_refused_under_a_fault_stop(arm, worker, authority):
    authority.raise_fault("gripper fault")
    done = Done()
    assert worker.open_guarded(done) == "fault-stop latched: gripper fault"
    assert done.outcomes == [
        outcome("refused", None, "fault-stop latched: gripper fault")
    ]
    assert arm.jaw_commands == []


def test_open_guarded_checks_the_control_owner_only_when_given_one(
    arm, worker, authority
):
    done = Done()
    reason = worker.open_guarded(done, Owner.NOBODY)
    assert reason == "control owner is agent, not nobody"
    assert done.outcomes == [outcome("refused", None, reason)]
    assert arm.jaw_commands == []
    authority.set_owner(Owner.NOBODY, "task over")
    assert open_fully(worker, Owner.NOBODY).status == "reached"
    assert open_fully(worker).status == "reached"


def test_open_guarded_sends_nothing_when_the_opening_cannot_be_read(
    arm, worker, jaw_state
):
    hold_an_object(arm, worker)
    arm.fail_gripper_reads(1)
    ended = open_fully(worker)
    assert ended.status == "refused"
    assert ended.detail == (
        "the jaw opening could not be read, so no opening step was sent"
    )
    assert len(arm.jaw_commands) == 1  # the close only
    assert jaw_state.may_be_holding is True
    assert open_fully(worker).status == "reached"  # the next try reads fine


def test_open_guarded_needs_referenced_jaws(sim_arm, make_worker):
    sim_arm.connect()
    ended = open_fully(make_worker(sim_arm))
    assert ended == outcome("refused", None, "the jaws are not referenced")
    assert sim_arm.jaw_commands == []


def test_open_guarded_waits_its_turn_behind_a_command(arm, worker):
    pause = JawPause(arm, lambda opening: opening <= 0.955)
    closing, opening = Done(), Done()
    worker.submit(0.9, 0.5, PLAIN, Owner.AGENT, closing)
    pause.wait()
    worker.open_guarded(opening)
    pause.release.set()
    assert closing.wait() == outcome("reached", 0.9)
    assert opening.wait() == outcome("reached", 1.0)
    targets = [command.opening for command in arm.jaw_commands]
    assert targets == pytest.approx([0.9, 0.95, 1.0])


# GripperWorker: driver faults


def test_a_driver_fault_is_reported_and_the_worker_keeps_running(
    arm, worker, jaw_state
):
    seen: list[str] = []
    done = Done()
    worker.on_fault = lambda reason: seen.append(f"fault: {reason}")
    inner = arm.gripper_command

    def faulty(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor over current")

    def on_done(ended: GripperOutcome) -> None:
        seen.append(f"done: {ended.status}")
        done(ended)

    arm.gripper_command = faulty
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, on_done)
    reason = "gripper fault: jaw motor over current"
    assert done.wait() == outcome("failed", None, reason)
    assert seen == [f"fault: {reason}", "done: failed"]
    assert jaw_state.may_be_holding is True  # a close whose end nobody measured
    arm.gripper_command = inner
    assert run(worker, 1.0) == outcome("reached", 1.0)


def test_any_other_driver_error_is_reported_as_a_fault_too(arm, make_worker):
    faults: list[str] = []
    worker = make_worker(arm, on_fault=faults.append)

    def broken(opening, max_effort, mode, cancel):
        raise RuntimeError("bus off")

    arm.gripper_command = broken
    reason = "jaw command error: RuntimeError('bus off')"
    assert run(worker, 1.0) == outcome("failed", None, reason)
    assert faults == [reason]


def test_a_fault_handler_may_latch_and_stop(arm, worker, authority):
    def on_fault(reason: str) -> None:
        authority.raise_fault(reason)
        worker.stop_now()

    def faulty(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor over current")

    worker.on_fault = on_fault
    arm.gripper_command = faulty
    assert run(worker, 0.0).status == "failed"
    assert len(arm.light_holds) == 1
    reason = refused_at_once(worker, 1.0, 0.5, PLAIN, Owner.AGENT)
    assert reason == "fault-stop latched: gripper fault: jaw motor over current"


def test_an_error_in_the_fault_handler_does_not_end_the_worker(arm, worker):
    def broken_handler(reason: str) -> None:
        raise RuntimeError("the handler failed")

    def faulty(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor over current")

    inner = arm.gripper_command
    worker.on_fault = broken_handler
    arm.gripper_command = faulty
    assert run(worker, 0.0).status == "failed"
    arm.gripper_command = inner
    assert run(worker, 1.0) == outcome("reached", 1.0)


def test_a_driver_fault_without_a_handler_only_fails_the_command(arm, worker):
    def faulty(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor over current")

    arm.gripper_command = faulty
    assert run(worker, 0.0).status == "failed"


def test_an_error_outside_the_arm_command_fails_the_request(arm, make_worker):
    faults: list[str] = []
    worker = make_worker(arm, on_fault=faults.append)
    inner = arm.gripper_referenced

    def broken() -> bool:
        raise RuntimeError("bus off")

    arm.gripper_referenced = broken
    reason = "jaw command error: RuntimeError('bus off')"
    assert open_fully(worker) == outcome("failed", None, reason)
    assert faults == [reason]
    assert arm.jaw_commands == []
    arm.gripper_referenced = inner
    assert open_fully(worker) == outcome("reached", 1.0)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ArmFault("no answer"), "gripper fault: the light hold failed: no answer"),
        (
            RuntimeError("bus off"),
            "jaw command error: the light hold failed: RuntimeError('bus off')",
        ),
    ],
)
def test_a_failed_light_hold_is_reported_once(arm, worker, error, reason):
    reported: list[str] = []

    def on_fault(text: str) -> None:
        reported.append(text)
        worker.stop_now()  # as a fault-stop does; its light hold fails again

    def no_answer() -> None:
        raise error

    worker.on_fault = on_fault
    arm.gripper_light_hold = no_answer
    worker.stop_now()  # does not raise
    assert reported == [reason]
    worker.stop_now()
    assert reported == [reason, reason]  # a later stop reports its own failure


def test_a_light_hold_that_fails_twice_in_one_stop_is_reported_once(arm, worker):
    reported: list[str] = []
    tries: list[int] = []

    def no_answer() -> None:
        tries.append(1)
        raise ArmFault("no answer")

    worker.on_fault = reported.append
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    done = Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, done)
    pause.wait()
    arm.gripper_light_hold = no_answer
    worker.stop_now()
    assert done.wait().status == "cancelled"
    assert len(tries) == 2  # at once, and again after the return
    assert reported == ["gripper fault: the light hold failed: no answer"]


# GripperWorker: close


def test_close_ends_the_thread_and_later_requests_are_refused(arm, make_worker):
    worker = make_worker(arm)
    run(worker, 0.5)
    worker.close()
    assert not worker._thread.is_alive()
    assert refused_at_once(worker, 1.0, 0.5, PLAIN, Owner.AGENT) == (
        "the gripper worker is closed"
    )
    done = Done()
    assert worker.open_guarded(done) == "the gripper worker is closed"
    assert done.outcomes == [outcome("refused", None, "the gripper worker is closed")]
    assert arm.light_holds == []  # nothing was running: nothing was sent
    worker.close()  # a second close does nothing


def test_close_stops_a_command_under_way(arm, make_worker):
    worker = make_worker(arm)
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    running, queued = Done(), Done()
    worker.submit(0.0, 0.5, PLAIN, Owner.AGENT, running)
    where = pause.wait()
    worker.submit(1.0, 0.5, PLAIN, Owner.AGENT, queued)
    worker.close()
    assert running.outcomes == [outcome("cancelled", where)]
    assert queued.outcomes == [DROPPED]
    assert not worker._thread.is_alive()
    assert arm.jaw_stiffness == "light"
    assert arm.gripper_read() == where


def test_close_may_be_called_from_on_done(arm, make_worker):
    worker = make_worker(arm)
    done = Done()

    def on_done(ended: GripperOutcome) -> None:
        worker.close()  # must not wait for its own thread
        done(ended)

    worker.submit(0.5, 0.5, PLAIN, Owner.AGENT, on_done)
    assert done.wait() == outcome("reached", 0.5)
    worker._thread.join(WAIT_S)
    assert not worker._thread.is_alive()
    assert arm.light_holds == []


def test_the_worker_thread_does_not_keep_the_interpreter_alive(arm, worker):
    assert worker._thread.daemon


# The jaw reference: asked for at the Thor terminal only


@pytest.fixture
def bare(sim_arm, make_worker, authority) -> tuple[SimArm, GripperWorker]:
    """The arm connected with its jaws not referenced, and nobody owning it."""
    authority.set_owner(Owner.NOBODY, "the task is over")
    sim_arm.connect()
    return sim_arm, make_worker(sim_arm)


def reference_refused_at_once(worker) -> str:
    done = Done()
    reason = worker.reference(done)
    assert reason is not None
    assert done.outcomes == [outcome("refused", None, reason)]
    assert done.threads == [threading.current_thread()]
    return reason


def nothing_sent_to_the_jaws(arm: SimArm) -> bool:
    return arm.jaw_commands == [] and arm.light_holds == []


def test_a_reference_references_the_jaws_and_keeps_may_be_holding(bare, jaw_state):
    arm, worker = bare
    # A close that ended short, before the reference was lost.
    jaw_state.note_command_result(0.0, PLAIN, outcome("stalled", 0.4))
    assert jaw_state.gripper_state(arm).opening is None  # not referenced: unknown
    done = Done()
    assert worker.reference(done) is None
    assert done.wait() == outcome("reached", 1.0)
    assert done.threads == [worker._thread]
    assert worker.wait_idle(WAIT_S)
    assert arm.gripper_referenced() and nothing_sent_to_the_jaws(arm)
    state = jaw_state.gripper_state(arm)
    assert (state.referenced, state.valid, state.opening) == (True, True, 1.0)
    # Only a measured full open after a command clears the flag.
    assert state.may_be_holding and state.load == "loaded"
    assert open_fully(worker) == outcome("reached", 1.0)
    assert not jaw_state.may_be_holding
    assert jaw_state.classify(arm) is JawLoad.EMPTY


def test_reference_is_refused_while_the_agent_owns_the_arm(bare, authority):
    arm, worker = bare
    authority.set_owner(Owner.AGENT, "new task")
    reason = reference_refused_at_once(worker)
    assert reason == "control owner is agent, not nobody"
    assert not arm.gripper_referenced() and nothing_sent_to_the_jaws(arm)


def test_reference_is_refused_while_a_fault_stop_is_latched(bare, authority):
    arm, worker = bare
    authority.raise_fault("motor 3 over temperature")
    reason = reference_refused_at_once(worker)
    assert reason == "fault-stop latched: motor 3 over temperature"
    assert not arm.gripper_referenced()


def test_reference_is_refused_when_the_jaws_are_referenced(bare):
    arm, worker = bare
    arm.gripper_reference()
    calls: list[object] = []
    arm.gripper_reference = lambda cancel=None: calls.append(cancel)
    assert reference_refused_at_once(worker) == ALREADY_REFERENCED
    assert ALREADY_REFERENCED == "the jaws are referenced already"
    assert calls == [] and nothing_sent_to_the_jaws(arm)


def test_reference_is_refused_while_a_jaw_command_runs(bare):
    arm, worker = bare
    arm.gripper_reference()
    pause = JawPause(arm, lambda opening: opening <= 0.8)
    opening = Done()
    assert worker.submit(0.5, 0.5, PLAIN, Owner.NOBODY, opening) is None
    pause.wait()
    # The jaws are referenced here; without that the refusal is the busy one.
    arm._jaw_referenced = False
    assert reference_refused_at_once(worker) == JAWS_BUSY
    arm._jaw_referenced = True
    pause.release.set()
    assert opening.wait().status == "reached"


def test_reference_is_not_a_jaw_mode_that_a_command_can_ask_for(bare):
    # "reference" is not a jaw mode: a GRIPPER link message cannot ask for one.
    arm, worker = bare
    assert REFERENCE not in JAW_MODES
    reason = refused_at_once(worker, 0.0, 0.0, REFERENCE, Owner.NOBODY)
    assert reason == "unknown mode 'reference'"
    assert not arm.gripper_referenced()


def test_a_stop_ends_a_reference_without_a_fault_stop(bare, authority):
    arm, worker = bare
    faults: list[str] = []
    worker.on_fault = faults.append
    entered, cancels = threading.Event(), []
    inner = arm.gripper_reference

    def slow_reference(cancel=None):
        # The driver's slow close: it ends at its next pass once cancel is set.
        cancels.append(cancel)
        entered.set()
        assert cancel.wait(WAIT_S)
        inner(cancel)

    arm.gripper_reference = slow_reference
    done = Done()
    assert worker.reference(done) is None
    assert entered.wait(WAIT_S)
    assert worker.busy()
    authority.raise_stop("operator pressed stop", "thor_terminal")
    worker.stop_now()
    ended = done.wait()
    assert (ended.status, ended.opening) == ("cancelled", None)
    assert ended.detail == "stopped before the jaws were referenced"
    assert faults == [] and authority.stop_level.value == "stop"
    assert not arm.gripper_referenced() and arm.jaw_commands == []
    assert arm.torque_on()
    # The stop's light hold went out; a reference can be asked for again.
    assert len(arm.light_holds) >= 1
    assert worker.wait_idle(WAIT_S)
    arm.gripper_reference = inner
    again = Done()
    assert worker.reference(again) is None
    assert again.wait().status == "reached"


def test_a_reference_the_driver_fails_is_a_fault(bare):
    arm, worker = bare
    faults: list[str] = []
    worker.on_fault = faults.append

    def stuck(cancel=None):
        raise ArmFault("the jaws stopped short of the closed stop")

    arm.gripper_reference = stuck
    done = Done()
    assert worker.reference(done) is None
    ended = done.wait()
    reason = "the jaw reference failed: the jaws stopped short of the closed stop"
    assert ended == outcome("failed", None, reason)
    assert faults == [reason]
    assert not arm.gripper_referenced() and arm.jaw_commands == []


def held_reference(arm: SimArm) -> tuple[threading.Event, threading.Event]:
    """Hold the arm's jaw reference inside its call; (entered, release)."""
    entered, release = threading.Event(), threading.Event()
    inner = arm.gripper_reference

    def slow_reference(cancel=None):
        entered.set()
        assert release.wait(WAIT_S)
        inner(cancel)

    arm.gripper_reference = slow_reference
    return entered, release


def test_referencing_is_true_from_the_request_until_its_outcome(bare):
    arm, worker = bare
    assert not worker.referencing()
    entered, release = held_reference(arm)
    seen: list[bool] = []
    done = Done()

    def on_done(ended: GripperOutcome) -> None:
        seen.append(worker.referencing())
        done(ended)

    assert worker.reference(on_done) is None
    assert worker.referencing()
    assert entered.wait(WAIT_S) and worker.referencing()
    release.set()
    assert done.wait().status == "reached"
    assert seen == [False] and not worker.referencing()
    # Another jaw command is no reference.
    pause = JawPause(arm, lambda opening: opening <= 0.8)
    closing = Done()
    assert worker.submit(0.5, 0.5, PLAIN, Owner.NOBODY, closing) is None
    pause.wait()
    assert worker.busy() and not worker.referencing()
    pause.release.set()
    assert closing.wait().status == "reached"


def test_a_stop_that_keeps_the_reference_drops_only_what_waits_behind_it(bare):
    arm, worker = bare
    entered, release = held_reference(arm)
    reference, queued = Done(), Done()
    assert worker.reference(reference) is None
    assert entered.wait(WAIT_S)
    assert worker.open_guarded(queued, Owner.NOBODY) is None
    worker.stop_now(keep_reference=True)
    assert queued.outcomes == [DROPPED]
    # The reference was not cancelled and no light hold went to the closing jaws.
    assert reference.outcomes == [] and worker.referencing()
    assert arm.light_holds == []
    release.set()
    assert reference.wait() == outcome("reached", 1.0)
    assert arm.gripper_referenced() and nothing_sent_to_the_jaws(arm)


def test_a_stop_that_keeps_the_reference_still_cancels_any_other_command(bare):
    arm, worker = bare
    arm.gripper_reference()
    cancels = spy_on_cancel(arm)
    pause = JawPause(arm, lambda opening: opening <= 0.605, until=lambda: cancels[-1])
    closing, queued = Done(), Done()
    worker.submit(0.0, 0.8, CLOSE_UNTIL_CONTACT, Owner.NOBODY, closing)
    where = pause.wait()
    worker.submit(1.0, 0.5, PLAIN, Owner.NOBODY, queued)
    worker.stop_now(keep_reference=True)
    assert queued.outcomes == [DROPPED]
    assert closing.wait() == outcome("cancelled", where)
    assert arm.jaw_stiffness == "light" and len(arm.light_holds) == 2
