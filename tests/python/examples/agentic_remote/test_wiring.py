# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app's modules wired together on the simulated arm, without app.py.

Each test crosses module borders; the modules' own rules are in their test files.
"""

from __future__ import annotations

import ast
import dataclasses
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
import isaaccapture_examples.agentic_remote as agentic_remote
from isaaccapture_examples.agentic_remote.protocol.carrier import receive_message, send
from isaaccapture_examples.agentic_remote.protocol.messages import Permit, RobotState
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    GripperWorker,
    JawState,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    CheckerSet,
    GripperOutcome,
    MotionState,
    MoveOutcome,
    Owner,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.permits import PermitGate
from isaaccapture_examples.agentic_remote.robot_app.rest import ReturnToRest
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.state import (
    IdleReader,
    JointHistory,
    StatePublisher,
)
from isaaccapture_examples.agentic_remote.robot_app.stop import StopController
from isaaccapture_examples.agentic_remote.robot_app.waits import AssistanceWait

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
PERIOD = 0.02
START_TOL = 0.02
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
RAISED = [0.5, -0.4, 0.3, 0.0, 0.2, -0.1]
LOST_FEEDBACK_S = 0.35


class MotionBox:
    """The arm's motion state, as the robot app keeps it for rest.py and the state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = MotionState.HOLD

    def get(self) -> MotionState:
        with self._lock:
            return self._state

    def set(self, state: MotionState) -> None:
        with self._lock:
            self._state = state


class Wired:
    """Every module, built in the order app.py has to build them."""

    def __init__(
        self,
        arm: SimArm,
        checkers: CheckerSet,
        config: RobotAppConfig,
        clock: FakeClock,
    ) -> None:
        self.arm, self.checkers, self.config, self.clock = arm, checkers, config, clock
        self.events: list[tuple[str, str, dict]] = []
        self.expired: list[tuple[bool, int]] = []  # (timed return, stop number then)
        self.motion = MotionBox()
        self._stop_reason: str | None = None
        self._stop_lock = threading.Lock()

        self.authority = Authority()
        self.history = JointHistory(config.n_joints, config.state.history_s)
        self.gate = PermitGate(self.authority, checkers.agent, config.permits, clock)
        self.executor = MoveExecutor(
            arm,
            self.authority,
            self.history,
            config.executor,
            clock,
            first_wait_s=config.permits.first_wait_s,
            on_missed_deadline=lambda: self.publisher.note_missed_deadline(),
        )
        self.jaw_state = JawState(config.gripper, clock)
        self.worker = GripperWorker(arm, self.jaw_state, self.authority, clock)
        self.wait = AssistanceWait(config.waits, clock, self._on_expire)
        self.stops = StopController(
            self.authority,
            checkers,
            self.worker,
            self.wait,
            self.emit,
            self.request_motion_stop,
        )
        self.worker.on_fault = self.stops.fault
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
            lambda: None,  # no grasp region was used
            lambda: False,  # no contact hold
            lambda: False,  # rest.py asks the executor itself
            jaws_busy=self.worker.busy,
        )
        self.idle = IdleReader(arm, self.history, config.state, clock)
        self.publisher = StatePublisher(
            arm,
            self.history,
            self.authority.snapshot,
            self.executor.progress,
            self.motion.get,
            lambda: self.jaw_state.gripper_state(arm),
            lambda: self.wait.remaining(clock.monotonic()),
            lambda: None,
            self.authority.session_id,
            config,
            clock,
            read_failures=self.idle.failures,
        )

    # What app.py gives the modules

    def emit(self, kind: str, text: str, data: dict) -> None:
        self.events.append((kind, text, data))

    def request_motion_stop(self, reason: str) -> None:
        with self._stop_lock:
            self._stop_reason = reason

    def should_stop(self) -> str | None:
        with self._stop_lock:
            return self._stop_reason

    def _on_expire(self, timed_return: bool) -> None:
        self.expired.append((timed_return, self.authority.stop_number))

    # What app.py does with them

    def start(self) -> None:
        """Connect once and reference the jaws, as the app does at rest."""
        self.arm.connect()
        self.arm.gripper_reference()

    def new_task(self) -> None:
        self.authority.set_owner(Owner.AGENT, "new task")

    def permit(self, through_index: int, move_id: str = "move-1", **changes) -> Permit:
        state = self.authority.snapshot()
        permit = Permit(
            move_id=move_id,
            session_id=state.session_id,
            owner_number=state.owner_number,
            stop_number=state.stop_number,
            through_index=through_index,
            evidence_wall_s=self.clock.wall(),
            expires_mono_s=self.clock.monotonic() + 5.0,
        )
        return dataclasses.replace(permit, **changes)

    def agent_move(
        self,
        path: Sequence[Sequence[float]] = PATH,
        start_q: Sequence[float] = REST,
        should_stop: Callable[[], str | None] | None = None,
    ) -> MoveOutcome:
        """Run one MOVE for the agent the way the motion worker does."""
        with self._stop_lock:
            self._stop_reason = None  # a new move starts with no stop pending
        self.motion.set(MotionState.MOVING)
        try:
            return self.executor.run(
                path,
                start_q,
                START_TOL,
                PERIOD,
                Owner.AGENT,
                self.checkers.agent,
                self.authority.snapshot(),
                "move-1",
                self.gate,
                None,
                should_stop or self.should_stop,
                permit_arrived=self.gate.arrived,
                halt_count=self.checkers.agent.halt_count(),
            )
        finally:
            self.motion.set(MotionState.HOLD)

    def sent(self) -> list[list[float]]:
        return [q for q, _ in self.arm.sent]


class Background:
    """Runs one call on another thread and hands back what it returned."""

    def __init__(self, call: Callable[[], object]) -> None:
        self._result: list[object] = []
        self._thread = threading.Thread(
            target=lambda: self._result.append(call()), daemon=True
        )
        self._thread.start()

    def result(self) -> object:
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the other thread did not finish"
        return self._result[0]


class Done:
    """An on_done for a jaw command that can be waited for."""

    def __init__(self) -> None:
        self.outcomes: list[GripperOutcome] = []
        self._called = threading.Event()

    def __call__(self, outcome: GripperOutcome) -> None:
        self.outcomes.append(outcome)
        self._called.set()

    def wait(self) -> GripperOutcome:
        assert self._called.wait(WAIT_S), "the jaw command did not end"
        return self.outcomes[0]


def pause_jaws_at(arm: SimArm, opening: float) -> tuple[threading.Event, list]:
    """Hold the next jaw command at `opening` until it is cancelled.

    Returns the event set once the jaws wait there and the cancel events seen.
    """
    reached = threading.Event()
    cancels: list[threading.Event] = []
    real = arm.gripper_command

    def gripper_command(target, max_effort, mode, cancel):
        cancels.append(cancel)
        return real(target, max_effort, mode, cancel)

    def on_step(at: float) -> None:
        if not reached.is_set() and at <= opening + 1e-9:
            reached.set()
            cancels[-1].wait(WAIT_S)

    arm.gripper_command = gripper_command
    arm.on_jaw_step = on_step
    return reached, cancels


def pause_move_after(
    arm: SimArm, count: int
) -> tuple[threading.Event, threading.Event]:
    """Hold the motion thread once `count` setpoints were sent, until released."""
    reached, release = threading.Event(), threading.Event()
    real = arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(arm.sent) == count:
            reached.set()
            assert release.wait(WAIT_S)

    arm.send_target = send_target
    return reached, release


@pytest.fixture
def app(sim_arm, checkers, config, clock) -> Iterator[Wired]:
    wired = Wired(sim_arm, checkers, config, clock)
    wired.start()
    yield wired
    wired.worker.close()


# Agent moves and permits


def test_an_agent_move_starts_on_its_first_permit_and_runs_to_its_end(app):
    app.new_task()
    asked: list[int] = []
    waiting = threading.Event()

    def should_stop() -> str | None:
        asked.append(1)
        if len(asked) >= 3:  # the wait for the first permit has begun
            waiting.set()
        return None

    move = Background(lambda: app.agent_move(should_stop=should_stop))
    assert waiting.wait(WAIT_S)
    assert app.arm.sent == []  # MOVE alone moves nothing
    # The permit is the only thing that feeds the agent checker's watchdog.
    assert app.gate.update(app.permit(len(PATH) - 1)) is None
    outcome = move.result()
    assert (outcome.status, outcome.index) == ("completed", len(PATH))
    assert app.sent() == PATH
    assert outcome.end_q == pytest.approx(PATH[-1], abs=0.03)
    # The move's own joint reads are in the history the state is built from.
    assert app.history.age(app.clock.monotonic()) < 0.2
    assert app.executor.progress() == ("", 0, 0)


def test_without_a_permit_no_setpoint_is_sent(sim_arm, checkers, config, clock):
    permits = dataclasses.replace(config.permits, first_wait_s=0.005)
    app = Wired(sim_arm, checkers, dataclasses.replace(config, permits=permits), clock)
    app.start()
    try:
        app.new_task()
        outcome = app.agent_move()
        assert outcome.status == "refused"
        assert outcome.reason == "no permit for this move (waited 0.005 s)"
        assert app.arm.sent == []
    finally:
        app.worker.close()


@pytest.mark.parametrize(
    "stale",
    [{"session_id": "another-run"}, {"owner_number": 0}, {"stop_number": 7}],
    ids=["session id", "owner number", "stop number"],
)
def test_a_permit_that_quotes_old_values_moves_nothing(
    sim_arm, checkers, config, clock, stale
):
    permits = dataclasses.replace(config.permits, first_wait_s=0.005)
    app = Wired(sim_arm, checkers, dataclasses.replace(config, permits=permits), clock)
    app.start()
    try:
        app.new_task()
        assert app.gate.update(app.permit(len(PATH) - 1, **stale)) is not None
        assert app.agent_move().status == "refused"
        assert app.arm.sent == []
    finally:
        app.worker.close()


def test_when_permits_stop_the_arm_holds_at_the_last_permitted_setpoint(app):
    app.new_task()
    app.gate.update(app.permit(9))
    outcome = app.agent_move()
    assert outcome.status == "refused"
    assert outcome.reason == "permit reaches setpoint 9, not 10"
    assert app.sent() == PATH[:10]
    assert app.arm.last_target()[0] == PATH[9]
    assert app.arm.torque_on()
    assert app.arm.holds == []


def test_a_shorter_permit_from_newer_evidence_shortens_a_running_move(app):
    app.new_task()
    app.gate.update(app.permit(len(PATH) - 1))
    real = app.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(app.arm.sent) == 5:
            assert app.gate.update(app.permit(7)) is None

    app.arm.send_target = send_target
    outcome = app.agent_move()
    assert outcome.status == "refused"
    assert outcome.reason == "permit reaches setpoint 7, not 8"
    assert app.sent() == PATH[:8]


# STOP


def test_stop_during_a_move_and_a_jaw_close_holds_both(app):
    app.new_task()
    jaws_waiting, _ = pause_jaws_at(app.arm, 0.6)
    closing = Done()
    assert app.worker.submit(0.0, 0.5, "plain", Owner.AGENT, closing) is None
    assert jaws_waiting.wait(WAIT_S)
    move_waiting, release = pause_move_after(app.arm, 6)
    app.gate.update(app.permit(len(PATH) - 1))
    quoted = app.authority.snapshot()
    move = Background(app.agent_move)
    assert move_waiting.wait(WAIT_S)

    number = app.stops.stop("operator pressed stop", "console")
    release.set()

    outcome = move.result()
    assert outcome.status == "stopped"
    assert outcome.reason == "STOP from console: operator pressed stop"
    assert app.sent() == PATH[:6]  # the next setpoint was not sent
    assert closing.wait().status == "cancelled"
    assert app.arm.gripper_read() == pytest.approx(0.6)  # where the jaws were
    assert app.arm.jaw_stiffness == "light"
    assert [command.opening for command in app.arm.jaw_commands] == [0.0]
    # Nothing was released and nobody owns the arm.
    assert app.arm.torque_on() and app.arm.released_at == []
    assert app.authority.owner is Owner.NOBODY
    assert number == app.authority.stop_number == quoted.stop_number + 1
    # Whatever quotes the values from before the STOP is refused.
    admit = app.authority.admit
    assert "stop number" in admit(
        quoted.session_id, quoted.owner_number, quoted.stop_number, Owner.AGENT
    )
    assert app.gate.update(app.permit(24, stop_number=quoted.stop_number)) is not None
    assert app.worker.submit(1.0, 0.5, "plain", Owner.AGENT) == (
        "control owner is nobody, not agent"
    )
    assert [kind for kind, _, _ in app.events] == ["stop"]


def test_only_a_stop_from_a_person_cancels_the_assistance_wait(app):
    app.wait.start("the agent is stuck", True)
    app.stops.stop("giving up", "agent")
    assert app.wait.active()
    app.stops.stop("right B", "controller")
    assert not app.wait.active()
    for _ in range(200):
        app.clock.advance(1.0)
        assert app.wait.step(app.clock.monotonic()) == []
    assert app.expired == []


def run_the_wait_out(app: Wired) -> None:
    for _ in range(int(app.config.waits.assistance_s) + 1):
        app.wait.step(app.clock.monotonic())
        app.clock.advance(1.0)


def test_the_timed_return_makes_one_attempt_and_ends_powered(app):
    app.arm.set_pose(RAISED)
    app.wait.start("the agent is stuck", True)
    run_the_wait_out(app)
    assert app.expired == [(True, 0)]
    stop_number = app.expired[0][1]
    outcome = app.rest.request(
        "wait", stop_number_seen=stop_number, should_stop=app.should_stop
    )
    assert outcome.status == "at_rest"
    assert app.arm.pose() == pytest.approx(REST, abs=app.config.rest.rest_tol_rad)
    assert app.arm.torque_on() and app.arm.released_at == []
    assert app.motion.get() is MotionState.REST_HOLD
    assert app.authority.owner is Owner.NOBODY
    # The wait is over: nothing follows by itself.
    run_the_wait_out(app)
    assert len(app.expired) == 1


def test_a_stop_after_the_wait_ran_out_keeps_the_timed_return_from_moving(app):
    app.arm.set_pose(RAISED)
    app.wait.start("the agent is stuck", True)
    run_the_wait_out(app)
    stop_number = app.expired[0][1]
    app.stops.stop("operator pressed stop", "thor_terminal")
    outcome = app.rest.request(
        "wait", stop_number_seen=stop_number, should_stop=app.should_stop
    )
    assert outcome.status == "refused"
    assert outcome.reason == "stop number 0 does not match the current 1"
    assert app.arm.sent == []
    assert app.arm.pose() == RAISED
    assert app.arm.torque_on()


# Return to rest, release and the published state


def published(app: Wired, carrier_pair) -> RobotState:
    """One ROBOT_STATE as the workstation end of the link receives it."""
    robot_end, workstation_end = carrier_pair
    send(robot_end, app.publisher.build(app.clock.monotonic()))
    state = receive_message(workstation_end, WAIT_S)
    assert isinstance(state, RobotState)
    return state


def test_rest_then_release_as_the_workstation_sees_them(app, carrier_pair):
    app.arm.set_pose(RAISED)
    assert app.rest.request("typed", True).status == "at_rest"
    state = published(app, carrier_pair)
    assert (state.motion, state.torque_on, state.owner) == ("rest_hold", True, "nobody")
    assert state.gripper.load == "empty"
    assert [sample.valid for sample in state.joints] == [True] * 6

    assert app.rest.release(app.authority.stop_number) is None
    state = published(app, carrier_pair)
    assert (state.motion, state.torque_on) == ("rest_unpowered", False)
    assert len(app.arm.released_at) == 1
    # Motors off is the end of the run: nothing reads, moves or returns any more.
    app.arm.fail_reads(0, 1000)
    for _ in range(50):
        assert app.idle.step(app.clock.monotonic()) is None
        app.clock.advance(0.02)
    assert app.idle.consecutive_failures_s(app.clock.monotonic()) == 0.0
    app.new_task()
    app.gate.update(app.permit(len(PATH) - 1))
    assert app.agent_move().reason == "the motors are off"
    assert app.rest.request("typed", True).reason == "the motors are off"
    assert app.arm.sent[-1][0] == REST  # the last setpoint ever sent


def test_a_jaw_command_under_way_keeps_the_arm_from_returning_or_releasing(app):
    app.new_task()
    jaws_waiting, _ = pause_jaws_at(app.arm, 0.8)
    done = Done()
    app.worker.submit(0.5, 0.5, "plain", Owner.AGENT, done)
    assert jaws_waiting.wait(WAIT_S)
    assert app.rest.request("typed", True).reason == "a jaw command is running"
    assert app.rest.release(app.authority.stop_number) == "a jaw command is running"
    assert app.authority.owner is Owner.AGENT  # nothing changed

    app.stops.stop("operator pressed stop", "console")
    assert done.wait().status == "cancelled"
    assert app.worker.wait_idle(WAIT_S)
    # Stopped part-way through a close, the jaws are not proven empty.
    refused = app.rest.request("typed", True)
    assert refused.status == "refused"
    assert "the jaws are not proven empty (loaded)" in refused.reason

    opened = Done()
    assert app.worker.open_guarded(opened, Owner.NOBODY) is None
    assert opened.wait().status == "reached"
    assert app.worker.wait_idle(WAIT_S)
    assert app.rest.request("typed", True).status == "at_rest"
    assert app.rest.release(app.authority.stop_number) is None
    assert app.arm.released_at == [REST]


# Faults


def test_a_motor_fault_latches_until_a_reset_that_starts_nothing(app):
    app.new_task()
    app.gate.update(app.permit(len(PATH) - 1))
    app.wait.start("the agent is stuck", True)
    app.arm.set_fault("motor 3 overheated")
    number = app.stops.fault(app.arm.faults()[0])

    assert app.authority.stop_level is StopLevel.FAULT
    assert not app.wait.active()
    assert all(checker.latched() for checker in app.checkers.all())
    latched = "fault-stop latched: motor 3 overheated"
    assert app.agent_move().status == "stopped"
    assert app.rest.request("typed", True).reason == latched
    assert app.rest.release(number) == latched
    assert app.worker.submit(1.0, 0.5, "plain", Owner.AGENT) == latched
    assert app.gate.update(app.permit(24)) == f"permit: {latched}"

    assert "still reports a fault" in app.stops.reset(number, app.arm)
    app.arm.clear_faults()
    assert "stop number" in app.stops.reset(number - 1, app.arm)
    assert app.stops.reset(number, app.arm) is None
    # The reset cleared the latch and started nothing.
    assert app.authority.stop_level is StopLevel.NONE
    assert app.authority.owner is Owner.NOBODY
    assert not app.wait.active()
    assert app.arm.sent == [] and app.arm.jaw_commands == []
    # A new task works again.
    app.new_task()
    app.gate.update(app.permit(len(PATH) - 1))
    assert app.agent_move().status == "completed"


def test_lost_joint_feedback_in_a_hold_becomes_a_fault_stop(app):
    app.arm.fail_reads(2, 10_000)
    raised: list[float] = []
    start = app.clock.monotonic()
    for _ in range(100):
        now = app.clock.monotonic()
        app.idle.step(now)
        lost = app.idle.consecutive_failures_s(app.clock.monotonic())
        if lost > LOST_FEEDBACK_S and app.authority.stop_level is not StopLevel.FAULT:
            app.stops.fault(f"joint feedback lost for {lost:.2f} s in hold")
            raised.append(app.clock.monotonic() - start)
        app.clock.advance(0.01)
    assert len(raised) == 1  # raised once, not on every step
    assert LOST_FEEDBACK_S < raised[0] < LOST_FEEDBACK_S + 0.15
    assert app.authority.stop_level is StopLevel.FAULT
    assert app.arm.torque_on() and app.arm.released_at == []


def test_lost_joint_feedback_during_a_move_is_reported_as_a_fault(app):
    app.new_task()
    app.gate.update(app.permit(len(PATH) - 1))
    real = app.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(app.arm.sent) == 1:
            app.arm.fail_reads(1, 10_000)

    app.arm.send_target = send_target
    outcome = app.agent_move()
    assert outcome.status == "failed"
    assert outcome.reason.startswith("fault: joint feedback lost")
    # The rule app.py applies to every move outcome.
    app.stops.fault(outcome.reason.removeprefix("fault: "))
    assert app.authority.stop_level is StopLevel.FAULT
    assert app.arm.torque_on()


# Rules that hold across the whole package


def method_calls(name: str) -> list[str]:
    """The package file of every call of a method called `name`; nothing is imported."""
    package = Path(agentic_remote.__file__).resolve().parent
    files: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        files += [
            path.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]
    return files


def test_torque_is_released_at_one_place_in_the_package():
    assert method_calls("release_torque") == ["rest.py"]


def test_only_the_app_connects_the_arm_and_only_once():
    # apai_backend.py connects its own backend inside ApaiArm.connect().
    callers = method_calls("connect")
    assert set(callers) <= {"apai_backend.py", "app.py"}
    assert callers.count("apai_backend.py") == 1
    assert callers.count("app.py") <= 1


@pytest.fixture
def guard_calls(monkeypatch) -> Iterator[list[SimArm]]:
    """The arms the release guard was run on; it must be one when the test is over."""
    import conftest

    calls: list[SimArm] = []
    monkeypatch.setattr(
        conftest, "check_released_at_rest", lambda arm, config: calls.append(arm)
    )
    yield calls
    assert len(calls) == 1, "the sim_arm fixture did not run the release guard"


def test_the_sim_arm_fixture_runs_the_guard_once_the_test_is_over(guard_calls, sim_arm):
    # guard_calls comes first, so it is checked after sim_arm has been torn down.
    assert guard_calls == []


def test_the_suite_wide_guard_fails_a_release_away_from_rest(clock, config):
    from conftest import check_released_at_rest

    arm = SimArm(clock=clock)  # not the fixture: its guard would fail this test
    arm.connect()
    arm.release_torque()
    check_released_at_rest(arm, config)
    arm.set_pose([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    arm.release_torque()
    with pytest.raises(AssertionError, match=r"released 0\.300 rad away from rest"):
        check_released_at_rest(arm, config)
