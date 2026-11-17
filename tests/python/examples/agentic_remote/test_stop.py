# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for StopController in isaaccapture_examples.agentic_remote.robot_app.stop."""

from __future__ import annotations

import ast
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import Event, decode
from isaaccapture_examples.agentic_remote.robot_app import stop as stop_module
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    GripperWorker,
    JawState,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    GripperOutcome,
    MoveOutcome,
    Owner,
    Refusal,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from isaaccapture_examples.agentic_remote.robot_app.stop import (
    AGENT_SENDER,
    FAULT_EVENT,
    PERSON_SENDERS,
    RESET_EVENT,
    STOP_EVENT,
    StopController,
)
from isaaccapture_examples.agentic_remote.robot_app.waits import AssistanceWait

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
DROPPED = GripperOutcome("cancelled", None, "dropped by a stop")


class AppSide:
    """The two callables the app gives the controller, and the wait's expiry."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self.motion_stops: list[str] = []
        self.expired: list[bool] = []
        self.order: list[str] = []  # the controller's steps, in call order

    def emit_event(self, kind: str, text: str, data: dict) -> None:
        self.order.append("event")
        self.events.append((kind, text, data))

    def request_motion_stop(self, reason: str) -> None:
        self.order.append("motion stop")
        self.motion_stops.append(reason)

    def should_stop(self) -> str | None:
        """The stop condition a running move asks, as the app would wire it."""
        return self.motion_stops[-1] if self.motion_stops else None

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.events]


class Done:
    """An on_done callback that keeps what it was given and can be waited for."""

    def __init__(self) -> None:
        self.outcomes: list[GripperOutcome] = []
        self._called = threading.Event()

    def __call__(self, outcome: GripperOutcome) -> None:
        self.outcomes.append(outcome)
        self._called.set()

    def wait(self) -> GripperOutcome:
        assert self._called.wait(WAIT_S), "on_done was not called"
        return self.outcomes[0]


class JawPause:
    """Holds the simulated jaws inside a command until that command is cancelled.

    It hooks sim.on_jaw_step, which runs on the thread that commands the jaws.
    """

    def __init__(self, arm: SimArm, at_or_below: float) -> None:
        self.reached = threading.Event()
        self.opening: float | None = None
        self._at_or_below = at_or_below
        self._cancels: list[threading.Event] = []
        inner = arm.gripper_command

        def gripper_command(opening, max_effort, mode, cancel):
            self._cancels.append(cancel)
            return inner(opening, max_effort, mode, cancel)

        arm.gripper_command = gripper_command
        arm.on_jaw_step = self._step

    def _step(self, opening: float) -> None:
        if self.reached.is_set() or opening > self._at_or_below:
            return
        self.opening = opening
        self.reached.set()
        self._cancels[-1].wait(WAIT_S)

    def wait(self) -> float:
        assert self.reached.wait(WAIT_S), "the jaws did not get there"
        return self.opening


class OnlyFaults:
    """An arm stand-in that answers faults() and fails on any other use."""

    def __init__(self, faults: tuple[str, ...] = ()) -> None:
        self._faults = list(faults)
        self.reads = 0

    def faults(self) -> list[str]:
        self.reads += 1
        return list(self._faults)

    def __getattr__(self, name: str):
        raise AssertionError(f"reset used arm.{name}")


def log_calls(order: list[str], obj: object, method: str, label: str) -> None:
    """Note `label` in order at every call of obj.method, then run it."""
    inner = getattr(obj, method)

    def spy(*args, **kwargs):
        order.append(label)
        return inner(*args, **kwargs)

    setattr(obj, method, spy)


def log_controller_steps(app, authority, checkers, worker, wait) -> None:
    log_calls(app.order, authority, "raise_stop", "raise_stop")
    log_calls(app.order, authority, "raise_fault", "raise_fault")
    log_calls(app.order, authority, "reset", "authority reset")
    for name, checker in zip(("agent", "human", "recovery"), checkers.all()):
        log_calls(app.order, checker, "halt", f"halt {name}")
        log_calls(app.order, checker, "latch", f"latch {name}")
        log_calls(app.order, checker, "reset_latch", f"unlatch {name}")
    log_calls(app.order, wait, "cancel", "wait cancel")
    log_calls(app.order, worker, "stop_now", "jaw stop")


def broken(*args, **kwargs):
    raise RuntimeError("it broke")


def run_wait_out(wait: AssistanceWait, clock) -> list[tuple[str, str]]:
    """Step the wait in 1 s steps well past its length; the events it gave."""
    events: list[tuple[str, str]] = []
    for _ in range(200):
        clock.advance(1.0)
        events += wait.step(clock.monotonic())
    return events


def start_close(arm: SimArm, worker: GripperWorker) -> tuple[float, Done]:
    """Start a close and hold it part-way; where the jaws are, and its on_done."""
    pause = JawPause(arm, at_or_below=0.605)
    closing = Done()
    assert worker.submit(0.0, 0.8, "close_until_contact", Owner.AGENT, closing) is None
    return pause.wait(), closing


def after_setpoint(arm: SimArm, count: int, action: Callable[[], object]) -> None:
    """Run action on the moving thread right after setpoint number `count` went out."""
    inner = arm.send_target

    def send_target(q):
        inner(q)
        if len(arm.sent) == count:
            action()

    arm.send_target = send_target


def run_agent_move(arm, authority, checkers, config, clock, should_stop=None):
    """Run a 20-setpoint agent move from rest; its outcome and its setpoints."""
    checkers.agent.feed_watchdog()
    executor = MoveExecutor(arm, authority, None, config.executor, clock)
    setpoints = [[0.004 * (i + 1)] * arm.n_joints for i in range(20)]
    extra = {} if should_stop is None else {"should_stop": should_stop}
    outcome = executor.run(
        setpoints,
        [0.0] * arm.n_joints,
        0.01,
        config.executor.period_s,
        Owner.AGENT,
        checkers.agent,
        authority.snapshot(),
        "move-1",
        **extra,
    )
    return outcome, setpoints


def assert_can_be_sent(event: tuple[str, str, dict]) -> None:
    kind, text, data = event
    sent = Event(seq=1, kind=kind, text=text, data=data)
    assert decode(Event.TYPE, sent.to_payload()) == sent


@pytest.fixture
def arm(sim_arm) -> Iterator[SimArm]:
    """The simulated arm, connected, its jaws referenced and fully open."""
    sim_arm.connect()
    sim_arm.gripper_reference()
    yield sim_arm
    # No stop, fault-stop or reset may ever cut torque.
    assert sim_arm.released_at == []
    assert sim_arm.torque_on() is True


@pytest.fixture
def authority() -> Authority:
    return Authority()


@pytest.fixture
def agent_owns(authority) -> None:
    authority.set_owner(Owner.AGENT, "new task")


@pytest.fixture
def jaw_state(config, clock) -> JawState:
    return JawState(config.gripper, clock)


@pytest.fixture
def worker(arm, jaw_state, authority, clock) -> Iterator[GripperWorker]:
    worker = GripperWorker(arm, jaw_state, authority, clock)
    yield worker
    worker.close()


@pytest.fixture
def app() -> AppSide:
    return AppSide()


@pytest.fixture
def wait(config, clock, app) -> AssistanceWait:
    return AssistanceWait(config.waits, clock, app.expired.append)


@pytest.fixture
def controller(authority, checkers, worker, wait, app) -> StopController:
    return StopController(
        authority, checkers, worker, wait, app.emit_event, app.request_motion_stop
    )


def test_the_senders():
    assert AGENT_SENDER == "agent"
    assert PERSON_SENDERS == ("console", "thor_terminal", "controller", "unknown")
    assert AGENT_SENDER not in PERSON_SENDERS


# STOP


@pytest.mark.parametrize("sender", [*PERSON_SENDERS, AGENT_SENDER])
def test_stop_raises_the_stop_number_and_ends_the_ownership(
    sender, controller, authority, agent_owns
):
    before = authority.snapshot()
    number = controller.stop("operator pressed stop", sender)
    after = authority.snapshot()
    assert number == after.stop_number == before.stop_number + 1
    assert after.owner is Owner.NOBODY
    assert after.owner_number == before.owner_number + 1
    assert after.stop_level is StopLevel.STOP
    assert after.session_id == before.session_id


def test_every_stop_raises_the_number_again(controller, authority):
    assert [controller.stop("again", "console") for _ in range(3)] == [1, 2, 3]
    assert authority.stop_number == 3


def test_stop_halts_every_checker(controller, checkers, arm):
    counts = [checker.halt_count() for checker in checkers.all()]
    rest, step = [0.0] * arm.n_joints, [0.001] * arm.n_joints
    controller.stop("operator pressed stop", "console")
    for checker, count in zip(checkers.all(), counts):
        assert checker.halt_count() == count + 1
        with pytest.raises(Refusal, match=r"^halted: operator pressed stop$"):
            checker.check_setpoint(rest, step, 0.02)
        with pytest.raises(Refusal, match=r"halt count is 1, the motion quotes 0"):
            checker.begin_motion(count)
        assert checker.latched() is False


def test_stop_asks_the_running_move_to_stop(controller, app):
    controller.stop("operator pressed stop", "thor_terminal")
    assert app.motion_stops == ["STOP from thor_terminal: operator pressed stop"]


def test_a_command_admitted_before_the_stop_is_refused_afterwards(
    controller, authority, agent_owns
):
    quoted = authority.snapshot()
    session = quoted.session_id

    def admit(owner_number, stop_number, required):
        return authority.admit(session, owner_number, stop_number, required)

    assert admit(quoted.owner_number, quoted.stop_number, Owner.AGENT) is None
    assert admit(None, quoted.stop_number, None) is None

    number = controller.stop("operator pressed stop", "console")

    stale = f"stop number {quoted.stop_number} does not match the current {number}"
    assert admit(quoted.owner_number, quoted.stop_number, Owner.AGENT) == stale
    # A command that carries no owner number is refused as well.
    assert admit(None, quoted.stop_number, None) == stale
    # Quoting the new stop number does not help the agent: it lost the arm.
    assert (
        admit(quoted.owner_number, number, Owner.AGENT)
        == "control owner is nobody, not agent"
    )


def test_stop_ends_a_jaw_close_under_way_at_the_light_hold(
    controller, arm, worker, jaw_state, agent_owns
):
    where, closing = start_close(arm, worker)
    queued = Done()
    assert worker.submit(1.0, 0.5, "plain", Owner.AGENT, queued) is None
    assert where == pytest.approx(0.6)
    assert arm.jaw_stiffness == "closing"
    assert arm.light_holds == []

    controller.stop("operator pressed stop", "console")

    assert closing.wait() == GripperOutcome("cancelled", where)
    assert queued.outcomes == [DROPPED]
    assert arm.gripper_read() == where
    assert arm.jaw_stiffness == "light"
    assert len(arm.light_holds) == 2  # at once, and again after the close returned
    # Only the close ever reached the arm: the jaws were not opened.
    assert [command.opening for command in arm.jaw_commands] == [0.0]
    assert arm.jaw_commands[0].status == "cancelled"
    assert jaw_state.may_be_holding is True
    assert worker.wait_idle(WAIT_S)
    # The agent's next jaw command is refused: the STOP took the arm from it.
    refusal = worker.submit(1.0, 0.5, "plain", Owner.AGENT)
    assert refusal == "control owner is nobody, not agent"
    assert [command.opening for command in arm.jaw_commands] == [0.0]


def test_stop_never_opens_jaws_that_hold_an_object(controller, arm, worker, agent_owns):
    arm.jaw_object(0.4)
    done = Done()
    assert worker.submit(0.0, 0.5, "close_until_contact", Owner.AGENT, done) is None
    held = done.wait().opening
    controller.stop("operator pressed stop", "console")
    controller.stop("and again", "controller")
    assert arm.gripper_read() == held == pytest.approx(0.4)
    assert arm.gripper_last_target() == 0.0
    assert arm.jaw_stiffness == "light"
    assert [command.opening for command in arm.jaw_commands] == [0.0]


@pytest.mark.parametrize("sender", PERSON_SENDERS)
def test_a_stop_from_a_person_cancels_the_assistance_wait(
    sender, controller, wait, clock, app
):
    wait.start("the agent is stuck")
    controller.stop("hold it there", sender)
    assert wait.active() is False
    assert run_wait_out(wait, clock) == []
    assert app.expired == []
    _, text, data = app.events[0]
    assert data["wait_cancelled"] is True
    assert "the assistance wait is cancelled" in text


@pytest.mark.parametrize("sender", [AGENT_SENDER, " agent "])
def test_a_stop_from_the_agent_leaves_the_assistance_wait_running(
    sender, controller, wait, clock, app
):
    wait.start("the agent is stuck")
    left = wait.remaining(clock.monotonic())
    controller.stop("giving up", sender)
    assert wait.active() is True
    assert wait.remaining(clock.monotonic()) == left
    _, text, data = app.events[0]
    assert data["wait_cancelled"] is False
    assert "assistance wait" not in text
    # The wait still runs out, once, and asks for the timed return.
    assert run_wait_out(wait, clock)[-1][0] == "expired"
    assert app.expired == [True]


@pytest.mark.parametrize("sender", ["", "  ", "Agent", "the agent", "service", None])
def test_any_sender_but_the_agent_counts_as_a_person(sender, controller, wait, app):
    wait.start("the agent is stuck")
    controller.stop("hold it there", sender)
    assert wait.active() is False
    assert app.events[0][2]["wait_cancelled"] is True


def test_a_stop_with_no_wait_running_reports_no_cancel(controller, wait, app):
    controller.stop("hold it there", "console")
    assert wait.active() is False
    assert app.events[0][2]["wait_cancelled"] is False
    assert "assistance wait" not in app.events[0][1]


def test_stop_emits_one_event_for_the_operator(controller, app, agent_owns):
    number = controller.stop("operator pressed stop", "console")
    assert len(app.events) == 1
    kind, text, data = app.events[0]
    assert kind == STOP_EVENT == "stop"
    assert text == (
        f"STOP from console: operator pressed stop; the arm holds, stop number {number}"
    )
    assert data == {
        "reason": "operator pressed stop",
        "sender": "console",
        "stop_number": number,
        "stop_level": "stop",
        "wait_cancelled": False,
        "problems": [],
    }
    assert_can_be_sent(app.events[0])


@pytest.mark.parametrize(
    ("reason", "sender", "start"),
    [
        ("", "console", "STOP from console: no reason given;"),
        ("   ", "agent", "STOP from agent: no reason given;"),
        ("STOP with a bad payload", "", "STOP from unknown: STOP with a bad payload;"),
        ("  key pressed ", " thor_terminal ", "STOP from thor_terminal: key pressed;"),
    ],
)
def test_a_missing_reason_or_sender_is_still_worded(
    reason, sender, start, controller, app, checkers
):
    controller.stop(reason, sender)
    assert app.events[0][1].startswith(start)
    assert app.motion_stops == [start.removesuffix(";")]
    halted = start.split(": ", 1)[1].removesuffix(";")
    with pytest.raises(Refusal, match=rf"^halted: {halted}$"):
        checkers.human.check_setpoint([0.0] * 6, [0.001] * 6, 0.02)


def test_the_text_of_a_stop_is_made_printable_before_anything_shows_it(
    controller, app, authority, checkers, caplog
):
    # What a terminal would act on: clear the screen, then a false line.
    caplog.set_level("DEBUG")
    controller.stop("\x1b[2J\r\nARM IS AT REST\n", "console\x07")
    _, text, data = app.events[0]
    assert data["reason"] == "[2J  ARM IS AT REST"
    assert data["sender"] == "console"
    assert text.startswith("STOP from console: [2J  ARM IS AT REST; the arm holds")
    assert app.motion_stops == ["STOP from console: [2J  ARM IS AT REST"]
    with pytest.raises(Refusal, match=r"^halted: \[2J  ARM IS AT REST$"):
        checkers.agent.check_setpoint([0.0] * 6, [0.001] * 6, 0.02)
    logged = [record.getMessage() for record in caplog.records]
    assert any("ARM IS AT REST" in line for line in logged)
    assert all(line.isprintable() for line in logged)


def test_a_long_stop_reason_and_sender_are_cut(controller, app):
    controller.stop("r" * 5000, "s" * 5000)
    _, text, data = app.events[0]
    assert data["reason"] == "r" * 253 + "..."
    assert data["sender"] == "s" * 61 + "..."
    assert len(text) < 400


@pytest.mark.parametrize(
    "sender",
    ["agent\x07", "\x00agent", "agent" + chr(0x200B), "agent" + " " * 100 + "x"],
    ids=["a bell", "a NUL", "a zero-width space", "a long tail"],
)
def test_a_sender_that_only_reads_as_the_agent_once_shown_counts_as_a_person(
    sender, controller, wait, app
):
    wait.start("the agent is stuck")
    controller.stop("giving up", sender)
    # Only the word itself leaves the wait running; this one cancels it.
    assert wait.active() is False
    _, text, data = app.events[0]
    assert data["wait_cancelled"] is True
    assert data["sender"] != "agent"
    assert text.isprintable()


def test_the_steps_of_a_stop_run_in_this_order(
    controller, app, authority, checkers, worker, wait
):
    wait.start("the agent is stuck")
    log_controller_steps(app, authority, checkers, worker, wait)
    controller.stop("operator pressed stop", "console")
    assert app.order == [
        "raise_stop",
        "halt agent",
        "halt human",
        "halt recovery",
        "motion stop",
        "wait cancel",
        "jaw stop",
        "event",
    ]


def test_a_stop_from_the_agent_does_not_touch_the_wait(
    controller, app, authority, checkers, worker, wait
):
    wait.start("the agent is stuck")
    log_controller_steps(app, authority, checkers, worker, wait)
    controller.stop("giving up", AGENT_SENDER)
    assert "wait cancel" not in app.order
    assert app.order[-3:] == ["motion stop", "jaw stop", "event"]


@pytest.mark.parametrize("wired", [True, False])
def test_stop_during_a_move_sends_no_further_setpoint(
    wired, controller, arm, authority, checkers, config, clock, app, agent_owns
):
    after_setpoint(arm, 5, lambda: controller.stop("operator pressed stop", "console"))
    should_stop = app.should_stop if wired else None
    outcome, setpoints = run_agent_move(
        arm, authority, checkers, config, clock, should_stop
    )
    assert outcome.status == "stopped"
    if wired:
        assert outcome.reason == "STOP from console: operator pressed stop"
    else:
        # Without the app's stop condition the stop number alone ends the move.
        assert outcome.reason == "stop number is now 1, the move was admitted at 0"
    assert outcome.index == 5
    assert [q for q, _ in arm.sent] == setpoints[:5]
    # The last setpoint stands and the motors stay on: a hold.
    assert arm.last_target()[0] == setpoints[4]
    assert arm.holds == []
    assert checkers.agent.in_motion is False
    # No jaw command ran: the jaws are sent nothing and keep their grip.
    assert arm.light_holds == [] and arm.jaw_commands == []


def test_a_move_does_not_start_after_a_stop(
    controller, arm, authority, checkers, config, clock, agent_owns
):
    admitted = authority.snapshot()
    controller.stop("operator pressed stop", "console")
    checkers.agent.feed_watchdog()
    executor = MoveExecutor(arm, authority, None, config.executor, clock)
    outcome = executor.run(
        [[0.004] * arm.n_joints],
        [0.0] * arm.n_joints,
        0.01,
        config.executor.period_s,
        Owner.AGENT,
        checkers.agent,
        admitted,
        "move-1",
    )
    assert outcome == MoveOutcome(
        "stopped", "stop number is now 1, the move was admitted at 0"
    )
    assert arm.sent == []


def test_a_stop_step_that_fails_does_not_keep_the_others_from_running(
    arm, authority, checkers, worker, wait, app, agent_owns
):
    def request_motion_stop(reason: str) -> None:
        app.request_motion_stop(reason)
        raise RuntimeError("the motion worker is gone")

    checkers.agent.halt = broken
    controller = StopController(
        authority, checkers, worker, wait, app.emit_event, request_motion_stop
    )
    wait.start("the agent is stuck")
    _, closing = start_close(arm, worker)  # a jaw command for the jaw stop to end

    assert controller.stop("operator pressed stop", "console") == 1

    assert authority.stop_number == 1
    assert [checker.halt_count() for checker in checkers.all()] == [0, 1, 1]
    assert len(app.motion_stops) == 1
    assert wait.active() is False
    # The jaw stop ran too: the close ended in the light hold.
    assert closing.wait().status == "cancelled"
    assert len(arm.light_holds) == 2 and arm.jaw_stiffness == "light"
    _, text, data = app.events[0]
    assert data["problems"] == [
        "the agent checker's halt failed: RuntimeError('it broke')",
        "the motion stop request failed: RuntimeError('the motion worker is gone')",
    ]
    assert text.endswith("; steps that failed: " + "; ".join(data["problems"]))
    assert_can_be_sent(app.events[0])


def test_a_failing_wait_or_jaw_stop_is_reported_too(
    authority, checkers, worker, wait, app
):
    wait.cancel = broken
    worker.stop_now = broken
    controller = StopController(
        authority, checkers, worker, wait, app.emit_event, app.request_motion_stop
    )
    assert controller.stop("operator pressed stop", "console") == 1
    assert [checker.halt_count() for checker in checkers.all()] == [1, 1, 1]
    data = app.events[0][2]
    assert data["wait_cancelled"] is False
    assert data["problems"] == [
        "the wait's cancel failed: RuntimeError('it broke')",
        "the jaw stop failed: RuntimeError('it broke')",
    ]


def test_an_event_that_cannot_be_emitted_does_not_undo_the_stop(
    arm, authority, checkers, worker, wait, app
):
    controller = StopController(
        authority, checkers, worker, wait, broken, app.request_motion_stop
    )
    assert controller.stop("operator pressed stop", "console") == 1
    assert controller.fault("motor 2 reports a fault") == 2
    assert authority.stop_level is StopLevel.FAULT
    assert all(checker.latched() for checker in checkers.all())
    assert arm.light_holds == []  # no jaw command ran: the jaws are sent nothing
    assert controller.reset(2, arm) is None
    assert authority.stop_level is StopLevel.NONE


def test_stops_from_several_threads_each_get_their_own_number(
    controller, authority, checkers, app
):
    count = 8
    barrier = threading.Barrier(count)
    numbers: list[int] = []

    def press(index: int) -> None:
        barrier.wait(WAIT_S)
        numbers.append(controller.stop(f"stop {index}", "console"))

    threads = [threading.Thread(target=press, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S)
    assert sorted(numbers) == list(range(1, count + 1))
    assert authority.stop_number == count
    assert [checker.halt_count() for checker in checkers.all()] == [count] * 3
    assert sorted(data["stop_number"] for _, _, data in app.events) == sorted(numbers)


# Fault-stop


def test_fault_latches_the_authority_and_every_checker(
    controller, authority, checkers, arm, agent_owns
):
    before = authority.snapshot()
    rest, step = [0.0] * arm.n_joints, [0.001] * arm.n_joints

    number = controller.fault("motor 2 reports a fault")

    after = authority.snapshot()
    assert number == after.stop_number == before.stop_number + 1
    assert after.stop_level is StopLevel.FAULT
    assert after.fault_reason == "motor 2 reports a fault"
    assert after.owner is Owner.NOBODY
    assert after.owner_number == before.owner_number + 1
    latched = r"^fault-stop latched: motor 2 reports a fault$"
    for checker in checkers.all():
        assert checker.latched() is True
        with pytest.raises(Refusal, match=latched):
            checker.check_setpoint(rest, step, 0.02)
        with pytest.raises(Refusal, match=latched):
            checker.check_pose(rest)
        with pytest.raises(Refusal, match=latched):
            checker.begin_motion(checker.halt_count())


def test_nothing_is_admitted_under_a_fault_stop(controller, authority, worker):
    number = controller.fault("motor 2 reports a fault")
    owner_number = authority.set_owner(Owner.AGENT, "a request that raced the fault")
    session = authority.session_id
    latched = "fault-stop latched: motor 2 reports a fault"
    # Current numbers and the right owner do not help.
    assert authority.admit(session, owner_number, number, Owner.AGENT) == latched
    assert authority.admit(session, None, number, None) == latched
    assert worker.submit(1.0, 0.5, "plain", Owner.AGENT) == latched
    assert worker.open_guarded() == latched


@pytest.mark.parametrize("timed_return", [True, False])
def test_fault_cancels_the_assistance_wait(timed_return, controller, wait, clock, app):
    wait.start("the agent is stuck", timed_return)
    controller.fault("joint feedback lost for 0.36 s")
    assert wait.active() is False
    assert run_wait_out(wait, clock) == []
    assert app.expired == []
    assert app.events[0][2]["wait_cancelled"] is True


def test_fault_asks_the_running_move_to_stop(controller, app):
    controller.fault("motor 2 reports a fault")
    assert app.motion_stops == ["fault-stop: motor 2 reports a fault"]


def test_fault_ends_a_jaw_close_under_way_at_the_light_hold(
    controller, arm, worker, agent_owns
):
    where, closing = start_close(arm, worker)

    controller.fault("motor 2 reports a fault")

    assert closing.wait() == GripperOutcome("cancelled", where)
    assert arm.gripper_read() == where
    assert arm.jaw_stiffness == "light"
    assert len(arm.light_holds) == 2  # at once, and again after the close returned
    assert [command.opening for command in arm.jaw_commands] == [0.0]


def test_fault_emits_one_event_for_the_operator(controller, app):
    number = controller.fault("motor 2 reports a fault")
    assert len(app.events) == 1
    kind, text, data = app.events[0]
    assert kind == FAULT_EVENT == "fault"
    assert text == (
        "FAULT-STOP: motor 2 reports a fault; the arm holds until a person types "
        f"reset, stop number {number}"
    )
    assert data == {
        "reason": "motor 2 reports a fault",
        "fault_reason": "motor 2 reports a fault",
        "stop_number": number,
        "stop_level": "fault",
        "wait_cancelled": False,
        "problems": [],
    }
    assert_can_be_sent(app.events[0])


@pytest.mark.parametrize(
    ("reason", "worded"),
    [
        ("", "no reason given"),
        ("  motor 2 reports a fault \n", "motor 2 reports a fault"),
    ],
)
def test_a_fault_reason_is_always_worded(reason, worded, controller, authority, app):
    controller.fault(reason)
    assert authority.fault_reason == worded
    assert app.motion_stops == [f"fault-stop: {worded}"]
    assert app.events[0][1].startswith(f"FAULT-STOP: {worded};")


def test_a_fault_reason_is_printable_and_at_most_1024_characters(
    controller, authority, app
):
    controller.fault("driver said:\n" + "y" * 5000)
    assert authority.fault_reason.startswith("driver said: yyyy")
    assert len(authority.fault_reason) == 1024
    assert authority.fault_reason.isprintable()
    assert_can_be_sent(app.events[0])


def test_the_steps_of_a_fault_stop_run_in_this_order(
    controller, app, authority, checkers, worker, wait
):
    wait.start("the agent is stuck")
    log_controller_steps(app, authority, checkers, worker, wait)
    controller.fault("motor 2 reports a fault")
    assert app.order == [
        "raise_fault",
        "latch agent",
        "latch human",
        "latch recovery",
        "motion stop",
        "wait cancel",
        "jaw stop",
        "event",
    ]


def test_fault_during_a_move_sends_no_further_setpoint(
    controller, arm, authority, checkers, config, clock, app, agent_owns
):
    after_setpoint(arm, 5, lambda: controller.fault("motor 2 reports a fault"))
    outcome, setpoints = run_agent_move(
        arm, authority, checkers, config, clock, app.should_stop
    )
    assert outcome.status == "stopped"
    assert outcome.reason == "fault-stop: motor 2 reports a fault"
    assert outcome.index == 5
    assert [q for q, _ in arm.sent] == setpoints[:5]
    assert arm.last_target()[0] == setpoints[4]
    assert authority.stop_level is StopLevel.FAULT


def test_a_second_fault_keeps_the_first_reason_and_raises_the_number(
    controller, authority, checkers, app
):
    assert controller.fault("motor 2 reports a fault") == 1
    assert controller.fault("joint feedback lost for 0.36 s") == 2
    assert authority.stop_level is StopLevel.FAULT
    assert authority.fault_reason == "motor 2 reports a fault"
    assert all(checker.latched() for checker in checkers.all())
    kind, text, data = app.events[1]
    assert kind == FAULT_EVENT
    assert text.startswith("FAULT-STOP: joint feedback lost for 0.36 s;")
    assert text.endswith("; latched earlier: motor 2 reports a fault")
    assert data["reason"] == "joint feedback lost for 0.36 s"
    assert data["fault_reason"] == "motor 2 reports a fault"


def test_a_stop_while_a_fault_is_latched_keeps_the_fault_level(
    controller, authority, checkers, wait, app
):
    controller.fault("motor 2 reports a fault")
    counts = [checker.halt_count() for checker in checkers.all()]

    number = controller.stop("operator pressed stop", "console")

    state = authority.snapshot()
    assert number == state.stop_number == 2
    assert state.stop_level is StopLevel.FAULT
    assert state.fault_reason == "motor 2 reports a fault"
    assert all(checker.latched() for checker in checkers.all())
    assert [c.halt_count() for c in checkers.all()] == [n + 1 for n in counts]
    kind, text, data = app.events[1]
    assert kind == STOP_EVENT
    assert data["stop_level"] == "fault"
    assert text.endswith("; the fault-stop stays latched: motor 2 reports a fault")


def test_a_fault_step_that_fails_does_not_keep_the_others_from_running(
    controller, arm, authority, checkers, worker, wait, app, agent_owns
):
    checkers.human.latch = broken
    wait.start("the agent is stuck")
    _, closing = start_close(arm, worker)  # a jaw command for the jaw stop to end
    assert controller.fault("motor 2 reports a fault") == 1
    assert authority.stop_level is StopLevel.FAULT
    assert [checker.latched() for checker in checkers.all()] == [True, False, True]
    assert len(app.motion_stops) == 1
    assert wait.active() is False
    # The jaw stop ran too: the close ended in the light hold.
    assert closing.wait().status == "cancelled"
    assert len(arm.light_holds) == 2 and arm.jaw_stiffness == "light"
    assert app.events[0][2]["problems"] == [
        "the human checker's latch failed: RuntimeError('it broke')"
    ]


def test_a_jaw_fault_the_gripper_worker_reports_becomes_a_fault_stop(
    controller, arm, authority, checkers, worker, app, agent_owns
):
    def faulty(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor over current")

    worker.on_fault = controller.fault  # runs on the worker thread
    arm.gripper_command = faulty
    done = Done()
    assert worker.submit(0.0, 0.5, "plain", Owner.AGENT, done) is None
    assert done.wait().status == "failed"
    assert authority.stop_level is StopLevel.FAULT
    assert authority.fault_reason == "gripper fault: jaw motor over current"
    assert all(checker.latched() for checker in checkers.all())
    assert app.kinds() == [FAULT_EVENT]
    assert len(arm.light_holds) == 1
    assert worker.wait_idle(WAIT_S)


def test_a_light_hold_that_fails_during_a_stop_becomes_a_fault_stop(
    controller, arm, authority, checkers, worker, app, agent_owns
):
    def no_hold() -> None:
        raise ArmFault("no answer from the jaw motor")

    worker.on_fault = controller.fault
    arm.gripper_light_hold = no_hold
    # A STOP sends the light hold only to jaws a command is moving.
    _, closing = start_close(arm, worker)

    assert controller.stop("operator pressed stop", "console") == 1
    assert closing.wait().status == "cancelled"

    reason = "gripper fault: the light hold failed: no answer from the jaw motor"
    assert authority.stop_level is StopLevel.FAULT
    assert authority.fault_reason == reason
    assert authority.stop_number == 2
    assert all(checker.latched() for checker in checkers.all())
    # The fault-stop ran inside the STOP's jaw step, so its event came first.
    assert app.kinds() == [FAULT_EVENT, STOP_EVENT]


# Reset


def test_reset_is_refused_when_nothing_is_latched(controller, authority, arm, app):
    assert controller.reset(0, arm) == "no fault-stop is latched"
    number = controller.stop("operator pressed stop", "console")
    before = authority.snapshot()
    # A STOP is not latched: there is nothing to reset.
    assert controller.reset(number, arm) == "no fault-stop is latched"
    assert authority.snapshot() == before
    assert app.kinds() == [STOP_EVENT]


@pytest.mark.parametrize("quoted", [0, 2, -1])
def test_reset_is_refused_with_a_stop_number_that_is_not_current(
    quoted, controller, authority, checkers, arm, app
):
    assert controller.fault("motor 2 reports a fault") == 1
    before = authority.snapshot()
    refusal = controller.reset(quoted, arm)
    assert refusal == f"stop number {quoted} does not match the current 1"
    assert authority.snapshot() == before
    assert all(checker.latched() for checker in checkers.all())
    assert app.kinds() == [FAULT_EVENT]


@pytest.mark.parametrize("quoted", [0, 2, -1])
def test_reset_is_refused_with_an_owner_number_that_is_not_current(
    quoted, controller, authority, checkers, arm, app
):
    assert controller.fault("motor 2 reports a fault") == 1
    before = authority.snapshot()
    assert before.owner_number == 1
    refusal = controller.reset(1, arm, quoted)
    assert refusal == f"owner number {quoted} does not match the current 1"
    assert authority.snapshot() == before
    assert all(checker.latched() for checker in checkers.all())
    assert app.kinds() == [FAULT_EVENT]
    # The stop number is compared first, as for every other command.
    assert controller.reset(0, arm, quoted) == (
        "stop number 0 does not match the current 1"
    )
    assert controller.reset(1, arm, 1) is None
    assert authority.stop_level is StopLevel.NONE


def test_reset_needs_the_number_of_a_stop_raised_under_the_fault(
    controller, authority, arm
):
    assert controller.fault("motor 2 reports a fault") == 1
    assert controller.stop("operator pressed stop", "console") == 2
    assert controller.reset(1, arm) == "stop number 1 does not match the current 2"
    assert controller.reset(2, arm) is None
    assert authority.stop_level is StopLevel.NONE


def test_reset_is_refused_while_the_arm_reports_a_fault(
    controller, authority, checkers, arm, app
):
    arm.set_fault("motor 2: over temperature")
    arm.set_fault("motor 5: over current")
    number = controller.fault("motor 2: over temperature")
    before = authority.snapshot()

    refusal = controller.reset(number, arm)

    assert refusal == (
        "the arm still reports a fault: motor 2: over temperature; "
        "motor 5: over current"
    )
    assert authority.snapshot() == before
    assert all(checker.latched() for checker in checkers.all())
    assert app.kinds() == [FAULT_EVENT]
    # Once the motors report no fault the same request is accepted.
    arm.clear_faults()
    assert controller.reset(number, arm) is None


def test_the_latch_and_the_number_are_checked_before_the_arm_is_asked(
    controller, authority
):
    faulty_arm = OnlyFaults(("motor 2: over temperature",))
    assert controller.reset(0, faulty_arm) == "no fault-stop is latched"
    controller.fault("motor 2: over temperature")
    stale = "stop number 0 does not match the current 1"
    assert controller.reset(0, faulty_arm) == stale
    assert faulty_arm.reads == 0


def test_reset_is_refused_when_the_arm_cannot_say_whether_it_has_a_fault(
    controller, authority, checkers, arm
):
    def no_answer() -> list[str]:
        raise ArmFault("the motor bus does not answer")

    number = controller.fault("motor 2 reports a fault")
    arm.faults = no_answer
    refusal = controller.reset(number, arm)
    assert refusal == (
        "the arm's faults could not be read: ArmFault('the motor bus does not answer')"
    )
    assert authority.stop_level is StopLevel.FAULT
    assert all(checker.latched() for checker in checkers.all())


def test_reset_clears_the_latch_and_starts_nothing(
    controller, authority, checkers, arm, worker, wait, clock, app, agent_owns
):
    wait.start("the agent is stuck")
    number = controller.fault("motor 2 reports a fault")
    latched = authority.snapshot()
    arm_before = (
        list(arm.sent),
        list(arm.holds),
        list(arm.jaw_commands),
        list(arm.light_holds),
        arm.last_target(),
        arm.gripper_last_target(),
    )
    motion_stops = list(app.motion_stops)
    halt_counts = [checker.halt_count() for checker in checkers.all()]

    assert controller.reset(number, arm) is None

    state = authority.snapshot()
    assert state.stop_level is StopLevel.NONE
    assert state.fault_reason == ""
    assert state.owner is Owner.NOBODY
    assert (state.owner_number, state.stop_number) == (
        latched.owner_number,
        latched.stop_number,
    )
    assert not any(checker.latched() for checker in checkers.all())
    assert [checker.halt_count() for checker in checkers.all()] == halt_counts
    # No motion, no jaw command, no hold and no wait were started.
    assert arm_before == (
        arm.sent,
        arm.holds,
        arm.jaw_commands,
        arm.light_holds,
        arm.last_target(),
        arm.gripper_last_target(),
    )
    assert app.motion_stops == motion_stops
    assert worker.busy() is False
    assert not any(checker.in_motion for checker in checkers.all())
    assert wait.active() is False
    assert run_wait_out(wait, clock) == []
    assert app.expired == []
    assert app.events[-1] == (
        RESET_EVENT,
        "fault-stop cleared by reset; the arm holds, nothing was started",
        {"stop_number": number, "problems": []},
    )
    assert_can_be_sent(app.events[-1])


def test_reset_asks_the_arm_for_its_faults_and_nothing_else(controller):
    quiet_arm = OnlyFaults()
    number = controller.fault("motor 2 reports a fault")
    assert controller.reset(number, quiet_arm) is None
    assert quiet_arm.reads == 1


def test_the_steps_of_a_reset_run_in_this_order(
    controller, app, authority, checkers, worker, wait, arm
):
    number = controller.fault("motor 2 reports a fault")
    log_controller_steps(app, authority, checkers, worker, wait)
    log_calls(app.order, arm, "faults", "arm faults")
    app.order.clear()
    assert controller.reset(number, arm) is None
    assert app.order == [
        "arm faults",
        "unlatch agent",
        "unlatch human",
        "unlatch recovery",
        "authority reset",
        "event",
    ]


def test_a_new_task_can_be_admitted_and_checked_after_a_reset(
    controller, authority, checkers, arm, worker
):
    number = controller.fault("motor 2 reports a fault")
    assert controller.reset(number, arm) is None
    session = authority.session_id
    # The stop number the console saw is still the current one.
    assert authority.admit(session, None, number, None) is None
    owner_number = authority.set_owner(Owner.AGENT, "new task")
    assert authority.admit(session, owner_number, number, Owner.AGENT) is None
    agent = checkers.agent
    agent.feed_watchdog()
    agent.begin_motion(agent.halt_count())
    agent.check_setpoint([0.0] * arm.n_joints, [0.001] * arm.n_joints, 0.02)
    agent.end_motion()
    done = Done()
    assert worker.submit(0.5, 0.5, "plain", Owner.AGENT, done) is None
    assert done.wait().status == "reached"


def test_a_second_reset_is_refused(controller, arm):
    number = controller.fault("motor 2 reports a fault")
    assert controller.reset(number, arm) is None
    assert controller.reset(number, arm) == "no fault-stop is latched"


@pytest.mark.parametrize("raised_by", ["fault", "stop", "past the controller"])
def test_a_stop_or_fault_raised_while_a_reset_is_checked_keeps_the_latch(
    raised_by, controller, authority, checkers, arm, app
):
    def faults() -> list[str]:
        # Raised between the reset's checks and its clearing of the latch.
        if raised_by == "fault":
            controller.fault("a second fault")
        elif raised_by == "stop":
            controller.stop("operator pressed stop", "console")
        else:
            authority.raise_fault("a second fault")
        return []

    number = controller.fault("motor 2 reports a fault")
    arm.faults = faults

    refusal = controller.reset(number, arm)

    assert refusal == "stop number 1 does not match the current 2"
    assert authority.stop_level is StopLevel.FAULT
    assert all(checker.latched() for checker in checkers.all())
    with pytest.raises(Refusal, match=r"^fault-stop latched: motor 2 reports a fault$"):
        checkers.recovery.check_pose([0.0] * arm.n_joints)
    assert RESET_EVENT not in app.kinds()


def test_a_reset_made_past_the_controller_leaves_no_checker_latched(
    controller, authority, checkers, arm, app
):
    def faults() -> list[str]:
        assert authority.reset(number) is None
        return []

    number = controller.fault("motor 2 reports a fault")
    arm.faults = faults

    assert controller.reset(number, arm) == "no fault-stop is latched"
    assert authority.stop_level is StopLevel.NONE
    assert not any(checker.latched() for checker in checkers.all())
    assert RESET_EVENT not in app.kinds()


def test_a_checker_whose_latch_cannot_be_cleared_refuses_the_reset(
    controller, authority, checkers, arm, app
):
    number = controller.fault("motor 2 reports a fault")
    checkers.human.reset_latch = broken

    refusal = controller.reset(number, arm)

    assert refusal == (
        "the human checker's latch reset failed: RuntimeError('it broke')"
    )
    assert authority.stop_level is StopLevel.FAULT
    assert authority.stop_number == number
    assert all(checker.latched() for checker in checkers.all())
    assert RESET_EVENT not in app.kinds()


def test_a_fault_from_another_thread_waits_for_a_running_reset(
    controller, authority, checkers, arm, app
):
    raised = threading.Event()
    seen_early: list[bool] = []

    def raise_fault() -> None:
        controller.fault("a fault from the motion worker")
        raised.set()

    other = threading.Thread(target=raise_fault)
    inner = checkers.agent.reset_latch

    def reset_latch() -> None:
        # Runs while the reset clears: the other thread's fault must not land yet.
        other.start()
        seen_early.append(raised.wait(0.005))
        inner()

    number = controller.fault("motor 2 reports a fault")
    checkers.agent.reset_latch = reset_latch

    assert controller.reset(number, arm) is None
    assert raised.wait(WAIT_S)
    other.join(WAIT_S)

    assert seen_early == [False]
    # The reset cleared the first fault; the second one then latched everything.
    assert app.kinds() == [FAULT_EVENT, RESET_EVENT, FAULT_EVENT]
    assert authority.stop_level is StopLevel.FAULT
    assert authority.fault_reason == "a fault from the motion worker"
    assert all(checker.latched() for checker in checkers.all())


def test_a_stop_from_another_thread_does_not_wait_for_a_running_reset(
    controller, authority, checkers, arm
):
    numbers: list[int] = []

    def press() -> None:
        numbers.append(controller.stop("operator pressed stop", "thor_terminal"))

    inner = checkers.agent.reset_latch

    def reset_latch() -> None:
        # Runs while the reset clears: a STOP on another thread goes through.
        other = threading.Thread(target=press)
        other.start()
        other.join(WAIT_S)
        inner()

    number = controller.fault("motor 2 reports a fault")
    checkers.agent.reset_latch = reset_latch

    refusal = controller.reset(number, arm)

    assert numbers == [2]
    # The STOP made the reset's stop number stale, so the fault stays latched.
    assert refusal == "stop number 1 does not match the current 2"
    assert authority.stop_level is StopLevel.FAULT
    assert all(checker.latched() for checker in checkers.all())


# What the module may never do


def test_the_module_names_no_call_that_moves_opens_starts_or_cuts_torque():
    source = Path(stop_module.__file__).read_text(encoding="utf-8")
    used = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }
    forbidden = {
        "release_torque",
        "connect",
        "torque_on",
        "send_target",
        "hold_measured",
        "gripper_command",
        "gripper_reference",
        "open_guarded",
        "submit",
        "set_owner",
        "start",
        "run",
    }
    assert used & forbidden == set()
    # The calls it does make on its collaborators.
    assert {"raise_stop", "raise_fault", "halt", "latch", "reset_latch"} <= used
    assert {"stop_now", "cancel", "faults"} <= used
