# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app without a robot connection, losing one and getting a new one.

On loopback links: RobotApp(carrier=None), attach_carrier(), the loss handled once
per carrier, nothing replayed and nothing restarted, and RobotDialer.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable

import pytest
from isaaccapture_examples.agentic_remote.protocol.carrier import (
    Carrier,
    LoopbackCarrier,
    loopback_pair,
    receive_message,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    Result,
    RobotState,
    Welcome,
)
from isaaccapture_examples.agentic_remote.robot_app.app import (
    LINK_CONNECTED,
    LINK_LOST,
    RobotApp,
    _Job,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (
    MIN_RETRY_S,
    ConfigError,
    ConnectionConfig,
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.connection import RobotDialer
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
    StopLevel,
)
from repo_paths import repo_root

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
DIALER_LOGGER = "isaaccapture.examples.agentic_remote.connection"
REST = [0.0] * 6
# A pose the agent may hold, well away from rest.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
# 25 setpoints at 0.2 rad/s on every joint, from rest and from RAISED.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
REACH = [[q + 0.004 * (index + 1) for q in RAISED] for index in range(25)]
# Longer than the assistance wait of the demo configuration.
LONG_S = 200.0
MADE = ("connection", LINK_CONNECTED, {"connected": True})


class Done:
    """The on_done of a request made at the Thor terminal."""

    def __init__(self) -> None:
        self.results: list[Result] = []
        self._called = threading.Event()

    def __call__(self, result: Result) -> None:
        self.results.append(result)
        self._called.set()

    def wait(self) -> Result:
        assert self._called.wait(WAIT_S), "the request did not end"
        return self.results[0]


def final(h, command_id: str) -> Result:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


def torque_kept(h) -> None:
    assert h.arm.torque_on(), "the motors are off"
    assert h.arm.released_at == [], "torque was released"


def nothing() -> None:
    return None


# No robot connection


def test_the_app_starts_and_holds_with_no_robot_connection(make_robot_app):
    h = make_robot_app(connected=False)
    assert h.app.running
    assert not h.app.link_alive
    assert h.app.wants_carrier
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.heard("connection") == []
    # The supervisor step runs without a carrier: a fault is still found.
    h.step()
    h.arm.set_fault("motor 3 overheated")
    h.step()
    assert h.app.snapshot().stop_level is StopLevel.FAULT
    torque_kept(h)


def test_an_app_built_without_a_carrier_has_no_link(config, sim_arm, checkers, clock):
    app = RobotApp(config, sim_arm, checkers, None, clock)
    assert not app.link_alive
    assert not app.wants_carrier  # not started yet
    app.close()  # ends the jaw worker's thread


def test_the_thor_terminal_is_served_with_no_robot_connection(make_robot_app):
    h = make_robot_app(connected=False)
    h.hold_at(RAISED)
    assert h.app.stop("typed stop") == 1
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().status == "completed"
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.heard("connection") == []
    torque_kept(h)


def test_the_idle_joint_reads_go_on_with_no_robot_connection(make_robot_app):
    h = make_robot_app(connected=False)
    h.advance(1.0, step_s=0.02)
    # Every joint was read again within the second that passed.
    age = h.app._history.age(h.clock.monotonic())
    assert age is not None and age < 0.3


# attach_carrier()


def test_attach_sends_welcome_then_a_state_and_the_connection_event(make_robot_app):
    h = make_robot_app(connected=False)
    robot_end, workstation_end = loopback_pair()
    assert h.app.attach_carrier(robot_end)
    assert h.app.link_alive
    assert not h.app.wants_carrier
    first = receive_message(workstation_end, WAIT_S)
    assert isinstance(first, Welcome)
    assert first.session_id == h.app.session_id
    others = [receive_message(workstation_end, WAIT_S) for _ in range(2)]
    (event,) = (one for one in others if isinstance(one, Event))
    (state,) = (one for one in others if isinstance(one, RobotState))
    assert (event.kind, event.text, event.data) == MADE
    assert state.session_id == h.app.session_id
    assert (state.owner, state.owner_number, state.stop_number) == ("nobody", 0, 0)
    assert h.heard("connection") == [MADE]
    robot_end.close()
    torque_kept(h)


def test_a_state_is_sent_on_a_new_connection_though_nothing_changed(robot_app):
    h = robot_app
    seq = h.client.state.seq
    h.lose_link()
    # The clock stands still: no ROBOT_STATE falls due by its period.
    client = h.connect()
    assert client.state.seq > seq
    assert client.welcome.session_id == h.app.session_id


def test_attach_is_refused_before_start_and_after_close(make_robot_app):
    h = make_robot_app(connected=False, start=False)
    robot_end, workstation_end = loopback_pair()
    assert not h.app.attach_carrier(robot_end)
    h.app.start()
    assert workstation_end.receive(0) is None
    assert not h.app.link_alive
    assert h.app.attach_carrier(robot_end)
    h.app.close()
    again, _ = loopback_pair()
    assert not h.app.attach_carrier(again)
    assert not h.app.wants_carrier
    assert again.alive  # left to the caller
    for end in (robot_end, again):
        end.close()


def test_a_dead_carrier_is_not_attached(make_robot_app):
    h = make_robot_app(connected=False)
    robot_end, _ = loopback_pair()
    robot_end.close()
    assert not h.app.attach_carrier(robot_end)
    assert h.app.wants_carrier
    assert h.heard("connection") == []


def test_attach_with_a_live_carrier_attached_returns_false_and_changes_nothing(
    robot_app,
):
    h = robot_app
    h.new_task()
    offered, other_end = loopback_pair()
    assert not h.app.attach_carrier(offered)
    assert not h.app.wants_carrier
    assert offered.alive  # the caller closes it
    assert other_end.receive(0) is None
    assert h.heard("connection") == []
    # The attached one still serves.
    assert final(h, h.client.checker_state("clear_contact_spots")).status == (
        "completed"
    )
    h.step()
    assert other_end.receive(0) is None
    offered.close()


def test_a_dead_carrier_is_replaced_only_once_its_loss_was_handled(robot_app):
    h = robot_app
    offered, _ = loopback_pair()
    with h.app._lock:  # the dispatcher cannot handle the loss
        h.robot_end.break_link()
        assert not h.app.link_alive
        assert not h.app.wants_carrier
        assert not h.app.attach_carrier(offered)
    h.wait_local_event("connection")
    assert h.app.wants_carrier
    assert h.app.attach_carrier(offered)
    assert [event[2] for event in h.heard("connection")] == [
        {"owner_before": "nobody", "wait_cancelled": False},
        {"connected": True},
    ]
    offered.close()


def test_the_loss_of_each_carrier_is_handled_once(robot_app):
    h = robot_app
    for lost in (1, 2, 3):
        h.lose_link()
        h.step()
        h.step()
        assert len(h.heard("connection")) == 2 * lost - 1
        h.connect()
        assert len(h.heard("connection")) == 2 * lost
    kinds = ["connected" in event[2] for event in h.heard("connection")]
    assert kinds == [False, True] * 3
    assert h.app.snapshot().owner_number == 0  # nobody owned the arm
    torque_kept(h)


# Loss in each motion state, then a new connection: nothing restarts.


def in_an_agent_move(h) -> Callable[[], None]:
    h.hold_at(RAISED)
    h.new_task()
    reached, release = h.hold_move_after(6)
    move_id = h.client.move(REACH, start_q=RAISED)
    h.client.permit(move_id, len(REACH) - 1)
    assert reached.wait(WAIT_S)
    return release.set


def in_the_assistance_wait(h) -> Callable[[], None]:
    h.hold_at(RAISED)
    h.new_task()
    final(h, h.client.request_owner("help_needed", reason="the agent is stuck"))
    h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    return nothing


def in_the_timed_return(h) -> Callable[[], None]:
    in_the_assistance_wait(h)
    reached, release = h.hold_move_after(5)
    h.advance_until(lambda: h.heard("expired"))
    assert reached.wait(WAIT_S)
    return release.set


def in_a_rest_asked_for_over_the_connection(h) -> Callable[[], None]:
    h.hold_at(RAISED)
    reached, release = h.hold_move_after(5)
    h.client.rest()
    assert reached.wait(WAIT_S)
    return release.set


def in_a_jaw_command(h) -> Callable[[], None]:
    h.hold_at(RAISED)
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    return nothing


def holding_for_the_agent(h) -> Callable[[], None]:
    h.hold_at(RAISED)
    h.new_task()
    return nothing


def holding_after_a_stop(h) -> Callable[[], None]:
    holding_for_the_agent(h)
    h.client.stop()
    h.client.wait_state(lambda s: s.stop_number == 1)
    return nothing


def holding_with_a_fault_latched(h) -> Callable[[], None]:
    holding_for_the_agent(h)
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    h.arm.clear_faults()
    return nothing


STATES = {
    "an agent move": in_an_agent_move,
    "the assistance wait": in_the_assistance_wait,
    "the timed return": in_the_timed_return,
    "a rest asked for over the connection": in_a_rest_asked_for_over_the_connection,
    "a jaw command": in_a_jaw_command,
    "a hold for the agent": holding_for_the_agent,
    "a hold after a STOP": holding_after_a_stop,
    "a fault-stop": holding_with_a_fault_latched,
}


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("reconnect", ["before the work ended", "after the work ended"])
def test_loss_ends_in_a_hold_and_a_new_connection_restarts_nothing(
    robot_app, state, reconnect
):
    h = robot_app
    finish = STATES[state](h)
    stop_number = h.app.snapshot().stop_number
    h.lose_link()
    if reconnect == "after the work ended":
        finish()
        h.idle()
    client = h.connect()
    finish()
    h.idle()
    assert h.app._worker.wait_idle(WAIT_S)
    h.step()
    sent, jaws = h.sent(), h.arm.gripper_read()
    expired = len(h.heard("expired"))
    authority = h.app.snapshot()
    assert authority.owner is Owner.NOBODY
    assert authority.stop_number == stop_number  # a loss is not a STOP
    assert h.app.motion_state is MotionState.HOLD
    assert not h.app._wait.active()
    # Longer than the assistance wait: no move, no wait, no return begins.
    h.advance(LONG_S, step_s=2.0)
    h.idle()
    assert h.sent() == sent
    assert h.arm.gripper_read() == jaws
    assert len(h.heard("expired")) == expired
    assert h.app.snapshot() == authority
    assert h.app.motion_state is MotionState.HOLD
    state_now = client.wait_state(lambda s: s.clock_mono_s >= h.clock.monotonic() - 1)
    assert state_now.assistance_wait_s is None
    assert (state_now.owner, state_now.motion) == ("nobody", "hold")
    assert [("connected" in e[2]) for e in h.heard("connection")] == [False, True]
    assert h.app.running and h.app.link_alive
    torque_kept(h)


# Nothing is replayed.


def held_move(h) -> tuple[str, threading.Event]:
    """An agent MOVE along PATH from rest, held once 6 setpoints were sent."""
    reached, release = h.hold_move_after(6)
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def test_a_result_that_became_final_with_no_connection_is_not_sent_later(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.lose_link()
    release.set()
    h.idle()
    client = h.connect()
    lost = {"command_id": move_id, "status": "cancelled", "reason": LINK_LOST}
    # ROBOT_STATE's latest command outcome shows it.
    state = client.wait_state(lambda s: s.last_result == lost)
    assert (state.owner, state.owner_number) == ("nobody", 2)
    h.step()
    assert client.results(move_id) == []
    assert h.sent() == PATH[:6]
    torque_kept(h)


def test_the_result_of_a_command_of_the_old_connection_stays_off_the_new_one(
    robot_app,
):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.lose_link()
    client = h.connect()
    # The move ends only now, with the new connection attached.
    release.set()
    h.idle()
    lost = {"command_id": move_id, "status": "cancelled", "reason": LINK_LOST}
    client.wait_state(lambda s: s.last_result == lost)
    h.step()
    assert client.results(move_id) == []
    assert h.sent() == PATH[:6]


def test_a_repeated_command_id_is_answered_from_the_store_and_not_run_again(
    robot_app,
):
    h = robot_app
    h.new_task()
    done = h.client.run_move(PATH[:6], start_q=REST)
    assert done.status == "completed"
    move_id = done.command_id
    cut_id, release = held_move_from(h, PATH[5])
    h.lose_link()
    release.set()
    h.idle()
    sent = h.sent()
    client = h.connect()
    for command_id, status, reason in (
        (move_id, "completed", ""),
        (cut_id, "cancelled", LINK_LOST),
    ):
        # The same command, sent again with numbers that are current now.
        client.move(PATH[:6], start_q=REST, command_id=command_id)
        answer = client.wait_result(command_id, timeout_s=WAIT_S)
        assert (answer.status, answer.reason) == (status, reason)
        assert [one.status for one in client.results(command_id)] == [status]
    h.idle()
    assert h.sent() == sent
    torque_kept(h)


def held_move_from(h, start: list[float]) -> tuple[str, threading.Event]:
    """An agent MOVE of 10 setpoints from `start`, held after its third."""
    path = [[q + 0.004 * (index + 1) for q in start] for index in range(10)]
    reached, release = h.hold_move_after(len(h.arm.sent) + 3)
    command_id = h.client.move(path, start_q=start)
    h.client.permit(command_id, len(path) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def test_a_command_with_an_old_owner_number_is_refused_on_the_new_connection(
    robot_app,
):
    h = robot_app
    h.new_task()
    assert h.app.snapshot().owner_number == 1
    h.lose_link()
    client = h.connect()
    # The agent's ownership ended with the connection: owner number 2.
    for send in (
        lambda: client.move(PATH, start_q=REST, owner_number=1),
        lambda: client.gripper(0.5, owner_number=1),
    ):
        refused = client.wait_result(send(), timeout_s=WAIT_S)
        assert refused.status == "refused"
        assert refused.reason == "control owner is nobody, not agent"
    for send in (
        lambda: client.rest(owner_number=1),
        lambda: client.release(owner_number=1),
    ):
        refused = client.wait_result(send(), timeout_s=WAIT_S)
        assert refused.status == "refused"
        assert refused.reason == "owner number 1 does not match the current 2"
    # With the agent the owner again, its old number is still refused.
    h.new_task()
    for send in (
        lambda: client.move(PATH, start_q=REST, owner_number=1),
        lambda: client.gripper(0.5, owner_number=1),
        lambda: client.request_owner("help_needed", owner_number=1),
    ):
        refused = client.wait_result(send(), timeout_s=WAIT_S)
        assert refused.status == "refused"
        assert refused.reason == "owner number 1 does not match the current 3"
    h.idle()
    assert h.sent() == []
    torque_kept(h)


def test_a_command_with_an_old_stop_number_is_refused_on_the_new_connection(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.lose_link()
    assert h.app.stop("typed stop") == 1  # at the Thor terminal, connection down
    client = h.connect()
    for send in (
        lambda: client.rest(stop_number=0),
        lambda: client.request_owner("agent", stop_number=0),
        lambda: client.shutdown(stop_number=0),
    ):
        refused = client.wait_result(send(), timeout_s=WAIT_S)
        assert refused.status == "refused"
        assert refused.reason == "stop number 0 does not match the current 1"
    h.idle()
    assert h.sent() == []
    torque_kept(h)


def test_link_messages_not_yet_handled_are_dropped_with_the_lost_connection(
    robot_app,
):
    h = robot_app
    # Outside the agent's envelope, inside the region a return to rest starts from.
    outside = [1.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    h.hold_at(outside)
    h.step()
    with h.app._lock:  # the dispatcher waits; both commands arrive in one pass
        # Refused after a joint read, which takes longer than one step period:
        # the rest behind it stays unhandled when the pass ends.
        task_id = h.client.request_owner("agent")
        rest_id = h.client.rest()
        h.workstation_end.close()  # the robot end still receives both
    h.wait_local_event("connection")
    assert h.app._results.get(task_id).status == "refused"
    client = h.connect()
    h.step()
    h.step()
    h.idle()
    assert h.app._results.get(rest_id) is None
    assert client.results(rest_id) == []
    assert h.sent() == []
    assert h.app.motion_state is MotionState.HOLD
    torque_kept(h)


def test_a_rest_typed_at_the_thor_terminal_goes_on_across_a_new_connection(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert reached.wait(WAIT_S)
    h.lose_link()
    client = h.connect()
    busy = client.wait_result(client.rest(), timeout_s=WAIT_S)
    assert (busy.status, busy.reason) == (
        "refused",
        "busy: a return to rest is running",
    )
    release.set()
    assert done.wait().status == "completed"
    h.idle()
    assert len(h.arm.sent) == 209  # one return, not two
    assert h.app.motion_state is MotionState.REST_HOLD
    torque_kept(h)


def test_a_new_task_runs_on_the_new_connection(robot_app):
    h = robot_app
    h.new_task()
    h.lose_link()
    client = h.connect()
    h.new_task()
    assert h.app.snapshot().owner_number == 3
    moved = client.run_move(PATH[:6], start_q=REST)
    assert moved.status == "completed", moved.reason
    assert h.sent() == PATH[:6]


def test_a_timed_return_stopped_by_a_lost_connection_does_not_begin_on_a_new_one(
    robot_app,
):
    h = robot_app
    h.hold_at(RAISED)
    # As the motion worker holds it once the loss was handled: told to stop.
    job = _Job("the timed return to rest", lambda job: None, None, needs_link=True)
    job.stop_reason = LINK_LOST
    assert h.app.link_alive
    ending = h.app._run_timed_return(job, 0)
    assert (ending.status, ending.reason) == (
        "refused",
        "no timed return to rest: robot connection lost",
    )
    assert h.sent() == []
    assert h.app.motion_state is MotionState.HOLD
    torque_kept(h)


# RobotDialer


class FakeApp:
    """What a RobotDialer asks of the robot app."""

    def __init__(self) -> None:
        self.wants_carrier = True
        self.accepts = True
        self.closed = False
        self.attached: list[Carrier] = []

    def attach_carrier(self, carrier: Carrier) -> bool:
        if not self.accepts:
            return False
        self.attached.append(carrier)
        self.wants_carrier = False
        return True

    def wait_closed(self, timeout: float | None = None) -> bool:
        return self.closed


class Workstation:
    """The connect() of a dialer on loopback links; keeps the far ends."""

    def __init__(self, *outcomes: Exception | str) -> None:
        """outcomes: what each call does in turn, an error to raise or "ok".

        Once they are used up every call connects.
        """
        self._outcomes = list(outcomes)
        self.far_ends: list[LoopbackCarrier] = []
        self.near_ends: list[LoopbackCarrier] = []
        self.threads: list[str] = []

    def __call__(self) -> LoopbackCarrier:
        self.threads.append(threading.current_thread().name)
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        near, far = loopback_pair()
        self.near_ends.append(near)
        self.far_ends.append(far)
        return near

    def close(self) -> None:
        for end in (*self.near_ends, *self.far_ends):
            end.close()


@pytest.fixture
def make_dialer(clock):
    """make_dialer(app, connect, **options): a started RobotDialer, closed after."""
    made: list[RobotDialer] = []
    connects: list[Workstation] = []

    def make(app, connect, start: bool = True, **options) -> RobotDialer:
        options.setdefault("poll_s", 0.001)
        dialer = RobotDialer(app, connect, clock, **options)
        made.append(dialer)
        if isinstance(connect, Workstation):
            connects.append(connect)
        if start:
            dialer.start()
        return dialer

    yield make
    for dialer in made:
        dialer.close()
    for connect in connects:
        connect.close()


def passes(dialer: RobotDialer, count: int = 3) -> None:
    """Let the dialer's thread make `count` passes that began after this call."""
    for _ in range(count):
        assert dialer.wait_pass(WAIT_S), "the dialer made no pass"


def warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == DIALER_LOGGER and record.levelno == logging.WARNING
    ]


def test_the_dialer_connects_and_attaches_when_the_app_wants_a_carrier(make_dialer):
    app, workstation = FakeApp(), Workstation()
    dialer = make_dialer(app, workstation)
    passes(dialer)
    assert dialer.attempts == 1
    assert app.attached == workstation.near_ends
    assert dialer.carrier is workstation.near_ends[0]
    assert dialer.last_error == ""
    assert workstation.threads == ["robot-dialer"]
    # A live link is left alone.
    passes(dialer)
    assert dialer.attempts == 1 and workstation.near_ends[0].alive


def test_the_dialer_waits_until_the_app_wants_a_carrier(make_dialer, clock):
    app, workstation = FakeApp(), Workstation()
    app.wants_carrier = False  # not started yet, or its own carrier is alive
    dialer = make_dialer(app, workstation)
    passes(dialer)
    clock.advance(60.0)
    passes(dialer)
    assert dialer.attempts == 0
    app.wants_carrier = True
    passes(dialer)
    assert dialer.attempts == 1


def test_a_failed_connect_is_tried_again_after_retry_s_on_the_clock(make_dialer, clock):
    refused = "cannot connect to 127.0.0.1:5000: connection refused"
    app = FakeApp()
    workstation = Workstation(RuntimeError(refused), RuntimeError(refused))
    dialer = make_dialer(app, workstation, retry_s=1.0)
    passes(dialer)
    assert dialer.attempts == 1
    assert dialer.last_error == refused
    assert dialer.carrier is None
    clock.advance(0.5)
    passes(dialer)
    assert dialer.attempts == 1
    clock.advance(0.5)
    passes(dialer)
    assert dialer.attempts == 2
    passes(dialer)
    assert dialer.attempts == 2
    clock.advance(1.0)
    passes(dialer)
    assert dialer.attempts == 3
    assert app.attached == workstation.near_ends
    assert dialer.last_error == ""


def test_a_connect_error_is_logged_once_per_distinct_text(make_dialer, clock, caplog):
    refused = "cannot connect to 127.0.0.1:5000: connection refused"
    app = FakeApp()
    workstation = Workstation(
        RuntimeError(refused),
        RuntimeError(refused),
        TimeoutError("no answer within 2 s"),
        RuntimeError(refused),
        TimeoutError("no answer within 2 s"),
    )
    with caplog.at_level(logging.INFO, logger=DIALER_LOGGER):
        dialer = make_dialer(app, workstation, retry_s=1.0)
        for attempt in range(1, 7):
            passes(dialer)
            assert dialer.attempts == attempt
            clock.advance(1.0)
        assert app.attached == workstation.near_ends
        logged = warnings(caplog)
        assert len(logged) == 2
        assert refused in logged[0] and "RuntimeError" in logged[0]
        assert "no answer within 2 s" in logged[1]
        # After a connection the same error is news again.
        caplog.clear()
        workstation._outcomes += [RuntimeError(refused), RuntimeError(refused)]
        workstation.far_ends[0].break_link()
        app.wants_carrier = True
        for attempt in (7, 8):
            passes(dialer)
            clock.advance(1.0)
            passes(dialer)
            assert dialer.attempts == attempt
        logged = warnings(caplog)
        assert len([line for line in logged if refused in line]) == 1


def test_a_carrier_the_app_does_not_take_is_closed_and_the_dial_repeated(
    make_dialer, clock, caplog
):
    app, workstation = FakeApp(), Workstation()
    app.accepts = False
    with caplog.at_level(logging.WARNING, logger=DIALER_LOGGER):
        dialer = make_dialer(app, workstation, retry_s=1.0)
        passes(dialer)
        assert dialer.attempts == 1
        assert not workstation.near_ends[0].alive
        assert dialer.carrier is None
        clock.advance(1.0)
        passes(dialer)
        assert dialer.attempts == 2
        assert not workstation.near_ends[1].alive
        assert len(warnings(caplog)) == 1
    app.accepts = True
    clock.advance(1.0)
    passes(dialer)
    assert dialer.attempts == 3
    assert dialer.carrier is workstation.near_ends[2]


def test_a_dead_link_is_closed_and_dialed_again_after_retry_s(make_dialer, clock):
    app, workstation = FakeApp(), Workstation()
    dialer = make_dialer(app, workstation, retry_s=2.0)
    passes(dialer)
    workstation.far_ends[0].break_link()
    passes(dialer)
    assert dialer.carrier is None
    # The app has not handled the loss yet.
    clock.advance(2.0)
    passes(dialer)
    assert dialer.attempts == 1
    app.wants_carrier = True
    passes(dialer)
    assert dialer.attempts == 2
    assert dialer.carrier is workstation.near_ends[1]


def test_no_dial_is_made_before_retry_s_has_passed_since_the_link_died(
    make_dialer, clock
):
    app, workstation = FakeApp(), Workstation()
    dialer = make_dialer(app, workstation, retry_s=2.0)
    passes(dialer)
    workstation.far_ends[0].break_link()
    app.wants_carrier = True
    passes(dialer)
    clock.advance(1.5)
    passes(dialer)
    assert dialer.attempts == 1
    clock.advance(0.5)
    passes(dialer)
    assert dialer.attempts == 2


def test_close_ends_the_dialer_and_closes_its_link(make_dialer, clock):
    app, workstation = FakeApp(), Workstation()
    dialer = make_dialer(app, workstation)
    passes(dialer)
    dialer.close()
    assert not dialer.running
    assert not workstation.near_ends[0].alive
    assert dialer.carrier is None
    assert "robot-dialer" not in {thread.name for thread in threading.enumerate()}
    app.wants_carrier = True
    clock.advance(60.0)
    assert not dialer.wait_pass(0.05)
    assert dialer.attempts == 1
    dialer.close()


def test_a_dialer_can_be_closed_unstarted_and_starts_once(make_dialer):
    app, workstation = FakeApp(), Workstation()
    unstarted = make_dialer(app, workstation, start=False)
    unstarted.close()
    with pytest.raises(RuntimeError, match=r"^the dialer was closed$"):
        unstarted.start()
    dialer = make_dialer(app, workstation)
    with pytest.raises(RuntimeError, match=r"^the dialer was already started$"):
        dialer.start()
    passes(dialer)
    assert dialer.attempts == 1


def test_close_during_a_connect_closes_what_it_returns(make_dialer):
    app = FakeApp()
    calling, go_on = threading.Event(), threading.Event()
    near, far = loopback_pair()

    def connect() -> LoopbackCarrier:
        calling.set()
        assert go_on.wait(WAIT_S)
        return near

    dialer = make_dialer(app, connect)
    assert calling.wait(WAIT_S)
    closing = threading.Thread(target=dialer.close, daemon=True)
    closing.start()
    go_on.set()
    closing.join(WAIT_S)
    assert not closing.is_alive()
    assert not dialer.running
    assert app.attached == []
    assert not near.alive
    far.close()


def test_the_dialer_ends_when_the_app_has_closed(make_dialer):
    app, workstation = FakeApp(), Workstation()
    dialer = make_dialer(app, workstation)
    passes(dialer)
    app.closed = True
    assert dialer.wait_ended(WAIT_S)
    assert not dialer.running
    assert not workstation.near_ends[0].alive
    assert dialer.attempts == 1


def test_an_error_in_the_app_does_not_end_the_dialer(make_dialer, clock, caplog):
    app, workstation = FakeApp(), Workstation()
    real = app.attach_carrier
    calls = []

    def attach_carrier(carrier: Carrier) -> bool:
        calls.append(carrier)
        if len(calls) == 1:
            raise RuntimeError("attach bug")
        return real(carrier)

    app.attach_carrier = attach_carrier
    with caplog.at_level(logging.ERROR, logger=DIALER_LOGGER):
        dialer = make_dialer(app, workstation, retry_s=1.0)
        passes(dialer)
        assert not workstation.near_ends[0].alive
        clock.advance(1.0)
        passes(dialer)
    assert dialer.carrier is workstation.near_ends[1]
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1


@pytest.mark.parametrize("retry_s", [0.0, -1.0, 0.19, float("nan"), float("inf")])
def test_the_dialer_refuses_a_retry_time_that_allows_a_burst_of_dials(clock, retry_s):
    with pytest.raises(ValueError, match=r"^retry_s must be finite and at least 0\.2"):
        RobotDialer(FakeApp(), Workstation(), clock, retry_s=retry_s)


@pytest.mark.parametrize("poll_s", [0.0, -0.1, float("nan"), float("inf")])
def test_the_dialer_refuses_a_bad_poll_time(clock, poll_s):
    with pytest.raises(ValueError, match=r"^poll_s must be finite and above 0"):
        RobotDialer(FakeApp(), Workstation(), clock, poll_s=poll_s)


def test_the_dialer_gives_the_app_its_connections_and_new_ones_after_a_loss(
    make_robot_app, make_dialer
):
    h = make_robot_app(connected=False, pose=RAISED)
    workstation = Workstation()
    arm_callers: set[str] = set()
    for name in dir(h.arm):
        call = getattr(h.arm, name)
        if name.startswith("_") or not callable(call):
            continue

        def recorded(*args, _call=call, **kwargs):
            arm_callers.add(threading.current_thread().name)
            return _call(*args, **kwargs)

        setattr(h.arm, name, recorded)
    dialer = make_dialer(h.app, workstation, retry_s=1.0)
    h.wait_local_events("connection", 1)
    first = receive_message(workstation.far_ends[0], WAIT_S)
    assert isinstance(first, Welcome) and first.session_id == h.app.session_id
    assert h.app.link_alive and dialer.attempts == 1
    workstation.far_ends[0].break_link()
    h.wait_local_events("connection", 2)
    passes(dialer)
    assert dialer.attempts == 1  # retry_s has not passed on the app's clock
    h.advance(1.0)
    h.wait_local_events("connection", 3)
    again = receive_message(workstation.far_ends[1], WAIT_S)
    assert isinstance(again, Welcome) and again.session_id == h.app.session_id
    assert dialer.attempts == 2
    assert [("connected" in e[2]) for e in h.heard("connection")] == [
        True,
        False,
        True,
    ]
    h.step()
    # The dialer never touches the arm.
    assert arm_callers and "robot-dialer" not in arm_callers
    assert h.sent() == []
    assert h.app.motion_state is MotionState.HOLD
    torque_kept(h)


def test_closing_the_dialer_leaves_the_app_holding(make_robot_app, make_dialer):
    h = make_robot_app(connected=False, pose=RAISED)
    dialer = make_dialer(h.app, Workstation())
    h.wait_local_events("connection", 1)
    dialer.close()
    h.wait_local_events("connection", 2)
    assert h.app.running and not h.app.link_alive
    assert h.app.motion_state is MotionState.HOLD
    assert h.sent() == []
    torque_kept(h)


def test_the_dialer_ends_when_the_real_app_closes(make_robot_app, make_dialer):
    h = make_robot_app(connected=False)
    workstation = Workstation()
    dialer = make_dialer(h.app, workstation)
    h.wait_local_events("connection", 1)
    h.app.close()
    assert dialer.wait_ended(WAIT_S)
    assert not workstation.near_ends[0].alive
    assert h.arm.torque_on()


# The connection settings


def test_the_retry_time_is_a_setting_with_a_floor(tmp_path):
    assert ConnectionConfig() == ConnectionConfig(retry_s=1.0)
    assert RobotAppConfig.default().connection == ConnectionConfig()
    assert MIN_RETRY_S == 0.2
    shipped = repo_root() / "examples" / "agentic_remote" / "configs" / "robot_app.yaml"
    assert load_config(shipped).connection.retry_s == 1.0
    text = shipped.read_text(encoding="utf-8")
    assert "connection:\n" in text
    for value, problem in (
        ("0.1", r"connection\.retry_s must be at least 0\.2, got 0\.1"),
        ("wrong", r"connection\.retry_s must be a number"),
        ("-1", r"connection\.retry_s must be at least 0\.2"),
    ):
        path = tmp_path / "robot_app.yaml"
        path.write_text(
            text.replace("  retry_s: 1.0", f"  retry_s: {value}"), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match=problem):
            load_config(path)
    changed = dataclasses.replace(
        RobotAppConfig.default(), connection=ConnectionConfig(retry_s=3)
    )
    assert changed.connection.retry_s == 3.0
    assert isinstance(changed.connection.retry_s, float)
