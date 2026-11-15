# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app, a real link on 127.0.0.1, the robot link service and its clients.

The gate of phase P1: a test arm completes commands over the real connection and
nothing replays after a reconnect. The robot app runs on the simulated arm, and
on APAI's mock arm in the tests marked "apai"; its clock is the fake clock, the
link and the local API run in real time.

Each test makes a few TCP connections to 127.0.0.1, none sooner than 0.3 s
after the one before. "Killed" means closed: a service or a client this test
started. test_link_programs.py ends a service process with signals.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
from isaaccapture_examples.agentic_remote.link_service.service import AGENT_GONE, LOST
from isaaccapture_examples.agentic_remote.protocol.link_carrier import LinkCarrier
from isaaccapture_examples.agentic_remote.protocol.messages import Stop
from isaaccapture_examples.agentic_remote.robot_app.app import LINK_LOST
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    CheckerSet,
    MotionState,
    Owner,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker, SimArm

import test_apai_app as apai
from conftest import AppHarness, check_released_at_rest
from link_stack import (
    CONNECT_S,
    HOST,
    PEER_TIMEOUT_S,
    RETRY_S,
    WAIT_S,
    Stack,
    Station,
    until,
)

pytestmark = pytest.mark.link

REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, from rest.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# How much sooner or later than the peer timeout a silent peer may be noticed.
EARLY_S, LATE_S = 1.0, 1.5


def onward(start: list[float], count: int = 10) -> list[list[float]]:
    """`count` setpoints at 0.2 rad/s on every joint, from start."""
    return [[q + 0.004 * (index + 1) for q in start] for index in range(count)]


def torque_kept(h: AppHarness) -> None:
    assert h.arm.torque_on(), "the motors are off"
    assert h.arm.released_at == [], "torque was released"


def held_move(h: AppHarness) -> tuple[str, threading.Event]:
    """An agent MOVE along PATH from rest, held once 6 setpoints were sent."""
    reached, release = h.hold_move_after(6)
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def statuses(station: Station, command_id: str) -> list[str]:
    return [result.status for result in station.ws.results(command_id)]


@pytest.fixture
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("ls")  # a short path, mode 0700


@pytest.fixture
def stack(make_robot_app, folder) -> Iterator[Stack]:
    """A started robot app on the simulated arm with no robot connection yet."""
    made = Stack(make_robot_app(connected=False), folder)
    yield made
    made.close()


def up(stack: Stack) -> Station:
    """Start the service and the dialer; the agent client, with the robot connected."""
    stack.start_service()
    stack.dial()
    return stack.agent()


def console_of(stack: Stack) -> Station:
    console = stack.client("console")
    stack.connected(console)
    return console


# A whole task


def test_a_whole_agent_task_completes_over_the_real_link(stack):
    h = stack.h
    agent = up(stack)
    console = console_of(stack)
    assert stack.dialer.attempts == 1
    assert agent.ws.welcome.session_id == h.app.session_id

    owned = h.new_task()
    moved = agent.ws.run_move(PATH, WAIT_S, start_q=REST)
    agent.final(moved.command_id)
    assert moved.status == "completed", moved.reason
    assert moved.detail == {"outcome": "completed", "sent": 25, "total": 25}
    assert h.sent() == PATH
    gave = agent.final(agent.ws.request_owner("nobody", reason="task done"))
    assert gave.status == "completed", gave.reason

    # The operator's part, from the console: back to rest, then motors off there.
    console.ws.wait_state(lambda s: s.owner_number == gave.detail["owner_number"])
    rest = console.final(console.ws.rest())
    assert rest.status == "completed", rest.reason
    console.ws.wait_state(lambda s: s.motion == "rest_hold" and s.torque_on)
    released = console.final(console.ws.release())
    assert released.status == "completed", released.reason
    state = console.ws.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on
    assert len(h.arm.released_at) == 1

    # Every RESULT of the agent's commands went to the console as well, and the
    # console's own went to the console alone.
    console.ws.wait_result(gave.command_id, timeout_s=WAIT_S)
    assert statuses(console, moved.command_id) == ["accepted", "completed"]
    assert statuses(agent, rest.command_id) == []
    assert owned.detail["owner_number"] == 1
    # The service measured the Thor's clock against its own with ping round trips.
    status = agent.end.wait_status(lambda s: s.clock_two_way, WAIT_S)
    assert status.clock_offset_ns is not None
    assert status.roles == {"agent": 1, "console": 1}
    assert agent.ws.unexpected == [] and console.ws.unexpected == []
    assert agent.end.notices == [] and console.end.notices == []


# STOP


@pytest.mark.parametrize("role", ["agent", "console", "viewer", "tool"])
def test_a_stop_from_each_role_ends_the_move_and_the_arm_holds(stack, role):
    h = stack.h
    agent = up(stack)
    stopper = agent if role == "agent" else stack.client(role)
    h.new_task()
    move_id, release = held_move(h)
    stopper.ws.stop("pressed stop", sender=role)
    state = agent.ws.wait_state(lambda s: s.stop_number == 1)
    assert (state.owner, state.stop_level) == ("nobody", "stop")
    release.set()
    result = agent.final(move_id)
    assert result.status == "cancelled"
    assert result.reason == f"STOP from {role}: pressed stop"
    assert h.sent() == PATH[:6]
    assert h.app.motion_state is MotionState.HOLD
    torque_kept(h)
    # The service has seen the higher stop number: the next task passes it.
    agent.end.wait_status(lambda s: not s.stop_unconfirmed, WAIT_S)
    h.new_task()
    at = agent.ws.measured_pose()
    again = agent.ws.run_move(onward(at), WAIT_S, start_q=at)
    assert again.status == "completed", again.reason


# The robot connection cut


def test_a_cut_during_a_move_holds_the_arm_and_nothing_is_replayed(stack):
    h = stack.h
    agent = up(stack)
    console = console_of(stack)
    h.new_task()
    move_id, release = held_move(h)
    before = agent.ws.wait_state(lambda s: s.owner == "agent")
    numbers = {
        "session_id": before.session_id,
        "owner_number": before.owner_number,
        "stop_number": before.stop_number,
    }

    stack.held = True
    stack.cut()
    # The service answers for the command the Thor can no longer answer.
    for station in (agent, console):
        lost = station.ws.wait_result(move_id, timeout_s=WAIT_S)
        assert (lost.status, lost.reason) == ("failed", LOST)
    status = agent.end.wait_status(lambda s: not s.robot_connected, WAIT_S)
    assert status.close_reason
    # A command sent meanwhile is refused by the service; a STOP gets a notice.
    meanwhile = agent.ws.move(PATH, start_q=REST, **numbers)
    refused = agent.ws.wait_result(meanwhile, timeout_s=WAIT_S)
    assert (refused.status, refused.reason) == ("refused", LOST)
    console.ws.stop()
    notice = console.end.wait_notice(lambda n: n.about_type == Stop.TYPE, WAIT_S)
    assert "the robot holds on its own" in notice.text

    # The Thor holds on its own: the move ended, nobody owns the arm.
    release.set()
    h.idle()
    assert h.sent() == PATH[:6]
    assert h.app.motion_state is MotionState.HOLD
    now = h.app.snapshot()
    assert (now.owner, now.owner_number, now.stop_number) == (Owner.NOBODY, 2, 0)
    torque_kept(h)

    # The robot app dials again; the session is the same one.
    stack.held = False
    stack.connected(agent)
    stack.connected(console)
    assert stack.dialer.last_error == ""
    state = agent.ws.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
    assert state.last_result == {
        "command_id": move_id,
        "status": "cancelled",
        "reason": LINK_LOST,
    }
    # A command with the numbers from before the cut is refused by the Thor.
    old = agent.ws.wait_result(
        agent.ws.move(onward(PATH[5]), start_q=PATH[5], **numbers), timeout_s=WAIT_S
    )
    assert old.status == "refused"
    # The cut move, sent again under its id with current numbers, is answered
    # from the Thor's result store and not run.
    agent.ws.move(PATH, start_q=REST, command_id=move_id)
    until(lambda: len(agent.ws.results(move_id)) >= 3, "the stored RESULT")
    assert statuses(agent, move_id) == ["accepted", "failed", "cancelled"]
    h.step()
    h.idle()
    assert h.sent() == PATH[:6]  # nothing moved since the cut
    assert agent.ws.unexpected == []

    # Until a new task: then the arm moves again.
    h.new_task()
    at = agent.ws.measured_pose()
    moved = agent.ws.run_move(onward(at), WAIT_S, start_q=at)
    assert moved.status == "completed", moved.reason
    assert len(h.sent()) == 16
    torque_kept(h)


def test_a_cut_in_a_hold_for_the_agent_ends_its_ownership(stack):
    h = stack.h
    agent = up(stack)
    h.new_task()
    stack.cut()
    stack.connected(agent)
    state = agent.ws.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.motion) == ("nobody", "rest_hold")
    assert state.assistance_wait_s is None
    assert stack.dialer.attempts == 2
    assert h.sent() == []
    torque_kept(h)


# The service closed and started again


def test_the_service_closed_and_started_again_moves_nothing(stack):
    h = stack.h
    agent = up(stack)
    h.new_task()
    moved = agent.ws.run_move(PATH, WAIT_S, start_q=REST)
    assert moved.status == "completed", moved.reason
    session_id = h.app.session_id

    stack.held = True
    events = stack.connection_events()
    stack.stop_service()
    # The Thor notices that its peer is gone and holds; closing does not stop it.
    h.wait_local_events("connection", events + 1)
    assert agent.end.wait_dead(WAIT_S)
    h.idle()
    now = h.app.snapshot()
    assert (now.owner, now.owner_number, now.stop_number) == (Owner.NOBODY, 2, 0)
    assert h.app.motion_state is MotionState.HOLD
    assert h.sent() == PATH
    torque_kept(h)

    stack.start_service()
    stack.held = False
    agent = stack.agent()
    assert h.app.session_id == session_id
    state = agent.ws.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
    h.step()
    h.idle()
    assert h.sent() == PATH
    # A new task on the new service moves the arm again.
    h.new_task()
    again = agent.ws.run_move(onward(PATH[-1]), WAIT_S, start_q=PATH[-1])
    assert again.status == "completed", again.reason
    torque_kept(h)


# The robot app closed and started again


def new_checkers(config, clock) -> CheckerSet:
    limits, grasp = config.limits, config.grasp
    return CheckerSet(
        agent=BasicChecker(limits.agent, clock, grasp=grasp),
        human=BasicChecker(limits.human, clock, grasp=grasp),
        recovery=BasicChecker(limits.recovery, clock, watchdog_s=None, grasp=grasp),
    )


def test_the_robot_app_closed_and_started_again_has_a_new_session(stack, config, clock):
    first = stack.h
    agent = up(stack)
    console = console_of(stack)
    first.new_task()
    moved = agent.ws.run_move(PATH, WAIT_S, start_q=REST)
    assert moved.status == "completed", moved.reason
    old = agent.ws.wait_state(lambda s: s.owner == "agent")
    numbers = {
        "session_id": old.session_id,
        "owner_number": old.owner_number,
        "stop_number": old.stop_number,
    }

    stack.dialer.close()
    first.app.close()
    status = console.end.wait_status(lambda s: not s.robot_connected, WAIT_S)
    assert status.session_id == old.session_id  # of the last WELCOME
    torque_kept(first)  # closing the robot app leaves the motors on

    # A new robot app on an arm that holds where the first one left it.
    arm = SimArm(n_joints=config.n_joints, clock=clock, q0=first.arm.pose())
    second = AppHarness(config, arm, new_checkers(config, clock), clock, None, False)
    try:
        second.start()
        assert second.app.session_id != old.session_id
        stack.dial(second)
        for station in (agent, console):
            stack.connected(station)
            assert station.ws.welcome.session_id == second.app.session_id
        state = agent.ws.wait_state(lambda s: s.session_id == second.app.session_id)
        assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
        # Commands of the old session are refused, whoever sends them.
        at = agent.ws.measured_pose()
        for command_id in (
            agent.ws.move(onward(at), start_q=at, **numbers),
            agent.ws.request_owner("agent", **numbers),
            console.ws.rest(**numbers),
        ):
            refused = console.ws.wait_result(command_id, timeout_s=WAIT_S)
            assert refused.status == "refused"
            assert refused.reason.startswith("session id ")
        second.step()
        second.idle()
        assert second.sent() == []
        # A new task of the new session moves the arm.
        second.client = agent.ws
        second.new_task()
        again = agent.ws.run_move(onward(at), WAIT_S, start_q=at)
        assert again.status == "completed", again.reason
        assert len(second.sent()) == 10
        assert arm.torque_on() and arm.released_at == []
    finally:
        stack.close()
        second.close()
        check_released_at_rest(arm, config)


# The agent client killed


def test_an_agent_client_killed_while_it_owns_the_arm_starts_the_assistance_wait(
    stack,
):
    h = stack.h
    agent = up(stack)
    console = console_of(stack)
    h.new_task()
    agent.kill()
    state = console.ws.wait_state(lambda s: s.assistance_wait_s is not None)
    assert (state.owner, state.owner_number) == ("nobody", 2)
    event = console.ws.wait_event("help_needed", timeout_s=WAIT_S)
    assert AGENT_GONE in event.text
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.sent() == []
    torque_kept(h)
    # A person's STOP ends the wait, as it does for an agent that asked itself.
    console.ws.stop()
    console.ws.wait_state(
        lambda s: s.stop_number == 1 and s.assistance_wait_s is None, WAIT_S
    )
    status = console.end.wait_status(lambda s: "agent" not in s.roles, WAIT_S)
    assert status.robot_connected


def test_an_agent_client_killed_during_its_move_ends_the_move_in_a_hold(stack):
    h = stack.h
    agent = up(stack)
    console = console_of(stack)
    h.new_task()
    move_id, release = held_move(h)
    agent.kill()
    console.ws.wait_state(lambda s: s.assistance_wait_s is not None)
    release.set()
    h.idle()
    # The console gets the RESULT of the move its sender no longer can.
    result = console.final(move_id)
    assert result.status == "cancelled"
    assert h.sent() == PATH[:6]
    assert h.app.motion_state is MotionState.HOLD
    torque_kept(h)


def test_an_agent_client_killed_when_it_does_not_own_the_arm_changes_nothing(stack):
    h = stack.h
    agent = up(stack)
    console = console_of(stack)
    agent.kill()
    console.end.wait_status(lambda s: "agent" not in s.roles, WAIT_S)
    h.step()
    state = console.ws.wait_state()
    assert (state.owner, state.owner_number) == ("nobody", 0)
    assert state.assistance_wait_s is None
    assert console.ws.events == []


# A silent peer


def test_a_silent_workstation_is_noticed_within_the_peer_timeout(stack):
    h = stack.h
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, 0))
        server.listen(1)
        server.settimeout(WAIT_S)
        stack.port = server.getsockname()[1]
        stack.dial()
        peer, _ = server.accept()  # it reads nothing and sends nothing
        with peer:
            h.wait_local_events("connection", 1)
            began = time.monotonic()
            stack.held = True  # the silent listener is dialed once
            assert h.app.link_alive
            until(
                lambda: stack.connection_events() >= 2,
                "the loss of the silent link",
                PEER_TIMEOUT_S + 2 * LATE_S,
            )
            took = time.monotonic() - began
    assert PEER_TIMEOUT_S - EARLY_S < took < PEER_TIMEOUT_S + LATE_S
    assert not h.app.link_alive
    assert h.app.motion_state is MotionState.REST_HOLD
    assert stack.dialer.attempts == 1
    torque_kept(h)


def test_a_silent_robot_is_dropped_within_the_peer_timeout_and_the_next_one_taken(
    stack,
):
    stack.start_service()
    tool = stack.client("tool")
    with socket.create_connection((HOST, stack.port), timeout=CONNECT_S):
        began = time.monotonic()
        tool.end.wait_status(lambda s: s.robot_connected, WAIT_S)
        status = tool.end.wait_status(
            lambda s: not s.robot_connected, PEER_TIMEOUT_S + 2 * LATE_S
        )
        took = time.monotonic() - began
    assert PEER_TIMEOUT_S - EARLY_S < took < PEER_TIMEOUT_S + LATE_S
    assert status.close_reason
    assert status.session_id == ""  # it never said WELCOME
    # The robot app's connection is taken once the silent one is dead.
    stack.dial()
    stack.connected(tool)
    assert stack.dialer.attempts == 1


# The Thor's address


def test_a_robot_app_at_an_address_that_is_not_allowed_gets_no_robot_connection(
    stack,
):
    h = stack.h
    stack.start_service(allowed_peers=["192.0.2.7"])
    tool = stack.client("tool")
    dialer = stack.dial()
    # Every dial is closed by the service at once; the dialer keeps its pace.
    began = time.monotonic()
    until(lambda: dialer.attempts >= 3, "three dials")
    stack.held = True
    assert time.monotonic() - began >= 2 * RETRY_S - 0.05
    until(
        lambda: not h.app.link_alive and stack.connection_events() % 2 == 0,
        "the loss of the refused links",
    )
    assert not stack.service.robot_connected
    assert tool.end.status.robot_connected is False
    assert tool.end.status.session_id == ""
    assert h.app.motion_state is MotionState.REST_HOLD
    torque_kept(h)


# Sends from several threads


def test_several_threads_can_send_on_one_link_carrier():
    # The service sends on its robot connection from one thread per local
    # client, the robot loop and the accept loop.
    senders, each = 4, 200
    ours, theirs = socket.socketpair()
    left = LinkCarrier.adopt(ours.detach(), name="left")
    right = LinkCarrier.adopt(theirs.detach(), name="right")
    try:

        def send(number: int) -> None:
            for index in range(each):
                left.send_reliable(Stop.TYPE, f"{number}:{index}".encode())

        threads = [threading.Thread(target=send, args=(n,)) for n in range(senders)]
        for thread in threads:
            thread.start()
        seen: dict[int, list[int]] = {number: [] for number in range(senders)}
        for _ in range(senders * each):
            received = right.receive(WAIT_S)
            assert received is not None, "a link message is missing"
            number, _, index = received[1].decode().partition(":")
            seen[int(number)].append(int(index))
        for thread in threads:
            thread.join(WAIT_S)
        # Nothing lost or doubled, and each thread's link messages in its order.
        assert all(indexes == list(range(each)) for indexes in seen.values())
        assert right.receive(0.05) is None
        assert left.alive and right.alive
    finally:
        left.close()
        right.close()


# On APAI's mock arm


class UnconnectedApai(apai.ApaiHarness):
    """The robot app on APAI's mock arm, built with no robot connection."""

    def __init__(self, config, profile, clock) -> None:
        backend = apai.apai_backend
        checkers = backend.build_apai_checkers(config, profile, clock)
        arm = backend.build_apai_arm(config, profile, clock)
        self.spy = apai.BackendSpy(arm, checkers)
        AppHarness.__init__(self, config, arm, checkers, clock, None, False)


@pytest.fixture
def apai_stack(clock, folder) -> Iterator[Stack]:
    """A started robot app on APAI's mock arm with no robot connection yet."""
    profile = apai.apai_backend.load_apai_profile("mock")
    harness = UnconnectedApai(apai.make_config(profile), profile, clock)
    harness.start()
    made = Stack(harness, folder)
    yield made
    made.close()
    harness.close()
    apai.check_driver_use(harness)


@pytest.mark.apai
def test_a_whole_agent_task_completes_on_apais_mock_arm(apai_stack):
    h = apai_stack.h
    agent = up(apai_stack)
    console = console_of(apai_stack)
    h.new_task()
    moved = agent.ws.run_move(apai.LIFT, WAIT_S, start_q=apai.REST)
    agent.final(moved.command_id)
    assert moved.status == "completed", moved.reason
    assert moved.end_q == pytest.approx(apai.RAISED)
    assert h.sent() == apai.LIFT
    # The agent's harness approved every joint target the driver took.
    assert h.spy.approved == [(number, 0, 0) for number in range(1, 26)]
    gave = agent.final(agent.ws.request_owner("nobody", reason="task done"))
    assert gave.status == "completed", gave.reason
    console.ws.wait_state(lambda s: s.owner_number == gave.detail["owner_number"])
    rest = console.final(console.ws.rest())
    assert rest.status == "completed", rest.reason
    assert h.sent()[-1] == apai.REST
    released = console.final(console.ws.release())
    assert released.status == "completed", released.reason
    assert h.spy.calls.count("disconnect") == 1
    assert h.spy.released_at[0] == pytest.approx(apai.REST, abs=0.05)


@pytest.mark.apai
def test_a_stop_from_the_console_halts_apais_harnesses_and_not_the_driver(apai_stack):
    h = apai_stack.h
    agent = up(apai_stack)
    console = console_of(apai_stack)
    h.new_task()
    move_id, release = apai.held_move(h, 6)
    console.ws.stop("operator pressed stop")
    agent.ws.wait_state(lambda s: s.stop_number == 1)
    for harness in h.harnesses:
        assert harness.halted == "operator pressed stop"
        assert not harness.estopped
    release.set()
    result = agent.final(move_id)
    assert result.status == "cancelled"
    assert h.sent() == apai.LIFT[:6]
    assert h.arm.torque_on()
    assert not {"stop", "hard_estop", "disconnect"} & set(h.spy.calls)


@pytest.mark.apai
def test_a_cut_during_a_move_on_apais_mock_arm_holds_and_nothing_restarts(apai_stack):
    h = apai_stack.h
    agent = up(apai_stack)
    h.new_task()
    move_id, release = apai.held_move(h, 6)
    apai_stack.held = True
    apai_stack.cut()
    lost = agent.ws.wait_result(move_id, timeout_s=WAIT_S)
    assert (lost.status, lost.reason) == ("failed", LOST)
    release.set()
    h.idle()
    assert h.sent() == apai.LIFT[:6]
    assert h.arm.last_target()[0] == apai.LIFT[5]
    assert h.arm.torque_on()
    assert "disconnect" not in h.spy.calls
    apai_stack.held = False
    apai_stack.connected(agent)
    state = agent.ws.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.motion, state.torque_on) == ("nobody", "hold", True)
    h.step()
    h.idle()
    assert h.sent() == apai.LIFT[:6]
    # A new task goes on from where the arm holds.
    h.new_task()
    again = agent.ws.run_move(apai.LIFT[6:], WAIT_S, start_q=apai.LIFT[5])
    agent.final(again.command_id)
    assert again.status == "completed", again.reason
    assert h.sent() == apai.LIFT
