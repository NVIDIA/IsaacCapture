# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentLink: the agent program's client of the robot link service's local API."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest
from link_bench import WAIT_S, Bench, FakeThor, robot_state, until, welcome

from isaaccapture_examples.agentic_remote.agent.link_client import (
    LOST,
    RESULTS_KEPT,
    AgentLink,
    CommandTimeout,
    LinkLost,
    Pinned,
    StaleState,
)
from isaaccapture_examples.agentic_remote.link_service import service
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from isaaccapture_examples.agentic_remote.protocol.messages import (
    CheckerState,
    Gripper,
    JointSample,
    Move,
    OwnerRequest,
    Permit,
    Result,
    RobotState,
    Stop,
)

PIN = Pinned("s1", 1, 0)
# The Thor's wall clock minus its monotonic clock in these tests.
SKEW = 1.7e9
# The Thor's monotonic clock minus the workstation's in these tests.
AHEAD = 500.0


class Now:
    """The workstation's monotonic clock, moved by the test."""

    def __init__(self) -> None:
        self.t = 500.0

    def __call__(self) -> float:
        return self.t


def a_state(
    seq: int = 1,
    mono: float = 1000.0,
    *,
    skew: float = SKEW,
    valid: bool = True,
    **fields: Any,
) -> RobotState:
    """A ROBOT_STATE of session s1 with the agent as owner and one joint."""
    fields = {"owner": "agent", **fields}
    return dataclasses.replace(
        robot_state(seq=seq),
        clock_mono_s=mono,
        clock_wall_s=mono + skew,
        joints=[JointSample(0, 0.25, mono, mono, valid)],
        **fields,
    )


class Rig:
    """An AgentLink on a loopback carrier whose robot side the test scripts."""

    def __init__(self) -> None:
        self.thor = FakeThor("test")
        self.now = Now()
        self.link = AgentLink(self.thor.service_end, monotonic=self.now)
        self._pool = ThreadPoolExecutor(4)

    def arrive(self, state: RobotState, at: float | None = None) -> RobotState:
        """Deliver a ROBOT_STATE while the workstation's clock reads `at`."""
        if at is not None:
            self.now.t = at
        before = self.link._states
        self.thor.send(state)
        until(lambda: self.link._states > before, "the ROBOT_STATE")
        return state

    def result(self, command_id: str, status: str, reason: str = "") -> None:
        before = self.link._changes
        self.thor.send(Result(command_id, status, reason))
        until(lambda: self.link._changes > before, "the RESULT")

    def ready(self) -> AgentLink:
        """WELCOME, one ROBOT_STATE and the pin."""
        self.thor.send(welcome())
        self.arrive(a_state())
        until(lambda: self.link.welcome is not None, "the WELCOME")
        self.link.pin(PIN)
        return self.link

    def later(self, call: Callable[..., Any], *args: Any) -> Future:
        """Run a blocking call on another thread."""
        return self._pool.submit(call, *args)

    def close(self) -> None:
        self.link.close()
        self.thor.carrier.close()
        self._pool.shutdown(wait=False)


@pytest.fixture
def rig() -> Iterator[Rig]:
    made = Rig()
    yield made
    made.close()


@pytest.fixture
def bench(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Bench]:
    made = Bench(tmp_path_factory.mktemp("al"))
    yield made
    made.close()


@pytest.fixture
def attached(bench: Bench) -> Iterator[tuple[FakeThor, AgentLink]]:
    """An AgentLink attached to a real service with a scripted robot side."""
    thor = bench.robot(None)
    thor.send(welcome())
    thor.send(a_state())
    link = AgentLink.attach(bench.path, timeout_s=WAIT_S)
    try:
        link.wait_ready(WAIT_S)
        link.pin(PIN)
        yield thor, link
    finally:
        link.close()


# Pinned numbers


def test_commands_quote_the_pin_although_the_state_has_newer_numbers(rig: Rig) -> None:
    link = rig.ready()
    rig.arrive(a_state(2, owner_number=7, stop_number=3, session_id="s2"))

    link.send_move("m-1", [0.0], 0.02, [[0.1], [0.2]], None, 1.7e9, 5.0)
    link.send_permit("m-1", 1, 1.7e9, 1.0)
    checker_id = link.checker_state("close_grasp")
    jaws = rig.later(link.gripper, 0.5, 0.4, "plain", WAIT_S)

    sent = [rig.thor.next_message() for _ in range(4)]
    assert [type(message) for message in sent] == [Move, Permit, CheckerState, Gripper]
    for message in sent:
        assert (message.session_id, message.owner_number, message.stop_number) == (
            "s1",
            1,
            0,
        )
    move, permit, checker, gripper = sent
    assert (move.command_id, move.setpoints, move.period_s) == (
        "m-1",
        [[0.1], [0.2]],
        0.02,
    )
    assert (permit.move_id, permit.through_index) == ("m-1", 1)
    assert (checker.command_id, checker.action) == (checker_id, "close_grasp")
    assert (gripper.opening, gripper.max_effort, gripper.mode) == (0.5, 0.4, "plain")
    rig.result(gripper.command_id, "accepted")
    rig.result(gripper.command_id, "completed")
    assert jaws.result(WAIT_S).status == "completed"
    assert link.pinned == PIN


def test_a_command_without_a_pin_raises_and_sends_nothing(rig: Rig) -> None:
    link = rig.ready()
    link.unpin()
    with pytest.raises(RuntimeError, match="pinned"):
        link.send_move("m-1", [0.0], 0.02, [[0.1]], None, None, 5.0)
    with pytest.raises(RuntimeError, match="pinned"):
        link.send_permit("m-1", 0, 1.7e9, 1.0)
    with pytest.raises(RuntimeError, match="pinned"):
        link.gripper(0.5, 0.5, "plain", 0.1)
    with pytest.raises(RuntimeError, match="pinned"):
        link.checker_state("close_grasp")
    rig.thor.nothing()


def test_pin_takes_only_a_pinned(rig: Rig) -> None:
    with pytest.raises(TypeError):
        rig.link.pin(("s1", 1, 0))  # type: ignore[arg-type]


def test_move_and_permit_expire_on_the_thors_monotonic_clock(rig: Rig) -> None:
    link = rig.ready()  # the state: Thor 1000.0 at workstation 500.0
    rig.now.t = 502.0
    link.send_move("m-1", [0.0], 0.02, [[0.1]], None, None, 5.0)
    link.send_permit("m-1", 0, 1.7e9 + 1001.5, 1.0)
    move, permit = rig.thor.next_message(), rig.thor.next_message()
    assert move.expires_mono_s == pytest.approx(1007.0)
    assert move.evidence_wall_s is None
    assert permit.expires_mono_s == pytest.approx(1003.0)
    assert permit.evidence_wall_s == 1.7e9 + 1001.5


def test_request_owner_quotes_the_newest_state_or_the_quote_given(rig: Rig) -> None:
    link = rig.ready()
    rig.arrive(a_state(2, owner="nobody", owner_number=4, stop_number=2))

    asked = rig.later(lambda: link.request_owner("agent", timed_return=False))
    request = rig.thor.next_message()
    assert isinstance(request, OwnerRequest)
    assert (request.request, request.timed_return) == ("agent", False)
    assert (request.session_id, request.owner_number, request.stop_number) == (
        "s1",
        4,
        2,
    )
    rig.result(request.command_id, "completed")
    assert asked.result(WAIT_S).status == "completed"
    assert link.final_result(request.command_id) is None  # dropped once returned

    ended = rig.later(
        lambda: link.request_owner("help_needed", reason="x" * 400, quote=PIN)
    )
    request = rig.thor.next_message()
    assert (request.session_id, request.owner_number, request.stop_number) == (
        "s1",
        1,
        0,
    )
    assert request.reason == "x" * 256
    rig.result(request.command_id, "refused", "owner number does not match")
    assert ended.result(WAIT_S).reason == "owner number does not match"


def test_request_owner_needs_a_fresh_state(rig: Rig) -> None:
    link = rig.ready()
    rig.now.t += 1.5
    with pytest.raises(StaleState):
        link.request_owner("agent")
    rig.thor.nothing()


# lost_reason


def test_lost_reason_is_none_while_the_pinned_ownership_stands(rig: Rig) -> None:
    link = rig.ready()
    assert link.lost_reason() is None
    rig.now.t += 0.9
    assert link.lost_reason() is None


@pytest.mark.parametrize(
    ("fields", "names"),
    [
        ({"session_id": "s2"}, "session"),
        ({"owner": "nobody"}, "control owner is nobody"),
        ({"owner_number": 2}, "owner number is 2"),
        ({"stop_number": 1}, "stop number is 1"),
        ({"stop_level": "stop"}, "stop level is stop"),
        ({"stop_level": "fault", "fault_reason": "feedback"}, "fault: feedback"),
    ],
)
def test_lost_reason_names_what_the_state_shows(
    rig: Rig, fields: dict[str, Any], names: str
) -> None:
    link = rig.ready()
    rig.arrive(a_state(2, **fields))
    assert names in link.lost_reason()


def test_lost_reason_without_a_pin_and_with_an_old_state(rig: Rig) -> None:
    link = rig.ready()
    rig.now.t += 1.01
    assert "old" in link.lost_reason()
    link.unpin()
    assert link.lost_reason() == "no ownership is pinned"
    assert link.pinned is None


def test_lost_reason_sees_a_welcome_of_another_session(rig: Rig) -> None:
    link = rig.ready()
    rig.thor.send(welcome("s2"))
    until(lambda: link.welcome.session_id == "s2", "the WELCOME")
    assert "new session (s2)" in link.lost_reason()


def test_a_lost_connection_result_ends_the_pin_for_good(rig: Rig) -> None:
    link = rig.ready()
    command_id = link.checker_state("close_grasp")
    rig.result(command_id, "failed", LOST)
    rig.arrive(a_state(2))  # the robot app is back and shows the same numbers
    assert LOST in link.lost_reason()
    assert link.wait_final(command_id, 0.1).reason == LOST
    link.pin(PIN)  # a new ownership starts clean
    assert link.lost_reason() is None


def test_the_lost_text_is_the_services() -> None:
    assert LOST == service.LOST


# Link loss


def test_a_dead_link_ends_a_wait_with_link_lost(rig: Rig) -> None:
    link = rig.ready()
    command_id = link.checker_state("close_grasp")
    waits = [
        rig.later(link.wait_final, command_id, 30.0),
        rig.later(link.wait_first, command_id, 30.0),
        rig.later(link.gripper, 0.5, 0.5, "plain", 30.0),
    ]
    began = time.monotonic()
    rig.thor.carrier.break_link()
    for wait in waits:
        with pytest.raises(LinkLost):
            wait.result(WAIT_S)
    assert time.monotonic() - began < 2.0
    assert "gone" in link.lost_reason()
    with pytest.raises(LinkLost):
        link.state()
    with pytest.raises(LinkLost):
        link.checker_state("close_grasp")
    with pytest.raises(LinkLost):
        link.send_move("m-1", [0.0], 0.02, [[0.1]], None, None, 5.0)
    with pytest.raises(LinkLost):
        link.wait_ready(0.1)


def test_a_result_that_is_here_is_returned_on_a_dead_link(rig: Rig) -> None:
    link = rig.ready()
    command_id = link.checker_state("close_grasp")
    rig.result(command_id, "completed")
    rig.thor.carrier.break_link()
    assert link.wait_final(command_id, 0.1).status == "completed"


def test_permit_and_stop_never_raise(rig: Rig) -> None:
    link = rig.link
    link.pin(PIN)
    link.send_permit("m-1", 0, 1.7e9, 1.0)  # no ROBOT_STATE yet: no Thor clock
    rig.thor.nothing()
    rig.ready()
    link.send_permit("m-1", 0, float("nan"), 1.0)  # cannot be encoded
    rig.thor.nothing()
    link.stop("r" * 300, "s" * 100)
    stop = rig.thor.next_message()
    assert isinstance(stop, Stop)
    assert (stop.reason, stop.sender) == ("r" * 256, "s" * 64)
    rig.thor.carrier.break_link()
    link.send_permit("m-1", 0, 1.7e9, 1.0)
    link.stop("ctrl+c", "agent_terminal")
    link.close()
    link.stop("ctrl+c", "agent_terminal")


def test_the_robot_connection_going_down_is_link_lost(
    attached: tuple[FakeThor, AgentLink],
) -> None:
    thor, link = attached
    assert link.lost_reason() is None
    command_id = link.checker_state("close_grasp")
    assert isinstance(thor.next_message(), CheckerState)
    wait = threading.Thread(target=thor.carrier.break_link)
    wait.start()
    try:
        result = link.wait_final(command_id, WAIT_S)
    except LinkLost as lost:
        assert LOST in str(lost)
    else:
        assert (result.status, result.reason) == ("failed", LOST)
    wait.join()
    until(lambda: not link.service_status.robot_connected, "the LOCAL_STATUS")
    assert LOST in link.lost_reason()
    with pytest.raises(LinkLost, match=LOST):
        link.state()
    with pytest.raises(LinkLost, match=LOST):
        link.request_owner("help_needed", quote=PIN)
    link.send_permit("m-1", 0, 1.7e9, 1.0)


# The newest ROBOT_STATE


def test_state_is_stale_before_the_first_and_past_the_age_asked(rig: Rig) -> None:
    link = rig.link
    with pytest.raises(StaleState):
        link.state()
    with pytest.raises(StaleState):
        link.state_arrived_s()
    with pytest.raises(StaleState):
        link.thor_now()
    with pytest.raises(StaleState):
        link.wall_to_thor_mono(1.7e9)
    sent = rig.arrive(a_state(), at=510.0)
    assert link.state() == sent
    assert link.state_arrived_s() == 510.0
    rig.now.t = 510.4
    assert link.state(0.5) == sent
    rig.now.t = 510.6
    with pytest.raises(StaleState, match=r"0\.60 s old"):
        link.state(0.5)
    assert link.state() == sent


def test_state_listeners_get_every_state_on_the_receive_thread(rig: Rig) -> None:
    link = rig.link
    seen: list[tuple[int, str]] = []

    def broken(state: RobotState) -> None:
        raise ValueError("a listener's own fault")

    link.add_state_listener(broken)
    link.add_state_listener(
        lambda state: seen.append((state.seq, threading.current_thread().name))
    )
    for seq in (1, 2, 3):
        rig.arrive(a_state(seq))
    until(lambda: len(seen) == 3, "the listener")
    assert [seq for seq, _ in seen] == [1, 2, 3]
    assert {name for _, name in seen} == {"workstation-client"}
    assert link.state().seq == 3


def test_wait_change_returns_at_a_state_a_result_or_the_timeout(rig: Rig) -> None:
    link = rig.ready()
    began = time.monotonic()
    link.wait_change(0.05)
    assert 0.04 < time.monotonic() - began < 1.0

    for send in (
        lambda: rig.thor.send(a_state(2)),
        lambda: rig.thor.send(Result("c-1", "completed")),
        rig.thor.carrier.break_link,
    ):
        began = time.monotonic()
        wait = rig.later(link.wait_change, 30.0)
        time.sleep(0.05)
        send()
        wait.result(WAIT_S)
        assert time.monotonic() - began < 2.0


def test_wait_ready_needs_the_welcome_and_every_joint_valid(rig: Rig) -> None:
    link = rig.link
    rig.arrive(a_state(1))
    with pytest.raises(TimeoutError, match="no WELCOME"):
        link.wait_ready(0.05)
    rig.thor.send(welcome())
    rig.arrive(a_state(2, valid=False))
    with pytest.raises(TimeoutError, match="every joint"):
        link.wait_ready(0.05)
    rig.arrive(a_state(3, session_id="s0"))
    with pytest.raises(TimeoutError):
        link.wait_ready(0.05)
    wait = rig.later(link.wait_ready, WAIT_S)
    rig.arrive(a_state(4))
    assert wait.result(WAIT_S) == welcome()
    assert link.welcome == welcome()


# The Thor's clocks


def test_thor_now_moves_the_newest_clock_pair_to_now(rig: Rig) -> None:
    link = rig.link
    rig.arrive(a_state(1, mono=1000.0), at=500.0)
    rig.now.t = 500.25
    mono, wall = link.thor_now()
    assert mono == pytest.approx(1000.25)
    assert wall == pytest.approx(1000.25 + SKEW)
    assert link.wall_age_s(1000.0 + SKEW) == pytest.approx(0.25 + 0.02, abs=1e-5)
    assert link.wall_to_thor_mono(999.5 + SKEW) == pytest.approx(999.5, abs=1e-5)


def test_one_late_state_does_not_move_the_clock_estimate(rig: Rig) -> None:
    link = rig.link
    local = 500.0
    for seq in range(1, 11):  # each 0.01 s under way
        local += 0.05
        rig.arrive(a_state(seq, mono=local + AHEAD - 0.01), at=local)
    assert link.thor_now()[0] == pytest.approx(local + AHEAD - 0.01)

    local += 0.34
    rig.arrive(a_state(11, mono=local + AHEAD - 0.3), at=local)  # 0.3 s under way
    assert link.thor_now()[0] == pytest.approx(local + AHEAD - 0.01)
    # The wall clock comes from the newest pair, on the same estimate.
    assert link.thor_now()[1] == pytest.approx(local + AHEAD - 0.01 + SKEW)


def test_the_clock_estimate_looks_back_over_64_states(rig: Rig) -> None:
    link = rig.link
    rig.arrive(a_state(1, mono=500.0 + AHEAD - 0.01), at=500.0)
    local = 500.0
    for seq in range(2, 65):  # 63 more, each 0.05 s under way
        local += 0.05
        rig.arrive(a_state(seq, mono=local + AHEAD - 0.05), at=local)
    assert link.thor_now()[0] == pytest.approx(local + AHEAD - 0.01)
    local += 0.05
    rig.arrive(a_state(65, mono=local + AHEAD - 0.05), at=local)
    assert link.thor_now()[0] == pytest.approx(local + AHEAD - 0.05)


@pytest.mark.parametrize(
    "restart",
    [
        {"session_id": "s2", "mono": 1000.0},  # the robot app was started again
        {"mono": 20.0},  # the Thor was started again: its monotonic clock is young
    ],
)
def test_the_clock_estimate_starts_again_with_the_thors_clock(
    rig: Rig, restart: dict[str, Any]
) -> None:
    link = rig.link
    rig.arrive(a_state(1, mono=1000.0), at=500.0)
    mono = restart.pop("mono")
    rig.arrive(a_state(2, mono=mono, **restart), at=501.0)
    assert link.thor_now()[0] == pytest.approx(mono)


def test_a_step_of_the_thors_wall_clock_is_counted_and_listeners_know(rig: Rig) -> None:
    link = rig.link
    seen: list[tuple[int, int]] = []
    link.add_state_listener(lambda state: seen.append((state.seq, link.clock_steps)))
    rig.arrive(a_state(1, mono=1000.0), at=500.0)
    rig.arrive(a_state(2, mono=1000.05, skew=SKEW + 0.015), at=500.05)
    assert link.clock_steps == 0
    assert link.clock_step_at_s() is None

    rig.arrive(a_state(3, mono=1000.1, skew=SKEW + 0.5), at=500.1)
    assert link.clock_steps == 1
    assert link.clock_step_at_s() == 500.1
    rig.arrive(a_state(4, mono=1000.15, skew=SKEW + 0.5), at=500.15)
    rig.arrive(a_state(5, mono=1000.2, skew=SKEW - 2.0), at=500.2)
    assert link.clock_steps == 2
    until(lambda: len(seen) == 5, "the listener")
    assert seen == [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2)]
    # The conversion follows the newest pair.
    assert link.wall_to_thor_mono(1000.0 + SKEW - 2.0) == pytest.approx(1000.0)


# RESULTs


def test_wait_first_takes_accepted_and_wait_final_the_final_result(rig: Rig) -> None:
    link = rig.ready()
    link.send_move("m-1", [0.0], 0.02, [[0.1]], None, None, 5.0)
    assert link.final_result("m-1") is None
    with pytest.raises(CommandTimeout, match="m-1") as timeout:
        link.wait_first("m-1", 0.05)
    assert timeout.value.command_id == "m-1"

    rig.result("m-1", "accepted")
    assert link.wait_first("m-1", 0.1).status == "accepted"
    assert link.final_result("m-1") is None
    with pytest.raises(CommandTimeout):
        link.wait_final("m-1", 0.05)

    wait = rig.later(link.wait_final, "m-1", WAIT_S)
    rig.result("m-1", "completed", "done")
    assert wait.result(WAIT_S).reason == "done"
    assert link.wait_first("m-1", 0.1).status == "accepted"
    assert link.final_result("m-1").status == "completed"

    link.forget("m-1")
    assert link.final_result("m-1") is None
    link.forget("m-1")  # twice is fine


def test_a_refusal_is_the_first_and_the_final_result(rig: Rig) -> None:
    link = rig.ready()
    rig.result("m-2", "refused", "start pose")
    assert link.wait_first("m-2", 0.1) == link.wait_final("m-2", 0.1)
    assert link.final_result("m-2").status == "refused"


def test_results_are_kept_for_the_newest_256_commands(rig: Rig) -> None:
    link = rig.ready()
    count = RESULTS_KEPT + 44
    for index in range(count):
        rig.thor.send(Result(f"c-{index}", "completed"))
    until(lambda: link.final_result(f"c-{count - 1}") is not None, "the last RESULT")
    assert RESULTS_KEPT == 256
    assert len(link._results) == RESULTS_KEPT
    assert link.final_result("c-43") is None
    assert link.final_result("c-44") is not None
    # The wrapped client keeps none of them.
    until(lambda: not link._client._results, "the client's store")


def test_wait_reflected_takes_the_state_that_names_the_command(rig: Rig) -> None:
    link = rig.ready()
    rig.arrive(a_state(2, last_result={"command_id": "m-1"}))  # ahead of the RESULT
    assert link.wait_reflected("m-1", 0.05) is None  # no final RESULT yet
    rig.result("m-1", "completed")
    assert link.wait_reflected("m-1", 0.05) is None  # no state after it yet

    wait = rig.later(link.wait_reflected, "m-1", WAIT_S)
    named = rig.arrive(a_state(3, last_result={"command_id": "m-1", "status": "x"}))
    assert wait.result(WAIT_S) == named


def test_wait_reflected_takes_the_second_state_after_the_result(rig: Rig) -> None:
    link = rig.ready()
    rig.result("m-1", "completed")
    rig.arrive(a_state(2, last_result={"command_id": "other"}))
    assert link.wait_reflected("m-1", 0.05) is None
    second = rig.arrive(a_state(3, last_result=None))
    assert link.wait_reflected("m-1", 0.05) == second
    rig.thor.carrier.break_link()
    assert link.wait_reflected("m-9", 5.0) is None  # a lost link ends the wait


# On the robot link service


def test_attach_takes_the_agent_role_and_only_one_at_a_time(bench: Bench) -> None:
    thor = bench.robot(None)
    link = AgentLink.attach(bench.path, timeout_s=WAIT_S)
    try:
        assert bench.service.status().roles == {"agent": 1}
        with pytest.raises(LinkLost, match="an agent client is attached already"):
            AgentLink.attach(bench.path, timeout_s=WAIT_S)
        thor.send(welcome())
        thor.send(a_state())
        assert link.wait_ready(WAIT_S).session_id == "s1"
        assert link.new_id().startswith("agent-")
    finally:
        link.close()
    until(lambda: not bench.service.status().roles, "the service to see the close")


def test_attach_with_nothing_listening_is_link_lost(tmp_path: Any) -> None:
    began = time.monotonic()
    with pytest.raises(LinkLost, match="cannot attach"):
        AgentLink.attach(str(tmp_path / "none.sock"), timeout_s=WAIT_S)
    assert time.monotonic() - began < 1.0


def test_wait_ready_waits_for_the_robot_connection(bench: Bench) -> None:
    link = AgentLink.attach(bench.path, timeout_s=WAIT_S)
    try:
        with pytest.raises(LinkLost, match="not connected"):
            link.wait_ready(0.1)
        wait_done: list[Any] = []
        wait = threading.Thread(
            target=lambda: wait_done.append(link.wait_ready(WAIT_S))
        )
        wait.start()
        thor = bench.robot(None)
        thor.send(welcome())
        thor.send(a_state())
        wait.join(WAIT_S)
        assert wait_done == [welcome()]
    finally:
        link.close()


def test_what_the_agent_role_may_not_send_comes_back_refused(
    attached: tuple[FakeThor, AgentLink],
) -> None:
    thor, link = attached
    result = link.request_owner("rest")
    assert result.status == "refused"
    assert "the agent role may not send" in result.reason
    thor.nothing()


def test_commands_reach_the_robot_through_the_service(
    attached: tuple[FakeThor, AgentLink],
) -> None:
    thor, link = attached
    link.send_move("m-1", [0.0], 0.02, [[0.1]], None, None, 5.0)
    link.send_permit("m-1", 0, 1.7e9 + 1000.0, 1.0)
    link.stop("ctrl+c", "agent_terminal")
    move, permit, stop = (thor.next_message() for _ in range(3))
    assert isinstance(move, Move)
    assert isinstance(permit, Permit)
    assert (stop.reason, stop.sender) == ("ctrl+c", "agent_terminal")
    thor.send(Result("m-1", "cancelled", "STOP"))
    assert link.wait_final("m-1", WAIT_S).status == "cancelled"
    # The service now waits for a higher stop number; the ownership is over.
    until(lambda: link.lost_reason() is not None, "the STOP to show")
    assert "STOP" in link.lost_reason()


# On the robot app


def test_an_old_pin_is_refused_by_the_robot_app_and_never_repaired(
    make_robot_app: Callable[..., Any],
) -> None:
    harness = make_robot_app(connected=False)
    robot_end, workstation_end = loopback_pair()
    link = AgentLink(workstation_end, monotonic=harness.clock.monotonic)
    try:
        assert harness.app.attach_carrier(robot_end)
        link.wait_ready(WAIT_S)
        granted = link.request_owner("agent", timed_return=False)
        assert granted.status == "completed", granted.reason
        number = granted.detail["owner_number"]
        until(
            lambda: (
                link.state().owner == "agent" and link.state().owner_number == number
            ),
            "the agent as owner",
        )
        state = link.state()
        link.pin(Pinned(state.session_id, state.owner_number, state.stop_number))
        assert link.lost_reason() is None
        done = link.wait_final(link.checker_state("clear_contact_spots"), WAIT_S)
        assert done.status == "completed", done.reason

        link.stop("pressed", "agent_terminal")
        until(lambda: link.lost_reason() is not None, "the STOP to show")
        newer = link.state()
        assert newer.stop_number > link.pinned.stop_number
        refused = link.wait_final(link.checker_state("clear_contact_spots"), WAIT_S)
        assert refused.status == "refused"
        assert link.pinned.stop_number == state.stop_number
    finally:
        link.close()
        robot_end.close()
