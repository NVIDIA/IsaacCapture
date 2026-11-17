# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator's commands through the robot app: rest, release, shutdown, ownership.

A command written by hand here carries only the fields the test names, as a
console that does not use the protocol module could send it.
"""

from __future__ import annotations

import json
import threading

import pytest
from isaaccapture_examples.agentic_remote.agent.console_cmd import (
    ASK_CLOSE,
    JAWS_NOT_CONFIRMED,
)
from isaaccapture_examples.agentic_remote.agent.console_cmd import (
    ASK_JAWS_EMPTY as ASK_JAWS,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    CLOSE_JAWS,
    OPEN_ACTIONS,
    GripperState,
    Open,
    OwnerRequest,
    Release,
    Reset,
    Result,
    Shutdown,
)
from isaaccapture_examples.agentic_remote.robot_app.app import (
    CONFIRM_CLOSE,
    CONFIRM_JAWS_EMPTY,
    CONFIRM_OPEN,
    CONFIRM_PATH,
    EVENT_JAWS,
    JAWS_EMPTY_WORD,
    PATH_CLEAR_PROMPT,
    REFERENCE_OFF,
)
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    CLOSE_UNTIL_CONTACT,
    JAWS_BUSY,
    OPEN_EFFORT,
    PLAIN,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    GripperOutcome,
    MotionState,
    Owner,
    ReadFailed,
)
from isaaccapture_examples.agentic_remote.robot_app.terminal import (
    ASK_JAWS_EMPTY,
    CONFIRM_REFERENCE,
    CONFIRM_REST,
)
from isaaccapture_examples.agentic_remote.robot_app.terminal import (
    CONFIRM_OPEN as CONFIRM_OPEN_ASKED,
)

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# 60 setpoints at the agent's speed limit: far enough for a contact to show.
LONG_PATH = [[0.008 * (index + 1)] * 6 for index in range(60)]
# A pose the agent may hold; the return to rest from it has 209 setpoints.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
RETURN_SETPOINTS = 209


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


def statuses(h, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def final(h, command_id: str) -> Result:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


def send_raw(h, kind: type, command_id: str, **fields) -> str:
    """Send a command as hand-written JSON with these fields and no others."""
    obj = {"command_id": command_id, "session_id": h.app.session_id, **fields}
    h.workstation_end.send_reliable(kind.TYPE, json.dumps(obj).encode())
    return command_id


def numbers(h) -> dict:
    """The owner number and stop number as the robot app has them now."""
    state = h.app.snapshot()
    return {"owner_number": state.owner_number, "stop_number": state.stop_number}


def ask_for_help(h, reason: str = "the agent is stuck") -> Result:
    """OWNER_REQUEST "help_needed", answered: the assistance wait runs."""
    result = final(h, h.client.request_owner("help_needed", reason=reason))
    assert result.status == "completed", result.reason
    return result


def settle(h) -> None:
    """Motion work is over and the workstation has every EVENT sent so far."""
    h.idle()
    h.step()
    heard = len(h.local_events)
    if h.app.link_alive:
        h.client.wait_until(lambda: len(h.client.events) >= heard, WAIT_S, "EVENTs")


def rest_texts(h) -> list[str]:
    """The texts of the rest EVENTs heard at the Thor terminal so far."""
    return [text for _, text, _ in h.heard("rest")]


# Every operator command quotes the owner number


@pytest.mark.parametrize(
    ("kind", "fields", "moved_first"),
    [
        (OwnerRequest, {"request": "rest", "confirmed": True}, True),
        (Shutdown, {"confirmed": True}, True),
        (Release, {}, False),
    ],
    ids=["rest", "shutdown", "release"],
)
def test_a_command_built_before_the_agent_took_the_arm_is_refused(
    robot_app, kind, fields, moved_first
):
    h = robot_app
    seen = numbers(h)  # what the console shows: nobody, owner number 0
    h.new_task()
    if moved_first:
        assert h.client.run_move(PATH, start_q=REST).status == "completed"
    sent = len(h.arm.sent)
    command_id = send_raw(h, kind, "built-earlier", **seen, **fields)
    result = final(h, command_id)
    h.idle()
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == "owner number 0 does not match the current 1"
    assert len(h.arm.sent) == sent
    assert h.arm.torque_on() and h.arm.released_at == [] and h.app.running
    state = h.app.snapshot()
    assert (state.owner, state.owner_number) == (Owner.AGENT, 1)


def test_an_open_built_before_the_jaws_took_an_object_is_refused(robot_app):
    h = robot_app
    seen = numbers(h)  # nobody, owner number 0, jaws empty
    h.new_task()
    h.arm.jaw_object(0.4)
    grasp = final(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert grasp.status == "completed", grasp.reason
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    jaw_commands = len(h.arm.jaw_commands)
    command_id = send_raw(h, Open, "built-earlier", **seen, confirmed=True)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == "owner number 0 does not match the current 2"
    assert len(h.arm.jaw_commands) == jaw_commands
    assert h.arm.gripper_read() == pytest.approx(0.4)  # the object is still held


def test_a_reset_quoting_an_old_owner_number_clears_nothing(robot_app):
    h = robot_app
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    h.arm.clear_faults()
    now = numbers(h)
    assert now == {"owner_number": 1, "stop_number": 1}
    command_id = send_raw(h, Reset, "old-owner", owner_number=0, stop_number=1)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == "owner number 0 does not match the current 1"
    assert h.app.snapshot().stop_level == "fault"
    assert all(checker.latched() for checker in h.checkers.all())
    # With the current numbers the latch is cleared.
    assert final(h, send_raw(h, Reset, "current", **now)).status == "completed"
    assert h.app.snapshot().stop_level == "none"


@pytest.mark.parametrize(
    ("request_kind", "given_back", "current"),
    [("agent", True, 2), ("help_needed", False, 1), ("nobody", False, 1)],
)
def test_an_owner_request_quoting_an_old_owner_number_is_refused(
    robot_app, request_kind, given_back, current
):
    h = robot_app
    seen = numbers(h)  # nobody, owner number 0
    h.new_task()
    if given_back:
        assert final(h, h.client.request_owner("nobody")).status == "completed"
    before = h.app.snapshot()
    command_id = send_raw(
        h, OwnerRequest, "built-earlier", **seen, request=request_kind
    )
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == f"owner number 0 does not match the current {current}"
    assert h.app.snapshot() == before
    assert not h.app._wait.active()


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        (OwnerRequest, {"request": "rest", "confirmed": True}),
        (Reset, {}),
        (Release, {}),
        (Open, {"confirmed": True}),
        (Shutdown, {"confirmed": True}),
    ],
    ids=["owner request", "reset", "release", "open", "shutdown"],
)
def test_an_operator_command_without_an_owner_number_is_refused(
    robot_app, kind, fields
):
    h = robot_app
    command_id = send_raw(h, kind, "no-owner-number", stop_number=0, **fields)
    result = final(h, command_id)
    h.idle()
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == f"{kind.__name__}: missing field 'owner_number'"
    assert h.arm.torque_on() and h.arm.jaw_commands == [] and h.app.running


def test_a_release_quoting_the_current_numbers_is_taken(robot_app):
    h = robot_app
    h.new_task()
    command_id = send_raw(h, Release, "current", **numbers(h))
    assert final(h, command_id).status == "completed"
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert not h.arm.torque_on()


# A shutdown that has to move the arm needs the path confirmed

QUIT_UNCONFIRMED = (
    "quit: refused: confirm the path is clear; the robot app keeps running"
)
QUIT_DONE = "quit: completed; the robot app closes"


def raised_by_the_agent(h) -> int:
    """The agent moves the arm away from rest and gives it back; setpoints sent."""
    h.new_task()
    assert h.client.run_move(PATH, start_q=REST).status == "completed"
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    assert h.app.motion_state is MotionState.HOLD
    return len(h.arm.sent)


def assert_holding_away_from_rest(h, sent: int) -> None:
    h.idle()
    assert len(h.arm.sent) == sent
    assert h.arm.pose() == pytest.approx(PATH[-1], abs=0.01)
    assert h.arm.torque_on() and h.arm.released_at == [] and h.app.running


@pytest.mark.parametrize(
    "fields", [{}, {"confirmed": False}], ids=["no confirmed field", "confirmed false"]
)
def test_a_shutdown_that_has_to_move_the_arm_is_refused_unconfirmed(robot_app, fields):
    h = robot_app
    sent = raised_by_the_agent(h)
    command_id = send_raw(h, Shutdown, "unconfirmed", **numbers(h), **fields)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == CONFIRM_PATH == "confirm the path is clear"
    assert_holding_away_from_rest(h, sent)
    assert h.heard("refused")[-1][1] == "shutdown refused: confirm the path is clear"


def test_a_confirmed_shutdown_moves_the_arm_to_rest_and_releases_there(robot_app):
    h = robot_app
    sent = raised_by_the_agent(h)
    command_id = send_raw(h, Shutdown, "confirmed", **numbers(h), confirmed=True)
    assert final(h, command_id).status == "completed"
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.sent) > sent
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)


@pytest.mark.parametrize("motors", ["on", "off"])
def test_a_shutdown_that_moves_nothing_needs_no_confirmation(robot_app, motors):
    h = robot_app
    if motors == "off":
        assert final(h, h.client.release()).status == "completed"
    command_id = send_raw(h, Shutdown, "at-rest", **numbers(h))
    assert final(h, command_id).status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert h.arm.sent == [] and len(h.arm.released_at) == 1


def test_an_unconfirmed_shutdown_moves_nothing_when_the_arm_is_found_off_rest(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)  # moved since the app last measured it at rest
    assert h.app.motion_state is MotionState.REST_HOLD
    command_id = send_raw(h, Shutdown, "unconfirmed", **numbers(h))
    result = final(h, command_id)
    h.idle()
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert result.reason == (
        "the return to rest ended refused: confirm the path is clear"
    )
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.arm.torque_on() and h.arm.released_at == [] and h.app.running
    # Confirmed, the same shutdown moves the arm from where it was found.
    assert final(h, h.client.shutdown()).status == "completed"
    assert len(h.arm.sent) == RETURN_SETPOINTS


def test_the_clients_shutdown_says_whether_the_path_was_confirmed(robot_app):
    h = robot_app
    sent = raised_by_the_agent(h)
    refused = final(h, h.client.shutdown(confirmed=False))
    assert (refused.status, refused.reason) == ("refused", CONFIRM_PATH)
    assert_holding_away_from_rest(h, sent)
    assert final(h, h.client.shutdown()).status == "completed"
    assert h.app.wait_closed(WAIT_S)


def test_a_shutdown_at_the_thor_terminal_needs_the_same_confirmation(make_robot_app):
    asked = []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return len(asked) > 1  # no the first time, yes after that

    h = make_robot_app(confirm=confirm)
    sent = raised_by_the_agent(h)
    done = Done()
    assert h.app.request_shutdown(False, done) == CONFIRM_PATH
    assert asked == []  # the terminal said no itself
    assert done.wait().status == "refused"
    done = Done()
    assert h.app.request_shutdown(None, done) == CONFIRM_PATH
    assert asked == [PATH_CLEAR_PROMPT]
    assert (done.wait().status, done.results[0].reason) == ("refused", CONFIRM_PATH)
    assert_holding_away_from_rest(h, sent)
    done = Done()
    assert h.app.request_shutdown(None, done) is None
    assert done.wait().status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert len(asked) == 2 and len(h.arm.released_at) == 1


@pytest.mark.parametrize("confirmed", [None, False])
def test_a_shutdown_at_the_thor_terminal_at_rest_asks_nothing(
    make_robot_app, confirmed
):
    asked = []
    h = make_robot_app(confirm=lambda prompt: bool(asked.append(prompt)))
    assert not h.app.shutdown_moves_arm()
    done = Done()
    assert h.app.request_shutdown(confirmed, done) is None
    assert done.wait().status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert asked == [] and h.arm.sent == [] and len(h.arm.released_at) == 1


def test_quit_typed_away_from_rest_moves_the_arm_only_after_the_typed_phrase(
    robot_app,
):
    h = robot_app
    sent = raised_by_the_agent(h)
    thor = h.thor_terminal()
    for answer in ("no", "", "yes", "path clear"):
        assert thor.ask("quit", answer, answer="quit: ") == QUIT_UNCONFIRMED
    assert len(thor.written(CONFIRM_REST)) == 4
    assert_holding_away_from_rest(h, sent)
    assert thor.ask("quit", " Path is  clear ", answer="quit: ") == QUIT_DONE
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.sent) > sent
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)


def test_stop_typed_where_quit_asks_is_a_stop_and_confirms_nothing(robot_app):
    h = robot_app
    sent = raised_by_the_agent(h)
    thor = h.thor_terminal()
    assert thor.ask("quit", "stop", answer="quit: ") == QUIT_UNCONFIRMED
    assert len(thor.written("[stop] STOP from thor_terminal: typed stop")) == 1
    assert h.app.snapshot().stop_number == 1
    assert_holding_away_from_rest(h, sent)


@pytest.mark.parametrize("motors", ["on", "off"])
def test_quit_typed_when_the_arm_need_not_move_asks_nothing(robot_app, motors):
    h = robot_app
    thor = h.thor_terminal()
    if motors == "off":
        assert thor.ask("release", answer="release: ") == "release: completed"
    assert thor.ask("quit", answer="quit: ") == QUIT_DONE
    assert h.app.wait_closed(WAIT_S)
    assert thor.written(CONFIRM_REST) == []
    assert h.arm.sent == [] and len(h.arm.released_at) == 1


def test_quit_asks_and_works_with_the_robot_connection_down(robot_app):
    h = robot_app
    sent = raised_by_the_agent(h)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    thor = h.thor_terminal()
    assert thor.ask("quit", "path is clear", answer="quit: ") == QUIT_DONE
    assert thor.written(CONFIRM_REST) == [CONFIRM_REST]
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.sent) > sent and len(h.arm.released_at) == 1


# The timed return is tried when the wait runs out, not after other motion work

HELD_AT = 60  # setpoints of the operator's return that are out when the wait ends
WAIT_LEFT_S = 0.7  # of the assistance wait when the operator asks


def waiting_raised(h) -> None:
    """A raised arm, 0.7 s before its assistance wait runs out.

    A return to rest uses 0.4 s of the app clock for its checks and 1.6 s until
    60 setpoints are out, and the wait counts at most 1 s per dispatcher pass:
    it runs out during the return's move, however the threads take turns.
    """
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    for margin_s in (2.0, WAIT_LEFT_S):  # the idle joint reads use clock time too
        h.advance(h.app._wait.remaining(h.clock.monotonic()) - margin_s)
    assert h.heard("expired") == []
    left = h.app._wait.remaining(h.clock.monotonic())
    assert WAIT_LEFT_S - 0.1 < left <= WAIT_LEFT_S


def wait_runs_out(h, reached: threading.Event) -> None:
    """The wait runs out while the motion worker is held in the work it runs.

    That work's own joint reads and setpoints took the clock past the wait's end.
    """
    assert reached.wait(WAIT_S), "the operator's work did not reach its hold"
    for _ in range(200):
        if h.heard("expired"):
            return
        h.step()
    raise AssertionError("the wait did not run out")


def make_the_move_late(h) -> None:
    """The next joint read takes 150 ms longer, as when the bus stalls."""
    real = h.arm.read_joint
    slowed = []

    def read_joint(joint: int):
        if not slowed:
            slowed.append(joint)
            h.clock.advance(0.15)
        return real(joint)

    h.arm.read_joint = read_joint


@pytest.mark.parametrize(
    "asked",
    [
        "rest at the Thor terminal",
        "rest at the console",
        "quit at the Thor terminal",
        "shutdown at the console",
    ],
)
def test_a_wait_that_runs_out_during_an_operators_return_moves_nothing_after_it(
    robot_app, asked
):
    h = robot_app
    waiting_raised(h)
    thor = h.thor_terminal()
    reached, release = h.hold_move_after(HELD_AT)
    done = Done()
    command_id = ""
    if asked == "rest at the Thor terminal":
        assert h.app.request_rest(True, done) is None
    elif asked == "rest at the console":
        command_id = h.client.rest()
    elif asked == "quit at the Thor terminal":
        assert h.app.request_shutdown(True, done) is None
    else:
        command_id = h.client.shutdown()
    work = "a shutdown" if "rest" not in asked else "a return to rest"
    wait_runs_out(h, reached)
    # The one attempt is made, and refused, at the moment the bells announced.
    refused = f"no timed return to rest: busy: {work} is running"
    assert rest_texts(h)[-1] == refused
    assert h.heard("rest")[-1][2] == {
        "source": "wait",
        "status": "refused",
        "reason": f"busy: {work} is running",
    }
    assert [job.what for job in h.app._jobs] == []
    make_the_move_late(h)
    release.set()
    result = final(h, command_id) if command_id else done.wait()
    assert result.status == "failed" and "late after 60 of 209" in result.reason
    settle(h)
    h.advance(5.0)  # nothing follows later either
    settle(h)
    assert len(h.arm.sent) == HELD_AT
    assert max(abs(at) for at in h.arm.pose()) > h.config.rest.rest_tol_rad
    assert h.arm.torque_on() and h.arm.released_at == [] and h.app.running
    assert len([text for text in rest_texts(h) if "the arm moves for" in text]) == 1
    # Both terminals were told, and the wait is over: it made its one attempt.
    assert h.client.wait_event("rest", lambda event: event.text == refused)
    thor.terminal.flush()
    assert thor.written(f"[rest] {refused}") == [f"[rest] {refused}"]
    assert not h.app._wait.active()


def test_a_wait_that_runs_out_behind_queued_motion_work_queues_nothing(robot_app):
    h = robot_app
    waiting_raised(h)
    done = Done()
    with h.app._lock:  # the motion worker cannot begin the return
        assert h.app.request_rest(True, done) is None
        h.clock.advance(1.5)
        h.app._step_wait()
        assert len(h.heard("expired")) == 1
        assert [job.what for job in h.app._jobs] == ["a return to rest"]
    assert rest_texts(h)[0] == (
        "no timed return to rest: busy: a return to rest is queued"
    )
    assert done.wait().status == "completed"
    settle(h)
    assert len(h.arm.sent) == RETURN_SETPOINTS  # one return, the operator's


def test_a_wait_that_runs_out_with_nothing_else_under_way_still_returns_the_arm(
    robot_app,
):
    h = robot_app
    waiting_raised(h)
    h.advance_until(lambda: h.heard("expired"))
    settle(h)
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert rest_texts(h)[-1] == "return to rest: the arm is at rest, motors on"
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_jaw_command_is_still_refused_while_a_timed_return_is_queued(robot_app):
    h = robot_app
    waiting_raised(h)
    with h.app._lock:  # the motion worker cannot begin the timed return
        h.clock.advance(1.5)
        h.app._step_wait()
        assert h.app._busy() == "busy: the timed return to rest is queued"
        h.app._handle(Open("open-1", h.app.session_id, **numbers(h), confirmed=True))
        stored = h.app._results.get("open-1")
    assert (stored.status, stored.reason) == (
        "refused",
        "busy: the timed return to rest is queued",
    )
    settle(h)
    assert h.arm.jaw_commands == [] and len(h.arm.sent) == RETURN_SETPOINTS


# A release or a shutdown that came over the link does not outlive the link

# The full joint read at which the motion worker is held. A release reads twice.
# A shutdown at rest reads twice for the return's conditions, twice to prove
# rest once recovery has the arm, and twice for its release.
LINK_WORK = [("release", 1), ("release", 2), ("shutdown", 3), ("shutdown", 5)]


@pytest.mark.parametrize(("command", "held_read"), LINK_WORK)
def test_connection_lost_while_a_link_release_is_checked_keeps_the_motors_on(
    robot_app, command, held_read
):
    h = robot_app  # at rest, jaws proven empty
    reached, release = h.hold_call("read_all", held_read)
    command_id = getattr(h.client, command)()
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    assert "the arm holds" in h.wait_local_event("connection")[1]
    release.set()
    h.idle()
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.heard("release") == []
    stored = h.app._results.get(command_id)
    if command == "release":
        assert (stored.status, stored.reason) == ("refused", "robot connection lost")
        assert h.heard("refused")[-1][1] == (
            "release refused: robot connection lost; the motors stay on"
        )
    else:
        assert (stored.status, stored.reason) == (
            "failed",
            "at rest with the motors on; release refused: robot connection lost",
        )
        assert h.heard("shutdown")[-1][1] == (
            f"shutdown failed: {stored.reason}; the robot app keeps running and "
            "the arm holds"
        )


@pytest.mark.parametrize(("command", "held_read"), LINK_WORK)
def test_connection_lost_lets_a_release_typed_at_the_thor_terminal_go_on(
    robot_app, command, held_read
):
    h = robot_app
    reached, release = h.hold_call("read_all", held_read)
    done = Done()
    if command == "release":
        assert h.app.request_release(done) is None
    else:
        assert h.app.request_shutdown(False, done) is None
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    assert done.wait().status == "completed"
    h.idle()
    assert not h.arm.torque_on() and len(h.arm.released_at) == 1
    assert h.app.running is (command == "release")


def test_closing_the_app_while_a_release_is_checked_keeps_the_motors_on(robot_app):
    h = robot_app
    reached, release = h.hold_call("read_all")
    done = Done()
    assert h.app.request_release(done) is None
    assert reached.wait(WAIT_S)
    closing = threading.Thread(target=h.app.close, daemon=True)
    closing.start()
    assert h.app._closing.wait(WAIT_S)
    h.app._lock.acquire()  # close() has told the release to stop
    h.app._lock.release()
    release.set()
    result = done.wait()
    closing.join(WAIT_S)
    assert (result.status, result.reason) == ("refused", "the robot app is closing")
    assert h.arm.torque_on() and h.arm.released_at == []


# A return that ends once recovery has the arm is "failed", never "refused"

START_READ_FAILED = (
    "refused after 0 of 209 setpoints: the start pose could not be read: "
    "joint index 2: the read failed"
)


def fail_full_read(h, call: int) -> None:
    """Make the arm's full joint read number `call`, counted from now, fail."""
    real = h.arm.read_all
    calls = []

    def read_all():
        calls.append(1)
        if len(calls) == call:
            raise ReadFailed("joint index 2: the read failed")
        return real()

    h.arm.read_all = read_all


def ask_for_rest(h, asked: str) -> Result:
    """A confirmed rest or shutdown at the console or the Thor terminal; its end."""
    if asked == "rest at the console":
        return final(h, h.client.rest())
    if asked == "shutdown at the console":
        return final(h, h.client.shutdown())
    done = Done()
    assert h.app.request_rest(True, done) is None
    return done.wait()


@pytest.mark.parametrize(
    "asked",
    ["rest at the console", "rest at the Thor terminal", "shutdown at the console"],
)
def test_a_return_that_ends_once_recovery_has_the_arm_is_failed_not_refused(
    robot_app, asked
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()  # the agent owns the arm and holds
    before = h.app.snapshot()
    # Reads 1 and 2 are the return's conditions; the third is the move's own
    # read of its start pose, taken once recovery owns the arm.
    fail_full_read(h, 3)
    result = ask_for_rest(h, asked)
    h.idle()
    assert result.status == "failed"
    assert result.reason.endswith(START_READ_FAILED)
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.arm.torque_on() and h.arm.released_at == [] and h.app.running
    # The agent's ownership ended, as after every return that had begun.
    after = h.app.snapshot()
    assert (before.owner, before.owner_number) == (Owner.AGENT, 1)
    assert (after.owner, after.owner_number) == (Owner.NOBODY, 3)
    assert rest_texts(h)[-1] == (
        f"return to rest failed: {START_READ_FAILED}; the arm holds"
    )
    if h.app.link_alive:
        old = h.client.run_move([RAISED], start_q=RAISED, owner_number=1)
        assert (old.status, old.reason) == (
            "refused",
            "control owner is nobody, not agent",
        )


@pytest.mark.parametrize("failing_read", [1, 2])
def test_a_return_refused_on_its_conditions_leaves_the_agent_the_owner(
    robot_app, failing_read
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    before = h.app.snapshot()
    fail_full_read(h, failing_read)
    result = final(h, h.client.rest())
    h.idle()
    assert result.status == "refused"
    assert result.reason == "a joint read failed: joint index 2: the read failed"
    assert h.app.snapshot() == before and h.arm.sent == []
    assert h.client.run_move([RAISED], start_q=RAISED).status == "completed"


# A rest typed at the Thor terminal goes on when the agent's ownership ends


def end_the_agents_ownership(h, change: str) -> None:
    if change == "the robot connection is lost":
        h.robot_end.break_link()
        h.wait_local_event("connection")
    else:
        ask_for_help(h)
    state = h.app.snapshot()
    assert (state.owner, state.owner_number) == (Owner.NOBODY, 2)


@pytest.mark.parametrize(
    "change", ["the robot connection is lost", "the agent asks for help"]
)
@pytest.mark.parametrize("held_read", [1, 2])
def test_a_rest_typed_at_the_thor_terminal_goes_on_when_the_agents_ownership_ends(
    robot_app, held_read, change
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()  # the agent owns the arm, owner number 1
    reached, release = h.hold_call("read_all", held_read)  # the return's checks
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert reached.wait(WAIT_S)
    end_the_agents_ownership(h, change)
    release.set()
    result = done.wait()
    h.idle()
    assert result.status == "completed", result.reason
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.arm.torque_on() and h.arm.released_at == []
    # Recovery took the arm from nobody, with the numbers of that moment.
    after = h.app.snapshot()
    assert (after.owner, after.owner_number, after.stop_number) == (Owner.NOBODY, 4, 0)
    assert h.app._wait.active() is (change == "the agent asks for help")


@pytest.mark.parametrize(
    "change", ["a person takes the arm", "a STOP", "a fault-stop", "a new task"]
)
def test_a_typed_rest_is_still_refused_when_anything_else_changes_meanwhile(
    robot_app, change
):
    h = robot_app
    h.arm.set_pose(RAISED)
    if change != "a new task":
        h.new_task()
    reached, release = h.hold_call("read_all")
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert reached.wait(WAIT_S)
    if change == "a person takes the arm":
        h.app._authority.set_owner(Owner.HUMAN, "takeover")
        reason = "the control owner changed while the return to rest was checked"
    elif change == "a STOP":
        h.app.stop("typed stop")
        reason = "a STOP was raised while the return to rest was checked"
    elif change == "a fault-stop":
        h.app._fault("motor 3 overheated")
        latched = "fault-stop latched: motor 3 overheated"
        reason = f"the path to rest was refused: {latched}"
    else:
        h.app._authority.set_owner(Owner.AGENT, "new task")
        reason = "the control owner changed while the return to rest was checked"
    owner = h.app.snapshot().owner
    release.set()
    result = done.wait()
    h.idle()
    assert (result.status, result.reason) == ("refused", reason)
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.app.snapshot().owner is owner  # recovery never took the arm
    assert h.arm.torque_on() and h.arm.released_at == []


@pytest.mark.parametrize("owner", ["nobody", "the agent"])
def test_a_rest_sent_over_the_link_announces_no_move_once_the_link_is_lost(
    robot_app, owner
):
    h = robot_app
    h.arm.set_pose(RAISED)
    if owner == "the agent":
        h.new_task()
    reached, release = h.hold_call("read_all")
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    h.idle()
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.arm.torque_on() and h.arm.released_at == []
    assert [text for text in rest_texts(h) if "the arm moves for" in text] == []
    assert h.app.snapshot().owner is Owner.NOBODY
    stored = h.app._results.get(rest_id)
    assert (stored.status, stored.reason) == (
        "cancelled",
        "robot connection lost (after 0 of 209 setpoints)",
    )


# A contact hold ends when the arm is next moved

AT_A_CONTACT = "the arm is holding at a contact"


def contact(h) -> list[float]:
    """An agent move meets something at joint index 0; the pose the arm holds at."""
    h.arm.block_joint(0, 0.05)
    result = h.client.run_move(LONG_PATH, start_q=REST)
    h.arm.block_joint(0, None)
    assert (result.status, result.reason) == ("failed", "contact")
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == ("refused", AT_A_CONTACT)
    return result.end_q


def retreat(start: list[float]) -> list[list[float]]:
    """40 setpoints that take every joint 0.16 rad back from `start`."""
    return [[q - 0.004 * (index + 1) for q in start] for index in range(40)]


@pytest.mark.parametrize("ended_by", ["its permits ran out", "help_needed", "a STOP"])
def test_a_contact_hold_ends_when_a_move_took_the_arm_away_and_ended_early(
    robot_app, ended_by
):
    h = robot_app
    h.new_task()
    at_contact = contact(h)
    spots = [list(checker.contact_spots) for checker in h.checkers.all()]
    move_id = h.client.new_id()
    if ended_by == "its permits ran out":
        h.client.permit(move_id, 29)
        h.client.move(retreat(at_contact), start_q=at_contact, command_id=move_id)
    else:
        reached, release = h.hold_move_after(len(h.arm.sent) + 30)
        h.client.permit(move_id, 39)
        h.client.move(retreat(at_contact), start_q=at_contact, command_id=move_id)
        assert reached.wait(WAIT_S)
        if ended_by == "help_needed":
            ask_for_help(h)
        else:
            h.client.stop()
            h.client.wait_state(lambda s: s.stop_number == 1)
        release.set()
    ended = final(h, move_id)
    h.idle()
    assert ended.status in ("refused", "cancelled") and ended.detail["sent"] == 30
    assert max(abs(a - b) for a, b in zip(h.arm.pose(), at_contact)) > 0.1
    # The agent moved the arm away from the contact: the hold there has ended.
    rest = final(h, h.client.rest())
    assert rest.status == "completed", rest.reason
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.arm.torque_on() and h.arm.released_at == []
    # Only the hold ended; the contact spots are the checkers' to keep.
    assert [list(checker.contact_spots) for checker in h.checkers.all()] == spots


def test_a_move_that_sent_no_setpoint_leaves_the_contact_hold(robot_app):
    h = robot_app
    h.new_task()
    at_contact = contact(h)
    elsewhere = [q + 0.2 for q in at_contact]
    refused = h.client.run_move(retreat(elsewhere), start_q=elsewhere)
    assert refused.status == "refused" and refused.detail["sent"] == 0
    again = final(h, h.client.rest())
    assert (again.status, again.reason) == ("refused", AT_A_CONTACT)


def test_a_move_that_ends_at_a_contact_again_keeps_the_contact_hold(robot_app):
    h = robot_app
    h.new_task()
    at_contact = contact(h)
    h.arm.block_joint(1, at_contact[1] - 0.01)
    second = h.client.run_move(retreat(at_contact), start_q=at_contact)
    h.arm.block_joint(1, None)
    assert (second.status, second.reason) == ("failed", "contact")
    assert second.detail["sent"] > 0
    again = final(h, h.client.rest())
    assert (again.status, again.reason) == ("refused", AT_A_CONTACT)
    assert len(h.checkers.recovery.contact_spots) == 2


# The typed jaw reference: the way out for a raised arm whose jaws lost their
# reference (their position is no longer known), so that it can neither rest
# nor release. Thor terminal only, behind the setting jaws.allow_reference.

FROM_RAISED = [[q + 0.004 * (index + 1) for q in RAISED] for index in range(25)]
REFERENCE_REFUSED = "reference typed at the Thor terminal refused: "


@pytest.fixture
def unreferenced(make_robot_app):
    """An app with the setting on that started raised: the jaws have no reference."""
    h = make_robot_app(pose=RAISED, allow_reference=True)
    assert h.app.allow_reference and not h.arm.gripper_referenced()
    assert h.app.motion_state is MotionState.HOLD
    return h


def reference_refused(h, confirmed: bool = True) -> str:
    """A typed reference that is refused at once; nothing went to the jaws."""
    done = Done()
    refused = len(h.heard("refused"))
    reason = h.app.request_reference(confirmed, done)
    assert reason is not None
    result = done.wait()
    assert (result.status, result.reason) == ("refused", reason)
    assert h.heard("refused")[refused][1] == REFERENCE_REFUSED + reason
    assert h.arm.jaw_commands == [] and h.arm.light_holds == []
    return reason


def test_reference_is_refused_while_its_setting_is_off(make_robot_app):
    h = make_robot_app(pose=RAISED)  # as every shipped configuration has it
    assert not h.app.allow_reference
    assert reference_refused(h) == REFERENCE_OFF
    assert REFERENCE_OFF == (
        "the typed jaw reference is switched off: the robot app setting "
        "jaws.allow_reference is false"
    )
    # The Thor terminal asks nothing where the command is off.
    thor = h.thor_terminal()
    told = thor.ask("reference", answer="reference: ")
    assert told == f"reference: refused: {REFERENCE_OFF}"
    assert thor.written(CONFIRM_REFERENCE) == []
    assert not h.arm.gripper_referenced()


def test_reference_is_refused_without_the_typed_phrase(unreferenced):
    h = unreferenced
    assert reference_refused(h, confirmed=False) == CONFIRM_JAWS_EMPTY
    thor = h.thor_terminal()
    for answer in ("yes", "path is clear", ""):
        told = thor.ask("reference", answer, answer="reference: ")
        assert told == "reference: refused: confirm that the jaws are empty"
    assert len(thor.written(CONFIRM_REFERENCE)) == 3
    assert not h.arm.gripper_referenced() and h.arm.jaw_commands == []


def test_reference_is_refused_while_the_agent_owns_the_arm(unreferenced):
    h = unreferenced
    h.new_task()
    assert reference_refused(h) == "control owner is agent, not nobody"
    assert not h.arm.gripper_referenced()
    assert h.app.snapshot().owner is Owner.AGENT


def test_reference_is_refused_while_a_move_runs(unreferenced):
    h = unreferenced
    h.new_task()
    reached, release = h.hold_move_after(6)
    move_id = h.client.move(FROM_RAISED, start_q=RAISED)
    h.client.permit(move_id, len(FROM_RAISED) - 1)
    assert reached.wait(WAIT_S)
    assert reference_refused(h) == "busy: a move is running"
    release.set()
    assert final(h, move_id).status == "completed"
    assert not h.arm.gripper_referenced()


def test_reference_is_refused_while_a_jaw_command_runs(make_robot_app):
    h = make_robot_app(allow_reference=True)
    held = h.hold_jaws_at(0.8)
    h.new_task()
    jaw_id = h.client.gripper(0.5)
    assert held.wait(WAIT_S)
    done = Done()
    assert h.app.request_reference(True, done) == JAWS_BUSY
    assert done.wait().status == "refused"
    h.app.stop("typed stop")
    assert final(h, jaw_id).status == "cancelled"


def test_reference_is_refused_when_the_jaws_are_referenced(make_robot_app):
    h = make_robot_app(allow_reference=True)  # started at rest: referenced
    assert h.arm.gripper_referenced()
    assert reference_refused(h) == "the jaws are referenced already"


@pytest.mark.parametrize("state", ["a fault-stop", "the motors off", "closed"])
def test_reference_is_refused_in_a_state_that_takes_no_jaw_command(
    make_robot_app, state
):
    h = make_robot_app(allow_reference=True)
    if state == "a fault-stop":
        h.arm.set_fault("motor 3 over temperature")
        h.wait_local_event("fault")
        h.arm.clear_faults()
        expected = "fault-stop latched: motor 3 over temperature"
    elif state == "the motors off":
        assert final(h, h.client.release()).status == "completed"
        expected = (
            "the motors are off; switch them on again: type on at the Thor "
            "terminal or run the console command on"
        )
    else:
        h.app.close()
        expected = "the robot app is not running"
    h.arm._jaw_referenced = False  # the reference was lost
    sent = list(h.arm.jaw_commands)  # a release closes the jaws before the cut
    done = Done()
    assert h.app.request_reference(True, done) == expected
    assert done.wait().status == "refused"
    assert not h.arm.gripper_referenced() and h.arm.jaw_commands == sent


def test_a_typed_reference_references_the_jaws_and_rest_is_accepted_after_an_open(
    unreferenced,
):
    h = unreferenced
    # The jaws had closed on something before their reference was lost.
    h.app._jaw_state.note_command_result(0.0, "plain", GripperOutcome("stalled", 0.4))
    thor = h.thor_terminal()
    # The defect: without the reference the arm can neither rest nor release.
    told = thor.ask("rest", "path is clear", answer="rest: ")
    assert told.startswith("rest: refused: the jaws are not proven empty (unknown)")
    assert "refused" in thor.ask("release", answer="release: ")
    refused = final(h, h.client.open_jaws())
    assert (refused.status, refused.reason) == (
        "refused",
        "the jaws are not referenced",
    )
    assert h.arm.sent == [] and h.arm.torque_on()

    told = thor.ask("reference", " Jaws are empty ", answer="reference: ")
    assert told == "reference: completed"
    assert thor.written(CONFIRM_REFERENCE) == [CONFIRM_REFERENCE]
    assert h.arm.gripper_referenced()
    state = h.client.wait_state(
        lambda s: "reference" in (s.last_result or {}).get("command_id", ""), WAIT_S
    )
    assert state.gripper.referenced
    # The reference proves nothing about what the jaws hold.
    assert state.gripper.may_be_holding and state.gripper.load == "loaded"
    assert (state.owner, state.stop_level, state.torque_on) == ("nobody", "none", True)
    told = thor.ask("rest", "path is clear", answer="rest: ")
    assert told.startswith("rest: refused: the jaws are not proven empty (loaded)")
    assert h.arm.sent == []

    # The guarded opening from the console clears the flag; then rest is taken.
    opened = final(h, h.client.open_jaws())
    assert opened.status == "completed" and opened.detail["opening"] == 1.0
    assert thor.ask("rest", "path is clear", answer="rest: ") == "rest: completed"
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD
    assert h.arm.torque_on() and h.arm.released_at == []
    # A referenced gripper is never referenced again.
    told = thor.ask("reference", "jaws are empty", answer="reference: ")
    assert told == "reference: refused: the jaws are referenced already"


def test_a_stop_ends_a_reference_without_a_fault_stop(unreferenced):
    h = unreferenced
    reached, release = h.hold_call("gripper_reference")
    done = Done()
    assert h.app.request_reference(True, done) is None
    assert reached.wait(WAIT_S)  # the gripper worker is about to close the jaws
    busy = Done()
    assert h.app.request_reference(True, busy) == JAWS_BUSY
    assert h.app.stop("typed stop") == 1
    release.set()
    result = done.wait()
    assert result.status == "cancelled"
    assert result.detail == {"outcome": "cancelled", "opening": None}
    state = h.app.snapshot()
    assert (state.stop_level.value, state.fault_reason) == ("stop", "")
    assert h.heard("fault") == []
    assert not h.arm.gripper_referenced() and h.arm.jaw_commands == []
    assert h.arm.torque_on() and h.arm.released_at == [] and h.arm.sent == []
    assert h.app.motion_state is MotionState.HOLD
    # After the STOP the reference can be typed again.
    assert h.app._worker.wait_idle(WAIT_S)
    again = Done()
    assert h.app.request_reference(True, again) is None
    assert again.wait().status == "completed"
    assert h.arm.gripper_referenced()


@pytest.fixture
def referencing(unreferenced):
    """A typed jaw reference held inside the arm's call: (app, release, done)."""
    h = unreferenced
    reached, release = h.hold_call("gripper_reference")
    done = Done()
    assert h.app.request_reference(True, done) is None
    assert reached.wait(WAIT_S)  # the jaws are closing to their stop
    return h, release, done


def test_a_new_task_is_refused_while_the_reference_runs(referencing):
    h, release, done = referencing
    checked: list[list[float]] = []
    agent_checker = h.app._checkers.agent
    check_pose = agent_checker.check_pose

    def noted_check(pose):
        checked.append(list(pose))
        return check_pose(pose)

    agent_checker.check_pose = noted_check
    refused = final(h, h.client.request_owner("agent"))
    assert (refused.status, refused.reason) == ("refused", JAWS_BUSY)
    assert JAWS_BUSY == "busy: a jaw command is running or waiting"
    assert h.app.snapshot().owner is Owner.NOBODY
    # Refused at the first look: the task's pose was not read and checked.
    assert checked == []
    agent_checker.check_pose = check_pose
    release.set()
    assert done.wait().status == "completed" and h.arm.gripper_referenced()
    # The reference is over: the task is taken.
    assert h.app._worker.wait_idle(WAIT_S)
    assert h.new_task().status == "completed"


def test_a_new_task_is_refused_when_the_reference_began_during_its_pose_read(
    unreferenced,
):
    h = unreferenced
    reading, read_on = threading.Event(), threading.Event()
    agent_checker = h.app._checkers.agent
    check_pose = agent_checker.check_pose

    def held_check(pose):
        reading.set()
        read_on.wait(WAIT_S)
        return check_pose(pose)

    agent_checker.check_pose = held_check
    held, release = h.hold_call("gripper_reference")
    command_id = h.client.request_owner("agent")
    assert reading.wait(WAIT_S)  # admitted; the measured pose is being checked
    done = Done()
    assert h.app.request_reference(True, done) is None
    assert held.wait(WAIT_S)
    read_on.set()
    refused = final(h, command_id)
    assert (refused.status, refused.reason) == ("refused", JAWS_BUSY)
    assert h.app.snapshot().owner is Owner.NOBODY
    release.set()
    assert done.wait().status == "completed"


def test_a_move_and_a_jaw_command_are_refused_while_the_reference_runs(referencing):
    """The second barrier behind the owner check.

    The app makes nobody the owner for the whole of a reference, so the test
    writes the agent into the authority by hand: no peer can get there.
    """
    h, release, done = referencing
    seen = h.app.snapshot()
    with h.app._lock:
        h.app._authority.set_owner_if(
            Owner.AGENT,
            "made the owner by the test",
            stop_number=seen.stop_number,
            owner_number=seen.owner_number,
        )
    assert h.app.snapshot().owner is Owner.AGENT
    move_id = h.client.move(FROM_RAISED, start_q=RAISED, **numbers(h))
    # With its permit a move that was taken would send its setpoints.
    h.client.permit(move_id, len(FROM_RAISED) - 1)
    moved = final(h, move_id)
    assert (moved.status, moved.reason) == ("refused", JAWS_BUSY)
    jaw_id = h.client.gripper(0.5, **numbers(h))
    jaws = final(h, jaw_id)
    assert (jaws.status, jaws.reason) == ("refused", JAWS_BUSY)
    # Neither was accepted first, and nothing waits behind the reference.
    assert statuses(h, move_id) == statuses(h, jaw_id) == ["refused"]
    h.idle()
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    release.set()
    done.wait()
    assert h.app._worker.wait_idle(WAIT_S)
    h.idle()
    assert h.arm.sent == [] and h.arm.jaw_commands == []


def test_without_a_task_a_move_and_a_jaw_command_meet_the_owner_check(referencing):
    h, release, done = referencing
    # What a peer sees: nobody owns the arm during a reference.
    moved = final(h, h.client.move(FROM_RAISED, start_q=RAISED, **numbers(h)))
    jaws = final(h, h.client.gripper(0.5, **numbers(h)))
    nobody = "control owner is nobody, not agent"
    assert (moved.status, moved.reason) == ("refused", nobody)
    assert (jaws.status, jaws.reason) == ("refused", nobody)
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    release.set()
    done.wait()


def test_an_open_is_refused_while_the_reference_runs_and_taken_after_it(referencing):
    h, release, done = referencing
    open_id = h.client.open_jaws()
    refused = final(h, open_id)
    assert (refused.status, refused.reason) == ("refused", JAWS_BUSY)
    assert statuses(h, open_id) == ["refused"]  # it never waited behind it
    assert h.heard("refused")[-1][1] == f"open refused: {JAWS_BUSY}"
    release.set()
    assert done.wait().status == "completed" and h.arm.gripper_referenced()
    assert h.app._worker.wait_idle(WAIT_S)
    # The jaws stay where the reference left them until an open that is asked now.
    assert h.arm.jaw_commands == []
    opened = final(h, h.client.open_jaws())
    assert opened.status == "completed"
    assert opened.detail == {"outcome": "reached", "opening": 1.0}


def test_a_typed_rest_is_refused_while_a_jaw_reference_runs(referencing):
    h, release, done = referencing
    rest = Done()
    refused = len(h.heard("refused"))
    assert h.app.request_rest(True, rest) == JAWS_BUSY
    result = rest.wait()
    assert (result.status, result.reason) == ("refused", JAWS_BUSY)
    told = f"rest typed at the Thor terminal refused: {JAWS_BUSY}"
    assert h.heard("refused")[refused][1] == told
    # It never waited behind the reference: no return to rest was asked for.
    assert h.app.wait_motion_idle(0.0)
    assert h.heard("rest") == [] and h.arm.sent == []
    thor = h.thor_terminal()
    typed = thor.ask("rest", "path is clear", answer="rest: ")
    assert typed == f"rest: refused: {JAWS_BUSY}"
    release.set()
    assert done.wait().status == "completed" and h.arm.gripper_referenced()
    assert h.app._worker.wait_idle(WAIT_S)
    # The arm stays where it is until a rest that is typed now.
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert thor.ask("rest", "path is clear", answer="rest: ") == "rest: completed"
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.arm.torque_on() and h.arm.released_at == []


NOT_RETURNED = (
    "a cancelled jaw command has not returned: until it does the jaws take no "
    "other command, and rest and release are refused"
)


def test_a_stop_says_that_the_cancelled_jaw_command_has_not_returned(referencing):
    h, release, done = referencing
    # The driver's call is held: the STOP gives up waiting for it.
    assert h.app.stop("typed stop", "thor_terminal") == 1
    _, text, _ = h.heard("stop")[-1]
    assert text == (
        "STOP from thor_terminal: typed stop; the arm holds, stop number 1; "
        f"{NOT_RETURNED}"
    )
    assert h.client.wait_event("stop").text == text  # the console is told too
    # The cancelled command is a jaw reference: the rest is refused at once.
    rest = Done()
    assert h.app.request_rest(True, rest) == JAWS_BUSY
    assert (rest.wait().status, rest.results[0].reason) == ("refused", JAWS_BUSY)
    release.set()
    assert done.wait().status == "cancelled"
    assert h.app._worker.wait_idle(WAIT_S)
    # A stop that finds no jaw command out says no such thing.
    assert h.app.stop("typed stop", "thor_terminal") == 2
    assert NOT_RETURNED not in h.heard("stop")[-1][1]
    assert h.arm.torque_on() and h.arm.released_at == [] and h.arm.sent == []


def test_a_fault_stop_says_that_the_cancelled_jaw_command_has_not_returned(
    referencing,
):
    h, release, done = referencing
    h.arm.set_fault("motor 3 overtemperature")
    h.step()
    _, text, _ = h.wait_local_event("fault")
    assert text.startswith("FAULT-STOP: ") and text.endswith(f"; {NOT_RETURNED}")
    release.set()
    done.wait()


def test_a_lost_robot_connection_lets_a_typed_reference_go_on(referencing):
    h, release, done = referencing
    holds = len(h.arm.light_holds)
    h.lose_link()
    h.step()
    # Not cancelled, and no light hold went to the closing jaws.
    assert done.results == [] and h.app._worker.referencing()
    assert len(h.arm.light_holds) == holds
    release.set()
    result = done.wait()
    assert result.status == "completed" and h.arm.gripper_referenced()
    state = h.app.snapshot()
    assert (state.stop_level.value, state.owner) == ("none", Owner.NOBODY)
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_lost_robot_connection_still_cancels_a_jaw_command_of_the_link(robot_app):
    h = robot_app
    h.new_task()
    waiting = h.hold_jaws_at(0.6)
    h.client.gripper(0.0)
    assert waiting.wait(WAIT_S)
    holds = len(h.arm.light_holds)
    h.lose_link()
    assert h.app._worker.wait_idle(WAIT_S)
    assert h.arm.gripper_read() == pytest.approx(0.6, abs=0.05)
    assert h.arm.jaw_stiffness == "light" and len(h.arm.light_holds) > holds


def test_a_reference_that_the_driver_fails_is_a_fault_stop_with_the_motors_on(
    unreferenced,
):
    h = unreferenced

    def stuck(cancel=None):
        raise ArmFault("the jaws stopped short of the closed stop")

    h.arm.gripper_reference = stuck
    done = Done()
    assert h.app.request_reference(True, done) is None
    result = done.wait()
    reason = "the jaw reference failed: the jaws stopped short of the closed stop"
    assert (result.status, result.reason) == ("failed", reason)
    state = h.app.snapshot()
    assert (state.stop_level.value, state.fault_reason) == ("fault", reason)
    assert h.arm.torque_on() and h.arm.released_at == []


# The typed open and the typed close of the jaws


def jaws_sent(h) -> list[tuple[float, float, str]]:
    """(opening, effort, mode) of every jaw command the simulated arm got."""
    return [(c.opening, c.max_effort, c.mode) for c in h.arm.jaw_commands]


def holding_an_object(h, at: float = 0.4) -> None:
    """The agent closes the jaws on an object and gives the arm back."""
    h.new_task()
    h.arm.jaw_object(at)
    grasp = final(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert grasp.status == "completed", grasp.reason
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    h.client.wait_state(lambda s: s.owner == "nobody" and s.gripper.may_be_holding)


def test_the_typed_open_is_one_opening_to_full_open(robot_app):
    h = robot_app
    holding_an_object(h)
    sent = len(h.arm.jaw_commands)
    assert final(h, h.client.open_jaws(confirmed=False)).reason == CONFIRM_OPEN
    assert len(h.arm.jaw_commands) == sent
    opened = final(h, h.client.open_jaws())
    assert opened.status == "completed", opened.reason
    assert opened.detail == {"outcome": "reached", "opening": 1.0}
    # One command, not steps of gripper.open_step.
    assert jaws_sent(h)[sent:] == [(1.0, OPEN_EFFORT, PLAIN)]
    state = h.client.wait_state(lambda s: s.gripper.load == "empty")
    assert state.gripper.may_be_holding is False


def test_the_typed_close_closes_the_jaws_and_asks_only_while_they_may_hold(robot_app):
    h = robot_app
    sent = len(h.arm.jaw_commands)
    # Jaws proven empty: no confirmation is needed.
    closed = final(h, h.client.open_jaws(confirmed=False, action=CLOSE_JAWS))
    assert closed.status == "completed", closed.reason
    assert jaws_sent(h)[sent:] == [(0.0, OPEN_EFFORT, CLOSE_UNTIL_CONTACT)]
    assert h.arm.gripper_read() == 0.0
    # Jaws that may hold something: they close on it only on the typed word.
    assert final(h, h.client.open_jaws()).status == "completed"
    holding_an_object(h)
    sent = len(h.arm.jaw_commands)
    refused = final(h, h.client.open_jaws(confirmed=False, action=CLOSE_JAWS))
    assert (refused.status, refused.reason) == ("refused", CONFIRM_CLOSE)
    assert len(h.arm.jaw_commands) == sent
    again = final(h, h.client.open_jaws(confirmed=True, action=CLOSE_JAWS))
    assert again.status == "completed", again.reason
    assert again.detail["outcome"] == "stalled"
    assert h.arm.gripper_read() == pytest.approx(0.4)  # the object is still held


def test_a_typed_open_or_close_needs_nobody_to_own_the_arm(robot_app):
    h = robot_app
    h.new_task()
    for action in OPEN_ACTIONS:
        refused = final(h, h.client.open_jaws(action=action))
        assert refused.status == "refused"
        assert refused.reason == "control owner is agent, not nobody"
    assert h.arm.jaw_commands == []


def test_an_open_with_an_action_the_robot_app_does_not_know_is_refused(robot_app):
    h = robot_app
    refused = final(h, h.client.open_jaws(action="squeeze"))
    assert (refused.status, refused.reason) == ("refused", "unknown action 'squeeze'")
    assert h.arm.jaw_commands == []


def test_an_open_without_an_action_is_the_opening(robot_app):
    # As a console that does not know the field sends it.
    h = robot_app
    command_id = send_raw(h, Open, "plain-open", **numbers(h), confirmed=True)
    assert final(h, command_id).status == "completed"
    assert jaws_sent(h) == [(1.0, OPEN_EFFORT, PLAIN)]


def test_open_and_close_are_typed_at_the_thor_terminal_too(robot_app):
    h = robot_app
    thor = h.thor_terminal()
    assert thor.ask("open", "no", answer="open: ") == f"open: refused: {CONFIRM_OPEN}"
    assert h.arm.jaw_commands == []
    assert thor.wait_line(CONFIRM_OPEN_ASKED)
    assert thor.ask("open", "yes", answer="open: completed") == "open: completed"
    # Empty jaws close without a question.
    assert thor.ask("close", answer="close: ") == "close: completed"
    assert jaws_sent(h) == [
        (1.0, OPEN_EFFORT, PLAIN),
        (0.0, OPEN_EFFORT, CLOSE_UNTIL_CONTACT),
    ]
    # Jaws that may hold something: the terminal asks first.
    thor.ask("open", "yes", answer="open: completed")
    holding_an_object(h)
    asked = thor.ask("close", "no", answer="Close the jaws: they are not proven empty")
    assert "(loaded: the jaws may be holding something)" in asked
    assert thor.wait_line(f"close: refused: {CONFIRM_CLOSE}")
    assert thor.ask("close", "yes", answer="close: completed") == "close: completed"
    assert h.arm.gripper_read() == pytest.approx(0.4)


# The jaw reading in ROBOT_STATE: the agent decides from it what the jaws hold


def test_the_jaws_are_read_anew_for_robot_state_while_the_arm_holds(robot_app):
    h = robot_app
    reads = []
    read = h.arm.gripper_read
    h.arm.gripper_read = lambda: reads.append(1) or read()
    h.arm._jaw = 0.7  # the jaws were pushed: no command told the robot app
    h.advance(3 * h.config.state.publish_period_s, h.config.state.publish_period_s)
    state = h.client.wait_state(lambda s: s.gripper.opening == 0.7)
    assert state.gripper.valid is True
    # One reading for each ROBOT_STATE period, not one for each dispatcher pass.
    assert 1 <= len(reads) <= 4
    # With the motors off nothing is read.
    assert final(h, h.client.release(jaws_empty=True)).status == "completed"
    reads.clear()
    h.advance(3 * h.config.state.publish_period_s, h.config.state.publish_period_s)
    assert reads == []


def test_the_jaws_are_read_anew_after_a_move_and_not_during_it(robot_app):
    h = robot_app
    h.new_task()
    reads = []
    read = h.arm.gripper_read
    h.arm.gripper_read = lambda: reads.append(len(h.arm.sent)) or read()
    h.arm._jaw = 0.6
    # The move takes 0.5 s of the app's clock: ten ROBOT_STATE periods.
    moved = h.client.run_move(PATH, start_q=REST)
    assert moved.status == "completed", moved.reason
    # No reading while setpoints went out: each took the bus before the move
    # began or after it had ended.
    assert reads and set(reads) <= {0, len(PATH)}
    # The state that shows the move's end has a reading newer than its last
    # setpoint: the agent's "held or closed on nothing" is decided from it.
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == moved.command_id
    )
    assert reads[-1] == len(PATH)
    assert state.gripper.opening == 0.6 and state.gripper.t >= state.target_t


# The operator's word that the jaws are empty

JAWS_REFUSAL = (
    "the jaws are not proven empty (loaded): the jaws may be holding something"
)


def test_a_release_refused_for_the_jaws_is_taken_on_the_operators_word(robot_app):
    h = robot_app
    holding_an_object(h)
    refused = final(h, h.client.release())
    assert (refused.status, refused.reason) == ("refused", JAWS_REFUSAL)
    assert h.arm.torque_on()
    taken = final(h, h.client.release(jaws_empty=True))
    assert taken.status == "completed", taken.reason
    assert not h.arm.torque_on()
    (told,) = h.heard(EVENT_JAWS)[:1]
    assert told[1] == JAWS_EMPTY_WORD.format(what="a torque release")


def test_the_operators_word_ends_with_the_next_jaw_command(robot_app):
    h = robot_app
    holding_an_object(h)
    # Away from rest the release is refused for the pose: the word was taken.
    h.arm.set_pose(RAISED)
    refused = final(h, h.client.release(jaws_empty=True))
    assert refused.status == "refused" and "jaws" not in refused.reason
    assert h.app._jaw_state.word_given
    state = h.client.wait_state(lambda s: s.gripper.load == "empty")
    assert state.gripper.may_be_holding is False
    # A new task of the agent ends it.
    h.arm.set_pose(REST)
    h.new_task()
    assert not h.app._jaw_state.word_given
    assert final(h, h.client.gripper(0.0, mode="close_until_contact")).status == (
        "completed"
    )
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    again = final(h, h.client.release())
    assert again.status == "refused" and again.reason.startswith(
        "the jaws are not proven empty"
    )


@pytest.mark.parametrize("command", ["release", "rest", "quit"])
def test_the_thor_terminal_asks_about_the_jaws_and_takes_the_typed_word(
    robot_app, command
):
    h = robot_app
    holding_an_object(h)
    if command == "rest":
        h.arm.set_pose(RAISED)
    thor = h.thor_terminal()
    # A quit with the arm at rest asks nothing about the path.
    typed = [command, "path is clear"] if command == "rest" else [command]
    line = thor.ask(*typed, answer=f"{command}: ")
    assert "the jaws are not proven empty" in line
    asked = thor.wait_line(f"{command} was refused because the jaws are not proven")
    assert asked == ASK_JAWS_EMPTY.format(what=command)
    assert h.arm.torque_on() and h.arm.sent == []
    # The typed word: the command is asked again, and goes on.
    typed[0] = "jaws are empty"
    assert thor.ask(*typed, answer=f"{command}: completed")
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)


def test_the_thor_terminals_question_about_the_jaws_stands_for_one_line(robot_app):
    h = robot_app
    holding_an_object(h)
    thor = h.thor_terminal()
    unknown = "unknown command 'jaws are empty'; type help"
    # No question was put: the phrase is no command.
    assert thor.ask("jaws are empty", answer="unknown command") == unknown
    thor.ask("release", answer="release: refused")
    thor.wait_line("release was refused because the jaws are not proven")
    # Anything else typed next withdraws the question.
    thor.ask("help", answer="stop ")
    assert thor.ask("jaws are empty", answer="unknown command") == unknown
    assert len(thor.written("release: ")) == 1
    assert h.arm.torque_on()


# The operator console: the typed close, and the question about the jaws


@pytest.fixture
def console_robot(tmp_path_factory):
    """A robot link service whose robot side answers as scripted."""
    from link_bench import Bench
    from test_console_cmd import Console as ScriptedRobot

    bench = Bench(tmp_path_factory.mktemp("oj"))
    robot = ScriptedRobot(bench)
    yield robot
    robot.close()
    bench.close()


@pytest.fixture
def console_desk(console_robot):
    """An operator console attached to that service; the test types and reads."""
    from test_console import Desk

    desk = Desk(console_robot.bench.path)
    desk.wait("attached to the robot link service: robot connection up")
    desk.sees(lambda state: True)
    yield desk
    desk.close()


def refuse_for_the_jaws(robot) -> None:
    """The robot app's answers: refused for the jaws, unless on the typed word."""

    def reply(message):
        if getattr(message, "jaws_empty", False):
            return [Result(message.command_id, "completed")]
        return [Result(message.command_id, "refused", JAWS_REFUSAL)]

    robot.reply = reply


def test_the_consoles_close_asks_only_while_the_jaws_are_not_proven_empty(
    console_robot, console_desk
):
    robot, desk = console_robot, console_desk
    # The robot's state shows jaws that are not proven empty.
    desk.type("close")
    asked = desk.wait('Type "yes" to close the jaws')
    assert asked == ASK_CLOSE.format(jaws="jaws unknown").rstrip()
    desk.type("no")
    desk.wait("close not confirmed; nothing sent")
    assert robot.settled() == []
    desk.type("close", "yes")
    desk.wait("close: RESULT completed")
    (sent,) = robot.settled()
    assert (type(sent), sent.action, sent.confirmed) == (Open, CLOSE_JAWS, True)
    # Jaws proven empty: sent without a question.
    empty = GripperState(1.0, 1000.0, True, True, 1.0, False, "empty")
    robot.show(gripper=empty)
    desk.sees(lambda state: state.gripper.load == "empty")
    desk.type("close")
    desk.wait("close: RESULT completed", 2)
    last = robot.settled()[-1]
    assert (type(last), last.action, last.confirmed) == (Open, CLOSE_JAWS, False)
    assert desk.count('Type "yes" to close the jaws') == 2


@pytest.mark.parametrize(
    ("command", "phrase", "kind"),
    [
        ("release", "yes", Release),
        ("rest", "path is clear", OwnerRequest),
        ("shutdown", "yes", Shutdown),
    ],
)
def test_the_console_asks_about_the_jaws_and_sends_the_typed_word(
    console_robot, console_desk, command, phrase, kind
):
    robot, desk = console_robot, console_desk
    refuse_for_the_jaws(robot)
    desk.type(command, phrase)
    desk.wait(f"{command}: RESULT refused: {JAWS_REFUSAL}")
    asked = desk.wait(f"{command} was refused because the jaws are not proven empty")
    assert asked == ASK_JAWS.format(command=command).rstrip()
    # The typed word: the command is asked again, with its own question.
    desk.type("jaws are empty", phrase)
    desk.wait(f"{command}: RESULT completed")
    first, second = robot.settled()
    assert type(first) is type(second) is kind
    assert (first.jaws_empty, second.jaws_empty) == (False, True)


def test_the_consoles_question_about_the_jaws_stands_for_one_line(
    console_robot, console_desk
):
    robot, desk = console_robot, console_desk
    desk.type("jaws are empty")
    desk.wait("unknown command 'jaws are empty'; type help")
    refuse_for_the_jaws(robot)
    desk.type("release", "yes")
    desk.wait("release was refused because the jaws are not proven empty")
    desk.type("state", "jaws are empty")
    desk.wait("unknown command 'jaws are empty'; type help", 2)
    assert [message.jaws_empty for message in robot.settled()] == [False]


def test_the_console_command_asks_about_the_jaws_and_sends_the_typed_word(
    console_robot,
):
    robot = console_robot
    refuse_for_the_jaws(robot)

    def ask(prompt: str) -> str:
        return "jaws are empty" if '"jaws are empty"' in prompt else robot.ask(prompt)

    assert robot.run("release", ask=ask) == 0
    assert [line for line in robot.said if line.startswith("release: ")] == [
        f"release: RESULT refused: {JAWS_REFUSAL}",
        "release: RESULT completed",
    ]
    assert [message.jaws_empty for message in robot.settled()] == [False, True]
    # Without the typed word nothing more is sent.
    robot.forwarded.clear()
    assert robot.run("release") == 1
    assert robot.said[-1] == f"release: {JAWS_NOT_CONFIRMED}"
    assert [message.jaws_empty for message in robot.settled()] == [False]


def test_the_console_commands_close_is_an_open_with_the_close_action(console_robot):
    robot = console_robot
    assert robot.run("close") == 0  # jaws not proven empty: asked, "yes"
    assert len(robot.asked) == 1 and "close on whatever they hold" in robot.asked[0]
    (sent,) = robot.settled()
    assert (type(sent), sent.action, sent.confirmed) == (Open, CLOSE_JAWS, True)
