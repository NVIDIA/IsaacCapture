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
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Open,
    OwnerRequest,
    Release,
    Reset,
    Result,
    Shutdown,
)
from isaaccapture_examples.agentic_remote.robot_app.app import (
    CONFIRM_PATH,
    PATH_CLEAR_PROMPT,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    Owner,
    ReadFailed,
)
from isaaccapture_examples.agentic_remote.robot_app.terminal import CONFIRM_REST

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


def test_quit_typed_away_from_rest_moves_the_arm_only_after_a_typed_yes(robot_app):
    h = robot_app
    sent = raised_by_the_agent(h)
    thor = h.thor_terminal()
    for answer in ("no", "", "y", "yes please"):
        assert thor.ask("quit", answer, answer="quit: ") == QUIT_UNCONFIRMED
    assert len(thor.written(CONFIRM_REST)) == 4
    assert_holding_away_from_rest(h, sent)
    assert thor.ask("quit", " Yes ", answer="quit: ") == QUIT_DONE
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
    assert thor.ask("quit", "yes", answer="quit: ") == QUIT_DONE
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
