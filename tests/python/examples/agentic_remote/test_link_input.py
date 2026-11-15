# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What arrives from the link, through the robot app: a STOP first, sizes, text.

The payloads here are written by hand, as a sender that does not use the
protocol module could write them.
"""

from __future__ import annotations

import json
import logging
import threading

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    MAX_PAYLOAD_BYTES,
    Result,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import Owner, Refusal

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# A pose the agent may hold; the return to rest from it is a real move.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
STALE = "stop number 0 does not match the current 1"
# What a terminal would act on: clear the screen, go home, then a false line.
NASTY = "\x1b[2J\x1b[H\r\nARM IS AT REST - SAFE TO APPROACH\n"

MOVE, GRIPPER, CHECKER_STATE, STOP, OWNER_REQUEST = 65, 66, 67, 68, 69
RESET, RELEASE, OPEN = 70, 71, 72
RESULT, EVENT, ROBOT_STATE, PERMIT, VIEW_STATUS = 74, 75, 96, 97, 98


def raw(obj: dict, ascii_only: bool = True) -> bytes:
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=ascii_only)
    return text.encode("utf-8")


def statuses(h, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def final(h, command_id: str) -> Result:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


def quoted(h) -> dict:
    """What a command quotes, as the robot app has it now."""
    state = h.app.snapshot()
    return {
        "session_id": state.session_id,
        "owner_number": state.owner_number,
        "stop_number": state.stop_number,
    }


def held_move(h, count: int = 6) -> tuple[str, threading.Event]:
    """An agent MOVE along PATH, held once `count` setpoints were sent."""
    reached, release = h.hold_move_after(count)
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


def hold_in_a_handler(h) -> threading.Event:
    """Hold the dispatcher inside a CHECKER_STATE of the agent; returns release."""
    reached, release = threading.Event(), threading.Event()
    h._releases.append(release)
    real = h.checkers.agent.set_grasp_region

    def held(region):
        reached.set()
        release.wait(WAIT_S)
        return real(region)

    h.checkers.agent.set_grasp_region = held
    h.client.checker_state("close_grasp")
    assert reached.wait(WAIT_S)
    return release


def move_payload(h, rows: int, row: str = "[0,0,0,0,0,0]", command_id="big") -> bytes:
    """A MOVE with `rows` equal rows, its command id first, as the module writes."""
    mono, wall = h.client.thor_now()
    head = {
        "command_id": command_id,
        **quoted(h),
        "start_q": REST,
        "start_tol_rad": 0.02,
        "period_s": 0.02,
        "grasp_region": None,
        "evidence_wall_s": wall,
        "expires_mono_s": mono + 5.0,
    }
    body = ",".join([row] * rows)
    return raw(head)[:-1] + b',"setpoints":[' + body.encode() + b"]}"


def unprintable(text: str) -> bool:
    return any(not char.isprintable() for char in text)


# A STOP is served before the link messages that arrived ahead of it


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (raw({"reason": "operator typed stop", "sender": "console"}), "typed stop"),
        (b"", "STOP with a bad payload"),
        (b"not json", "STOP with a bad payload"),
    ],
    ids=["a STOP", "an empty STOP", "a STOP that is not JSON"],
)
def test_a_stop_passes_the_link_messages_that_wait_ahead_of_it(
    robot_app, payload, reason
):
    h = robot_app
    h.new_task()
    release = hold_in_a_handler(h)
    # 40 jaw commands the app would take, then the STOP, all waiting on the link.
    ids = [h.client.gripper(0.5) for _ in range(40)]
    h.workstation_end.send_reliable(STOP, payload)
    release.set()
    results = [final(h, command_id) for command_id in ids]
    # The STOP was served first: every command it passed quotes an old stop number.
    assert [statuses(h, command_id) for command_id in ids] == [["refused"]] * 40
    assert {result.reason for result in results} == {STALE}
    assert h.arm.jaw_commands == []
    assert reason in h.heard("stop")[0][1]
    assert h.app.snapshot().stop_number == 1


@pytest.fixture
def waiting(robot_app):
    """The app in an assistance wait that runs out at the next clock second."""
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    help_id = h.client.request_owner("help_needed", reason="the agent is stuck")
    result = final(h, help_id)
    assert result.status == "completed", result.reason
    h.client.wait_state(lambda s: s.owner_number == result.detail["owner_number"])
    h.advance(119.0)
    assert h.heard("expired") == []
    return h


def test_a_stop_behind_a_burst_still_prevents_the_timed_return(waiting):
    h = waiting

    def refuse_pose(q):
        raise Refusal("outside the agent's limits")

    # The dispatcher is held in the joint read of a new task that will be refused.
    h.checkers.agent.check_pose = refuse_pose
    reached, release = h.hold_call("read_all")
    task = h.client.request_owner("agent")
    assert reached.wait(WAIT_S)
    burst = [h.client.gripper(0.5, owner_number=10**6) for _ in range(200)]
    h.client.stop("operator typed stop")
    h.clock.advance(1.5)  # the wait is over at its next step
    release.set()
    assert final(h, task).status == "refused"
    for command_id in burst:
        final(h, command_id)
    h.idle()
    h.step()
    # The STOP came before the wait ran out, so it cancelled the wait.
    assert h.heard("expired") == []
    assert h.heard("rest") == []
    assert h.arm.sent == []
    assert len(h.heard("stop")) == 1
    assert h.heard("stop")[0][2]["wait_cancelled"] is True
    assert h.arm.pose() == RAISED and h.arm.torque_on()
    assert [statuses(h, command_id) for command_id in burst] == [["refused"]] * 200


def test_the_link_messages_that_wait_keep_their_order(robot_app):
    h = robot_app
    h.new_task()
    release = hold_in_a_handler(h)
    # What the third command gets depends on the second being handled before it.
    ids = [
        h.client.checker_state("clear_contact_spots"),
        h.client.request_owner("nobody", reason="the task is done"),
        h.client.checker_state("close_grasp"),
    ]
    release.set()
    first, gave_up, late = (final(h, command_id) for command_id in ids)
    assert (first.status, gave_up.status) == ("completed", "completed")
    assert (late.status, late.reason) == (
        "refused",
        "control owner is nobody, not agent",
    )


def test_the_supervisor_step_runs_between_link_messages_that_take_long(robot_app):
    h = robot_app

    def refuse_pose(q):
        raise Refusal("outside the agent's limits")

    # A new task reads all six joints: 48 ms of the clock, more than a step period.
    h.checkers.agent.check_pose = refuse_pose
    reached, release = h.hold_call("read_all")
    ids = [h.client.request_owner("agent")]
    assert reached.wait(WAIT_S)
    ids += [h.client.request_owner("agent") for _ in range(5)]
    # How many of the six were answered when each supervisor step began.
    answered: list[int] = []
    real = h.app._supervisor_step

    def supervisor_step() -> None:
        answered.append(sum(h.app._results.get(i) is not None for i in ids))
        real()

    h.app._supervisor_step = supervisor_step
    release.set()
    for command_id in ids:
        assert final(h, command_id).status == "refused"
    h.step()
    # A supervisor step ran after each one: none had to wait for all six.
    assert set(answered) >= {1, 2, 3, 4, 5, 6}


def test_many_quick_link_messages_are_handled_in_one_pass(robot_app):
    h = robot_app
    h.new_task()
    release = hold_in_a_handler(h)
    ids = [h.client.gripper(0.5, owner_number=10**6) for _ in range(200)]
    passes = h.app._passes_finished
    release.set()
    with h.app._pass_cond:
        ended = h.app._pass_cond.wait_for(
            lambda: h.app._passes_finished > passes, WAIT_S
        )
    assert ended
    # The clock did not move, so all had room in the pass that was under way.
    assert all(h.app._results.get(command_id) is not None for command_id in ids)
    for command_id in ids:
        assert final(h, command_id).status == "refused"


def test_link_messages_beyond_the_backlog_bound_are_refused_as_busy(robot_app, caplog):
    h = robot_app
    caplog.set_level(logging.WARNING)
    h.new_task()
    h.app._inbox.max_messages = 8  # INBOX_MESSAGES in the robot app
    release = hold_in_a_handler(h)
    ids = [h.client.gripper(0.5) for _ in range(11)]
    # One whose command id is its last field, as another sender may write it.
    jaws = {"opening": 0.5, "max_effort": 0.5, "mode": "plain"}
    h.workstation_end.send_reliable(
        GRIPPER, raw({**quoted(h), **jaws, "command_id": "id-last"})
    )
    ids.append("id-last")
    h.client.permit("some-move", 5)  # no room either: dropped
    h.client.stop("operator typed stop")
    release.set()
    results = [final(h, command_id) for command_id in ids]
    assert [statuses(h, command_id) for command_id in ids] == [["refused"]] * 12
    # Eight waited and were passed by the STOP; four found no room.
    assert [result.reason for result in results[:8]] == [STALE] * 8
    busy = "busy: 8 link messages wait at the robot app"
    assert [result.reason for result in results[8:]] == [busy] * 4
    assert h.arm.jaw_commands == []
    # The STOP was served although the backlog was full.
    assert h.app.snapshot().stop_number == 1
    logged = [record.getMessage() for record in caplog.records]
    assert f"dropped a link message of type 97: {busy}" in logged
    assert sum("type 66" in line for line in logged) == 1  # once, then counted
    # With the backlog worked off, link messages are taken again.
    h.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    assert h.new_task().status == "completed"


def test_what_does_not_decode_under_a_known_id_is_answered_as_a_repeat(robot_app):
    h = robot_app
    h.new_task()
    done = final(h, h.client.checker_state("close_grasp"))
    assert done.status == "completed"
    bad = {"command_id": done.command_id, **quoted(h), "opening": "wide"}
    h.workstation_end.send_reliable(GRIPPER, raw(bad))
    h.client.wait_until(lambda: len(h.client.results(done.command_id)) == 2, WAIT_S)
    assert h.client.results(done.command_id) == [done, done]
    assert h.arm.jaw_commands == []


# Sizes: nothing over its limit is parsed, and a move's rows are counted first


@pytest.fixture
def parsed(monkeypatch) -> list[int]:
    """The length of every text given to the JSON parser during the test."""
    sizes: list[int] = []
    real = json.loads

    def loads(text, *args, **kwargs):
        sizes.append(len(text))
        return real(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)
    return sizes


def test_a_move_over_its_size_limit_is_refused_and_never_parsed(robot_app, parsed):
    h = robot_app
    h.new_task()
    payload = move_payload(h, 72_000)  # 1 MB: under the limit of the link itself
    assert 1_000_000 < len(payload) < MAX_PAYLOAD_BYTES
    h.workstation_end.send_reliable(MOVE, payload)
    result = final(h, "big")
    assert statuses(h, "big") == ["refused"]
    assert "over the limit" in result.reason
    assert max(parsed) < 100_000
    assert h.arm.sent == []


@pytest.mark.parametrize(
    ("rows", "row", "reason"),
    [
        (40_000, "[0,0,0,0,0,0]", "the move has 40000 setpoints, the limit is 3000"),
        (1, "[" + "0," * 99_999 + "0]", "setpoint 0 needs 6 finite joint values"),
    ],
    ids=["too many rows", "a row of 100000 values"],
)
def test_a_move_of_the_wrong_shape_is_refused_before_its_rows_are_converted(
    robot_app, rows, row, reason
):
    h = robot_app
    h.new_task()
    h.workstation_end.send_reliable(MOVE, move_payload(h, rows, row))
    result = final(h, "big")
    # Refused when it was decoded: it was never accepted for the motion worker.
    assert statuses(h, "big") == ["refused"]
    assert reason in result.reason
    assert h.arm.sent == []


def test_the_largest_valid_move_is_taken(robot_app):
    h = robot_app
    h.new_task()
    # 3000 rows of six numbers at their longest: 17 digits, a sign and an exponent.
    row = "[" + ",".join(["-1.2345678901234567e-308"] * 6) + "]"
    payload = move_payload(h, 3000, row)
    assert len(payload) > 450_000
    reached, release = h.hold_move_after(10)
    h.workstation_end.send_reliable(MOVE, payload)
    h.client.permit("big", 2999)
    assert reached.wait(WAIT_S)
    h.client.stop("end of the test")
    h.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    release.set()
    assert statuses(h, "big")[0] == "accepted"
    assert final(h, "big").status == "cancelled"
    assert len(h.arm.sent) == 10


def test_link_messages_the_app_does_not_take_are_not_decoded(robot_app, caplog):
    h = robot_app
    caplog.set_level(logging.DEBUG)
    for msg_type in (64, RESULT, EVENT):
        h.workstation_end.send_reliable(msg_type, b"not json")
    h.workstation_end.send_latest(ROBOT_STATE, b"not json")
    h.step()
    h.step()
    logged = [record.getMessage() for record in caplog.records]
    assert [text for text in logged if "bad JSON" in text] == []
    assert sorted(text for text in logged if "takes no" in text) == [
        f"the robot app takes no {kind} link message; dropped"
        for kind in ("Event", "Result", "RobotState", "Welcome")
    ]
    assert h.new_task().status == "completed"


# The results store and the jaw queue


def test_a_running_move_keeps_its_result_under_a_flood_of_jaw_commands(robot_app):
    h = robot_app
    h.new_task()
    # The jaw command first: its travel moves the fake clock, and the held move
    # must not take that for lost joint feedback.
    jaws_held = h.hold_jaws_at(0.5)
    h.client.gripper(0.0)
    assert jaws_held.wait(WAIT_S)
    h.client.wait_state()
    h.step()
    move_id, release = held_move(h, 3)
    ids = [h.client.gripper(0.0) for _ in range(300)]
    for command_id in ids:
        h.client.wait_result(command_id, final=False, timeout_s=WAIT_S)
    # The MOVE is sent again under its id while it runs.
    h.client.move(PATH, start_q=REST, command_id=move_id)
    h.client.wait_until(lambda: len(h.client.results(move_id)) == 2, WAIT_S)
    assert statuses(h, move_id) == ["accepted", "accepted"]
    release.set()
    assert final(h, move_id).status == "completed"
    h.idle()
    h.step()
    assert statuses(h, move_id) == ["accepted", "accepted", "completed"]
    assert h.sent() == PATH
    # Few of the 300 were queued; the others were refused as busy, each once.
    refused = [i for i in ids if statuses(h, i) == ["refused"]]
    assert len(refused) >= 290
    assert all(final(h, i).reason.startswith("busy: ") for i in refused)


def test_a_jaw_command_sent_again_while_it_runs_is_not_queued_again(robot_app):
    h = robot_app
    h.new_task()
    jaws_held = h.hold_jaws_at(0.5)
    first = h.client.gripper(0.0)
    assert jaws_held.wait(WAIT_S)
    ids = [h.client.gripper(0.6) for _ in range(256)]
    for command_id in ids:
        h.client.wait_result(command_id, final=False, timeout_s=WAIT_S)
    queued = len(h.app._worker._queue)
    h.client.gripper(0.0, command_id=first)
    h.client.wait_until(lambda: len(h.client.results(first)) == 2, WAIT_S)
    assert statuses(h, first) == ["accepted", "accepted"]
    assert len(h.app._worker._queue) == queued
    assert h.app._results.get(first).status == "accepted"


# Text from the link: cut where it enters, and every answer can be sent


def test_a_stop_with_megabytes_of_reason_is_served_and_reported(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    payload = raw({"reason": "é" * 700_000, "sender": "console"}, ascii_only=False)
    assert 1_400_000 < len(payload) < MAX_PAYLOAD_BYTES
    h.workstation_end.send_reliable(STOP, payload)
    h.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    release.set()
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "cancelled"]
    assert h.sent() == PATH[:6]
    event = h.client.wait_event("stop", timeout_s=WAIT_S)
    assert len(result.reason) < 600 and len(event.text) < 600
    # ROBOT_STATE goes on, with the move's outcome in it.
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == move_id, WAIT_S
    )
    assert len(state.last_result["reason"]) < 600


def test_a_long_stop_reason_is_cut_and_the_sender_is_kept(robot_app):
    h = robot_app
    h.new_task()
    h.client.stop("x" * 10_000, sender="agent")
    event = h.client.wait_event("stop", timeout_s=WAIT_S)
    assert event.data["sender"] == "agent"
    assert event.data["reason"].startswith("x" * 200)
    assert len(event.data["reason"]) <= 256
    assert len(event.text) < 600


def test_a_command_with_megabytes_of_id_is_refused_under_a_cut_id(robot_app):
    h = robot_app
    long_id = "é" * 700_000
    cut = "é" * 125 + "..."
    payload = raw({"command_id": long_id, **quoted(h)}, ascii_only=False)
    before = h.client.wait_state().seq
    h.workstation_end.send_reliable(RESET, payload)
    result = final(h, cut)
    assert statuses(h, cut) == ["refused"]
    assert "over the limit" in result.reason
    assert h.client.results(long_id) == []
    # Neither id is kept, and ROBOT_STATE goes on.
    assert h.app._results.get(long_id) is None and h.app._results.get(cut) is None
    h.advance(0.1, step_s=0.05)
    assert h.client.wait_state(lambda s: s.seq > before, WAIT_S).last_result == {
        "command_id": cut,
        "status": "refused",
        "reason": result.reason,
    }


def test_two_long_ids_that_begin_alike_are_not_one_command(robot_app):
    h = robot_app
    cut = "m" * 125 + "..."
    for last in "12":
        payload = raw({"command_id": "m" * 200 + last, **quoted(h)})
        h.workstation_end.send_reliable(RESET, payload)
    h.client.wait_until(lambda: len(h.client.results(cut)) == 2, WAIT_S)
    assert statuses(h, cut) == ["refused", "refused"]
    assert "command id" in h.client.results(cut)[0].reason
    assert h.app._results.get(cut) is None
    # A command whose id is the cut text is a command of its own.
    command_id = h.client.request_owner("human", command_id=cut)
    h.client.wait_until(lambda: len(h.client.results(cut)) == 3, WAIT_S)
    assert h.client.results(command_id)[2].reason == "human path not built yet"


@pytest.mark.parametrize("command_id", ["\x1b[2Jid", "line\nbreak", "bell\x07"])
def test_a_command_id_that_cannot_be_printed_is_refused(robot_app, command_id, caplog):
    h = robot_app
    caplog.set_level(logging.DEBUG)
    shown = "".join(char if char.isprintable() else " " for char in command_id)
    h.workstation_end.send_reliable(RESET, raw({"command_id": command_id, **quoted(h)}))
    result = final(h, shown)
    assert statuses(h, shown) == ["refused"]
    assert "command id" in result.reason
    assert h.client.results(command_id) == []
    assert h.app._results.get(command_id) is None
    assert [r.getMessage() for r in caplog.records if unprintable(r.getMessage())] == []


def test_words_of_a_link_message_are_cut_where_they_are_echoed(robot_app):
    h = robot_app
    h.new_task()
    base = quoted(h)
    sent = {
        "request": (OWNER_REQUEST, {"request": "r" * 5000}),
        "action": (CHECKER_STATE, {"action": "a" * 5000}),
        "mode": (GRIPPER, {"opening": 0.5, "max_effort": 0.5, "mode": "m" * 5000}),
    }
    for name, (msg_type, fields) in sent.items():
        h.workstation_end.send_reliable(
            msg_type, raw({"command_id": name, **base, **fields})
        )
    for name in sent:
        result = final(h, name)
        assert statuses(h, name) == ["refused"]
        assert name[0] * 20 in result.reason
        assert len(result.reason) < 200
    assert all(len(text) < 300 for _, text, _ in h.heard("refused"))


def test_a_long_help_reason_is_cut_and_the_wait_still_starts(robot_app):
    h = robot_app
    h.new_task()
    help_id = h.client.request_owner("help_needed", reason="stuck " * 2000)
    assert final(h, help_id).status == "completed"
    _, text, data = h.heard("help_needed")[0]
    assert data["reason"].startswith("stuck stuck")
    assert len(data["reason"]) <= 256 and len(text) < 600
    assert h.client.wait_state(lambda s: s.assistance_wait_s is not None, WAIT_S)


def test_a_result_too_long_to_send_goes_out_with_its_reason_cut(robot_app):
    h = robot_app
    before = h.client.wait_state().seq
    h.app._send_result(Result("long-1", "failed", "y" * (MAX_PAYLOAD_BYTES + 1)))
    result = final(h, "long-1")
    assert result.status == "failed"
    assert result.reason.startswith("y" * 100) and len(result.reason) <= 1024
    h.advance(0.1, step_s=0.05)
    state = h.client.wait_state(lambda s: s.seq > before, WAIT_S)
    assert state.last_result["command_id"] == "long-1"
    assert len(state.last_result["reason"]) <= 1024


# A command that does not decode is refused, with the decoder's reason


def undecodable(h) -> list[tuple[str, int, bytes, str]]:
    """(command id, type number, payload, what the decoder says) per bad command."""
    base = quoted(h)
    mono, wall = h.client.thor_now()
    move = {
        **base,
        "start_q": REST,
        "start_tol_rad": 0.02,
        "period_s": 0.02,
        "grasp_region": None,
        "evidence_wall_s": wall,
        "expires_mono_s": mono + 5.0,
    }
    step = [0.002] * 6
    jaws = {"opening": 0.5, "max_effort": 0.5, "mode": "plain"}

    def command(command_id: str, fields: dict) -> bytes:
        return raw({"command_id": command_id, **fields})

    return [
        (
            "bad-nan",
            MOVE,
            command("bad-nan", {**move, "setpoints": [[float("nan")] * 6]}),
            "Move: non-finite number",
        ),
        (
            "bad-null",
            MOVE,
            command("bad-null", {**move, "setpoints": [[None] * 6]}),
            "Move.setpoints[0][0]: expected a number, got NoneType",
        ),
        (
            "bad-row",
            MOVE,
            command("bad-row", {**move, "setpoints": ["0,0,0,0,0,0"]}),
            "Move.setpoints[0]: expected a list, got str",
        ),
        (
            "bad-missing",
            MOVE,
            command(
                "bad-missing",
                {
                    k: v
                    for k, v in {**move, "setpoints": [step]}.items()
                    if k != "period_s"
                },
            ),
            "Move: missing field 'period_s'",
        ),
        (
            "bad-jaws",
            GRIPPER,
            command("bad-jaws", {**base, **jaws, "opening": True}),
            "Gripper.opening: expected a number, got bool",
        ),
        (
            "bad-release",
            RELEASE,
            command("bad-release", {**base, "stop_number": 0.0}),
            "Release.stop_number: expected an integer, got float",
        ),
        (
            "bad-rest",
            OWNER_REQUEST,
            command("bad-rest", {**base, "request": "rest", "confirmed": "yes"}),
            "OwnerRequest.confirmed: expected a boolean, got str",
        ),
        (
            # The command id is not the first field, as another sender may write it.
            "bad-order",
            GRIPPER,
            raw({**base, **jaws, "max_effort": "full", "command_id": "bad-order"}),
            "Gripper.max_effort: expected a number, got str",
        ),
    ]


def test_a_command_that_does_not_decode_is_refused_with_the_decoders_reason(
    robot_app, caplog
):
    h = robot_app
    caplog.set_level(logging.WARNING)
    h.new_task()
    owner = h.app.snapshot()
    cases = undecodable(h)
    for _, msg_type, payload, _ in cases:
        h.workstation_end.send_reliable(msg_type, payload)
    for command_id, _, _, text in cases:
        result = final(h, command_id)
        assert statuses(h, command_id) == ["refused"], command_id
        assert text in result.reason
    h.step()
    # Nothing was run from a payload that did not decode.
    assert h.app.snapshot() == owner
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    assert h.arm.torque_on() and h.arm.released_at == []
    # An operator's command is also refused at both terminals.
    refusals = [text for _, text, _ in h.heard("refused")]
    assert len(refusals) == 2
    assert refusals[0].startswith("release refused: Release.stop_number")
    assert refusals[1].startswith("owner request refused: OwnerRequest.confirmed")
    # The Thor's log names the command.
    logged = [record.getMessage() for record in caplog.records]
    assert any("bad-nan" in text and "non-finite number" in text for text in logged)


def test_only_a_command_is_answered_when_it_does_not_decode(robot_app):
    h = robot_app
    h.new_task()
    base = {"command_id": "not-a-command", **quoted(h)}
    h.workstation_end.send_reliable(99, raw(base))
    h.workstation_end.send_latest(PERMIT, raw({**base, "move_id": 5}))
    h.workstation_end.send_latest(VIEW_STATUS, raw({**base, "live": "yes"}))
    h.workstation_end.send_reliable(STOP, raw({**base, "reason": 5}))
    probe = h.client.request_owner("human", stop_number=1)
    final(h, probe)  # sent after them: an answer would have arrived by now
    assert h.client.results("not-a-command") == []
    # The STOP with a bad payload was still a STOP.
    assert h.app.snapshot().stop_number == 1


# A refused permit is named when its move is refused


def with_first_wait(config, first_wait_s: float):
    import dataclasses

    permits = dataclasses.replace(config.permits, first_wait_s=first_wait_s)
    return dataclasses.replace(config, permits=permits)


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"evidence_ahead_s": 2.0}, "permit evidence is in the future"),
        ({"evidence_ahead_s": -4.0}, "permit evidence is too old"),
        ({"expires_mono_s": 5.0}, "permit expired"),
        (
            {"owner_number": -1},
            "permit: owner number -1 does not match the current 1",
        ),
    ],
    ids=["clock ahead", "old evidence", "expiry on another clock", "old owner number"],
)
def test_a_move_refused_for_want_of_a_permit_names_the_permits_refusal(
    make_robot_app, config, fields, why
):
    h = make_robot_app(config=with_first_wait(config, 0.02))
    h.new_task()
    fields = dict(fields)
    ahead = fields.pop("evidence_ahead_s", None)
    if ahead is not None:
        fields["evidence_wall_s"] = h.client.thor_now()[1] + ahead
    move_id = h.client.new_id()
    events = len(h.client.events)
    for _ in range(5):
        h.client.permit(move_id, len(PATH) - 1, **fields)
        h.step()
    h.client.move(PATH, start_q=REST, command_id=move_id)
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "refused"]
    assert result.reason == (
        f"no permit for this move; its last permit was refused: {why} (waited 0.02 s)"
    )
    assert h.arm.sent == []
    # Permits come every 50 ms: a refused one is no EVENT.
    h.step()
    assert h.client.events[events:] == []


def test_another_moves_refused_permit_is_not_named(make_robot_app, config):
    h = make_robot_app(config=with_first_wait(config, 0.02))
    h.new_task()
    h.client.permit("another-move", len(PATH) - 1, owner_number=-1)
    h.step()
    move_id = h.client.move(PATH, start_q=REST)
    result = final(h, move_id)
    assert result.reason == "no permit for this move (waited 0.02 s)"


# Text from the link in the log and at the Thor terminal; floods


def test_text_from_a_stop_is_printable_in_the_log_and_at_the_terminal(
    make_robot_app, caplog
):
    caplog.set_level(logging.DEBUG)
    h = make_robot_app(pose=RAISED)
    h.arm.gripper_reference()  # started away from rest: the jaws had no reference
    thor = h.thor_terminal()
    reached, release = h.hold_move_after(3)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    h.client.stop(NASTY, sender="console\x07")
    h.client.wait_state(lambda s: s.stop_number == 1, WAIT_S)
    release.set()
    result = final(h, rest_id)
    h.idle()
    thor.terminal.flush()
    assert result.status == "cancelled"
    assert "ARM IS AT REST - SAFE TO APPROACH" in result.reason
    assert not unprintable(result.reason)
    assert [line for line in thor.lines if unprintable(line)] == []
    logged = [record.getMessage() for record in caplog.records]
    assert [text for text in logged if unprintable(text)] == []
    assert any("SAFE TO APPROACH" in text for text in logged)
    assert h.app.snapshot().owner is Owner.NOBODY


def test_a_flood_of_bad_link_messages_adds_few_lines(robot_app, caplog):
    h = robot_app
    caplog.set_level(logging.WARNING)
    thor = h.thor_terminal()
    base = {**quoted(h), "confirmed": False}
    count = 300
    for index in range(count):
        h.workstation_end.send_reliable(120, b"{}")  # no such link message
        h.workstation_end.send_reliable(MOVE, b"{not json")
        h.workstation_end.send_reliable(OPEN, raw({"command_id": f"o-{index}", **base}))
    for index in range(count):
        assert final(h, f"o-{index}").reason == "confirm the jaw opening"
    h.step()
    thor.terminal.flush()
    # Every command was answered and every refusal is an EVENT on the link.
    refused = [event for event in h.client.events if event.kind == "refused"]
    assert len(refused) == count
    # The first line of each kind is there; the repeats are counted, not written.
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    assert "type 120" in warnings[0] and "type 65" in warnings[1]
    assert thor.written("[refused]") == [
        "[refused] open refused: confirm the jaw opening"
    ]
    # Once the second is over the counts are written.
    h.advance(1.5, step_s=0.5)
    h.app.stop("typed stop")
    thor.terminal.flush()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 4
    assert all(f"{count - 1} more" in text for text in warnings[2:])
    assert len(thor.written("[refused]")) == 2
    assert f"{count - 1} more" in thor.written("[refused]")[1]
