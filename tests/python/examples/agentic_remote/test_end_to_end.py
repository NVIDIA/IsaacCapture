# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The P0 gate: a started robot app on the simulated arm, driven over a loopback link.

Commands come from a workstation client or the Thor terminal's calls; evidence
is what the simulated arm recorded and what the workstation received. Tests
named test_gate_* show the gate of phase P0 in section 17 of the design: a
refused or expired command cannot move the arm, no torque is cut away from rest.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    GraspRegion,
    Result,
    RobotState,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    JointReading,
    ReadFailed,
)

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
BLOCKED_S = 0.05  # real seconds given to something that must not happen
PERIOD_S = 0.02
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# 60 setpoints at the agent's speed limit: far enough for a contact to show.
LONG_PATH = [[0.008 * (index + 1)] * 6 for index in range(60)]
# 200 setpoints, 4 s: longer than one permit of the client lasts.
SLOW_PATH = [[0.004 * (index + 1)] * 6 for index in range(200)]
# A pose the agent may hold; the return to rest from it has 209 setpoints.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
RETURN_SETPOINTS = 209
LATCHED = "fault-stop latched: motor 3 overheated"


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


def statuses(h, command_id: str) -> list[str]:
    return [result.status for result in h.client.results(command_id)]


def events(h, kind: str) -> list[Event]:
    """The EVENTs of this kind the workstation received so far."""
    return [event for event in h.client.events if event.kind == kind]


def settle(h) -> None:
    """Motion work is over and the workstation has every EVENT sent so far."""
    h.idle()
    h.step()
    heard = len(h.local_events)
    h.client.wait_until(lambda: len(h.client.events) >= heard, WAIT_S, "the EVENTs")


def state_after(h, command_id: str, **shown) -> RobotState:
    """The ROBOT_STATE that shows this command's final RESULT as the newest one.

    A ROBOT_STATE is not read in one step: give in `shown` the fields that
    another thread changed along with the RESULT, and they are waited for too.
    """

    def found(s: RobotState) -> bool:
        ended = (s.last_result or {}).get("command_id") == command_id
        return ended and all(getattr(s, name) == value for name, value in shown.items())

    return h.client.wait_state(found, WAIT_S)


def now_state(h) -> RobotState:
    """A ROBOT_STATE built after every command sent so far was answered."""
    probe = h.client.checker_state("probe")  # refused whoever owns the arm
    assert final(h, probe).status == "refused"
    return state_after(h, probe)


def facts(state: RobotState) -> tuple:
    """What a refused command must leave as it was."""
    return (
        state.owner,
        state.owner_number,
        state.stop_number,
        state.stop_level,
        state.motion,
        state.torque_on,
    )


def stop(h, sender: str = "console", reason: str = "operator pressed stop"):
    """STOP over the link; the ROBOT_STATE that shows the new stop number."""
    number = h.client.state.stop_number + 1
    h.client.stop(reason, sender=sender)
    return h.client.wait_state(lambda s: s.stop_number == number, WAIT_S)


def held_move(
    h, count: int = 6, path=PATH, start_q=REST
) -> tuple[str, threading.Event]:
    """An agent MOVE with a permit for all of it, held once `count` setpoints left."""
    reached, release = h.hold_move_after(len(h.arm.sent) + count)
    move_id = h.client.move(path, start_q=start_q)
    h.client.permit(move_id, len(path) - 1)
    assert reached.wait(WAIT_S), "the move did not reach its hold"
    return move_id, release


def fed_move(h, path, ahead: int = 20, every: int = 10, **move_fields):
    """A MOVE whose permits reach `ahead` setpoints past the arm, renewed in step.

    Returns the move id and, for each permit, (setpoints sent when it left, the
    last setpoint it covers).
    """
    move_id = h.client.new_id()
    permits: list[tuple[int, int]] = []
    before = len(h.arm.sent)
    real = h.arm.send_target

    def permit() -> None:
        done = len(h.arm.sent) - before
        through = min(len(path) - 1, done + ahead - 1)
        permits.append((done, through))
        h.client.permit(move_id, through, evidence_wall_s=h.clock.wall())

    def send_target(q: Sequence[float]) -> None:
        real(q)
        done = len(h.arm.sent) - before
        if done < len(path) and done % every == 0:
            permit()
            h.step()  # the motion worker waits until the dispatcher has the permit

    h.arm.send_target = send_target
    permit()
    h.client.move(path, command_id=move_id, **move_fields)
    return move_id, permits


def ask_for_help(h, reason: str = "the agent is stuck") -> Result:
    """OWNER_REQUEST "help_needed", answered, with its ROBOT_STATE received."""
    result = final(h, h.client.request_owner("help_needed", reason=reason))
    assert result.status == "completed", result.reason
    h.client.wait_state(lambda s: s.owner_number == result.detail["owner_number"])
    return result


def run_wait_out(h) -> None:
    """Advance the clock until the assistance wait has run out, and no further."""
    h.advance_until(lambda: h.heard("expired"))
    h.idle()


def close_on_an_object(h, at: float = 0.4) -> Result:
    """The agent closes the jaws until they meet an object at this opening."""
    h.arm.jaw_object(at)
    result = final(h, h.client.gripper(0.0, mode="close_until_contact"))
    assert result.status == "completed", result.reason
    return result


def contact(h) -> Result:
    """An agent move that meets something: joint index 0 cannot pass 0.05 rad."""
    h.arm.block_joint(0, 0.05)
    result = h.client.run_move(LONG_PATH, start_q=REST)
    h.arm.block_joint(0, None)
    return result


def with_config(config, **blocks):
    """The configuration with fields of its blocks replaced: permits={...}."""
    changed = {
        block: dataclasses.replace(getattr(config, block), **fields)
        for block, fields in blocks.items()
    }
    return dataclasses.replace(config, **changed)


def assert_holding_at(h, pose: Sequence[float], sent: int = 0) -> None:
    """The arm is where it was, powered, and was sent nothing more."""
    assert len(h.arm.sent) == sent
    assert h.arm.pose() == pytest.approx(pose, abs=1e-9)
    assert h.arm.torque_on() and h.arm.released_at == []


# Start


def test_start_sends_welcome_and_a_state_with_the_arm_powered_at_rest(robot_app):
    h = robot_app
    welcome, state = h.client.welcome, h.client.state
    assert welcome.protocol_version == 1
    assert welcome.session_id == state.session_id != ""
    assert welcome.rest_q == REST
    assert welcome.setpoint_period_s == PERIOD_S
    assert welcome.max_move_setpoints == h.config.executor.max_setpoints
    assert (state.owner, state.owner_number) == ("nobody", 0)
    assert (state.stop_number, state.stop_level) == (0, "none")
    assert (state.motion, state.torque_on) == ("rest_hold", True)
    assert state.gripper.referenced and state.gripper.load == "empty"
    assert [sample.valid for sample in state.joints] == [True] * 6
    assert h.arm.torque_on()
    assert h.arm.sent == [] and h.arm.jaw_commands == [] and h.arm.released_at == []


# A whole agent task


def test_a_whole_agent_task_from_ownership_to_the_next_task(robot_app):
    h, client = robot_app, robot_app.client
    task = h.new_task()
    assert task.detail == {"owner": "agent", "owner_number": 1}

    # A move to the object, fed by permits that reach 20 setpoints past the arm.
    above = [0.3, -0.2, 0.25, 0.0, 0.1, 0.0]
    reach = client.plan_straight(REST, above, 0.3)
    region = GraspRegion(center_xy=(0.3, 0.0), radius_m=0.05, z_min=0.0, z_max=0.1)
    move_id, permits = fed_move(h, reach, start_q=REST, grasp_region=region)
    first = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "completed"]
    assert h.sent() == reach
    assert first.end_q == pytest.approx(above, abs=0.03)
    assert len(permits) == 1 + (len(reach) - 1) // 10
    assert all(through - done == 19 for done, through in permits[:-2])

    # The grasp: close until the jaws meet the object, then hold it lightly.
    grasp = close_on_an_object(h, 0.4)
    assert grasp.detail == {"outcome": "stalled", "opening": pytest.approx(0.4)}
    assert h.arm.jaw_stiffness == "light"
    assert client.wait_state(lambda s: s.gripper.may_be_holding).gripper.load == (
        "loaded"
    )
    assert final(h, client.checker_state("close_grasp")).status == "completed"

    # A second move carries the object.
    carry = client.plan_straight(first.end_q, [0.1, -0.3, 0.2, 0.0, 0.1, 0.2], 0.3)
    second = client.run_move(carry, start_q=first.end_q)
    assert second.status == "completed", second.reason
    assert h.sent() == reach + carry

    # The agent fails: its ownership ends and the assistance wait starts.
    helped = ask_for_help(h, "the place step failed three times")
    assert helped.detail["owner"] == "nobody" and helped.detail["owner_number"] == 2
    waiting = client.wait_state(lambda s: s.assistance_wait_s is not None)
    assert (waiting.owner, waiting.stop_level) == ("nobody", "none")
    assert "the place step failed three times" in client.wait_event("help_needed").text
    late = client.run_move(PATH, start_q=second.end_q, owner_number=1)
    assert late.reason == "control owner is nobody, not agent"

    # A new task cancels the wait; the object stayed in the jaws throughout.
    assert h.new_task().detail == {"owner": "agent", "owner_number": 3}
    assert client.state.assistance_wait_s is None
    h.advance(125.0)
    settle(h)
    assert events(h, "expired") == []
    assert h.sent() == reach + carry
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    assert h.arm.gripper_read() == pytest.approx(0.4)
    assert h.arm.torque_on() and h.arm.released_at == []


def test_robot_state_shows_a_move_under_way(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    state = h.client.wait_state(lambda s: s.move_id == move_id)
    assert (state.motion, state.move_total, state.owner) == ("moving", 25, "agent")
    assert 0 <= state.move_index <= 6
    release.set()
    done = final(h, move_id)
    assert done.detail == {"outcome": "completed", "sent": 25, "total": 25}
    after = state_after(h, move_id, move_id="")
    assert (after.motion, after.move_total) == ("hold", 0)
    assert after.target_q == PATH[-1]
    assert after.last_result == {
        "command_id": move_id,
        "status": "completed",
        "reason": "",
    }


# Stale numbers and moves that may not start


@pytest.fixture
def second_task(make_robot_app, config):
    """The agent's second task, after a STOP: every number has moved on."""
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    stop(h)
    h.new_task()
    state = h.client.state
    assert (state.owner, state.owner_number, state.stop_number) == ("agent", 3, 1)
    return h


OLD = {
    "session id": {"session_id": "an-earlier-run"},
    "owner number": {"owner_number": 1},
    "stop number": {"stop_number": 0},
}
old_values = pytest.mark.parametrize("old", OLD.values(), ids=OLD.keys())


@old_values
def test_gate_a_move_quoting_an_old_value_sends_no_setpoint(second_task, old):
    h = second_task
    result = h.client.run_move(PATH, start_q=REST, **old)
    assert statuses(h, result.command_id) == ["refused"]
    assert "does not match" in result.reason
    h.idle()
    assert_holding_at(h, REST)


@old_values
def test_a_jaw_command_quoting_an_old_value_moves_no_jaw(second_task, old):
    h = second_task
    command_id = h.client.gripper(0.0, **old)
    assert "does not match" in final(h, command_id).reason
    assert statuses(h, command_id) == ["refused"]
    h.step()
    assert h.arm.jaw_commands == []
    assert h.arm.gripper_read() == 1.0


@old_values
def test_a_permit_quoting_an_old_value_lets_no_setpoint_out(second_task, old):
    h = second_task
    move_id = h.client.new_id()
    h.client.permit(move_id, len(PATH) - 1, **old)
    h.client.move(PATH, start_q=REST, command_id=move_id)
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "refused"]
    # The move's refusal names why its permit was refused.
    told = "no permit for this move; its last permit was refused: permit: "
    assert result.reason.startswith(told)
    assert "does not match" in result.reason
    assert result.reason.endswith(" (waited 0.02 s)")
    assert_holding_at(h, REST)


def test_gate_an_expired_move_sends_no_setpoint(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=REST, expires_in_s=0.0)
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason.startswith("the move expired")
    h.idle()
    assert_holding_at(h, REST)


def test_a_move_that_expires_before_its_permit_comes_sends_no_setpoint(
    make_robot_app, config
):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 30.0}))
    h.new_task()
    move_id = h.client.move(PATH, start_q=REST, expires_in_s=0.05)
    assert h.client.wait_result(move_id, final=False).status == "accepted"
    h.advance(0.1)
    result = final(h, move_id)
    assert result.status == "refused"
    assert result.reason.startswith("the move expired")
    # The permit comes too late to start it.
    h.client.permit(move_id, len(PATH) - 1)
    h.step()
    h.idle()
    assert h.arm.sent == []


def test_a_move_from_another_start_pose_sends_no_setpoint(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=[0.1] * 6)
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason.startswith("start pose: joint index 0 is at 0.0000 rad")
    assert_holding_at(h, REST)


def test_a_move_longer_than_the_thor_allows_sends_no_setpoint(robot_app):
    h = robot_app
    h.new_task()
    limit = h.client.welcome.max_move_setpoints
    result = h.client.run_move([REST] * (limit + 1), start_q=REST)
    assert result.status == "refused"
    assert result.reason == f"the move has {limit + 1} setpoints, the limit is {limit}"
    assert_holding_at(h, REST)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"period_s": 0.01}, "period_s is 0.01, the configured period is 0.02 s"),
        ({"start_tol_rad": 0.2}, "start_tol is 0.2 rad, the limit is 0.05 rad"),
        ({"setpoints": []}, "the move has no setpoints"),
        ({"setpoints": [[0.004] * 5]}, "setpoint 0 needs 6 finite joint values"),
    ],
    ids=["another period", "a wide start tolerance", "no setpoints", "five joints"],
)
def test_a_move_the_thor_cannot_take_sends_no_setpoint(robot_app, fields, reason):
    h = robot_app
    h.new_task()
    setpoints = fields.pop("setpoints", PATH)
    result = h.client.run_move(setpoints, start_q=REST, **fields)
    assert (result.status, result.reason) == ("refused", reason)
    assert_holding_at(h, REST)


def test_a_setpoint_over_the_agents_speed_limit_is_refused_not_clamped(robot_app):
    h = robot_app
    h.new_task()
    jump = [PATH[4][0] + 0.02, *PATH[4][1:]]  # 1 rad/s on joint index 0
    path = PATH[:5] + [jump] + PATH[6:]
    result = h.client.run_move(path, start_q=REST)
    assert result.status == "refused"
    assert "would move at 1.000 rad/s, over the agent speed limit" in result.reason
    assert result.detail == {"outcome": "refused", "sent": 5, "total": 25}
    assert h.sent() == PATH[:5]  # nothing was altered to fit and nothing followed
    assert h.arm.last_target()[0] == PATH[4]
    assert h.arm.torque_on()


def test_a_setpoint_outside_the_agents_envelope_is_refused_not_clamped(robot_app):
    h = robot_app
    edge = h.config.limits.agent.envelope_max[0]
    start = [edge - 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    h.arm.set_pose(start)
    h.new_task()
    path = [[start[0] + 0.008 * (index + 1), *start[1:]] for index in range(12)]
    inside = [q for q in path if q[0] <= edge]
    result = h.client.run_move(path, start_q=start)
    assert result.status == "refused"
    assert "is outside the agent envelope" in result.reason
    assert 0 < len(inside) < len(path)
    assert h.sent() == inside
    assert h.arm.last_target()[0] == inside[-1]


def test_a_new_task_outside_the_agents_limits_needs_a_rest_first(robot_app):
    h = robot_app
    h.arm.set_pose([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    refused = h.client.start_task()
    assert refused.status == "refused"
    assert refused.reason.startswith("type rest first")
    assert h.client.state.owner == "nobody"
    assert final(h, h.client.rest()).status == "completed"
    assert h.client.start_task().status == "completed"


def test_a_move_is_refused_unless_the_agent_owns_the_arm(robot_app):
    h = robot_app
    result = h.client.run_move(PATH, start_q=REST)
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason == "control owner is nobody, not agent"
    assert_holding_at(h, REST)


# Permits


def test_gate_without_a_permit_the_arm_does_not_advance(make_robot_app, config):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    move_id = h.client.move(PATH, start_q=REST)
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "refused"]
    assert result.reason == "no permit for this move (waited 0.02 s)"
    assert result.detail == {"outcome": "refused", "sent": 0, "total": 25}
    assert_holding_at(h, REST)


def test_a_permit_for_another_move_lets_no_setpoint_out(make_robot_app, config):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    h.client.permit("another-move", len(PATH) - 1)  # current numbers, good evidence
    h.step()
    move_id = h.client.move(PATH, start_q=REST)
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "refused"]
    assert result.reason == "no permit for this move (waited 0.02 s)"
    assert_holding_at(h, REST)


def test_a_move_starts_only_when_its_first_permit_has_arrived(make_robot_app, config):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 30.0}))
    h.new_task()
    move_id = h.client.move(PATH, start_q=REST)
    h.advance(0.2, step_s=0.05)  # ROBOT_STATE goes out as the clock moves
    state = h.client.wait_state(lambda s: s.move_id == move_id)
    assert (state.motion, state.move_index, state.move_total) == ("moving", 0, 25)
    assert h.arm.sent == []
    h.client.permit(move_id, len(PATH) - 1)
    assert final(h, move_id).status == "completed"
    assert h.sent() == PATH


def test_gate_when_permits_stop_the_arm_holds_at_the_last_permitted_setpoint(
    robot_app,
):
    h = robot_app
    h.new_task()
    move_id = h.client.move(PATH, start_q=REST)
    h.client.permit(move_id, 9)
    result = final(h, move_id)
    assert (result.status, result.reason) == (
        "refused",
        "permit reaches setpoint 9, not 10",
    )
    assert result.detail == {"outcome": "refused", "sent": 10, "total": 25}
    assert result.end_q == pytest.approx(PATH[9], abs=0.03)
    assert h.sent() == PATH[:10]
    assert h.arm.last_target()[0] == PATH[9]
    assert h.arm.torque_on() and h.arm.released_at == []
    state = state_after(h, move_id)
    assert (state.motion, state.owner, state.stop_level) == ("hold", "agent", "none")


def test_a_move_that_held_for_want_of_a_permit_is_never_resumed(robot_app):
    h = robot_app
    h.new_task()
    move_id = h.client.move(PATH, start_q=REST)
    h.client.permit(move_id, 9)
    held = final(h, move_id)
    # A permit for the rest of it comes after the move has ended.
    h.client.permit(move_id, len(PATH) - 1)
    h.step()
    h.idle()
    assert h.sent() == PATH[:10]
    # The agent may send a new MOVE from the measured pose.
    onward = h.client.plan_straight(held.end_q, PATH[-1], 0.2)
    assert h.client.run_move(onward, start_q=held.end_q).status == "completed"
    assert h.sent() == PATH[:10] + onward


def test_a_shorter_permit_shortens_how_far_a_running_move_goes(robot_app):
    h = robot_app
    h.new_task()
    move_id = h.client.new_id()
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(h.arm.sent) == 5:
            # Newer camera evidence: less of the path is clear than before.
            h.client.permit(move_id, 7, evidence_wall_s=h.clock.wall())
            h.step()

    h.arm.send_target = send_target
    h.client.permit(move_id, len(PATH) - 1, evidence_wall_s=h.clock.wall())
    h.client.move(PATH, start_q=REST, command_id=move_id)
    result = final(h, move_id)
    assert (result.status, result.reason) == (
        "refused",
        "permit reaches setpoint 7, not 8",
    )
    assert h.sent() == PATH[:8]
    assert h.arm.last_target()[0] == PATH[7]


@pytest.mark.parametrize(
    ("evidence_age_s", "expires_in_s", "why"),
    [
        (0.0, 0.0, "permit expired"),
        (3.5, 2.5, "permit evidence is too old"),
        (-1.0, 2.5, "permit evidence is in the future"),
    ],
    ids=["expired", "old camera evidence", "camera evidence from the future"],
)
def test_a_permit_that_is_expired_or_rests_on_bad_evidence_is_not_used(
    make_robot_app, config, evidence_age_s, expires_in_s, why
):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    move_id = h.client.new_id()
    h.client.permit(
        move_id,
        len(PATH) - 1,
        evidence_wall_s=h.clock.wall() - evidence_age_s,
        expires_in_s=expires_in_s,
    )
    h.client.move(PATH, start_q=REST, command_id=move_id)
    result = final(h, move_id)
    # The move's refusal names why its permit was refused.
    assert (result.status, result.reason) == (
        "refused",
        f"no permit for this move; its last permit was refused: {why} (waited 0.02 s)",
    )
    assert_holding_at(h, REST)


def test_a_move_holds_when_its_permit_expires_on_the_way(robot_app):
    h = robot_app
    h.new_task()
    move_id = h.client.move(SLOW_PATH, start_q=REST)
    h.client.permit(move_id, len(SLOW_PATH) - 1)  # lasts 2.5 s of the 4 s
    result = final(h, move_id)
    assert (result.status, result.reason) == ("refused", "permit expired")
    sent = result.detail["sent"]
    assert 110 < sent < 126
    assert h.sent() == SLOW_PATH[:sent]
    assert h.arm.last_target()[0] == SLOW_PATH[sent - 1]
    assert h.arm.torque_on()


def test_a_move_holds_when_its_permits_evidence_grows_too_old_on_the_way(robot_app):
    h = robot_app
    h.new_task()
    move_id = h.client.move(SLOW_PATH, start_q=REST)
    # The camera evidence is 2.5 s old: 0.5 s short of the 3 s limit.
    wall = h.clock.wall() - 2.5
    h.client.permit(move_id, len(SLOW_PATH) - 1, evidence_wall_s=wall, expires_in_s=5)
    result = final(h, move_id)
    assert (result.status, result.reason) == ("refused", "permit evidence is too old")
    sent = result.detail["sent"]
    assert 15 < sent < 25
    assert h.sent() == SLOW_PATH[:sent]
    assert h.arm.last_target()[0] == SLOW_PATH[sent - 1]


# STOP


def test_gate_stop_during_a_move_and_a_jaw_close_holds_both_and_releases_nothing(
    robot_app,
):
    h = robot_app
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    move_id, release = held_move(h)
    state = stop(h)
    release.set()
    move, jaws = final(h, move_id), final(h, jaw_id)
    assert (move.status, jaws.status) == ("cancelled", "cancelled")
    assert move.reason == "STOP from console: operator pressed stop"
    # The next setpoint was not sent: the arm holds the last one.
    assert h.sent() == PATH[:6]
    assert h.arm.last_target()[0] == PATH[5]
    # The close ended where it was, at the light stiffness, and was never opened.
    # The light hold went out at once and again when the command had returned.
    assert h.arm.jaw_stiffness == "light"
    assert len(h.arm.light_holds) == 2
    assert h.arm.gripper_read() == pytest.approx(0.6)
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    assert h.arm.torque_on() and h.arm.released_at == []
    assert (state.stop_level, state.owner, state.owner_number) == ("stop", "nobody", 2)
    # Later commands that quote the stop number from before the STOP are refused.
    h.new_task()
    refusal = "stop number 0 does not match the current 1"
    again = h.client.run_move(PATH, start_q=move.end_q, stop_number=0)
    assert (again.status, again.reason) == ("refused", refusal)
    assert final(h, h.client.gripper(1.0, stop_number=0)).reason == refusal
    assert final(h, h.client.rest(stop_number=0)).reason == refusal
    assert final(h, h.client.release(stop_number=0)).reason == refusal
    h.idle()
    assert h.sent() == PATH[:6]
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    assert h.arm.torque_on() and h.arm.released_at == []


@pytest.mark.parametrize("sender", ["console", "controller", "thor_terminal", "agent"])
def test_stop_from_any_sender_during_a_move_keeps_the_next_setpoint_back(
    robot_app, sender
):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    state = stop(h, sender, "stop now")
    release.set()
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "cancelled"]
    assert result.reason == f"STOP from {sender}: stop now"
    assert result.detail == {"outcome": "stopped", "sent": 6, "total": 25}
    assert h.sent() == PATH[:6]
    assert h.arm.last_target()[0] == PATH[5]
    assert h.arm.torque_on() and h.arm.released_at == []
    assert (state.stop_level, state.owner, state.owner_number) == ("stop", "nobody", 2)
    told = h.client.wait_event("stop")
    assert (told.data["sender"], told.data["stop_number"]) == (sender, 1)
    after = state_after(h, move_id)
    assert (after.motion, after.torque_on) == ("hold", True)


@pytest.mark.parametrize("key", ["typed stop", "Ctrl+C"])
def test_stop_at_the_thor_terminal_during_a_move_keeps_the_next_setpoint_back(
    robot_app, key
):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    if key == "Ctrl+C":
        h.app.request_stop(key)  # what a signal handler may call
    else:
        assert h.app.stop(key) == 1
    state = h.client.wait_state(lambda s: s.stop_number == 1 and s.owner == "nobody")
    release.set()
    result = final(h, move_id)
    assert (result.status, result.reason) == (
        "cancelled",
        f"STOP from thor_terminal: {key}",
    )
    assert h.sent() == PATH[:6]
    assert state.stop_level == "stop"
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.running  # Ctrl+C is a STOP: the robot app keeps running


def test_after_a_stop_a_typed_rest_and_a_new_task_work_again(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h, 20, LONG_PATH)
    stop(h)
    release.set()
    assert final(h, move_id).status == "cancelled"
    assert h.sent() == LONG_PATH[:20]
    rest = final(h, h.client.rest())
    assert rest.status == "completed", rest.reason
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    h.new_task()
    assert h.client.state.stop_level == "none"
    assert h.client.run_move(PATH, start_q=rest.end_q).status == "completed"


def test_after_a_stop_in_mid_close_the_operator_opens_the_jaws_and_then_rests(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    stop(h)
    assert final(h, jaw_id).status == "cancelled"
    # Jaws that stopped short of their target may hold something: no rest yet.
    held = final(h, h.client.rest())
    assert "the jaws are not proven empty (loaded)" in held.reason
    unconfirmed = final(h, h.client.open_jaws(confirmed=False))
    assert (unconfirmed.status, unconfirmed.reason) == (
        "refused",
        "confirm the jaw opening",
    )
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    open_id = h.client.open_jaws()
    assert final(h, open_id).status == "completed"
    assert statuses(h, open_id) == ["accepted", "completed"]
    openings = [command.opening for command in h.arm.jaw_commands[1:]]
    assert openings == pytest.approx([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    assert h.arm.sent == []
    rest = final(h, h.client.rest())
    assert rest.status == "completed", rest.reason
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.torque_on() and h.arm.released_at == []


@pytest.mark.parametrize(
    ("sender", "wait_goes_on"),
    [
        ("agent", True),
        ("console", False),
        ("controller", False),
        ("thor_terminal", False),
    ],
)
def test_only_a_stop_from_a_person_cancels_the_assistance_wait(
    robot_app, sender, wait_goes_on
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    state = stop(h, sender)
    assert (state.assistance_wait_s is not None) is wait_goes_on
    assert h.client.wait_event("stop").data["wait_cancelled"] is not wait_goes_on
    if wait_goes_on:
        run_wait_out(h)
        settle(h)
        assert len(events(h, "expired")) == 1
        assert len(h.arm.sent) == RETURN_SETPOINTS
        assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    else:
        h.advance(125.0)
        settle(h)
        assert events(h, "expired") == []
        assert_holding_at(h, RAISED)
    assert h.arm.torque_on() and h.arm.released_at == []


# The assistance wait


def test_the_wait_counts_down_and_rings_at_both_terminals(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    helped = ask_for_help(h)
    assert helped.detail == {
        "owner": "nobody",
        "owner_number": 2,
        "timed_return": True,
        "assistance_s": 120.0,
    }
    h.advance(50.0)
    state = h.client.wait_state(
        lambda s: s.assistance_wait_s is not None and s.assistance_wait_s < 70.5
    )
    assert (state.owner, state.owner_number, state.motion) == ("nobody", 2, "rest_hold")
    assert state.assistance_wait_s > 69.0
    run_wait_out(h)
    settle(h)
    received = h.client.events
    assert [event.seq for event in received] == list(range(1, len(received) + 1))
    # The Thor terminal's listener heard the same lines in the same order.
    assert [(event.kind, event.text) for event in received] == [
        (kind, text) for kind, text, _ in h.local_events
    ]
    # A line every 10 s, a bell at 30 s, at 10 s and on each of the last 5 s.
    assert [event.kind for event in received] == (
        ["owner", "help_needed"]
        + ["countdown"] * 10
        + ["bell", "countdown", "countdown", "bell"]
        + ["bell"] * 5
        + ["expired", "rest", "rest"]
    )
    bells = events(h, "bell")
    assert [bell.text for bell in bells] == [
        f"ARM WILL MOVE TO REST in {mark:g} s" for mark in (30, 10, 5, 4, 3, 2, 1)
    ]
    # Each rang at the first step of the wait at or under its mark.
    for bell, mark in zip(bells, (30, 10, 5, 4, 3, 2, 1)):
        assert mark - 1.0 < bell.data["remaining_s"] <= mark
    countdowns = [event.text for event in events(h, "countdown")]
    assert countdowns[0].startswith("assistance wait: 120 s left, then the arm moves")
    assert countdowns[-1] == "assistance wait: 10 s left"


def test_help_needed_during_a_move_ends_it_in_a_hold_and_starts_the_wait(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h, 20, LONG_PATH)
    ask_for_help(h, "the model gave up")
    release.set()
    result = final(h, move_id)
    assert result.status == "cancelled"
    assert h.sent() == LONG_PATH[:20]  # the next setpoint was not sent
    assert h.arm.last_target()[0] == LONG_PATH[19]
    state = h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    assert (state.owner, state.owner_number, state.stop_level) == ("nobody", 2, "none")
    assert h.arm.torque_on() and h.arm.released_at == []


def test_gate_the_wait_makes_exactly_one_return_attempt_and_it_ends_powered(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    run_wait_out(h)
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.sent[-1][0] == REST
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    # The state of the wait's last second also shows rest_hold and nobody: the
    # one waited for is a state from after the wait.
    state = h.client.wait_state(
        lambda s: (
            (s.motion, s.owner, s.assistance_wait_s) == ("rest_hold", "nobody", None)
        )
    )
    assert (state.torque_on, state.assistance_wait_s) == (True, None)
    assert h.arm.torque_on() and h.arm.released_at == []
    # Nothing follows by itself: no second wait, no second return.
    h.advance(130.0)
    settle(h)
    assert len(events(h, "expired")) == 1
    assert [event.data.get("status") for event in events(h, "rest")] == [
        None,
        "at_rest",
    ]
    assert len(h.arm.sent) == RETURN_SETPOINTS


def test_a_timed_return_that_is_refused_leaves_the_arm_holding_and_powered(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    close_on_an_object(h)
    ask_for_help(h)
    run_wait_out(h)
    settle(h)
    (refusal,) = events(h, "rest")
    assert refusal.data["status"] == "refused"
    assert "the jaws are not proven empty (loaded)" in refusal.data["reason"]
    assert_holding_at(h, RAISED)
    assert h.arm.gripper_read() == pytest.approx(0.4)  # the object is still held
    h.client.wait_state(lambda s: s.owner == "nobody" and s.assistance_wait_s is None)
    # One attempt: nothing automatic follows.
    h.advance(130.0)
    settle(h)
    assert len(events(h, "expired")) == 1 and len(events(h, "rest")) == 1
    assert_holding_at(h, RAISED)


@pytest.mark.parametrize(
    ("task_allows", "auto_return"),
    [(False, True), (True, False)],
    ids=["the task forbids it", "the configuration forbids it"],
)
def test_without_the_timed_return_the_arm_keeps_holding_when_the_wait_ends(
    make_robot_app, config, task_allows, auto_return
):
    h = make_robot_app(config=with_config(config, rest={"auto_return": auto_return}))
    h.arm.set_pose(RAISED)
    h.new_task(timed_return=task_allows)
    assert ask_for_help(h).detail["timed_return"] is False
    run_wait_out(h)
    settle(h)
    assert events(h, "bell")[0].text.startswith("ARM WILL KEEP HOLDING")
    assert [event.text for event in events(h, "expired")] == [
        "assistance wait over: the arm keeps holding"
    ]
    assert events(h, "rest") == []
    assert_holding_at(h, RAISED)
    h.advance(130.0)
    settle(h)
    assert_holding_at(h, RAISED)


# A STOP that races the end of the assistance wait


@pytest.fixture
def waiting(robot_app):
    """A raised arm in an assistance wait that runs out at the next clock second."""
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    h.advance(119.0)
    assert h.heard("expired") == []
    return h


def raise_stop(h, sender: str) -> None:
    """A STOP that is in force when this returns: by link, or typed at the Thor."""
    if sender == "thor_terminal":
        assert h.app.stop("typed stop") == 1
    else:
        h.client.stop("operator pressed stop", sender=sender)
        h.client.wait_state(lambda s: s.stop_number == 1)


def assert_no_return_began(h) -> None:
    settle(h)
    assert_holding_at(h, RAISED)
    state = h.client.wait_state(
        lambda s: s.stop_number == 1 and s.assistance_wait_s is None
    )
    assert (state.owner, state.torque_on) == ("nobody", True)
    # Nothing follows later either.
    h.advance(130.0)
    settle(h)
    assert_holding_at(h, RAISED)


def test_gate_a_stop_typed_as_the_wait_runs_out_leaves_no_timed_return(waiting):
    h = waiting
    # The dispatcher is held between two steps of the wait, outside the app's lock.
    reached, release = h.hold_call("faults")
    assert reached.wait(WAIT_S)
    h.clock.advance(1.5)  # the wait is over at its next step
    assert h.app.stop("typed stop") == 1  # takes the app's lock before that step
    release.set()
    assert_no_return_began(h)
    assert events(h, "expired") == [] and events(h, "rest") == []


def test_gate_a_stop_by_link_as_the_wait_runs_out_leaves_no_timed_return(waiting):
    h = waiting
    reached, release = h.hold_call("faults")
    assert reached.wait(WAIT_S)
    h.clock.advance(1.5)  # the wait is over at its next step
    h.client.stop()  # the dispatcher takes it before it steps the wait again
    release.set()
    assert_no_return_began(h)
    assert events(h, "expired") == [] and events(h, "rest") == []


def test_gate_the_wait_cannot_run_out_while_a_stop_is_being_raised(waiting):
    h = waiting
    held, go = threading.Event(), threading.Event()
    real = h.checkers.agent.halt

    def halt(reason: str) -> int:
        held.set()  # the stop number is up; the wait is not cancelled yet
        go.wait(WAIT_S)
        return real(reason)

    h.checkers.agent.halt = halt
    stopping = threading.Thread(target=h.app.stop, args=("typed stop",), daemon=True)
    stopping.start()
    assert held.wait(WAIT_S)
    h.clock.advance(1.5)  # the wait would be over at its next step
    h.app.wait_step(BLOCKED_S)  # time for that step, if the STOP did not bar it
    assert h.heard("expired") == []
    go.set()
    stopping.join(WAIT_S)
    assert_no_return_began(h)
    assert events(h, "expired") == [] and events(h, "rest") == []
    assert h.client.wait_event("stop").data["wait_cancelled"] is True


@pytest.mark.parametrize("sender", ["console", "thor_terminal"])
@pytest.mark.parametrize(
    "read",
    [1, 3],
    ids=["while the return is checked", "just before the first setpoint"],
)
def test_gate_a_stop_between_wait_expired_and_the_first_setpoint_sends_no_setpoint(
    waiting, read, sender
):
    h = waiting
    # The timed return reads every joint twice for its checks; the third full
    # read is the move's own, right before its first setpoint.
    reached, release = h.hold_call("read_all", call=read)
    h.advance_until(lambda: h.heard("expired"))  # the wait's step took the lock
    assert reached.wait(WAIT_S)
    raise_stop(h, sender)
    release.set()
    assert_no_return_began(h)
    assert len(events(h, "expired")) == 1
    ended = events(h, "rest")[-1]
    assert ended.data["status"] in ("refused", "stopped")
    assert "STOP" in ended.data["reason"]


@pytest.mark.parametrize("sender", ["console", "thor_terminal"])
def test_gate_a_wait_that_runs_out_behind_other_motion_work_starts_no_timed_return(
    waiting, sender
):
    h = waiting
    # A release is being checked when the wait runs out: the timed return is
    # refused then and there, not queued behind the release.
    reached, release = h.hold_call("read_all")
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    h.advance_until(lambda: h.heard("expired"))
    assert h.app._busy() == "busy: a torque release is running"
    raise_stop(h, sender)
    release.set()
    assert final(h, release_id).status == "refused"
    assert_no_return_began(h)
    assert len(events(h, "expired")) == 1
    # The wait's one attempt was refused when it ran out: no return was tried.
    assert [(event.text, event.data["status"]) for event in events(h, "rest")] == [
        ("no timed return to rest: busy: a torque release is running", "refused")
    ]


# Return to rest refused: nothing changes


def assert_rest_refused(h, reason: str, confirmed: bool = True) -> None:
    """A typed rest is refused for this reason and changes nothing."""
    before = now_state(h)
    sent, jaws = len(h.arm.sent), len(h.arm.jaw_commands)
    rest_id = h.client.rest(confirmed=confirmed)
    result = final(h, rest_id)
    assert result.status == "refused"
    assert reason in result.reason
    assert facts(state_after(h, rest_id)) == facts(before)
    h.idle()
    assert (len(h.arm.sent), len(h.arm.jaw_commands)) == (sent, jaws)
    assert h.arm.torque_on() and h.arm.released_at == []
    settle(h)
    assert reason in h.client.events[-1].text  # the operator is told why


def test_gate_a_return_to_rest_is_refused_with_loaded_jaws(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    close_on_an_object(h)
    assert_rest_refused(h, "the jaws are not proven empty (loaded)")
    assert h.arm.gripper_read() == pytest.approx(0.4)


def test_gate_a_return_to_rest_is_refused_with_jaws_that_were_never_referenced(
    make_robot_app,
):
    h = make_robot_app(pose=RAISED)  # away from rest at start: no jaw reference
    assert not h.client.state.gripper.referenced
    assert_rest_refused(h, "the jaws are not proven empty (unknown)")


def test_gate_a_return_to_rest_is_refused_when_a_jaw_read_fails(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.arm.fail_gripper_reads(1)
    assert_rest_refused(h, "the jaws are not proven empty (unknown): a jaw read failed")


def test_gate_a_return_to_rest_is_refused_while_a_fault_stop_is_latched(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    assert_rest_refused(h, LATCHED)


def test_a_return_to_rest_is_refused_while_the_arm_holds_at_a_contact(robot_app):
    h = robot_app
    h.new_task()
    assert contact(h).reason == "contact"
    assert_rest_refused(h, "the arm is holding at a contact")


def test_gate_a_return_to_rest_is_refused_outside_the_recovery_region(robot_app):
    h = robot_app
    h.arm.set_pose([2.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert_rest_refused(h, "joint index 0 at 2.6000 rad is outside the tested recovery")


def test_gate_a_return_to_rest_is_refused_when_a_joint_read_fails(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.arm.read_all
    calls = []

    def read_all() -> list[JointReading]:
        calls.append(1)
        if len(calls) == 1:  # the return's first full read
            raise ReadFailed("joint index 3: the read failed")
        return real()

    h.arm.read_all = read_all
    assert_rest_refused(h, "a joint read failed: joint index 3: the read failed")
    assert len(calls) == 1  # nothing was read again and nothing was retried


def test_a_return_to_rest_is_refused_when_two_reads_disagree(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.arm.read_all
    calls = []

    def read_all() -> list[JointReading]:
        readings = real()
        calls.append(1)
        if len(calls) == 1:  # the arm is found elsewhere 0.2 s later
            h.arm.set_pose([RAISED[0] + 0.05, *RAISED[1:]])
        return readings

    h.arm.read_all = read_all
    assert_rest_refused(h, "two joint reads 0.2 s apart disagree: joint index 0")


def test_a_return_to_rest_typed_without_the_confirmation_is_refused(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    assert_rest_refused(h, "confirm the path is clear", confirmed=False)


def test_a_return_to_rest_is_refused_while_a_jaw_command_runs(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    assert_rest_refused(h, "a jaw command is running")


def test_a_return_to_rest_is_refused_while_the_gripper_tip_is_blocked(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    low = "the gripper tip is 0.010 m above the table, under the clearance"
    h.checkers.recovery.set_tip_hook(lambda q, region: low)
    assert_rest_refused(h, low)


def start_a_jaw_command(h, still_closing: bool) -> bool:
    """The agent closes the jaws; True if the robot app took the command.

    The close ends on an object, or is held part-way while still closing.
    """
    if still_closing:
        waiting = h.hold_jaws_at(0.6)
    else:
        h.arm.jaw_object(0.4)
    mode = "plain" if still_closing else "close_until_contact"
    jaw_id = h.client.gripper(0.0, mode=mode)
    taken = h.client.wait_result(jaw_id, final=False).status == "accepted"
    if taken and still_closing:
        assert waiting.wait(WAIT_S)
    elif taken:
        assert final(h, jaw_id).status == "completed"
    return taken


jaw_commands = pytest.mark.parametrize(
    "still_closing", [False, True], ids=["closed on an object", "still closing"]
)


@jaw_commands
def test_a_return_to_rest_does_not_start_when_a_jaw_command_comes_while_it_is_checked(
    robot_app, still_closing
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    # The jaws were proven empty; the return now takes its second full read.
    reached, release = h.hold_call("read_all", call=2)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    taken = start_a_jaw_command(h, still_closing)
    release.set()
    result = final(h, rest_id)
    # One of the two must be refused: the arm never returns with jaws that are
    # closing or closed on something.
    if taken:
        assert result.status == "refused", "the arm returned to rest all the same"
        assert_holding_at(h, RAISED)
    else:
        assert h.arm.jaw_commands == []


# Return to rest accepted, and the torque release


def test_gate_a_return_to_rest_is_one_straight_move_at_recovery_speed_and_ends_powered(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    rest_id = h.client.rest()
    result = final(h, rest_id)
    assert statuses(h, rest_id) == ["accepted", "completed"]
    sent = h.arm.sent
    assert len(sent) == RETURN_SETPOINTS
    assert sent[-1][0] == REST
    # One straight line in joint space: every setpoint is the start pose scaled.
    shares = [q[0] / RAISED[0] for q, _ in sent]
    assert shares == sorted(shares, reverse=True) and shares[0] < 1.0
    for (q, _), share in zip(sent, shares):
        assert q == pytest.approx([share * at for at in RAISED], abs=1e-12)
    # One setpoint per period, never faster than the recovery speed.
    times = [t for _, t in sent]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert gaps == pytest.approx([PERIOD_S] * len(gaps), abs=1e-9)
    poses = [RAISED] + [q for q, _ in sent]
    fastest = max(
        abs(b - a) / PERIOD_S
        for q0, q1 in zip(poses, poses[1:])
        for a, b in zip(q0, q1)
    )
    limit = h.config.rest.recovery_speed
    assert 0.85 * limit < fastest <= limit
    assert limit == h.config.limits.recovery.max_joint_speed
    # It ends at rest with the motors on, and nobody owns the arm.
    assert result.end_q == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    state = state_after(h, rest_id, owner="nobody")
    assert (state.motion, state.torque_on) == ("rest_hold", True)
    assert h.arm.torque_on() and h.arm.released_at == []
    settle(h)
    texts = [event.text for event in events(h, "rest")]
    assert texts == [
        "return to rest: the arm moves for 4.2 s (largest joint travel 0.20 rad)",
        "return to rest: the arm is at rest, motors on",
    ]


def test_while_the_arm_returns_to_rest_the_agent_can_neither_move_nor_grip(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    state = h.client.wait_state(lambda s: s.owner == "recovery")
    assert state.limit_set_version == h.config.limits.recovery.version
    for command_id in (
        h.client.move(PATH, start_q=RAISED, owner_number=1),
        h.client.gripper(0.0, owner_number=1),
        h.client.request_owner("agent"),
    ):
        assert final(h, command_id).status == "refused"
    release.set()
    assert final(h, rest_id).status == "completed"
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.jaw_commands == []


def test_stop_during_a_return_to_rest_ends_it_in_a_hold(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    stop(h)
    release.set()
    result = final(h, rest_id)
    assert result.status == "cancelled"
    assert result.reason.startswith("STOP from console: operator pressed stop")
    assert len(h.arm.sent) == 5  # the next setpoint was not sent
    state = state_after(h, rest_id, owner="nobody")
    assert (state.motion, state.torque_on) == ("hold", True)
    assert max(abs(at) for at in h.arm.pose()) > h.config.rest.rest_tol_rad
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_contact_during_a_return_to_rest_ends_it_in_a_hold_and_is_not_retried(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.arm.block_joint(0, 0.15)  # something lies in the way of joint index 0
    rest_id = h.client.rest()
    result = final(h, rest_id)
    assert result.status == "failed"
    assert "contact: joint index 0 trails" in result.reason
    sent = len(h.arm.sent)
    assert 0 < sent < RETURN_SETPOINTS
    ((held_at, _),) = h.arm.holds  # held where it was measured
    assert held_at[0] == pytest.approx(0.15, abs=1e-6)
    state = state_after(h, rest_id, owner="nobody")
    assert (state.stop_level, state.motion) == ("none", "hold")
    # Nothing is retried, and the arm stays at the contact until it is moved.
    h.arm.block_joint(0, None)
    again = final(h, h.client.rest())
    assert (again.status, again.reason) == (
        "refused",
        "the arm is holding at a contact",
    )
    h.idle()
    assert len(h.arm.sent) == sent
    assert h.arm.torque_on() and h.arm.released_at == []


def test_no_robot_state_reports_a_rest_as_ended_with_the_owner_from_before_its_end(
    robot_app,
):
    h = robot_app
    put_on_the_link: list[RobotState] = []
    real = h.robot_end.send_latest

    def send_latest(msg_type: int, payload: bytes) -> None:
        if msg_type == RobotState.TYPE:
            put_on_the_link.append(decode(msg_type, payload))
        real(msg_type, payload)

    h.robot_end.send_latest = send_latest
    h.arm.set_pose(RAISED)
    at_last_setpoint, finish = h.hold_move_after(RETURN_SETPOINTS)
    rest_id = h.client.rest()
    assert at_last_setpoint.wait(WAIT_S)
    # The next ROBOT_STATE is being put together, the control owner already
    # read for it, while the return ends and its RESULT goes out.
    building, go_on = h.hold_call("gripper_last_target")
    h.clock.advance(0.06)
    assert building.wait(WAIT_S)
    finish.set()
    assert final(h, rest_id).status == "completed"
    go_on.set()
    assert state_after(h, rest_id, owner="nobody").motion == "rest_hold"
    wrong = [
        (state.seq, state.owner)
        for state in put_on_the_link
        if (state.last_result or {}).get("command_id") == rest_id
        and state.owner != "nobody"
    ]
    assert wrong == [], (
        "a ROBOT_STATE shows the rest as ended and its owner from before"
    )


def test_a_rest_typed_at_the_thor_terminal_asks_for_the_confirmation(make_robot_app):
    asked = []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return len(asked) > 1  # no the first time, yes after that

    h = make_robot_app(confirm=confirm)
    h.arm.set_pose(RAISED)
    done = Done()
    assert h.app.request_rest(None, done) == "confirm the path is clear"
    assert done.wait().status == "refused"
    h.idle()
    assert_holding_at(h, RAISED)
    done = Done()
    assert h.app.request_rest(None, done) is None
    assert done.wait().status == "completed"
    assert len(asked) == 2 and "path" in asked[0]
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.torque_on() and h.arm.released_at == []


def test_release_away_from_rest_is_refused_and_the_motors_stay_on(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    release_id = h.client.release()
    result = final(h, release_id)
    assert statuses(h, release_id) == ["accepted", "refused"]
    assert result.reason.startswith("the arm is not at rest: joint index 0")
    state = state_after(h, release_id)
    assert state.torque_on and state.motion != "rest_unpowered"
    assert_holding_at(h, RAISED)


def test_release_at_rest_with_loaded_jaws_is_refused(robot_app):
    h = robot_app
    h.new_task()
    close_on_an_object(h)
    result = final(h, h.client.release())
    assert result.status == "refused"
    assert "the jaws are not proven empty (loaded)" in result.reason
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.arm.gripper_read() == pytest.approx(0.4)


@jaw_commands
def test_release_cuts_no_torque_when_a_jaw_command_comes_while_it_is_checked(
    robot_app, still_closing
):
    h = robot_app
    h.new_task()
    # The jaws were proven empty; the release now takes its second full read.
    reached, release = h.hold_call("read_all", call=2)
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    taken = start_a_jaw_command(h, still_closing)
    release.set()
    result = final(h, release_id)
    # One of the two must be refused: torque is cut only with jaws proven empty.
    if taken:
        assert result.status == "refused", "torque was released all the same"
        assert h.arm.torque_on() and h.arm.released_at == []
    else:
        assert h.arm.jaw_commands == []


@pytest.mark.parametrize("sender", ["console", "thor_terminal"])
def test_a_stop_while_a_release_is_checked_keeps_the_motors_on(robot_app, sender):
    h = robot_app
    # The release has read the joints at rest once and takes its second read.
    reached, release = h.hold_call("read_all", call=2)
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    raise_stop(h, sender)
    release.set()
    result = final(h, release_id)
    assert (result.status, result.reason) == (
        "refused",
        "a STOP was raised while the release was checked",
    )
    h.idle()
    assert h.arm.torque_on() and h.arm.released_at == []
    assert state_after(h, release_id).motion == "rest_hold"


def test_release_at_measured_rest_switches_the_motors_off_for_the_rest_of_the_run(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    assert final(h, h.client.rest()).status == "completed"
    assert h.arm.released_at == []  # the return alone released nothing
    h.new_task()
    release_id = h.client.release()
    assert final(h, release_id).status == "completed"
    assert statuses(h, release_id) == ["accepted", "completed"]
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on and not h.arm.torque_on()
    # Every motion command is refused from here on, whoever owns the arm.
    for owner in ("agent", "nobody"):
        for command_id in (
            h.client.move(PATH, start_q=REST),
            h.client.gripper(0.5),
            h.client.open_jaws(),
            h.client.rest(),
            h.client.release(),
            h.client.request_owner("agent"),
        ):
            refused = final(h, command_id)
            assert refused.status == "refused", (owner, refused)
        if owner == "agent":
            assert final(h, h.client.request_owner("nobody")).status == "completed"
            h.client.wait_state(lambda s: s.owner == "nobody")
    assert refused.reason == "the motors are off; restart the robot app"
    h.advance(5.0)
    h.idle()
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.jaw_commands == [] and len(h.arm.released_at) == 1
    assert now_state(h).stop_level == "none"  # no fault for the silent joints


# Late setpoints


def slow_read_after(h, count: int, seconds: float) -> None:
    """Make the first joint read after setpoint number `count` take this much longer."""
    real = h.arm.read_joint
    slowed = []

    def read_joint(joint: int) -> JointReading:
        if len(h.arm.sent) == count and not slowed:
            slowed.append(joint)
            h.clock.advance(seconds)
        return real(joint)

    h.arm.read_joint = read_joint


def test_gate_a_small_delay_shifts_the_rest_of_the_move_and_skips_nothing(robot_app):
    h = robot_app
    h.new_task()
    slow_read_after(h, 10, 0.03)
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "completed", result.reason
    assert h.sent() == PATH  # none skipped, none sent twice, in order
    times = [t for _, t in h.arm.sent]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    # Setpoint 11 left 18 ms late; every later one moved back with it and none
    # followed sooner than one period to catch up.
    assert gaps[9] == pytest.approx(0.038, abs=1e-9)
    assert gaps[:9] + gaps[10:] == pytest.approx([PERIOD_S] * 23, abs=1e-9)
    assert times[-1] - times[0] == pytest.approx(24 * PERIOD_S + 0.018, abs=1e-9)
    assert state_after(h, result.command_id).health["missed_deadlines"] == 1


def test_gate_a_large_delay_ends_the_move_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    slow_read_after(h, 10, 0.15)
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "failed"
    assert result.reason.startswith("setpoint 10 is 0.1")
    assert "late, over the limit of 0.1 s" in result.reason
    assert result.detail == {"outcome": "late", "sent": 10, "total": 25}
    assert h.sent() == PATH[:10]  # nothing after the delay, nothing skipped before
    assert h.arm.last_target()[0] == PATH[9]  # the arm holds its last setpoint
    assert h.arm.torque_on() and h.arm.released_at == []
    state = state_after(h, result.command_id)
    assert (state.motion, state.stop_level, state.owner) == ("hold", "none", "agent")
    # The move is not resumed; a new one from the measured pose is taken.
    onward = h.client.plan_straight(result.end_q, PATH[-1], 0.2)
    assert h.client.run_move(onward, start_q=result.end_q).status == "completed"


# Faults


def test_gate_a_motor_fault_latches_until_a_reset_with_the_current_stop_number(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    h.arm.set_fault("motor 3 overheated")
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == "motor 3 overheated"
    assert (state.stop_number, state.owner, state.assistance_wait_s) == (
        1,
        "nobody",
        None,
    )
    assert h.client.wait_event("fault").data["stop_number"] == 1
    # Commands are refused while the fault-stop is latched.
    for command_id in (
        h.client.request_owner("agent"),
        h.client.move(PATH, start_q=RAISED),
        h.client.gripper(0.0),
        h.client.rest(),
        h.client.release(),
        h.client.open_jaws(),
        h.client.shutdown(),
    ):
        refused = final(h, command_id)
        assert (refused.status, refused.reason) == ("refused", LATCHED)
    # RESET needs the current stop number, and an arm that reports no fault.
    old = final(h, h.client.reset(stop_number=0))
    assert old.reason == "stop number 0 does not match the current 1"
    still = final(h, h.client.reset())
    assert still.reason == "the arm still reports a fault: motor 3 overheated"
    assert h.client.state.stop_level == "fault"
    h.arm.clear_faults()
    reset_id = h.client.reset()
    assert final(h, reset_id).status == "completed"
    state = state_after(h, reset_id)
    assert (state.stop_level, state.stop_number, state.owner) == ("none", 1, "nobody")
    assert state.assistance_wait_s is None
    # The reset started no motion and no wait.
    h.advance(130.0)
    settle(h)
    assert events(h, "expired") == [] and events(h, "rest") == []
    assert_holding_at(h, RAISED)
    assert h.arm.jaw_commands == []
    assert len(events(h, "fault")) == 1  # one fault-stop for the one fault
    # A new task works again.
    h.new_task()
    up = h.client.plan_straight(RAISED, [0.25, -0.1, 0.1, 0.0, 0.05, -0.05], 0.2)
    assert h.client.run_move(up, start_q=RAISED).status == "completed"


def test_a_motor_fault_during_a_move_keeps_the_next_setpoint_back(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.arm.set_fault("motor 3 overheated")
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert (state.owner, state.stop_number) == ("nobody", 1)
    release.set()
    result = final(h, move_id)
    assert (result.status, result.reason) == (
        "cancelled",
        "fault-stop: motor 3 overheated",
    )
    assert h.sent() == PATH[:6]
    assert h.arm.last_target()[0] == PATH[5]
    assert h.arm.torque_on() and h.arm.released_at == []


def test_lost_joint_feedback_in_a_hold_latches_a_fault_stop(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.arm.fail_reads(2, 10_000)
    h.advance(0.8, step_s=0.02)
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost for 0.")
    assert (state.stop_number, state.torque_on) == (1, True)
    assert_holding_at(h, RAISED)
    assert final(h, h.client.rest()).reason.startswith("fault-stop latched")
    h.idle()
    assert_holding_at(h, RAISED)


def test_lost_joint_feedback_during_a_move_latches_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()
    real = h.arm.send_target

    def send_target(q: Sequence[float]) -> None:
        real(q)
        if len(h.arm.sent) == 1:
            h.arm.fail_reads(1, 10_000)

    h.arm.send_target = send_target
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "failed"
    assert result.reason.startswith("fault: joint feedback lost: joint index 1")
    assert 0 < result.detail["sent"] < len(PATH)
    assert h.sent() == PATH[: result.detail["sent"]]
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert (state.owner, state.stop_number, state.torque_on) == ("nobody", 1, True)
    assert h.arm.torque_on() and h.arm.released_at == []
    again = h.client.run_move(PATH, start_q=REST)
    assert again.reason.startswith("fault-stop latched: joint feedback lost")


# Contact


def test_a_blocked_joint_ends_the_move_in_a_hold_at_the_measured_pose(robot_app):
    h = robot_app
    h.new_task()
    result = contact(h)
    assert (result.status, result.reason) == ("failed", "contact")
    assert result.detail["outcome"] == "contact"
    assert result.detail["contact"].startswith("contact: joint index 0 trails")
    sent = result.detail["sent"]
    assert 0 < sent < len(LONG_PATH)
    assert h.sent() == LONG_PATH[:sent]
    # The arm holds where it was measured, so it does not keep pressing.
    ((held_at, _),) = h.arm.holds
    assert held_at[0] == pytest.approx(0.05, abs=1e-6)
    assert h.arm.last_target()[0] == held_at
    assert result.end_q == pytest.approx(held_at, abs=1e-6)
    # The agent and recovery checkers were told where the contact was.
    assert h.checkers.agent.contact_spots == [result.end_q]
    assert h.checkers.recovery.contact_spots == [result.end_q]
    # Not a fault-stop: nothing is latched and the agent still owns the arm.
    state = state_after(h, result.command_id)
    assert (state.stop_level, state.stop_number, state.owner) == ("none", 0, "agent")
    assert (state.motion, state.torque_on) == ("hold", True)
    assert h.arm.torque_on() and h.arm.released_at == []


@pytest.mark.parametrize("ended_by", ["a completed move", "clear_contact_spots"])
def test_a_return_to_rest_is_refused_until_the_contact_hold_has_ended(
    robot_app, ended_by
):
    h = robot_app
    h.new_task()
    met = contact(h)
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == (
        "refused",
        "the arm is holding at a contact",
    )
    if ended_by == "a completed move":
        away = h.client.plan_straight(met.end_q, [q - 0.02 for q in met.end_q], 0.2)
        moved = h.client.run_move(away, start_q=met.end_q)
        assert moved.status == "completed", moved.reason
    else:
        cleared = final(h, h.client.checker_state("clear_contact_spots"))
        assert cleared.status == "completed"
    rest = final(h, h.client.rest())
    assert rest.status == "completed", rest.reason
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.arm.torque_on() and h.arm.released_at == []


# Robot connection lost


def assert_nothing_restarts(h, sent: int, expired: int = 0) -> None:
    """Long after the connection was lost the arm was sent nothing more."""
    h.advance(130.0)
    h.idle()
    assert len(h.arm.sent) == sent
    assert len(h.heard("expired")) == expired
    assert len(h.heard("connection")) == 1
    assert h.app.snapshot().owner == "nobody"
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.running


def test_connection_lost_while_idle_changes_nothing(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.robot_end.break_link()
    _, text, data = h.wait_local_event("connection")
    assert text.startswith("robot connection lost: the arm holds")
    assert data == {"owner_before": "nobody", "wait_cancelled": False}
    assert h.app.snapshot().owner_number == 0
    assert_nothing_restarts(h, sent=0)
    assert h.arm.pose() == RAISED


def test_connection_lost_while_the_agent_moves_ends_the_move_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    _, release = held_move(h)
    h.robot_end.break_link()
    _, text, data = h.wait_local_event("connection")
    assert "the agent's ownership ended" in text
    assert data == {"owner_before": "agent", "wait_cancelled": False}
    release.set()
    h.idle()
    assert h.sent() == PATH[:6]  # the next setpoint was not sent
    assert h.arm.last_target()[0] == PATH[5]
    state = h.app.snapshot()
    assert (state.owner, state.owner_number, state.stop_number) == ("nobody", 2, 0)
    assert_nothing_restarts(h, sent=6)


def test_connection_lost_during_the_wait_cancels_it(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    h.advance(60.0)
    h.robot_end.break_link()
    _, text, data = h.wait_local_event("connection")
    assert "the assistance wait is cancelled" in text
    assert data == {"owner_before": "nobody", "wait_cancelled": True}
    assert_nothing_restarts(h, sent=0)
    assert h.arm.pose() == RAISED


def test_connection_lost_during_a_timed_return_ends_it_in_a_hold(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    reached, release = h.hold_move_after(5)
    h.advance_until(lambda: h.heard("expired"))
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    h.idle()
    assert len(h.arm.sent) == 5  # the next setpoint was not sent
    ended = h.heard("rest")[-1][2]
    assert ended["status"] == "stopped"
    assert "robot connection lost" in ended["reason"]
    assert max(abs(at) for at in h.arm.pose()) > h.config.rest.rest_tol_rad
    assert_nothing_restarts(h, sent=5, expired=1)


def test_connection_lost_during_a_rest_typed_at_the_thor_terminal_lets_it_go_on(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    assert done.wait().status == "completed"
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.heard("rest")[-1][2]["status"] == "at_rest"
    assert_nothing_restarts(h, sent=RETURN_SETPOINTS)


def test_with_the_connection_down_stop_and_rest_still_work_at_the_thor_terminal(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    assert h.app.stop("typed stop") == 1
    refused = Done()
    assert h.app.request_rest(False, refused) == "confirm the path is clear"
    assert refused.wait().status == "refused"
    h.idle()
    assert_holding_at(h, RAISED)
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().status == "completed"
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.torque_on() and h.arm.released_at == []


# Shutdown


def test_shutdown_by_link_returns_to_rest_releases_there_and_ends_the_app(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    shutdown_id = h.client.shutdown()
    result = final(h, shutdown_id)
    assert statuses(h, shutdown_id) == ["accepted", "completed"]
    assert result.end_q == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.wait_closed(WAIT_S) and not h.app.running
    assert len(h.arm.sent) == RETURN_SETPOINTS
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert not h.arm.torque_on()
    assert h.client.wait_event("release").text.startswith("torque released at rest")
    assert h.client.wait_event("shutdown").data == {"status": "completed"}


def test_shutdown_at_the_thor_terminal_does_the_same(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    done = Done()
    assert h.app.request_shutdown(True, done) is None
    assert done.wait().status == "completed"
    assert h.app.wait_closed(WAIT_S) and not h.app.running
    assert len(h.arm.sent) == RETURN_SETPOINTS
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert not h.arm.torque_on()


@pytest.mark.parametrize("asked_at", ["the console", "the Thor terminal"])
def test_a_refused_shutdown_keeps_the_app_running_and_the_arm_holding(
    robot_app, asked_at
):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    close_on_an_object(h)
    if asked_at == "the console":
        result = final(h, h.client.shutdown())
    else:
        done = Done()
        assert h.app.request_shutdown(True, done) is None
        result = done.wait()
    assert result.status == "refused"
    assert "the jaws are not proven empty (loaded)" in result.reason
    h.idle()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert_holding_at(h, RAISED)
    assert h.arm.gripper_read() == pytest.approx(0.4)
    # The app still takes commands.
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    assert "keeps running and the arm holds" in h.client.wait_event("shutdown").text


def test_a_shutdown_stopped_on_the_way_releases_nothing_and_keeps_the_app_running(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    shutdown_id = h.client.shutdown()
    assert reached.wait(WAIT_S)
    stop(h)
    release.set()
    result = final(h, shutdown_id)
    assert result.status == "cancelled"
    assert "STOP from console" in result.reason
    assert len(h.arm.sent) == 5
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.running
    assert state_after(h, shutdown_id).motion == "hold"


# The Thor terminal: what the operator types and reads at the robot


def test_ctrl_c_at_the_thor_terminal_is_a_stop_and_the_app_runs_on(robot_app):
    h = robot_app
    thor = h.thor_terminal()
    h.new_task()
    move_id, release = held_move(h)
    thor.terminal.interrupt()
    thor.wait_line("[stop] STOP from thor_terminal: Ctrl+C")
    release.set()
    result = final(h, move_id)
    assert (result.status, result.reason) == (
        "cancelled",
        "STOP from thor_terminal: Ctrl+C",
    )
    assert h.sent() == PATH[:6]  # the next setpoint was not sent
    # A second and a third Ctrl+C are STOPs too, and nothing more.
    thor.terminal.interrupt()
    thor.terminal.interrupt()
    thor.wait_line("[stop] STOP from thor_terminal: Ctrl+C", 3)
    assert h.app.snapshot().stop_number == 3
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.sent() == PATH[:6]
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_stop_typed_at_the_thor_terminal_ends_a_move_and_the_wait(robot_app):
    h = robot_app
    thor = h.thor_terminal()
    h.new_task()
    move_id, release = held_move(h)
    told = thor.ask("stop", answer="[stop] ")
    assert told.startswith("[stop] STOP from thor_terminal: typed stop; the arm holds")
    release.set()
    assert final(h, move_id).status == "cancelled"
    assert h.sent() == PATH[:6]
    h.client.wait_state(lambda s: s.stop_number == 1)  # quoted by the next task
    h.new_task()
    ask_for_help(h)
    thor.wait_line("[countdown] assistance wait: 120 s left, then the arm moves")
    told = thor.ask(" STOP ", answer="[stop] ")
    assert told.endswith("stop number 2; the assistance wait is cancelled")
    h.advance(125.0)
    settle(h)
    assert thor.written("[expired]") == [] and thor.written("[rest]") == []
    assert h.sent() == PATH[:6]
    assert h.arm.torque_on() and h.arm.released_at == []


def test_stop_typed_where_the_thor_terminal_asks_is_a_stop_and_confirms_nothing(
    robot_app,
):
    h = robot_app
    thor = h.thor_terminal()
    h.arm.set_pose(RAISED)
    told = thor.ask("rest", "stop", answer="rest: ")
    assert told == "rest: refused: confirm the path is clear"
    assert len(thor.written("[stop] STOP from thor_terminal: typed stop")) == 1
    assert h.app.snapshot().stop_number == 1
    h.idle()
    assert_holding_at(h, RAISED)


def test_the_thor_terminal_rings_before_the_arm_moves_by_itself(robot_app):
    h = robot_app
    thor = h.thor_terminal()
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h, "bell\a in the reason")
    run_wait_out(h)
    settle(h)
    thor.wait_line("[rest] return to rest: the arm is at rest, motors on")
    # Only the wait's bells ring, each at the start of its line.
    rang = [line for line in thor.lines if "\a" in line]
    assert rang == [
        f"\a[bell] ARM WILL MOVE TO REST in {mark} s"
        for mark in (30, 10, 5, 4, 3, 2, 1)
    ]
    moved = thor.lines.index(
        "[rest] return to rest: the arm moves for 4.2 s (largest joint travel 0.20 rad)"
    )
    assert thor.lines.index(rang[-1]) < moved
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_rest_typed_at_the_thor_terminal_moves_only_after_the_typed_phrase(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    # The question is asked with the robot connection up, and with it down.
    for connection in ("up", "down"):
        if connection == "down":
            h.robot_end.break_link()
            h.wait_local_event("connection")
        thor = h.thor_terminal()
        for answer in ("no", "", "yes", "path clear"):
            told = thor.ask("rest", answer, answer="rest: ")
            assert told == "rest: refused: confirm the path is clear"
        asked = thor.written("Return to rest: is the path to the rest pose clear?")
        assert len(asked) == 4
        h.idle()
        assert_holding_at(h, RAISED)
    assert thor.ask("rest", " Path is clear ", answer="rest: ") == "rest: completed"
    assert len(h.arm.sent) == RETURN_SETPOINTS
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.running
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_release_typed_at_the_thor_terminal_is_refused_away_from_rest(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    thor = h.thor_terminal()
    told = thor.ask("release", answer="release: ")
    assert told.startswith("release: refused: the arm is not at rest: joint index 0")
    h.idle()
    assert_holding_at(h, RAISED)
    assert h.app.running


def test_a_release_typed_at_the_thor_terminal_at_rest_switches_the_motors_off(
    robot_app,
):
    h = robot_app
    h.robot_end.break_link()  # the Thor terminal needs no robot connection
    h.wait_local_event("connection")
    thor = h.thor_terminal()
    assert thor.ask("release", answer="release: ") == "release: completed"
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert not h.arm.torque_on()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.arm.sent == []


def test_quit_typed_at_the_thor_terminal_returns_releases_at_rest_and_ends_the_app(
    robot_app,
):
    h = robot_app
    h.hold_at(RAISED)
    thor = h.thor_terminal()
    # Away from rest a quit moves the arm: it asks first, as a typed rest does.
    told = thor.ask("quit", "no", answer="quit: ")
    assert (
        told == "quit: refused: confirm the path is clear; the robot app keeps running"
    )
    h.idle()
    assert_holding_at(h, RAISED)
    told = thor.ask("quit", "path is clear", answer="quit: ")
    assert told == "quit: completed; the robot app closes"
    assert len(thor.written("Return to rest: is the path to the rest pose clear?")) == 2
    assert h.app.wait_closed(WAIT_S) and not h.app.running
    assert len(h.arm.sent) == RETURN_SETPOINTS
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert not h.arm.torque_on()


def test_a_quit_that_is_refused_leaves_the_app_running_and_the_arm_holding(robot_app):
    h = robot_app
    h.hold_at(RAISED)
    h.new_task()
    close_on_an_object(h)
    thor = h.thor_terminal()
    told = thor.ask("quit", "path is clear", answer="quit: ")
    assert told.startswith("quit: refused: the return to rest ended refused")
    assert "the jaws are not proven empty (loaded)" in told
    assert told.endswith("; the robot app keeps running")
    h.idle()
    assert h.app.running and not h.app.wait_closed(0.0)
    assert_holding_at(h, RAISED)


def test_a_closed_thor_terminal_and_unknown_lines_change_nothing(robot_app):
    h = robot_app
    thor = h.thor_terminal()
    h.new_task()
    move_id, release = held_move(h)
    for line in ("limp", "torque off", "exit", "yes"):
        assert thor.ask(line, answer="unknown command") == (
            f"unknown command {line!r}; type help"
        )
    thor.close()  # the input ends, as when the terminal is closed
    assert thor.lines[-1] == "the Thor terminal's input ended; the robot app runs on"
    release.set()
    assert final(h, move_id).status == "completed"
    assert h.sent() == PATH
    assert h.app.running and h.app.snapshot().stop_number == 0
    assert h.arm.torque_on() and h.arm.released_at == []


# RESULTs and repeated commands


def test_queued_commands_are_accepted_first_and_every_command_ends_once(robot_app):
    h = robot_app
    h.new_task()
    move = h.client.run_move(PATH, start_q=REST)
    jaw_id = h.client.gripper(0.5)
    assert final(h, jaw_id).status == "completed"
    spots_id = h.client.checker_state("clear_contact_spots")
    assert final(h, spots_id).status == "completed"
    nobody_id = h.client.request_owner("nobody")
    assert final(h, nobody_id).status == "completed"
    h.client.wait_state(lambda s: s.owner_number == 2)
    open_id = h.client.open_jaws()
    assert final(h, open_id).status == "completed"
    rest_id = h.client.rest()
    assert final(h, rest_id).status == "completed"
    release_id = h.client.release()
    assert final(h, release_id).status == "completed"
    h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    shutdown_id = h.client.shutdown()
    assert final(h, shutdown_id).status == "completed"
    for command_id in (move.command_id, jaw_id, open_id, rest_id, release_id):
        assert statuses(h, command_id) == ["accepted", "completed"]
    assert statuses(h, shutdown_id) == ["accepted", "completed"]
    for command_id in (spots_id, nobody_id):
        assert statuses(h, command_id) == ["completed"]
    assert h.client.unexpected == []
    assert len(h.arm.released_at) == 1


def test_a_repeated_move_is_answered_again_and_not_run_again(robot_app):
    h = robot_app
    h.new_task()
    move = h.client.run_move(PATH, start_q=REST)
    again = h.client.move(PATH[::-1], start_q=move.end_q, command_id=move.command_id)
    h.client.permit(again, len(PATH) - 1)
    h.client.wait_until(lambda: len(h.client.results(again)) == 3)
    assert h.client.results(again)[2] == move
    h.idle()
    assert h.sent() == PATH  # the arm did not move a second time


def test_a_command_repeated_while_it_runs_is_not_started_twice(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.client.move(PATH, start_q=REST, command_id=move_id)
    h.client.wait_until(lambda: len(h.client.results(move_id)) == 2)
    assert statuses(h, move_id) == ["accepted", "accepted"]
    release.set()
    assert final(h, move_id).status == "completed"
    h.idle()
    assert statuses(h, move_id) == ["accepted", "accepted", "completed"]
    assert h.sent() == PATH


def test_repeated_jaw_owner_and_release_commands_are_not_executed_twice(robot_app):
    h = robot_app
    task = h.new_task()
    h.client.request_owner("agent", command_id=task.command_id)
    jaw_id = h.client.gripper(0.5)
    first = final(h, jaw_id)
    h.client.gripper(0.9, command_id=jaw_id)
    h.client.wait_until(lambda: len(h.client.results(jaw_id)) == 3)
    assert h.client.results(jaw_id)[2] == first
    assert [command.opening for command in h.arm.jaw_commands] == [0.5]
    assert h.client.results(task.command_id) == [task, task]
    assert now_state(h).owner_number == 1  # the arm was not taken a second time
    # A repeated RELEASE is answered with the refusal it got, not tried again.
    release_id = h.client.release()
    refused = final(h, release_id)
    assert "the jaws are not proven empty" in refused.reason
    assert final(h, h.client.gripper(1.0)).status == "completed"
    h.client.release(command_id=release_id)
    h.client.wait_until(lambda: len(h.client.results(release_id)) == 3)
    assert h.client.results(release_id)[2] == refused
    h.idle()
    assert h.arm.torque_on() and h.arm.released_at == []
