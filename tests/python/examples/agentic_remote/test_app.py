# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app through a loopback link and a workstation client, on the simulated arm.

The modules' own rules are in their test files, the fuller scenarios in
test_end_to_end.py.
"""

from __future__ import annotations

import dataclasses
import inspect
import threading
from collections.abc import Callable

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    Event,
    GraspRegion,
    Gripper,
    OwnerRequest,
    Release,
    Result,
    Shutdown,
    ViewStatus,
)
from isaaccapture_examples.agentic_remote.robot_app import app as app_module
from isaaccapture_examples.agentic_remote.robot_app import gripper as gripper_module
from isaaccapture_examples.agentic_remote.robot_app.app import (
    CLOSING,
    CONFIRM_OPEN,
    CONFIRM_PATH,
    LINK_LOST,
    MOTORS_OFF,
    NOT_RUNNING,
    PATH_CLEAR_PROMPT,
    RESULT_IDS_KEPT,
    RobotApp,
    _Results,
)
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    ArmFault,
    GripperOutcome,
    MotionState,
    Owner,
    ReadFailed,
    Refusal,
    StopLevel,
)

WAIT_S = 5.0  # real seconds a test gives another thread before it fails
BLOCKED_S = 0.05  # real seconds after which a thread counts as kept waiting
REST = [0.0] * 6
# 25 setpoints at 0.2 rad/s on every joint, inside the agent's limits.
PATH = [[0.004 * (index + 1)] * 6 for index in range(25)]
# 60 setpoints at the agent's speed limit, far enough for a contact to show.
LONG_PATH = [[0.008 * (index + 1)] * 6 for index in range(60)]
# 200 setpoints, 4 s: longer than one permit of the client lasts.
SLOW_PATH = [[0.004 * (index + 1)] * 6 for index in range(200)]
# A pose the agent may hold; the return to rest from it has 209 setpoints.
RAISED = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
RACED = "a STOP or an owner change arrived while the request was checked"


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


def held_move(h, count: int = 6) -> tuple[str, threading.Event]:
    """An agent MOVE along PATH, held once `count` setpoints were sent."""
    reached, release = h.hold_move_after(count)
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    return command_id, release


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


def with_config(config, **blocks):
    """The configuration with fields of its blocks replaced: permits={...}."""
    changed = {
        block: dataclasses.replace(getattr(config, block), **fields)
        for block, fields in blocks.items()
    }
    return dataclasses.replace(config, **changed)


def in_background(call: Callable[[], object]) -> threading.Event:
    """Run a call on another thread; the event is set when it has returned."""
    returned = threading.Event()

    def run() -> None:
        call()
        returned.set()

    threading.Thread(target=run, daemon=True).start()
    return returned


# Start and WELCOME


def test_start_connects_the_arm_and_sends_welcome_and_state(robot_app):
    h, config = robot_app, robot_app.config
    welcome = h.client.welcome
    assert welcome.session_id == h.app.session_id
    assert (welcome.robot_id, welcome.profile_id) == ("sim-arm", "sim-6dof")
    assert welcome.joint_names == list(config.joint_names)
    assert welcome.joint_min == list(config.limits.agent.joint_min)
    assert welcome.joint_max == list(config.limits.agent.joint_max)
    assert welcome.rest_q == REST
    assert welcome.setpoint_period_s == config.executor.period_s
    assert welcome.max_move_setpoints == config.executor.max_setpoints
    versions = {"agent": "sim-1", "human": "sim-1", "recovery": "sim-1"}
    assert welcome.limit_set_versions == versions
    assert welcome.camera_calibration_id == ""  # the shipped file names none
    state = h.client.state
    assert state.session_id == welcome.session_id
    assert (state.owner, state.owner_number) == ("nobody", 0)
    assert (state.stop_number, state.stop_level) == (0, "none")
    assert (state.motion, state.torque_on) == ("rest_hold", True)
    assert state.gripper.referenced and state.gripper.load == "empty"
    assert [sample.valid for sample in state.joints] == [True] * 6
    assert h.app.running and h.app.link_alive
    assert h.arm.torque_on()


def test_welcome_carries_the_configured_camera_calibration_id(make_robot_app, config):
    named = dataclasses.replace(config, camera_calibration_id="rig-2026-11-14")
    h = make_robot_app(config=named)
    assert h.client.welcome.camera_calibration_id == "rig-2026-11-14"


def test_the_arm_is_connected_once(robot_app):
    with pytest.raises(RuntimeError, match=r"start\(\) was already called"):
        robot_app.app.start()
    assert robot_app.app.running


def test_an_arm_with_another_joint_count_is_refused(sim_arm, checkers, config, clock):
    wrong = dataclasses.replace(
        config,
        n_joints=6,
        joint_names=tuple(f"j{index}" for index in range(6)),
    )
    sim_arm.n_joints = 5
    with pytest.raises(ValueError, match=r"the arm has 5 joints"):
        RobotApp(wrong, sim_arm, checkers, None, clock)


def test_away_from_rest_the_app_holds_and_leaves_the_jaws_unreferenced(
    make_robot_app,
):
    h = make_robot_app(pose=RAISED)
    state = h.client.state
    assert (state.motion, state.torque_on) == ("hold", True)
    assert not state.gripper.referenced
    assert state.gripper.load == "unknown"
    assert h.arm.sent == []


# Three full joint reads of the rig lying folded at rest with the motors off, rad
# (feedback-connected, rest-recovery-readings, operator-hold-reset). Joint
# indices 0, 4 and 5 have no stop: they lie up to 0.078 rad from the rest pose.
RIG_REST_READINGS = [
    [-0.019729, -0.007883, 0.000980, 0.007785, -0.059097, 0.056450],
    [-0.054968, -0.006605, 0.000426, 0.007938, -0.008744, 0.077696],
    [-0.020453, -0.007585, 0.001150, 0.006558, -0.009089, 0.077811],
]
RIG_START_TOL_RAD = 0.12  # the rig profile's tolerance for its unpowered rest


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_the_rigs_rest_reading_is_at_rest_at_start_with_the_start_tolerance(
    make_robot_app, config, reading
):
    wide = with_config(config, rest={"start_rest_tol_rad": RIG_START_TOL_RAD})
    h = make_robot_app(config=wide, pose=reading)
    state = h.client.state
    assert state.gripper.referenced and state.gripper.load == "empty"
    # Not "rest_hold": the release keeps its 0.05 rad and refuses this pose.
    assert (state.motion, state.torque_on) == ("hold", True)
    refused = final(h, h.client.release())
    assert refused.status == "refused"
    assert refused.reason.startswith("the arm is not at rest: joint index ")
    assert h.arm.torque_on() and h.arm.sent == []
    # A typed rest brings every joint inside the release tolerance.
    assert final(h, h.client.rest()).status == "completed"
    assert h.app.motion_state is MotionState.REST_HOLD
    assert final(h, h.client.release()).status == "completed"
    (released_at,) = h.arm.released_at
    assert released_at == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
def test_the_rigs_rest_reading_is_away_from_rest_with_the_release_tolerance(
    make_robot_app, config, reading
):
    assert config.rest.start_rest_tol_rad == config.rest.rest_tol_rad == 0.05
    h = make_robot_app(pose=reading)
    state = h.client.state
    assert (state.motion, state.torque_on) == ("hold", True)
    assert not state.gripper.referenced and state.gripper.load == "unknown"


@pytest.mark.parametrize(
    ("off_rest", "motion", "referenced"),
    [(0.049, "rest_hold", True), (0.051, "hold", True), (0.121, "hold", False)],
)
def test_the_start_tolerance_decides_the_jaw_reference_and_never_rest_hold(
    make_robot_app, config, off_rest, motion, referenced
):
    wide = with_config(config, rest={"start_rest_tol_rad": RIG_START_TOL_RAD})
    h = make_robot_app(config=wide, pose=[0.0, 0.0, 0.0, 0.0, 0.0, off_rest])
    state = h.client.state
    assert state.motion == motion
    assert state.gripper.referenced is referenced
    assert h.app.shutdown_moves_arm() is (motion == "hold")


def test_a_start_inside_the_start_tolerance_only_is_logged(
    make_robot_app, config, caplog
):
    wide = with_config(config, rest={"start_rest_tol_rad": RIG_START_TOL_RAD})
    with caplog.at_level("WARNING"):
        make_robot_app(config=wide, pose=RIG_REST_READINGS[1])
    assert [r.getMessage() for r in caplog.records if "at start" in r.getMessage()] == [
        "the arm lies at rest at start (within 0.12 rad) but joint index 5 is "
        "0.078 rad from the rest pose, outside the 0.05 rad of a release: the "
        "motion state is hold until a typed rest"
    ]


def test_the_jaws_are_not_referenced_when_the_configuration_says_so(
    make_robot_app, config
):
    h = make_robot_app(
        config=with_config(config, gripper={"reference_at_start": False})
    )
    assert not h.client.state.gripper.referenced
    assert h.client.state.motion == "rest_hold"


def test_a_jaw_reference_that_fails_at_start_is_a_fault_stop(make_robot_app):
    h = make_robot_app(start=False)

    def reference() -> None:
        raise ArmFault("no index pulse")

    h.arm.gripper_reference = reference
    h.start()
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == "the jaw reference failed: no index pulse"
    assert h.arm.torque_on()


def test_an_arm_that_connects_at_rest_with_its_motors_off_is_unpowered(
    make_robot_app, config
):
    # Neither arm port of P0 does this; a later driver may.
    h = make_robot_app(
        config=with_config(config, gripper={"reference_at_start": False}),
        start=False,
    )
    connect = h.arm.connect

    def connect_unpowered() -> None:
        connect()
        h.arm._torque = False

    h.arm.connect = connect_unpowered
    h.start()
    state = h.client.state
    assert (state.motion, state.torque_on) == ("rest_unpowered", False)
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    rest = h.client.wait_result(h.client.rest(), timeout_s=WAIT_S)
    assert (rest.status, rest.reason) == ("refused", "the motors are off")
    assert h.arm.sent == [] and h.arm.released_at == []


# OWNER_REQUEST


def test_a_new_task_makes_the_agent_the_control_owner(robot_app):
    h = robot_app
    result = h.new_task()
    assert result.detail == {"owner": "agent", "owner_number": 1}
    assert statuses(h, result.command_id) == ["completed"]
    state = h.client.state
    assert (state.owner, state.owner_number) == ("agent", 1)
    assert state.limit_set_version == h.config.limits.agent.version
    assert h.client.wait_event("owner").text == (
        "new task: the agent owns the arm, owner number 1"
    )
    assert h.arm.sent == []


def test_a_new_task_is_refused_while_the_agent_owns_the_arm(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.start_task()
    assert result.status == "refused"
    assert result.reason == "control owner is agent, not nobody"
    assert h.app.snapshot().owner_number == 1
    assert h.client.wait_event("refused").text == (
        "owner request 'agent' refused: control owner is agent, not nobody"
    )


@pytest.mark.parametrize(
    ("quoted", "reason"),
    [
        ({"session_id": "another-run"}, "session id does not match"),
        ({"stop_number": 7}, "stop number 7 does not match the current 0"),
        ({"owner_number": 7}, "owner number 7 does not match the current {owner}"),
    ],
    ids=["session id", "stop number", "owner number"],
)
@pytest.mark.parametrize("request_kind", ["agent", "help_needed", "nobody", "rest"])
def test_an_owner_request_quoting_an_old_value_is_refused(
    robot_app, request_kind, quoted, reason
):
    h = robot_app
    if request_kind == "help_needed":
        h.new_task()
    reason = reason.format(owner=h.app.snapshot().owner_number)
    command_id = h.client.request_owner(request_kind, confirmed=True, **quoted)
    result = final(h, command_id)
    assert (result.status, result.reason) == ("refused", reason)
    assert statuses(h, command_id) == ["refused"]
    assert h.arm.sent == []


def test_a_new_task_outside_the_agents_limits_needs_a_rest_first(robot_app):
    h = robot_app
    h.arm.set_pose([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = h.client.start_task()
    assert result.status == "refused"
    assert result.reason.startswith("type rest first: pose: joint index 0 at 1.6000")
    assert h.app.snapshot().owner is Owner.NOBODY
    # The typed rest brings the arm back inside, and the task is accepted.
    assert final(h, h.client.rest()).status == "completed"
    assert h.client.start_task().status == "completed"


def test_a_new_task_is_refused_when_the_pose_cannot_be_read(robot_app):
    h = robot_app
    h.arm.fail_reads(3, 1)
    result = h.client.start_task()
    assert result.status == "refused"
    assert result.reason.startswith("the measured pose could not be read")
    assert h.app.snapshot().owner is Owner.NOBODY


def test_a_new_task_is_refused_while_a_fault_stop_is_latched(robot_app):
    h = robot_app
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    result = h.client.start_task()
    assert result.status == "refused"
    assert result.reason == "fault-stop latched: motor 3 overheated"


def test_a_stop_while_a_new_task_is_checked_wins(robot_app):
    h = robot_app
    reached, release = h.hold_call("read_all")  # the measured pose of the check
    command_id = h.client.request_owner("agent")
    assert reached.wait(WAIT_S)
    assert h.app.stop("operator pressed stop") == 1
    release.set()
    result = final(h, command_id)
    assert (result.status, result.reason) == ("refused", RACED)
    assert h.app.snapshot().owner is Owner.NOBODY


def test_a_new_task_is_refused_while_motion_work_is_under_way(robot_app):
    h = robot_app
    reads = []
    real = h.arm.read_all

    def read_all():
        reads.append(1)
        return real()

    h.arm.read_all = read_all
    reached, release = h.hold_call("read_all")  # the release's first joint read
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    result = h.client.start_task()
    assert result.status == "refused"
    assert result.reason == "busy: a torque release is running"
    assert reads == []  # refused before the pose was read
    release.set()
    assert final(h, release_id).status == "completed"


def test_a_new_task_is_refused_when_a_release_began_while_its_pose_was_read(
    robot_app,
):
    h = robot_app
    reading, go_on = h.hold_call("read_all", call=1)  # the new task's pose read
    checking, finish = h.hold_call("read_all", call=2)  # the release's first read
    task_id = h.client.request_owner("agent")
    assert reading.wait(WAIT_S)
    done = Done()
    assert h.app.request_release(done) is None
    assert checking.wait(WAIT_S)
    go_on.set()
    result = final(h, task_id)
    assert (result.status, result.reason) == (
        "refused",
        "busy: a torque release is running",
    )
    finish.set()
    assert done.wait().status == "completed"
    assert h.app.snapshot().owner is Owner.NOBODY


def test_a_new_task_is_refused_when_the_motors_went_off_while_its_pose_was_read(
    robot_app,
):
    h = robot_app
    reading, go_on = h.hold_call("read_all")  # the new task's pose read
    task_id = h.client.request_owner("agent")
    assert reading.wait(WAIT_S)
    done = Done()
    assert h.app.request_release(done) is None
    assert done.wait().status == "completed"
    go_on.set()
    result = final(h, task_id)
    assert (result.status, result.reason) == ("refused", MOTORS_OFF)
    assert h.app.snapshot().owner is Owner.NOBODY


def test_help_needed_ends_the_agents_ownership_and_starts_the_wait(robot_app):
    h = robot_app
    h.new_task()
    result = ask_for_help(h)
    assert result.detail == {
        "owner": "nobody",
        "owner_number": 2,
        "timed_return": True,
        "assistance_s": 120.0,
    }
    state = h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    assert (state.owner, state.owner_number) == ("nobody", 2)
    assert state.assistance_wait_s == pytest.approx(120.0, abs=1.0)
    assert (state.stop_number, state.stop_level) == (0, "none")
    event = h.client.wait_event("help_needed")
    assert event.text == (
        "HELP NEEDED: help needed: the agent is stuck; the arm holds and nobody owns it"
    )
    assert h.heard("help_needed")[0][1] == event.text


def test_help_needed_stops_a_running_move_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    ask_for_help(h, "the model gave up")
    release.set()
    result = final(h, move_id)
    assert result.status == "cancelled"
    assert result.reason == "help needed: the model gave up"
    assert h.sent() == PATH[:6]
    assert h.arm.last_target()[0] == PATH[5]
    assert h.arm.torque_on()


def test_help_needed_stops_a_running_jaw_command(robot_app):
    h = robot_app
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    ask_for_help(h)
    assert final(h, jaw_id).status == "cancelled"
    assert h.arm.jaw_stiffness == "light"
    assert h.arm.gripper_read() == pytest.approx(0.6)


def test_help_needed_is_refused_unless_the_agent_owns_the_arm(robot_app):
    h = robot_app
    result = final(h, h.client.request_owner("help_needed", reason="stuck"))
    assert result.status == "refused"
    assert result.reason == "control owner is nobody, not agent"
    h.step()
    assert h.client.state.assistance_wait_s is None


def test_a_new_task_cancels_the_assistance_wait(robot_app):
    h = robot_app
    h.new_task()
    ask_for_help(h)
    h.client.wait_state(lambda s: s.assistance_wait_s is not None)
    h.new_task()
    assert h.client.state.assistance_wait_s is None
    h.advance(125.0)
    assert h.heard("expired") == []


@pytest.mark.parametrize(
    ("task", "auto_return", "returns"),
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=["both allow it", "the task forbids it", "the configuration forbids it"],
)
def test_the_timed_return_needs_the_task_and_the_configuration_to_allow_it(
    make_robot_app, config, task, auto_return, returns
):
    h = make_robot_app(config=with_config(config, rest={"auto_return": auto_return}))
    h.arm.set_pose(RAISED)
    h.new_task(timed_return=task)
    assert ask_for_help(h).detail["timed_return"] is returns
    run_wait_out(h)
    if returns:
        assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
        assert len(h.sent()) == 209
        assert h.app.motion_state is MotionState.REST_HOLD
        assert h.heard("bell")[0][1] == "ARM WILL MOVE TO REST in 30 s"
    else:
        assert h.arm.sent == []
        assert h.arm.pose() == RAISED
        assert h.heard("bell")[0][1].startswith("ARM WILL KEEP HOLDING")
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.snapshot().owner is Owner.NOBODY
    # The wait made one attempt: nothing follows by itself.
    sent = len(h.arm.sent)
    h.advance(125.0)
    assert len(h.heard("expired")) == 1
    assert len(h.arm.sent) == sent


def test_the_wait_counts_down_and_rings_at_both_terminals(robot_app):
    h = robot_app
    h.new_task()
    ask_for_help(h)
    run_wait_out(h)
    h.step()
    heard = [(kind, text) for kind, text, _ in h.local_events]
    h.client.wait_until(lambda: len(h.client.events) >= len(heard))
    events = h.client.events
    assert [(event.kind, event.text) for event in events] == heard
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    countdowns = [text for kind, text in heard if kind == "countdown"]
    assert countdowns[0] == (
        "assistance wait: 120 s left, then the arm moves to rest (the agent is stuck)"
    )
    assert countdowns[1:] == [
        f"assistance wait: {left} s left" for left in range(110, 0, -10)
    ]
    bells = [text for kind, text in heard if kind == "bell"]
    assert bells == [
        f"ARM WILL MOVE TO REST in {mark:g} s" for mark in h.config.waits.bell_at_s
    ]
    assert [text for kind, text in heard if kind == "expired"] == [
        "assistance wait over: timed return to rest requested"
    ]
    # The arm was at rest already: the timed return moved nothing.
    assert heard[-1] == ("rest", "return to rest: the arm is at rest, motors on")
    assert h.arm.sent == []


def test_the_agent_gives_the_arm_back_and_its_move_ends_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    result = final(h, h.client.request_owner("nobody", reason="task done"))
    assert result.status == "completed"
    assert result.detail == {"owner": "nobody", "owner_number": 2}
    release.set()
    move = final(h, move_id)
    assert move.status == "cancelled"
    assert move.reason == "the agent gave the arm back: task done"
    assert h.sent() == PATH[:6]
    state = h.client.wait_state(lambda s: s.owner_number == 2)
    assert (state.owner, state.assistance_wait_s) == ("nobody", None)
    # Asked again, nothing changes.
    again = final(h, h.client.request_owner("nobody"))
    assert again.detail == {"owner": "nobody", "owner_number": 2}


def test_the_agent_giving_the_arm_back_stops_a_running_jaw_command(robot_app):
    h = robot_app
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    assert final(h, jaw_id).status == "cancelled"
    assert h.arm.jaw_stiffness == "light"


def test_a_return_to_rest_is_not_ended_by_an_owner_request(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    result = final(h, h.client.request_owner("nobody"))
    assert result.status == "refused"
    assert result.reason == "control owner is recovery, not agent"
    release.set()
    assert final(h, rest_id).status == "completed"


@pytest.mark.parametrize(
    ("request_kind", "reason"),
    [("human", "human path not built yet"), ("dance", "unknown request 'dance'")],
)
def test_owner_requests_the_app_does_not_take_are_refused(
    robot_app, request_kind, reason
):
    h = robot_app
    result = final(h, h.client.request_owner(request_kind))
    assert (result.status, result.reason) == ("refused", reason)
    assert h.app.snapshot().owner is Owner.NOBODY


# MOVE


def test_a_move_is_accepted_then_completed_with_the_measured_end_pose(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=REST)
    assert statuses(h, result.command_id) == ["accepted", "completed"]
    assert h.sent() == PATH
    assert result.end_q == pytest.approx(PATH[-1], abs=0.03)
    assert result.detail == {"outcome": "completed", "sent": 25, "total": 25}
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == result.command_id
    )
    assert state.last_result == {
        "command_id": result.command_id,
        "status": "completed",
        "reason": "",
    }
    assert state.motion == "hold"
    assert (state.move_id, state.move_index, state.move_total) == ("", 0, 0)


def test_the_client_plans_a_move_from_the_measured_pose(robot_app):
    h = robot_app
    h.new_task()
    start = h.client.measured_pose()
    assert start == pytest.approx(REST, abs=1e-6)
    goal = [0.1, -0.1, 0.05, 0.0, 0.0, 0.02]
    path = h.client.plan_straight(start, goal, 0.3)
    # 0.1 rad at a peak of 0.3 rad/s: 0.625 s, rounded up to whole periods.
    assert len(path) == 32
    assert path[-1] == goal
    steps = [abs(b[0] - a[0]) for a, b in zip([start, *path], path)]
    assert max(steps) <= 0.3 * 0.02 + 1e-12
    assert max(steps) > 0.29 * 0.02
    command_id = h.client.new_id()
    h.client.permit(command_id, len(path) - 1)
    h.client.move(path, command_id=command_id)  # start_q: the measured pose
    result = final(h, command_id)
    assert result.status == "completed", result.reason
    assert result.end_q == pytest.approx(goal, abs=0.03)
    assert h.sent() == path


def test_a_move_is_refused_unless_the_agent_owns_the_arm(robot_app):
    h = robot_app
    result = h.client.run_move(PATH, start_q=REST)
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason == "control owner is nobody, not agent"
    assert h.arm.sent == []


@pytest.mark.parametrize(
    ("quoted", "reason"),
    [
        ({"session_id": "another-run"}, "session id does not match"),
        ({"owner_number": 0}, "owner number 0 does not match the current 1"),
        ({"stop_number": 7}, "stop number 7 does not match the current 0"),
    ],
    ids=["session id", "owner number", "stop number"],
)
def test_a_move_quoting_an_old_value_sends_no_setpoint(robot_app, quoted, reason):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=REST, **quoted)
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason == reason
    h.step()
    assert h.arm.sent == []


def test_an_expired_move_sends_no_setpoint(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=REST, expires_mono_s=h.clock.monotonic())
    assert statuses(h, result.command_id) == ["refused"]
    assert result.reason.startswith("the move expired 0.0")
    h.step()
    assert h.arm.sent == []


def test_a_second_move_is_refused_as_busy_and_the_first_goes_on(robot_app):
    h = robot_app
    h.new_task()
    first, release = held_move(h)
    second = h.client.move(PATH, start_q=REST)
    result = final(h, second)
    assert statuses(h, second) == ["refused"]
    assert result.reason == "busy: a move is running"
    state = h.client.wait_state(lambda s: s.move_id == first)
    assert (state.motion, state.move_total) == ("moving", 25)
    release.set()
    assert final(h, first).status == "completed"
    assert h.sent() == PATH


def test_a_move_is_refused_as_busy_while_a_release_runs(robot_app):
    h = robot_app
    h.new_task()
    reached, release = h.hold_call("read_all")  # the release's first joint read
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "refused"
    assert result.reason == "busy: a torque release is running"
    release.set()
    assert final(h, release_id).status == "completed"
    assert h.arm.sent == []


def test_a_move_from_the_wrong_start_pose_is_refused_and_rest_hold_stays(robot_app):
    h = robot_app
    h.new_task()
    result = h.client.run_move(PATH, start_q=[0.1] * 6)
    assert statuses(h, result.command_id) == ["accepted", "refused"]
    assert result.reason.startswith("start pose: joint index 0 is at 0.0000 rad")
    assert result.end_q == pytest.approx(REST, abs=1e-6)
    assert h.arm.sent == []
    # Nothing was sent, so the arm is still measured at rest.
    assert h.app.motion_state is MotionState.REST_HOLD


def test_without_a_permit_no_setpoint_is_sent(make_robot_app, config):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    command_id = h.client.move(PATH, start_q=REST)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert result.reason == "no permit for this move (waited 0.02 s)"
    assert h.arm.sent == []
    # The MOVE's own evidence feeds nothing while permits are required.
    assert h.checkers.agent._fed_at is None


@pytest.mark.parametrize(
    "quoted",
    [{"session_id": "another-run"}, {"owner_number": 0}, {"stop_number": 7}],
    ids=["session id", "owner number", "stop number"],
)
def test_a_permit_quoting_an_old_value_moves_nothing(make_robot_app, config, quoted):
    h = make_robot_app(config=with_config(config, permits={"first_wait_s": 0.02}))
    h.new_task()
    command_id = h.client.new_id()
    h.client.permit(command_id, len(PATH) - 1, **quoted)
    h.client.move(PATH, start_q=REST, command_id=command_id)
    assert final(h, command_id).status == "refused"
    assert h.arm.sent == []


# The stage of a move in ROBOT_STATE


def patient(config, **state):
    """The configuration with a first-permit wait no test runs into."""
    return with_config(config, permits={"first_wait_s": 30.0}, state=state)


def armed_move(h, path=PATH) -> str:
    """An agent MOVE that ROBOT_STATE shows as armed; no permit was sent."""
    command_id = h.client.move(path, start_q=REST)
    h.client.wait_state(lambda s: s.move_id == command_id and s.move_stage == "armed")
    return command_id


def test_robot_state_shows_a_move_armed_then_sending_then_none(make_robot_app, config):
    # No ROBOT_STATE falls due in this test: each one seen was sent at once.
    h = make_robot_app(config=patient(config, publish_period_s=1000.0))
    h.new_task()
    assert h.client.state.move_stage == ""
    command_id = h.client.move(PATH, start_q=REST)
    armed = h.client.wait_state(
        lambda s: s.move_id == command_id and s.move_stage == "armed"
    )
    assert (armed.move_index, armed.move_total) == (0, len(PATH))
    assert armed.motion == "moving"
    assert h.arm.sent == []
    assert not h.checkers.agent.in_motion
    reached, release = h.hold_move_after(6)
    h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    sending = h.client.wait_state(
        lambda s: s.move_id == command_id and s.move_stage == "sending"
    )
    assert sending.move_total == len(PATH)
    release.set()
    assert final(h, command_id).status == "completed"
    after = h.client.wait_state(lambda s: s.move_id == "")
    assert (after.move_stage, after.move_index) == ("", 0)


def test_a_move_from_another_start_pose_is_refused_without_a_wait_for_a_permit(
    make_robot_app, config
):
    h = make_robot_app(config=patient(config))
    h.new_task()
    stages: list[str] = []
    real = h.arm.read_all

    def read_all() -> list:
        stages.append(h.app._executor.stage())
        return real()

    h.arm.read_all = read_all
    command_id = h.client.move(PATH, start_q=[0.1] * 6)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert result.reason.startswith("start pose: joint index 0 is at 0.0000 rad")
    assert result.end_q == REST
    assert result.detail == {"outcome": "refused", "sent": 0, "total": len(PATH)}
    assert stages == [""]  # never armed
    assert h.arm.sent == []
    assert h.app.motion_state is MotionState.REST_HOLD


def test_no_permit_starts_a_move_from_another_start_pose(make_robot_app, config):
    h = make_robot_app(config=patient(config))
    h.new_task()
    command_id = h.client.new_id()
    h.client.permit(command_id, len(PATH) - 1)
    h.client.move(PATH, start_q=[0.1] * 6, command_id=command_id)
    assert final(h, command_id).reason.startswith("start pose: joint index 0")
    assert h.arm.sent == []


def test_the_joints_are_still_read_while_a_move_is_armed(make_robot_app, config):
    h = make_robot_app(config=patient(config))
    h.new_task()
    command_id = armed_move(h)
    h.advance(0.4, step_s=0.01)
    # A ROBOT_STATE goes out every 0.05 s: one of the last two is enough.
    newest = h.clock.monotonic() - 0.1
    state = h.client.wait_state(lambda s: s.clock_mono_s >= newest)
    assert (state.move_id, state.move_stage) == (command_id, "armed")
    assert state.health["reading_age_s"] < 0.2
    assert state.stop_level == "none"
    assert h.arm.sent == []
    h.client.permit(command_id, len(PATH) - 1)
    assert final(h, command_id).status == "completed"
    assert h.sent() == PATH


def test_lost_joint_feedback_while_a_move_is_armed_is_a_fault_stop(
    make_robot_app, config
):
    h = make_robot_app(config=patient(config))
    h.new_task()
    command_id = armed_move(h)
    h.arm.fail_reads(2, 10_000)
    h.advance(0.8, step_s=0.02)
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost for 0.")
    result = final(h, command_id)
    assert result.status == "cancelled"
    assert h.arm.sent == []
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_stop_ends_an_armed_move_with_nothing_sent(make_robot_app, config):
    h = make_robot_app(config=patient(config))
    h.new_task()
    command_id = armed_move(h)
    h.client.stop("operator")
    assert final(h, command_id).status == "cancelled"
    state = h.client.wait_state(lambda s: s.move_id == "")
    assert state.move_stage == ""
    assert h.arm.sent == []
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_cancel_permit_ends_a_running_move_at_its_next_setpoint(robot_app):
    h = robot_app
    h.new_task()
    reached, release = h.hold_move_after(6)
    command_id = h.client.move(PATH, start_q=REST)
    permit = h.client.permit(command_id, len(PATH) - 1)
    assert reached.wait(WAIT_S)
    # What the agent sends to cancel: through setpoint -1, on the same evidence.
    h.client.permit(command_id, -1, evidence_wall_s=permit.evidence_wall_s)
    for _ in range(1000):  # until the dispatcher has taken the cancel
        if h.app._gate.covers(command_id, 6) is not None:
            break
        h.step()
    release.set()
    result = final(h, command_id)
    assert (result.status, result.reason) == (
        "refused",
        "permit reaches setpoint -1, not 6",
    )
    assert h.sent() == PATH[:6]
    assert h.arm.last_target()[0] == PATH[5]
    assert h.arm.torque_on() and h.arm.holds == [] and h.arm.released_at == []
    assert h.app.motion_state is MotionState.HOLD


def test_when_permits_stop_the_arm_holds_at_the_last_permitted_setpoint(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.move(PATH, start_q=REST)
    h.client.permit(command_id, 9)
    result = final(h, command_id)
    assert (result.status, result.reason) == (
        "refused",
        "permit reaches setpoint 9, not 10",
    )
    assert result.detail == {"outcome": "refused", "sent": 10, "total": 25}
    assert h.sent() == PATH[:10]
    assert h.arm.last_target()[0] == PATH[9]
    assert h.arm.torque_on() and h.arm.holds == []
    assert h.app.motion_state is MotionState.HOLD


def test_a_move_holds_when_its_permit_runs_out(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.move(SLOW_PATH, start_q=REST)
    h.client.permit(command_id, len(SLOW_PATH) - 1)  # lasts 2.5 s of the 4 s
    result = final(h, command_id)
    assert (result.status, result.reason) == ("refused", "permit expired")
    sent = result.detail["sent"]
    assert 110 < sent < 126
    assert h.sent() == SLOW_PATH[:sent]
    assert h.arm.last_target()[0] == SLOW_PATH[sent - 1]
    assert h.arm.torque_on()


def test_a_long_move_runs_to_its_end_while_its_permit_is_renewed(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.new_id()
    h.client.permit(command_id, len(SLOW_PATH) - 1)
    h.permit_every(command_id, len(SLOW_PATH) - 1, every=50)
    h.client.move(SLOW_PATH, start_q=REST, command_id=command_id)
    result = final(h, command_id)
    assert result.status == "completed", result.reason
    assert h.sent() == SLOW_PATH


def test_with_permits_off_the_moves_own_evidence_is_checked(make_robot_app, config):
    h = make_robot_app(
        config=with_config(config, permits={"required_for_agent": False})
    )
    h.new_task()
    wall = h.client.thor_now()[1]
    refusals = {
        None: "the move carries no evidence_wall_s and permits are off",
        wall - 3.5: "the move's evidence is 3.5",
        wall + 1.0: "the move's evidence is in the future",
    }
    for evidence, reason in refusals.items():
        command_id = h.client.move(PATH, start_q=REST, evidence_wall_s=evidence)
        assert final(h, command_id).reason.startswith(reason)
        assert statuses(h, command_id) == ["refused"]
    assert h.arm.sent == []
    # Fresh evidence feeds the agent checker's watchdog, and no permit is needed.
    command_id = h.client.move(PATH, start_q=REST, evidence_wall_s=wall - 1.0)
    assert final(h, command_id).status == "completed"
    assert h.sent() == PATH
    offset = h.clock.wall() - h.clock.monotonic()
    assert h.checkers.agent._fed_at == pytest.approx(wall - 1.0 - offset, abs=0.05)


def test_a_contact_ends_the_move_in_a_hold_that_is_not_latched(robot_app):
    h = robot_app
    h.new_task()
    h.arm.block_joint(0, 0.05)
    result = h.client.run_move(LONG_PATH, start_q=REST)
    assert (result.status, result.reason) == ("failed", "contact")
    assert result.detail["outcome"] == "contact"
    assert result.detail["contact"].startswith("contact: joint index 0 trails")
    assert result.end_q[0] == pytest.approx(0.05, abs=1e-6)
    assert len(h.arm.holds) == 1  # held at the measured pose
    # Not a fault-stop: the agent still owns the arm.
    state = h.app.snapshot()
    assert (state.stop_level, state.owner) == (StopLevel.NONE, Owner.AGENT)
    assert h.checkers.agent.contact_spots == [result.end_q]
    assert h.checkers.recovery.contact_spots == [result.end_q]
    # While the arm holds at the contact it does not return to rest.
    h.arm.block_joint(0, None)
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == (
        "refused",
        "the arm is holding at a contact",
    )
    cleared = final(h, h.client.checker_state("clear_contact_spots"))
    assert cleared.status == "completed"
    # The recovery checker keeps the spots the agent checker has, no others.
    assert h.checkers.agent.contact_spots == []
    assert h.checkers.recovery.contact_spots == []
    assert final(h, h.client.rest()).status == "completed"


def test_a_completed_move_ends_a_contact_hold(robot_app):
    h = robot_app
    h.new_task()
    h.arm.block_joint(0, 0.05)
    contact = h.client.run_move(LONG_PATH, start_q=REST)
    assert contact.reason == "contact"
    h.arm.block_joint(0, None)
    start = h.arm.pose()
    away = h.client.plan_straight(start, [q - 0.02 for q in start], 0.2)
    moved = h.client.run_move(away, start_q=start)
    assert moved.status == "completed", moved.reason
    assert final(h, h.client.rest()).status == "completed"


def test_a_move_that_does_not_settle_fails_without_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()
    h.arm.block_joint(0, 0.06)
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "failed"
    assert "is not within 0.03 rad of the last setpoint" in result.reason
    assert result.detail == {"outcome": "not_settled", "sent": 25, "total": 25}
    assert h.app.snapshot().stop_level is StopLevel.NONE


def test_the_grasp_region_of_the_last_grasp_bars_the_return_to_rest(robot_app):
    h = robot_app
    h.new_task()
    region = GraspRegion(center_xy=(0.3, 0.0), radius_m=0.05, z_min=0.0, z_max=0.1)
    seen = []

    def tip_blocked(q, grasp_region):
        seen.append(grasp_region)
        return None if grasp_region is None else "the gripper tip is in the grasp"

    h.checkers.recovery.set_tip_hook(tip_blocked)
    assert h.client.run_move(PATH, start_q=REST, grasp_region=region).status == (
        "completed"
    )
    assert h.checkers.agent.grasp_region is None  # the executor cleared it
    refused = final(h, h.client.rest())
    assert (refused.status, refused.reason) == (
        "refused",
        "the gripper tip is in the grasp",
    )
    assert seen == [region]
    assert final(h, h.client.checker_state("close_grasp")).status == "completed"
    assert final(h, h.client.rest()).status == "completed"
    assert seen[-1] is None


# GRIPPER, OPEN and CHECKER_STATE


def test_a_jaw_command_is_accepted_then_completed(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.gripper(0.5)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert result.detail == {"outcome": "reached", "opening": pytest.approx(0.5)}
    assert result.end_q is None
    assert [command.opening for command in h.arm.jaw_commands] == [0.5]


def test_a_jaw_command_that_ends_at_once_is_still_accepted_first(robot_app):
    h = robot_app
    h.new_task()
    real = h.app._worker.submit

    def submit(*args):
        refusal = real(*args)
        # The command has ended before the app hears that it was queued.
        assert h.app._worker.wait_idle(WAIT_S)
        return refusal

    h.app._worker.submit = submit
    command_id = h.client.gripper(0.5)
    assert final(h, command_id).status == "completed"
    assert statuses(h, command_id) == ["accepted", "completed"]


@pytest.mark.parametrize(
    ("mode", "status", "reason"),
    [
        ("close_until_contact", "completed", ""),
        ("plain", "failed", "the jaws stalled short of their target"),
    ],
)
def test_a_close_that_stalls_on_an_object(robot_app, mode, status, reason):
    h = robot_app
    h.new_task()
    h.arm.jaw_object(0.4)
    result = final(h, h.client.gripper(0.0, mode=mode))
    assert (result.status, result.reason) == (status, reason)
    assert result.detail == {"outcome": "stalled", "opening": pytest.approx(0.4)}
    state = h.client.wait_state(lambda s: s.gripper.may_be_holding)
    assert state.gripper.load == "loaded"
    # In either mode the closed target stands at the light holding stiffness.
    assert (state.gripper.last_target, h.arm.jaw_stiffness) == (0.0, "light")


def test_a_jaw_command_that_times_out_fails(robot_app):
    h = robot_app
    h.new_task()
    h.arm.jaw_timeout_s = 0.05
    result = final(h, h.client.gripper(0.0))
    assert (result.status, result.reason) == ("failed", "the jaw command timed out")
    assert result.detail["outcome"] == "timeout"
    # A close that ran out of time short of its target is not left closing.
    assert h.arm.jaw_stiffness == "light"


@pytest.mark.parametrize("next_command", ["release", "rest", "shutdown"])
def test_the_command_that_follows_a_jaw_commands_result_does_not_find_the_jaws_busy(
    make_robot_app, monkeypatch, next_command
):
    h = make_robot_app()
    if next_command != "release":
        h.hold_at(RAISED)
    sent, go_on = threading.Event(), threading.Event()
    real = app_module._JawReply._final

    def final_then_wait(reply, outcome: GripperOutcome) -> None:
        real(reply, outcome)
        if threading.current_thread().name == "gripper-worker":
            # The RESULT is out; the gripper worker loses the processor here.
            sent.set()
            go_on.wait(WAIT_S)

    monkeypatch.setattr(app_module._JawReply, "_final", final_then_wait)
    h.new_task()
    # Held until "accepted" is out, so the final RESULT comes from the worker.
    starting, start = h.hold_call("gripper_command")
    jaw_id = h.client.gripper(1.0)
    assert starting.wait(WAIT_S)
    assert h.client.wait_result(jaw_id, final=False).status == "accepted"
    start.set()
    assert final(h, jaw_id).status == "completed"
    assert sent.wait(WAIT_S)
    try:
        send = {
            "release": h.client.release,
            "rest": h.client.rest,
            "shutdown": h.client.shutdown,
        }[next_command]
        result = final(h, send())
        assert (result.status, result.reason) == ("completed", "")
    finally:
        go_on.set()


@pytest.mark.parametrize(
    ("task", "fields", "reason"),
    [
        (False, {}, "control owner is nobody, not agent"),
        (True, {"owner_number": 0}, "owner number 0 does not match the current 1"),
        (True, {"stop_number": 3}, "stop number 3 does not match the current 0"),
        (True, {"session_id": "another-run"}, "session id does not match"),
        (True, {"mode": "squeeze"}, "unknown mode 'squeeze'"),
        (True, {"opening": 1.5}, "opening must be between 0 and 1, got 1.5"),
    ],
    ids=["no owner", "owner number", "stop number", "session id", "mode", "opening"],
)
def test_a_jaw_command_that_is_refused_moves_nothing(robot_app, task, fields, reason):
    h = robot_app
    if task:
        h.new_task()
    opening = fields.pop("opening", 0.5)
    command_id = h.client.gripper(opening, **fields)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == reason
    assert h.arm.jaw_commands == []


@pytest.mark.parametrize(
    ("work", "what"),
    [
        ("rest", "a return to rest"),
        ("release", "a torque release"),
        ("shutdown", "a shutdown"),
    ],
)
def test_no_jaw_command_is_taken_while_the_jaws_must_stay_proven_empty(
    robot_app, work, what
):
    h = robot_app
    if work != "release":
        h.arm.set_pose(RAISED)
    h.new_task()
    # The work has proven the jaws empty and now takes its first full joint read.
    reached, release = h.hold_call("read_all")
    work_id = getattr(h.client, work)()
    assert reached.wait(WAIT_S)
    jaw_id = h.client.gripper(0.0)
    result = final(h, jaw_id)
    assert (result.status, result.reason) == ("refused", f"busy: {what} is running")
    assert statuses(h, jaw_id) == ["refused"]
    release.set()
    assert final(h, work_id).status == "completed"
    assert h.arm.jaw_commands == []


def test_no_jaw_command_is_taken_once_a_return_to_rest_is_queued(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    owner_number = h.new_task().detail["owner_number"]
    session = h.app.session_id
    with h.app._lock:  # the motion worker cannot begin the return
        h.app._handle(
            OwnerRequest("rest-1", session, owner_number, 0, "rest", confirmed=True)
        )
        h.app._handle(Gripper("jaw-1", session, owner_number, 0, 0.0, 0.5, "plain"))
        stored = h.app._results.get("jaw-1")
        assert (stored.status, stored.reason) == (
            "refused",
            "busy: a return to rest is queued",
        )
    assert final(h, "rest-1").status == "completed"
    assert h.arm.jaw_commands == []


def test_a_jaw_command_is_taken_while_a_move_runs(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert h.client.wait_result(jaw_id, final=False).status == "accepted"
    assert jaws_waiting.wait(WAIT_S)
    h.client.stop()
    assert final(h, jaw_id).status == "cancelled"
    release.set()
    # The jaw travel moved the fake clock, so how the held move ends says nothing.
    final(h, move_id)


def test_open_needs_a_confirmation_and_nobody_owning_the_arm(robot_app):
    h = robot_app
    result = final(h, h.client.open_jaws(confirmed=False))
    assert (result.status, result.reason) == ("refused", CONFIRM_OPEN)
    h.new_task()
    result = final(h, h.client.open_jaws())
    assert (result.status, result.reason) == (
        "refused",
        "control owner is agent, not nobody",
    )
    assert h.arm.jaw_commands == []
    assert [text for _, text, _ in h.heard("refused")] == [
        f"open refused: {CONFIRM_OPEN}",
        "open refused: control owner is agent, not nobody",
    ]


def test_open_goes_to_full_open_in_guarded_steps(robot_app):
    h = robot_app
    h.new_task()
    assert final(h, h.client.gripper(0.8)).status == "completed"
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    command_id = h.client.open_jaws()
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert result.detail == {"outcome": "reached", "opening": pytest.approx(1.0)}
    openings = [command.opening for command in h.arm.jaw_commands]
    assert openings == pytest.approx([0.8, 0.85, 0.9, 0.95, 1.0])


def test_open_is_refused_while_motion_work_is_under_way(robot_app):
    h = robot_app
    reached, release = h.hold_call("read_all")  # the release's first joint read
    release_id = h.client.release()
    assert reached.wait(WAIT_S)
    result = final(h, h.client.open_jaws())
    assert (result.status, result.reason) == (
        "refused",
        "busy: a torque release is running",
    )
    release.set()
    assert final(h, release_id).status == "completed"


@pytest.mark.parametrize(
    ("task", "fields", "reason"),
    [
        (False, {}, "control owner is nobody, not agent"),
        (True, {"owner_number": 0}, "owner number 0 does not match the current 1"),
        (True, {"action": "forget"}, "unknown action 'forget'"),
    ],
    ids=["no owner", "owner number", "action"],
)
def test_a_checker_state_that_is_refused_changes_nothing(
    robot_app, task, fields, reason
):
    h = robot_app
    h.checkers.agent.note_contact(REST)
    h.checkers.recovery.note_contact(REST)
    if task:
        h.new_task()
    action = fields.pop("action", "clear_contact_spots")
    result = final(h, h.client.checker_state(action, **fields))
    assert (result.status, result.reason) == ("refused", reason)
    assert h.checkers.agent.contact_spots == [REST]
    assert h.checkers.recovery.contact_spots == [REST]


# STOP


def test_stop_from_the_link_during_a_move_sends_no_further_setpoint(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.client.stop("operator pressed stop")
    state = h.client.wait_state(lambda s: s.stop_number == 1)
    assert (state.stop_level, state.owner, state.owner_number) == ("stop", "nobody", 2)
    release.set()
    result = final(h, move_id)
    assert statuses(h, move_id) == ["accepted", "cancelled"]
    assert result.reason == "STOP from console: operator pressed stop"
    assert h.sent() == PATH[:6]  # the next setpoint was not sent
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.client.wait_event("stop").data["stop_number"] == 1
    # A command that quotes the stop number from before the STOP is refused.
    again = h.client.run_move(PATH, start_q=REST, stop_number=0, owner_number=1)
    assert again.status == "refused"
    assert again.reason == "stop number 0 does not match the current 1"
    assert h.sent() == PATH[:6]
    # A new task works at once: the STOP level ends with it.
    h.new_task()
    assert h.client.state.stop_level == "none"


def test_stop_during_a_jaw_close_ends_it_at_the_light_stiffness(robot_app):
    h = robot_app
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    h.client.stop()
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert final(h, jaw_id).status == "cancelled"
    assert h.arm.jaw_stiffness == "light"
    assert h.arm.gripper_read() == pytest.approx(0.6)  # never opened
    assert [command.opening for command in h.arm.jaw_commands] == [0.0]
    refused = final(h, h.client.gripper(1.0, stop_number=0, owner_number=1))
    assert refused.reason == "stop number 0 does not match the current 1"


def test_stop_typed_at_the_thor_terminal_is_the_same_stop(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    assert h.app.stop("typed stop") == 1
    release.set()
    result = final(h, move_id)
    assert result.status == "cancelled"
    assert result.reason == "STOP from thor_terminal: typed stop"
    assert h.sent() == PATH[:6]
    assert h.heard("stop")[0][2]["sender"] == "thor_terminal"
    assert h.app.snapshot().owner is Owner.NOBODY


def test_a_stop_asked_for_by_a_signal_handler_is_raised_by_the_dispatcher(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.app.request_stop("Ctrl+C")
    h.client.wait_state(lambda s: s.stop_number == 1)
    release.set()
    assert final(h, move_id).reason == "STOP from thor_terminal: Ctrl+C"
    assert h.sent() == PATH[:6]
    assert h.app.running


@pytest.mark.parametrize(
    ("sender", "wait_goes_on"),
    [("agent", True), ("console", False), ("thor_terminal", False)],
)
def test_only_a_stop_from_a_person_cancels_the_assistance_wait(
    robot_app, sender, wait_goes_on
):
    h = robot_app
    h.new_task()
    ask_for_help(h)
    h.client.stop("stop", sender=sender)
    state = h.client.wait_state(lambda s: s.stop_number == 1)
    assert (state.assistance_wait_s is not None) is wait_goes_on
    assert h.client.wait_event("stop").data["wait_cancelled"] is not wait_goes_on


def test_a_stop_cancels_motion_work_that_has_not_begun(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    done = Done()
    with h.app._lock:  # no motion work begins while this thread has the lock
        assert h.app.request_rest(True, done) is None
        assert not done._called.wait(BLOCKED_S)
        h.app.stop("typed stop")
    result = done.wait()
    assert (result.status, result.reason) == (
        "cancelled",
        "STOP from thor_terminal: typed stop",
    )
    h.idle()
    assert h.arm.sent == []
    assert h.arm.pose() == RAISED


@pytest.mark.parametrize(
    "call",
    [lambda app: app.stop("typed stop"), lambda app: app._fault("motor fault")],
    ids=["stop", "fault-stop"],
)
def test_a_stop_from_another_thread_waits_for_the_apps_lock(robot_app, call):
    h = robot_app
    with h.app._lock:
        returned = in_background(lambda: call(h.app))
        assert not returned.wait(BLOCKED_S)
        assert h.app.snapshot().stop_number == 0
    assert returned.wait(WAIT_S)
    assert h.app.snapshot().stop_number == 1


@pytest.mark.parametrize("work", ["an agent move", "a return to rest"])
def test_a_stop_the_motion_worker_never_hears_of_still_ends_its_move(robot_app, work):
    h = robot_app
    if work == "an agent move":
        h.new_task()
        command_id, release = held_move(h)
    else:
        h.arm.set_pose(RAISED)
        reached, release = h.hold_move_after(6)
        command_id = h.client.rest()
        assert reached.wait(WAIT_S)
    # Nothing asks the move to stop: only the stop number tells it.
    h.app._authority.raise_stop("unheard", "console")
    release.set()
    result = final(h, command_id)
    assert result.status == "cancelled"
    assert "stop number is now 1, the move was admitted at 0" in result.reason
    assert len(h.arm.sent) == 6
    assert h.arm.torque_on() and h.arm.released_at == []


# The stop that races the end of the assistance wait


@pytest.fixture
def waiting(robot_app):
    """The app in an assistance wait that runs out at the next clock second."""
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    h.advance(119.0)
    assert h.heard("expired") == []
    return h


def assert_held_where_it_was(h) -> None:
    h.idle()
    assert h.arm.sent == []
    assert h.arm.pose() == RAISED
    assert h.arm.torque_on()


def test_the_wait_is_stepped_under_the_apps_lock(waiting):
    h = waiting
    with h.app._lock:
        h.clock.advance(1.5)
        assert not h.app.wait_step(BLOCKED_S)  # the dispatcher waits for the lock
        assert h.heard("expired") == []
    h.step()
    assert len(h.heard("expired")) == 1


def test_a_stop_just_before_the_wait_runs_out_leaves_no_timed_return(waiting):
    h = waiting
    with h.app._lock:
        h.clock.advance(1.5)  # the wait is over at its next step
        h.app.stop("typed stop")
    h.step()
    assert h.heard("expired") == []
    assert_held_where_it_was(h)
    assert h.heard("rest") == []


def test_a_stop_just_after_the_wait_ran_out_drops_the_timed_return(waiting):
    h = waiting
    with h.app._lock:
        h.clock.advance(1.5)
        h.app._step_wait()  # the wait runs out here and queues the timed return
        assert len(h.heard("expired")) == 1
        assert h.app._busy() == "busy: the timed return to rest is queued"
        h.app.stop("typed stop")
        assert h.app._busy() is None
    assert_held_where_it_was(h)
    assert h.heard("rest") == []


def test_a_stop_while_the_timed_return_reads_the_arm_sends_no_setpoint(waiting):
    h = waiting
    reached, release = h.hold_call("read_all")  # the return's first joint read
    h.advance_until(lambda: h.heard("expired"))
    assert reached.wait(WAIT_S)
    h.app.stop("typed stop")
    release.set()
    assert_held_where_it_was(h)
    data = h.heard("rest")[-1][2]
    assert data["status"] == "refused"
    assert data["reason"] == "a STOP was raised while the return to rest was checked"
    assert h.app.snapshot().owner is Owner.NOBODY


def test_the_timed_return_quotes_the_stop_number_of_the_expiry(waiting):
    h = waiting
    with h.app._lock:
        h.clock.advance(1.5)
        h.app._step_wait()
        # A STOP that the motion worker never hears of: only the number tells.
        h.app._authority.raise_stop("unheard", "console")
    assert_held_where_it_was(h)
    assert h.heard("rest")[-1][2]["reason"] == (
        "stop number 0 does not match the current 1"
    )


@pytest.mark.parametrize(
    "command",
    [
        lambda session: OwnerRequest("quoting", session, 0, 0, "rest", confirmed=True),
        lambda session: Shutdown("quoting", session, 0, 0),
        lambda session: Release("quoting", session, 0, 0),
    ],
    ids=["rest", "shutdown", "release"],
)
def test_motion_work_quotes_the_stop_number_its_command_was_admitted_with(
    robot_app, command
):
    h = robot_app
    h.arm.set_pose(RAISED)
    with h.app._lock:
        h.app._handle(command(h.app.session_id))  # admitted and queued
        # A STOP that the motion worker never hears of: only the number tells.
        h.app._authority.raise_stop("unheard", "console")
    result = final(h, "quoting")
    assert statuses(h, "quoting") == ["accepted", "refused"]
    assert result.reason.endswith("stop number 0 does not match the current 1")
    assert h.arm.sent == [] and h.arm.released_at == []
    assert h.arm.torque_on() and h.app.running


@pytest.mark.parametrize("request_kind", ["rest", "shutdown"])
def test_work_typed_at_the_thor_terminal_quotes_the_stop_number_too(
    robot_app, request_kind
):
    h = robot_app
    h.arm.set_pose(RAISED)
    done = Done()
    with h.app._lock:
        if request_kind == "rest":
            assert h.app.request_rest(True, done) is None
        else:
            assert h.app.request_shutdown(True, done) is None
        h.app._authority.raise_stop("unheard", "console")
    result = done.wait()
    assert result.status == "refused"
    assert result.reason.endswith("stop number 0 does not match the current 1")
    assert h.arm.sent == [] and h.app.running


def test_a_release_typed_at_the_thor_terminal_quotes_the_stop_number_too(robot_app):
    h = robot_app
    done = Done()
    with h.app._lock:
        assert h.app.request_release(done) is None
        # A STOP that the motion worker never hears of: only the number tells.
        h.app._authority.raise_stop("unheard", "console")
    result = done.wait()
    assert (result.status, result.reason) == (
        "refused",
        "stop number 0 does not match the current 1",
    )
    assert h.arm.torque_on() and h.arm.released_at == []


def test_no_timed_return_runs_without_the_robot_connection(waiting):
    h = waiting
    with h.app._lock:
        h.robot_end.break_link()
        h.clock.advance(1.5)
        h.app._step_wait()
        assert h.app._busy() is None
    assert_held_where_it_was(h)
    assert h.heard("rest")[-1][1] == (
        "no timed return to rest: robot connection lost; the arm holds"
    )
    # Asked again when the motion worker gets to a timed return.
    ending = h.app._run_timed_return(None, 0)
    assert (ending.status, ending.reason) == (
        "refused",
        "no timed return to rest: robot connection lost",
    )
    assert h.arm.sent == []


def test_a_timed_return_queued_as_the_connection_dies_does_not_start(waiting):
    h = waiting
    with h.app._lock:
        h.clock.advance(1.5)
        h.app._step_wait()
        assert h.app._busy() == "busy: the timed return to rest is queued"
        h.robot_end.break_link()
    h.wait_local_event("connection")
    assert_held_where_it_was(h)


# Faults


def test_a_motor_fault_latches_once_until_a_reset_that_starts_nothing(robot_app):
    h = robot_app
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
    h.advance(1.0, step_s=0.1)
    assert len(h.heard("fault")) == 1  # raised once, not on every step
    assert h.app.snapshot().stop_number == 1
    assert all(checker.latched() for checker in h.checkers.all())
    latched = "fault-stop latched: motor 3 overheated"
    assert final(h, h.client.rest()).reason == latched
    assert final(h, h.client.release()).reason == latched

    refused = final(h, h.client.reset(stop_number=0))
    assert refused.reason == "stop number 0 does not match the current 1"
    refused = final(h, h.client.reset(session_id="another-run"))
    assert refused.reason == "session id does not match"
    refused = final(h, h.client.reset())
    assert refused.reason == "the arm still reports a fault: motor 3 overheated"
    assert h.app.snapshot().stop_level is StopLevel.FAULT
    h.arm.clear_faults()
    reset_id = h.client.reset()
    assert statuses(h, reset_id) == ["completed"] or final(h, reset_id).status == (
        "completed"
    )
    # The reset cleared the latch and started nothing.
    state = h.client.wait_state(lambda s: s.stop_level == "none")
    assert (state.owner, state.assistance_wait_s) == ("nobody", None)
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    assert not any(checker.latched() for checker in h.checkers.all())
    assert h.arm.torque_on()
    h.new_task()
    assert h.client.run_move(PATH, start_q=REST).status == "completed"


def test_lost_joint_feedback_in_a_hold_is_a_fault_stop(robot_app):
    h = robot_app
    h.arm.fail_reads(2, 10_000)
    h.advance(0.8, step_s=0.02)
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost for 0.")
    assert state.fault_reason.endswith("s in hold")
    assert len(h.heard("fault")) == 1
    assert h.arm.torque_on() and h.arm.released_at == []


def test_joint_feedback_lost_during_a_move_is_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()
    real = h.arm.send_target

    def send_target(q) -> None:
        real(q)
        if len(h.arm.sent) == 1:
            h.arm.fail_reads(1, 10_000)

    h.arm.send_target = send_target
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "failed"
    assert result.reason.startswith("fault: joint feedback lost: joint index 1")
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost: joint index 1")
    assert len(h.heard("fault")) == 1
    assert h.arm.torque_on()


def test_a_driver_fault_during_a_move_is_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()
    real = h.arm.send_target

    def send_target(q) -> None:
        if len(h.arm.sent) == 3:
            raise ArmFault("motor 2 reports overcurrent")
        real(q)

    h.arm.send_target = send_target
    result = h.client.run_move(PATH, start_q=REST)
    assert (result.status, result.reason) == (
        "failed",
        "fault: motor 2 reports overcurrent",
    )
    assert h.sent() == PATH[:3]
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == "motor 2 reports overcurrent"
    assert h.arm.torque_on()


def test_an_error_on_the_motion_worker_is_a_fault_stop_and_the_worker_lives(
    robot_app,
):
    h = robot_app
    h.new_task()
    real = h.arm.send_target

    def send_target(q) -> None:
        raise RuntimeError("driver bug")

    h.arm.send_target = send_target
    result = h.client.run_move(PATH, start_q=REST)
    assert result.status == "failed"
    assert result.reason == "robot app error: RuntimeError('driver bug')"
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == (
        "robot app error in a move: RuntimeError('driver bug')"
    )
    assert state.motion == "hold"
    # After a reset the same worker runs the next motion work.
    h.arm.send_target = real
    assert final(h, h.client.reset()).status == "completed"
    h.arm.set_pose(RAISED)
    assert final(h, h.client.rest()).status == "completed"


def test_an_error_while_a_final_result_goes_out_does_not_end_the_motion_worker(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.robot_end.send_reliable
    failed = []

    def send_reliable(msg_type: int, payload: bytes) -> None:
        if msg_type == Result.TYPE and b'"completed"' in payload and not failed:
            failed.append(payload)
            raise OSError("link write failed")
        real(msg_type, payload)

    h.robot_end.send_reliable = send_reliable
    lost_id = h.client.rest()
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == (
        "robot app error reporting a return to rest: OSError('link write failed')"
    )
    assert len(failed) == 1 and statuses(h, lost_id) == ["accepted"]
    assert h.app._motion_worker.is_alive()
    # The same worker runs and answers the next motion work.
    assert final(h, h.client.reset()).status == "completed"
    assert final(h, h.client.release()).status == "completed"
    assert h.arm.released_at == [pytest.approx(REST, abs=h.config.rest.rest_tol_rad)]


def test_a_driver_fault_outside_a_move_is_a_fault_stop(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)

    def read_all():
        raise ArmFault("encoder fault on joint 4")

    real = h.arm.read_all
    h.arm.read_all = read_all  # the return to rest's first joint read
    result = final(h, h.client.rest())
    assert (result.status, result.reason) == (
        "failed",
        "fault: encoder fault on joint 4",
    )
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == "encoder fault on joint 4"
    assert h.arm.sent == [] and h.arm.torque_on()
    # The worker lives on.
    h.arm.read_all = real
    assert final(h, h.client.reset()).status == "completed"
    assert final(h, h.client.rest()).status == "completed"


def test_a_return_to_rest_that_loses_joint_feedback_is_a_fault_stop(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.arm.send_target

    def send_target(q) -> None:
        real(q)
        if len(h.arm.sent) == 1:
            h.arm.fail_reads(1, 10_000)

    h.arm.send_target = send_target
    result = final(h, h.client.rest())
    assert result.status == "failed"
    assert "fault: joint feedback lost" in result.reason
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason.startswith("joint feedback lost: joint index 1")
    assert h.arm.torque_on() and h.arm.released_at == []


def test_an_error_in_a_link_message_handler_is_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()

    def clear_contact_spots() -> None:
        raise RuntimeError("checker bug")

    h.checkers.agent.clear_contact_spots = clear_contact_spots
    command_id = h.client.checker_state("clear_contact_spots")
    result = final(h, command_id)
    assert statuses(h, command_id) == ["failed"]
    assert result.reason == "robot app error: RuntimeError('checker bug')"
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == (
        "robot app error in a link message: RuntimeError('checker bug')"
    )
    # The dispatcher goes on: the next command is answered.
    assert final(h, h.client.reset()).status == "completed"


def test_an_error_in_the_supervisor_step_is_a_fault_stop(robot_app):
    h = robot_app
    real = h.arm.faults
    calls = []

    def faults() -> list[str]:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("bus error")
        return real()

    h.arm.faults = faults
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert "RuntimeError('bus error')" in state.fault_reason
    assert state.fault_reason.startswith("robot app error in the ")
    h.step()
    assert len(h.heard("fault")) == 1
    assert final(h, h.client.reset()).status == "completed"


def test_an_error_while_receiving_is_a_fault_stop_and_the_dispatcher_goes_on(
    robot_app,
):
    h = robot_app
    real = h.robot_end.receive
    calls = []

    def receive(timeout_s: float):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("socket error")
        return real(timeout_s)

    h.robot_end.receive = receive
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == (
        "robot app error in receiving a link message: OSError('socket error')"
    )
    assert final(h, h.client.reset()).status == "completed"
    assert h.new_task().status == "completed"


@pytest.mark.parametrize("link", ["alive", "lost"])
def test_an_error_before_the_wait_for_a_link_message_is_a_fault_stop(robot_app, link):
    h = robot_app
    if link == "lost":
        h.robot_end.break_link()
        h.wait_local_event("connection")
    real = h.arm.torque_on
    raised = []

    def torque_on() -> bool:
        # The first call the dispatcher makes for how long it may wait.
        if not raised and "_link_wait_s" in [f.function for f in inspect.stack()]:
            raised.append(1)
            raise RuntimeError("driver bug")
        return real()

    h.arm.torque_on = torque_on
    _, text, _ = h.wait_local_event("fault")
    assert text.startswith(
        "FAULT-STOP: robot app error in serving the link: RuntimeError('driver bug')"
    )
    # The dispatcher goes on: it makes its passes and serves a STOP.
    h.step()
    h.app.request_stop("typed stop")
    h.step()
    assert h.app.snapshot().stop_number == 2
    assert h.arm.torque_on() and h.arm.released_at == []


def test_an_error_that_repeats_on_every_step_is_one_fault_stop(robot_app):
    h = robot_app

    def temperatures() -> list[float]:
        raise RuntimeError("sensor bug")

    h.arm.temperatures = temperatures  # read for every ROBOT_STATE
    h.clock.advance(0.1)
    for _ in range(5):
        h.step()
    state = h.app.snapshot()
    assert state.stop_level is StopLevel.FAULT
    assert state.fault_reason == (
        "robot app error in the state publisher: RuntimeError('sensor bug')"
    )
    assert state.stop_number == 1
    assert len(h.heard("fault")) == 1
    assert h.app.running


def test_a_gripper_fault_is_a_fault_stop(robot_app):
    h = robot_app
    h.new_task()

    def gripper_command(opening, max_effort, mode, cancel):
        raise ArmFault("jaw motor overcurrent")

    h.arm.gripper_command = gripper_command
    result = final(h, h.client.gripper(0.5))
    assert result.status == "failed"
    state = h.client.wait_state(lambda s: s.stop_level == "fault")
    assert state.fault_reason == "gripper fault: jaw motor overcurrent"


# Robot connection lost


def test_connection_lost_during_an_agent_move_ends_it_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.robot_end.break_link()
    _, text, data = h.wait_local_event("connection")
    assert text == (
        "robot connection lost: the arm holds; the agent's ownership ended; "
        "rest can still be typed at the Thor terminal"
    )
    assert data == {"owner_before": "agent", "wait_cancelled": False}
    release.set()
    h.idle()
    assert h.sent() == PATH[:6]
    stored = h.app._results.get(move_id)
    assert (stored.status, stored.reason) == ("cancelled", LINK_LOST)
    state = h.app.snapshot()
    assert (state.owner, state.owner_number, state.stop_number) == (Owner.NOBODY, 2, 0)
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.running and not h.app.link_alive
    # Handled once.
    h.step()
    h.step()
    assert len(h.heard("connection")) == 1


def test_connection_lost_stops_a_running_jaw_command(robot_app):
    h = robot_app
    h.new_task()
    jaws_waiting = h.hold_jaws_at(0.6)
    jaw_id = h.client.gripper(0.0)
    assert jaws_waiting.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    assert h.app._worker.wait_idle(WAIT_S)
    assert h.app._results.get(jaw_id).status == "cancelled"
    assert h.arm.jaw_stiffness == "light"
    assert h.arm.gripper_read() == pytest.approx(0.6)


def test_connection_lost_cancels_the_assistance_wait(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.new_task()
    ask_for_help(h)
    h.robot_end.break_link()
    _, text, data = h.wait_local_event("connection")
    assert data == {"owner_before": "nobody", "wait_cancelled": True}
    assert text == (
        "robot connection lost: the arm holds; the assistance wait is cancelled; "
        "rest can still be typed at the Thor terminal"
    )
    assert h.app.snapshot().owner_number == 2  # nobody owned the arm already
    h.advance(125.0)
    assert h.heard("expired") == []
    assert h.arm.sent == []


def test_connection_lost_during_the_timed_return_ends_it_in_a_hold(robot_app):
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
    assert len(h.arm.sent) == 5
    assert h.heard("rest")[-1][2]["status"] == "stopped"
    assert h.app.motion_state is MotionState.HOLD
    assert h.app.snapshot().owner is Owner.NOBODY
    assert h.arm.torque_on() and h.arm.released_at == []


def test_connection_lost_ends_a_rest_asked_for_over_it_in_a_hold(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    h.idle()
    assert len(h.arm.sent) == 5
    stored = h.app._results.get(rest_id)
    assert stored.status == "cancelled"
    assert stored.reason.startswith(LINK_LOST)
    assert h.app.snapshot().owner is Owner.NOBODY
    assert h.arm.torque_on()


def test_connection_lost_lets_a_rest_typed_at_the_thor_terminal_go_on(robot_app):
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
    assert len(h.arm.sent) == 209
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.motion_state is MotionState.REST_HOLD


def test_connection_lost_lets_a_shutdown_typed_at_the_thor_terminal_go_on(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    done = Done()
    assert h.app.request_shutdown(True, done) is None
    assert reached.wait(WAIT_S)
    h.robot_end.break_link()
    h.wait_local_event("connection")
    release.set()
    assert done.wait().status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.sent) == 209
    assert len(h.arm.released_at) == 1


@pytest.mark.parametrize("request_kind", ["rest", "release", "shutdown"])
def test_connection_lost_does_not_drop_work_typed_at_the_thor_terminal(
    robot_app, request_kind
):
    h = robot_app
    done = Done()
    with h.app._lock:  # the motion worker cannot begin the work
        if request_kind == "rest":
            assert h.app.request_rest(True, done) is None
        elif request_kind == "release":
            assert h.app.request_release(done) is None
        else:
            assert h.app.request_shutdown(True, done) is None
        h.robot_end.break_link()
        h.app._check_link()
        assert h.heard("connection")
    assert done.wait().status == "completed"
    h.idle()
    assert h.arm.torque_on() == (request_kind == "rest")


def test_the_thor_terminal_still_works_with_the_connection_down(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.robot_end.break_link()
    data = h.wait_local_event("connection")[2]
    assert data == {"owner_before": "nobody", "wait_cancelled": False}
    assert h.app.stop("typed stop") == 1
    done = Done()
    assert h.app.request_rest(True, done) is None
    assert done.wait().status == "completed"
    done = Done()
    assert h.app.request_shutdown(True, done) is None
    assert done.wait().status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.released_at) == 1 and not h.arm.torque_on()


# Rest, release and shutdown


def test_a_typed_rest_returns_the_arm_and_ends_powered(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    command_id = h.client.rest()
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert result.end_q == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert result.detail == {"outcome": "at_rest"}
    state = h.client.wait_state(
        lambda s: s.motion == "rest_hold" and s.owner == "nobody"
    )
    assert state.torque_on
    assert h.arm.released_at == []
    texts = [text for _, text, _ in h.heard("rest")]
    assert texts[0].startswith("return to rest: the arm moves for 4.2 s")
    assert texts[1:] == ["return to rest: the arm is at rest, motors on"]


def test_a_typed_rest_needs_the_path_confirmed(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    command_id = h.client.rest(confirmed=False)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == CONFIRM_PATH
    h.idle()
    assert h.arm.sent == []


@pytest.mark.parametrize("command", ["rest", "release", "shutdown"])
def test_an_operator_command_is_refused_as_busy_while_a_move_runs(robot_app, command):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    command_id = getattr(h.client, command)()
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == "busy: a move is running"
    release.set()
    assert final(h, move_id).status == "completed"
    assert h.sent() == PATH
    assert h.arm.torque_on() and h.app.running


def test_a_return_whose_path_the_recovery_checker_refuses_sends_no_setpoint(
    robot_app,
):
    h = robot_app
    h.arm.set_pose(RAISED)
    recovery = h.checkers.recovery
    path_check, setpoint_check = recovery.check_path, recovery.check_setpoint

    def barred(what: str, q) -> None:
        if q[0] < 0.1:  # half-way down the path
            raise Refusal(f"{what}: joint index 0 may not pass 0.1 rad")

    def check_path(q_start, setpoints, dt) -> None:
        path_check(q_start, setpoints, dt)
        for index, q in enumerate(setpoints):
            barred(f"setpoint {index}", q)

    def check_setpoint(q_prev, q_next, dt) -> None:
        setpoint_check(q_prev, q_next, dt)
        barred("setpoint", q_next)

    recovery.check_path, recovery.check_setpoint = check_path, check_setpoint
    result = final(h, h.client.rest())
    assert result.status == "refused"
    assert result.reason.startswith("the path to rest was refused: setpoint ")
    h.idle()
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.app.snapshot().owner_number == 0  # recovery never took the arm


def test_a_return_that_cannot_prove_rest_ends_failed_in_a_hold(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.arm.read_all
    calls = []

    def read_all():
        calls.append(1)
        if len(calls) == 5:  # the first of the two reads that prove rest
            raise ReadFailed("joint index 3: the read failed")
        return real()

    h.arm.read_all = read_all
    rest_id = h.client.rest()
    result = final(h, rest_id)
    assert (result.status, result.reason) == (
        "failed",
        "a joint read failed: joint index 3: the read failed",
    )
    assert len(h.arm.sent) == 209
    state = h.client.wait_state(
        lambda s: (
            (s.last_result or {}).get("command_id") == rest_id and s.owner == "nobody"
        )
    )
    assert state.motion == "hold"  # not rest_hold: rest was not proven
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_rest_typed_at_the_thor_terminal_is_refused_under_a_fault_stop(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    h.arm.clear_faults()  # the latch stays until a reset
    before = h.app.snapshot()
    done = Done()
    assert h.app.request_rest(True, done) is None
    result = done.wait()
    latched = "fault-stop latched: motor 3 overheated"
    assert (result.status, result.reason) == ("refused", latched)
    h.idle()
    assert h.app.snapshot() == before  # recovery never took the arm
    assert [text for _, text, _ in h.heard("rest")] == [
        f"return to rest refused: {latched}"
    ]
    assert h.arm.sent == [] and h.arm.pose() == RAISED


def test_stop_ends_a_typed_rest_in_a_hold(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    rest_id = h.client.rest()
    assert reached.wait(WAIT_S)
    h.client.stop()
    h.client.wait_state(lambda s: s.stop_number == 1)
    release.set()
    result = final(h, rest_id)
    assert result.status == "cancelled"
    assert result.reason.startswith("STOP from console: operator pressed stop")
    assert len(h.arm.sent) == 5
    state = h.client.wait_state(lambda s: s.motion == "hold")
    assert (state.owner, state.torque_on) == ("nobody", True)


def test_release_switches_the_motors_off_at_measured_rest(robot_app):
    h = robot_app
    h.new_task()
    command_id = h.client.release()
    assert final(h, command_id).status == "completed"
    assert statuses(h, command_id) == ["accepted", "completed"]
    state = h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert not state.torque_on
    assert h.arm.released_at == [pytest.approx(REST, abs=1e-6)]
    # Motors off is the end of this run: nothing moves or starts any more.
    refused = h.client.run_move(PATH, start_q=REST)
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    refused = final(h, h.client.gripper(0.5))
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    assert final(h, h.client.request_owner("nobody")).status == "completed"
    refused = h.client.start_task()
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    refused = final(h, h.client.open_jaws())
    assert (refused.status, refused.reason) == ("refused", MOTORS_OFF)
    assert final(h, h.client.rest()).reason == "the motors are off"
    assert final(h, h.client.release()).reason == "the motors are already off"
    assert h.arm.sent == [] and h.arm.jaw_commands == []
    h.advance(1.0, step_s=0.1)
    assert h.app.snapshot().stop_level is StopLevel.NONE


def test_release_needs_both_joint_reads_at_rest(robot_app):
    h = robot_app
    h.arm.set_pose([0.04, 0.0, 0.0, 0.0, 0.0, 0.0])
    real = h.arm.read_all
    calls = []

    def read_all():
        readings = real()
        calls.append(1)
        if len(calls) == 1:  # 0.2 s later the arm is 0.08 rad from rest
            h.arm.set_pose([0.08, 0.0, 0.0, 0.0, 0.0, 0.0])
        return readings

    h.arm.read_all = read_all
    result = final(h, h.client.release())
    assert result.status == "refused"
    assert result.reason.startswith(
        "the arm is not at rest: joint index 0 is at 0.0800 rad"
    )
    assert h.arm.torque_on() and h.arm.released_at == []


def test_release_away_from_rest_is_refused_and_the_motors_stay_on(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    command_id = h.client.release()
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "refused"]
    assert result.reason.startswith("the arm is not at rest: joint index 0")
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.heard("refused")[-1][1] == (
        f"release refused: {result.reason}; the motors stay on"
    )


@pytest.mark.parametrize(
    ("quoted", "reason"),
    [
        ({"session_id": "another-run"}, "session id does not match"),
        ({"stop_number": 7}, "stop number 7 does not match the current 0"),
        ({"owner_number": 7}, "owner number 7 does not match the current 0"),
    ],
    ids=["session id", "stop number", "owner number"],
)
@pytest.mark.parametrize("command", ["release", "shutdown", "open_jaws"])
def test_an_operator_command_quoting_an_old_value_is_refused(
    robot_app, command, quoted, reason
):
    h = robot_app
    command_id = getattr(h.client, command)(**quoted)
    result = final(h, command_id)
    assert statuses(h, command_id) == ["refused"]
    assert result.reason == reason
    h.idle()
    assert h.arm.torque_on() and h.app.running
    assert h.arm.jaw_commands == []


def test_shutdown_returns_to_rest_releases_there_and_closes_the_app(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    command_id = h.client.shutdown()
    result = final(h, command_id)
    assert statuses(h, command_id) == ["accepted", "completed"]
    assert h.app.wait_closed(WAIT_S)
    assert not h.app.running
    assert len(h.arm.sent) == 209
    assert len(h.arm.released_at) == 1  # the fixture checks that it was at rest
    assert not h.arm.torque_on()
    assert h.heard("shutdown")[0][1] == (
        "shutdown: the motors are off at rest; the robot app closes"
    )
    assert result.detail == {"outcome": "at_rest"}


def test_a_shutdown_whose_return_is_refused_leaves_the_app_running(robot_app):
    h = robot_app
    h.new_task()
    h.arm.jaw_object(0.4)
    assert final(h, h.client.gripper(0.0, mode="close_until_contact")).status == (
        "completed"
    )
    result = final(h, h.client.shutdown())
    assert result.status == "refused"
    assert result.reason.startswith(
        "the return to rest ended refused: the jaws are not proven empty (loaded)"
    )
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.heard("shutdown")[0][1].endswith(
        "the robot app keeps running and the arm holds"
    )
    # The app still takes commands.
    assert final(h, h.client.request_owner("nobody")).status == "completed"


def test_a_shutdown_whose_release_is_refused_leaves_the_motors_on(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    real = h.app._rest.request

    def request(*args, **kwargs):
        outcome = real(*args, **kwargs)
        h.app.stop("second thoughts")  # after the return, before the release
        return outcome

    h.app._rest.request = request
    result = final(h, h.client.shutdown())
    assert result.status == "failed"
    assert result.reason == (
        "at rest with the motors on; release refused: stop number 0 does not "
        "match the current 1"
    )
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.app.running
    assert h.app.motion_state is MotionState.REST_HOLD


def test_a_shutdown_with_the_motors_off_already_only_closes_the_app(robot_app):
    h = robot_app
    assert final(h, h.client.release()).status == "completed"
    h.client.wait_state(lambda s: s.motion == "rest_unpowered")
    assert final(h, h.client.shutdown()).status == "completed"
    assert h.app.wait_closed(WAIT_S)
    assert len(h.arm.released_at) == 1
    assert h.heard("rest") == []  # no return to rest was asked for


def test_a_stop_as_the_shutdowns_return_ends_leaves_the_motors_on(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)

    def stop_at_rest(kind: str, text: str, data: dict) -> None:
        if kind == "rest" and data.get("status") == "at_rest":
            # A STOP between the return and the release; only the number tells.
            h.app._authority.raise_stop("unheard", "console")

    h.app.add_event_listener(stop_at_rest)
    result = final(h, h.client.shutdown())
    assert (result.status, result.reason) == (
        "failed",
        "at rest with the motors on; release refused: stop number 0 does not "
        "match the current 1",
    )
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    assert h.app.running and not h.app.wait_closed(0.0)
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_release_is_refused_while_a_fault_stop_is_latched(robot_app):
    h = robot_app
    h.arm.set_fault("motor 3 overheated")
    h.client.wait_state(lambda s: s.stop_level == "fault")
    h.arm.clear_faults()  # the latch stays until a reset
    latched = "fault-stop latched: motor 3 overheated"
    refused = final(h, h.client.release())
    assert (refused.status, refused.reason) == ("refused", latched)
    done = Done()
    assert h.app.request_release(done) is None
    result = done.wait()
    assert (result.status, result.reason) == ("refused", latched)
    h.idle()
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_stop_ends_a_shutdown_and_nothing_is_released(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    command_id = h.client.shutdown()
    assert reached.wait(WAIT_S)
    h.client.stop()
    h.client.wait_state(lambda s: s.stop_number == 1)
    release.set()
    result = final(h, command_id)
    assert result.status == "cancelled"
    assert result.reason.startswith("the return to rest ended cancelled: STOP from")
    assert h.app.running
    assert h.arm.torque_on() and h.arm.released_at == []


def test_a_rest_typed_at_the_thor_terminal(make_robot_app):
    asked = []

    def confirm(prompt: str) -> bool:
        asked.append(prompt)
        return len(asked) > 1  # no the first time, yes after that

    h = make_robot_app(confirm=confirm)
    h.arm.set_pose(RAISED)
    done = Done()
    assert h.app.request_rest(False, done) == CONFIRM_PATH
    assert asked == []  # the terminal said no itself
    assert done.wait().status == "refused"
    done = Done()
    assert h.app.request_rest(None, done) == CONFIRM_PATH
    assert asked == [PATH_CLEAR_PROMPT]
    assert (done.wait().status, done.results[0].reason) == ("refused", CONFIRM_PATH)
    h.idle()
    assert h.arm.sent == []
    done = Done()
    assert h.app.request_rest(None, done) is None
    result = done.wait()
    assert result.status == "completed"
    assert result.command_id.startswith("thor-terminal-rest-")
    assert len(done.results) == 1
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
    # Nothing typed there is answered over the link.
    h.step()
    assert not any(
        command_id.startswith("thor-terminal") for command_id in h.client._results
    )
    state = h.client.wait_state(
        lambda s: (s.last_result or {}).get("command_id") == result.command_id
    )
    assert state.motion == "rest_hold"


def test_a_release_typed_at_the_thor_terminal_needs_no_robot_connection(robot_app):
    h = robot_app
    h.robot_end.break_link()
    h.wait_local_event("connection")
    done = Done()
    assert h.app.request_release(done) is None
    result = done.wait()
    assert (result.status, result.reason) == ("completed", "")
    assert result.command_id == "thor-terminal-release-1"
    assert h.arm.released_at == [pytest.approx(REST, abs=1e-6)]
    assert h.app.motion_state is MotionState.REST_UNPOWERED
    assert h.heard("release")[-1][1] == "torque released at rest: the motors are off"
    # Only a shutdown ends the app.
    assert h.app.running and not h.app.wait_closed(0.0)


def test_a_release_typed_at_the_thor_terminal_away_from_rest_is_refused(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    done = Done()
    assert h.app.request_release(done) is None
    result = done.wait()
    assert result.status == "refused"
    assert result.reason.startswith("the arm is not at rest: joint index 0")
    assert len(done.results) == 1
    h.idle()
    assert h.arm.torque_on() and h.arm.released_at == []
    assert h.arm.sent == [] and h.arm.pose() == RAISED
    assert h.heard("refused")[-1][1] == (
        f"release refused: {result.reason}; the motors stay on"
    )


def test_without_a_confirm_function_nothing_is_ever_confirmed(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    assert h.app.request_rest(None) == CONFIRM_PATH
    h.idle()
    assert h.arm.sent == []


def test_a_confirm_function_that_raises_confirms_nothing(make_robot_app):
    def confirm(prompt: str) -> bool:
        raise OSError("the terminal is gone")

    h = make_robot_app(confirm=confirm)
    assert h.app.request_rest(None) == CONFIRM_PATH


def test_requests_typed_at_the_thor_terminal_are_refused_as_busy(robot_app):
    h = robot_app
    h.arm.set_pose(RAISED)
    reached, release = h.hold_move_after(5)
    first = Done()
    assert h.app.request_rest(True, first) is None
    assert reached.wait(WAIT_S)
    for request in (
        lambda done: h.app.request_rest(True, done),
        h.app.request_release,
        lambda done: h.app.request_shutdown(True, done),
    ):
        done = Done()
        assert request(done) == "busy: a return to rest is running"
        assert done.wait().status == "refused"
        assert len(done.results) == 1
    release.set()
    assert first.wait().status == "completed"
    assert h.app.running


def test_close_away_from_rest_leaves_the_motors_on_and_holding(robot_app):
    h = robot_app
    h.new_task()
    assert h.client.run_move(PATH, start_q=REST).status == "completed"
    h.app.close()
    assert h.app.wait_closed(0.0) and not h.app.running
    assert not h.app._dispatcher.is_alive()
    assert not h.app._motion_worker.is_alive()
    assert not h.app._worker._thread.is_alive()
    assert h.arm.torque_on()
    assert h.arm.released_at == []
    assert h.arm.last_target()[0] == PATH[-1]  # still holding its last target
    # Closed, the app takes no request any more.
    done = Done()
    assert h.app.request_rest(True, done) == NOT_RUNNING
    assert h.app.request_release() == NOT_RUNNING
    assert h.app.request_shutdown(True) == NOT_RUNNING
    assert done.wait().status == "refused"
    h.app.close()  # a second close changes nothing


def test_close_during_a_move_ends_it_in_a_hold(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    closed = in_background(h.app.close)
    with h.app._motion_cond:  # until close() has told the move to stop
        assert h.app._motion_cond.wait_for(
            lambda: getattr(h.app._running, "stop_reason", None) == CLOSING, WAIT_S
        )
    assert not closed.wait(BLOCKED_S)  # close() waits for the motion worker
    assert not h.app.running
    release.set()
    assert closed.wait(WAIT_S)
    result = final(h, move_id)
    assert (result.status, result.reason) == ("cancelled", CLOSING)
    assert h.sent() == PATH[:6]
    assert h.arm.torque_on() and h.arm.released_at == []


def test_close_before_start_ends_the_gripper_worker(make_robot_app):
    h = make_robot_app(start=False)
    h.app.close()
    assert h.app.wait_closed(0.0)
    assert not h.app._worker._thread.is_alive()
    assert h.arm.sent == [] and h.arm.released_at == []


# RESULTs, EVENTs and link messages the app does not take


def test_a_repeated_command_is_answered_again_and_not_executed_again(robot_app):
    h = robot_app
    task = h.new_task()
    h.client.request_owner("agent", command_id=task.command_id)
    h.client.wait_until(lambda: len(h.client.results(task.command_id)) == 2)
    assert h.client.results(task.command_id)[1] == task
    assert h.app.snapshot().owner_number == 1  # not taken a second time

    move = h.client.run_move(PATH, start_q=REST)
    again = h.client.move(PATH, start_q=PATH[-1], command_id=move.command_id)
    h.client.permit(again, len(PATH) - 1)
    h.client.wait_until(lambda: len(h.client.results(again)) == 3)
    assert h.client.results(again)[2] == move
    h.idle()
    assert h.sent() == PATH  # the arm did not move a second time


def test_a_command_repeated_while_it_runs_is_answered_accepted(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    h.client.move(PATH, start_q=REST, command_id=move_id)
    h.client.wait_until(lambda: len(h.client.results(move_id)) == 2)
    assert statuses(h, move_id) == ["accepted", "accepted"]
    release.set()
    assert final(h, move_id).status == "completed"
    assert statuses(h, move_id) == ["accepted", "accepted", "completed"]
    assert h.sent() == PATH


def test_the_results_of_the_last_256_commands_are_kept(robot_app):
    h = robot_app
    assert RESULT_IDS_KEPT == 256
    first = h.client.request_owner("help_needed", reason="one")
    assert final(h, first).reason == "control owner is nobody, not agent"
    last = ""
    for _ in range(RESULT_IDS_KEPT - 1):
        last = h.client.checker_state("close_grasp")
    final(h, last)
    h.step()
    refusals = len(h.heard("refused"))
    # 256 commands are kept: the first is answered again, not executed again.
    h.client.request_owner("help_needed", reason="one", command_id=first)
    h.client.wait_until(lambda: len(h.client.results(first)) == 2)
    h.step()
    assert len(h.heard("refused")) == refusals
    # One more command pushes it out: now it is a new command.
    h.new_task()
    h.client.request_owner("help_needed", reason="one", command_id=first)
    h.client.wait_until(lambda: len(h.client.results(first)) == 3)
    assert h.client.results(first)[2].status == "completed"


def test_a_running_command_is_not_pushed_out_of_the_results(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    last = ""
    for _ in range(RESULT_IDS_KEPT + 4):
        last = h.client.checker_state("forget")
    final(h, last)
    # Still known: the repeated MOVE is answered and not run a second time.
    h.client.move(PATH, start_q=REST, command_id=move_id)
    h.client.wait_until(lambda: len(h.client.results(move_id)) == 2)
    assert statuses(h, move_id) == ["accepted", "accepted"]
    release.set()
    assert final(h, move_id).status == "completed"
    h.idle()
    h.step()
    assert statuses(h, move_id) == ["accepted", "accepted", "completed"]
    assert h.sent() == PATH


def test_the_results_store_never_forgets_a_command_that_has_not_ended():
    results, sent = _Results(keep=3), []
    for index in range(3):
        assert results.put(Result(f"c-{index}", "accepted"), sent.append)
    assert results.room_for("c-0")  # known already
    assert not results.room_for("new")  # it would push an unfinished one out
    # A RESULT that is stored all the same is delivered and not kept.
    assert results.put(Result("new", "refused", "busy"), sent.append)
    assert results.get("new") is None
    assert [results.get(f"c-{index}").status for index in range(3)] == ["accepted"] * 3
    # A further unfinished command is kept beside the others, over the bound.
    assert results.put(Result("more", "accepted"), sent.append)
    assert results.get("c-0").status == "accepted"
    # Once commands have ended, the oldest of those make room again.
    assert results.put(Result("c-1", "completed"), sent.append)
    assert results.get("c-1") is None  # four were kept, one over the bound
    assert results.put(Result("c-0", "cancelled"), sent.append)
    assert results.get("c-0").status == "cancelled"
    assert results.room_for("next")
    assert results.put(Result("next", "accepted"), sent.append)
    assert results.get("c-0") is None
    assert [results.get(name).status for name in ("c-2", "more", "next")] == (
        ["accepted"] * 3
    )
    assert [result.command_id for result in sent] == [
        *("c-0", "c-1", "c-2", "new", "more", "c-1", "c-0", "next")
    ]


def test_a_command_is_refused_when_only_unfinished_commands_are_kept(
    robot_app, monkeypatch
):
    h = robot_app
    # Only jaw commands can be unfinished in numbers: their queue is unbounded here.
    monkeypatch.setattr(gripper_module, "MAX_WAITING", 10 * RESULT_IDS_KEPT)
    h.new_task()
    jaws_held = h.hold_jaws_at(0.5)
    first = h.client.gripper(0.0)
    assert jaws_held.wait(WAIT_S)
    kept = [h.client.gripper(0.6) for _ in range(RESULT_IDS_KEPT - 1)]
    over = [h.client.gripper(0.6) for _ in range(3)]
    for command_id in over:
        result = final(h, command_id)
        assert statuses(h, command_id) == ["refused"]
        assert result.reason == "busy: 256 commands have not ended"
        assert h.app._results.get(command_id) is None
    assert [statuses(h, command_id) for command_id in kept] == [["accepted"]] * 255
    # The running command is still known: sent again, it is not queued again.
    h.client.gripper(0.0, command_id=first)
    h.client.wait_until(lambda: len(h.client.results(first)) == 2)
    assert statuses(h, first) == ["accepted", "accepted"]
    assert len(h.app._worker._queue) == RESULT_IDS_KEPT - 1
    # A STOP ends them all, each with one final RESULT, and commands are taken again.
    h.client.stop()
    for command_id in [first, *kept]:
        final(h, command_id)
    assert all(
        [status for status in statuses(h, command_id) if status != "accepted"]
        == ["cancelled"]
        for command_id in [first, *kept]
    )
    h.client.wait_state(lambda s: s.stop_number == 1)
    assert h.new_task().status == "completed"


def test_every_command_gets_exactly_one_final_result(robot_app):
    h = robot_app
    h.new_task()
    move_id, release = held_move(h)
    ids = [
        move_id,
        h.client.move(PATH, start_q=REST),
        h.client.gripper(0.5, owner_number=0),
        h.client.checker_state("close_grasp"),
        h.client.rest(),
        h.client.release(),
        h.client.open_jaws(),
        h.client.shutdown(),
        h.client.reset(),
        h.client.request_owner("human"),
    ]
    h.client.stop()
    release.set()
    for command_id in ids:
        final(h, command_id)
    h.idle()
    h.step()
    for command_id in ids:
        ended = [status for status in statuses(h, command_id) if status != "accepted"]
        assert len(ended) == 1, (command_id, statuses(h, command_id))
    assert h.app.running and h.arm.torque_on()


def test_a_second_final_result_for_a_command_is_never_sent(robot_app):
    h = robot_app
    command_id = h.client.request_owner("human")
    assert final(h, command_id).status == "refused"
    h.app._send_result(Result(command_id, "completed"))
    h.app._send_result(Result(command_id, "accepted"))
    probe = h.client.request_owner("human")
    final(h, probe)  # sent after them: they would have arrived by now
    assert statuses(h, command_id) == ["refused"]
    assert h.app._results.get(command_id).status == "refused"


def test_a_result_that_cannot_be_encoded_is_sent_without_its_detail(robot_app):
    h = robot_app
    h.new_task()

    def gripper_command(opening, max_effort, mode, cancel):
        return GripperOutcome("reached", float("nan"))

    h.arm.gripper_command = gripper_command
    command_id = h.client.gripper(0.5)
    result = final(h, command_id)
    assert (result.status, result.detail) == ("completed", {})
    assert statuses(h, command_id) == ["accepted", "completed"]


def test_events_reach_every_listener_even_when_one_raises(robot_app):
    h = robot_app
    heard = []

    def broken(kind, text, data):
        raise RuntimeError("listener bug")

    h.app.add_event_listener(broken)
    h.app.add_event_listener(lambda kind, text, data: heard.append((kind, data)))
    assert h.app.stop("typed stop") == 1
    _, text, data = h.heard("stop")[0]
    assert text.startswith("STOP from thor_terminal: typed stop; the arm holds")
    assert heard == [("stop", data)]
    event = h.client.wait_event("stop")
    assert (event.seq, event.text, event.data) == (1, text, data)
    assert h.app.snapshot().stop_level is StopLevel.STOP


def test_link_messages_the_app_does_not_take_are_dropped(robot_app, caplog):
    h = robot_app
    for message in (
        h.client.state,
        h.client.welcome,
        Result("some-command", "completed"),
        Event(1, "stop", "not for the robot app"),
        h.client.state,
        ViewStatus(live=True, age_s=0.1),
    ):
        h.client.send(message)
    h.step()
    h.step()
    dropped = [r.getMessage() for r in caplog.records if "dropped" in r.getMessage()]
    # Each kind is logged once, however many arrive.
    assert sorted(dropped) == [
        f"the robot app takes no {kind} link message; dropped"
        for kind in ("Event", "Result", "RobotState", "Welcome")
    ]
    state = h.app.snapshot()
    assert (state.stop_level, state.stop_number, state.owner_number) == ("none", 0, 0)
    assert h.client.unexpected == []
    assert h.new_task().status == "completed"
