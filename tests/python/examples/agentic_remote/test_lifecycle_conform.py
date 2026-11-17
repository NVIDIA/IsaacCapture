# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rest, motors off, motors on again and the ends of a session on the Thor.

They decide as the functional code decides: APAI's rest tool
(apai.skills.rest_pose.move_to_rest) and the launcher that ran it on the
robot's own computer. The functional rest is: up to the ready pose unless the
arm is there, the fold onto the rest pose in measured steps, the proof of rest
by measurement, the jaws closed with the driver's close, the motors off. What
the Thor keeps of its own: the motors go off only at measured rest.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest
from isaaccapture_examples.agentic_remote.protocol.messages import (
    REQUEST_MOTORS_OFF,
    REQUEST_PARK,
    Result,
)
from isaaccapture_examples.agentic_remote.robot_app import __main__ as program
from isaaccapture_examples.agentic_remote.robot_app import app as app_module
from isaaccapture_examples.agentic_remote.robot_app import rest as rest_module
from isaaccapture_examples.agentic_remote.robot_app.app import RobotApp
from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    MotionState,
    MoveOutcome,
    StopLevel,
)
from isaaccapture_examples.agentic_remote.robot_app.rest import (
    ReturnToRest,
    SwitchedArm,
)
from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
from test_rest import RAISED, REST, Rig
from test_tools_conform import HOME, functional_rest

apai = pytest.mark.apai

WAIT_S = 10.0


def with_route(
    rig: Rig, closes: list[str] | None = None, why: str | None = None
) -> Rig:
    """The rig's return to rest on the functional route, with a jaw close."""

    def close_jaws(should_stop: Any) -> str | None:
        closes.append("close")
        rig.events.append(("close", "the driver's close"))
        return why

    rig.rest = ReturnToRest(
        rig.arm,
        rig.authority,
        rig.checkers,
        rig.executor,
        rig.jaw_state,
        rig.history,
        rig.config,
        rig.clock,
        rig.emit,
        rig.motion,
        lambda: rig.region,
        lambda: rig.contact_hold,
        lambda: rig.moving,
        jaws_busy=lambda: rig.jaws_busy,
        route=rest_module.functional_route(HOME),
        close_jaws=None if closes is None else close_jaws,
    )
    return rig


@pytest.fixture
def up(sim_arm, checkers, config, clock, monkeypatch) -> Rig:
    """The arm holding at RAISED, the return to rest on the functional route."""
    rig = Rig(sim_arm, checkers, config, clock, monkeypatch)
    rig.arm.set_pose(RAISED)
    rig.closes = []
    return with_route(rig, rig.closes)


# The route


@apai
def test_the_route_reads_its_numbers_from_apais_rest_tool() -> None:
    from apai.skills import rest_pose

    route = rest_module.functional_route(HOME)
    assert route.track_tol_rad == rest_pose.TRACK_TOL_RAD
    assert (route.home_s, route.segment_s) == (rest_pose.HOME_S, rest_pose.SEGMENT_S)
    assert route.waypoints is rest_pose.fold_waypoints
    assert route.home_q == tuple(HOME)
    # The rig's file measures rest with the tolerance APAI's tool releases at.
    assert RobotAppConfig.default().rest.rest_tol_rad <= rest_pose.REST_TOL_RAD


@apai
def test_the_thors_rest_drives_the_joint_targets_of_apais_rest_tool(
    up: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What APAI's tool commands from the same pose, on an arm that records.
    functional: list[tuple] = []
    recorder = types.SimpleNamespace(motion_rate_hz=50.0)
    functional_rest(monkeypatch, recorder, RAISED, functional)
    goals = [entry[1] for entry in functional]
    seconds = [entry[2] for entry in functional]
    assert len(goals) == 5  # up to the ready pose, then the fold's four steps
    # The Thor plans the same goals, in order, each no faster than APAI's.
    legs = up.rest.legs(RAISED)
    assert [list(leg.goal) for leg in legs] == [pytest.approx(goal) for goal in goals]
    assert [leg.min_duration_s for leg in legs] == seconds
    # And it drives them: one move per leg, each ending on its goal. As in
    # APAI's tool the fold starts from the pose measured after the leg up.
    runs = up.spy_run()
    outcome = up.request()
    assert outcome.status == rest_module.AT_REST, outcome.reason
    driven = [args[0][-1] for args, _ in runs]
    assert driven == [pytest.approx(goal, abs=1e-4) for goal in goals]
    assert driven[0] == pytest.approx(HOME, abs=1e-12)
    assert driven[-1] == pytest.approx(REST, abs=1e-12)
    assert up.arm.pose() == pytest.approx(REST, abs=up.config.rest.rest_tol_rad)
    # Every leg is told at the terminal, by its name.
    told = [text for kind, text in up.events if kind == "rest" and "moves for" in text]
    assert [text.split(": ")[1] for text in told] == [
        rest_module.LEG_UP,
        *[rest_module.LEG_FOLD.format(step=step, steps=4) for step in (1, 2, 3, 4)],
    ]


@apai
def test_near_the_ready_pose_the_arm_folds_without_the_leg_up(up: Rig) -> None:
    # APAI: the leg up is left out within its tracking tolerance of that pose.
    near = [value + 0.05 for value in HOME]
    assert up.rest.legs(near)[0].name == "fold step 1 of 4"
    far = [value + 0.09 for value in HOME]
    assert up.rest.legs(far)[0].name == rest_module.LEG_UP


@apai
def test_a_leg_that_does_not_track_ends_in_a_hold_with_the_motors_on(
    up: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    off = [value + 0.2 for value in HOME]  # where the leg up really ended
    real = up.executor.run

    def short(setpoints: Any, *args: Any, **kwargs: Any) -> MoveOutcome:
        ended = real(setpoints, *args, **kwargs)
        return MoveOutcome("completed", "", ended.index, off)

    monkeypatch.setattr(up.executor, "run", short)
    outcome = up.request()
    assert outcome.status == rest_module.FAILED
    assert outcome.reason.startswith(f"{rest_module.LEG_UP} did not track")
    assert up.motion.get() is MotionState.HOLD
    assert up.arm.torque_on() and up.arm.released_at == [] and up.closes == []


# The end of the rest: the jaws closed, then the motors off, at measured rest


@apai
def test_at_measured_rest_the_jaws_close_and_then_the_motors_go_off(up: Rig) -> None:
    assert up.request().status == rest_module.AT_REST
    assert up.arm.torque_on()  # the return alone leaves the motors on
    assert up.rest.power_down(up.authority.stop_number) is None
    assert not up.arm.torque_on() and up.motion.get() is MotionState.REST_UNPOWERED
    assert up.arm.released_at == [pytest.approx(REST, abs=up.config.rest.rest_tol_rad)]
    # In order: the last fold step, rest by measurement, the close, the cut.
    kinds = [kind for kind, _ in up.events]
    close, cut = kinds.index("close"), kinds.index("release")
    at_rest = [text for _, text in up.events].index(
        "return to rest: the arm is at rest, motors on"
    )
    assert at_rest < close < cut and up.closes == ["close"]
    said = [text for kind, text in up.events if kind == rest_module.EVENT_JAWS]
    assert said == [rest_module.CLOSING_JAWS, rest_module.JAWS_CLOSED]
    assert up.events[-1] == ("release", rest_module.RELEASED)


def test_away_from_measured_rest_the_motors_stay_on_and_nothing_is_sent(
    sim_arm, checkers, config, clock, monkeypatch
) -> None:
    rig = Rig(sim_arm, checkers, config, clock, monkeypatch)
    rig.closes = []
    rig = with_route_or_plain(rig)
    rig.arm.set_pose(RAISED)
    before = rig.state()
    reason = rig.rest.power_down(rig.authority.stop_number)
    assert reason is not None and reason.startswith("the arm is not at rest")
    assert rig.state() == before and rig.closes == []
    assert rig.arm.torque_on() and rig.arm.released_at == []


def with_route_or_plain(rig: Rig) -> Rig:
    """A jaw close on the rig's return to rest, without APAI's route."""

    def close_jaws(should_stop: Any) -> str | None:
        rig.closes.append("close")
        return None

    rig.rest._close_jaws = close_jaws
    return rig


def test_a_jaw_close_that_fails_is_told_and_the_motors_still_go_off(
    sim_arm, checkers, config, clock, monkeypatch
) -> None:
    # APAI: at rest the jaws hold nothing up; a failed close is noted.
    rig = Rig(sim_arm, checkers, config, clock, monkeypatch)
    rig.rest._close_jaws = lambda should_stop: "the jaws are not referenced"
    assert rig.rest.power_down(rig.authority.stop_number) is None
    left = rest_module.JAWS_LEFT.format(why="the jaws are not referenced")
    assert (rest_module.EVENT_JAWS, left) in rig.events
    assert not rig.arm.torque_on()


def test_a_possible_load_refuses_the_rest_until_the_operator_says_empty(
    sim_arm, checkers, config, clock, monkeypatch
) -> None:
    rig = Rig(sim_arm, checkers, config, clock, monkeypatch)
    rig.arm.set_pose(RAISED)
    rig.close_on_object()
    before = rig.state()
    refused = rig.request()
    assert refused.status == rest_module.REFUSED
    assert refused.reason.startswith("the jaws are not proven empty")
    assert rig.state() == before  # nothing moved: the load is not carried in
    # Never a dead end: the operator's typed word is taken in place of the proof.
    rig.jaw_state.take_word()
    assert rig.request().status == rest_module.AT_REST


# Motors on again, without a new start of the robot app


def test_a_switched_arm_connects_a_new_port_after_a_release(clock) -> None:
    made: list[SimArm] = []

    def build() -> SimArm:
        made.append(SimArm(n_joints=6, clock=clock, q0=list(REST)))
        return made[-1]

    arm = SwitchedArm(build(), build)
    arm.connect()
    assert arm.torque_on() and len(made) == 1
    with pytest.raises(RuntimeError, match=rest_module.MOTORS_ARE_ON):
        arm.connect()  # never while the motors are on
    arm.release_torque()
    assert not arm.torque_on()
    arm.connect()  # as APAI's LazyArm: a new backend at each wake
    assert len(made) == 2 and arm.torque_on() and made[0].torque_on() is False
    assert arm.n_joints == 6 and arm.released_at == []  # the port in use answers


class Done:
    """Takes the final RESULT of a request typed at the Thor terminal."""

    def __init__(self) -> None:
        self.results: list[Result] = []
        self._came = threading.Event()

    def __call__(self, result: Result) -> None:
        self.results.append(result)
        self._came.set()

    def wait(self) -> Result:
        assert self._came.wait(WAIT_S), "no final RESULT"
        self._came.clear()
        return self.results[-1]


@pytest.fixture
def thor():
    """A robot app in real time on a simulated arm that can be switched on again."""
    clock, config = SystemClock(), RobotAppConfig.default()
    arms: list[SimArm] = []

    def build() -> SimArm:
        rest_q = list(config.rest.rest_q)
        arms.append(SimArm(n_joints=config.n_joints, clock=clock, q0=rest_q))
        return arms[-1]

    app = RobotApp(
        config,
        build(),
        program.sim_checkers(config, clock),
        None,
        clock,
        rebuild_arm=build,
    )
    app.arms = arms
    app.start()
    yield app
    app.close()
    for arm in arms:
        for released in arm.released_at:  # never away from rest
            assert released == pytest.approx(list(config.rest.rest_q), abs=0.05)


def test_the_motors_are_switched_on_again_without_a_new_start(thor: RobotApp) -> None:
    session, done = thor.session_id, Done()
    assert thor.request_release(done) is None
    assert done.wait().status == "completed"
    assert thor.motors_off_at_rest and not thor.arms[-1].torque_on()
    # The operator's question comes first.
    assert thor.request_power_on(False, done) == app_module.CONFIRM_POWER_ON
    assert done.wait().status == "refused" and not thor.arms[-1].torque_on()
    assert thor.request_power_on(True, done) is None
    result = done.wait()
    assert (result.status, result.detail["motors"]) == ("completed", "on")
    assert thor.motion_state is MotionState.REST_HOLD and thor.arms[-1].torque_on()
    assert thor.running and thor.session_id == session and len(thor.arms) == 2
    # With the motors on there is nothing to switch on.
    assert thor.request_power_on(True, done) == rest_module.MOTORS_ARE_ON
    assert done.wait().status == "refused"
    # And the arm can be released again, at measured rest.
    assert thor.request_release(done) is None
    assert done.wait().status == "completed" and thor.motors_off_at_rest


def test_a_reset_typed_at_the_thor_clears_a_fault_stop(thor: RobotApp) -> None:
    done = Done()
    assert thor.request_reset(done) is not None  # nothing is latched
    assert done.wait().status == "refused"
    thor._fault("a jaw fault of the test")
    assert thor.snapshot().stop_level is StopLevel.FAULT
    assert thor.request_reset(done) is None
    assert done.wait().status == "completed"
    assert thor.snapshot().stop_level is StopLevel.NONE
    assert thor.arms[-1].torque_on()  # it starts nothing and cuts nothing


# SIGTERM


def test_sigterm_at_measured_rest_ends_as_the_functional_launcher(
    thor: RobotApp,
) -> None:
    lines = program._sigterm_lines(thor)
    assert lines == [
        "robot app: SIGTERM; closing; the arm lies at measured rest: the motors are off"
    ]
    assert thor.motors_off_at_rest and not thor.arms[-1].torque_on()
    # Asked again, nothing is left to do.
    assert program._sigterm_lines(thor) == [
        "robot app: SIGTERM; closing; the motors are off at rest"
    ]


def test_sigterm_away_from_rest_holds_and_says_why(thor: RobotApp) -> None:
    raised = [0.2, -0.15, 0.1, 0.0, 0.05, -0.05]
    thor.arms[-1].set_pose(raised)
    first, second = program._sigterm_lines(thor)
    assert first == "robot app: SIGTERM; closing with the motors left on"
    assert second.startswith("robot app: the motors stay on because the arm is not at")
    assert thor.arms[-1].torque_on() and thor.arms[-1].released_at == []
    assert thor.arms[-1].pose() == pytest.approx(raised)  # nothing moved


# What the agent program asks of the Thor


def final(h: Any, command_id: str) -> Result:
    return h.client.wait_result(command_id, timeout_s=WAIT_S)


def test_the_rest_tools_motors_off_is_done_only_at_measured_rest(robot_app) -> None:
    h = robot_app
    h.new_task()  # the agent owns the arm, as inside its rest tool
    h.hold_at([0.2, -0.15, 0.1, 0.0, 0.05, -0.05])
    refused = final(h, h.client.request_owner(REQUEST_MOTORS_OFF))
    assert refused.status == "refused"
    assert refused.reason.startswith("the arm is not at rest")
    assert h.arm.torque_on() and h.arm.released_at == []


def test_the_rest_tools_motors_off_at_rest_sends_the_jaws_nothing(robot_app) -> None:
    h = robot_app
    h.new_task()
    done = final(h, h.client.request_owner(REQUEST_MOTORS_OFF))
    assert (done.status, done.detail["motors"]) == ("completed", "off")
    # APAI's tool has judged the load and closed the jaws itself.
    assert not h.arm.torque_on() and h.arm.jaw_commands == []
    assert h.app.motion_state is MotionState.REST_UNPOWERED


def test_the_agent_programs_park_is_the_typed_rest(robot_app) -> None:
    h = robot_app
    h.hold_at([0.2, -0.15, 0.1, 0.0, 0.05, -0.05])
    # Sent once the operator typed "path is clear" at the agent terminal.
    done = final(h, h.client.request_owner(REQUEST_PARK))
    assert done.status == "completed" and done.detail["outcome"] == "at_rest"
    assert h.arm.pose() == pytest.approx(REST, abs=h.config.rest.rest_tol_rad)
