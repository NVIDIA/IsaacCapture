# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanLogic: the buttons, the new-squeeze rule and the workstation's own ends.

The logic runs on stand-ins for the retargeting graph and the solver, pass by
pass in the order the human program uses, on ROBOT_STATEs built by hand. A pass
is one turn of the human program's loop (0.02 s). The clutch lets the arm
follow the hand only while the grip is squeezed; an engagement is one such
stretch. The Thor is the machine on the robot; its robot app sends ROBOT_STATE.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from isaaccapture_examples.agentic_remote.human import logic as logic_module
from isaaccapture_examples.agentic_remote.human.__main__ import as_sent
from isaaccapture_examples.agentic_remote.human.ik import nearest_step
from isaaccapture_examples.agentic_remote.human.logic import (
    BEHIND_NOTE_PASSES,
    COUNT_KEYS,
    FOLLOW_WAIT_S,
    HOLD_BACK_STEPS,
    JOINT_SPEED_SHARE,
    MAX_AHEAD_TURN_RAD,
    NO_FOLLOW_S,
    NOT_FOLLOWED,
    NOTE_AT_JOINT_LIMIT,
    NOTE_AT_LIMIT,
    NOTE_AT_REST,
    NOTE_AT_REST_JAWS,
    NOTE_CATCHING_UP,
    NOTE_CUT,
    NOTE_DRIVE_ON,
    NOTE_MAX_CHARS,
    NOTE_OUT_OF_REACH,
    NOTE_RELEASE_A,
    NOTE_RELEASE_B,
    NOTE_RELEASE_FIRST,
    NOTE_RETURNING,
    NOTE_SQUEEZE,
    NOTE_SQUEEZE_AGAIN,
    NOTE_STOPPING,
    NOTE_TAKE_OVER,
    NOTE_THOR_BEHIND,
    NOTE_TRIGGER_EASE,
    NOTE_TRIGGER_PULL,
    REACH_SHARES,
    STANDING_PERIODS,
    STEP_HALVINGS,
    TIP_BOX_MARGIN_M,
    TIP_FLOOR_MARGIN_M,
    TRIGGER_CLOSED_MIN,
    TRIGGER_OPEN_MAX,
    UNNAMED_JOINT_SPEED,
    WALL_NOTE_M,
    HumanLogic,
    jaw_opening,
)
from isaaccapture_examples.agentic_remote.human.types import (
    FeedStep,
    GraphStep,
    HumanSettings,
    IkSettings,
    IkStep,
    RetargetSettings,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    HUMAN_STAGES,
    GripperState,
    HumanSetpoint,
    JointSample,
    Message,
    OwnerRequest,
    Result,
    RobotState,
    Stop,
    Welcome,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
from isaaccapture_examples.agentic_remote.robot_app.gripper import (
    NO_OPENING,
    NO_TIGHTER,
    NOT_MATCHED,
)
from isaaccapture_examples.agentic_remote.robot_app.human import SPEED_MARGIN
from isaaccapture_examples.agentic_remote.robot_app.interfaces import JawLoad, Owner

PASS_S = 0.02
T0 = 100.0
N_JOINTS = 6
THRESHOLD = HumanSettings().retarget.squeeze_threshold
REST = (0.0,) * N_JOINTS
# The rest pose moved inside the stand-in solver's margin.
REST_INSIDE = (0.0, 0.025, 0.025, 0.0, 0.0, 0.0)
STOP_B = Stop(reason="right B", sender="controller")
AGAIN = "release the grip and squeeze again: "
# The buttons of a pass are known when the controller feed gave the graph's
# outputs with a sample age that is a finite number of 0 or more. A data gap is
# a pass without known buttons; these are the ways one comes about, as
# Bench.step takes them.
GAPS = {
    "stale SAMPLES": {"present": False, "stage": "stale", "age": None},
    "no outputs": {"outputs": False, "stage": "no_server", "age": None},
    "no sample age": {"age": None},
    "a sample age that is no number": {"age": float("nan")},
}
every_gap = pytest.mark.parametrize("gap", list(GAPS.values()), ids=list(GAPS))
# The controller is active and its buttons are read; its pose is not valid.
POSE_LOST = {"pose_valid": False, "stage": "no_controller"}


class FakeGraph:
    """A stand-in for the retargeting graph that keeps the clutch's rules.

    The clutch engages on a squeeze above the threshold while the pass is not
    stopped and the engage is permitted, and starts at the home. It lets go
    when the squeeze falls, the controller is absent, its pose is not valid or
    the pass is stopped.
    """

    def __init__(self) -> None:
        self.homes: list[np.ndarray] = []  # every home that was set
        self.handed: list[tuple[bool, bool]] = []  # (stopped, permitted) per pass
        self.steps: list[GraphStep] = []
        self.reads = 0
        self._home = np.eye(4)
        self._engaged = False
        self._origin = np.zeros(3)
        self._base = np.eye(4)
        self._held = np.eye(4)

    def set_home(self, base_T_ee: np.ndarray) -> None:
        assert not self._engaged, "the home was set while the clutch was engaged"
        self._home = np.array(base_T_ee, dtype=float)
        self._held = self._home.copy()
        self.homes.append(self._home.copy())

    def read(self, outputs: Any) -> GraphStep:
        self.reads += 1
        return outputs["step"]

    def run(
        self,
        *,
        stopped: bool,
        permitted: bool,
        hand: np.ndarray,
        squeeze: float,
        a: bool,
        b: bool,
        present: bool,
        pose_valid: bool = True,
        trigger: float = 0.0,
        lag_m: float = 0.0,
        limiter: str = "passed",
        clutch: bool | None = None,
        without: str | None = None,
    ) -> dict[str, GraphStep]:
        """One pass of the graph: the outputs a controller feed hands on.

        present says the controller is active; with pose_valid False its pose
        is not valid, and its buttons, squeeze and trigger are read all the same.
        lag_m puts the limited target that far below the clutch's; clutch
        forces the clutch's word, against its own rules; without names a
        field of the GraphStep that is None on this pass.
        """
        self.handed.append((stopped, permitted))
        tracked = present and pose_valid
        engage = not self._engaged and permitted
        if not tracked or stopped or not squeeze > THRESHOLD:
            self._engaged = False
        elif engage or (clutch and not self._engaged):
            self._engaged = True
            self._origin = np.array(hand, dtype=float)
            self._base = self._home.copy()
        if clutch is not None:
            self._engaged = clutch
        if self._engaged:
            self._held = self._base.copy()
            self._held[:3, 3] += np.asarray(hand, dtype=float) - self._origin
        limited = self._held.copy()
        limited[2, 3] -= lag_m
        step = GraphStep(
            present=tracked,
            squeeze=squeeze if present else 0.0,
            trigger=trigger if present else 0.0,
            a=present and a,
            b=present and b,
            engaged=self._engaged,
            ee_target=limited,
            commanded_target=self._held.copy(),
            limiter=limiter,
            hand=np.array(hand, dtype=float) if tracked else None,
        )
        if without is not None:
            step = dataclasses.replace(step, **{without: None})
        self.steps.append(step)
        return {"step": step}


NO_SOLUTION = "no IK solution (error 0.0063 after 50 iterations)"
NO_NEARER = "no nearer pose than the one of the last joint target"


class FakeSolver:
    """A stand-in for the solver: the tip's position is linear in three joints.

    The second and third joint have their lower stop at 0, as the arm at rest.
    The tip does not turn. A test may give the third joint an upper stop:
    a pose above what it reaches then has no solution.
    """

    n_joints = N_JOINTS
    LOW = (-2.0, 0.0, 0.0, -2.0, -2.0, -2.0)
    HIGH = (2.0,) * N_JOINTS
    MARGIN = 0.025
    BASE = np.array([0.2, 0.0, 0.1])
    M_PER_RAD = 0.2
    JUMP_RAD = 0.05  # the farthest a joint of a nearest pose lies from the seed

    def __init__(self) -> None:
        self.solves: list[tuple[np.ndarray, tuple[float, ...]]] = []
        # Set by a test: what solve() answers in place of the arm's own solution.
        self.answer: Callable[[np.ndarray, tuple[float, ...]], IkStep] | None = None
        # Set by a test: the third joint's upper stop, rad.
        self.top: float | None = None
        # Set by a test: a pose that is turned has no solution.
        self.turns = True
        self.nearests: list[tuple[np.ndarray, tuple[float, ...]]] = []
        # Set by a test: what nearest() answers in place of the arm's own.
        self.near: Callable[[np.ndarray, tuple[float, ...]], IkStep] | None = None

    def inside(self, q: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            min(max(float(value), low + self.MARGIN), high - self.MARGIN)
            for value, low, high in zip(q, self.LOW, self.HIGH)
        )

    def fk(self, q: Sequence[float]) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = self.BASE + self.M_PER_RAD * np.array(q[:3], dtype=float)
        return pose

    def solve(self, base_T_ee: np.ndarray, q_seed: Sequence[float]) -> IkStep:
        seed = tuple(q_seed)
        self.solves.append((np.array(base_T_ee), seed))
        if self.answer is not None:
            return self.answer(base_T_ee, seed)
        first = (base_T_ee[:3, 3] - self.BASE) / self.M_PER_RAD
        if self.top is not None and first[2] > self.top + 1e-12:
            return IkStep(None, NO_SOLUTION)
        if not self.turns and not np.allclose(base_T_ee[:3, :3], np.eye(3)):
            return IkStep(None, NO_SOLUTION)
        return IkStep((*(float(value) for value in first), *seed[3:]), "")

    def nearest(
        self,
        base_T_ee: np.ndarray,
        q_seed: Sequence[float],
        low: Sequence[float] | None = None,
        high: Sequence[float] | None = None,
    ) -> IkStep:
        """One step of human/ik.py's nearest pose on this arm's own FK."""
        seed = tuple(q_seed)
        self.nearests.append((np.array(base_T_ee), seed))
        if self.near is not None:
            return self.near(base_T_ee, seed)
        floor = np.array(self.LOW) + self.MARGIN
        ceiling = np.array(self.HIGH) - self.MARGIN
        if self.top is not None:
            ceiling[2] = self.top
        if low is not None:
            floor = np.maximum(floor, low)
        if high is not None:
            ceiling = np.minimum(ceiling, high)
        step = nearest_step(
            self.fk, np.array(base_T_ee), np.array(seed), floor, ceiling, self.JUMP_RAD
        )
        if step is None:
            return IkStep(None, NO_NEARER)
        return IkStep(tuple(float(value) for value in step), "")


# The human joint speed limit the stand-in robot app names in WELCOME, rad/s:
# limits.human.max_joint_speed of the shipped robot file.
HUMAN_SPEED = 1.5
# The farthest a joint of the joint target then moves in one pass, rad.
JOINT_STEP = JOINT_SPEED_SHARE * SPEED_MARGIN * HUMAN_SPEED * PASS_S


def welcome(
    session_id: str = "s1", human_max_joint_speed: float | None = HUMAN_SPEED
) -> Welcome:
    return Welcome(
        session_id=session_id,
        robot_id="sim-arm",
        profile_id="sim-6dof",
        joint_names=[f"j{index}" for index in range(1, N_JOINTS + 1)],
        joint_min=[-3.0] * N_JOINTS,
        joint_max=[3.0] * N_JOINTS,
        rest_q=[0.0] * N_JOINTS,
        setpoint_period_s=PASS_S,
        max_move_setpoints=3000,
        limit_set_versions={"human": "sim-1"},
        driver="sim",
        human_max_joint_speed=human_max_joint_speed,
    )


def robot_state(
    *,
    owner: str = "human",
    owner_number: int = 3,
    stop_number: int = 1,
    stop_level: str = "none",
    q: Sequence[float] = REST,
    valid: Sequence[bool] = (True,) * N_JOINTS,
    target_q: Sequence[float] | None = None,
    motion: str = "hold",
    clock_mono_s: float = 500.0,
    session_id: str = "s1",
    move_index: int = 0,
    **human: Any,
) -> RobotState:
    """A ROBOT_STATE as the robot app sends it; human goes into health["human"].

    move_index is how many setpoints the human drive has sent.
    """
    facts = {
        "enabled": True,
        "takeover_open": owner == "human",
        "driving": False,
        "needs_release": False,
        "ended": "",
        **human,
    }
    return RobotState(
        seq=1,
        session_id=session_id,
        clock_mono_s=clock_mono_s,
        clock_wall_s=1.7e9,
        joints=[
            JointSample(joint, value, 499.9, 499.95, ok)
            for joint, (value, ok) in enumerate(zip(q, valid))
        ],
        recent=[],
        target_q=None if target_q is None else list(target_q),
        target_t=None,
        gripper=GripperState(None, None, False, False, None, False, "unknown"),
        owner=owner,
        owner_number=owner_number,
        stop_number=stop_number,
        stop_level=stop_level,
        fault_reason="",
        limit_set_version="sim-1",
        motion=motion,
        move_id="",
        move_index=move_index,
        move_total=0,
        last_result=None,
        assistance_wait_s=None,
        torque_on=True,
        view_age_s=0.1,
        teleop_stage="ready",
        controller_age_s=0.01,
        health={"human": facts},
    )


def read_at(state: RobotState, began: float) -> RobotState:
    """state with every joint's reading taken anew, begun at this time."""
    joints = [
        dataclasses.replace(sample, t_start=began, t_end=began + 0.005)
        for sample in state.joints
    ]
    return dataclasses.replace(state, joints=joints)


def following(state: RobotState | None = None, sent: int = 1) -> RobotState:
    """state as the robot app gives it once its human drive has sent setpoints.

    The logic takes the arm as following an engagement from such a state on.
    """
    state = robot_state() if state is None else state
    human = {**state.health["human"], "driving": True}
    health = {**state.health, "human": human}
    return dataclasses.replace(state, move_index=sent, health=health)


def nobody(stop_number: int, owner_number: int = 5, **human: Any) -> RobotState:
    """The ROBOT_STATE after a STOP: nobody owns the arm."""
    return robot_state(
        owner="nobody",
        owner_number=owner_number,
        stop_number=stop_number,
        stop_level="stop",
        **human,
    )


def with_jaws(state: RobotState, load: str) -> RobotState:
    """state with this jaw load; the jaws are the gripper's fingers."""
    gripper = dataclasses.replace(state.gripper, load=load)
    return dataclasses.replace(state, gripper=gripper)


def of(messages: Sequence[Message], kind: Any) -> list[Any]:
    return [message for message in messages if isinstance(message, kind)]


def setpoint(messages: Sequence[Message]) -> HumanSetpoint:
    """The one HUMAN_SETPOINT among the link messages of a pass."""
    [found] = of(messages, HumanSetpoint)
    return found


def notes(messages: Sequence[Message]) -> set[str]:
    """The notes of the HUMAN_SETPOINTs among these link messages."""
    return {sent.note for sent in of(messages, HumanSetpoint)}


def none_engaged(messages: Sequence[Message]) -> bool:
    """Whether HUMAN_SETPOINTs went out and none of them carries a joint target."""
    sent = of(messages, HumanSetpoint)
    return bool(sent) and all(not m.engaged and m.q is None for m in sent)


def all_engaged(messages: Sequence[Message]) -> bool:
    """Whether HUMAN_SETPOINTs went out and each of them carries a joint target."""
    sent = of(messages, HumanSetpoint)
    return bool(sent) and all(m.engaged and m.q is not None for m in sent)


class Bench:
    """A HumanLogic on stand-ins, run pass by pass as the human program runs it."""

    def __init__(self, source: str = HUMAN_SOURCE_SYNTHETIC, **settings: Any) -> None:
        self.settings = dataclasses.replace(HumanSettings(), **settings)
        self.graph = FakeGraph()
        self.solver = FakeSolver()
        self.logic = HumanLogic(self.graph, self.solver, self.settings, source)
        self.welcome: Welcome | None = welcome()
        self.state: RobotState | None = robot_state()
        self.state_age_s: float | None = 0.01
        self.hand = np.array([0.0, 1.2, -0.3])
        self.passes = 0
        # Whether each pass's ROBOT_STATE carries new readings of every
        # joint, as from a robot app that reads them between two passes.
        self.reads = True

    def seen(self) -> RobotState | None:
        """The ROBOT_STATE this pass's logic gets: self.state, as read just now."""
        if self.state is None or not self.reads:
            return self.state
        return read_at(self.state, 400.0 + self.passes * PASS_S)

    def step(
        self,
        *,
        squeeze: float = 0.0,
        a: bool = False,
        b: bool = False,
        present: bool = True,
        stage: str = "ready",
        age: float | None = 0.01,
        note: str = "",
        outputs: bool = True,
        results: Sequence[Result] = (),
        **graph: Any,
    ) -> list[Message]:
        """One pass: the graph runs on what armed says, then the logic steps."""
        self.passes += 1
        armed = self.logic.armed
        produced = None
        if outputs:
            produced = self.graph.run(
                stopped=not armed,
                permitted=armed,
                hand=self.hand,
                squeeze=squeeze,
                a=a,
                b=b,
                present=present,
                **graph,
            )
        return self.logic.step(
            T0 + self.passes * PASS_S,
            FeedStep(produced, age, stage, note),
            self.welcome,
            self.seen(),
            self.state_age_s,
            results,
        )

    def run(self, passes: int, **controls: Any) -> list[Message]:
        """The link messages of this many equal passes."""
        out: list[Message] = []
        for _ in range(passes):
            out += self.step(**controls)
        return out

    def lost(self, stage: str = "stale", age: float | None = None) -> list[Message]:
        """One pass without usable controller data."""
        return self.step(present=False, stage=stage, age=age)

    def stall(self, seconds: float) -> None:
        """Let this long go by without a pass, as when the program's loop stands.

        The next pass then comes this long and one pass period after the last.
        """
        self.passes += round(seconds / PASS_S)

    def status(self, **controls: Any) -> HumanSetpoint:
        """The next HUMAN_SETPOINT; at the status rate it takes up to three passes."""
        for _ in range(3):
            sent = of(self.step(**controls), HumanSetpoint)
            if sent:
                return sent[0]
        raise AssertionError("no HUMAN_SETPOINT within three passes")

    def engage(self, follow: bool = True) -> HumanSetpoint:
        """Release the grip, then squeeze it: the HUMAN_SETPOINT of the squeeze.

        With follow the ROBOT_STATE of the passes after it shows the Thor's
        drive following: the joint target leaves the squeeze's first one.
        After an engagement the release takes up to three passes: the squeeze
        counts once a ROBOT_STATE has shown no drive and two have shown the
        same pose in new readings.
        """
        self.halted()
        self.step()
        for _ in range(2):
            if not self.logic.armed:
                self.step()
        sent = setpoint(self.step(squeeze=1.0))
        assert sent.engaged, sent.note
        if follow:
            self.state = following(self.state)
        return sent

    def halted(self) -> None:
        """The ROBOT_STATE from here on shows no human drive, as after one ended.

        A new squeeze counts only on such a state, made after the pass that
        ended the engagement before it.
        """
        human = {**self.state.health["human"], "driving": False}
        health = {**self.state.health, "human": human}
        self.state = dataclasses.replace(self.state, move_index=0, health=health)

    def drive(self, passes: int, **controls: Any) -> list[HumanSetpoint]:
        """The HUMAN_SETPOINTs of this many squeezed passes under a Thor that follows.

        Each ROBOT_STATE shows the joint target of the pass before as the
        Thor's last joint target and as the arm's pose.
        """
        out: list[HumanSetpoint] = []
        for _ in range(passes):
            sent = setpoint(self.step(squeeze=1.0, **controls))
            out.append(sent)
            if sent.q is not None:
                at = robot_state(q=sent.q, target_q=sent.q, seq=sent.seq)
                self.state = following(at, sent=len(out))
        return out


# Buttons


def test_a_press_of_b_is_a_stop_at_once():
    bench = Bench()
    bench.state = robot_state(owner="agent")
    assert of(bench.step(), Stop) == []
    pressed = bench.step(b=True)
    assert pressed[0] == STOP_B
    assert len(of(pressed, Stop)) == 1 and of(pressed, OwnerRequest) == []
    assert of(bench.run(20, b=True), Stop) == []
    assert bench.logic.counts()["b_pressed"] == 1

    # A STOP quotes nothing: it goes out before any WELCOME or ROBOT_STATE.
    early = Bench()
    early.welcome, early.state, early.state_age_s = None, None, None
    assert early.step() == []
    assert early.step(b=True) == [STOP_B]


def test_b_held_one_second_with_the_grip_released_asks_for_the_takeover_once():
    bench = Bench()
    bench.state = robot_state(owner="agent", owner_number=4, stop_number=7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    assert of(bench.run(49, b=True), OwnerRequest) == []  # 0.98 s after the press
    due = bench.step(b=True)
    [request] = of(due, OwnerRequest)
    expected = OwnerRequest(
        request.command_id, "s1", 5, 8, "human", "right B held 1.0 s"
    )
    assert request == expected and request.command_id
    assert of(due, Stop) == [] and isinstance(due[-1], HumanSetpoint)
    assert of(bench.run(100, b=True), OwnerRequest) == []
    assert bench.logic.counts()["takeover_asked"] == 1


def test_a_pass_sends_the_stop_then_one_owner_request_then_the_setpoint():
    def kinds(messages: Sequence[Message]) -> list[type]:
        return [type(message) for message in messages]

    bench = Bench()
    bench.state = nobody(7, takeover_open=True)
    bench.step()
    assert kinds(bench.step(a=True, b=True)) == [Stop, HumanSetpoint]
    bench.state = nobody(8, takeover_open=True)
    bench.run(49, a=True, b=True)
    # Both holds are over on this pass: one request now, the other on the next.
    due = bench.step(a=True, b=True)
    assert kinds(due) == [OwnerRequest, HumanSetpoint]
    after = bench.step(a=True, b=True)
    assert kinds(after) == [OwnerRequest, HumanSetpoint]
    assert (due[0].request, after[0].request) == ("human", "handback")
    assert due[0].command_id != after[0].command_id

    # A press of B on the pass on which A's hold is over: the STOP goes alone,
    # and the hand-back quotes the ROBOT_STATE that shows it.
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    bench.run(50, a=True)
    assert kinds(bench.step(a=True, b=True)) == [Stop, HumanSetpoint]
    assert kinds(bench.step(a=True, b=True)) == [HumanSetpoint]
    bench.state = nobody(2, takeover_open=True, motion="rest_hold")
    shown = bench.step(a=True, b=True)
    assert kinds(shown) == [OwnerRequest, HumanSetpoint]
    assert (shown[0].request, shown[0].stop_number) == ("handback", 2)


def test_the_takeover_waits_for_a_fresh_robot_state_that_shows_the_stop():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    waiting = bench.run(75, b=True)  # 1.5 s, still number 7
    assert of(waiting, OwnerRequest) == []
    # The hold can still ask, once its STOP shows: the note stays the usual one.
    assert notes(waiting) == {NOTE_TAKE_OVER}
    bench.state, bench.state_age_s = nobody(8, owner_number=6), 0.5
    assert of(bench.run(5, b=True), OwnerRequest) == []  # shown, but too old
    bench.state_age_s = 0.01
    [request] = of(bench.step(b=True), OwnerRequest)
    assert (request.owner_number, request.stop_number) == (6, 8)

    # A ROBOT_STATE of another session cannot show this press's STOP.
    other = Bench()
    other.state = nobody(7)
    other.step()
    other.step(b=True)
    other.welcome, other.state = welcome("s2"), nobody(9, session_id="s2")
    assert of(other.run(100, b=True), OwnerRequest) == []


def test_b_held_with_the_grip_squeezed_asks_for_nothing():
    bench = Bench()
    bench.state = nobody(7)
    bench.step(squeeze=1.0)
    assert bench.step(b=True, squeeze=1.0)[0] == STOP_B
    bench.state = nobody(8)
    assert of(bench.run(150, b=True, squeeze=1.0), OwnerRequest) == []

    # One squeezed pass during the hold is enough.
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(9)
    bench.run(20, b=True)
    bench.step(b=True, squeeze=1.0)
    assert of(bench.run(150, b=True), OwnerRequest) == []

    # So is a squeeze on the pass of the press alone.
    bench.step()
    assert bench.step(b=True, squeeze=1.0)[0] == STOP_B
    bench.state = nobody(10)
    assert of(bench.run(150, b=True), OwnerRequest) == []

    # The same hold with the grip released does ask.
    bench.step()
    bench.step(b=True)
    bench.state = nobody(11)
    assert len(of(bench.run(50, b=True), OwnerRequest)) == 1


def test_a_button_down_before_a_data_loss_counts_only_when_pressed_again():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    bench.run(10, b=True)
    bench.lost()
    still_down = bench.run(100, b=True)
    assert of(still_down, (Stop, OwnerRequest)) == []
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(9)
    assert len(of(bench.run(50, b=True), OwnerRequest)) == 1

    # A button that is down when the first data arrive was never seen up: it
    # asks for nothing. Right B is one STOP all the same.
    start = Bench()
    start.state = nobody(7, takeover_open=True)
    assert start.step(a=True, b=True)[0] == STOP_B
    start.state = nobody(8, takeover_open=True)  # the STOP shows
    assert of(start.run(100, a=True, b=True), (Stop, OwnerRequest)) == []
    counts = start.logic.counts()
    assert (counts["a_pressed"], counts["b_pressed"]) == (0, 0)


def test_right_b_down_as_the_first_data_arrive_sends_one_stop():
    bench = Bench()
    bench.state = robot_state(owner="agent")
    waiting = {"outputs": False, "stage": "waiting_for_headset", "age": None}
    assert of(bench.run(20, **waiting), Stop) == []
    began = bench.run(100, b=True)
    assert began[0] == STOP_B and len(of(began, Stop)) == 1

    # On the program's first pass, before any WELCOME: a STOP quotes nothing.
    early = Bench()
    early.welcome, early.state, early.state_age_s = None, None, None
    assert early.step(b=True) == [STOP_B]
    assert early.run(20, b=True) == []


@every_gap
def test_right_b_pressed_during_a_data_gap_sends_one_stop_when_data_return(gap):
    # The arm follows the hand; one pass has no data and B goes down on it.
    bench = Bench()
    bench.engage()
    bench.run(3, squeeze=1.0)
    assert of(bench.step(squeeze=1.0, b=True, **gap), Stop) == []
    back = bench.run(100, squeeze=1.0, b=True)
    assert back[0] == STOP_B and len(of(back, Stop)) == 1
    assert of(back, OwnerRequest) == [] and none_engaged(back)

    # A long gap is one stretch as well.
    bench = Bench()
    bench.state = robot_state(owner="agent")
    bench.step()
    assert of(bench.run(150, b=True, **gap), Stop) == []
    back = bench.run(20, b=True)
    assert back[0] == STOP_B and len(of(back, Stop)) == 1


@every_gap
def test_right_b_held_through_a_gap_sends_no_second_stop(gap):
    # The gaps here last 0.5 s; a stretch is kept over a gap of up to 1 s.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    bench.run(10, b=True)
    held = bench.run(25, b=True, **gap) + bench.run(100, b=True)
    assert of(held, (Stop, OwnerRequest)) == []

    # The same after a STOP for a B that went down in a gap.
    bench.step()
    bench.run(5, b=True, **gap)
    assert bench.step(b=True)[0] == STOP_B
    held = bench.run(25, b=True, **gap) + bench.run(100, b=True)
    assert of(held, (Stop, OwnerRequest)) == []

    # Seen up, B is a new stretch.
    bench.step()
    assert bench.step(b=True)[0] == STOP_B


@every_gap
def test_right_b_down_after_a_gap_of_more_than_a_second_is_a_stop_again(gap):
    def held_through(gap_passes: int) -> tuple[Bench, list[Message]]:
        """B is a STOP and stays down; a gap; an agent task; the data return."""
        bench = Bench()
        bench.state = robot_state(owner="agent")
        bench.step()
        assert bench.step(b=True)[0] == STOP_B
        bench.state = nobody(2)
        bench.run(5, b=True)
        assert of(bench.run(gap_passes, b=True, **gap), Stop) == []
        # In the gap B may have been released and pressed again, unseen.
        bench.state = robot_state(owner="agent", owner_number=7, stop_number=2)
        return bench, bench.run(10, b=True)

    # Two passes with known buttons 1.0 s apart: still the same stretch.
    bench, back = held_through(49)
    assert of(back + bench.run(100, b=True), (Stop, OwnerRequest)) == []

    # Further apart, B that is down is a STOP at once, and one STOP.
    for gap_passes in (50, 250):
        bench, back = held_through(gap_passes)
        assert back[0] == STOP_B and len(of(back, Stop)) == 1
        # It is no press that counts: held on, B asks for no takeover.
        bench.state = nobody(3, owner_number=8)  # the STOP shows
        assert of(bench.run(150, b=True), (Stop, OwnerRequest)) == []
        assert bench.logic.counts()["b_pressed"] == 1

    # The cost: B truly held through such a gap takes the arm from the person,
    # and nothing arms until a ROBOT_STATE shows that STOP.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = robot_state(owner_number=6, stop_number=8)  # the grant
    bench.run(5, b=True)
    assert bench.logic.armed
    bench.run(50, b=True, **gap)
    again = bench.step(b=True)
    assert again[0] == STOP_B and not bench.logic.armed
    assert setpoint(again).note == "STOP by right B"
    assert of(bench.run(100, b=True), (Stop, OwnerRequest)) == []
    assert not bench.logic.armed
    bench.state = nobody(9, takeover_open=True)
    assert none_engaged(bench.run(5, b=True))


def test_right_b_is_a_stop_while_the_controllers_pose_is_not_valid():
    bench = Bench()
    bench.state = robot_state(owner="agent")
    bench.step()
    assert of(bench.run(5, **POSE_LOST), Stop) == []
    pressed = bench.step(b=True, **POSE_LOST)
    assert pressed[0] == STOP_B and len(of(pressed, Stop)) == 1
    assert (bench.logic.last.present, bench.logic.last.b) == (False, True)
    # One stretch, also when the pose comes back under the held button.
    held = bench.run(50, b=True, **POSE_LOST) + bench.run(50, b=True)
    assert of(held, Stop) == []

    # B is seen up and goes down again, all without a valid pose.
    bench.step(**POSE_LOST)
    assert bench.step(b=True, **POSE_LOST)[0] == STOP_B

    # Without a sample age the buttons are not known, with one they are.
    bench.step(**POSE_LOST)
    assert of(bench.run(5, b=True, **POSE_LOST, age=None), Stop) == []
    assert bench.step(b=True, **POSE_LOST)[0] == STOP_B

    # The pose goes and B goes down on the same pass, while the arm follows.
    bench = Bench()
    bench.engage()
    ended = bench.step(squeeze=1.0, b=True, **POSE_LOST)
    assert ended[0] == STOP_B and none_engaged(ended)
    assert not bench.logic.armed

    # The stage word does not decide: outputs with a sample age are enough.
    for stage in (*HUMAN_STAGES, "", "a word no feed sends"):
        for pose_valid in (True, False):
            odd = Bench()
            odd.state = robot_state(owner="agent")
            odd.step()
            pressed = odd.step(b=True, stage=stage, pose_valid=pose_valid)
            assert pressed[0] == STOP_B, (stage, pose_valid)


def test_one_pass_with_a_pose_that_is_not_valid_ends_the_hold_of_a_button():
    # The hold runs on fresh passes. B, still down, is the same stretch.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    sent = bench.run(30, b=True) + bench.step(b=True, **POSE_LOST)
    sent += bench.run(100, b=True)
    assert of(sent, (Stop, OwnerRequest)) == []
    assert bench.logic.counts()["takeover_asked"] == 0

    # The same hold without that pass does ask.
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(9)
    assert len(of(bench.run(50, b=True), OwnerRequest)) == 1

    # Right A and the hand-back keep the same rule.
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    sent = bench.run(30, a=True) + bench.step(a=True, **POSE_LOST)
    sent += bench.run(100, a=True)
    assert of(sent, OwnerRequest) == []
    bench.step()
    assert len(of(bench.run(51, a=True), OwnerRequest)) == 1


def test_a_stop_without_a_press_that_counts_asks_for_no_takeover():
    # B went down in a data gap and is held for 3 s with the grip released.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.lost()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)  # the STOP shows
    held = bench.run(150, b=True)
    assert of(held, (Stop, OwnerRequest)) == []
    # The note says so: B must be released first.
    assert {sent.note for sent in of(held, HumanSetpoint)} == {NOTE_RELEASE_B}

    # B went down while the pose was not valid; the pose comes back under it.
    bench.step()
    bench.step(**POSE_LOST)
    assert bench.step(b=True, **POSE_LOST)[0] == STOP_B
    bench.state = nobody(9)
    held = bench.run(150, b=True)
    assert of(held, (Stop, OwnerRequest)) == []
    assert {sent.note for sent in of(held, HumanSetpoint)} == {NOTE_RELEASE_B}
    counts = bench.logic.counts()
    assert (counts["b_pressed"], counts["takeover_asked"]) == (0, 0)

    # Seen up on a fresh pass and pressed again, the hold asks.
    assert bench.status().note == NOTE_TAKE_OVER
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(10)
    held = bench.run(50, b=True)
    [request] = of(held, OwnerRequest)
    assert (request.request, request.stop_number) == ("human", 10)
    assert {sent.note for sent in of(held, HumanSetpoint)} == {NOTE_TAKE_OVER}


def test_a_stop_without_a_press_that_counts_closes_the_gates_until_it_shows():
    def stopped() -> Bench:
        """The person owns the arm; B went down in a data gap: one STOP."""
        bench = Bench()
        bench.step()
        assert bench.logic.armed
        bench.lost()
        assert bench.step(b=True)[0] == STOP_B
        return bench

    # While B stays down nothing arms until a ROBOT_STATE shows the STOP.
    bench = stopped()
    held = bench.run(100, b=True)  # 2 s
    assert none_engaged(held) and not bench.logic.armed
    assert setpoint(held[-1:]).note == "STOP by right B"
    bench.state = robot_state(owner_number=4, stop_number=2)
    bench.step(b=True)
    assert bench.logic.armed

    # With B up again the wait ends 1 s after the STOP.
    bench = stopped()
    waited = bench.run(49)  # until 0.98 s after the STOP
    assert none_engaged(waited) and not bench.logic.armed
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged

    # The hand-back waits the same way, so that it quotes the STOP's numbers.
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    bench.lost()
    assert bench.step(b=True)[0] == STOP_B
    assert of(bench.run(60, a=True, b=True), OwnerRequest) == []  # A's hold is over
    bench.state = nobody(2, takeover_open=True, motion="rest_hold")
    [request] = of(bench.step(a=True, b=True), OwnerRequest)
    assert (request.request, request.stop_number) == ("handback", 2)


def test_a_button_held_on_after_its_request_asks_for_nothing_more():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    [request] = of(bench.run(50, b=True), OwnerRequest)
    granted = Result(request.command_id, "completed")
    bench.state = robot_state(owner="human", owner_number=6, stop_number=8)
    held = bench.step(b=True, results=[granted]) + bench.run(150, b=True)
    assert of(held, (Stop, OwnerRequest)) == []
    bench.step()
    assert bench.step(b=True)[0] == STOP_B


def test_the_result_of_an_overtaken_request_is_not_taken():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    [first] = of(bench.run(50, b=True), OwnerRequest)
    bench.step()
    bench.step(b=True)  # pressed again before the first RESULT came
    bench.state = nobody(9)
    [second] = of(bench.run(50, b=True), OwnerRequest)
    assert second.command_id != first.command_id

    late = [Result(first.command_id, "refused", "stale stop number")]
    assert setpoint(bench.step(b=True, results=late)).note == NOTE_TAKE_OVER
    # The second request is still open: a HUMAN_SETPOINT on every pass.
    assert len(of(bench.run(50, b=True), HumanSetpoint)) == 50
    bench.step(b=True, results=[Result(second.command_id, "refused", "busy")])
    assert bench.status(b=True).note == "takeover refused: busy"


def test_a_held_at_rest_asks_for_the_handback():
    bench = Bench()
    bench.state = robot_state(owner_number=6, stop_number=8, motion="rest_hold")
    bench.step()
    assert of(bench.run(50, a=True), OwnerRequest) == []  # 0.98 s after the press
    due = bench.step(a=True)
    [request] = of(due, OwnerRequest)
    expected = OwnerRequest(
        request.command_id, "s1", 6, 8, "handback", "right A held 1.0 s"
    )
    assert request == expected and request.command_id
    after = bench.run(150, a=True)
    assert of(due + after, Stop) == [] and of(after, OwnerRequest) == []
    assert bench.logic.counts()["handback_asked"] == 1

    # After a STOP nobody owns the arm and the takeover is still open.
    stopped = Bench()
    stopped.state = nobody(8, takeover_open=True, motion="rest_hold")
    stopped.step()
    assert len(of(stopped.run(51, a=True), OwnerRequest)) == 1

    # With the grip squeezed, or with no takeover open, A asks for nothing.
    squeezed = Bench()
    squeezed.state = nobody(8, takeover_open=True, motion="rest_hold")
    squeezed.step(squeeze=1.0)
    assert of(squeezed.run(150, a=True, squeeze=1.0), OwnerRequest) == []
    closed = Bench()
    closed.state = nobody(8, motion="rest_hold")
    closed.step()
    assert of(closed.run(150, a=True), (Stop, OwnerRequest)) == []


# A stall is a stretch without a pass: the human program's loop stood, as
# behind a log call that blocks. A button that is down before and after it may
# have been released and pressed again meanwhile, unseen.


def test_a_hold_of_b_does_not_count_across_a_stall_of_more_than_a_second():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B  # a press that counts
    bench.state = nobody(8)  # the Thor shows the STOP
    assert notes(bench.run(10, b=True)) == {NOTE_TAKE_OVER}  # 0.2 s of the hold
    bench.stall(2.0)
    back = bench.step(b=True)
    assert back[0] == STOP_B  # more than 1 s unseen: a new stretch of right B
    bench.state = nobody(9)  # the Thor shows that STOP too
    # The 2 s count for nothing, and the press that was on counts no more:
    # held on for 2 s, B asks for no takeover. The note says what to do.
    held = back + bench.run(100, b=True)
    assert of(held, OwnerRequest) == [] and of(held, Stop) == [STOP_B]
    assert notes(held) == {NOTE_RELEASE_B}
    counts = bench.logic.counts()
    assert (counts["b_pressed"], counts["takeover_asked"]) == (1, 0)

    # Seen up and then held for 1 s, B asks once.
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(10)
    assert of(bench.run(49, b=True), OwnerRequest) == []  # 0.98 s after the press
    [request] = of(bench.run(100, b=True), OwnerRequest)
    assert (request.request, request.stop_number) == ("human", 10)
    assert bench.logic.counts()["takeover_asked"] == 1


def test_a_hold_of_a_does_not_count_across_a_stall_of_more_than_a_second():
    # After a STOP at rest, with the takeover still open.
    bench = Bench()
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    bench.step()
    assert notes(bench.run(10, a=True)) == {NOTE_AT_REST}  # 0.2 s of the hold
    assert bench.logic.counts()["a_pressed"] == 1
    bench.stall(2.0)
    # The 2 s count for nothing, and the press that was on counts no more:
    # held on for 2 s, A asks for no hand-back. The note says what to do.
    held = bench.run(100, a=True)
    assert of(held, (Stop, OwnerRequest)) == []
    assert notes(held) == {NOTE_RELEASE_A}
    counts = bench.logic.counts()
    assert (counts["a_pressed"], counts["handback_asked"]) == (1, 0)

    # Seen up and then held for 1 s, A asks once.
    bench.step()
    assert of(bench.run(50, a=True), OwnerRequest) == []  # 0.98 s after the press
    [request] = of(bench.run(100, a=True), OwnerRequest)
    assert request.request == "handback"
    assert bench.logic.counts()["handback_asked"] == 1

    # The same while the person owns the arm.
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    bench.run(10, a=True)
    bench.stall(2.0)
    assert of(bench.run(100, a=True), OwnerRequest) == []
    bench.step()
    assert len(of(bench.run(51, a=True), OwnerRequest)) == 1


def test_a_hold_runs_on_through_a_stall_of_a_second_or_less():
    # The threshold is _STRETCH_KEEP_S, 1 s, as for a stretch of right B: two
    # passes with known buttons that far apart or less are one hold. So a
    # hold can be over with the button seen down on a few passes only.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    bench.run(10, b=True)  # 0.2 s after the press
    bench.stall(0.5)
    # The next pass comes 0.72 s after the press; the hold is over at 1 s.
    assert of(bench.run(14, b=True), (Stop, OwnerRequest)) == []
    due = bench.step(b=True)
    assert len(of(due, OwnerRequest)) == 1 and of(due, Stop) == []

    bench = Bench()
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    bench.step()
    bench.run(11, a=True)  # 0.2 s after the press
    bench.stall(0.5)
    assert of(bench.run(14, a=True), OwnerRequest) == []
    assert len(of(bench.step(a=True), OwnerRequest)) == 1

    # Two passes 1.0 s apart: the hold is over on the second one.
    bench = Bench()
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    bench.step()
    assert of(bench.step(a=True), OwnerRequest) == []
    bench.stall(1.0 - PASS_S)
    assert len(of(bench.step(a=True), OwnerRequest)) == 1


def test_a_button_seen_up_on_the_first_pass_after_a_stall_counts_as_seen_up():
    # That pass is fresh: a press on the pass after it counts.
    bench = Bench()
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    bench.step()
    bench.run(10, a=True)
    bench.stall(2.0)
    bench.step()  # A is up, 2 s after the pass before
    assert of(bench.run(50, a=True), OwnerRequest) == []  # 0.98 s after the press
    assert len(of(bench.run(100, a=True), OwnerRequest)) == 1
    assert bench.logic.counts()["a_pressed"] == 2

    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    bench.run(10, b=True)
    bench.stall(2.0)
    assert of(bench.step(), Stop) == []  # B is up, 2 s after the pass before
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(9)
    assert of(bench.run(49, b=True), OwnerRequest) == []
    assert len(of(bench.run(100, b=True), OwnerRequest)) == 1
    assert bench.logic.counts()["b_pressed"] == 2


# Gates, the home


@pytest.mark.parametrize(
    "state",
    [
        nobody(1),
        robot_state(owner="agent"),
        robot_state(owner="recovery"),
        robot_state(owner="human", stop_level="stop"),
    ],
    ids=["nobody", "agent", "recovery", "human under a stop"],
)
def test_no_engage_without_ownership(state):
    bench = Bench()
    bench.state = state
    bench.step()
    assert not bench.logic.armed
    # Even a clutch that says engaged gets no joint target.
    assert none_engaged(bench.run(10, squeeze=1.0, clutch=True))
    assert bench.solver.solves == []

    owned = Bench()
    owned.step()
    assert owned.logic.armed and setpoint(owned.step(squeeze=1.0)).engaged

    # Armed under the person: without the ownership that is gone. The passes
    # after the release see a ROBOT_STATE that arrived after it, and then one
    # whose new readings show the same pose: the arm is at rest.
    owned.run(3)
    assert owned.logic.armed
    owned.state = state
    owned.step()
    assert not owned.logic.armed
    assert none_engaged(owned.run(10, squeeze=1.0, clutch=True))


def test_no_engage_on_an_old_robot_state():
    bench = Bench()
    bench.state_age_s = bench.settings.state_max_age_s  # not younger than the limit
    bench.step()
    assert not bench.logic.armed
    assert none_engaged(bench.run(10, squeeze=1.0, clutch=True))
    assert bench.solver.solves == [] and bench.graph.homes == []

    bench.state_age_s = None
    assert none_engaged(bench.run(3))
    assert not bench.logic.armed

    bench.state_age_s = 0.19
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged


def test_no_engage_on_a_joint_without_a_valid_reading():
    bench = Bench()
    bench.state = robot_state(valid=(True, True, False, True, True, True))
    bench.step()
    assert not bench.logic.armed
    assert none_engaged(bench.run(10, squeeze=1.0, clutch=True))
    assert bench.solver.solves == [] and bench.graph.homes == []

    # A joint that is missing from ROBOT_STATE counts the same.
    whole = robot_state()
    bench.state = dataclasses.replace(whole, joints=whole.joints[:-1])
    bench.step()
    assert not bench.logic.armed and bench.graph.homes == []

    # So do a joint that is named twice and a reading that is no number.
    twice = [*whole.joints[:-1], whole.joints[0]]
    bench.state = dataclasses.replace(whole, joints=twice)
    bench.step()
    assert not bench.logic.armed and bench.graph.homes == []
    no_number = dataclasses.replace(whole.joints[0], value=float("nan"))
    bench.state = dataclasses.replace(whole, joints=[no_number, *whole.joints[1:]])
    bench.step()
    assert not bench.logic.armed and bench.graph.homes == []

    bench.state = whole
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged


@pytest.mark.parametrize(
    "health",
    [{}, {"human": None}, {"human": "on"}, {"human": ["takeover_open"]}],
    ids=["no entry", "none", "a word", "a list"],
)
def test_a_robot_state_without_the_human_facts_reads_them_as_false(health):
    bench = Bench()
    bench.state = dataclasses.replace(robot_state(), health=health)
    assert setpoint(bench.step()).grip_released
    assert bench.logic.armed  # no released grip is asked for
    assert setpoint(bench.step(squeeze=1.0)).engaged
    # No takeover is open: right A asks for nothing.
    bench.step()
    assert of(bench.run(60, a=True), OwnerRequest) == []


def test_the_home_is_the_measured_pose_moved_inside_the_ik_margin():
    measured = (0.3, 0.0, 0.01, 0.1, -0.2, 0.4)
    inside = (0.3, 0.025, 0.025, 0.1, -0.2, 0.4)
    bench = Bench()
    bench.state = robot_state(q=measured)
    bench.step()
    [home] = bench.graph.homes
    assert np.allclose(home, bench.solver.fk(inside))
    assert not np.allclose(home, bench.solver.fk(measured))

    # Every pass with the clutch not engaged sets it again, in joint order.
    moved = robot_state(q=(0.5, 0.2, 0.3, 0.0, 0.0, 0.0))
    bench.state = dataclasses.replace(moved, joints=moved.joints[::-1])
    bench.step()
    assert len(bench.graph.homes) == 2
    assert np.allclose(bench.graph.homes[-1], bench.solver.fk((0.5, 0.2, 0.3)))

    # While the clutch is engaged the home stays.
    assert setpoint(bench.step(squeeze=1.0)).engaged
    bench.run(5, squeeze=1.0)
    assert len(bench.graph.homes) == 2
    bench.step()
    assert len(bench.graph.homes) == 3


def test_the_first_joint_target_and_the_first_seed_of_an_engagement_are_the_inside_pose():
    bench = Bench()  # at rest: two joints lie on their stops
    first = bench.engage()
    # The squeeze's joint target is the home: nothing is solved for it.
    assert bench.solver.solves == []
    assert first.q == pytest.approx(list(REST_INSIDE))

    # From then on the seed is the last joint target sent.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.002])
    second = setpoint(bench.step(squeeze=1.0))
    target, seed = bench.solver.solves[0]
    assert seed == tuple(first.q) == REST_INSIDE
    assert np.allclose(
        target[:3, 3], bench.solver.fk(REST_INSIDE)[:3, 3] + [0, 0, 0.002]
    )
    assert second.q == pytest.approx([0.0, 0.025, 0.035, 0.0, 0.0, 0.0])
    setpoint(bench.step(squeeze=1.0))
    assert bench.solver.solves[1][1] == tuple(second.q)

    # The next engagement starts at the measured pose again, not at the last
    # joint target of the one before.
    moved = (0.5, 0.2, 0.3, 0.1, 0.0, 0.0)
    bench.step()
    bench.state = robot_state(q=moved)
    bench.run(2)  # two readings of each joint that agree: the arm is at rest
    again = setpoint(bench.step(squeeze=1.0))
    assert again.q == pytest.approx(list(moved))
    bench.state = following(robot_state(q=moved))
    setpoint(bench.step(squeeze=1.0))
    assert bench.solver.solves[-1][1] == moved


# The new-squeeze rule


def test_a_squeeze_held_through_the_grant_moves_nothing_until_released():
    bench = Bench()
    bench.state = nobody(1)
    bench.run(5, squeeze=1.0)
    bench.state = robot_state(owner_number=6)  # the takeover is granted
    held = bench.run(100, squeeze=1.0)
    assert len(of(held, HumanSetpoint)) == 100 and none_engaged(held)
    assert not any(step.engaged for step in bench.graph.steps)
    assert none_engaged(bench.run(5, squeeze=1.0, clutch=True))
    assert not bench.logic.armed and bench.solver.solves == []

    assert setpoint(bench.step()).grip_released
    assert bench.logic.armed
    assert setpoint(bench.step(squeeze=1.0)).engaged


def test_a_squeeze_held_through_a_data_loss_does_not_engage_when_the_data_come_back():
    bench = Bench()
    bench.step()
    assert bench.logic.armed
    bench.lost()
    assert not bench.logic.armed
    # The grip was squeezed during the loss and is still down.
    assert none_engaged(bench.run(50, squeeze=1.0))
    assert none_engaged(bench.run(5, squeeze=1.0, clutch=True))
    assert bench.solver.solves == []

    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged


def test_one_pass_without_the_controller_ends_the_engagement_and_needs_a_new_squeeze():
    bench = Bench()
    bench.engage()
    bench.run(3, squeeze=1.0)
    ended = setpoint(bench.lost(stage="no_controller", age=0.01))
    assert (ended.engaged, ended.q, ended.grip_released) == (False, None, False)
    assert ended.note == AGAIN + "controller data lost"
    assert not bench.logic.armed

    solves = len(bench.solver.solves)
    back = bench.run(50, squeeze=1.0)
    assert none_engaged(back) and len(bench.solver.solves) == solves
    assert setpoint(back[-1:]).note == AGAIN + "controller data lost"

    bench.halted()
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged
    counts = bench.logic.counts()
    assert (counts["engagements"], counts["data_lost"]) == (2, 1)


def test_an_old_robot_state_ends_the_engagement_and_needs_a_new_squeeze():
    bench = Bench()
    bench.engage()
    bench.state_age_s = 0.25
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the robot state is 0.25 s old"
    assert not bench.logic.armed

    bench.state_age_s = 0.01
    solves = len(bench.solver.solves)
    assert none_engaged(bench.run(50, squeeze=1.0))
    assert len(bench.solver.solves) == solves
    bench.halted()
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.counts()["state_old"] == 1


def test_an_owner_change_ends_the_engagement_and_needs_a_new_squeeze():
    bench = Bench()
    bench.engage()
    bench.state = robot_state(owner="recovery", owner_number=4)
    assert none_engaged(bench.run(3, squeeze=1.0))
    assert not bench.logic.armed
    bench.state = robot_state(owner_number=5)  # the arm is the person's again
    assert none_engaged(bench.run(50, squeeze=1.0))
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.counts()["not_owner"] == 1


def test_the_thors_needs_release_is_mirrored():
    bench = Bench()
    bench.state = robot_state(needs_release=True)
    released = setpoint(bench.step())
    # The released grip is reported; arming waits for the Thor to take it.
    assert released.grip_released and released.note == NOTE_RELEASE_FIRST
    assert not bench.logic.armed
    assert none_engaged(bench.run(10, squeeze=1.0, clutch=True))
    assert bench.solver.solves == []

    bench.state = robot_state()
    bench.step()
    assert bench.logic.armed
    assert setpoint(bench.step(squeeze=1.0)).engaged

    # Raised during an engagement, it ends it with the Thor's reason.
    why = "the live view is not fresh (0.7 s old, the limit is 0.5 s)"
    bench.state = robot_state(needs_release=True, ended=why)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + why
    assert not bench.logic.armed
    assert bench.logic.counts()["needs_release"] == 1


def test_a_release_asked_for_while_other_work_ran_names_that_work():
    def ended_note(**human: Any) -> str:
        """The note of the pass on which the Thor's needs_release ends a squeeze."""
        bench = Bench()
        bench.engage()
        bench.state = robot_state(needs_release=True, **human)
        ended = setpoint(bench.step(squeeze=1.0))
        assert (ended.engaged, ended.q) == (False, None)
        return ended.note

    # The squeeze began while other work ran on the Thor, which refused it as
    # busy. No drive ran, so "ended" still holds the end of the drive before.
    before = "the grip was released"
    busy = "busy: a hand-back is running"
    assert ended_note(ended=before, refusal=busy) == AGAIN + busy
    assert ended_note(ended="", refusal=busy) == AGAIN + busy

    # A refusal that names no work: the drive's own end is quoted.
    why = "the live view is not fresh (0.7 s old, the limit is 0.5 s)"
    for refusal in ("", None, "no fresh controller data", "the arm is busy: no"):
        assert ended_note(ended=why, refusal=refusal) == AGAIN + why, refusal
    # With neither, the logic's own words.
    own = AGAIN + "the robot app asks for a released grip"
    assert ended_note(ended="", refusal="") == own
    assert ended_note(ended="", refusal="no fresh controller data") == own


def test_armed_reaches_the_graph_as_the_permission_and_the_running_state():
    bench = Bench()
    assert not bench.logic.armed
    bench.step(squeeze=1.0)  # not armed at the start: stopped, nothing engages
    bench.step()  # handed in as not armed; the released grip arms
    bench.step(squeeze=1.0)  # armed: running and permitted, the clutch engages
    bench.lost()  # handed in as armed; the loss takes it away
    bench.step(squeeze=1.0)  # stopped again: the held squeeze engages nothing
    bench.step()  # the released grip arms again
    bench.lost()  # handed in as armed; a loss with nothing engaged
    bench.step(squeeze=1.0)  # stopped: the squeeze that began in the loss
    assert bench.graph.handed == [
        (True, False),
        (True, False),
        (False, True),
        (False, True),
        (True, False),
        (True, False),
        (False, True),
        (True, False),
    ]
    engaged = [step.engaged for step in bench.graph.steps]
    assert engaged == [False, False, True, False, False, False, False, False]


# The workstation's own ends


def test_a_jump_ends_the_engagement_and_needs_a_new_squeeze():
    bench = Bench()
    bench.engage()
    jump = "IK jump of 0.300 rad on joint index 1, the limit is 0.05"
    bench.solver.answer = lambda target, seed: IkStep(None, jump)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + jump
    assert not bench.logic.armed

    bench.solver.answer = None
    solves = len(bench.solver.solves)
    assert none_engaged(bench.run(50, squeeze=1.0))
    assert len(bench.solver.solves) == solves
    bench.halted()
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.counts()["jump"] == 1


# Out of reach: a pose with no solution does not end the engagement. The arm
# follows as far as it can, the tip's position first and the wrist's angle
# second: the position with the orientation of the last joint target, by
# shares of the way, and then the solver's nearest pose.


def no_solution(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
    return IkStep(None, NO_SOLUTION)


def no_nearer(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
    return IkStep(None, NO_NEARER)


def test_where_nothing_is_nearer_the_last_joint_target_stands_and_the_engagement_goes_on():
    bench = Bench()
    first = bench.engage()
    start = bench.hand.copy()
    bench.hand = start + np.array([0.0, 0.0, 0.1])
    bench.solver.answer, bench.solver.near = no_solution, no_nearer
    for _ in range(4):
        again = setpoint(bench.step(squeeze=1.0))
        assert again.engaged and again.q == first.q  # the last joint values
        assert NOTE_OUT_OF_REACH not in again.note and bench.logic.stand == ""
    bench.solver.answer = None
    moved = setpoint(bench.step(squeeze=1.0))  # the count starts again
    assert moved.engaged and moved.q != first.q
    bench.solver.answer = no_solution
    assert all_engaged(bench.run(4, squeeze=1.0))
    # From the fifth pass in a row the note says what holds the arm. However
    # long nothing is nearer, no joint target but the last one goes out, and
    # nothing ends the engagement.
    held = bench.run(100, squeeze=1.0)
    assert all_engaged(held) and bench.logic.armed
    assert {tuple(sent.q) for sent in of(held, HumanSetpoint)} == {tuple(moved.q)}
    assert all(NOTE_OUT_OF_REACH in sent.note for sent in of(held, HumanSetpoint))
    assert bench.logic.stand == NOTE_OUT_OF_REACH == "out of reach"
    assert bench.logic.counts()["no_solution"] == 0
    # The pose solves again: the arm follows again at once, under the speed
    # limit, and the note is gone.
    bench.solver.answer = None
    on = setpoint(bench.step(squeeze=1.0))
    assert on.engaged and NOTE_OUT_OF_REACH not in on.note
    step = max(abs(a - b) for a, b in zip(on.q, moved.q))
    assert step == pytest.approx(0.01)  # one pass at the hand speed limit
    # A released grip ends it as any engagement.
    assert none_engaged(bench.run(1))

    # The hand's own bounds still end it: the hand further from the tip than
    # is believed.
    bench = Bench()
    bench.engage()
    bench.solver.answer, bench.solver.near = no_solution, no_nearer
    assert all_engaged(bench.run(10, squeeze=1.0))
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.51])
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note.startswith(AGAIN + "the hand is 51 cm from the gripper tip")


def test_a_pose_with_no_solution_is_tried_by_position_and_then_by_the_nearest_pose():
    assert REACH_SHARES == (1.0, 0.5, 0.25)
    bench = Bench()
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.1])
    bench.graph._base[:3, :3] = turned_about_z(0.3)  # the hand turned as well
    bench.solver.answer, bench.solver.near = no_solution, no_nearer
    solves = len(bench.solver.solves)
    held = setpoint(bench.step(squeeze=1.0))
    assert held.engaged and held.q == first.q
    at = bench.solver.fk(first.q)
    whole = bench.settings.retarget.max_linear_velocity * PASS_S  # 2 mm
    turn = bench.settings.retarget.max_angular_velocity * PASS_S
    asked = bench.solver.solves[solves:]
    assert all(seed == tuple(first.q) for _, seed in asked)
    # The pass's pose: one pass of both hand speed limits from the arm's.
    assert np.allclose(asked[0][0][:3, :3], turned_about_z(turn), atol=1e-12)
    assert asked[0][0][2, 3] - at[2, 3] == pytest.approx(whole)
    # Then its position, the whole way, half and a quarter of it, with the
    # orientation of the last joint target.
    assert [pose[2, 3] - at[2, 3] for pose, _ in asked[1:]] == pytest.approx(
        [whole, whole / 2, whole / 4]
    )
    assert all(np.array_equal(pose[:3, :3], at[:3, :3]) for pose, _ in asked[1:])
    # Then the nearest pose to that position with the hand's own orientation,
    # from the same joint values.
    [(pose, seed)] = bench.solver.nearests
    assert seed == tuple(first.q)
    assert pose[2, 3] - at[2, 3] == pytest.approx(whole)
    assert np.allclose(pose[:3, :3], turned_about_z(0.3), atol=1e-12)


def test_a_hand_turned_out_of_reach_is_followed_by_its_position_alone():
    bench = Bench()
    first = bench.engage()
    bench.solver.turns = False  # no pose that is turned has a solution
    bench.graph._base[:3, :3] = turned_about_z(0.5)
    bench.hand = bench.hand + np.array([0.0, 0.03, 0.02])
    sent = bench.drive(40)
    assert all(m.engaged for m in sent)
    # The tip went the hand's way at the hand speed limit, 2 mm a pass, with
    # the orientation the arm has; the nearest pose was not asked for.
    tips = np.array([bench.solver.fk(m.q)[:3, 3] for m in [first, *sent]])
    np.testing.assert_allclose(tips[-1] - tips[0], [0.0, 0.03, 0.02], atol=1e-9)
    assert np.linalg.norm(np.diff(tips, axis=0), axis=1).max() <= 0.002 + 1e-4
    assert bench.solver.nearests == []
    # The note says so from the fifth pass on, while the hand's pose is not
    # reached, also once the tip stands at the hand's point.
    said = [NOTE_OUT_OF_REACH in m.note for m in sent]
    assert said == [False] * 4 + [True] * 36 and bench.logic.stand == NOTE_OUT_OF_REACH
    # The hand turns back: its pose has a solution again and the note goes,
    # with no new squeeze.
    bench.graph._base[:3, :3] = np.eye(3)
    bench.hand = bench.hand + np.array([0.0, -0.01, 0.0])
    back = bench.drive(10)
    assert all(m.engaged and NOTE_OUT_OF_REACH not in m.note for m in back)
    np.testing.assert_allclose(
        bench.solver.fk(back[-1].q)[:3, 3] - tips[0], [0.0, 0.02, 0.02], atol=1e-9
    )
    assert bench.logic.stand == "" and bench.logic.counts()["engagements"] == 1
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert not any(bench.logic.counts()[key] for key in ended)


def test_a_hand_beyond_the_reach_is_followed_to_its_edge_and_along_it():
    bench = Bench()
    first = bench.engage()
    # The third joint, which alone raises the stand-in arm's tip, stops at
    # 0.2 rad: the tip reaches 3.5 cm above where it is.
    bench.solver.top = 0.2
    edge = FakeSolver.BASE[2] + 0.2 * FakeSolver.M_PER_RAD
    assert edge - tip(bench, first)[2] == pytest.approx(0.035)
    # The hand goes 4.5 cm up and 6 cm to the side, at once.
    bench.hand = bench.hand + np.array([0.0, 0.06, 0.045])
    sent = bench.drive(120)
    assert all(m.engaged for m in sent) and bench.logic.armed
    tips = np.array([tip(bench, m) for m in [first, *sent]])
    # The tip rose to the edge of reach and no further, and went on to the
    # side along it, to under the hand within a millimetre: no faster than
    # the hand speed limit, and never back.
    assert tips[:, 2].max() <= edge + 1e-9
    np.testing.assert_allclose(tips[-1], HOME_TIP + [0.0, 0.06, 0.035], atol=1e-3)
    assert np.linalg.norm(np.diff(tips, axis=0), axis=1).max() <= 0.002 + 1e-4
    assert np.all(np.diff(tips[:, 1]) >= -1e-12) and np.all(
        np.diff(tips[:, 2]) >= -1e-12
    )
    # The last part of the way was the nearest pose's; the arm then stands.
    assert len(bench.solver.nearests) > 20
    assert sent[-1].q == sent[-20].q
    assert NOTE_OUT_OF_REACH in sent[-1].note and bench.logic.stand == NOTE_OUT_OF_REACH
    # The hand comes back into reach: it is followed as everywhere, with no
    # new squeeze, and the note is gone.
    bench.hand = bench.hand + np.array([0.0, 0.0, -0.025])
    back = bench.drive(40)
    assert all(m.engaged for m in back)
    np.testing.assert_allclose(
        tip(bench, back[-1]), HOME_TIP + [0.0, 0.06, 0.02], atol=1e-9
    )
    assert NOTE_OUT_OF_REACH not in back[-1].note and bench.logic.stand == ""
    counts = bench.logic.counts()
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert counts["engagements"] == 1 and not any(counts[key] for key in ended)


def test_a_nearest_pose_keeps_the_hand_speed_limits_and_the_tip_box():
    # A nearest pose 4 mm up and 3 mm to the side of the tip, on a pass whose
    # hand speed limit is 2 mm: the joint target goes the share of the way
    # that the limit allows.
    def aside(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
        return IkStep((seed[0], seed[1] + 0.015, seed[2] + 0.02, *seed[3:]), "")

    bench = boxed()
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.03])
    bench.solver.answer, bench.solver.near = no_solution, aside
    [moved] = bench.drive(1)
    way = tip(bench, moved) - tip(bench, first)
    np.testing.assert_allclose(way, [0.0, 0.0012, 0.0016], atol=1e-9)
    # One that would take the tip out of the tip box is not taken: the box
    # ends 4 cm above the tip, less its margin.
    top = TIP_BOX[1][2] - TIP_BOX_MARGIN_M
    bench.solver.answer = None
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.02])
    [*_, at_the_top] = bench.drive(40)
    assert tip(bench, at_the_top)[2] == pytest.approx(top, abs=1e-9)
    bench.solver.answer = no_solution
    stood = bench.drive(10)
    assert all(m.engaged and m.q == at_the_top.q for m in stood)
    assert len(bench.solver.nearests) == 11


def test_a_jump_among_the_poses_tried_out_of_reach_does_not_end_the_engagement():
    jump = "IK jump of 0.300 rad on joint index 1, the limit is 0.05"

    def answer(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
        turned = not np.allclose(target[:3, :3], np.eye(3))
        return IkStep(None, NO_SOLUTION if turned else jump)

    bench = Bench()
    first = bench.engage()
    bench.graph._base[:3, :3] = turned_about_z(0.5)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.01])
    bench.solver.answer = answer
    sent = bench.drive(10)
    # The hand's pose has no solution and its position alone is a jump: the
    # nearest pose is taken, and the tip comes to the hand's point.
    assert all(m.engaged for m in sent) and bench.logic.counts()["jump"] == 0
    assert tip(bench, sent[-1])[2] - tip(bench, first)[2] == pytest.approx(
        0.01, abs=2e-4
    )


@pytest.mark.parametrize(
    "q",
    [
        (float("nan"),) * N_JOINTS,
        (float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0,) * (N_JOINTS - 1),
        (0.0,) * (N_JOINTS + 1),
        None,
    ],
    ids=["nan", "inf", "five values", "seven values", "none"],
)
def test_joint_values_that_are_not_six_finite_numbers_count_as_a_failed_solve(q):
    bench = Bench()
    first = bench.engage()
    bench.solver.answer = lambda target, seed: IkStep(q, "")
    for _ in range(20):
        again = setpoint(bench.step(squeeze=1.0))
        assert again.engaged and again.q == first.q  # the last joint values
    # No value of such a solution reaches the arm, and the note says why
    # the arm stands.
    assert again.note == NOTE_OUT_OF_REACH and bench.logic.armed
    assert bench.logic.counts()["no_solution"] == 0


@pytest.mark.parametrize("field", ["ee_target", "commanded_target"])
def test_an_engaged_pass_without_a_target_ends_the_engagement(field):
    bench = Bench()
    bench.engage()
    solves = len(bench.solver.solves)
    ended = setpoint(bench.step(squeeze=1.0, without=field))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the retargeting graph gave no target"
    assert not bench.logic.armed and len(bench.solver.solves) == solves
    assert none_engaged(bench.run(20, squeeze=1.0))
    assert bench.logic.counts()["no_solution"] == 1


def test_a_joint_target_too_far_ahead_of_an_arm_that_stays_ends_the_engagement():
    bench = Bench()  # the measured pose is the rest pose
    bench.engage()  # the ROBOT_STATE goes on showing the arm at rest
    bench.solver.answer = lambda target, seed: IkStep((0.5, *seed[1:]), "")
    # The joint target goes toward the solution a step at a time.
    limit = bench.settings.ik.max_ahead_rad
    assert 3 * JOINT_STEP < limit < 4 * JOINT_STEP
    for steps in (1, 2, 3):
        sent = setpoint(bench.step(squeeze=1.0))
        assert sent.q[0] == pytest.approx(steps * JOINT_STEP)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the solution is 0.106 rad ahead of the arm"
    assert not bench.logic.armed
    assert none_engaged(bench.run(20, squeeze=1.0))
    assert bench.logic.counts()["solution_ahead"] == 1

    # The joint values that go out again after a failed solve are checked too.
    bench = Bench()
    bench.engage()
    bench.solver.answer = lambda target, seed: IkStep(None, "no IK solution")
    assert setpoint(bench.step(squeeze=1.0)).engaged
    bench.state = robot_state(q=(0.3, 0.025, 0.025, 0.0, 0.0, 0.0))
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the solution is 0.300 rad ahead of the arm"
    assert not bench.logic.armed


def test_a_hand_further_ahead_of_the_arm_than_is_followed_ends_the_engagement():
    bench = Bench(ik=IkSettings(max_ahead_m=0.05))
    bench.engage()
    assert HumanSettings().ik.max_ahead_m == 0.5  # as the two shipped files
    start = bench.hand.copy()
    bench.hand = start + np.array([0.0, 0.0, 0.049])
    sent = setpoint(bench.step(squeeze=1.0))
    assert sent.engaged and sent.note == ""
    assert bench.logic.ahead_m == pytest.approx(0.049)
    solves = len(bench.solver.solves)
    # The arm's joint target has come 2 mm: the hand is 6 cm ahead of its tip.
    bench.hand = start + np.array([0.0, 0.0, 0.062])
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    why = "the hand is 6 cm from the gripper tip, over the 5 cm believed"
    assert ended.note == AGAIN + why
    assert not bench.logic.armed and len(bench.solver.solves) == solves
    assert bench.logic.ahead_m is None
    assert none_engaged(bench.run(20, squeeze=1.0))
    assert bench.logic.counts()["hand_ahead"] == 1


def turned_about_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_a_hand_turned_further_ahead_of_the_arm_than_is_followed_ends_the_engagement():
    assert MAX_AHEAD_TURN_RAD == 2.0
    bench = Bench()
    bench.engage()
    # The stand-in arm does not turn: the hand's turn stays ahead of it.
    bench.graph._base[:3, :3] = turned_about_z(1.9)
    assert all_engaged(bench.run(5, squeeze=1.0))
    # Each pass solved for a pose one pass of the angular limit from the arm's.
    pose, _ = bench.solver.solves[-1]
    most = bench.settings.retarget.max_angular_velocity * PASS_S
    assert np.allclose(pose[:3, :3], turned_about_z(most), atol=1e-12)
    bench.graph._base[:3, :3] = turned_about_z(2.1)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    why = "the hand is turned 120 deg ahead of the arm (115 are followed)"
    assert ended.note == AGAIN + why
    assert bench.logic.counts()["hand_ahead"] == 1


def test_the_arm_catches_up_with_a_hand_that_is_ahead_and_stops_where_it_stopped():
    bench = Bench(ik=dataclasses.replace(IkSettings(), max_ahead_m=0.5))
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.2])  # 20 cm, at once
    sent = [first, *bench.drive(130)]
    assert all(m.engaged for m in sent) and bench.logic.armed
    # The tip of the joint target moves at the hand speed limit, 2 mm a pass,
    # which is 0.01 rad on the stand-in arm's third joint; then it stands.
    steps = np.diff([m.q for m in sent], axis=0)
    per_pass = bench.settings.retarget.max_linear_velocity * PASS_S / 0.2
    assert np.allclose(steps[:100, 2], per_pass, atol=1e-9)
    assert np.allclose(steps[100:], 0.0, atol=1e-9)
    assert np.allclose(steps[:, [0, 1, 3, 4, 5]], 0.0, atol=1e-12)
    assert sent[-1].q[2] == pytest.approx(first.q[2] + 0.2 / 0.2)
    assert bench.logic.ahead_m == pytest.approx(0.0, abs=1e-9)
    # The note says how far the hand is ahead, in steps, down to 5 cm.
    said = [m.note for m in sent]
    changes = [said[0]] + [b for a, b in zip(said, said[1:]) if a != b]
    assert changes == [
        "",
        "hand 20 cm ahead: the arm is catching up",
        "hand 10 cm ahead: the arm is catching up",
        "hand 5 cm ahead: the arm is catching up",
        "",
    ]
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert not any(bench.logic.counts()[key] for key in ended)
    # A released grip ends it on its pass, wherever the arm is on its way.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.2])
    assert all(m.engaged for m in bench.drive(10))
    released = setpoint(bench.step())
    assert (released.engaged, released.q, released.grip_released) == (False, None, True)


@pytest.mark.parametrize(
    "cm, shown, gives",
    [
        (4.9, None, None),
        (5.0, None, 5),
        (9.9, None, 5),
        (10.0, None, 10),
        (27.0, None, 20),
        (49.0, None, 40),
        # A note stands until the hand is 2 cm outside its step.
        (3.0, 5, 5),
        (2.9, 5, None),
        (11.9, 5, 5),
        (12.0, 5, 10),
        (8.0, 10, 10),
        (7.9, 10, 5),
        (21.9, 10, 10),
        (22.0, 10, 20),
        (18.0, 20, 20),
        (17.9, 20, 10),
    ],
)
def test_the_catching_up_note_changes_in_steps_and_not_on_every_pass(cm, shown, gives):
    assert logic_module._ahead_step(cm, shown) == gives
    assert NOTE_CATCHING_UP.format(cm=5) == "hand 5 cm ahead: the arm is catching up"


def test_no_joint_of_the_target_moves_faster_than_a_share_of_the_thors_clamp():
    assert logic_module.THOR_SPEED_MARGIN == SPEED_MARGIN == 0.98
    assert JOINT_SPEED_SHARE == 0.9
    # A hand speed limit of 2 cm a pass asks the stand-in arm for 0.1 rad.
    fast = dataclasses.replace(RetargetSettings(), max_linear_velocity=1.0)
    wide = dataclasses.replace(IkSettings(), max_ahead_m=0.5)
    bench = Bench(retarget=fast, ik=wide)
    first = bench.engage()
    way = np.array([0.03, -0.06, 0.12])  # m: 0.15, -0.3 and 0.6 rad
    bench.hand = bench.hand + way
    sent = [first, *bench.drive(40)]
    steps = np.diff([m.q for m in sent], axis=0)
    moving = steps[np.abs(steps).max(axis=1) > 1e-9]
    # The joint that has the farthest to go moves at the share of the clamp,
    # and the others by the same share of their way: the direction is kept.
    assert len(moving) == 23 and np.allclose(steps[23:], 0.0, atol=1e-9)
    assert np.allclose(np.abs(moving[:-1]).max(axis=1), JOINT_STEP, atol=1e-12)
    assert np.abs(steps).max() <= JOINT_STEP + 1e-12
    for step in moving:
        assert np.allclose(step[:3] / step[2], way / way[2], atol=1e-9)
    assert np.allclose(np.array(sent[-1].q[:3]) - first.q[:3], way / 0.2, atol=1e-9)
    assert bench.logic.joint_speed(bench.welcome) == pytest.approx(JOINT_STEP / PASS_S)


@pytest.mark.parametrize(
    "named, period_s, speed",
    [
        (0.8, PASS_S, 0.8),
        (1.5, PASS_S, 1.5),
        (None, PASS_S, UNNAMED_JOINT_SPEED),  # a robot app that names none
        (0.0, PASS_S, UNNAMED_JOINT_SPEED),
        (0.8, 0.01, 0.4),  # a robot app whose setpoints come twice as often
        (0.8, 0.05, 0.8),  # one whose setpoints come slower: this program's pass
    ],
)
def test_the_joint_speed_is_the_one_the_robot_apps_welcome_names(
    named, period_s, speed
):
    assert UNNAMED_JOINT_SPEED == 0.1
    fast = dataclasses.replace(RetargetSettings(), max_linear_velocity=1.0)
    bench = Bench(retarget=fast, ik=dataclasses.replace(IkSettings(), max_ahead_m=0.5))
    bench.welcome = dataclasses.replace(
        welcome(human_max_joint_speed=named), setpoint_period_s=period_s
    )
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.3])
    sent = [first, *bench.drive(5)]
    steps = np.diff([m.q[2] for m in sent])
    assert np.allclose(steps, JOINT_SPEED_SHARE * SPEED_MARGIN * speed * PASS_S)
    # What the program says of its speed is that step per pass.
    said = bench.logic.joint_speed(bench.welcome)
    assert said == pytest.approx(JOINT_SPEED_SHARE * SPEED_MARGIN * speed)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 1.5, 3.0, np.pi - 1e-8, np.pi])
@pytest.mark.parametrize(
    "axis", [(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.6, 0.0, -0.8), (-0.48, 0.6, 0.64)]
)
def test_the_step_toward_a_pose_is_bounded_in_its_way_and_in_its_turn(axis, angle):
    unit = np.array(axis)
    at = np.eye(4)
    at[:3, :3] = turned_about_z(0.7)
    at[:3, 3] = [0.3, -0.1, 0.2]
    goal = np.eye(4)
    goal[:3, :3] = at[:3, :3] @ logic_module._turn(unit, angle)
    goal[:3, 3] = at[:3, 3] + np.array([0.03, 0.0, -0.04])  # 5 cm away
    # The axis and the angle of a turn are read back, a half turn included.
    found, turned = logic_module._axis_angle(at[:3, :3].T @ goal[:3, :3])
    assert turned == pytest.approx(angle, abs=1e-7)
    if angle > 1e-6:
        back = logic_module._turn(found, turned)
        assert np.allclose(back, logic_module._turn(unit, angle), atol=1e-6)
    # A goal within both bounds, or within the slack of them, is given whole.
    whole = logic_module._toward(at, goal, 0.05, 3.2, 1e-4)
    assert np.array_equal(whole, goal)
    assert np.array_equal(logic_module._toward(at, goal, 0.04991, 3.2, 1e-4), goal)
    # Else the pose lies 2 cm and 0.1 rad on the way to it, each by itself.
    pose = logic_module._toward(at, goal, 0.02, 0.1, 1e-4)
    way = pose[:3, 3] - at[:3, 3]
    assert np.allclose(way, 0.4 * (goal[:3, 3] - at[:3, 3]), atol=1e-12)
    step = logic_module._axis_angle(at[:3, :3].T @ pose[:3, :3])[1]
    assert step == pytest.approx(min(angle, 0.1), abs=1e-7)
    if angle > 0.1 + 1e-4:
        left = logic_module._axis_angle(pose[:3, :3].T @ goal[:3, :3])[1]
        assert left == pytest.approx(angle - 0.1, abs=1e-6)
        assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)


def test_a_late_pass_moves_the_joint_target_no_further_and_an_early_one_less():
    fast = dataclasses.replace(RetargetSettings(), max_linear_velocity=1.0)
    bench = Bench(retarget=fast, ik=dataclasses.replace(IkSettings(), max_ahead_m=0.5))
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.3])
    [one] = bench.drive(1)
    assert one.q[2] - first.q[2] == pytest.approx(JOINT_STEP)
    bench.stall(0.1)  # the loop stood for 0.1 s
    [late] = bench.drive(1)
    assert late.q[2] - one.q[2] == pytest.approx(JOINT_STEP)
    bench.passes -= 0.5  # the next pass comes after half a period
    [early] = bench.drive(1)
    assert early.q[2] - late.q[2] == pytest.approx(0.5 * JOINT_STEP)


def test_the_first_joint_target_stands_until_the_thor_drives_for_it():
    bench = Bench(ik=dataclasses.replace(IkSettings(), max_ahead_m=0.5))
    first = bench.engage(follow=False)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.02])
    solves = len(bench.solver.solves)
    # No drive; one that still reads the arm and has sent nothing; one whose
    # newest human target is older than this engagement's first; one whose
    # own joint target is still more than one of its steps from the first.
    older = following(robot_state(seq=first.seq - 1))
    step = SPEED_MARGIN * HUMAN_SPEED * PASS_S
    behind = (first.q[0], first.q[1] - step - 0.001, *first.q[2:])
    on_its_way = following(robot_state(seq=first.seq, target_q=behind))
    for state in (robot_state(), following(sent=0), older, on_its_way):
        bench.state = state
        held = setpoint(bench.step(squeeze=1.0))
        assert held.engaged and held.q == first.q
    assert len(bench.solver.solves) == solves
    assert bench.logic.ahead_m == pytest.approx(0.02)
    # The drive has sent a setpoint for this engagement, and its joint target
    # is within a step of the first target: the target moves on.
    near = (first.q[0], first.q[1] - step + 0.001, *first.q[2:])
    bench.state = following(robot_state(seq=first.seq, target_q=near))
    moved = setpoint(bench.step(squeeze=1.0))
    assert moved.q[2] == pytest.approx(first.q[2] + 0.01)
    # A ROBOT_STATE on its way since before need not show it again.
    bench.state = older
    assert setpoint(bench.step(squeeze=1.0)).q[2] == pytest.approx(first.q[2] + 0.02)

    # A robot app that names no seq is taken at the drive and its setpoint.
    bench = Bench()
    first = bench.engage(follow=False)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.002])
    assert "seq" not in following().health["human"]
    bench.state = following(sent=0)
    assert setpoint(bench.step(squeeze=1.0)).q == first.q
    bench.state = following()
    assert setpoint(bench.step(squeeze=1.0)).q[2] == pytest.approx(first.q[2] + 0.01)


def test_an_arm_that_does_not_follow_a_squeeze_ends_the_engagement_after_a_second():
    assert FOLLOW_WAIT_S == 1.0
    bench = Bench()
    first = bench.engage(follow=False)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.03])
    held = of(bench.run(round(FOLLOW_WAIT_S / PASS_S), squeeze=1.0), HumanSetpoint)
    assert len(held) == 50 and all(m.engaged and m.q == first.q for m in held)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the arm did not follow the squeeze within 1 s"
    assert not bench.logic.armed
    assert bench.logic.counts()["solution_ahead"] == 1


def test_a_jump_halves_the_step_of_the_tip_and_a_jump_that_stays_ends_the_engagement():
    assert STEP_HALVINGS == 3
    bench = Bench(ik=dataclasses.replace(IkSettings(), max_ahead_m=0.5))
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.1])
    at = bench.solver.fk(first.q)[2, 3]
    whole = bench.settings.retarget.max_linear_velocity * PASS_S  # 2 mm

    def jumps_above(metres: float) -> Callable[[np.ndarray, Any], IkStep]:
        """The stand-in arm's solution, a jump for a step of the tip above metres."""

        def answer(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
            if target[2, 3] - at > metres + 1e-9:
                return IkStep(None, "IK jump of 0.300 rad on joint index 2")
            first_three = (target[:3, 3] - FakeSolver.BASE) / FakeSolver.M_PER_RAD
            return IkStep((*(float(v) for v in first_three), *seed[3:]), "")

        return answer

    # A quarter of the step is no jump: the third solve of the pass is taken.
    bench.solver.answer = jumps_above(whole / 4)
    solves = len(bench.solver.solves)
    sent = setpoint(bench.step(squeeze=1.0))
    asked = [pose[2, 3] - at for pose, _ in bench.solver.solves[solves:]]
    assert asked == pytest.approx([whole, whole / 2, whole / 4])
    assert sent.engaged and sent.q[2] == pytest.approx(first.q[2] + whole / 4 / 0.2)
    assert bench.logic.counts()["jump"] == 0

    # Every step down to an eighth is a jump: the engagement ends.
    at = bench.solver.fk(sent.q)[2, 3]
    bench.solver.answer = jumps_above(whole / 16)
    solves = len(bench.solver.solves)
    ended = setpoint(bench.step(squeeze=1.0))
    asked = [pose[2, 3] - at for pose, _ in bench.solver.solves[solves:]]
    assert asked == pytest.approx([whole, whole / 2, whole / 4, whole / 8])
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "IK jump of 0.300 rad on joint index 2"
    assert bench.logic.counts()["jump"] == 1


def test_the_lead_is_measured_from_the_arm_until_the_thor_drives():
    bench = Bench()
    first = bench.engage(follow=False)
    assert first.q == pytest.approx(list(REST_INSIDE))
    # No human drive runs: the Thor's last joint target is of an earlier move.
    far = (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    bench.state = robot_state(target_q=far)
    assert setpoint(bench.step(squeeze=1.0)).engaged
    # A drive that has sent nothing yet: the target is still the earlier one.
    bench.state = following(robot_state(target_q=far), sent=0)
    assert setpoint(bench.step(squeeze=1.0)).engaged
    # The arm's measured pose counts, and nothing is taken off for its age.
    bench.state = following(robot_state(q=(0.09, 0.0, 0.0, 0.0, 0.0, 0.0)), sent=0)
    bench.state_age_s = 0.19
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the solution is 0.090 rad ahead of the arm"


@pytest.mark.parametrize("age_s, follows", [(0.01, False), (0.03, True), (0.19, True)])
def test_once_the_thor_drives_the_lead_is_measured_from_where_its_target_can_be(
    age_s, follows
):
    bench = Bench()
    first = bench.engage()
    bench.solver.answer = lambda target, seed: IkStep(seed, "")  # the target stands
    # The Thor's last joint target counts, the measured pose does not.
    measured = (0.3, 0.3, 0.3, 0.0, 0.0, 0.0)
    bench.state = following(robot_state(q=measured, target_q=first.q))
    assert setpoint(bench.step(squeeze=1.0)).engaged
    bench.state = following(robot_state(q=first.q))  # no target known
    assert setpoint(bench.step(squeeze=1.0)).engaged
    # A joint target that is not six finite numbers is no reference either.
    for target in ((0.5, 0.5), (float("nan"),) * N_JOINTS):
        bench.state = following(robot_state(q=first.q, target_q=target))
        assert setpoint(bench.step(squeeze=1.0)).engaged

    # The Thor's joint target was 0.115 rad behind when its ROBOT_STATE came.
    # Since then its speed clamp can have moved it age x 0.98 x 1.5 rad nearer.
    behind = (first.q[0] - 0.115, *first.q[1:])
    bench.state = following(robot_state(q=first.q, target_q=behind))
    bench.state_age_s = age_s
    nearer = age_s * SPEED_MARGIN * HUMAN_SPEED
    assert (0.115 - nearer <= bench.settings.ik.max_ahead_rad) == follows
    sent = setpoint(bench.step(squeeze=1.0))
    assert sent.engaged == follows
    if not follows:
        assert sent.note == AGAIN + "the solution is 0.100 rad ahead of the arm"
        assert bench.logic.counts()["solution_ahead"] == 1


# A new squeeze after an engagement: it counts only on a ROBOT_STATE that shows
# that engagement's drive as ended. One made earlier gives a pose the arm has
# left, and a first target there would pull the arm back.


def test_a_new_squeeze_waits_for_the_robot_state_that_shows_the_last_drive_ended():
    bench = Bench()
    bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.2])
    sent = bench.drive(6)
    behind, at = sent[-3].q, sent[-1].q  # where a late state has the arm, and is
    assert at[2] - behind[2] == pytest.approx(0.02)
    released = setpoint(bench.step())
    assert released.grip_released and not released.engaged
    assert released.note == NOTE_SQUEEZE and not bench.logic.armed
    end, homes = released.seq, len(bench.graph.homes)

    def waits(state: RobotState, squeeze: float = 1.0) -> None:
        bench.state = state
        waiting = setpoint(bench.step(squeeze=squeeze))
        assert not waiting.engaged and not bench.logic.armed
        assert waiting.note == (NOTE_STOPPING if squeeze else NOTE_SQUEEZE)
        assert len(bench.graph.homes) == homes  # no home from such a state

    # States of the drive that still ran: the grip may be released or squeezed.
    late = robot_state(q=behind, target_q=behind, seq=end - 1)
    waits(following(late, sent=5), squeeze=0.0)
    waits(following(late, sent=5))
    # No drive in it, and made before the robot app took the pass that ended
    # the engagement: the drive had not begun, or the state is older still.
    waits(late)
    # Made after that pass, with the drive not ended yet.
    waits(following(robot_state(q=at, target_q=at, seq=end), sent=6))
    # The state that shows the drive as ended: its readings are the first
    # since, and the arm may still move.
    waits(robot_state(q=at, target_q=at, seq=end + 3))
    # New readings of every joint that agree with those: the arm is at rest.
    # The squeeze counts from here on, and the home is the arm's pose of
    # this state.
    shown = setpoint(bench.step(squeeze=1.0))
    assert bench.logic.armed and not shown.engaged and shown.note == NOTE_STOPPING
    assert len(bench.graph.homes) == homes + 1
    np.testing.assert_allclose(bench.graph.homes[-1], bench.solver.fk(at))
    first = setpoint(bench.step(squeeze=1.0))
    assert first.engaged and first.q == pytest.approx(at)
    counts = bench.logic.counts()
    assert (counts["engagements"], counts["released"]) == (2, 1)


def test_a_robot_app_that_names_no_seq_is_taken_at_a_state_that_arrived_after_the_end():
    bench = Bench()
    bench.engage()
    bench.run(3, squeeze=1.0)
    assert "seq" not in bench.state.health["human"]
    bench.step()  # the release ends the engagement
    # No drive in it, but it arrived before that pass: 0.05 s ago.
    bench.halted()
    bench.state_age_s = 0.05
    assert not setpoint(bench.step(squeeze=1.0)).engaged and not bench.logic.armed
    # It arrived after that pass: the drive has ended. The next one shows the
    # arm at rest.
    bench.state_age_s = 0.03
    assert not setpoint(bench.step(squeeze=1.0)).engaged and not bench.logic.armed
    assert not setpoint(bench.step(squeeze=1.0)).engaged and bench.logic.armed
    assert setpoint(bench.step(squeeze=1.0)).engaged


@pytest.mark.parametrize("how", ["jump", "data lost", "error"])
def test_every_end_of_an_engagement_is_waited_for_and_asks_for_a_release_as_before(
    how,
):
    bench = Bench()
    bench.engage()
    [moved] = bench.drive(1)
    if how == "jump":
        bench.solver.answer = lambda target, seed: IkStep(None, "IK jump of 0.3 rad")
        ended = setpoint(bench.step(squeeze=1.0))
    elif how == "data lost":
        ended = setpoint(bench.lost(stage="no_controller", age=0.01))
    else:

        def raising(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
            raise RuntimeError("the solver broke")

        bench.solver.answer = raising
        with pytest.raises(RuntimeError):
            bench.step(squeeze=1.0)
        ended = setpoint(bench.step(squeeze=1.0))
    bench.solver.answer = None
    assert not ended.engaged and not bench.logic.armed
    # The drive is shown as ended, but the grip was not released since: the
    # held squeeze is no new one.
    bench.state = robot_state(q=moved.q, seq=ended.seq + 1)
    held = bench.run(5, squeeze=1.0)
    assert none_engaged(held) and not bench.logic.armed
    assert NOTE_STOPPING not in notes(held)
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged


# The hold-back: the joint target goes no further than two of the robot app's
# steps ahead of where its own joint target can be.

THOR_STEP = SPEED_MARGIN * HUMAN_SPEED * PASS_S  # of the stand-in robot app, rad
# A hand speed limit that asks the stand-in arm for more than a joint may do.
FAST = dataclasses.replace(RetargetSettings(), max_linear_velocity=1.0)


def held_drive() -> tuple[Bench, HumanSetpoint]:
    """An engagement at full joint speed; the Thor's last joint target is `one`."""
    bench = Bench(retarget=FAST)
    bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.45])
    [one] = bench.drive(1)
    return bench, one


def timed(state: RobotState, made_at: float, sent_at: float, sent: int) -> RobotState:
    """state as made at made_at, its drive's last of `sent` setpoints sent at sent_at."""
    return dataclasses.replace(
        state, clock_mono_s=made_at, target_t=sent_at, move_index=sent
    )


def test_the_joint_target_is_held_back_two_thor_steps_ahead_of_where_the_thors_can_be():
    assert HOLD_BACK_STEPS == 2.0
    bench, one = held_drive()
    # The Thor's joint target stands from here on, in ROBOT_STATEs 0.01 s old:
    # by then its clamp can have brought it 0.0147 rad nearer.
    credit = bench.state_age_s * SPEED_MARGIN * HUMAN_SPEED
    room = HOLD_BACK_STEPS * THOR_STEP + credit
    ahead = [setpoint(bench.step(squeeze=1.0)).q[2] - one.q[2] for _ in range(6)]
    # The joint target runs on at its own speed and stops at that room.
    assert ahead == pytest.approx(
        [min(passes * JOINT_STEP, room) for passes in range(1, 7)]
    )
    assert ahead[1] < room == ahead[2] and bench.logic.armed
    assert room - credit <= bench.settings.ik.max_ahead_rad  # the lead does not end it
    # The Thor's joint target comes on by half a step: so does the joint target.
    target = list(one.q)
    for _ in range(4):
        target[2] += 0.5 * THOR_STEP
        bench.state = following(robot_state(q=target, target_q=target, seq=one.seq))
        ahead.append(setpoint(bench.step(squeeze=1.0)).q[2] - one.q[2])
    assert np.diff(ahead[5:]) == pytest.approx([0.5 * THOR_STEP] * 4)
    # The other joints kept their share of the way: the direction is kept.
    assert bench.logic.counts()["solution_ahead"] == 0


def test_a_robot_app_whose_loop_stands_or_runs_slow_gets_less_room():
    assert STANDING_PERIODS == 1.5
    bench, one = held_drive()
    age = bench.state_age_s
    base = following(robot_state(q=one.q, target_q=one.q, seq=one.seq))

    def room(state: RobotState) -> float:
        """How far ahead of the Thor's joint target the joint target comes to stand."""
        fresh, start = held_drive()
        fresh.state = state
        last = [setpoint(fresh.step(squeeze=1.0)).q[2] for _ in range(8)][-1]
        return last - start.q[2]

    full = HOLD_BACK_STEPS * THOR_STEP + age * SPEED_MARGIN * HUMAN_SPEED
    # Its last setpoint went out one period before the state was made: on time.
    assert room(timed(base, 500.0, 500.0 - PASS_S, 1)) == pytest.approx(full)
    # More than a period and a half before: its loop stands, and its joint
    # target is taken not to have moved since the state.
    assert room(timed(base, 500.0, 500.0 - 1.6 * PASS_S, 1)) == pytest.approx(
        HOLD_BACK_STEPS * THOR_STEP
    )
    # Two setpoints in 0.08 s between two states: a loop at half its pace is
    # given half as much for the time since the state.
    fresh, start = held_drive()
    fresh.state = timed(base, 500.0, 500.0 - 0.01, 3)
    fresh.step(squeeze=1.0)
    fresh.state = timed(base, 500.08, 500.08 - 0.01, 5)
    last = [setpoint(fresh.step(squeeze=1.0)).q[2] for _ in range(8)][-1]
    half = HOLD_BACK_STEPS * THOR_STEP + 0.5 * age * SPEED_MARGIN * HUMAN_SPEED
    assert last - start.q[2] == pytest.approx(half)


def test_a_joint_target_that_is_held_back_for_a_second_ends_the_engagement():
    assert NO_FOLLOW_S == 1.0
    assert NOT_FOLLOWED == (
        "the arm does not follow: no new setpoint of the robot app in 1 s"
    )
    assert len(AGAIN + NOT_FOLLOWED) <= NOTE_MAX_CHARS
    bench, one = held_drive()
    # The Thor reports its drive and sends no new setpoint. The joint target
    # is at its room on the third pass and held back from then on.
    free = bench.run(2, squeeze=1.0)
    held = bench.run(round(NO_FOLLOW_S / PASS_S) + 1, squeeze=1.0)
    assert all_engaged(free + held)
    said = [m.note for m in of(free + held, HumanSetpoint)]
    behind = NOTE_THOR_BEHIND + "; " + NOTE_CATCHING_UP.format(cm=40)
    assert said[0] == NOTE_CATCHING_UP.format(cm=40) and said[-1] == behind
    assert said.index(behind) == 2 + BEHIND_NOTE_PASSES // 2 - 1
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + NOT_FOLLOWED and not bench.logic.armed
    assert bench.logic.counts()["solution_ahead"] == 1

    # A joint target of the Thor that comes on, however slowly, ends nothing.
    bench, one = held_drive()
    target = list(one.q)
    for index in range(150):
        if index % 40 == 39:
            target[2] += 0.002
            bench.state = following(robot_state(q=target, target_q=target))
        assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.behind

    # One that only wobbles, by less than a twentieth of a Thor step, is no
    # new one: the engagement ends as with a target that stands.
    assert logic_module.NEW_TARGET_SHARE == 0.05
    bench, one = held_drive()
    ended_at = None
    for index in range(round(NO_FOLLOW_S / PASS_S) + 5):
        wobble = list(one.q)
        wobble[2] += 0.9 * logic_module.NEW_TARGET_SHARE * THOR_STEP * (index % 2)
        bench.state = following(robot_state(q=wobble, target_q=wobble))
        if not setpoint(bench.step(squeeze=1.0)).engaged:
            ended_at = index
            break
    assert ended_at == 2 + round(NO_FOLLOW_S / PASS_S) + 1

    # Nor does a hand that stands: its joint target is not held back.
    bench = Bench(retarget=FAST)
    bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.004])
    [one] = bench.drive(1)
    assert all_engaged(bench.run(150, squeeze=1.0)) and not bench.logic.behind


@pytest.mark.parametrize(
    "seed, q, reference, room, share",
    [
        ((0.0,), (0.02,), (0.0,), 0.05, 1.0),
        ((0.0,), (0.02,), (0.0,), 0.01, 0.5),
        ((0.0,), (-0.02,), (0.0,), 0.01, 0.5),
        # The slowest joint sets the share of all.
        ((0.0, 0.0), (0.02, 0.02), (0.0, -0.015), 0.02, 0.25),
        # A joint that is further than the room already: no further ...
        ((0.03,), (0.05,), (0.0,), 0.02, 0.0),
        # ... but it may come back toward the reference, and past it.
        ((0.03,), (0.01,), (0.0,), 0.02, 1.0),
        ((0.03,), (-0.04,), (0.0,), 0.02, 5 / 7),
        ((0.03,), (0.03,), (0.0,), 0.02, 1.0),  # it stands
    ],
)
def test_the_share_of_its_way_the_joint_target_may_go(seed, q, reference, room, share):
    assert logic_module._share_within(seed, q, reference, room) == pytest.approx(share)


# The tip box: the robot app names the box its checker holds a person's gripper
# tip in; the pose solved for stays 1 cm inside its walls.

TIP_BOX = [[0.1, -0.3, 0.0], [0.5, 0.3, 0.155]]  # m, around the stand-in arm
HOME_TIP = FakeSolver().fk(REST_INSIDE)[:3, 3]  # 0.2, 0.005, 0.105


def boxed(box: Any = TIP_BOX, **settings: Any) -> Bench:
    """A bench whose robot app names this tip box in WELCOME."""
    bench = Bench(**settings)
    bench.welcome = dataclasses.replace(bench.welcome, human_tip_box=box)
    return bench


def tip(bench: Bench, sent: HumanSetpoint) -> np.ndarray:
    return bench.solver.fk(sent.q)[:3, 3]


def test_the_tip_stops_at_a_wall_of_the_tip_box_and_slides_along_it():
    assert TIP_BOX_MARGIN_M == 0.01 and WALL_NOTE_M == 0.01
    bench = boxed()
    first = bench.engage()
    top = TIP_BOX[1][2] - TIP_BOX_MARGIN_M  # 4 cm above the tip
    assert top - HOME_TIP[2] == pytest.approx(0.04)
    # The hand rises 30 cm and goes 5 cm to the side, at once.
    bench.hand = bench.hand + np.array([0.0, 0.05, 0.30])
    sent = [first, *bench.drive(60)]
    assert all(m.engaged for m in sent)
    tips = np.array([tip(bench, m) for m in sent])
    # The tip went up to 1 cm under the wall and no further, and on to the
    # side: 2 mm a pass, the hand speed limit, in all.
    assert tips[:, 2].max() == pytest.approx(top, abs=1e-9)
    np.testing.assert_allclose(tips[-1], HOME_TIP + [0.0, 0.05, 0.04], atol=1e-9)
    slack = bench.settings.ik.tol  # a step this near its end is taken whole
    assert np.linalg.norm(np.diff(tips, axis=0), axis=1).max() <= 0.002 + slack
    # The hand beyond the wall is not ahead of the arm: the arm is where the
    # hand's point on the wall is, and the note names the wall.
    assert bench.logic.ahead_m == pytest.approx(0.0, abs=1e-9)
    assert bench.logic.walls == ("top",)
    said = [m.note for m in sent]
    changes = [said[0]] + [b for a, b in zip(said, said[1:]) if a != b]
    # The way to the hand's point on the wall counts as ahead, 6.4 cm at first;
    # the wall is named once the tip is within a centimetre of it.
    assert changes == [
        "",
        "hand 5 cm ahead: the arm is catching up",
        "",
        "at the workspace limit: top",
    ]
    # Two walls at once, in the order of the axes; back inside, none.
    bench.hand = bench.hand + np.array([0.0, 0.40, 0.0])
    [*_, corner] = bench.drive(160)
    assert corner.note == "at the workspace limit: left, top"
    np.testing.assert_allclose(tip(bench, corner), [0.2, 0.29, top], atol=1e-9)
    bench.hand = bench.hand + np.array([0.0, -0.30, -0.30])
    [*_, inside] = bench.drive(200)
    assert inside.note == "" and bench.logic.walls == ()
    np.testing.assert_allclose(
        tip(bench, inside), HOME_TIP + [0.0, 0.15, 0.0], atol=1e-9
    )
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert not any(bench.logic.counts()[key] for key in ended)


def test_a_hand_beyond_a_wall_is_believed_up_to_the_bound_from_the_tip_and_no_further():
    # The hand is believed up to ik.max_ahead_m from the gripper tip, in all,
    # beyond a wall as inside the box. "Ahead" is the way from the tip to the
    # hand's point held inside the box, which is never longer than that.
    bench = boxed()
    bench.engage()
    assert bench.settings.ik.max_ahead_m == 0.5
    top = TIP_BOX[1][2] - TIP_BOX_MARGIN_M
    # 25 cm to the side and 38 cm beyond the top wall: 49 cm from the tip.
    bench.hand = bench.hand + np.array([0.0, 0.25, 0.04 + 0.38])
    [sent] = bench.drive(1)
    assert sent.engaged and bench.logic.ahead_m == pytest.approx(
        float(np.hypot(0.25, 0.04)), abs=1e-3
    )
    # 4 cm further beyond the wall: 52 cm from the tip, of which only 25 cm
    # are ahead of the arm.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.04])
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    why = "the hand is 52 cm from the gripper tip, over the 50 cm believed"
    assert ended.note == AGAIN + why and len(AGAIN + why) <= NOTE_MAX_CHARS
    assert bench.logic.counts()["hand_ahead"] == 1
    assert tip(bench, sent)[2] < top

    # Inside the box, and without one, it is the same bound.
    for bench in (boxed([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), Bench()):
        bench.engage()
        bench.hand = bench.hand + np.array([0.0, 0.49, 0.0])
        assert setpoint(bench.step(squeeze=1.0)).engaged
        bench.hand = bench.hand + np.array([0.0, 0.03, 0.0])
        ended = setpoint(bench.step(squeeze=1.0))
        assert ended.note == AGAIN + why


def test_a_tip_outside_the_tip_box_at_a_squeeze_may_only_come_toward_it():
    # The checker's own rule for a tip that is outside already, kept axis by
    # axis: under the box here, 9.5 cm, and inside it on the other two axes.
    box = [[0.1, -0.3, 0.19], [0.5, 0.3, 0.5]]
    bench = boxed(box)
    first = bench.engage()
    np.testing.assert_allclose(tip(bench, first), HOME_TIP, atol=1e-12)
    # A hand that goes down and forward: the tip goes forward, not down.
    bench.hand = bench.hand + np.array([0.04, 0.0, -0.06])
    [*_, low] = bench.drive(40)
    np.testing.assert_allclose(tip(bench, low), HOME_TIP + [0.04, 0.0, 0.0], atol=1e-9)
    assert low.note == "at the workspace limit: bottom"
    # A hand that comes up: the tip comes up, toward the box and into it.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.06 + 0.05])
    [*_, higher] = bench.drive(40)
    np.testing.assert_allclose(
        tip(bench, higher), HOME_TIP + [0.04, 0.0, 0.05], atol=1e-9
    )
    assert higher.note == ""
    # From there it may not go back down: what it has come in stays.
    bench.hand = bench.hand + np.array([0.0, 0.0, -0.05])
    [*_, kept] = bench.drive(40)
    np.testing.assert_allclose(tip(bench, kept), tip(bench, higher), atol=1e-9)
    assert kept.note == "at the workspace limit: bottom"
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.25])
    [*_, inside] = bench.drive(120)
    assert tip(bench, inside)[2] == pytest.approx(HOME_TIP[2] + 0.25, abs=1e-9)
    assert not any(bench.logic.counts()[key] for key in ("hand_ahead", "jump"))


@pytest.mark.parametrize(
    "named, corners",
    [
        (None, None),
        (TIP_BOX, ([0.11, -0.29, 0.01], [0.49, 0.29, 0.145])),
        # An axis narrower than twice the margin is held at its middle.
        (
            [[0.0, 0.0, 0.0], [1.0, 0.015, 1.0]],
            ([0.01, 0.0075, 0.01], [0.99, 0.0075, 0.99]),
        ),
        ([[0.1, -0.3, 0.0]], None),
        ([[0.1, -0.3], [0.5, 0.3]], None),
        ([[0.1, -0.3, 0.0], [0.5, 0.3, float("nan")]], None),
        ([[0.1, -0.3, 0.0], [0.5, 0.3, float("inf")]], None),
        ([[0.6, -0.3, 0.0], [0.5, 0.3, 0.2]], None),  # the corners the wrong way
        ([["a", "b", "c"], [0.5, 0.3, 0.2]], None),
        ("box", None),
    ],
)
def test_the_tip_box_is_the_one_welcome_names_less_the_margin(named, corners):
    named_in = dataclasses.replace(welcome(), human_tip_box=named)
    logic = Bench().logic
    box = logic.tip_box(named_in)
    if corners is None:
        assert box is None
        # No box: nothing is bounded, as under a robot app that names none.
        bench = boxed(named)
        bench.engage()
        bench.hand = bench.hand + np.array([0.0, 0.0, 0.3])
        [*_, far] = bench.drive(160)
        assert tip(bench, far)[2] == pytest.approx(HOME_TIP[2] + 0.3, abs=1e-9)
    else:
        assert box[0] == pytest.approx(corners[0]) and box[1] == pytest.approx(
            corners[1]
        )
    assert logic.tip_box(None) is None


@pytest.mark.parametrize(
    "past, tip_z, named, walls",
    [
        (0.011, 0.145, (), ("top",)),
        (0.010, 0.145, (), ()),  # not yet 1 cm beyond
        (0.001, 0.145, ("top",), ("top",)),  # named: until the hand is back inside
        (0.0, 0.145, ("top",), ()),
        (0.2, 0.134, (), ()),  # the tip is not at the wall yet
        (0.2, 0.136, (), ("top",)),
    ],
)
def test_a_wall_is_named_from_a_centimetre_beyond_it_until_the_hand_is_back(
    past, tip_z, named, walls
):
    low, high = np.array([0.11, -0.29, 0.01]), np.array([0.49, 0.29, 0.145])
    at = np.array([0.3, 0.0, tip_z])
    beyond = np.array([0.0, 0.0, past])
    assert logic_module._walls_at(beyond, at, low, high, named, "bottom") == walls
    assert logic_module.WALL_NAMES == (
        ("back", "front"),
        ("right", "left"),
        ("bottom", "top"),
    )
    under = np.array([-0.02, -0.02, -0.02])
    every = logic_module._walls_at(under, low, low, high, (), "bottom")
    assert every == ("back", "right", "bottom")
    # The lowest wall has the name it is given: the table's, where it is the
    # height the robot app names for a low gripper tip.
    assert logic_module.WALL_TABLE == "table"
    every = logic_module._walls_at(under, low, low, high, (), "table")
    assert every == ("back", "right", "table")
    assert NOTE_AT_LIMIT.format(walls="front") == "at the workspace limit: front"


def test_the_notes_of_an_engagement_stand_together_as_far_as_a_note_holds_them():
    wall = NOTE_AT_LIMIT.format(walls="front, left, top")
    ahead = NOTE_CATCHING_UP.format(cm=40)
    joined = logic_module._joined
    assert joined([wall]) == wall and joined([ahead]) == ahead
    assert joined([wall, ahead]) == f"{wall}; {ahead}"
    assert joined([NOTE_THOR_BEHIND, ahead]) == f"{NOTE_THOR_BEHIND}; {ahead}"
    # All three are more than a note holds, with the shortest wall too: the
    # wall and the slow robot app are said, how far the hand is ahead is not.
    short = NOTE_AT_LIMIT.format(walls="top")
    for first in (wall, short):
        assert len(f"{first}; {NOTE_THOR_BEHIND}; {ahead}") > NOTE_MAX_CHARS
        said = joined([first, NOTE_THOR_BEHIND, ahead])
        assert said == f"{first}; {NOTE_THOR_BEHIND}"


def test_a_hand_pose_the_rate_limiter_rejects_ends_the_engagement():
    bench = Bench()
    bench.engage()
    assert setpoint(bench.step(squeeze=1.0, limiter="clamped")).engaged
    solves = len(bench.solver.solves)
    ended = setpoint(bench.step(squeeze=1.0, limiter="rejected"))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the hand moved too fast for the rate limiter"
    assert not bench.logic.armed and len(bench.solver.solves) == solves
    assert bench.logic.counts()["limiter"] == 1


def test_a_clutch_that_lets_go_under_a_held_squeeze_needs_a_new_squeeze():
    bench = Bench()
    bench.engage()
    ended = setpoint(bench.step(squeeze=1.0, clutch=False))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the clutch let go of the hand"
    assert not bench.logic.armed
    assert bench.logic.counts()["clutch"] == 1


def test_a_press_of_b_ends_the_engagement_on_the_same_pass():
    bench = Bench()
    bench.engage()
    pressed = bench.step(squeeze=1.0, b=True)
    assert pressed[0] == STOP_B
    ended = setpoint(pressed)
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "STOP by right B"
    assert not bench.logic.armed

    # B stays down and the ROBOT_STATE does not show the STOP yet.
    held = bench.run(10, b=True)
    assert none_engaged(held) and not bench.logic.armed
    held = bench.run(100, squeeze=1.0, b=True)
    assert of(held, OwnerRequest) == [] and none_engaged(held)
    assert bench.logic.counts()["stop"] == 1


def test_no_engage_after_a_press_of_b_until_a_robot_state_shows_its_stop():
    def tapped() -> Bench:
        """B went down for one pass while the arm followed the hand."""
        bench = Bench()
        bench.engage()
        assert bench.step(squeeze=1.0, b=True)[0] == STOP_B
        return bench

    # B is up again and the ROBOT_STATE still shows the person as the owner:
    # a released grip arms nothing and a new squeeze moves nothing.
    bench = tapped()
    released = setpoint(bench.step())
    assert released.grip_released and not bench.logic.armed
    assert released.note == AGAIN + "STOP by right B"
    assert none_engaged(bench.run(10, squeeze=1.0))
    assert bench.solver.solves == []  # nothing was solved for

    # The usual end: the ROBOT_STATE shows the STOP, nobody owns the arm.
    bench.state = nobody(2, takeover_open=True)
    assert none_engaged(bench.run(5)) and not bench.logic.armed

    # A ROBOT_STATE with a higher stop number ends the wait for the STOP at
    # once; the next one shows the arm at rest.
    bench = tapped()
    bench.state = robot_state(owner_number=4, stop_number=2)
    bench.run(2)
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged

    # A STOP that never shows is waited for 1 s from the press, not longer.
    bench = tapped()
    bench.halted()  # the STOP ended the drive; its numbers never show
    waited = bench.run(49)  # until 0.98 s after the press
    assert none_engaged(waited) and not bench.logic.armed
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged

    # The hand-back waits the same way, so that it quotes the STOP's numbers.
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    bench.run(49, a=True)
    assert bench.step(a=True, b=True)[0] == STOP_B
    assert of(bench.run(10, a=True), OwnerRequest) == []  # B is up again
    bench.state = nobody(2, takeover_open=True, motion="rest_hold")
    [request] = of(bench.step(a=True), OwnerRequest)
    assert (request.request, request.stop_number) == ("handback", 2)


def test_the_stop_of_a_press_goes_out_on_a_pass_that_fails_afterwards():
    def broken(q: Sequence[float]) -> np.ndarray:
        raise ValueError("no arm model")

    bench = Bench()
    bench.step()
    assert bench.logic.armed
    bench.solver.fk = broken  # the home cannot be set
    assert bench.step(b=True) == [STOP_B]
    assert not bench.logic.armed and bench.logic.counts()["b_pressed"] == 1
    # A pass with no link message of a button to send fails as it did.
    with pytest.raises(ValueError, match="no arm model"):
        bench.step(b=True)

    # A pass that fails while the arm follows the hand ends the engagement:
    # the held squeeze moves nothing afterwards.
    def raising(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
        raise RuntimeError("the solver broke")

    bench = Bench()
    bench.engage()
    bench.solver.answer = raising
    with pytest.raises(RuntimeError, match="the solver broke"):
        bench.step(squeeze=1.0)
    bench.solver.answer = None
    assert not bench.logic.armed
    solves = len(bench.solver.solves)
    held = bench.run(20, squeeze=1.0)
    assert none_engaged(held) and len(bench.solver.solves) == solves
    assert setpoint(held[-1:]).note == NOTE_RELEASE_FIRST
    assert bench.logic.counts()["clutch"] == 0
    bench.halted()
    bench.run(2)
    assert setpoint(bench.step(squeeze=1.0)).engaged


def test_the_stop_of_right_b_goes_out_on_a_pass_whose_request_fails(caplog):
    name = logic_module._log.name
    caplog.set_level("INFO", logger=name)

    def broken(p: Any) -> OwnerRequest | None:
        raise RuntimeError("no request")

    # The arm follows the hand. Right B goes down on a pass on which the
    # request of a held button cannot be built: the code right behind the STOP.
    bench = Bench()
    bench.engage()
    whole = bench.logic._request
    bench.logic._request = broken
    # The STOP goes out, alone: no HUMAN_SETPOINT of a pass that failed.
    assert bench.step(squeeze=1.0, b=True) == [STOP_B]
    assert not bench.logic.armed and bench.logic.counts()["b_pressed"] == 1
    # The error is logged as that of any pass that fails behind its buttons.
    bench.logic.log_pass()
    lines = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == name]
    assert lines[-2:] == [
        ("INFO", "right B is down: STOP sent"),
        ("ERROR", "the pass failed after its buttons were read"),
    ]
    assert caplog.records[-1].exc_info[0] is RuntimeError

    # B stays down: its stretch has had its STOP. A pass that fails with no
    # link message of a button to send fails as it did.
    with pytest.raises(RuntimeError, match="no request"):
        bench.step(squeeze=1.0, b=True)
    bench.logic._request = whole
    held = bench.run(20, squeeze=1.0, b=True)
    assert of(held, Stop) == [] and none_engaged(held)
    assert not bench.logic.armed

    # Seen up and pressed again, B is a STOP again, on such a pass too.
    assert of(bench.step(), Stop) == []
    bench.logic._request = broken
    assert bench.step(b=True) == [STOP_B]
    assert bench.logic.counts()["b_pressed"] == 2

    # Before any WELCOME or ROBOT_STATE, as the first data arrive with B down.
    early = Bench()
    early.welcome, early.state, early.state_age_s = None, None, None
    early.logic._request = broken
    assert early.step(b=True) == [STOP_B]


def test_an_error_before_the_buttons_are_read_sends_nothing_and_stays_armed():
    def broken(outputs: Any) -> GraphStep:
        raise ValueError("no graph outputs")

    # Right B goes down on the pass that fails: no STOP on it, one on the next.
    bench = Bench()
    bench.step()
    read = bench.graph.read
    bench.graph.read = broken
    with pytest.raises(ValueError, match="no graph outputs"):
        bench.step(b=True)
    bench.graph.read = read
    assert bench.logic.counts()["b_pressed"] == 0
    assert bench.step(b=True)[0] == STOP_B
    assert bench.logic.counts()["b_pressed"] == 1

    # The arm follows the hand; the pass that fails leaves the logic armed,
    # and the held squeeze is followed on the next pass with no new squeeze.
    bench = Bench()
    bench.engage()
    read = bench.graph.read
    bench.graph.read = broken
    with pytest.raises(ValueError, match="no graph outputs"):
        bench.step(squeeze=1.0)
    bench.graph.read = read
    assert bench.logic.armed
    assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.counts()["engagements"] == 1


def test_a_released_grip_ends_the_engagement_and_the_next_squeeze_engages():
    bench = Bench()
    bench.engage()
    bench.run(3, squeeze=1.0)
    released = setpoint(bench.step())
    assert (released.engaged, released.q, released.grip_released) == (
        False,
        None,
        True,
    )
    assert released.note == NOTE_SQUEEZE
    # The next squeeze counts once a ROBOT_STATE has shown that drive as
    # ended and the next one the arm at rest: here on its third pass.
    assert not bench.logic.armed
    bench.halted()
    waiting = setpoint(bench.step(squeeze=1.0))
    assert not waiting.engaged and waiting.note == NOTE_STOPPING
    assert not bench.logic.armed
    waiting = setpoint(bench.step(squeeze=1.0))
    assert not waiting.engaged and waiting.note == NOTE_STOPPING
    assert bench.logic.armed
    assert setpoint(bench.step(squeeze=1.0)).engaged
    counts = bench.logic.counts()
    assert (counts["engagements"], counts["released"]) == (2, 1)


# At rest. After an engagement the arm goes on for a moment, and the first
# ROBOT_STATE that shows the drive as ended still has readings taken while it
# moved. A new squeeze counts once two readings of every joint, both begun
# after the Thor's last setpoint, agree: the home is then where the arm
# stands.

SENT_AT = 450.0  # when the Thor's last setpoint went out, on its clock


def released_at_speed() -> tuple[Bench, list[float], int]:
    """An engagement ended by a released grip: the bench, the arm's pose, the seq.

    From here on the test gives each ROBOT_STATE its own readings.
    """
    bench = Bench()
    bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.2])
    at = bench.drive(6)[-1].q
    released = setpoint(bench.step())
    bench.reads = False
    return bench, at, released.seq


def read(
    q: Sequence[float], began: float, seq: int, sent_at: float | None
) -> RobotState:
    """A ROBOT_STATE with no drive, every joint read anew at `began`."""
    state = read_at(robot_state(q=q, target_q=q, seq=seq), began)
    return dataclasses.replace(state, target_t=sent_at)


def test_the_arm_is_at_rest_once_two_readings_of_every_joint_after_the_last_setpoint_agree():
    assert logic_module.REST_TOL_RAD == 0.002
    bench, at, end = released_at_speed()
    homes = len(bench.graph.homes)

    def armed(q: Sequence[float], began: float) -> bool:
        bench.state = read(q, began, end + 1, SENT_AT)
        waiting = setpoint(bench.step(squeeze=1.0))
        assert not waiting.engaged and waiting.note == NOTE_STOPPING
        return bench.logic.armed

    def off(by: float, joint: int | None = None) -> list[float]:
        """The pose `by` short of where the arm comes to rest, on one joint or all."""
        return [v - by * (joint in (None, i)) for i, v in enumerate(at)]

    # The drive is shown as ended, with readings that began before the Thor's
    # last setpoint went out: the arm has moved since, however well they agree.
    assert not armed(off(0.03), SENT_AT - 0.02)
    assert not armed(off(0.03), SENT_AT - 0.01)
    # The first readings since that setpoint: nothing to compare them with.
    assert not armed(off(0.010), SENT_AT + 0.11)
    # The same readings in a later ROBOT_STATE are no new ones.
    assert not armed(off(0.010), SENT_AT + 0.11)
    # New ones, 0.004 rad on: the arm still moves.
    assert not armed(off(0.006), SENT_AT + 0.17)
    # One joint alone that still moves holds the squeeze back.
    assert not armed(off(0.003, joint=2), SENT_AT + 0.23)
    assert len(bench.graph.homes) == homes  # no home from any of these
    # Every joint within 0.002 rad of its reading before: at rest. The home
    # is the arm's pose of this state, and the squeeze counts from it on.
    assert armed(off(0.0015, joint=2), SENT_AT + 0.29)
    np.testing.assert_allclose(
        bench.graph.homes[-1], bench.solver.fk(off(0.0015, joint=2))
    )
    first = setpoint(bench.step(squeeze=1.0))
    assert first.engaged and first.q == pytest.approx(off(0.0015, joint=2))

    # A robot app that names no send time of a setpoint has sent none: any
    # two readings of a joint count.
    bench, at, end = released_at_speed()
    bench.state = read(at, 400.0, end + 1, None)
    bench.step(squeeze=1.0)
    assert not bench.logic.armed
    bench.state = read(at, 400.06, end + 1, None)
    bench.step(squeeze=1.0)
    assert bench.logic.armed


def test_a_squeeze_waits_half_a_second_at_most_for_the_arm_to_be_shown_at_rest():
    assert logic_module.REST_WAIT_S == 0.5
    # A robot app whose readings never show the arm at rest: here the same
    # ones in every ROBOT_STATE. The squeeze is taken on the newest state
    # half a second after the engagement ended, so the person is not left
    # without a squeeze; the robot app's own step limit stays the backstop.
    bench, at, end = released_at_speed()
    bench.state = read(at, SENT_AT + 0.11, end + 1, SENT_AT)
    waited = 0
    while not bench.logic.armed:
        waiting = setpoint(bench.step(squeeze=1.0))
        assert not waiting.engaged and waiting.note == NOTE_STOPPING
        waited += 1
        assert waited < 40
    assert waited == round(logic_module.REST_WAIT_S / PASS_S) + 1
    first = setpoint(bench.step(squeeze=1.0))
    assert first.engaged and first.q == pytest.approx(at)


def test_a_robot_app_that_goes_on_reporting_its_drive_is_named_after_a_second():
    assert logic_module.DRIVE_ON_NOTE_S == 1.0
    assert NOTE_DRIVE_ON == (
        "wait: the robot app still reports the last drive; if this stays, type stop"
    )
    assert len(NOTE_DRIVE_ON) <= NOTE_MAX_CHARS
    bench = Bench()
    bench.engage()
    bench.run(3, squeeze=1.0)
    assert setpoint(bench.step()).note == NOTE_SQUEEZE
    # The ROBOT_STATE goes on showing the drive. For a second the notes are
    # the usual ones; then they say what is waited for, grip squeezed or not.
    said = [setpoint(bench.step(squeeze=index % 2)).note for index in range(49)]
    assert set(said[0::2]) == {NOTE_SQUEEZE} and set(said[1::2]) == {NOTE_STOPPING}
    setpoint(bench.step())
    later = [setpoint(bench.step(squeeze=index % 2)).note for index in range(200)]
    assert set(later) == {NOTE_DRIVE_ON} and not bench.logic.armed
    assert len(bench.solver.solves) == 3  # nothing was solved for meanwhile
    # The drive ends after all, long after the half second the arm is given
    # to come to rest: the squeeze counts on the state that shows it.
    bench.halted()
    assert setpoint(bench.step(squeeze=1.0)).note == NOTE_STOPPING
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged


# The table. The robot app names the height its checker keeps a person's
# gripper tip above: its table clearance. The pose solved for stays 0.5 cm
# above that height and moves freely above it: the tip stops there and slides
# along it while the hand pushes down. A squeeze with the tip close to the
# table starts as any other.


def floored(floor: float, box: Any = TIP_BOX, **settings: Any) -> Bench:
    """A bench whose robot app names this height for its table clearance."""
    bench = boxed(box, **settings)
    bench.welcome = dataclasses.replace(bench.welcome, human_tip_floor=floor)
    return bench


def test_the_tip_stops_half_a_centimetre_above_the_named_height_and_slides_along_it():
    assert TIP_FLOOR_MARGIN_M == logic_module.TIP_FLOOR_MARGIN_M == 0.005
    bench = floored(0.05)
    first = bench.engage()
    stop = 0.05 + TIP_FLOOR_MARGIN_M  # 5 cm under the tip
    assert tip(bench, first)[2] - stop == pytest.approx(0.05)
    # The hand goes 20 cm down and 5 cm to the side, at once.
    bench.hand = bench.hand + np.array([0.0, 0.05, -0.20])
    sent = bench.drive(80)
    assert all(m.engaged for m in sent)
    tips = np.array([tip(bench, m) for m in sent])
    assert tips[:, 2].min() == pytest.approx(stop, abs=1e-9)
    np.testing.assert_allclose(tips[-1], HOME_TIP + [0.0, 0.05, -0.05], atol=1e-9)
    # The note names the table, not the bottom of the box.
    assert sent[-1].note == "at the workspace limit: table"
    assert bench.logic.walls == ("table",)
    # The hand pushes further down and goes on to the side: the tip slides
    # along the stop at the hand speed limit, 2 mm a pass.
    bench.hand = bench.hand + np.array([0.0, 0.04, -0.02])
    slid = np.array([tip(bench, m) for m in bench.drive(30)])
    np.testing.assert_allclose(slid[:, 2], stop, atol=1e-9)
    assert np.diff(slid[:10, 1]) == pytest.approx([0.002] * 9)
    np.testing.assert_allclose(slid[-1], HOME_TIP + [0.0, 0.09, -0.05], atol=1e-9)
    # And it rises from the stop as from anywhere.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.24])
    [*_, risen] = bench.drive(40)
    np.testing.assert_allclose(
        tip(bench, risen), HOME_TIP + [0.0, 0.09, 0.02], atol=1e-9
    )
    assert risen.note == "" and bench.logic.walls == ()
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert not any(bench.logic.counts()[key] for key in ended)


def test_a_level_hand_close_to_the_table_moves_the_tip_sideways():
    # The tip is at the stop, 0.5 cm above the named height, at the squeeze.
    bench = floored(0.1)
    first = bench.engage()
    assert tip(bench, first)[2] == pytest.approx(0.1 + TIP_FLOOR_MARGIN_M)
    bench.hand = bench.hand + np.array([0.03, 0.04, 0.0])
    # A ROBOT_STATE that still names a waiting drive is read as any other.
    state = robot_state(q=first.q, target_q=first.q, seq=first.seq, waiting=True)
    bench.state = following(state)
    moved = setpoint(bench.step(squeeze=1.0))
    way = tip(bench, moved) - tip(bench, first)
    np.testing.assert_allclose(way, [0.0012, 0.0016, 0.0], atol=1e-9)
    sent = [moved, *bench.drive(40)]
    assert all(m.engaged for m in sent)
    tips = np.array([tip(bench, m) for m in [first, *sent]])
    # From the first pass the Thor follows, the tip goes the hand's way at
    # the hand speed limit, level, and stands where the hand is.
    np.testing.assert_allclose(tips[:, 2], tips[0, 2], atol=1e-9)
    steps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    assert steps[:25] == pytest.approx([0.002] * 25) and steps[25:].max() < 1e-9
    np.testing.assert_allclose(tips[-1], HOME_TIP + [0.03, 0.04, 0.0], atol=1e-9)
    assert not any("table" in m.note or "raise" in m.note for m in sent)
    assert bench.logic.walls == ()


def test_a_squeeze_on_an_arm_that_sags_under_its_last_joint_target_follows_at_once():
    # The stand-in arm is raised: its tip is 20 cm up, 4.5 cm above the stop.
    bench = floored(0.15, box=None)
    raised = (0.0, 0.3, 0.5, 0.0, 0.0, 0.0)
    bench.state = robot_state(q=raised)
    bench.engage()
    home = bench.solver.fk(raised)[:3, 3]
    bench.hand = bench.hand + np.array([0.0, 0.0, -0.20])
    [*_, down] = bench.drive(40)
    stop = 0.15 + TIP_FLOOR_MARGIN_M
    assert tip(bench, down)[2] == pytest.approx(stop, abs=1e-9)
    bench.step()  # the grip is released
    # The arm holds 3 mm under that joint target: under the stop, and above
    # the named height.
    sagged = [*down.q[:2], down.q[2] - 0.003 / FakeSolver.M_PER_RAD, *down.q[3:]]
    bench.state = robot_state(q=sagged)
    first = bench.engage()
    assert first.q == pytest.approx(sagged)
    assert tip(bench, first)[2] == pytest.approx(stop - 0.003)
    # A level hand that goes to the side: the first pass the Thor follows
    # moves the joint target that way, and up to the stop.
    bench.hand = bench.hand + np.array([0.0, 0.03, 0.0])
    sent = bench.drive(25)
    assert all(m.engaged for m in sent) and sent[0].q != first.q
    tips = np.array([tip(bench, m) for m in [first, *sent]])
    assert np.all(np.diff(tips[:, 1]) >= -1e-12) and tips[1, 1] > tips[0, 1] + 1e-3
    assert np.all(np.diff(tips[:, 2]) >= -1e-12)
    np.testing.assert_allclose(tips[-1], [home[0], home[1] + 0.03, stop], atol=1e-9)
    assert np.linalg.norm(np.diff(tips, axis=0), axis=1).max() <= 0.002 + 1e-4
    assert not any("raise" in m.note for m in sent)
    counts = bench.logic.counts()
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert counts["engagements"] == 2 and not any(counts[key] for key in ended)


def test_a_tip_under_the_named_height_rises_straight_up_and_then_follows():
    # The stand-in arm's tip is 10.5 cm up at the squeeze; the robot app
    # names 12 cm, so the stop is at 12.5 cm.
    bench = floored(0.12)
    first = bench.engage()
    assert tip(bench, first)[2] == pytest.approx(0.105)
    # The squeeze's joint target is the arm's pose, under the height.
    np.testing.assert_allclose(tip(bench, first), HOME_TIP, atol=1e-12)
    # A level hand that goes to the side.
    bench.hand = bench.hand + np.array([0.0, 0.04, 0.0])
    sent = bench.drive(40)
    assert all(m.engaged for m in sent)
    tips = np.array([tip(bench, m) for m in [first, *sent]])
    # Under the named height the tip only rises, straight up, at the hand
    # speed limit: the checker takes a step there only when the tip rises.
    under = tips[:-1, 2] < 0.12
    assert int(under.sum()) == 8 and bool(np.all(under[:8]))
    np.testing.assert_allclose(tips[1:9, :2], [HOME_TIP[:2]] * 8, atol=1e-12)
    assert np.diff(tips[:9, 2]) == pytest.approx([0.002] * 8)
    # From there it goes on to the stop and to the side.
    np.testing.assert_allclose(tips[-1], HOME_TIP + [0.0, 0.04, 0.02], atol=1e-9)
    assert np.all(np.diff(tips[:, 2]) >= -1e-12)
    # The hand is under the stop: the note names the table.
    assert sent[-1].note == "at the workspace limit: table"
    ended = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    assert not any(bench.logic.counts()[key] for key in ended)


@pytest.mark.parametrize(
    "box, floor, low_z, table",
    [
        (TIP_BOX, None, 0.01, False),  # no height named: the box's bottom
        (TIP_BOX, 0.02, 0.025, True),
        (TIP_BOX, 0.006, 0.011, True),
        (TIP_BOX, 0.004, 0.01, False),  # the box's bottom is the higher one
        (TIP_BOX, -0.05, 0.01, False),
        (None, 0.02, 0.025, True),  # no box named: the table alone
        (TIP_BOX, 0.5, 0.145, True),  # above the box: its top, less the margin
        (TIP_BOX, float("nan"), 0.01, False),
        (TIP_BOX, "low", 0.01, False),
        (TIP_BOX, True, 0.01, False),
    ],
)
def test_the_lowest_wall_is_the_higher_of_the_boxs_bottom_and_the_stop_above_the_table(
    box, floor, low_z, table
):
    named = dataclasses.replace(welcome(), human_tip_box=box, human_tip_floor=floor)
    low, high, is_table = logic_module._named_box(named)
    assert low[2] == pytest.approx(low_z) and is_table is table
    assert (high[2] == pytest.approx(0.145)) == (box is not None)
    if box is None:
        assert np.all(np.isinf(low[:2])) and np.all(np.isinf(high))
    # Neither named: nothing is bounded.
    nothing = dataclasses.replace(named, human_tip_box=None, human_tip_floor=None)
    assert logic_module._named_box(nothing) is None


# The joint box and the reach. The robot app names the joint box its checker
# holds a person's joint target in. The joint target stops 0.01 rad inside
# it and the engagement goes on, where the checker's refusal ended the drive.

JOINT_BOX = [[-1.0, -0.05, -0.05, -1.0, -1.0, -1.0], [1.0, 1.0, 0.3, 1.0, 1.0, 1.0]]


def test_the_joint_target_stops_inside_the_joint_box_and_follows_again():
    assert logic_module.JOINT_BOX_MARGIN_RAD == 0.01
    bench = Bench()
    bench.welcome = dataclasses.replace(bench.welcome, human_joint_box=JOINT_BOX)
    first = bench.engage()
    stop = JOINT_BOX[1][2] - logic_module.JOINT_BOX_MARGIN_RAD  # 0.29 rad
    # The hand goes 10 cm up, which the third joint alone makes on the
    # stand-in arm: 0.5 rad, past its bound of 0.3 rad.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.10])
    sent = bench.drive(60)
    assert all(m.engaged for m in sent)
    third = [m.q[2] for m in sent]
    assert max(third) == pytest.approx(stop) and third[-1] == pytest.approx(stop)
    # The last step went as far as the box allows, not a whole pass's.
    assert np.diff([first.q[2], *third]).max() == pytest.approx(0.01)
    said = NOTE_AT_JOINT_LIMIT.format(joint=3)
    assert said == "at a joint limit: joint 3"
    assert said in sent[-1].note and bench.logic.stand == said
    assert bench.logic.counts()["no_solution"] == 0
    # The hand goes on to the side, still 4.7 cm above what the joint
    # allows: every step toward it asks that joint for more. The joint
    # target goes on along the box, by the solver's nearest pose inside it:
    # the second joint takes the tip to the side, to under the hand within
    # 3 mm, and the third stays at its stop.
    bench.hand = bench.hand + np.array([0.0, 0.05, 0.0])
    along = bench.drive(160)
    assert all(m.engaged and said in m.note for m in along)
    assert [m.q[2] for m in along] == pytest.approx([stop] * 160)
    second = [first.q[1], *(m.q[1] for m in along)]
    assert np.all(np.diff(second) >= 0.0) and np.diff(second).max() <= 0.01 + 1e-9
    assert second[-1] == pytest.approx(first.q[1] + 0.25, abs=0.015)
    assert along[-1].q == along[-20].q and len(bench.solver.nearests) > 20
    # The hand comes back down: the arm follows again, to the side as well,
    # and the note is gone.
    bench.hand = bench.hand + np.array([0.0, 0.0, -0.10])
    back = bench.drive(60)
    assert back[-1].q[2] == pytest.approx(first.q[2], abs=1e-9)
    assert back[-1].q[1] == pytest.approx(first.q[1] + 0.25)
    assert said not in back[-1].note and bench.logic.stand == ""

    # A joint that is outside the box at the squeeze may come back toward it
    # and goes no further out: the checker's own rule.
    bench = Bench()
    bench.welcome = dataclasses.replace(bench.welcome, human_joint_box=JOINT_BOX)
    outside = (0.0, 0.025, 0.4, 0.0, 0.0, 0.0)
    bench.state = robot_state(q=outside)
    first = bench.engage()
    assert first.q == pytest.approx(outside)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.02])
    assert bench.drive(10)[-1].q[2] == pytest.approx(0.4)
    bench.hand = bench.hand + np.array([0.0, 0.0, -0.06])
    inward = bench.drive(40)
    assert inward[-1].q[2] == pytest.approx(0.2)
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.10])
    assert bench.drive(60)[-1].q[2] == pytest.approx(stop)


@pytest.mark.parametrize(
    "named",
    [
        None,
        "box",
        [[0.0] * 6],
        [[0.0] * 5, [1.0] * 5],
        [[1.0] * 6, [0.0] * 6],
        [[0.0] * 6, [float("inf")] * 6],
        [[0.0] * 6, [float("nan")] * 6],
    ],
    ids=["none", "text", "one corner", "five joints", "min above max", "inf", "nan"],
)
def test_a_joint_box_that_cannot_be_used_bounds_nothing(named):
    bench = Bench()
    bench.welcome = dataclasses.replace(bench.welcome, human_joint_box=named)
    assert not bench.logic.joint_box(bench.welcome)
    first = bench.engage()
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.10])
    assert bench.drive(60)[-1].q[2] == pytest.approx(first.q[2] + 0.5)


# The hold-back, once the next ROBOT_STATE is overdue.


def test_no_more_is_given_for_the_time_since_a_robot_state_once_the_next_is_overdue():
    assert logic_module.STATE_DUE_PERIODS == 1.5
    assert logic_module.STATE_GAPS_MIN == 3
    base = HOLD_BACK_STEPS * THOR_STEP

    def room(period_s: float, age_s: float) -> float:
        """How far ahead of the Thor's joint target the joint target comes to stand.

        The Thor's drive has made a ROBOT_STATE every period_s, the newest
        age_s ago, and its joint target has stood since.
        """
        bench, start = held_drive()
        state = following(robot_state(q=start.q, target_q=start.q, seq=start.seq))
        for index in range(5):
            made = 500.0 + index * period_s
            numbered = dataclasses.replace(state, seq=10 + index)
            sent = 1 + index * round(period_s / PASS_S)  # its loop is on time
            bench.state = timed(numbered, made, made - 0.001, sent)
            bench.state_age_s = 0.001
            bench.step(squeeze=1.0)
        bench.state_age_s = age_s
        last = [setpoint(bench.step(squeeze=1.0)).q[2] for _ in range(8)][-1]
        return last - start.q[2]

    credit = SPEED_MARGIN * HUMAN_SPEED
    # Where the joint target stood on the passes before, with ROBOT_STATEs
    # 0.001 s old: it is not taken back from there.
    stood = base + 0.001 * credit
    # A ROBOT_STATE with every setpoint: the next is overdue 0.03 s after
    # the newest arrived, and nothing more is given from then on.
    assert room(0.02, 0.029) == pytest.approx(base + 0.029 * credit)
    assert room(0.02, 0.031) == pytest.approx(stood)
    # One every third setpoint, about as an older robot app sends them:
    # 0.09 s. Until then all is as with that robot app before.
    assert room(0.06, 0.089) == pytest.approx(base + 0.089 * credit)
    assert room(0.06, 0.091) == pytest.approx(stood)


def sent_per_pass(bench: Bench, passes: int, **controls: Any) -> list[bool]:
    """For each pass, whether a HUMAN_SETPOINT went out."""
    return [bool(of(bench.step(**controls), HumanSetpoint)) for _ in range(passes)]


def test_the_setpoint_rate_is_fifty_when_owning_and_twenty_otherwise():
    bench = Bench()
    assert sum(sent_per_pass(bench, 50)) == 50  # one second

    for state in (nobody(2), robot_state(owner="agent", owner_number=6)):
        bench.state = state
        sent = sent_per_pass(bench, 50)
        assert sum(sent) == 20
        gaps = "".join("x" if one else "." for one in sent).split("x")
        assert max(len(gap) for gap in gaps) <= 2  # at most 0.06 s apart

    status = bench.status(squeeze=1.0, clutch=True)
    assert (status.q, status.engaged) == (None, False)


def test_the_setpoint_rate_is_fifty_while_b_is_down():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert sum(sent_per_pass(bench, 50, b=True)) == 50
    assert sum(sent_per_pass(bench, 50)) == 20  # B was seen up

    # With B still down, the RESULT of its request ends it too.
    bench.step(b=True)
    bench.state = nobody(9)
    held = bench.run(50, b=True)
    [request] = of(held, OwnerRequest)
    assert len(of(held, HumanSetpoint)) == 50
    assert sum(sent_per_pass(bench, 50, b=True)) == 50  # no RESULT yet
    refused = Result(request.command_id, "refused", "no fresh controller data")
    bench.step(b=True, results=[refused])
    assert sum(sent_per_pass(bench, 50, b=True)) == 20

    # A B that went down in a data gap asks for no takeover: the rate stays.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.lost()
    assert bench.step(b=True) == [STOP_B]
    assert sum(sent_per_pass(bench, 50, b=True)) == 20


def test_every_setpoint_quotes_the_newest_robot_state_and_its_clock():
    bench = Bench(source=HUMAN_SOURCE_TELEOP)
    state = robot_state(owner_number=4, stop_number=2, clock_mono_s=512.5)
    bench.welcome, bench.state = None, state
    assert bench.run(5) == []  # none before the first WELCOME
    bench.welcome, bench.state, bench.state_age_s = welcome(), None, None
    assert bench.run(5) == []  # none before the first ROBOT_STATE

    bench.state, bench.state_age_s = state, 0.01
    first = setpoint(bench.step(age=0.03))
    assert first == HumanSetpoint(
        session_id="s1",
        owner_number=4,
        stop_number=2,
        seq=1,
        source="teleop",
        stage="ready",
        sample_age_s=0.03,
        engaged=False,
        grip_released=True,
        q=None,
        jaw_opening=1.0,  # the trigger is released
        state_clock_mono_s=512.5,
        note=NOTE_SQUEEZE,
    )

    bench.state = robot_state(owner_number=5, stop_number=3, clock_mono_s=512.6)
    second = setpoint(bench.step(squeeze=1.0))
    assert (second.owner_number, second.stop_number, second.seq) == (5, 3, 2)
    assert second.state_clock_mono_s == 512.6
    assert second.engaged and second.q == pytest.approx(list(REST_INSIDE))
    assert (second.jaw_opening, second.grip_released, second.note) == (1.0, False, "")
    assert decode(HumanSetpoint.TYPE, second.to_payload()) == second

    # The source is the controller feed's, as the logic was given it.
    assert setpoint(Bench(source=HUMAN_SOURCE_SYNTHETIC).step()).source == "synthetic"


def test_a_robot_state_of_another_session_is_not_used():
    bench = Bench()
    bench.welcome = welcome("s2")  # the robot app was started again
    assert of(bench.run(10), HumanSetpoint) == []
    assert not bench.logic.armed and bench.graph.homes == []
    assert bench.step(b=True) == [STOP_B]

    bench.state = robot_state(session_id="s2")
    assert setpoint(bench.step()).session_id == "s2"


def test_grip_released_needs_fresh_data():
    bench = Bench()
    assert setpoint(bench.step()).grip_released
    assert not setpoint(bench.step(squeeze=1.0)).grip_released
    assert not setpoint(bench.step(squeeze=THRESHOLD + 0.01)).grip_released
    assert setpoint(bench.step(squeeze=THRESHOLD)).grip_released

    # The squeeze reads 0.0 on each of these passes.
    assert not setpoint(bench.step(stage="stale", age=None)).grip_released
    assert not setpoint(bench.step(stage="no_controller")).grip_released
    assert not setpoint(bench.step(present=False)).grip_released
    assert not setpoint(bench.step(age=None)).grip_released
    absent = setpoint(bench.step(outputs=False, stage="no_server", age=None))
    assert not absent.grip_released

    for age in (float("nan"), float("inf"), -0.1):
        unknown = setpoint(bench.step(age=age))
        assert not unknown.grip_released and unknown.sample_age_s is None
    assert setpoint(bench.step()).grip_released


def test_the_last_graph_step_is_kept_and_is_none_without_outputs():
    bench = Bench()
    assert bench.logic.last is None
    bench.step(squeeze=0.3, a=True)
    last = bench.logic.last
    assert last is bench.graph.steps[-1] and bench.graph.reads == 1
    assert (last.present, last.squeeze, last.a, last.b) == (True, 0.3, True, False)
    assert not last.engaged and np.array_equal(last.hand, bench.hand)

    bench.step(outputs=False, stage="no_server", age=None)
    assert bench.logic.last is None and bench.graph.reads == 1

    bench.step(squeeze=1.0)
    assert bench.logic.last is bench.graph.steps[-1] and bench.graph.reads == 2
    assert bench.logic.last.squeeze == 1.0


# Notes and counts


def test_the_notes_name_the_operators_next_step():
    bench = Bench()
    bench.state = robot_state(owner="agent")
    assert bench.status().note == NOTE_TAKE_OVER
    bench.state = nobody(2)
    assert bench.status().note == NOTE_TAKE_OVER

    # The grant, at rest, with the grip squeezed.
    at_rest = {"owner_number": 6, "stop_number": 2, "motion": "rest_hold"}
    bench.state = robot_state(needs_release=True, **at_rest)
    assert setpoint(bench.step(squeeze=1.0)).note == NOTE_RELEASE_FIRST
    assert setpoint(bench.step()).note == NOTE_RELEASE_FIRST
    bench.state = robot_state(**at_rest)  # the Thor took the released grip
    assert setpoint(bench.step()).note == NOTE_SQUEEZE
    assert setpoint(bench.step(squeeze=1.0)).note == ""  # the arm follows
    bench.state = robot_state(owner_number=6, stop_number=2)
    assert setpoint(bench.step()).note == NOTE_SQUEEZE

    # Back at rest after the arm was moved, and after a STOP there.
    bench.state = robot_state(**at_rest)
    assert setpoint(bench.step()).note == NOTE_AT_REST
    assert setpoint(bench.step()).note == NOTE_AT_REST  # shown at rest
    assert setpoint(bench.step(squeeze=1.0)).note == ""
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    assert bench.status().note == NOTE_AT_REST
    bench.state = nobody(3, takeover_open=True)
    assert bench.status().note == NOTE_TAKE_OVER

    # The hand-back was taken: the next takeover starts with the squeeze again.
    bench.state = nobody(3, motion="rest_hold")
    assert bench.status().note == NOTE_TAKE_OVER
    bench.state = robot_state(owner_number=8, stop_number=3, motion="rest_hold")
    assert setpoint(bench.step()).note == NOTE_SQUEEZE


def test_the_note_asks_for_a_released_b_while_b_is_down_without_a_press_that_counts():
    assert len(NOTE_RELEASE_B) <= NOTE_MAX_CHARS

    # B is down as the first data arrive: one STOP, and no hold to wait for.
    bench = Bench()
    bench.state = robot_state(owner="agent")
    assert bench.step(b=True)[0] == STOP_B
    assert bench.status(b=True).note == NOTE_RELEASE_B
    bench.state = nobody(2)
    assert bench.status(b=True).note == NOTE_RELEASE_B
    assert bench.status().note == NOTE_TAKE_OVER  # B is up

    # A press that counts: its hold runs under the usual note.
    assert setpoint(bench.step(b=True)).note == NOTE_TAKE_OVER
    bench.state = nobody(3)
    assert setpoint(bench.step(b=True)).note == NOTE_TAKE_OVER
    # One pass that is not fresh ends the hold, and B is still down.
    bench.step(b=True, **POSE_LOST)
    assert bench.status(b=True).note == NOTE_RELEASE_B

    # At rest with a takeover open, the hand-back is the next step.
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    assert bench.status(b=True).note == NOTE_AT_REST


def test_the_note_says_release_b_while_b_is_held_after_a_press_that_cannot_ask():
    def held_on(bench: Bench, stop_number: int, **human: Any) -> None:
        """B stays down for 3 s with the grip released; then a new press asks."""
        held = bench.run(150, b=True)
        assert of(held, (Stop, OwnerRequest)) == []
        assert notes(held) == {NOTE_RELEASE_B}
        assert bench.status().note == NOTE_TAKE_OVER  # B is up
        assert bench.step(b=True)[0] == STOP_B
        bench.state = nobody(stop_number + 1, **human)
        again = bench.run(50, b=True)
        assert len(of(again, OwnerRequest)) == 1
        assert notes(again) == {NOTE_TAKE_OVER}

    # The grip was squeezed as B went down: the press counts, and its hold
    # asks for nothing, also once the grip is released.
    bench = Bench()
    bench.state = robot_state(owner="agent", owner_number=4, stop_number=7)
    bench.step(squeeze=1.0)
    assert bench.step(b=True, squeeze=1.0)[0] == STOP_B
    assert bench.logic.counts()["b_pressed"] == 1
    bench.state = nobody(8)
    assert notes(bench.run(10, b=True, squeeze=1.0)) == {NOTE_RELEASE_B}
    held_on(bench, 8)

    # Right B as the STOP of an engagement: the grip is squeezed at its press.
    bench = Bench()
    bench.engage()
    assert bench.step(b=True, squeeze=1.0)[0] == STOP_B
    bench.state = nobody(2, takeover_open=True)
    held_on(bench, 2, takeover_open=True)

    # One squeezed pass in the middle of a hold that began with the grip
    # released.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(8)
    assert notes(bench.run(20, b=True)) == {NOTE_TAKE_OVER}
    assert setpoint(bench.step(b=True, squeeze=1.0)).note == NOTE_RELEASE_B
    held_on(bench, 8)


@pytest.mark.parametrize(
    "case",
    ["no ROBOT_STATE", "the ROBOT_STATE of the session before", "a new session"],
)
def test_the_note_says_release_b_while_no_robot_state_can_show_the_stop_of_its_press(
    case,
):
    bench = Bench()
    bench.state = nobody(7)
    bench.step()  # B is seen up: the next press counts
    session = "s1"  # of the robot app while right B is held
    if case == "no ROBOT_STATE":
        bench.state = None
    elif case == "the ROBOT_STATE of the session before":
        # The robot app is back; its first ROBOT_STATE has not arrived.
        bench.welcome, session = welcome("s2"), "s2"
    assert bench.step(b=True)[0] == STOP_B
    assert bench.logic.counts()["b_pressed"] == 1
    if case == "a new session":
        bench.welcome, session = welcome("s2"), "s2"  # while B stays down
    # The STOP quoted no ROBOT_STATE of this session, so none can show it:
    # the hold asks for nothing, 3 s long, and the note says what to do.
    bench.state = nobody(8, session_id=session)
    held = bench.run(150, b=True)
    assert of(held, (Stop, OwnerRequest)) == []
    assert len(of(held, HumanSetpoint)) == 150
    assert notes(held) == {NOTE_RELEASE_B}
    # B up, then a new press: its STOP is shown, and its hold asks.
    assert bench.status().note == NOTE_TAKE_OVER
    assert bench.step(b=True)[0] == STOP_B
    bench.state = nobody(9, session_id=session)
    again = bench.run(50, b=True)
    assert len(of(again, OwnerRequest)) == 1
    assert notes(again) == {NOTE_TAKE_OVER}


def test_a_hold_that_can_ask_keeps_the_takeover_note_until_its_result():
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    assert notes(bench.run(49, b=True)) == {NOTE_TAKE_OVER}
    due = bench.step(b=True)  # the pass that sends the request
    [request] = of(due, OwnerRequest)
    assert setpoint(due).note == NOTE_TAKE_OVER
    # B stays down while the RESULT is pending: 2 s without one, an
    # "accepted", 0.4 s more. "release right B" would be wrong here.
    pending = bench.run(100, b=True)
    accepted = [Result(request.command_id, "accepted")]
    pending += bench.step(b=True, results=accepted) + bench.run(20, b=True)
    assert of(pending, (Stop, OwnerRequest)) == []
    assert len(of(pending, HumanSetpoint)) == 121
    assert notes(pending) == {NOTE_TAKE_OVER}


def test_at_rest_with_loaded_jaws_the_note_names_their_opening_first():
    # The robot app refuses a hand-back while the jaws may hold something.
    assert logic_module._JAWS_LOADED == JawLoad.LOADED.value
    assert len(NOTE_AT_REST_JAWS) <= NOTE_MAX_CHARS
    assert NOTE_AT_REST_JAWS == (
        "at rest: jaws not proven empty: at the console type stop, open, yes; "
        "then hold right A for 1 s"
    )
    wanted = {
        "loaded": NOTE_AT_REST_JAWS,
        "empty": NOTE_AT_REST,
        "unknown": NOTE_AT_REST,
    }

    # After a STOP at rest, with the takeover still open.
    stopped = nobody(3, takeover_open=True, motion="rest_hold")
    for load, note in wanted.items():
        bench = Bench()
        bench.state = with_jaws(stopped, load)
        assert bench.status().note == note, load

    # While the person owns the arm, back at rest after moving it.
    at_rest = robot_state(motion="rest_hold")
    for load, note in wanted.items():
        bench = Bench()
        bench.engage()
        bench.state = with_jaws(at_rest, load)
        assert setpoint(bench.step()).note == note, load

    # Loaded jaws change no other note: before the arm was moved, away from
    # rest, and with no takeover open.
    bench = Bench()
    bench.state = with_jaws(at_rest, "loaded")
    assert setpoint(bench.step()).note == NOTE_SQUEEZE
    bench.state = with_jaws(nobody(2, takeover_open=True), "loaded")
    assert bench.status().note == NOTE_TAKE_OVER
    bench.state = with_jaws(nobody(2, motion="rest_hold"), "loaded")
    assert bench.status().note == NOTE_TAKE_OVER


def test_the_note_says_release_a_while_a_is_down_with_no_press_that_can_ask():
    assert NOTE_RELEASE_A == (
        "release right A, then hold it for 1 s with the grip released "
        "to give the arm back"
    )
    assert len(NOTE_RELEASE_A) <= NOTE_MAX_CHARS
    # After a STOP at rest, with the takeover still open.
    open_at_rest = nobody(3, takeover_open=True, motion="rest_hold")

    def held_on(bench: Bench) -> None:
        """A stays down for 3 s with the grip released; then a new press asks."""
        held = bench.run(150, a=True)
        assert of(held, (Stop, OwnerRequest)) == []
        assert notes(held) == {NOTE_RELEASE_A}
        assert bench.status().note == NOTE_AT_REST  # A is up
        again = bench.run(51, a=True)
        assert len(of(again, OwnerRequest)) == 1
        assert notes(again) == {NOTE_AT_REST}

    # The grip was squeezed as A went down: the press counts, and its hold
    # asks for nothing, also once the grip is released.
    bench = Bench()
    bench.state = open_at_rest
    bench.step(squeeze=1.0)
    assert notes(bench.run(10, a=True, squeeze=1.0)) == {NOTE_RELEASE_A}
    assert bench.logic.counts()["a_pressed"] == 1
    held_on(bench)

    # One squeezed pass in the middle of a hold that began with the grip
    # released.
    bench = Bench()
    bench.state = open_at_rest
    bench.step()
    assert notes(bench.run(20, a=True)) == {NOTE_AT_REST}
    bench.step(a=True, squeeze=1.0)
    held_on(bench)

    # One pass that is not fresh ends the hold, and A is still down.
    bench = Bench()
    bench.state = open_at_rest
    bench.step()
    assert notes(bench.run(30, a=True)) == {NOTE_AT_REST}
    bench.step(a=True, **POSE_LOST)
    held_on(bench)

    # A is down as the first data arrive: no press that counts.
    bench = Bench()
    bench.state = open_at_rest
    held_on(bench)
    assert bench.logic.counts()["a_pressed"] == 1  # the new press alone

    # A's last known reading stays over passes without known buttons.
    bench = Bench()
    bench.state = open_at_rest
    bench.step()
    bench.run(5, a=True)
    assert notes(bench.run(10, a=True, age=None)) == {NOTE_RELEASE_A}
    held_on(bench)

    # While the person owns the arm, back at rest after moving it.
    bench = Bench()
    bench.engage()
    bench.state = robot_state(motion="rest_hold")
    assert setpoint(bench.step()).note == NOTE_AT_REST
    bench.step(a=True, **POSE_LOST)
    held_on(bench)

    # Loaded jaws come first: their opening is the next step, A down or not.
    bench = Bench()
    bench.state = with_jaws(open_at_rest, "loaded")
    assert notes(bench.run(20, a=True)) == {NOTE_AT_REST_JAWS}
    bench.state = with_jaws(open_at_rest, "empty")  # opened at the console
    held_on(bench)

    # A that is down changes no other note: away from rest, with no takeover
    # open, and before the arm was moved.
    bench = Bench()
    bench.state = nobody(3, takeover_open=True)
    assert notes(bench.run(20, a=True)) == {NOTE_TAKE_OVER}
    bench.state = nobody(3, motion="rest_hold")
    assert notes(bench.run(20, a=True)) == {NOTE_TAKE_OVER}
    bench.state = robot_state(motion="rest_hold")
    assert notes(bench.run(20, a=True)) == {NOTE_SQUEEZE}


def test_a_hold_of_a_that_can_ask_keeps_the_hand_back_note_until_its_result():
    bench = Bench()
    bench.state = nobody(3, takeover_open=True, motion="rest_hold")
    bench.step()
    held = bench.run(51, a=True)
    [request] = of(held, OwnerRequest)
    # A stays down while the RESULT is pending: 2 s without one, an
    # "accepted", 0.4 s more. "release right A" would be wrong here.
    pending = bench.run(100, a=True)
    accepted = [Result(request.command_id, "accepted")]
    pending += bench.step(a=True, results=accepted) + bench.run(20, a=True)
    assert of(pending, (Stop, OwnerRequest)) == []
    assert notes(held + pending) == {NOTE_AT_REST}


def returning(**fields: Any) -> RobotState:
    """The ROBOT_STATE of a return to rest while a takeover is open.

    Recovery, the return to rest, is the control owner while the arm moves
    and while it is measured at rest; fields replace what the state shows.
    """
    shown = {"owner": "recovery", "owner_number": 4, "motion": "moving", **fields}
    return robot_state(**{"takeover_open": True, **shown})


def test_during_a_return_to_rest_with_a_takeover_open_the_note_says_b_is_a_stop():
    assert NOTE_RETURNING == "the arm returns to rest; right B is a STOP"
    assert len(NOTE_RETURNING) <= NOTE_MAX_CHARS
    assert logic_module._OWNER_RECOVERY == Owner.RECOVERY.value

    # The person moved the arm and holds it away from rest.
    bench = Bench()
    bench.engage()
    assert setpoint(bench.step()).note == NOTE_SQUEEZE
    # A typed rest: recovery owns the arm while it moves ...
    bench.state = returning()
    moving = bench.run(50)
    assert notes(moving) == {NOTE_RETURNING} and none_engaged(moving)
    # ... and while it is measured at rest, before the person has it again.
    bench.state = returning(motion="rest_hold")
    assert notes(bench.run(10)) == {NOTE_RETURNING}
    # The return is over: the notes are those of a person who owns the arm.
    at_rest = {"owner_number": 5, "motion": "rest_hold"}
    bench.state = robot_state(needs_release=True, **at_rest)
    assert setpoint(bench.step()).note == NOTE_RELEASE_FIRST
    bench.state = robot_state(**at_rest)
    assert setpoint(bench.step()).note == NOTE_AT_REST

    # Right B during the return: the STOP, under the same note until a
    # ROBOT_STATE shows it. Held on, B then asks for the arm, as after any STOP.
    bench = Bench()
    bench.state = returning()
    bench.step()
    pressed = bench.step(b=True)
    assert pressed[0] == STOP_B and setpoint(pressed).note == NOTE_RETURNING
    bench.state = nobody(2, takeover_open=True)
    held = bench.run(50, b=True)
    assert notes(held) == {NOTE_TAKE_OVER}
    assert [sent.request for sent in of(held, OwnerRequest)] == ["human"]


def test_the_note_of_a_return_to_rest_comes_ahead_of_the_others_that_apply_there():
    # Away from rest the notes of a takeover apply: the one for right B that
    # is down with no press that can ask, here as the first data arrive, and
    # the plain one.
    bench = Bench()
    bench.state = returning()
    assert bench.step(b=True)[0] == STOP_B
    assert notes(bench.run(20, b=True)) == {NOTE_RETURNING}
    assert bench.status().note == NOTE_RETURNING

    # At rest those of the hand-back apply: the one for right A that is down
    # with no press that can ask, the plain one, the one for loaded jaws.
    at_rest = returning(motion="rest_hold")
    bench = Bench()
    bench.state = at_rest
    assert notes(bench.run(20, a=True)) == {NOTE_RETURNING}
    assert bench.status().note == NOTE_RETURNING
    bench.state = with_jaws(at_rest, "loaded")
    assert bench.status().note == NOTE_RETURNING

    # A hand-back asked for during the return is refused. The refusal is not
    # shown while the return goes on, and it has gone when the return is over.
    bench = Bench()
    bench.engage()
    bench.state = returning()
    bench.step()
    held = bench.run(51, a=True)
    [request] = of(held, OwnerRequest)
    refused = [Result(request.command_id, "refused", "control owner is recovery")]
    held += bench.step(a=True, results=refused) + bench.run(20)
    assert notes(held) == {NOTE_RETURNING}
    bench.state = robot_state(owner_number=5, motion="rest_hold")
    assert setpoint(bench.step()).note == NOTE_AT_REST


def test_the_note_of_a_return_to_rest_is_given_nowhere_else():
    # Recovery owns the arm with no takeover open, as in the timed return to
    # rest of a help request nobody answered: the note of a takeover.
    bench = Bench()
    bench.state = returning(takeover_open=False)
    assert bench.status().note == NOTE_TAKE_OVER
    bench.state = returning(takeover_open=False, motion="rest_hold")
    assert bench.status().note == NOTE_TAKE_OVER

    # A takeover is open and recovery does not own the arm.
    bench.state = nobody(2, takeover_open=True)
    assert bench.status().note == NOTE_TAKE_OVER
    bench.state = nobody(2, takeover_open=True, motion="rest_hold")
    assert bench.status().note == NOTE_AT_REST
    bench.state = robot_state(owner_number=6, stop_number=2)
    assert setpoint(bench.step()).note == NOTE_SQUEEZE

    # The feed's stage is not ready during such a return: right B may not be
    # read, so the note does not say that it is a STOP. The feed's note stays.
    text = "the SAMPLES of the Teleop Session Server are too old"
    bench.state = returning()
    assert bench.status().note == NOTE_RETURNING
    stale = {"present": False, "stage": "stale", "age": None}
    assert bench.status(note=text, **stale).note == text
    assert bench.status(note="no valid pose", **POSE_LOST).note == "no valid pose"
    absent = {"outputs": False, "stage": "no_server", "age": None}
    assert bench.status(note="no server", **absent).note == "no server"
    assert bench.status().note == NOTE_RETURNING


def test_the_note_of_a_return_to_rest_changes_nothing_but_the_note(monkeypatch):
    def run() -> list[Message]:
        """A takeover with a return to rest in it, and both buttons during it."""
        bench = Bench()
        out = bench.step() + bench.run(5, squeeze=1.0)  # the arm follows the hand
        bench.state = returning()
        out += bench.run(5, squeeze=1.0) + bench.run(5)
        held = bench.run(60, a=True)  # right A: a hand-back is asked for
        [request] = of(held, OwnerRequest)
        refused = [Result(request.command_id, "refused", "control owner is recovery")]
        out += held + bench.step(a=True, results=refused) + bench.run(5)
        out += bench.run(5, b=True)  # right B: the STOP of the return
        bench.state = nobody(2, takeover_open=True)
        held = bench.run(60, b=True)  # held on, B asks for the arm again
        [request] = of(held, OwnerRequest)
        granted = [Result(request.command_id, "completed")]
        bench.state = robot_state(owner_number=6, stop_number=2)
        out += held + bench.step(b=True, results=granted) + bench.run(3)
        return out + bench.run(5, squeeze=1.0)  # the arm follows the hand again

    def plain(messages: Sequence[Message]) -> list[Message]:
        """The link messages without their notes.

        A command id keeps its number: what stands before it differs from one
        HumanLogic to the next.
        """
        out: list[Message] = []
        for message in messages:
            if isinstance(message, HumanSetpoint):
                message = dataclasses.replace(message, note="")
            elif isinstance(message, OwnerRequest):
                number = message.command_id.rsplit("-", 1)[-1]
                message = dataclasses.replace(message, command_id=number)
            out.append(message)
        return out

    with_note = run()
    # The same run with the note switched off: no owner has this word.
    monkeypatch.setattr(logic_module, "_OWNER_RECOVERY", "an owner there is not")
    without = run()
    assert NOTE_RETURNING in notes(with_note)
    assert NOTE_RETURNING not in notes(without)
    assert plain(with_note) == plain(without)
    # The run holds what could differ: both requests, the STOP, joint targets
    # before and after the return and the status HUMAN_SETPOINTs between.
    assert [sent.request for sent in of(with_note, OwnerRequest)] == [
        "handback",
        "human",
    ]
    assert of(with_note, Stop) == [STOP_B]
    sent = of(with_note, HumanSetpoint)
    assert sent[1].engaged and sent[-1].engaged
    assert sum(not one.engaged for one in sent) > 50


# The trigger and the jaws. The jaws are the gripper's fingers; a jaw opening
# is a share of their travel, 1.0 fully open and 0.0 closed. The grip level is
# the opening in force when the person took the arm over: the robot app lets
# the jaws follow the trigger once the opening it asks for has reached it.


def openings(messages: Sequence[Message]) -> list[float | None]:
    """The jaw opening of each HUMAN_SETPOINT among these link messages."""
    return [sent.jaw_opening for sent in of(messages, HumanSetpoint)]


def test_the_jaw_opening_is_one_minus_the_trigger_the_grip_squeezed_or_not():
    bench = Bench()
    # The grip is released: the arm holds, and the trigger asks all the same.
    assert setpoint(bench.step()).jaw_opening == 1.0
    released = setpoint(bench.step(trigger=0.3))
    assert released.grip_released and not released.engaged
    assert released.jaw_opening == 1.0 - 0.3
    # The grip is squeezed and the arm follows the hand.
    squeezed = setpoint(bench.step(squeeze=1.0, trigger=0.3))
    assert squeezed.engaged and squeezed.jaw_opening == 1.0 - 0.3
    assert setpoint(bench.step(squeeze=1.0, trigger=0.8)).jaw_opening == 1.0 - 0.8
    # The grip is squeezed and the arm may not follow: the gates are the arm's.
    bench.step()
    bench.state = robot_state(needs_release=True)
    held = setpoint(bench.step(squeeze=1.0, trigger=0.6))
    assert not held.engaged and held.note == NOTE_RELEASE_FIRST
    assert held.jaw_opening == 1.0 - 0.6
    bench.state = robot_state(valid=(True, False, True, True, True, True))
    blind = setpoint(bench.step(trigger=0.6))
    assert blind.note == "a joint has no valid reading"
    assert blind.jaw_opening == 1.0 - 0.6
    # On every pass while the person owns the arm.
    assert openings(bench.run(50, trigger=1.0)) == [0.0] * 50


@pytest.mark.parametrize(
    "trigger, opening",
    [
        (0.0, 1.0),
        (0.03, 1.0),
        (TRIGGER_OPEN_MAX, 1.0),
        (0.06, 1.0 - 0.06),
        (0.5, 0.5),
        (0.94, 1.0 - 0.94),
        (TRIGGER_CLOSED_MIN, 0.0),
        (0.97, 0.0),
        (1.0, 0.0),
        # A value outside 0 to 1 asks for what the nearest end asks for.
        (-0.2, 1.0),
        (1.3, 0.0),
    ],
)
def test_the_trigger_has_a_dead_zone_at_each_end(trigger, opening):
    assert (TRIGGER_OPEN_MAX, TRIGGER_CLOSED_MIN) == (0.05, 0.95)
    assert jaw_opening(trigger) == opening
    sent = setpoint(Bench().step(trigger=trigger))
    assert sent.jaw_opening == opening and 0.0 <= sent.jaw_opening <= 1.0
    assert decode(HumanSetpoint.TYPE, sent.to_payload()) == sent


def test_a_trigger_that_rests_in_its_dead_zone_asks_for_fully_open_jaws():
    # The robot app counts jaws as fully open, which proves them empty, from
    # this opening on. A trigger that rests at 0.05 would ask for less.
    full_open = 1.0 - RobotAppConfig.default().gripper.full_open_tol
    assert 1.0 - TRIGGER_OPEN_MAX < full_open
    assert jaw_opening(TRIGGER_OPEN_MAX) >= full_open


NOT_USABLE = {
    **GAPS,
    "a pose that is not valid": POSE_LOST,
    "a controller that is not active": {"present": False, "stage": "no_controller"},
    "a stage that is not ready": {"stage": "waiting_for_headset"},
    "a trigger that reads nan": {"trigger": float("nan")},
    "a trigger that reads inf": {"trigger": float("inf")},
    "a trigger that reads -inf": {"trigger": float("-inf")},
}


@pytest.mark.parametrize("controls", list(NOT_USABLE.values()), ids=list(NOT_USABLE))
def test_no_jaw_opening_on_a_pass_whose_trigger_is_not_usable(controls):
    bench = Bench()
    assert setpoint(bench.step(trigger=0.4)).jaw_opening == 1.0 - 0.4
    # None: the jaws keep their state. Never fully open, never closed.
    lost = setpoint(bench.step(**{"trigger": 0.4, **controls}))
    assert lost.jaw_opening is None
    assert setpoint(bench.step(squeeze=1.0, **controls)).jaw_opening is None
    # Usable again on the next pass.
    assert setpoint(bench.step(trigger=0.4)).jaw_opening == 1.0 - 0.4


@pytest.mark.parametrize("value", [None, "0.5", True, False, [0.5], 16**400])
def test_a_trigger_value_that_is_no_number_asks_for_no_jaw_opening(value):
    assert jaw_opening(value) is None


def test_no_jaw_opening_unless_the_robot_state_shows_the_person_as_the_owner():
    # The robot app takes a jaw opening from a person who owns the arm only.
    bench = Bench()
    for state in (
        robot_state(owner="agent", owner_number=6),
        nobody(2),
        nobody(2, takeover_open=True),
        returning(),
    ):
        bench.state = state
        assert bench.status(trigger=0.4).jaw_opening is None, state.owner

    # Right B is held for the takeover: a HUMAN_SETPOINT on every pass, and
    # still none of them asks anything of the jaws.
    bench = Bench()
    bench.state = nobody(2)
    bench.step(trigger=0.4)
    assert bench.step(b=True, trigger=0.4)[0] == STOP_B
    bench.state = nobody(3)
    held = bench.run(60, b=True, trigger=0.4)
    [request] = of(held, OwnerRequest)
    assert openings(held) == [None] * 60
    # The takeover is granted: from the first ROBOT_STATE that shows it.
    granted = [Result(request.command_id, "completed")]
    bench.state = robot_state(owner_number=6, stop_number=3, needs_release=True)
    first = setpoint(bench.step(b=True, trigger=0.4, results=granted))
    assert (first.owner_number, first.jaw_opening) == (6, 1.0 - 0.4)


def test_the_trigger_changes_nothing_of_a_pass_but_the_jaw_opening():
    def run(trigger: float) -> list[Message]:
        """A takeover, an engagement, a data gap and the hand-back's request."""
        bench = Bench()
        bench.state = robot_state(owner="agent")
        out = bench.run(3, trigger=trigger) + bench.run(2, b=True, trigger=trigger)
        bench.state = nobody(2)
        held = bench.run(60, b=True, trigger=trigger)
        [request] = of(held, OwnerRequest)
        granted = [Result(request.command_id, "completed")]
        owned = {"owner_number": 6, "stop_number": 2}
        bench.state = robot_state(needs_release=True, **owned)
        out += held + bench.step(b=True, trigger=trigger, results=granted)
        out += bench.run(2, squeeze=1.0, trigger=trigger)
        bench.state = robot_state(**owned)
        out += bench.run(3, trigger=trigger)
        out += bench.run(10, squeeze=1.0, trigger=trigger)  # the arm follows
        out += bench.lost() + bench.run(3, squeeze=1.0, trigger=trigger)
        out += bench.run(3, trigger=trigger)
        bench.state = robot_state(motion="rest_hold", **owned)
        return out + bench.run(55, a=True, trigger=trigger)

    def plain(messages: Sequence[Message]) -> list[Message]:
        """The link messages without their jaw openings.

        A command id keeps its number: what stands before it differs from one
        HumanLogic to the next.
        """
        out: list[Message] = []
        for message in messages:
            if isinstance(message, HumanSetpoint):
                message = dataclasses.replace(message, jaw_opening=None)
            elif isinstance(message, OwnerRequest):
                number = message.command_id.rsplit("-", 1)[-1]
                message = dataclasses.replace(message, command_id=number)
            out.append(message)
        return out

    released, pulled = run(0.0), run(0.7)
    assert set(openings(released)) == {None, 1.0}
    assert set(openings(pulled)) == {None, 1.0 - 0.7}
    assert plain(released) == plain(pulled)
    # The run holds what could differ: the STOP, both requests, joint targets,
    # and HUMAN_SETPOINTs with the grip released, squeezed and without data.
    assert of(pulled, Stop) == [STOP_B]
    assert [sent.request for sent in of(pulled, OwnerRequest)] == ["human", "handback"]
    sent = of(pulled, HumanSetpoint)
    assert sum(one.engaged for one in sent) == 10
    assert {one.stage for one in sent} == {"ready", "stale"}
    assert {one.grip_released for one in sent} == {True, False}
    assert len({one.note for one in sent}) >= 6


def jaws(**facts: Any) -> dict[str, Any]:
    """health["human"]["jaws"]: the robot app's facts of the jaws under the trigger.

    As after a takeover from an agent that left the jaws closed, with the
    trigger released; facts replace any of them.
    """
    short = {
        "matched": False,
        "grip_level": 0.0,
        "opened": False,
        "asked": 1.0,
        "refusal": NOT_MATCHED,
    }
    return {**short, **facts}


def test_the_note_says_which_way_the_trigger_goes_while_the_arm_follows_the_hand():
    assert NOTE_TRIGGER_PULL == "pull the trigger to the grip level to move the jaws"
    assert NOTE_TRIGGER_EASE == (
        "ease the trigger back to the grip level to move the jaws"
    )
    assert max(len(NOTE_TRIGGER_PULL), len(NOTE_TRIGGER_EASE)) <= NOTE_MAX_CHARS
    # The robot app's own words for it.
    assert logic_module._JAWS_NOT_MATCHED == NOT_MATCHED

    def note(**facts: Any) -> str:
        """The note of one more pass of the engagement, on these jaw facts."""
        bench.state = robot_state(jaws=jaws(**facts))
        sent = setpoint(bench.step(squeeze=1.0))
        assert sent.engaged
        return sent.note

    bench = Bench()
    bench.engage()
    # The jaws are closed further than the trigger asks: pull it.
    assert note() == NOTE_TRIGGER_PULL
    assert note(asked=0.4) == NOTE_TRIGGER_PULL
    assert note(grip_level=0.5, asked=0.8) == NOTE_TRIGGER_PULL
    # The trigger asks for more closing than the grip level: ease it.
    assert note(grip_level=1.0, asked=0.4) == NOTE_TRIGGER_EASE
    assert note(grip_level=0.5, asked=0.2) == NOTE_TRIGGER_EASE
    # Without both numbers: the plain one.
    assert note(asked=None) == NOTE_TRIGGER_PULL
    assert note(grip_level="0.5", asked=float("nan")) == NOTE_TRIGGER_PULL

    # The jaws follow the trigger: nothing to say, whatever holds them wider.
    assert note(matched=True, refusal="") == ""
    assert note(matched=True, refusal=NO_TIGHTER) == ""
    assert note(matched=True) == ""
    # The trigger moves nothing for another reason: no advice on the trigger.
    assert note(refusal=NO_OPENING, asked=None) == ""
    unknown = "the grip level is not known: the jaws are not referenced"
    assert note(grip_level=None, refusal=unknown) == ""
    assert note(refusal="") == ""
    # A robot app that gives no jaw facts, as with human.trigger_jaws false.
    for state in (robot_state(), robot_state(jaws=None), robot_state(jaws=NOT_MATCHED)):
        bench.state = state
        assert setpoint(bench.step(squeeze=1.0)).note == ""
    assert bench.logic.counts()["engagements"] == 1  # one engagement throughout


def test_the_trigger_note_comes_behind_every_note_about_the_arm():
    def walk(**human: Any) -> list[HumanSetpoint]:
        """A takeover through the notes about the arm.

        human goes into health["human"] of every ROBOT_STATE.
        """
        out: list[HumanSetpoint] = []

        def take(messages: Sequence[Message]) -> None:
            out.extend(of(messages, HumanSetpoint))

        def owned(number: int = 6, **fields: Any) -> RobotState:
            return robot_state(owner_number=number, stop_number=2, **human, **fields)

        gap = {"present": False, "stage": "stale", "age": None, "note": "too old"}
        bench = Bench()
        bench.state = owned(needs_release=True)
        take(bench.run(2, squeeze=1.0))  # release the grip first
        bench.state = owned()
        take(bench.run(2))  # squeeze the grip
        take(bench.run(3, squeeze=1.0))  # the arm follows the hand
        take(bench.step(**gap) + bench.run(2, squeeze=1.0))  # squeeze again
        take(bench.step())
        take(bench.step(**gap))  # the feed's note
        bench.state_age_s = 1.0
        take(bench.step())  # no fresh robot state
        bench.state_age_s = 0.01
        take(bench.step() + bench.run(3, squeeze=1.0))  # the arm follows again
        recovery = {"owner": "recovery", "takeover_open": True, "motion": "moving"}
        bench.state = robot_state(owner_number=7, stop_number=2, **human, **recovery)
        out.append(bench.status())  # the arm returns to rest
        at_rest = owned(8, motion="rest_hold")
        bench.state = at_rest
        take(bench.run(2))  # at rest: right A
        bench.state = with_jaws(at_rest, "loaded")
        take(bench.step())  # the jaws first
        bench.state = at_rest
        take(bench.step(**gap) + bench.run(2, a=True))  # release right A
        take(bench.step())
        held = bench.run(51, a=True)
        [request] = of(held, OwnerRequest)
        refused = [Result(request.command_id, "refused", "not proven empty")]
        take(held + bench.step(a=True, results=refused))  # the refusal
        bench.state = nobody(3, owner_number=9, takeover_open=True, **human)
        out.append(bench.status())  # right B takes over
        bench.step(**gap)
        take(bench.run(3, b=True))  # release right B
        return out

    plain, short = walk(), walk(jaws=jaws())
    about_the_arm = [
        NOTE_RELEASE_FIRST,
        NOTE_SQUEEZE,
        "",
        AGAIN + "controller data lost",
        NOTE_SQUEEZE,
        "too old",
        "no fresh robot state",
        NOTE_SQUEEZE,
        "",
        NOTE_RETURNING,
        NOTE_AT_REST,
        NOTE_AT_REST_JAWS,
        "too old",
        NOTE_RELEASE_A,
        NOTE_AT_REST,
        "handback refused: not proven empty",
        NOTE_TAKE_OVER,
        NOTE_RELEASE_B,
    ]
    told = [one.note for one in plain]
    assert [told[0]] + [b for a, b in zip(told, told[1:]) if a != b] == about_the_arm
    # No note is given only while the arm follows the hand.
    assert {one.engaged for one in plain if one.note == ""} == {True}
    assert {one.note for one in plain if one.engaged} == {""}
    # The trigger's note takes that place and no other, and changes nothing else.
    assert len(short) == len(plain)
    assert [one.note for one in short] == [note or NOTE_TRIGGER_PULL for note in told]
    assert [dataclasses.replace(one, note="") for one in short] == [
        dataclasses.replace(one, note="") for one in plain
    ]


def test_the_feeds_note_is_sent_while_the_stage_is_not_ready():
    text = "server did not answer at 127.0.0.1:47600"
    bench = Bench()
    bench.state = nobody(2)
    absent = {"outputs": False, "stage": "no_server", "age": None}
    sent = bench.status(note=text, **absent)
    assert (sent.stage, sent.note, sent.sample_age_s) == ("no_server", text, None)
    bench.state = robot_state()
    assert setpoint(bench.step(note=text, **absent)).note == text
    assert setpoint(bench.step(note=text)).note == NOTE_SQUEEZE  # ready again

    # After an engagement the loss ended, the reason stays until a release.
    bench.engage()
    lost = AGAIN + "controller data lost"
    assert setpoint(bench.step(note=text, **absent)).note == lost
    assert setpoint(bench.step(note=text, **absent)).note == lost
    assert setpoint(bench.step(squeeze=1.0)).note == lost
    assert setpoint(bench.step()).note == NOTE_SQUEEZE


def test_a_refused_request_is_repeated_in_the_note_until_the_next_press():
    reason = "the live view is not fresh: no VIEW_STATUS has arrived"
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    [request] = of(bench.run(50, b=True), OwnerRequest)

    # RESULTs of other commands and an "accepted" change nothing.
    others = [Result("rest-1", "refused", "x"), Result(request.command_id, "accepted")]
    assert setpoint(bench.step(b=True, results=others)).note == NOTE_TAKE_OVER

    bench.step(b=True, results=[Result(request.command_id, "refused", reason)])
    assert bench.status(b=True).note == "takeover refused: " + reason
    assert bench.status().note == "takeover refused: " + reason
    bench.step(b=True)  # a new press
    assert bench.status(b=True).note == NOTE_TAKE_OVER

    # The same for the hand-back.
    away = robot_state(owner_number=6, stop_number=9)
    bench.state = away
    bench.step()
    [request] = of(bench.run(51, a=True), OwnerRequest)
    why = "the arm is not at rest; type rest first"
    results = [Result(request.command_id, "accepted")]
    assert setpoint(bench.step(a=True, results=results)).note == NOTE_SQUEEZE
    results = [Result(request.command_id, "refused", why)]
    assert setpoint(bench.step(a=True, results=results)).note == (
        "handback refused: " + why
    )
    assert setpoint(bench.step()).note == "handback refused: " + why
    # A squeeze is the person's next act: the arm follows and the note goes.
    assert setpoint(bench.step(squeeze=1.0)).note == ""
    assert setpoint(bench.step()).note == NOTE_SQUEEZE


def refused_handback(bench: Bench, reason: str, **controls: Any) -> HumanSetpoint:
    """Right A held for 1 s and its hand-back refused: the HUMAN_SETPOINT after."""
    bench.step(**controls)
    [request] = of(bench.run(51, a=True, **controls), OwnerRequest)
    results = [Result(request.command_id, "refused", reason)]
    return setpoint(bench.step(a=True, results=results, **controls))


def test_a_refusal_leaves_the_note_once_the_robot_state_shows_a_change():
    why = "the arm is not at rest; type rest first"
    refusal = "handback refused: " + why
    bench = Bench()
    bench.state = robot_state(owner_number=6, stop_number=9)
    bench.engage()  # the arm was moved
    assert refused_handback(bench, why).note == refusal
    # A newer ROBOT_STATE that shows the same keeps it.
    bench.state = dataclasses.replace(bench.state, seq=2, clock_mono_s=501.0)
    assert setpoint(bench.step()).note == refusal

    # The typed rest: the return to rest owns the arm, then the person again.
    going = {"owner": "recovery", "motion": "moving", "takeover_open": True}
    bench.state = robot_state(owner_number=7, stop_number=9, **going)
    assert bench.status().note != refusal
    bench.state = robot_state(owner_number=8, stop_number=9, motion="rest_hold")
    assert setpoint(bench.step()).note == NOTE_AT_REST

    # Each of these alone is a change.
    base = robot_state(owner_number=6, stop_number=9)
    changed = [
        dataclasses.replace(base, owner_number=7),
        dataclasses.replace(base, stop_number=10),
        dataclasses.replace(base, motion="moving"),
        dataclasses.replace(base, session_id="s2"),
    ]
    for state in changed:
        bench = Bench()
        bench.state = base
        assert refused_handback(bench, "busy").note == "handback refused: busy"
        bench.welcome, bench.state = welcome(state.session_id), state
        assert setpoint(bench.step()).note == NOTE_SQUEEZE, state

    # A refused takeover, and then the agent is given a task.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    [request] = of(bench.run(50, b=True), OwnerRequest)
    bench.step(b=True, results=[Result(request.command_id, "refused", "busy")])
    assert bench.status().note == "takeover refused: busy"
    working = {"owner": "agent", "motion": "moving"}
    bench.state = robot_state(owner_number=6, stop_number=8, **working)
    assert bench.status().note == NOTE_TAKE_OVER

    # A refusal that came with no ROBOT_STATE of the session goes with the
    # first one.
    bench = Bench()
    bench.state = nobody(7)
    bench.step()
    bench.step(b=True)
    bench.state = nobody(8)
    [request] = of(bench.run(50, b=True), OwnerRequest)
    bench.welcome = welcome("s2")  # the robot app was started again
    assert bench.step(results=[Result(request.command_id, "refused", "busy")]) == []
    bench.state = nobody(8, session_id="s2")
    assert bench.status().note == NOTE_TAKE_OVER


def cut(text: str) -> str:
    """A note of more than 100 characters as it is sent: without its middle."""
    return text[:40] + NOTE_CUT + text[-57:]


# What every HUMAN_SETPOINT stays under as the human program sends it, in bytes.
SETPOINT_BYTES = 600


def sent_size(setpoint: HumanSetpoint) -> int:
    """The bytes of a HUMAN_SETPOINT as the human program sends it.

    The program stamps the sample age and the send time with as_sent(); here
    each has every digit. The sequence number has seven digits, as it has
    hours into a run.
    """
    late = dataclasses.replace(setpoint, seq=1234567)
    sent = as_sent(late, 100.0, 100.0 - 1.0 / 70.0, 100.0 + 1.0 / 300.0)
    assert sent.sent_mono_s not in (None, setpoint.state_clock_mono_s)
    if setpoint.sample_age_s is not None:
        assert sent.sample_age_s > setpoint.sample_age_s
    return len(sent.to_payload())


def test_a_long_note_is_cut_and_every_setpoint_stays_under_600_bytes_as_it_is_sent():
    long = "the Teleop Session Server did not answer; " * 8
    value, age = 1.2345678901234567, 0.012345678901234567
    session_id = "0123456789abcdef0123456789abcdef"
    bench = Bench()
    bench.welcome = welcome(session_id)
    bench.state = robot_state(
        owner_number=123456789,
        stop_number=123456789,
        q=(-value, value, value, -value, value, -value),
        clock_mono_s=1234567.8901234567,
        session_id=session_id,
        # The trigger has not reached the grip level: a note while engaged.
        jaws=jaws(asked=0.1, grip_level=0.5),
    )
    absent = {"outputs": False, "stage": "waiting_for_headset", "age": None}
    status = setpoint(bench.step(note=long, **absent))
    assert status.note == cut(long) and len(status.note) == NOTE_MAX_CHARS == 100
    assert sent_size(status) < SETPOINT_BYTES
    # A note of 100 characters goes out whole, one of 101 does not.
    assert setpoint(bench.step(note=long[:100], **absent)).note == long[:100]
    assert setpoint(bench.step(note=long[:101], **absent)).note == cut(long[:101])

    # The largest: a joint target, the longer note about the trigger and a
    # jaw opening, each number with every digit.
    pulled = {"squeeze": 1.0, "trigger": 0.123456789012345, "age": age}
    bench.step(age=age)
    engaged = setpoint(bench.step(**pulled))
    assert engaged.engaged and len(engaged.q) == N_JOINTS
    assert engaged.note == NOTE_TRIGGER_EASE
    assert len(repr(engaged.jaw_opening)) >= 17
    assert max(len(NOTE_TRIGGER_EASE), len(NOTE_TRIGGER_PULL)) == len(engaged.note)
    # The stamps of the send are part of the size.
    assert len(engaged.to_payload()) < sent_size(engaged) < SETPOINT_BYTES

    # The reason of a refusal is cut as well.
    refused = refused_handback(bench, long, age=age)
    assert refused.note == cut("handback refused: " + long)
    assert sent_size(refused) < SETPOINT_BYTES

    # Behind "release the grip and squeeze again: " the reason is cut by
    # itself: its first 40 characters, "..." and its last 21. The robot app
    # removes that prefix from the note, so the prefix stays whole.
    bench.step(age=age)
    bench.solver.answer = lambda target, seed: IkStep(None, "IK jump: " + long)
    assert setpoint(bench.step(squeeze=1.0, age=age)).engaged
    bench.state = following(bench.state)
    ended = setpoint(bench.step(squeeze=1.0, age=age))
    why = "IK jump: " + long
    assert ended.note == NOTE_SQUEEZE_AGAIN + why[:40] + NOTE_CUT + why[-21:]
    assert ended.note.startswith(NOTE_SQUEEZE_AGAIN)
    assert len(ended.note) <= NOTE_MAX_CHARS
    assert sent_size(ended) < SETPOINT_BYTES

    # The cut keeps what the robot app's refusals end with: what to do next.
    measured = "joint index 0 is at 0.1180 rad, rest is 0.0000 rad (tolerance 0.02)"
    for remedy in (
        "; type rest first",
        "; type stop, then open, then yes, then hold right A again",
        "; show the viewer's live picture in a visible browser tab",
    ):
        reason = f"the arm is not at rest: {measured}{remedy}"
        note = refused_handback(bench, reason, age=age).note
        assert len(note) == 100 and note.endswith(remedy)
        assert note.startswith("handback refused: the arm is not at rest")


def ended_by_the_thor(why: str) -> HumanSetpoint:
    """The HUMAN_SETPOINT of the pass on which the Thor's reason ends an engagement."""
    bench = Bench()
    bench.engage()
    bench.state = robot_state(needs_release=True, ended=why)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    return ended


def test_a_long_reason_behind_squeeze_again_keeps_its_first_forty_characters():
    # The Thor's reason for a joint target too far from the arm: 75 characters.
    why = "the target is 0.123 rad from the arm on joint index 2, the limit is 0.1 rad"
    ended = ended_by_the_thor(why)
    assert ended.note == (
        "release the grip and squeeze again: the target is 0.123 rad from the"
        " arm on ... the limit is 0.1 rad"
    )
    assert ended.note == AGAIN + why[:40] + NOTE_CUT + why[-21:]
    assert len(ended.note) == NOTE_MAX_CHARS and sent_size(ended) < SETPOINT_BYTES

    # Behind the prefix there is room for 64 characters: a reason of 64 goes
    # out whole, one of 65 does not.
    letters = "abcdefghijklmnopqrstuvwxyz" * 3
    whole = ended_by_the_thor(letters[:64]).note
    assert whole == AGAIN + letters[:64] and len(whole) == NOTE_MAX_CHARS
    shortened = ended_by_the_thor(letters[:65]).note
    assert shortened == AGAIN + letters[:40] + NOTE_CUT + letters[44:65]
    assert len(shortened) == NOTE_MAX_CHARS

    # A character outside printable ASCII in the reason becomes "?".
    odd = "caf\xe9: " + "the hand \N{RIGHTWARDS ARROW} the arm, " * 3 + "tab\there"
    plain = "caf?: " + "the hand ? the arm, " * 3 + "tab?here"
    assert ended_by_the_thor(odd).note == AGAIN + plain[:40] + NOTE_CUT + plain[-21:]
    assert ended_by_the_thor("caf\xe9").note == AGAIN + "caf?"


def test_a_note_goes_out_as_printable_ascii_and_stays_under_600_bytes_as_sent():
    session_id = "0123456789abcdef0123456789abcdef"
    bench = Bench()
    bench.welcome = welcome(session_id)
    bench.state = nobody(123456789, owner_number=123456789, session_id=session_id)
    absent = {"outputs": False, "stage": "waiting_for_headset", "age": None}

    mixed = "caf\xe9 \N{RIGHTWARDS ARROW} \U0001f600 tab\there\nnull\x00 del\x7f ~ok"
    assert bench.status(note=mixed, **absent).note == "caf? ? ? tab?here?null? del? ~ok"

    # In JSON each of these takes more than one byte per character.
    for char in ("\xe4", "\U0001f600", "\x00", "\n", '"', "\\"):
        sent = bench.status(note=char * 300, **absent)
        assert len(sent.note) == 100 and sent_size(sent) < SETPOINT_BYTES, repr(char)
        assert decode(HumanSetpoint.TYPE, sent.to_payload()) == sent
    assert bench.status(note="\xe4" * 300, **absent).note == cut("?" * 300)


def test_counts_start_at_zero_under_the_keys_the_report_line_reads():
    bench = Bench()
    counts = bench.logic.counts()
    assert tuple(counts) == COUNT_KEYS and set(counts.values()) == {0}
    report = {"no_solution", "jump", "solution_ahead", "hand_ahead", "limiter"}
    assert report <= set(counts)
    counts["jump"] = 5  # a copy: the logic's own counts stay
    bench.step()
    bench.step(a=True)
    assert bench.logic.counts() == {**dict.fromkeys(COUNT_KEYS, 0), "a_pressed": 1}


# The log lines


def test_a_pass_logs_nothing_itself_and_log_pass_writes_its_lines_in_order(caplog):
    name = logic_module._log.name
    caplog.set_level("INFO", logger=name)

    def written() -> list[tuple[str, str]]:
        """What the logic's logger has written so far."""
        records = [record for record in caplog.records if record.name == name]
        return [(record.levelname, record.getMessage()) for record in records]

    # Right B while the arm follows the hand. step() logs nothing: a log call
    # can block, and the program sends the STOP before it asks for the lines.
    bench = Bench()
    bench.engage()
    assert bench.step(squeeze=1.0, b=True)[0] == STOP_B
    assert written() == []
    bench.logic.log_pass()
    assert written() == [
        ("INFO", "engagement started"),
        ("INFO", "right B is down: STOP sent"),
        ("INFO", "engagement ended: STOP by right B"),
    ]
    bench.logic.log_pass()  # each line once
    assert len(written()) == 3

    # A request, then a pass with its refusal and right B that fails after its
    # buttons were read.
    def broken(q: Sequence[float]) -> np.ndarray:
        raise ValueError("no arm model")

    caplog.clear()
    bench = Bench()
    bench.state = robot_state(motion="rest_hold")
    bench.step()
    [request] = of(bench.run(51, a=True), OwnerRequest)
    refused = Result(request.command_id, "refused", "not at rest")
    bench.solver.fk = broken  # the home cannot be set
    assert bench.step(a=True, b=True, results=[refused]) == [STOP_B]
    assert written() == []
    bench.logic.log_pass()
    assert written() == [
        ("INFO", "right A held 1.0 s: asking for the handback"),
        ("WARNING", "handback refused: not at rest"),
        ("INFO", "right B is down: STOP sent"),
        ("ERROR", "the pass failed after its buttons were read"),
    ]
    assert caplog.records[-1].exc_info[0] is ValueError

    # Lines nobody asks for are not kept without end: the newest ones stay.
    caplog.clear()
    bench = Bench()
    for _ in range(logic_module.LOG_LINES_KEPT):
        bench.engage()  # one line for the squeeze, one for the release before
    assert written() == []
    bench.logic.log_pass()
    assert len(written()) == logic_module.LOG_LINES_KEPT
    assert written()[-1] == ("INFO", "engagement started")


# The module


def test_the_logic_imports_numpy_the_shared_types_and_the_link_messages_only():
    tree = ast.parse(Path(logic_module.__file__).read_text(encoding="utf-8"))
    relative: set[str] = set()
    absolute: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level:
            relative.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.ImportFrom):
            absolute.add(node.module.split(".")[0])
    assert relative == {"..protocol.messages", ".types"}
    assert absolute <= set(sys.stdlib_module_names) | {"numpy"}
