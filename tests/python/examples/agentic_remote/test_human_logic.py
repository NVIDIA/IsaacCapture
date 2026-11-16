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
from isaaccapture_examples.agentic_remote.human.logic import (
    COUNT_KEYS,
    NOTE_AT_REST,
    NOTE_CUT,
    NOTE_MAX_CHARS,
    NOTE_RELEASE_B,
    NOTE_RELEASE_FIRST,
    NOTE_SQUEEZE,
    NOTE_SQUEEZE_AGAIN,
    NOTE_TAKE_OVER,
    HumanLogic,
)
from isaaccapture_examples.agentic_remote.human.types import (
    FeedStep,
    GraphStep,
    HumanSettings,
    IkStep,
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
        lag_m: float = 0.0,
        limiter: str = "passed",
        clutch: bool | None = None,
        without: str | None = None,
    ) -> dict[str, GraphStep]:
        """One pass of the graph: the outputs a controller feed hands on.

        present says the controller is active; with pose_valid False its pose
        is not valid, and its buttons and squeeze are read all the same.
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
            trigger=0.0,
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


class FakeSolver:
    """A stand-in for the solver: the tip's position is linear in three joints.

    The second and third joint have their lower stop at 0, as the arm at rest.
    """

    n_joints = N_JOINTS
    LOW = (-2.0, 0.0, 0.0, -2.0, -2.0, -2.0)
    HIGH = (2.0,) * N_JOINTS
    MARGIN = 0.025
    BASE = np.array([0.2, 0.0, 0.1])
    M_PER_RAD = 0.2

    def __init__(self) -> None:
        self.solves: list[tuple[np.ndarray, tuple[float, ...]]] = []
        # Set by a test: what solve() answers in place of the arm's own solution.
        self.answer: Callable[[np.ndarray, tuple[float, ...]], IkStep] | None = None

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
        return IkStep((*(float(value) for value in first), *seed[3:]), "")


def welcome(session_id: str = "s1") -> Welcome:
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
    **human: Any,
) -> RobotState:
    """A ROBOT_STATE as the robot app sends it; human goes into health["human"]."""
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
        move_index=0,
        move_total=0,
        last_result=None,
        assistance_wait_s=None,
        torque_on=True,
        view_age_s=0.1,
        teleop_stage="ready",
        controller_age_s=0.01,
        health={"human": facts},
    )


def nobody(stop_number: int, owner_number: int = 5, **human: Any) -> RobotState:
    """The ROBOT_STATE after a STOP: nobody owns the arm."""
    return robot_state(
        owner="nobody",
        owner_number=owner_number,
        stop_number=stop_number,
        stop_level="stop",
        **human,
    )


def of(messages: Sequence[Message], kind: Any) -> list[Any]:
    return [message for message in messages if isinstance(message, kind)]


def setpoint(messages: Sequence[Message]) -> HumanSetpoint:
    """The one HUMAN_SETPOINT among the link messages of a pass."""
    [found] = of(messages, HumanSetpoint)
    return found


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
            self.state,
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

    def status(self, **controls: Any) -> HumanSetpoint:
        """The next HUMAN_SETPOINT; at the status rate it takes up to three passes."""
        for _ in range(3):
            sent = of(self.step(**controls), HumanSetpoint)
            if sent:
                return sent[0]
        raise AssertionError("no HUMAN_SETPOINT within three passes")

    def engage(self) -> HumanSetpoint:
        """Release the grip, then squeeze it: the HUMAN_SETPOINT of the squeeze."""
        self.step()
        sent = setpoint(self.step(squeeze=1.0))
        assert sent.engaged, sent.note
        return sent


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
    assert of(bench.run(75, b=True), OwnerRequest) == []  # 1.5 s, still number 7
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

    # Armed under the person: without the ownership that is gone.
    owned.step()
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


def test_the_first_seed_of_an_engagement_is_the_inside_pose():
    bench = Bench()  # at rest: two joints lie on their stops
    first = bench.engage()
    target, seed = bench.solver.solves[0]
    assert seed == REST_INSIDE
    assert np.allclose(target, bench.solver.fk(REST_INSIDE))
    assert first.q == pytest.approx(list(REST_INSIDE))

    # From then on the seed is the last joint target sent.
    bench.hand = bench.hand + np.array([0.0, 0.0, 0.002])
    second = setpoint(bench.step(squeeze=1.0))
    assert bench.solver.solves[1][1] == tuple(first.q)
    assert second.q == pytest.approx([0.0, 0.025, 0.035, 0.0, 0.0, 0.0])
    setpoint(bench.step(squeeze=1.0))
    assert bench.solver.solves[2][1] == tuple(second.q)

    # The next engagement starts at the measured pose again, not at the last
    # joint target of the one before.
    moved = (0.5, 0.2, 0.3, 0.1, 0.0, 0.0)
    bench.step()
    bench.state = robot_state(q=moved)
    bench.step()
    again = setpoint(bench.step(squeeze=1.0))
    assert bench.solver.solves[-1][1] == moved
    assert again.q == pytest.approx(list(moved))


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
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged
    assert bench.logic.counts()["jump"] == 1


def test_five_failed_solves_end_the_engagement():
    problem = "no IK solution (error 0.0063 after 50 iterations)"

    def failed(target: np.ndarray, seed: tuple[float, ...]) -> IkStep:
        return IkStep(None, problem)

    bench = Bench()
    first = bench.engage()
    bench.solver.answer = failed
    for _ in range(4):
        again = setpoint(bench.step(squeeze=1.0))
        assert again.engaged and again.q == first.q  # the last joint values
    bench.solver.answer = None
    assert setpoint(bench.step(squeeze=1.0)).engaged  # the count starts again
    bench.solver.answer = failed
    assert all_engaged(bench.run(4, squeeze=1.0))
    assert bench.logic.armed
    ended = setpoint(bench.step(squeeze=1.0))  # the fifth in a row
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + problem
    assert not bench.logic.armed
    assert bench.logic.counts()["no_solution"] == 1

    # Each engagement has its own count.
    bench = Bench()
    bench.engage()
    bench.solver.answer = failed
    assert all_engaged(bench.run(3, squeeze=1.0))
    bench.step()
    assert all_engaged(bench.run(4, squeeze=1.0))
    assert bench.logic.armed


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
    for _ in range(4):
        again = setpoint(bench.step(squeeze=1.0))
        assert again.engaged and again.q == first.q  # the last joint values
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "no IK solution (no usable joint values)"
    assert not bench.logic.armed
    assert bench.logic.counts()["no_solution"] == 1


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


def test_a_solution_too_far_ahead_of_the_arm_ends_the_engagement():
    bench = Bench()  # the measured pose is the rest pose
    bench.engage()
    bench.solver.answer = lambda target, seed: IkStep((0.079, *seed[1:]), "")
    assert setpoint(bench.step(squeeze=1.0)).q[0] == 0.079
    bench.solver.answer = lambda target, seed: IkStep((0.09, *seed[1:]), "")
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the solution is 0.090 rad ahead of the arm"
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


def test_a_hand_too_far_ahead_of_the_limited_target_ends_the_engagement():
    bench = Bench()
    bench.engage()
    bench.solver.answer = lambda target, seed: IkStep(seed, "")
    assert setpoint(bench.step(squeeze=1.0, lag_m=0.049)).engaged
    solves = len(bench.solver.solves)
    ended = setpoint(bench.step(squeeze=1.0, lag_m=0.06))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the hand is 6.0 cm ahead of the arm's target"
    assert not bench.logic.armed and len(bench.solver.solves) == solves
    assert none_engaged(bench.run(20, squeeze=1.0))
    assert bench.logic.counts()["hand_ahead"] == 1


def test_ahead_is_measured_against_the_measured_pose_until_the_thor_drives():
    near = (0.05, 0.05, 0.05, 0.0, 0.0, 0.0)
    bench = Bench()
    bench.engage()
    bench.solver.answer = lambda target, seed: IkStep(near, "")

    # No human drive runs: the Thor's last joint target is of an earlier move.
    bench.state = robot_state(target_q=(0.5, 0.5, 0.5, 0.0, 0.0, 0.0))
    assert setpoint(bench.step(squeeze=1.0)).engaged

    # The Thor drives: its last joint target counts, the measured pose does not.
    target = (0.02, 0.02, 0.02, 0.0, 0.0, 0.0)
    measured = (0.3, 0.3, 0.3, 0.0, 0.0, 0.0)
    bench.state = robot_state(q=measured, target_q=target, driving=True)
    assert setpoint(bench.step(squeeze=1.0)).engaged
    bench.state = robot_state(q=near, driving=True)  # no target known
    assert setpoint(bench.step(squeeze=1.0)).engaged
    # A joint target that is not six finite numbers is no reference either.
    for target in ((0.5, 0.5), (float("nan"),) * N_JOINTS):
        bench.state = robot_state(q=near, target_q=target, driving=True)
        assert setpoint(bench.step(squeeze=1.0)).engaged

    target = (0.2, 0.2, 0.2, 0.0, 0.0, 0.0)
    bench.state = robot_state(q=near, target_q=target, driving=True)
    ended = setpoint(bench.step(squeeze=1.0))
    assert (ended.engaged, ended.q) == (False, None)
    assert ended.note == AGAIN + "the solution is 0.150 rad ahead of the arm"


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
    assert len(bench.solver.solves) == 1  # that of the engagement before

    # The usual end: the ROBOT_STATE shows the STOP, nobody owns the arm.
    bench.state = nobody(2, takeover_open=True)
    assert none_engaged(bench.run(5)) and not bench.logic.armed

    # A ROBOT_STATE with a higher stop number ends the wait at once.
    bench = tapped()
    bench.state = robot_state(owner_number=4, stop_number=2)
    bench.step()
    assert bench.logic.armed and setpoint(bench.step(squeeze=1.0)).engaged

    # A STOP that never shows is waited for 1 s from the press, not longer.
    bench = tapped()
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
    bench.step()
    assert setpoint(bench.step(squeeze=1.0)).engaged


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
    assert bench.logic.armed and released.note == NOTE_SQUEEZE
    assert setpoint(bench.step(squeeze=1.0)).engaged
    counts = bench.logic.counts()
    assert (counts["engagements"], counts["released"]) == (2, 1)


# Rates, numbers, fields


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
        jaw_opening=None,
        state_clock_mono_s=512.5,
        note=NOTE_SQUEEZE,
    )

    bench.state = robot_state(owner_number=5, stop_number=3, clock_mono_s=512.6)
    second = setpoint(bench.step(squeeze=1.0))
    assert (second.owner_number, second.stop_number, second.seq) == (5, 3, 2)
    assert second.state_clock_mono_s == 512.6
    assert second.engaged and second.q == pytest.approx(list(REST_INSIDE))
    assert (second.jaw_opening, second.grip_released, second.note) == (None, False, "")
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


def test_a_long_note_is_cut_and_every_setpoint_stays_under_512_bytes():
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
    )
    absent = {"outputs": False, "stage": "waiting_for_headset", "age": None}
    status = setpoint(bench.step(note=long, **absent))
    assert status.note == cut(long) and len(status.note) == NOTE_MAX_CHARS == 100
    assert len(status.to_payload()) < 512
    # A note of 100 characters goes out whole, one of 101 does not.
    assert setpoint(bench.step(note=long[:100], **absent)).note == long[:100]
    assert setpoint(bench.step(note=long[:101], **absent)).note == cut(long[:101])

    bench.step(age=age)
    engaged = setpoint(bench.step(squeeze=1.0, age=age))
    assert engaged.engaged and len(engaged.q) == N_JOINTS
    assert len(engaged.to_payload()) < 512

    # The reason of a refusal and of an ended engagement are cut as well.
    refused = refused_handback(bench, long, age=age)
    assert refused.note == cut("handback refused: " + long)
    assert len(refused.to_payload()) < 512

    bench.step(age=age)
    bench.solver.answer = lambda target, seed: IkStep(None, "IK jump: " + long)
    ended = setpoint(bench.step(squeeze=1.0, age=age))
    assert ended.note == cut(NOTE_SQUEEZE_AGAIN + "IK jump: " + long)
    assert len(ended.to_payload()) < 512

    # The cut keeps what the robot app's refusals end with: what to do next.
    measured = "joint index 0 is at 0.1180 rad, rest is 0.0000 rad (tolerance 0.02)"
    for remedy in (
        "; type rest first",
        "; type stop, then open, then yes, then hold right A again",
        "; open the live view page in a visible browser tab",
    ):
        reason = f"the arm is not at rest: {measured}{remedy}"
        note = refused_handback(bench, reason, age=age).note
        assert len(note) == 100 and note.endswith(remedy)
        assert note.startswith("handback refused: the arm is not at rest")


def test_a_note_goes_out_as_printable_ascii_and_stays_under_512_bytes():
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
        assert len(sent.note) == 100 and len(sent.to_payload()) < 512, repr(char)
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
