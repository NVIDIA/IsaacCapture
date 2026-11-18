# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""human/__main__.py: the human program, and its parts together.

The human program is the workstation program through which a person drives the
arm with the headset's right controller. A takeover makes the person the control
owner, the one who may move the arm; the hand-back ends it at rest. The clutch
lets the arm follow the hand only while the grip is squeezed; an engagement is
one such stretch. A pass is one turn of the program's loop (0.02 s).

Three kinds of test. The real retargeting graph, solver and logic run pass by
pass on a script feed (a synthetic controller: rows of timed controller data)
against a stand-in for the Thor, the machine on the robot. The program's main()
runs on a thread against a real robot link service whose robot side the test
scripts, and once against the robot app on the simulated arm. The files and
the start script are read and started. No test has a headset. The file is
skipped without Isaac Capture.
"""

from __future__ import annotations

import ast
import dataclasses
import fcntl
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("isaaccapture.retargeters")

import isaaccapture  # noqa: E402
from isaaccapture_examples.agentic_remote import console_log  # noqa: E402
from isaaccapture_examples.agentic_remote.human import (  # noqa: E402
    __main__ as human_main,
)
from isaaccapture_examples.agentic_remote.human import feed as feed_module  # noqa: E402
from isaaccapture_examples.agentic_remote.human import (  # noqa: E402
    logic as logic_module,
)
from isaaccapture_examples.agentic_remote.human.__main__ import (  # noqa: E402
    HumanConfigError,
    human_settings,
    load_human_settings,
    report_line,
)
from isaaccapture_examples.agentic_remote.human.feed import ScriptFeed  # noqa: E402
from isaaccapture_examples.agentic_remote.human.ik import Solver  # noqa: E402
from isaaccapture_examples.agentic_remote.human.logic import (  # noqa: E402
    JOINT_SPEED_SHARE,
    NOTE_TAKE_OVER,
    UNNAMED_JOINT_SPEED,
    HumanLogic,
)
from isaaccapture_examples.agentic_remote.human.retarget import (  # noqa: E402
    PERMITTED_LEAF,
    HumanGraph,
)
from isaaccapture_examples.agentic_remote.human.types import (  # noqa: E402
    FeedStep,
    GraphStep,
    HumanSettings,
    IkSettings,
    RetargetSettings,
    TeleopSettings,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (  # noqa: E402
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    HumanSetpoint,
    Message,
    OwnerRequest,
    Result,
    RobotState,
    Stop,
    decode,
)
from isaaccapture_examples.agentic_remote.robot_app.config import (  # noqa: E402
    RobotAppConfig,
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.gripper import (  # noqa: E402
    NOT_MATCHED,
)
from isaaccapture_examples.agentic_remote.robot_app.human import (  # noqa: E402
    SPEED_MARGIN,
)
from link_bench import Bench, FakeThor, until  # noqa: E402
from test_apai_backend import RIG_REST_READINGS  # noqa: E402
from test_human_logic import (  # noqa: E402
    HUMAN_SPEED,
    of,
    read_at,
    robot_state,
    welcome,
)
from test_human_retarget import HOME, MID_Q, along  # noqa: E402
from test_human_retarget import Bench as GraphBench  # noqa: E402

pytestmark = [pytest.mark.link, pytest.mark.apai]

WAIT_S = 10.0  # real seconds a test gives another thread before it fails
PASS_S = 0.02
T0 = 100.0  # the time at which a rig starts
EXAMPLE = Path(human_main.__file__).resolve().parents[4]
CONFIGS = EXAMPLE / "configs"
PACKAGE = Path(human_main.__file__).resolve().parents[1]
MODULE = "isaaccapture_examples.agentic_remote.human"
USAGE = ["usage:", "python", "-m", MODULE]  # how the program's usage text begins
# A port of this process's own, bound and never listening: a dial is refused.
# Every human.yaml a test writes names it as the Teleop Session Server's.
_HELD = socket.socket()
_HELD.bind(("127.0.0.1", 0))
NO_SERVER = f"127.0.0.1:{_HELD.getsockname()[1]}"
REST = (0.0,) * 6
HAND = np.array([0.0, 1.2, -0.3])  # where the hand is in the stage space, m
BASE = {"t": 0.0, "pos": list(HAND), "quat": [0, 0, 0, 1]}
# Hand moves in the stage space (x right, y up, z toward the person), as unit
# vectors; the gripper tip then moves along +z, +x and +y of the robot's base.
UP, AWAY, LEFT = (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)
BASE_AXIS = {UP: 2, AWAY: 0, LEFT: 1}
END_STOP = Stop(
    reason="the human program ends (the stop event was set)", sender="human_program"
)
RIGHT_B = Stop(reason="right B", sender="controller")
EDGE = re.compile(r"human program: (A|B|grip|trigger) (down|up)$")
NUMBER = r"-?\d+\.\d{3}"
REPORT = re.compile(
    r"human program: stage (?P<stage>\w+); "
    r"tick age (?P<tick>\d+/\d+/\d+|-/-/-) ms; "
    r"state age (?P<state>\d+/\d+/\d+|-/-/-) ms; "
    r"round trip (?P<trip>\d+|-) ms; owner (?P<owner>[\w-]+); "
    r"engaged (?P<engaged>yes|no); "
    rf"hand x ({NUMBER}|-) y ({NUMBER}|-) z ({NUMBER}|-) m; "
    rf"tip target x ({NUMBER}|-) y ({NUMBER}|-) z ({NUMBER}|-) m, "
    r"roll (-?\d+\.\d|-) pitch (-?\d+\.\d|-) yaw (-?\d+\.\d|-) deg; "
    r"IK (?P<ik>\d+\.\d\d/\d+\.\d\d|-/-) ms; "
    r"ended: no solution \d+, jump \d+, solution ahead \d+, hand ahead \d+, "
    r"limiter \d+"
    # While the gripper tip stands at a wall of the robot app's box for it
    # with the hand beyond that wall.
    r"(; at the workspace limit: (?P<walls>[a-z]+(, [a-z]+)*))?"
    # While the joint target stands though the hand goes on.
    r"(; (?P<stand>out of reach|at a joint limit: joint \d))?"
    # While the joint target is held back for a slow robot app.
    r"(?P<behind>; the robot app is behind: the arm follows slower)?"
    # While the arm follows a hand that is 5 cm or more ahead of it.
    r"(; hand (?P<ahead>\d+) cm ahead: the arm is catching up)?"
    # While a takeover is open: the jaws, the gripper's fingers, under the
    # trigger. An opening is a share of the jaw travel, 1.00 fully open.
    r"(?P<jaws>; jaws follow the trigger (?P<following>yes|no), "
    r"grip level (?P<level>\d\.\d\d|-), trigger asks (?P<asks>\d\.\d\d|-)"
    r"(?P<why>: .+?)?)?"
    r"(?P<refused>; refused by the Thor: .+)?"
)


def is_report(line: str) -> bool:
    return line.startswith("human program: stage ")


# The parts together: the real graph, solver and logic on a script feed


class Thor:
    """A stand-in for the robot app: it answers the human program's link messages.

    It grants a takeover from nobody and a hand-back at rest, asks for a
    released grip after each grant, and follows an engaged joint target by at
    most max_step on a joint per pass (None: at once). speed is the human
    joint speed limit its WELCOME names. A drive is one stretch of engaged
    targets; its first start_passes passes send nothing, as while the robot
    app reads the arm. With a step_limit a target farther than that from the
    measured pose at a drive's first step, and from the last joint target
    sent after it, is refused: the arm holds and a released grip is asked
    for. A joint in blocked does not follow.
    """

    def __init__(
        self,
        q: Sequence[float] = REST,
        *,
        owner: str = "nobody",
        max_step: float | None = None,
        speed: float | None = HUMAN_SPEED,
        step_limit: float | None = None,
        start_passes: int = 0,
    ) -> None:
        self.owner = owner
        self.owner_number, self.stop_number = 1, 0
        self.stop_level = "none"
        self.q = list(q)  # the measured pose
        self.target_q: list[float] | None = None  # the last joint target sent
        self.driving = False
        self.takeover_open = owner == "human"
        self.needs_release = False
        self.motion = "rest_hold" if tuple(q) == REST else "hold"
        self.max_step = max_step
        self.speed = speed
        self.step_limit = step_limit
        self.start_passes = start_passes
        # The box for the gripper tip that its WELCOME names; None: none.
        self.box: list[list[float]] | None = None
        self.blocked: set[int] = set()
        self.received: list[Message] = []
        self.sent = 0  # setpoints the drive under way has sent
        self.seq: int | None = None  # of the newest HUMAN_SETPOINT taken
        self._reading = 0  # passes of the start read still to come
        self.steps: list[float] = []  # how far each target it took lay away, rad
        self.refused: list[float] = []  # the same for each target it refused
        self.given = 0  # ROBOT_STATEs given so far

    def state(self) -> RobotState:
        """Its ROBOT_STATE of now, with every joint read anew for it."""
        self.given += 1
        return read_at(self._state(), 400.0 + self.given * PASS_S)

    def _state(self) -> RobotState:
        return robot_state(
            owner=self.owner,
            owner_number=self.owner_number,
            stop_number=self.stop_number,
            stop_level=self.stop_level,
            q=self.q,
            target_q=self.target_q,
            motion=self.motion,
            move_index=self.sent if self.driving else 0,
            takeover_open=self.takeover_open,
            driving=self.driving,
            needs_release=self.needs_release,
            seq=self.seq,
        )

    def rest(self) -> None:
        """What a typed rest ends in: the arm at rest, still the person's."""
        self.q, self.target_q = list(REST), None
        self.driving, self.motion = False, "rest_hold"

    def place(self, q: Sequence[float]) -> None:
        """The arm holds at another pose, as after someone else moved it."""
        self.q, self.target_q = list(q), None
        self.driving, self.motion = False, "hold"

    def take(self, messages: Sequence[Message]) -> list[Result]:
        """Take the link messages of one pass; the RESULTs they are answered with."""
        results: list[Result] = []
        for message in messages:
            self.received.append(message)
            if isinstance(message, Stop):
                self.stop_number += 1
                self._own("nobody")
                self.stop_level, self.driving = "stop", False
            elif isinstance(message, OwnerRequest):
                results.append(self._request(message))
            elif isinstance(message, HumanSetpoint):
                self._setpoint(message)
        return results

    def _quoted(self, message: Any) -> bool:
        numbers = (message.owner_number, message.stop_number)
        return numbers == (self.owner_number, self.stop_number)

    def _own(self, owner: str) -> None:
        if owner != self.owner:
            self.owner, self.owner_number = owner, self.owner_number + 1

    def _request(self, message: OwnerRequest) -> Result:
        if not self._quoted(message):
            return Result(message.command_id, "refused", "stale numbers")
        if message.request == "human" and self.owner == "nobody":
            self._own("human")
            self.stop_level, self.takeover_open = "none", True
            self.needs_release = True
        elif (
            message.request == "handback"
            and self.takeover_open
            and self.motion == "rest_hold"
        ):
            self._own("nobody")
            self.takeover_open = False
        else:
            return Result(message.command_id, "refused", "not now")
        return Result(message.command_id, "completed")

    def _setpoint(self, message: HumanSetpoint) -> None:
        was_driving, self.driving = self.driving, False
        self.seq = message.seq
        if not self._quoted(message) or self.owner != "human":
            return
        if message.grip_released:
            self.needs_release = False
        if not message.engaged or message.q is None or self.needs_release:
            if self.motion == "moving":
                self.motion = "hold"
            return
        if not was_driving:
            self.sent, self._reading = 0, self.start_passes
        self.driving, self.motion = True, "moving"
        if self._reading > 0:
            self._reading -= 1  # the start read: nothing is sent yet
            return
        last = self.q if self.sent == 0 or self.target_q is None else self.target_q
        off = max(abs(goal - at) for at, goal in zip(last, message.q))
        if self.step_limit is not None and off > self.step_limit:
            self.refused.append(off)
            self.needs_release, self.driving, self.motion = True, False, "hold"
            return
        self.steps.append(off)
        limit = math.inf if self.max_step is None else self.max_step
        self.target_q = [
            at if joint in self.blocked else at + min(limit, max(-limit, goal - at))
            for joint, (at, goal) in enumerate(zip(last, message.q))
        ]
        self.q = list(self.target_q)
        self.sent += 1


class Rig:
    """The real graph, solver and logic on a script feed and a stand-in Thor.

    Each pass runs in the program's order, on a clock of the test's; the Thor
    answers within the pass, so every ROBOT_STATE is one pass old. With
    state_passes the logic sees the ROBOT_STATE of that many passes ago.
    """

    def __init__(
        self,
        rows: Any,
        thor: Thor,
        settings: HumanSettings | None = None,
        *,
        state_passes: int = 1,
    ) -> None:
        self.settings = settings or HumanSettings()
        ik = self.settings.ik
        self.graph = HumanGraph(self.settings.retarget)
        self.solver = Solver.from_apai(
            ik.apai_arm, max_iter=ik.max_iter, tol=ik.tol, max_jump_rad=ik.max_jump_rad
        )
        self.thor = thor
        # The ROBOT_STATEs on their way to the logic; it sees the oldest.
        self._states = deque([thor.state()], maxlen=state_passes)
        self.state = self._states[0]
        newest = lambda: self.state  # noqa: E731
        if isinstance(rows, str):
            self.feed = ScriptFeed.named(self.graph, rows, state=newest)
        else:
            self.feed = ScriptFeed(self.graph, rows, state=newest)
        self.logic = HumanLogic(
            self.graph, self.solver, self.settings, HUMAN_SOURCE_SYNTHETIC
        )
        named = welcome(human_max_joint_speed=thor.speed)
        self.welcome = dataclasses.replace(named, human_tip_box=thor.box)
        self.now = T0
        self.results: list[Result] = []
        self.sent: list[Message] = []
        # What the feed and the graph gave on each pass.
        self.fed: list[FeedStep] = []
        self.steps: list[GraphStep] = []
        # How far each engaged target ran ahead of what the Thor had sent, rad.
        self.ahead: list[float] = []
        # The same as the logic measures it. Before the ROBOT_STATE it saw
        # shows a drive that has sent a setpoint: against that state's
        # measured pose. From then on: against its joint target, less what
        # the Thor's speed clamp can have moved that target since.
        self.lead: list[float] = []

    def step(self) -> list[Message]:
        logic, graph = self.logic, self.graph
        events = graph.events(stopped=not logic.armed)
        inputs = graph.external_inputs(logic.armed)
        fed = self.feed.step(inputs, events, self.now)
        seen = self.state
        age_s = len(self._states) * PASS_S
        messages = logic.step(self.now, fed, self.welcome, seen, age_s, self.results)
        thor = self.thor
        reference = thor.target_q if thor.driving and thor.target_q else thor.q
        held_to, nearer = [sample.value for sample in seen.joints], 0.0
        moves = seen.health["human"]["driving"] and seen.move_index > 0
        if moves and seen.target_q is not None:
            held_to = seen.target_q
            nearer = age_s * SPEED_MARGIN * (thor.speed or UNNAMED_JOINT_SPEED)
        for message in of(messages, HumanSetpoint):
            if message.engaged:
                gaps = [abs(a - b) for a, b in zip(message.q, reference)]
                self.ahead.append(max(gaps))
                lead = max(abs(a - b) for a, b in zip(message.q, held_to))
                self.lead.append(lead - nearer)
        self.results = thor.take(messages)
        logic.log_pass()  # behind the link messages, as the program logs
        self._states.append(thor.state())
        self.state = self._states[0]
        self.sent += messages
        self.fed.append(fed)
        self.steps.append(logic.last)
        self.now += PASS_S
        return messages

    def run(self, seconds: float) -> list[Message]:
        """The link messages of the passes of this long."""
        out: list[Message] = []
        for _ in range(round(seconds / PASS_S)):
            out += self.step()
        return out

    def run_until(self, done: Callable[[], bool], limit_s: float = 30.0) -> None:
        for _ in range(round(limit_s / PASS_S)):
            if done():
                return
            self.step()
        raise AssertionError(f"not done within {limit_s:g} s; {self.logic.counts()}")

    def tip(self, q: Sequence[float]) -> np.ndarray:
        """Where the gripper tip is at these joint values, in the base axes."""
        return self.solver.fk(q)[:3, 3]

    def ended(self) -> dict[str, int]:
        """How many engagements the workstation itself ended, by cause."""
        counts = self.logic.counts()
        causes = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
        return {key: counts[key] for key in (*causes, "clutch", "data_lost")}


def engaged(messages: Sequence[Message]) -> list[HumanSetpoint]:
    return [m for m in of(messages, HumanSetpoint) if m.engaged]


def test_the_first_joint_target_after_an_engage_is_within_the_jump_limit_after_the_arm_moved():
    lifted = list(HAND + 0.03 * np.array(UP))
    rows = [
        BASE,
        {"t": 0.2, "squeeze": 1.0},
        {"t": 0.4},
        {"t": 1.4, "pos": lifted},
        {"t": 1.6, "squeeze": 0.0},
        {"t": 2.6, "squeeze": 1.0},
    ]
    rig = Rig(rows, Thor(owner="human"))
    limit = rig.settings.ik.max_jump_rad
    rest_inside = rig.solver.inside(REST)

    # The first engagement starts at the measured pose moved inside the IK's
    # margin, and lifts the arm 3 cm.
    first = engaged(rig.run(1.7))
    gaps = np.abs(np.array(first[0].q) - rest_inside)
    assert gaps.max() < 1e-3 and rig.logic.counts()["engagements"] == 1
    rise = rig.tip(rig.thor.q) - rig.tip(rest_inside)
    np.testing.assert_allclose(rise, [0.0, 0.0, 0.03], atol=2e-3)
    assert not rig.logic.last.engaged

    # The arm is moved far away while the grip is released.
    rig.thor.place(MID_Q)
    assert engaged(rig.run(0.8)) == []
    again = engaged(rig.run(0.4))
    assert again, rig.ended()
    # The next engagement starts where the arm is now, not where the hand left
    # the target: no jump, and nothing ended on the workstation.
    mid_inside = rig.solver.inside(MID_Q)
    gaps = np.abs(np.array(again[0].q) - mid_inside)
    assert gaps.max() <= limit and gaps.max() < 1e-3
    far = np.abs(np.array(again[0].q) - np.array(first[-1].q)).max()
    assert far > 10 * limit
    assert rig.logic.counts()["engagements"] == 2
    assert not any(rig.ended().values())


def test_a_script_lift_from_rest_engages_and_follows():
    speed = yaml.safe_load((CONFIGS / "robot_app.yaml").read_text())
    speed = speed["limits"]["human"]["max_joint_speed"]
    thor = Thor(max_step=SPEED_MARGIN * speed * PASS_S)
    rig = Rig("lift", thor)
    inside = rig.solver.inside(REST)

    # Right B: a STOP, then the takeover; the squeeze lifts the arm; the release.
    rig.run_until(lambda: rig.logic.counts()["released"] == 1)
    assert of(rig.sent, Stop) == [Stop(reason="right B", sender="controller")]
    requests = of(rig.sent, OwnerRequest)
    assert [request.request for request in requests] == ["human"]
    assert thor.owner == "human" and thor.takeover_open
    targets = engaged(rig.sent)
    assert len(targets) > 150  # the 4 s of the lift, at 50 per second
    gaps = np.abs(np.array(targets[0].q) - inside)
    assert gaps.max() <= rig.settings.ik.max_jump_rad and gaps.max() < 1e-3
    # Every pass of the lift had a solution within the limits.
    assert rig.logic.counts()["engagements"] == 1
    assert not any(rig.ended().values())
    assert max(rig.ahead) <= rig.settings.ik.max_ahead_rad
    # The arm followed: the gripper tip rose 5 cm.
    rise = rig.tip(thor.q) - rig.tip(inside)
    np.testing.assert_allclose(rise, [0.0, 0.0, 0.05], atol=2e-3)
    np.testing.assert_allclose(thor.q, targets[-1].q, atol=1e-9)

    # The operator's rest, then right A: the hand-back.
    thor.rest()
    rig.run_until(lambda: thor.owner == "nobody")
    requests = of(rig.sent, OwnerRequest)
    assert [request.request for request in requests] == ["human", "handback"]
    assert not thor.takeover_open and len(of(rig.sent, Stop)) == 1
    rig.run(2.0)
    last = rig.logic.last
    assert last.present and not (last.a or last.b) and last.squeeze == 0.0
    assert len(engaged(rig.sent)) == len(targets)  # nothing moved after the release
    # The trigger was never pulled: under the person every HUMAN_SETPOINT asked
    # for fully open jaws, and none asked anything while nobody owned the arm.
    owned = thor.owner_number - 1  # the owner number of the takeover
    asked = {
        (m.owner_number == owned, m.jaw_opening) for m in of(rig.sent, HumanSetpoint)
    }
    assert asked == {(True, 1.0), (False, None)}


def test_a_script_grasp_asks_for_closed_and_then_open_jaws_and_moves_no_joint():
    thor = Thor()
    rig = Rig("grasp", thor)
    rig.run_until(lambda: thor.owner == "human")
    owned = thor.owner_number
    rig.run_until(lambda: thor.owner == "nobody")
    rig.run(1.0)
    requests = of(rig.sent, OwnerRequest)
    assert [request.request for request in requests] == ["human", "handback"]
    assert of(rig.sent, Stop) == [Stop(reason="right B", sender="controller")]
    sent = of(rig.sent, HumanSetpoint)
    # The jaw opening the trigger asks for goes out only under the person:
    # fully open, closed for the 4 s the trigger is pulled, fully open again.
    under = [m.jaw_opening for m in sent if m.owner_number == owned]
    assert {m.jaw_opening for m in sent if m.owner_number != owned} == {None}
    changes = [under[0]] + [b for a, b in zip(under, under[1:]) if a != b]
    assert changes == [1.0, 0.0, 1.0]
    assert 195 <= under.count(0.0) <= 205  # 50 passes per second
    # The grip is never squeezed: no joint target, and the arm stays at rest.
    assert engaged(rig.sent) == [] and rig.logic.counts()["engagements"] == 0
    assert tuple(thor.q) == REST and thor.motion == "rest_hold"
    last = rig.logic.last
    assert last.present and not (last.a or last.b)
    assert (last.squeeze, last.trigger) == (0.0, 0.0)


def test_right_b_is_a_stop_through_the_real_graph_while_the_pose_is_not_valid():
    rows = [
        {**BASE, "pose_valid": False},
        {"t": 0.3, "b": True},
        {"t": 2.3, "b": False},
        {"t": 2.6, "pose_valid": True},
        {"t": 2.8, "b": True},
    ]
    rig = Rig(rows, Thor(owner="agent"))
    rig.run(0.2)
    assert of(rig.sent, Stop) == [] and of(rig.sent, HumanSetpoint)
    sent = rig.run(2.0)  # B is held for longer than buttons.hold_s
    assert of(sent, Stop) == [Stop(reason="right B", sender="controller")]
    assert of(rig.sent, OwnerRequest) == []
    held = [fed for fed, step in zip(rig.fed, rig.steps) if step.b]
    assert len(held) > 90 and not any(step.present for step in rig.steps)
    assert {(fed.stage, fed.sample_age_s) for fed in held} == {("no_controller", 0.0)}
    assert rig.thor.owner == "nobody" and rig.thor.stop_number == 1

    # With the pose back, a press that counts: a STOP again, then the takeover.
    rig.run_until(lambda: rig.thor.owner == "human")
    assert len(of(rig.sent, Stop)) == 2
    assert [request.request for request in of(rig.sent, OwnerRequest)] == ["human"]


# The rig's two files together. The Thor follows a person's joint target at
# 0.98 of limits.human.max_joint_speed of configs/robot_app_rebot.yaml, which
# its WELCOME names, and refuses a target that lies more than
# limits.human.max_step_rad from the arm at a drive's first step and from its
# own last joint target after it. The human program of
# configs/human_rebot.yaml moves the joint target it sends at 0.9 of the
# Thor's speed clamp, toward a hand that may be far ahead, so the Thor's own
# joint target is never left behind: a hand at any speed is caught up with,
# and no engagement ends for it.

RIG_ROBOT_FILE = CONFIGS / "robot_app_rebot.yaml"
# How many passes old the ROBOT_STATE is that the logic sees. The Thor sends
# one every 0.05 s, so with the pass that waits for it up to four; the logic
# follows none of 0.2 s or more (state.max_age_s), so at most nine.
STATE_PASSES = [1, 4, 9]
START = (0.0, 0.6, 0.9, 0.0, 0.0, 0.0)
# The passes of a drive in which the Thor reads the arm and sends nothing
# yet: about 0.06 s on the rig.
START_READ = 3
# Hand moves in the stage space by name; DOWN, BACK and RIGHT are the others.
DOWN, BACK, RIGHT = (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
WAYS = {"up": UP, "down": DOWN, "forward": AWAY, "back": BACK, "left": LEFT}
WAYS["right"] = RIGHT
every_way = pytest.mark.parametrize("way", list(WAYS))
# The ends of an engagement the human program itself decides on.
OWN_ENDS = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")


def rig_settings(**retarget: float) -> HumanSettings:
    """configs/human_rebot.yaml; retarget replaces keys of its retarget block."""
    settings = load_human_settings(CONFIGS / "human_rebot.yaml")
    changed = dataclasses.replace(settings.retarget, **retarget)
    return dataclasses.replace(settings, retarget=changed)


def tip_way(direction: Sequence[float]) -> np.ndarray:
    """The way the gripper tip goes, in the robot's base axes, for a hand move."""
    x, y, z = direction
    return np.array([-z, -x, y])


def moving(way: str, metres: float, speed: float, at: float = 0.4) -> list[dict]:
    """A script: a squeeze at 0.2 s, then the hand moves metres at speed from `at`."""
    goal = HAND + metres * np.array(WAYS[way])
    rows = [BASE, {"t": 0.2, "squeeze": 1.0}]
    if at > 0.2:
        rows.append({"t": at})
    return [*rows, {"t": at + metres / speed, "pos": list(goal)}]


def turning(axis: int, angle: float, speed: float) -> list[dict]:
    """A script: a squeeze at 0.2 s, then the hand turns by angle at speed from 0.4 s."""
    rows = [BASE, {"t": 0.2, "squeeze": 1.0}]
    passes = math.ceil(angle / speed / PASS_S)
    for index in range(1, passes + 1):
        half = 0.5 * angle * index / passes
        quat = [0.0, 0.0, 0.0, math.cos(half)]
        quat[axis] = math.sin(half)
        rows.append({"t": 0.4 + index * PASS_S, "quat": quat})
    return rows


def turn_between(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between the orientations of two 4x4 poses, rad."""
    cosine = 0.5 * (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0)
    return math.acos(min(1.0, max(-1.0, cosine)))


class Drive:
    """What one run of a Rig on the rig's files showed."""

    def __init__(self, rig: Rig, thor: Thor, start: Sequence[float]) -> None:
        self.rig, self.thor = rig, thor
        self.home = rig.solver.fk(rig.solver.inside(start))
        self.targets = np.array([m.q for m in engaged(rig.sent)])
        # How far a joint of the joint target moved from one pass to the next.
        self.steps = np.abs(np.diff(self.targets, axis=0)).max(axis=1)
        tips = np.array([rig.tip(q) for q in self.targets])
        self.tip_speeds = np.linalg.norm(np.diff(tips, axis=0), axis=1) / PASS_S
        self.ended = {key: count for key, count in rig.ended().items() if count}
        self.counts = rig.logic.counts()

    def healthy(self, off_the_arm: float = 0.0) -> None:
        """One engagement that nothing ended, under a Thor that refused nothing.

        off_the_arm is how far the squeeze's first target lay from the arm.
        """
        thor, settings = self.thor, self.rig.settings
        assert self.ended == {} and self.counts["engagements"] == 1
        assert thor.refused == [] and self.counts["needs_release"] == 0
        # No joint of the joint target was faster than 0.9 of the Thor's clamp.
        most = JOINT_SPEED_SHARE * thor.clamp
        assert self.steps.max() <= most + 1e-9
        # So no target lay further from the Thor's own last joint target than
        # the first one of the squeeze, or than one such step: the Thor's
        # joint target was never left behind, and its step limit never met.
        assert max(thor.steps) <= max(off_the_arm, most) + 1e-9 < thor.step_limit
        # The tip of the joint target was no faster than the hand speed limit.
        assert self.tip_speeds.max() <= 1.02 * settings.retarget.max_linear_velocity
        assert max(self.rig.lead) <= max(off_the_arm, 0.01) < settings.ik.max_ahead_rad

    def at_the_cap(self) -> bool:
        """Whether the fastest joint of the joint target reached its speed limit."""
        return self.steps.max() == pytest.approx(JOINT_SPEED_SHARE * self.thor.clamp)

    def off_the_hand(self) -> tuple[float, float]:
        """How far the arm stands from the hand's pose: m and rad."""
        arm = self.rig.solver.fk(self.thor.q)
        hand = self.rig.logic.last.commanded_target
        metres = float(np.linalg.norm(arm[:3, 3] - hand[:3, 3]))
        return metres, turn_between(arm, hand)


def drive_on_the_rig(
    rows: list[dict],
    seconds: float,
    *,
    start: Sequence[float] = START,
    state_passes: int = 1,
    settings: HumanSettings | None = None,
    thor: Thor | None = None,
) -> Drive:
    """Run a script for this long on the rig's two files, from the pose start."""
    thor = thor or RigThor(start, owner="human", start_passes=START_READ)
    rig = Rig(rows, thor, settings or rig_settings(), state_passes=state_passes)
    rig.run(seconds)
    return Drive(rig, thor, start)


class RigThor(Thor):
    """The stand-in with the two bounds of the rig's robot file on a human target.

    A target farther than limits.human.max_step_rad from the measured pose at
    a squeeze, and from the last joint target sent after it, is refused: the
    arm holds and the Thor asks for a released grip.
    """

    def __init__(self, q: Sequence[float], **keywords: Any) -> None:
        human = yaml.safe_load(RIG_ROBOT_FILE.read_text())["limits"]["human"]
        speed, step_limit = human["max_joint_speed"], human["max_step_rad"]
        self.clamp = SPEED_MARGIN * speed * PASS_S  # the farthest step of a joint
        super().__init__(
            q, max_step=self.clamp, speed=speed, step_limit=step_limit, **keywords
        )


def test_the_rig_files_name_the_speeds_and_the_limits_of_a_person():
    settings, thor = rig_settings(), RigThor(START)
    assert (thor.speed, thor.step_limit) == (1.6, 0.10)  # the owner's speed
    assert thor.clamp == pytest.approx(0.03136)
    step = JOINT_SPEED_SHARE * thor.clamp  # of the joint target, in one pass
    assert step == pytest.approx(0.028224)
    # The tip's target may move at 50 cm a second and turn at 2 rad a second;
    # a hand up to half a metre ahead of the arm is caught up with.
    retarget, ik = settings.retarget, settings.ik
    assert (retarget.max_linear_velocity, retarget.max_angular_velocity) == (0.5, 2.0)
    assert ik.max_ahead_m == 0.5 and ik.max_ahead_rad == 0.09 < thor.step_limit
    # A joint target that follows the hand is held back to two of the Thor's
    # steps ahead of the Thor's own: inside the lead that ends an engagement.
    assert logic_module.HOLD_BACK_STEPS * thor.clamp < ik.max_ahead_rad
    # A solution is a jump from 0.10 rad: a step of the tip that was halved
    # for one still moves the fastest joint at its speed limit.
    assert ik.max_jump_rad == 0.10 and 0.5 * ik.max_jump_rad >= step
    assert settings.state_max_age_s == pytest.approx((max(STATE_PASSES) + 1) * PASS_S)
    assert (settings.ik.apai_arm, Solver.from_apai("rebot_rs").n_joints) == (
        "rebot_rs",
        6,
    )


@every_way
@pytest.mark.parametrize("speed", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_a_hand_at_any_steady_speed_is_followed_to_where_it_stops(
    way, speed, state_passes
):
    # 10 cm at 5, 20 and 50 cm a second; the hand then stays where it is.
    metres = 0.10
    seconds = 0.4 + metres / speed + 2.5
    rows = moving(way, metres, speed)
    drive = drive_on_the_rig(rows, seconds, state_passes=state_passes)
    drive.healthy()
    rig, thor = drive.rig, drive.thor
    assert len(drive.targets) > 100 and rig.logic.last.engaged
    # The arm arrived at the hand's pose and stands there: the hand's way,
    # with the orientation it had at the squeeze.
    moved = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(moved, metres * tip_way(WAYS[way]), atol=1e-3)
    assert drive.off_the_hand() == pytest.approx((0.0, 0.0), abs=1e-3)
    assert drive.steps[-25:].max() < 1e-6 and rig.logic.ahead_m < 1e-3
    in_motion = drive.tip_speeds[drive.tip_speeds > 1e-4]
    if speed >= 0.5:
        # Faster than the joints allow here: the fastest joint ran at its
        # limit, and the tip stayed behind the hand.
        assert drive.at_the_cap() and drive.tip_speeds.max() < speed
    elif speed <= 0.05 or not drive.at_the_cap():
        # At 20 cm a second the fastest joint ran at its limit in some runs:
        # to the side, where the first joint alone moves the tip, and where
        # the arm had the start of the Thor's drive to make up.
        # The arm kept the hand's pace.
        assert float(np.median(in_motion)) == pytest.approx(speed, rel=0.02)
    print(
        f"\n{way} at {speed} m/s, state {state_passes} passes old: tip up to "
        f"{drive.tip_speeds.max():.3f} m/s, lead {max(rig.lead):.4f} rad"
    )


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
@pytest.mark.parametrize("speed", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("state_passes", STATE_PASSES[::2])
def test_a_hand_that_turns_at_any_steady_speed_is_followed_to_where_it_stops(
    axis, speed, state_passes
):
    angle = 0.6  # 34 degrees, at 0.3, 1 and 3 rad a second
    seconds = 0.4 + angle / speed + 4.0
    rows = turning(axis, angle, speed)
    drive = drive_on_the_rig(rows, seconds, state_passes=state_passes)
    drive.healthy()
    rig, thor = drive.rig, drive.thor
    # The arm turned its tool as the hand turned, about the tip, and stands.
    arm = rig.solver.fk(thor.q)
    assert turn_between(drive.home, arm) == pytest.approx(angle, abs=2e-3)
    assert drive.off_the_hand() == pytest.approx((0.0, 0.0), abs=2e-3)
    assert drive.steps[-25:].max() < 1e-6
    # At 3 rad a second the fastest joint ran at its limit.
    assert drive.at_the_cap() or speed < 3.0


@pytest.mark.parametrize("way", ["forward", "back", "left", "right"])
def test_a_hand_moved_30_cm_in_0_4_s_is_followed_and_the_arm_arrives(way):
    drive = drive_on_the_rig(moving(way, 0.30, 0.75), 0.4 + 0.4 + 3.0, state_passes=4)
    drive.healthy()
    rig, thor = drive.rig, drive.thor
    moved = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(moved, 0.30 * tip_way(WAYS[way]), atol=1e-3)
    assert drive.at_the_cap() and drive.steps[-25:].max() < 1e-6
    # The arm was behind the hand, and the note said so while it caught up.
    said = [m.note for m in engaged(rig.sent)]
    changes = [said[0]] + [b for a, b in zip(said, said[1:]) if a != b]
    assert changes[0] == changes[-1] == ""
    assert "hand 10 cm ahead: the arm is catching up" in changes
    notes = (f"hand {cm} cm ahead: the arm is catching up" for cm in (5, 10, 20))
    assert set(changes) <= {"", *notes}
    # The hand was there after 0.4 s; the arm was on its way for about one.
    on_its_way = int(np.sum(drive.steps > 1e-6)) * PASS_S
    assert 0.7 < on_its_way < 1.5
    print(f"\n30 cm {way} in 0.4 s: the arm was on its way for {on_its_way:.2f} s")


@pytest.mark.parametrize("reading", RIG_REST_READINGS)
@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_a_squeeze_at_rest_and_a_fast_lift_run_on_the_rig_files(reading, state_passes):
    # The arm lies folded on the stops of its second and third joint, as the
    # rig read it there. The squeeze's first solution lies inside the IK's
    # margin from those stops, about 0.03 rad above the measured pose. The
    # hand rises at 20 cm a second from the pass of the squeeze on, while the
    # Thor still reads the arm for the drive.
    metres = 0.10
    rows = moving("up", metres, 0.2, at=0.2)
    rows.append({"t": 0.2 + 0.5 + 2.0, "squeeze": 0.0})
    thor = RigThor(reading, owner="human", start_passes=START_READ)
    drive = drive_on_the_rig(
        rows, 0.2 + 0.5 + 2.6, start=reading, state_passes=state_passes, thor=thor
    )
    rig = drive.rig
    inside = rig.solver.inside(reading)
    off_the_stops = max(abs(a - b) for a, b in zip(inside, reading))
    assert 0.031 < off_the_stops < 0.033
    # One engagement, ended by the released grip; the Thor refused nothing.
    assert (drive.counts["engagements"], drive.counts["released"]) == (1, 1)
    drive.healthy(off_the_arm=off_the_stops)
    # The first target is the farthest the Thor was asked to go: off the
    # stops, as the logic measured it from the arm before the Thor drove. It
    # stood until a ROBOT_STATE showed the Thor's drive following it.
    assert thor.steps[0] == pytest.approx(off_the_stops, abs=1e-9)
    assert max(rig.lead) == pytest.approx(off_the_stops, abs=1e-9)
    # The Thor's first setpoint came START_READ passes after the squeeze and
    # brought its joint target within a step of the first target, 0.031 rad
    # of the way; the ROBOT_STATE with that came state_passes later.
    held = int(np.argmax(drive.steps > 1e-9))
    assert held == START_READ + state_passes - 1
    np.testing.assert_allclose(drive.targets[: held + 1], [inside] * (held + 1))
    # The gripper tip rose with the hand, which was a pass on its way when
    # the squeeze was read.
    rise = rig.tip(thor.q) - rig.tip(inside)
    assert rise[:2] == pytest.approx([0.0, 0.0], abs=1e-3)
    assert metres - 0.2 * 2 * PASS_S <= rise[2] <= metres + 1e-3
    print(f"\nrest {reading[1]:.4f}, {state_passes} passes: held {held} passes")


@pytest.mark.parametrize(("reads", "lifted"), [(-0.03, True), (-0.07, False)])
def test_how_far_under_0_a_joint_on_its_stop_may_read_for_a_squeeze_at_rest(
    reads, lifted
):
    # A known limit of the two limits. The first solution lies 0.025 rad, the
    # IK's margin, above the 0 of the arm model: a second joint that reads
    # further than 0.065 rad under 0 is more than ik.max_ahead_rad away. The
    # rig read -0.007 to -0.008 there.
    reading = [*RIG_REST_READINGS[0]]
    reading[1] = reads
    rows = [
        BASE,
        {"t": 0.2, "squeeze": 1.0},
        {"t": 2.2, "pos": list(HAND + 0.02 * np.array(UP))},
    ]
    thor = RigThor(reading, owner="human")
    rig = Rig(rows, thor, rig_settings(), state_passes=4)
    sent = rig.run(2.6)
    ended = {key: count for key, count in rig.ended().items() if count}
    if lifted:
        assert ended == {} and len(engaged(sent)) > 100
        assert thor.steps[0] == pytest.approx(0.025 - reads, abs=1e-3)
    else:
        # The human program ends the engagement on its first pass: the Thor
        # gets no target, refuses none, and the arm stays where it lies.
        assert ended == {"solution_ahead": 1} and engaged(sent) == []
        assert thor.steps == [] and list(thor.q) == reading
    assert thor.refused == [] and rig.logic.counts()["needs_release"] == 0


@pytest.mark.parametrize("way", ["forward", "left"])
def test_a_hand_half_a_metre_ahead_of_the_arm_is_not_believed(way):
    # 80 cm in 0.5 s, under the speed the rate limiter rejects: the arm goes
    # after the hand until the hand is 50 cm ahead of it.
    drive = drive_on_the_rig(moving(way, 0.8, 1.6), 2.0, state_passes=4)
    rig, thor = drive.rig, drive.thor
    assert drive.ended == {"hand_ahead": 1} and drive.counts["engagements"] == 1
    assert thor.refused == [] and drive.counts["needs_release"] == 0
    said = [m.note for m in of(rig.sent, HumanSetpoint)]
    changes = [said[0]] + [b for a, b in zip(said, said[1:]) if a != b]
    assert changes[:-1] == [
        "squeeze the grip to move",
        "",
        *(f"hand {cm} cm ahead: the arm is catching up" for cm in (5, 10, 20, 30, 40)),
    ]
    ended = r"the hand is 5[12] cm from the gripper tip, over the 50 cm believed"
    assert re.fullmatch("release the grip and squeeze again: " + ended, changes[-1])
    # The arm holds where it was when the engagement ended, on its way.
    assert not thor.driving and thor.motion == "hold"
    travelled = float(np.linalg.norm(rig.tip(thor.q) - drive.home[:3, 3]))
    assert 0.01 < travelled < 0.20
    np.testing.assert_allclose(thor.q, drive.targets[-1], atol=1e-9)


@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_a_released_grip_stops_the_arm_on_its_pass_at_any_speed(state_passes):
    # The hand is 30 cm ahead and the arm at its speed limit when the grip
    # is released: no joint target follows the release.
    rows = [*moving("forward", 0.30, 1.0), {"t": 0.7, "squeeze": 0.0}]
    drive = drive_on_the_rig(rows, 2.0, state_passes=state_passes)
    drive.healthy()
    rig, thor = drive.rig, drive.thor
    assert drive.counts["released"] == 1 and drive.at_the_cap()
    assert drive.steps[-1] == pytest.approx(JOINT_SPEED_SHARE * thor.clamp)
    sent = of(rig.sent, HumanSetpoint)
    last = max(index for index, m in enumerate(sent) if m.engaged)
    after = sent[last + 1]
    assert (after.engaged, after.q, after.grip_released) == (False, None, True)
    assert all(not m.engaged for m in sent[last + 1 :])
    # The arm stands at the last joint target it was sent, short of the hand:
    # it travelled one step of the Thor's, 1.8 degrees a joint, past the
    # target of the pass before.
    np.testing.assert_allclose(thor.q, drive.targets[-1], atol=1e-12)
    assert not thor.driving and thor.motion == "hold"
    short = 0.30 - float(np.linalg.norm(rig.tip(thor.q) - drive.home[:3, 3]))
    assert short > 0.10


@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_right_b_stops_the_arm_on_its_pass_at_any_speed(state_passes):
    rows = [*moving("forward", 0.30, 1.0), {"t": 0.7, "b": True}]
    drive = drive_on_the_rig(rows, 2.0, state_passes=state_passes)
    rig, thor = drive.rig, drive.thor
    assert drive.ended == {} and thor.refused == [] and drive.at_the_cap()
    # The STOP went out on the pass that read right B down, ahead of that
    # pass's HUMAN_SETPOINT; the Thor followed no target after it.
    [stop] = [index for index, m in enumerate(rig.sent) if isinstance(m, Stop)]
    assert rig.sent[stop] == RIGHT_B
    before = [m for m in rig.sent[:stop] if isinstance(m, HumanSetpoint)]
    assert before[-1].engaged
    np.testing.assert_allclose(thor.q, before[-1].q, atol=1e-12)
    assert thor.owner == "nobody" and thor.stop_level == "stop" and not thor.driving
    assert drive.counts["stop"] + drive.counts["not_owner"] == 1


class Stuck(RigThor):
    """The rig's stand-in with a joint that stops following after some setpoints."""

    def __init__(self, joint: int, after: int, **keywords: Any) -> None:
        super().__init__(START, owner="human", start_passes=START_READ, **keywords)
        self.joint, self.after = joint, after
        self.stuck_at: float | None = None  # where the joint stands
        self.stuck_seq = 0  # the seq of the first HUMAN_SETPOINT it did not follow

    def _setpoint(self, message: HumanSetpoint) -> None:
        if self.sent >= self.after and self.stuck_at is None:
            self.blocked.add(self.joint)
            self.stuck_at, self.stuck_seq = self.q[self.joint], message.seq
        super()._setpoint(message)

    def ahead(self, sent: Sequence[Message]) -> np.ndarray:
        """How far each joint target since then lay ahead of the joint, rad."""
        since = [m.q[self.joint] for m in engaged(sent) if m.seq >= self.stuck_seq]
        return np.abs(np.array(since) - self.stuck_at)


def test_a_joint_target_the_thor_does_not_follow_is_held_back_and_ends_as_not_followed():
    # The first joint of the Thor's own joint target stands still from its
    # 20th setpoint on, while the hand goes on to the left at 50 cm a second.
    # The ROBOT_STATE that shows it is one pass old.
    thor = Stuck(0, 20)
    drive = drive_on_the_rig(moving("left", 0.3, 0.5), 3.0, thor=thor)
    rig = drive.rig
    # The joint target was held back two of the Thor's steps, and what its
    # clamp could have done in the state's age, ahead of the joint that stood:
    # under the Thor's step limit, so the Thor refused nothing.
    assert thor.refused == [] and drive.counts["needs_release"] == 0
    ahead = thor.ahead(rig.sent)
    room = (logic_module.HOLD_BACK_STEPS + 1) * thor.clamp
    assert ahead.max() == pytest.approx(room) and room < thor.step_limit
    # A second after it was first held back the human program ended the
    # engagement, and the arm holds.
    assert drive.ended == {"solution_ahead": 1} and not rig.logic.armed
    held_s = (int(np.sum(ahead > room - 1e-9)) - 1) * PASS_S
    assert logic_module.NO_FOLLOW_S <= held_s <= logic_module.NO_FOLLOW_S + 2 * PASS_S
    assert not thor.driving and thor.motion == "hold"
    sent = of(rig.sent, HumanSetpoint)
    last = max(index for index, m in enumerate(sent) if m.engaged)
    again = "release the grip and squeeze again: "
    assert sent[last + 1].note == again + logic_module.NOT_FOLLOWED
    assert sent[last].note.startswith("the robot app is behind: the arm follows slower")
    assert abs(thor.q[0] - thor.stuck_at) < 1e-12


@pytest.mark.parametrize("state_passes", STATE_PASSES[1:])
def test_a_joint_target_the_thor_does_not_follow_ends_the_drive_at_the_thors_step_limit(
    state_passes,
):
    # The same with ROBOT_STATEs 4 and 9 passes old. The hold-back sees the
    # joint that stands only in such a state: until then the joint target has
    # run on, and the Thor's step limit is what ends the drive, as before.
    thor = Stuck(0, 20)
    drive = drive_on_the_rig(
        moving("left", 0.3, 0.5), 3.0, state_passes=state_passes, thor=thor
    )
    rig = drive.rig
    # The Thor refused the first target that lay beyond its step limit from
    # its own joint target, and the arm holds; the human program ended the
    # engagement on the ROBOT_STATE that said so.
    assert len(thor.refused) == 1 and thor.step_limit < thor.refused[0]
    assert thor.refused[0] <= thor.step_limit + JOINT_SPEED_SHARE * thor.clamp
    assert drive.ended == {} and drive.counts["needs_release"] == 1
    assert not thor.driving and thor.motion == "hold" and not rig.logic.armed
    # The joint target ran ahead of the joint that stood by the step limit
    # and, until the ROBOT_STATE came, by one more step for each pass of its
    # age; the Thor followed none of those.
    ahead = thor.ahead(rig.sent).max()
    most = thor.step_limit + (state_passes + 1) * JOINT_SPEED_SHARE * thor.clamp
    assert thor.step_limit < ahead <= most
    assert abs(thor.q[0] - thor.stuck_at) < 1e-12


@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_without_the_thors_step_limit_a_drive_that_is_not_followed_is_held_back(
    state_passes,
):
    # The same under a Thor that refuses no target for its distance: the human
    # program alone is left to end the drive.
    thor = Stuck(0, 20)
    thor.step_limit = math.inf
    settings = rig_settings()
    drive = drive_on_the_rig(
        moving("left", 0.3, 0.5), 3.0, state_passes=state_passes, thor=thor
    )
    rig = drive.rig
    assert drive.ended == {"solution_ahead": 1} and thor.refused == []
    assert not thor.driving and thor.motion == "hold" and not rig.logic.armed
    # The joint target stops two of the Thor's steps further ahead than the
    # Thor's speed clamp could have brought the Thor's own joint target since
    # the ROBOT_STATE: with the oldest state followed, 0.34 rad. The lead is
    # what is left of that less the clamp's way, two of the Thor's steps:
    # inside ik.max_ahead_rad, so the lead ends nothing here.
    age_s = state_passes * PASS_S
    room = logic_module.HOLD_BACK_STEPS * thor.clamp
    most = room + age_s * SPEED_MARGIN * thor.speed
    ahead = thor.ahead(rig.sent)
    assert ahead.max() == pytest.approx(most) and most <= 0.346
    assert room < settings.ik.max_ahead_rad
    # A second after the last ROBOT_STATE with a new joint target of the
    # Thor, with the joint target held back, the engagement ends.
    sent = of(rig.sent, HumanSetpoint)
    last = max(index for index, m in enumerate(sent) if m.engaged)
    again = "release the grip and squeeze again: "
    assert sent[last + 1].note == again + logic_module.NOT_FOLLOWED
    held_s = (int(np.sum(ahead > most - 1e-9)) - 1) * PASS_S
    bound = logic_module.NO_FOLLOW_S + (state_passes + 1) * PASS_S
    assert logic_module.NO_FOLLOW_S <= held_s <= bound
    print(
        f"\nstate {state_passes} passes old: held {ahead.max():.3f} rad ahead, ended "
        f"{len(ahead) * PASS_S:.2f} s after the joint stood"
    )


# A second squeeze right after a fast move. The ROBOT_STATE the logic sees at
# that squeeze may have been made while the arm still moved: its measured pose
# lies behind the arm. The squeeze counts only once a state has shown the
# drive before it as ended and two readings of every joint have agreed, so
# the engagement starts where the arm stands. The stand-in's arm is at its
# target at once: test_human_realtime.py has the robot app on the simulated arm.


@pytest.mark.parametrize("gap_passes", [2, 3, 5])
@pytest.mark.parametrize("state_passes", STATE_PASSES)
def test_a_second_squeeze_right_after_a_fast_move_starts_where_the_arm_stands(
    gap_passes, state_passes
):
    # The hand is 30 cm ahead and the arm at full speed when the grip is
    # released; it is squeezed again 2, 3 and 5 passes later.
    release = 1.2
    again = release + gap_passes * PASS_S
    rows = [
        *moving("left", 0.30, 1.0),
        {"t": release, "squeeze": 0.0},
        {"t": again, "squeeze": 1.0},
        {"t": again + 1.5, "squeeze": 0.0},
    ]
    thor = RigThor(START, owner="human", start_passes=START_READ)
    rig = Rig(rows, thor, rig_settings(), state_passes=state_passes)
    poses: list[list[float]] = []
    firsts: list[tuple[int, list[float], list[float]]] = []
    said: list[str] = []
    was = False
    for index in range(round((again + 2.5) / PASS_S)):
        before = list(thor.q)
        made = of(rig.step(), HumanSetpoint)
        poses.append(list(thor.q))
        said += [m.note for m in made]
        now = bool(made and made[-1].engaged)
        if now and not was:
            firsts.append((index, list(made[-1].q), before))
        was = now
    counts = rig.logic.counts()
    assert (counts["engagements"], counts["released"]) == (2, 2)
    assert thor.refused == [] and counts["needs_release"] == 0
    assert not any(rig.ended().values())
    # The first target of the second squeeze is the arm's pose, to the last
    # bit: the arm had stood since the release.
    at, first_q, arm = firsts[1]
    assert first_q == arm == poses[round(release / PASS_S) + 1]
    # So the arm never moved back: its first joint turns one way throughout.
    turned = np.diff(np.array(poses)[:, 0])
    turned = turned * np.sign(poses[-1][0] - poses[0][0])
    assert turned.max() > 0.01 and turned.min() >= 0.0
    # The squeeze is read a pass after its row, and it waited for the
    # ROBOT_STATE that showed the drive as ended and for the next one, whose
    # new readings showed the same pose: at most 0.2 s from the release on
    # this stand-in, whose arm is at its target at once. The note said so on
    # each pass of the wait.
    waited = at - round(again / PASS_S) - 1
    assert waited == max(0, state_passes - gap_passes + 2)
    assert said.count("wait: the arm is still stopping") == waited
    # The clutch took the hand anew at that pose and the hand stood: so did
    # the arm, through the 1.5 s of the second engagement.
    np.testing.assert_allclose(thor.q, arm, atol=1e-4)
    assert thor.sent > 60 and max(thor.steps[-thor.sent :]) < 1e-4
    print(
        f"\nsqueezed {gap_passes} passes after the release, state {state_passes} "
        f"passes old: waited {waited} passes, first target on the arm"
    )


# The hold-back: a pause of the Thor's loop, or a loop that runs slow, costs
# speed and not the engagement. The joint target goes no further than two of
# the Thor's steps ahead of where the Thor's own joint target can be, by the
# newest ROBOT_STATE. What the logic cannot see in time is left to the
# Thor's step limit, as before: a pause that no ROBOT_STATE has shown yet.


class Paused(RigThor):
    """The rig's stand-in whose loop stands for some passes at full speed.

    It takes no HUMAN_SETPOINT on those passes and reports its drive as
    running, with the joint target it had.
    """

    def __init__(self, after: int, passes: int) -> None:
        super().__init__(START, owner="human", start_passes=START_READ)
        self.after, self.passes = after, passes
        self.skipped = 0

    def _setpoint(self, message: HumanSetpoint) -> None:
        if self.sent >= self.after and self.skipped < self.passes and message.engaged:
            self.skipped += 1
            self.seq = message.seq
            return
        super()._setpoint(message)


@pytest.mark.parametrize("passes", [3, 4, 5, 10])
def test_a_pause_of_the_thors_loop_costs_speed_and_not_the_engagement(passes):
    # 30 cm to the left in 0.3 s: the arm is at full joint speed when the
    # Thor's loop stands for 3, 4, 5 and 10 passes after its 20th setpoint.
    thor = Paused(20, passes)
    drive = drive_on_the_rig(moving("left", 0.30, 1.0), 4.5, thor=thor)
    rig = drive.rig
    assert thor.skipped == passes
    # Nothing ended and the Thor refused nothing: the joint target stood two
    # of the Thor's steps, and one for the state's age, ahead of the Thor's
    # own, which is inside the Thor's step limit of 0.10 rad.
    assert drive.ended == {} and thor.refused == []
    assert (drive.counts["engagements"], drive.counts["needs_release"]) == (1, 0)
    room = (logic_module.HOLD_BACK_STEPS + 1) * thor.clamp
    assert max(thor.steps) == pytest.approx(room) and room < thor.step_limit
    # The arm arrived at the hand, later by about the pause.
    moved = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(moved, 0.30 * tip_way(WAYS["left"]), atol=1e-3)
    on_time = drive_on_the_rig(moving("left", 0.30, 1.0), 4.5)
    arrived = int(np.nonzero(drive.steps > 1e-6)[0][-1])
    late = arrived - int(np.nonzero(on_time.steps > 1e-6)[0][-1])
    assert 0 <= late <= passes
    print(
        f"\na pause of {passes} passes: the Thor's largest step {max(thor.steps):.4f} "
        f"rad, the arm arrived {late} passes later"
    )


class TimedThor(RigThor):
    """The rig's stand-in on a clock of its own, as the robot app runs.

    Its loop takes the newest HUMAN_SETPOINT every loop_s, first at phase_s;
    it stands for pause_s once it has sent pause_after setpoints, and sends
    none at all after silent_after of them while it reports its drive. A
    ROBOT_STATE is made every state_s; with named_times it carries its own
    time and the time its last setpoint went out, as the robot app's does.
    With each_setpoint one is made behind every setpoint too, and the timed
    one a state_s after the last of those, as the robot app does while a
    person drives. With whole the pause stops
    the ROBOT_STATEs too: the whole robot app stands, not its drive alone.
    """

    def __init__(
        self,
        *,
        loop_s: float = PASS_S,
        phase_s: float = 0.007,
        state_s: float = 0.05,
        state_phase_s: float = 0.003,
        pause_after: int | None = None,
        pause_s: float = 0.0,
        silent_after: int | None = None,
        named_times: bool = True,
        each_setpoint: bool = False,
        whole: bool = False,
    ) -> None:
        super().__init__(START, owner="human", start_passes=START_READ)
        self.loop_s, self.state_s = loop_s, state_s
        self.next_tick, self.next_state = T0 + phase_s, T0 + state_phase_s
        self.pause_after, self.pause_s = pause_after, pause_s
        self.silent_after, self.named_times = silent_after, named_times
        self.each_setpoint, self.whole = each_setpoint, whole
        self.newest: HumanSetpoint | None = None
        self.target_t: float | None = None  # when the last setpoint went out
        self.silent_at: float | None = None
        self.states: list[tuple[float, RobotState]] = []  # (when made, the state)
        self._make(T0)
        self.total = 0  # setpoints sent

    def take(self, messages: Sequence[Message]) -> list[Result]:
        for message in of(messages, HumanSetpoint):
            self.newest = message
        return []

    def _make(self, at: float) -> None:
        """Make the ROBOT_STATE of this moment; each has the next seq."""
        made = dataclasses.replace(self.state(), seq=len(self.states) + 1)
        if self.named_times and self.states:
            made = dataclasses.replace(made, clock_mono_s=at, target_t=self.target_t)
        self.states.append((at, made))

    def advance(self, now: float) -> None:
        """Run the loop and make the ROBOT_STATEs that are due by now."""
        while min(self.next_tick, self.next_state) <= now:
            if self.next_tick <= self.next_state:
                self._tick(self.next_tick)
                continue
            self._make(self.next_state)
            self.next_state += self.state_s

    def _tick(self, at: float) -> None:
        self.next_tick = at + self.loop_s
        if self.newest is None:
            return
        if self.silent_after is not None and self.total >= self.silent_after:
            if self.silent_at is None:
                self.silent_at = at
            self.seq = self.newest.seq
            return
        if self.pause_after is not None and self.total >= self.pause_after:
            self.pause_after = None
            self.next_tick = at + self.pause_s  # the loop stands
            if self.whole:
                self.next_state = max(self.next_state, self.next_tick)
            return
        before = self.sent
        self._setpoint(self.newest)
        if self.sent != before:
            self.target_t, self.total = at, self.total + 1
            if self.each_setpoint:
                self._make(at)
                self.next_state = at + self.state_s


class Timed:
    """What a script showed on the rig's files against a TimedThor."""

    def __init__(self, rows: list[dict], thor: TimedThor, seconds: float) -> None:
        settings = rig_settings()
        ik = settings.ik
        graph = HumanGraph(settings.retarget)
        solver = Solver.from_apai(
            ik.apai_arm, max_iter=ik.max_iter, tol=ik.tol, max_jump_rad=ik.max_jump_rad
        )
        seen = [thor.states[0][1]]
        feed = ScriptFeed(graph, rows, state=lambda: seen[0])
        logic = HumanLogic(graph, solver, settings, HUMAN_SOURCE_SYNTHETIC)
        named = welcome(human_max_joint_speed=thor.speed)
        self.thor, self.logic = thor, logic
        self.notes: list[str] = []
        self.targets: list[list[float]] = []
        self.ended_at: float | None = None  # when the logic ended an engagement
        now = T0
        for _ in range(round(seconds / PASS_S)):
            thor.advance(now)
            made_at, seen[0] = thor.states[-1]
            events = graph.events(stopped=not logic.armed)
            fed = feed.step(graph.external_inputs(logic.armed), events, now)
            messages = logic.step(now, fed, named, seen[0], now - made_at, [])
            logic.log_pass()
            for message in of(messages, HumanSetpoint):
                self.notes.append(message.note)
                if message.engaged:
                    self.targets.append(list(message.q))
                elif self.targets and self.ended_at is None:
                    self.ended_at = now
            thor.take(messages)
            now += PASS_S
        counts = logic.counts()
        self.ended = {key: counts[key] for key in OWN_ENDS if counts[key]}
        self.needs_release = counts["needs_release"]
        steps = np.abs(np.diff(np.array(self.targets), axis=0)).max(axis=1)
        self.moving_s = float(np.sum(steps > 1e-6)) * PASS_S
        home = solver.fk(solver.inside(START))[:3, 3]
        self.moved = solver.fk(thor.q)[:3, 3] - home

    def followed(self) -> bool:
        """Nothing ended, the Thor refused nothing, and the arm is at the hand."""
        there = np.allclose(self.moved, 0.30 * tip_way(WAYS["left"]), atol=1e-3)
        return not self.ended and not self.thor.refused and bool(there)


FAST_LEFT = moving("left", 0.30, 1.0)  # 30 cm in 0.3 s: the joints at their limit
BEHIND = "the robot app is behind: the arm follows slower"


@pytest.mark.parametrize("phase_s", [0.001, 0.007, 0.013, 0.019])
@pytest.mark.parametrize("state_phase_s", [0.003, 0.023, 0.043])
def test_a_thor_on_time_is_followed_as_fast_as_before_on_its_own_clock(
    phase_s, state_phase_s
):
    # The Thor's loop and its ROBOT_STATEs, one every 0.05 s, at any phase
    # to the human program's passes: the hold-back never holds the joint
    # target, which is on its way exactly as long as under the stand-in that
    # answers within the pass.
    run = Timed(FAST_LEFT, TimedThor(phase_s=phase_s, state_phase_s=state_phase_s), 4.0)
    assert run.followed() and run.needs_release == 0
    assert not any(BEHIND in note for note in run.notes)
    on_time = drive_on_the_rig(FAST_LEFT, 4.0)
    assert run.moving_s == pytest.approx(float(np.sum(on_time.steps > 1e-6)) * PASS_S)
    assert max(run.thor.steps) == pytest.approx(JOINT_SPEED_SHARE * run.thor.clamp)


PAUSES = [
    (passes, after, phase_s)
    for passes in (3, 4, 5, 10)
    for after in (20, 21, 22, 23, 24)
    for phase_s in (0.003, 0.013)
]


def test_a_pause_on_the_thors_own_clock_is_ridden_out_where_a_state_shows_it_in_time():
    # The Thor's loop stands for 3, 4, 5 and 10 of its periods at full joint
    # speed, begun at ten moments to the ROBOT_STATEs, which come every 0.05 s.
    # A state shows the pause once it was made a period and a half after the
    # Thor's last setpoint. Where that state comes before the joint target is
    # 0.10 rad ahead, the pause costs speed only.
    outcomes = {
        case: Timed(
            FAST_LEFT,
            TimedThor(
                phase_s=case[2], pause_after=case[1], pause_s=(case[0] + 1) * PASS_S
            ),
            4.5,
        )
        for case in PAUSES
    }
    lost = {case: run for case, run in outcomes.items() if not run.followed()}
    for run in outcomes.values():
        if run.followed():
            assert run.needs_release == 0 and max(run.thor.steps) < run.thor.step_limit
    # The known limit: a pause that begins just after a ROBOT_STATE was made
    # is shown by the next one, 0.05 s later, and by then three passes have
    # moved the joint target on. 4 of these 40 pauses: the Thor refused two,
    # at its step limit, and the lead ended two; the arm holds either way.
    assert sorted(lost) == [
        (3, 23, 0.013),
        (4, 23, 0.013),
        (5, 23, 0.013),
        (10, 23, 0.013),
    ]
    for (passes, _, _), run in lost.items():
        if passes < 5:
            assert run.ended == {} and len(run.thor.refused) == 1
            assert run.thor.step_limit < run.thor.refused[0] < 0.12
        else:
            assert run.ended == {"solution_ahead": 1} and run.thor.refused == []
        assert not run.thor.driving and run.thor.motion == "hold"
    print(f"\npauses ridden out: {len(PAUSES) - len(lost)} of {len(PAUSES)}")


# The robot app sends a ROBOT_STATE behind every setpoint of a person's
# drive. When the next one is overdue, one and a half times the time from
# one to the next in this drive, the joint target is given no more for the
# time since the last: it stands two of the Thor's steps ahead of the Thor's
# reported joint target, whatever stands on the Thor, its drive or all of it.
STALLS = [
    (passes, after, phase_s, whole)
    for passes in (3, 4, 5)
    for after in (20, 22, 24)
    for phase_s in (0.001, 0.005, 0.009, 0.013, 0.017)
    for whole in (False, True)
]


def test_a_stall_of_a_thor_that_sends_a_state_with_every_setpoint_is_ridden_out():
    # 60, 80 and 100 ms at full joint speed, begun at 15 moments to the human
    # program's passes; the Thor's drive alone, and with its ROBOT_STATEs.
    assert logic_module.STATE_DUE_PERIODS == 1.5
    largest = 0.0
    for passes, after, phase_s, whole in STALLS:
        thor = TimedThor(
            phase_s=phase_s,
            pause_after=after,
            pause_s=(passes + 1) * PASS_S,
            each_setpoint=True,
            whole=whole,
        )
        run = Timed(FAST_LEFT, thor, 4.5)
        case = (passes, after, phase_s, whole)
        assert run.followed() and run.needs_release == 0, (case, run.ended)
        largest = max(largest, *thor.steps)
    # At the Thor's return the joint target lay at most two of the Thor's
    # steps from the Thor's last one: 0.063 rad, 0.037 rad inside the Thor's
    # step limit. While all is well it is up to two of its own steps ahead,
    # 0.056 rad, and the pass on which the ROBOT_STATE is overdue adds no
    # more than what is left of those two steps of the Thor.
    room = logic_module.HOLD_BACK_STEPS * thor.clamp
    assert largest <= room + 1e-9 and room < thor.step_limit - 0.037
    print(f"\n{len(STALLS)} stalls ridden out: the Thor's largest step {largest:.4f}")


@pytest.mark.parametrize("phase_s", [0.001, 0.007, 0.013, 0.019])
def test_a_thor_that_sends_a_state_with_every_setpoint_is_followed_as_fast(phase_s):
    # No ROBOT_STATE is overdue while the Thor's loop is on time: nothing is
    # held, and the arm is on its way as long as under the stand-in that
    # answers within the pass.
    run = Timed(FAST_LEFT, TimedThor(phase_s=phase_s, each_setpoint=True), 4.0)
    assert run.followed() and run.needs_release == 0
    assert not any(BEHIND in note for note in run.notes)
    on_time = drive_on_the_rig(FAST_LEFT, 4.0)
    assert run.moving_s == pytest.approx(float(np.sum(on_time.steps > 1e-6)) * PASS_S)
    assert max(run.thor.steps) == pytest.approx(JOINT_SPEED_SHARE * run.thor.clamp)


@pytest.mark.parametrize("loop_ms", [22, 25, 27, 40])
def test_a_thor_whose_loop_runs_slow_is_followed_at_its_pace(loop_ms):
    # The Thor moves one clamped step per pass of its own loop, which takes
    # 22, 25, 27 and 40 ms in place of 20. The human program reads the pace
    # off two ROBOT_STATEs and holds its joint target back to it.
    run = Timed(FAST_LEFT, TimedThor(loop_s=loop_ms / 1000.0), 6.0)
    assert run.followed() and run.needs_release == 0
    assert max(run.thor.steps) < run.thor.step_limit
    on_time = Timed(FAST_LEFT, TimedThor(), 6.0)
    # A loop at 22 ms still keeps up with 0.9 of its clamp; a slower one
    # costs speed, and the note says so.
    slower = run.moving_s - on_time.moving_s
    said = any(BEHIND in note for note in run.notes)
    if loop_ms == 22:
        assert slower == pytest.approx(0.0, abs=1e-9) and not said
    else:
        assert slower > 0.05 and said
    assert slower <= (loop_ms / 20.0 - 0.9) * on_time.moving_s
    print(f"\na loop of {loop_ms} ms: on its way {run.moving_s:.2f} s")


def test_a_slow_loop_that_names_no_times_is_left_to_the_thors_step_limit():
    # A robot app whose ROBOT_STATE names no send time of its joint target:
    # its pace cannot be read. Down to 27 ms the hold-back still keeps the
    # joint target inside the step limit; at 40 ms the Thor refuses.
    for loop_ms in (25, 27):
        run = Timed(
            FAST_LEFT, TimedThor(loop_s=loop_ms / 1000.0, named_times=False), 6.0
        )
        assert run.followed()
    run = Timed(FAST_LEFT, TimedThor(loop_s=0.040, named_times=False), 6.0)
    assert run.ended == {} and len(run.thor.refused) == 1


def test_a_thor_that_sends_no_more_setpoints_ends_the_engagement_within_a_second():
    # It reports its drive as running and its loop sends nothing from its
    # 21st setpoint on.
    run = Timed(FAST_LEFT, TimedThor(silent_after=20), 4.0)
    thor = run.thor
    assert run.ended == {"solution_ahead": 1} and thor.refused == []
    assert run.notes[-1] == (
        "release the grip and squeeze again: the arm does not follow: no new "
        "setpoint of the robot app in 1 s"
    )
    # The joint target stood two of the Thor's steps ahead of the Thor's own.
    ahead = np.abs(np.array(run.targets[-1]) - np.array(thor.target_q)).max()
    assert ahead == pytest.approx(logic_module.HOLD_BACK_STEPS * thor.clamp)
    # The engagement ended within the bound, and the ROBOT_STATE's age, of
    # the first setpoint that did not come.
    took = run.ended_at - thor.silent_at
    assert logic_module.NO_FOLLOW_S < took <= logic_module.NO_FOLLOW_S + 0.15
    print(f"\nno more setpoints: ended {took:.2f} s later, {ahead:.4f} rad ahead")


# Near a singular pose a short step of the tip asks a joint for a long one.
# FOLDED: a little above the rest pose. From START the arm's reach ends 20 cm
# higher: 19 cm up it is stretched out.
FOLDED = (0.0, 0.1, 0.1, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "start, way, metres",
    [(FOLDED, "forward", 0.05), (START, "up", 0.19)],
    ids=["folded", "stretched out"],
)
def test_near_a_singular_pose_the_tip_slows_and_the_drive_goes_on(start, way, metres):
    thor = RigThor(start, owner="human", start_passes=START_READ)
    rig = Rig(moving(way, metres, 0.5), thor, rig_settings(), state_passes=4)
    solves: list[str] = []
    solve = rig.solver.solve

    def counted(pose: np.ndarray, seed: Sequence[float]) -> Any:
        solved = solve(pose, seed)
        solves.append(solved.problem)
        return solved

    rig.solver.solve = counted
    rig.run(0.4 + metres / 0.5 + 3.0)
    drive = Drive(rig, thor, start)
    drive.healthy()
    # Some steps of the tip asked a joint for more than ik.max_jump_rad: they
    # were halved, and no jump was left to end the engagement.
    jumps = sum(problem.startswith("IK jump") for problem in solves)
    assert jumps >= 5 and drive.counts["jump"] == 0
    # The joints ran at their limit and the tip came behind the hand, slower
    # than in the open: at half the hand's 50 cm a second or less.
    assert drive.at_the_cap()
    moving_tip = drive.tip_speeds[drive.tip_speeds > 1e-4]
    assert float(np.median(moving_tip)) < 0.25
    moved = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(moved, metres * tip_way(WAYS[way]), atol=1e-3)
    print(f"\n{way} from {start}: {jumps} of {len(solves)} solves were jumps")


# The owner's speeds of 2026-11-18 on the rig's files: a hand at the hand
# speed limit, 50 cm a second, is followed with the fastest joint at its
# limit, 0.9 of the Thor's clamp of 1.6 rad/s, from three poses, and no
# engagement ends for it. READY: the arm raised over the table. STRETCHED:
# reaching forward, the tip 2.7 cm above the table plane. NEAR: folded, the
# tip near the arm's base.
READY = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
STRETCHED = (0.0, 1.5, 0.6, 0.0, 0.3, 0.0)
NEAR = (0.0, 0.3, 0.3, 0.0, 0.0, 0.0)
POSES = {"ready": READY, "stretched": STRETCHED, "near": NEAR}


@every_way
@pytest.mark.parametrize("pose", list(POSES))
def test_a_hand_at_50_cm_a_second_is_followed_at_the_joint_speed_limit(pose, way):
    metres, start = 0.10, POSES[pose]
    thor = RigThor(start, owner="human", start_passes=START_READ)
    drive = drive_on_the_rig(
        moving(way, metres, 0.5),
        0.4 + 0.2 + 2.5,
        start=start,
        state_passes=4,
        thor=thor,
    )
    drive.healthy()
    rig = drive.rig
    assert JOINT_SPEED_SHARE * thor.clamp == pytest.approx(0.9 * 0.98 * 1.6 * PASS_S)
    assert drive.at_the_cap() and drive.tip_speeds.max() < 0.5
    # The arm arrived where the hand stopped and stands there. From NEAR the
    # ways down and back end at the edge of the arm's reach, with two joints
    # near their stops: the tip comes within 5 mm of the hand's point.
    edge = pose == "near" and way in ("down", "back")
    moved = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(
        moved, metres * tip_way(WAYS[way]), atol=5e-3 if edge else 1e-3
    )
    assert drive.steps[-25:].max() < 1e-6
    assert rig.logic.stand == ("out of reach" if edge else "")
    print(f"\n{pose}, {way}: the tip at up to {drive.tip_speeds.max():.2f} m/s")


# The table and the reach on the rig's files. The Thor names the box and the
# joint box of a person on the rig, and the table plane as its table
# clearance: the tip stops 0.5 cm above the plane and moves freely above it.


def hand_to(tip: Sequence[float]) -> np.ndarray:
    """Where the hand is, in the stage space, for a tip moved by this from the squeeze."""
    x, y, z = tip
    return HAND + np.array([-y, z, -x])


def legs(*moves: tuple[Sequence[float], float]) -> list[dict]:
    """A script: a squeeze at 0.2 s; from 0.4 s the hand goes from one point to the next.

    Each move is where the tip is asked to be, from the squeeze's, and how
    long the hand takes to get there, s.
    """
    rows, at = [BASE, {"t": 0.2, "squeeze": 1.0}, {"t": 0.4}], 0.4
    for tip, seconds in moves:
        at += seconds
        rows.append({"t": at, "pos": list(hand_to(tip))})
    return rows


def on_the_rig_table(rows: list[dict], seconds: float, start: Sequence[float]) -> Drive:
    """Run a script from start under a Thor that names the rig's box and table."""
    thor = RigThor(start, owner="human", start_passes=START_READ)
    rig = Rig(rows, thor, rig_settings(), state_passes=4)
    rig.welcome = dataclasses.replace(
        rig.welcome,
        human_tip_box=RIG_BOX,
        human_tip_floor=RIG_FLOOR,
        human_joint_box=RIG_JOINT_BOX,
    )
    rig.run(seconds)
    return Drive(rig, thor, start)


def test_on_the_rig_a_hand_that_pushes_down_stops_the_tip_over_the_table_and_slides():
    stop = RIG_FLOOR + logic_module.TIP_FLOOR_MARGIN_M
    assert stop == 0.005
    # From STRETCHED the hand goes 10 cm down in 0.2 s, far under the table,
    # then 15 cm to the left in 0.3 s while it stays down there.
    rows = legs(((0.0, 0.0, -0.10), 0.2), ((0.0, 0.15, -0.10), 0.3))
    drive = on_the_rig_table(rows, 0.4 + 0.5 + 2.5, STRETCHED)
    drive.healthy()
    rig = drive.rig
    tips = np.array([rig.tip(q) for q in drive.targets])
    assert tips[0, 2] == pytest.approx(0.027, abs=1e-3)
    # No joint target had its tip under the stop, within what a solve leaves
    # of its pose; the tip came down to it and went on to the left along it.
    slack = rig.settings.ik.tol
    assert stop - slack <= tips[:, 2].min() and tips[-1, 2] <= stop + slack
    np.testing.assert_allclose(tips[-1, :2] - tips[0, :2], [0.0, 0.15], atol=1e-3)
    at_stop = np.nonzero(tips[:, 2] < stop + 1e-3)[0]
    assert tips[at_stop[-1], 1] - tips[at_stop[0], 1] > 0.10
    # The note names the table while the hand is under the stop.
    assert engaged(rig.sent)[-1].note == "at the workspace limit: table"
    assert not any("raise" in m.note for m in of(rig.sent, HumanSetpoint))
    print(f"\nthe tip came down to {1000 * tips[:, 2].min():.2f} mm over the table")


def test_on_the_rig_a_squeeze_with_the_tip_at_the_table_starts_as_any_other():
    # The arm holds with the tip 2 mm over the table plane: under the stop,
    # as an arm that sags under a joint target at the stop.
    solver = Solver.from_apai("rebot_rs", max_jump_rad=1.0)
    low = solver.fk(solver.inside(STRETCHED))
    low[2, 3] = 0.002
    start = solver.solve(low, solver.inside(STRETCHED)).q
    assert solver.fk(start)[2, 3] == pytest.approx(0.002, abs=2e-4)
    # A level hand that goes 10 cm to the left at 20 cm a second.
    drive = on_the_rig_table(legs(((0.0, 0.10, 0.0), 0.5)), 0.4 + 0.5 + 2.0, start)
    drive.healthy()
    rig = drive.rig
    tips = np.array([rig.tip(q) for q in drive.targets])
    # The first joint target is the arm's pose. From the first pass the Thor
    # follows, the tip goes to the left, and up to the stop: no wait.
    moved = int(np.argmax(drive.steps > 1e-9))
    assert moved == START_READ + 4 - 1 and tips[0, 2] == pytest.approx(0.002, abs=2e-4)
    assert np.all(np.diff(tips[:, 2]) >= -1e-5)
    np.testing.assert_allclose(tips[-1], tips[0] + [0.0, 0.10, 0.003], atol=1e-3)
    assert not any("raise" in m.note for m in of(rig.sent, HumanSetpoint))


def test_on_the_rig_a_hand_beyond_the_joint_box_is_followed_along_it():
    # From STRETCHED the hand goes 40 cm forward in 0.8 s and then 15 cm to
    # the left in 0.3 s. The front wall of the tip box is 16 cm away; the
    # second joint comes to its bound for a person before the tip is there.
    rows = legs(((0.40, 0.0, 0.0), 0.8), ((0.40, 0.15, 0.0), 0.3))
    drive = on_the_rig_table(rows, 0.4 + 1.1 + 4.0, STRETCHED)
    rig, thor = drive.rig, drive.thor
    # Nothing ended the engagement, and the Thor refused nothing.
    assert drive.ended == {} and drive.counts["engagements"] == 1
    assert thor.refused == [] and drive.counts["needs_release"] == 0
    assert rig.logic.last.engaged and drive.steps.max() <= 0.9 * thor.clamp + 1e-9
    low = np.array(RIG_JOINT_BOX[0]) + logic_module.JOINT_BOX_MARGIN_RAD
    high = np.array(RIG_JOINT_BOX[1]) - logic_module.JOINT_BOX_MARGIN_RAD
    assert np.all(drive.targets >= low - 1e-9) and np.all(drive.targets <= high + 1e-9)
    tips = np.array([rig.tip(q) for q in drive.targets])
    # The tip went forward as far as the second joint allows and then to the
    # left with the hand, along that bound: never back from the hand's point
    # as the front wall holds it, 1 cm inside the box. Then the arm stands.
    front = RIG_BOX[1][0] - logic_module.TIP_BOX_MARGIN_M
    held = np.array([front, tips[0, 1] + 0.15, tips[0, 2]])
    off = np.linalg.norm(held - tips, axis=1)
    assert np.all(np.diff(off) <= 1e-4) and drive.steps[-25:].max() < 1e-9
    forward, left = tips[-1, 0] - tips[0, 0], tips[-1, 1] - tips[0, 1]
    assert 0.05 < forward < front - tips[0, 0] and left > 0.12
    assert max(q[1] for q in drive.targets) == pytest.approx(high[1])
    # The note names the joint; and the front wall while the tip was within
    # a centimetre of it.
    notes = [m.note for m in engaged(rig.sent)]
    assert notes[-1] == "at a joint limit: joint 2"
    assert "at the workspace limit: front; at a joint limit: joint 2" in notes
    note = notes[-1]
    print(
        f"\nbeyond the joint box: {100 * forward:.1f} cm forward, {100 * left:.1f} cm "
        f"to the left; the note: {note}"
    )


def test_on_the_rig_a_hand_out_of_reach_and_back_is_followed_with_no_new_squeeze():
    # Under a Thor that names no box: the reach of the arm model itself. The
    # hand goes 45 cm forward in 0.9 s, 20 cm to the left in 0.4 s, stays for
    # a second and comes back to 10 cm forward in 0.9 s.
    rows = legs(
        ((0.45, 0.0, 0.0), 0.9),
        ((0.45, 0.20, 0.0), 0.4),
        ((0.45, 0.20, 0.0), 1.0),
        ((0.10, 0.20, 0.0), 0.9),
    )
    thor = RigThor(STRETCHED, owner="human", start_passes=START_READ)
    rig = Rig(rows, thor, rig_settings(), state_passes=4)
    rig.run(0.4 + 2.3)
    out = engaged(rig.sent)[-1]
    reached = rig.tip(thor.q) - rig.tip(rig.solver.inside(STRETCHED))
    # The arm's reach ends 28 cm forward of the squeeze: the tip went on by
    # the nearest pose and to the left along the edge, and the note said so.
    assert "out of reach" in out.note and rig.logic.stand == "out of reach"
    assert 0.25 < reached[0] < 0.45 and reached[1] > 0.10
    rig.run(0.9 + 3.0)
    drive = Drive(rig, thor, STRETCHED)
    assert drive.ended == {} and drive.counts["engagements"] == 1
    assert thor.refused == [] and drive.counts["needs_release"] == 0
    assert drive.steps.max() <= 0.9 * thor.clamp + 1e-9
    # Back in reach the arm is at the hand's pose again, with the hand's
    # orientation, and the note is gone.
    back = rig.tip(thor.q) - drive.home[:3, 3]
    np.testing.assert_allclose(back, [0.10, 0.20, 0.0], atol=1e-3)
    assert drive.off_the_hand() == pytest.approx((0.0, 0.0), abs=2e-3)
    assert engaged(rig.sent)[-1].note == "" and rig.logic.stand == ""
    print(f"\nout of reach: the tip came {100 * reached[0]:.1f} cm forward")


# The files


def flat(block: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """The settings of a file by dotted key."""
    out: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def shipped(name: str) -> dict[str, Any]:
    return flat(yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8")))


def comment_above(text: str, key: str) -> str:
    """The comment lines right above a key of a YAML file, as one line."""
    lines = [line.strip() for line in text.splitlines()]
    [at] = [index for index, line in enumerate(lines) if line.startswith(f"{key}:")]
    words: list[str] = []
    while at > 0 and lines[at - 1].startswith("#"):
        at -= 1
        words.insert(0, lines[at].lstrip("# "))
    return " ".join(words)


# How the comment beside rates.loop_hz ends in both shipped files.
LOOP_HZ_BOUND = (
    "A pass that comes sooner than 10 ms after the one before counts as 10 ms; "
    "above 100 both hand speed limits would grow in proportion, so values above "
    "100 are refused."
)


def test_the_shipped_human_files_load():
    demo = load_human_settings(CONFIGS / "human.yaml")
    assert demo == HumanSettings()  # the first values
    assert demo.teleop == TeleopSettings() and demo.ik == IkSettings()
    assert demo.retarget == RetargetSettings()
    rig = load_human_settings(CONFIGS / "human_rebot.yaml")
    assert rig.ik.apai_arm == "rebot_rs" and rig.teleop.attempts == 8640

    # Each file names every setting there is.
    names = {"socket_path", "log_level", "buttons.hold_s", "state.max_age_s"}
    names |= {"rates.loop_hz", "rates.idle_hz"}
    for block, kind in (
        ("teleop", TeleopSettings),
        ("retarget", RetargetSettings),
        ("ik", IkSettings),
    ):
        names |= {f"{block}.{field.name}" for field in dataclasses.fields(kind)}
    for name in ("human.yaml", "human_rebot.yaml"):
        assert set(shipped(name)) == names, name
        text = (CONFIGS / name).read_text(encoding="utf-8")
        assert text.startswith("# SPDX-FileCopyrightText: ")
        beside = comment_above(text, "loop_hz")
        assert "LIMITER_PASS_S" in beside and beside.endswith(LOOP_HZ_BOUND), name

    # What each file asks for can be built: the graph, and the arm model.
    for settings in (demo, rig):
        HumanGraph(settings.retarget)
        solver = Solver.from_apai(settings.ik.apai_arm)
        assert solver.n_joints == 6
        assert settings.teleop.retry_s >= 2.0
        # The Thor shows the program's stage for longer than an attempt lasts.
        keep_s = RobotAppConfig.default().human.stage_keep_s
        assert settings.teleop.connect_timeout_s < keep_s


def test_the_rig_human_file_differs_from_the_shipped_one_in_eight_keys():
    demo, rig = shipped("human.yaml"), shipped("human_rebot.yaml")
    differing = {key: (demo[key], rig[key]) for key in demo if demo[key] != rig[key]}
    assert differing == {
        "socket_path": ("", "~/.agentic_remote/human.sock"),
        "teleop.endpoint": ("127.0.0.1:47600", "192.0.2.1:47600"),
        "teleop.attempts": (720, 8640),
        # The bounds of the tip's target, the owner's values of 2026-11-18:
        # with them the joints follow at 0.9 of the Thor's clamp of the
        # owner's 1.6 rad/s.
        "retarget.max_linear_velocity": (0.10, 0.5),
        "retarget.max_angular_velocity": (0.4, 2.0),
        "ik.apai_arm": ("mock", "rebot_rs"),
        # A pass of those bounds asks a joint for twice as much.
        "ik.max_jump_rad": (0.05, 0.10),
        # Below limits.human.max_step_rad of the rig's robot file, 0.10.
        "ik.max_ahead_rad": (0.08, 0.09),
    }
    # In both files a hand up to half a metre ahead of the arm is caught up
    # with: the one-machine file follows the code, as the first value does.
    assert demo["ik.max_ahead_m"] == rig["ik.max_ahead_m"] == 0.5
    assert IkSettings().max_ahead_m == 0.5
    text = (CONFIGS / "human_rebot.yaml").read_text(encoding="utf-8")
    assert "It differs from configs/human.yaml in eight keys:" in text
    # The file says in plain words what each of the new values does, and
    # whose value the 1.6 rad/s is.
    speed = comment_above(text, "max_linear_velocity")
    assert "the owner's value of 2026-11-18" in speed and "1.6 rad/s a joint" in speed
    assert "0.9 of the Thor's speed clamp, 1.41 rad/s" in speed
    assert "The fastest the gripper tip's target moves" in speed
    turn = comment_above(text, "max_angular_velocity")
    assert turn == (
        "The fastest the gripper tip's target turns: 0.04 rad a pass. The owner's "
        "value of 2026-11-18."
    )
    # The tip's speed as computed: slower from the ready pose than the bound,
    # up to the bound where the arm is stretched out and the tip goes sideways.
    assert "27 to 36 cm a second from the ready pose" in speed
    assert "up to the 50 cm a second of the bound below" in speed
    ahead = comment_above(text, "max_ahead_m")
    assert "the arm catches up with the hand and stops where the hand stopped" in ahead
    assert "the tip stops 1 cm inside a wall of it and slides along the wall" in ahead
    # The table and the joint box, which the Thor names as well.
    assert "the tip stops 0.5 cm above that and slides along it" in ahead
    assert "a joint stops 0.01 rad inside it" in ahead
    for stale in ("a tip under it only rises", "to be lifted", "0.045", "0.71"):
        assert stale not in text
    # The one-machine file says the same of the hand, and no longer what the
    # code did before.
    demo_text = (CONFIGS / "human.yaml").read_text(encoding="utf-8")
    assert comment_above(demo_text, "max_ahead_m") == (
        "The farthest the hand may be from the arm's gripper tip, in all. Up to it "
        "the arm catches up with the hand and stops where the hand stopped; a hand "
        "further away is not believed, and the engagement ends."
    )
    # A pose with no solution ends no engagement: both files say what the
    # arm does then, and what the count is for.
    for file_text in (text, demo_text):
        assert comment_above(file_text, "max_failures") == (
            'Passes in a row from which the note says "out of reach". A hand\'s pose '
            "that has no solution does not end the engagement: the arm follows as "
            "far as it can, the tip's position first and the wrist's angle second, "
            "and follows the hand again once its pose has a solution. A whole number."
        )
    for stale in ("rate-limited target", "raising them needs", "and ends the engage"):
        assert stale not in demo_text


@pytest.mark.parametrize("key", ["loop_hz", "idle_hz"])
def test_rates_that_are_not_above_0_are_refused(key, tmp_path, capsys):
    for value, said in (
        (0, "must be above 0"),
        (0.0, "must be above 0"),
        (-50.0, "must be above 0"),
        (float("inf"), "must be a finite number"),
        (float("nan"), "must be a finite number"),
        (True, "must be a number"),
        ("50", "must be a number"),
        (None, "must be a number"),
    ):
        with pytest.raises(HumanConfigError, match=f"rates.{key} {said}"):
            human_settings({"rates": {key: value}})
    assert getattr(human_settings({"rates": {key: 1}}), key) == 1.0

    # Through the program: exit code 2, before anything is attached.
    file = write_config(tmp_path, str(tmp_path / "nobody.sock"), rates={key: 0})
    assert human_main.main(["--config", str(file)], attach=never) == 2
    assert f"rates.{key} must be above 0" in capsys.readouterr().err


def loop_rate_refusal(value: Any) -> str:
    """How the settings refuse this rates.loop_hz: one line."""
    return (
        f"rates.loop_hz must be at most 100, got {value!r}: the rate limiter "
        "counts a pass as at least 10 ms, so above 100 both hand speed limits "
        "would rise above their settings"
    )


def test_a_loop_rate_above_100_is_refused(tmp_path, capsys):
    assert human_main.MAX_LOOP_HZ == 100.0
    for value in (100, 100.0, 99.5):
        assert human_settings({"rates": {"loop_hz": value}}).loop_hz == value
    for value in (100.5, 200, 1e6):
        with pytest.raises(HumanConfigError) as refused:
            human_settings({"rates": {"loop_hz": value}})
        assert str(refused.value) == loop_rate_refusal(value)
    # The bound is on the program's passes alone.
    assert human_settings({"rates": {"idle_hz": 200}}).idle_hz == 200.0
    assert human_settings({"teleop": {"tick_rate_hz": 120}}).teleop.tick_rate_hz == 120

    # Through the program: exit code 2 and one line, before anything is attached.
    file = write_config(tmp_path, str(tmp_path / "nobody.sock"), rates={"loop_hz": 200})
    assert human_main.main(["--config", str(file)], attach=never) == 2
    line = f"human program: the settings cannot be used: {loop_rate_refusal(200)}\n"
    assert capsys.readouterr().err == line


@pytest.mark.parametrize("rate, times", [(50, 1), (100, 1), (200, 2), (400, 4)])
def test_above_100_passes_per_second_the_hand_speed_limit_grows_in_proportion(
    rate, times
):
    # Why rates.loop_hz ends at MAX_LOOP_HZ: up to it the limited target moves
    # at retarget.max_linear_velocity, above it faster by rate / MAX_LOOP_HZ.
    assert times == max(1.0, rate / human_main.MAX_LOOP_HZ)
    settings = RetargetSettings()
    seconds, hand_speed = 0.2, 0.5  # the hand: above the limit, not rejected
    assert settings.max_linear_velocity < hand_speed < settings.reject_linear_velocity
    graph = GraphBench(settings)
    first = graph.engage().hand
    for index in range(1, round(seconds * rate) + 1):
        hand = along(first, UP, hand_speed * index / rate)
        step = graph.step(hand, squeeze=1.0, pass_ns=round(1e9 / rate))
        assert step.limiter == "clamped"
    speed = (step.ee_target[2, 3] - HOME[2, 3]) / seconds
    assert speed == pytest.approx(times * settings.max_linear_velocity, rel=0.01)


@pytest.mark.parametrize(
    "rate, early_ms, late_ms, times",
    [
        (100, 9.9, 10.1, 1.005),
        (100, 9.0, 11.0, 1.05),
        (66, 5.0, 25.3, 0.99),
        (50, 15.0, 25.0, 0.875),
    ],
)
def test_uneven_passes_raise_the_hand_speed_limit_only_near_100_passes_per_second(
    rate, early_ms, late_ms, times
):
    # A known limit of MAX_LOOP_HZ itself. The rate limiter counts a pass as
    # 10 ms at least and 20 ms at most. At 100 passes per second a pass that
    # comes early counts for more than its time, and the late one behind it
    # for no less: the limited target is faster than the setting by half the
    # share of that unevenness. With a period of 15 ms or more, up to 66 per
    # second, an early pass and a late one never count for more than their
    # time together; a late pass is not made up, so the target is slower.
    assert early_ms + late_ms == pytest.approx(2000.0 / rate, rel=0.002)
    settings = RetargetSettings()
    hand_speed = 0.5  # the hand: above the limit, not rejected
    graph = GraphBench(settings)
    first = graph.engage().hand
    elapsed_ns = 0
    for index in range(1, rate + 1):
        pass_ns = round((early_ms if index % 2 else late_ms) * 1e6)
        elapsed_ns += pass_ns
        hand = along(first, UP, hand_speed * elapsed_ns / 1e9)
        step = graph.step(hand, squeeze=1.0, pass_ns=pass_ns)
        assert step.limiter == "clamped"
    speed = (step.ee_target[2, 3] - HOME[2, 3]) / (elapsed_ns / 1e9)
    assert speed == pytest.approx(times * settings.max_linear_velocity, rel=0.002)


# A whole number of more digits than Python prints, and what the settings say
# in its place.
HUGE = 16**5000
TOO_LARGE = "a whole number too large to print"


@pytest.mark.parametrize(
    "data, said",
    [
        ({"speed": 1}, "unknown setting(s): speed"),
        ({"teleop": {"port": 1}}, "unknown setting(s): teleop.port"),
        ({"buttons": {"hold_s": 1.0, "tap_s": 1}}, "unknown setting(s): buttons.tap_s"),
        ({"state": {"max_age": 1}}, "unknown setting(s): state.max_age"),
        ({"ik": [1]}, "ik must be a mapping"),
        ([1], "the file must be a mapping"),
        ({"log_level": "loud"}, "log_level must be one of debug, info"),
        ({"socket_path": 3}, "socket_path must be text"),
        ({"teleop": {"endpoint": " "}}, "teleop.endpoint must be text"),
        ({"teleop": {"retry_s": 1.9}}, "teleop.retry_s must be at least 2"),
        ({"teleop": {"attempts": 0}}, "teleop.attempts must be a whole number"),
        ({"teleop": {"attempts": 7.5}}, "teleop.attempts must be a whole number"),
        ({"ik": {"max_iter": True}}, "ik.max_iter must be a whole number"),
        ({"ik": {"apai_arm": ""}}, "ik.apai_arm must be text"),
        ({"ik": {"max_jump_rad": 0}}, "ik.max_jump_rad must be above 0"),
        ({"buttons": {"hold_s": -1}}, "buttons.hold_s must be above 0"),
        ({"state": {"max_age_s": "1"}}, "state.max_age_s must be a number"),
        (
            {"retarget": {"controller_pose": "palm"}},
            "retarget.controller_pose must be one of grip, aim",
        ),
        (
            {"retarget": {"squeeze_threshold": 1.0}},
            "retarget.squeeze_threshold must be above 0 and below 1",
        ),
        (
            {"retarget": {"reject_linear_velocity": 0}},
            "retarget.reject_linear_velocity must be above 0",
        ),
        (
            {"retarget": {"controller_to_base": [[1, 0, 0], [0, 1, 0]]}},
            "retarget.controller_to_base must be 3 rows of 3 numbers",
        ),
        (
            {"retarget": {"controller_to_base": [[1, 0, 0], [0, 1, 0], [0, 0, "z"]]}},
            "retarget.controller_to_base[2][2] must be a number",
        ),
        # A whole number too large for a float, and one too large to print.
        (
            {"rates": {"loop_hz": 10**400}},
            "rates.loop_hz must be a finite number, got 1000",
        ),
        ({"buttons": {"hold_s": -(10**400)}}, "buttons.hold_s must be a finite number"),
        ({"state": {"max_age_s": HUGE}}, f"must be a finite number, got {TOO_LARGE}"),
        ({"teleop": {"attempts": -HUGE}}, f"number of 1 or more, got {TOO_LARGE}"),
        ({"ik": {"apai_arm": HUGE}}, f"ik.apai_arm must be text, got {TOO_LARGE}"),
        ({"log_level": HUGE}, f"warning, error, got {TOO_LARGE}"),
        ({"ik": HUGE}, f"ik must be a mapping, got {TOO_LARGE}"),
        ({"retarget": {"controller_to_base": [HUGE]}}, f"numbers, got {TOO_LARGE}"),
        ({"buttons": {"hold_s": [HUGE]}}, f"must be a number, got {TOO_LARGE}"),
        ({HUGE: 1}, f"unknown setting(s): {TOO_LARGE}"),
    ],
)
def test_a_setting_the_file_does_not_know_or_cannot_use_is_refused(data, said):
    with pytest.raises(HumanConfigError, match=re.escape(said)):
        human_settings(data)


def test_what_python_cannot_read_or_print_in_the_file_ends_with_exit_code_2(
    tmp_path, capsys
):
    finite = "rates.loop_hz must be a finite number, got "
    for value, said in (
        # Too large for a float.
        ("1" + "0" * 400, finite + "1000"),
        # In hexadecimal, of more digits than Python prints.
        ("0x" + "f" * 5000, finite + TOO_LARGE),
        # In decimal, of more digits than Python reads.
        ("1" + "0" * 5000, "cannot read "),
        # Not a number, and not UTF-8 either.
        (b"\xff\xfe", "cannot read "),
    ):
        file = tmp_path / f"human_{len(value)}.yaml"
        if isinstance(value, bytes):
            file.write_bytes(b"rates:\n  loop_hz: " + value + b"\n")
        else:
            file.write_text(f"rates:\n  loop_hz: {value}\n", encoding="utf-8")
        with pytest.raises(HumanConfigError):
            load_human_settings(file)
        # Through the program: exit code 2 and one line, cut to a line's length.
        assert human_main.main(["--config", str(file)], attach=never) == 2
        error = capsys.readouterr().err
        assert error.startswith("human program: the settings cannot be used: ")
        assert said in error and error.count("\n") == 1 and len(error) < 500, error


def test_a_file_names_only_what_differs_from_the_first_values(tmp_path):
    assert human_settings(None) == HumanSettings() == human_settings({})
    turned = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    settings = human_settings(
        {
            "socket_path": "~/robot.sock",
            "log_level": "debug",
            "teleop": {"endpoint": "10.0.0.2:47600", "attempts": 3, "retry_s": 2},
            "retarget": {
                "controller_pose": "aim",
                "reject_linear_velocity": None,
                "controller_to_base": turned,
                "stage_yaw_deg": -90,
            },
            "ik": {"apai_arm": "rebot_rs", "max_failures": 9},
            "buttons": {"hold_s": 2},
            "rates": {"loop_hz": 25, "idle_hz": 10},
            "state": None,  # a block with nothing in it
        }
    )
    assert settings == HumanSettings(
        socket_path=os.path.expanduser("~/robot.sock"),
        log_level="debug",
        teleop=TeleopSettings(endpoint="10.0.0.2:47600", attempts=3, retry_s=2.0),
        retarget=RetargetSettings(
            controller_pose="aim",
            reject_linear_velocity=None,
            controller_to_base=((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            stage_yaw_deg=-90.0,
        ),
        ik=IkSettings(apai_arm="rebot_rs", max_failures=9),
        hold_s=2.0,
        loop_hz=25.0,
        idle_hz=10.0,
    )
    assert isinstance(settings.loop_hz, float) and settings.teleop.attempts == 3
    with pytest.raises(HumanConfigError, match=r"cannot read .*missing\.yaml"):
        load_human_settings(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("rates: [", encoding="utf-8")
    with pytest.raises(HumanConfigError, match=r"cannot read .*broken\.yaml"):
        load_human_settings(broken)


# The program


def never(*args: Any) -> Any:
    """In place of main()'s attach: the program must not get that far."""
    raise AssertionError(f"the program attached: {args}")


@pytest.fixture(autouse=True)
def log_levels(monkeypatch) -> list[str]:
    """The console log levels main() asked for; none is set in a test."""
    levels: list[str] = []
    monkeypatch.setattr(console_log, "set_console_level", levels.append)
    return levels


@pytest.fixture
def folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("hp")  # a short path, mode 0700


@pytest.fixture
def bench(folder) -> Iterator[Bench]:
    made = Bench(folder)
    yield made
    made.close()


def write_config(folder: Path, socket_path: str, **blocks: Any) -> Path:
    """A human.yaml of the test: its socket, NO_SERVER, else the first values."""
    file = folder / f"human_{len(list(folder.iterdir()))}.yaml"
    settings = {"socket_path": socket_path, **blocks}
    # No test dials the Teleop Session Server's own port, whatever it builds.
    settings["teleop"] = {"endpoint": NO_SERVER, **settings.get("teleop", {})}
    file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return file


def robot(
    bench: Bench,
    driver: str = "sim",
    joints: int = 6,
    speed: float | None = HUMAN_SPEED,
    **state: Any,
) -> FakeThor:
    """A robot connection that has sent its WELCOME and one ROBOT_STATE.

    speed is the human joint speed limit the WELCOME names; None: it names none.
    """
    thor = bench.dial()
    until(lambda: bench.service._carrier is thor.service_end, "the robot connection")
    names = [f"j{index}" for index in range(1, joints + 1)]
    named = welcome(human_max_joint_speed=speed)
    thor.send(dataclasses.replace(named, driver=driver, joint_names=names))
    show(bench, thor, **{"owner": "nobody", **state})
    return thor


def show(bench: Bench, thor: FakeThor, **state: Any) -> RobotState:
    """Send a ROBOT_STATE; the service has it on return."""
    sent = robot_state(**state)
    thor.send(sent)
    until(lambda: bench.service.robot_state == sent, "the ROBOT_STATE")
    return sent


class Forwarded:
    """The link messages the service forwarded to a scripted Thor, kept in order.

    Of a newest-wins link message, such as HUMAN_SETPOINT, only the newest one
    not yet read is there.
    """

    def __init__(self, thor: FakeThor) -> None:
        self._thor = thor
        self.messages: list[Message] = []

    def of(self, kind: type) -> list[Any]:
        """Every one of this kind so far."""
        self.messages += [decode(*record) for record in self._thor.drain()]
        return [message for message in self.messages if isinstance(message, kind)]


class Running:
    """The human program's main() on a thread, until the test ends it."""

    def __init__(self, config: Path, *args: str, **keywords: Any) -> None:
        self.said: list[str] = []
        self.said_at: list[float] = []  # when each line was said
        self.code: int | None = None
        self.stop = threading.Event()
        argv = ["--config", str(config), *args]
        self._thread = threading.Thread(target=self._run, args=(argv, keywords))
        self._thread.start()

    def _say(self, line: str) -> None:
        self.said_at.append(time.monotonic())
        self.said.append(line)

    def _run(self, argv: list[str], keywords: dict[str, Any]) -> None:
        # A test may give its own say in place of the one that keeps the lines.
        keywords = {"say": self._say, **keywords}
        self.code = human_main.main(argv, stop=self.stop, **keywords)

    def wait_said(self, part: str, count: int = 1) -> str:
        """Wait for the line number `count` that contains `part`."""

        def found() -> list[str]:
            return [line for line in list(self.said) if part in line]

        end = time.monotonic() + WAIT_S
        while len(found()) < count:
            assert time.monotonic() < end, f"no line {count} with {part!r}: {self.said}"
            assert self._thread.is_alive(), (self.code, self.said)
            time.sleep(0.005)
        return found()[count - 1]

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def end(self) -> int | None:
        """End the program as Ctrl+C does; its exit code."""
        self.stop.set()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the human program did not end"
        return self.code


@pytest.fixture
def start() -> Iterator[Callable[..., Running]]:
    """Starts human programs; each is ended when the test is over."""
    made: list[Running] = []

    def run(config: Path, *args: str, **keywords: Any) -> Running:
        made.append(Running(config, *args, **keywords))
        return made[-1]

    yield run
    for program in made:
        program.end()


def run_main(*argv: str, **keywords: Any) -> int:
    """main() on this thread, for a program that is to end by itself.

    One that does not is ended after WAIT_S through its stop event: main()
    then returns 0, which no test of an early end expects.
    """
    stop = threading.Event()
    watchdog = threading.Timer(WAIT_S, stop.set)
    watchdog.start()
    try:
        return human_main.main(list(argv), stop=stop, **keywords)
    finally:
        watchdog.cancel()


class HeadsetScript(ScriptFeed):
    """A script feed that says it is the headset path, as the teleop feed does."""

    source = HUMAN_SOURCE_TELEOP


# The program's second line, with the shipped file and the stand-in robot
# app's WELCOME: how fast the arm follows the hand. The joint speed is 0.9 of
# the robot app's speed clamp, 0.98 of the limit its WELCOME names.
SPEED_LINE = (
    "human program: the arm follows the hand at up to 1.32 rad/s a joint (the "
    "robot app's limit for a person is 1.5 rad/s), its tip at up to 0.1 m/s and "
    "0.4 rad/s"
)
SPEED_LINE_UNNAMED = (
    "human program: the arm follows the hand at up to 0.09 rad/s a joint (the "
    "robot app names no speed limit for a person), its tip at up to 0.1 m/s and "
    "0.4 rad/s"
)
# Its third line: where the gripper tip is held. The stand-in robot app names
# no box for it, as the robot app on the simulated arm.
BOX_LINE_UNNAMED = (
    "human program: the robot app names no workspace box for the gripper tip: a "
    "target outside its limits ends the drive"
)
# With the box of a person on the rig, as the rig's robot app names it: the
# tip stops 1 cm inside each of its walls.
RIG_BOX = [[0.10, -0.50, -0.01], [0.55, 0.50, 0.70]]
BOX_LINE_RIG_NO_TABLE = (
    "human program: the gripper tip stops at the robot app's workspace limits: "
    "x 0.110 to 0.540, y -0.490 to 0.490, z 0.000 to 0.690 m from the arm's "
    "base; it names no table clearance: a tip lowered under it ends the drive"
)
# With the height the rig's robot app names for a person's table clearance,
# the table plane: the tip stops 0.5 cm above it.
RIG_FLOOR = 0.0
BOX_LINE_RIG = (
    "human program: the gripper tip stops at the robot app's workspace limits: "
    "x 0.110 to 0.540, y -0.490 to 0.490, z 0.005 to 0.690 m from the arm's "
    "base; its lowest is 0.5 cm above the table clearance"
)
# Its fourth line: where the joints are held. The stand-in robot app names no
# joint box for a person, as a robot app older than this line.
JOINTS_LINE_UNNAMED = (
    "human program: the robot app names no joint limits for a person: a joint "
    "target outside them ends the drive"
)
RIG_JOINT_BOX = [
    [-2.0744, -0.05, -0.05, -1.67, -1.55, -2.0744],
    [2.0744, 2.0744, 2.0744, 1.77, 1.55, 2.0744],
]
JOINTS_LINE = (
    "human program: the joints stop at the robot app's joint limits for a "
    "person, and the arm follows as far as it can while the hand is out of reach"
)


def test_the_start_line_names_the_feed_the_driver_and_the_model(
    bench, folder, start, caplog
):
    caplog.set_level("INFO", logger=human_main._log.name)
    config = write_config(folder, bench.path)
    program = start(config, "--controller", "synthetic:idle")
    # Nothing is said before the robot app's WELCOME.
    time.sleep(0.3)
    assert program.said == [] and program.running
    thor = robot(bench, driver="apai-mock")
    line = program.wait_said("human program: ")
    assert line == (
        'human program: synthetic controller "idle"; driver apai-mock; '
        "arm model mock, 6 joints"
    )
    assert program.said[0] == line
    # Then how fast the arm follows the hand under this robot app.
    assert HUMAN_SPEED == 1.5
    assert JOINT_SPEED_SHARE * SPEED_MARGIN * HUMAN_SPEED == pytest.approx(1.323)
    assert program.wait_said("human program: the arm follows") == SPEED_LINE
    assert program.said[1] == SPEED_LINE
    # And where the gripper tip is held: this robot app names no box.
    assert program.wait_said("workspace") == BOX_LINE_UNNAMED
    assert program.said[2] == BOX_LINE_UNNAMED
    # And where the joints are held: it names no joint box for a person.
    assert program.wait_said("joint limits") == JOINTS_LINE_UNNAMED
    assert program.said[3] == JOINTS_LINE_UNNAMED
    # Once: a second WELCOME of that robot app says nothing new.
    thor.send(dataclasses.replace(welcome(), driver="apai-mock"))
    program.wait_said("right controller seen")
    time.sleep(0.2)
    assert [text for text in program.said if "arm model" in text] == [line]
    assert [text for text in program.said if "follows" in text] == [SPEED_LINE]
    assert [text for text in program.said if "workspace" in text] == [BOX_LINE_UNNAMED]
    said = [text for text in program.said if "joint limits" in text]
    assert said == [JOINTS_LINE_UNNAMED]
    assert program.end() == 0
    # A synthetic controller has no Teleop Session Server to speak of.
    assert not any("Teleop Session Server" in text for text in program.said)
    # The WELCOME was checked once: no pass took it for a new session.
    logged = [record.getMessage() for record in caplog.records]
    # The socket may be the robot link service's or the robot app's own.
    attached = "attached to the robot link service or the robot app; "
    assert logged.count(attached + "waiting for a WELCOME") == 1
    assert not any("new session" in text for text in logged)


def test_the_start_line_of_the_headset_path_and_its_stage_lines(bench, folder, start):
    rows = [BASE, {"t": 0.4, "absent": True}, {"t": 0.8, "absent": False}]
    robot(bench, driver="")
    program = start(
        write_config(folder, bench.path),
        feed=lambda graph, state: HeadsetScript(graph, rows, state=state),
    )
    program.wait_said("Teleop Session Server ready", 2)
    assert program.end() == 0
    lines = [text for text in program.said if not is_report(text)]
    assert lines == [
        "human program: teleop; driver not declared (treated as real); "
        "arm model mock, 6 joints",
        SPEED_LINE,
        BOX_LINE_UNNAMED,
        JOINTS_LINE_UNNAMED,
        "human program: Teleop Session Server ready",
        "human program: right controller seen",
        "human program: Teleop Session Server no_controller: "
        "the synthetic controller is absent",
        "human program: right controller lost",
        "human program: Teleop Session Server ready",
        "human program: right controller seen",
    ]


def test_the_speed_line_follows_the_limit_each_robot_app_names(bench, folder, start):
    # A robot app whose WELCOME names no human speed limit: the arm follows
    # at 0.9 of the clamp of 0.1 rad/s.
    thor = robot(bench, speed=None)
    program = start(write_config(folder, bench.path), "--controller", "synthetic:idle")
    assert program.wait_said("human program: the arm follows") == SPEED_LINE_UNNAMED
    assert program.wait_said("workspace") == BOX_LINE_UNNAMED
    assert program.wait_said("joint limits") == JOINTS_LINE_UNNAMED
    # The robot app starts again with the rig's limit and the rig's box for
    # the gripper tip, as a robot app that names neither a table clearance
    # nor a joint box: the lines are said anew.
    again = dataclasses.replace(
        welcome("s2", human_max_joint_speed=1.6), driver="sim", human_tip_box=RIG_BOX
    )
    thor.send(again)
    show(bench, thor, owner="nobody", session_id="s2")
    line = program.wait_said("human program: the arm follows the hand at up to 1.41")
    assert line == (
        "human program: the arm follows the hand at up to 1.41 rad/s a joint (the "
        "robot app's limit for a person is 1.6 rad/s), its tip at up to 0.1 m/s and "
        "0.4 rad/s"
    )
    assert program.wait_said("workspace limits") == BOX_LINE_RIG_NO_TABLE
    # And once more as the rig's robot app of now: with both.
    full = dataclasses.replace(
        again,
        session_id="s3",
        human_tip_floor=RIG_FLOOR,
        human_joint_box=RIG_JOINT_BOX,
    )
    thor.send(full)
    show(bench, thor, owner="nobody", session_id="s3")
    assert program.wait_said("table clearance", 2) == BOX_LINE_RIG
    assert program.wait_said("the joints stop") == JOINTS_LINE
    assert program.end() == 0
    said = [text for text in program.said if "the arm follows the hand" in text]
    assert said == [SPEED_LINE_UNNAMED, line, line]
    said = [text for text in program.said if "workspace" in text]
    assert said == [BOX_LINE_UNNAMED, BOX_LINE_RIG_NO_TABLE, BOX_LINE_RIG]
    said = [text for text in program.said if "joint limits" in text]
    assert said == [JOINTS_LINE_UNNAMED, JOINTS_LINE_UNNAMED, JOINTS_LINE]


def test_each_button_edge_is_printed_once(bench, folder, start):
    changes = (
        {"a": True},
        {"a": False},
        {"b": True},
        {"b": False},
        # Below retarget.squeeze_threshold, 0.5, and back: no edge.
        {"squeeze": 0.4},
        {"squeeze": 0.0},
        {"squeeze": 0.6},
        {"squeeze": 0.0},
        # The trigger's edge is at 0.5 too.
        {"trigger": 0.4},
        {"trigger": 0.0},
        {"trigger": 0.6},
        {"trigger": 0.0},
        {"a": True, "b": True},
        {"absent": True},
    )
    rows = [BASE]
    for index, change in enumerate(changes):
        rows.append({"t": 0.2 + 0.15 * index, **change})
    thor = robot(bench)
    program = start(write_config(folder, bench.path), feed=script(rows))
    program.wait_said("right controller lost")
    time.sleep(0.2)
    assert program.end() == 0
    edges = [text for text in program.said if EDGE.match(text)]
    assert edges == [
        "human program: A down",
        "human program: A up",
        "human program: B down",
        "human program: B up",
        "human program: grip down",
        "human program: grip up",
        "human program: trigger down",
        "human program: trigger up",
        "human program: A down",
        "human program: B down",
        # A controller that is not active reads every button as up.
        "human program: A up",
        "human program: B up",
    ]
    seen = [text for text in program.said if "right controller" in text]
    assert seen == [
        "human program: right controller seen",
        "human program: right controller lost",
    ]
    # Each press of right B was a STOP on its way to the Thor.
    stops = Forwarded(thor).of(Stop)
    assert stops == [Stop(reason="right B", sender="controller")] * 2


def script(rows: Sequence[dict[str, Any]]) -> Callable[..., ScriptFeed]:
    """main()'s feed keyword for a script feed on these rows."""
    return lambda graph, state: ScriptFeed(graph, rows, state=state)


def test_the_report_line_is_printed_once_per_period_and_not_with_report_0(
    bench, folder, start
):
    robot(bench, owner="agent")
    config = write_config(folder, bench.path)
    program = start(config, "--controller", "synthetic:idle", "--report-s", "0.25")
    program.wait_said("human program: stage ", 5)
    assert program.end() == 0
    began = program.said_at[0]  # the start line: the loop begins
    reports = [
        (at, text) for at, text in zip(program.said_at, program.said) if is_report(text)
    ]
    # One line per period, from the start on.
    times = [at for at, _ in reports]
    gaps = np.diff([began, *times])
    assert len(reports) >= 5 and np.all(gaps > 0.15) and np.all(gaps < 1.0), gaps
    assert abs(len(reports) - (times[-1] - began) / 0.25) <= 1.5
    for _, text in reports:
        found = REPORT.fullmatch(text)
        assert found is not None, text
        assert (found["stage"], found["owner"]) == ("ready", "agent")
        # The script's sample age is 0; nothing engages, so nothing is solved.
        assert (found["tick"], found["engaged"], found["ik"]) == ("0/0/0", "no", "-/-")
        assert found["trip"] == "-" and found["refused"] is None
        assert "hand x 0.000 y 1.200 z -0.300 m" in text

    # With --report-s 0 there is none.
    silent = start(config, "--controller", "synthetic:idle", "--report-s", "0")
    silent.wait_said("right controller seen")
    time.sleep(0.8)
    assert silent.end() == 0
    assert not any(is_report(text) for text in silent.said)


def test_the_report_line_shows_the_thors_refusal(bench, folder, start):
    thor = robot(bench, owner="human", round_trip_s=0.0214)
    program = start(
        write_config(folder, bench.path),
        *("--controller", "synthetic:idle", "--report-s", "0.2"),
    )
    plain = program.wait_said("human program: stage ")
    found = REPORT.fullmatch(plain)
    assert found is not None and found["refused"] is None, plain
    assert (found["owner"], found["trip"]) == ("human", "21")

    refusal = "the live view is not fresh: the view is not live"
    show(bench, thor, owner="human", refusal=refusal)
    count = len([text for text in program.said if is_report(text)])
    line = program.wait_said("human program: stage ", count + 2)
    assert line.endswith(f"; refused by the Thor: {refusal}"), line
    assert REPORT.fullmatch(line) is not None

    # It goes with the refusal.
    show(bench, thor, owner="human", refusal="")
    count = len([text for text in program.said if is_report(text)])
    line = program.wait_said("human program: stage ", count + 2)
    assert "refused by the Thor" not in line and REPORT.fullmatch(line) is not None


def test_the_words_of_the_report_line():
    turn = np.eye(4)
    # A quarter turn about z, the tip 30 cm forward and 20 cm up.
    turn[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    turn[:3, 3] = [0.3, 0.0, 0.2]
    last = GraphStep(
        present=True,
        squeeze=1.0,
        trigger=0.0,
        a=False,
        b=False,
        engaged=True,
        ee_target=turn,
        commanded_target=turn,
        limiter="passed",
        hand=np.array([0.01, 1.2345, -0.3]),
    )
    state = robot_state(round_trip_s=0.0316, refusal="")
    counts = {
        "no_solution": 1,
        "jump": 2,
        "solution_ahead": 3,
        "hand_ahead": 4,
        "limiter": 5,
        "released": 9,
    }
    line = report_line(
        "ready",
        [0.012, 0.004, 0.0081],
        [0.03, 0.01, 0.02, 0.04],
        state,
        last,
        [0.0002, 0.0001, 0.0015],
        counts,
    )
    assert line == (
        "human program: stage ready; tick age 4/8/12 ms; state age 10/25/40 ms; "
        "round trip 32 ms; owner human; engaged yes; "
        "hand x 0.010 y 1.234 z -0.300 m; "
        "tip target x 0.300 y 0.000 z 0.200 m, roll 0.0 pitch 0.0 yaw 90.0 deg; "
        "IK 0.20/1.50 ms; ended: no solution 1, jump 2, solution ahead 3, "
        "hand ahead 4, limiter 5"
    )
    assert REPORT.fullmatch(line) is not None

    # A value that is not there prints as "-".
    nothing = report_line("no_server", [], [], None, None, [], {})
    assert nothing == (
        "human program: stage no_server; tick age -/-/- ms; state age -/-/- ms; "
        "round trip - ms; owner -; engaged no; hand x - y - z - m; "
        "tip target x - y - z - m, roll - pitch - yaw - deg; IK -/- ms; "
        "ended: no solution 0, jump 0, solution ahead 0, hand ahead 0, limiter 0"
    )
    assert REPORT.fullmatch(nothing) is not None
    refused = report_line(
        "ready", [], [], robot_state(refusal="no fresh\tcontroller data"), None, [], {}
    )
    assert refused.endswith("; refused by the Thor: no fresh controller data")
    # Without the robot app's jaw facts the line says nothing of the jaws.
    assert "jaws" not in line + nothing + refused

    # While the arm follows a hand that is 5 cm or more ahead of it, the line
    # says how far, in whole cm, ahead of what it says of the jaws.
    def with_ahead(metres: float | None) -> str:
        times = [0.0002, 0.0001, 0.0015]
        ages = [0.012, 0.004, 0.0081], [0.03, 0.01, 0.02, 0.04]
        return report_line("ready", *ages, state, last, times, counts, metres)

    assert with_ahead(None) == with_ahead(0.0) == with_ahead(0.044) == line
    assert with_ahead(float("nan")) == line
    behind = with_ahead(0.1234)
    assert behind == line + "; hand 12 cm ahead: the arm is catching up"
    assert with_ahead(0.0496) == line + "; hand 5 cm ahead: the arm is catching up"
    assert REPORT.fullmatch(behind)["ahead"] == "12" and REPORT.fullmatch(line)
    jaws = robot_state(jaws=jaw_facts(), refusal="busy")
    both = report_line("ready", [], [], jaws, last, [], counts, 0.2)
    found = REPORT.fullmatch(both)
    assert (found["ahead"], found["refused"]) == ("20", "; refused by the Thor: busy")
    assert both.index("; hand 20 cm ahead") < both.index("; jaws follow the trigger")
    # Ahead of that: the walls of the robot app's tip box that the tip stands
    # at, and that the joint target is held back for a slow robot app. The
    # line holds all three, which a note does not.
    held = report_line(
        "ready", [], [], state, last, [], counts, 0.07, ("front", "top"), True
    )
    assert held.endswith(
        "; at the workspace limit: front, top; the robot app is behind: the arm "
        "follows slower; hand 7 cm ahead: the arm is catching up"
    )
    found = REPORT.fullmatch(held)
    assert (found["walls"], found["ahead"]) == ("front, top", "7") and found["behind"]
    at_wall = report_line("ready", [], [], state, last, [], counts, 0.0, ("left",))
    assert at_wall.endswith("limiter 5; at the workspace limit: left")
    assert REPORT.fullmatch(at_wall)["walls"] == "left"
    assert REPORT.fullmatch(line)["walls"] is None
    assert REPORT.fullmatch(line)["behind"] is None
    # Between the walls and the rest: why the joint target stands though the
    # hand goes on, as the note says it.
    for stand in ("out of reach", "at a joint limit: joint 3"):
        stood = report_line(
            "ready", [], [], state, last, [], counts, 0.07, ("table",), False, stand
        )
        assert stood.endswith(
            f"; at the workspace limit: table; {stand}; hand 7 cm ahead: the arm "
            "is catching up"
        )
        found = REPORT.fullmatch(stood)
        assert (found["walls"], found["stand"]) == ("table", stand)
    assert REPORT.fullmatch(line)["stand"] is None


def jaw_facts(**facts: Any) -> dict[str, Any]:
    """health["human"]["jaws"] as the robot app gives it while a takeover is open."""
    return {
        "matched": False,
        "grip_level": None,
        "opened": False,
        "asked": None,
        "refusal": "",
        **facts,
    }


@pytest.mark.parametrize(
    "facts, said, why",
    [
        (
            jaw_facts(grip_level=0.0, asked=1.0, refusal=NOT_MATCHED),
            "; jaws follow the trigger no, grip level 0.00, trigger asks 1.00"
            ": the trigger has not reached the grip level",
            ": the trigger has not reached the grip level",
        ),
        (
            jaw_facts(matched=True, grip_level=0.3, asked=0.456, opened=True),
            "; jaws follow the trigger yes, grip level 0.30, trigger asks 0.46",
            None,
        ),
        (
            jaw_facts(refusal="the grip level is not known:\tno reading"),
            "; jaws follow the trigger no, grip level -, trigger asks -"
            ": the grip level is not known: no reading",
            ": the grip level is not known: no reading",
        ),
        # Values that are no numbers print as "-".
        (
            jaw_facts(grip_level=float("nan"), asked=True),
            "; jaws follow the trigger no, grip level -, trigger asks -",
            None,
        ),
        (
            jaw_facts(grip_level="0.5", asked=16**400),
            "; jaws follow the trigger no, grip level -, trigger asks -",
            None,
        ),
    ],
    ids=["not reached", "following", "no grip level", "no numbers", "no floats"],
)
def test_the_report_line_says_whether_the_jaws_follow_the_trigger(facts, said, why):
    def line(**human: Any) -> str:
        return report_line("ready", [], [], robot_state(**human), None, [], {})

    plain = line()
    with_jaws = line(jaws=facts)
    # The jaw part comes behind the line of today and ahead of the refusal.
    assert with_jaws == plain + said
    found = REPORT.fullmatch(with_jaws)
    assert found is not None and (found["why"], found["refused"]) == (why, None)
    assert found["following"] == ("yes" if facts["matched"] else "no")
    refused = line(jaws=facts, refusal="no fresh controller data")
    assert refused == plain + said + "; refused by the Thor: no fresh controller data"
    found = REPORT.fullmatch(refused)
    assert found is not None and found["why"] == why
    assert found["refused"] == "; refused by the Thor: no fresh controller data"
    # Jaw facts that are no mapping say nothing.
    assert line(jaws="matched") == plain and line(jaws=None) == plain


@pytest.mark.parametrize("driver", ["real", ""])
def test_the_program_refuses_a_synthetic_controller_against_a_driver_that_is_not_simulated(
    driver, bench, folder, capsys
):
    thor = robot(bench, driver=driver)
    config = write_config(folder, bench.path)
    said: list[str] = []
    for argv, keywords in (
        (["--controller", "synthetic:idle"], {}),
        # The feed's source decides, also for one given to main().
        ([], {"feed": script([BASE])}),
    ):
        code = run_main("--config", str(config), *argv, say=said.append, **keywords)
        assert code == 1 and said == []
        error = capsys.readouterr().err
        assert error.count("\n") == 1
        assert "a synthetic controller may drive only a simulated arm" in error
        assert ("driver real" if driver else "no driver") in error
    # Nothing went to the Thor, and the program has left the service.
    until(lambda: not bench.service.roles().get("human"), "the program's leaving")
    assert thor.drain() == []


def test_a_new_session_of_a_real_arm_ends_a_synthetic_controller(
    bench, folder, start, capsys
):
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=script([BASE]))
    program.wait_said("right controller seen")
    # The robot app comes back as another session, on the real driver.
    thor.send(dataclasses.replace(welcome("s2"), driver="real"))
    until(lambda: not program.running, "the program's end")
    assert program.code == 1
    assert "a synthetic controller may drive only" in capsys.readouterr().err
    # The person owned the arm, as far as the program knew: one STOP.
    until(lambda: seen.of(Stop), "the STOP")
    time.sleep(0.2)
    [stop] = seen.of(Stop)
    assert stop.sender == "human_program"
    assert stop.reason.startswith("the human program ends (")


def test_an_arm_model_that_does_not_fit_welcome_ends_with_exit_code_1(
    bench, folder, capsys
):
    thor = robot(bench, joints=5)
    config = write_config(folder, bench.path)
    said: list[str] = []
    code = run_main("--config", str(config), feed=script([BASE]), say=said.append)
    assert code == 1 and said == []
    error = capsys.readouterr().err
    assert "the arm model 'mock' has 6 joints" in error and "names 5" in error
    time.sleep(0.1)
    assert thor.drain() == []


def test_a_second_human_program_ends_with_exit_code_1(bench, folder, start, capsys):
    robot(bench)
    config = write_config(folder, bench.path)
    first = start(config, "--controller", "synthetic:idle")
    first.wait_said("arm model")
    said: list[str] = []
    code = run_main(
        "--config", str(config), "--controller", "synthetic:idle", say=said.append
    )
    assert code == 1 and said == []
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and error.startswith("human program: cannot attach")
    assert "a human program is attached already; one at a time" in error
    # The first one runs on.
    assert first.running and bench.service.roles().get("human") == 1
    assert first.end() == 0


def test_without_a_robot_link_service_the_program_ends_with_exit_code_1(folder, capsys):
    config = write_config(folder, str(folder / "nobody.sock"))
    said: list[str] = []
    code = run_main(
        "--config", str(config), "--controller", "synthetic:idle", say=said.append
    )
    assert code == 1 and said == []
    error = capsys.readouterr().err
    # The socket may be the robot link service's or the robot app's own.
    assert error.startswith(
        "human program: cannot attach to the robot link service or the robot app: "
    )


def test_the_program_ends_with_exit_code_1_when_the_teleop_attempts_run_out(
    bench, folder, capsys
):
    thor = robot(bench, driver="real")  # the headset path may drive a real arm
    seen = Forwarded(thor)
    said: list[str] = []
    # Bound and not listening: the port is this test's, and a dial is refused.
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{held.getsockname()[1]}"
        teleop = {
            "endpoint": endpoint,
            "connect_timeout_s": 0.3,
            "retry_s": 2.0,
            "attempts": 2,
        }
        config = write_config(folder, bench.path, teleop=teleop)
        started = time.monotonic()
        # --controller is teleop unless given: Isaac Capture's own remote session.
        code = run_main("--config", str(config), say=said.append)
        took = time.monotonic() - started
    assert code == 1 and 2.3 < took < WAIT_S
    error = capsys.readouterr().err
    assert f"human program: no remote session at {endpoint}: 2 attempts in a" in error
    lines = [text for text in said if not is_report(text)]
    assert lines[0] == "human program: teleop; driver real; arm model mock, 6 joints"
    assert lines[1:4] == [SPEED_LINE, BOX_LINE_UNNAMED, JOINTS_LINE_UNNAMED]
    assert lines[4].startswith(
        "human program: Teleop Session Server no_server: Teleop Session Server "
        f"at {endpoint} was not ready after 0.3 s: server did not answer"
    )
    assert len(lines) == 5
    # Meanwhile the Thor was told the stage, for the operator console.
    status = seen.of(HumanSetpoint)[-1]
    assert (status.source, status.stage) == ("teleop", "no_server")
    assert status.sample_age_s is None and not status.engaged and status.q is None
    assert status.note.startswith("Teleop Session Server at ")
    assert seen.of(Stop) == []


def test_ctrl_c_while_owning_sends_one_stop(bench, folder, start):
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    config = write_config(folder, bench.path)
    program = start(config, "--controller", "synthetic:idle")
    program.wait_said("right controller seen")
    # The program sends its HUMAN_SETPOINTs, and no STOP of its own accord.
    until(lambda: seen.of(HumanSetpoint), "a HUMAN_SETPOINT")
    assert seen.of(Stop) == []
    assert program.end() == 0
    until(lambda: seen.of(Stop), "the STOP")
    time.sleep(0.2)
    assert seen.of(Stop) == [END_STOP]

    # Not while somebody else owns the arm, or nobody.
    for owner in ("agent", "nobody"):
        # A higher stop number: the service has seen the Thor take the STOP.
        show(bench, thor, owner=owner, stop_number=5)
        setpoints = len(seen.of(HumanSetpoint))
        other = start(config, "--controller", "synthetic:idle")
        other.wait_said("right controller seen")
        end = time.monotonic() + WAIT_S
        while len(seen.of(HumanSetpoint)) == setpoints:
            assert time.monotonic() < end, "no HUMAN_SETPOINT was forwarded"
            time.sleep(0.01)
        assert other.end() == 0
        time.sleep(0.2)
        assert seen.of(Stop) == [END_STOP]


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_ends_the_program_as_the_stop_event_does(number, bench, folder):
    from link_stack import Program

    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    config = write_config(folder, bench.path)
    # The program in a process of its own: only a main thread waits for signals.
    program = Program(MODULE, "--config", str(config), "--controller", "synthetic:idle")
    try:
        program.wait_line("human program: right controller seen")
        until(lambda: seen.of(HumanSetpoint), "a HUMAN_SETPOINT")
        program.signal(number)
        assert program.wait_exit() == 0
        until(lambda: seen.of(Stop), "the STOP")
        time.sleep(0.2)
        reason = f"the human program ends ({number.name})"
        assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
    finally:
        program.close()


def test_a_standard_output_nobody_reads_holds_up_neither_the_passes_nor_the_end(
    bench, folder
):
    from link_stack import program_env

    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    config = write_config(folder, bench.path)
    # The program's standard output: a pipe of one page that nobody reads, as
    # a terminal whose output is paused. A report line per pass fills it.
    read_end, write_end = os.pipe()
    size = fcntl.fcntl(write_end, fcntl.F_SETPIPE_SZ, 4096)
    if size > 4096:  # a larger memory page: the pipe would not fill in time
        os.close(read_end)
        os.close(write_end)
        pytest.skip(f"the smallest pipe here holds {size} bytes, not 4096")
    argv = [sys.executable, "-m", MODULE, "--config", str(config)]
    argv += ["--controller", "synthetic:idle", "--report-s", "0.02"]
    errors = open(folder / "stderr.txt", "w+", encoding="utf-8")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=write_end,
        stderr=errors,
        env=program_env(),
    )
    os.close(write_end)
    try:
        until(lambda: seen.of(HumanSetpoint), "a HUMAN_SETPOINT")
        time.sleep(1.0)  # 50 report lines: several times what the pipe holds
        seq = seen.of(HumanSetpoint)[-1].seq
        # The passes go on behind the output that blocks.
        until(lambda: seen.of(HumanSetpoint)[-1].seq >= seq + 25, "more passes")
        assert seen.of(Stop) == []
        # So does the program's end: one STOP, then exit code 0.
        process.send_signal(signal.SIGINT)
        assert process.wait(WAIT_S) == 0
        until(lambda: seen.of(Stop), "the STOP")
        time.sleep(0.2)
        reason = "the human program ends (SIGINT)"
        assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
        errors.seek(0)
        said = errors.read()
        assert "Fatal Python error" not in said and "Traceback" not in said
        # What the pipe took: the program's first lines, and nothing since.
        os.set_blocking(read_end, False)
        written = os.read(read_end, 1 << 16).decode("utf-8")
        assert written.startswith('human program: synthetic controller "idle"; ')
        assert len(written) <= size and written.count("human program: stage ") < 25
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
            process.wait(WAIT_S)
        os.close(read_end)
        errors.close()


def test_a_feed_given_to_main_takes_the_place_of_the_controller_argument(
    bench, folder, start, monkeypatch
):
    def no_teleop(*args: Any, **keywords: Any) -> Any:
        raise AssertionError("the teleop feed was built")

    monkeypatch.setattr(feed_module, "TeleopFeed", no_teleop)
    given: list[tuple[Any, Any]] = []

    def feed(graph: Any, newest_state: Any) -> ScriptFeed:
        given.append((graph, newest_state))
        return ScriptFeed(graph, [{**BASE, "squeeze": 1.0}], state=newest_state)

    thor = robot(bench)
    # --controller is left at teleop.
    program = start(write_config(folder, bench.path), feed=feed)
    line = program.wait_said("arm model")
    assert line.startswith('human program: synthetic controller "rows"; driver sim')
    # The program runs on the given feed's controller.
    program.wait_said("human program: grip down")
    [(graph, newest_state)] = given
    assert isinstance(graph, HumanGraph)
    # newest_state() gives the newest ROBOT_STATE.
    shown = show(bench, thor, owner="agent", owner_number=7)
    until(lambda: newest_state() == shown, "the newest ROBOT_STATE")
    assert program.end() == 0


def test_the_graph_runs_stopped_and_unpermitted_while_the_logic_is_not_armed(
    bench, folder, start
):
    seen: list[tuple[str, bool]] = []  # (execution state, engage permitted) per pass

    class Watched(ScriptFeed):
        def step(self, external_inputs: Any, events: Any, now_mono: float) -> FeedStep:
            permitted = bool(external_inputs[PERMITTED_LEAF]["value"][0])
            seen.append((events.execution_state.value, permitted))
            return super().step(external_inputs, events, now_mono)

    thor = robot(bench, owner="agent")
    ended, silent = threading.Event(), threading.Event()
    owner = ["agent"]

    def keep_fresh() -> None:
        # The logic arms only on a ROBOT_STATE younger than state.max_age_s.
        while not ended.wait(0.02):
            if not silent.is_set():
                thor.send(robot_state(owner=owner[0], stop_number=1))

    feeder = threading.Thread(target=keep_fresh, daemon=True)
    feeder.start()
    try:
        program = start(
            write_config(folder, bench.path, state={"max_age_s": 0.5}),
            feed=lambda graph, state: Watched(graph, [BASE], state=state),
        )
        program.wait_said("right controller seen")
        time.sleep(0.3)
        # Somebody else owns the arm: every pass ran stopped, nothing permitted.
        assert len(seen) > 5 and set(seen) == {("stopped", False)}
        # The person owns it and the grip is released: armed, from the next pass.
        owner[0] = "human"
        until(lambda: seen[-1] == ("running", True), "an armed pass")
        # The ROBOT_STATEs stop: the newest one grows old, and with it the arming.
        silent.set()
        until(lambda: seen[-1] == ("stopped", False), "a pass on an old state")
        silent.clear()
        until(lambda: seen[-1] == ("running", True), "an armed pass")
        owner[0] = "agent"
        until(lambda: seen[-1] == ("stopped", False), "a pass that is not armed")
        # Never the one without the other.
        assert set(seen) == {("stopped", False), ("running", True)}
        assert program.end() == 0
    finally:
        ended.set()
        feeder.join(WAIT_S)


def test_every_result_reaches_the_logic_and_a_refusal_shows_in_the_note(
    bench, folder, start
):
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)  # read by the Thor's thread alone
    shown = {"owner": "agent", "stop_number": 1}
    requests: list[OwnerRequest] = []
    notes: list[str] = []
    ended = threading.Event()

    def thor_side() -> None:
        """Show the STOP in ROBOT_STATE and refuse every request."""
        while not ended.wait(0.02):
            thor.send(robot_state(**shown))
            if seen.of(Stop):
                shown.update(owner="nobody", stop_number=2, stop_level="stop")
            for request in seen.of(OwnerRequest)[len(requests) :]:
                requests.append(request)
                thor.send(Result(request.command_id, "refused", "not today"))
            notes[:] = [message.note for message in seen.of(HumanSetpoint)]

    side = threading.Thread(target=thor_side, daemon=True)
    side.start()
    try:
        start(
            write_config(folder, bench.path, state={"max_age_s": 1.0}),
            feed=script([BASE, {"t": 0.3, "b": True}]),
        )
        until(lambda: "takeover refused: not today" in notes, "the refusal's note")
        [request] = requests
        asked = (request.request, request.reason, request.stop_number)
        assert asked == ("human", "right B held 1.0 s", 2)
        assert NOTE_TAKE_OVER in notes  # what the note said before
    finally:
        ended.set()
        side.join(WAIT_S)


def test_a_pass_that_fails_in_the_logic_does_not_end_the_program(
    bench, folder, start, monkeypatch, caplog
):
    real = HumanLogic.step
    calls: list[None] = []

    def step(self: HumanLogic, *args: Any, **keywords: Any) -> list[Message]:
        calls.append(None)
        if 3 < len(calls) <= 13:
            raise ValueError("the solver broke")
        return real(self, *args, **keywords)

    monkeypatch.setattr(HumanLogic, "step", step)
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    with caplog.at_level("ERROR", logger=human_main._log.name):
        program = start(write_config(folder, bench.path), feed=script([BASE]))
        until(lambda: len(calls) > 30, "the passes after the failed ones")
        # It runs on, and so does right B: its HUMAN_SETPOINTs go out again.
        before = len(seen.of(HumanSetpoint))
        until(lambda: len(seen.of(HumanSetpoint)) > before, "a HUMAN_SETPOINT")
        assert program.running and program.end() == 0
    # Ten passes failed within a quarter of a second: logged, and not ten times.
    failed = [r for r in caplog.records if "a pass failed" in r.getMessage()]
    assert 1 <= len(failed) <= 3 and failed[0].exc_info[0] is ValueError


def test_right_b_is_a_stop_on_a_pass_whose_request_cannot_be_built(
    bench, folder, start, monkeypatch, caplog
):
    def broken(self: HumanLogic, p: Any) -> None:
        raise RuntimeError("no request")

    # Every pass fails right behind its buttons, where the logic builds the
    # OWNER_REQUEST of a held button.
    monkeypatch.setattr(HumanLogic, "_request", broken)
    # The agent owns the arm: right B is the wearer's STOP of the agent's move.
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    name = logic_module._log.name
    with caplog.at_level("ERROR", logger=name):
        program = start(
            write_config(folder, bench.path), feed=script([BASE, {"t": 0.3, "b": True}])
        )
        until(lambda: seen.of(Stop), "the STOP of right B")
        program.wait_said("human program: B down")
        time.sleep(0.2)  # ten more passes with B down
        assert program.running and program.end() == 0
    # One STOP for the stretch, and no other link message of a pass that failed.
    assert seen.of(Stop) == [RIGHT_B] and seen.of(HumanSetpoint) == []
    told = [record for record in caplog.records if record.name == name]
    assert [record.getMessage() for record in told] == [
        "the pass failed after its buttons were read"
    ]
    assert told[0].exc_info[0] is RuntimeError


def test_the_link_messages_of_a_pass_go_out_in_the_logics_order(
    bench, folder, start, monkeypatch
):
    from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient

    sent: list[Message] = []  # what the program's client was given, in order
    real = WorkstationClient.send

    def send(self: Any, message: Message) -> None:
        sent.append(message)
        real(self, message)

    monkeypatch.setattr(WorkstationClient, "send", send)
    thor = robot(bench, owner="human")
    ended = threading.Event()

    def keep_fresh() -> None:
        # The person owns the arm, on a ROBOT_STATE that stays young.
        while not ended.wait(0.02):
            thor.send(robot_state(owner="human"))

    feeder = threading.Thread(target=keep_fresh, daemon=True)
    feeder.start()
    try:
        # The squeeze engages; a second later right B ends the engagement.
        rows = [BASE, {"t": 0.4, "squeeze": 1.0}, {"t": 1.4, "b": True}]
        program = start(
            write_config(folder, bench.path, state={"max_age_s": 1.0}),
            feed=script(rows),
        )
        program.wait_said("human program: B down")
        until(lambda: of(sent, Stop), "the STOP of right B")
        time.sleep(0.1)
        assert program.end() == 0
    finally:
        ended.set()
        feeder.join(WAIT_S)
    at = sent.index(Stop(reason="right B", sender="controller"))
    before, behind = sent[at - 1], sent[at + 1]
    # The STOP went out ahead of the HUMAN_SETPOINT of its own pass: right
    # behind the last joint target, and before the first one without.
    assert isinstance(before, HumanSetpoint) and before.engaged
    assert isinstance(behind, HumanSetpoint) and not behind.engaged
    assert behind.q is None and behind.seq == before.seq + 1
    assert engaged(sent[at:]) == []


def test_a_standard_output_that_blocks_holds_up_no_stop(
    bench, folder, start, monkeypatch
):
    monkeypatch.setattr(human_main, "_WRITER_JOIN_S", 0.3)
    # The agent owns the arm: right B is the wearer's STOP of the agent's move.
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    blocked, release = threading.Event(), threading.Event()
    taken: list[str] = []

    def say(line: str) -> None:
        taken.append(line)
        if is_report(line):
            blocked.set()
            release.wait(3 * WAIT_S)  # the terminal takes no more output

    program = start(
        write_config(folder, bench.path),
        *("--report-s", "0.2"),
        feed=script([BASE, {"t": 0.8, "b": True}]),
        say=say,
    )
    try:
        assert blocked.wait(WAIT_S)
        setpoints = len(seen.of(HumanSetpoint))
        # The passes go on behind the line that blocks: right B is a STOP.
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert seen.of(Stop) == [Stop(reason="right B", sender="controller")]
        assert len(seen.of(HumanSetpoint)) > setpoints
        # The stop event ends the program while its output still blocks.
        assert program.running and program.end() == 0
        assert is_report(taken[-1]) and not release.is_set()
        assert len([line for line in taken if is_report(line)]) == 1
    finally:
        release.set()


def test_a_line_that_cannot_be_said_does_not_end_the_program(
    bench, folder, start, caplog
):
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    tried: list[str] = []

    def say(line: str) -> None:
        tried.append(line)
        raise BrokenPipeError("the terminal is gone")

    with caplog.at_level("WARNING", logger=human_main._log.name):
        program = start(
            write_config(folder, bench.path),
            feed=script([BASE, {"t": 0.3, "b": True}]),
            say=say,
        )
        # From the start line on no line can be said; right B is a STOP still.
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert seen.of(Stop) == [Stop(reason="right B", sender="controller")]
        assert program.running and program.end() == 0
    assert tried[0].startswith('human program: synthetic controller "rows"')
    assert "human program: B down" in tried
    # Logged once for the stretch, not once per line.
    warned = [r for r in caplog.records if "cannot be written" in r.getMessage()]
    assert len(warned) == 1 and "the terminal is gone" in warned[0].getMessage()


class PausedLog(logging.Handler):
    """A log handler a test can pause, on the loggers of the program's passes.

    It stands in for Isaac Capture's console handler on a terminal that is
    paused, or on a standard error nobody reads: that handler writes on the
    thread that logs, so the log call waits there.
    """

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.pause = threading.Event()  # set by a test: no line is taken
        self.resume = threading.Event()  # set by a test: lines are taken again
        self.waiting = threading.Event()  # a log call waits
        self.held: list[str] = []  # the lines whose log call waited
        self.written: list[str] = []  # the lines that were written, in order

    def emit(self, record: logging.LogRecord) -> None:
        line = record.getMessage()
        if self.pause.is_set() and not self.resume.is_set():
            self.held.append(line)
            self.waiting.set()
            self.resume.wait(3 * WAIT_S)
        self.written.append(line)


@pytest.fixture
def paused_log() -> Iterator[PausedLog]:
    """A PausedLog on the three loggers a pass of the program logs with."""
    handler = PausedLog()
    loggers = [human_main._log, logic_module._log, feed_module._log]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        handler.resume.set()
        for logger, level in zip(loggers, levels):
            logger.removeHandler(handler)
            logger.setLevel(level)


@pytest.fixture
def handed(monkeypatch) -> list[Message]:
    """What the program's client is given to send, in order.

    The robot link service drops a HUMAN_SETPOINT behind a STOP that no
    ROBOT_STATE shows yet: what it forwards is not all that was sent.
    """
    from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient

    sent: list[Message] = []
    real = WorkstationClient.send

    def send(self: Any, message: Message) -> None:
        sent.append(message)
        real(self, message)

    monkeypatch.setattr(WorkstationClient, "send", send)
    return sent


class ScriptedSession:
    """In place of a TeleopSession: a script's controller data, 10 ms old."""

    def __init__(self, graph: Any, rows: Sequence[dict[str, Any]]) -> None:
        self._script = ScriptFeed(graph, rows)
        self.last_step_info = SimpleNamespace(remote_tick_age_s=0.01)
        self.lost = threading.Event()  # set by a test: the next step loses it

    def __enter__(self) -> ScriptedSession:
        return self

    def __exit__(self, *error: object) -> bool:
        return False

    def step(self, *, external_inputs: Any, execution_events: Any) -> Any:
        if self.lost.is_set():
            raise RuntimeError("Remote session lost: connection closed")
        now = time.monotonic()
        return self._script.step(external_inputs, execution_events, now).outputs


def test_right_b_is_a_stop_ahead_of_the_log_line_of_its_pass(
    bench, folder, start, paused_log
):
    line = "right B is down: STOP sent"
    # The agent owns the arm: right B is the wearer's STOP of the agent's move.
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    program = start(
        write_config(folder, bench.path), feed=script([BASE, {"t": 0.6, "b": True}])
    )
    program.wait_said("right controller seen")
    paused_log.pause.set()  # from here on the terminal takes no log line
    try:
        # The pass that reads right B has a line to log, and its call waits ...
        assert paused_log.waiting.wait(WAIT_S)
        # ... behind the STOP of that pass, which is on its way to the Thor.
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert seen.of(Stop) == [RIGHT_B]
        assert paused_log.held == [line] and line not in paused_log.written
    finally:
        paused_log.resume.set()
    # The line is written once the terminal takes lines again.
    until(lambda: line in paused_log.written, "the log line")
    assert program.end() == 0
    # One line per STOP. Right B held through a wait of more than 1 s would
    # be a second STOP: a new stretch.
    stops = seen.of(Stop)
    assert set(stops) == {RIGHT_B} and paused_log.written.count(line) == len(stops)


def test_right_b_is_a_stop_ahead_of_the_log_line_of_the_session_it_arrives_with(
    bench, folder, start, paused_log
):
    entering, go = threading.Event(), threading.Event()

    class Late(ScriptedSession):
        def __enter__(self) -> ScriptedSession:
            entering.set()
            go.wait(WAIT_S)
            return self

    def feed(graph: Any, newest_state: Any) -> Any:
        # The teleop feed itself; right B is down in its session's first data.
        rows = [{**BASE, "b": True}]
        return feed_module.TeleopFeed(
            graph,
            TeleopSettings(endpoint=NO_SERVER),
            session_factory=lambda config: Late(graph, rows),
        )

    line = f"entered a remote session at {NO_SERVER}"
    # The agent owns the arm: right B is the wearer's STOP of the agent's move.
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=feed)
    assert entering.wait(WAIT_S)  # the first pass is entering the session
    paused_log.pause.set()  # from here on the terminal takes no log line
    go.set()
    try:
        # That pass reads right B behind the entry. Its STOP is on its way to
        # the Thor while the line of the entry waits.
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert seen.of(Stop) == [RIGHT_B]
        assert paused_log.waiting.wait(WAIT_S)
        assert paused_log.held == [line] and line not in paused_log.written
    finally:
        paused_log.resume.set()
    until(lambda: "right B is down: STOP sent" in paused_log.written, "the lines")
    assert program.end() == 0
    lines = [text for text in paused_log.written if text == line or "right B" in text]
    assert lines[:2] == [line, "right B is down: STOP sent"] and lines.count(line) == 1


class HeldB:
    """main()'s feed keyword: a script feed whose passes a test can hold up.

    Right B goes down once a ROBOT_STATE shows nobody as the owner.
    """

    def __init__(self, b: bool = True) -> None:
        """b False: right B stays up."""
        self.b = b
        self.hold = threading.Event()  # set by a test: the next pass waits
        self.holding = threading.Event()  # a pass waits
        self.go = threading.Event()  # set by a test: that pass goes on
        self.newest: Callable[[], RobotState | None] = lambda: None

    def __call__(self, graph: Any, newest_state: Any) -> ScriptFeed:
        held = self

        class Held(ScriptFeed):
            def step(
                self, external_inputs: Any, events: Any, now_mono: float
            ) -> FeedStep:
                if held.hold.is_set():
                    held.holding.set()
                    held.go.wait(WAIT_S)
                return super().step(external_inputs, events, now_mono)

        self.newest = newest_state
        rows = [BASE, {"wait_for": "owner_nobody"}, {"t": 0.0, "b": self.b}]
        return Held(graph, rows, state=newest_state)

    def between_two_passes(self, bench: Bench, thor: FakeThor, new: Message) -> None:
        """The robot app comes back with this WELCOME, of the session s2.

        Right B goes down with it: the next pass, which waits for go, reads
        both.
        """
        self.hold.set()
        assert self.holding.wait(WAIT_S)
        thor.send(new)
        shown = show(bench, thor, owner="nobody", session_id="s2")
        until(lambda: self.newest() == shown, "the new session's ROBOT_STATE")


SESSION_LINE = "the robot app has a new session: s2"  # a log line of the program


def test_right_b_is_a_stop_ahead_of_the_log_line_of_a_new_session(
    bench, folder, start, paused_log
):
    feed = HeldB()
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=feed)
    program.wait_said("right controller seen")
    feed.between_two_passes(bench, thor, welcome("s2"))
    paused_log.pause.set()
    feed.go.set()
    try:
        assert paused_log.waiting.wait(WAIT_S)
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert seen.of(Stop) == [RIGHT_B]
        assert paused_log.held == [SESSION_LINE]
    finally:
        paused_log.resume.set()
    until(lambda: "right B is down: STOP sent" in paused_log.written, "the lines")
    assert program.end() == 0
    lines = [text for text in paused_log.written if "right B" in text or "s2" in text]
    assert lines[:2] == [SESSION_LINE, "right B is down: STOP sent"]


@pytest.fixture
def paused_error(paused_log, monkeypatch) -> list[str]:
    """What the program wrote to standard error, which waits while the log does.

    The lines are kept without the program's name in front.
    """
    written: list[str] = []

    def complain(text: str) -> None:
        if paused_log.pause.is_set():
            paused_log.resume.wait(3 * WAIT_S)  # a standard error that takes nothing
        written.append(text)

    monkeypatch.setattr(human_main, "_complain", complain)
    return written


@pytest.mark.parametrize("why", ["a real arm", "another arm model"])
def test_right_b_is_a_stop_on_the_pass_that_ends_the_program_for_a_new_session(
    why, bench, folder, start, paused_log, paused_error, handed
):
    feed = HeldB()
    # The agent owns the arm: right B is the wearer's STOP of the agent's move.
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=feed)
    program.wait_said("right controller seen")
    # The new session is one this program must not drive: its last pass.
    if why == "a real arm":
        new = dataclasses.replace(welcome("s2"), driver="real")
        said = "a synthetic controller may drive only a simulated arm; "
    else:
        names = [f"j{index}" for index in range(1, 6)]
        new = dataclasses.replace(welcome("s2"), joint_names=names)
        said = "the arm model 'mock' has 6 joints and the robot app's WELCOME names 5"
    feed.between_two_passes(bench, thor, new)
    paused_log.pause.set()  # neither the log nor standard error takes a line
    feed.go.set()
    try:
        # Right B, read on that pass, is a STOP ahead of every line of the end.
        until(lambda: seen.of(Stop), "the STOP of right B")
        assert paused_log.waiting.wait(WAIT_S)
        assert paused_log.held == [SESSION_LINE] and paused_error == []
        assert program.running
    finally:
        paused_log.resume.set()
    until(lambda: not program.running, "the program's end")
    assert program.code == 1
    assert len(paused_error) == 1 and paused_error[0].startswith(said)
    lines = [text for text in paused_log.written if "right B" in text or "s2" in text]
    assert lines == [SESSION_LINE, "right B is down: STOP sent"]
    # The agent owned the arm: no STOP of the program's own. Nothing else went
    # out for the session it must not drive.
    time.sleep(0.2)
    assert seen.of(Stop) == [RIGHT_B]
    others = [message for message in seen.messages if message != RIGHT_B]
    assert all(message.session_id == "s1" for message in others)
    # Nor was anything handed to the client behind that STOP: the service
    # would have dropped a HUMAN_SETPOINT there, so `seen` cannot tell.
    assert handed[handed.index(RIGHT_B) :] == [RIGHT_B]
    assert all(message.session_id == "s1" for message in handed[:-1])


def test_the_last_pass_for_a_session_not_to_be_driven_sends_no_human_setpoint(
    bench, folder, start, handed
):
    # Right B stays up: no STOP of that pass makes the service drop what follows.
    feed = HeldB(b=False)
    thor = robot(bench, owner="agent")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=feed)
    program.wait_said("right controller seen")
    until(lambda: handed, "a HUMAN_SETPOINT for the session the program took")
    new = dataclasses.replace(welcome("s2"), driver="real")
    feed.between_two_passes(bench, thor, new)
    time.sleep(0.2)  # a HUMAN_SETPOINT is due: 20 per second as a status
    feed.go.set()
    until(lambda: not program.running, "the program's end")
    assert program.code == 1
    time.sleep(0.2)
    # Every link message of the program was a HUMAN_SETPOINT for the session
    # it took. For the one it must not drive it sent nothing, also no status.
    assert handed and {type(message) for message in handed} == {HumanSetpoint}
    assert {message.session_id for message in handed} == {"s1"}
    assert {message.session_id for message in seen.of(HumanSetpoint)} == {"s1"}
    assert seen.of(Stop) == []


@pytest.mark.parametrize(
    "why, said",
    [
        ("it failed", "failed: ValueError('the feed broke')"),
        ("no Teleop Session Server", "no remote session at x: 3 attempts failed"),
        (
            "the robot app's new session cannot be driven",
            "a synthetic controller may drive only a simulated arm; "
            "the robot app declares driver real",
        ),
        # As Ctrl+C and SIGTERM end the program: exit code 0, no line of its own.
        ("the stop event was set", None),
    ],
    ids=["an error", "the attempts ran out", "a new session", "the stop event"],
)
def test_the_programs_own_stop_goes_out_ahead_of_the_lines_of_its_end(
    why, said, bench, folder, start, paused_log, paused_error
):
    errors: list[BaseException] = []

    class Failing(ScriptFeed):
        def step(self, external_inputs: Any, events: Any, now_mono: float) -> FeedStep:
            if errors:
                raise errors[0]
            return super().step(external_inputs, events, now_mono)

    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    program = start(
        write_config(folder, bench.path),
        feed=lambda graph, state: Failing(graph, [BASE], state=state),
    )
    program.wait_said("right controller seen")
    paused_log.pause.set()  # neither the log nor standard error takes a line
    if why == "it failed":
        errors.append(ValueError("the feed broke"))
    elif why == "no Teleop Session Server":
        errors.append(feed_module.AttemptsRanOut(said))
    elif why == "the stop event was set":
        program.stop.set()
    else:
        thor.send(dataclasses.replace(welcome("s2"), driver="real"))
    reason = f"the human program ends ({why})"
    stop_line = f"STOP sent: the person owned the arm; {reason}"
    try:
        # The person owns the arm: the end is a STOP, a hold, and it is on its
        # way to the Thor while the first line that tells of it waits.
        until(lambda: seen.of(Stop), "the program's STOP")
        assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
        assert paused_log.waiting.wait(WAIT_S)
        assert paused_log.held == [stop_line]
        assert paused_error == [] and program.running
    finally:
        paused_log.resume.set()
    until(lambda: not program.running, "the program's end")
    if said is None:
        assert program.code == 0 and paused_error == []
        assert paused_log.written[-2:] == [stop_line, f"{why}; closing"]
    else:
        assert program.code == 1 and paused_error == [said]
    time.sleep(0.2)
    assert len(seen.of(Stop)) == 1


def test_the_stop_after_the_last_teleop_attempt_waits_for_no_log_line_of_the_feed(
    bench, folder, start, paused_log, paused_error
):
    go = threading.Event()  # set by the test: the second attempt goes on
    errors = [
        TimeoutError(
            f"Teleop Session Server at {NO_SERVER} was not ready after 3 s: "
            "server did not answer"
        ),
        RuntimeError("The Teleop Session Server is busy"),
    ]
    entered: list[None] = []

    class NoSession:
        """In place of a TeleopSession: one that cannot be entered."""

        def __init__(self, config: Any) -> None:
            self.config = config

        def __enter__(self) -> Any:
            entered.append(None)
            if len(entered) == 2:
                go.wait(WAIT_S)
            raise errors[len(entered) - 1]

    # The teleop feed itself, with two attempts; nothing is dialed.
    settings = TeleopSettings(endpoint=NO_SERVER, retry_s=0.5, attempts=2)
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    program = start(
        write_config(folder, bench.path),
        feed=lambda graph, state: feed_module.TeleopFeed(
            graph, settings, session_factory=NoSession
        ),
    )
    first = (
        f"attempt 1 of 2 to enter a remote session at {NO_SERVER} failed "
        f"(no_server): {errors[0]}"
    )
    until(lambda: first in paused_log.written, "the log line of the first attempt")
    # The program has the ROBOT_STATE that shows the person as the owner.
    until(lambda: seen.of(HumanSetpoint), "a HUMAN_SETPOINT")
    paused_log.pause.set()  # neither the log nor standard error takes a line
    go.set()
    reason = "the human program ends (no Teleop Session Server)"
    try:
        # The last attempt failed: the program's STOP is on its way to the
        # Thor, and the first line that waits is the one behind it.
        until(lambda: seen.of(Stop), "the program's STOP")
        assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
        assert paused_log.waiting.wait(WAIT_S)
        assert paused_log.held == [f"STOP sent: the person owned the arm; {reason}"]
        assert paused_error == [] and program.running
    finally:
        paused_log.resume.set()
    until(lambda: not program.running, "the program's end")
    assert program.code == 1 and len(entered) == 2
    # The line of the end names the last attempt's error; the log has a line
    # of the attempt before it only.
    assert paused_error == [
        f"no remote session at {NO_SERVER}: 2 attempts in a row failed, 0.5 s "
        f"apart; the last one: {errors[1]}"
    ]
    attempts = [text for text in paused_log.written if text.startswith("attempt ")]
    assert attempts == [first]


def test_a_log_line_that_blocks_on_an_earlier_pass_still_holds_up_right_b_afterwards(
    bench, folder, start, monkeypatch, paused_log
):
    # A known limit. A log call writes on the thread that logs, so one that
    # blocks after a pass has sent its link messages holds up every pass
    # behind it: right B is read, and is a STOP, only once the call returns.
    from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient

    sent: list[Message] = []  # what the program's client was given, in order
    real = WorkstationClient.send

    def send(self: Any, message: Message) -> None:
        sent.append(message)
        real(self, message)

    monkeypatch.setattr(WorkstationClient, "send", send)
    newest: list[Callable[[], RobotState | None]] = []

    def feed(graph: Any, newest_state: Any) -> ScriptFeed:
        newest.append(newest_state)
        # Right A is held from 0.2 s on; right B goes down once a ROBOT_STATE
        # shows the arm at rest.
        rows = [BASE, {"t": 0.2, "a": True}, {"wait_for": "rest_hold"}]
        return ScriptFeed(graph, [*rows, {"t": 0.2, "b": True}], state=newest_state)

    # The person owns the arm, on a ROBOT_STATE that counts as young throughout.
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    config = write_config(folder, bench.path, state={"max_age_s": 60.0})
    program = start(config, feed=feed)
    program.wait_said("human program: A down")
    paused_log.pause.set()  # from here on the terminal takes no log line
    try:
        # A's hold of 1 s asks for the hand-back; that pass's line waits.
        assert paused_log.waiting.wait(WAIT_S)
        assert paused_log.held == ["right A held 1.0 s: asking for the handback"]
        # Its link messages went out first, and a HUMAN_SETPOINT on every pass
        # before it.
        count = len(sent)
        assert isinstance(sent[-2], OwnerRequest) and sent[-2].request == "handback"
        assert isinstance(sent[-1], HumanSetpoint) and count > 20

        # The passes stand. Meanwhile the ROBOT_STATE at rest arrives, on which
        # the script holds right B down: B is down when a pass runs again.
        shown = show(bench, thor, owner="human", motion="rest_hold")
        until(lambda: newest[0]() == shown, "the ROBOT_STATE at rest")
        time.sleep(0.5)  # 25 passes
        assert len(sent) == count and of(sent, Stop) == [] and seen.of(Stop) == []
        assert "human program: B down" not in program.said
    finally:
        paused_log.resume.set()
    # The passes run again: the STOP of right B goes out with the first one.
    until(lambda: seen.of(Stop), "the STOP of right B")
    assert sent[count] == RIGHT_B and seen.of(Stop) == [RIGHT_B]
    until(lambda: len(of(sent[count:], HumanSetpoint)) > 10, "more passes")
    program.wait_said("human program: B down")
    assert program.end() == 0


def test_the_writer_says_the_lines_in_order_and_keeps_only_so_many(monkeypatch):
    monkeypatch.setattr(human_main, "_WRITER_JOIN_S", 0.2)
    said: list[str] = []
    taking, release = threading.Event(), threading.Event()

    def say(line: str) -> None:
        taking.set()
        release.wait(WAIT_S)
        said.append(line)

    out = human_main._Writer(say)
    try:
        out("line 0")
        assert taking.wait(WAIT_S)  # the output takes no more
        kept = human_main.LINES_KEPT
        started = time.monotonic()
        for index in range(1, kept + 20):
            out(f"line {index}")
        assert time.monotonic() - started < 1.0  # no line waited for the output
        # Closing waits for the lines only so long.
        assert not out.close() and said == []
        assert 0.15 < time.monotonic() - started < 2.0
    finally:
        release.set()
    # Once the output takes lines again, every line that was kept is said
    # and the writer ends; the newest ones, which had no room, were dropped.
    monkeypatch.setattr(human_main, "_WRITER_JOIN_S", WAIT_S)
    assert out.close()
    assert said == [f"line {index}" for index in range(kept + 1)]


def test_the_default_say_prints_a_line_to_a_stream_with_no_file_descriptor(capsys):
    # capsys makes standard output such a stream: there is nothing to wait for.
    with pytest.raises(OSError):
        sys.stdout.fileno()
    human_main._say("human program: a line")
    assert capsys.readouterr().out == "human program: a line\n"


@pytest.mark.parametrize(
    "error, line, why",
    [
        (
            ValueError("the feed broke"),
            "human program: failed: ValueError('the feed broke')",
            "it failed",
        ),
        (
            feed_module.AttemptsRanOut("no remote session at x: 3 attempts failed"),
            "human program: no remote session at x: 3 attempts failed",
            "no Teleop Session Server",
        ),
    ],
    ids=["an error", "the attempts ran out"],
)
def test_a_feed_that_fails_ends_the_program_with_exit_code_1_and_is_closed(
    error, line, why, bench, folder, capsys
):
    calls: list[str] = []

    class Failing(ScriptFeed):
        def open(self) -> None:
            calls.append("open")
            super().open()

        def step(self, external_inputs: Any, events: Any, now_mono: float) -> FeedStep:
            calls.append("step")
            if calls.count("step") > 10:
                raise error
            return super().step(external_inputs, events, now_mono)

        def close(self) -> None:
            calls.append("close")
            super().close()

    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    code = run_main(
        "--config",
        str(write_config(folder, bench.path)),
        feed=lambda graph, state: Failing(graph, [BASE], state=state),
        say=lambda line: None,
    )
    assert code == 1
    said = capsys.readouterr().err.splitlines()
    assert [text for text in said if text.startswith("human program: ")] == [line]
    # The feed was closed, once, after its last pass.
    assert calls.count("step") == 11 and calls.count("close") == 1
    assert calls[-1] == "close" and "open" in calls
    # The person owned the arm: the program's end is one STOP, a hold.
    until(lambda: seen.of(Stop), "the STOP")
    time.sleep(0.2)
    reason = f"the human program ends ({why})"
    assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
    assert len(seen.of(HumanSetpoint)) > 0


def test_an_endpoint_isaac_capture_cannot_use_ends_the_program_with_exit_code_2(
    bench, folder, capsys
):
    # The person owns the arm, from a human program that ran before.
    thor = robot(bench, driver="real", owner="human")
    endpoint = "127.0.0.1:notaport"  # it names no port, so nothing is dialed
    config = write_config(folder, bench.path, teleop={"endpoint": endpoint})
    said: list[str] = []
    # --controller is teleop unless given: Isaac Capture's own remote session.
    code = run_main("--config", str(config), say=said.append)
    assert code == 2
    error = capsys.readouterr().err.splitlines()
    assert [text for text in error if text.startswith("human program: ")] == [
        f"human program: the settings cannot be used: teleop.endpoint {endpoint!r}: "
        f"invalid port in endpoint '{endpoint}'"
    ]
    start_line = "human program: teleop; driver real; arm model mock, 6 joints"
    assert said == [start_line, SPEED_LINE, BOX_LINE_UNNAMED, JOINTS_LINE_UNNAMED]
    # This program drove nothing: no link message of it went to the Thor.
    until(lambda: not bench.service.roles().get("human"), "the program's leaving")
    assert thor.drain() == []


def test_a_value_error_at_a_later_teleop_attempt_ends_the_program_with_its_stop(
    bench, folder, start, capsys
):
    sessions: list[ScriptedSession] = []
    # As for an answer of the peer at teleop.endpoint that is not UTF-8 text.
    wrong = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    def feed(graph: Any, newest_state: Any) -> Any:
        def factory(config: Any) -> ScriptedSession:
            if sessions:
                raise wrong
            sessions.append(ScriptedSession(graph, [BASE]))
            return sessions[0]

        # The teleop feed itself, on a session that is entered and then lost.
        settings = TeleopSettings(endpoint=NO_SERVER, retry_s=0.3, attempts=5)
        return feed_module.TeleopFeed(graph, settings, session_factory=factory)

    # The person owns the arm, and this program sends its HUMAN_SETPOINTs.
    thor = robot(bench, driver="real", owner="human")
    seen = Forwarded(thor)
    program = start(write_config(folder, bench.path), feed=feed)
    program.wait_said("right controller seen")
    until(lambda: len(seen.of(HumanSetpoint)) > 5, "HUMAN_SETPOINTs")
    sessions[0].lost.set()
    until(lambda: not program.running, "the program's end")
    # Isaac Capture read teleop.endpoint before: this is no error of the file.
    # The program ends as on any error: its STOP, a hold, and exit code 1.
    assert program.code == 1
    until(lambda: seen.of(Stop), "the program's STOP")
    time.sleep(0.2)
    reason = "the human program ends (it failed)"
    assert seen.of(Stop) == [Stop(reason=reason, sender="human_program")]
    error = RuntimeError(
        f"the attempt to enter a remote session at {NO_SERVER} failed: "
        f"UnicodeDecodeError: {wrong}"
    )
    said = capsys.readouterr().err.splitlines()
    assert [text for text in said if text.startswith("human program: ")] == [
        f"human program: failed: {error!r}"
    ]


def test_a_robot_state_of_an_earlier_session_is_no_reason_for_the_stop_at_the_end(
    bench, folder, start, caplog
):
    thor = robot(bench, owner="human")
    seen = Forwarded(thor)
    config = write_config(folder, bench.path)
    with caplog.at_level("INFO", logger=human_main._log.name):
        program = start(config, feed=script([BASE]))
        program.wait_said("right controller seen")
        until(lambda: seen.of(HumanSetpoint), "a HUMAN_SETPOINT")
        # The robot app comes back as another session, which has no owner yet;
        # its first ROBOT_STATE has not arrived.
        thor.send(welcome("s2"))
        logged = lambda: [r.getMessage() for r in caplog.records]  # noqa: E731
        until(lambda: any("new session" in text for text in logged()), "the WELCOME")
        assert program.end() == 0
    time.sleep(0.3)
    assert seen.of(Stop) == []

    # Once that session shows the person as its owner, the end is a STOP again.
    show(bench, thor, owner="human", session_id="s2")
    other = start(config, feed=script([BASE]))
    other.wait_said("right controller seen")
    until(
        lambda: any(sent.session_id == "s2" for sent in seen.of(HumanSetpoint)),
        "a HUMAN_SETPOINT for the new session",
    )
    assert other.end() == 0
    until(lambda: seen.of(Stop), "the STOP")
    time.sleep(0.2)
    assert seen.of(Stop) == [END_STOP]


def test_the_program_ends_with_exit_code_1_when_the_robot_link_service_goes(
    folder, start, capsys
):
    served = Bench(folder)
    try:
        robot(served)
        program = start(write_config(folder, served.path), feed=script([BASE]))
        program.wait_said("right controller seen")
    finally:
        served.close()
    until(lambda: not program.running, "the program's end")
    assert program.code == 1
    assert "human program: the robot link service is gone (" in capsys.readouterr().err


def test_the_program_waits_for_a_welcome_and_ends_on_the_stop_event(
    bench, folder, start
):
    program = start(write_config(folder, bench.path), feed=script([BASE]))
    until(lambda: bench.service.roles().get("human") == 1, "the program's attaching")
    time.sleep(0.2)
    assert program.running and program.said == []
    assert program.end() == 0 and program.said == []


def test_the_program_takes_the_simulated_arm_over_and_drives_it(
    make_realtime_robot_app, folder, start
):
    from link_stack import Stack
    from test_takeover import human_config

    rows = [
        BASE,
        {"t": 0.3, "b": True},
        {"wait_for": "owner_human"},
        {"t": 0.5, "b": False},
        {"t": 1.0, "squeeze": 1.0},
        {"t": 1.2},
        {"t": 2.2, "pos": list(HAND + 0.02 * np.array(UP))},
        {"t": 2.6, "squeeze": 0.0},
    ]
    config = human_config(
        RobotAppConfig.default(), require_live_view=False, sample_age_limit_s=1.0
    )
    h = make_realtime_robot_app(config=config, connected=False)
    stack = Stack(h, folder)
    solver = Solver.from_apai("mock")
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        program = start(
            write_config(folder, stack.path, state={"max_age_s": 1.0}),
            *("--report-s", "0.4"),
            feed=script(rows),
        )
        # Right B: a STOP from the controller, then the takeover after the hold.
        watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
        assert [event[2]["sender"] for event in h.heard("stop")] == ["controller"]
        assert h.app.takeover_open
        # The squeeze, the lift and the release: the arm rose with the hand.
        program.wait_said("human program: grip down")
        program.wait_said("human program: grip up")
        watcher.ws.wait_state(
            lambda s: not s.health["human"]["driving"] and s.motion == "hold", WAIT_S
        )
        assert len(h.arm.sent) > 20
        measured = watcher.ws.measured_pose()
        rise = solver.fk(measured)[:3, 3] - solver.fk(solver.inside(REST))[:3, 3]
        np.testing.assert_allclose(rise, [0.0, 0.0, 0.02], atol=3e-3)
        # The report lines of the drive: engaged, with the times of its solves
        # and the Thor's measure of the round trip.
        reports = [REPORT.fullmatch(text) for text in program.said if is_report(text)]
        assert reports and None not in reports, program.said
        driving = [found for found in reports if found["engaged"] == "yes"]
        assert driving and all(found["owner"] == "human" for found in driving)
        assert all(found["ik"] != "-/-" for found in driving)
        assert any(found["trip"] != "-" for found in driving)
        assert all(found["refused"] is None for found in driving)
        # The trigger is released and the jaws are open: they follow it from
        # the takeover on, and that asks no jaw command of the arm.
        open_jaws = ("yes", "1.00", "1.00", None)
        for found in driving:
            said = (found["following"], found["level"], found["asks"], found["why"])
            assert said == open_jaws, found.group(0)
        assert h.arm.jaw_commands == []
        assert h.arm.torque_on() and h.arm.released_at == []

        # The program ends while the person owns the arm: one more STOP, a hold.
        assert watcher.ws.state.owner == "human"
        assert program.end() == 0
        watcher.ws.wait_state(lambda s: s.owner == "nobody", WAIT_S)
        senders = [event[2]["sender"] for event in h.heard("stop")]
        assert senders == ["controller", "human_program"]
        assert h.app.takeover_open and h.arm.torque_on()
    finally:
        stack.close()


def test_the_program_lifts_the_arm_off_its_stops_on_the_rig_files(
    make_realtime_robot_app, folder, start
):
    """The rig's human file and the rig's robot file, from a rest reading.

    The human program runs on configs/human_rebot.yaml and the robot app on
    configs/robot_app_rebot.yaml, on the simulated arm at a pose the rig read
    at rest. What differs from the files, in memory only: human driving is
    on; no live view is asked for; the arm counts as at rest at start within
    the 0.12 rad that --arm apai takes from the profile; and the oldest
    controller data and the oldest ROBOT_STATE that are followed are 1 s old,
    since both programs run in real time on a machine that runs other tests.
    """
    from link_stack import Stack

    reading = RIG_REST_READINGS[0]
    robot = load_config(RIG_ROBOT_FILE)
    step_limit = robot.limits.human.max_step_rad
    human = dataclasses.replace(
        robot.human, enabled=True, require_live_view=False, sample_age_limit_s=1.0
    )
    rest = dataclasses.replace(robot.rest, start_rest_tol_rad=0.12)
    config = dataclasses.replace(robot, human=human, rest=rest)
    settings = yaml.safe_load((CONFIGS / "human_rebot.yaml").read_text())
    assert settings["ik"]["max_ahead_rad"] < step_limit == 0.10
    rows = [
        BASE,
        {"t": 0.3, "b": True},
        {"wait_for": "owner_human"},
        {"t": 0.5, "b": False},
        # The hand rises 2 cm at the file's hand speed limit from the squeeze on.
        {"t": 1.0, "squeeze": 1.0},
        {"t": 3.0, "pos": list(HAND + 0.02 * np.array(UP))},
        {"t": 3.4, "squeeze": 0.0},
    ]
    h = make_realtime_robot_app(config=config, pose=reading, connected=False)
    stack = Stack(h, folder)
    solver = Solver.from_apai(settings["ik"]["apai_arm"])
    inside = solver.inside(reading)
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        settings["socket_path"] = stack.path
        settings["teleop"]["endpoint"] = NO_SERVER
        settings["state"]["max_age_s"] = 1.0
        file = folder / "human_rig.yaml"
        file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        program = start(file, *("--report-s", "0.4"), feed=script(rows))
        watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
        program.wait_said("human program: grip down")
        program.wait_said("human program: grip up")
        watcher.ws.wait_state(
            lambda s: not s.health["human"]["driving"] and s.motion == "hold", WAIT_S
        )
        # One human drive, ended by the released grip: the Thor refused no
        # target, and the human program ended no engagement itself.
        sent = [q for q, _ in h.arm.sent]
        assert len(sent) > 90 and h.heard("human") == []
        reports = [REPORT.fullmatch(text) for text in program.said if is_report(text)]
        assert reports and None not in reports, program.said
        assert all(found["refused"] is None for found in reports)
        none_ended = (
            "ended: no solution 0, jump 0, solution ahead 0, hand ahead 0, limiter 0"
        )
        assert all(none_ended in found.group(0) for found in reports)
        # The first target lay off the two joint stops, 0.033 rad from the
        # arm: inside the Thor's step limit, and taken at its speed clamp.
        clamp = SPEED_MARGIN * robot.limits.human.max_joint_speed * PASS_S
        steps = np.abs(np.diff(np.array([reading, *sent]), axis=0))
        off_the_stops = np.abs(np.array(inside) - np.array(reading)).max()
        assert 0.031 < off_the_stops < settings["ik"]["max_ahead_rad"]
        assert steps[0].max() == pytest.approx(clamp, abs=1e-9)
        assert steps.max() <= clamp + 1e-9
        # The arm rose with the hand.
        rise = solver.fk(watcher.ws.measured_pose())[:3, 3] - solver.fk(inside)[:3, 3]
        np.testing.assert_allclose(rise, [0.0, 0.0, 0.02], atol=3e-3)
        assert h.arm.torque_on() and h.arm.released_at == []
        assert program.end() == 0
    finally:
        stack.close()


@pytest.mark.parametrize("short", [0.095, 0.12, 0.18])
def test_a_squeeze_drives_an_arm_that_holds_short_of_its_last_joint_target(
    make_realtime_robot_app, folder, start, short
):
    """After an agent's move that did not settle, on the rig files.

    The arm holds raised with one joint blocked `short` rad from the last joint
    target sent, more than ik.max_ahead_rad. Until the person's drive has sent
    a setpoint the lead is measured from the arm, not from that target, so
    every squeeze drives. Changed in memory as in the test above.
    """
    from link_stack import Stack

    raised = [0.3, 1.2, 1.2, 0.2, 0.3, 0.1]
    robot = load_config(RIG_ROBOT_FILE)
    human = dataclasses.replace(
        robot.human, enabled=True, require_live_view=False, sample_age_limit_s=1.0
    )
    config = dataclasses.replace(robot, human=human)
    settings = yaml.safe_load((CONFIGS / "human_rebot.yaml").read_text())
    assert short > settings["ik"]["max_ahead_rad"]
    rows = [
        BASE,
        {"t": 0.3, "b": True},
        {"wait_for": "owner_human"},
        {"t": 0.5, "b": False},
    ]
    squeezes = 3
    for at in (1.0, 2.8, 4.6):  # each squeeze with the hand rising 1 cm
        rows += [
            {"t": at, "squeeze": 1.0},
            {"t": at + 1.0, "pos": list(HAND + 0.01 * np.array(UP))},
            {"t": at + 1.2, "squeeze": 0.0, "pos": list(HAND)},
        ]
    h = make_realtime_robot_app(config=config, pose=raised, connected=False)
    h.arm.block_joint(0, raised[0])
    h.arm.send_target([raised[0] + short, *raised[1:]])
    before = len(h.arm.sent)
    stack = Stack(h, folder)
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        settings["socket_path"] = stack.path
        settings["teleop"]["endpoint"] = NO_SERVER
        settings["state"]["max_age_s"] = 1.0
        file = folder / f"human_short_{short}.yaml"
        file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        program = start(file, *("--report-s", "0.4"), feed=script(rows))
        watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
        for _ in range(squeezes):
            program.wait_said("human program: grip down")
            program.wait_said("human program: grip up")
            del program.said[: program.said.index("human program: grip up") + 1]
        watcher.ws.wait_state(
            lambda s: not s.health["human"]["driving"] and s.motion == "hold", WAIT_S
        )
        # Every squeeze drove: nothing was ended by the human program or the Thor.
        assert h.heard("human") == []
        assert len(h.arm.sent) - before > 30 * squeezes
        assert h.arm.torque_on() and h.arm.released_at == []
        assert program.end() == 0
    finally:
        stack.close()


def test_the_program_follows_a_hand_moved_20_cm_in_a_quarter_second_on_the_rig_files(
    make_realtime_robot_app, folder, start
):
    """A hand faster than the arm, in real time, on the rig's two files.

    The human program's main() and the robot app run on the machine's clock,
    on the simulated arm raised over the table. The hand goes 20 cm to the
    left in a quarter of a second and stays there for 3 s; then the grip is
    released. Changed in memory as in the two tests above.
    """
    from link_stack import Stack

    raised = [0.3, 1.2, 1.2, 0.2, 0.3, 0.1]
    robot = load_config(RIG_ROBOT_FILE)
    human = dataclasses.replace(
        robot.human, enabled=True, require_live_view=False, sample_age_limit_s=1.0
    )
    config = dataclasses.replace(robot, human=human)
    settings = yaml.safe_load((CONFIGS / "human_rebot.yaml").read_text())
    metres = 0.20
    rows = [
        BASE,
        {"t": 0.3, "b": True},
        {"wait_for": "owner_human"},
        {"t": 0.5, "b": False},
        {"t": 1.0, "squeeze": 1.0},
        {"t": 1.2},
        {"t": 1.45, "pos": list(HAND + metres * np.array(LEFT))},
        {"t": 4.45, "squeeze": 0.0},
    ]
    h = make_realtime_robot_app(config=config, pose=raised, connected=False)
    stack = Stack(h, folder)
    solver = Solver.from_apai(settings["ik"]["apai_arm"])
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        settings["socket_path"] = stack.path
        settings["teleop"]["endpoint"] = NO_SERVER
        settings["state"]["max_age_s"] = 1.0
        file = folder / "human_fast.yaml"
        file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        program = start(file, *("--report-s", "0.2"), feed=script(rows))
        # The program says how fast the arm follows under this robot app.
        assert program.wait_said("human program: the arm follows") == (
            "human program: the arm follows the hand at up to 1.41 rad/s a joint "
            "(the robot app's limit for a person is 1.6 rad/s), its tip at up to "
            "0.5 m/s and 2 rad/s"
        )
        watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)
        program.wait_said("human program: grip down")
        program.wait_said("human program: grip up")
        watcher.ws.wait_state(
            lambda s: not s.health["human"]["driving"] and s.motion == "hold", WAIT_S
        )
        # One human drive, ended by the released grip: the Thor refused no
        # target, and the human program ended no engagement itself.
        sent = [q for q, _ in h.arm.sent]
        assert len(sent) > 150 and h.heard("human") == []
        reports = [REPORT.fullmatch(text) for text in program.said if is_report(text)]
        assert reports and None not in reports, program.said
        assert all(found["refused"] is None for found in reports)
        none_ended = (
            "ended: no solution 0, jump 0, solution ahead 0, hand ahead 0, limiter 0"
        )
        assert all(none_ended in found.group(0) for found in reports)
        # The fastest joint ran at 0.9 of the Thor's clamp, the human
        # program's bound. The two loops are not in step in real time: where
        # two passes of the human program fell between two setpoints, the
        # Thor took the way at its clamp, and no joint was faster than that.
        clamp = SPEED_MARGIN * robot.limits.human.max_joint_speed * PASS_S
        steps = np.abs(np.diff(np.array([raised, *sent]), axis=0)).max(axis=1)
        assert steps.max() <= clamp + 1e-9
        # The arm was on its way for about 0.6 s: the farthest joint turns
        # 0.79 rad, which takes 0.56 s at 0.9 of the clamp, 1.41 rad/s, and
        # no less. A margin of a second is left for a loaded machine.
        times = np.array([when for _, when in h.arm.sent])
        moving = np.nonzero(steps > 1e-6)[0]
        on_its_way_s = float(times[moving[-1]] - times[moving[0]])
        turned = float(np.abs(np.array(sent[-1]) - np.array(raised)).max())
        assert turned / (JOINT_SPEED_SHARE * clamp / PASS_S) < on_its_way_s + 0.05
        assert on_its_way_s < 1.6
        print(f"\n20 cm: {turned:.3f} rad on the farthest joint, {on_its_way_s:.2f} s")
        # While the arm was behind, the report line said how far.
        behind = [found["ahead"] for found in reports if found["ahead"]]
        assert behind and all(5 <= int(cm) <= 20 for cm in behind)
        # The arm stands where the hand stopped: 20 cm to the left.
        moved = solver.fk(sent[-1])[:3, 3] - solver.fk(raised)[:3, 3]
        np.testing.assert_allclose(moved, metres * tip_way(LEFT), atol=2e-3)
        assert h.arm.torque_on() and h.arm.released_at == []
        assert program.end() == 0
    finally:
        stack.close()


def child_env(*paths: Path) -> dict[str, str]:
    """This process's environment with only these folders to import from."""
    drop = ("PYTHONPATH", "AGENTIC_REMOTE_APAI_SRC", "AGENTIC_REMOTE_ISAAC_CAPTURE_PY")
    env = {key: value for key, value in os.environ.items() if key not in drop}
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_child(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_without_isaac_capture_the_program_ends_with_exit_code_2(folder):
    # The socket is the test's: a program that got as far as attaching would
    # find nothing there.
    config = write_config(folder, str(folder / "nobody.sock"))
    env = child_env(EXAMPLE / "python")
    done = run_child(env, "-m", MODULE, "--config", str(config))
    assert done.returncode == 2 and done.stdout == "", done.stderr
    assert done.stderr.count("\n") == 1
    assert done.stderr.startswith("human program: Isaac Capture cannot be imported")
    assert "AGENTIC_REMOTE_ISAAC_CAPTURE_PY" in done.stderr

    # The usage text and the settings need neither Isaac Capture nor APAI.
    usage = run_child(env, "-m", MODULE, "--help")
    assert usage.returncode == 0 and usage.stderr == ""
    assert usage.stdout.split()[:4] == USAGE
    for word in ("--config PATH", "--controller FEED", "--report-s S", "--log-level L"):
        assert word in usage.stdout
    code = (
        f"from {MODULE}.__main__ import load_human_settings as load; "
        f"print(load({str(CONFIGS / 'human_rebot.yaml')!r}).ik.apai_arm)"
    )
    loaded = run_child(env, "-c", code)
    assert (loaded.returncode, loaded.stdout, loaded.stderr) == (0, "rebot_rs\n", "")


def test_without_apai_the_program_ends_with_exit_code_2(folder):
    config = write_config(folder, str(folder / "nobody.sock"))
    build = Path(isaaccapture.__file__).resolve().parents[1]
    env = child_env(EXAMPLE / "python", build)
    done = run_child(
        env, "-m", MODULE, "--config", str(config), "--controller", "synthetic:idle"
    )
    assert done.returncode == 2 and done.stdout == "", done.stderr
    ours = [line for line in done.stderr.splitlines() if line.startswith("human pro")]
    assert len(ours) == 1
    assert ours[0].startswith("human program: APAI cannot be imported")
    assert "AGENTIC_REMOTE_APAI_SRC" in ours[0]
    assert "Isaac Capture cannot be imported" not in done.stderr


def test_what_the_program_cannot_use_ends_with_exit_code_2(folder, capsys):
    said: list[str] = []

    def run(*argv: str) -> tuple[int, str]:
        code = human_main.main(list(argv), attach=never, say=said.append)
        assert said == []
        return code, capsys.readouterr().err

    nobody = str(folder / "nobody.sock")
    good = write_config(folder, nobody)
    # A retarget.controller_to_base that is no rotation is the file's error.
    mirrored = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    bad = write_config(folder, nobody, retarget={"controller_to_base": mirrored})
    code, error = run("--config", str(bad), "--controller", "synthetic:idle")
    assert code == 2 and error.count("\n") == 1
    assert "the settings cannot be used: retarget.controller_to_base must be" in error
    # What Isaac Capture's nodes refuse is the file's error too.
    slow = write_config(folder, nobody, retarget={"reject_linear_velocity": 0.05})
    code, error = run("--config", str(slow), "--controller", "synthetic:idle")
    assert code == 2 and "reject_linear_velocity" in error

    code, error = run("--config", str(good), "--controller", "synthetic:wave")
    assert code == 2 and "no synthetic controller 'wave'" in error
    for controller in ("headset", "synthetic:", "synthetic"):
        code, error = run("--config", str(good), "--controller", controller)
        assert code == 2 and "--controller must be teleop or synthetic:<name>" in error
    code, error = run("--config", str(good), "--report-s", "-1")
    assert code == 2 and "--report-s must be 0 or more" in error
    code, error = run("--config", str(folder / "missing.yaml"))
    assert code == 2 and "cannot read" in error
    unknown = write_config(folder, nobody, ik={"apai_arm": "no_such_arm"})
    code, error = run("--config", str(unknown), "--controller", "synthetic:idle")
    assert code == 2 and "ik.apai_arm 'no_such_arm'" in error
    assert run("--report-s", "1")[0] == 2  # no --config
    # Without a stop event the signals are waited for: on the main thread only.
    codes: list[int] = []
    waiter = threading.Thread(
        target=lambda: codes.append(
            human_main.main(["--config", str(good)], attach=never)
        )
    )
    waiter.start()
    waiter.join(WAIT_S)
    assert codes == [2]
    assert "on the main thread only" in capsys.readouterr().err


def test_the_log_level_is_the_arguments_else_the_files(
    bench, folder, start, log_levels
):
    robot(bench)
    config = write_config(folder, bench.path, log_level="warning")
    assert start(config, "--controller", "synthetic:idle").end() == 0
    program = start(config, "--controller", "synthetic:idle", "--log-level", "DEBUG")
    assert program.end() == 0
    assert log_levels == ["warning", "debug"]


# The start script and the imports


def test_run_human_sh_names_its_variables_and_execs_the_human_program(tmp_path):
    script_file = EXAMPLE / "run_human.sh"
    text = script_file.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n# SPDX-FileCopyrightText: ")
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert "set -euo pipefail" in text
    assert "[ " not in text.replace("[[ ", "")  # [[ ]] only

    # A folder that stands for APAI's src. Python imports sitecustomize from
    # it at its start, which shows the environment the script sets.
    apai_src = tmp_path / "apai_src"
    apai_src.mkdir()
    shown = (
        "APAI_ROBOT APAI_BASE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES "
        "PYTHONDONTWRITEBYTECODE AGENTIC_REMOTE_APAI_SRC PYTHONPATH"
    )
    (apai_src / "sitecustomize.py").write_text(
        "import os, sys\n"
        f"names = {shown.split()!r}\n"
        "print('ENV', *(os.environ.get(name, 'unset') for name in names),"
        " file=sys.stderr)\n",
        encoding="utf-8",
    )
    package = tmp_path / "isaac_capture_py"
    package.mkdir()

    def run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script_file), *args],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )

    env = {"PATH": os.environ["PATH"]}
    needed = {
        "AGENTIC_REMOTE_HUMAN_PYTHON": (sys.executable, "must name a Python"),
        "AGENTIC_REMOTE_APAI_SRC": (str(apai_src), "must name the src folder"),
        "AGENTIC_REMOTE_ISAAC_CAPTURE_PY": (str(package), "must name the folder"),
    }
    for name, (value, said) in needed.items():
        # One line and exit code 2 without the variable, and with a wrong one.
        for wrong in ({}, {name: str(tmp_path / "missing")}):
            done = run({**env, **wrong}, "--help")
            assert done.returncode == 2 and done.stdout == ""
            assert done.stderr.startswith("run_human.sh: ") and name in done.stderr
            assert said in done.stderr and done.stderr.count("\n") == 1
        env[name] = value

    # With them: the human program, here its usage text.
    env.update(APAI_ROBOT="another", APAI_BASE="another", PYTHONPATH="/before")
    env.update(CUDA_VISIBLE_DEVICES="1")
    done = run(env, "--help")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[:4] == USAGE
    path = f"{EXAMPLE}/python:{package}:{apai_src}:/before"
    assert done.stderr.split() == [
        "ENV",
        *("unset", "unset", "PCI_BUS_ID", "0", "1"),
        str(apai_src),
        path,
    ]
    # The arguments reach the program: it reads the file it is given.
    done = run(env, "--config", "missing.yaml")
    assert done.returncode == 2
    assert "human program: the settings cannot be used: cannot read" in done.stderr

    # Without its own interpreter the agent program's is taken.
    del env["AGENTIC_REMOTE_HUMAN_PYTHON"]
    env["AGENTIC_REMOTE_AGENT_PYTHON"] = sys.executable
    done = run(env, "--help")
    assert done.returncode == 0 and done.stdout.split()[:4] == USAGE
    assert list(tmp_path.rglob("*.pyc")) == []


def test_run_human_sh_reads_its_variables_from_an_env_file(tmp_path):
    script_file = EXAMPLE / "run_human.sh"
    fake = tmp_path / "python"
    fake.write_text('#!/usr/bin/env bash\necho "$PYTHONPATH $*"\n', encoding="utf-8")
    fake.chmod(0o755)
    apai_src, package = tmp_path / "apai_src", tmp_path / "isaac_capture_py"
    apai_src.mkdir()
    package.mkdir()
    file = tmp_path / "thor.env"
    file.write_text(
        f"AGENTIC_REMOTE_HUMAN_PYTHON={fake}\n"
        f"AGENTIC_REMOTE_APAI_SRC={apai_src}\n"
        f"AGENTIC_REMOTE_ISAAC_CAPTURE_PY={package}\n",
        encoding="utf-8",
    )

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script_file), *args],
            check=False,
            env={"PATH": os.environ["PATH"]},  # none of the three variables
            capture_output=True,
            text=True,
            timeout=60,
            cwd=tmp_path,
        )

    # As the first argument: the file's variables count, the program gets the rest.
    done = run("--env", str(file), "--config", "configs/human_rebot.yaml")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [
        f"{EXAMPLE}/python:{package}:{apai_src}",
        *("-m", "isaaccapture_examples.agentic_remote.human"),
        *("--config", "configs/human_rebot.yaml"),
    ]
    # Anywhere else the script reads no file: its variables are missing.
    done = run("--config", "configs/human_rebot.yaml", "--env", str(file))
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr.startswith("run_human.sh: AGENTIC_REMOTE_HUMAN_PYTHON ")
    # Without a file, or with one that is not there: one line and exit code 2.
    for args, named in ((["--env"], ""), (["--env", "none.env"], "none.env")):
        done = run(*args)
        assert done.returncode == 2 and done.stdout == ""
        assert done.stderr == (
            f"run_human.sh: --env needs a file of KEY=value lines (it is '{named}')\n"
        )


def imports(source: str, top_only: bool = False) -> list[tuple[str, tuple[str, ...]]]:
    """(module, names) of the import statements of a source; nothing is imported.

    A relative module keeps its dots. top_only: only the statements that run
    when the module is imported, not those in a function or under an `if`.
    """
    tree = ast.parse(source)
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in tree.body if top_only else ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            found.append(("." * node.level + (node.module or ""), names))
        elif isinstance(node, ast.Import):
            found += [(alias.name, ()) for alias in node.names]
    return found


# The folders of the workstation's programs, and the test helpers.
WORKSTATION_PROGRAMS = {"link_service", "console", "agent", "viewer_live", "testing"}


def programs_imported(source: str) -> set[str]:
    """The workstation programs a source imports from, wherever it does."""
    words: set[str] = set()
    for module, names in imports(source):
        words |= {*module.strip(".").split("."), *names}
    return words & WORKSTATION_PROGRAMS


def test_the_human_path_modules_import_no_workstation_program():
    for name in ("retarget.py", "ik.py", "logic.py", "feed.py"):
        source = (PACKAGE / "human" / name).read_text(encoding="utf-8")
        assert imports(source), name
        assert programs_imported(source) == set(), name
    # Such an import would be found however it is written.
    for line, program in (
        ("from ..link_service.local import connect_local", "link_service"),
        ("from .. import console", "console"),
        ("import isaaccapture_examples.agentic_remote.agent.config", "agent"),
        ("def late():\n    from ..testing import client", "testing"),
        (
            "if True:\n    from ..viewer_live.view_model import ViewModel",
            "viewer_live",
        ),
    ):
        assert programs_imported(line) == {program}, line
    # Only the program itself attaches to the robot link service.
    main_source = (PACKAGE / "human" / "__main__.py").read_text(encoding="utf-8")
    assert programs_imported(main_source) == {"link_service", "testing"}


def test_the_program_imports_isaac_capture_and_apai_inside_main_only():
    source = (PACKAGE / "human" / "__main__.py").read_text(encoding="utf-8")
    found = imports(source, top_only=True)
    ours = {module: names for module, names in found if module.startswith(".")}
    assert set(ours) == {
        "..link_service.local",
        "..protocol.local_messages",
        "..protocol.messages",
        ".types",
    }
    assert set(ours["..link_service.local"]) == {"LocalCarrier", "connect_local"}
    others = {module.split(".")[0] for module, _ in found if not module.startswith(".")}
    assert "yaml" in others
    assert others - {"yaml"} <= set(sys.stdlib_module_names)
