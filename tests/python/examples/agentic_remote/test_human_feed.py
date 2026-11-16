# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""human/feed.py: the controller feeds of the human program.

A controller feed runs one pass of the retargeting graph on the newest data of
the right controller and gives the sample age (how old those data are) and its
stage. The script feed is a synthetic controller: rows of timed controller
data. The teleop feed is a remote session, a TeleopSession that is a client of
Isaac Capture's Teleop Session Server; SAMPLES is the server's link message
with the controller data, READY the one that says a headset is tracked, and the
tick age is the age of the newest SAMPLES. Here the teleop feed runs on a
stand-in session, and on a real remote session against a server in miniature
inside this process, on 127.0.0.1 with a port the system picks. No test has a
headset: SAMPLES never carry controller data. The file is skipped without
Isaac Capture.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("isaaccapture.retargeters")

import isaaccapture  # noqa: E402
from isaaccapture.retargeting_engine.interface import (  # noqa: E402
    ComputeContext,
    GraphTime,
)
from isaaccapture.teleop_session_manager import (  # noqa: E402
    RetargetingExecutionMode,
    SessionMode,
    TeleopSession,
    _remote,
)
from isaaccapture_examples.agentic_remote.human import feed as feed_module  # noqa: E402
from isaaccapture_examples.agentic_remote.human.feed import (  # noqa: E402
    NOTE_CHARS,
    SCRIPTS,
    WAITS,
    AttemptsRanOut,
    EndpointRefused,
    ScriptFeed,
    TeleopFeed,
    entry_stage,
    feed_note,
)
from isaaccapture_examples.agentic_remote.human.logic import HumanLogic  # noqa: E402
from isaaccapture_examples.agentic_remote.human.retarget import (  # noqa: E402
    COMMANDED_KEY,
    CONTROLLER_KEY,
    EE_KEY,
    LIMITER_KEY,
    HumanGraph,
)
from isaaccapture_examples.agentic_remote.human.types import (  # noqa: E402
    FeedStep,
    GraphStep,
    HumanSettings,
    RetargetSettings,
    TeleopSettings,
)
from isaaccapture_examples.agentic_remote.protocol.messages import (  # noqa: E402
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    HUMAN_STAGES,
    Stop,
)
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock  # noqa: E402
from test_human_logic import FakeSolver, robot_state, welcome  # noqa: E402
from test_human_retarget import HOME, snapshot  # noqa: E402

PASS_S = 0.02  # the human program runs 50 passes per second
T0 = 100.0  # the monotonic time at which a bench starts its script
HAND = (0.0, 1.2, -0.3)  # where the hand is in the stage space, m
BASE = {"t": 0.0, "pos": list(HAND), "quat": [0, 0, 0, 1]}
# The poses arrive as float32.
TOL = 1e-6
# An endpoint no test dials: its feeds get a stand-in session.
ENDPOINT = "server.example:47600"
WAIT_S = 10.0  # real seconds a test gives a real remote session
WAIT_MS = 5000


class Bench:
    """A script feed stepped pass by pass, in the order the human program uses.

    The script's clock starts at T0. armed stands for the logic's word: while
    it is False the graph runs stopped and nothing may engage.
    """

    def __init__(self, rows: Any, *, with_state: bool = True) -> None:
        self.graph = HumanGraph(RetargetSettings(), HOME)
        self.state = None  # the newest ROBOT_STATE, as the script sees it
        newest = (lambda: self.state) if with_state else None
        if isinstance(rows, str):
            self.feed = ScriptFeed.named(self.graph, rows, state=newest)
        else:
            self.feed = ScriptFeed(self.graph, rows, state=newest)
        self.armed = True
        self.now = T0
        self.fed, self.step = self._pass()

    def _pass(self) -> tuple[FeedStep, GraphStep]:
        events = self.graph.events(stopped=not self.armed)
        inputs = self.graph.external_inputs(self.armed)
        fed = self.feed.step(inputs, events, self.now)
        return fed, self.graph.read(fed.outputs)

    def go(self, seconds: float) -> GraphStep:
        """Run passes for this long; the graph's result of the last one."""
        for _ in range(round(seconds / PASS_S)):
            self.now += PASS_S
            self.fed, self.step = self._pass()
        return self.step

    def at(self, t: float) -> GraphStep:
        """Run passes up to this time after the start; with no wait, script time."""
        return self.go(t - (self.now - T0))


def test_the_script_feed_runs_the_graph_on_controller_snapshots():
    turned = [0.0, 0.0, 0.6, 0.8]
    rows = [
        {**BASE, "quat": turned, "trigger": 0.25, "a": True},
        {"t": 0.1, "squeeze": 1.0},
        {"t": 0.3},  # the hand starts to move here
        {"t": 0.8, "pos": [0.0, 1.22, -0.3]},
    ]
    bench = Bench(rows)
    assert bench.feed.source == HUMAN_SOURCE_SYNTHETIC
    # The first pass: the pipeline's four outputs, read by the graph itself.
    fed, step = bench.fed, bench.step
    assert (fed.sample_age_s, fed.stage, fed.note) == (0.0, "ready", "")
    assert set(fed.outputs) == {COMMANDED_KEY, EE_KEY, LIMITER_KEY, CONTROLLER_KEY}
    assert step.present and step.a and not step.b and not step.engaged
    assert step.squeeze == 0.0 and step.trigger == pytest.approx(0.25)
    np.testing.assert_allclose(step.hand, HAND, atol=TOL)

    # The squeeze reaches the clutch: it engages at the home, wherever the hand is.
    engaged = bench.at(0.2)
    assert engaged.engaged and engaged.squeeze == 1.0
    np.testing.assert_allclose(engaged.ee_target, HOME, atol=TOL)
    # The hand rises 2 cm: the target rises 2 cm along z of the robot's base.
    moved = bench.at(0.9)
    assert moved.engaged and moved.limiter == "passed"
    for target in (moved.commanded_target, moved.ee_target):
        np.testing.assert_allclose(
            target[:3, 3] - HOME[:3, 3], [0.0, 0.0, 0.02], atol=1e-5
        )

    # The graph runs as the program says: stopped, the clutch lets go.
    bench.armed = False
    assert not bench.go(PASS_S).engaged


def test_a_script_row_keeps_the_values_before_it():
    rows = [
        {**BASE, "squeeze": 0.3, "trigger": 0.7, "a": True, "b": False},
        {"t": 1.0, "b": True},
        {"t": 2.0, "squeeze": 0.9},
        {"t": 3.0},
    ]
    bench = Bench(rows)

    def controls(step: GraphStep) -> tuple[float, float, bool, bool]:
        return (round(step.squeeze, 6), round(step.trigger, 6), step.a, step.b)

    assert controls(bench.at(0.96)) == (0.3, 0.7, True, False)  # not before its time
    assert controls(bench.at(1.04)) == (0.3, 0.7, True, True)
    assert controls(bench.at(1.96)) == (0.3, 0.7, True, True)
    assert controls(bench.at(2.5)) == (0.9, 0.7, True, True)
    assert controls(bench.at(9.0)) == (0.9, 0.7, True, True)  # after the last row
    np.testing.assert_allclose(bench.step.hand, HAND, atol=TOL)


def test_pos_moves_only_between_the_two_rows_that_name_it():
    start, end = np.array(HAND), np.array([0.2, 1.3, -0.5])
    rows = [BASE, {"t": 1.0, "a": True}, {"t": 3.0, "pos": list(end)}, {"t": 4.0}]
    bench = Bench(rows)
    for t, share in (
        (0.5, 0.0),
        (1.0, 0.0),  # the row before the one that names pos: the move starts here
        (1.5, 0.25),
        (2.0, 0.5),
        (3.0, 1.0),
        (3.5, 1.0),
        (8.0, 1.0),
    ):
        hand = bench.at(t).hand
        np.testing.assert_allclose(hand, start + share * (end - start), atol=2e-6)
    assert bench.step.a  # the other values changed at their row's time

    # Two rows at one time: the position is there at once.
    bench = Bench([BASE, {"t": 0.0, "pos": list(end)}])
    np.testing.assert_allclose(bench.step.hand, end, atol=TOL)


def test_wait_for_stops_the_scripts_clock():
    rows = [
        BASE,
        {"t": 1.0, "b": True},
        {"wait_for": "owner_human"},
        {"t": 1.5, "b": False},
        {"wait_for": "rest_hold"},
        {"t": 2.0, "a": True},
        {"wait_for": "owner_nobody"},
        {"t": 2.5, "a": False},
    ]
    bench = Bench(rows)
    assert not bench.at(0.96).b and bench.at(1.04).b
    # No ROBOT_STATE, then one that does not show it: the clock stands.
    assert bench.go(3.0).b
    bench.state = robot_state(owner="nobody")
    assert bench.go(3.0).b
    # It goes on from where it stood: 0.5 s to the next row.
    bench.state = robot_state(owner="human")
    assert bench.go(0.48).b and not bench.go(0.04).b

    assert not bench.go(3.0).a  # the motion is "hold"
    bench.state = robot_state(owner="human", motion="rest_hold")
    assert not bench.go(0.48).a and bench.go(0.04).a

    assert bench.go(3.0).a
    bench.state = robot_state(owner="nobody", motion="rest_hold")
    assert bench.go(0.48).a and not bench.go(0.04).a

    # A wait that already holds stops nothing; without a state callable none ends.
    bench = Bench(rows[:4])
    bench.state = robot_state(owner="human")
    assert bench.at(1.48).b and not bench.at(1.52).b
    assert set(WAITS) == {"owner_human", "owner_nobody", "rest_hold"}
    alone = Bench(rows[:4], with_state=False)
    alone.state = robot_state(owner="human")
    assert alone.at(9.0).b


def test_an_absent_controller_has_no_sample_age():
    rows = [
        {**BASE, "b": True},
        {"t": 1.0, "absent": True},
        {"t": 2.0, "absent": False, "age_s": 0.04},
        {"t": 3.0, "pose_valid": False},
    ]
    bench = Bench(rows)
    assert (bench.fed.sample_age_s, bench.fed.stage) == (0.0, "ready")

    # Not active: the graph ran on no controller, and there is no sample age.
    absent = bench.at(1.5)
    assert (bench.fed.sample_age_s, bench.fed.stage) == (None, "no_controller")
    assert bench.fed.outputs is not None and bench.fed.note
    assert not absent.present and absent.hand is None and not absent.b

    back = bench.at(2.5)
    assert (bench.fed.sample_age_s, bench.fed.stage, bench.fed.note) == (
        0.04,
        "ready",
        "",
    )
    assert back.present and back.b

    # Active with a pose that is not valid: the row's sample age, and the
    # buttons are still read.
    lost = bench.at(3.5)
    assert (bench.fed.sample_age_s, bench.fed.stage) == (0.04, "no_controller")
    assert bench.fed.note and not lost.present and lost.hand is None and lost.b


def test_a_squeeze_that_is_no_number_is_no_controller_data():
    bench = Bench([BASE, {"t": 0.5, "squeeze": float("nan"), "b": True}])
    assert bench.at(0.4).present and bench.fed.stage == "ready"
    lost = bench.at(0.6)
    assert (bench.fed.sample_age_s, bench.fed.stage) == (0.0, "no_controller")
    assert bench.fed.note and not lost.present and lost.squeeze == 0.0 and lost.b


def test_open_sets_the_script_back_to_its_start():
    rows = [BASE, {"t": 1.0, "b": True}, {"wait_for": "owner_human"}, {"t": 1.5}]
    bench = Bench(rows)
    assert bench.at(3.0).b  # the clock stands at the wait
    # The program opens the feed once the robot app's WELCOME is there.
    bench.feed.open()
    assert not bench.go(0.96).b and bench.go(0.08).b


def test_the_script_lift_releases_b_and_a():
    bench = Bench("lift")
    assert bench.feed.name == "lift"
    assert not bench.at(2.96).b and bench.at(3.04).b
    assert bench.go(4.0).b  # held until a person owns the arm
    bench.state = robot_state(owner="human")
    assert bench.go(0.48).b and not bench.go(0.04).b  # script time 3.52

    # The squeezed hand rises 5 cm from script time 5.0 to 9.0.
    low = bench.go(1.46)  # 4.98
    assert low.squeeze == 0.0 and low.hand[1] == pytest.approx(1.2, abs=TOL)
    assert bench.go(0.04).squeeze == 1.0  # 5.02
    assert bench.go(1.98).hand[1] == pytest.approx(1.225, abs=1e-5)  # 7.0
    top = bench.go(2.4)  # 9.4
    assert top.squeeze == 1.0 and top.hand[1] == pytest.approx(1.25, abs=TOL)
    assert bench.go(0.12).squeeze == 0.0  # 9.52

    # A is pressed once the arm is at rest and released once it is given back.
    assert not bench.go(4.0).a
    bench.state = robot_state(owner="human", motion="rest_hold")
    assert not bench.go(1.48).a and bench.go(0.04).a
    assert bench.go(4.0).a
    bench.state = robot_state(owner="nobody", motion="rest_hold")
    assert bench.go(0.48).a and not bench.go(0.04).a
    end = bench.go(5.0)
    assert end.present and not (end.a or end.b) and end.squeeze == 0.0

    # The other built-in script holds a controller with nothing pressed.
    idle = Bench("idle")
    for step in (idle.step, idle.go(20.0)):
        assert step.present and not (step.a or step.b)
        assert (step.squeeze, step.trigger) == (0.0, 0.0)
    assert (idle.feed.name, idle.fed.stage) == ("idle", "ready")
    assert set(SCRIPTS) == {"lift", "idle", "grasp"}
    known = r"no synthetic controller 'wave'.*grasp, idle, lift"
    with pytest.raises(ValueError, match=known):
        ScriptFeed.named(idle.graph, "wave")


def test_the_script_grasp_pulls_and_releases_the_trigger_and_never_squeezes():
    bench = Bench("grasp")
    assert bench.feed.name == "grasp"
    hands, squeezes = [bench.step.hand], {bench.step.squeeze}

    def go(seconds: float) -> GraphStep:
        """Run on; the hand and the squeeze of the last pass are kept."""
        step = bench.go(seconds)
        hands.append(step.hand)
        squeezes.add(step.squeeze)
        return step

    # Right B is held until a person owns the arm.
    assert not go(2.96).b and go(0.08).b  # script time 3.04
    assert go(4.0).b
    bench.state = robot_state(owner="human")
    assert go(0.48).b and not go(0.04).b  # 3.52

    # The trigger: pulled all the way from script time 5.0 to 9.0, then released.
    assert go(1.46).trigger == 0.0  # 4.98
    assert go(0.04).trigger == 1.0  # 5.02
    assert go(3.96).trigger == 1.0  # 8.98
    assert go(0.04).trigger == 0.0  # 9.02

    # The script's clock stands at 9.0 until the arm holds at rest; right A
    # goes down 6 s later and up once the arm was given back.
    assert not go(8.0).a
    bench.state = robot_state(owner="human", motion="rest_hold")
    assert not go(5.96).a and go(0.08).a  # 15.04
    assert go(4.0).a
    bench.state = robot_state(owner="nobody", motion="rest_hold")
    assert go(0.48).a and not go(0.04).a
    end = go(5.0)
    assert end.present and not (end.a or end.b) and end.trigger == 0.0
    # The arm is not asked to move: no squeeze, and the hand stays where it is.
    assert squeezes == {0.0}
    np.testing.assert_allclose(hands, [HAND] * len(hands), atol=TOL)


@pytest.mark.parametrize(
    "rows, said",
    [
        ([{**BASE, "speed": 1.0}], "script row 0 names unknown key"),
        ([BASE, {"t": 1.0, "wait_for": "rest_hold"}], "script row 1: a wait is"),
        ([{"wait_for": "owner_agent"}], "script row 0: a wait is"),
        ([BASE, {"t": 2.0}, {"t": 1.0}], "script row 2: t must be a time of 2 s"),
        ([{"pos": list(HAND)}], "script row 0: t must be a number"),
        ([{"t": 0.0, "pos": [0.0, 1.2]}], "script row 0: pos must be 3 numbers"),
        ([{"t": 0.0, "quat": [0, 0, 0, 0]}], "script row 0: quat cannot be used"),
        ([{"t": 0.0, "b": 1}], "script row 0: b must be true or false"),
        ([{"t": 0.0, "age_s": -0.1}], "script row 0: age_s must be 0 or more"),
        (["squeeze"], "script row 0 must be a mapping"),
    ],
)
def test_a_script_row_that_cannot_be_used_is_refused(rows, said):
    with pytest.raises(ValueError, match=said):
        ScriptFeed(HumanGraph(RetargetSettings(), HOME), rows)


# The teleop feed on a stand-in session


class StandInServer:
    """What the stand-in sessions of one teleop feed do, as the test sets it."""

    def __init__(self, graph: HumanGraph, clock: FakeClock) -> None:
        self.graph = graph
        self.clock = clock
        self.sessions: list[StandInSession] = []
        # The error of each coming attempt to enter, first first; none: entered.
        self.entry_errors: list[BaseException] = []
        self.entry_takes_s = 0.0  # how long an attempt blocks
        self.step_error: BaseException | None = None
        # The right controller: a ControllerSnapshot, None when not active.
        self.right: Any = snapshot(HAND)
        self.tick_age_s: Any = 0.012

    def factory(self, config: Any) -> StandInSession:
        session = StandInSession(config, self)
        self.sessions.append(session)
        return session


class StandInSession:
    """In place of a TeleopSession: it runs the graph's pipeline on one snapshot."""

    def __init__(self, config: Any, server: StandInServer) -> None:
        self.config = config
        self.server = server
        self.entered = self.left = False
        self.steps: list[tuple[Any, Any]] = []  # (external_inputs, execution_events)
        self.last_step_info = SimpleNamespace(remote_tick_age_s=None)

    def __enter__(self) -> StandInSession:
        server = self.server
        server.clock.advance(server.entry_takes_s)
        if server.entry_errors:
            raise server.entry_errors.pop(0)
        self.entered = True
        return self

    def __exit__(self, *error: object) -> bool:
        self.left = True
        return False

    def step(self, *, external_inputs: Any, execution_events: Any) -> Any:
        server = self.server
        if server.step_error is not None:
            raise server.step_error
        self.steps.append((external_inputs, execution_events))
        now_ns = round(server.clock.monotonic() * 1e9)
        self.last_step_info = SimpleNamespace(remote_tick_age_s=server.tick_age_s)
        return server.graph.pipeline.execute_pipeline(
            {**server.graph.controller_inputs(server.right), **external_inputs},
            context=ComputeContext(
                graph_time=GraphTime(now_ns, now_ns),
                execution_events=execution_events,
            ),
        )


class TeleopBench:
    """A teleop feed on stand-in sessions and a fake clock."""

    def __init__(self, **settings: Any) -> None:
        self.settings = TeleopSettings(**{"endpoint": ENDPOINT, **settings})
        self.graph = HumanGraph(RetargetSettings(), HOME)
        self.clock = FakeClock()
        self.server = StandInServer(self.graph, self.clock)
        self.feed = TeleopFeed(
            self.graph,
            self.settings,
            session_factory=self.server.factory,
            clock=self.clock,
        )

    def step(self, after_s: float = 0.0) -> FeedStep:
        """One pass, this long after the pass before; the graph runs stopped."""
        self.clock.advance(after_s)
        self.events = self.graph.events(stopped=True)
        self.inputs = self.graph.external_inputs(False)
        return self.feed.step(self.inputs, self.events, self.clock.monotonic())

    @property
    def attempts(self) -> int:
        return len(self.server.sessions)


def not_ready(stage: str, timeout_s: float | None = 3.0) -> TimeoutError:
    """The TimeoutError of a TeleopSession that was not entered, in its words."""
    return TimeoutError(_remote.not_ready_message(ENDPOINT, stage, timeout_s))


def test_the_teleop_feed_builds_a_remote_session_config():
    bench = TeleopBench(
        tick_rate_hz=72.0, max_tick_age_s=0.2, connect_timeout_s=1.5, retry_s=2.0
    )
    assert bench.feed.source == HUMAN_SOURCE_TELEOP
    assert bench.attempts == 0  # nothing is entered before the first pass
    bench.server.entry_errors = [not_ready(_remote.SERVER_DID_NOT_ANSWER)]
    bench.step()
    fed = bench.step(2.0)
    assert fed.stage == "ready" and bench.attempts == 2

    first, second = bench.server.sessions
    assert first.config is not second.config  # a new session per attempt
    for session in (first, second):
        config = session.config
        assert config.app_name == "agentic_remote_human"
        assert config.pipeline is bench.graph.pipeline
        assert config.mode is SessionMode.REMOTE
        assert config.remote_endpoint == ENDPOINT
        assert config.remote_tick_rate_hz == 72.0
        assert config.remote_max_tick_age_s == 0.2
        assert config.remote_connect_timeout_s == 1.5
        # Retargeting runs inside step(), and nothing of CloudXR is started.
        assert config.retargeting_execution.mode is RetargetingExecutionMode.SYNC
        assert config.plugins == [] and config.oxr_handles is None
        assert config.teleop_control_pipeline is None
    # The session is stepped with what the program handed to the feed.
    assert len(second.steps) == 1
    assert second.steps[0][0] is bench.inputs and second.steps[0][1] is bench.events

    # Without a factory the session is Isaac Capture's own, on that config.
    plain = TeleopFeed(bench.graph, bench.settings)
    made = feed_module._teleop_session(plain.session_config())
    assert type(made) is TeleopSession and made.config.remote_endpoint == ENDPOINT

    bench.feed.close()
    assert second.entered and second.left and not first.left


WAITING_TEXT = "waiting for the headset to connect to CloudXR"


@pytest.mark.parametrize(
    "error, stage",
    [
        (not_ready(_remote.SERVER_DID_NOT_ANSWER), "no_server"),
        (
            not_ready(
                f"{_remote.SERVER_DID_NOT_ANSWER} (cannot connect to "
                "127.0.0.1:47600: connect: Connection refused)"
            ),
            "no_server",
        ),
        (not_ready(_remote.describe_stage(None)), "no_server"),
        (not_ready(_remote.describe_stage({})), "no_server"),
        (
            not_ready(
                _remote.describe_stage(
                    {"stage": "connecting_openxr", "message": WAITING_TEXT}
                )
            ),
            "connecting_openxr",
        ),
        (
            not_ready(_remote.describe_stage({"stage": "waiting_for_headset"})),
            "waiting_for_headset",
        ),
        (
            not_ready(
                _remote.describe_stage(
                    {"stage": "waiting_for_headset", "message": "put on the headset"}
                ),
                None,
            ),
            "waiting_for_headset",
        ),
        # Any other stage of the server, and any other TimeoutError.
        (not_ready(_remote.describe_stage({"stage": "ready"})), "no_server"),
        (not_ready(_remote.describe_stage({"stage": "running"})), "no_server"),
        (not_ready(_remote.describe_stage({"stage": ""})), "no_server"),
        (TimeoutError("timed out"), "no_server"),
        (TimeoutError("waiting_for_headset"), "no_server"),
        # Every RuntimeError: the server's ERROR text, a lost connection.
        (RuntimeError("The Teleop Session Server is busy"), "no_server"),
        (RuntimeError("Remote session lost: connection closed"), "no_server"),
        (RuntimeError(str(not_ready("waiting_for_headset"))), "no_server"),
        (ConnectionRefusedError("connection refused"), "no_server"),
    ],
)
def test_the_stage_of_a_failed_entry_is_read_from_isaac_captures_texts(error, stage):
    assert stage in HUMAN_STAGES and entry_stage(error) == stage
    bench = TeleopBench()
    bench.server.entry_errors = [error]
    expected = FeedStep(None, None, stage, str(error)[:NOTE_CHARS])
    assert bench.step() == expected
    # Until the next attempt every pass says the same.
    assert bench.step(1.0) == expected and bench.attempts == 1


def test_the_note_of_a_failed_entry_is_the_first_150_characters_of_its_text():
    text = _remote.not_ready_message(ENDPOINT, "waiting_for_headset (" + "x" * 300, 3)
    bench = TeleopBench()
    bench.server.entry_errors = [TimeoutError(text)]
    fed = bench.step()
    assert NOTE_CHARS == 150 and len(text) > 300
    assert (fed.stage, fed.note) == ("waiting_for_headset", text[:150])


def test_the_attempts_run_out_only_in_a_row():
    bench = TeleopBench(attempts=3, retry_s=2.0, connect_timeout_s=1.0)
    server = bench.server
    refused = not_ready(_remote.SERVER_DID_NOT_ANSWER)
    server.entry_errors = [refused, refused]
    assert bench.step().stage == "no_server" and bench.attempts == 1
    # The pause: no attempt before retry_s has passed.
    assert bench.step(1.9).outputs is None and bench.attempts == 1
    assert bench.step(0.2).outputs is None and bench.attempts == 2
    # The third attempt is entered: the count starts again.
    assert bench.step(2.1).stage == "ready" and bench.attempts == 3
    assert bench.step(0.02).stage == "ready" and bench.attempts == 3

    # A lost session is left; the next attempt comes after the pause.
    entered = server.sessions[-1]
    server.step_error = RuntimeError("Remote session lost: connection closed")
    lost = bench.step(0.02)
    assert lost == FeedStep(None, None, "no_server", str(server.step_error))
    assert entered.left
    server.step_error = None
    server.entry_errors = [refused, refused, refused]
    assert bench.step(1.9).outputs is None and bench.attempts == 3
    assert bench.step(0.2).outputs is None and bench.attempts == 4
    assert bench.step(2.1).outputs is None and bench.attempts == 5
    with pytest.raises(AttemptsRanOut, match="3 attempts in a row failed") as raised:
        bench.step(2.1)
    assert bench.attempts == 6 and ENDPOINT in str(raised.value)
    assert isinstance(raised.value, RuntimeError)
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert not isinstance(raised.value, EndpointRefused)

    # The pause runs from the end of an attempt, which may take seconds.
    slow = TeleopBench(attempts=5, retry_s=2.0)
    slow.server.entry_takes_s = 1.5
    slow.server.entry_errors = [refused, refused]
    slow.step()
    assert slow.step(1.9).outputs is None and slow.attempts == 1
    assert slow.step(0.2).outputs is None and slow.attempts == 2


def test_the_last_failed_attempt_raises_without_a_log_line_of_its_own(caplog):
    name = feed_module._log.name
    caplog.set_level("INFO", logger=name)

    def attempts_logged() -> list[tuple[str, str]]:
        """The feed's log lines so far that tell of an attempt."""
        records = [record for record in caplog.records if record.name == name]
        lines = [(record.levelname, record.getMessage()) for record in records]
        return [line for line in lines if line[1].startswith("attempt ")]

    def line(number: int, stage: str, error: BaseException) -> tuple[str, str]:
        return (
            "INFO",
            f"attempt {number} of 3 to enter a remote session at {ENDPOINT} "
            f"failed ({stage}): {error}",
        )

    bench = TeleopBench(attempts=3, retry_s=2.0)
    absent = not_ready(_remote.SERVER_DID_NOT_ANSWER)
    asleep = not_ready(_remote.describe_stage({"stage": "waiting_for_headset"}))
    busy = RuntimeError("The Teleop Session Server is busy")
    bench.server.entry_errors = [absent, asleep, busy]
    # An attempt before the last one is logged, with its stage and its error.
    assert bench.step().stage == "no_server"
    assert attempts_logged() == [line(1, "no_server", absent)]
    assert bench.step(2.1).stage == "waiting_for_headset"
    logged = [line(1, "no_server", absent), line(2, "waiting_for_headset", asleep)]
    assert attempts_logged() == logged

    # The last one is not: a log call can block, and the human program sends
    # its STOP before the line of its end. That line is the error's text,
    # which names the last attempt's own error.
    with pytest.raises(AttemptsRanOut) as raised:
        bench.step(2.1)
    assert bench.attempts == 3 and attempts_logged() == logged
    assert str(raised.value) == (
        f"no remote session at {ENDPOINT}: 3 attempts in a row failed, 2 s apart; "
        f"the last one: {busy}"
    )
    assert raised.value.__cause__ is busy


def test_an_endpoint_isaac_capture_cannot_use_is_refused_and_no_failed_attempt():
    endpoint = "127.0.0.1:notaport"
    bench = TeleopBench(endpoint=endpoint, attempts=1)
    # Isaac Capture's words for a port that is no number; nothing was dialed.
    wrong = ValueError(f"invalid port in endpoint '{endpoint}'")
    bench.server.entry_errors = [wrong]
    with pytest.raises(EndpointRefused) as raised:
        bench.step()
    assert str(raised.value) == f"teleop.endpoint {endpoint!r}: {wrong}"
    assert isinstance(raised.value, ValueError) and raised.value.__cause__ is wrong
    # Not an attempt that failed: another one would end the same way.
    assert not isinstance(raised.value, AttemptsRanOut) and bench.attempts == 1
    assert not bench.server.sessions[0].entered


def test_the_feed_says_ready_and_never_running():
    bench = TeleopBench()
    seen = []
    # Entered, fresh SAMPLES, an active right controller with a valid pose.
    fed = bench.step()
    assert (fed.stage, fed.sample_age_s, fed.note) == ("ready", 0.012, "")
    assert bench.graph.read(fed.outputs).present
    seen.append(fed.stage)
    for right, age in (
        (None, 0.012),  # not active
        (snapshot(HAND, valid=False), 0.012),
        (snapshot(HAND), None),  # SAMPLES older than the limit
        (snapshot(HAND), 0.0),
    ):
        bench.server.right, bench.server.tick_age_s = right, age
        seen.append(bench.step(PASS_S).stage)
    assert seen == ["ready", "no_controller", "no_controller", "stale", "ready"]
    # The server's own words for a session that runs never come through.
    stages = {
        value for name, value in vars(feed_module).items() if name.startswith("STAGE_")
    }
    assert stages == set(HUMAN_STAGES) and "running" not in stages
    assert entry_stage(not_ready("running")) == "no_server"


def test_the_teleop_feed_keeps_the_outputs_and_the_age_of_a_pass_with_no_valid_pose():
    bench = TeleopBench()
    bench.server.right = snapshot(HAND, valid=False, b=True)
    bench.server.tick_age_s = 0.03
    fed = bench.step()
    assert (fed.sample_age_s, fed.stage) == (0.03, "no_controller")
    assert fed.outputs is not None and fed.note
    step = bench.graph.read(fed.outputs)
    assert not step.present and step.b

    # So the logic reads right B from it: a STOP, before any ROBOT_STATE.
    logic = HumanLogic(bench.graph, FakeSolver(), HumanSettings(), HUMAN_SOURCE_TELEOP)
    sent = logic.step(bench.clock.monotonic(), fed, welcome(), None, None)
    assert sent == [Stop(reason="right B", sender="controller")]

    # Stale SAMPLES keep the outputs too, without an age; an age that is no
    # number of 0 or more is no age.
    for age in (None, float("nan"), -0.01, True):
        bench.server.tick_age_s = age
        stale = bench.step(PASS_S)
        assert (stale.sample_age_s, stale.stage) == (None, "stale")
        assert stale.outputs is not None and stale.note


def test_no_note_of_a_feed_begins_with_fault():
    for text in ("fault: the headset fell", "  Fault: x", "FAULT:y"):
        assert not feed_note(text).lower().lstrip().startswith("fault:")
        assert text.strip() in feed_note(text)
        assert len(feed_note(text + "z" * 200)) == NOTE_CHARS
    assert feed_note("no fault: all is well") == "no fault: all is well"

    bench = TeleopBench(retry_s=2.0)
    bench.server.entry_errors = [RuntimeError("fault: the server said so")]
    refused = bench.step()
    assert "fault: the server said so" in refused.note
    assert bench.step(2.1).stage == "ready"
    bench.server.step_error = RuntimeError("fault: lost")
    lost = bench.step(PASS_S)
    for fed in (refused, lost):
        assert fed.stage == "no_server" and not fed.note.startswith("fault: ")
    notes = [
        value for name, value in vars(feed_module).items() if name.startswith("NOTE_")
    ]
    assert len(notes) >= 4
    assert not any(str(note).lower().startswith("fault") for note in notes)


# The teleop feed on a real remote session and a server in miniature

CONNECTED = {"state": "AVAILABLE", "headset_state": "CONNECTED", "reason": "NONE"}
# What the server sends before it has an OpenXR session.
NO_HEADSET = {"state": "", "headset_state": "", "reason": ""}


def status(stage: str, message: str = "", headset: dict | None = None) -> str:
    """A STATUS of the Teleop Session Server, as JSON."""
    return json.dumps(
        {"stage": stage, "message": message, "headset": headset or CONNECTED}
    )


class MiniServer:
    """A Teleop Session Server in miniature, after Isaac Capture's own test.

    It listens on 127.0.0.1 with a port the system picks, takes one connection
    after the other on a thread and runs `script(server, host)` on each. The
    script returns when the test sets `drop`; the connection is closed then.
    """

    def __init__(self, script: Callable[[MiniServer, Any], None]) -> None:
        from isaaccapture.deviceio_session import RemoteSessionListener

        self.listener = RemoteSessionListener("127.0.0.1", 0)
        self.endpoint = f"127.0.0.1:{self.listener.port}"
        self.hellos: list[dict[str, Any]] = []  # the HELLO of each connection
        self.error: BaseException | None = None
        self.drop = threading.Event()
        self.ticking = threading.Event()  # cleared: no SAMPLES go out
        self.ticking.set()
        self._script = script
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from isaaccapture.deviceio_session import RemoteSessionHost

        try:
            while not self._ended.is_set():
                accepted = self.listener.accept(50)
                if accepted is None:
                    continue
                with RemoteSessionHost.adopt(accepted[0], False) as host:
                    hello = host.wait_hello(WAIT_MS)
                    assert hello is not None, "no HELLO"
                    self.hellos.append(json.loads(hello))
                    self._script(self, host)
                self.drop.clear()
        except Exception as error:  # reported by close()
            self.error = error
        finally:
            self.listener.close()

    def idle(self, host: Any) -> None:
        while not self.drop.is_set() and host.alive:
            time.sleep(0.005)

    def close(self) -> None:
        self._ended.set()
        self.drop.set()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive(), "the server in miniature did not end"
        if self.error is not None:
            raise self.error


def acknowledge(host: Any) -> None:
    ack = {"session_id": "mini", "server_version": isaaccapture.__version__}
    host.send_hello_ack(json.dumps(ack))


def wait_for_the_headset(server: MiniServer, host: Any) -> None:
    """HELLO_ACK, then the stage of a server whose headset is not tracked."""
    acknowledge(host)
    host.send_status(status("waiting_for_headset", "put on the headset"))
    server.idle(host)


def serve(server: MiniServer, host: Any) -> None:
    """HELLO_ACK, the first stage, READY, then a tick every 5 ms.

    The session reads the keyboard, which needs no OpenXR. The feed asks for
    the controllers, so the topic filter drops every record: each SAMPLES is
    empty and still carries its tick time.
    """
    from isaaccapture import oxr
    from isaaccapture.deviceio_session import VendorConfig
    from isaaccapture.deviceio_trackers import KeyboardTracker

    acknowledge(host)
    host.send_status(status("connecting_openxr", WAITING_TEXT, NO_HEADSET))
    host.start_session(
        [(KeyboardTracker(), "0")],
        oxr.OpenXRSessionHandles(1, 1, 1, 1),
        VendorConfig(),
        server.hellos[-1]["topics"],
    )
    host.send_ready(status("ready"))
    host.send_status(status("running"))  # the server's word from here on
    while not server.drop.is_set() and host.alive:
        if server.ticking.is_set():
            host.tick()
        time.sleep(0.005)


class RemoteBench:
    """A teleop feed with Isaac Capture's own TeleopSession, in real time."""

    def __init__(self, endpoint: str, **settings: Any) -> None:
        values = {"connect_timeout_s": 5.0, "retry_s": 0.3, "attempts": 20, **settings}
        self.settings = TeleopSettings(endpoint=endpoint, **values)
        self.graph = HumanGraph(RetargetSettings(), HOME)
        self.feed = TeleopFeed(self.graph, self.settings)
        self.stages: list[str] = []  # of every pass so far

    def step(self) -> FeedStep:
        events = self.graph.events(stopped=True)
        inputs = self.graph.external_inputs(False)
        fed = self.feed.step(inputs, events, time.monotonic())
        self.stages.append(fed.stage)
        return fed

    def until(self, stage: str) -> FeedStep:
        """Run passes, 10 ms apart, until one has this stage; that pass."""
        end = time.monotonic() + WAIT_S
        while True:
            fed = self.step()
            if fed.stage == stage:
                return fed
            assert time.monotonic() < end, f"no stage {stage}; had {self.stages}"
            time.sleep(0.01)


@pytest.fixture
def mini() -> Iterator[Callable[..., MiniServer]]:
    """Starts servers in miniature; each is closed when the test is over."""
    pytest.importorskip("isaaccapture.deviceio_session")
    made: list[MiniServer] = []

    def start(script: Callable[[MiniServer, Any], None]) -> MiniServer:
        made.append(MiniServer(script))
        return made[-1]

    yield start
    for server in made:
        server.close()


@pytest.fixture
def remote() -> Iterator[Callable[..., RemoteBench]]:
    """Builds teleop feeds on real remote sessions; each is closed afterwards."""
    pytest.importorskip("isaaccapture.deviceio_session")
    made: list[RemoteBench] = []

    def build(endpoint: str, **settings: Any) -> RemoteBench:
        made.append(RemoteBench(endpoint, **settings))
        return made[-1]

    yield build
    for bench in made:
        bench.feed.close()


def test_no_listener_gives_no_server(remote):
    # Bound and not listening: the port is this test's, and a dial is refused.
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{held.getsockname()[1]}"
        bench = remote(endpoint, connect_timeout_s=0.3, retry_s=2.0)
        started = time.monotonic()
        fed = bench.step()
        took = time.monotonic() - started
        assert (fed.outputs, fed.sample_age_s, fed.stage) == (None, None, "no_server")
        assert "server did not answer" in fed.note and len(fed.note) <= NOTE_CHARS
        assert 0.25 < took < 5.0  # the attempt lasted its time limit
        # No attempt during the pause: the pass returns at once.
        started = time.monotonic()
        assert bench.step() == fed
        assert time.monotonic() - started < 0.2


@pytest.mark.parametrize("endpoint", ["127.0.0.1:notaport", "127.0.0.1:"])
def test_a_port_that_is_no_number_is_refused_by_isaac_capture_itself(endpoint, remote):
    # Neither text names a port, so nothing is dialed.
    bench = remote(endpoint, connect_timeout_s=3.0, retry_s=2.0, attempts=3)
    started = time.monotonic()
    with pytest.raises(EndpointRefused) as raised:
        bench.step()
    said = f"teleop.endpoint {endpoint!r}: invalid port in endpoint '{endpoint}'"
    assert str(raised.value) == said
    assert time.monotonic() - started < 2.0  # at once: no attempt ran its time


def test_a_server_waiting_for_the_headset_gives_that_stage(mini, remote):
    server = mini(wait_for_the_headset)
    bench = remote(server.endpoint, connect_timeout_s=0.5, tick_rate_hz=72.0)
    fed = bench.step()
    assert (fed.outputs, fed.sample_age_s) == (None, None)
    assert fed.stage == "waiting_for_headset"
    assert "waiting_for_headset (put on the headset)" in fed.note
    # What the feed asked the server for: the two controllers, at its tick rate.
    (hello,) = server.hellos
    assert hello["app_name"] == "agentic_remote_human"
    assert hello["tick_rate_hz"] == 72.0
    assert [tracker["type"] for tracker in hello["trackers"]] == ["ControllerTracker"]


def test_a_ready_server_with_no_controller_gives_no_controller_and_a_fresh_age(
    mini, remote
):
    server = mini(serve)
    bench = remote(server.endpoint, max_tick_age_s=0.15)
    fed = bench.until("no_controller")
    assert fed.outputs is not None and fed.note
    assert 0.0 <= fed.sample_age_s < 0.15
    step = bench.graph.read(fed.outputs)
    assert not step.present and not step.engaged and step.hand is None
    # It goes on so, pass after pass, with the server saying "running".
    for _ in range(20):
        time.sleep(0.01)
        fed = bench.step()
        assert fed.stage == "no_controller" and 0.0 <= fed.sample_age_s < 0.15
    assert set(bench.stages) <= {"stale", "no_controller"}
    assert len(server.hellos) == 1


def test_ticks_that_stop_give_stale_and_no_age(mini, remote):
    server = mini(serve)
    bench = remote(server.endpoint, max_tick_age_s=0.15)
    bench.until("no_controller")
    server.ticking.clear()
    stale = bench.until("stale")
    assert stale.sample_age_s is None and stale.outputs is not None and stale.note
    assert not bench.graph.read(stale.outputs).present
    # The session stays entered, and fresh SAMPLES bring the age back.
    server.ticking.set()
    fresh = bench.until("no_controller")
    assert 0.0 <= fresh.sample_age_s < 0.15
    assert "no_server" not in bench.stages and len(server.hellos) == 1


def test_a_server_that_closes_gives_no_server_and_is_entered_again(mini, remote):
    server = mini(serve)
    bench = remote(server.endpoint, retry_s=0.3)
    bench.until("no_controller")
    server.drop.set()
    lost = bench.until("no_server")
    assert (lost.outputs, lost.sample_age_s) == (None, None)
    assert "Remote session lost" in lost.note
    again = bench.until("no_controller")
    assert again.outputs is not None and again.sample_age_s is not None
    assert len(server.hellos) == 2
