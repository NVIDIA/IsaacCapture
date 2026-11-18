# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The human program in real time against the rig's robot app on the simulated arm.

The human program is the program through which a person drives the arm with
the headset's right controller; an engagement is one stretch in which the arm
follows the hand, a pass one turn of its loop (0.02 s). Its main() runs on
configs/human_rebot.yaml with a script feed (a synthetic controller: rows of
timed controller data). The robot app, the program that talks to the motors,
runs on configs/robot_app_rebot.yaml, with human driving switched on and no
live view asked, in memory. Its arm is the simulated one, which comes after
its joint target with a time constant of 0.05 s and takes 8 ms for one joint
read; between the two is a robot link service on a real link.

Both run on the machine's clock: a time measured here depends on what else
the machine does. Nothing here touches an arm, a headset or a network beyond
127.0.0.1. The file is skipped without Isaac Capture.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("isaaccapture.retargeters")

from isaaccapture_examples.agentic_remote.human import (  # noqa: E402
    logic as logic_module,
)
from isaaccapture_examples.agentic_remote.human.feed import ScriptFeed  # noqa: E402
from isaaccapture_examples.agentic_remote.human.ik import Solver  # noqa: E402
from isaaccapture_examples.agentic_remote.robot_app.config import (  # noqa: E402
    load_config,
)
from isaaccapture_examples.agentic_remote.robot_app.human import (  # noqa: E402
    SPEED_MARGIN,
)
from test_human_program import (  # noqa: E402
    BASE,
    CONFIGS,
    HAND,
    LEFT,
    NO_SERVER,
    PASS_S,
    RIG_ROBOT_FILE,
    WAIT_S,
    folder,  # noqa: F401
    start,  # noqa: F401
)

pytestmark = [pytest.mark.link, pytest.mark.apai]

# The arm raised over the table, away from its stops.
RAISED = [0.3, 1.2, 1.2, 0.2, 0.3, 0.1]
# A script's first rows: right B held until the person owns the arm.
TAKE = [
    BASE,
    {"t": 0.3, "b": True},
    {"wait_for": "owner_human"},
    {"t": 0.5, "b": False},
]
# The hand goes 30 cm to the left in 0.3 s: the joints at their limit for
# about a second and a half.
FAR_LEFT = list(HAND + 0.30 * np.array(LEFT))
# The fastest a joint of the rig's robot app moves in one setpoint, rad, and
# the farthest a human target may lie from its last setpoint.
CLAMP = SPEED_MARGIN * 0.8 * PASS_S
STEP_LIMIT = 0.05


class Recorded(ScriptFeed):
    """A script feed that keeps, per pass, when it ran and the grip's squeeze."""

    def __init__(self, *args: Any, **keywords: Any) -> None:
        super().__init__(*args, **keywords)
        self.passes: list[tuple[float, float]] = []

    def step(self, external_inputs: Any, events: Any, now_mono: float) -> Any:
        out = super().step(external_inputs, events, now_mono)
        self.passes.append((now_mono, float(self._values["squeeze"])))
        return out

    def first(self, squeezed: bool, after: float = 0.0) -> float:
        """When the first pass after `after` ran with the grip squeezed, or not."""
        for now, squeeze in self.passes:
            if now > after and (squeeze > 0.5) == squeezed:
                return now
        raise AssertionError("no such pass")


class Trace:
    """The simulated arm's own pose, sampled every 2 ms."""

    def __init__(self, arm: Any) -> None:
        self._arm = arm
        self.rows: list[tuple[float, np.ndarray]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        arm = self._arm
        while not self._stop.is_set():
            with arm._lock:
                arm._advance()
                self.rows.append((arm._t, arm._q.copy()))
            time.sleep(0.002)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)

    def since(self, when: float) -> np.ndarray:
        """The poses sampled from `when` on."""
        return np.array([q for t, q in self.rows if t >= when])


@contextlib.contextmanager
def running(
    make_app: Callable[..., Any],
    folder: Path,  # noqa: F811
    start: Callable[..., Any],  # noqa: F811
    rows: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[types.SimpleNamespace]:
    """Both programs, started; the person owns the arm when the block begins.

    Gives h (the robot app's harness), program, feed, logic, trace, solver,
    ended (how each human drive ended), leads (how far each human target
    lay from the robot app's last setpoint when it took it, rad) and notes
    (the note of each HUMAN_SETPOINT the program made).
    """
    from link_stack import Stack
    from test_takeover import drives

    robot = load_config(RIG_ROBOT_FILE)
    human = dataclasses.replace(robot.human, enabled=True, require_live_view=False)
    settings = yaml.safe_load((CONFIGS / "human_rebot.yaml").read_text())
    config = dataclasses.replace(robot, human=human)
    h = make_app(config=config, pose=RAISED, connected=False)
    ended = drives(h)
    leads: list[float] = []
    clamped = h.app._human_drive._clamped

    def noted(q: Any, q_prev: Any) -> list[float]:
        leads.append(max(abs(a - b) for a, b in zip(q, q_prev)))
        return clamped(q, q_prev)

    h.app._human_drive._clamped = noted
    feeds: list[Recorded] = []
    logics: list[Any] = []

    notes: list[str] = []

    class Kept(logic_module.HumanLogic):
        def __init__(self, *args: Any, **keywords: Any) -> None:
            super().__init__(*args, **keywords)
            logics.append(self)

        def step(self, *args: Any, **keywords: Any) -> list[Any]:
            out = super().step(*args, **keywords)
            notes.extend(message.note for message in out if hasattr(message, "note"))
            return out

    monkeypatch.setattr(logic_module, "HumanLogic", Kept)

    def feed(graph: Any, state: Any) -> Recorded:
        feeds.append(Recorded(graph, rows, state=state))
        return feeds[-1]

    stack = Stack(h, folder)
    trace = Trace(h.arm)
    try:
        stack.start_service()
        stack.dial()
        watcher = stack.client("tool")
        stack.connected(watcher)
        settings["socket_path"] = stack.path
        settings["teleop"]["endpoint"] = NO_SERVER
        file = folder / "human_realtime.yaml"
        file.write_text(yaml.safe_dump(settings), encoding="utf-8")
        program = start(file, *("--report-s", "0.2"), feed=feed)
        watcher.ws.wait_state(lambda s: s.owner == "human", WAIT_S)

        def wait_hold() -> None:
            watcher.ws.wait_state(
                lambda s: not s.health["human"]["driving"] and s.motion == "hold",
                WAIT_S,
            )

        yield types.SimpleNamespace(
            h=h,
            program=program,
            feed=feeds[0],
            logic=logics[0],
            trace=trace,
            solver=Solver.from_apai(settings["ik"]["apai_arm"]),
            ended=ended,
            leads=leads,
            notes=notes,
            wait_hold=wait_hold,
        )
    finally:
        trace.close()
        stack.close()


def sent_of(h: Any) -> tuple[np.ndarray, np.ndarray]:
    """When each setpoint went to the arm, and the setpoints."""
    rows = list(h.arm.sent)
    return np.array([t for _, t in rows]), np.array([q for q, _ in rows])


def own_ends(logic: Any) -> dict[str, int]:
    """How many engagements the human program itself ended, by cause."""
    counts = logic.counts()
    keys = ("no_solution", "jump", "solution_ahead", "hand_ahead", "limiter")
    return {key: counts[key] for key in keys if counts[key]}


# A quick second squeeze. The grip is released with the arm at full joint
# speed and squeezed again a moment later. The first ROBOT_STATE that shows
# the drive as ended still has joint readings taken while the arm moved, and
# the arm goes on for a moment after it: an engagement begun there starts
# behind the arm, and the robot app refuses it or drives the arm back. The
# human program takes the squeeze once two readings of every joint, both
# begun after the robot app's last setpoint, agree.

RELEASE_S = 2.0


@pytest.mark.parametrize("run_", [0, 1, 2])
@pytest.mark.parametrize("gap_s", [0.04, 0.06, 0.10, 0.20])
def test_a_second_squeeze_right_after_a_release_at_full_speed_starts_at_rest(
    make_realtime_robot_app,
    folder,  # noqa: F811
    start,  # noqa: F811
    monkeypatch,
    gap_s,
    run_,
):
    rows = [
        *TAKE,
        {"t": 1.0, "squeeze": 1.0},
        {"t": 1.2},
        {"t": 1.5, "pos": FAR_LEFT},
        {"t": RELEASE_S, "squeeze": 0.0},
        {"t": RELEASE_S + gap_s, "squeeze": 1.0},
        {"t": RELEASE_S + gap_s + 1.0, "squeeze": 0.0},
    ]
    with running(make_realtime_robot_app, folder, start, rows, monkeypatch) as run:
        h, program, feed = run.h, run.program, run.feed
        program.wait_said("human program: grip up", 2)
        time.sleep(0.3)
        run.wait_hold()
        released = feed.first(False, feed.first(True))
        again = feed.first(True, released)
        times, targets = sent_of(h)
        first, second = targets[times <= again], targets[times > again]
        # The arm was at full joint speed when the grip was released.
        speed = np.abs(np.diff(first, axis=0)).max(axis=1)[-6:-1].max() / PASS_S
        assert speed > 0.68
        # Both squeezes drove, each until its grip was released: the robot
        # app refused nothing and the human program ended nothing.
        assert [end.status for end in run.ended] == ["released", "released"]
        assert h.heard("human") == [] and own_ends(run.logic) == {}
        assert run.logic.counts()["engagements"] == 2 and len(second) > 30
        # The arm did not go back: on its fastest joint, from the release on,
        # it never stood further back than it had been.
        joint = int(np.argmax(np.abs(first[-1] - np.array(RAISED))))
        way = np.sign(first[-1][joint] - RAISED[joint])
        along = way * run.trace.since(released)[:, joint]
        back = float((np.maximum.accumulate(along) - along).max())
        assert back <= 0.002
        # The second drive's first setpoint: within the bound on the wait,
        # 0.5 s from the engagement's end, and what a start costs.
        began = float(times[times > again][0] - released)
        assert began < logic_module.REST_WAIT_S + 0.25
        # The note said to wait meanwhile. How many passes depends on how
        # long before the arm's rest the squeeze came.
        waited = run.notes.count(logic_module.NOTE_STOPPING)
        assert waited >= 1
        print(
            f"\nsqueezed again {1000 * gap_s:.0f} ms after a release at {speed:.2f} "
            f"rad/s: the second drive's first setpoint {began:.2f} s after the "
            f"release; the arm went back {back:.4f} rad; the note said to wait "
            f"on {waited} passes"
        )
        assert program.end() == 0


# A pause of the robot app at full joint speed. With every setpoint of a
# person's drive the robot app sends a ROBOT_STATE. When the next one is
# overdue, one and a half times the time from one to the next, the human
# program gives the robot app no more credit for the time since and holds
# its joint target two of the robot app's steps ahead of the last joint
# target it reported: when the robot app goes on, the joint target is inside
# its step limit, and the pause has cost speed and not the engagement.

MOVE = [
    *TAKE,
    {"t": 1.0, "squeeze": 1.0},
    {"t": 1.2},
    {"t": 1.5, "pos": FAR_LEFT},
    {"t": 3.6, "squeeze": 0.0},
]


def paused_run(
    make_app: Any,
    folder: Path,  # noqa: F811
    start: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    pause_s: float,
) -> types.SimpleNamespace:
    """One 30 cm move with the robot app paused once at full joint speed.

    kind "drive": its drive thread stands, in a send to the motors. "app":
    its dispatcher stands too, which serves the link and sends the timed
    ROBOT_STATEs. "": no pause.
    """
    with running(make_app, folder, start, MOVE, monkeypatch) as run:
        h, program = run.h, run.program
        send = h.arm.send_target
        before = len(h.arm.sent)
        paused: list[float] = []

        def hold_app() -> None:
            with h.app._lock:
                time.sleep(pause_s)

        def send_target(q: Any) -> None:
            if kind and not paused and len(h.arm.sent) - before >= 30:
                paused.append(time.monotonic())
                if kind == "app":
                    threading.Thread(target=hold_app, daemon=True).start()
                time.sleep(pause_s)
            send(q)

        h.arm.send_target = send_target
        program.wait_said("human program: grip down")
        program.wait_said("human program: grip up")
        time.sleep(0.2)
        run.wait_hold()
        times, targets = sent_of(h)
        times, targets = times[before:], targets[before:]
        steps = np.abs(np.diff(targets, axis=0)).max(axis=1)
        moving = np.nonzero(steps > 1e-6)[0]
        tip = run.solver.fk(targets[-1])[:3, 3] - run.solver.fk(RAISED)[:3, 3]
        out = types.SimpleNamespace(
            ended=[(end.status, end.sent) for end in run.ended],
            reasons=[end.reason for end in run.ended],
            heard=[event[1] for event in h.heard("human")],
            own=own_ends(run.logic),
            steps=steps,
            on_its_way_s=float(times[moving[-1] + 1] - times[moving[0]]),
            moved=tip,
            gap_s=float(np.diff(times).max()),
            lead=max(run.leads) if run.leads else 0.0,
            paused=bool(paused),
            states=run.logic.counts(),
        )
        assert program.end() == 0
    return out


def test_a_hand_30_cm_ahead_is_caught_up_with_at_the_joint_speed_limit(
    make_realtime_robot_app,
    folder,  # noqa: F811
    start,  # noqa: F811
    monkeypatch,
):
    # The healthy case: nothing pauses. The arm arrives where the hand
    # stopped, 30 cm to the left, in about a second and a half.
    out = paused_run(make_realtime_robot_app, folder, start, monkeypatch, "", 0.0)
    assert [status for status, _ in out.ended] == ["released"]
    assert out.heard == [] and out.own == {}
    assert out.steps.max() <= CLAMP + 1e-9
    np.testing.assert_allclose(out.moved, [0.0, 0.30, 0.0], atol=2e-3)
    # No target lay further from the robot app's last setpoint than its step
    # limit: two of the human program's own steps, 0.028 rad, on a machine
    # that does nothing else.
    assert out.lead < STEP_LIMIT
    assert out.on_its_way_s < 1.9
    print(
        f"\n30 cm at the joint speed limit: on its way {out.on_its_way_s:.2f} s; the "
        f"farthest a target lay from the robot app's last setpoint {out.lead:.4f} rad"
    )


@pytest.mark.parametrize("run_", range(6))
@pytest.mark.parametrize("pause_ms", [60, 90])
@pytest.mark.parametrize("kind", ["drive", "app"])
def test_a_short_pause_of_the_robot_app_costs_speed_and_not_the_engagement(
    make_realtime_robot_app,
    folder,  # noqa: F811
    start,  # noqa: F811
    monkeypatch,
    kind,
    pause_ms,
    run_,
):
    out = paused_run(
        make_realtime_robot_app, folder, start, monkeypatch, kind, pause_ms / 1000.0
    )
    assert out.paused and out.gap_s >= pause_ms / 1000.0
    # The pause was ridden out: one drive, ended by the released grip, with
    # nothing refused and nothing ended, and the arm at the hand.
    assert [status for status, _ in out.ended] == ["released"], (out.reasons, out.own)
    assert out.heard == [] and out.own == {}
    np.testing.assert_allclose(out.moved, [0.0, 0.30, 0.0], atol=2e-3)
    # When the robot app went on, the joint target was inside its step limit,
    # and no setpoint of the robot app was a larger step than its clamp.
    assert out.lead < STEP_LIMIT
    assert out.steps.max() <= CLAMP + 1e-9
    print(
        f"\n{kind} paused {pause_ms} ms: ridden out; the farthest a target lay from "
        f"the robot app's last setpoint {out.lead:.4f} rad (the limit is "
        f"{STEP_LIMIT}); on its way {out.on_its_way_s:.2f} s"
    )


@pytest.mark.parametrize("pause_ms", [200, 500])
def test_a_long_pause_of_the_robot_apps_drive_ends_by_its_own_late_setpoint_rule(
    make_realtime_robot_app,
    folder,  # noqa: F811
    start,  # noqa: F811
    monkeypatch,
    pause_ms,
):
    out = paused_run(
        make_realtime_robot_app, folder, start, monkeypatch, "drive", pause_ms / 1000.0
    )
    # executor.large_delay_s of the rig file, 0.1 s: the robot app's own rule.
    assert [status for status, _ in out.ended] == ["late"]
    assert "s late, over the limit of 0.1 s" in out.reasons[0]
    assert out.steps.max() <= CLAMP + 1e-9
    assert len(out.heard) == 1 and "late" in out.heard[0]
