# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program end to end: APAI's agent, the robot link, the robot app.

The whole runs put the launcher (launcher.main, in this process) on APAI's
mock scene with the robot app on APAI's mock arm in real time: on a loopback
carrier, and behind the real robot link service on a link of 127.0.0.1. A
Trace notes, on the robot app, every PERMIT it got and the PERMIT held when
the driver took each joint target.

The fast tests put the arm adapter on FakeLink, a robot side that plays the
setpoints it holds a PERMIT for: with FakeCheckAhead, and with APAI's checker
on a grid fed by the link's stereo camera from a FakePairSource.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
import re
import signal
import threading
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, NamedTuple
from unittest import mock

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import checkahead, launcher, task_gate
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent.arm_adapter import (
    AGENT_TERMINAL,
    ArmSettings,
    RobotLinkArm,
    RobotLinkError,
)
from isaaccapture_examples.agentic_remote.agent.camera.frame_source import StereoPair
from isaaccapture_examples.agentic_remote.agent.checkahead import (
    CheckAhead,
    Evidence,
    FrameTimes,
    Grant,
)
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink
from isaaccapture_examples.agentic_remote.protocol.messages import MOVE_SENDING
from link_bench import decode, until
from test_arm_adapter import PERIOD_S, PROFILE, ScriptedRig, ScriptedThor
from test_launcher import (
    LOOK,
    PICK,
    QUICK,
    SPEED,
    Stage,
    scene_robot_config,
    settings_file,
)

from conftest import APAI_SRC_ENV, EXPERIMENT_DIR_ENV, FakeGridClient, apai_runtime_env

apai = pytest.mark.apai
link = pytest.mark.link
experiment = pytest.mark.experiment

# Real seconds one whole run may take; the pick and place takes about 22 s.
RUN_BUDGET_S = 60.0
# A pose of APAI's arm model at which its checker lets the arm move.
HOME = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)


# What a run did, noted on the robot app


class Target(NamedTuple):
    """One joint target the driver took, and the PERMIT held at that moment."""

    number: int  # of the joint target, from 1
    move_id: str  # the running move; "" when none ran
    index: int  # the setpoint's index in its move
    owner: str  # the control owner
    permit_move_id: str  # "" without a held permit
    through_index: int
    evidence_age_s: float  # of the held permit's camera frame, on the app's clock
    expires_in_s: float


class Trace:
    """What a run did, noted on the robot app and on the agent's link."""

    def __init__(self, harness: Any) -> None:
        self.h = harness
        self.targets: list[Target] = []
        # (move id, through_index, evidence_wall_s, why refused or None) of
        # every PERMIT the robot app got.
        self.thor_permits: list[tuple[str, int, float, str | None]] = []
        # (move id, through_index, evidence_wall_s, cleared through) of every
        # PERMIT the agent's link sent.
        self.agent_permits: list[tuple[str, int, float, int]] = []
        self.moves: dict[str, int] = {}  # move id -> setpoints, as sent
        self.owners: list[str] = []  # the owners ROBOT_STATE showed, changes only
        # The final RESULT the robot app sent for each command, by command id.
        self.results: dict[str, Any] = {}
        self.built: dict[str, Any] = {}  # what the launcher's hooks built
        # The last setpoint the checker callback cleared in the newest move.
        self.cleared = -1
        gate = harness.app._gate
        update = gate.update

        def noted_update(permit: Any, *args: Any, **kwargs: Any) -> str | None:
            reason = update(permit, *args, **kwargs)
            self.thor_permits.append(
                (permit.move_id, permit.through_index, permit.evidence_wall_s, reason)
            )
            return reason

        gate.update = noted_update
        harness.spy.after_target.append(self._target)

    def _target(self, number: int) -> None:
        """Runs on the robot app's motion thread, inside the driver's send."""
        app = self.h.app
        move_id, index, _ = app._executor.progress()
        held = app._gate._held
        now = self.h.clock.monotonic()
        if held is None:
            covered = ("", -1, math.inf, 0.0)
        else:
            covered = (
                held.permit.move_id,
                held.permit.through_index,
                now - held.evidence_mono,
                held.permit.expires_mono_s - now,
            )
        self.targets.append(
            Target(number, move_id, index, app.snapshot().owner.value, *covered)
        )

    def watch(self, made: AgentLink) -> None:
        """Note the link's MOVEs, PERMITs and RESULTs and its states' owners."""
        send_move, send_permit = made.send_move, made.send_permit
        take_result = made._take_result

        def noted_result(result: Any) -> None:  # on the link's receive thread
            if result.status != "accepted":
                self.results.setdefault(result.command_id, result)
            take_result(result)

        def noted_move(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            self.moves[move_id] = len(q)
            self.cleared = -1
            send_move(move_id, start, tol, q, *rest)

        def noted_permit(
            move_id: str, through: int, evidence: float, *rest: Any
        ) -> None:
            self.agent_permits.append((move_id, through, evidence, self.cleared))
            send_permit(move_id, through, evidence, *rest)

        def state(shown: Any) -> None:  # on the link's receive thread
            if not self.owners or self.owners[-1] != shown.owner:
                self.owners.append(shown.owner)

        made.send_move, made.send_permit = noted_move, noted_permit
        made._take_result = noted_result
        made.add_state_listener(state)

    @contextmanager
    def hooked(self) -> Iterator[None]:
        """While open: the launcher's hooks and every check-ahead pass are noted."""
        hooks, step = launcher.agent_hooks, checkahead.MoveCheck.step
        trace = self

        @contextmanager
        def noted_hooks(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
            with hooks(*args, **kwargs) as built:
                trace.built = built
                yield built

        def noted_step(move_check: Any, sent: int, reach: int) -> Any:
            grant = step(move_check, sent, reach)
            trace.cleared = move_check.approved_through
            return grant

        with (
            mock.patch.object(launcher, "agent_hooks", noted_hooks),
            mock.patch.object(checkahead.MoveCheck, "step", noted_step),
        ):
            yield

    def check_every_target_was_permitted(self) -> None:
        """Each joint target: inside an agent move, under a live covering PERMIT."""
        limit = self.h.config.permits.evidence_age_limit_s
        for target in self.targets:
            assert target.move_id in self.moves and target.owner == "agent", target
            assert target.permit_move_id == target.move_id, target
            assert target.through_index >= target.index, target
            assert target.expires_in_s > 0.0, target
            assert 0.0 <= target.evidence_age_s < limit, target

    def check_permits_follow_the_check_ahead(self) -> None:
        """No PERMIT reached past what the checker callback had cleared."""
        newest: dict[str, float] = {}
        for move_id, through, evidence, cleared in self.agent_permits:
            assert move_id in self.moves
            assert through <= cleared < self.moves[move_id], (through, cleared)
            # The camera evidence of a move's permits never goes back.
            assert evidence >= newest.get(move_id, -math.inf)
            newest[move_id] = evidence
        # What the robot app got is what the agent sent; none was refused.
        sent = {row[:3] for row in self.agent_permits}
        assert {row[:3] for row in self.thor_permits} <= sent
        assert [row for row in self.thor_permits if row[3] is not None] == []


class Scene(Stage):
    """Stage of test_launcher.py with a Trace, and the grid class of the test."""

    grid: type = FakeGridClient

    def __init__(self, harness: Any, folder: Path) -> None:
        super().__init__(harness, folder)
        self.trace = Trace(harness)
        # The robot link service's socket; None attaches on a loopback carrier.
        self.socket_path: str | None = None
        self.stack: Any = None  # the link_stack.Stack behind the service
        self.source: Any = None  # the stereo pair source handed to the launcher
        # Called once, on the thread that sent it, after the link's next STOP.
        self.after_stop: Callable[[], None] | None = None
        # Called with the new AgentLink before anything watches it.
        self.on_attach: Callable[[AgentLink], None] | None = None
        # (joint targets, MOVEs and GRIPPERs sent) at the answer of each owner
        # request that ended a task.
        self.at_end: list[tuple[int, int]] = []

    def attach(self, socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
        if self.socket_path is None:
            return super().attach(socket_path, timeout_s=timeout_s)
        assert socket_path == self.socket_path
        made = AgentLink.attach(socket_path, timeout_s=timeout_s)
        self.watch(made)
        return made

    def watch(self, made: AgentLink) -> None:
        if self.on_attach is not None:
            self.on_attach(made)
        super().watch(made)
        self.trace.watch(made)
        request, stop = made.request_owner, made.stop

        def request_owner(name: str, **kwargs: Any) -> Any:
            result = request(name, **kwargs)
            if name != "agent":
                self.at_end.append(self.counts())
            return result

        def stop_then(reason: str, sender: str) -> None:
            stop(reason, sender)
            hook, self.after_stop = self.after_stop, None
            if hook is not None:
                hook()

        made.request_owner, made.stop = request_owner, stop_then

    def counts(self) -> tuple[int, int]:
        return len(self.h.spy.targets), len(self.motion)

    def run(self, *argv: str, attach: Any = None, **blocks: dict[str, Any]) -> int:
        occupancy = importlib.import_module("apai.perception.occupancy")
        if self.socket_path is not None:
            blocks["link"] = {**blocks.get("link", {}), "socket_path": self.socket_path}
        path = settings_file(self.folder, **blocks)
        arguments = ["--config", str(path), "--run-dir", str(self.run_dir)]
        if "--camera" not in argv:
            arguments += ["--camera", "mock"]
        with (
            mock.patch.object(occupancy, "OccupancyClient", self.grid),
            self.trace.hooked(),
        ):
            return launcher.main(
                [*arguments, *argv],
                attach=attach or self.attach,
                ask=self.ask,
                say=self.lines.append,
                task_input=self.next_line,
                source=self.source,
            )

    def holds(self) -> None:
        """The arm holds with its motors on: no joint target comes any more."""
        spy = self.h.spy
        sent = len(spy.targets)
        time.sleep(0.25)
        assert len(spy.targets) == sent
        assert spy.released_at == [] and self.h.arm.torque_on()
        if sent:
            assert spy.pose() == pytest.approx(spy.targets[-1], abs=1e-6)


class ObstacleGrid(FakeGridClient):
    """FakeGridClient whose grid a test can put a box into.

    with_box() gives the grid a bridge reports with an obstacle there: each
    cell's distance to the box. .made are the clients built so far.
    """

    made: ClassVar[list[ObstacleGrid]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        ObstacleGrid.made.append(self)

    def with_box(self, center: Any, half_m: float) -> np.ndarray:
        """The grid with a cube of this half width at center (map frame, m)."""
        cells = np.indices(self.grid.shape, dtype=np.float32)
        gap = [
            np.maximum(np.abs(self.origin[i] + cells[i] * self.voxel - c) - half_m, 0.0)
            for i, c in enumerate(center)
        ]
        return np.sqrt(gap[0] ** 2 + gap[1] ** 2 + gap[2] ** 2).astype(np.float32)


def raise_arm(harness: Any) -> None:
    """Put the mock arm off its joint stops, holding, with an object to grip."""
    import test_apai_app
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState

    # The app started at rest, so the jaws are referenced.
    harness.spy.backend._q[:] = test_apai_app.RAISED
    harness.app._motion.set(MotionState.HOLD)
    harness.spy.backend.object_stop_frac = 0.5


def tree_state(folder: Path) -> dict[str, tuple[int, int]]:
    """(size, time of change) of every file under a folder, and each folder's name.

    A folder counts as (-1, 0): a new one is seen also while it is empty. A
    .venv is left out. Of a .git folder only the index is looked at, which a
    git status without GIT_OPTIONAL_LOCKS=0 rewrites.
    """
    state = {}
    for root, folders, files in os.walk(folder):
        folders[:] = [name for name in folders if name != ".venv"]
        if os.path.basename(root) == ".git":
            folders[:] = []
            files = [name for name in files if name == "index"]
        for name in folders:
            state[os.path.join(root, name)] = (-1, 0)
        for name in files:
            path = os.path.join(root, name)
            try:
                found = os.stat(path)
            except OSError:
                continue
            state[path] = (found.st_size, found.st_mtime_ns)
    return state


def test_the_tree_state_sees_a_new_empty_folder_and_a_rewritten_git_index(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    (tree / "tools" / ".git" / "objects").mkdir(parents=True)
    (tree / "tools" / ".git" / "index").write_bytes(b"index 1")
    (tree / ".venv" / "lib").mkdir(parents=True)
    (tree / "code.py").write_text("x = 1\n")
    before = tree_state(tree)
    assert set(before) == {
        str(tree / "tools"),
        str(tree / "tools" / ".git"),
        str(tree / "tools" / ".git" / "index"),
        str(tree / "code.py"),
    }
    # Another program's environment and git's own objects are not looked at.
    (tree / ".venv" / "lib" / "module.pyc").write_bytes(b"")
    (tree / "tools" / ".git" / "objects" / "ab").mkdir()
    assert tree_state(tree) == before
    # A new folder without a file in it, a new file, and the index rewritten.
    (tree / ".state" / "cache").mkdir(parents=True)
    assert set(tree_state(tree)) - set(before) == {
        str(tree / ".state"),
        str(tree / ".state" / "cache"),
    }
    (tree / "tools" / ".git" / "index").write_bytes(b"index 22")
    index = str(tree / "tools" / ".git" / "index")
    assert tree_state(tree)[index] != before[index]
    (tree / "__pycache__").mkdir()
    (tree / "__pycache__" / "code.pyc").write_bytes(b"")
    assert len(tree_state(tree)) == len(before) + 4


def read_only_trees() -> list[Path]:
    """APAI's checkout and, when it is named, the experiment folder."""
    trees = [Path(os.environ[APAI_SRC_ENV]).resolve().parent]
    if os.environ.get(EXPERIMENT_DIR_ENV):
        trees.append(Path(os.environ[EXPERIMENT_DIR_ENV]).resolve())
    return trees


@pytest.fixture
def scene(make_realtime_apai: Callable[..., Any], tmp_path: Path) -> Iterator[Scene]:
    """The mock scene's robot app on a loopback carrier, and one launcher run."""
    harness = make_realtime_apai(config=scene_robot_config(), connected=False)
    raise_arm(harness)
    with apai_runtime_env(tmp_path) as folder:
        yield Scene(harness, folder)


@pytest.fixture
def service_scene(
    make_realtime_apai: Callable[..., Any],
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Scene]:
    """The same robot app behind the robot link service, on a link of 127.0.0.1.

    One dial to a port the system picked; scene.stack is the link_stack.Stack.
    """
    from link_stack import Stack

    harness = make_realtime_apai(config=scene_robot_config(), connected=False)
    raise_arm(harness)
    # A short path: a socket path holds 107 bytes at most.
    stack = Stack(harness, tmp_path_factory.mktemp("ls"))
    try:
        stack.start_service()
        stack.dial()
        stack.connected(stack.client("console"))
        with apai_runtime_env(tmp_path) as folder:
            made = Scene(harness, folder)
            made.socket_path, made.stack = stack.path, stack
            yield made
    finally:
        stack.close()


@pytest.fixture(params=["loopback", pytest.param("service", marks=link)])
def any_scene(request: pytest.FixtureRequest) -> Scene:
    """The scene on a loopback carrier, and again behind the real service."""
    name = "scene" if request.param == "loopback" else "service_scene"
    return request.getfixturevalue(name)


# The whole agent program on APAI's mock scene


@apai
def test_two_tasks_run_end_to_end_and_every_setpoint_was_permitted(
    any_scene: Scene,
) -> None:
    scene, trace = any_scene, any_scene.trace
    before = [tree_state(tree) for tree in read_only_trees()]
    scene.typed = [QUICK, PICK]
    scene.answers = ["yes", "yes"]
    began = time.monotonic()
    code = scene.run("--interactive")
    took = time.monotonic() - began
    assert took < RUN_BUDGET_S

    # The life cycle: nobody, the agent for the first task, nobody when it
    # succeeded; the agent again, and nobody when the model ended the second
    # without success (the mock scene cannot confirm the place): the arm is
    # given back and nobody is called, as the experiment's launcher goes on to
    # its next task.
    assert trace.owners == ["nobody", "agent", "nobody", "agent", "nobody"]
    assert scene.order == [
        "agent:completed",
        "nobody:completed",
        "agent:completed",
        "nobody:completed",
        "close:nobody",  # only after the end request's RESULT
    ]
    # One typed yes per task, then the question at the session's end.
    first, second, at_the_end = scene.asked
    assert json.dumps(QUICK) in first and "First tool open_gripper" in first
    assert json.dumps(PICK) in second and "First tool pick_and_place" in second
    assert at_the_end == task_gate.END_QUESTION and scene.said(task_gate.END_HOLDS)
    owners = [event[2].get("owner") for event in scene.h.heard("owner")]
    # Granted, given back, granted, given back.
    assert owners == ["agent", None, "agent", None]
    assert "task ended without success: " in scene.h.heard("owner")[-1][1]
    assert scene.h.heard("help_needed") == []
    # One task of the session ended without success.
    assert code == launcher.EXIT_FAILED

    # The tasks are judged by the run folder's task reports.
    opened, picked = scene.reports()
    assert opened["success"] is True and picked["success"] is False
    (result,) = scene.results("pick_and_place")
    assert result["ok"] is True and result["picked"] == "red object"
    assert result["destination"] == "drop zone"
    assert result["return_home"]["ok"] is True

    # Every joint target was a setpoint of an agent MOVE, sent under a PERMIT
    # that reached it, had not expired and rested on a camera frame; every
    # PERMIT reached no further than the checker callback had cleared.
    assert len(trace.moves) == 8 and scene.motion.count("gripper") == 4
    assert len(trace.targets) == sum(trace.moves.values()) > 500
    trace.check_every_target_was_permitted()
    trace.check_permits_follow_the_check_ahead()
    assert len(trace.thor_permits) > 8 * len(trace.moves)
    assert all(through >= 0 for _, through, _, _ in trace.agent_permits)  # no cancel
    for move_id, count in trace.moves.items():
        assert trace.results[move_id].status == "completed"
        final = max(row[1] for row in trace.agent_permits if row[0] == move_id)
        assert final == count - 1

    # A normal end: no STOP, no torque cut, and APAI's shutdown moved nothing.
    snapshot = scene.h.app.snapshot()
    assert snapshot.stop_number == 0 and snapshot.owner.value == "nobody"
    assert "stop" not in " ".join(scene.order) and scene.h.heard("stop") == []
    assert scene.at_end[-1] == scene.counts()
    scene.holds()
    if scene.stack is not None:
        # The service saw the agent leave as nobody's owner: it asked for no help.
        until(lambda: scene.stack.service.robot_state.owner == "nobody", "nobody")
        assert scene.h.heard("help_needed") == []
    # Nothing was written into APAI's checkout or the experiment folder.
    assert [tree_state(tree) for tree in read_only_trees()] == before

    # Shown with pytest -rP.
    ages = [target.evidence_age_s for target in trace.targets]
    print(
        f"measured: run {took:.1f} s, {len(trace.targets)} joint targets, "
        f"{len(trace.agent_permits)} PERMITs sent, {len(trace.thor_permits)} "
        f"taken, camera evidence {min(ages):.3f} to {max(ages):.3f} s old"
    )


@apai
def test_the_agent_plans_within_the_speed_limit_the_robot_app_names_in_its_welcome(
    make_realtime_apai: Callable[..., Any], tmp_path: Path
) -> None:
    """The Thor's own agent speed limit travels in its WELCOME.

    The agent's file names the speed of the mock scene for the task and for
    its copy of the Thor's limit; this robot app's file holds half of it.
    """
    slower = SPEED / 2
    config = scene_robot_config()
    agent = dataclasses.replace(config.limits.agent, max_joint_speed=slower)
    limits = dataclasses.replace(config.limits, agent=agent)
    harness = make_realtime_apai(
        config=dataclasses.replace(config, limits=limits), connected=False
    )
    raise_arm(harness)
    with apai_runtime_env(tmp_path) as folder:
        scene = Scene(harness, folder)
        code = scene.run("--task", QUICK)
        assert code == launcher.EXIT_OK
        assert scene.link.welcome.agent_max_joint_speed == slower
        # Said once, at the start, with where each speed came from.
        (line,) = scene.said("joint speed: ")
        assert line == (
            f"joint speed: the agent plans at {slower:g} rad/s, the lowest of "
            f"{SPEED:g} (task.max_joint_speed) and {SPEED:g} "
            f"(thor.agent_max_joint_speed); the Thor's own agent speed limit is "
            f"{slower:g} (limits.agent.max_joint_speed, from the robot app's WELCOME)"
        )
        assert scene.lines.index(line) < scene.lines.index(scene.said("ready. ")[0])
        # The operator's question, APAI's checker and the arm adapter all hold
        # the Thor's limit, not the file's.
        assert f"Joint speed at most {slower:g} rad/s" in scene.asked[0]
        built = scene.trace.built
        assert built["harness"].limits.max_joint_vel == slower
        assert built["adapter"]._settings.agent_max_joint_speed == slower
        resolved = yaml.safe_load(
            (scene.run_dir / agent_config.RESOLVED_CONFIG).read_text()
        )
        assert resolved["safety"]["max_joint_vel"] == slower
        # What the file itself says is kept as it is.
        kept = json.loads((scene.run_dir / "agent-settings.json").read_text())
        assert kept["thor"]["agent_max_joint_speed"] == SPEED


@apai
def test_an_obstacle_on_the_path_ends_the_move_in_a_hold_and_the_arm_is_given_back(
    scene: Scene,
) -> None:
    ObstacleGrid.made.clear()
    scene.grid = ObstacleGrid
    trace = scene.trace
    with_box: list[np.ndarray] = []

    def on_attach(made: AgentLink) -> None:
        send_move = made.send_move

        def first_move(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            if not with_box:
                # A box where the gripper tip is at the end of the first move.
                kinematics = trace.built["harness"].kin
                tip = kinematics.fk(np.asarray(q[-1], dtype=float))[:3, 3]
                with_box.append(ObstacleGrid.made[-1].with_box(tip, 0.04))
            send_move(move_id, start, tol, q, *rest)

        made.send_move = first_move

    def switch_in(number: int) -> None:
        if number == 5:  # the arm is under way; the grid's next query shows it
            ObstacleGrid.made[-1].grid = with_box[0]

    scene.on_attach = on_attach
    scene.h.spy.after_target.append(switch_in)
    code = scene.run("--task", PICK)

    # The workstation checker refused a setpoint ahead of the arm.
    (result,) = scene.results("pick_and_place")
    # A failed step, as with APAI's own arm: it does not end the task.
    assert result["ok"] is False and "motion_blocked" not in result
    assert "clearance" in result["error"] and "occupancy map" in result["error"]
    assert "e-stop latched" in result["error"]  # APAI's skill stopped the arm
    ((move_id, count),) = trace.moves.items()
    # Refused by the new grid's check of a setpoint cleared before, or by the
    # checker callback at the next setpoint to clear.
    again = re.search(r"check ahead refused setpoint (\d+) of", result["error"])
    index = int(again.group(1)) if again else trace.cleared + 1
    assert 5 < index < count
    # The permits ended: one cancel, quoting the evidence of the PERMIT before
    # it; no PERMIT after it.
    permits = [row for row in trace.agent_permits if row[0] == move_id]
    assert [row[1] for row in permits].count(-1) == 1 and permits[-1][1] == -1
    assert permits[-1][2] == permits[-2][2]
    assert trace.thor_permits[-1][:2] == (move_id, -1)
    # The arm stopped short of the refused setpoint and holds; what it was
    # sent was permitted.
    sent = len(trace.targets)
    assert 5 <= sent <= index and sent < count
    trace.check_every_target_was_permitted()
    trace.check_permits_follow_the_check_ahead()
    scene.holds()
    # What the Thor reports: the move refused where its permit ended, no STOP,
    # no fault-stop.
    ended = trace.results[move_id]
    assert ended.status == "refused"
    assert ended.reason == f"permit reaches setpoint -1, not {sent}"
    assert ended.detail["sent"] == sent
    snapshot = scene.h.app.snapshot()
    assert snapshot.stop_number == 0 and snapshot.stop_level.value == "none"
    assert scene.h.heard("stop") == [] and scene.h.heard("fault") == []
    # APAI got a failed step and its skill stopped the arm; the model went on
    # to its report, without success, and the arm was given back: no help
    # request. The skill stop stands for the next task's typed yes; the
    # program asks for no restart (D4).
    assert scene.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert trace.owners == ["nobody", "agent", "nobody"]
    assert scene.h.heard("help_needed") == []
    assert scene.said("task ended without success: ")
    assert scene.said(task_gate.SKILL_STOP_TOLD) and not scene.said(task_gate.RESTART)
    assert trace.built["harness"].estopped is True and code == launcher.EXIT_FAILED
    assert trace.built["adapter"]._latched is True
    assert scene.motion.count("send_move") == 1  # nothing was sent after it
    print(
        f"measured: setpoint {index} of {count} refused, {sent} joint targets "
        f"sent, {len(permits)} PERMITs, cancel to RESULT "
        f"{trace.built['adapter'].last_move['cancel_to_result_s'] * 1e3:.0f} ms"
    )


@apai
def test_after_an_obstacle_refusal_the_next_task_runs_without_a_restart(
    scene: Scene,
) -> None:
    ObstacleGrid.made.clear()
    scene.grid = ObstacleGrid
    trace = scene.trace
    grids: dict[str, np.ndarray] = {}
    latched: list[bool] = []  # the workstation checker, at each question
    resets: list[tuple[str, str, int]] = []

    def on_attach(made: AgentLink) -> None:
        send_move = made.send_move

        def first_move(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            if not grids:
                # A box where the gripper tip is at the end of the first move.
                kinematics = trace.built["harness"].kin
                tip = kinematics.fk(np.asarray(q[-1], dtype=float))[:3, 3]
                grids["clear"] = ObstacleGrid.made[-1].grid
                grids["box"] = ObstacleGrid.made[-1].with_box(tip, 0.04)
            send_move(move_id, start, tol, q, *rest)

        made.send_move = first_move

    def switch_in(number: int) -> None:
        if number == 5:  # the arm is under way; the grid's next query shows it
            ObstacleGrid.made[-1].grid = grids["box"]

    def on_ask() -> None:
        harness = trace.built["harness"]
        latched.append(harness.estopped)
        if len(scene.asked) != 2:
            return
        # A person took the obstacle away before typing yes to the next task.
        ObstacleGrid.made[-1].grid = grids["clear"]
        reset = harness.reset_estop

        def noted_reset(**kwargs: Any) -> None:
            snapshot = scene.h.app.snapshot()
            resets.append(
                (snapshot.owner.value, snapshot.stop_level.value, len(scene.motion))
            )
            reset(**kwargs)

        harness.reset_estop = noted_reset

    scene.on_attach = on_attach
    scene.on_ask = on_ask
    scene.h.spy.after_target.append(switch_in)
    scene.typed = [PICK, PICK]
    scene.answers = ["yes", "yes"]
    began = time.monotonic()
    code = scene.run("--interactive")
    assert time.monotonic() - began < RUN_BUDGET_S

    # The first task: the check ahead refused a setpoint, APAI's skill stopped
    # the arm, the model ended the task without success and the arm was given
    # back. No STOP, no fault-stop.
    blocked, picked = scene.reports()
    first, second = scene.results("pick_and_place")
    assert first["ok"] is False and "motion_blocked" not in first
    assert "clearance" in first["error"] and blocked["success"] is False
    first_move = next(iter(trace.moves))
    assert trace.results[first_move].status == "refused"
    assert scene.said(task_gate.SKILL_STOP_TOLD)
    # The same program took the next task: it said no restart, the question
    # named the skill stop, and the latch stood until the typed yes.
    assert not scene.said(task_gate.RESTART)
    assert task_gate.SKILL_STOP_NOTE not in scene.asked[0]
    assert task_gate.SKILL_STOP_NOTE in scene.asked[1]
    # At the two tasks' questions and at the question of the session's end.
    assert latched == [False, True, False]
    assert scene.asked[2] == task_gate.END_QUESTION
    # It was cleared once, with the agent as owner on the Thor and before any
    # command of the second task.
    sent_before = scene.motion.index("send_move") + 1
    assert resets == [("agent", "none", sent_before)]
    assert scene.said("the skill stop of the last task is cleared")
    assert trace.built["harness"].estopped is False
    # The second task moved the arm: the whole pick and place.
    assert second["ok"] is True and second["picked"] == "red object"
    assert second["return_home"]["ok"] is True
    later = [move_id for move_id in trace.moves if move_id != first_move]
    assert len(later) >= 5
    assert all(trace.results[move_id].status == "completed" for move_id in later)
    assert len(trace.targets) > 300
    trace.check_every_target_was_permitted()
    trace.check_permits_follow_the_check_ahead()
    # Each task took the arm with its own grant and gave it back when the
    # model ended it without success (the mock scene cannot confirm the place).
    assert scene.order == [
        "agent:completed",
        "nobody:completed",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
    ]
    assert scene.h.heard("help_needed") == []
    assert trace.owners == ["nobody", "agent", "nobody", "agent", "nobody"]
    assert picked["success"] is False and code == launcher.EXIT_FAILED
    snapshot = scene.h.app.snapshot()
    assert snapshot.stop_number == 0 and snapshot.stop_level.value == "none"
    assert scene.h.heard("stop") == [] and scene.h.heard("fault") == []
    scene.holds()


@apai
def test_a_second_ctrl_c_sends_stop_again_and_ends_the_program_with_130(
    scene: Scene, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(number: int) -> None:
        if number == 40:
            signal.raise_signal(signal.SIGINT)

    scene.h.spy.after_target.append(interrupt)
    # The second Ctrl+C comes while APAI handles the first: right after the
    # STOP of the first one left.
    scene.after_stop = lambda: signal.raise_signal(signal.SIGINT)
    code = scene.run("--task", PICK)
    assert code == launcher.EXIT_INTERRUPTED == 130
    assert "interrupted; STOP was sent" in capsys.readouterr().err
    stop = f"stop:{AGENT_TERMINAL}"
    assert scene.order == ["agent:completed", stop, stop, "close:nobody"]
    # The second STOP left just before the link was closed.
    until(lambda: len(scene.h.heard("stop")) == 2, "the second STOP on the Thor")
    assert all(AGENT_TERMINAL in event[1] for event in scene.h.heard("stop"))
    snapshot = scene.h.app.snapshot()
    assert snapshot.stop_number == 2 and snapshot.owner.value == "nobody"
    # The move was cancelled by permit before the first STOP; a person's STOP
    # starts no help request and no assistance wait.
    assert scene.trace.agent_permits[-1][1] == -1
    assert 40 <= len(scene.h.spy.targets) < 80
    assert scene.h.heard("help_needed") == []
    assert scene.h.app._wait.remaining(scene.h.clock.monotonic()) is None
    assert scene.said("the task was cancelled")
    scene.trace.check_every_target_was_permitted()
    scene.holds()


@apai
@link
def test_a_stop_from_the_console_ends_the_task_and_the_next_one_runs(
    service_scene: Scene,
) -> None:
    scene, trace = service_scene, service_scene.trace
    console = scene.stack.stations[0]

    def press(number: int) -> None:
        if number == 40:
            console.ws.stop("typed stop", "console")

    scene.h.spy.after_target.append(press)
    scene.typed = [PICK, QUICK]
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive")

    # The STOP ended the first move where it stood.
    (stop,) = scene.h.heard("stop")
    assert stop[1].startswith("STOP from console: typed stop")
    first = next(iter(trace.moves))
    ended = trace.results[first]
    assert ended.status == "cancelled"
    assert ended.reason == "STOP from console: typed stop"
    assert 40 <= ended.detail["sent"] <= 45 < trace.moves[first]
    trace.check_every_target_was_permitted()
    # A PERMIT that crossed the STOP quotes the old stop number: refused.
    for row in trace.thor_permits:
        assert row[3] is None or "stop number 0 does not match" in row[3]
    # APAI got a halted motion: a failed step, after which the model ended
    # the task without success. Nobody owned the arm any more, so the
    # give-back, with the numbers from before the STOP, was refused and
    # nothing else was sent.
    stopped, opened = scene.reports()
    assert stopped["success"] is False
    (halted,) = scene.results("pick_and_place")
    assert halted["error"] == (
        "MotionHalted: the robot app stopped the move: STOP from console: typed stop"
    )
    assert scene.h.heard("help_needed") == []
    assert scene.said("refused by the robot: stop number 0 does not match")
    # The STOP latched the adapter and halted the workstation checker. A halt
    # is no latch: the next task took the arm with new numbers, without a
    # restart of the program and without a skill stop to clear.
    assert not scene.said(task_gate.RESTART)
    assert task_gate.SKILL_STOP_NOTE not in scene.asked[1]
    assert scene.order == [
        "agent:completed",
        "nobody:refused",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
    ]
    assert trace.owners == ["nobody", "agent", "nobody", "agent", "nobody"]
    # The stopped task did not succeed: exit code 1 for the session.
    assert opened["success"] is True and code == launcher.EXIT_FAILED
    # The next task's jaw command was not held back by the halt, and cleared
    # none: APAI clears a halt at the next skill's first arm motion.
    harness = trace.built["harness"]
    assert harness.estopped is False and harness._halt_generation == 1
    assert harness.halted is not None
    assert scene.h.app.snapshot().stop_number == 1
    scene.holds()


def watch_grasp(
    scene: Scene, at_first: Callable[[], None] | None = None
) -> tuple[list[tuple[str, np.ndarray, Any]], dict[str, Any]]:
    """Note a run's MOVEs, where a grasp's descent begins and what a STOP left.

    Returns (moves, seen). moves gets (move id, setpoints, grasp region) of
    every MOVE the agent sends. seen gets "descent", the place in moves of
    the descent's MOVE, and at the second task's question "kept", the
    approach column APAI kept, "checker", the workstation checker's (halt
    reason, e-stop latched, halts so far), and "sent", the count of MOVEs
    and GRIPPERs sent. at_first runs at the first task's question.
    """
    trace = scene.trace
    stop_retrace = importlib.import_module("apai.skills.stop_retrace")
    moves: list[tuple[str, np.ndarray, Any]] = []
    seen: dict[str, Any] = {}

    def on_attach(made: AgentLink) -> None:
        send_move = made.send_move

        def noted(move_id: str, start: Any, tol: Any, q: Any, *rest: Any) -> None:
            moves.append((move_id, np.asarray(q, dtype=float), rest[0]))
            send_move(move_id, start, tol, q, *rest)

        made.send_move = noted

    def on_ask() -> None:
        runtime, harness = trace.built["runtime"], trace.built["harness"]
        if len(scene.asked) == 1:
            # As in the rig's arm profile: the approach may enter the column
            # above its target, which is what a grasp keeps.
            runtime.cfg._data["grasp"]["approach_target_exemption"] = True
            allow = harness.allow_grasp_descent

            def descent_opened(*args: Any, **kwargs: Any) -> Any:
                seen["descent"] = len(moves)  # the place of the next MOVE
                return allow(*args, **kwargs)

            harness.allow_grasp_descent = descent_opened
            if at_first is not None:
                at_first()
        elif len(scene.asked) == 2:
            seen["kept"] = stop_retrace.live(runtime)
            seen["checker"] = (
                harness.halted,
                harness.estopped,
                harness._halt_generation,
            )
            seen["sent"] = len(scene.motion)

    scene.on_attach, scene.on_ask = on_attach, on_ask
    return moves, seen


def check_the_kept_column_and_its_retrace(
    scene: Scene, seen: dict[str, Any], approach: tuple, retrace: tuple, home: tuple
) -> None:
    """After a STOP below a grasp's hover and the task "go to the initial pose".

    approach, retrace and home are rows of watch_grasp()'s moves: the grasp's
    approach, and the two MOVEs of the second task.
    """
    trace = scene.trace
    stop_retrace = importlib.import_module("apai.skills.stop_retrace")
    # At the next task's question the workstation checker was halted and not
    # latched, and the grasp had kept its approach column and its hover.
    reason, latched, halts = seen["checker"]
    assert reason is not None and latched is False and halts == 1
    assert task_gate.SKILL_STOP_NOTE not in scene.asked[1]
    assert not scene.said(task_gate.RESTART)
    kept, column = seen["kept"], approach[2]
    assert kept is not None and column is not None
    assert (kept.column[1], kept.column[2]) == (column.radius_m, column.z_min)
    assert kept.q_pre == pytest.approx(approach[1][-1], abs=1e-6)

    # The next task: move_home retraced up the kept column to the hover, in
    # one MOVE whose every setpoint raised the gripper tip, and then went home.
    (result,) = scene.results("move_home")
    assert result["ok"] is True and result["at"] == "home"
    assert result["stop_retrace"].startswith("completed: the TCP rose")
    region = retrace[2]
    assert (region.radius_m, region.z_min) == (column.radius_m, column.z_min)
    assert region.z_max == pytest.approx(kept.z_max)
    assert retrace[1][-1] == pytest.approx(kept.q_pre, abs=1e-6)
    harness = trace.built["harness"]
    heights = [float(harness.kin.fk(q)[2, 3]) for q in retrace[1]]
    assert all(above > below for below, above in zip(heights, heights[1:]))
    assert heights[0] < kept.hover_z - 0.01
    assert heights[-1] == pytest.approx(kept.hover_z, abs=1e-6)
    assert home[2] is None
    for move_id, _, _ in (retrace, home):
        assert trace.results[move_id].status == "completed"
    trace.check_every_target_was_permitted()
    # The halt was cleared where APAI clears its own, at the retrace's first
    # motion; the column is used up and nothing is latched.
    assert harness.halted is None and harness.estopped is False
    assert harness._halt_generation == 1
    assert stop_retrace.kept(trace.built["runtime"]) is None
    assert scene.order == [
        "agent:completed",
        "nobody:refused",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
    ]
    stopped, back = scene.reports()
    assert stopped["success"] is False and back["success"] is True
    assert scene.h.app.snapshot().stop_number == 1
    scene.holds()


@apai
@link
def test_after_a_stop_in_a_grasp_descent_the_next_task_retraces_up_the_kept_column(
    service_scene: Scene,
) -> None:
    """A console STOP below the pregrasp hover, then "go to the initial pose".

    The hover is where a grasp's approach ends, above its target. APAI's
    grasp keeps the column it approached through when a stop of its checker
    ended it below the hover, and the next move_home first retraces up that
    column to the hover (apai.skills.stop_retrace). Without the retrace a
    gripper tip beside the object cannot leave it: no step gains clearance.
    """
    scene, trace = service_scene, service_scene.trace
    console = scene.stack.stations[0]
    moves, seen = watch_grasp(scene)

    def press(number: int) -> None:
        at = seen.get("descent")
        if "pressed" in seen or at is None or at >= len(moves):
            return
        move_id, index, _ = scene.h.app._executor.progress()
        if move_id == moves[at][0] and index >= 0.6 * len(moves[at][1]):
            seen["pressed"] = index
            console.ws.stop("typed stop", "console")

    scene.h.spy.after_target.append(press)
    scene.typed = [PICK, "go to the initial pose"]  # the second: move_home
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive")

    # The STOP ended the descent below the hover. The grasp ended on it as on
    # a stop of APAI's own: no retreat, no retry, nothing more was sent.
    assert "pressed" in seen
    approach, descent, retrace, home = moves[seen["descent"] - 1 :]
    ended = trace.results[descent[0]]
    assert (ended.status, ended.reason) == (
        "cancelled",
        "STOP from console: typed stop",
    )
    (halted,) = scene.results("pick_and_place")
    assert halted["error"] == (
        "MotionHalted: the robot app stopped the move: STOP from console: typed stop"
    )
    check_the_kept_column_and_its_retrace(scene, seen, approach, retrace, home)
    assert code == launcher.EXIT_FAILED  # the stopped task did not succeed


# Where a console STOP lands below a grasp's hover with no command of the
# agent in flight, and what pick_and_place then ends with.
OWNER_GONE = (
    "the agent no longer owns the arm: the control owner is nobody, not the agent"
)
STALE_NUMBER = "stop number 0 does not match the current 1"
STOP_MOMENTS = {
    # The descent's MOVE has ended and the close is not sent yet.
    "after_descent": f"{OWNER_GONE}; no jaw command sent",
    # The close's GRIPPER has ended and the lift is not sent yet.
    "after_close": f"{OWNER_GONE}; no motion sent",
    # The STOP reaches the Thor just before the close's GRIPPER.
    "crossing_close": f"robot app refused the jaw command: {STALE_NUMBER}",
    # The same, with the STOP's effect not yet shown to the robot link
    # service: the service refuses the GRIPPER itself.
    "crossing_close_at_the_service": (
        "robot app refused the jaw command: a STOP was sent after stop number 0; "
        "the GRIPPER quotes stop number 0"
    ),
    # The STOP reaches the Thor just before the lift's MOVE.
    "crossing_lift": f"robot app refused the move: {STALE_NUMBER}",
    "crossing_lift_at_the_service": (
        "robot app refused the move: a STOP was sent after stop number 0; "
        "the MOVE quotes stop number 0"
    ),
}


@apai
@link
@pytest.mark.parametrize("moment", list(STOP_MOMENTS))
def test_a_stop_that_cancels_no_command_of_a_grasp_ends_it_as_a_halt_all_the_same(
    service_scene: Scene, moment: str
) -> None:
    """A console STOP below the hover between two commands, or just before one.

    No command comes back cancelled. The next one finds the ownership ended
    in ROBOT_STATE, or the robot app or the robot link service refuses it by
    its numbers. Either ends the grasp as a STOP in the descent does: the
    workstation checker is halted once, APAI keeps the approach column and
    retries nothing, and "go to the initial pose" retraces up the column.
    """
    scene, trace = service_scene, service_scene.trace
    console, service = scene.stack.stations[0], scene.stack.service
    # What the adapter refused unsent in the first task, after the STOP.
    unsent: list[str] = []

    def due() -> bool:
        """The descent's MOVE is the last one sent and no STOP was pressed."""
        at = seen.get("descent")
        return "pressed" not in seen and at is not None and len(moves) == at + 1

    def press(shown: Callable[[], bool], what: str) -> None:
        seen["pressed"] = len(scene.motion)
        console.ws.stop("typed stop", "console")
        until(shown, what)

    def hook() -> None:
        adapter = trace.built["adapter"]
        made = adapter._link
        move, jaw, refuse = adapter._move, adapter._jaw, adapter._refuse_unless_task
        gripper, send_move = made.gripper, made.send_move

        def in_state() -> bool:
            return made.state().owner == "nobody"

        def at_the_thor() -> bool:
            # And shown to the service, which then lets the command through.
            return scene.h.app.snapshot().stop_number == 1 and service._gate is None

        def at_the_service() -> bool:
            return service._gate is not None

        def press_before_a_command() -> None:
            if not moment.endswith("at_the_service"):
                press(at_the_thor, "the STOP at the robot app")
                return
            # The service does not see the STOP take effect until the
            # command has come back refused (seen_again).
            seen["lift"], service._lift_gate = service._lift_gate, lambda: None
            press(at_the_service, "the STOP at the service")

        def seen_again() -> None:
            if "lift" in seen:
                service._lift_gate = seen.pop("lift")

        def move_then(*args: Any, **kwargs: Any) -> Any:
            try:
                out = move(*args, **kwargs)
            finally:
                seen_again()
            if moment == "after_descent" and due():
                press(in_state, "the STOP in ROBOT_STATE")
            return out

        def jaw_then(*args: Any, **kwargs: Any) -> Any:
            out = jaw(*args, **kwargs)
            if moment == "after_close" and due():
                seen["close"] = out.status
                press(in_state, "the STOP in ROBOT_STATE")
            return out

        def stop_then_gripper(*args: Any, **kwargs: Any) -> Any:
            if not (moment.startswith("crossing_close") and due()):
                return gripper(*args, **kwargs)
            press_before_a_command()
            try:
                out = gripper(*args, **kwargs)
            finally:
                seen_again()
            seen["crossed"] = (out.status, out.reason)
            return out

        def stop_then_move(*args: Any, **kwargs: Any) -> None:
            if moment.startswith("crossing_lift") and due():
                press_before_a_command()
            send_move(*args, **kwargs)

        def noted_refusal(what: str) -> None:
            try:
                refuse(what)
            except Exception as error:
                if "pressed" in seen and len(scene.asked) == 1:
                    unsent.append(f"{type(error).__name__}: {error}")
                raise

        adapter._move, adapter._jaw = move_then, jaw_then
        adapter._refuse_unless_task = noted_refusal
        made.gripper, made.send_move = stop_then_gripper, stop_then_move

    moves, seen = watch_grasp(scene, hook)
    scene.typed = [PICK, "go to the initial pose"]  # the second: move_home
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive")

    # The grasp ended as a halt at the first command after the STOP: one
    # failed step, no retry against the adapter's latch, no help request.
    # Nothing was sent after the STOP but the command that crossed it.
    assert "pressed" in seen
    crossing = moment.startswith("crossing")
    error = STOP_MOMENTS[moment]
    assert [step["error"] for step in scene.results("pick_and_place")] == [
        f"MotionHalted: {error}"
    ]
    assert unsent == ([] if crossing else [f"MotionHalted: {error}"])
    assert seen["sent"] == seen["pressed"] + (1 if crossing else 0)
    assert scene.h.heard("help_needed") == []
    assert len(scene.h.heard("stop")) == 1
    # The checker was halted once, with the text APAI got.
    assert seen["checker"] == (error, False, 1)
    # The STOP came below the hover, after the descent had ended.
    at = seen["descent"]
    assert trace.results[moves[at][0]].status == "completed"
    if moment.startswith("crossing_lift"):
        approach, _, lift, retrace, home = moves[at - 1 :]
        ended = trace.results[lift[0]]
        seen["crossed"] = (ended.status, ended.reason)
    else:
        approach, _, retrace, home = moves[at - 1 :]
    refused = ("refused", error.partition(": ")[2]) if crossing else None
    assert seen.get("crossed") == refused
    assert seen.get("close") == ("completed" if moment == "after_close" else None)
    check_the_kept_column_and_its_retrace(scene, seen, approach, retrace, home)
    assert code == launcher.EXIT_FAILED  # the stopped task did not succeed


@apai
@link
def test_a_robot_app_restarted_as_real_gets_no_task_of_a_program_on_invented_times(
    service_scene: Scene, make_realtime_apai: Callable[..., Any]
) -> None:
    scene, stack, first = service_scene, service_scene.stack, service_scene.h
    typed = [QUICK, QUICK, QUICK]
    seen: dict[str, Any] = {}

    def on_the_new_session() -> bool:
        session = seen["second"].app.session_id
        try:
            return (
                scene.link.welcome.session_id == session
                and scene.link.state().session_id == session
            )
        except RuntimeError:  # the link is lost while the robot app is away
            return False

    def next_line(prompt: str = "") -> str:
        if not typed:
            raise EOFError
        if len(typed) == 2:  # before the second task: the robot app is restarted
            stack.dialer.close()
            first.app.close()
            second = make_realtime_apai(
                config=scene_robot_config(), connected=False, driver="real"
            )
            raise_arm(second)
            seen["second"], seen["jaws"] = second, len(second.spy.jaw_commands)
            stack.dial(second)
            until(on_the_new_session, "the new session at the agent program")
        return typed.pop(0)

    scene.next_line = next_line
    scene.answers = ["yes", "yes", "yes"]
    code = scene.run("--interactive")
    second = seen["second"]
    # The first task ran on the mock arm; the robot app that declares "real"
    # got no question, no owner request and no command.
    assert scene.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert len(scene.asked) == 1
    assert second.heard("owner") == [] and second.spy.targets == []
    assert len(second.spy.jaw_commands) == seen["jaws"]
    (refused,) = scene.said("motion refused: ")
    assert "camera.allow_invented_times is true, but the robot app declares" in refused
    assert "the driver 'real'" in refused and refused.endswith("; no command sent")
    # The driver was said again for the new session; no third task was read.
    up = scene.said("robot connection up")
    assert [line.split("driver ")[1].split(",")[0] for line in up] == [
        "apai-mock",
        "real",
    ]
    assert f"session {second.app.session_id}" in up[1]
    assert typed == [QUICK] and scene.said(task_gate.NO_NEW_TASK)
    assert code == launcher.EXIT_FAILED
    assert second.spy.released_at == [] and second.arm.torque_on()


def restart_before_the_second_task(
    scene: Scene, make_realtime_apai: Callable[..., Any], typed: list[str], **changed
) -> dict[str, Any]:
    """Type `typed`; before its second task another robot app takes the link.

    The second robot app's file differs from the first one's in `changed`.
    Returns a dict that holds it as "second" once it runs.
    """
    stack, first = scene.stack, scene.h
    seen: dict[str, Any] = {}
    second_at = len(typed) - 1

    def on_the_new_session() -> bool:
        session = seen["second"].app.session_id
        try:
            return (
                scene.link.welcome.session_id == session
                and scene.link.state().session_id == session
            )
        except RuntimeError:  # the link is lost while the robot app is away
            return False

    def next_line(prompt: str = "") -> str:
        if not typed:
            raise EOFError
        if len(typed) == second_at:
            stack.dialer.close()
            first.app.close()
            config = dataclasses.replace(scene_robot_config(), **changed)
            second = make_realtime_apai(config=config, connected=False)
            raise_arm(second)
            seen["second"], seen["jaws"] = second, len(second.spy.jaw_commands)
            stack.dial(second)
            until(on_the_new_session, "the new session at the agent program")
        return typed.pop(0)

    scene.next_line = next_line
    return seen


@apai
@link
@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        (
            {"camera_calibration_id": "another-calibration"},
            "the robot app's camera calibration is 'another-calibration', this "
            "program's 'apai-mock-calibration'",
        ),
        (
            {"profile_id": "another-profile"},
            "the robot app's profile is 'another-profile', expected 'apai-rs-model'",
        ),
    ],
    ids=["calibration", "profile"],
)
def test_a_robot_app_restarted_with_another_welcome_gets_no_task(
    service_scene: Scene,
    make_realtime_apai: Callable[..., Any],
    changed: dict[str, str],
    reason: str,
) -> None:
    scene = service_scene
    typed = [QUICK, QUICK, QUICK]
    seen = restart_before_the_second_task(scene, make_realtime_apai, typed, **changed)
    scene.answers = ["yes", "yes", "yes"]
    code = scene.run("--interactive")
    second = seen["second"]
    # The driver it declares is allowed; what else it declares is not this
    # program's: no question, no owner request, no command.
    assert scene.link.welcome.driver == "apai-mock"
    assert scene.order == ["agent:completed", "nobody:completed", "close:nobody"]
    assert len(scene.asked) == 1
    assert second.heard("owner") == [] and second.spy.targets == []
    assert len(second.spy.jaw_commands) == seen["jaws"]
    assert scene.said("motion refused: ") == [
        f"motion refused: {reason}; no command sent"
    ]
    # No third task was read, and the program ended with exit code 1.
    assert typed == [QUICK]
    assert scene.said(f"{reason}: {task_gate.NO_NEW_TASK}")
    assert code == launcher.EXIT_FAILED
    assert second.spy.released_at == [] and second.arm.torque_on()


@apai
@link
def test_a_robot_app_restarted_on_the_same_file_gets_the_next_task(
    service_scene: Scene, make_realtime_apai: Callable[..., Any]
) -> None:
    scene = service_scene
    typed = [QUICK, QUICK]
    seen = restart_before_the_second_task(scene, make_realtime_apai, typed)
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive")
    second = seen["second"]
    assert scene.order == ["agent:completed", "nobody:completed"] * 2 + ["close:nobody"]
    # A question per task, and the one at the session's end.
    assert len(scene.asked) == 3 and scene.asked[2] == task_gate.END_QUESTION
    assert not scene.said("motion refused")
    assert second.heard("owner")[0][2].get("owner") == "agent"
    assert len(second.spy.jaw_commands) > seen["jaws"]
    assert code == launcher.EXIT_OK


@apai
def test_ownership_whose_result_never_came_is_given_back(scene: Scene) -> None:
    """The answer to OWNER_REQUEST "agent" times out, the ownership is there."""

    def on_attach(made: AgentLink) -> None:
        take = made._take_result
        dropped: list[Any] = []

        def take_result(result: Any) -> None:
            if not dropped:  # the RESULT of the first command: the owner request
                dropped.append(result)
                return
            take(result)

        made._take_result = take_result

    scene.on_attach = on_attach
    code = scene.run("--task", QUICK, task={"end_wait_s": 0.3})
    (result,) = scene.results("open_gripper")
    # Not admitted, as a tool of the experiment's launcher: the task goes on
    # and its end is the model's, which the mock model calls a success.
    assert result["refused"] == "ownership" and "motion_blocked" not in result
    assert result["error"].startswith(
        "open_gripper not admitted in this session; no command sent: "
        "the robot did not answer: no final RESULT"
    )
    assert scene.reports()[0]["success"] is True and code == launcher.EXIT_OK
    assert len(scene.asked) == 1  # the arm was not used: no question at the end
    # The gate found the agent as owner and gave the arm back before the end.
    assert scene.order == ["nobody:completed", "close:nobody"]
    assert scene.trace.owners == ["nobody", "agent", "nobody"]
    granted, returned = scene.h.heard("owner")
    assert granted[2]["owner"] == "agent"
    assert "the ownership arrived late" in returned[1]
    # Nothing was commanded and nobody was called.
    assert scene.motion == [] and scene.h.spy.targets == []
    assert scene.h.heard("help_needed") == []
    snapshot = scene.h.app.snapshot()
    assert snapshot.owner.value == "nobody" and snapshot.stop_number == 0
    scene.holds()


@apai
def test_ownership_granted_after_the_gate_gave_up_is_given_back_before_the_end(
    any_scene: Scene,
) -> None:
    """The Thor answers OWNER_REQUEST "agent" later than the gate waits.

    The gate keeps watching: the program does not end before the late grant
    is there and the arm is given back.
    """
    scene = any_scene
    app = scene.h.app
    new_task = app._new_task

    def slow(message: Any, reply: Any) -> Any:
        app._new_task = new_task
        time.sleep(0.6)  # on the dispatcher thread
        return new_task(message, reply)

    app._new_task = slow
    code = scene.run("--task", QUICK, task={"end_wait_s": 0.3})
    (result,) = scene.results("open_gripper")
    assert result["refused"] == "ownership" and "motion_blocked" not in result
    # The tool was not admitted; the task's end is the mock model's report.
    assert scene.reports()[0]["success"] is True and code == launcher.EXIT_OK
    # The arm was given back by the program itself, and only then the link closed.
    assert scene.order == ["nobody:completed", "close:nobody"]
    assert scene.trace.owners == ["nobody", "agent", "nobody"]
    granted, returned = scene.h.heard("owner")
    assert granted[2]["owner"] == "agent"
    assert "the ownership arrived late" in returned[1]
    # Nobody had to end the ownership for the program: no lost link's
    # ownership, no help request of the service.
    time.sleep(0.3)
    assert scene.h.heard("help_needed") == []
    assert not [e for e in scene.h.heard("connection") if "ownership ended" in e[1]]
    snapshot = app.snapshot()
    assert snapshot.owner.value == "nobody" and snapshot.stop_number == 0
    assert scene.motion == [] and scene.h.spy.targets == []
    scene.holds()


@apai
def test_a_late_ownership_is_given_back_and_the_next_task_runs(scene: Scene) -> None:
    """--interactive: the grant of the first task comes after the gate gave up."""
    app = scene.h.app
    new_task = app._new_task

    def slow(message: Any, reply: Any) -> Any:
        app._new_task = new_task
        time.sleep(0.6)  # on the dispatcher thread
        return new_task(message, reply)

    app._new_task = slow
    scene.typed = [QUICK, QUICK]
    scene.answers = ["yes", "yes"]
    code = scene.run("--interactive", task={"end_wait_s": 0.3})
    first, second = scene.reports()
    refused, opened = scene.results("open_gripper")
    assert refused["refused"] == "ownership" and opened["ok"] is True
    # The first task's tool was not admitted and nothing moved; its end is the
    # mock model's report, which calls it a success.
    assert "motion_blocked" not in refused
    assert first["success"] is True and second["success"] is True
    # The late ownership was given back, then the second task took the arm.
    assert scene.order == [
        "nobody:completed",
        "agent:completed",
        "nobody:completed",
        "close:nobody",
    ]
    assert scene.trace.owners == ["nobody", "agent", "nobody", "agent", "nobody"]
    assert "the ownership arrived late" in scene.h.heard("owner")[1][1]
    assert scene.h.heard("help_needed") == []
    assert code == launcher.EXIT_OK
    scene.holds()


@apai
@experiment
def test_recorded_stereo_pairs_run_through_the_launcher(
    scene: Scene, experiment_dir: Path
) -> None:
    """--camera recorded:<folder>: the link's stereo camera, depth on the CPU."""
    import test_stereo_camera as stereo

    camera_profile = agent_config._camera_profile

    def with_mock_detector(*args: Any) -> dict[str, Any]:
        # APAI's default detector asks a model server; the mock one asks nothing.
        return {**camera_profile(*args), "detector": {"type": "mock"}}

    with mock.patch.object(agent_config, "_camera_profile", with_mock_detector):
        code = scene.run(
            "--task",
            LOOK,
            "--camera",
            f"recorded:{experiment_dir / stereo.RAW_PAIR}",
            camera={
                "calibration_dir": str(experiment_dir / stereo.STEREO_CALIBRATION),
                "hand_eye_json": str(experiment_dir / stereo.HAND_EYE),
                "depth_backend": "opencv_cpu",
            },
        )
    assert code == launcher.EXIT_OK
    # The grid went live on a camera frame of the recorded pair. That needs the
    # self mask's pose at the capture time, from the ROBOT_STATE samples.
    assert scene.said("obstacle grid: live")
    grid = scene.trace.built["grid"]
    assert grid.refreshes >= 1 and grid.last_error is None
    profile = (scene.run_dir / "camera_profile.yaml").read_text()
    assert "type: link_stereo" in profile and "recorded_dir:" in profile
    used = json.loads((scene.run_dir / "experiment-files.json").read_text())
    assert {"thor_camera.py", "thor_self_mask.py"} <= set(used)
    # Looking takes no arm.
    assert scene.reports()[0]["success"] is True
    assert scene.order == ["close:nobody"] and scene.asked == []
    assert scene.h.heard("owner") == [] and scene.h.spy.targets == []


# The fakes: a robot side that plays its moves, a scripted check, a pair source


class PlayedThor(ScriptedThor):
    """ScriptedThor of test_arm_adapter.py that plays its moves as the robot app does.

    From a move's first PERMIT on, one setpoint goes "out" every period while
    the newest PERMIT reaches it and has not expired; the joint readings
    follow. The move ends completed after its last setpoint and refused where
    its permit ends. Nothing here checks a setpoint.
    """

    def __init__(self, q: tuple[float, ...] = (0.0,) * 6) -> None:
        # (move id, setpoint index, through_index of the permit over it).
        self.played: list[tuple[str, int, int]] = []
        # (status, reason, setpoints sent) of each move's end.
        self.results: list[tuple[str, str, int]] = []
        self._due = 0.0
        self._publish_due = 0.0
        super().__init__()
        self.on_cancel = None  # a cancel ends the move where it stands
        with self.lock:
            self.q = [float(value) for value in q]

    def _run(self) -> None:
        while not self._off.wait(0.002):
            while (item := self.thor.carrier.receive(0)) is not None:
                self._take(decode(*item))
            self._play()
            now = time.monotonic()
            if self.publishing and now >= self._publish_due:
                self._publish_due = now + 0.01
                self._publish()

    def _play(self) -> None:
        with self.lock:
            move, stage, index = self._move, self._stage, self._index
        now = time.monotonic()
        if move is None or stage != MOVE_SENDING or now < self._due:
            return
        permit = next(p for p in reversed(self.permits) if p.move_id == move.command_id)
        total = len(move.setpoints)
        if index >= total:
            self._end("completed", "", "reached", index, total)
        elif permit.through_index < index:
            reason = f"permit reaches setpoint {permit.through_index}, not {index}"
            self._end("refused", reason, "refused", index, total)
        elif now >= permit.expires_mono_s:
            self._end("refused", "permit expired", "refused", index, total)
        else:
            with self.lock:
                self.q = [float(value) for value in move.setpoints[index]]
                self._index = index + 1
            self.played.append((move.command_id, index, permit.through_index))
            self._due = now + PERIOD_S

    def _end(self, status: str, reason: str, outcome: str, sent: int, total: int):
        detail = {"outcome": outcome, "sent": sent, "total": total}
        self.results.append((status, reason, sent))
        self.end_move(status, reason, detail, list(self.q))


class FakeLink:
    """An AgentLink on a loopback carrier with a PlayedThor as its robot side."""

    PIN = ScriptedRig.PIN  # the numbers the robot side shows

    def __init__(self, q: tuple[float, ...] = (0.0,) * 6) -> None:
        self.thor = PlayedThor(q)
        self.link = AgentLink(self.thor.agent_end)

    def adapter(self, check: Any, **settings: Any) -> RobotLinkArm:
        """An arm adapter on this link with the task open; for tests marked apai."""
        chosen = {
            "expected_profile_id": "sim-6dof",
            "allow_unnamed_calibration": True,
            **settings,
        }
        joint_feed = chosen.pop("joint_feed", None)
        made = RobotLinkArm(
            PROFILE,
            link=self.link,
            settings=ArmSettings(**chosen),
            joint_feed=joint_feed,
        )
        made.connect()
        made.bind_checker(check)
        made.open_task(self.PIN)
        return made

    def at_setpoint(self, index: int, act: Callable[[], None]) -> threading.Thread:
        """Call act() on another thread once this many setpoints were played."""

        def later() -> None:
            until(lambda: len(self.thor.played) >= index, f"{index} setpoints played")
            act()

        thread = threading.Thread(target=later, daemon=True)
        thread.start()
        return thread

    def cut(self) -> None:
        """Cut the link at once, as a lost connection does."""
        self.thor.thor.carrier.break_link()

    def close(self) -> None:
        self.link.close()
        self.thor.close()


class _FakeMoveCheck:
    def __init__(self, check: FakeCheckAhead, approve: Any, start: Any, q: Any) -> None:
        self._check, self._approve = check, approve
        self._q = [np.asarray(start, dtype=float), *(np.asarray(row) for row in q)]
        self.count = len(q)
        self.approved_through = -1
        self.refusal: BaseException | None = None

    def step(self, sent: int, reach: int) -> Grant | None:
        check = self._check
        check.passes.append((sent, reach))
        if self.refusal is not None:
            return None
        last = min(self.count - 1, sent + reach - 1)
        for index in range(self.approved_through + 1, last + 1):
            if index == check.refuse_at:
                self.refusal = checkahead.CheckRefused(
                    index, "scripted refusal", self.count
                )
                return None
            self._approve(self._q[index], self._q[index + 1], check.period_s)
            self.approved_through = index
        if self.approved_through < 0:
            return None
        return Grant(self.approved_through, check.evidence())


class FakeCheckAhead:
    """In place of CheckAhead: clears every setpoint but the one a test names.

    refuse_at is the setpoint index it refuses, with CheckRefused; whole is the
    refusal of the whole-move check. Each grant quotes evidence(), a time
    shortly before the link's estimate of the Thor's wall clock.
    """

    def __init__(
        self, link: AgentLink, *, refuse_at: int | None = None, whole: str = ""
    ) -> None:
        self._link = link
        self.refuse_at, self.whole = refuse_at, whole
        self.period_s = PERIOD_S
        # What the arm adapter reads of the workstation checker.
        self.harness = types.SimpleNamespace(
            estopped=False,
            halted=None,
            _grasp_exempt=None,
            violations=[],
            clear_grasp_exemption=lambda: None,
            clear_contact_zones=lambda reason="cleared": 0,
        )
        self.passes: list[tuple[int, int]] = []  # (sent, reach) of every pass
        self._evidence = -math.inf

    def evidence(self) -> float:
        self._evidence = max(self._evidence, self._link.thor_now()[1] - 0.05)
        return self._evidence

    def wait_fresh_grid(self, max_age_s: float, timeout_s: float) -> None:
        return None

    def vet_whole(self, start_q: Any, setpoints: Any) -> Evidence:
        if self.whole:
            raise checkahead.CheckRefused(0, self.whole, len(setpoints))
        return Evidence(wall_s=self.evidence(), grid_stamp=(1, 0))

    def begin(self, approve: Any, start_q: Any, setpoints: Any) -> _FakeMoveCheck:
        return _FakeMoveCheck(self, approve, start_q, setpoints)


class FakePairSource:
    """A PairSource that makes a stereo pair every period, with times of the link.

    Each pair was "captured" exposure_age_s before the link's estimate of the
    Thor's wall clock and arrives now. While .paused it gives none. .handed
    are the pairs given out.
    """

    def __init__(
        self,
        link: AgentLink,
        *,
        period_s: float = 0.08,
        exposure_age_s: float = 0.05,
        size_wh: tuple[int, int] = (64, 48),
    ) -> None:
        self._link, self._period_s, self._age_s = link, period_s, exposure_age_s
        width, height = size_wh
        self._eye = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
        self._eye[:height] = 90
        self.paused = False
        self.opened = self.closed = 0
        self.handed: list[StereoPair] = []
        self.last_none = ""
        self._due = 0.0
        self._last_ns = 0

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def next_pair(self, timeout_s: float) -> StereoPair | None:
        wait = self._due - time.monotonic()
        if self.paused or wait > timeout_s:
            self.last_none = "paused" if self.paused else "not due yet"
            time.sleep(max(0.0, min(timeout_s, 0.02)))
            return None
        if wait > 0:
            time.sleep(wait)
        now = time.monotonic()
        self._due = now + self._period_s
        capture_ns = round((self._link.thor_now()[1] - self._age_s) * 1e9)
        self._last_ns = capture_ns = max(capture_ns, self._last_ns + 1)
        pair = StereoPair(
            session_id=1,
            seq=len(self.handed) + 1,
            left=self._eye,
            right=self._eye,
            sof_wall_ns=(capture_ns + 1_000_000, capture_ns),
            device_index=(5, 7),
            rotation=0,
            receiver_rx_mono_s=now,
            check={"source": "test"},
        )
        self.handed.append(pair)
        return pair


@pytest.fixture
def fake() -> Iterator[FakeLink]:
    made = FakeLink()
    made.link.wait_ready(5.0)
    yield made
    made.close()


def a_move(count: int = 100, joint0: float = 0.3) -> list[list[float]]:
    """Setpoints from the zero pose along the first joint, evenly spaced."""
    return [[joint0 * (i + 1) / count, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(count)]


def test_the_fake_robot_side_plays_what_a_permit_reaches_and_no_more(
    fake: FakeLink,
) -> None:
    made, thor = fake.link, fake.thor
    made.pin(fake.PIN)
    wall = made.thor_now()[1]
    made.send_move("m1", [0.0] * 6, 0.004, a_move(40), None, wall, 2.0)
    assert made.wait_first("m1", 5.0).status == "accepted"
    until(lambda: made.state().move_stage == "armed", "the move armed")
    time.sleep(0.1)
    assert thor.played == []  # nothing goes out without a PERMIT
    made.send_permit("m1", 9, wall, 1.0)
    result = made.wait_final("m1", 5.0)
    assert result.status == "refused"
    assert result.reason == "permit reaches setpoint 9, not 10"
    assert result.detail["sent"] == 10 and result.end_q[0] == pytest.approx(0.075)
    assert [index for _, index, _ in thor.played] == list(range(10))
    # A whole move, permitted to its end; then one whose permit runs out.
    made.send_move("m2", result.end_q, 0.004, a_move(20), None, wall, 2.0)
    made.send_permit("m2", 19, wall, 1.0)
    assert made.wait_final("m2", 5.0).status == "completed"
    made.send_move("m3", [0.3, 0, 0, 0, 0, 0], 0.004, a_move(40), None, wall, 2.0)
    made.send_permit("m3", 39, wall, 0.2)
    late = made.wait_final("m3", 5.0)
    assert (late.status, late.reason) == ("refused", "permit expired")
    assert 5 <= late.detail["sent"] <= 12  # 0.2 s of setpoints


def test_the_fake_pair_source_gives_newer_pairs_with_the_links_times(
    fake: FakeLink,
) -> None:
    source = FakePairSource(fake.link, period_s=0.02)
    source.open()
    pairs = [source.next_pair(1.0) for _ in range(3)]
    assert [pair.seq for pair in pairs] == [1, 2, 3]
    captures = [min(pair.sof_wall_ns) for pair in pairs]
    assert captures == sorted(set(captures))
    newest = pairs[-1]
    assert newest.left.shape == (72, 64) and newest.device_index == (5, 7)
    age = fake.link.thor_now()[1] - min(newest.sof_wall_ns) / 1e9
    assert 0.04 < age < 0.2 and newest.receiver_rx_mono_s <= time.monotonic()
    assert source.next_pair(0.0) is None and source.last_none == "not due yet"
    source.paused = True
    assert source.next_pair(0.05) is None and source.last_none == "paused"
    assert len(source.handed) == 3


# The arm adapter on the fakes


def moved(
    fake: FakeLink, adapter: RobotLinkArm, joint0: float, duration_s: float
) -> list[int]:
    """adapter.stream_to along the first joint; the callback's calls, numbered."""
    cleared: list[int] = []
    goal = np.array(fake.thor.q, dtype=float)
    goal[0] += joint0
    adapter.stream_to(
        goal, duration_s, approve=lambda prev, nxt, dt: cleared.append(len(cleared))
    )
    return cleared


@apai
def test_a_scripted_refusal_ends_the_permits_before_the_refused_setpoint(
    fake: FakeLink,
) -> None:
    check = FakeCheckAhead(fake.link, refuse_at=40)
    adapter = fake.adapter(check)
    with pytest.raises(checkahead.CheckRefused, match="setpoint 40 of 100") as raised:
        moved(fake, adapter, 0.3, 2.0)
    assert raised.value.index == 40
    permits = [permit.through_index for permit in fake.thor.permits]
    assert permits[-1] == -1 and permits.count(-1) == 1 and max(permits) == 39
    ((status, reason, sent),) = fake.thor.results
    assert status == "refused" and reason == f"permit reaches setpoint -1, not {sent}"
    assert 15 <= sent <= 40  # the first PERMIT with no setpoint left to clear
    assert all(through >= index for _, index, through in fake.thor.played)
    assert fake.thor.stops == [] and adapter._latched is False
    # The whole-move check refuses before anything is sent.
    check.whole, check.refuse_at = "obstacle at the goal", None
    with pytest.raises(checkahead.CheckRefused, match="obstacle at the goal"):
        moved(fake, adapter, 0.1, 1.0)
    assert len(fake.thor.moves) == 1
    # Without a refusal the next move runs to its end.
    check.whole = ""
    cleared = moved(fake, adapter, 0.1, 1.0)
    assert cleared == list(range(50)) and fake.thor.results[-1][0] == "completed"


@apai
def test_a_link_lost_mid_move_ends_the_permits_and_the_arm_runs_out_its_last(
    fake: FakeLink,
) -> None:
    adapter = fake.adapter(FakeCheckAhead(fake.link))
    thread = fake.at_setpoint(20, fake.cut)
    with pytest.raises(RobotLinkError, match="robot link service is gone") as raised:
        moved(fake, adapter, 0.3, 2.0)
    thread.join(5.0)
    apai_types = importlib.import_module("apai.types")
    assert not isinstance(raised.value, apai_types.SafetyViolation)
    assert adapter._latched is True  # nothing more is sent in this task
    # No cancel could leave. The robot side goes as far as its last PERMIT
    # reached and holds there: at most the reach beyond the cut.
    until(lambda: fake.thor.results, "the move's end on the robot side")
    ((status, reason, sent),) = fake.thor.results
    last = fake.thor.permits[-1].through_index
    assert last >= 20 and status == "refused" and sent == last + 1 < 100
    assert reason == f"permit reaches setpoint {last}, not {sent}"
    assert sent <= 20 + 25 + 5  # the cut, the reach, the setpoints of one pass
    with pytest.raises(apai_types.SafetyViolation, match="arm stop latched"):
        moved(fake, adapter, 0.05, 0.5)
    assert len(fake.thor.moves) == 1


@apai
def test_a_stale_robot_state_mid_move_cancels_and_latches_the_adapter(
    fake: FakeLink,
) -> None:
    adapter = fake.adapter(FakeCheckAhead(fake.link))

    def quiet() -> None:
        fake.thor.publishing = False

    thread = fake.at_setpoint(20, quiet)
    began = time.monotonic()
    with pytest.raises(RobotLinkError, match=r"ROBOT_STATE is 0\.5") as raised:
        moved(fake, adapter, 0.3, 2.0)
    thread.join(5.0)
    assert "the permits were stopped" in str(raised.value)
    assert time.monotonic() - began < 2.0  # before the move would have ended
    assert adapter._latched is True
    # No PERMIT reached further once the states stopped; the robot side ended
    # the move within the last one.
    throughs = [permit.through_index for permit in fake.thor.permits]
    assert max(throughs) <= 20 + 25 + 5
    until(lambda: fake.thor.results, "the move's end on the robot side")
    ((status, _, sent),) = fake.thor.results
    assert status == "refused" and sent <= max(throughs) + 1 < 100
    assert fake.thor.stops == []  # a cancel by permit or by its end, never a STOP


@apai
def test_a_latched_checker_refuses_the_jaws_as_it_refuses_a_move(
    fake: FakeLink,
) -> None:
    check = FakeCheckAhead(fake.link)
    adapter = fake.adapter(check)
    apai_types = importlib.import_module("apai.types")
    # The checker alone is latched: no stop of the adapter, the task is open.
    check.harness.estopped = True
    assert adapter._latched is False and fake.link.lost_reason() is None
    for command in (
        adapter.close_gripper_two_stage,  # APAI calls it past its SafeArm
        lambda: adapter.set_gripper(PROFILE["gripper"]["open_pos"]),
    ):
        with pytest.raises(
            apai_types.SafetyViolation, match="e-stop latched; no jaw command sent"
        ):
            command()
    with pytest.raises(apai_types.SafetyViolation, match="e-stop latched"):
        moved(fake, adapter, 0.05, 0.5)
    time.sleep(0.1)
    assert fake.thor.grippers == [] and fake.thor.moves == []
    # Without the latch the same close leaves.
    check.harness.estopped = False
    adapter.close_gripper_two_stage()
    until(lambda: fake.thor.grippers, "the GRIPPER on the robot side")


class PoseHistory:
    """In place of the experiment's JointHistory: every joint's samples, kept."""

    def __init__(self, n_joints: int) -> None:
        self._samples: list[list[tuple[float, float]]] = [[] for _ in range(n_joints)]

    def add(self, joints: list[int], values: list[float], start: float, end: float):
        for joint, value in zip(joints, values):
            self._samples[joint].append((end, value))

    def invalidate(self) -> None:
        for samples in self._samples:
            samples.clear()

    def latest(self) -> float:
        return min((s[-1][0] if s else -math.inf) for s in self._samples)

    def state_at(self, stamp: float) -> Any:
        from apai.perception.robot_mask import BodyMaskUnavailable
        from apai.types import RobotState

        q = []
        for samples in self._samples:
            before = [value for end, value in samples if end <= stamp]
            if not before or samples[-1][0] <= stamp:
                raise BodyMaskUnavailable("no joint sample on both sides of the time")
            q.append(before[-1])
        return RobotState(q=np.array(q), gripper_pos=0.0, gripper_valid=False, t=stamp)


class CameraRig:
    """The adapter under APAI's checker, on a grid fed by the link's stereo camera.

    FakePairSource -> LinkStereoCamera -> APAI's OccupancyMap (required, with
    the adapter's body_mask_state() as its self mask) -> CheckAhead -> PERMITs
    to a PlayedThor. A thread stands in for APAI's perception loop.
    """

    def __init__(self, folder: Path, grid_max_age_s: float = 3.0) -> None:
        from apai.config import Cfg
        from apai.perception.occupancy import OccupancyMap
        from apai.safety.harness import SafetyHarness, SafetyLimits
        from isaaccapture_examples.agentic_remote.agent import self_mask
        from isaaccapture_examples.agentic_remote.agent.camera import stereo_camera
        from isaaccapture_examples.agentic_remote.robot_app import apai_backend
        from test_stereo_camera import FakeDepth, calibration_files

        profile = apai_backend.load_apai_profile("mock")
        self.fake = FakeLink(HOME)
        made = self.fake.link
        made.wait_ready(5.0)
        self.states: list[tuple[float, list[float]]] = []  # (the Thor's time, pose)
        made.add_state_listener(
            lambda s: self.states.append((s.clock_mono_s, [j.value for j in s.joints]))
        )
        self.feed = self_mask.JointFeed(6, history=PoseHistory(6))
        made.add_state_listener(self.feed.on_state)
        self.times = FrameTimes()
        self.source = FakePairSource(made)
        directory, hand_eye = calibration_files(folder)
        self.camera = stereo_camera.LinkStereoCamera(
            Cfg(
                {
                    "type": stereo_camera.CAMERA_TYPE,
                    "name": "head",
                    "calibration_dir": str(directory),
                    "extrinsics": {"hand_eye_json": str(hand_eye)},
                    "depth_backend": "stand_in",
                    "frame_timeout_s": 0.2,
                    "max_frame_age_s": 3,
                }
            ),
            link=made,
            frame_times=self.times,
            source=self.source,
            depth=FakeDepth,
        )
        self.grid = OccupancyMap(
            client=FakeGridClient(),
            region_min=np.full(3, -2.0),
            region_max=np.full(3, 2.0),
            required=True,
            max_age_s=grid_max_age_s,
        )
        self.harness = SafetyHarness(
            SafetyLimits.from_config(profile.apai_cfg.safety),
            kinematics=profile.kinematics,
            occupancy=self.grid,
        )
        self.adapter = self.fake.adapter(
            CheckAhead(self.harness, self.times, period_s=PERIOD_S),
            joint_feed=self.feed,
        )
        # (the camera frame the grid masked, the pose it was masked with).
        self.masked: list[tuple[Any, Any]] = []
        self.unmasked: list[str] = []  # why a camera frame got no pose
        self.grid.add_robot_body_mask(self._mask)
        self.frames: list[Any] = []
        self._off = threading.Event()
        self._thread = threading.Thread(
            target=self._perceive, args=(self_mask,), daemon=True
        )
        # A camera frame captured before the first joint sample gets no pose.
        until(lambda: len(self.states) >= 12, "0.1 s of joint samples")
        self.camera.open()
        self._thread.start()
        until(lambda: self.grid.usable(), "a usable grid")

    def _mask(self, depth: np.ndarray, intrinsics: Any, camera_pose: Any) -> np.ndarray:
        from isaaccapture_examples.agentic_remote.agent import self_mask

        try:
            pose = self.adapter.body_mask_state()
        except Exception as error:
            self.unmasked.append(str(error))
            raise
        self.masked.append((self_mask.FRAME.get(), pose))
        return np.zeros(depth.shape, dtype=bool)

    def _perceive(self, self_mask: Any) -> None:
        with self_mask.frame_context():
            while not self._off.is_set():
                try:
                    frame = self.camera.get_frame()
                except RuntimeError:  # CameraError: no stereo pair in time
                    continue
                self.harness.heartbeat(frame.t)
                self.grid.refresh(frame, np.eye(4))
                self.frames.append(frame)

    def move(self, joint0: float, duration_s: float) -> bool:
        """adapter.stream_to with APAI's checker callback, as SafeArm calls it."""
        goal = np.array(self.fake.thor.q, dtype=float)
        goal[0] += joint0
        self.harness.begin_motion()
        try:
            return self.adapter.stream_to(
                goal, duration_s, approve=self.harness.approve
            )
        finally:
            self.harness.end_motion()

    def close(self) -> None:
        self._off.set()
        self._thread.join(5.0)
        self.camera.close()
        self.fake.close()


@pytest.fixture
def make_camera_rig(tmp_path: Path) -> Iterator[Callable[..., CameraRig]]:
    made: list[CameraRig] = []

    def make(**options: Any) -> CameraRig:
        made.append(CameraRig(tmp_path, **options))
        return made[-1]

    yield make
    for rig in made:
        rig.close()


@apai
def test_each_permit_quotes_a_stereo_pairs_capture_time_and_the_mask_its_pose(
    make_camera_rig: Callable[..., CameraRig],
) -> None:
    rig = make_camera_rig()
    thor = rig.fake.thor
    assert rig.move(0.3, 1.5) is True
    assert [status for status, _, _ in thor.results] == ["completed"]
    assert len(thor.played) == 75 and thor.q[0] == pytest.approx(0.3)

    # Every PERMIT quotes the capture time of a stereo pair the source handed
    # out: the time went pair -> camera frame -> grid -> check ahead -> PERMIT.
    captures = [min(pair.sof_wall_ns) / 1e9 for pair in rig.source.handed]
    quoted = [permit.evidence_wall_s for permit in thor.permits]
    assert len(quoted) > 10 and set(quoted) <= set(captures)
    assert quoted == sorted(quoted) and len(set(quoted)) >= 5  # newer grids came
    # ... of a camera frame at most a few camera periods old when it was sent.
    for permit in thor.permits:
        age = (permit.expires_mono_s - 1.0) - rig.fake.link.wall_to_thor_mono(
            permit.evidence_wall_s
        )
        assert 0.0 < age < 0.6, age
    # The MOVE itself quotes the grid of its whole-move check.
    assert thor.moves[0].evidence_wall_s in captures

    # The self mask of each camera frame used the pose at its capture time,
    # from the ROBOT_STATE samples: the newest pose published up to then.
    frames = len(rig.frames)
    masked = list(rig.masked)
    assert len(masked) >= frames > 10 and rig.unmasked == []
    moving = 0
    for frame, pose in masked:
        stamp = frame.capture["capture_thor_mono_s"]
        assert pose.t == frame.t and pose.gripper_valid is False
        assert pose.mask_alignment["capture_thor_mono_s"] == stamp
        assert pose.mask_alignment["source"].startswith("ROBOT_STATE samples")
        expected = [q for at, q in rig.states if at <= stamp][-1]
        assert list(pose.q) == expected
        moving += 0.0 < pose.q[0] < 0.3
    assert moving >= 5  # camera frames taken while the arm moved


@apai
def test_without_a_new_camera_frame_the_permits_end_and_the_arm_holds(
    make_camera_rig: Callable[..., CameraRig],
) -> None:
    rig = make_camera_rig(grid_max_age_s=0.6)
    thor = rig.fake.thor
    apai_types = importlib.import_module("apai.types")

    def camera_stops() -> None:
        rig.source.paused = True

    thread = rig.fake.at_setpoint(20, camera_stops)
    began = time.monotonic()
    with pytest.raises(apai_types.SafetyViolation) as raised:
        rig.move(0.3, 3.0)
    thread.join(5.0)
    assert time.monotonic() - began < 2.5  # the move would have taken 3 s
    assert "distance grid is stale" in str(raised.value)
    # The last PERMITs all quote the last camera frame; then one cancel.
    last_capture = min(rig.source.handed[-1].sof_wall_ns) / 1e9
    assert thor.permits[-1].through_index == -1
    assert thor.permits[-1].evidence_wall_s == thor.permits[-2].evidence_wall_s
    assert thor.permits[-2].evidence_wall_s <= last_capture
    ((status, reason, sent),) = thor.results
    assert status == "refused" and reason == f"permit reaches setpoint -1, not {sent}"
    assert 20 <= sent < 150 and thor.stops == []
    held = list(thor.q)
    time.sleep(0.2)
    assert thor.q == held and len(thor.played) == sent
    # With camera frames again the next move runs.
    rig.source.paused = False
    until(lambda: not rig.grid.is_stale(), "a fresh grid")
    assert rig.adapter._latched is False and rig.move(0.05, 0.5) is True
