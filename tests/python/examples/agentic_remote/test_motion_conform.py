# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arm motion of the split against the experiment's own code.

The experiment is the folder of the rig's single-machine program, which moves
the real arm with APAI's driver on one machine (AGENTIC_REMOTE_EXPERIMENT_DIR,
with APAI under it). The split puts the agent on the workstation and the
motors on the Thor. Each test here holds one thing the split decides about a
move against what the experiment's code decides: the experiment's routine is
called, or its file is read, for the expected answer. No expected number is
typed here.

A setpoint is one joint target of a move; a permit is the workstation's leave
for the Thor to send setpoints up to an index.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import time
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import arm_adapter
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent.arm_adapter import ArmSettings
from isaaccapture_examples.agentic_remote.agent.setpoints import sample_move
from isaaccapture_examples.agentic_remote.protocol.messages import Move, Result
from isaaccapture_examples.agentic_remote.robot_app import executor as executor_module
from isaaccapture_examples.agentic_remote.robot_app.authority import Authority
from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
from isaaccapture_examples.agentic_remote.robot_app.config import load_config
from isaaccapture_examples.agentic_remote.robot_app.executor import MoveExecutor
from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
    JointReading,
    MoveOutcome,
    Owner,
    ReadFailed,
    Refusal,
)
from isaaccapture_examples.agentic_remote.robot_app.state import JointHistory
from repo_paths import repo_root

import test_arm_adapter
from test_arm_adapter import SHORT, ScriptedRig, ScriptedThor, SimRig

try:
    from isaaccapture_examples.agentic_remote.robot_app import apai_backend
except ModuleNotFoundError as missing:
    # Without an APAI checkout every test here is skipped: all are marked "apai".
    if missing.name != "apai":
        raise
    apai_backend = None

pytestmark = pytest.mark.apai
experiment = pytest.mark.experiment

# The fixtures of test_arm_adapter.py: the adapter on the real robot app with a
# simulated arm, and on a robot side the test scripts.
make_sim = test_arm_adapter.make_sim
sim = test_arm_adapter.sim
make_scripted = test_arm_adapter.make_scripted

CONFIGS = repo_root() / "examples" / "agentic_remote" / "configs"
ROBOT_RIG = CONFIGS / "robot_app_rebot.yaml"
AGENT_RIG = CONFIGS / "agent_rebot.yaml"
RATE_HZ = 50.0
# The rest pose as the rig reads it: the second and third joint lie on their
# stops and read 0.002 rad below them.
AT_REST = (0.0, -0.002, -0.002, 0.0, 0.0, 0.0)


def apai(name: str) -> Any:
    return importlib.import_module(name)


# The experiment's side


@pytest.fixture(scope="module")
def mock_profile() -> Any:
    """APAI's software arm, which has the arm model of the rig's profile."""
    return apai_backend.load_apai_profile("mock")


@pytest.fixture(scope="module")
def rs_profile() -> Any:
    return apai_backend.load_apai_profile("rebot_rs")


@pytest.fixture(scope="module")
def launcher(experiment_dir: Path) -> Iterator[Any]:
    """The experiment's launcher module, thor_runtime.

    The tests' interpreter may lack scipy, which one module the launcher
    imports needs and nothing called here: that module is then a stand-in.
    """
    folder = str(experiment_dir)
    if folder not in sys.path:
        sys.path.append(folder)
    stand_in = None
    try:
        try:
            importlib.import_module("thor_occupancy")
        except ModuleNotFoundError as missing:
            if missing.name != "scipy":
                raise
            stand_in = types.ModuleType("thor_occupancy")
            stand_in.install_grid_sampler = lambda occupancy: None
            sys.modules["thor_occupancy"] = stand_in
        module = importlib.import_module("thor_runtime")
    except ModuleNotFoundError as missing:
        pytest.skip(f"the experiment's launcher cannot be imported here: {missing}")
    try:
        yield module
    finally:
        if stand_in is not None:
            sys.modules.pop("thor_runtime", None)
            if sys.modules.get("thor_occupancy") is stand_in:
                del sys.modules["thor_occupancy"]


@pytest.fixture(scope="module")
def thor_folder(experiment_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The experiment's launcher files as they lie on the Thor, in a test folder.

    The launcher's settings, its speed source and its motion bounds module are
    the experiment's own files. Its arm profile extends a file that exists
    only on the Thor; here it extends APAI's rebot_rs profile, which that
    file extends too.
    """
    root = tmp_path_factory.mktemp("experiment")
    (root / "configs").mkdir()
    for name in ("thor_runtime.yaml", "rebot_workstation.yaml"):
        text = (experiment_dir / "configs" / name).read_text(encoding="utf-8")
        (root / "configs" / name).write_text(text, encoding="utf-8")
    profile = yaml.safe_load(
        (experiment_dir / "configs" / "thor_operation.yaml").read_text("utf-8")
    )
    profile["extends"] = "rebot_rs"
    (root / "configs" / "thor_operation.yaml").write_text(yaml.safe_dump(profile))
    module = f"{apai_backend.BOUNDS_MODULE}.py"
    (root / module).write_text((experiment_dir / module).read_text("utf-8"), "utf-8")
    return root


@pytest.fixture(scope="module")
def thor_profile(thor_folder: Path) -> Any:
    """The rig's arm profile as the robot app loads it on the Thor."""
    return apai_backend.load_apai_profile(str(thor_folder / "configs/thor_operation"))


@pytest.fixture
def agent_rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """The rig's agent file; the folders its paths name are test folders."""
    for name in agent_config.PATH_VARIABLES:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return agent_config.load_settings(AGENT_RIG)


def experiment_home(experiment_dir: Path) -> list[float]:
    """The home pose of the experiment's arm profile."""
    text = (experiment_dir / "configs" / "thor_operation.yaml").read_text("utf-8")
    return [float(value) for value in yaml.safe_load(text)["home_q"]]


def functional_harness(profile: Any, speed: float) -> Any:
    """APAI's checker as the experiment builds it, without an obstacle grid."""
    harness_module = apai("apai.safety.harness")
    limits = dataclasses.replace(
        harness_module.SafetyLimits.from_config(profile.apai_cfg.safety),
        max_joint_vel=speed,
    )
    return harness_module.SafetyHarness(limits, kinematics=profile.kinematics)


def functional_arm(start: Sequence[float], kinematics: Any = None) -> Any:
    """APAI's software arm at a pose: its streaming code is every driver's."""
    cfg = apai("apai.config").Cfg({"home_q": list(start), "n_joints": len(start)})
    arm = apai("apai.control.mock_arm").MockArm(cfg, kinematics)
    arm.connect()
    return arm


def functional_move(
    profile: Any, speed: float, start: Sequence[float], goal: Sequence[float], s: float
) -> bool:
    """Whether APAI's own stream completes this move from this reading."""
    harness = functional_harness(profile, speed)
    arm = functional_arm(start, profile.kinematics)
    harness.heartbeat()
    harness.begin_motion()
    try:
        return arm.stream_to(np.asarray(goal, float), s, approve=harness.approve)
    except apai("apai.types").SafetyViolation:
        return False
    finally:
        harness.end_motion()


# The split's side: the Thor


def rig_file_on(profile: Any) -> Any:
    """The rig's robot file for an arm profile the experiment's launcher does not run.

    Such a profile's speed cap bounds a person too, and the file's
    limits.human.max_joint_speed lies above it: here a person gets that cap.
    The agent's limits, which every move of these tests is checked with, are
    the file's.
    """
    config = load_config(ROBOT_RIG)
    assert profile.launcher_dir is None
    cap = apai_backend.human_speed_cap(profile)
    assert config.limits.human.max_joint_speed > cap
    human = dataclasses.replace(config.limits.human, max_joint_speed=cap)
    limits = dataclasses.replace(config.limits, human=human)
    return apai_backend.config_for_profile(
        dataclasses.replace(config, limits=limits), profile
    )


class Thor:
    """The Thor's executor on APAI's software arm, with the rig's robot file."""

    def __init__(self, profile: Any, start: Sequence[float]) -> None:
        self.clock = FakeClock()
        self.config = rig_file_on(profile)
        self.arm = apai_backend.build_apai_arm(self.config, profile, self.clock)
        self.arm.connect()
        self.arm.backend._q = np.array(start, dtype=float)
        self.checkers = apai_backend.build_apai_checkers(
            self.config, profile, self.clock
        )
        self.authority = Authority()
        self.authority.set_owner(Owner.AGENT, "new task")
        self.history = JointHistory(self.config.n_joints, self.config.state.history_s)
        self.executor = MoveExecutor(
            self.arm,
            self.authority,
            self.history,
            self.config.executor,
            self.clock,
            start_read_tries=self.arm.start_read_tries,
            pause_limit_s=self.arm.move_pause_limit_s,
        )
        self.sent: list[tuple[float, list[float]]] = []  # (clock time, target)
        send = self.arm.send_target

        def spied(q: Sequence[float]) -> None:
            self.sent.append((self.clock.monotonic(), [float(v) for v in q]))
            send(q)

        self.arm.send_target = spied

    def read_off_by(self, noise: Sequence[float]) -> None:
        """The Thor's start reading differs from the pose by this, joint by joint."""
        read, calls = self.arm.read_all, []

        def read_all() -> list[JointReading]:
            readings = read()
            if not calls:
                readings = [
                    dataclasses.replace(r, value=r.value + noise[r.joint])
                    for r in readings
                ]
            calls.append(1)
            return readings

        self.arm.read_all = read_all

    def run(
        self, rows: Sequence[Any], start: Sequence[float], tol: float, gate: Any = None
    ) -> MoveOutcome:
        self.checkers.agent.feed_watchdog()
        return self.executor.run(
            [np.asarray(row, float).tolist() for row in rows],
            list(start),
            tol,
            self.config.executor.period_s,
            Owner.AGENT,
            self.checkers.agent,
            self.authority.snapshot(),
            "move-1",
            gate,
        )


def quoted_tol(speed: float) -> float:
    """The start tolerance the adapter quotes at this agent speed limit."""
    return ArmSettings(agent_max_joint_speed=speed).start_tol(1.0 / RATE_HZ)


def setpoints(start: Sequence[float], goal: Sequence[float], s: float) -> list[Any]:
    """The setpoints the adapter samples for a plain move."""
    min_jerk = apai("apai.control.motion_profile").min_jerk
    return sample_move(start, goal, s, RATE_HZ, min_jerk)


# Row: the first setpoint of a move from rest


@experiment
@pytest.mark.parametrize("noise", [9e-5, -9e-5, 5e-4])
def test_the_first_setpoint_from_rest_is_decided_as_apais_stream_decides_it(
    mock_profile: Any, experiment_dir: Path, noise: float
) -> None:
    """Seen on the rig: 'joint 2 target -0.002 rad outside [0.020, 3.120]'.

    APAI's stream checks the first setpoint from the reading the move was
    sampled from, so a joint below its stop always moves back toward its
    limits. The Thor reads the joints a second time; the two readings of a
    resting joint differ by about 0.00009 rad.
    """
    speed = load_config(ROBOT_RIG).limits.agent.max_joint_speed
    goal = experiment_home(experiment_dir)
    theirs = functional_move(mock_profile, speed, AT_REST, goal, 0.6)
    assert theirs is True  # the experiment's arm leaves rest

    thor = Thor(mock_profile, AT_REST)
    thor.read_off_by([0.0, noise, noise, 0.0, 0.0, 0.0])
    rows = setpoints(AT_REST, goal, 0.6)
    ours = thor.run(rows, AT_REST, quoted_tol(speed))
    assert (ours.status == "completed") is theirs, ours.reason
    assert ours.index == len(rows)


@experiment
def test_from_a_second_reading_the_same_first_setpoint_is_refused(
    mock_profile: Any, experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule the Thor had: its refusal is the one seen on the rig."""
    speed = load_config(ROBOT_RIG).limits.agent.max_joint_speed
    goal = experiment_home(experiment_dir)
    monkeypatch.setattr(executor_module, "_PLANNED_START_TOL_RAD", -1.0)
    thor = Thor(mock_profile, AT_REST)
    thor.read_off_by([0.0, 9e-5, 9e-5, 0.0, 0.0, 0.0])
    ours = thor.run(setpoints(AT_REST, goal, 0.6), AT_REST, quoted_tol(speed))
    assert (ours.status, ours.index) == ("refused", 0)
    assert "joint 2 target -0.002 rad outside [0.020, 3.120]" in ours.reason
    # APAI's check says the same of a step that starts at the second reading.
    second = np.array(AT_REST) + [0.0, 9e-5, 9e-5, 0.0, 0.0, 0.0]
    harness = functional_harness(mock_profile, speed)
    first = setpoints(AT_REST, goal, 0.6)[0]
    assert "joint 2 target" in harness.vet_step(second, first, 1.0 / RATE_HZ)
    assert harness.vet_step(np.array(AT_REST), first, 1.0 / RATE_HZ) is None


# Row: how far the arm may be from the pose a move was sampled from


@experiment
def test_the_quoted_start_tolerance_is_that_of_the_launchers_speed(
    thor_profile: Any, agent_rig: Any
) -> None:
    """Seen on the rig: a start pose refused at a tolerance of 0.001 rad.

    At the speed the agent plans with, the launcher's and at most the Thor's
    limit, the tolerance is half a period's travel, and so small that the
    Thor checks the first setpoint from the move's own start.
    """
    cap = apai_backend.speed_cap(thor_profile)
    speed = min(cap, agent_rig.thor.agent_max_joint_speed)
    tol = quoted_tol(speed)
    assert tol == pytest.approx(0.5 * speed / RATE_HZ)
    assert tol <= executor_module._PLANNED_START_TOL_RAD
    assert agent_config.arm_settings(agent_rig).start_tol(1.0 / RATE_HZ) == tol


@pytest.mark.parametrize(("share", "starts"), [(0.95, True), (1.05, False)])
def test_the_thor_starts_a_move_inside_the_quoted_tolerance_and_none_outside(
    mock_profile: Any, share: float, starts: bool
) -> None:
    speed = load_config(ROBOT_RIG).limits.agent.max_joint_speed
    tol = quoted_tol(speed)
    start = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    goal = [0.1, 0.5, 0.5, 0.0, 0.0, 0.0]
    thor = Thor(mock_profile, start)
    thor.read_off_by([share * tol, 0.0, 0.0, 0.0, 0.0, 0.0])
    ours = thor.run(setpoints(start, goal, 1.0), start, tol)
    assert (ours.status == "completed") is starts, ours.reason
    if not starts:
        assert ours.reason.startswith(arm_adapter.START_POSE)
        assert ours.index == 0 and thor.sent == []


class DriftingThor(ScriptedThor):
    """A robot side that finds the arm away from the first MOVEs' start pose."""

    refusals = 1  # how many MOVEs it refuses for their start pose

    def _take(self, message: Any) -> None:
        if isinstance(message, Move) and len(self.moves) < self.refusals:
            self.moves.append(message)
            with self.lock:
                self.q[4] -= 0.0012  # the joint creeps on, as on the rig
            reason = (
                f"{arm_adapter.START_POSE}joint index 4 is at -0.0578 rad, the "
                "move starts at -0.0566 rad (tolerance 0.004 rad)"
            )
            detail = {"outcome": "refused", "sent": 0, "total": 25}
            self.thor.send(Result(message.command_id, "refused", reason, None, detail))
            return
        super()._take(message)


@pytest.fixture
def drifting(monkeypatch: pytest.MonkeyPatch) -> Iterator[ScriptedRig]:
    monkeypatch.setattr(test_arm_adapter, "ScriptedThor", DriftingThor)
    rig = ScriptedRig()
    rig.thor.at_first_permit = (
        "completed",
        "",
        {"outcome": "completed", "sent": 25, "total": 25},
        None,
    )
    rig.own()
    yield rig
    rig.close()


def test_a_move_whose_start_the_arm_has_left_is_sampled_again_as_apai_does(
    drifting: ScriptedRig,
) -> None:
    """Seen on the rig: 'start pose: joint index 4 is at -0.0578 rad, ...'.

    APAI's prepare_stream takes a new reading when the joints moved during a
    route's check and checks again from it. The split does the same when the
    Thor's reading differs from the workstation's.
    """
    assert drifting.move(SHORT, 0.5) is True
    first, second = drifting.thor.moves
    assert first.start_q[4] == 0.0 and second.start_q[4] == pytest.approx(-0.0012)
    # The second MOVE's setpoints are sampled from its own start pose.
    again = setpoints(second.start_q, SHORT, 0.5)
    assert np.allclose(second.setpoints, again, atol=0.0, rtol=0.0)
    assert drifting.adapter.last_move["start_retries"] == 1


def test_the_tries_and_their_budget_are_apais_own(
    drifting: ScriptedRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm_base = apai("apai.control.arm_base")
    violation = apai("apai.types").SafetyViolation
    drifting.thor.refusals = 10**6
    with pytest.raises(violation, match="start pose: joint index 4"):
        drifting.move(SHORT, 0.5)
    assert len(drifting.thor.moves) == arm_base.PREFLIGHT_MAX_CHECKS
    # Nothing was sent and the checker callback saw no setpoint.
    assert not drifting.of("approve") and not drifting.of("permit")
    assert drifting.checker.approved_commands == 0
    # The experiment's launcher sets the budget in APAI's module (thor_motion).
    sent = len(drifting.thor.moves)
    monkeypatch.setattr(arm_base, "PREFLIGHT_REBIND_BUDGET_S", 0.0)
    with pytest.raises(violation, match="start pose: joint index 4"):
        drifting.move(SHORT, 0.5)
    assert len(drifting.thor.moves) == sent + 1


# Row: how close to its goal a move counts as settled


@pytest.mark.parametrize("away", [0.049, 0.05, 0.051, 0.03])
def test_a_joint_counts_as_settled_as_apais_driver_counts_it(
    rs_profile: Any, away: float
) -> None:
    """The rig's driver settles within its own tolerance, compared with <."""
    driver = apai("apai.control.arm_base").make_arm(rs_profile.arm_cfg)
    goal = np.zeros(6)
    # APAI's wait for the settled pose, on its software arm this far away.
    theirs = functional_arm([away, 0, 0, 0, 0, 0]).wait_settled(
        goal, driver.settle_tol, 0.01
    )
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), rs_profile)
    assert config.executor.settle_timeout_s == driver.settle_timeout_s
    ours_executor = MoveExecutor(
        types.SimpleNamespace(n_joints=6), None, None, config.executor, FakeClock()
    )
    near = [False] * 6
    readings = [JointReading(0, away, 0.0, 0.0)]
    readings += [JointReading(joint, 0.0, 0.0, 0.0) for joint in range(1, 6)]
    ours_executor._note_near(near, readings, goal.tolist())
    assert all(near) is theirs


def test_a_real_arm_is_not_built_with_another_settle_rule(rs_profile: Any) -> None:
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), rs_profile)
    other = dataclasses.replace(config.executor, settle_tol_rad=0.03)
    with pytest.raises(ValueError, match="use config_for_profile"):
        apai_backend.build_apai_arm(
            dataclasses.replace(config, executor=other), rs_profile, FakeClock()
        )


# Row: the joint speed of an agent move


@experiment
def test_the_thor_caps_the_joint_speed_where_the_launcher_does(
    launcher: Any, thor_folder: Path, thor_profile: Any
) -> None:
    """The launcher runs without --speed-profile: its speed profile "reference"."""
    theirs = launcher.load_joint_speed("reference")
    assert apai_backend.launcher_folder(str(thor_folder / "configs/thor_operation"))
    assert apai_backend.speed_cap(thor_profile) == theirs
    # The profile's own number is only the launcher's fallback.
    own = yaml.safe_load((thor_folder / "configs/thor_operation.yaml").read_text())
    assert own["overrides"]["safety"]["max_joint_vel"] != theirs
    # A profile the launcher does not run keeps its own cap.
    assert apai_backend.launcher_folder("rebot_rs") is None
    assert apai_backend.launcher_folder(str(thor_folder / "configs/other")) is None


# The owner's value of 2026-11-16 for the agent's moves and the return to rest.
OWNER_SPEED = 0.2


@experiment
def test_the_rig_files_run_at_the_owners_speed_within_the_launchers(
    launcher: Any, agent_rig: Any
) -> None:
    theirs = launcher.load_joint_speed("reference")
    config = load_config(ROBOT_RIG)
    speeds = {
        "limits.agent.max_joint_speed": config.limits.agent.max_joint_speed,
        "limits.recovery.max_joint_speed": config.limits.recovery.max_joint_speed,
        "rest.recovery_speed": config.rest.recovery_speed,
        # The agent's copy of the Thor's limit: its tasks are planned at the
        # launcher's speed and at most at this one.
        "thor.agent_max_joint_speed": agent_rig.thor.agent_max_joint_speed,
    }
    for name, speed in speeds.items():
        assert speed == OWNER_SPEED <= theirs, name


@experiment
def test_an_agent_move_at_the_rig_files_speed_passes_the_thor_and_no_faster_one(
    launcher: Any, thor_profile: Any
) -> None:
    """APAI stretches a move so its peak is 0.9 of the cap; the Thor takes it."""
    theirs = launcher.load_joint_speed("reference")
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), thor_profile)
    # The agent plans at the launcher's speed and at most at the Thor's limit.
    speed = min(theirs, config.limits.agent.max_joint_speed)
    start = np.array([0.0, 0.6, 0.6, 0.0, 0.0, 0.0])
    goal = start + [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    rows = setpoints(start, goal, stretched(0.5, speed, 0.0))
    peak = max(
        float(np.max(np.abs(b - a))) * RATE_HZ for a, b in zip([start, *rows], rows)
    )
    assert 0.8 * speed < peak <= speed
    checkers = apai_backend.build_apai_checkers(config, thor_profile, FakeClock())
    assert checkers.agent.harness.limits.max_joint_vel == speed
    checkers.agent.check_path(start, [row.tolist() for row in rows], 1.0 / RATE_HZ)
    # The functional checker passes the same steps.
    harness = functional_harness(thor_profile, speed)
    prev = start
    for row in rows:
        assert harness.vet_step(prev, row, 1.0 / RATE_HZ) is None
        prev = row
    # A move planned at a higher speed than the rig file's is refused: the
    # launcher's own speed reaches the arm only as far as the file allows.
    if theirs > speed:
        fast = setpoints(start, goal, stretched(0.5, theirs, 0.0))
        with pytest.raises(Refusal, match=rf"velocity .* exceeds {speed:.2f}"):
            checkers.agent.check_path(
                start, [row.tolist() for row in fast], 1.0 / RATE_HZ
            )


# Row: the experiment's motion bounds on the Thor


@experiment
def test_the_agents_thor_checker_has_the_launchers_motion_bounds(
    thor_folder: Path, thor_profile: Any, experiment_dir: Path
) -> None:
    """The launcher's restrict_motion on its harness, with the profile's bounds."""
    bounds = thor_profile.arm_cfg.get("motion_bounds").as_dict()
    theirs = functional_harness(thor_profile, apai_backend.speed_cap(thor_profile))
    sys.path.append(str(experiment_dir))
    try:
        rig = importlib.import_module(apai_backend.BOUNDS_MODULE)
    finally:
        sys.path.remove(str(experiment_dir))
    rig.restrict_motion(theirs, bounds, label=apai_backend.BOUNDS_LABEL)
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), thor_profile)
    checkers = apai_backend.build_apai_checkers(config, thor_profile, FakeClock())
    home = np.array(experiment_home(experiment_dir))
    far = home + [0.0, 1.2, 0.0, 0.0, 0.0, 0.0]  # the gripper leaves the radius
    step = 1.0 / RATE_HZ
    for pose in (home, far):
        expected = theirs.vet_step(pose, pose + 1e-4, step)
        ours = checkers.agent.harness.vet_step(pose, pose + 1e-4, step)
        assert ours == expected
    assert "gripper leaves configured radius" in expected
    # The rest pose and the home pose lie inside the bounds: a task can start.
    for pose in (np.array(config.rest.rest_q), home):
        inside = checkers.agent.harness.vet_step(pose, pose + 1e-4, step)
        assert "motion bounds" not in (inside or "")
    # The recovery checker keeps its own limits: its path is not the launcher's.
    kept = checkers.recovery.harness.vet_step(far, far + 1e-4, step)
    assert kept is None or "motion bounds" not in kept


# Row: a dense joint path


def test_a_joint_path_is_one_move_with_the_setpoints_of_apais_path_ticks(
    sim: SimRig,
) -> None:
    """APAI's point_at backs the jaws out along such a path."""
    waypoints = [
        np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.03, 0.0, 0.0, 0.04, 0.0, 0.0]),
        np.array([0.08, 0.0, 0.0, 0.04, 0.0, 0.0]),
    ]
    sim.new_grid()
    done = sim.adapter.stream_path(waypoints, 0.8, approve=sim.checker.approve)
    assert done is True
    (move,) = sim.of("move")
    start = np.asarray(move[2], dtype=float)
    # What APAI's own driver streams for this path from the same reading.
    theirs: list[np.ndarray] = []
    driver = functional_arm(start)
    driver.stream_path(
        waypoints, 0.8, approve=lambda prev, q, dt: theirs.append(np.array(q))
    )
    ours = sim.setpoints()
    assert len(ours) == len(theirs)
    assert all((a == b).all() for a, b in zip(ours, theirs))
    # The arm passed the corner of the path and ended at its last pose.
    assert any(np.allclose(q, waypoints[1], atol=5e-3) for q in ours)
    assert sim.h.sent()[-1] == pytest.approx(waypoints[-1].tolist())
    assert sim.approved() == list(range(len(ours)))


def test_a_joint_path_without_poses_is_refused_as_apai_refuses_it(
    sim: SimRig,
) -> None:
    violation = apai("apai.types").SafetyViolation
    with pytest.raises(violation) as theirs:
        functional_arm([0.0] * 6).stream_path([], 0.5)
    with pytest.raises(violation) as ours:
        sim.adapter.stream_path([], 0.5, approve=sim.checker.approve)
    assert str(ours.value) == str(theirs.value)
    assert not sim.of("move")


# Row: a permit that is late


class LateGate:
    """A permit gate whose cover of setpoint `index` comes `late_s` late."""

    def __init__(self, clock: FakeClock, index: int, late_s: float) -> None:
        self._clock, self._index, self._late_s = clock, index, late_s
        self._since: float | None = None  # when the setpoint first lacked cover
        self.renews = True

    def covers(self, move_id: str, index: int, now: float) -> str | None:
        if index < self._index:
            return None
        if self._since is None:
            self._since = now
        if now - self._since >= self._late_s:
            return None
        return f"permit reaches setpoint {self._index - 1}, not {index}"

    def may_renew(self, move_id: str) -> bool:
        return self.renews

    def last_refusal(self, move_id: str) -> str | None:
        return None


# What one joint read costs APAI's reBot driver, in s (rebot_rs_arm.py: "~8 ms
# each"): two fit into a period, so a joint is read every third period.
RS_READ_S = 0.008


def stretched(travel_rad: float, speed: float, wanted_s: float) -> float:
    """The duration APAI gives a move that asks for less time than its speed allows.

    SafeArm.move_joints stretches such a move until the peak joint speed of
    its minimum-jerk profile is 0.9 of the speed cap.
    """
    return max(wanted_s, 1.875 * travel_rad / (0.9 * speed))


def functional_pause(
    profile: Any, speed: float, pause_s: float, reads_per_cycle: int
) -> bool:
    """Whether APAI's stream completes a move whose check pauses this long once.

    The arm is APAI's software arm that reads one joint at a time, at the
    cost of the rig's driver: the contact monitor then ends a stream when a
    joint went too long without a reading, as on the rig. The move is 0.1 rad
    of one joint, as long as APAI makes it at this speed.
    """
    harness = functional_harness(profile, speed)
    contact = dataclasses.replace(
        harness.limits.contact or apai("apai.safety.contact").ContactMonitorConfig(),
        enabled=True,
        mode="enforce",
        min_reads_per_cycle=reads_per_cycle,
    )
    harness.limits = dataclasses.replace(harness.limits, contact=contact)
    start = [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
    # The software arm streams in steps without a clock; a hardware driver
    # streams with ArmBase's own stream_to, on the machine's clock.
    software = apai("apai.control.mock_arm").MockArm
    paced = apai("apai.control.arm_base").ArmBase.stream_to
    kind = type("PacedArm", (software,), {"stream_to": paced})
    cfg = apai("apai.config").Cfg({"home_q": start, "n_joints": len(start)})
    arm = kind(cfg, profile.kinematics)
    arm.connect()

    def read_one(joint: int, timeout_s: float) -> float:
        time.sleep(RS_READ_S)
        return float(arm._q[joint])

    arm.contact_joint_reader = lambda: read_one
    seen: list[int] = []

    def approve(prev: Any, q: Any, dt: float) -> None:
        seen.append(1)
        if len(seen) == 11:
            time.sleep(pause_s)  # a slow check: the stream waits for it
        harness.approve(prev, q, dt)

    harness.heartbeat()
    harness.begin_motion()
    try:
        return arm.stream_to(
            np.array([0.1, 0.6, 0.6, 0.0, 0.0, 0.0]),
            stretched(0.1, speed, 0.5),
            approve=approve,
            contact_monitor=harness.contact_monitor("test"),
        )
    except apai("apai.types").SafetyViolation:
        return False
    finally:
        harness.end_motion()


@experiment
@pytest.mark.parametrize("late_s", [0.2, 0.5])
def test_a_late_permit_is_waited_for_as_long_as_apais_stream_waits(
    mock_profile: Any, experiment_dir: Path, late_s: float
) -> None:
    """APAI's stream waits for a slow check and ends at a long pause.

    The Thor holds while a permit is late and ends the move where APAI's
    stream would: at the pause its contact monitor allows without a reading.
    """
    config = load_config(ROBOT_RIG)
    speed = config.limits.agent.max_joint_speed
    # The joint reads per period the experiment's launcher asks for.
    settings = yaml.safe_load(
        (experiment_dir / "configs" / "thor_runtime.yaml").read_text("utf-8")
    )
    reads = settings["overrides"]["safety"]["contact_monitor"]["min_reads_per_cycle"]
    assert config.executor.reads_per_step == reads
    theirs = functional_pause(mock_profile, speed, late_s, reads)

    start = [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
    goal = [0.1, 0.6, 0.6, 0.0, 0.0, 0.0]
    thor = Thor(mock_profile, start)
    rows = setpoints(start, goal, stretched(0.1, speed, 0.5))
    ours = thor.run(rows, start, quoted_tol(speed), LateGate(thor.clock, 10, late_s))
    assert (ours.status == "completed") is theirs, ours.reason
    times = [at for at, _ in thor.sent]
    # Nothing went out while the permit was away: the arm held setpoint 9.
    gap = times[10] - times[9] if len(times) > 10 else None
    if theirs:
        assert gap == pytest.approx(late_s + 0.02, abs=0.021)
        assert ours.index == len(rows)
    else:
        assert (ours.status, ours.index) == ("refused", 10)
        assert ours.reason == "permit reaches setpoint 9, not 10"
        assert thor.sent[-1][1] == pytest.approx(rows[9].tolist())


def test_a_cancel_is_not_waited_for(mock_profile: Any) -> None:
    """A permit that takes its cover back ends the move at the next setpoint."""
    speed = load_config(ROBOT_RIG).limits.agent.max_joint_speed
    start = [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
    thor = Thor(mock_profile, start)
    gate = LateGate(thor.clock, 10, 0.05)
    gate.renews = False
    goal = [0.1, 0.6, 0.6, 0.0, 0.0, 0.0]
    rows = setpoints(start, goal, stretched(0.1, speed, 0.5))
    ours = thor.run(rows, start, quoted_tol(speed), gate)
    assert (ours.status, ours.index) == ("refused", 10)
    assert thor.sent[-1][0] - thor.sent[0][0] == pytest.approx(9 * 0.02)


def test_the_pause_a_move_survives_is_the_contact_monitors(
    rs_profile: Any, mock_profile: Any
) -> None:
    contact = apai("apai.safety.contact").ContactMonitorConfig.from_config(
        rs_profile.apai_cfg.safety.get("contact_monitor")
    )
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), rs_profile)
    executor = config.executor
    limit = apai_backend.move_pause_limit(config, rs_profile)
    # One round of joint reads before a joint's next reading is due.
    round_s = config.n_joints / executor.reads_per_step * executor.period_s
    assert limit == pytest.approx(contact.max_feedback_gap_s - round_s)
    arm = apai_backend.build_apai_arm(config, rs_profile, FakeClock())
    assert arm.move_pause_limit_s == limit


# Row: a move the Thor refuses before it sent anything


@pytest.mark.parametrize("sent", [0, 3])
def test_a_move_the_thor_sent_nothing_of_counts_as_apai_counts_it(
    make_scripted: Callable[..., ScriptedRig], sent: int
) -> None:
    """APAI stops the arm after a move only when a command was cleared to go.

    The check ahead clears setpoints before the Thor sends one. A move the
    Thor refused at its first setpoint must read as refused before any
    motion, as in the experiment: the skill then stops nothing.
    """
    runtime = apai("apai.skills.runtime").SkillRuntime
    rig = make_scripted()
    rig.own()
    reason = "joint 2 target -0.002 rad outside [0.020, 3.120]"
    detail = {"outcome": "refused", "sent": sent, "total": 25}
    rig.thor.at_first_permit = ("refused", reason, detail, None)
    skill = types.SimpleNamespace(arm=types.SimpleNamespace(harness=rig.checker))
    before = runtime._approved_commands(skill)
    with pytest.raises(apai("apai.types").SafetyViolation) as raised:
        rig.move(SHORT, 0.5)
    assert rig.of("approve")  # the check ahead had cleared setpoints
    # APAI's own question, asked as its skills ask it.
    assert runtime._refused_before_motion(skill, raised.value, before) is (sent == 0)


# Row: a plain move whose later part is blocked


def test_a_plain_move_blocked_far_ahead_starts_and_stops_as_apais_stream(
    sim: SimRig,
) -> None:
    """APAI checks a plain move setpoint by setpoint, as it sends it.

    A move blocked beyond the reach of its first permit is not refused whole:
    it starts and ends before the blocked setpoint, with the arm holding.
    """
    goal = np.asarray(test_arm_adapter.LONG, dtype=float)
    blocked_at = 0.15  # a pose the grid refuses, 0.75 of the way
    violation = apai("apai.types").SafetyViolation

    # APAI's own stream on its software arm, with the same obstacle.
    driver = functional_arm([0.0] * 6)

    def approve(prev: Any, q: Any, dt: float) -> None:
        if q[0] > blocked_at:
            raise violation("obstacle ahead")

    with pytest.raises(violation, match="obstacle ahead"):
        driver.stream_to(goal, 1.2, approve=approve)
    theirs = len(driver.commands)  # the setpoints its motors got
    assert theirs > sim.settings.reach_setpoints

    sim.obstacle_beyond(blocked_at)
    with pytest.raises(violation, match="obstacle ahead"):
        sim.move(goal, 1.2)
    ours = sim.h.sent()
    # It started, and stopped no later than APAI's stream would have.
    assert 0 < len(ours) <= theirs
    assert max(q[0] for q in ours) <= blocked_at
    assert sim.link.state().owner == "agent"


# Row: a failed joint read at the start of a move


class FailingReads:
    """An arm whose first full reads fail."""

    n_joints = 6

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.reads = 0

    def torque_on(self) -> bool:
        return True

    def read_all(self) -> list[JointReading]:
        self.reads += 1
        if self.reads <= self.failures:
            raise ReadFailed("no answer")
        return [JointReading(joint, 0.0, 0.0, 0.0) for joint in range(6)]


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_a_move_starts_over_as_many_missed_answers_as_apais_driver_takes(
    rs_profile: Any, failures: int
) -> None:
    """The rig's driver serves a start pose over one or two failed reads."""
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), rs_profile)
    driver = apai_backend.build_apai_arm(config, rs_profile, FakeClock())
    backend = driver.backend
    # APAI's driver, not connected: its reads fail, its jaws are not asked.
    backend._read_failures = 0
    backend._gripper_pos = lambda: None

    def no_answer() -> np.ndarray:
        raise RuntimeError("no answer")

    backend._fresh_positions = no_answer
    theirs = True
    for _ in range(failures):
        try:
            backend.get_state()
        except RuntimeError:
            theirs = False

    arm = FailingReads(failures)
    authority = Authority()
    authority.set_owner(Owner.AGENT, "new task")

    class Stopped(Exception):
        pass

    def read_passed(*_args: Any) -> None:
        raise Stopped  # the start read has passed: the move would go on

    checker = types.SimpleNamespace(
        halt_count=lambda: 0,
        set_grasp_region=lambda region: None,
        begin_motion=read_passed,
        end_motion=lambda: None,
    )
    ours_executor = MoveExecutor(
        arm,
        authority,
        None,
        config.executor,
        FakeClock(),
        start_read_tries=driver.start_read_tries,
    )
    try:
        outcome = ours_executor.run(
            [[0.0] * 6],
            [0.0] * 6,
            0.001,
            config.executor.period_s,
            Owner.AGENT,
            checker,
            authority.snapshot(),
            "move-1",
        )
        ours = False
        assert outcome.reason.startswith("the start pose could not be read")
    except Stopped:
        ours = True
    assert ours is theirs


# Row: the pose a skill reads


def test_a_pose_is_never_older_than_the_link_allows(
    make_scripted: Callable[..., ScriptedRig],
) -> None:
    """The experiment's place check takes a pose's time for a fresh reading."""
    rig = make_scripted(state_max_age_s=0.3, fresh_wait_s=0.2)
    rig.own()
    rig.thor.health["reading_age_s"] = 0.2
    test_arm_adapter.until(
        lambda: rig.link.state().health["reading_age_s"] == 0.2, "the older readings"
    )
    state = rig.adapter.get_state()
    assert time.monotonic() - state.t <= 0.3 + 0.02
    # The readings and their ROBOT_STATE together grow older than the limit.
    rig.thor.publishing = False
    time.sleep(0.15)
    began = time.monotonic()
    with pytest.raises(RuntimeError, match="robot link feedback lost"):
        rig.adapter.get_state()
    assert time.monotonic() - began >= 0.2 - 0.02  # it waited for a newer state


# What the motor bus exposes about a hold


def held_arm(profile: Any) -> tuple[Any, FakeClock]:
    clock = FakeClock()
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), profile)
    arm = apai_backend.build_apai_arm(config, profile, clock)
    arm.connect()
    return arm, clock


def test_a_joint_that_leaves_its_hold_is_said_plainly(mock_profile: Any) -> None:
    """APAI's bus has no read of a motor's enable state: the pose is the evidence."""
    monitor = apai("apai.safety.contact").ContactMonitorConfig
    block = mock_profile.apai_cfg.safety.get("contact_monitor")
    contact = monitor.from_config(block) or monitor()
    band = float(contact.tolerances(6)[2])
    arm, clock = held_arm(mock_profile)
    assert arm.hold_evidence()[0] == apai_backend.MOTORS_UNKNOWN  # nothing sent yet
    arm.hold_measured()
    clock.advance(arm._hold._after_s + 0.01)
    arm.read_all()
    assert arm.faults() == []
    assert arm.hold_evidence()[0] == apai_backend.MOTORS_HOLDING
    # Inside the band the joint still holds.
    arm.backend._q[2] += 0.9 * band
    for _ in range(contact.debounce_samples):
        arm.read_joint(2)
    assert arm.faults() == []
    # Beyond it, for as many readings as the contact monitor asks for.
    arm.backend._q[2] += 0.2 * band
    arm.read_joint(2)
    if contact.debounce_samples > 1:
        assert arm.faults() == []
    for _ in range(contact.debounce_samples):
        arm.read_joint(2)
    (fault,) = arm.faults()
    assert fault.startswith("the motors do not hold the arm: joint3 reads ")
    assert "its motor is off" in fault
    state, why = arm.hold_evidence()
    assert (state, why) == (apai_backend.MOTORS_NOT_HOLDING, fault)
    # torque_on() only says that this program has not released the motors.
    assert arm.torque_on() is True


def test_a_joint_on_its_way_to_a_new_target_is_no_lost_hold(
    mock_profile: Any,
) -> None:
    arm, clock = held_arm(mock_profile)
    arm.hold_measured()
    arm.backend._q[2] += 0.5  # as if the arm had not followed yet
    clock.advance(arm._hold._after_s - 0.01)
    for _ in range(5):
        arm.read_joint(2)
    assert arm.faults() == []


class FaultyMotor:
    def __init__(self, fault: int) -> None:
        self.fault = fault

    def robstride_get_fault_report(self) -> tuple[int, int]:
        return self.fault, 0


def test_a_motor_that_reports_a_fault_after_the_connect_is_named(
    mock_profile: Any,
) -> None:
    clock = FakeClock()
    config = apai_backend.config_for_profile(load_config(ROBOT_RIG), mock_profile)
    arm = apai_backend.build_apai_arm(config, mock_profile, clock)
    motors = {"joint2": FaultyMotor(0), "joint5": FaultyMotor(4)}
    arm.backend._arm = types.SimpleNamespace(_motor_map=motors)
    arm.connect()
    # A fault that stood at the connect is not this session's.
    assert arm.faults() == []
    motors["joint2"].fault = 4
    clock.advance(1.0)
    (fault,) = arm.faults()
    assert fault.startswith("motor joint2 reports the fault 0x00000004")
    assert arm.hold_evidence()[0] == apai_backend.MOTORS_NOT_HOLDING
