# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program: its settings, APAI's configuration and the launcher.

The runs put the whole launcher (python -m ...agent, in this process) on APAI's
mock scene, with the robot app on APAI's mock arm in real time: on a loopback
carrier, and behind the real robot link service on a link of 127.0.0.1.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher, task_gate
from isaaccapture_examples.agentic_remote.agent.arm_adapter import AGENT_TERMINAL
from isaaccapture_examples.agentic_remote.agent.config import ConfigError
from isaaccapture_examples.agentic_remote.agent.link_client import AgentLink, LinkLost
from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
from link_bench import until

from conftest import EXPERIMENT_DIR_ENV, FakeGridClient, apai_runtime_env

apai = pytest.mark.apai
link = pytest.mark.link
experiment = pytest.mark.experiment

EXAMPLE = Path(agent_config.__file__).resolve().parents[4]
PACKAGE = Path(agent_config.__file__).resolve().parents[1]
AGENT = PACKAGE / "agent"
SHIPPED = EXAMPLE / "configs" / "agent.yaml"
# The agent program's own files (unit F).
PROGRAM_FILES = ("config.py", "task_gate.py", "launcher.py", "__main__.py")
# The agent's joint speed in the runs, in rad/s: the mock profile's cap.
SPEED = 1.2
# Tasks APAI's reflex tier turns into one skill, without a model.
PICK = "pick up and place the red object"  # pick_and_place to the drop zone
QUICK = "open the gripper"  # one jaw command
LOOK = "look at the table and report what objects you see"


def settings_file(folder: Path, **blocks: dict[str, Any]) -> Path:
    """An agent.yaml for the mock scene on the robot app of test_apai_app.py."""
    data: dict[str, Any] = {
        "state_dir": str(folder / "state"),
        "link": {
            "expected_profile_id": "apai-rs-model",
            "calibration_id": "apai-mock-calibration",
            "connect_timeout_s": 5.0,
        },
        "thor": {"agent_max_joint_speed": SPEED},
        "task": {"max_joint_speed": SPEED, "grid_wait_s": 10.0},
        "experiment": {"stack": "minimal"},
    }
    for name, block in blocks.items():
        data[name] = {**data.get(name, {}), **block}
    path = folder / "agent.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# The settings


def test_the_package_imports_without_apai_torch_or_the_experiment() -> None:
    code = (
        "import sys\n"
        "from isaaccapture_examples.agentic_remote.agent import "
        "config, task_gate, launcher, console_cmd\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in "
        "('apai', 'torch', 'cupy', 'warp', 'isaaccapture', 'thor_runtime')]\n"
        "assert not bad, bad\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("AGENTIC_REMOTE_APAI_SRC", EXPERIMENT_DIR_ENV, "PYTHONPATH")
    }
    env["PYTHONPATH"] = str(EXAMPLE / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_the_shipped_file_loads_with_the_first_values() -> None:
    settings = agent_config.load_settings(SHIPPED)
    assert settings.experiment.stack == "full"
    assert settings.task.timed_return is False
    assert settings.task.failed_steps_limit == 3
    assert settings.task.max_joint_speed == 0.1
    assert settings.permits.reach_setpoints == 25
    assert (settings.permits.life_s, settings.permits.renew_s) == (1.0, 0.05)
    assert settings.move.start_tol_rad == 0.02
    assert settings.state_dir == EXAMPLE / ".state"
    assert settings.rig.park_q == (0.0,) * 6 and len(settings.rig.home_q) == 6
    assert settings.rig.body_mask_dilate_px == 6
    assert settings.rig.motion_bounds["max_tcp_distance_m"] == 0.25
    arm = agent_config.arm_settings(settings, "cal-1")
    assert arm.calibration_id == "cal-1" and arm.agent_max_joint_speed == 0.1
    assert arm.reach_setpoints == 25 and arm.life_s == 1.0


@experiment
def test_the_rig_values_are_those_of_the_experiments_arm_profile(
    experiment_dir: Path,
) -> None:
    theirs = yaml.safe_load(
        (experiment_dir / "configs/thor_operation.yaml").read_text()
    )
    rig = agent_config.load_settings(SHIPPED).rig
    assert list(rig.home_q) == theirs["home_q"]
    assert list(rig.park_q) == theirs["park_q"]
    assert rig.body_mask_dilate_px == theirs["body_mask"]["dilate_px"]
    assert dict(rig.motion_bounds) == theirs["motion_bounds"]


@experiment
def test_the_read_only_tools_are_the_experiments(experiment_dir: Path) -> None:
    # thor_runtime.py imports its motor side; its list is read, not imported.
    tree = ast.parse((experiment_dir / "thor_runtime.py").read_text())
    (theirs,) = (
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "READ_ONLY_TOOLS"
    )
    names = {item.value for item in theirs.args[0].elts}
    assert names == set(task_gate.READ_ONLY_TOOLS)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"task": {"max_joint_speed": 9.0}}, "above the Thor's agent limit"),
        ({"task": {"max_joint_speed": 0.0}}, "task.max_joint_speed must be"),
        ({"task": {"max_joint_speed": float("nan")}}, "task.max_joint_speed must be"),
        ({"task": {"failed_steps_limit": 0}}, "failed_steps_limit must be"),
        ({"task": {"timed_return": "no"}}, "timed_return must be true or false"),
        ({"experiment": {"stack": "half"}}, "experiment.stack must be"),
        ({"permits": {"reach": 3}}, "permits: unknown key"),
        ({"camera": {"streams": ["one"]}}, "two stream names"),
        ({"rig": {"home_q": [0.0, "a"]}}, "rig.home_q must be"),
    ],
)
def test_a_file_that_cannot_be_used_is_refused(
    tmp_path: Path, change: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        agent_config.load_settings(settings_file(tmp_path, **change))


def test_an_unknown_block_or_a_missing_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("speed: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        agent_config.load_settings(path)
    with pytest.raises(ConfigError, match="cannot read"):
        agent_config.load_settings(tmp_path / "none.yaml")


def test_a_relative_path_is_taken_from_the_files_folder(tmp_path: Path) -> None:
    path = settings_file(tmp_path, camera={"calibration_dir": "calibration"})
    data = yaml.safe_load(path.read_text())
    data["state_dir"] = "../state"
    path.write_text(yaml.safe_dump(data))
    settings = agent_config.load_settings(path)
    assert settings.camera.calibration_dir == str(tmp_path / "calibration")
    assert settings.state_dir == tmp_path.parent / "state"


@pytest.mark.parametrize("name", agent_config.REFUSED_ENV)
def test_an_environment_that_changes_apai_is_refused(name: str) -> None:
    with pytest.raises(ConfigError, match=name):
        agent_config.refused_environment({name: "1"})
    agent_config.refused_environment({name: " "})


def test_the_camera_argument_has_three_forms() -> None:
    assert agent_config.camera_kind("link") == ("link", "")
    assert agent_config.camera_kind("mock") == ("mock", "")
    assert agent_config.camera_kind("recorded:/x/y") == ("recorded", "/x/y")
    for bad in ("recorded:", "usb", ""):
        with pytest.raises(ConfigError, match="--camera must be"):
            agent_config.camera_kind(bad)


def test_the_stores_of_apai_go_into_the_state_folder(tmp_path: Path) -> None:
    settings = agent_config.load_settings(settings_file(tmp_path))
    env = {"APAI_BELIEFS_PATH": "/kept/beliefs.json"}
    folder = agent_config.redirect_apai_stores(settings, env)
    assert folder == tmp_path / "state" / "apai" and folder.is_dir()
    assert env["APAI_BELIEFS_PATH"] == "/kept/beliefs.json"  # the operator's own
    for name, leaf in agent_config.APAI_STORES[1:]:
        assert env[name] == str(folder / leaf)


def test_the_calibration_id_is_the_files_or_the_one_named(tmp_path: Path) -> None:
    named = agent_config.load_settings(settings_file(tmp_path))
    assert agent_config.calibration_id(named, "link") == "apai-mock-calibration"
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "manifest.json").write_text("{}")
    (tmp_path / "hand_eye.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()[:12]
    by_files = agent_config.load_settings(
        settings_file(
            tmp_path,
            link={"calibration_id": ""},
            camera={
                "calibration_dir": str(calibration),
                "hand_eye_json": str(tmp_path / "hand_eye.json"),
            },
        )
    )
    assert agent_config.calibration_id(by_files, "mock") == ""
    assert (
        agent_config.calibration_id(by_files, "link")
        == f"stereo-{digest}+mount-{digest}"
    )
    (tmp_path / "hand_eye.json").unlink()
    with pytest.raises(ConfigError, match="cannot read the calibration file"):
        agent_config.calibration_id(by_files, "recorded:/x")


def test_the_gpu_is_index_0_in_pci_order_whatever_was_set() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    launcher.pin_gpu(env)
    assert env == {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    # The module run with -m sets both before it imports the launcher.
    tree = ast.parse((AGENT / "__main__.py").read_text())
    first_import = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.ImportFrom) and node.level
    )
    set_before = ast.unparse(ast.Module(tree.body[:first_import], []))
    assert "os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'" in set_before
    assert "os.environ['CUDA_VISIBLE_DEVICES'] = '0'" in set_before


def test_the_scripts_set_the_environment_and_name_what_is_missing(
    tmp_path: Path,
) -> None:
    agent = (EXAMPLE / "run_agent.sh").read_text()
    receiver = (EXAMPLE / "run_camera_receiver.sh").read_text()
    for script in (agent, receiver):
        assert "export CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
        assert "export CUDA_VISIBLE_DEVICES=0" in script
        assert "export PYTHONDONTWRITEBYTECODE=1" in script
        assert "unset LD_LIBRARY_PATH" in script
        assert "[ " not in script.replace("[[ ", "")  # [[ ]] only
    assert (
        'exec "${python}" -m isaaccapture_examples.agentic_remote.agent "$@"' in agent
    )
    env = {"PATH": os.environ["PATH"]}
    for script, missing in (
        ("run_agent.sh", "AGENTIC_REMOTE_AGENT_PYTHON must name a Python"),
        ("run_camera_receiver.sh", "AGENTIC_REMOTE_CAMERA_PR must name"),
    ):
        done = subprocess.run(
            [str(EXAMPLE / script)],
            check=False,
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 2 and missing in done.stderr
    # With an interpreter: the program runs under it with the environment set.
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$CUDA_DEVICE_ORDER $CUDA_VISIBLE_DEVICES $APAI_REQUIRE_CUDA '
        '$PYTHONDONTWRITEBYTECODE ${LD_LIBRARY_PATH:-unset} $PYTHONPATH $*"\n'
    )
    fake.chmod(0o755)
    env.update(
        AGENTIC_REMOTE_AGENT_PYTHON=str(fake),
        AGENTIC_REMOTE_APAI_SRC=str(tmp_path),
        CUDA_VISIBLE_DEVICES="1",
        LD_LIBRARY_PATH="/somewhere",
    )
    done = subprocess.run(
        [str(EXAMPLE / "run_agent.sh"), "--check"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [
        "PCI_BUS_ID",
        "0",
        "1",
        "1",
        "unset",
        f"{EXAMPLE}/python:{tmp_path}",
        "-m",
        "isaaccapture_examples.agentic_remote.agent",
        "--check",
    ]


# Source rules of the agent program's own files


def program_sources() -> Iterator[tuple[str, ast.AST]]:
    for name in PROGRAM_FILES:
        yield name, ast.parse((AGENT / name).read_text(encoding="utf-8"))


def test_the_agent_program_sends_owner_requests_and_stop_and_nothing_else() -> None:
    # Of the link's sending calls the launcher and the gate use two; MOVE,
    # PERMIT, GRIPPER and CHECKER_STATE are the arm adapter's.
    sending = {
        "request_owner",
        "stop",
        "send",
        "send_move",
        "send_permit",
        "gripper",
        "checker_state",
        "send_reliable",
        "send_latest",
        "reset",
        "release",
        "rest",
        "open_jaws",
        "shutdown",
    }
    used: dict[str, set[str]] = {}
    requests: set[Any] = set()
    for name, tree in program_sources():
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            target = ast.unparse(node.func.value)
            if node.func.attr in sending and target != "self._current":  # a ContextVar
                used.setdefault(name, set()).add(node.func.attr)
                assert target in ("link", "self._link"), (name, target)
            if node.func.attr == "request_owner":
                requests.add(ast.unparse(node.args[0]))
    assert used == {"task_gate.py": {"request_owner"}, "launcher.py": {"stop"}}
    # The owner requests of the gate: agent, nobody and help_needed.
    assert requests == {"'agent'", "'nobody'", "request"}
    gate = ast.parse((AGENT / "task_gate.py").read_text())
    (ending,) = (
        node
        for node in ast.walk(gate)
        if isinstance(node, ast.FunctionDef) and node.name == "_ending"
    )
    ends = {
        ast.unparse(node.value.elts[0])
        for node in ast.walk(ending)
        if isinstance(node, ast.Return)
    }
    assert ends == {"None", "'help_needed'", "'nobody'"}
    # No link message class is built in these files at all.
    for name, tree in program_sources():
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("messages")
            for alias in node.names
        }
        assert imported <= {"RobotState"}, (name, imported)


def test_nothing_in_the_agent_program_can_cut_torque_or_start_a_return_to_rest() -> (
    None
):
    import test_no_limp

    files = {f"agent/{name}" for name in (*PROGRAM_FILES, "console_cmd.py")}
    assert not [site for site in test_no_limp.package_sites() if site.file in files]
    for name, tree in program_sources():
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names & {
            "Release",
            "Shutdown",
            "Open",
            "Reset",
            "release",
            "open_jaws",
            "move_to_rest",
            "disconnect",
            "resume",
        }, name


# The launcher on APAI's mock scene


class Stage:
    """The robot app on APAI's mock arm in real time, and one launcher run on it."""

    def __init__(self, harness: Any, folder: Path) -> None:
        self.h, self.folder = harness, folder
        self.lines: list[str] = []  # what the program said
        self.asked: list[str] = []  # the prompts of the typed yes
        self.answers = ["yes"]  # what is typed, one per prompt; then "no"
        self.typed: list[str] = []  # the task lines of --interactive
        self.order: list[str] = []  # the link's requests, STOPs and close
        self.motion: list[str] = []  # the MOVEs and GRIPPERs the agent sent
        self.link: AgentLink | None = None
        self.on_ask: Callable[[], None] | None = None
        self.run_dir = folder / "run"
        # The robot app referenced the jaws when it started.
        self._jaws_before = len(harness.spy.jaw_commands)

    def jaw_commands(self) -> list[tuple[float, float]]:
        """The jaw commands the driver took since the stage was made."""
        return self.h.spy.jaw_commands[self._jaws_before :]

    def attach(self, socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
        robot_end, agent_end = loopback_pair()
        made = AgentLink(agent_end)
        assert self.h.app.attach_carrier(robot_end)
        self.watch(made)
        return made

    def watch(self, made: AgentLink) -> None:
        """Note the link's owner requests, STOPs and close, in order."""
        self.link = made
        request, stop, close = made.request_owner, made.stop, made.close

        def request_owner(name: str, **kwargs: Any) -> Any:
            result = request(name, **kwargs)
            self.order.append(f"{name}:{result.status}")
            return result

        def stop_now(reason: str, sender: str) -> None:
            self.order.append(f"stop:{sender}")
            stop(reason, sender)

        def close_now() -> None:
            try:
                owner = made.state().owner
            except RuntimeError:
                owner = "?"
            self.order.append(f"close:{owner}")
            close()

        made.request_owner, made.stop, made.close = request_owner, stop_now, close_now
        for name in ("send_move", "gripper"):
            setattr(made, name, self._noted(name, getattr(made, name)))

    def _noted(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def noted(*args: Any, **kwargs: Any) -> Any:
            self.motion.append(name)
            return real(*args, **kwargs)

        return noted

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        if self.on_ask is not None:
            self.on_ask()
        return self.answers.pop(0) if self.answers else "no"

    def next_line(self, prompt: str = "") -> str:
        if not self.typed:
            raise EOFError
        return self.typed.pop(0)

    def run(self, *argv: str, attach: Any = None, **blocks: dict[str, Any]) -> int:
        occupancy = importlib.import_module("apai.perception.occupancy")
        path = settings_file(self.folder, **blocks)
        arguments = ["--config", str(path), "--camera", "mock"]
        arguments += ["--run-dir", str(self.run_dir), *argv]
        with mock.patch.object(occupancy, "OccupancyClient", FakeGridClient):
            return launcher.main(
                arguments,
                attach=attach or self.attach,
                ask=self.ask,
                say=self.lines.append,
                task_input=self.next_line,
            )

    def said(self, piece: str) -> list[str]:
        return [line for line in self.lines if piece in line]

    def reports(self) -> list[dict[str, Any]]:
        file = self.run_dir / "task-reports.jsonl"
        return [json.loads(line) for line in file.read_text().splitlines()]

    def results(self, tool: str) -> list[dict[str, Any]]:
        return [
            row["result"]
            for report in self.reports()
            for row in report["tool_log"]
            if row["tool"] == tool
        ]


def scene_robot_config() -> Any:
    """The robot app's file for the mock scene.

    That of test_apai_app.py with the agent's envelope as wide as its joint
    box (the mock scene's grasp turns the fourth joint past 1 rad) and the
    agent's speed at the mock profile's cap, which keeps the moves short.
    """
    import dataclasses

    import test_apai_app

    backend = test_apai_app.apai_backend
    config = test_apai_app.make_config(backend.load_apai_profile("mock"))
    agent = dataclasses.replace(
        config.limits.agent,
        envelope_min=tuple(config.limits.agent.joint_min),
        envelope_max=tuple(config.limits.agent.joint_max),
        max_joint_speed=SPEED,
    )
    return dataclasses.replace(
        config, limits=dataclasses.replace(config.limits, agent=agent)
    )


@pytest.fixture
def stage(make_realtime_apai: Callable[..., Any], tmp_path: Path) -> Iterator[Stage]:
    import test_apai_app
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState

    # Started at rest, so the jaws are referenced; then put off the joint
    # stops, holding, where APAI's checker lets the arm move.
    harness = make_realtime_apai(config=scene_robot_config(), connected=False)
    harness.spy.backend._q[:] = test_apai_app.RAISED
    harness.app._motion.set(MotionState.HOLD)
    harness.spy.backend.object_stop_frac = 0.5  # an object between the jaws
    with apai_runtime_env(tmp_path) as folder:
        yield Stage(harness, folder)


@apai
def test_a_mock_pick_and_place_runs_through_the_launcher(stage: Stage) -> None:
    demo = importlib.import_module("apai.apps.demo")
    names = ("build_runtime", "_build_arm", "make_arm", "make_camera")
    names += ("_make_detector", "AgentOrchestrator", "_print_report")
    before = {name: getattr(demo, name) for name in names}
    camera_base = importlib.import_module("apai.perception.camera_base")
    orchestrator = importlib.import_module("apai.agent.orchestrator")
    others = (camera_base.make_camera, orchestrator.tool_specs_for_config)
    began = time.monotonic()
    code = stage.run("--task", PICK)
    assert time.monotonic() - began < 60.0
    # APAI's names are its own again: nothing stays patched after the run.
    assert {name: getattr(demo, name) for name in names} == before
    assert not hasattr(demo, "input")
    assert (camera_base.make_camera, orchestrator.tool_specs_for_config) == others
    # The skill picked the object and put it down at the drop zone.
    (result,) = stage.results("pick_and_place")
    assert result["ok"] is True and result["picked"] == "red object"
    assert result["destination"] == "drop zone"
    assert result["return_home"]["ok"] is True
    assert len(stage.h.spy.targets) > 500  # the motors got the moves' setpoints
    assert len(stage.jaw_commands()) >= 2  # and the jaws their commands
    # One typed yes, where the task was typed, before the first motion skill.
    (prompt,) = stage.asked
    assert json.dumps(PICK) in prompt and "First tool pick_and_place" in prompt
    assert f"at most {SPEED:g} rad/s" in prompt and "Type yes to proceed" in prompt
    # The mock scene cannot confirm the place, so the task ends without
    # success while the agent owns the arm: a help request (D6), answered
    # before the link is closed.
    assert stage.order == ["agent:completed", "help_needed:completed", "close:nobody"]
    assert code == launcher.EXIT_FAILED
    (help_needed,) = stage.h.heard("help_needed")
    assert help_needed[2]["reason"].startswith("task ended without success: ")
    assert "\n" not in help_needed[2]["reason"]
    assert help_needed[2]["timed_return"] is False
    assert stage.said("HELP NEEDED: task ended without success")
    assert stage.said("obstacle grid: live") and stage.said("robot connection up")
    # No torque was cut and the workstation checker is not latched.
    assert stage.h.spy.released_at == [] and stage.h.app.snapshot().owner == "nobody"
    assert not stage.said("restart the agent program")
    # What the run left: in its folder and the state folder, nowhere else.
    for name in ("arm_profile.yaml", "agent-settings.json", "task-reports.jsonl"):
        assert (stage.run_dir / name).is_file()
    profile = yaml.safe_load((stage.run_dir / "arm_profile.yaml").read_text())
    assert profile == {"extends": "mock", "type": "robot_link"}
    assert (stage.folder / "state" / ".gitignore").read_text() == "*\n"


@apai
def test_looking_takes_no_arm_and_asks_nothing(stage: Stage) -> None:
    code = stage.run("--task", LOOK)
    assert stage.asked == [] and stage.order == ["close:nobody"]
    assert stage.h.heard("owner") == [] and stage.h.spy.targets == []
    assert stage.reports()[0]["tool_log"]  # the model looked
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED)


@apai
@pytest.mark.parametrize("answer", ["no", "", "y", "yes please"])
def test_without_the_typed_yes_nothing_is_asked_of_the_robot(
    stage: Stage, answer: str
) -> None:
    stage.answers = [answer]
    code = stage.run("--task", QUICK)
    assert len(stage.asked) == 1
    assert stage.order == ["close:nobody"] and stage.h.heard("owner") == []
    assert stage.motion == [] and stage.h.spy.targets == []
    assert stage.jaw_commands() == []
    (result,) = stage.results("open_gripper")
    assert result["refused"] == "consent" and result["motion_blocked"] is True
    assert result["error"] == "the operator did not type yes; no command sent"
    assert stage.h.heard("help_needed") == []  # the agent never owned the arm
    assert code == launcher.EXIT_FAILED


@apai
def test_a_short_task_gives_the_arm_back_and_then_closes_the_link(
    stage: Stage,
) -> None:
    code = stage.run("--task", QUICK)
    assert stage.order[0] == "agent:completed" and stage.order[-1] == "close:nobody"
    assert len(stage.order) == 3 and "stop" not in " ".join(stage.order)
    (result,) = stage.results("open_gripper")
    assert result["ok"] is True
    owners = [event[2].get("owner") for event in stage.h.heard("owner")]
    assert owners[0] == "agent"
    assert stage.reports()[0]["success"] is True, stage.reports()[0]["summary"]
    assert stage.order[1] == "nobody:completed" and code == launcher.EXIT_OK
    assert stage.h.heard("help_needed") == []
    assert stage.said("task completed: the arm holds where it is")


@apai
def test_ctrl_c_at_the_prompt_is_a_stop_and_no_task_starts(stage: Stage) -> None:
    stage.on_ask = lambda: signal.raise_signal(signal.SIGINT)
    before = stage.h.app.snapshot().stop_number
    code = stage.run("--task", QUICK)
    assert stage.order == [f"stop:{AGENT_TERMINAL}", "close:nobody"]
    after = stage.h.app.snapshot()
    assert after.stop_number == before + 1 and after.owner == "nobody"
    assert stage.motion == [] and stage.h.heard("owner") == []
    assert code == launcher.EXIT_OK  # ended by the operator


@apai
def test_ctrl_c_during_a_move_is_a_persons_stop_and_no_help_request(
    stage: Stage,
) -> None:
    def interrupt(number: int) -> None:
        if number == 40:
            signal.raise_signal(signal.SIGINT)

    stage.h.spy.after_target.append(interrupt)
    code = stage.run("--task", PICK)
    assert stage.order == ["agent:completed", f"stop:{AGENT_TERMINAL}", "close:nobody"]
    sent = len(stage.h.spy.targets)
    assert 40 <= sent < 80  # the move ended within the reach of its last permit
    snapshot = stage.h.app.snapshot()
    assert snapshot.owner == "nobody" and snapshot.stop_number == 1
    # A person's STOP: no help request, no assistance wait, torque on.
    assert stage.h.heard("help_needed") == []
    assert stage.h.app._wait.remaining(stage.h.clock.monotonic()) is None
    assert stage.h.spy.released_at == []
    assert stage.said("the task was cancelled")
    time.sleep(0.2)
    assert len(stage.h.spy.targets) == sent  # the arm holds
    assert code == launcher.EXIT_OK


@apai
def test_after_a_skill_stop_the_program_says_restart_and_takes_no_new_task(
    stage: Stage,
) -> None:
    # The driver refuses a joint target of the first move: a fault on the
    # Thor, a failed move for APAI, whose skill then stops the arm.
    stage.h.spy.fail_target = 30
    stage.typed = [PICK, QUICK]
    code = stage.run("--interactive")
    harness_of_run = stage.reports()[0]
    assert harness_of_run["success"] is False
    # The skill stop sent no STOP; the help request found the Thor's own
    # fault-stop, which had made nobody the owner already.
    assert stage.order == ["agent:completed", "help_needed:refused", "close:nobody"]
    assert stage.said("refused by the robot") and stage.said("nothing else is sent")
    # The second task was never read: the program said to restart and ended.
    assert stage.typed == [QUICK] and len(stage.reports()) == 1
    assert stage.said(task_gate.RESTART) and len(stage.asked) == 1
    assert code == launcher.EXIT_FAILED
    assert stage.h.spy.released_at == []
    # Nothing was sent after the failed move: no MOVE, no GRIPPER, no STOP.
    assert stage.motion.count("send_move") == 1


@apai
def test_check_connects_to_nothing_and_prints_the_resolved_values(
    stage: Stage, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(*args: Any, **kwargs: Any) -> AgentLink:
        raise AssertionError("--check attached to the robot link service")

    code = stage.run("--check", attach=never)
    assert code == launcher.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["configuration"] == "valid"
    assert report["calibration_id"] == "apai-mock-calibration"
    assert report["arm_type"] == "robot_link" and report["camera_type"] == "mock"
    assert report["max_joint_vel_rad_s"] == SPEED
    assert report["obstacle_grid"]["required"] is True
    assert report["settings"]["task"]["failed_steps_limit"] == 3
    assert not (stage.folder / "state").exists() and not stage.run_dir.exists()


@apai
def test_a_program_that_cannot_start_says_why_and_sends_nothing(
    stage: Stage, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def nobody_there(*args: Any, **kwargs: Any) -> AgentLink:
        raise LinkLost("cannot attach to the robot link service at x: nothing")

    assert stage.run("--task", QUICK, attach=nobody_there) == launcher.EXIT_FAILED
    assert "no robot connection: cannot attach" in capsys.readouterr().err
    # Another camera calibration than the robot app's: the arm does not connect.
    code = stage.run("--task", QUICK, link={"calibration_id": "another"})
    assert code == launcher.EXIT_FAILED
    assert "camera calibration is 'apai-mock-calibration'" in capsys.readouterr().err
    assert stage.order == ["close:nobody"] and stage.asked == []
    assert stage.h.heard("owner") == [] and stage.h.spy.targets == []
    # What cannot be used at all: exit code 2, before anything is attached.
    monkeypatch.setenv("APAI_START_STOPPED", "1")
    assert stage.run("--task", QUICK) == launcher.EXIT_NOT_RUN
    assert "APAI_START_STOPPED is set" in capsys.readouterr().err
    monkeypatch.delenv("APAI_START_STOPPED")
    assert stage.run("--task", QUICK, task={"max_joint_speed": 9.0}) == 2
    assert stage.order == ["close:nobody"]


@apai
def test_apais_configuration_is_built_for_the_robot_link(
    apai_scratch: Path,
) -> None:
    settings = agent_config.load_settings(settings_file(apai_scratch))
    cfg = agent_config.build_apai_config(settings, apai_scratch / "run", camera="mock")
    data = cfg.as_dict()
    assert len(data["arms"]) == 1
    for view in (data, data["arms"][0]["resolved"]):
        assert view["arm"]["type"] == "robot_link"
        assert view["occupancy"]["enabled"] is True
        assert view["occupancy"]["required"] is True
        assert view["safety"]["max_joint_vel"] == SPEED
    assert cfg.memory.experience_path == str(
        apai_scratch / "state" / "apai" / "experience.json"
    )
    assert cfg.llm.type == "mock" and cfg.camera.type == "mock"
    # A stereo camera needs its calibration; a speed above the Thor's is
    # refused by the loader already.
    with pytest.raises(ConfigError, match=r"camera\.calibration_dir"):
        agent_config.build_apai_config(settings, apai_scratch / "run", camera="link")
    with pytest.raises(ConfigError, match="APAI_ROBOT"):
        agent_config.build_apai_config(
            settings, apai_scratch / "run", camera="mock", environ={"APAI_ROBOT": "x"}
        )


@apai
def test_a_stereo_camera_profile_names_the_bridge_or_the_recorded_pairs(
    apai_scratch: Path,
) -> None:
    calibration = apai_scratch / "calibration"
    calibration.mkdir()
    blocks = {
        "camera": {
            "calibration_dir": str(calibration),
            "hand_eye_json": str(apai_scratch / "hand_eye.json"),
            "bridge_python": "/consumer/python",
            "depth_backend": "opencv_cpu",
        }
    }
    settings = agent_config.load_settings(settings_file(apai_scratch, **blocks))
    run = apai_scratch / "run"
    live = agent_config._camera_profile(settings, run, "")
    assert live["type"] == "link_stereo" and live["name"] == "head"
    assert (live["width"], live["height"], live["fps"]) == (1280, 992, 7.0)
    assert live["map_depth"] is True and live["fuse_beliefs"] is False
    assert live["bridge"] == {
        "python": "/consumer/python",
        "receiver_socket": "/tmp/camera_receiver_avs1_cuda_ipc.sock",
        "run_dir": str(run),
        "streams": ["head.left", "head.right"],
        "max_skew_ms": 5.0,
    }
    recorded = agent_config._camera_profile(settings, run, str(apai_scratch))
    assert recorded["recorded_dir"] == str(apai_scratch) and "bridge" not in recorded
    # A hand-eye file APAI does not accept as calibrated is refused.
    with pytest.raises(ConfigError):
        agent_config.build_apai_config(settings, run, camera=f"recorded:{run}")


@apai
@link
def test_the_launcher_runs_behind_the_real_service_and_closes_after_the_result(
    realtime_apai_stack: Any, tmp_path: Path
) -> None:
    import test_apai_app
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import MotionState

    stack = realtime_apai_stack
    stack.h.spy.backend._q[:] = test_apai_app.RAISED
    stack.h.app._motion.set(MotionState.HOLD)
    stack.start_service()
    stack.dial()
    console = stack.client("console")
    stack.connected(console)
    with apai_runtime_env(tmp_path) as folder:
        stage = Stage(stack.h, folder)

        def attach(socket_path: str | None = None, *, timeout_s: float) -> AgentLink:
            assert socket_path == stack.path
            made = AgentLink.attach(socket_path, timeout_s=timeout_s)
            stage.watch(made)
            return made

        code = stage.run(
            "--task",
            QUICK,
            attach=attach,
            link={"socket_path": stack.path},
            thor={"agent_max_joint_speed": 0.4},
            task={"max_joint_speed": 0.4},
        )
    assert stage.order[0] == "agent:completed" and stage.order[-1] == "close:nobody"
    assert stage.order[1].endswith(":completed") and len(stage.order) == 3
    assert stage.results("open_gripper")[0]["ok"] is True
    assert code in (launcher.EXIT_OK, launcher.EXIT_FAILED)
    # The service saw the agent leave with nobody as owner: it asked for no help
    # in the agent's name.
    until(lambda: stack.service.robot_state.owner == "nobody", "owner nobody")
    time.sleep(0.3)
    reasons = [event[2].get("reason", "") for event in stack.h.heard("help_needed")]
    assert "agent program gone" not in reasons
    assert stack.h.spy.released_at == []


# The full stack's configuration, and what the launcher records


@apai
@experiment
def test_the_full_stack_merges_the_experiments_files_over_apais(
    apai_scratch: Path, experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
    settings = agent_config.load_settings(SHIPPED)
    cfg = agent_config.build_apai_config(settings, apai_scratch / "run", camera="mock")
    data = cfg.as_dict()
    for view in (data, data["arms"][0]["resolved"]):
        assert view["arm"]["type"] == "robot_link"
        assert view["safety"]["max_joint_vel"] == 0.1  # task.max_joint_speed
        assert view["occupancy"]["required"] is True
        assert view["occupancy"]["max_age_s"] == 3.0  # the experiment's
        assert view["perception_loop"]["rate_hz"] == 7
        assert view["llm"]["type"] == "codex_exec"
        assert view["detector"]["type"] == "thor_astra"
        assert view["memory"]["persist_beliefs"] is False
    arm = data["arm"]
    assert arm["home_q"] == list(settings.rig.home_q)
    assert arm["motion_bounds"]["max_tcp_distance_m"] == 0.25
    assert arm["body_mask"]["dilate_px"] == 6
    assert arm["body_mask"]["attachments"]  # of the experiment's arm profile
    profile = yaml.safe_load((apai_scratch / "run" / "arm_profile.yaml").read_text())
    assert profile["extends"] == str(experiment_dir / "configs" / "rebot_workstation")
    # Without the experiment folder the full stack is refused, with the name
    # of the variable.
    monkeypatch.delenv(EXPERIMENT_DIR_ENV)
    with pytest.raises(ConfigError, match=EXPERIMENT_DIR_ENV):
        agent_config.build_apai_config(settings, apai_scratch / "run", camera="mock")


@apai
@experiment
def test_the_experiments_hooks_are_found_where_the_launcher_looks(
    experiment_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
    try:
        parts = launcher._experiment_parts()
    except ImportError as missing:
        pytest.skip(f"an experiment module cannot be imported here: {missing}")
    for name, part in parts.items():
        assert part is not None, name
    assert "move_to_rest" not in parts["guidance"]


def test_the_hashes_of_the_experiment_files_a_run_imported(tmp_path: Path) -> None:
    folder = tmp_path / "experiment"
    (folder / "sub").mkdir(parents=True)
    (folder / "thor_x.py").write_text("x = 1\n")
    (folder / "sub" / "thor_y.py").write_text("y = 2\n")
    (tmp_path / "other.py").write_text("z = 3\n")
    modules = {
        "thor_x": mock.Mock(__file__=str(folder / "thor_x.py")),
        "thor_y": mock.Mock(__file__=str(folder / "sub" / "thor_y.py")),
        "other": mock.Mock(__file__=str(tmp_path / "other.py")),
        "builtin": mock.Mock(__file__=None),
    }
    assert agent_config.file_hashes(folder, modules) == {
        "sub/thor_y.py": hashlib.sha256(b"y = 2\n").hexdigest(),
        "thor_x.py": hashlib.sha256(b"x = 1\n").hexdigest(),
    }
    with mock.patch.dict(os.environ, {EXPERIMENT_DIR_ENV: str(folder)}):
        with mock.patch.dict(sys.modules, {"thor_x": modules["thor_x"]}):
            launcher._record_experiment_files(tmp_path)
    recorded = json.loads((tmp_path / "experiment-files.json").read_text())
    assert recorded["thor_x.py"] == hashlib.sha256(b"x = 1\n").hexdigest()


def test_camera_frames_of_a_local_camera_get_evidence_that_never_goes_back() -> None:
    from isaaccapture_examples.agentic_remote.agent.checkahead import FrameTimes
    from isaaccapture_examples.agentic_remote.agent.link_client import StaleState

    class Link:
        wall = 5000.0

        def thor_now(self) -> tuple[float, float]:
            if self.wall is None:
                raise StaleState("no ROBOT_STATE has arrived yet")
            return 0.0, self.wall

    made, times = Link(), FrameTimes()
    stamps = launcher.LocalFrameStamps(made, times)
    now = time.monotonic()
    first = mock.Mock(t=now - 0.2)
    stamps.note(first)
    assert times.lookup(first.t) == pytest.approx(5000.0 - 0.2, abs=0.05)
    made.wall = 4999.0  # the estimate of the Thor's clock moved back
    second = mock.Mock(t=now)
    stamps.note(second)
    assert times.lookup(second.t) == times.lookup(first.t)
    # No clock of the Thor yet, or a time that is not this machine's: no evidence.
    made.wall = None
    stamps.note(mock.Mock(t=now + 0.001))
    made.wall = 5001.0
    stamps.note(mock.Mock(t=12.0))
    assert len(times) == 2


@apai
def test_the_typed_yes_is_never_taken_inside_an_agents_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(RuntimeError, match="outside an agent environment"):
        launcher.terminal_ask("Type yes: ")
    # Without a terminal on stdin nobody is asked either.
    arm_hold = importlib.import_module("apai.control.arm_hold")
    monkeypatch.setattr(arm_hold, "agent_environment", lambda: None)
    with pytest.raises(RuntimeError, match="no interactive terminal"):
        launcher.terminal_ask("Type yes: ")
