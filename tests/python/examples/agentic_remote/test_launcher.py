# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program: its settings, APAI's configuration and the launcher.

The runs put the whole launcher (python -m ...agent, in this process) on APAI's
mock scene, with the robot app on APAI's mock arm in real time: on a loopback
carrier, and behind the real robot link service on a link of 127.0.0.1.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
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
# A rig.motion_bounds block that can be used.
BOUNDS: dict[str, Any] = {
    "joint_center_rad": [0.0] * 6,
    "max_joint_deviation_rad": 2.0,
    "tcp_center_m": [0.3, 0.0, 0.2],
    "max_tcp_distance_m": 0.25,
}


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
        # The mock camera and recorded pairs: their times are invented.
        "camera": {"allow_invented_times": True},
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
        ({"task": {"max_joint_speed": 2.0}}, "above the Thor's agent limit"),
        ({"task": {"max_joint_speed": 9.0}}, "max_joint_speed must be at most 3"),
        ({"task": {"max_joint_speed": 0.0}}, "task.max_joint_speed must be"),
        ({"task": {"max_joint_speed": float("nan")}}, "task.max_joint_speed must be"),
        ({"task": {"failed_steps_limit": 0}}, "failed_steps_limit must be"),
        ({"task": {"timed_return": "no"}}, "timed_return must be true or false"),
        ({"experiment": {"stack": "half"}}, "experiment.stack must be"),
        ({"permits": {"reach": 3}}, "permits: unknown key"),
        ({"camera": {"streams": ["one"]}}, "two stream names"),
        ({"rig": {"home_q": [0.0, "a"]}}, "rig.home_q must be"),
        ({"camera": {"streams": ["head.left", "head.left"]}}, "two different"),
        ({"permits": {"renew_s": 1.0, "life_s": 1.0}}, "renew_s must be below"),
        ({"permits": {"renew_s": 0.5, "life_s": 0.1}}, "renew_s must be below"),
        ({"camera": {"allow_invented_times": 1}}, "must be true or false"),
        ({"rig": {"home_q": [0.0] * 6, "park_q": [0.0] * 5}}, "differ in length"),
        (
            {
                "rig": {
                    "home_q": [0.0] * 6,
                    "motion_bounds": {**BOUNDS, "joint_center_rad": [0.0] * 5},
                }
            },
            "differ in length",
        ),
        (
            {"rig": {"motion_bounds": {**BOUNDS, "no_such_key": 1}}},
            "rig.motion_bounds: unknown key",
        ),
        (
            {"rig": {"motion_bounds": {**BOUNDS, "max_tcp_distance_m": -5}}},
            "max_tcp_distance_m must be",
        ),
        (
            {"rig": {"motion_bounds": {**BOUNDS, "max_joint_deviation_rad": 0}}},
            "max_joint_deviation_rad must be",
        ),
        (
            {"rig": {"motion_bounds": {**BOUNDS, "tcp_center_m": [0.0, 0.0]}}},
            "tcp_center_m must be three",
        ),
        (
            {"rig": {"motion_bounds": {"max_tcp_distance_m": 0.2}}},
            "rig.motion_bounds needs",
        ),
    ],
)
def test_a_file_that_cannot_be_used_is_refused(
    tmp_path: Path, change: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        agent_config.load_settings(settings_file(tmp_path, **change))


def test_every_wait_age_and_count_has_an_upper_bound(tmp_path: Path) -> None:
    limits = agent_config.LIMITS
    # No number of the file is without one.
    for block, kind in agent_config._SECTIONS.items():
        for item in dataclasses.fields(kind):
            number = isinstance(item.default, (int, float)) or item.name.endswith("px")
            if number and not isinstance(item.default, bool):
                assert item.name in limits[block], f"{block}.{item.name}"
    for block, keys in limits.items():
        for key, limit in keys.items():
            over = limit + 1 if isinstance(limit, int) else limit * 1.001
            with pytest.raises(ConfigError, match=f"{block}.{key} must be at most"):
                agent_config.load_settings(
                    settings_file(tmp_path, **{block: {key: over}})
                )
    # The shipped file names the bounds, and its own values are inside them.
    shipped = SHIPPED.read_text().lower()
    assert "at most 60 s" in shipped and "at most 3 rad/s" in shipped
    agent_config.load_settings(SHIPPED)


@pytest.mark.parametrize("empty", ["[]", "0", "false", '""', "text"])
def test_a_block_that_is_no_mapping_is_refused(tmp_path: Path, empty: str) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(f"task: {empty}\n")
    with pytest.raises(ConfigError, match="task must be a mapping"):
        agent_config.load_settings(path)
    path.write_text(f"{empty}\n")
    with pytest.raises(ConfigError, match="must hold a mapping"):
        agent_config.load_settings(path)
    # A block or a file with nothing in it is the defaults.
    path.write_text("task:\nlink:\n")
    assert agent_config.load_settings(path).task == agent_config.TaskSettings()
    path.write_text("")
    assert agent_config.load_settings(path).link == agent_config.LinkSettings()


@pytest.mark.parametrize(
    "text",
    [
        "task:\n  max_joint_speed: 0.05\ntask:\n  max_joint_speed: 0.1\n",
        "task:\n  end_wait_s: 1.0\n  end_wait_s: 2.0\n",
        "rig:\n  motion_bounds:\n    tcp_center_m: [0]\n    tcp_center_m: [1]\n",
    ],
    ids=["block", "key", "nested"],
)
def test_a_key_given_twice_is_refused(tmp_path: Path, text: str) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="is given twice"):
        agent_config.load_settings(path)


def test_the_rigs_poses_have_the_arms_joint_count(tmp_path: Path) -> None:
    rig = {"home_q": [0.0] * 6, "park_q": [0.0] * 6, "motion_bounds": BOUNDS}
    settings = agent_config.load_settings(settings_file(tmp_path, rig=rig))
    agent_config.check_joint_count(settings, 6)
    with pytest.raises(ConfigError, match=r"rig\.home_q has 6 values; the arm has 7"):
        agent_config.check_joint_count(settings, 7)
    # Nothing to compare without the rig's values.
    agent_config.check_joint_count(
        agent_config.load_settings(settings_file(tmp_path)), 7
    )


def test_an_unknown_block_or_a_missing_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("speed: 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        agent_config.load_settings(path)
    with pytest.raises(ConfigError, match="cannot read"):
        agent_config.load_settings(tmp_path / "none.yaml")
    for text, message in (
        ("state_dir: 12\n", "state_dir must be text"),
        ("state_dir: [a]\n", "state_dir must be text"),
        ("log_level: 3\n", "log_level must be one of"),
        ("log_level: loud\n", "log_level must be one of"),
    ):
        path.write_text(text)
        with pytest.raises(ConfigError, match=message):
            agent_config.load_settings(path)


def test_a_relative_path_is_taken_from_the_files_folder(tmp_path: Path) -> None:
    path = settings_file(tmp_path, camera={"calibration_dir": "calibration"})
    data = yaml.safe_load(path.read_text())
    data["state_dir"] = "../state"
    path.write_text(yaml.safe_dump(data))
    settings = agent_config.load_settings(path)
    assert settings.camera.calibration_dir == str(tmp_path / "calibration")
    assert settings.state_dir == tmp_path.parent / "state"


@pytest.mark.parametrize(
    "name",
    [
        "APAI_ROBOT",
        "APAI_BASE",
        "APAI_START_STOPPED",
        "APAI_OBSERVED_FINGER_GATE",
        "APAI_OCCUPANCY",
        "APAI_GRASP_EXECUTOR",
        "APAI_GRASP_BACKEND",
        "APAI_AGENT_HARNESS",
        "APAI_PROGRAMS",
        "APAI_BELIEFS",
        "APAI_PREMOTION_CHECK",
        "APAI_ANYTHING_NEW",
    ],
)
def test_an_apai_variable_the_program_does_not_know_is_refused(name: str) -> None:
    with pytest.raises(ConfigError, match=f"{name} is set in the environment"):
        agent_config.refused_environment({name: "1", "APAI_REQUIRE_CUDA": "1"})
    agent_config.refused_environment({name: " "})


def test_the_apai_variables_the_program_takes_are_listed() -> None:
    stores = {name for name, _ in agent_config.APAI_STORES}
    assert stores < set(agent_config.ALLOWED_ENV)
    assert set(agent_config.ALLOWED_ENV) - stores == {
        "APAI_REQUIRE_CUDA",  # run_agent.sh sets it
        "APAI_ASPIRE_LIBRARY",  # the learned-notes library, which is only read
        "APAI_GRASPGENX_PORT",  # the port of a grasp server on this machine
    }
    agent_config.refused_environment(dict.fromkeys(agent_config.ALLOWED_ENV, "x"))
    # Other programs' variables are not looked at.
    agent_config.refused_environment({"APAIX": "1", "MY_APAI_ROBOT": "1", "PATH": "/"})
    # Every variable that is refused is named.
    with pytest.raises(ConfigError, match="APAI_BASE, APAI_ROBOT are set"):
        agent_config.refused_environment({"APAI_ROBOT": "x", "APAI_BASE": "y"})
    assert "APAI_REQUIRE_CUDA" in (EXAMPLE / "run_agent.sh").read_text()


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
    # A variable that is set and empty names no place: APAI would fall back to
    # its own checkout.
    env = {"APAI_BELIEFS_PATH": "", "APAI_EPISODIC_PATH": "  "}
    agent_config.redirect_apai_stores(settings, env)
    for name, leaf in agent_config.APAI_STORES:
        assert env[name] == str(folder / leaf)


def test_the_minimal_stack_pins_apais_mock_detector_for_a_stereo_camera(
    tmp_path: Path,
) -> None:
    camera = {
        "calibration_dir": str(tmp_path),
        "hand_eye_json": str(tmp_path / "hand_eye.json"),
        "bridge_python": "/consumer/python",
    }
    minimal = agent_config.load_settings(settings_file(tmp_path, camera=camera))
    for recorded in ("", str(tmp_path)):
        profile = agent_config._camera_profile(minimal, tmp_path, recorded)
        assert profile["detector"] == {"type": "mock"}
    # The full stack takes the experiment's detector (see _merge_experiment).
    full = agent_config.load_settings(
        settings_file(tmp_path, camera=camera, experiment={"stack": "full"})
    )
    assert "detector" not in agent_config._camera_profile(full, tmp_path, "")


def test_a_camera_with_invented_times_needs_its_setting(tmp_path: Path) -> None:
    refused = agent_config.load_settings(
        settings_file(tmp_path, camera={"allow_invented_times": False})
    )
    assert agent_config.load_settings(SHIPPED).camera.allow_invented_times is False
    assert agent_config.invented_times(refused, "link") is False
    for camera in ("mock", "recorded:/x"):
        with pytest.raises(ConfigError, match=r"camera\.allow_invented_times"):
            agent_config.invented_times(refused, camera)
    allowed = agent_config.load_settings(settings_file(tmp_path))
    assert agent_config.invented_times(allowed, "mock") is True
    assert agent_config.invented_times(allowed, "recorded:/x") is True
    assert agent_config.invented_times(allowed, "link") is False


@pytest.mark.parametrize("camera", ["mock", "recorded:/x"])
@pytest.mark.parametrize("more", [["--task", QUICK], ["--check"]])
def test_the_program_does_not_start_on_invented_times_without_the_setting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], camera: str, more: list[str]
) -> None:
    def never(*args: Any, **kwargs: Any) -> AgentLink:
        raise AssertionError("attached to the robot link service")

    path = settings_file(tmp_path, camera={"allow_invented_times": False})
    code = launcher.main(
        ["--config", str(path), "--camera", camera, *more], attach=never
    )
    assert code == launcher.EXIT_NOT_RUN
    error = capsys.readouterr().err
    assert "camera.allow_invented_times" in error and "invented" in error
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--task", ""], "--task must name a task"),
        (["--task", "  \n"], "--task must name a task"),
        (["--max-steps", "0"], "--max-steps must be at least 1"),
        (["--max-steps", "-3"], "--max-steps must be at least 1"),
    ],
)
def test_arguments_that_cannot_be_used_end_with_exit_code_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    def never(*args: Any, **kwargs: Any) -> AgentLink:
        raise AssertionError("attached to the robot link service")

    path = settings_file(tmp_path)
    code = launcher.main(["--config", str(path), *arguments], attach=never)
    assert code == launcher.EXIT_NOT_RUN
    assert message in capsys.readouterr().err


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
    first = mock.Mock(t=now - 0.2, capture=None)
    stamps.note(first)
    assert times.lookup(first.t) == pytest.approx(5000.0 - 0.2, abs=0.05)
    made.wall = 4999.0  # the estimate of the Thor's clock moved back
    second = mock.Mock(t=now, capture=None)
    stamps.note(second)
    assert times.lookup(second.t) == times.lookup(first.t)
    # No clock of the Thor yet, or a time that is not this machine's: no evidence.
    made.wall = None
    stamps.note(mock.Mock(t=now + 0.001))
    made.wall = 5001.0
    stamps.note(mock.Mock(t=12.0))
    assert len(times) == 2
    # Each camera frame that got such a time says that it was invented.
    assert first.capture == {"invented_times": True}
    kept = mock.Mock(t=time.monotonic(), capture={"backend": "mock"})
    stamps.note(kept)
    assert kept.capture == {"backend": "mock", "invented_times": True}


@apai
def test_a_run_on_invented_times_says_so_at_its_start(
    stage: Stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks, seen = launcher.agent_hooks, []

    @contextlib.contextmanager
    def watched(*args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        with hooks(*args, **kwargs) as built:
            seen.append(built)
            yield built

    monkeypatch.setattr(launcher, "agent_hooks", watched)
    stage.run("--task", LOOK)
    (warning,) = stage.said("camera.allow_invented_times")
    assert warning.startswith("WARNING: the camera frames of --camera mock carry")
    assert "never against a real arm" in warning
    assert stage.lines.index(warning) < stage.lines.index(
        stage.said("robot connection up")[0]
    )
    assert seen[0]["invented_times"] is True
    recorded = json.loads((stage.run_dir / "agent-settings.json").read_text())
    assert recorded["camera"]["allow_invented_times"] is True


@apai
def test_a_registration_that_fails_leaves_apais_names_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera_base = importlib.import_module("apai.perception.camera_base")
    demo = importlib.import_module("apai.apps.demo")
    before = (camera_base.make_camera, demo.make_camera, demo._make_detector)
    settings = agent_config.load_settings(
        settings_file(tmp_path, experiment={"stack": "full"})
    )

    def register_detector() -> None:
        demo._make_detector = lambda *args, **kwargs: None  # then it fails
        raise RuntimeError("the detector cannot be registered")

    monkeypatch.setattr(
        launcher, "_experiment_parts", lambda: {"register_detector": register_detector}
    )
    try:
        with pytest.raises(RuntimeError, match="cannot be registered"):
            with launcher.agent_hooks(None, settings, mock.Mock(), None, tmp_path):
                pass
        after = (camera_base.make_camera, demo.make_camera, demo._make_detector)
        assert after == before
    finally:
        camera_base.make_camera, demo.make_camera, demo._make_detector = before


def test_the_end_of_the_program_waits_until_the_agent_no_longer_owns() -> None:
    class Link:
        def __init__(self, owners: list[str]) -> None:
            self.owners = owners

        def state(self) -> Any:
            owner = self.owners.pop(0) if len(self.owners) > 1 else self.owners[0]
            return mock.Mock(owner=owner)

        def wait_change(self, timeout_s: float) -> None:
            time.sleep(min(timeout_s, 0.01))

    assert launcher._settle(Link(["agent", "agent", "nobody"]), 1.0) is True
    began = time.monotonic()
    assert launcher._settle(Link(["agent"]), 0.2) is False  # it gave up
    assert 0.2 <= time.monotonic() - began < 1.0


@apai
def test_an_end_request_without_an_answer_ends_the_program_with_1(
    stage: Stage, capsys: pytest.CaptureFixture[str]
) -> None:
    from isaaccapture_examples.agentic_remote.agent.link_client import CommandTimeout

    def attach(socket_path: Any = None, *, timeout_s: float) -> Any:
        made = stage.attach(socket_path, timeout_s=timeout_s)
        real = made.request_owner

        def request_owner(name: str, **kwargs: Any) -> Any:
            if name == "nobody":  # the request, or its RESULT, never gets through
                stage.order.append("nobody:no answer")
                raise CommandTimeout("no RESULT within 5 s")
            return real(name, **kwargs)

        made.request_owner = request_owner
        return made

    began = time.monotonic()
    code = stage.run("--task", QUICK, attach=attach)
    # The task succeeded, and the arm is still the agent's.
    assert stage.reports()[0]["success"] is True
    assert stage.order == ["agent:completed", "nobody:no answer", "close:agent"]
    assert code == launcher.EXIT_FAILED
    assert stage.said("was not answered")
    error = capsys.readouterr().err
    assert "the agent still owned the arm when the program ended" in error
    assert time.monotonic() - began > launcher.SETTLE_S  # it waited for the state


@apai
def test_an_interactive_session_ends_with_1_when_a_task_did_not_succeed(
    stage: Stage,
) -> None:
    stage.typed = [QUICK]
    assert stage.run("--interactive") == launcher.EXIT_OK
    assert stage.reports()[-1]["success"] is True
    # The second session: one task succeeds, the other is not authorized.
    stage.typed = [QUICK, QUICK]
    stage.answers = ["yes", "no"]
    assert stage.run("--interactive") == launcher.EXIT_FAILED
    assert [row["success"] for row in stage.reports()[-2:]] == [True, False]
    assert stage.order[-1] == "close:nobody"


@apai
def test_sigterm_is_a_stop_and_ends_the_program_with_1(stage: Stage) -> None:
    def interrupt(number: int) -> None:
        if number == 40:
            signal.raise_signal(signal.SIGTERM)

    stage.h.spy.after_target.append(interrupt)
    code = stage.run("--task", PICK)
    assert code == launcher.EXIT_FAILED
    assert f"stop:{AGENT_TERMINAL}" in stage.order and stage.order[-1] == "close:nobody"
    snapshot = stage.h.app.snapshot()
    assert snapshot.owner == "nobody" and snapshot.stop_number >= 1
    assert stage.h.spy.released_at == []


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


@apai
def test_a_yes_typed_before_the_question_does_not_answer_it() -> None:
    from test_console_cmd import on_a_terminal

    def ask() -> str:
        arm_hold = importlib.import_module("apai.control.arm_hold")
        arm_hold.agent_environment = lambda: None  # in the child process only
        return launcher.terminal_ask("Authorize motion. Type yes to proceed: ")

    # Typed while the task was thinking: dropped. The answer is what is typed
    # once the question is on the terminal.
    assert on_a_terminal(ask, b"yes\n", b"no\n") == (repr("no\n"), True)
    assert on_a_terminal(ask, b"yes\nyes\n", b"\n") == (repr("\n"), True)
    assert on_a_terminal(ask, b"", b"yes\n") == (repr("yes\n"), True)
