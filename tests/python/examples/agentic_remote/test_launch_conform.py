# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's start against the experiment's own launcher.

The experiment is the folder of the rig's single-machine program, which picks
and places on the real arm; its launcher is thor_runtime.sh with
thor_runtime.py. Each test here holds one thing the agent program decides at
its start against what that launcher decides: the environment, APAI's
configuration key by key, what the model reads, and the files of a run's
start. The launcher's own routines and files are the reference; no value is
typed here. A mounting record is the launcher's record of the accepted fit of
the camera's place beside the arm, with the hash of its hand-eye file.

The tests' interpreter may lack scipy, which one module the launcher imports
(thor_occupancy) needs and none of the routines called here: that module is
then a stand-in while the launcher is imported.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import test_launcher
import yaml
from isaaccapture_examples.agentic_remote.agent import config as agent_config
from isaaccapture_examples.agentic_remote.agent import launcher
from isaaccapture_examples.agentic_remote.agent.config import ConfigError
from test_launcher import LOOK, Stage

from conftest import EXPERIMENT_DIR_ENV

apai = pytest.mark.apai
experiment = pytest.mark.experiment
# The fixtures of test_launcher.py: the launcher on APAI's mock scene.
make_stage = test_launcher.make_stage
stage = test_launcher.stage

EXAMPLE = Path(agent_config.__file__).resolve().parents[4]
AGENT_RIG = EXAMPLE / "configs" / "agent_rebot.yaml"
RUN_AGENT = EXAMPLE / "run_agent.sh"
RUN_WORKSTATION = EXAMPLE / "run_workstation.sh"

# What run_agent.sh exports that the experiment's launcher does not, and why a
# second machine needs it.
EXPORTED = {
    "AGENTIC_REMOTE_APAI_SRC": "the package's own variables, for its children",
    "AGENTIC_REMOTE_EXPERIMENT_DIR": "the package's own variables",
    "AGENTIC_REMOTE_CAMERA_PR": "the package's own variables",
    "AGENTIC_REMOTE_ENV_DIR": "the package's own variables",
    "CUDA_DEVICE_ORDER": "the workstation has several cards; the Thor has one",
    "CUDA_VISIBLE_DEVICES": "the workstation has several cards; the Thor has one",
    "TRITON_CACHE_DIR": "the experiment folder is only read here",
    "TORCHINDUCTOR_CACHE_DIR": "the experiment folder is only read here",
    "YOLO_CONFIG_DIR": "the experiment folder is only read here",
    "PYTHONDONTWRITEBYTECODE": "the experiment folder is only read here",
    "GIT_OPTIONAL_LOCKS": "the experiment folder is only read here",
    "PYTHONPATH": "the example is not installed in the agent's environment",
}
# The keys of APAI's configuration that differ from the launcher's, and why a
# split over two machines needs each to.
DIFFERS = {
    "arm.type": "the arm is the robot link, not the motor driver",
    "arm.channel": "the motor bus is the Thor's",
    "arm.hw_yaml": "the motor file is the Thor's",
    "camera.type": "the camera frames come over the camera link",
    "camera.bridge.python": "the camera link's receiver, workstation only",
    "camera.bridge.receiver_socket": "the camera link's receiver",
    "camera.bridge.run_dir": "the camera link's receiver",
    "camera.bridge.streams": "the camera link's receiver",
    "camera.bridge.max_skew_ms": "the camera link's receiver",
    "camera.calibration_dir": "a path: the workstation's copy of the same files",
    "camera.capture_binary": "the capture program runs on the Thor",
    "camera.capture_format": "the capture program runs on the Thor",
    "camera.lock_file": "the capture program runs on the Thor",
    "camera.state_dir": "the capture program runs on the Thor",
    "llm.codex_bin": "a path: the model client's program on the workstation",
    "memory.experience_path": "a path: the experiment folder is only read here",
}
# The one more key that differs with the shipped rig file, and why: the robot
# app's own speed limit, which the motion unit and the owner raise on the Thor.
THOR_LIMIT = "safety.max_joint_vel"


def flat(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested configuration as {dotted key: value}."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flat(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
        for index, item in enumerate(value):
            out.update(flat(item, f"{prefix}[{index}]"))
    else:
        out[prefix] = value
    return out


def top_level(key: str) -> str:
    """A key of an arm's own view of the configuration, as its top-level key."""
    for prefix in ("arms[0].resolved.", "arm.resolved."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if key.startswith("arms[0]."):
        key = "arm." + key[len("arms[0].") :]
    return key.replace("cameras[0].", "camera.")


def differing(theirs: dict[str, Any], ours: dict[str, Any]) -> set[str]:
    a, b = flat(theirs), flat(ours)
    missing = object()
    return {
        top_level(key)
        for key in set(a) | set(b)
        if a.get(key, missing) != b.get(key, missing)
    }


@pytest.fixture(scope="module")
def functional(experiment_dir: Path) -> Iterator[Any]:
    """The experiment's launcher module, thor_runtime."""
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
            vars(stand_in)["install_grid_sampler"] = lambda occupancy: None
            sys.modules["thor_occupancy"] = stand_in
        module = importlib.import_module("thor_runtime")
    except ModuleNotFoundError as missing:
        _forget(stand_in)
        pytest.skip(f"the experiment's launcher cannot be imported here: {missing}")
    try:
        yield module
    finally:
        _forget(stand_in)


def _forget(stand_in: types.ModuleType | None) -> None:
    """Take the stand-in, and the launcher imported with it, away again."""
    if stand_in is not None:
        sys.modules.pop("thor_runtime", None)
        if sys.modules.get("thor_occupancy") is stand_in:
            del sys.modules["thor_occupancy"]


class Rig:
    """The shipped rig file on this machine's copy of the experiment folder."""

    def __init__(self, experiment_dir: Path, folder: Path, launcher_module: Any):
        self.experiment_dir, self.folder = experiment_dir, folder
        self.theirs_module = launcher_module
        shipped = agent_config.load_settings(AGENT_RIG)
        self.settings = dataclasses.replace(shipped, state_dir=folder / "state")
        self._count = 0

    def with_limit(self, limit: float) -> Any:
        """The settings with another copy of the Thor's agent speed limit."""
        thor = dataclasses.replace(self.settings.thor, agent_max_joint_speed=limit)
        return dataclasses.replace(self.settings, thor=thor)

    def ours(self, settings: Any = None, **arguments: Any) -> Any:
        """APAI's configuration as the agent program builds it."""
        self._count += 1
        run = self.folder / f"run-{self._count}"
        return agent_config.build_apai_config(
            settings or self.settings, run, camera="link", **arguments
        )

    def theirs(
        self, speed_profile: str = "reference", depth_backend: str | None = None
    ) -> Any:
        """APAI's configuration as the experiment's launcher builds it.

        From its own files. Three of them hold paths of the Thor: the record
        of the mounting, the camera profile it names and the arm profile's
        chain. Each is read from this machine's copy, and only its paths are
        rewritten.
        """
        self._count += 1
        out = self.folder / f"theirs-{self._count}"
        out.mkdir()
        root = self.experiment_dir
        hand_eye = Path(agent_config.resolve_mounting(self.settings).hand_eye_json)
        # The camera profile of the fit; an exported fit has none of its own.
        profiles = [
            hand_eye.parent / "camera-candidate.yaml",
            *sorted(root.glob(".state/thor/mounting/*/camera-candidate.yaml")),
        ]
        chosen = next(path for path in profiles if path.is_file())
        profile = yaml.safe_load(chosen.read_text())
        thor = profile["extrinsics"]["hand_eye_json"].split("/.state/")[0]

        def local(text: str) -> str:
            return text.replace(thor, str(root))

        (out / "arm.yaml").write_text(
            local((root / ".state/thor/robot/arm.yaml").read_text())
        )
        operation = (root / "configs/thor_operation.yaml").read_text()
        (out / "thor_operation.yaml").write_text(
            operation.replace(f"{thor}/.state/thor/robot/arm", str(out / "arm"))
        )
        settings = yaml.safe_load((root / "configs/thor_runtime.yaml").read_text())
        settings["arm_profile"] = str(out / "thor_operation.yaml")
        (out / "thor_runtime.yaml").write_text(yaml.safe_dump(settings))
        profile = yaml.safe_load(local(yaml.safe_dump(profile)))
        profile["extrinsics"]["hand_eye_json"] = str(hand_eye)
        (out / "camera-candidate.yaml").write_text(yaml.safe_dump(profile))
        record = {
            "accepted_fit": True,
            "handeye_file": str(hand_eye),
            "candidate_camera_profile": str(out / "camera-candidate.yaml"),
            "sha256": hashlib.sha256(hand_eye.read_bytes()).hexdigest(),
        }
        (out / "latest.json").write_text(json.dumps(record))
        cfg, _ = self.theirs_module.load_config(
            out / "thor_runtime.yaml", out / "latest.json", depth_backend, speed_profile
        )
        return cfg


@pytest.fixture(scope="module")
def rig(
    experiment_dir: Path, tmp_path_factory: pytest.TempPathFactory, functional: Any
) -> Iterator[Rig]:
    folder = tmp_path_factory.mktemp("conform")
    program = folder / "env" / "codex" / "node_modules" / ".bin" / "codex"
    program.parent.mkdir(parents=True)
    program.write_text("#!/bin/sh\n")
    program.chmod(0o755)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
        patch.setenv("AGENTIC_REMOTE_CAMERA_PR", str(folder / "camera_pr"))
        patch.setenv("AGENTIC_REMOTE_ENV_DIR", str(folder / "env"))
        yield Rig(experiment_dir, folder, functional)


# The environment


def exported(script: str) -> set[str]:
    return set(re.findall(r"^\s*export ([A-Z_]+)", script, re.M))


@experiment
def test_run_agent_sets_no_apai_variable_as_the_experiments_launcher(
    experiment_dir: Path, tmp_path: Path
) -> None:
    theirs = (experiment_dir / "thor_runtime.sh").read_text()
    ours = RUN_AGENT.read_text()
    # The experiment's launcher sets nothing and takes LD_LIBRARY_PATH away.
    assert exported(theirs) == set() and "APAI_" not in theirs
    assert "env -u LD_LIBRARY_PATH" in theirs and "unset LD_LIBRARY_PATH" in ours
    # Every variable run_agent.sh sets is one a second machine needs; none is
    # APAI's, whose strict CUDA mode fails in the grasp code.
    assert exported(ours) == set(EXPORTED)
    assert not [name for name in exported(ours) if name.startswith("APAI_")]
    # What the program gets: APAI's own default, and the operator's own value.
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "${APAI_REQUIRE_CUDA-not set} ${LD_LIBRARY_PATH-not set}"\n'
    )
    fake.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "AGENTIC_REMOTE_AGENT_PYTHON": str(fake),
        "AGENTIC_REMOTE_APAI_SRC": str(tmp_path),
        "LD_LIBRARY_PATH": "/somewhere",
    }
    for given, got in ((None, "not set"), ("0", "0"), ("1", "1")):
        if given is not None:
            env["APAI_REQUIRE_CUDA"] = given
        done = subprocess.run(
            [str(RUN_AGENT)], check=False, env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == f"{got} not set"


@experiment
def test_the_workstation_script_types_the_experiment_launchers_command(
    experiment_dir: Path,
) -> None:
    script = RUN_WORKSTATION.read_text()
    lines = re.findall(r'^agent\+?="(.*)"$', script, re.M)
    command = " ".join(lines).replace("${prefix} ", "")
    assert command.startswith("./run_agent.sh --config ${agent_config} ")
    given = command.split()[3:]
    # The experiment's own documented command for a timed run on the rig.
    told = (experiment_dir / "THOR_TIMING.md").read_text()
    theirs = re.search(r"^\./thor_runtime\.sh (--operate .*)$", told, re.M)
    assert theirs is not None
    assert given[: len(theirs.group(1).split())] == theirs.group(1).split()
    # And the launcher's switch for one model request per camera frame.
    assert given[len(theirs.group(1).split()) :] == ["--batch-milestones"]
    # Every one of them is an argument of both programs.
    source = (experiment_dir / "thor_runtime.py").read_text()
    ours = {
        option
        for action in launcher._parser()._actions
        for option in action.option_strings
    }
    for option in (word for word in given if word.startswith("--")):
        assert f'"{option}"' in source and option in ours, option
    # The rate of the debug recording's map pictures is the agent program's
    # own argument: the launcher has none, and the script types none.
    assert "--debug-map-hz" in ours and "debug-map-hz" not in source
    assert "--debug-map-hz" not in script
    # So is what the debug recording does at its size.
    assert "--debug-at-limit" in ours and "debug-at-limit" not in source
    assert "--debug-at-limit" not in script
    # And what it stores while the program waits for a task.
    assert "--debug-idle-map-s" in ours and "debug-idle-map-s" not in source
    assert "--debug-idle-map-s" not in script


@experiment
def test_the_debug_recordings_limit_stands_on_the_experiments_recorder_and_viewer(
    experiment_dir: Path,
) -> None:
    from isaaccapture_examples.agentic_remote.agent import debug_recording

    # The default rate is what the viewer's page can show: it asks for the
    # state once per period.
    page = (experiment_dir / "thor_viewer.html").read_text()
    period_ms = re.search(r"setInterval\(tick,(\d+)\)", page)
    assert period_ms is not None
    assert debug_recording.DEBUG_MAP_HZ == 1000 / int(period_ms.group(1))
    # The size's default is the launcher's own.
    theirs = (experiment_dir / "thor_runtime.py").read_text()
    size = re.search(r'"--debug-max-gib",\s+type=float,\s+default=(\d+)', theirs)
    assert size is not None and float(size.group(1)) == agent_config.DEBUG_MAX_GIB
    # Every kind of event the experiment records is either one the limit may
    # leave out or one of these, which are stored whatever the rate.
    recorder = (experiment_dir / "thor_recording.py").read_text()
    routes = (experiment_dir / "thor_route_diagnostic.py").read_text()
    families = "map|mask|bridge|object|robot|safety|verification|session"
    kinds = set(re.findall(rf'"((?:{families})\.[a-z_.]+)"', recorder))
    kinds |= set(re.findall(r'recorder\.record\(\s*"([a-z_.]+)"', routes))
    always = {
        "session.start",
        "session.end",
        "robot.feedback",
        "robot.call",
        "robot.return",
        "safety.refusal",
        "verification.measurement",
        "route.segment.begin",
        "route.segment.end",
    }
    assert kinds - debug_recording.MAP_KINDS == always
    assert debug_recording.MAP_KINDS <= kinds
    assert {debug_recording.REFUSAL, *debug_recording.SEGMENTS} <= always
    assert launcher.STATE_EVENT not in debug_recording.MAP_KINDS
    # The fields the limit reads of them, as the recorder's hooks write them:
    # the map revision a refusal and a segment name, and the ids that join
    # the events of one refresh.
    for written in (
        '"map_id": id(obj.occupancy)',
        '"gate_refreshes": ctx.refreshes',
        '"map_id": id(obj)',
        '"refreshes": obj.refreshes',
        '"query_request_id": getattr(rec.local, "query_id", None)',
        'rec.emit("bridge.response", request_id=request_id, result=result)',
        "rec.local.refresh_id = rec.record(",
        '"refresh_id": getattr(self.local, "refresh_id", None)',
    ):
        assert written in recorder, written
    assert '"map": map_state(args[0] if args else kwargs["harness"])' in routes
    assert '"map_id": None if occupancy is None else id(occupancy)' in routes
    assert '"refreshes",' in routes and '"call_id": event_id' in routes
    # What the viewer shows as the map of a refusal: the newest commit before
    # it, compared by the revision the refusal's report names.
    viewer = (experiment_dir / "thor_viewer.py").read_text()
    assert 'report.get("gate_refreshes")' in viewer
    assert '== mapped["commit"]["data"]["refreshes"]' in viewer
    # What the line of a full recording is said for: the writer's own words.
    words = "recording size limit reached (conservative uncompressed reservation)"
    assert f'self.writer_error = "{words}"' in recorder
    assert debug_recording._SIZE_REASON in words
    assert "self.disabled = True" in recorder and "self.start_mono_ns" in recorder


@experiment
def test_the_debug_recordings_keeper_stands_on_the_experiments_writer_and_viewer(
    experiment_dir: Path,
) -> None:
    from isaaccapture_examples.agentic_remote.agent import debug_recording

    recorder = (experiment_dir / "thor_recording.py").read_text()
    writer = recorder[
        recorder.index("    def _writer(self):") : recorder.index(
            "    def _status(self, finalized):"
        )
    ]
    # Where the keeper stands: the recorder's thread takes an event from its
    # queue and then checks it against the size, by these words.
    taken = "row, arrays, size = self.events.get(timeout=0.1)"
    check = (
        f"if self.written_bytes + size + {debug_recording._CHECK_BYTES} "
        "> self.max_bytes:"
    )
    assert "self.events = queue.Queue(" in recorder
    assert writer.count(taken) == 1 == writer.count(check)
    assert writer.index(taken) < writer.index(check)
    # The written bytes are that thread's alone: it adds to them, the status
    # reads them, and nothing else of the recorder names them.
    assert recorder.count("self.written_bytes") == 5
    assert writer.count("self.written_bytes += ") == 2
    assert "self.seq = self.pending_bytes = self.written_bytes = 0" in recorder
    assert '"written_bytes": self.written_bytes,' in recorder
    # The file the keeper removes is the one the thread wrote, by the event's
    # id, and it wrote it before the event's row: a row whose arrays are not
    # there is one the keeper dropped.
    named = "rel = f\"arrays/{row['id']:09d}.npz\""
    assert named in writer and named.count("09d") == 1
    assert writer.index("self._write_arrays(path, arrays)") < writer.index(
        "stream.write(line)"
    )
    assert "self.written_bytes += path.stat().st_size" in writer
    # The marks of a tick: the recorder numbers its events one after another,
    # in the order its thread writes them, and each row names the thread that
    # recorded it.
    emit = recorder[recorder.index("    def emit(self, kind, **data):") :]
    emit = emit[: emit.index("    def record(self, kind, builder):")]
    numbered = "self.seq += 1\n                event_id = self.seq\n"
    assert emit.count(numbered) == 1 and recorder.count("self.seq +=") == 1
    assert emit.index("with self.lock:") < emit.index(numbered)
    assert emit.index(numbered) < emit.index("self.events.put_nowait(")
    assert '"id": event_id,' in emit
    assert '"thread": threading.current_thread().name,' in emit
    # The status: the names of its file, what makes it "complete", and that
    # the thread writes the last one itself, also after it stopped.
    assert f'temp = self.path / "{debug_recording._STATUS_PARTIAL}"' in recorder
    assert f'temp.replace(self.path / "{debug_recording.STATUS_FILE}")' in recorder
    assert re.search(
        r'"finalized": finalized,\s+"complete": bool\(\s+finalized\b', recorder
    )
    assert "self._status(self.stop.is_set() and self.events.empty())" in writer
    assert "self.start_wall_ns" in recorder
    # The viewer. Its index reads rows only; the state of a time reads the
    # arrays of the bridge's answer, the picture those of the camera frame
    # and the map's input. So the pictures can go first.
    viewer = (experiment_dir / "thor_viewer.py").read_text()
    index = viewer[
        viewer.index("    def consume(self, row):") : viewer.index("    def load(")
    ]
    assert "load(" not in index and "arrays" not in index
    assert 'query = self.load(mapped["query"]).get("result", {})' in viewer
    assert 'frame = self.load(mapped["frame"])["frame"]' in viewer
    assert 'inp = self.load(mapped["input"])' in viewer
    pictures = debug_recording.PICTURE_KINDS
    assert {"map.frame", "map.depth_input"} <= pictures < debug_recording.MAP_KINDS
    assert "bridge.response" not in pictures and "mask.pose" not in pictures
    # A file that is not there is an error of that one request, by its name.
    assert "except (ValueError, FileNotFoundError, KeyError) as error:" in viewer
    assert "self.send_error(400, str(error))" in viewer
    # The page: a failed state leaves what is shown and its time, so "Play
    # recording" would stand still at a time without state; "Follow live"
    # asks for the newest again.
    page = (experiment_dir / "thor_viewer.html").read_text()
    assert "if(!res.ok)throw Error(await res.text());" in page
    assert "render();$('error').textContent='';if(playing){at=Math.min(" in page
    assert "catch(e){$('error').textContent=String(e);}" in page
    assert "follow=true;playing=false;at=null;initial=false;imageKey='';" in page
    assert "$('photo').src='/api/image?'" in page


# APAI's configuration


@apai
@experiment
def test_the_configuration_is_the_experiment_launchers_key_by_key(rig: Rig) -> None:
    theirs = rig.theirs().as_dict()
    # With the Thor's limit at the launcher's speed nothing else differs.
    speed = rig.theirs_module.load_joint_speed("reference")
    ours = rig.ours(rig.with_limit(speed)).as_dict()
    assert differing(theirs, ours) == set(DIFFERS)
    assert len(flat(theirs)) > 1000  # the whole configuration was compared
    # The shipped rig file: one more key, held down by the Thor's own limit.
    shipped = rig.ours().as_dict()
    limit = rig.settings.thor.agent_max_joint_speed
    assert differing(theirs, shipped) - set(DIFFERS) == (
        {THOR_LIMIT} if limit < speed else set()
    )
    assert shipped["safety"]["max_joint_vel"] == min(speed, limit)
    # What the model reads of the hardware: the arm by its name, the camera's
    # place beside the arm.
    assert ours["arm"]["name"] == theirs["arm"]["name"]
    placed = theirs["camera"]["extrinsics"]["mode"]
    assert placed and ours["camera"]["extrinsics"]["mode"] == placed
    for key in agent_config.CAMERA_KEYS:
        assert ours["camera"][key] == theirs["camera"][key], key
    for key in ("home_q", "park_q", "motion_bounds", "body_mask"):
        assert ours["arm"][key] == theirs["arm"][key], key


@apai
@experiment
def test_the_model_is_told_the_same_hardware(rig: Rig) -> None:
    prompts = importlib.import_module("apai.agent.prompts")
    speed = rig.theirs_module.load_joint_speed("reference")
    their_cfg, our_cfg = rig.theirs(), rig.ours(rig.with_limit(speed))
    theirs = prompts.build_system_prompt(their_cfg)
    ours = prompts.build_system_prompt(our_cfg)
    # The arm by its name and the camera's place, not a run folder's path.
    assert f"Arm {their_cfg.arm.name}:" in theirs
    assert f"Camera {their_cfg.camera.name}: " in theirs
    assert their_cfg.camera.extrinsics.mode in theirs
    assert str(rig.folder) not in ours
    # One word differs: how the arm is reached.
    assert ours != theirs
    assert ours == theirs.replace(
        f"transport {their_cfg.arm.type}", f"transport {our_cfg.arm.type}"
    )


@apai
@experiment
def test_the_joint_speed_is_the_experiment_launchers_speed_profile(rig: Rig) -> None:
    load = rig.theirs_module.load_joint_speed
    for profile in agent_config.SPEED_PROFILES:
        assert agent_config.functional_joint_speed(profile) == load(profile)
    assert agent_config.CONSERVATIVE_SPEED == load(agent_config.SPEED_CONSERVATIVE)
    # Another name is refused by both.
    with pytest.raises(ValueError):
        load("fast")
    with pytest.raises(ConfigError, match="--speed-profile must be one of"):
        agent_config.functional_joint_speed("fast")
    # The default is the launcher's: its own argument's default.
    default = re.search(
        r'"--speed-profile",.*?default="(\w+)"',
        inspect.getsource(rig.theirs_module.main),
        re.S,
    )
    assert default is not None and default.group(1) == agent_config.SPEED_REFERENCE
    free = rig.with_limit(3.0)
    assert agent_config.joint_speed(free) == load("reference")
    for profile in agent_config.SPEED_PROFILES:
        assert agent_config.joint_speed(free, profile) == load(profile)
        cfg = rig.ours(free, speed_profile=profile)
        assert cfg.safety.max_joint_vel == rig.theirs(profile).safety.max_joint_vel
    # Never above the Thor's own limit, which refuses a faster MOVE.
    low = load("conservative") / 2
    assert agent_config.joint_speed(rig.with_limit(low)) == low
    # The task gate's question names the speed of the configuration.
    cfg = rig.ours(free)
    assert agent_config.task_settings(free, cfg).max_joint_speed == load("reference")


@apai
@experiment
def test_the_stereo_matcher_follows_the_argument_as_in_the_experiment(
    rig: Rig,
) -> None:
    matchers = importlib.import_module("thor_stereo").BACKENDS
    default = rig.theirs().camera.depth_backend
    assert rig.ours().camera.depth_backend == default
    other = next(name for name in matchers if name != default)
    assert rig.ours(depth_backend=other).camera.depth_backend == other
    assert rig.theirs(depth_backend=other).camera.depth_backend == other
    with pytest.raises(ValueError):
        rig.theirs(depth_backend="none-such")
    with pytest.raises(ConfigError, match="'none-such' is not one of"):
        rig.ours(depth_backend="none-such")
    # The argument is taken as the launcher's: any name, checked with its list.
    args = launcher._parser().parse_args(["--config", "x", "--depth-backend", other])
    assert launcher._config_arguments(args)["depth_backend"] == other


@apai
@experiment
def test_a_move_waits_for_the_obstacle_grid_as_the_experiments_route_check(
    rig: Rig,
) -> None:
    motion = importlib.import_module("thor_motion")
    shipped = rig.ours()
    # Another age limit of the grid than the file's own value would fit.
    data = shipped.as_dict()
    data["occupancy"]["max_age_s"] = 6.0 * rig.settings.move.start_grid_age_s
    cfg = importlib.import_module("apai.config").Cfg(data)
    arm = agent_config.arm_settings(rig.settings, "id", cfg)
    assert arm.start_grid_wait_s == motion.timing_settings(cfg)[0]
    assert arm.start_grid_wait_s != rig.settings.move.start_grid_wait_s
    assert arm.start_grid_age_s != rig.settings.move.start_grid_age_s

    # The oldest grid the experiment's route check starts on is the same age.
    @contextmanager
    def held(harness: Any, deadline: float | None = None) -> Iterator[None]:
        yield

    def starts_on(age_s: float) -> bool:
        grid = SimpleNamespace(
            required=True,
            max_age_s=cfg.occupancy.max_age_s,
            _last_refresh=time.monotonic() - age_s,
            _check_required_age=lambda: None,
        )
        harness = SimpleNamespace(occupancy=grid)
        try:
            with motion.fresh_geometry_guard(
                held, harness, deadline=time.monotonic() + 0.05
            ):
                return True
        except Exception:
            return False

    assert starts_on(arm.start_grid_age_s - 0.1)
    assert not starts_on(arm.start_grid_age_s + 0.1)
    # The minimal stack keeps the file's two values.
    minimal = dataclasses.replace(
        rig.settings,
        experiment=dataclasses.replace(rig.settings.experiment, stack="minimal"),
    )
    kept = agent_config.arm_settings(minimal, "id", cfg)
    assert (kept.start_grid_age_s, kept.start_grid_wait_s) == (
        minimal.move.start_grid_age_s,
        minimal.move.start_grid_wait_s,
    )


@apai
@experiment
def test_no_run_without_motion_bounds_as_in_the_experiment(rig: Rig) -> None:
    data = rig.ours().as_dict()
    views = [data, *(arm["resolved"] for arm in data["arms"])]
    agent_config._refuse(data, views, rig.settings)
    del data["arm"]["motion_bounds"]
    with pytest.raises(ConfigError, match="needs motion_bounds"):
        agent_config._refuse(data, views, rig.settings)
    # The launcher's own words for the same file.
    assert "requires explicit motion_bounds" in inspect.getsource(
        rig.theirs_module.load_config
    )
    data["arm"]["motion_bounds"] = {"joint_center_rad": [0.0]}
    with pytest.raises(ConfigError, match="the experiment's check"):
        agent_config._refuse(data, views, rig.settings)


@apai
@experiment
def test_the_agent_plans_at_the_lowest_of_the_launchers_speed_the_key_and_the_thors(
    rig: Rig,
) -> None:
    """The Thor's own agent speed limit is the one its WELCOME names."""
    speed = rig.theirs_module.load_joint_speed("reference")
    slow = rig.theirs_module.load_joint_speed("conservative")
    # The owner's word of 2026-11-16 for the rig: the agent runs at 0.2 rad/s.
    key = rig.settings.thor.agent_max_joint_speed
    assert key == 0.2 < speed

    def planned(settings: Any, thor: float | None, profile: str | None = None) -> Any:
        built = rig.ours(settings, speed_profile=profile)
        agreed, cfg, line = agent_config.agree_joint_speed(
            settings, built, thor, profile
        )
        data = cfg.as_dict()
        speeds = {data["safety"]["max_joint_vel"]}
        speeds |= {arm["resolved"]["safety"]["max_joint_vel"] for arm in data["arms"]}
        (one,) = speeds  # every view of the configuration holds the same speed
        # What the arm adapter takes for the Thor's limit, and the task gate
        # for its question, is that speed.
        assert agent_config.arm_settings(agreed, "id", cfg).agent_max_joint_speed == one
        assert agent_config.task_settings(agreed, cfg).max_joint_speed == one
        assert line.startswith(f"joint speed: the agent plans at {one:g} rad/s, ")
        return one, line

    # The shipped file against a Thor that names the same limit: the key.
    one, line = planned(rig.settings, key)
    assert one == key
    assert line.endswith(
        f"the lowest of {speed:g} (--speed-profile reference) and {key:g} "
        f"(thor.agent_max_joint_speed); the Thor's own agent speed limit is {key:g} "
        "(limits.agent.max_joint_speed, from the robot app's WELCOME)"
    )
    # Each of the three is the lowest in turn.
    free = rig.with_limit(2 * speed)
    assert planned(free, 2 * speed)[0] == speed
    assert planned(free, speed / 4)[0] == speed / 4
    assert planned(rig.with_limit(speed / 8), speed / 4)[0] == speed / 8
    assert planned(free, 2 * speed, "conservative")[0] == slow
    # A robot app from before its WELCOME named the limit: the key stands.
    one, line = planned(rig.with_limit(speed / 2), None)
    assert one == speed / 2
    assert line.endswith(
        "the robot app's WELCOME names no agent speed limit: that key stands for it"
    )
    assert agent_config.thor_agent_limit(SimpleNamespace()) is None
    assert agent_config.thor_agent_limit(SimpleNamespace(agent_max_joint_speed=0.2))


@apai
@experiment
def test_the_start_names_the_arm_and_the_hand_eye_file(rig: Rig) -> None:
    args = launcher._parser().parse_args(["--config", "x"])
    lines = launcher._start_lines(rig.settings, rig.ours(), args)
    # The joint speed is said once the robot app's WELCOME is here.
    assert not [line for line in lines if "speed" in line]
    found = agent_config.resolve_mounting(rig.settings)
    assert any(line.startswith(f"arm {rig.theirs().arm.name};") for line in lines)
    assert any(
        found.hand_eye_json in line and found.sha256[:12] in line for line in lines
    )
    # The minimal stack takes nothing of the experiment and says nothing of it.
    minimal = dataclasses.replace(
        rig.settings,
        experiment=dataclasses.replace(rig.settings.experiment, stack="minimal"),
    )
    assert launcher._start_lines(minimal, rig.ours(), args) == []


@apai
def test_apais_strict_cuda_mode_in_the_operators_shell_is_said_at_the_start(
    stage: Stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert stage.run("--task", LOOK) == launcher.EXIT_OK
    assert not stage.said("APAI_REQUIRE_CUDA")
    stage.lines.clear()
    monkeypatch.setenv("APAI_REQUIRE_CUDA", "1")
    assert stage.run("--task", LOOK) == launcher.EXIT_OK
    (warning,) = stage.said("APAI_REQUIRE_CUDA")
    assert warning.startswith("WARNING: APAI_REQUIRE_CUDA is 1 in the environment")


# The mounting record


def record_tree(folder: Path, stamp: str, checked: str, **changes: Any) -> Path:
    """A fit <stamp> with its hand-eye file and a record that names Thor paths."""
    fit = folder / ".state" / "thor" / "mounting" / stamp
    fit.mkdir(parents=True)
    (fit / "handeye.json").write_text(json.dumps({"mode": "eye_to_hand", "id": stamp}))
    thor = "/home/thor-user/experiment"
    record = {
        "checked_at_utc": checked,
        "accepted_fit": True,
        "handeye_file": f"{thor}/.state/thor/mounting/{stamp}/handeye.json",
        "candidate_camera_profile": f"{thor}/.state/thor/mounting/{stamp}/c.yaml",
        "sha256": hashlib.sha256((fit / "handeye.json").read_bytes()).hexdigest(),
        **changes,
    }
    path = fit / agent_config.MOUNTING_EXPORTED
    path.write_text(json.dumps(record))
    return path


def full_settings(folder: Path, **camera: Any) -> Any:
    """Full-stack settings in a temporary folder."""
    data = {
        "state_dir": str(folder / "state"),
        "camera": {"calibration_dir": str(folder), **camera},
        "experiment": {"stack": "full"},
    }
    path = folder / "agent.yaml"
    path.write_text(yaml.safe_dump(data))
    return agent_config.load_settings(path)


@apai
@experiment
def test_the_hand_eye_file_follows_the_mounting_record_as_in_the_experiment(
    functional: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "experiment"
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(root))
    old = record_tree(root, "20260101T000000Z", "2026-01-01T00:00:00+00:00")
    latest = root / agent_config.MOUNTING_DIR / agent_config.MOUNTING_LATEST
    old = old.rename(latest)  # the fit the launcher's own record names
    hand_eye = latest.parent / "20260101T000000Z" / "handeye.json"
    settings = full_settings(tmp_path)
    # The launcher's own record, whose paths are the Thor's.
    found = agent_config.resolve_mounting(settings)
    assert found.record == str(latest)
    assert found.hand_eye_json == str(hand_eye)
    assert found.sha256 == json.loads(latest.read_text())["sha256"]
    # A new fit is in use once the Thor's export of it is unpacked.
    new = record_tree(root, "20260202T000000Z", "2026-02-02T00:00:00+00:00")
    found = agent_config.resolve_mounting(settings)
    assert found.record == str(new)
    assert found.hand_eye_json == str(new.parent / "handeye.json")
    name = agent_config.calibration_id
    (tmp_path / "manifest.json").write_text("{}")
    assert name(settings, "link").endswith(f"+mount-{found.sha256[:12]}")
    # The file's own key is used in the record's place; --mounting before both.
    named = full_settings(tmp_path, hand_eye_json=str(hand_eye))
    assert agent_config.resolve_mounting(named).record == ""
    assert agent_config.resolve_mounting(named).hand_eye_json == str(hand_eye)
    assert agent_config.resolve_mounting(named, new).record == str(new)

    # What the launcher refuses is refused, and what it takes is taken.
    def theirs(record: Path) -> str:
        """Why the launcher's load_config ends on this record, with local paths."""
        data = json.loads(record.read_text())
        data["handeye_file"] = str(record.parent / "handeye.json")
        local = tmp_path / "local.json"
        local.write_text(json.dumps(data))
        own = functional.ROOT / "configs" / "thor_runtime.yaml"
        with pytest.raises(Exception) as ended:  # it has no camera profile
            functional.load_config(own, local)
        return f"{type(ended.value).__name__}: {ended.value}"

    refusal = "ValueError: Mounting record is not accepted or its hand-eye"
    assert not theirs(new).startswith(refusal)  # it got past the record
    assert agent_config.resolve_mounting(settings, new).record == str(new)
    bad = [
        record_tree(root, "3", "2026-03-03T00:00:00+00:00", accepted_fit=False),
        record_tree(root, "4", "2026-03-04T00:00:00+00:00", accepted_fit="yes"),
        record_tree(root, "5", "2026-03-05T00:00:00+00:00", sha256="0" * 64),
    ]
    for record in bad:
        assert theirs(record).startswith(refusal)
        with pytest.raises(ConfigError, match="is not accepted or its hand-eye"):
            agent_config.resolve_mounting(settings, record)
    # The newest record is a refused one: the program does not fall back.
    with pytest.raises(ConfigError, match="is not accepted or its hand-eye"):
        agent_config.resolve_mounting(settings)
    # Without any record the key is asked for.
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="no mounting record in"):
        agent_config.resolve_mounting(settings)


@experiment
def test_the_experiments_own_record_is_read_from_this_machines_copy(
    experiment_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(EXPERIMENT_DIR_ENV, str(experiment_dir))
    record = agent_config.mounting_record(experiment_dir)
    theirs = json.loads(record.read_text())  # its paths may be the Thor's
    found = agent_config.resolve_mounting(full_settings(tmp_path))
    assert found.record == str(record) and found.sha256 == theirs["sha256"]
    assert Path(found.hand_eye_json).is_relative_to(experiment_dir)
    assert Path(found.hand_eye_json).name == Path(theirs["handeye_file"]).name


# The arm profile


@experiment
def test_the_arm_profile_is_the_experiment_launchers_without_the_motor_bus(
    experiment_dir: Path,
) -> None:
    profile = agent_config.experiment_arm_profile(experiment_dir)
    named = yaml.safe_load((experiment_dir / "configs/thor_runtime.yaml").read_text())
    theirs = yaml.safe_load((experiment_dir / named["arm_profile"]).read_text())
    local = yaml.safe_load((experiment_dir / ".state/thor/robot/arm.yaml").read_text())
    for key, value in theirs.items():
        if key != "extends":
            assert profile[key] == value, key
    assert profile["name"] == local["name"]
    assert set(agent_config.MOTOR_KEYS) <= set(local)
    assert not set(agent_config.MOTOR_KEYS) & set(profile)
    # It extends the profile the Thor's arm file extends, on this machine.
    assert Path(local["extends"]).name == Path(profile["extends"]).name
    assert Path(profile["extends"] + ".yaml").is_file()


# What the model reads


def stand_in_parts(functional: Any, bridge: Any) -> dict[str, Any]:
    """The experiment's hooks as stand-ins, with its launcher's own two parts."""

    @contextmanager
    def nothing(*args: Any, **kwargs: Any) -> Iterator[None]:
        yield

    return {
        "Admission": functional.Admission,
        "save_frame": functional.save_frame,
        "OwnedOccupancyBridge": lambda *args, **kwargs: bridge,
        "install_verification": lambda runtime, output: None,
        "register_detector": lambda: None,
        "motion_adapters": nothing,
        "perception_adapters": nothing,
        "placement_adapters": nothing,
        "target_map_adapters": nothing,
        "colors": SimpleNamespace(mask_color=None),
        "mask_color": None,
        "guidance": "",
    }


class Start:
    """agent_hooks of the full stack on stand-ins, up to APAI's built runtime."""

    def __init__(
        self, rig: Rig, run: Path, monkeypatch: pytest.MonkeyPatch, camera: Any
    ) -> None:
        self.demo = importlib.import_module("apai.apps.demo")
        self.rig, self.run, self.cfg = rig, run, rig.ours()
        self.said: list[str] = []
        self.torn_down: list[Any] = []
        grid = SimpleNamespace(
            status={"backend": "stand-in"},
            refreshes=0,
            report=lambda: {"detail": "no camera frame yet"},
            wait_until_usable=lambda timeout_s: False,
        )
        self.runtime = SimpleNamespace(
            arm=SimpleNamespace(raw="adapter", harness=SimpleNamespace(occupancy=grid)),
            camera=camera,
            execute=lambda name, args: {"ok": True, "tool": name},
        )
        bridge = SimpleNamespace(port=1, verify=lambda status: None)
        bridge.start, bridge.close = (lambda: bridge), (lambda: None)
        parts = stand_in_parts(rig.theirs_module, bridge)
        cloud = SimpleNamespace(make_client=lambda config: None)
        built = (self.runtime, "raw")
        monkeypatch.setattr(self.demo, "build_runtime", lambda *a, **k: built)
        monkeypatch.setattr(
            self.demo, "shutdown_runtime", lambda *a: self.torn_down.append(a)
        )
        monkeypatch.setattr(launcher, "_experiment_parts", lambda: parts)
        monkeypatch.setattr(launcher, "experiment_module", lambda name: cloud)
        self.gate = mock.Mock()
        self.gate.execute = lambda original, name, args: original(name, args)

    def build(self) -> Any:
        """Run the hooks' build of the runtime; returns what APAI would get."""
        run, settings = self.run, self.rig.settings
        run.mkdir(exist_ok=True)
        hooks = launcher.agent_hooks(
            self.cfg, settings, mock.Mock(), self.gate, run, say=self.said.append
        )
        with hooks:
            return self.demo.build_runtime(self.cfg, run)


@apai
@experiment
def test_the_start_saves_a_camera_frame_and_goes_on_without_a_grid(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    frame_type = importlib.import_module("apai.types").Frame
    frame = frame_type(
        np.zeros((4, 6, 3), np.uint8), np.ones((4, 6), np.float32), np.eye(3)
    )
    camera = SimpleNamespace(get_fresh_frame=lambda timeout_s: frame)
    start = Start(rig, tmp_path / "run", monkeypatch, camera)
    made, _ = start.build()
    run, cfg, theirs = start.run, start.cfg, rig.theirs_module
    assert made is start.runtime and start.torn_down == []
    # The experiment launcher's start, by its own save_frame: one camera frame
    # saved, and what was found.
    for name in ("startup-rgbd.npz", "startup-rgb.jpg", "startup-depth-m.npy"):
        assert (run / name).is_file(), name
    status = json.loads((run / launcher.STATUS_FILE).read_text())
    assert status["startup"]["shape"] == [4, 6, 3]
    assert status["occupancy_backend"] == {"backend": "stand-in"}
    assert "startup_error" not in status
    hand_eye = Path(cfg.camera.extrinsics.hand_eye_json)
    assert status["mounting"]["mounting_calibration_sha256"] == (
        hashlib.sha256(hand_eye.read_bytes()).hexdigest()
    )
    # Its bounds are said once, in its own words, and a grid that is not
    # usable yet does not end the program: the launcher starts without one.
    own = theirs.Admission(
        True, run, cfg.arm.motion_bounds, cfg.safety.max_joint_vel, "task"
    )
    assert own.motion_description() in start.said
    assert any("the obstacle grid is not usable yet" in line for line in start.said)
    assert any(line.startswith("ready.") for line in start.said)
    # A tool goes through the task gate, which admits it, and nothing else.
    assert made.execute("task_done", {}) == {"ok": True, "tool": "task_done"}


@apai
@experiment
def test_a_camera_that_gives_no_frame_ends_the_start_as_in_the_experiment(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_frame(timeout_s: float) -> Any:
        raise TimeoutError("no fresh camera frame")

    # The launcher's own wait with its clock moved on: its deadline has passed.
    theirs = rig.theirs_module
    clock = iter([0.0, 0.0])
    late = SimpleNamespace(monotonic=lambda: next(clock, 3600.0))
    monkeypatch.setattr(theirs, "time", late)
    camera = SimpleNamespace(get_fresh_frame=no_frame)
    with pytest.raises(TimeoutError):
        theirs.save_frame(SimpleNamespace(camera=camera), tmp_path)
    clock = iter([0.0, 0.0])
    start = Start(rig, tmp_path / "run", monkeypatch, camera)
    with pytest.raises(TimeoutError, match="no fresh camera frame"):
        start.build()
    # The runtime is torn down, and the run folder says why the start failed.
    assert len(start.torn_down) == 1
    status = json.loads((start.run / launcher.STATUS_FILE).read_text())
    assert status["startup_error"] == {
        "type": "TimeoutError",
        "message": "no fresh camera frame",
    }
    assert not any(line.startswith("ready.") for line in start.said)


@experiment
def test_a_task_carries_the_experiment_launchers_texts(
    functional: Any, tmp_path: Path
) -> None:
    source = inspect.getsource(functional.main)
    assert f'default="{launcher.DEFAULT_TASK}"' in source
    assert f'+ "{launcher.DEPTH_TASK_NOTE}"' in source
    full = full_settings(tmp_path)
    parser = launcher._parser()

    def arguments(*given: str) -> list[str]:
        args = parser.parse_args(["--config", "x", *given])
        return launcher.task_arguments(args, full, tmp_path)

    def value(found: list[str], option: str) -> str:
        return found[found.index(option) + 1]

    # A run without a task looks and does not move; a named task gets the note.
    task = value(arguments(), "--task")
    assert task == launcher.DEFAULT_TASK + launcher.DEPTH_TASK_NOTE
    assert value(arguments("--task", "pick the cup"), "--task") == (
        "pick the cup" + launcher.DEPTH_TASK_NOTE
    )
    # APAI's start line names the profiles of this run, not its own defaults.
    found = arguments("--interactive")
    assert value(found, "--camera") == str(tmp_path / "camera_profile")
    assert value(found, "--arm") == str(tmp_path / "arm_profile")
    assert found[-1] == "--interactive"
    # Every argument the launcher passes to APAI is passed here.
    given = source[source.index("arguments = [") :]
    passed = set(re.findall(r'"(--[a-z-]+)",\n', given))
    assert passed and passed <= set(found)


@apai
@experiment
def test_the_task_gate_gets_the_speed_and_no_count_of_failed_steps(
    rig: Rig, tmp_path: Path
) -> None:
    cfg = rig.ours()

    def failing(name: str, args: dict) -> dict:
        return {"ok": False, "error": "missed"}

    # The launcher's Admission passes forty failures in a row as they are: it
    # has no rule on failed steps, APAI ends a task at its step limit.
    own = rig.theirs_module.Admission(
        True, tmp_path, cfg.arm.motion_bounds, cfg.safety.max_joint_vel, "task"
    )
    steps = 40
    passed = [own.execute(failing, "task_done", {}) for _ in range(steps)]
    assert [result.get("outcome") for result in passed] == [None] * steps
    # So the full stack's task block names a count no task reaches, and the
    # speed of the configuration; everything else is the file's.
    task = agent_config.task_settings(rig.settings, cfg)
    assert task.failed_steps_limit > 1000 * steps
    assert task.max_joint_speed == cfg.safety.max_joint_vel
    assert dataclasses.replace(
        task,
        failed_steps_limit=rig.settings.task.failed_steps_limit,
        max_joint_speed=rig.settings.task.max_joint_speed,
    ) == (rig.settings.task)
    minimal = dataclasses.replace(
        rig.settings,
        experiment=dataclasses.replace(rig.settings.experiment, stack="minimal"),
    )
    assert agent_config.task_settings(minimal, cfg) is minimal.task


@apai
def test_the_task_gate_is_built_with_the_settings_and_their_task_block(
    stage: Stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    given: list[Any] = []
    real = launcher.TaskGate

    def gate(link: Any, arm: Any, settings: Any, **kwargs: Any) -> Any:
        given.append(settings)
        return real(link, arm, settings, **kwargs)

    monkeypatch.setattr(launcher, "TaskGate", gate)
    assert stage.run("--task", LOOK) == launcher.EXIT_OK
    (settings,) = given
    # The gate reads the stack and the task block of what it is given.
    assert settings.minimal is True
    assert settings.task.max_joint_speed == test_launcher.SPEED


# The learned stores


@apai
@experiment
def test_the_learned_stores_start_from_where_the_experiment_keeps_them(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    sources = agent_config.learned_store_sources()
    # Where the launcher's run keeps them: APAI's own defaults, and its file.
    runtime = importlib.import_module("apai.skills.runtime")
    apai_source = inspect.getsource(runtime.SkillRuntime.__init__)
    places = dict(agent_config.LEARNED_STORES)
    for leaf in ("envelope.json", "grasp_memory.json"):
        assert f'"{places[leaf]}"' in apai_source, leaf
        assert sources[leaf] == home / ".apai" / leaf
    assert str(sources["experience.json"]) == rig.theirs().memory.experience_path
    # This program keeps its own, outside the experiment folder and the home.
    settings = dataclasses.replace(rig.settings, state_dir=tmp_path / "state")
    env: dict[str, str] = {}
    store = agent_config.redirect_apai_stores(settings, env)
    ours = {
        "envelope.json": Path(env["APAI_ENVELOPE_PATH"]),
        "grasp_memory.json": Path(env["APAI_GRASP_MEMORY_PATH"]),
        "experience.json": Path(rig.ours(settings).memory.experience_path),
    }
    assert set(ours) == set(sources)
    for leaf, path in ours.items():
        assert path == store / leaf and path != sources[leaf]
    # Nothing to start from: each store starts empty, and the line says where
    # the Thor's record goes.
    lines = agent_config.seed_learned_stores(settings, env)
    assert len(lines) == 3 and all("starts empty" in line for line in lines)
    assert not any(path.exists() for path in ours.values())
    # The launcher's stores on this machine: each is taken once.
    (home / ".apai").mkdir(parents=True)
    sources["envelope.json"].write_text('{"from": "the experiment"}')
    sources["grasp_memory.json"].write_text('{"wins": 3}')
    lines = agent_config.seed_learned_stores(settings, env)
    assert ours["envelope.json"].read_text() == '{"from": "the experiment"}'
    assert ours["grasp_memory.json"].read_text() == '{"wins": 3}'
    assert sum("started from" in line for line in lines) == 2
    # A store this program has written is never replaced.
    ours["envelope.json"].write_text('{"from": "this program"}')
    agent_config.seed_learned_stores(settings, env)
    assert ours["envelope.json"].read_text() == '{"from": "this program"}'
    assert sources["envelope.json"].read_text() == '{"from": "the experiment"}'
    # The minimal stack has none of this.
    minimal = dataclasses.replace(
        settings,
        experiment=dataclasses.replace(settings.experiment, stack="minimal"),
    )
    assert agent_config.seed_learned_stores(minimal, env) == []


# The start


@apai
@experiment
def test_operate_without_the_operators_terminal_is_refused_at_the_start(
    functional: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The launcher: exit code 2 before anything is loaded.
    never_loaded = mock.Mock(side_effect=AssertionError)
    monkeypatch.setattr(functional, "load_config", never_loaded)
    with pytest.raises(SystemExit) as theirs:
        functional.main(["--operate"])
    assert theirs.value.code == 2
    assert "--operate requires the operator's interactive terminal" in (
        capsys.readouterr().err
    )

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("attached to the robot link service")

    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "state_dir": str(tmp_path / "state"),
                "experiment": {"stack": "minimal"},
                "camera": {"allow_invented_times": True},
            }
        )
    )
    arguments = ["--config", str(path), "--camera", "mock", "--task", "look"]
    code = launcher.main([*arguments, "--operate"], attach=never)
    assert code == launcher.EXIT_NOT_RUN
    assert "--operate requires the operator's interactive terminal" in (
        capsys.readouterr().err
    )
    assert not (tmp_path / "state").exists()


# What the launcher adds for the operator


def test_a_skills_type_error_leaves_its_traceback_in_the_run_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Skills:
        def skill_grasp(self, name: str) -> str:
            return max(name, 3)  # type: ignore[call-overload]

        def skill_look(self) -> str:
            return "seen"

        def helper(self) -> None:
            raise TypeError("not a skill")

    monkeypatch.setattr(launcher, "_skill_runtime", lambda: Skills)
    traced = dict(launcher._skills_with_traceback(tmp_path))
    assert set(traced) == {"skill_grasp", "skill_look"}
    assert traced["skill_look"](Skills()) == "seen"
    assert not (tmp_path / launcher.SKILL_ERRORS).exists()
    # The error goes on to APAI as it was; its traceback is kept.
    with pytest.raises(TypeError, match="not supported between"):
        traced["skill_grasp"](Skills(), "cup")
    kept = (tmp_path / launcher.SKILL_ERRORS).read_text()
    assert kept.startswith("--- skill_grasp\nTraceback (most recent call last):")
    assert "return max(name, 3)" in kept


@apai
def test_the_skills_are_traced_while_apai_runs_and_its_own_afterwards(
    tmp_path: Path,
) -> None:
    runtime = importlib.import_module("apai.skills.runtime").SkillRuntime
    before = {
        name: skill
        for name, skill in vars(runtime).items()
        if name.startswith("skill_")
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"experiment": {"stack": "minimal"}}))
    minimal = agent_config.load_settings(path)
    gate = mock.Mock()
    with launcher.agent_hooks({}, minimal, mock.Mock(), gate, tmp_path):
        during = {name: vars(runtime)[name] for name in before}
    assert before and all(
        inspect.isfunction(skill) is (during[name] is not skill)
        for name, skill in before.items()
    )
    for name, skill in before.items():
        if inspect.isfunction(skill):
            assert during[name].__wrapped__ is skill
        assert vars(runtime)[name] is skill


def test_the_task_prompt_is_the_last_line_while_the_program_waits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    shown: list[str] = []

    def background() -> None:
        time.sleep(0.2)
        print("[agent] learned store envelope.json: started")  # behind the prompt
        time.sleep(launcher.PROMPT_QUIET_S + 0.5)
        shown.append(capsys.readouterr().out)
        sys.stderr.write("a warning of another thread\n")
        time.sleep(launcher.PROMPT_QUIET_S + 0.5)
        shown.append(capsys.readouterr().out)
        os.write(write_end, b"pick the cup\n")

    thread = threading.Thread(target=background)
    thread.start()
    try:
        line = launcher.terminal_input("task> ")
    finally:
        thread.join()
        os.close(write_end)
        stdin.close()
    assert line == "pick the cup"
    # The prompt, the line that pushed it up, and the prompt again as the last
    # line; and again behind a line on the other stream.
    assert shown[0] == "task> [agent] learned store envelope.json: started\ntask> "
    assert shown[1] == "task> "
    # Nothing is written again once nothing new came.
    assert capsys.readouterr().out == ""
    assert sys.stdin is stdin  # the streams are the program's own again


def test_the_prompt_comes_back_once_other_output_is_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import logging

    class Console(logging.Handler):
        """In place of the console's handler; it writes nothing."""

        def emit(self, record: logging.LogRecord) -> None:
            pass

    logger = logging.getLogger("isaaccapture.examples.agentic_remote")
    console = Console()
    logger.addHandler(console)
    keeper = launcher.PromptKeeper("task> ")
    out, err = sys.stdout, sys.stderr
    try:
        with keeper:
            assert sys.stdout is not out and sys.stderr is not err
            assert keeper.restore() is False  # nothing was written behind it
            sys.stdout.write("half a line")
            assert keeper.restore() is False  # not quiet yet
            time.sleep(launcher.PROMPT_QUIET_S + 0.05)
            assert keeper.restore() is True and keeper.restore() is False
            # The prompt begins a line of its own.
            assert capsys.readouterr().out == "half a line\ntask> "
            # A log line of the console's handler counts as output too.
            logger.warning("the camera link is late")
            time.sleep(launcher.PROMPT_QUIET_S + 0.05)
            assert keeper.restore() is True
            assert capsys.readouterr().out == "task> "
        assert keeper not in console.filters
    finally:
        logger.removeHandler(console)
    assert sys.stdout is out and sys.stderr is err
    assert keeper.restored == 2
