# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program: APAI's agent on the workstation, the arm on the Thor.

main() attaches to the robot link service, builds APAI's configuration and
runs APAI's own command line (apai.apps.demo.main) inside agent_hooks(). The
hooks give APAI the robot link arm and the link's stereo camera in place of
its motor driver and camera, bind the check ahead to APAI's checker, and put
the task gate around every task and every skill. APAI is not edited: names of
its modules are patched for the lifetime of demo.main.

Every run measures what the split over two machines costs: Measures writes
the robot connection's delays, each move's stretches, the Thor's single joint
reads and the camera's deliveries into the run folder, on a thread of its own.

Ctrl+C at this terminal is a STOP. The program cuts no torque. APAI's rest
tool asks the robot app for the motors off, and when the program ends the
operator is asked whether the arm returns to rest; the robot app runs that
return and switches the motors off, only at measured rest.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import logging
import math
import os
import queue
import select
import sys
import tempfile
import termios
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ..console_log import set_console_level
from . import config as agent_config
from .arm_adapter import (
    AGENT_TERMINAL,
    ARM_TYPE,
    MEASURED_JAW,
    MEASURED_JAW_BEGIN,
    MEASURED_MOVE,
    RobotLinkArm,
    arm_factory,
)
from .checkahead import CheckAhead, FrameTimes, MoveCheck
from .clear_grid import ClearGrid
from .config import AgentSettings, ConfigError, RunOptions
from .link_client import AgentLink, Arrival, LinkLost, StaleState
from .self_mask import EXPERIMENT_DIR_ENV, JointFeed, experiment_module, frame_context
from .task_gate import READ_ONLY_TOOLS, TaskGate

_log = logging.getLogger("isaaccapture.examples.agentic_remote.launcher")

EXIT_OK = 0
EXIT_FAILED = 1  # the robot link, APAI's runtime or the task did not succeed
EXIT_NOT_RUN = 2  # the arguments, the file or the installation cannot be used
EXIT_INTERRUPTED = 130  # a second Ctrl+C

PROGRAM = "agent program"
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"
STOP_REASON = "Ctrl+C at the agent terminal"
# How long the end of the program waits for a ROBOT_STATE in which the agent
# no longer owns the arm, before the local socket is closed.
SETTLE_S = 3.0
_INPUT_POLL_S = 0.05
# How long other output must be quiet before the task prompt is written again.
PROMPT_QUIET_S = 0.3
# The oldest a camera frame of a local camera may claim to be, in s.
_LOCAL_AGE_LIMIT_S = 5.0

Say = Callable[[str], None]


def pin_gpu(environ: dict[str, str] | None = None) -> None:
    """This process and its children see one card: index 0 in PCI order."""
    env = os.environ if environ is None else environ
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0"


def _say(text: str) -> None:
    print(f"[agent] {text}", flush=True)


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def terminal_ask(prompt: str) -> str:
    """What the operator types at this process's terminal; never a pipe.

    RuntimeError inside an agent's environment and without a terminal. What
    was typed before the question is shown is dropped.
    """
    from apai.control import arm_hold

    if arm_hold.agent_environment() is not None:
        raise RuntimeError(
            "motion needs the operator's terminal, outside an agent environment"
        )
    _drop_typed_ahead()
    try:
        return arm_hold._answer_at_terminal(prompt)
    except arm_hold.NoTerminal as missing:
        raise RuntimeError(str(missing)) from None


def _drop_typed_ahead() -> None:
    """Discard what waits unread at the controlling terminal of this session.

    APAI's helper reads the answer from /dev/tty; a "yes" typed while the task
    was thinking would answer a question the operator has not seen.
    """
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return
        terminal = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    except (AttributeError, OSError, ValueError):
        return  # no terminal: APAI's helper says so
    try:
        termios.tcflush(terminal, termios.TCIFLUSH)
    except termios.error:
        pass
    finally:
        os.close(terminal)


class _Noted:
    """A text stream of this process that notes when something is written to it."""

    def __init__(self, stream: Any, written: Callable[[str], None]) -> None:
        self._stream, self._written = stream, written

    def write(self, text: str) -> Any:
        count = self._stream.write(text)
        if text:
            self._written(text)
        return count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class PromptKeeper:
    """Keeps the task prompt the last line while the program waits for a task.

    While open, what any thread prints or logs behind the prompt is noted:
    sys.stdout and sys.stderr are wrapped, and the console handlers of the
    logger tree get a filter that passes every record. restore() writes the
    prompt again once such output has been quiet for PROMPT_QUIET_S, so a
    program that waits for a task shows it. Nothing of APAI is changed.
    """

    def __init__(self, prompt: str) -> None:
        self._prompt = str(prompt)
        self._at = 0.0  # the monotonic time of the last write behind the prompt
        self._dirty = False
        self._line_open = False  # the last write did not end its line
        self._streams: list[tuple[str, Any, Any]] = []
        self._handlers: list[logging.Handler] = []
        self.restored = 0  # how often the prompt was written again

    def _note(self, text: str = "\n") -> None:
        self._line_open = not text.endswith("\n")
        self._at, self._dirty = time.monotonic(), True

    def filter(self, record: logging.LogRecord) -> bool:
        """A logging filter that passes every record and notes it."""
        self._note()
        return True

    def __enter__(self) -> PromptKeeper:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            wrapped = _Noted(stream, self._note)
            setattr(sys, name, wrapped)
            self._streams.append((name, stream, wrapped))
        for owner in ("", "isaaccapture", "isaaccapture.examples.agentic_remote"):
            for handler in logging.getLogger(owner).handlers:
                handler.addFilter(self)
                self._handlers.append(handler)
        return self

    def __exit__(self, *_exc: object) -> None:
        for handler in self._handlers:
            handler.removeFilter(self)
        for name, stream, wrapped in reversed(self._streams):
            if getattr(sys, name) is wrapped:
                setattr(sys, name, stream)
        self._handlers, self._streams = [], []

    def restore(self) -> bool:
        """Write the prompt again when output behind it has gone quiet."""
        if not self._dirty or time.monotonic() - self._at < PROMPT_QUIET_S:
            return False
        out = self._streams[0][1] if self._streams else sys.stdout
        out.write(("\n" if self._line_open else "") + self._prompt)
        out.flush()
        self._dirty, self._line_open = False, False
        self.restored += 1
        return True


def terminal_input(prompt: str = "", idle: Callable[[], None] | None = None) -> str:
    """One line of the task terminal, read so that a signal is seen at once.

    idle() is called while no key comes, about every 50 ms. A prompt that
    other output pushed up is written again (PromptKeeper), until the first
    character of the line is read.
    """
    stream = sys.stdin
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return builtins.input(prompt)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("the task terminal belongs to the main thread")
    sys.stdout.write(str(prompt))
    sys.stdout.flush()
    data = bytearray()
    with PromptKeeper(prompt) as keeper:
        while True:
            readable, _, _ = select.select([fd], [], [], _INPUT_POLL_S)
            if not readable:
                if idle is not None:
                    idle()
                if prompt and not data:
                    keeper.restore()
                continue
            char = os.read(fd, 1)
            if char == b"\n" or not char:
                if not char and not data:
                    raise EOFError
                return data.decode(stream.encoding or "utf-8", "replace")
            data.extend(char)


class LocalFrameStamps:
    """Evidence times for camera frames that did not come over the camera link.

    APAI's mock camera stamps a camera frame on this machine's monotonic
    clock; its capture time on the Thor's wall clock is that age taken off the
    link's estimate of the Thor's clock. The times never go back. They are
    invented: nothing was captured on the Thor then, and each camera frame
    says so in capture["invented_times"].
    """

    def __init__(self, link: AgentLink, frame_times: FrameTimes) -> None:
        self._link, self._frame_times = link, frame_times
        self._lock = threading.Lock()
        self._last = float("-inf")

    def note(self, frame: Any) -> None:
        try:
            _, wall = self._link.thor_now()
        except (StaleState, LinkLost):
            return
        age = time.monotonic() - float(frame.t)
        if not 0.0 <= age <= _LOCAL_AGE_LIMIT_S:
            return  # not a time of this machine's monotonic clock: no evidence
        frame.capture = {**(frame.capture or {}), "invented_times": True}
        with self._lock:
            self._last = max(self._last, wall - age)
            self._frame_times.note(frame.t, self._last)


# What the model is told about a tool APAI's own instructions name and this
# program, as the functional launcher, does not offer.
TOOLS_GUIDANCE = (
    "Tools on this robot: the tool list you were given is complete. "
    "preview_grasp is not offered: grasp with grasp_object without a preview."
)


# Two texts of the experiment's launcher (thor_runtime.py, main), which holds
# them inline. The task of a run that names none, and what it adds to the task
# of a run that is not interactive.
DEFAULT_TASK = "Observe the scene and describe what you see. Do not move the robot."
DEPTH_TASK_NOTE = (
    " Camera depth is stereo-derived; mounting fit accepted, independent metric "
    "accuracy not yet verified."
)
# A skill's traceback, in the run folder.
SKILL_ERRORS = "skill-errors.log"
# What the experiment's launcher writes at its start, in the run folder.
STATUS_FILE = "runtime-status.json"


def _skill_runtime() -> Any:
    from apai.skills.runtime import SkillRuntime

    return SkillRuntime


def _skills_with_traceback(run_dir: Path) -> list[tuple[str, Any]]:
    """Each skill of APAI's runtime, writing a TypeError's traceback to the run.

    APAI reports a TypeError inside a skill as "bad arguments" and drops its
    traceback; here it goes to SKILL_ERRORS and the error is raised again.
    """
    import functools
    import inspect
    import traceback

    def traced(skill: Any) -> Any:
        @functools.wraps(skill)
        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return skill(*args, **kwargs)
            except TypeError:
                try:
                    with (run_dir / SKILL_ERRORS).open("a", encoding="utf-8") as out:
                        out.write(f"--- {skill.__name__}\n{traceback.format_exc()}\n")
                except OSError:
                    _log.warning("a skill's traceback could not be written")
                raise

        return call

    return [
        (name, traced(skill))
        for name, skill in vars(_skill_runtime()).items()
        if name.startswith("skill_") and inspect.isfunction(skill)
    ]


def task_arguments(
    args: argparse.Namespace, settings: AgentSettings, run_dir: Path
) -> list[str]:
    """The arguments of APAI's command line for this run.

    With the full stack as the experiment's launcher gives them: the profiles
    by name, and always a task, its own default when none is named, with its
    note on the depth. APAI reads no --task of an interactive run.
    """
    # The model is the configuration's; naming it keeps APAI from probing
    # for another one.
    model = settings.experiment.llm if settings.minimal else "codex_astra"
    arguments = ["--no-view", "--no-serve", "--run-dir", str(run_dir)]
    arguments += ["--llm", model, "--max-steps", str(args.max_steps)]
    if settings.minimal:
        if args.task is not None:
            arguments += ["--task", args.task]
    else:
        kind = agent_config.camera_kind(args.camera)[0]
        mock_camera = kind == agent_config.CAMERA_MOCK
        camera = kind if mock_camera else str(Path(run_dir) / "camera_profile")
        arguments += ["--camera", camera, "--arm", str(Path(run_dir) / "arm_profile")]
        arguments += ["--task", (args.task or DEFAULT_TASK) + DEPTH_TASK_NOTE]
    if args.interactive:
        arguments.append("--interactive")
    return arguments


def _detector_client(
    original: Callable[[dict[str, Any]], Any], settings: AgentSettings
) -> Callable[[dict[str, Any]], Any]:
    """The experiment's detector client on the model client's program of this run."""
    named = settings.experiment.codex_bin

    def make_client(config: dict[str, Any]) -> Any:
        if not named:
            return original(config)
        # An absolute path: the experiment joins it to its own folder.
        return original({**config, "llm": {**config["llm"], "codex_bin": named}})

    return make_client


@contextmanager
def agent_hooks(
    cfg: Any,
    settings: AgentSettings,
    link: AgentLink,
    gate: TaskGate,
    run_dir: Path,
    *,
    frame_times: FrameTimes | None = None,
    joint_feed: Any = None,
    source: Any = None,
    calibration: str = "",
    invented_times: bool = False,
    say: Say = _say,
    task_input: Callable[[str], str] = terminal_input,
    timing: Any = None,
    measures: Measures | None = None,
) -> Iterator[dict[str, Any]]:
    """APAI's demo.main, while this is open, runs on the robot link.

    Yields a dict the hooks fill: "runtime", "adapter", "harness", "grid";
    "invented_times" is there from the start. Only with invented_times does a
    camera frame that did not come over the camera link get an evidence time.
    timing is the recorder of the experiment's timing, which then times the
    wait for each task line. With thor_batch_milestones in cfg the agent gets
    the experiment's milestone batching. measures gets what the arm's moves
    and jaw commands measured and every camera frame the camera returns.
    """
    from apai.agent import orchestrator
    from apai.apps import demo
    from apai.perception import camera_base

    from .camera import stereo_camera

    run_dir = Path(run_dir)
    full = not settings.minimal
    frame_times = FrameTimes() if frame_times is None else frame_times
    built: dict[str, Any] = {"invented_times": bool(invented_times)}
    make_arm = arm_factory(
        link,
        agent_config.arm_settings(settings, calibration, cfg),
        joint_feed,
        None if measures is None else measures.note,
    )
    stamps = LocalFrameStamps(link, frame_times) if invented_times else None
    mounting = _mounting_facts(cfg) if full else {}
    original_build, original_arm = demo.build_runtime, demo._build_arm
    original_agent, original_report = demo.AgentOrchestrator, demo._print_report
    original_specs = orchestrator.tool_specs_for_config
    experiment = _experiment_parts() if full else {}
    # The two registrations (below) replace a name of APAI for good; both are
    # put back when the hooks close.
    unregistered_camera = camera_base.make_camera
    unregistered_detector = demo._make_detector
    registered_camera: Any = None
    bridge: Any = None

    def make_camera(ccfg: Any) -> Any:
        camera = registered_camera(ccfg)
        grab = camera.get_frame
        stereo = ccfg.get("type") == stereo_camera.CAMERA_TYPE

        def get_frame() -> Any:
            frame = grab()
            if stereo:
                frame.capture = {**(frame.capture or {}), **mounting}
            elif stamps is not None:
                stamps.note(frame)
            if measures is not None:
                measures.note_frame(frame)  # queues only: the camera goes on
            return frame

        camera.get_frame = get_frame
        return camera

    def build_arm(
        acfg: Any, lazy_arm: bool, occupancy: Any, fallback_cfg: Any, safety: Any = None
    ) -> Any:
        if full:
            experiment["install_grid_sampler"](occupancy)
        # An eager arm: connecting sends nothing and cuts no torque here.
        raw, safe, kin = original_arm(acfg, False, occupancy, fallback_cfg, safety)
        if not callable(getattr(raw, "bind_checker", None)):
            raise RuntimeError(f"the arm profile's type must be {ARM_TYPE}")
        if full:
            experiment["restrict_motion"](
                safe.harness, acfg.get("motion_bounds"), label="operation"
            )
        raw.bind_checker(
            CheckAhead(
                safe.harness, frame_times, period_s=link.welcome.setpoint_period_s
            )
        )
        return raw, safe, kin

    def build(selected: Any, output: Any, **_kwargs: Any) -> Any:
        nonlocal bridge
        if full:
            low, high = demo._shared_map_region(selected, demo._arm_cfgs(selected))
            bridge = experiment["OwnedOccupancyBridge"](
                output, low, high, device="cuda:0", target_retirement=True
            ).start()
            selected._data["occupancy"].update(host="127.0.0.1", port=bridge.port)
        with _grid_client(settings):
            runtime, raw = original_build(
                selected, output, lazy_arm=False, view=False, serve=False
            )
        # From here demo.main owns the runtime; a failure below tears it down.
        try:
            safe = runtime.arm
            adapter, harness = safe.raw, safe.harness
            grid = harness.occupancy
            if full:
                bridge.verify(grid.status)
                experiment["install_verification"](runtime, output)
            original_execute = runtime.execute
            runtime.execute = lambda name, args: gate.execute(
                original_execute, name, args
            )
            gate.bind_arm(adapter, harness)
            built.update(runtime=runtime, adapter=adapter, harness=harness, grid=grid)
            if full:
                _startup_files(
                    experiment["save_frame"], runtime, Path(output), mounting
                )
                # How far the arm may go, in the experiment launcher's words.
                say(
                    experiment["Admission"](
                        True,
                        Path(output),
                        selected.arm.motion_bounds,
                        selected.safety.max_joint_vel,
                        motion_approval=agent_config.APPROVAL_TASK,
                    ).motion_description()
                )
            usable = grid is not None and grid.wait_until_usable(
                settings.task.grid_wait_s
            )
            say(f"obstacle grid: {_grid_state(grid)}")
            if not usable and not full:
                raise RuntimeError(
                    "the obstacle grid is not usable: no camera frame reached "
                    f"it within {settings.task.grid_wait_s:g} s"
                )
            if not usable:
                # As the experiment's launcher, which starts without a grid.
                say(
                    "the obstacle grid is not usable yet: the program goes on, "
                    "and no move starts without a fresh obstacle grid"
                )
            say(
                "ready. Ctrl+C at this terminal is STOP: the arm holds with the "
                "motors on. When the program ends you are asked whether the arm "
                "returns to rest; there the robot app switches the motors off"
            )
        except BaseException:
            demo.shutdown_runtime(runtime, raw)
            raise
        return runtime, raw

    def make_agent(*args: Any, **kwargs: Any) -> Any:
        agent = original_agent(*args, **kwargs)
        if cfg.get("thor_batch_milestones", False):
            _batched_milestones()(agent)
        agent.run_task = gate.bind_run_task(agent.run_task)
        if full:
            agent._base_system_prompt += "\n" + experiment["guidance"]
            agent._base_system_prompt += "\n" + TOOLS_GUIDANCE
            agent.set_harness(agent.harness)
        return agent

    def print_report(report: Any) -> None:
        row = asdict(report) if is_dataclass(report) else {"report": repr(report)}
        with (run_dir / "task-reports.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, default=str) + "\n")
        original_report(report)

    def specs(config: Any) -> list[Any]:
        # The functional launcher's tool list with --operate: its read-only
        # tools and every motion skill of APAI, the rest tool among them.
        motion = _motion_skills()
        return [
            item
            for item in original_specs(config)
            if item["name"] in READ_ONLY_TOOLS or item["name"] in motion
        ]

    def next_task(prompt: str = "") -> str:
        # APAI's task prompt ends at an empty line. Here an empty or blank
        # line is read again: only the end of the input (Ctrl+D) ends.
        while True:
            if gate.tell_end():
                raise EOFError  # ends APAI's task prompt: no new task is taken
            gate.return_late_ownership()
            if task_input is terminal_input:
                line = terminal_input(prompt, idle=gate.return_late_ownership)
            else:
                line = task_input(prompt)
            if line.strip():
                return line

    if timing is not None:
        # The span the experiment's timing gives the wait for a task line.
        next_task = timing.wrap(next_task, "operator.terminal_input", "Operator")

    try:
        stereo_camera.register_camera(link, frame_times, source)
        registered_camera = camera_base.make_camera
        if full:
            experiment["register_detector"]()
        with ExitStack() as stack:
            if full:
                stack.enter_context(experiment["motion_adapters"](cfg))
                stack.enter_context(experiment["perception_adapters"]())
                stack.enter_context(experiment["placement_adapters"]())
                stack.enter_context(experiment["target_map_adapters"](run_dir))
                colors = experiment["colors"]
                stack.enter_context(
                    patch.object(colors, "mask_color", new=experiment["mask_color"])
                )
                # The detector builds a model client of its own from the
                # experiment's file: it gets this run's program too.
                cloud = experiment_module("thor_cloud")
                stack.enter_context(
                    patch.object(
                        cloud,
                        "make_client",
                        new=_detector_client(cloud.make_client, settings),
                    )
                )
            for skill_name, skill in _skills_with_traceback(run_dir):
                stack.enter_context(
                    patch.object(_skill_runtime(), skill_name, new=skill)
                )
            stack.enter_context(frame_context())
            for target, name, new in (
                (demo, "load_demo_config", lambda **_kwargs: cfg),
                (demo, "build_runtime", build),
                (demo, "make_arm", make_arm),
                (demo, "_build_arm", build_arm),
                (demo, "make_camera", make_camera),
                (camera_base, "make_camera", make_camera),
                (demo, "AgentOrchestrator", make_agent),
                (orchestrator, "tool_specs_for_config", specs),
                (demo, "_print_report", print_report),
            ):
                stack.enter_context(patch.object(target, name, new=new))
            stack.enter_context(patch.object(demo, "input", new=next_task, create=True))
            yield built
    finally:
        camera_base.make_camera = unregistered_camera
        demo._make_detector = unregistered_detector
        if bridge is not None:
            bridge.close()
        _record_experiment_files(run_dir)


@contextmanager
def _grid_client(settings: AgentSettings) -> Iterator[None]:
    """While open, APAI makes the client of its obstacle map as the file says.

    With the minimal stack and experiment.grid clear that client is ClearGrid,
    the stand-in that reports nothing near; else APAI's own, which dials a grid
    bridge. The map itself, the check ahead and the PERMITs are the same.
    """
    clear = settings.experiment.grid == agent_config.GRID_CLEAR
    if not (settings.minimal and clear):
        yield
        return
    from apai.perception import occupancy

    with patch.object(occupancy, "OccupancyClient", ClearGrid):
        yield


def _experiment_parts() -> dict[str, Any]:
    """The experiment's behaviour hooks; none of them touches a motor driver."""
    from apai.perception import colors

    rig = experiment_module("thor_calibration_rig")
    perception = experiment_module("thor_perception")
    placement = experiment_module("thor_placement")
    theirs = experiment_module("thor_runtime")  # the experiment's launcher
    return {
        "Admission": theirs.Admission,
        "save_frame": theirs.save_frame,
        "OwnedOccupancyBridge": rig.OwnedOccupancyBridge,
        "restrict_motion": rig.restrict_motion,
        "install_grid_sampler": experiment_module(
            "thor_occupancy"
        ).install_grid_sampler,
        "install_verification": experiment_module(
            "thor_verification"
        ).install_verification,
        "motion_adapters": experiment_module("thor_motion").motion_adapters,
        "perception_adapters": perception.perception_adapters,
        "placement_adapters": placement.placement_adapters,
        "target_map_adapters": experiment_module(
            "thor_target_lifecycle"
        ).target_map_adapters,
        "guidance": perception.TARGET_GUIDANCE + placement.PLACEMENT_GUIDANCE,
        "mask_color": experiment_module("thor_colors").mask_color,
        "register_detector": experiment_module("thor_cloud").register_detector,
        "colors": colors,
    }


def _mounting_facts(cfg: Any) -> dict[str, Any]:
    """What the experiment's perception reads of a camera frame's mounting.

    The hand-eye file is the one APAI's configuration names for the camera.
    """
    try:
        data = cfg.as_dict() if hasattr(cfg, "as_dict") else cfg
        file = str(data["camera"]["extrinsics"]["hand_eye_json"])
        digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    except (KeyError, TypeError, OSError):
        file, digest = "", ""
    return {
        "camera_to_robot_base_calibrated": True,
        "mounting_calibration_file": file,
        "mounting_calibration_sha256": digest,
        "absolute_accuracy_verified": False,
    }


def _startup_files(
    save_frame: Callable[[Any, Path], dict[str, Any]],
    runtime: Any,
    run_dir: Path,
    mounting: dict[str, Any],
) -> None:
    """The experiment launcher's start: one fresh camera frame, and STATUS_FILE.

    save_frame is thor_runtime.save_frame: it waits for a fresh camera frame,
    saves it as startup-rgbd.npz, startup-rgb.jpg and startup-depth-m.npy, and
    raises TimeoutError when the camera gives none. STATUS_FILE says what was
    found, or why the start failed.
    """
    status: dict[str, Any] = {
        "mode": "operate",
        "mounting": mounting,
        "motion_approval": agent_config.APPROVAL_TASK,
    }
    try:
        status["startup"] = save_frame(runtime, run_dir)
        status["occupancy_backend"] = runtime.arm.harness.occupancy.status
    except BaseException as error:
        status["startup_error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            agent_config.write_json(run_dir / STATUS_FILE, status)
        except OSError as problem:
            _log.warning("%s was not written: %s", STATUS_FILE, problem)


def _motion_skills() -> frozenset[str]:
    from apai.skills.runtime import _MOTION_SKILLS

    return frozenset(_MOTION_SKILLS)


def _grid_state(grid: Any) -> str:
    if grid is None:
        return "none"
    try:
        report = grid.report()
        text = report.get("detail") or report
    except Exception as problem:
        text = f"no report ({problem})"
    return f"{text}; refreshes {getattr(grid, 'refreshes', '?')}"


def _record_experiment_files(run_dir: Path) -> None:
    """experiment-files.json: the hash of every experiment file imported."""
    folder = os.environ.get(EXPERIMENT_DIR_ENV)
    if not folder or not Path(folder).is_dir():
        return
    try:
        hashes = agent_config.file_hashes(Path(folder), dict(sys.modules))
        agent_config.write_json(run_dir / "experiment-files.json", hashes)
    except OSError as problem:
        _log.warning("experiment-files.json was not written: %s", problem)


def _record_resolved_config(run_dir: Path, cfg: Any) -> None:
    """resolved-config.yaml: APAI's configuration of this run."""
    try:
        agent_config.write_resolved_config(run_dir, cfg)
    except (ConfigError, OSError) as problem:
        _log.warning(
            "%s was not written, so the experiment's viewer will not list this run: %s",
            agent_config.RESOLVED_CONFIG,
            problem,
        )


def _batched_milestones() -> Callable[[Any], None]:
    """The experiment's install_batched_milestones(agent)."""
    return experiment_module("thor_milestones").install_batched_milestones


class _TimedParts:
    """What the experiment's timing takes in place of its own launcher's module.

    It times that module's Admission.execute as the tools of a task and its
    operator_answer as the wait for the typed yes. Here they are the task
    gate's execute and the program's question.
    """

    Admission = TaskGate

    def __init__(self, ask: Callable[[str], str]) -> None:
        self.operator_answer = ask


# Spans (timed stretches) the experiment's timing takes from its motor arm and
# its camera, which this program does not build. The robot link arm and the
# link's stereo camera give them under the same names, and the parts of a move
# and of a task's start and end that exist only with two machines under names
# of their own: (method, span, category).
_ARM_SPANS = (
    ("stream_to", "motor.stream_to", "Execution"),
    ("close_gripper_two_stage", "gripper.close_two_stage", "Execution"),
    # Inside motor.stream_to, in this order: the wait for joint readings taken
    # after the call began; MOVE sent to its first RESULT, inside MOVE sent to
    # the move shown as armed; the PERMITs until the final RESULT.
    ("_fresh_state", "motor.fresh_joint_readings", "Execution"),
    ("_await_first", "motor.move_to_first_result", "Execution"),
    ("_await_armed", "motor.move_to_armed", "Execution"),
    ("_permit_loop", "motor.permits", "Execution"),
)
# A span without length per point of the permit loop: motor.mark:first_permit,
# motor.mark:first_setpoint, motor.mark:all_sent (MARKS in arm_adapter.py).
_MARK_SPANS = (("_mark", "motor.mark", "Execution"),)
# The wait for a fresh obstacle grid and the check of the whole move, before
# the MOVE is sent.
_CHECK_SPANS = (
    ("wait_fresh_grid", "safety.wait_fresh_grid", "Safety"),
    ("vet_whole", "safety.check_whole_move", "Safety"),
)
# One pass of the check ahead, counted in bulk (count, total, longest) inside
# the motor.permits span of its move.
_PASS_SPANS = (("step", "safety.check_ahead_pass", "Safety"),)
# Taking the arm before a task's first motion tool (the typed yes is inside
# it) and giving it back, or asking for help, after the task.
_GATE_SPANS = (
    ("_start", "task.take_control", "Task"),
    ("_end", "task.give_back_control", "Task"),
)
_CAMERA_SPANS = (("_grab", "camera.pair_wait_read_depth", "Look"),)


def _link_spans(stack: ExitStack, timing: Any) -> None:
    """Time the spans of the tables above until the stack closes."""
    from .camera import stereo_camera

    plain: dict[str, Any] = {}
    for owner, spans, options in (
        (RobotLinkArm, _ARM_SPANS, plain),
        (RobotLinkArm, _MARK_SPANS, {"named_argument": 1}),
        (CheckAhead, _CHECK_SPANS, plain),
        (MoveCheck, _PASS_SPANS, {"hot": True}),
        (TaskGate, _GATE_SPANS, plain),
        (stereo_camera.LinkStereoCamera, _CAMERA_SPANS, plain),
    ):
        for method, name, category in spans:
            measured = timing.wrap(getattr(owner, method), name, category, **options)
            stack.enter_context(patch.object(owner, method, measured))
            timing.hooks.append(f"{owner.__module__}.{owner.__name__}.{method}")


# What a robot.feedback event of the debug recording says of itself here. The
# experiment records that event on its motor arm, which this program does not
# build; here it is made from ROBOT_STATE as it arrives.
FEEDBACK_SOURCE = "ROBOT_STATE over the robot connection"
FEEDBACK_CLOCK = (
    "state.t is the ROBOT_STATE's receipt on the workstation's monotonic clock "
    "less the reading's age; no time of the motors"
)
# The shortest time between two robot.feedback events, on the Thor's
# monotonic clock: at most 10 a second.
FEEDBACK_PERIOD_S = 0.1
# The event of the debug recording with what else a ROBOT_STATE reports of the
# motors: each single joint read once, the last target sent, the jaw reading.
STATE_EVENT = "robot.state"

# What a run measures, in its run folder. Each of the first five holds one
# JSON object per line.
MOVES_FILE = "moves.jsonl"  # one line per move
JAWS_FILE = "jaw-commands.jsonl"  # one line per jaw command
STATES_FILE = "robot-state.jsonl"  # what ROBOT_STATE reports of the motors
CAMERA_FILE = "camera-deliveries.jsonl"  # one line per camera frame returned
LATENCY_FILE = "link-latency.jsonl"  # the robot connection, once a second
SUMMARY_FILE = "latency-summary.txt"  # written when the program ends
MEASURE_FILES = (MOVES_FILE, JAWS_FILE, STATES_FILE, CAMERA_FILE, LATENCY_FILE)
# The one thread that writes them.
MEASURE_THREAD = "agent-measures"
# What may wait for that thread; what comes on a full queue is dropped.
MEASURE_QUEUE = 4096
# While the arm does not move and no jaw command runs, STATES_FILE gets one
# line per this many s; LATENCY_FILE one line per this many s at most.
IDLE_STATE_PERIOD_S = 1.0
LATENCY_PERIOD_S = 1.0
# Values kept per kind for the median and the 95th percentile.
SAMPLES_KEPT = 65536
# Readings kept of one jaw command, 30 s of ROBOT_STATEs at 20 a second.
JAW_TRACE_KEPT = 600
# Moves whose ROBOT_STATEs are counted at one time.
_MOVES_TRACKED = 8
# How long close() waits for the thread, in s.
_CLOSE_WAIT_S = 3.0
_EPSILON_S = 1e-6


def _open_append(path: Path) -> Any:
    """Open a file of the measurements for appending, line by line.

    Every file of MEASURE_FILES and SUMMARY_FILE is opened here and nowhere
    else, on MEASURE_THREAD.
    """
    return path.open("a", encoding="utf-8", buffering=1)


def _number(value: Any) -> float | None:
    """value as a float when it is a finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _rounded(value: Any, digits: int = 6) -> float | None:
    number = _number(value)
    return None if number is None else round(number, digits)


class _Bulk:
    """Durations counted in bulk: how many, their sum and the longest, in s."""

    def __init__(self) -> None:
        self.count, self.total, self.longest = 0, 0.0, 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.longest = max(self.longest, value)

    def row(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_s": round(self.total, 6),
            "longest_s": round(self.longest, 6),
        }


class _Samples(_Bulk):
    """Values of one kind with a bounded share of them for the quantiles.

    At `keep` values every second kept one is dropped, and from then on only
    every second of those that come is kept, and so on: the median and the
    95th percentile are those of an even share of the values.
    """

    def __init__(self, keep: int = SAMPLES_KEPT) -> None:
        super().__init__()
        self.shortest = math.inf
        self._keep, self._every = max(2, int(keep)), 1
        self._kept: list[float] = []

    def add(self, value: float) -> None:
        super().add(value)
        self.shortest = min(self.shortest, value)
        if self.count % self._every == 0:
            self._kept.append(value)
            if len(self._kept) >= self._keep:
                self._kept = self._kept[1::2]
                self._every *= 2

    def quantile(self, share: float) -> float | None:
        """The value at this share of the kept ones in order; None without any."""
        if not self._kept:
            return None
        ordered = sorted(self._kept)
        return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


class _Lines:
    """One file of JSON lines. A failure to write is logged once; then it is shut."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.written = 0
        self.failed = ""
        self._file: Any = None

    def open(self) -> None:
        if self._file is None and not self.failed:
            try:
                self._file = _open_append(self.path)
            except OSError as problem:
                self._fail(problem)

    def write(self, row: dict[str, Any]) -> None:
        self.open()
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(row, default=str) + "\n")
            self.written += 1
        except (OSError, TypeError, ValueError) as problem:
            self._fail(problem)

    def close(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError as problem:
                self._fail(problem)

    def _fail(self, problem: Exception) -> None:
        if not self.failed:
            self.failed = f"{type(problem).__name__}: {problem}"
            _log.warning(
                "%s cannot be written (%s); the run goes on without it",
                self.path,
                self.failed,
            )
        self._file = None


# The durations the summary lists, in this order: (key, what it is). The keys
# are also the names in LATENCY_FILE and MOVES_FILE where a line holds one.
_SUMMARY = (
    ("transit_above_shortest_s", "ROBOT_STATE transit above the shortest"),
    ("arrival_gap_s", "ROBOT_STATE arrival to the next"),
    ("joint_read_s", "single joint read, on the Thor's clock"),
    ("fresh_wait_s", "move: wait for fresh joint readings"),
    ("grid_wait_s", "move: wait for a fresh obstacle grid"),
    ("whole_check_s", "move: check of the whole move"),
    ("first_result_after_s", "MOVE sent to its first RESULT"),
    ("armed_after_s", "MOVE sent to armed"),
    ("first_permit_after_s", "MOVE sent to the first PERMIT"),
    ("first_sent_after_s", "MOVE sent to the first setpoint reported"),
    ("sending_s", "move: first setpoint reported to the last rise of the count"),
    ("after_last_progress_s", "move: last rise of the count to final RESULT"),
    ("settle_s", "move: every setpoint reported to final RESULT"),
    ("result_after_s", "MOVE sent to its final RESULT"),
    ("max_pass_s", "move: longest check ahead pass"),
    ("jaw_result_s", "jaw command sent to its RESULT"),
    ("permit_frame_age_s", "camera frame age under a PERMIT, on the Thor"),
    ("frame_age_at_return_s", "camera frame: its time to its return"),
    ("stop_shown_s", "STOP sent to the ROBOT_STATE that shows it"),
)
# Those of them a line of MOVES_FILE holds, on this machine's monotonic clock.
_MOVE_DURATIONS = (
    "fresh_wait_s",
    "grid_wait_s",
    "whole_check_s",
    "first_result_after_s",
    "armed_after_s",
    "first_permit_after_s",
    "first_sent_after_s",
    "sending_s",
    "after_last_progress_s",
    "settle_s",
    "result_after_s",
    "max_pass_s",
)


class Measures:
    """What a run measures of the split over two machines, and its files.

    Nothing here runs on a thread that drives the arm except a hand-over to a
    bounded queue: on_state() on the link's receive thread, note() on the
    skill thread (the thread that sends the PERMITs) and note_frame() on the
    camera's thread. One thread, MEASURE_THREAD, takes the queue, counts and
    writes MEASURE_FILES. What comes on a full queue is dropped and counted.
    A file that cannot be written is logged once and left out. With the
    experiment's debug recorder the same thread gives it one robot.feedback
    event per FEEDBACK_PERIOD_S and one STATE_EVENT per ROBOT_STATE.
    """

    def __init__(
        self, run_dir: Path, recording: Any = None, *, queue_size: int = MEASURE_QUEUE
    ) -> None:
        self.run_dir = Path(run_dir)
        self._recording = recording
        self._queue: queue.Queue[tuple[Any, ...]] = queue.Queue(maxsize=queue_size)
        self._count_lock = threading.Lock()
        self.dropped = 0  # what a full queue turned away
        self._closing = threading.Event()
        self._closed = False
        self._trouble = False
        self._summary_lines: list[str] = []
        self._link: Any = None
        self._wall_offset = time.time() - time.monotonic()
        self._files = {name: _Lines(self.run_dir / name) for name in MEASURE_FILES}
        self.samples = {key: _Samples() for key, _ in _SUMMARY}
        self.pace = _Samples()  # setpoints a second, one value per move
        self.states = 0  # ROBOT_STATEs counted
        self.reads_failed = 0  # single joint reads the Thor reported as not valid
        self.read_gaps = 0  # times single joint reads may have been missed
        self.seq_missing = 0  # ROBOT_STATEs that did not arrive, by seq
        self.missed_deadlines = 0  # the rise of the Thor's count over the run
        self._clock: Arrival | None = None  # of the newest ROBOT_STATE
        # What the thread keeps between two ROBOT_STATEs.
        self._session: str | None = None
        self._before: Any = None  # the ROBOT_STATE before this one
        self._before_at = 0.0
        self._last_read: dict[int, float] = {}  # joint -> t_end of its last read
        self._missed: float | None = None  # the Thor's running count
        self._moves: dict[str, dict[str, Any]] = {}  # move id -> what was seen
        self._jaw: dict[str, Any] | None = None  # the jaw command that runs
        self._feedback_clock = -math.inf
        self._state_line_at = -math.inf
        self._state_reads = _Bulk()  # since the last line of STATES_FILE
        self._latency_at: float | None = None
        self._second = self._new_second()
        if recording is not None and hasattr(recording, "hooks"):
            recording.hooks.append(f"{__name__}.Measures.on_state")
        self._thread = threading.Thread(
            target=self._run, name=MEASURE_THREAD, daemon=True
        )
        self._thread.start()

    # What the other threads call: each only queues.

    def attach(self, link: Any) -> None:
        """Measure every ROBOT_STATE of this link from now on."""
        self._link = link
        link.add_arrival_listener(self.on_state)

    def on_state(self, state: Any, arrival: Arrival) -> None:
        """An arrival listener of the link; on its receive thread."""
        self._put(("state", arrival, state))

    def note(self, kind: str, facts: dict[str, Any]) -> None:
        """What the arm adapter measured (Measure in arm_adapter.py)."""
        self._put((kind, time.monotonic(), dict(facts)))

    def note_frame(self, frame: Any) -> None:
        """A camera frame the agent's camera returns; never raises."""
        try:
            capture = dict(getattr(frame, "capture", None) or {})
            self._put(("frame", time.monotonic(), getattr(frame, "t", None), capture))
        except Exception:
            self._log_trouble()

    def _put(self, item: tuple[Any, ...]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._count_lock:
                self.dropped += 1

    # The end of the run.

    def finish(self) -> None:
        """Take nothing more and end the thread, which writes SUMMARY_FILE last.

        Waits for it _CLOSE_WAIT_S at most; never raises. Called before the
        link is closed, so that nothing of this is left to do behind it.
        """
        if self._closed:
            return
        self._closed = True
        self._closing.set()
        try:
            self._queue.put_nowait(("end",))  # wakes the thread at once
        except queue.Full:
            pass  # it is at work and looks at _closing
        self._thread.join(_CLOSE_WAIT_S)

    def close(self, say: Say | None = None) -> list[str]:
        """finish(), then say the summary's lines; returns them. Never raises.

        Nothing is said when no ROBOT_STATE arrived. Writes no file: the
        thread wrote the summary when it ended.
        """
        self.finish()
        lines = list(self._summary_lines)
        try:
            if self._thread.is_alive():
                lines = [
                    *self.summary(),
                    f"the thread that writes the measurements did not end in "
                    f"{_CLOSE_WAIT_S:g} s: {SUMMARY_FILE} and the last lines of "
                    "the other files may be missing",
                ]
            if say is not None and self.states:
                for line in lines:
                    say(line)
        except Exception:
            self._log_trouble()
        return lines

    def summary(self) -> list[str]:
        """The latency summary: one line per kind of duration that was measured."""
        lines = [
            f"latency summary of {self.run_dir}",
            "in ms: count, median, 95th percentile, longest",
        ]
        for key, what in _SUMMARY:
            kind = self.samples[key]
            if kind.count:
                lines.append(
                    f"  {what}: {kind.count}, "
                    f"{_ms(kind.quantile(0.5))}, {_ms(kind.quantile(0.95))}, "
                    f"{_ms(kind.longest)}"
                )
        if len(lines) == 2:
            lines.append("  nothing was measured: no ROBOT_STATE arrived")
        clock = self._clock
        if clock is not None:
            steps = getattr(self._link, "clock_steps", 0)
            lines.append(
                f"clock: this machine's monotonic clock less the Thor's is "
                f"{clock.offset_s:.6f} s with the shortest transit in it; the "
                f"newest {clock.samples} ROBOT_STATEs spread over "
                f"{_ms(clock.spread_s)} ms; {steps} step(s) of the Thor's wall clock"
            )
        if self.pace.count:
            lines.append(
                f"setpoint pace: {self.pace.count} move(s), median "
                f"{self.pace.quantile(0.5):.1f} a second, lowest "
                f"{self.pace.shortest:.1f}, highest {self.pace.longest:.1f}"
            )
        lines.append(
            f"ROBOT_STATEs {self.states}, not arrived {self.seq_missing}; missed "
            f"setpoint deadlines {self.missed_deadlines}; single joint reads "
            f"not valid {self.reads_failed}, possibly missed at {self.read_gaps} "
            f"gap(s); measurements dropped on a full queue {self.dropped}"
        )
        failed = [name for name, file in self._files.items() if file.failed]
        if failed:
            lines.append(f"not written: {', '.join(failed)}")
        return lines

    # The thread.

    def _run(self) -> None:
        for file in self._files.values():
            file.open()
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    break
                continue
            if item[0] == "end":
                break
            try:
                self._take(item)
            except Exception:
                self._log_trouble()
        try:
            self._latency_line(self._clock)
        except Exception:
            self._log_trouble()
        for file in self._files.values():
            file.close()
        try:
            self._summary_lines = self.summary()
            with _open_append(self.run_dir / SUMMARY_FILE) as out:
                out.write("\n".join(self._summary_lines) + "\n")
        except OSError as problem:
            _log.warning("%s was not written: %s", SUMMARY_FILE, problem)
        except Exception:
            self._log_trouble()

    def _log_trouble(self) -> None:
        if not self._trouble:
            self._trouble = True
            _log.warning(
                "a measurement failed; the run goes on without it", exc_info=True
            )

    def _take(self, item: tuple[Any, ...]) -> None:
        kind = item[0]
        if kind == "state":
            self._state(item[1], item[2])
        elif kind == "frame":
            self._frame(item[1], item[2], item[3])
        elif kind == MEASURED_MOVE:
            self._move(item[1], item[2])
        elif kind == MEASURED_JAW_BEGIN:
            self._jaw = {"trace": [], "states": 0, "gap_max": None}
        elif kind == MEASURED_JAW:
            self._jaw_done(item[1], item[2])

    def _utc(self, mono_s: float) -> str:
        """A time of this machine's monotonic clock as UTC text."""
        stamp = datetime.fromtimestamp(mono_s + self._wall_offset, timezone.utc)
        return stamp.isoformat(timespec="milliseconds")

    def _new_second(self) -> dict[str, Any]:
        """What LATENCY_FILE's next line counts."""
        return {
            "states": 0,
            "seq_missing": 0,
            "arrival_gap_max_s": None,
            "transit": [],
            "reads": _Bulk(),
            "read_gaps": 0,
            "permit_frame_age_max_s": None,
            "first_result_after_s": [],
            "jaw_result_s": [],
            "stop_shown_s": [],
        }

    def _add(self, key: str, value: Any) -> float | None:
        """Count a duration of the summary when it is a number."""
        number = _number(value)
        if number is not None:
            self.samples[key].add(number)
        return number

    # One ROBOT_STATE.

    def _state(self, arrival: Arrival, state: Any) -> None:
        second = self._second
        before = self._before
        if before is not None and (
            state.session_id != self._session
            or state.clock_mono_s < before.clock_mono_s
        ):
            before = None  # another run of the Thor's clock
            self._last_read.clear()
            self._feedback_clock = -math.inf
        self._session = state.session_id
        self.states += 1
        self._clock = arrival
        second["states"] += 1
        if before is not None:
            gap = self._add("arrival_gap_s", arrival.at_s - self._before_at)
            if gap is not None:
                second["arrival_gap_max_s"] = max(
                    second["arrival_gap_max_s"] or 0.0, gap
                )
            missing = max(0, int(state.seq) - int(before.seq) - 1)
            second["seq_missing"] += missing
            self.seq_missing += missing
            if missing and state.recent:
                # ROBOT_STATEs were lost, and the window of recent reads of
                # this one does not reach back to the one before.
                if min(read.t_end for read in state.recent) > before.clock_mono_s:
                    self.read_gaps += 1
                    second["read_gaps"] += 1
        above = self._add("transit_above_shortest_s", arrival.transit_above_shortest_s)
        if above is not None:
            second["transit"].append(above)
        shown = self._add("stop_shown_s", arrival.stop_shown_s)
        if shown is not None:
            second["stop_shown_s"].append(round(shown, 6))
        reads = self._new_reads(state)
        health = state.health if isinstance(state.health, dict) else {}
        age = self._add("permit_frame_age_s", self._move_seen(state, health, reads))
        if age is not None:
            second["permit_frame_age_max_s"] = max(
                second["permit_frame_age_max_s"] or 0.0, age
            )
        missed = _number(health.get("missed_deadlines"))
        if missed is not None:
            if self._missed is not None and missed > self._missed:
                self.missed_deadlines += int(missed - self._missed)
            self._missed = missed
        pose = _pose_of(state)
        jaw = _jaw_row(state.gripper)
        if self._jaw is not None:
            self._jaw_seen(state, jaw)
        busy = bool(state.move_id) or state.motion == "moving" or self._jaw is not None
        if busy or arrival.at_s - self._state_line_at >= IDLE_STATE_PERIOD_S:
            self._state_line(arrival, state, health, pose, jaw, above)
        if self._recording is not None:
            self._record(arrival, state, health, pose, jaw, reads)
        self._before, self._before_at = state, arrival.at_s
        if self._latency_at is None:
            self._latency_at = arrival.at_s
        elif arrival.at_s - self._latency_at >= LATENCY_PERIOD_S:
            self._latency_line(arrival)

    def _new_reads(self, state: Any) -> list[Any]:
        """The single joint reads this ROBOT_STATE reports for the first time.

        A read is known by its joint and the end of its read. Each is counted
        with its duration, the end of the read less its start, both of the
        Thor's monotonic clock.
        """
        samples = sorted((*state.recent, *state.joints), key=lambda read: read.t_end)
        new = []
        for read in samples:
            if not read.t_end > self._last_read.get(read.joint, -math.inf):
                continue
            self._last_read[read.joint] = read.t_end
            if not read.valid:
                # A joint without any reading is sent with the times 0.
                self.reads_failed += int(read.t_end > 0)
                continue
            took = self._add("joint_read_s", read.t_end - read.t_start)
            if took is not None:
                self._second["reads"].add(took)
                self._state_reads.add(took)
                new.append(read)
        return new

    def _move_seen(
        self, state: Any, health: dict[str, Any], reads: list[Any]
    ) -> float | None:
        """Count this ROBOT_STATE for the agent's move it names, if it names one.

        Returns the age of the camera frame behind the permit the Thor holds
        for that move, as the Thor counts it; None without one.
        """
        move_id = state.move_id
        if not move_id or state.owner != "agent":
            return None  # no move, or a person's drive or the return to rest
        seen = self._moves.get(move_id)
        if seen is None:
            seen = self._moves[move_id] = {
                "states": 0,
                "missed_before": self._missed,
                "first": None,  # (the Thor's clock, setpoints sent), the first
                "last": None,  # the same when the count last rose
                "reads": _Bulk(),
                "permit_lead_min": None,
                "frame_age_first_s": None,
                "frame_age_max_s": None,
            }
            while len(self._moves) > _MOVES_TRACKED:
                del self._moves[next(iter(self._moves))]
        seen["states"] += 1
        index = int(state.move_index)
        if index >= 1:
            if seen["first"] is None:
                seen["first"] = (state.clock_mono_s, index)
            if seen["last"] is None or index > seen["last"][1]:
                seen["last"] = (state.clock_mono_s, index)
        for read in reads:
            seen["reads"].add(read.t_end - read.t_start)
        permit = health.get("permit")
        if not isinstance(permit, dict) or permit.get("move_id") != move_id:
            return None
        through = _number(permit.get("through_index"))
        if through is not None:
            # Setpoints the held permit covers beyond those sent.
            lead = int(through) + 1 - index
            least = seen["permit_lead_min"]
            seen["permit_lead_min"] = lead if least is None else min(least, lead)
        age = _number(permit.get("evidence_age_s"))
        if age is not None:
            if seen["frame_age_first_s"] is None:
                seen["frame_age_first_s"] = age
            seen["frame_age_max_s"] = max(seen["frame_age_max_s"] or age, age)
        return age

    def _state_line(
        self,
        arrival: Arrival,
        state: Any,
        health: dict[str, Any],
        pose: tuple[list[float], float] | None,
        jaw: dict[str, Any],
        above: Any,
    ) -> None:
        self._state_line_at = arrival.at_s
        reads, self._state_reads = self._state_reads, _Bulk()
        self._files[STATES_FILE].write(
            {
                "arrived_mono_s": round(arrival.at_s, 6),
                "seq": state.seq,
                "clock_mono_s": state.clock_mono_s,
                "clock_wall_s": state.clock_wall_s,
                "transit_above_shortest_s": _rounded(above),
                "q": None if pose is None else pose[0],
                "sample_span_s": None if pose is None else round(pose[1], 6),
                "reading_age_s": _rounded(health.get("reading_age_s")),
                "reads": reads.row(),
                "target_q": state.target_q,
                "target_t": state.target_t,
                "jaw": jaw,
                "motion": state.motion,
                "move_id": state.move_id,
                "move_index": state.move_index,
                "move_total": state.move_total,
                "move_stage": state.move_stage,
                "owner": state.owner,
                "stop_number": state.stop_number,
                "stop_level": state.stop_level,
                "missed_deadlines": health.get("missed_deadlines"),
                "permit": health.get("permit"),
                "view_age_s": state.view_age_s,
            }
        )

    def _record(
        self,
        arrival: Arrival,
        state: Any,
        health: dict[str, Any],
        pose: tuple[list[float], float] | None,
        jaw: dict[str, Any],
        reads: list[Any],
    ) -> None:
        """The events of the experiment's debug recording for one ROBOT_STATE."""
        recording = self._recording
        age = _number(health.get("reading_age_s"))
        since = state.clock_mono_s - self._feedback_clock
        due = since >= FEEDBACK_PERIOD_S - _EPSILON_S
        if pose is not None and age is not None and due:
            self._feedback_clock = state.clock_mono_s
            q, span = pose
            recording.record(
                "robot.feedback",
                lambda: {
                    "seq": int(state.seq),
                    "state": {"q": q, "t": arrival.at_s - age},
                    "read_failures": 0,
                    "source": FEEDBACK_SOURCE,
                    "clock": FEEDBACK_CLOCK,
                    "sample_span_s": span,
                },
            )
        recording.record(
            STATE_EVENT,
            lambda: {
                "seq": int(state.seq),
                "clock_mono_s": float(state.clock_mono_s),
                "arrived_mono_s": arrival.at_s,
                # Each single joint read once: joint, value, start, end.
                "reads": [
                    [int(read.joint), float(read.value), read.t_start, read.t_end]
                    for read in reads
                ],
                "target_q": state.target_q,
                "target_t": state.target_t,
                "jaw": jaw,
                "move_id": state.move_id,
                "move_index": int(state.move_index),
                "move_total": int(state.move_total),
                "missed_deadlines": _number(health.get("missed_deadlines")),
            },
        )

    def _latency_line(self, arrival: Arrival | None) -> None:
        """One line of LATENCY_FILE for what came since the last one."""
        second, self._second = self._second, self._new_second()
        if arrival is None or not second["states"]:
            self._second = second  # nothing arrived: what was noted waits
            return
        self._latency_at = arrival.at_s
        transit = sorted(second["transit"])
        state = self._before
        health = state.health if isinstance(state.health, dict) else {}
        self._files[LATENCY_FILE].write(
            {
                "utc": self._utc(arrival.at_s),
                "mono_s": round(arrival.at_s, 6),
                "states": second["states"],
                "seq_missing": second["seq_missing"],
                "arrival_gap_max_s": _rounded(second["arrival_gap_max_s"]),
                "transit_above_shortest_s": {
                    "median": _rounded(transit[(len(transit) - 1) // 2])
                    if transit
                    else None,
                    "longest": _rounded(transit[-1]) if transit else None,
                },
                "clock_offset_s": round(arrival.offset_s, 6),
                "clock_spread_s": round(arrival.spread_s, 6),
                "clock_samples": arrival.samples,
                "clock_steps": getattr(self._link, "clock_steps", None),
                "joint_reads": second["reads"].row(),
                "read_gaps": second["read_gaps"],
                "reading_age_s": _rounded(health.get("reading_age_s")),
                "missed_deadlines": health.get("missed_deadlines"),
                "view_age_s": state.view_age_s,
                "permit_frame_age_max_s": _rounded(second["permit_frame_age_max_s"]),
                "first_result_after_s": second["first_result_after_s"],
                "jaw_result_s": second["jaw_result_s"],
                "stop_shown_s": second["stop_shown_s"],
                "dropped": self.dropped,
            }
        )

    # A move, a jaw command, a camera frame.

    def _move(self, noted_at: float, facts: dict[str, Any]) -> None:
        row: dict[str, Any] = {"utc": self._utc(noted_at), **facts}
        for key, value in facts.items():
            if isinstance(value, float):
                row[key] = round(value, 6)
        # The stretches between the points of the permit loop. Settling is
        # known only when a ROBOT_STATE showed every setpoint as sent.
        first = _number(facts.get("first_sent_after_s"))
        rose = _number(facts.get("last_progress_after_s"))
        whole = _number(facts.get("all_sent_after_s"))
        result = _number(facts.get("result_after_s"))
        row["sending_s"] = row["settle_s"] = row["after_last_progress_s"] = None
        if first is not None and rose is not None:
            row["sending_s"] = round(rose - first, 6)
        if whole is not None and result is not None:
            row["settle_s"] = round(result - whole, 6)
        if rose is not None and result is not None:
            row["after_last_progress_s"] = round(result - rose, 6)
        seen = self._moves.pop(str(facts.get("move_id")), None)
        row["thor"] = None if seen is None else self._thor_move(seen)
        for key in _MOVE_DURATIONS:
            self._add(key, row.get(key))
        accepted = _number(row.get("first_result_after_s"))
        if accepted is not None:
            self._second["first_result_after_s"].append(accepted)
        self._files[MOVES_FILE].write(row)

    def _thor_move(self, seen: dict[str, Any]) -> dict[str, Any]:
        """What the ROBOT_STATEs that named a move say of it, on the Thor's clock."""
        first, last = seen["first"], seen["last"]
        pace = None
        if first is not None and last[0] > first[0] and last[1] > first[1]:
            pace = (last[1] - first[1]) / (last[0] - first[0])
            self.pace.add(pace)
        missed = None
        if self._missed is not None and seen["missed_before"] is not None:
            missed = int(self._missed - seen["missed_before"])
        return {
            "states": seen["states"],
            "setpoints_per_s": _rounded(pace, 3),
            "missed_deadlines": missed,
            "joint_reads": seen["reads"].row(),
            "permit_lead_min": seen["permit_lead_min"],
            "frame_age_first_s": _rounded(seen["frame_age_first_s"]),
            "frame_age_max_s": _rounded(seen["frame_age_max_s"]),
        }

    def _jaw_seen(self, state: Any, jaw: dict[str, Any]) -> None:
        """Keep the jaw reading of a ROBOT_STATE that came while a command runs."""
        running = self._jaw
        running["states"] += 1
        opening, target = _number(jaw["opening"]), _number(jaw["last_target"])
        if jaw["valid"] and opening is not None and target is not None:
            gap = abs(opening - target)
            running["gap_max"] = max(running["gap_max"] or 0.0, gap)
        if len(running["trace"]) < JAW_TRACE_KEPT:
            running["trace"].append([state.clock_mono_s, jaw["opening"], target])
        running["load"] = jaw["load"]

    def _jaw_done(self, noted_at: float, facts: dict[str, Any]) -> None:
        running, self._jaw = self._jaw or {"trace": [], "states": 0}, None
        row: dict[str, Any] = {"utc": self._utc(noted_at), **facts}
        for key, value in facts.items():
            if isinstance(value, float):
                row[key] = round(value, 6)
        row["thor"] = {
            "states": running["states"],
            "target_gap_max": _rounded(running.get("gap_max")),
            "load": running.get("load"),
            # Per ROBOT_STATE: the Thor's clock, the opening, the last target.
            "trace": running["trace"],
        }
        took = self._add("jaw_result_s", facts.get("sent_to_result_s"))
        if took is not None:
            self._second["jaw_result_s"].append(round(took, 6))
        self._files[JAWS_FILE].write(row)

    def _frame(self, returned_at: float, frame_t: Any, capture: dict[str, Any]) -> None:
        age = None
        if _number(frame_t) is not None:
            age = self._add("frame_age_at_return_s", returned_at - float(frame_t))
        self._files[CAMERA_FILE].write(
            {
                "utc": self._utc(returned_at),
                "returned_mono_s": round(returned_at, 6),
                "frame_t": _number(frame_t),
                "age_at_return_s": _rounded(age),
                "capture": capture,
            }
        )


def _ms(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds * 1e3:.2f}"


def _pose_of(state: Any) -> tuple[list[float], float] | None:
    """(the measured joints, their sample span) of a ROBOT_STATE; None if not whole.

    The sample span is the newest end of a read less the oldest among the
    joints' newest readings, on the Thor's clock: the readings are not of one
    moment.
    """
    joints = sorted(state.joints, key=lambda read: read.joint)
    if not joints or any(
        not read.valid or _number(read.value) is None for read in joints
    ):
        return None
    ends = [read.t_end for read in joints]
    return [float(read.value) for read in joints], max(ends) - min(ends)


def _jaw_row(jaws: Any) -> dict[str, Any]:
    """The jaw reading of a ROBOT_STATE: the opening is 0 closed to 1 open."""
    return {
        "opening": jaws.opening,
        "t": jaws.t,
        "valid": bool(jaws.valid),
        "last_target": jaws.last_target,
        "load": jaws.load,
    }


def _measure_line(run_dir: Path) -> str:
    """What every run measures, and what it cannot; said at the start."""
    return (
        f"measurements: {', '.join(MEASURE_FILES)} in {run_dir}, and "
        f"{SUMMARY_FILE} when the program ends, which is also printed. The "
        "single joint reads and the setpoint pace in them are what the robot "
        "app reports in ROBOT_STATE, on the Thor's clock. Not measured here: "
        "how long the Thor takes to send one setpoint to the motors, and the "
        "camera's capture and encoding apart from the network"
    )


@contextmanager
def experiment_sessions(
    options: RunOptions, run_dir: Path, ask: Callable[[str], str]
) -> Iterator[tuple[Callable[[str], str], Any, Any]]:
    """The experiment's recordings around a run, in its own launcher's order.

    The timing, then the model transcript, then the debug recording, each
    only when its option is on; they end in the reverse order. Yields the
    question the task gate asks with, which is timed while the timing runs,
    the timing's recorder or None, and the debug recorder or None.
    """
    parts = _TimedParts(ask)
    timing = None
    recording = None
    with ExitStack() as stack:
        if options.batch_milestones:
            _batched_milestones()  # missing: said before anything is attached
        if options.timing:
            session = experiment_module("thor_timing").timing_session
            timing = stack.enter_context(session(run_dir, runtime_module=parts))
            if timing is not None:
                _link_spans(stack, timing)
        if options.model_trace:
            session = experiment_module("thor_model_trace").model_trace_session
            stack.enter_context(session(run_dir, timing_recorder=timing))
        if options.record_debug:
            session = experiment_module("thor_recording").recording_session
            limit = int(options.debug_max_gib * 1024**3)
            recording = stack.enter_context(session(run_dir, max_bytes=limit))
        yield parts.operator_answer, timing, recording


def _option_lines(options: RunOptions, run_dir: Path) -> list[str]:
    """What this run records, and what it cannot; said at the start."""
    lines = []
    if options.timing:
        lines.append(
            f"timing: {run_dir / 'timing.jsonl'}; timing-report.txt and "
            "timeline.html are written when the program ends. Not timed here: "
            "the motor reads and sends and the camera's capture, which run on "
            "the Thor. A move is timed from its MOVE to its final RESULT"
        )
    if options.model_trace:
        lines.append(f"model transcript: {run_dir / 'model-trace' / 'report.txt'}")
    if options.record_debug:
        lines.append(
            f"debug recording: {run_dir / 'debug'}, at most "
            f"{options.debug_max_gib:g} GiB; ./run_viewer.sh shows it in a "
            "browser while the run goes on. Not recorded here: the motor "
            "commands and the single motor reads (robot.call, robot.return, "
            "mask.state_source), which are the robot app's on the Thor. Its "
            "joint feedback, robot.feedback, is what this program read from "
            "ROBOT_STATE, timed at receipt on this machine. The camera frames in "
            "it are the ones APAI took on this machine"
        )
    if options.batch_milestones:
        lines.append(
            "milestone batching: one model request for the milestones that are "
            "checked on one camera frame"
        )
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.agent",
        description=(
            "The agent program of the agentic remote execution example: APAI's "
            "agent on the workstation, driving the arm through the robot link "
            "service. Ctrl+C is STOP."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="the agent's YAML file"
    )
    parser.add_argument("--task", default=None, metavar="TEXT", help="run one task")
    parser.add_argument(
        "--interactive", action="store_true", help="type one task after another"
    )
    parser.add_argument("--max-steps", type=int, default=30, metavar="N")
    parser.add_argument(
        "--camera",
        default=agent_config.CAMERA_LINK,
        metavar="link|mock|recorded:<dir>",
        help="the camera connection (default), APAI's mock camera, or a folder "
        "of recorded stereo pairs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load and check everything, connect to nothing, print the "
        "resolved values and the calibration id",
    )
    parser.add_argument(
        "--run-dir", default=None, metavar="PATH", help="this run's folder"
    )
    parser.add_argument(
        "--motion-approval",
        default=agent_config.APPROVAL_TASK,
        choices=agent_config.APPROVALS,
        help="task (default): one typed yes before the first motion tool of "
        "each task, which is what the program does. tool is not built",
    )
    # Arguments of the experiment's launcher, so that its command line can be
    # reused.
    parser.add_argument(
        "--operate",
        action="store_true",
        help="every run may move the arm after the typed yes; needs a terminal",
    )
    parser.add_argument(
        "--depth-backend",
        default=None,
        metavar="NAME",
        help="the stereo matcher of this run, in place of the camera profile's; "
        "with the full stack one of the experiment's stereo matchers",
    )
    parser.add_argument(
        "--speed-profile",
        default=None,
        choices=agent_config.SPEED_PROFILES,
        help="full stack: the joint speed cap. reference (default): the one of "
        "the experiment's configs/rebot_workstation.yaml; conservative: "
        f"{agent_config.CONSERVATIVE_SPEED:g} rad/s. Never above the Thor's "
        "agent limit (thor.agent_max_joint_speed)",
    )
    parser.add_argument(
        "--mounting",
        default=None,
        metavar="PATH",
        help="the mounting record (the accepted fit of the camera's place) "
        "whose hand-eye file this run uses; default: camera.hand_eye_json, "
        "else the experiment's newest record",
    )
    parser.add_argument(
        "--batch-milestones",
        action="store_true",
        help="full stack: the experiment's milestone batching, one model "
        "request for the milestones that are checked on one camera frame",
    )
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="full stack: run without the experiment's timing",
    )
    parser.add_argument(
        "--record-debug",
        action="store_true",
        help="full stack: the experiment's debug recording (obstacle map "
        "inputs, masks, map revisions) in the run folder; needs the timing",
    )
    parser.add_argument(
        "--debug-max-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="the debug recording's size on disk at which it stops (default "
        f"{agent_config.DEBUG_MAX_GIB:g})",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.lower,
        choices=agent_config.LOG_LEVELS,
        metavar="L",
        help="console log level (default: the file's)",
    )
    return parser


def _apai_on_path() -> None:
    """Put APAI's src folder on the import path when the variable names it."""
    source = os.environ.get(APAI_SRC_ENV)
    if source and source not in sys.path:
        sys.path.insert(0, source)


def _run_folder(settings: AgentSettings, given: str | None) -> Path:
    if given:
        folder = agent_config.user_path("--run-dir", given)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        folder = settings.state_dir / "runs" / f"{stamp}-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _state_folder(settings: AgentSettings) -> None:
    """Make the state folder; git ignores everything in it."""
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    ignore = settings.state_dir / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")


def _config_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """The experiment launcher's arguments that APAI's configuration takes."""
    return {
        "speed_profile": args.speed_profile,
        "depth_backend": args.depth_backend,
        "mounting": args.mounting,
    }


def _check(
    settings: AgentSettings, options: RunOptions, args: argparse.Namespace, say: Say
) -> int:
    camera = args.camera
    with tempfile.TemporaryDirectory(prefix="agentic-remote-check-") as folder:
        cfg = agent_config.build_apai_config(
            settings,
            folder,
            camera=camera,
            batch_milestones=options.batch_milestones,
            **_config_arguments(args),
        )
        report = agent_config.summary(
            settings,
            cfg,
            camera,
            speed_profile=args.speed_profile,
            mounting=args.mounting,
        )
    report["run_options"] = asdict(options)
    say(json.dumps(report, indent=2, default=str))
    return EXIT_OK


def _start_lines(
    settings: AgentSettings, cfg: Any, args: argparse.Namespace
) -> list[str]:
    """What the full stack took of the experiment's files; said at the start.

    The joint speed is said once the robot app's WELCOME is here
    (agent_config.agree_joint_speed).
    """
    if settings.minimal:
        return []
    lines = [f"arm {cfg.arm.get('name')}; the motors are switched by the robot app"]
    if agent_config.camera_kind(args.camera)[0] != agent_config.CAMERA_MOCK:
        found = agent_config.resolve_mounting(settings, args.mounting)
        where = "camera.hand_eye_json"
        if found.record:
            where = f"mounting record {found.record}"
        lines.append(
            f"hand-eye file {found.hand_eye_json} (sha256 {found.sha256[:12]}), "
            f"from {where}; stereo matcher {cfg.camera.get('depth_backend')}"
        )
    return lines


def _settle(link: AgentLink, timeout_s: float = SETTLE_S) -> bool:
    """Wait until ROBOT_STATE no longer shows the agent as control owner.

    A local socket closed while it does makes the robot link service ask for
    help in the agent's name ("agent program gone"). False when the agent is
    still the owner after timeout_s.
    """
    end = time.monotonic() + timeout_s
    while True:
        try:
            if link.state().owner != "agent":
                return True
        except (LinkLost, StaleState):
            return True
        left = end - time.monotonic()
        if left <= 0:
            _log.warning("the agent still owns the arm; closing the link anyway")
            return False
        link.wait_change(min(left, 0.1))


def _end_session(gate: TaskGate, link: AgentLink) -> None:
    """Offer the session's end once: the arm to rest, the motors off there.

    The functional launcher returns to rest and releases torque by itself
    when it ends. Here the operator is asked first, and the robot app does
    it. A Ctrl+C while the arm returns is a STOP: the arm holds, motors on.
    """
    try:
        gate.end_session()
    except KeyboardInterrupt:
        link.stop(STOP_REASON, AGENT_TERMINAL)
        _complain("interrupted; STOP was sent: the arm holds with the motors on")
    except Exception as problem:
        _log.debug("the end of the session failed", exc_info=True)
        _complain(
            "the end of the session was not offered "
            f"({type(problem).__name__}: {problem}): the arm holds with the "
            "motors on"
        )


class _Announcer:
    """Says what a WELCOME declares, once per robot, driver and session.

    Called from the program's thread and from the link's receive thread.
    """

    def __init__(self, say: Say) -> None:
        self._say = say
        self._lock = threading.Lock()
        self._last = ""

    def __call__(self, welcome: Any) -> None:
        driver = getattr(welcome, "driver", "")
        line = (
            f"robot connection up: robot {welcome.robot_id}, profile "
            f"{welcome.profile_id}, driver {driver or '(not declared)'}, session "
            f"{welcome.session_id}, camera calibration "
            f"{welcome.camera_calibration_id or '(unnamed)'}"
        )
        with self._lock:
            if line == self._last:
                return
            self._last = line
        self._say(line)


def _check_arguments(args: argparse.Namespace) -> None:
    """ConfigError for an argument APAI would take for something else."""
    if args.task is not None and not args.task.strip():
        raise ConfigError("--task must name a task; it is empty")
    if args.max_steps < 1:
        raise ConfigError(f"--max-steps must be at least 1, not {args.max_steps}")


def _operator_terminal(args: argparse.Namespace, ask: Callable[[str], str]) -> None:
    """ConfigError for --operate without the operator's terminal.

    As the experiment's launcher, at the start: the typed yes of a task could
    not be given. Only when the question is this program's own (terminal_ask).
    """
    if not args.operate or ask is not terminal_ask:
        return
    from apai.control import arm_hold

    if arm_hold.agent_environment() is not None or not arm_hold._interactive():
        raise ConfigError(
            "--operate requires the operator's interactive terminal, outside an "
            "agent environment"
        )


def _run_options(args: argparse.Namespace, settings: AgentSettings) -> RunOptions:
    """What the run takes of the experiment's launcher; ConfigError when refused."""
    return agent_config.run_options(
        settings,
        motion_approval=args.motion_approval,
        batch_milestones=args.batch_milestones,
        no_timing=args.no_timing,
        record_debug=args.record_debug,
        debug_max_gib=args.debug_max_gib,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    attach: Callable[..., AgentLink] | None = None,
    ask: Callable[[str], str] = terminal_ask,
    say: Say = _say,
    source: Any = None,
    task_input: Callable[[str], str] = terminal_input,
) -> int:
    """Run the agent program; returns its exit code.

    attach, ask, task_input and source stand in for the robot link service's
    socket, the operator's terminal and the stereo pair source in the tests.
    """
    pin_gpu()
    sys.dont_write_bytecode = True  # nothing lands in APAI or the experiment
    args = _parser().parse_args(argv)
    try:
        _check_arguments(args)
        settings = agent_config.load_settings(args.config)
        options = _run_options(args, settings)
        if settings.minimal and args.speed_profile is not None:
            agent_config.joint_speed(settings, args.speed_profile)  # refused
        invented = agent_config.invented_times(settings, args.camera)
        agent_config.refused_environment()
        caches = agent_config.experiment_caches(settings)
    except ConfigError as problem:
        _complain(str(problem))
        return EXIT_NOT_RUN
    # Before any module of the experiment is imported, which would name its
    # own folder. Taken back at the end: a test runs main() again.
    os.environ.update(caches)
    try:
        return _program(
            args, settings, options, invented, attach, ask, say, source, task_input
        )
    finally:
        for name in caches:
            os.environ.pop(name, None)


def _program(
    args: argparse.Namespace,
    settings: AgentSettings,
    options: RunOptions,
    invented: bool,
    attach: Callable[..., AgentLink] | None,
    ask: Callable[[str], str],
    say: Say,
    source: Any,
    task_input: Callable[[str], str],
) -> int:
    """main() from where the settings and the environment are taken."""
    set_console_level(args.log_level or settings.log_level)
    _apai_on_path()
    try:
        import apai.apps.demo as demo
    except ImportError as missing:
        _complain(f"APAI cannot be imported ({missing}); set {APAI_SRC_ENV}")
        return EXIT_NOT_RUN
    try:
        _operator_terminal(args, ask)
        if args.check:
            return _check(settings, options, args, lambda text: print(text, flush=True))
        _state_folder(settings)
        agent_config.redirect_apai_stores(settings)
        run_dir = _run_folder(settings, args.run_dir)
        cfg = agent_config.build_apai_config(
            settings,
            run_dir,
            camera=args.camera,
            batch_milestones=options.batch_milestones,
            **_config_arguments(args),
        )
        calibration = agent_config.calibration_id(settings, args.camera, args.mounting)
        told = _start_lines(settings, cfg, args)
        told += agent_config.seed_learned_stores(settings)
    except (ConfigError, OSError) as problem:
        _complain(str(problem))
        return EXIT_NOT_RUN
    agent_config.write_json(run_dir / "agent-settings.json", settings.as_dict())
    _record_resolved_config(run_dir, cfg)
    if options != RunOptions():
        agent_config.write_json(run_dir / "run-options.json", asdict(options))
    say(f"run folder {run_dir}; camera calibration {calibration or '(unnamed)'}")
    for line in told:
        say(line)
    if os.environ.get("APAI_REQUIRE_CUDA", "").strip() == "1":
        say(
            "WARNING: APAI_REQUIRE_CUDA is 1 in the environment: APAI then keeps "
            "an object's points as CUDA tensors and its grasp code fails on "
            "them. The experiment's launcher never sets it: unset it"
        )
    if invented:
        say(
            f"WARNING: the camera frames of --camera {args.camera} carry invented "
            "capture times (camera.allow_invented_times is true): a permit rests "
            "on evidence that nobody captured. For a simulated or mock arm only, "
            "never against a real arm: the program ends unless the robot app "
            "declares a simulated driver"
        )
    if settings.experiment.grid == agent_config.GRID_CLEAR:
        say(
            "WARNING: experiment.grid is clear: the obstacle grid is a stand-in "
            "with nothing near; no move is checked against obstacles. For a "
            "simulated or mock arm only: the program ends unless the robot app "
            "declares a simulated driver"
        )

    for line in _option_lines(options, run_dir):
        say(line)
    # The recordings stand around the whole robot connection: a STOP and the
    # close of the link come before any of them writes its report.
    code = EXIT_FAILED
    try:
        with ExitStack() as sessions:
            try:
                recordings = experiment_sessions(options, run_dir, ask)
                ask, timing, recording = sessions.enter_context(recordings)
            except Exception as problem:
                _log.debug("the recordings did not start", exc_info=True)
                _complain(
                    "the experiment's recordings cannot start "
                    f"({type(problem).__name__}: {problem}); nothing was attached"
                )
                return EXIT_NOT_RUN
            # It ends before the link is closed, so before the debug recording,
            # which takes its events; its summary is said behind the close.
            say(_measure_line(run_dir))
            measures = Measures(run_dir, recording)
            try:
                code = _drive(
                    demo,
                    args,
                    settings,
                    (cfg, run_dir, calibration, invented),
                    attach,
                    ask,
                    say,
                    source,
                    task_input,
                    timing,
                    measures,
                )
            finally:
                measures.close(say)
    except Exception as problem:
        _log.debug("a recording did not end", exc_info=True)
        _complain(
            "a recording of this run did not end as it should: "
            f"{type(problem).__name__}: {problem}"
        )
    return code


def _drive(
    demo: Any,
    args: argparse.Namespace,
    settings: AgentSettings,
    run: tuple[Any, Path, str, bool],
    attach: Callable[..., AgentLink] | None,
    ask: Callable[[str], str],
    say: Say,
    source: Any,
    task_input: Callable[[str], str],
    timing: Any,
    measures: Measures | None = None,
) -> int:
    """Attach to the robot link service, run APAI's command line, close the link.

    run is (APAI's configuration, the run folder, the camera calibration's
    name, whether the camera's capture times are invented).
    """
    cfg, run_dir, calibration, invented = run
    link: AgentLink | None = None
    gate: TaskGate | None = None
    code, settled = EXIT_FAILED, True
    try:
        attach = attach or AgentLink.attach
        link = attach(
            settings.link.socket_path or None,
            timeout_s=settings.link.connect_timeout_s,
        )
        # Said for the first WELCOME and for each one that says something else,
        # as that of a restarted robot app does.
        announce = _Announcer(say)
        link.add_welcome_listener(announce)
        welcome = link.wait_ready(settings.link.connect_timeout_s)
        announce(welcome)
        driver = getattr(welcome, "driver", "")
        agent_config.check_robot_driver(settings, driver, invented)

        def driver_check(declared: str) -> str | None:
            try:
                agent_config.check_robot_driver(settings, declared, invented)
            except ConfigError as problem:
                return str(problem)
            return None

        agent_config.check_joint_count(settings, len(welcome.joint_names), cfg)
        # The speed the agent plans at: never above the Thor's own limit.
        built_for = float(cfg.safety.max_joint_vel)
        settings, cfg, speed_line = agent_config.agree_joint_speed(
            settings, cfg, agent_config.thor_agent_limit(welcome), args.speed_profile
        )
        say(speed_line)
        if float(cfg.safety.max_joint_vel) != built_for:
            _record_resolved_config(run_dir, cfg)
        if measures is not None:
            measures.attach(link)
        frame_times = FrameTimes()
        joint_feed = None
        if agent_config.camera_kind(args.camera)[0] != agent_config.CAMERA_MOCK:
            joint_feed = JointFeed(len(welcome.joint_names))
            link.add_state_listener(joint_feed.on_state)
        gate = TaskGate(
            link,
            None,
            replace(settings, task=agent_config.task_settings(settings, cfg)),
            ask=ask,
            say=say,
            motion_skills=_motion_skills(),
            read_only_tools=READ_ONLY_TOOLS,
            driver_check=driver_check,
        )
        arguments = task_arguments(args, settings, run_dir)
        with agent_hooks(
            cfg,
            settings,
            link,
            gate,
            run_dir,
            frame_times=frame_times,
            joint_feed=joint_feed,
            source=source,
            calibration=calibration,
            invented_times=invented,
            say=say,
            task_input=task_input,
            timing=timing,
            measures=measures,
        ):
            status = demo.main(arguments)
        if status == 0 and not gate.restart_told and not gate.wrong_robot_app:
            code = EXIT_OK
        if code == EXIT_OK and args.interactive and gate.failed_tasks:
            say(f"{gate.failed_tasks} task(s) of this session ended without success")
            code = EXIT_FAILED
    except KeyboardInterrupt:
        # A second Ctrl+C, or one before APAI's own handling stood.
        if link is not None:
            link.stop(STOP_REASON, AGENT_TERMINAL)
        _complain("interrupted; STOP was sent")
        code = EXIT_INTERRUPTED
    except (LinkLost, TimeoutError) as problem:
        _complain(f"no robot connection: {problem}")
    except ConfigError as problem:
        _complain(str(problem))
    except Exception as problem:
        _log.debug("the agent program failed", exc_info=True)
        _complain(f"{type(problem).__name__}: {problem}")
    finally:
        if link is not None:
            if gate is not None:
                gate.return_late_ownership(SETTLE_S)
            settled = _settle(link)
            if gate is not None and settled:
                _end_session(gate, link)
            if measures is not None:
                # Before the socket closes: nothing of it is left to do behind.
                measures.finish()
            link.close()
    if not settled:
        _complain(
            "the agent still owned the arm when the program ended: the robot "
            "side ends that ownership; look at the Thor terminal"
        )
    elif gate is not None and gate.end_unanswered:
        _complain("an owner request that ended a task was not answered")
    if code == EXIT_OK and (not settled or (gate and gate.end_unanswered)):
        code = EXIT_FAILED
    return code
