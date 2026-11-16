# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program: APAI's agent on the workstation, the arm on the Thor.

main() attaches to the robot link service, builds APAI's configuration and
runs APAI's own command line (apai.apps.demo.main) inside agent_hooks(). The
hooks give APAI the robot link arm and the link's stereo camera in place of
its motor driver and camera, bind the check ahead to APAI's checker, and put
the task gate around every task and every skill. APAI is not edited: names of
its modules are patched for the lifetime of demo.main.

Ctrl+C at this terminal is a STOP. The program cuts no torque and sends no
return to rest: both belong to the robot app.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import logging
import os
import select
import sys
import tempfile
import termios
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ..console_log import set_console_level
from . import config as agent_config
from .arm_adapter import AGENT_TERMINAL, ARM_TYPE, arm_factory
from .checkahead import CheckAhead, FrameTimes
from .config import AgentSettings, ConfigError
from .link_client import AgentLink, LinkLost, StaleState
from .self_mask import EXPERIMENT_DIR_ENV, JointFeed, experiment_module, frame_context
from .task_gate import READ_ONLY_TOOLS, REST_SKILL, TaskGate

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


def terminal_input(prompt: str = "", idle: Callable[[], None] | None = None) -> str:
    """One line of the task terminal, read so that a signal is seen at once.

    idle() is called while no key comes, about every 50 ms.
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
    while True:
        readable, _, _ = select.select([fd], [], [], _INPUT_POLL_S)
        if not readable:
            if idle is not None:
                idle()
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
) -> Iterator[dict[str, Any]]:
    """APAI's demo.main, while this is open, runs on the robot link.

    Yields a dict the hooks fill: "runtime", "adapter", "harness", "grid";
    "invented_times" is there from the start. Only with invented_times does a
    camera frame that did not come over the camera link get an evidence time.
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
        link, agent_config.arm_settings(settings, calibration), joint_feed
    )
    stamps = LocalFrameStamps(link, frame_times) if invented_times else None
    mounting = _mounting_facts(settings) if full else {}
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
            usable = grid is not None and grid.wait_until_usable(
                settings.task.grid_wait_s
            )
            say(f"obstacle grid: {_grid_state(grid)}")
            if not usable:
                raise RuntimeError(
                    "the obstacle grid is not usable: no camera frame reached "
                    f"it within {settings.task.grid_wait_s:g} s"
                )
            say(
                "ready. Ctrl+C at this terminal is STOP; the return to rest and "
                "the torque release belong to the robot app"
            )
        except BaseException:
            demo.shutdown_runtime(runtime, raw)
            raise
        return runtime, raw

    def make_agent(*args: Any, **kwargs: Any) -> Any:
        agent = original_agent(*args, **kwargs)
        agent.run_task = gate.bind_run_task(agent.run_task)
        if full:
            agent._base_system_prompt += "\n" + experiment["guidance"]
            agent.set_harness(agent.harness)
        return agent

    def print_report(report: Any) -> None:
        row = asdict(report) if is_dataclass(report) else {"report": repr(report)}
        with (run_dir / "task-reports.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, default=str) + "\n")
        original_report(report)

    def specs(config: Any) -> list[Any]:
        motion = _motion_skills() - {REST_SKILL}
        return [
            item
            for item in original_specs(config)
            if item["name"] in READ_ONLY_TOOLS or item["name"] in motion
        ]

    def next_task(prompt: str = "") -> str:
        if gate.tell_restart():
            raise EOFError  # ends APAI's task prompt: no new task is taken
        gate.return_late_ownership()
        if task_input is terminal_input:
            return terminal_input(prompt, idle=gate.return_late_ownership)
        return task_input(prompt)

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


def _experiment_parts() -> dict[str, Any]:
    """The experiment's behaviour hooks; none of them touches a motor driver."""
    from apai.perception import colors

    rig = experiment_module("thor_calibration_rig")
    perception = experiment_module("thor_perception")
    placement = experiment_module("thor_placement")
    return {
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


def _mounting_facts(settings: AgentSettings) -> dict[str, Any]:
    """What the experiment's perception reads of a camera frame's mounting."""
    file = settings.camera.hand_eye_json
    try:
        digest = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    except OSError:
        digest = ""
    return {
        "camera_to_robot_base_calibrated": True,
        "mounting_calibration_file": file,
        "mounting_calibration_sha256": digest,
        "absolute_accuracy_verified": False,
    }


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
        folder = Path(given).expanduser()
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


def _check(settings: AgentSettings, camera: str, say: Say) -> int:
    with tempfile.TemporaryDirectory(prefix="agentic-remote-check-") as folder:
        cfg = agent_config.build_apai_config(settings, folder, camera=camera)
        report = agent_config.summary(settings, cfg, camera)
    say(json.dumps(report, indent=2, default=str))
    return EXIT_OK


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


def _check_arguments(args: argparse.Namespace) -> None:
    """ConfigError for an argument APAI would take for something else."""
    if args.task is not None and not args.task.strip():
        raise ConfigError("--task must name a task; it is empty")
    if args.max_steps < 1:
        raise ConfigError(f"--max-steps must be at least 1, not {args.max_steps}")


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
        invented = agent_config.invented_times(settings, args.camera)
        agent_config.refused_environment()
    except ConfigError as problem:
        _complain(str(problem))
        return EXIT_NOT_RUN
    set_console_level(args.log_level or settings.log_level)
    _apai_on_path()
    try:
        import apai.apps.demo as demo
    except ImportError as missing:
        _complain(f"APAI cannot be imported ({missing}); set {APAI_SRC_ENV}")
        return EXIT_NOT_RUN
    try:
        if args.check:
            return _check(settings, args.camera, lambda text: print(text, flush=True))
        _state_folder(settings)
        agent_config.redirect_apai_stores(settings)
        run_dir = _run_folder(settings, args.run_dir)
        cfg = agent_config.build_apai_config(settings, run_dir, camera=args.camera)
        calibration = agent_config.calibration_id(settings, args.camera)
    except (ConfigError, OSError) as problem:
        _complain(str(problem))
        return EXIT_NOT_RUN
    agent_config.write_json(run_dir / "agent-settings.json", settings.as_dict())
    say(f"run folder {run_dir}; camera calibration {calibration or '(unnamed)'}")
    if invented:
        say(
            f"WARNING: the camera frames of --camera {args.camera} carry invented "
            "capture times (camera.allow_invented_times is true): a permit rests "
            "on evidence that nobody captured. For a simulated or mock arm only, "
            "never against a real arm"
        )

    link: AgentLink | None = None
    gate: TaskGate | None = None
    code, settled = EXIT_FAILED, True
    try:
        attach = attach or AgentLink.attach
        link = attach(
            settings.link.socket_path or None,
            timeout_s=settings.link.connect_timeout_s,
        )
        welcome = link.wait_ready(settings.link.connect_timeout_s)
        say(
            f"robot connection up: robot {welcome.robot_id}, profile "
            f"{welcome.profile_id}, session {welcome.session_id}, camera "
            f"calibration {welcome.camera_calibration_id or '(unnamed)'}"
        )
        agent_config.check_joint_count(settings, len(welcome.joint_names))
        frame_times = FrameTimes()
        joint_feed = None
        if agent_config.camera_kind(args.camera)[0] != agent_config.CAMERA_MOCK:
            joint_feed = JointFeed(len(welcome.joint_names))
            link.add_state_listener(joint_feed.on_state)
        gate = TaskGate(
            link,
            None,
            settings,
            ask=ask,
            say=say,
            motion_skills=_motion_skills(),
            read_only_tools=READ_ONLY_TOOLS,
        )
        # The model is the configuration's; naming it keeps APAI from probing
        # for another one.
        model = settings.experiment.llm if settings.minimal else "codex_astra"
        arguments = ["--no-view", "--no-serve", "--run-dir", str(run_dir)]
        arguments += ["--llm", model, "--max-steps", str(args.max_steps)]
        if args.task is not None:
            arguments += ["--task", args.task]
        if args.interactive:
            arguments.append("--interactive")
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
        ):
            status = demo.main(arguments)
        if status == 0 and not gate.restart_told:
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
