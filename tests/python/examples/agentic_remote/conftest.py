# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The package under test lives in examples/agentic_remote/python and is not installed.

from __future__ import annotations

import contextlib
import importlib
import os
import socket
import sys
import threading
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

_tests_python = Path(__file__).resolve().parents[2]
if str(_tests_python) not in sys.path:
    sys.path.insert(0, str(_tests_python))

from repo_paths import repo_root  # noqa: E402

if TYPE_CHECKING:
    from isaaccapture_examples.agentic_remote.protocol.carrier import LoopbackCarrier
    from isaaccapture_examples.agentic_remote.protocol.messages import (
        Result,
        RobotState,
    )
    from isaaccapture_examples.agentic_remote.robot_app.app import RobotApp
    from isaaccapture_examples.agentic_remote.robot_app.clock import (
        Clock,
        FakeClock,
        SystemClock,
    )
    from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import CheckerSet
    from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm
    from isaaccapture_examples.agentic_remote.robot_app.terminal import ThorTerminal
    from isaaccapture_examples.agentic_remote.testing.client import WorkstationClient

# The src folder of an APAI checkout; tests marked "apai" skip without it.
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"
# What the tests marked "apai" import besides APAI: pinocchio, which comes with
# pin in the "apai" extra of pyproject.toml. APAI imports it late, at first use.
APAI_EXTRA_MODULES = ("pinocchio",)
# The experiment folder (read-only); tests marked "experiment" skip without it.
EXPERIMENT_DIR_ENV = "AGENTIC_REMOTE_EXPERIMENT_DIR"
# The camera_streamer example of the camera pull request (read-only).
CAMERA_PR_ENV = "AGENTIC_REMOTE_CAMERA_PR"

_apai_src = os.environ.get(APAI_SRC_ENV)
if _apai_src:
    sys.path.insert(0, _apai_src)
sys.path.insert(0, str(repo_root() / "examples" / "agentic_remote" / "python"))


def _apai_unavailable() -> str | None:
    """Why tests marked "apai" are skipped, or None when they run.

    They are skipped without a checkout and without a package of the "apai"
    extra. An import that fails any other way is left to the tests to show.
    """
    if not _apai_src:
        return f"{APAI_SRC_ENV} is not set"
    for name in ("apai", *APAI_EXTRA_MODULES):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as missing:
            if missing.name != name:
                continue
            if name == "apai":
                return f"{APAI_SRC_ENV}={_apai_src} has no package apai"
            return (
                "the apai extra of this folder's pyproject.toml is not installed: "
                f"{missing}"
            )
        except Exception:
            continue
    return None


def _link_unavailable() -> str | None:
    """Why tests marked "link" are skipped, or None when they run."""
    try:
        importlib.import_module("isaaccapture.remote_link")
    except ImportError as missing:
        return f"isaaccapture.remote_link cannot be imported: {missing}"
    return None


def _folder_unavailable(env: str) -> str | None:
    """Why the folder an environment variable names cannot be used, or None."""
    folder = os.environ.get(env)
    if not folder:
        return f"{env} is not set"
    if not Path(folder).is_dir():
        return f"{env}={folder} is not a folder"
    return None


def _experiment_unavailable() -> str | None:
    """Why tests marked "experiment" are skipped, or None when they run."""
    return _folder_unavailable(EXPERIMENT_DIR_ENV)


def _gpu_unavailable() -> str | None:
    """Why tests marked "gpu" are skipped, or None when they run.

    They need torch with CUDA in this interpreter; the tests' own environment
    has no torch, so there they are always skipped.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError as missing:
        return f"needs torch with CUDA: {missing}"
    if not torch.cuda.is_available():
        return "torch finds no CUDA device"
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for marker, unavailable in (
        ("apai", _apai_unavailable),
        ("link", _link_unavailable),
        ("experiment", _experiment_unavailable),
        ("gpu", _gpu_unavailable),
    ):
        marked = [item for item in items if item.get_closest_marker(marker)]
        reason = unavailable() if marked else None
        if reason is None:
            continue
        for item in marked:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session", autouse=True)
def _apai_files_out_of_home(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Keep APAI's hold markers and rest log out of the home folder."""
    if not _apai_src:
        yield
        return
    scratch = tmp_path_factory.mktemp("apai")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("APAI_ARM_HOLD_DIR", str(scratch / "arm_holds"))
        patch.setenv("APAI_REST_LOG", str(scratch / "rest_readings.jsonl"))
        yield


def _env_folder(env: str) -> Path:
    reason = _folder_unavailable(env)
    if reason is not None:
        pytest.skip(reason)
    return Path(os.environ[env]).resolve()


@pytest.fixture(scope="session")
def experiment_dir() -> Path:
    """The experiment folder; the test is skipped without it. Read-only."""
    return _env_folder(EXPERIMENT_DIR_ENV)


@pytest.fixture(scope="session")
def camera_pr_dir() -> Path:
    """The camera pull request's example folder; skipped without it. Read-only."""
    return _env_folder(CAMERA_PR_ENV)


def import_experiment(name: str) -> types.ModuleType:
    """Import a module of the experiment folder, for a test marked "experiment".

    The test is skipped, with the import error as the reason, when the folder
    is not set or the module needs a package this interpreter lacks.
    """
    folder = str(_env_folder(EXPERIMENT_DIR_ENV))
    if folder not in sys.path:
        sys.path.append(folder)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as missing:
        if missing.name == name:
            raise
        pytest.skip(f"the experiment's {name} cannot be imported here: {missing}")


# Where APAI keeps a store: the environment variable and the name in the folder.
_APAI_STORES = (
    ("APAI_BELIEFS_PATH", "beliefs.json"),
    ("APAI_ARM_HOLD_DIR", "arm_holds"),
    ("APAI_REST_LOG", "rest_readings.jsonl"),
    ("APAI_EPISODIC_PATH", "episodic.json"),
    ("APAI_GRASP_MEMORY_PATH", "grasp_memory.json"),
    ("APAI_ENVELOPE_PATH", "envelope.json"),
    ("APAI_PROGRAMS_PATH", "programs.jsonl"),
    ("APAI_ASPIRE_LIBRARY", "no-aspire-library"),
)
# Settings of a developer's shell that would change what APAI's runtime builds.
_APAI_AMBIENT = (
    "APAI_BELIEFS",
    "APAI_EPISODIC",
    "APAI_RUN_PURPOSE",
    "APAI_MEMORY_EMBEDDER",
    "APAI_MEMORY_EMBEDDER_MODEL",
    "APAI_PROGRAMS",
    "APAI_GRASP_EXECUTOR",
    "APAI_VLA_PORT",
    "APAI_JUDGE",
    "APAI_JUDGE_CONFIG",
    "APAI_OBSERVED_FINGER_GATE",
    "APAI_GRASPGENX_HOST",
    "APAI_OCCUPANCY",
    "APAI_START_STOPPED",
    "DISPLAY",
)


@contextlib.contextmanager
def apai_runtime_env(folder: Path) -> Iterator[Path]:
    """While open, a whole APAI runtime writes into `folder` and sees nothing ambient.

    For tests that build APAI's runtime (demo.build_runtime, demo.main): its
    stores go into this folder instead of the checkout or the home folder, its
    learned-notes library is an absent folder, and its grasp server port is one
    this process holds bound without listening, so the start-up probe is
    refused at once. The occupancy grid is left as the configuration has it.
    The run folder and memory.experience_path are the caller's to place here.
    """
    with contextlib.ExitStack() as stack:
        patch = stack.enter_context(pytest.MonkeyPatch.context())
        for env, leaf in _APAI_STORES:
            patch.setenv(env, str(folder / leaf))
        for env in _APAI_AMBIENT:
            patch.delenv(env, raising=False)
        notes = importlib.import_module("apai.skills.notes")
        patch.setattr(notes, "DEFAULT_LIBRARY", folder / "no-aspire-library")
        dead = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        dead.bind(("127.0.0.1", 0))
        patch.setenv("APAI_GRASPGENX_PORT", str(dead.getsockname()[1]))
        yield folder


@pytest.fixture
def apai_scratch(tmp_path: Path) -> Iterator[Path]:
    """apai_runtime_env() on the test's temporary folder; for tests marked "apai"."""
    with apai_runtime_env(tmp_path) as folder:
        yield folder


class FakeGridClient:
    """In place of APAI's OccupancyClient: a distance grid with nothing near.

    After the pattern of FakeClient and GridClient in APAI's test_occupancy.py;
    it needs no pyzmq and no msgpack. Every query answers with .grid (distances
    in metres, here 1 m everywhere) from .origin with cubes of .voxel metres; a
    test may replace .grid. .requests are the actions asked so far. Patch it in
    with patch.object(apai.perception.occupancy, "OccupancyClient",
    FakeGridClient) while the runtime is built, or hand an instance to
    OccupancyMap(client=...).
    """

    def __init__(self, host: str = "", port: int = 0, timeout_ms: int = 0) -> None:
        import numpy as np

        self.voxel = 0.05
        self.origin = np.full(3, -2.0)
        self.grid = np.full((80, 80, 80), 1.0, dtype=np.float32)
        self.requests: list[str] = []
        self.closed = False

    def request(self, payload: dict, timeout_ms: int | None = None) -> dict:
        import numpy as np

        action = payload["action"]
        self.requests.append(action)
        if action == "probe":
            return {
                "ok": True,
                "backend": "fake",
                "device": "cpu",
                "voxel": self.voxel,
                "esdf": True,
            }
        if action == "query":
            return {
                "points": np.empty((0, 3), dtype=np.float32),
                "grid": self.grid,
                "origin": self.origin,
                "voxel": self.voxel,
            }
        if action in ("integrate", "integrate_depth", "clear"):
            return {}
        # As an old bridge answers: APAI then falls back to an action it knows.
        occupancy = importlib.import_module("apai.perception.occupancy")
        raise occupancy.OccupancyError(
            f"occupancy bridge error: unknown action {action!r}"
        )

    def probe(self, timeout_ms: int = 300) -> dict:
        return self.request({"action": "probe"})

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def clock() -> FakeClock:
    from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock

    return FakeClock()


@pytest.fixture
def config() -> RobotAppConfig:
    from isaaccapture_examples.agentic_remote.robot_app.config import RobotAppConfig

    return RobotAppConfig.default()


@pytest.fixture
def carrier_pair() -> Iterator[tuple[LoopbackCarrier, LoopbackCarrier]]:
    """Both ends of a loopback link: (robot app end, workstation end)."""
    from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair

    pair = loopback_pair()
    yield pair
    for end in pair:
        end.close()


def check_released_at_rest(arm: SimArm, config: RobotAppConfig) -> None:
    """Fail unless every torque release the simulated arm recorded was at rest."""
    for pose in arm.released_at:
        off = max(abs(at - rest) for at, rest in zip(pose, config.rest.rest_q))
        assert off <= config.rest.rest_tol_rad, (
            f"torque was released {off:.3f} rad away from rest"
        )


@pytest.fixture
def sim_arm(
    clock: FakeClock, config: RobotAppConfig, request: pytest.FixtureRequest
) -> Iterator[SimArm]:
    """A simulated arm that starts at the rest pose.

    After the test, every torque release it recorded must have been at measured
    rest; a test of the simulated arm itself opts out with the marker
    "releases_away_from_rest".
    """
    from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm

    arm = SimArm(n_joints=config.n_joints, clock=clock, q0=list(config.rest.rest_q))
    yield arm
    if not request.node.get_closest_marker("releases_away_from_rest"):
        check_released_at_rest(arm, config)


@pytest.fixture
def checkers(clock: FakeClock, config: RobotAppConfig) -> CheckerSet:
    """One BasicChecker per control owner; the recovery one has no watchdog."""
    return basic_checkers(config, clock)


def basic_checkers(config: RobotAppConfig, clock: Clock) -> CheckerSet:
    """One BasicChecker per control owner on this clock; recovery has no watchdog."""
    from isaaccapture_examples.agentic_remote.robot_app.interfaces import CheckerSet
    from isaaccapture_examples.agentic_remote.robot_app.sim import BasicChecker

    limits, grasp = config.limits, config.grasp
    return CheckerSet(
        agent=BasicChecker(limits.agent, clock, grasp=grasp),
        human=BasicChecker(limits.human, clock, grasp=grasp),
        recovery=BasicChecker(limits.recovery, clock, watchdog_s=None, grasp=grasp),
    )


# Real seconds a test of the robot app gives another thread before it fails.
APP_WAIT_S = 5.0
# The robot app's dispatcher period in tests, in real seconds.
APP_STEP_S = 0.0002


class ThorSession:
    """A Thor terminal on a robot app: the test types its lines and reads them.

    interrupt() on .terminal is what Ctrl+C does. The terminal counts repeated
    refusal lines on `clock`, the app's fake clock.
    """

    def __init__(self, app: RobotApp, clock: FakeClock) -> None:
        from isaaccapture_examples.agentic_remote.robot_app.terminal import (
            ThorTerminal,
            TypedLines,
        )

        self._typed = TypedLines()
        self._written = threading.Condition()
        self.lines: list[str] = []  # every line the terminal wrote, in order
        self.terminal: ThorTerminal = ThorTerminal(
            app, self._typed, self._write, clock=clock
        )
        self._reader = threading.Thread(
            target=self.terminal.run, name="thor-session-input", daemon=True
        )
        self._reader.start()

    def written(self, start: str) -> list[str]:
        """The lines written so far that begin with `start`."""
        with self._written:
            return [line for line in self.lines if line.startswith(start)]

    def wait_line(self, start: str, count: int = 1) -> str:
        """Wait for the terminal's line number `count` that begins with `start`."""

        def found() -> list[str]:
            return [line for line in self.lines if line.startswith(start)]

        with self._written:
            assert self._written.wait_for(lambda: len(found()) >= count, APP_WAIT_S), (
                f"no line number {count} beginning {start!r}; written: {self.lines}"
            )
            return found()[count - 1]

    def ask(self, *typed: str, answer: str) -> str:
        """Type these lines; the next line written that begins with `answer`."""
        count = len(self.written(answer))
        for line in typed:
            self._typed.type(line)
        return self.wait_line(answer, count + 1)

    def close(self) -> None:
        """End the input, as a closed terminal does, and the terminal's threads."""
        self._typed.end()
        self._reader.join(APP_WAIT_S)
        self.terminal.close()

    def _write(self, line: str) -> None:
        with self._written:
            self.lines.append(line)
            self._written.notify_all()


class AppHarness:
    """A robot app on the simulated arm, with a workstation client on its link.

    The fake clock moves only when the test, a joint read, a jaw command or a
    running move moves it. advance() lets the app's dispatcher see every step.

    On a SystemClock (the real-time variant, see make_realtime_robot_app) a
    move takes its real length and waits for its permits; advance(),
    advance_until() and permit_every() are for the fake clock only.
    """

    def __init__(
        self,
        config: RobotAppConfig,
        arm: SimArm,
        checkers: CheckerSet,
        clock: Clock,
        confirm: Callable[[str], bool] | None = None,
        connected: bool = True,
        step_period_s: float | None = APP_STEP_S,
        app_options: Mapping[str, Any] | None = None,
    ) -> None:
        """connected=False builds the app with no robot connection; see connect().

        step_period_s None leaves the app its own dispatcher period. app_options
        are further keywords of RobotApp, such as driver or allow_reference.
        """
        from isaaccapture_examples.agentic_remote.robot_app.app import RobotApp
        from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock

        self.config, self.arm, self.checkers, self.clock = config, arm, checkers, clock
        # True when moves and waits take real time.
        self.real_time = not isinstance(clock, FakeClock)
        step = {} if step_period_s is None else {"step_period_s": step_period_s}
        # None until the app has a robot connection.
        self.robot_end: LoopbackCarrier | None = None
        self.workstation_end: LoopbackCarrier | None = None
        self.client: WorkstationClient | None = None
        self._ends: list[LoopbackCarrier] = []  # of every link made so far
        self._clients: list[WorkstationClient] = []
        if connected:
            self._new_link()
        self.app: RobotApp = RobotApp(
            config,
            arm,
            checkers,
            self.robot_end,
            clock,
            confirm=confirm,
            **step,
            **dict(app_options or {}),
        )
        # What a listener at the Thor terminal heard: (kind, text, data).
        self.local_events: list[tuple[str, str, dict[str, Any]]] = []
        self._heard = threading.Condition()
        self._releases: list[threading.Event] = []
        self._thor_sessions: list[ThorSession] = []
        self.app.add_event_listener(self._hear)

    def start(self) -> AppHarness:
        """Start the app and wait for its WELCOME and its first ROBOT_STATE."""
        self.app.start()
        if self.client is not None:
            self.client.wait_welcome(APP_WAIT_S)
            self.client.wait_state(timeout_s=APP_WAIT_S)
        return self

    def connect(self) -> WorkstationClient:
        """Attach a new loopback link to the app, with a new client on it.

        The app must take it: it has no robot connection, or has handled the
        loss of the last one. Returns the client, which has the WELCOME and a
        ROBOT_STATE of the new link.
        """
        earlier = self.robot_end, self.workstation_end, self.client
        self._new_link()
        if not self.app.attach_carrier(self.robot_end):
            self.robot_end, self.workstation_end, self.client = earlier
            raise AssertionError("the robot app did not take the new link")
        self.client.wait_welcome(APP_WAIT_S)
        self.client.wait_state(timeout_s=APP_WAIT_S)
        return self.client

    def lose_link(self) -> None:
        """Cut the robot connection and wait until the app has handled the loss."""
        lost = len(self.heard("connection"))
        self.robot_end.break_link()
        self.wait_local_events("connection", lost + 1)

    def _new_link(self) -> None:
        from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
        from isaaccapture_examples.agentic_remote.testing.client import (
            WorkstationClient,
        )

        self.robot_end, self.workstation_end = loopback_pair()
        self._ends += [self.robot_end, self.workstation_end]
        self.client = WorkstationClient(
            self.workstation_end, self.clock, quotable=self._shows_the_apps_numbers
        )
        self._clients.append(self.client)

    def thor_terminal(self) -> ThorSession:
        """A Thor terminal on this app; closed with the harness."""
        session = ThorSession(self.app, self.clock)
        self._thor_sessions.append(session)
        return session

    def close(self) -> None:
        for release in self._releases:
            release.set()  # no thread stays held in a paused call
        for session in self._thor_sessions:
            session.close()
        self.app.close()
        for client in self._clients:
            client.close()
        for end in self._ends:
            end.close()

    def step(self) -> None:
        """Wait for one dispatcher pass that began after this call."""
        assert self.app.wait_step(APP_WAIT_S), "the dispatcher made no pass"

    def advance(self, seconds: float, step_s: float = 1.0) -> None:
        """Move the fake clock on in steps; the dispatcher makes a pass after each."""
        assert not self.real_time, "advance() needs the fake clock"
        left = seconds
        while left > 1e-9:
            dt = min(step_s, left)
            self.clock.advance(dt)
            self.step()
            left -= dt

    def advance_until(
        self, done: Callable[[], bool], step_s: float = 1.0, limit_s: float = 200.0
    ) -> None:
        """Move the fake clock on in steps until done() holds; no step after that.

        Use it to run the assistance wait out: a step taken while the timed
        return is moving the arm would make that move late.
        """
        assert not self.real_time, "advance_until() needs the fake clock"
        passed = 0.0
        while not done():
            assert passed < limit_s, f"not done after {limit_s:g} s of fake time"
            self.clock.advance(step_s)
            self.step()
            passed += step_s

    def heard(self, kind: str) -> list[tuple[str, str, dict[str, Any]]]:
        """The events of this kind the Thor terminal's listener heard so far."""
        with self._heard:
            return [event for event in self.local_events if event[0] == kind]

    def idle(self) -> None:
        """Wait until the motion worker has nothing queued or running."""
        assert self.app.wait_motion_idle(APP_WAIT_S), "motion work did not end"

    def sent(self) -> list[list[float]]:
        """Every setpoint the arm was sent so far."""
        return [q for q, _ in self.arm.sent]

    def hold_at(self, pose: Sequence[float]) -> None:
        """Put the arm at this pose, holding, as after a move that ended there.

        arm.set_pose() alone leaves the app's motion state at "rest_hold".
        """
        from isaaccapture_examples.agentic_remote.robot_app.interfaces import (
            MotionState,
        )

        self.arm.set_pose(pose)
        self.app._motion.set(MotionState.HOLD)

    def new_task(self, timed_return: bool = True) -> Result:
        """The agent becomes control owner; the client then quotes the new numbers."""
        result = self.client.start_task(timed_return, APP_WAIT_S)
        assert result.status == "completed", result.reason
        return result

    def hold_move_after(self, count: int) -> tuple[threading.Event, threading.Event]:
        """Hold the motion worker once the arm was sent `count` setpoints in all.

        Returns (reached, release): reached is set when the worker waits inside
        that send; set release to let it go on.
        """
        reached, release = threading.Event(), threading.Event()
        self._releases.append(release)
        real = self.arm.send_target

        def send_target(q: Sequence[float]) -> None:
            real(q)
            if len(self.arm.sent) == count:
                reached.set()
                release.wait(APP_WAIT_S)

        self.arm.send_target = send_target
        return reached, release

    def permit_every(self, move_id: str, through_index: int, every: int = 50) -> None:
        """Renew the move's permit after every `every` setpoints, in step with it.

        The motion worker waits, with the fake clock standing still, until the
        dispatcher has taken the permit: a move longer than one permit's life
        then runs the same way every time.
        """
        real = self.arm.send_target
        first = len(self.arm.sent)

        def send_target(q: Sequence[float]) -> None:
            real(q)
            if (len(self.arm.sent) - first) % every == 0:
                self.client.permit(move_id, through_index)
                self.step()

        self.arm.send_target = send_target

    def hold_call(
        self, name: str, call: int = 1
    ) -> tuple[threading.Event, threading.Event]:
        """Hold the thread that makes the arm's call number `call` of `name`.

        It waits before the call is made. Returns (reached, release).
        """
        reached, release = threading.Event(), threading.Event()
        self._releases.append(release)
        real = getattr(self.arm, name)
        calls = [0]

        def held(*args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            if calls[0] == call:
                reached.set()
                release.wait(APP_WAIT_S)
            return real(*args, **kwargs)

        setattr(self.arm, name, held)
        return reached, release

    def hold_jaws_at(self, opening: float) -> threading.Event:
        """Hold the next jaw command once the jaws pass `opening`, until a stop.

        Returns the event set when the gripper worker waits there; a STOP, a
        fault-stop or close() lets the command go on, cancelled.
        """
        reached = threading.Event()
        cancels: list[threading.Event] = []
        real = self.arm.gripper_command

        def gripper_command(target, max_effort, mode, cancel):
            cancels.append(cancel)
            return real(target, max_effort, mode, cancel)

        def on_step(at: float) -> None:
            if not reached.is_set() and at <= opening + 1e-9:
                reached.set()
                cancels[-1].wait(APP_WAIT_S)

        self.arm.gripper_command = gripper_command
        self.arm.on_jaw_step = on_step
        return reached

    def wait_local_event(self, kind: str) -> tuple[str, str, dict[str, Any]]:
        """Wait for an event of this kind at the Thor terminal's listener."""

        def found() -> tuple[str, str, dict[str, Any]] | None:
            return next((e for e in self.local_events if e[0] == kind), None)

        with self._heard:
            assert self._heard.wait_for(lambda: found() is not None, APP_WAIT_S), (
                f"no {kind!r} event; heard {[e[0] for e in self.local_events]}"
            )
            return found()

    def wait_local_events(
        self, kind: str, count: int
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Wait until the Thor terminal's listener heard `count` events of this kind."""

        def found() -> list[tuple[str, str, dict[str, Any]]]:
            return [event for event in self.local_events if event[0] == kind]

        with self._heard:
            assert self._heard.wait_for(lambda: len(found()) >= count, APP_WAIT_S), (
                f"{len(found())} of {count} {kind!r} events; heard "
                f"{[e[0] for e in self.local_events]}"
            )
            return found()

    def _hear(self, kind: str, text: str, data: dict[str, Any]) -> None:
        with self._heard:
            self.local_events.append((kind, text, data))
            self._heard.notify_all()

    def _shows_the_apps_numbers(self, state: RobotState) -> bool:
        """Whether a ROBOT_STATE has the owner and stop numbers the app has now.

        The client fills a command's numbers in from such a state only: a
        command sent right after an owner change would else quote the old one.
        """
        if not (self.app.running and self.app.link_alive):
            return True  # no newer ROBOT_STATE will come
        now = self.app.snapshot()
        shown = (state.owner_number, state.stop_number)
        return shown == (now.owner_number, now.stop_number)


@pytest.fixture
def make_robot_app(
    sim_arm: SimArm, checkers: CheckerSet, config: RobotAppConfig, clock: FakeClock
) -> Iterator[Callable[..., AppHarness]]:
    """Builds a robot app with a client on the simulated arm; closed after the test.

    make_robot_app(config=None, pose=None, confirm=None, start=True, connected=True):
    pose puts the arm there before the app connects; start=False leaves the app
    unstarted; connected=False builds it with no robot connection. Further
    keywords, such as driver or allow_reference, go to RobotApp.
    """
    made: list[AppHarness] = []

    def make(
        config: RobotAppConfig = config,
        pose: Sequence[float] | None = None,
        confirm: Callable[[str], bool] | None = None,
        start: bool = True,
        connected: bool = True,
        **app_options: Any,
    ) -> AppHarness:
        if pose is not None:
            sim_arm.set_pose(pose)
        harness = AppHarness(
            config,
            sim_arm,
            checkers,
            clock,
            confirm,
            connected,
            app_options=app_options,
        )
        made.append(harness)
        return harness.start() if start else harness

    yield make
    for harness in made:
        harness.close()


@pytest.fixture
def robot_app(make_robot_app: Callable[..., AppHarness]) -> AppHarness:
    """A started robot app at rest with a workstation client."""
    return make_robot_app()


# The real-time variant: the same harnesses on the machine's clock.
#
# On the fake clock a move's setpoints go out without real time passing, so a
# move never waits for a later permit. Whatever drives the robot app with
# permits paced in real time (the arm adapter's permit loop, the agent program)
# needs the app on SystemClock and its own dispatcher period: then a move of 25
# setpoints takes 0.5 s. Keep such moves short and give each test a budget.

# Real seconds one real-time test may take for all of its moves.
REALTIME_BUDGET_S = 30.0


@pytest.fixture
def system_clock() -> SystemClock:
    from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock

    return SystemClock()


@pytest.fixture
def make_realtime_robot_app(
    config: RobotAppConfig, system_clock: SystemClock, request: pytest.FixtureRequest
) -> Iterator[Callable[..., AppHarness]]:
    """Builds a robot app on the simulated arm in real time; closed after the test.

    make_realtime_robot_app(config=None, pose=None, confirm=None, start=True,
    connected=True), as make_robot_app, with a simulated arm and checkers of
    its own on the machine's clock. After the test every torque release of each
    arm must have been at measured rest, unless the test is marked
    "releases_away_from_rest".
    """
    from isaaccapture_examples.agentic_remote.robot_app.sim import SimArm

    made: list[AppHarness] = []

    def make(
        config: RobotAppConfig = config,
        pose: Sequence[float] | None = None,
        confirm: Callable[[str], bool] | None = None,
        start: bool = True,
        connected: bool = True,
    ) -> AppHarness:
        arm = SimArm(
            n_joints=config.n_joints,
            clock=system_clock,
            q0=list(config.rest.rest_q if pose is None else pose),
        )
        harness = AppHarness(
            config,
            arm,
            basic_checkers(config, system_clock),
            system_clock,
            confirm,
            connected,
            step_period_s=None,
        )
        made.append(harness)
        return harness.start() if start else harness

    yield make
    for harness in made:
        harness.close()
    if not request.node.get_closest_marker("releases_away_from_rest"):
        for harness in made:
            check_released_at_rest(harness.arm, harness.config)


@pytest.fixture
def realtime_robot_app(
    make_realtime_robot_app: Callable[..., AppHarness],
) -> AppHarness:
    """A started robot app at rest in real time, with a workstation client."""
    return make_realtime_robot_app()


def realtime_apai_harness(
    config: RobotAppConfig | None = None,
    profile: Any = None,
    connected: bool = True,
) -> AppHarness:
    """The robot app on APAI's mock arm and checkers, on the machine's clock.

    An ApaiHarness of test_apai_app.py (its spy is .spy), not started. config
    defaults to that file's make_config(profile), profile to APAI's mock
    profile. connected=False builds it with no robot connection, as
    UnconnectedApai of test_link_end_to_end.py, for a link_stack.Stack. APAI's
    checker paces itself on a clock that is not a fake one.
    """
    import test_apai_app as apai
    from isaaccapture_examples.agentic_remote.robot_app.clock import SystemClock

    backend = apai.apai_backend
    profile = backend.load_apai_profile("mock") if profile is None else profile
    config = apai.make_config(profile) if config is None else config
    clock = SystemClock()

    class RealtimeApai(apai.ApaiHarness):
        def __init__(self) -> None:
            checkers = backend.build_apai_checkers(config, profile, clock)
            arm = backend.build_apai_arm(config, profile, clock)
            self.spy = apai.BackendSpy(arm, checkers)
            AppHarness.__init__(
                self, config, arm, checkers, clock, None, connected, step_period_s=None
            )

    return RealtimeApai()


def check_realtime_apai(harness: AppHarness) -> None:
    """check_driver_use() of test_apai_app.py, also for a harness with no client.

    A harness driven by the agent program has no test client of its own; the
    rule about the driver (connected once, never stopped, disconnected at
    measured rest only) holds for it all the same.
    """
    import test_apai_app as apai

    if harness.client is None:
        harness.client = types.SimpleNamespace(unexpected=[])
    apai.check_driver_use(harness)


@pytest.fixture
def make_realtime_apai() -> Iterator[Callable[..., AppHarness]]:
    """Builds and starts a robot app on APAI's mock arm in real time.

    make_realtime_apai(config=None, profile=None, connected=True, start=True);
    see realtime_apai_harness(). After the test each app is closed and
    check_realtime_apai() is applied to it. For tests marked "apai".
    """
    made: list[AppHarness] = []

    def make(
        config: RobotAppConfig | None = None,
        profile: Any = None,
        connected: bool = True,
        start: bool = True,
    ) -> AppHarness:
        harness = realtime_apai_harness(config, profile, connected)
        made.append(harness)
        return harness.start() if start else harness

    yield make
    for harness in made:
        harness.close()
    for harness in made:
        check_realtime_apai(harness)


@pytest.fixture
def realtime_apai_app(make_realtime_apai: Callable[..., AppHarness]) -> AppHarness:
    """A started robot app on APAI's mock arm at rest in real time, with a client."""
    return make_realtime_apai()


@pytest.fixture
def realtime_apai_stack(
    make_realtime_apai: Callable[..., AppHarness],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Any]:
    """A link_stack.Stack on a started real-time app on APAI's mock arm.

    The app has no robot connection yet: start_service() and dial() make it,
    as with the fixture apai_stack of test_link_end_to_end.py. For tests
    marked "apai" and "link".
    """
    from link_stack import Stack

    harness = make_realtime_apai(connected=False)
    # A short path: a socket path holds 107 bytes at most.
    made = Stack(harness, tmp_path_factory.mktemp("ls"))
    yield made
    made.close()
