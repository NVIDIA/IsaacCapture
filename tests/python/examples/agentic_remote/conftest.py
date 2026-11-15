# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The package under test lives in examples/agentic_remote/python and is not installed.

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
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
    from isaaccapture_examples.agentic_remote.robot_app.clock import FakeClock
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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    apai_items = [item for item in items if item.get_closest_marker("apai")]
    if not apai_items:
        return
    reason = _apai_unavailable()
    if reason is None:
        return
    for item in apai_items:
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
    """

    def __init__(
        self,
        config: RobotAppConfig,
        arm: SimArm,
        checkers: CheckerSet,
        clock: FakeClock,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        from isaaccapture_examples.agentic_remote.protocol.carrier import loopback_pair
        from isaaccapture_examples.agentic_remote.robot_app.app import RobotApp
        from isaaccapture_examples.agentic_remote.testing.client import (
            WorkstationClient,
        )

        self.config, self.arm, self.checkers, self.clock = config, arm, checkers, clock
        self.robot_end, self.workstation_end = loopback_pair()
        self.app: RobotApp = RobotApp(
            config,
            arm,
            checkers,
            self.robot_end,
            clock,
            confirm=confirm,
            step_period_s=APP_STEP_S,
        )
        self.client: WorkstationClient = WorkstationClient(
            self.workstation_end, clock, quotable=self._shows_the_apps_numbers
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
        self.client.wait_welcome(APP_WAIT_S)
        self.client.wait_state(timeout_s=APP_WAIT_S)
        return self

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
        self.client.close()
        for end in (self.robot_end, self.workstation_end):
            end.close()

    def step(self) -> None:
        """Wait for one dispatcher pass that began after this call."""
        assert self.app.wait_step(APP_WAIT_S), "the dispatcher made no pass"

    def advance(self, seconds: float, step_s: float = 1.0) -> None:
        """Move the fake clock on in steps; the dispatcher makes a pass after each."""
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

    make_robot_app(config=None, pose=None, confirm=None, start=True): pose puts the
    arm there before the app connects; start=False leaves the app unstarted.
    """
    made: list[AppHarness] = []

    def make(
        config: RobotAppConfig = config,
        pose: Sequence[float] | None = None,
        confirm: Callable[[str], bool] | None = None,
        start: bool = True,
    ) -> AppHarness:
        if pose is not None:
            sim_arm.set_pose(pose)
        harness = AppHarness(config, sim_arm, checkers, clock, confirm)
        made.append(harness)
        return harness.start() if start else harness

    yield make
    for harness in made:
        harness.close()


@pytest.fixture
def robot_app(make_robot_app: Callable[..., AppHarness]) -> AppHarness:
    """A started robot app at rest with a workstation client."""
    return make_robot_app()
