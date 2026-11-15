# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.robot_app --config PATH
--arm sim|apai [--selftest].

Phase P0 builds no link to the workstation, so the one thing that runs is the
self-test: the robot app on the simulated arm, driven over a loopback link by a
scripted workstation client and at a scripted Thor terminal.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ..protocol.carrier import loopback_pair
from .app import RobotApp
from .clock import SystemClock
from .config import ConfigError, load_config
from .interfaces import CheckerSet
from .rest import planned_duration_s
from .sim import BasicChecker, SimArm
from .terminal import CONFIRM_REST, ThorTerminal, TypedLines
from ..testing.client import WorkstationClient

if TYPE_CHECKING:
    from ..protocol.messages import Result, RobotState
    from .clock import Clock
    from .config import RobotAppConfig

_log = logging.getLogger("isaaccapture.examples.agentic_remote.main")

EXIT_OK = 0
EXIT_FAILED = 1  # a step of the self-test went wrong
EXIT_NOT_RUN = 2  # nothing to run in phase P0, or the arguments cannot be used

LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
# The src folder of an APAI checkout, as the tests name it.
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"

NO_LINK = (
    "phase P0 builds no link to the workstation, so the robot app has nothing to "
    "connect to"
)

_LOGGER_NAME = "isaaccapture.examples.agentic_remote"
_HANDLER_NAME = "agentic_remote.console"

# Longest one wait of the self-test takes, in real time.
STEP_WAIT_S = 10.0
# How far the self-test's two moves take every joint, in rad.
FIRST_MOVE_RAD = 0.08
SECOND_MOVE_RAD = 0.30
# The unpermitted move is this short, so little moves if it is not refused.
REFUSED_MOVE_RAD = 0.02
# The moves' peak joint speed as a share of the agent's speed limit.
SPEED_SHARE = 0.75
# Setpoints of the stopped move that are out before the STOP is sent.
STOP_AFTER = 3
TERMINAL_PREFIX = "    thor terminal | "

Say = Callable[[str], None]


class _Lines:
    """Writes whole lines to the standard output, from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self, text: str) -> None:
        with self._lock:
            print(text, flush=True)


_say = _Lines()


def _complain(text: str) -> None:
    print(f"robot app: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.robot_app",
        description="The robot app of the agentic remote execution example.",
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="the robot app's YAML file"
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=("sim", "apai"),
        help="sim: the simulated arm; apai: APAI's checker and motor driver",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="with --arm sim: run the scripted self-test and exit",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        type=str.lower,
        choices=LOG_LEVELS,
        metavar="L",
        help=f"console log level, one of {', '.join(LOG_LEVELS)} (default: warning)",
    )
    parser.add_argument(
        "--apai-profile",
        default="mock",
        metavar="NAME",
        help="with --arm apai: the APAI arm profile to load (default: mock)",
    )
    return parser


def _configure_logging(level: str) -> None:
    """Set the console log level, through Isaac Capture's logging when it is there."""
    try:
        from isaaccapture import logging_config
    except ImportError:
        logging_config = None
    if logging_config is not None:
        logging_config.set_console_level(level)
        return
    # Without Isaac Capture nothing shows this logger tree: the example's one
    # stream handler is attached here and nowhere else.
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the robot app's command line; returns the exit code."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse has printed the usage or the help
        code = exit_.code
        return code if isinstance(code, int) else EXIT_NOT_RUN
    if args.selftest and args.arm != "sim":
        _complain("--selftest runs on the simulated arm only: use --arm sim")
        return EXIT_NOT_RUN
    _configure_logging(args.log_level)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        _complain(f"the configuration cannot be used: {error}")
        return EXIT_NOT_RUN
    if args.arm == "apai":
        return _describe_apai(config, args.config, args.apai_profile)
    if not args.selftest:
        _complain(f"{NO_LINK}; --arm sim runs only with --selftest")
        return EXIT_NOT_RUN
    return run_selftest(config)


# --arm apai


def _describe_apai(config: RobotAppConfig, config_path: str, profile_name: str) -> int:
    """Build APAI's checkers and arm port, connect nothing, say what would run."""
    src = os.environ.get(APAI_SRC_ENV)
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        # APAI is imported in this branch only, so --arm sim runs without it.
        apai_backend = importlib.import_module(
            "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
        )
    except ImportError as error:
        _complain(
            f"--arm apai needs APAI on the Python path (set {APAI_SRC_ENV} to its "
            f"src folder): {error}"
        )
        return EXIT_NOT_RUN
    clock = SystemClock()
    try:
        profile = apai_backend.load_apai_profile(profile_name)
        config = apai_backend.config_for_profile(config, profile)
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        arm = apai_backend.build_apai_arm(config, profile, clock)
    except Exception as error:
        _log.debug("building the APAI backend failed", exc_info=True)
        _complain(
            f"APAI arm profile {profile_name!r} cannot be used with {config_path}: "
            f"{error}"
        )
        return EXIT_NOT_RUN
    backend = type(arm.backend).__name__
    versions = ", ".join(f"{c.name} {c.version}" for c in checkers.all())
    _say("robot app with --arm apai would run:")
    _say(
        f"  configuration {config_path}: robot {config.robot_id}, profile "
        f"{config.profile_id}, {config.n_joints} joints"
    )
    _say(f"  APAI arm profile {profile_name}: driver {backend}, built, not connected")
    _say(f"  one APAI checker per control owner, no obstacle grid: {versions}")
    travel, jaw_tol = apai_backend.jaw_travel_rad(profile), config.gripper.target_tol
    in_rad = (
        "" if travel is None else f" ({jaw_tol * travel:.3g} rad of {travel:g} rad)"
    )
    _say(
        f"  jaws proven empty: within {jaw_tol:.3g} of the jaw travel of the last "
        f"jaw target{in_rad}"
    )
    _say(
        f"  at rest at start: within {config.rest.start_rest_tol_rad:g} rad of the "
        f"rest pose (the profile's); a release: within {config.rest.rest_tol_rad:g} rad"
    )
    _say(f"{NO_LINK}: the arm was not connected and nothing runs (exit code 2)")
    return EXIT_NOT_RUN


# --arm sim --selftest


def sim_checkers(config: RobotAppConfig, clock: Clock) -> CheckerSet:
    """One BasicChecker per control owner; the recovery one has no watchdog."""
    limits = config.limits
    settings = {"reads_per_step": config.executor.reads_per_step, "grasp": config.grasp}
    return CheckerSet(
        agent=BasicChecker(limits.agent, clock, **settings),
        human=BasicChecker(limits.human, clock, **settings),
        recovery=BasicChecker(limits.recovery, clock, watchdog_s=None, **settings),
    )


def run_selftest(config: RobotAppConfig, say: Say = _say) -> int:
    """Run the self-test on the simulated arm; EXIT_OK when every step passed."""
    say("selftest: the robot app on the simulated arm, over a loopback link")
    return _SelfTest(config, say).run()


class _StepFailed(Exception):
    """A step of the self-test did not go as it must."""


class _TerminalLines:
    """What the scripted Thor terminal wrote: shown, and kept for the script."""

    def __init__(self, say: Say) -> None:
        self._say = say
        self._written = threading.Condition()
        self._lines: list[str] = []

    def __call__(self, line: str) -> None:
        self._say(TERMINAL_PREFIX + line)
        with self._written:
            self._lines.append(line)
            self._written.notify_all()

    def wait_for(self, line: str, timeout_s: float) -> bool:
        """Wait in real time until the terminal has written this line."""
        with self._written:
            return self._written.wait_for(lambda: line in self._lines, timeout_s)

    def count(self, line: str) -> int:
        """How often the terminal has written this line so far."""
        with self._written:
            return self._lines.count(line)


class _SelfTest:
    """The scripted run: one line per step, and the Thor terminal's own lines."""

    def __init__(self, config: RobotAppConfig, say: Say) -> None:
        self._config = config
        self._say = say
        self._clock = SystemClock()
        rest_q = list(config.rest.rest_q)
        self._arm = SimArm(
            n_joints=config.n_joints,
            clock=self._clock,
            q0=rest_q,
            joint_names=config.joint_names,
        )
        self._robot_end, self._workstation_end = loopback_pair()
        self._app = RobotApp(
            config,
            self._arm,
            sim_checkers(config, self._clock),
            self._robot_end,
            self._clock,
        )
        self._client = WorkstationClient(self._workstation_end, self._clock)
        self._typed = TypedLines()
        self._written = _TerminalLines(say)
        self._terminal = ThorTerminal(self._app, self._typed, self._written)
        self._reader = threading.Thread(
            target=self._terminal.run, name="thor-terminal-input", daemon=True
        )
        self._at = rest_q  # the pose the arm holds
        self._stop_number = 0  # as the last step left it
        self._steps: tuple[tuple[str, Callable[[], str]], ...] = (
            ("start the robot app", self._start),
            ("take ownership for the agent", self._take_ownership),
            ("a move with permits", self._permitted_move),
            ("a move without a permit", self._unpermitted_move),
            ("STOP during a move", self._stop_during_move),
            ("a new task after the STOP, no reset", self._task_after_stop),
            ("the agent asks for help", self._help_needed),
            ("stop typed at the Thor terminal", self._typed_stop),
            ("rest typed at the Thor terminal", self._typed_rest),
            ("release at rest", self._release),
            ("quit typed at the Thor terminal", self._typed_quit),
        )

    def run(self) -> int:
        total = len(self._steps)
        began = self._clock.monotonic()
        failed = 0  # the number of the step that went wrong
        try:
            for number, (title, step) in enumerate(self._steps, start=1):
                try:
                    detail = step()
                except Exception as error:
                    _log.debug("self-test step %r failed", title, exc_info=True)
                    self._terminal.flush()
                    self._say(f"selftest {number:2d}/{total} FAILED  {title}: {error}")
                    failed = number
                    break
                self._terminal.flush()
                self._say(f"selftest {number:2d}/{total} ok  {title}: {detail}")
        finally:
            self._close()
        if failed:
            self._say(f"selftest: FAILED at step {failed} of {total}")
            return EXIT_FAILED
        took = self._clock.monotonic() - began
        self._say(f"selftest: all {total} steps passed in {took:.1f} s")
        return EXIT_OK

    def _close(self) -> None:
        """End every thread; the simulated motors stay as the run left them."""
        self._typed.end()
        if self._reader.is_alive():
            self._reader.join(STEP_WAIT_S)
        self._terminal.close()
        self._app.close()
        self._client.close()
        for end in (self._robot_end, self._workstation_end):
            end.close()

    # Steps; each returns what its line says

    def _start(self) -> str:
        self._app.start()
        self._reader.start()
        welcome = self._client.wait_welcome(STEP_WAIT_S)
        state = self._state(lambda s: s.motion == "rest_hold" and s.torque_on)
        _need(state.gripper.load == "empty", f"the jaws read {state.gripper.load}")
        return (
            f"session {welcome.session_id[:8]}, {len(welcome.joint_names)} joints, "
            "at rest with the motors on, jaws proven empty"
        )

    def _take_ownership(self) -> str:
        result = _ended(self._client.start_task(timeout_s=STEP_WAIT_S), "completed")
        return f"the agent owns the arm, owner number {result.detail['owner_number']}"

    def _permitted_move(self) -> str:
        goal = self._toward_middle(self._at, FIRST_MOVE_RAD)
        setpoints = self._plan(goal)
        result = self._client.run_move(setpoints, STEP_WAIT_S, start_q=self._at)
        _ended(result, "completed")
        self._at = goal
        off = _distance(result.end_q or [], goal)
        _need(
            off <= self._config.executor.settle_tol_rad,
            f"the arm ended {off:.3f} rad from the goal",
        )
        return f"{len(setpoints)} setpoints sent, ended {off:.3f} rad from the goal"

    def _unpermitted_move(self) -> str:
        sent = len(self._arm.sent)
        setpoints = self._plan(self._toward_middle(self._at, REFUSED_MOVE_RAD))
        command_id = self._client.move(setpoints, start_q=self._at)
        result = self._client.wait_result(command_id, timeout_s=STEP_WAIT_S)
        if result.status != "refused":
            needed = self._config.permits.required_for_agent
            why = "" if needed else ": this configuration switches permits off"
            raise _StepFailed(f"the move ended {result.status}, not refused{why}")
        _need(len(self._arm.sent) == sent, "a setpoint was sent without a permit")
        return f"refused, no setpoint sent: {result.reason}"

    def _stop_during_move(self) -> str:
        before = self._state()
        setpoints = self._plan(self._toward_middle(self._at, SECOND_MOVE_RAD))
        move_id = self._client.new_id()

        def under_way_or_ended() -> bool:
            state = self._client.state
            if state is not None and state.move_id == move_id:
                return state.move_index >= STOP_AFTER
            return any(r.status != "accepted" for r in self._client.results(move_id))

        feed = self._client.feed_permits(move_id, len(setpoints) - 1)
        try:
            self._client.move(setpoints, start_q=self._at, command_id=move_id)
            self._client.wait_until(under_way_or_ended, STEP_WAIT_S, "the move's start")
            self._client.stop("self-test: the operator pressed stop")
            result = self._client.wait_result(move_id, timeout_s=STEP_WAIT_S)
        finally:
            feed.stop()
        _ended(result, "cancelled")
        sent = result.detail.get("sent")
        _need(
            isinstance(sent, int) and 0 < sent < len(setpoints),
            f"{sent} of {len(setpoints)} setpoints were sent",
        )
        self._stop_number = before.stop_number + 1
        self._state(
            lambda s: (
                s.stop_number == self._stop_number
                and s.owner == "nobody"
                and s.motion == "hold"
                and s.torque_on
            )
        )
        return (
            f"cancelled after {sent} of {len(setpoints)} setpoints, the arm holds, "
            f"stop number {self._stop_number}"
        )

    def _task_after_stop(self) -> str:
        owns = self._take_ownership()
        # Only the stop number can refuse this move: it quotes the new owner
        # number, starts where the arm is and asks for no travel.
        at, old = self._client.measured_pose(), self._stop_number - 1
        command_id = self._client.move([at], start_q=at, stop_number=old)
        result = self._client.wait_result(command_id, timeout_s=STEP_WAIT_S)
        wanted = f"stop number {old} does not match the current {self._stop_number}"
        _need(
            _ended(result, "refused").reason == wanted,
            f"a move quoting stop number {old} was refused with {result.reason!r}, "
            f"not with {wanted!r}",
        )
        return f"{owns}; a move quoting the old stop number is refused"

    def _help_needed(self) -> str:
        command_id = self._client.request_owner(
            "help_needed", reason="self-test: the agent gives up"
        )
        _ended(self._client.wait_result(command_id, timeout_s=STEP_WAIT_S), "completed")
        state = self._state(
            lambda s: s.assistance_wait_s is not None and s.owner == "nobody"
        )
        return (
            f"the arm holds, nobody owns it, {state.assistance_wait_s:.0f} s of the "
            "assistance wait left (no configuration may shorten it below 90 s)"
        )

    def _typed_stop(self) -> str:
        self._type("stop")
        self._stop_number += 1
        self._state(
            lambda s: (
                s.stop_number == self._stop_number
                and s.assistance_wait_s is None
                and s.motion == "hold"
                and s.torque_on
            )
        )
        return (
            "the assistance wait is cancelled, the arm holds, stop number "
            f"{self._stop_number}"
        )

    def _typed_rest(self) -> str:
        self._type("rest")
        _need(
            self._written.wait_for(CONFIRM_REST, STEP_WAIT_S),
            "the Thor terminal did not ask whether the path is clear",
        )
        self._type("yes")
        rest = self._config.rest
        longest = planned_duration_s(
            FIRST_MOVE_RAD + SECOND_MOVE_RAD, rest.recovery_speed, rest.min_duration_s
        )
        state = self._state(
            lambda s: _typed_result(s, "rest") is not None, STEP_WAIT_S + longest
        )
        result = _typed_result(state, "rest") or {}
        _need(
            result.get("status") == "completed",
            f"the return to rest ended {result.get('status')}: {result.get('reason')}",
        )
        _need(state.motion == "rest_hold" and state.torque_on, "not holding at rest")
        off = _distance(self._arm.pose(), rest.rest_q)
        return f"the arm holds at rest with the motors on, {off:.3f} rad from rest"

    def _release(self) -> str:
        command_id = self._client.release()
        _ended(self._client.wait_result(command_id, timeout_s=STEP_WAIT_S), "completed")
        self._state(lambda s: s.motion == "rest_unpowered" and not s.torque_on)
        _need(len(self._arm.released_at) == 1, "torque was not released exactly once")
        off = _distance(self._arm.released_at[0], self._config.rest.rest_q)
        _need(
            off <= self._config.rest.rest_tol_rad,
            f"torque was released {off:.3f} rad away from rest",
        )
        return f"the motors are off, released {off:.3f} rad from rest"

    def _typed_quit(self) -> str:
        asked = self._written.count(CONFIRM_REST)
        self._type("quit")
        _need(self._app.wait_closed(STEP_WAIT_S), "the robot app did not close")
        _need(len(self._arm.released_at) == 1, "torque was released a second time")
        self._terminal.flush()
        # Away from rest a quit asks first that the path is clear, as rest does.
        _need(
            self._written.count(CONFIRM_REST) == asked,
            "quit asked whether the path is clear with the motors off at rest",
        )
        return "the robot app has closed; nothing was asked: the arm did not move"

    # Helpers

    def _type(self, line: str) -> None:
        """Type one line at the Thor terminal, and show it."""
        self._terminal.flush()
        self._say(f"{TERMINAL_PREFIX}> {line}")
        self._typed.type(line)

    def _state(
        self,
        wanted: Callable[[RobotState], bool] | None = None,
        timeout_s: float = STEP_WAIT_S,
    ) -> RobotState:
        return self._client.wait_state(wanted, timeout_s)

    def _plan(self, goal: Sequence[float]) -> list[list[float]]:
        speed = SPEED_SHARE * self._config.limits.agent.max_joint_speed
        return self._client.plan_straight(self._at, goal, speed)

    def _toward_middle(self, pose: Sequence[float], by: float) -> list[float]:
        """The pose with every joint moved by `by` toward the middle of the box.

        The box is the agent's envelope, or its joint limits without one.
        """
        agent = self._config.limits.agent
        low = agent.envelope_min or agent.joint_min
        high = agent.envelope_max or agent.joint_max
        goal = []
        for at, lowest, highest in zip(pose, low, high):
            step = by if at <= (lowest + highest) / 2.0 else -by
            goal.append(min(max(at + step, lowest), highest))
        return goal


def _need(holds: bool, problem: str) -> None:
    if not holds:
        raise _StepFailed(problem)


def _ended(result: Result, status: str) -> Result:
    """The result, if its command ended with this status; else the step fails."""
    if result.status != status:
        why = f": {result.reason}" if result.reason else ""
        raise _StepFailed(f"RESULT {result.status}, not {status}{why}")
    return result


def _typed_result(state: RobotState, what: str) -> dict | None:
    """ROBOT_STATE's last result, if a request typed at the Thor terminal ended."""
    result = state.last_result or {}
    typed = str(result.get("command_id", "")).startswith(f"thor-terminal-{what}-")
    return result if typed else None


def _distance(pose: Sequence[float], goal: Sequence[float]) -> float:
    """The largest joint distance between two poses, in rad."""
    if len(pose) != len(goal):
        return float("inf")
    return max((abs(at - to) for at, to in zip(pose, goal)), default=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
