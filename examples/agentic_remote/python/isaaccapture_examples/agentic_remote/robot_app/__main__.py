# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.robot_app --config PATH
--arm sim|apai [--connect HOST:PORT | --selftest | --selftest-link].

--connect runs the robot app with the Thor terminal and dials the robot link
service. --selftest drives the app on the simulated arm over a loopback link;
--selftest-link drives it through a robot link service over a real link on
127.0.0.1. --arm apai --driver real connects APAI's motor driver, whose connect
cuts motor torque: only after its checks and three typed confirmations, and
APAI's driver itself refuses an arm it does not measure at rest.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import select
import signal
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..console_log import set_console_level
from ..link_service.listen import RobotListener
from ..link_service.local import connect_local
from ..link_service.service import LOST, RobotLinkService
from ..protocol.carrier import loopback_pair
from ..protocol.confirm import PATH_IS_CLEAR, YES, agent_environment, confirms
from ..protocol.link_carrier import LinkCarrier
from ..protocol.local_messages import ROLE_AGENT, ROLE_CONSOLE
from ..protocol.messages import DRIVER_APAI_MOCK, DRIVER_REAL, DRIVER_SIM
from .app import RobotApp, live_view_problem
from .clock import SystemClock
from .config import ConfigError, load_config
from .connection import RobotDialer
from .interfaces import CheckerSet, MotionState, Owner
from .rest import planned_duration_s
from .sim import BasicChecker, SimArm
from .terminal import CONFIRM_REST, ThorTerminal, TypedLines, drop_typed_ahead
from .text import MAX_WORD_CHARS, printable
from ..testing.client import WorkstationClient

try:
    import termios
except ImportError:  # no terminal control on this platform
    termios = None

if TYPE_CHECKING:
    from ..link_service.local import LocalCarrier
    from ..protocol.messages import Result, RobotState
    from .clock import Clock
    from .config import RobotAppConfig
    from .interfaces import ArmPort

_log = logging.getLogger("isaaccapture.examples.agentic_remote.main")

EXIT_OK = 0
EXIT_FAILED = 1  # a step of a self-test went wrong, or the robot app did not start
# Nothing ran and nothing was connected: the arguments, the installation, or a
# check or a typed confirmation of the real driver.
EXIT_NOT_RUN = 2

LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
# The src folder of an APAI checkout, as the tests name it.
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"

NO_MODE = "needs --connect HOST:PORT, --selftest or --selftest-link"
NO_CONNECT = "no --connect HOST:PORT was given"
NO_REMOTE_LINK = (
    "Isaac Capture's remote link cannot be imported (isaaccapture.remote_link on "
    "the Python path)"
)
NOT_CONFIRMED = "not confirmed: nothing was connected"
NOT_CONNECTED = "the real driver was not connected"
NO_REAL_CONNECT = (
    "--driver real needs the robot link service's address: --connect HOST:PORT"
)
NO_TERMINAL = "the confirmations are typed by a person: standard input is no terminal"
NO_OUTPUT_TERMINAL = (
    "the questions are shown to a person: standard output is no terminal"
)
# After a start that failed past APAI's own refusal: the connect may have
# enabled the motors, and this program switches nothing off away from a release.
MOTORS_MAY_BE_ON = (
    "the start failed after the connect began: the motors may be enabled and "
    "holding, and this program ends without releasing them. APAI's driver "
    "enables them only at measured rest: with the arm lying at rest, switch the "
    "48 V off (safe only there or with the arm held by hand), then start again"
)
# What the operator confirms at the Thor terminal before the real driver may be
# connected: connecting cuts motor torque.
REAL_DRIVER_QUESTIONS = (
    "Is the arm powered?",
    "Connecting cuts motor torque. Is the arm at rest and supported?",
    "Is the calibration in use {calibration}?",
)
# The connect that follows the three answers. APAI's driver reads every joint
# twice before anything that cuts torque and refuses an arm that is not at rest.
CONNECTING = (
    "robot app: confirmed three times; connecting the arm. APAI's driver measures "
    "it first and refuses an arm that is not at rest"
)
# APAI's refusal of a connect: a hold marker, or an arm not measured at rest.
_ARM_HELD = "ArmHeldError"
NOT_AT_REST = (
    "APAI's driver refused the connect before anything that cuts torque. This "
    "program never connects on an operator's confirmation: bring the arm to rest "
    "(48 V off only with the arm held by hand or at rest) and start again"
)

# The longest one dial of the robot link service takes; the dialer's close()
# waits for a dial under way.
CONNECT_TIMEOUT_S = 2.0
LINK_NAME = "robot-app"
# How often the program looks at the robot app, the dialer and its signals.
_WATCH_S = 0.2
_READ_BYTES = 4096
# The most reads that take typed-ahead lines off a terminal before a question.
_DROP_READS = 64

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
    """One line on the standard error; a terminal that is gone loses it."""
    try:
        print(f"robot app: {text}", file=sys.stderr, flush=True)
    except (OSError, ValueError) as error:
        _log.warning("the standard error cannot be written to (%r): %s", error, text)


def _lossy(say: Say) -> Say:
    """say for the lines the program writes itself once the robot app runs.

    A terminal that is gone loses them and ends nothing: the robot app runs on
    without its lines, as it does without the Thor terminal's.
    """
    lost = [False]

    def tell(text: str) -> None:
        try:
            say(text)
        except Exception as error:
            if not lost[0]:
                _log.warning("the terminal cannot be written to: %r", error)
            lost[0] = True
        else:
            lost[0] = False

    return tell


def _let_go_of_dead_streams() -> None:
    """Point a standard stream whose terminal is gone at the null device.

    The text it did not take would fail the interpreter's last flush, and the
    exit code would be 120 in place of main()'s.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, ValueError):
            continue  # no such stream, or a closed one
        except OSError:
            with contextlib.suppress(AttributeError, OSError, ValueError):
                null = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(null, stream.fileno())
                finally:
                    os.close(null)
                stream.flush()


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
        "--connect",
        type=_endpoint,
        metavar="HOST:PORT",
        help="run the robot app and dial the robot link service at this address",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="with --arm sim: run the scripted self-test on a loopback link and exit",
    )
    parser.add_argument(
        "--selftest-link",
        action="store_true",
        help=(
            "with --arm sim: run the scripted self-test through a robot link "
            "service over a real link on 127.0.0.1 and exit"
        ),
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
        "--driver",
        choices=("mock", "real"),
        help=(
            "with --arm apai: mock is APAI's software arm (default); real is APAI's "
            "motor driver, connected after its checks and three typed "
            "confirmations (needs --apai-profile, --calibration and --connect)"
        ),
    )
    parser.add_argument(
        "--apai-profile",
        metavar="NAME",
        help="with --arm apai: the APAI arm profile to load (default: mock)",
    )
    parser.add_argument(
        "--calibration",
        metavar="NAME",
        help=(
            "with --driver real: the name of the calibration in use; it must be "
            "camera_calibration_id of the file"
        ),
    )
    return parser


def _endpoint(text: str) -> str:
    """A "host:port" the robot app can dial; the port is from 1 to 65535."""
    host, colon, port = text.strip().rpartition(":")
    usable = colon and host and port.isascii() and port.isdigit()
    if not usable or not 1 <= int(port) <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} is not HOST:PORT")
    return f"{host}:{int(port)}"


def _remote_link_missing() -> str | None:
    """Why Isaac Capture's remote link cannot be used, or None when it can."""
    try:
        importlib.import_module("isaaccapture.remote_link")
    except ImportError as error:
        return f"{NO_REMOTE_LINK}: {error}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the robot app's command line; returns the exit code."""
    try:
        return _main(argv)
    finally:
        _let_go_of_dead_streams()


def _main(argv: Sequence[str] | None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse has printed the usage or the help
        code = exit_.code
        return code if isinstance(code, int) else EXIT_NOT_RUN
    problem = _unusable(args)
    if problem is not None:
        _complain(problem)
        return EXIT_NOT_RUN
    set_console_level(args.log_level)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as error:
        _complain(f"the configuration cannot be used: {error}")
        return EXIT_NOT_RUN
    if args.arm == "apai":
        return _run_apai(config, args)
    if args.selftest:
        return run_selftest(config)
    if args.selftest_link:
        return run_link_selftest(config)
    if args.connect is None:
        _complain(f"--arm sim {NO_MODE}")
        return EXIT_NOT_RUN
    missing = _remote_link_missing()
    if missing is not None:
        _complain(f"--connect cannot run: {missing}")
        return EXIT_NOT_RUN
    clock = SystemClock()
    arm = SimArm(
        n_joints=config.n_joints,
        clock=clock,
        q0=list(config.rest.rest_q),
        joint_names=config.joint_names,
    )
    return run_connected(
        config,
        arm,
        sim_checkers(config, clock),
        clock,
        args.connect,
        driver=DRIVER_SIM,
    )


def _unusable(args: argparse.Namespace) -> str | None:
    """Why these arguments cannot be used together, or None when they can."""
    if args.selftest and args.arm != "sim":
        return "--selftest runs on the simulated arm only: use --arm sim"
    if args.selftest_link and args.arm != "sim":
        return "--selftest-link runs on the simulated arm only: use --arm sim"
    modes = [bool(args.selftest), bool(args.selftest_link), args.connect is not None]
    if sum(modes) > 1:
        return "--connect, --selftest and --selftest-link exclude each other"
    apai_only = {
        "--driver": args.driver,
        "--apai-profile": args.apai_profile,
        "--calibration": args.calibration,
    }
    given = [name for name, value in apai_only.items() if value is not None]
    if args.arm != "apai" and given:
        return f"{', '.join(given)} can be used with --arm apai only"
    if args.driver != "real" and args.calibration is not None:
        return "--calibration can be used with --driver real only"
    if args.driver == "real":
        if not args.apai_profile:
            return "--driver real needs the arm's profile: --apai-profile NAME"
        if not args.calibration or not args.calibration.strip():
            return "--driver real needs the calibration's name: --calibration NAME"
        if args.connect is None:
            return NO_REAL_CONNECT
    return None


# --connect


class _FdLines:
    """Lines typed at a file descriptor.

    The standard input object is not used: a thread that waits in its readline
    holds a lock the interpreter needs at exit. Iterating waits for the next
    line; the input ends with the descriptor's end or soon after close().
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""
        self._ended = False
        self._closed = threading.Event()
        # Whether the lines are a person's: decided once, while the terminal is there.
        self._terminal = termios is not None and os.isatty(fd)
        # The line handed over last ended with the input, not with a newline.
        self.unterminated = False

    def close(self) -> None:
        self._closed.set()

    def drop_typed_ahead(self) -> list[str]:
        """On a terminal, drop what was typed and not yet handed over.

        Returns the whole lines dropped, so that a stop among them is not lost;
        a line that was begun and not entered goes too. A terminal that is gone
        ends the input. Called by the thread that iterates. Input that is no
        terminal, a pipe, is kept: its lines are a script's, not a person's.
        """
        if not self._terminal:
            return []
        held, self._buffer = self._buffer, b""
        try:
            # The terminal cannot say what it flushes: its whole lines are read.
            for _ in range(_DROP_READS):
                if not select.select([self._fd], [], [], 0)[0]:
                    break
                chunk = os.read(self._fd, _READ_BYTES)
                if not chunk:
                    # A typed end of the input, which goes too, or a terminal
                    # that is gone: its flush then fails.
                    break
                held += chunk
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except (OSError, ValueError, termios.error):
            self._ended = True  # the terminal is gone: nothing more is handed over
        whole = held.split(b"\n")[:-1]
        return [line.decode("utf-8", "replace") for line in whole]

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        while True:
            line, newline, rest = self._buffer.partition(b"\n")
            if newline or (self._ended and line):
                self._buffer = rest
                self.unterminated = not newline
                return line.decode("utf-8", "replace")
            if self._ended or self._closed.is_set():
                raise StopIteration
            ready, _, _ = select.select([self._fd], [], [], _WATCH_S)
            if not ready:
                continue
            chunk = os.read(self._fd, _READ_BYTES)
            self._ended = not chunk
            self._buffer += chunk


def typed_lines() -> Iterator[str]:
    """The lines typed at the standard input."""
    try:
        return _FdLines(sys.stdin.fileno())
    except (AttributeError, OSError, ValueError):  # no descriptor behind it
        return iter(sys.stdin)


def run_connected(
    config: RobotAppConfig,
    arm: ArmPort,
    checkers: CheckerSet,
    clock: Clock,
    endpoint: str,
    *,
    typed: Iterator[str] | None = None,
    say: Say = _say,
    driver: str = "",
    signals: _StartSignals | None = None,
) -> int:
    """Run the robot app with the Thor terminal and dial the robot link service.

    Returns when the robot app has closed (a typed quit, or a SHUTDOWN) or on
    SIGTERM, which closes the app with the motors left on. typed yields the
    lines of the Thor terminal; None reads the standard input. Ctrl+C is a STOP
    when this runs on the main thread: one pressed while the app starts is a
    STOP once it has started, and a SIGTERM then closes the app after the
    start, so no start is left half-way with the motors enabled. A terminal
    that closes ends neither the start nor the program: its SIGHUP is ignored
    and the lines it does not take are lost. signals are the caller's, noted
    since before this call; None notes them from here on. driver is what
    WELCOME declares. A configuration this driver may not run with gives one
    line and EXIT_NOT_RUN, with nothing connected.
    """
    lines = typed_lines() if typed is None else typed
    signals = _StartSignals() if signals is None else signals
    tell = _lossy(say)
    try:
        app = RobotApp(
            config,
            arm,
            checkers,
            None,
            clock,
            driver=driver,
            allow_reference=config.jaws.allow_reference,
        )
    except ConfigError as error:
        # A setting this driver may not run with: nothing was connected.
        _complain(f"the configuration cannot be used: {error}")
        _end_input(lines)
        return EXIT_NOT_RUN
    terminal = ThorTerminal(app, lines, say)
    dialer = RobotDialer(
        app,
        lambda: LinkCarrier.connect(endpoint, CONNECT_TIMEOUT_S, name=LINK_NAME),
        clock,
        retry_s=config.connection.retry_s,
    )
    reader = threading.Thread(
        target=terminal.run, name="thor-terminal-input", daemon=True
    )
    on_main = threading.current_thread() is threading.main_thread()
    try:
        if on_main:
            signals.install()
        try:
            app.start()
        except Exception as error:
            _log.debug("the robot app did not start", exc_info=True)
            _complain(f"the robot app did not start: {error}")
            if driver == DRIVER_REAL:
                held = type(error).__name__ == _ARM_HELD
                _complain(NOT_AT_REST if held else MOTORS_MAY_BE_ON)
            return EXIT_FAILED
        if driver == DRIVER_REAL:
            tell(_connected_line(app, arm))
        if on_main:
            terminal.install_signal_handlers()
            if signals.interrupts:  # Ctrl+C while the app started
                terminal.interrupt()
        reader.start()
        tell(
            f"robot app: session {app.session_id[:8]}; dialing the robot link "
            f"service at {endpoint}"
        )
        dialer.start()
        shown = ""
        while not app.wait_closed(_WATCH_S):
            if signals.terminated:
                tell("robot app: SIGTERM; closing with the motors left on")
                break
            error = "" if app.link_alive else dialer.last_error
            if error and error != shown:
                tell(
                    f"robot app: no robot connection: {error}; dialing again every "
                    f"{config.connection.retry_s:g} s"
                )
            shown = error
    finally:
        dialer.close()
        app.close()
        _end_input(lines)
        if reader.is_alive():
            reader.join(2 * _WATCH_S)  # one that cannot be ended is left behind
        terminal.close()
        if on_main:
            signals.restore()
    return EXIT_OK


class _StartSignals:
    """SIGINT and SIGTERM, noted and not acted on, from before the app's start.

    A start that Ctrl+C, SIGTERM or a closed terminal's SIGHUP ended half-way
    would leave the motors enabled with no program. SIGHUP is ignored.
    install() and restore() are for the main thread.
    """

    def __init__(self) -> None:
        # Filled by the handlers: list.append() is safe in a signal handler.
        self.interrupts: list[int] = []
        self.terminated: list[int] = []
        self._replaced: dict[int, Any] = {}

    def install(self) -> None:
        """Note the two signals and ignore SIGHUP from here on.

        Does nothing when installed.
        """
        if self._replaced:
            return
        noted = {signal.SIGINT: self.interrupts, signal.SIGTERM: self.terminated}
        for number, notes in noted.items():
            # Kept first: a Ctrl+C still raises until its handler is in place.
            handler = signal.getsignal(number)
            self._replaced[number] = signal.SIG_DFL if handler is None else handler
            signal.signal(number, lambda *_, notes=notes: notes.append(1))
        hangup = getattr(signal, "SIGHUP", None)  # absent on Windows
        if hangup is not None:
            handler = signal.getsignal(hangup)
            self._replaced[hangup] = signal.SIG_DFL if handler is None else handler
            signal.signal(hangup, signal.SIG_IGN)

    def restore(self) -> None:
        """Put back the handlers install() replaced."""
        while self._replaced:
            number, handler = self._replaced.popitem()
            signal.signal(number, handler)


def _connected_line(app: RobotApp, arm: ArmPort) -> str:
    """What the real driver's connect left: motors, motion state and jaws."""
    motors = "motors on" if arm.torque_on() else "MOTORS OFF"
    jaws = "referenced" if arm.gripper_referenced() else "NOT referenced"
    # APAI's driver sets this when it went ahead on an operator's confirmation.
    on_word = getattr(getattr(arm, "backend", None), "connected_on_confirmation", False)
    how = (
        "on an operator's confirmation, NOT at measured rest"
        if on_word
        else "at measured rest"
    )
    return (
        f"robot app: the real driver is connected {how}: {motors}, motion "
        f"{app.motion_state.value}, jaws {jaws}"
    )


def _end_input(lines: Iterator[str]) -> None:
    """End typed input that can be ended, so that a wait for a line returns."""
    for name in ("close", "end"):
        end = getattr(lines, name, None)
        if end is not None:
            end()
            return


# --arm apai


def _apai_backend() -> Any:
    """The module that puts APAI behind the ports; None, and a line, without APAI."""
    src = os.environ.get(APAI_SRC_ENV)
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        # APAI is imported in this branch only, so --arm sim runs without it.
        return importlib.import_module(
            "isaaccapture_examples.agentic_remote.robot_app.apai_backend"
        )
    except ImportError as error:
        _complain(
            f"--arm apai needs APAI on the Python path (set {APAI_SRC_ENV} to its "
            f"src folder): {error}"
        )
        return None


def _run_apai(config: RobotAppConfig, args: argparse.Namespace) -> int:
    """--arm apai: the real driver's confirmations, a description, or the app."""
    apai_backend = _apai_backend()
    if apai_backend is None:
        return EXIT_NOT_RUN
    if args.driver == "real":
        return _real_driver(apai_backend, config, args)
    profile_name = args.apai_profile or "mock"
    if args.connect is None:
        return _describe_apai(apai_backend, config, args.config, profile_name)
    missing = _remote_link_missing()
    if missing is not None:
        _complain(f"--connect cannot run: {missing}")
        return EXIT_NOT_RUN
    clock = SystemClock()
    try:
        profile = apai_backend.load_apai_profile(profile_name)
        if profile.arm_cfg.get("type") != "mock":
            _complain(
                f"APAI arm profile {profile_name!r} drives hardware: it needs "
                "--driver real"
            )
            return EXIT_NOT_RUN
        config = apai_backend.config_for_profile(config, profile)
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        arm = apai_backend.build_apai_arm(config, profile, clock)
    except Exception as error:
        _log.debug("building the APAI backend failed", exc_info=True)
        _complain(
            f"APAI arm profile {profile_name!r} cannot be used with {args.config}: "
            f"{error}"
        )
        return EXIT_NOT_RUN
    return run_connected(
        config, arm, checkers, clock, args.connect, driver=DRIVER_APAI_MOCK
    )


def confirm_real_driver(calibration: str, typed: Iterator[str], say: Say) -> bool:
    """Ask the three confirmations; True only when each was answered with yes.

    The first other answer, or the end of the input, ends the asking. A
    question is written first: on a terminal, what was typed before it was
    written is dropped. A question that cannot be written and a last line that
    the end of the input cut short answer nothing.
    """
    named = repr(printable(calibration.strip(), MAX_WORD_CHARS))
    total = len(REAL_DRIVER_QUESTIONS)
    for number, question in enumerate(REAL_DRIVER_QUESTIONS, start=1):
        asked = question.format(calibration=named)
        try:
            say(f"Real driver, {number} of {total}: {asked} Type {YES} to confirm.")
            drop_typed_ahead(typed)
            answer = next(typed, None)
        except (OSError, ValueError):  # the terminal is gone
            answer = None
        if getattr(typed, "unterminated", False) or not confirms(answer, YES):
            return False
    return True


def _stdin_is_terminal() -> bool:
    return _is_terminal(sys.stdin)


def _stdout_is_terminal() -> bool:
    return _is_terminal(sys.stdout)


def _is_terminal(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no such stream, or a closed one
        return False


def real_driver_refusal(
    config: RobotAppConfig,
    profile: Any,
    calibration: str,
    apai_backend: Any,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Why the real driver may not be connected; None when nothing speaks against it.

    In turn, with nothing built or sent: an AI agent's environment (environ
    None: the process's); a --calibration that is not the file's
    camera_calibration_id; APAI's hold marker for the arm's motor bus; a hold
    folder of APAI's that cannot be listed; any operator's confirmation file
    in that folder, whatever arm it names: with one APAI's driver would
    connect an arm it does not measure at rest.
    """
    marker = agent_environment(environ)
    if marker is not None:
        return (
            f"{NOT_CONFIRMED}: the real driver is started by a person, outside "
            f"an agent environment ({marker} is set)"
        )
    in_file = config.camera_calibration_id
    if not in_file or calibration != in_file:
        named = repr(printable(calibration, MAX_WORD_CHARS))
        return (
            f"{NOT_CONNECTED}: --calibration names {named} but "
            f"camera_calibration_id of the file is {in_file!r}; the two must be "
            "the same name, and not empty"
        )
    hold = apai_backend.unresolved_hold(profile)
    if hold is not None:
        when = hold.get("at_text") or "an unknown time"
        return (
            f"{NOT_CONNECTED}: the arm on {hold.get('key')} has an unresolved "
            f"hold: APAI's hold marker of {when} says {hold.get('reason')!r}. "
            "Connecting would cut the torque of an arm that may be raised. Do "
            "APAI's operator recovery first; a typed yes never releases a hold"
        )
    unlistable = apai_backend.hold_folder_unlistable()
    if unlistable is not None:
        return (
            f"{NOT_CONNECTED}: APAI's hold folder cannot be listed ({unlistable}): "
            "an operator's confirmation may lie in it, and APAI's driver would "
            "find it by its name. Let the user who starts this program read and "
            "enter the folder (its permissions), then start again"
        )
    on_file = apai_backend.confirmation_on_file()
    if on_file is not None:
        return (
            f"{NOT_CONNECTED}: an operator's confirmation for APAI's connect is "
            f"on file ({on_file}): with one on file APAI's driver would connect "
            "an arm it does not measure at rest. Revoke it with APAI's own tool "
            "(arm_hold.py revoke), then start again"
        )
    return None


def _real_driver(
    apai_backend: Any,
    config: RobotAppConfig,
    args: argparse.Namespace,
    *,
    typed: Iterator[str] | None = None,
    say: Say = _say,
) -> int:
    """--driver real: the checks, the three typed confirmations, then the app.

    EXIT_NOT_RUN when a check or a confirmation failed: nothing was connected.
    Else run_connected()'s exit code; it connects the arm once, and APAI's
    driver refuses there an arm it does not measure at rest. typed yields the
    answers and then the Thor terminal's lines; None reads the standard input,
    which must be a terminal.
    """
    problem = live_view_problem(config, DRIVER_REAL)
    if problem is not None:
        # Said before any check or question, as every configuration error.
        _complain(f"the configuration cannot be used: {problem}")
        return EXIT_NOT_RUN
    name = args.apai_profile
    try:
        profile = apai_backend.load_apai_profile(name)
    except Exception as error:
        _log.debug("loading the APAI profile failed", exc_info=True)
        _complain(f"APAI arm profile {name!r} cannot be used: {error}")
        return EXIT_NOT_RUN
    if profile.arm_cfg.get("type") == "mock":
        _complain(
            f"APAI arm profile {name!r} is APAI's software arm: use --driver mock"
        )
        return EXIT_NOT_RUN
    if typed is None:
        for is_terminal, why in (
            (_stdin_is_terminal, NO_TERMINAL),
            (_stdout_is_terminal, NO_OUTPUT_TERMINAL),
        ):
            if not is_terminal():
                _complain(f"{NOT_CONFIRMED}: {why}")
                return EXIT_NOT_RUN
    refusal = real_driver_refusal(config, profile, args.calibration, apai_backend)
    if refusal is not None:
        _complain(refusal)
        return EXIT_NOT_RUN
    clock = SystemClock()
    try:
        config = apai_backend.config_for_profile(config, profile)
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        arm = apai_backend.build_apai_arm(config, profile, clock)
        cap = apai_backend.speed_cap(profile)
    except Exception as error:
        _log.debug("building the APAI backend failed", exc_info=True)
        _complain(
            f"{NOT_CONNECTED}: APAI arm profile {name!r} cannot be used with "
            f"{args.config}: {error}"
        )
        return EXIT_NOT_RUN
    missing = _remote_link_missing()
    if missing is not None:
        _complain(f"{NOT_CONNECTED}: --connect cannot run: {missing}")
        return EXIT_NOT_RUN
    say(
        f"robot app: APAI arm profile {name}, the real driver on "
        f"{apai_backend.hold_bus(profile)}: built, not connected. No hold marker; "
        f"calibration {config.camera_calibration_id}; speed cap {cap:g} rad/s; "
        f"at rest at start within {config.rest.start_rest_tol_rad:g} rad"
    )
    lines = typed_lines() if typed is None else typed
    signals = _StartSignals()
    try:
        try:
            confirmed = confirm_real_driver(args.calibration, lines, say)
            if confirmed and threading.current_thread() is threading.main_thread():
                # From here on Ctrl+C is noted: it must not end a connect half-way.
                signals.install()
        except KeyboardInterrupt:  # Ctrl+C at a question confirms nothing
            confirmed = False
        # A hold marker or a confirmation may have been written during the
        # questions.
        late = (
            real_driver_refusal(config, profile, args.calibration, apai_backend)
            if confirmed
            else NOT_CONFIRMED
        )
        if late is None and (signals.interrupts or signals.terminated):
            late = NOT_CONFIRMED  # Ctrl+C or SIGTERM while the checks were made
        if late is None:
            try:
                say(CONNECTING)
            except (OSError, ValueError):
                late = NOT_CONFIRMED  # the terminal is gone, and its person
        if late is not None:
            if typed is None:
                _end_input(lines)
            _complain(late)
            return EXIT_NOT_RUN
        return run_connected(
            config,
            arm,
            checkers,
            clock,
            args.connect,
            typed=lines,
            say=say,
            driver=DRIVER_REAL,
            signals=signals,
        )
    finally:
        signals.restore()


def _describe_apai(
    apai_backend: Any, config: RobotAppConfig, config_path: str, profile_name: str
) -> int:
    """Build APAI's checkers and arm port, connect nothing, say what would run."""
    clock = SystemClock()
    try:
        profile = apai_backend.load_apai_profile(profile_name)
        config = apai_backend.config_for_profile(config, profile)
        checkers = apai_backend.build_apai_checkers(config, profile, clock)
        arm = apai_backend.build_apai_arm(config, profile, clock)
        cap = apai_backend.speed_cap(profile)
    except Exception as error:
        _log.debug("building the APAI backend failed", exc_info=True)
        _complain(
            f"APAI arm profile {profile_name!r} cannot be used with {config_path}: "
            f"{error}"
        )
        return EXIT_NOT_RUN
    backend = type(arm.backend).__name__
    versions = ", ".join(f"{c.name} {c.version}" for c in checkers.all())
    limits = config.limits
    speeds = ", ".join(
        f"{limit_set.name} {limit_set.max_joint_speed:g}"
        for limit_set in (limits.agent, limits.human, limits.recovery)
    )
    _say("robot app with --arm apai would run:")
    _say(
        f"  configuration {config_path}: robot {config.robot_id}, profile "
        f"{config.profile_id}, {config.n_joints} joints"
    )
    _say(f"  APAI arm profile {profile_name}: driver {backend}, built, not connected")
    _say(f"  one APAI checker per control owner, no obstacle grid: {versions}")
    _say(
        f"  the profile's speed cap: {cap:g} rad/s; the limit sets' joint speeds: "
        f"{speeds} rad/s"
    )
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
    _say(f"{NO_CONNECT}: the arm was not connected and nothing runs (exit code 2)")
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


def run_link_selftest(
    config: RobotAppConfig, say: Say = _say, socket_folder: str | None = None
) -> int:
    """Run the self-test over a real link; EXIT_OK when every step passed.

    In this process: a robot link service on 127.0.0.1 and a free port, the
    robot app on the simulated arm dialing it, and two clients of the local
    API. socket_folder holds the local API's socket; None makes a temporary
    folder. EXIT_NOT_RUN without Isaac Capture's remote link.
    """
    missing = _remote_link_missing()
    if missing is not None:
        say(f"selftest-link: skipped: {missing} (exit code 2)")
        return EXIT_NOT_RUN
    say(
        "selftest-link: the robot app on the simulated arm, through a robot link "
        "service over a real link on 127.0.0.1"
    )
    if socket_folder is not None:
        return _LinkSelfTest(config, say, socket_folder).run()
    with tempfile.TemporaryDirectory(prefix="agentic_remote-") as folder:
        return _LinkSelfTest(config, say, folder).run()


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
            driver=DRIVER_SIM,
            allow_reference=config.jaws.allow_reference,
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

    _label = "selftest"  # how each line of the run begins

    def run(self) -> int:
        total, label = len(self._steps), self._label
        began = self._clock.monotonic()
        failed = 0  # the number of the step that went wrong
        try:
            for number, (title, step) in enumerate(self._steps, start=1):
                try:
                    detail = step()
                except Exception as error:
                    _log.debug("self-test step %r failed", title, exc_info=True)
                    self._flush()
                    self._say(f"{label} {number:2d}/{total} FAILED  {title}: {error}")
                    failed = number
                    break
                self._flush()
                self._say(f"{label} {number:2d}/{total} ok  {title}: {detail}")
        finally:
            self._close()
        if failed:
            self._say(f"{label}: FAILED at step {failed} of {total}")
            return EXIT_FAILED
        took = self._clock.monotonic() - began
        self._say(f"{label}: all {total} steps passed in {took:.1f} s")
        return EXIT_OK

    def _flush(self) -> None:
        """Let the Thor terminal's lines come out before the step's own line."""
        self._terminal.flush()

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
        return self._stopped_move(self._client)

    def _stopped_move(self, stopper: WorkstationClient) -> str:
        """The agent's move, ended by a STOP that `stopper` sends."""
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
            stopper.stop("self-test: the operator pressed stop")
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
        self._type(PATH_IS_CLEAR)
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
        return self._released_by(self._client)

    def _released_by(self, operator: WorkstationClient) -> str:
        """A RELEASE this client sends, at rest."""
        command_id = operator.release()
        _ended(operator.wait_result(command_id, timeout_s=STEP_WAIT_S), "completed")
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


class _LinkSelfTest(_SelfTest):
    """The scripted run through a robot link service, over a real link.

    The agent's commands come from a client in the agent role, the operator's
    from one in the console role.
    """

    _label = "selftest-link"

    def __init__(self, config: RobotAppConfig, say: Say, socket_folder: str) -> None:
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
        self._app = RobotApp(
            config,
            self._arm,
            sim_checkers(config, self._clock),
            None,
            self._clock,
            driver=DRIVER_SIM,
            allow_reference=config.jaws.allow_reference,
        )
        self._socket_path = os.path.join(socket_folder, "robot_link.sock")
        self._listener: RobotListener | None = None
        self._service: RobotLinkService | None = None
        self._dialer: RobotDialer | None = None
        self._ends: list[LocalCarrier] = []  # agent, then console
        self._client: WorkstationClient | None = None  # the agent role
        self._console: WorkstationClient | None = None
        self._at = rest_q
        self._stop_number = 0
        self._owner_before_cut = 0
        self._sent_at_cut = 0  # setpoints the arm had been sent when it held
        self._steps = (
            ("start the robot link service", self._start_service),
            ("start the robot app; it dials the service", self._start),
            ("take ownership for the agent", self._take_ownership),
            ("a move with permits", self._permitted_move),
            ("a move without a permit", self._unpermitted_move),
            ("STOP from the console during a move", self._console_stop),
            ("a new task after the STOP, no reset", self._task_after_stop),
            ("the robot connection is cut during a move", self._cut_during_move),
            ("the robot app dials again", self._connected_again),
            ("rest from the console", self._console_rest),
            ("release from the console", self._console_release),
            ("close the robot app", self._close_app),
        )

    def _flush(self) -> None:
        return None  # no Thor terminal in this run

    def _close(self) -> None:
        """End every thread and connection; the simulated motors stay as they are."""
        for client in (self._client, self._console):
            if client is not None:
                client.close()
        for end in self._ends:
            end.close()
        if self._dialer is not None:
            self._dialer.close()
        self._app.close()
        if self._service is not None:
            self._service.close()
        if self._listener is not None:
            self._listener.close()

    # Steps

    def _start_service(self) -> str:
        self._listener = RobotListener("127.0.0.1", 0)
        self._service = RobotLinkService(
            self._listener.accept,
            socket_path=self._socket_path,
            allowed_peers=("127.0.0.1",),
        )
        self._service.start()
        return (
            f"waits for the robot app on {self._listener.endpoint}; local API at "
            f"{self._service.socket_path}"
        )

    def _start(self) -> str:
        endpoint = self._listener.endpoint
        self._dialer = RobotDialer(
            self._app,
            lambda: LinkCarrier.connect(endpoint, CONNECT_TIMEOUT_S, name=LINK_NAME),
            self._clock,
            retry_s=self._config.connection.retry_s,
        )
        self._app.start()
        self._dialer.start()
        for role in (ROLE_AGENT, ROLE_CONSOLE):
            end = connect_local(
                self._socket_path, role, f"selftest-{role}", STEP_WAIT_S
            )
            self._ends.append(end)
        agent_end, console_end = self._ends
        self._client = WorkstationClient(
            agent_end, self._clock, name="agent", quotable=self._current
        )
        self._console = WorkstationClient(
            console_end, self._clock, name="console", quotable=self._current
        )
        agent_end.wait_status(lambda s: s.robot_connected, STEP_WAIT_S)
        welcome = self._client.wait_welcome(STEP_WAIT_S)
        _need(welcome.session_id == self._app.session_id, "WELCOME of another session")
        state = self._state(lambda s: s.motion == "rest_hold" and s.torque_on)
        _need(state.gripper.load == "empty", f"the jaws read {state.gripper.load}")
        return (
            f"robot connection up after {self._dialer.attempts} dial(s), session "
            f"{welcome.session_id[:8]}, at rest with the motors on; an agent and a "
            "console client are attached"
        )

    def _console_stop(self) -> str:
        return self._stopped_move(self._console)

    def _cut_during_move(self) -> str:
        before = self._state(self._current)
        self._owner_before_cut = before.owner_number
        self._at = self._client.measured_pose()
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
            link = self._dialer.carrier
            _need(link is not None, "the dialer holds no robot connection")
            link.close()
            result = self._client.wait_result(move_id, timeout_s=STEP_WAIT_S)
        finally:
            feed.stop()
        _need(
            _ended(result, "failed").reason == LOST,
            f"the move's RESULT gives the reason {result.reason!r}, not {LOST!r}",
        )
        self._ends[0].wait_status(lambda s: not s.robot_connected, STEP_WAIT_S)
        # A command sent while there is no robot connection goes nowhere.
        numbers = {
            "session_id": before.session_id,
            "owner_number": before.owner_number,
            "stop_number": before.stop_number,
        }
        command_id = self._client.move([self._at], start_q=self._at, **numbers)
        _ended(self._client.wait_result(command_id, timeout_s=STEP_WAIT_S), "refused")
        _need(self._app.wait_motion_idle(STEP_WAIT_S), "the move did not end")
        now = self._app.snapshot()
        _need(now.owner is Owner.NOBODY, f"the control owner is {now.owner}")
        _need(self._app.motion_state is MotionState.HOLD, "the arm does not hold")
        _need(self._arm.torque_on(), "the motors are off")
        self._sent_at_cut = len(self._arm.sent)
        return (
            f"the move's RESULT is failed: {LOST}; the arm holds with the motors "
            "on, nobody owns it; a command sent meanwhile is refused"
        )

    def _connected_again(self) -> str:
        retry_s = self._config.connection.retry_s
        status = self._ends[0].wait_status(
            lambda s: s.robot_connected, retry_s + STEP_WAIT_S
        )
        _need(status.session_id == self._app.session_id, "another session id")
        old = self._owner_before_cut
        state = self._state(
            lambda s: s.owner_number > old and s.motion == "hold" and s.torque_on
        )
        _need(state.owner == "nobody", f"the control owner is {state.owner}")
        at = self._client.measured_pose()
        command_id = self._client.move([at], start_q=at, owner_number=old)
        result = self._client.wait_result(command_id, timeout_s=STEP_WAIT_S)
        _ended(result, "refused")
        _need(
            len(self._arm.sent) == self._sent_at_cut,
            "a setpoint was sent after the robot connection was lost",
        )
        self._stop_number = state.stop_number
        return (
            f"robot connection up again after {self._dialer.attempts} dial(s) in "
            f"all, the same session; nothing moved; the agent's move quoting owner "
            f"number {old} is refused: {result.reason}"
        )

    def _console_rest(self) -> str:
        rest = self._config.rest
        longest = planned_duration_s(
            FIRST_MOVE_RAD + 2.0 * SECOND_MOVE_RAD,
            rest.recovery_speed,
            rest.min_duration_s,
        )
        command_id = self._console.rest(confirmed=True)
        result = self._console.wait_result(command_id, timeout_s=STEP_WAIT_S + longest)
        _ended(result, "completed")
        self._state(lambda s: s.motion == "rest_hold" and s.torque_on)
        off = _distance(self._arm.pose(), rest.rest_q)
        return f"the arm holds at rest with the motors on, {off:.3f} rad from rest"

    def _console_release(self) -> str:
        return self._released_by(self._console)

    def _close_app(self) -> str:
        self._dialer.close()
        self._app.close()
        _need(self._app.wait_closed(STEP_WAIT_S), "the robot app did not close")
        self._ends[0].wait_status(lambda s: not s.robot_connected, STEP_WAIT_S)
        _need(len(self._arm.released_at) == 1, "torque was released a second time")
        return (
            "the robot app has closed; the service shows the robot connection as lost"
        )

    # Helpers

    def _current(self, state: RobotState) -> bool:
        """Whether a ROBOT_STATE shows the numbers the robot app has now."""
        if not (self._app.running and self._app.link_alive):
            return True  # no newer ROBOT_STATE will come
        now = self._app.snapshot()
        shown = (state.session_id, state.owner_number, state.stop_number)
        return shown == (now.session_id, now.owner_number, now.stop_number)


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
