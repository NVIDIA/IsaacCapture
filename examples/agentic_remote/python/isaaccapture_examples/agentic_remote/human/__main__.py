# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""python -m isaaccapture_examples.agentic_remote.human --config PATH.

The human program: the program through which a person drives the arm with the
headset's right controller. It attaches in the role "human" to the socket its
file names: on the rig the robot app's own on the Thor, the machine on the
robot; on one machine the robot link service's. On every pass of its loop it
takes controller data from a controller feed, runs the retargeting graph and
the inverse kinematics (IK) on them and sends link messages to the robot app,
the program that talks to the motors: joint targets in HUMAN_SETPOINT, a STOP
for right B, and the requests for the takeover (a person becomes the control
owner, the one who may move the arm) and for the hand-back that ends it at rest.

Isaac Capture and APAI are imported inside main(): the settings and the usage
text work without them.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import queue
import select
import signal
import statistics
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import yaml

from ..link_service.local import LocalCarrier, connect_local
from ..protocol.local_messages import ROLE_HUMAN
from ..protocol.messages import (
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    SIMULATED_DRIVERS,
    HumanSetpoint,
    Message,
    Result,
    RobotState,
    Stop,
    Welcome,
)
from .types import (
    ControllerFeed,
    FeedStep,
    GraphStep,
    HumanSettings,
    IkSettings,
    IkStep,
    RetargetSettings,
    SolverPort,
    TeleopSettings,
)

if TYPE_CHECKING:
    from .retarget import HumanGraph

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_main")

EXIT_OK = 0
# Neither a robot link service nor a robot app at the socket, a second human
# program, an arm model that does not fit WELCOME, a synthetic controller
# against a real arm, no Teleop Session Server after teleop.attempts attempts,
# or an error in a pass.
EXIT_FAILED = 1
EXIT_NOT_RUN = 2  # the arguments, the file or the installation cannot be used

PROGRAM = "human program"
LOG_LEVELS = ("debug", "info", "warning", "error")
CONTROLLER_TELEOP = "teleop"
CONTROLLER_SYNTHETIC = "synthetic:"
ATTACH_TIMEOUT_S = 5.0
APAI_SRC_ENV = "AGENTIC_REMOTE_APAI_SRC"
ISAAC_CAPTURE_ENV = "AGENTIC_REMOTE_ISAAC_CAPTURE_PY"
# The STOP the program sends when it ends while the person owns the arm: on
# SIGINT, SIGTERM or the stop event, and when it fails, finds no Teleop
# Session Server or must not drive the robot app's new session.
END_STOP_REASON = "the human program ends"
END_STOP_SENDER = "human_program"
# WELCOME.driver "" declares nothing; every reader treats that as the real arm.
NOT_DECLARED = "not declared (treated as real)"
TRIGGER_DOWN = 0.5  # the trigger counts as down above this
# Samples kept for one report line, and RESULTs kept for one pass.
SAMPLES_KEPT = 4096
RESULTS_KEPT = 1024
# Lines that may wait for a standard output that takes none; more are dropped.
LINES_KEPT = 256
_WRITER_JOIN_S = 2.0  # longest wait, at the end, for the lines still to say
_WELCOME_POLL_S = 0.05
_ERROR_LOG_S = 1.0  # a pass that keeps failing is logged this often
_OWNER_HUMAN = "human"
# Of a text from the Thor or a feed, or of a value of the file, in one line.
_MAX_TEXT_CHARS = 200
_NEW_SESSION = "the robot app has a new session: %s"  # a log line

Say = Callable[[str], None]


class HumanConfigError(ValueError):
    """A setting is missing, unknown, of the wrong kind or out of range."""


# The blocks of the file whose keys are the fields of a dataclass.
_BLOCK_TYPES: dict[str, type] = {
    "teleop": TeleopSettings,
    "retarget": RetargetSettings,
    "ik": IkSettings,
}
# The other blocks: key -> the field of HumanSettings it fills.
_BLOCK_KEYS: dict[str, dict[str, str]] = {
    "buttons": {"hold_s": "hold_s"},
    "rates": {"loop_hz": "loop_hz", "idle_hz": "idle_hz"},
    "state": {"max_age_s": "state_max_age_s"},
}
# The keys at the top of the file, which are fields of HumanSettings.
_TOP_KEYS = {"socket_path": "socket_path", "log_level": "log_level"}
# How a key's value is checked; a key in none of these is a number above 0.
_TEXT = ("socket_path", "endpoint", "apai_arm")
_WHOLE = ("attempts", "max_iter", "max_failures")  # 1 or more
_MAY_BE_NONE = ("reject_linear_velocity", "reject_angular_velocity")
_ANY_NUMBER = ("stage_yaw_deg",)
_AT_LEAST = {"retry_s": 2.0}
_BELOW = {"squeeze_threshold": 1.0}  # and above 0
_CONTROLLER_POSES = ("grip", "aim")
# The most rates.loop_hz may be. The rate limiter counts a pass as at least
# half of LIMITER_PASS_S of human/retarget.py, 10 ms: with more passes per
# second the target would move faster than both hand speed limits allow.
MAX_LOOP_HZ = 100.0


def load_human_settings(path: str | os.PathLike[str]) -> HumanSettings:
    """The settings in a YAML file; HumanConfigError names the one that is wrong.

    A setting the file leaves out keeps its first value; one it does not know
    is an error.
    """
    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    # A ValueError: a file that is not UTF-8, or a whole number of more digits
    # than Python reads.
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise HumanConfigError(f"cannot read {os.fspath(path)}: {error}") from error
    return human_settings(data)


def human_settings(data: object) -> HumanSettings:
    """The settings in a mapping as the YAML file holds it."""
    top = _block("the file", data)
    values: dict[str, Any] = {}
    for name, kind in _BLOCK_TYPES.items():
        names = [field.name for field in dataclasses.fields(kind)]
        block = _block(name, top.pop(name, None))
        values[name] = kind(**_taken(f"{name}.", block, dict(zip(names, names))))
    for name, keys in _BLOCK_KEYS.items():
        values.update(_taken(f"{name}.", _block(name, top.pop(name, None)), keys))
    values.update(_taken("", top, _TOP_KEYS))  # what is left of the file
    if values.get("socket_path"):
        values["socket_path"] = os.path.expanduser(values["socket_path"])
    return HumanSettings(**values)


def _block(where: str, value: object) -> dict[Any, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HumanConfigError(f"{where} must be a mapping, got {_got(value)}")
    return dict(value)


def _taken(
    where: str, block: Mapping[Any, Any], keys: Mapping[str, str]
) -> dict[str, Any]:
    """The checked values of one block, by the field each key fills."""
    unknown = sorted(_got(key, str) for key in block if key not in keys)
    if unknown:
        raise HumanConfigError(
            "unknown setting(s): " + ", ".join(where + key for key in unknown)
        )
    return {
        keys[key]: _checked(where + key, key, value) for key, value in block.items()
    }


def _checked(where: str, key: str, value: Any) -> Any:
    """One value of the file, of the kind and in the range its key asks for."""
    if key == "log_level":
        return _one_of(where, value, LOG_LEVELS)
    if key == "controller_pose":
        return _one_of(where, value, _CONTROLLER_POSES)
    if key == "controller_to_base":
        rows = value if isinstance(value, (list, tuple)) else ()
        if len(rows) != 3 or not all(
            isinstance(row, (list, tuple)) and len(row) == 3 for row in rows
        ):
            raise HumanConfigError(
                f"{where} must be 3 rows of 3 numbers, got {_got(value)}"
            )
        return tuple(
            tuple(_number(f"{where}[{i}][{j}]", item) for j, item in enumerate(row))
            for i, row in enumerate(rows)
        )
    if key in _TEXT:
        if not isinstance(value, str) or not (value.strip() or key == "socket_path"):
            raise HumanConfigError(f"{where} must be text, got {_got(value)}")
        return value
    if key in _WHOLE:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise HumanConfigError(
                f"{where} must be a whole number of 1 or more, got {_got(value)}"
            )
        return int(value)
    if key in _MAY_BE_NONE and value is None:
        return None
    number = _number(where, value)
    if key in _ANY_NUMBER:
        return number
    if key in _AT_LEAST:
        if number < _AT_LEAST[key]:
            raise HumanConfigError(
                f"{where} must be at least {_AT_LEAST[key]:g}, got {value!r}"
            )
    elif number <= 0.0:
        raise HumanConfigError(f"{where} must be above 0, got {value!r}")
    elif number >= _BELOW.get(key, math.inf):
        raise HumanConfigError(
            f"{where} must be above 0 and below {_BELOW[key]:g}, got {value!r}"
        )
    elif key == "loop_hz" and number > MAX_LOOP_HZ:
        raise HumanConfigError(
            f"{where} must be at most {MAX_LOOP_HZ:g}, got {value!r}: the rate "
            f"limiter counts a pass as at least {1000.0 / MAX_LOOP_HZ:g} ms, so "
            f"above {MAX_LOOP_HZ:g} both hand speed limits would rise above "
            "their settings"
        )
    return number


def _one_of(where: str, value: Any, words: Sequence[str]) -> str:
    if value not in words:
        raise HumanConfigError(
            f"{where} must be one of {', '.join(words)}, got {_got(value)}"
        )
    return value


def _number(where: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HumanConfigError(f"{where} must be a number, got {_got(value)}")
    try:
        number = float(value)
    except OverflowError:  # a whole number too large for a float
        number = math.inf
    if not math.isfinite(number):
        raise HumanConfigError(f"{where} must be a finite number, got {_got(value)}")
    return number


def _got(value: object, text: Callable[[object], str] = repr) -> str:
    """A value or a key of the file for an error's text, cut to one line.

    For one that was not checked to be a finite number: a whole number can
    have more digits than Python prints.
    """
    try:
        return _plain(text(value))
    except ValueError:
        return "a whole number too large to print"


class _MachineClock:
    def monotonic(self) -> float:
        return time.monotonic()


class _Inbox:
    """What the robot app sent: the newest WELCOME and ROBOT_STATE, the RESULTs.

    on_message runs on the receive thread of the program's client.
    """

    def __init__(self, clock: _MachineClock) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._welcome: Welcome | None = None
        self._state: RobotState | None = None
        self._state_at: float | None = None  # when the ROBOT_STATE arrived
        self._results: deque[Result] = deque(maxlen=RESULTS_KEPT)

    def on_message(self, message: Message) -> None:
        now = self._clock.monotonic()
        with self._lock:
            if isinstance(message, RobotState):
                self._state, self._state_at = message, now
            elif isinstance(message, Result):
                self._results.append(message)
            elif isinstance(message, Welcome):
                self._welcome = message

    def welcome(self) -> Welcome | None:
        with self._lock:
            return self._welcome

    def state(self) -> RobotState | None:
        """The newest ROBOT_STATE, None before the first one."""
        with self._lock:
            return self._state

    def take(
        self,
    ) -> tuple[Welcome | None, RobotState | None, float | None, list[Result]]:
        """The WELCOME, the ROBOT_STATE, when it arrived, and the RESULTs since."""
        with self._lock:
            results = list(self._results)
            self._results.clear()
            return self._welcome, self._state, self._state_at, results


class _Ending:
    """Says when the program is to end: the stop event, or SIGINT or SIGTERM."""

    def __init__(self, stop: threading.Event | None) -> None:
        """Without a stop event the two signals are caught; main thread only."""
        self._stop = stop
        # The handler only queues: SimpleQueue.put() is reentrant.
        self._came: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._replaced: dict[int, Any] = {}
        if stop is None:
            for number in (signal.SIGINT, signal.SIGTERM):
                self._replaced[number] = signal.signal(
                    number, lambda came, _: self._came.put(came)
                )

    def wait(self, seconds: float) -> str | None:
        """Wait up to seconds; why the program is to end, or None to go on."""
        seconds = max(0.0, seconds)
        if self._stop is not None:
            return "the stop event was set" if self._stop.wait(seconds) else None
        try:
            # A wait of no time at all would raise at once.
            number = self._came.get(timeout=max(seconds, 1e-4))
        except queue.Empty:
            return None
        return signal.Signals(number).name

    def restore(self) -> None:
        """Put the signal handlers back as they were."""
        for number, handler in self._replaced.items():
            signal.signal(number, signal.SIG_DFL if handler is None else handler)
        self._replaced = {}


class _Writer:
    """Says the program's lines on a thread of its own, as the console does.

    A standard output that blocks then holds up no pass: right B stays a STOP.
    """

    def __init__(self, say: Say) -> None:
        self._say = say
        # A line, or None to end the writer.
        self._lines: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="human-out", daemon=True)
        self._thread.start()

    def __call__(self, line: str) -> None:
        """Queue one line; it is dropped while LINES_KEPT lines wait."""
        if self._lines.qsize() < LINES_KEPT:
            self._lines.put(line)

    def _run(self) -> None:
        failed = False
        while (line := self._lines.get()) is not None:
            try:
                self._say(line)
            except Exception as error:
                if not failed:
                    _log.warning("the standard output cannot be written to: %r", error)
                failed = True
            else:
                failed = False

    def close(self) -> bool:
        """End the writer behind the queued lines; False while it still blocks."""
        self._lines.put(None)
        self._thread.join(_WRITER_JOIN_S)
        return not self._thread.is_alive()


class _TimedSolver:
    """The solver with the time of each solve; it satisfies SolverPort."""

    def __init__(self, solver: SolverPort) -> None:
        self._solver = solver
        self.n_joints = solver.n_joints
        self.times_s: deque[float] = deque(maxlen=SAMPLES_KEPT)

    def fk(self, q: Sequence[float]) -> Any:
        return self._solver.fk(q)

    def inside(self, q: Sequence[float]) -> tuple[float, ...]:
        return self._solver.inside(q)

    def solve(self, base_T_ee: Any, q_seed: Sequence[float]) -> IkStep:  # noqa: N803
        started = time.perf_counter()
        try:
            return self._solver.solve(base_T_ee, q_seed)
        finally:
            self.times_s.append(time.perf_counter() - started)


def _plain(text: object, limit: int = _MAX_TEXT_CHARS) -> str:
    """str(text) for one printed line: printable, at most `limit` characters."""
    text = "".join(char if char.isprintable() else " " for char in str(text))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fixed(value: float, digits: int) -> str:
    """A number with this many decimals; a zero has no sign."""
    text = f"{value:.{digits}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def _ms(seconds: Any, digits: int = 0) -> str:
    """A time in ms; "-" for one that is not known."""
    if isinstance(seconds, bool) or not isinstance(seconds, Real):
        return "-"
    if not math.isfinite(seconds):
        return "-"
    return _fixed(float(seconds) * 1000.0, digits)


def _spread(seconds: Sequence[float], digits: int = 0) -> str:
    """The lowest, the middle and the highest of some times, in ms."""
    if not seconds:
        return "-/-/-"
    three = (min(seconds), statistics.median(seconds), max(seconds))
    return "/".join(_ms(value, digits) for value in three)


def _xyz(point: Any) -> str:
    """A position in m; "-" for each value when there is none."""
    if point is None:
        return "x - y - z -"
    x, y, z = (_fixed(float(value), 3) for value in point)
    return f"x {x} y {y} z {z}"


def _turn(pose: Any) -> str:
    """The turn of a 4x4 pose about x, y and z (roll, pitch, yaw), in degrees."""
    if pose is None:
        return "roll - pitch - yaw -"
    roll = math.atan2(float(pose[2][1]), float(pose[2][2]))
    pitch = math.asin(min(1.0, max(-1.0, -float(pose[2][0]))))
    yaw = math.atan2(float(pose[1][0]), float(pose[0][0]))
    roll, pitch, yaw = (_fixed(math.degrees(turn), 1) for turn in (roll, pitch, yaw))
    return f"roll {roll} pitch {pitch} yaw {yaw}"


def _human_facts(state: RobotState | None) -> dict[str, Any]:
    """health["human"] of a ROBOT_STATE; empty when the robot app reports none."""
    facts = None if state is None else state.health.get("human")
    return facts if isinstance(facts, dict) else {}


def as_sent(
    setpoint: HumanSetpoint,
    asked_mono: float,
    state_at: float | None,
    now_mono: float,
) -> HumanSetpoint:
    """setpoint as it is sent at now_mono, on this machine's monotonic clock.

    asked_mono is when the pass asked the feed for the controller data: the
    sample age gains the time since, the pass's own work. state_at is when the
    ROBOT_STATE the setpoint was computed on arrived: sent_mono_s is that
    one's clock_mono_s plus the time since, so never later than the robot
    app's clock reads now. The robot app adds the time from then to the age.
    """
    age = setpoint.sample_age_s
    if age is not None:
        age += max(0.0, now_mono - asked_mono)
    sent = None
    if setpoint.state_clock_mono_s is not None and state_at is not None:
        sent = setpoint.state_clock_mono_s + max(0.0, now_mono - state_at)
    return dataclasses.replace(setpoint, sample_age_s=age, sent_mono_s=sent)


def report_line(
    stage: str,
    tick_ages_s: Sequence[float],
    state_ages_s: Sequence[float],
    state: RobotState | None,
    last: GraphStep | None,
    ik_times_s: Sequence[float],
    counts: Mapping[str, int],
) -> str:
    """The report line: the last pass, and the ages and times since the last line.

    tick age is the sample age, how old the controller data of a pass are: on
    the headset path the age of the Teleop Session Server's newest data. hand
    is the right controller's position in the stage space, the space its poses
    are given in; tip target is the target of the gripper tip in the robot's
    base axes. round trip is the Thor's measure of the way of a ROBOT_STATE to
    this program and of the HUMAN_SETPOINT back.
    """
    human = _human_facts(state)
    tip = None if last is None else last.ee_target
    ik = "-/-"
    if ik_times_s:
        ik = f"{_ms(statistics.median(ik_times_s), 2)}/{_ms(max(ik_times_s), 2)}"
    line = (
        f"{PROGRAM}: stage {_plain(stage) or '-'}; "
        f"tick age {_spread(tick_ages_s)} ms; "
        f"state age {_spread(state_ages_s)} ms; "
        f"round trip {_ms(human.get('round_trip_s'))} ms; "
        f"owner {'-' if state is None else _plain(state.owner)}; "
        f"engaged {'yes' if last is not None and last.engaged else 'no'}; "
        f"hand {_xyz(None if last is None else last.hand)} m; "
        f"tip target {_xyz(None if tip is None else tip[:3, 3])} m, "
        f"{_turn(tip)} deg; "
        f"IK {ik} ms; "
        f"ended: no solution {counts.get('no_solution', 0)}, "
        f"jump {counts.get('jump', 0)}, "
        f"solution ahead {counts.get('solution_ahead', 0)}, "
        f"hand ahead {counts.get('hand_ahead', 0)}, "
        f"limiter {counts.get('limiter', 0)}"
    )
    line += _jaws_text(human.get("jaws"))
    refusal = human.get("refusal")
    if refusal:
        line += f"; refused by the Thor: {_plain(refusal)}"
    return line


def _jaws_text(jaws: Any) -> str:
    """The report line's part about the jaws under the trigger; "" without facts.

    jaws is health["human"]["jaws"] of a ROBOT_STATE, which the robot app
    gives while a takeover is open. The grip level is the jaw opening in force
    at the takeover; the jaws follow the trigger once the opening it asks for
    has reached that level. An opening is a share of the jaw travel, 1.00
    fully open.
    """
    if not isinstance(jaws, dict):
        return ""
    text = (
        f"; jaws follow the trigger {'yes' if jaws.get('matched') else 'no'}, "
        f"grip level {_share(jaws.get('grip_level'))}, "
        f"trigger asks {_share(jaws.get('asked'))}"
    )
    refusal = jaws.get("refusal")
    return f"{text}: {_plain(refusal)}" if refusal else text


def _share(value: Any) -> str:
    """A share of the jaw travel with two decimals; "-" for one not known."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return "-"
    try:
        number = float(value)
    except OverflowError:  # a whole number too large for a float
        return "-"
    return _fixed(number, 2) if math.isfinite(number) else "-"


class _Edges:
    """Says each change of the feed's stage, of the controller and of a button."""

    def __init__(self, say: Say, squeeze_threshold: float, stages: bool) -> None:
        """stages: the feed is a remote session, whose stage changes are said."""
        self._say = say
        self._threshold = squeeze_threshold
        self._stages = stages
        self._stage: str | None = None
        self._seen = False
        self._down = dict.fromkeys(("A", "B", "grip", "trigger"), False)

    def note(self, feed_step: FeedStep, last: GraphStep | None) -> None:
        """Take one pass; last is the graph's result, None when the feed gave none."""
        if self._stages and feed_step.stage != self._stage:
            self._stage = feed_step.stage
            text = f": {_plain(feed_step.note)}" if feed_step.note else ""
            self._say(f"{PROGRAM}: Teleop Session Server {_plain(self._stage)}{text}")
        seen = last is not None and bool(last.present)
        if seen != self._seen:
            self._seen = seen
            self._say(f"{PROGRAM}: right controller {'seen' if seen else 'lost'}")
        if last is None:
            return  # the buttons are not known
        now = {
            "A": bool(last.a),
            "B": bool(last.b),
            "grip": bool(last.squeeze > self._threshold),
            "trigger": bool(last.trigger > TRIGGER_DOWN),
        }
        for name, down in now.items():
            if down != self._down[name]:
                self._down[name] = down
                self._say(f"{PROGRAM}: {name} {'down' if down else 'up'}")


def _feed_text(feed: ControllerFeed) -> str:
    """How the start line names a feed."""
    if feed.source == HUMAN_SOURCE_SYNTHETIC:
        return f'synthetic controller "{_plain(getattr(feed, "name", ""))}"'
    return _plain(feed.source)


class _Loop:
    """The human program on an attached carrier: WELCOME first, then the passes."""

    def __init__(
        self,
        settings: HumanSettings,
        graph: HumanGraph,
        solver: _TimedSolver,
        logic: Any,
        feed: ControllerFeed,
        ran_out: type[BaseException],
        refused: type[BaseException],
        carrier: LocalCarrier,
        client: Any,
        inbox: _Inbox,
        clock: _MachineClock,
        ending: _Ending,
        say: Say,
        report_s: float,
    ) -> None:
        """logic is the HumanLogic.

        ran_out and refused are the feed's errors for no Teleop Session Server
        and for a teleop.endpoint that cannot be used.
        """
        self._settings = settings
        self._graph = graph
        self._solver = solver
        self._logic = logic
        self._feed = feed
        self._ran_out = ran_out
        self._refused = refused
        self._carrier = carrier
        self._client = client
        self._inbox = inbox
        self._clock = clock
        self._ending = ending
        self._say = say
        self._report_s = report_s
        self._edges = _Edges(
            say,
            settings.retarget.squeeze_threshold,
            stages=feed.source == HUMAN_SOURCE_TELEOP,
        )
        self._session_id: str | None = None  # of the WELCOME that was checked
        self._tick_ages: deque[float] = deque(maxlen=SAMPLES_KEPT)
        self._state_ages: deque[float] = deque(maxlen=SAMPLES_KEPT)
        self._stage = ""
        self._report_due = math.inf
        self._error_due = -math.inf

    def run(self) -> int:
        """Run until the program is to end; the exit code."""
        # The socket is the robot link service's or the robot app's own.
        _log.info(
            "attached to the robot link service or the robot app; waiting for a WELCOME"
        )
        while True:
            welcome = self._inbox.welcome()
            if welcome is not None:
                break
            if not self._carrier.alive:
                return self._gone()
            why = self._ending.wait(_WELCOME_POLL_S)
            if why is not None:
                _log.info("%s before a WELCOME came; closing", why)
                return EXIT_OK
        problem = self._problem(welcome)
        if problem is not None:
            _complain(problem)
            return EXIT_FAILED
        self._session_id = welcome.session_id
        self._say(
            f"{PROGRAM}: {_feed_text(self._feed)}; driver "
            f"{_plain(welcome.driver) or NOT_DECLARED}; arm model "
            f"{_plain(self._settings.ik.apai_arm)}, {self._solver.n_joints} joints"
        )
        self._feed.open()
        try:
            return self._passes()
        finally:
            self._feed.close()

    def _problem(self, welcome: Welcome) -> str | None:
        """Why this program must not drive a WELCOME's session; None: it may."""
        joints, n_joints = len(welcome.joint_names), self._solver.n_joints
        if joints != n_joints:
            return (
                f"the arm model {self._settings.ik.apai_arm!r} has {n_joints} "
                f"joints and the robot app's WELCOME names {joints}: set "
                "ik.apai_arm to the model of this arm"
            )
        if (
            self._feed.source != HUMAN_SOURCE_TELEOP
            and welcome.driver not in SIMULATED_DRIVERS
        ):
            declared = f"driver {_plain(welcome.driver)}" if welcome.driver else None
            return (
                "a synthetic controller may drive only a simulated arm; the robot "
                f"app declares {declared or 'no driver, which counts as real'}"
            )
        return None

    def _passes(self) -> int:
        """One pass every 1 / rates.loop_hz seconds; a late pass is not made up.

        Where the program ends, its STOP goes out ahead of the lines that tell
        of the end: a log call and a write to standard error can block.
        """
        period = 1.0 / self._settings.loop_hz
        due = self._clock.monotonic()
        if self._report_s > 0.0:
            self._report_due = due + self._report_s
        while True:
            if not self._carrier.alive:
                return self._gone()
            try:
                drivable = self._pass()
            except self._ran_out as error:
                self._stop_if_owning("no Teleop Session Server")
                _complain(str(error))
                return EXIT_FAILED
            except self._refused as error:
                # The file's error, which the feed raises at its first attempt
                # only: nothing was sent.
                _complain(f"the settings cannot be used: {_plain(error)}")
                return EXIT_NOT_RUN
            except Exception as error:
                self._stop_if_owning("it failed")
                _log.exception("the human program failed")
                _complain(f"failed: {_plain(repr(error))}")
                return EXIT_FAILED
            if not drivable:
                return EXIT_FAILED
            due += period
            now = self._clock.monotonic()
            due = max(due, now)
            why = self._ending.wait(due - now)
            if why is not None:
                self._stop_if_owning(why)
                _log.info("%s; closing", why)
                return EXIT_OK

    def _pass(self) -> bool:
        """Controller data in, link messages out, in the order the logic asks for.

        False when a new session of the robot app came that this program must
        not drive: nothing but STOPs went out.
        """
        logic, graph = self._logic, self._graph
        # While the logic is not armed the graph runs stopped: nothing engages.
        events = graph.events(stopped=not logic.armed)
        inputs = graph.external_inputs(logic.armed)
        asked = self._clock.monotonic()
        feed_step = self._feed.step(inputs, events, asked)
        # Read after the feed: entering a remote session can take seconds.
        now = self._clock.monotonic()
        welcome, state, state_at, results = self._inbox.take()
        new_session = welcome is not None and welcome.session_id != self._session_id
        problem = self._problem(welcome) if new_session else None
        if problem is not None:
            self._end_on_session(welcome, problem, now, feed_step)
            return False
        if new_session:
            self._session_id = welcome.session_id
        state_age = None if state_at is None else max(0.0, now - state_at)
        try:
            messages = logic.step(now, feed_step, welcome, state, state_age, results)
        except Exception:
            # Nothing of this pass is sent. The logic un-arms itself when the
            # error came after its buttons were read.
            messages = []
            if now >= self._error_due:
                self._error_due = now + _ERROR_LOG_S
                _log.exception("a pass failed; no HUMAN_SETPOINT of it is sent")
        for message in messages:
            if isinstance(message, HumanSetpoint):
                # Its age is the one at this moment: the graph's and the IK's
                # time are in it.
                message = as_sent(message, asked, state_at, self._clock.monotonic())
            self._client.send(message)
        # A log call can block, as on a terminal that is paused: this pass's
        # log lines come behind its link messages, so right B's STOP is out.
        self._log_feed()
        if new_session:
            _log.info(_NEW_SESSION, _plain(welcome.session_id))
        logic.log_pass()
        self._edges.note(feed_step, logic.last)
        self._stage = feed_step.stage
        if feed_step.sample_age_s is not None:
            self._tick_ages.append(feed_step.sample_age_s)
        if state_age is not None:
            self._state_ages.append(state_age)
        if now >= self._report_due:
            self._report(now, state)
        return True

    def _end_on_session(
        self, welcome: Welcome, problem: str, now: float, feed_step: FeedStep
    ) -> None:
        """The last pass: a new session came that this program must not drive.

        Only STOPs go out: right B's when this pass read it down, then the
        program's own when the person owns the arm. The lines come behind.
        """
        # With no WELCOME the logic takes no ROBOT_STATE: of this pass it
        # gives right B's STOP and nothing else.
        for message in self._logic.step(now, feed_step, None, None, None):
            if isinstance(message, Stop):
                self._client.send(message)
        self._stop_if_owning("the robot app's new session cannot be driven")
        self._log_feed()
        _log.info(_NEW_SESSION, _plain(welcome.session_id))
        _complain(problem)
        self._logic.log_pass()

    def _log_feed(self) -> None:
        """Have the feed log the lines it kept; a feed need not keep any.

        The teleop feed keeps the line of a session it entered: that pass
        reads right B behind the entry.
        """
        log_pass = getattr(self._feed, "log_pass", None)
        if log_pass is not None:
            log_pass()

    def _report(self, now: float, state: RobotState | None) -> None:
        """Say the report line; the next one is due a period after this one was."""
        self._report_due += self._report_s
        if self._report_due <= now:
            self._report_due = now + self._report_s
        self._say(
            report_line(
                self._stage,
                self._tick_ages,
                self._state_ages,
                state,
                self._logic.last,
                self._solver.times_s,
                self._logic.counts(),
            )
        )
        for samples in (self._tick_ages, self._state_ages, self._solver.times_s):
            samples.clear()

    def _stop_if_owning(self, why: str) -> None:
        """One STOP when the newest ROBOT_STATE shows the person as the owner.

        Only a ROBOT_STATE of the session whose WELCOME this program took
        counts: one of a session before it does not say who owns the arm now.
        """
        state = self._inbox.state()
        if (
            state is None
            or state.session_id != self._session_id
            or state.owner != _OWNER_HUMAN
            or not self._carrier.alive
        ):
            return
        reason = f"{END_STOP_REASON} ({why})"
        self._client.send(Stop(reason=reason, sender=END_STOP_SENDER))
        _log.warning("STOP sent: the person owned the arm; %s", reason)

    def _gone(self) -> int:
        reason = _plain(self._carrier.close_reason)
        _complain(f"the robot link service is gone ({reason})")
        return EXIT_FAILED


def _say(text: str) -> None:
    """One line to standard output, once it takes one.

    The wait holds no lock, as a write that blocks does: the interpreter, and
    with it the program, can still end with exit code 0.
    """
    try:
        select.select([], [sys.stdout.fileno()], [])
    except (AttributeError, OSError, ValueError):
        pass  # a stream with no file descriptor: nothing to wait for
    print(text, flush=True)


def _complain(text: str) -> None:
    print(f"{PROGRAM}: {text}", file=sys.stderr, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaaccapture_examples.agentic_remote.human",
        description=(
            "The human program of the agentic remote execution example: a "
            "person drives the arm with the headset's right controller. Right B "
            "is a STOP; held for a second with the grip released it takes the "
            "arm over. The arm follows the hand while the grip is squeezed. "
            "The jaws follow the trigger once it has reached the jaw opening "
            "they had at the takeover. Right A held gives the arm back at rest."
        ),
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="the program's YAML file"
    )
    parser.add_argument(
        "--controller",
        default=CONTROLLER_TELEOP,
        metavar="FEED",
        help=(
            f"{CONTROLLER_TELEOP} (default): the headset, through the Teleop "
            f"Session Server; {CONTROLLER_SYNTHETIC}<name>: a scripted "
            "controller, lift, grasp or idle, for a simulated arm only"
        ),
    )
    parser.add_argument(
        "--report-s",
        default=1.0,
        type=float,
        metavar="S",
        help="seconds from one report line to the next (default 1.0; 0: none)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.lower,
        choices=LOG_LEVELS,
        metavar="L",
        help=f"console log level, one of {', '.join(LOG_LEVELS)} (default: the file's)",
    )
    return parser


def _script_name(controller: str) -> str | None:
    """The script --controller names; None for the headset."""
    if controller == CONTROLLER_TELEOP:
        return None
    name = controller.removeprefix(CONTROLLER_SYNTHETIC)
    if not controller.startswith(CONTROLLER_SYNTHETIC) or not name:
        raise HumanConfigError(
            f"--controller must be {CONTROLLER_TELEOP} or "
            f"{CONTROLLER_SYNTHETIC}<name>, got {controller!r}"
        )
    return name


def main(
    argv: Sequence[str] | None = None,
    *,
    attach: Callable[..., LocalCarrier] = connect_local,
    feed: (
        Callable[[HumanGraph, Callable[[], RobotState | None]], ControllerFeed] | None
    ) = None,
    stop: threading.Event | None = None,
    say: Say = _say,
) -> int:
    """Run the human program's command line; returns the exit code.

    attach(path, role, name, timeout_s) gives the carrier of the local API.
    feed(graph, newest_state) builds the controller feed in place of the one
    --controller names; newest_state() gives the newest ROBOT_STATE or None.
    stop, when given, ends the program once it is set, as SIGINT and SIGTERM
    do; without it the two signals are waited for, which only the main thread
    can. say gets every line of standard output, on a writer thread.
    """
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse has printed the usage or the help
        code = exit_.code
        return code if isinstance(code, int) else EXIT_NOT_RUN
    try:
        settings = load_human_settings(args.config)
        script = _script_name(args.controller)
        if not (math.isfinite(args.report_s) and args.report_s >= 0.0):
            raise HumanConfigError(f"--report-s must be 0 or more, got {args.report_s}")
    except HumanConfigError as error:
        _complain(f"the settings cannot be used: {error}")
        return EXIT_NOT_RUN
    if stop is None and threading.current_thread() is not threading.main_thread():
        _complain("signals can be waited for on the main thread only")
        return EXIT_NOT_RUN

    # Isaac Capture first, then APAI: each one that is missing has its own line.
    try:
        from . import feed as feeds
        from .retarget import HumanGraph
    except ImportError as error:
        _complain(
            f"Isaac Capture cannot be imported ({_plain(error)}); set "
            f"{ISAAC_CAPTURE_ENV} to the folder of its Python package and start "
            "the program with run_human.sh"
        )
        return EXIT_NOT_RUN
    from .ik import Solver

    ik = settings.ik
    try:
        solver = Solver.from_apai(
            ik.apai_arm, max_iter=ik.max_iter, tol=ik.tol, max_jump_rad=ik.max_jump_rad
        )
    except ImportError as error:
        _complain(
            f"APAI cannot be imported ({_plain(error)}); set {APAI_SRC_ENV} to the "
            "src folder of the APAI checkout"
        )
        return EXIT_NOT_RUN
    except (OSError, ValueError) as error:
        _complain(
            f"the settings cannot be used: ik.apai_arm {ik.apai_arm!r}: {_plain(error)}"
        )
        return EXIT_NOT_RUN
    from .. import console_log
    from ..testing.client import WorkstationClient
    from .logic import HumanLogic

    clock = _MachineClock()
    inbox = _Inbox(clock)
    try:
        graph = HumanGraph(settings.retarget)
        if feed is not None:
            made = feed(graph, inbox.state)
        elif script is None:
            made = feeds.TeleopFeed(graph, settings.teleop)
        else:
            made = feeds.ScriptFeed.named(graph, script, state=inbox.state)
    except ValueError as error:  # a rotation that is none, an unknown script
        _complain(f"the settings cannot be used: {_plain(error)}")
        return EXIT_NOT_RUN
    timed = _TimedSolver(solver)
    logic = HumanLogic(graph, timed, settings, made.source)

    console_log.set_console_level(args.log_level or settings.log_level)
    try:
        carrier = attach(
            settings.socket_path or None, ROLE_HUMAN, "human", ATTACH_TIMEOUT_S
        )
    except OSError as problem:  # also LocalRefused and TimeoutError
        # The socket is the robot link service's or the robot app's own.
        _complain(
            f"cannot attach to the robot link service or the robot app: {problem}"
        )
        return EXIT_FAILED
    try:
        client = WorkstationClient(
            carrier, clock, name="human", listeners=[inbox.on_message]
        )
        ending = _Ending(stop)
        out = _Writer(say)
        try:
            loop = _Loop(
                settings,
                graph,
                timed,
                logic,
                made,
                feeds.AttemptsRanOut,
                feeds.EndpointRefused,
                carrier,
                client,
                inbox,
                clock,
                ending,
                out,
                args.report_s,
            )
            return loop.run()
        finally:
            out.close()
            ending.restore()
            client.close()
    finally:
        carrier.close()


if __name__ == "__main__":
    raise SystemExit(main())
