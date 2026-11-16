# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The controller feeds of the human program: where its controller data come from.

A controller feed runs one pass of the retargeting graph on the newest data of
the right controller. With the graph's outputs it gives the sample age, which
is how old those data are, and its stage: how far it is with the data, one of
HUMAN_STAGES. A pass is one turn of the human program's loop.

The teleop feed is a remote session: a TeleopSession that is a client of Isaac
Capture's Teleop Session Server, the program that reads the headset. SAMPLES
is the server's link message with the newest controller data and READY the one
that says a headset is tracked. A tick is one pass of the server's loop; the
tick age is now minus the tick time of the newest SAMPLES. The script feed is
a synthetic controller: timed controller data from a list of rows, with no
headset.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from isaaccapture.retargeting_engine.interface import ComputeContext, GraphTime
from isaaccapture.schema import (
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    Point,
    Pose,
    Quaternion,
)

from ..protocol.messages import (
    HUMAN_SOURCE_SYNTHETIC,
    HUMAN_SOURCE_TELEOP,
    RobotState,
)
from .retarget import HumanGraph
from .types import FeedStep, TeleopSettings

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_feed")

# The stages a feed gives; each is one of HUMAN_STAGES.
STAGE_NO_SERVER = "no_server"
STAGE_CONNECTING = "connecting_openxr"
STAGE_WAITING = "waiting_for_headset"
STAGE_READY = "ready"
STAGE_STALE = "stale"
STAGE_NO_CONTROLLER = "no_controller"

# The name a remote session gives the Teleop Session Server for its client.
APP_NAME = "agentic_remote_human"
# How much of an error's text goes into a feed's note.
NOTE_CHARS = 150
NOTE_STALE = "the SAMPLES of the Teleop Session Server are too old"
NOTE_NO_CONTROLLER = "the right controller is not active or its pose is not valid"
NOTE_ABSENT = "the synthetic controller is absent"
# The robot app reads a reason that begins with this as a fault of the arm.
_FAULT = "fault:"
_NOT_A_FAULT = "Teleop Session Server: "

# Isaac Capture's words in the TimeoutError of a remote session that was not
# entered: no server took the connection up.
_NO_ANSWER = ("server did not answer", "no status from the server yet")
# What follows this in that error is the server's stage, then its message.
_NOT_READY = re.compile(r"was not ready(?: after [^:]* s)?: ")
_STARTING = (STAGE_CONNECTING, STAGE_WAITING)

# The values a script row may name besides "t", and what a script starts with.
_FIRST_VALUES: dict[str, Any] = {
    "pos": (0.0, 0.0, 0.0),  # m, in the stage space
    "quat": (0.0, 0.0, 0.0, 1.0),  # [x, y, z, w]
    "squeeze": 0.0,
    "trigger": 0.0,
    "a": False,
    "b": False,
    "absent": False,  # the controller is not active
    "pose_valid": True,  # False: active, with a pose that is not valid
    "age_s": 0.0,  # the sample age
}
# What "wait_for" may name, and what the newest ROBOT_STATE then shows.
WAITS: dict[str, Callable[[RobotState], bool]] = {
    "owner_human": lambda state: state.owner == "human",
    "owner_nobody": lambda state: state.owner == "nobody",
    "rest_hold": lambda state: state.motion == "rest_hold",
}
_HAND = [0.0, 1.2, -0.3]
_IDLE: dict[str, Any] = {
    "t": 0.0,
    "pos": _HAND,
    "quat": [0, 0, 0, 1],
    "squeeze": 0.0,
    "trigger": 0.0,
    "a": False,
    "b": False,
}
# The built-in scripts. "lift": right B takes the arm over, the squeezed hand
# rises 5 cm in 4 s, and right A gives the arm back once it is at rest.
# "grasp" moves the jaws, the gripper's fingers, and not the arm: right B takes
# the arm over, the trigger is pulled all the way for 4 s, which closes the
# jaws on nothing, and released for 6 s, which opens them again, and right A
# gives the arm back once it is at rest.
SCRIPTS: dict[str, tuple[Mapping[str, Any], ...]] = {
    "idle": (_IDLE,),
    "grasp": (
        _IDLE,
        {"t": 3.0, "b": True},
        {"wait_for": "owner_human"},
        {"t": 3.5, "b": False},
        {"t": 5.0, "trigger": 1.0},
        {"t": 9.0, "trigger": 0.0},
        {"wait_for": "rest_hold"},
        {"t": 15.0, "a": True},
        {"wait_for": "owner_nobody"},
        {"t": 15.5, "a": False},
    ),
    "lift": (
        _IDLE,
        {"t": 3.0, "b": True},
        {"wait_for": "owner_human"},
        {"t": 3.5, "b": False},
        {"t": 5.0, "squeeze": 1.0},
        {"t": 9.0, "pos": [0.0, 1.25, -0.3]},
        {"t": 9.5, "squeeze": 0.0},
        {"wait_for": "rest_hold"},
        {"t": 11.0, "a": True},
        {"wait_for": "owner_nobody"},
        {"t": 11.5, "a": False},
    ),
}


class Clock(Protocol):
    """What a feed needs of a clock: seconds that never go back."""

    def monotonic(self) -> float: ...


class _MachineClock:
    def monotonic(self) -> float:
        return time.monotonic()


class AttemptsRanOut(RuntimeError):
    """teleop.attempts attempts to enter a remote session failed in a row."""


class EndpointRefused(ValueError):
    """Isaac Capture cannot use teleop.endpoint as it is written."""


def feed_note(text: object) -> str:
    """The first NOTE_CHARS characters of a text, as a feed's note.

    No note begins with "fault: ": the robot app would read it as a fault.
    """
    note = str(text)
    if note.lstrip().lower().startswith(_FAULT):
        note = _NOT_A_FAULT + note.lstrip()
    return note[:NOTE_CHARS]


def entry_stage(error: BaseException) -> str:
    """The stage of a remote session that was not entered, from its error."""
    if isinstance(error, TimeoutError):
        text = str(error)
        if any(words in text for words in _NO_ANSWER):
            return STAGE_NO_SERVER
        found = _NOT_READY.search(text)
        if found is not None:
            for stage in _STARTING:
                if text.startswith(stage, found.end()):
                    return stage
    return STAGE_NO_SERVER


def _known_age(value: Any) -> float | None:
    """A sample age as a finite number of 0 or more; None for anything else."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    age = float(value)
    return age if math.isfinite(age) and age >= 0.0 else None


def _teleop_session(config: Any) -> Any:
    from isaaccapture.teleop_session_manager import TeleopSession

    return TeleopSession(config)


class TeleopFeed:
    """The headset path: a remote session on the graph's pipeline.

    A session that cannot be entered, or that was lost, is tried again
    teleop.retry_s later, with a new TeleopSession. An attempt runs inside
    step() and takes up to teleop.connect_timeout_s.
    """

    source = HUMAN_SOURCE_TELEOP

    def __init__(
        self,
        graph: HumanGraph,
        settings: TeleopSettings,
        *,
        session_factory: Callable[[Any], Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        """session_factory(config) gives a session to enter; None: a TeleopSession."""
        self._graph = graph
        self._settings = settings
        self._make = session_factory or _teleop_session
        self._clock = clock or _MachineClock()
        self._session: Any | None = None  # the entered session
        self._failures = 0  # failed attempts in a row
        self._next_attempt = -math.inf
        # Of the last attempt, while there is no session.
        self._stage = STAGE_NO_SERVER
        self._note = ""

    def session_config(self) -> Any:
        """The TeleopSessionConfig of each attempt."""
        from isaaccapture.teleop_session_manager import (
            SessionMode,
            TeleopSessionConfig,
        )

        settings = self._settings
        return TeleopSessionConfig(
            app_name=APP_NAME,
            pipeline=self._graph.pipeline,
            mode=SessionMode.REMOTE,
            remote_endpoint=settings.endpoint,
            remote_tick_rate_hz=settings.tick_rate_hz,
            remote_max_tick_age_s=settings.max_tick_age_s,
            remote_connect_timeout_s=settings.connect_timeout_s,
        )

    def open(self) -> None:
        """Nothing yet: the first attempt runs inside the first step()."""

    def step(
        self, external_inputs: dict[str, Any], events: Any, now_mono: float
    ) -> FeedStep:
        """One pass of the session; AttemptsRanOut when the attempts ran out.

        EndpointRefused when Isaac Capture cannot use teleop.endpoint.
        Without a session: no outputs, the stage and the text of the last
        attempt. With one the graph's outputs are given also while the
        controller's pose is not valid, with the tick age: its buttons are read.
        """
        if self._session is None:
            if now_mono < self._next_attempt:
                return FeedStep(None, None, self._stage, self._note)
            self._enter()
            if self._session is None:
                return FeedStep(None, None, self._stage, self._note)
        session = self._session
        try:
            outputs = session.step(
                external_inputs=external_inputs, execution_events=events
            )
        except RuntimeError as error:
            _log.warning("the remote session was lost: %s", error)
            self._leave()
            self._failed(error, STAGE_NO_SERVER)
            return FeedStep(None, None, self._stage, self._note)
        age = _known_age(session.last_step_info.remote_tick_age_s)
        if age is None:
            return FeedStep(outputs, None, STAGE_STALE, NOTE_STALE)
        if self._graph.read(outputs).present:
            return FeedStep(outputs, age, STAGE_READY)
        return FeedStep(outputs, age, STAGE_NO_CONTROLLER, NOTE_NO_CONTROLLER)

    def close(self) -> None:
        """Leave the session, if one is entered."""
        self._leave()

    def _enter(self) -> None:
        """One attempt to enter a remote session."""
        settings = self._settings
        try:
            session = self._make(self.session_config())
            session.__enter__()
        except (OSError, RuntimeError) as error:  # a TimeoutError is an OSError
            self._failures += 1
            self._failed(error, entry_stage(error))
            if self._failures >= settings.attempts:
                # Not logged here: the program sends its STOP before the line
                # of its end, which names the last attempt.
                raise AttemptsRanOut(
                    f"no remote session at {settings.endpoint}: {self._failures} "
                    f"attempts in a row failed, {settings.retry_s:g} s apart; the "
                    f"last one: {self._note}"
                ) from error
            _log.info(
                "attempt %d of %d to enter a remote session at %s failed (%s): %s",
                self._failures,
                settings.attempts,
                settings.endpoint,
                self._stage,
                error,
            )
            return
        except ValueError as error:
            # Isaac Capture's error for an endpoint it cannot read, such as a
            # port that is no number: no later attempt can end another way.
            raise EndpointRefused(
                f"teleop.endpoint {settings.endpoint!r}: {error}"
            ) from error
        self._failures = 0
        self._session = session
        _log.info("entered a remote session at %s", settings.endpoint)

    def _failed(self, error: BaseException, stage: str) -> None:
        """Note why there is no session; the next attempt comes after the pause."""
        self._stage = stage
        self._note = feed_note(error)
        self._next_attempt = self._clock.monotonic() + self._settings.retry_s

    def _leave(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.__exit__(None, None, None)
        except Exception:
            _log.exception("leaving the remote session failed")


@dataclass(frozen=True)
class _Row:
    """One row of a script: values from a time on, or a wait."""

    t: float = 0.0
    values: Mapping[str, Any] | None = None  # None: a wait
    wait_for: str = ""


class ScriptFeed:
    """The synthetic controller: the right controller as a script's rows give it.

    A row is {"t": seconds, ...values} or {"wait_for": word}. t is on the
    script's own clock, which starts with the first step(). A row keeps every
    value of the row before it that it does not name: "pos" (m, the stage
    space), "quat" [x, y, z, w], "squeeze", "trigger", "a", "b", "absent" (the
    controller is not active), "pose_valid" and "age_s" (the sample age).
    "pos" moves in a straight line from the row before a row that names it to
    that row, over the time between the two; every other value changes at its
    row's time. A wait stops the script's clock until the newest ROBOT_STATE
    shows what it names, one of WAITS.
    """

    source = HUMAN_SOURCE_SYNTHETIC
    name = "rows"  # the script's name; named() sets it

    def __init__(
        self,
        graph: HumanGraph,
        rows: Sequence[Mapping[str, Any]],
        *,
        state: Callable[[], RobotState | None] | None = None,
    ) -> None:
        """state() gives the newest ROBOT_STATE or None; a wait needs it."""
        self._graph = graph
        self._state = state
        self._rows = _rows(rows)
        self.open()

    @classmethod
    def named(cls, graph: HumanGraph, name: str, **kw: Any) -> ScriptFeed:
        """A built-in script, one of SCRIPTS; kw are the keywords of ScriptFeed."""
        if name not in SCRIPTS:
            raise ValueError(
                f"no synthetic controller {name!r}; there are "
                f"{', '.join(sorted(SCRIPTS))}"
            )
        feed = cls(graph, SCRIPTS[name], **kw)
        feed.name = name
        return feed

    def open(self) -> None:
        """Set the script back to its start."""
        self._t = 0.0  # the script's clock
        self._last_now: float | None = None
        self._next = 0  # index of the first row not reached yet
        self._values = dict(_FIRST_VALUES)
        self._row_t: float | None = None  # time of the last timed row reached

    def step(
        self, external_inputs: dict[str, Any], events: Any, now_mono: float
    ) -> FeedStep:
        """Run the graph on the controller as the script has it now."""
        passed = 0.0 if self._last_now is None else now_mono - self._last_now
        self._last_now = now_mono
        self._advance(max(0.0, passed))
        values = self._values
        right = None if values["absent"] else self._snapshot()
        now_ns = round(now_mono * 1e9)
        outputs = self._graph.pipeline.execute_pipeline(
            {**self._graph.controller_inputs(right), **external_inputs},
            context=ComputeContext(
                graph_time=GraphTime(now_ns, now_ns), execution_events=events
            ),
        )
        if right is None:
            return FeedStep(outputs, None, STAGE_NO_CONTROLLER, NOTE_ABSENT)
        if self._graph.read(outputs).present:
            return FeedStep(outputs, values["age_s"], STAGE_READY)
        return FeedStep(
            outputs, values["age_s"], STAGE_NO_CONTROLLER, NOTE_NO_CONTROLLER
        )

    def close(self) -> None:
        """Nothing to end."""

    def _advance(self, passed: float) -> None:
        """Move the script's clock on, up to a wait that does not hold yet."""
        until = self._t + passed
        while self._next < len(self._rows):
            row = self._rows[self._next]
            if row.values is None:
                if not self._shown(row.wait_for):
                    return  # the clock stands
            elif row.t > until:
                break
            else:
                self._values.update(row.values)
                self._t = self._row_t = row.t
            self._next += 1
        self._t = until

    def _shown(self, wait_for: str) -> bool:
        state = None if self._state is None else self._state()
        return state is not None and WAITS[wait_for](state)

    def _hand(self) -> tuple[float, float, float]:
        """Where the hand is: on its way when the next timed row names "pos"."""
        at = self._values["pos"]
        timed = (row for row in self._rows[self._next :] if row.values is not None)
        ahead = next(timed, None)
        if ahead is None or "pos" not in ahead.values or self._row_t is None:
            return at
        span = ahead.t - self._row_t
        if span <= 0.0:
            return at
        share = min(1.0, max(0.0, (self._t - self._row_t) / span))
        return tuple(a + share * (b - a) for a, b in zip(at, ahead.values["pos"]))

    def _snapshot(self) -> ControllerSnapshot:
        """The right controller now; its grip pose and its aim pose are one."""
        values = self._values
        pose = ControllerPose(
            Pose(Point(*self._hand()), Quaternion(*values["quat"])),
            values["pose_valid"],
        )
        buttons = ControllerInputState(
            values["a"],
            values["b"],
            False,
            False,
            0.0,
            0.0,
            values["squeeze"],
            values["trigger"],
        )
        return ControllerSnapshot(pose, pose, buttons)


def _rows(rows: Sequence[Mapping[str, Any]]) -> tuple[_Row, ...]:
    """The rows of a script, checked; ValueError names the row that is wrong."""
    made: list[_Row] = []
    last_t = 0.0
    for index, row in enumerate(rows):
        where = f"script row {index}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{where} must be a mapping, got {row!r}")
        if "wait_for" in row:
            word = row["wait_for"]
            if len(row) != 1 or word not in WAITS:
                raise ValueError(
                    f"{where}: a wait is {{'wait_for': word}} alone, the word one "
                    f"of {', '.join(WAITS)}; got {dict(row)!r}"
                )
            made.append(_Row(wait_for=word))
            continue
        unknown = sorted(str(key) for key in row if key not in ("t", *_FIRST_VALUES))
        if unknown:
            raise ValueError(f"{where} names unknown key(s): {', '.join(unknown)}")
        t = _number(where, "t", row.get("t"))
        if not (math.isfinite(t) and t >= last_t):
            raise ValueError(
                f"{where}: t must be a time of {last_t:g} s or later, got {t!r}"
            )
        last_t = t
        values = {key: _value(where, key, row[key]) for key in row if key != "t"}
        made.append(_Row(t=t, values=values))
    return tuple(made)


def _value(where: str, key: str, value: Any) -> Any:
    """One value of a script row, checked."""
    if key in ("a", "b", "absent", "pose_valid"):
        if not isinstance(value, bool):
            raise ValueError(f"{where}: {key} must be true or false, got {value!r}")
        return value
    if key in ("pos", "quat"):
        count = 3 if key == "pos" else 4
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError(f"{where}: {key} must be {count} numbers, got {value!r}")
        numbers = tuple(_number(where, key, item) for item in value)
        norm = math.sqrt(sum(item * item for item in numbers))
        if not math.isfinite(norm) or (key == "quat" and norm < 1e-6):
            raise ValueError(f"{where}: {key} cannot be used: {value!r}")
        return numbers if key == "pos" else tuple(item / norm for item in numbers)
    number = _number(where, key, value)
    # A squeeze or a trigger may be no finite number: no controller data then.
    if key == "age_s" and not (math.isfinite(number) and number >= 0.0):
        raise ValueError(f"{where}: age_s must be 0 or more, got {value!r}")
    return number


def _number(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}")
    return float(value)
