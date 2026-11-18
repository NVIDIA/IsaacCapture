# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live arm: the newest ROBOT_STATE for the experiment's viewer page.

LiveRobot keeps the newest ROBOT_STATE that the viewer's attachment to the
robot link service received (view_link.py), with a number that only grows.
Nothing else is kept and nothing is stored. Its answer() is what
GET /api/live/robot.json gives: the state's number, its age on this machine,
the measured joints, the last joint target, the jaw opening, the owner, the
stop level, the motion and the torque. With ?after=N the answer waits until
the newest state is not number N: a page that asks again at once gets every
state as it arrives, and one that asks slowly gets the newest and misses the
ones between.

A ROBOT_STATE carries each joint's newest reading however old it is: with the
motors off, and when the joint feedback is lost, the robot app reads no joint
and goes on publishing. The arm is live only while the readings themselves
are fresh: the state's age since its arrival plus the age of its readings is
within STALE_S. Older readings are the last read pose, with their age and why
they are old; the jaw reading has its age the same way.

"robot" in the answer is the arm as the experiment's page draws it: the
points of the links and the corners of the gripper's collision volumes.
ArmModel computes them with the experiment's own routine, geometry() of its
viewer's RunIndex, built on the newest run folder that holds
resolved-config.yaml. No other geometry of the arm is computed here.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..protocol.messages import RobotState
from ..robot_app.text import printable, torque_text

_log = logging.getLogger("isaaccapture.examples.agentic_remote.viewer_live")

# How long an answer waits for a state other than number ?after=N.
ROBOT_WAIT_S = 1.0
# The arm is live while its joints' readings are no older than this: the time
# since the ROBOT_STATE arrived plus the age of the readings it carries. The
# rig's robot app reads a joint every 0.01 s and publishes 20 states a second.
STALE_S = 1.0
# The file of a run folder that names the arm's model, as the experiment's
# viewer reads it.
RESOLVED_CONFIG = "resolved-config.yaml"
# The runs folder is looked through for a newer run folder this often.
MODEL_CHECK_S = 2.0
# What the page's own text says of a pose that came this way.
LIVE_SOURCE = "ROBOT_STATE as it arrives; shown, not stored"
# And of a pose whose readings are older than STALE_S.
LAST_READ_SOURCE = "ROBOT_STATE: the joints' last reading; shown, not stored"
# Why a ROBOT_STATE carries old readings: the robot app reads no joint while
# the motors are off, and none arrives while the joint feedback is lost.
TORQUE_OFF_WHY = "torque off: the joints are not read"
FEEDBACK_LOST = "joint feedback lost"
NO_READING_AGE = "the ROBOT_STATE does not say how old its joint readings are"
# The most characters of a fault reason in an answer's "why".
_REASON_CHARS = 120
# ROBOT_STATE's motion after a torque release whose driver call failed.
_RELEASE_FAILED = "release_failed"
NOT_ATTACHED = "the viewer is not attached to the robot link service"
# What /api/live/robot.json answers in a viewer without a live arm.
NO_LIVE_ROBOT = "this viewer has no live robot state"

Geometry = Callable[[Sequence[float]], dict[str, Any]]


class ArmModelError(Exception):
    """The arm cannot be drawn; the text says why."""


class ArmModel:
    """The experiment's own geometry of the arm, for the newest run's arm.

    viewer is the experiment's viewer module and runs its Viewer: the run
    folders lie in runs.root. geometry(q) is geometry(q) of viewer.RunIndex on
    the run folder whose name sorts last among those with resolved-config.yaml,
    recorded or not; that RunIndex reads no recording. run() names that folder.
    """

    def __init__(
        self,
        viewer: Any,
        runs: Any,
        *,
        check_s: float = MODEL_CHECK_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._viewer = viewer
        self._root = getattr(runs, "root", None)
        self._check_s = float(check_s)
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._checked: float | None = None
        self._key: tuple[str, int, int] | None = None  # of the run folder in use
        self._index: Any = None
        self._problem = ""

    def geometry(self, q: Sequence[float]) -> dict[str, Any]:
        """The experiment's {"joints", "q", "boxes"} of this pose.

        ArmModelError when no run folder names the arm, when it cannot be
        built, and when the pose has another number of joints than the model.
        """
        with self._lock:  # the experiment's kinematics is not made for two threads
            self._look()
            if self._index is None:
                raise ArmModelError(self._problem)
            joints = getattr(getattr(self._index, "kin", None), "n", None)
            if isinstance(joints, int) and len(q) != joints:
                raise ArmModelError(
                    f"the ROBOT_STATE has {len(q)} joints and the arm model of run "
                    f"{self._key[0]} has {joints}"
                )
            try:
                return self._index.geometry(q)
            except Exception as problem:
                raise ArmModelError(
                    f"the experiment's viewer cannot place the arm: {problem}"
                ) from problem

    def run(self) -> str:
        """The name of the run folder whose arm model is in use; "" without one."""
        with self._lock:
            return self._key[0] if self._index is not None and self._key else ""

    def _look(self) -> None:
        """Build the RunIndex of the newest run folder when it is another one."""
        now = self._monotonic()
        if self._checked is not None and now - self._checked < self._check_s:
            return
        self._checked = now
        found = self._newest()
        if found is None:
            self._key = self._index = None
            self._problem = (
                f"no run folder in {self._root} holds {RESOLVED_CONFIG}, which "
                "names the arm's model"
            )
            return
        path, key = found
        if key == self._key:
            return
        self._key = key
        try:
            self._index = self._viewer.RunIndex(path)
            self._problem = ""
        except Exception as problem:
            self._index = None
            self._problem = (
                f"the experiment's viewer cannot build the arm of {path.name}: "
                f"{type(problem).__name__}: {problem}"
            )

    def _newest(self) -> tuple[Path, tuple[str, int, int]] | None:
        try:
            root = Path(self._root)
            names = sorted(
                (
                    entry.name
                    for entry in root.iterdir()
                    if (entry / RESOLVED_CONFIG).is_file()
                ),
                reverse=True,
            )
            if not names:
                return None
            stat = (root / names[0] / RESOLVED_CONFIG).stat()
        except (OSError, TypeError):
            return None
        return root / names[0], (names[0], stat.st_mtime_ns, stat.st_size)


class LiveRobot:
    """The newest ROBOT_STATE and its number; every method may be called from any thread.

    geometry(q) gives the arm of a pose as the experiment's page draws it, or
    raises ArmModelError; None: no arm is drawn. model_run() names the run
    whose arm model geometry() uses. on_state() takes a state at its arrival,
    on the clock monotonic() reads. on_link() takes why no ROBOT_STATE can
    arrive, "" when one can.
    """

    def __init__(
        self,
        geometry: Geometry | None = None,
        *,
        model_run: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._geometry = geometry
        self._model_run = model_run
        self._monotonic = monotonic
        self._wall = wall
        self._changed = threading.Condition()
        self._ended = False
        self._n = 0
        self._state: RobotState | None = None
        self._arrived = 0.0
        # Until when the newest state's readings are fresh; None: they are not.
        self._fresh_until: float | None = None
        self._arrivals: deque[float] = deque()  # of the last second's states
        self._link_why = NOT_ATTACHED
        self._draw_lock = threading.Lock()
        # Of the state drawn last: its number, its geometry, why there is
        # none, and the run whose arm model drew it.
        self._drawn: tuple[int, dict[str, Any] | None, str, str] = (0, None, "", "")

    def on_state(self, state: RobotState, now_mono: float) -> None:
        """A ROBOT_STATE arrived: it is the newest, with the next number."""
        reading_age = _reading_age(state)
        with self._changed:
            self._n += 1
            self._state = state
            self._arrived = now_mono
            self._fresh_until = (
                None
                if reading_age is None or reading_age > STALE_S
                else now_mono + STALE_S - reading_age
            )
            self._arrivals.append(now_mono)
            while self._arrivals and now_mono - self._arrivals[0] > 1.0:
                self._arrivals.popleft()
            self._changed.notify_all()

    def on_link(self, why: str) -> None:
        """Why no ROBOT_STATE can arrive now; "" when the robot is connected."""
        with self._changed:
            self._link_why = why

    def close(self) -> None:
        """No answer waits any longer; safe to call twice."""
        with self._changed:
            self._ended = True
            self._changed.notify_all()

    def answer(
        self, after: int | None = None, timeout_s: float = ROBOT_WAIT_S
    ) -> dict[str, Any]:
        """What /api/live/robot.json answers, as JSON values.

        With `after` it waits, at most timeout_s, until the newest state is
        not number `after`; the wait also ends when that state is live no
        longer, or too old to be drawn. "available" says whether the arm is live: a
        ROBOT_STATE with a valid reading of every joint, whose arrival age
        plus the age of its readings is within STALE_S, and its geometry.
        "why" says why not. "last_read" is there when the readings are older:
        their age and why; "robot" is then the last read pose.
        """
        deadline = time.monotonic() + timeout_s
        with self._changed:
            # When the answer about the newest state changes by itself: its
            # readings pass the limit, and later the state does.
            began = self._monotonic()
            turns = [self._fresh_until]
            if self._state is not None:
                turns.append(self._arrived + STALE_S)
            turn = min(
                (at for at in turns if at is not None and at >= began), default=None
            )
            while after is not None and self._n == after and not self._ended:
                left = deadline - time.monotonic()
                if turn is not None:
                    until_turn = turn - self._monotonic()
                    if until_turn < 0.0:
                        break
                    left = min(left, until_turn + 0.001)
                if left <= 0:
                    break
                self._changed.wait(left)
            n, state, arrived = self._n, self._state, self._arrived
            link_why = self._link_why
            now = self._monotonic()
            per_second = sum(now - at <= 1.0 for at in self._arrivals)
        told: dict[str, Any] = {
            "n": n,
            "available": False,
            "why": "",
            "age_s": None,
            "per_second": per_second,
            # This machine's wall clock now: the page dates the recording by it.
            "wall_s": self._wall(),
            # The limit of "live", and the age it is held against: the page
            # goes on counting it between two answers.
            "stale_s": STALE_S,
            "pose_age_s": None,
            "last_read": None,
            "model_run": None,
            "robot": None,
        }
        if state is None:
            told["why"] = link_why or "no ROBOT_STATE has arrived"
            return told
        age = max(0.0, now - arrived)
        told["age_s"] = round(age, 3)
        told.update(_state_json(state, age))
        if age > STALE_S:
            told["why"] = link_why or f"no ROBOT_STATE for {age:.1f} s"
            return told
        pose = _pose_of(state)
        if pose is None:
            told["why"] = (
                "a joint has no valid reading"
                if state.joints
                else "the ROBOT_STATE holds no joint reading"
            )
            return told
        reading_age = _reading_age(state)
        if reading_age is None:
            told["why"] = NO_READING_AGE
            return told
        pose_age = age + reading_age
        told["pose_age_s"] = round(pose_age, 3)
        drawn, problem, run = self._draw(n, pose[0])
        told["model_run"] = run or None
        if drawn is None:
            told["why"] = problem
            return told
        live = pose_age <= STALE_S
        told["robot"] = {
            **drawn,
            # What the page's own text reads of a pose.
            "source": LIVE_SOURCE if live else LAST_READ_SOURCE,
            "sample_span_s": pose[1],
            "age_s": pose_age,
            # When the joints were read, on this machine's wall clock.
            "utc": datetime.fromtimestamp(
                told["wall_s"] - pose_age, timezone.utc
            ).isoformat(timespec="milliseconds"),
        }
        if live:
            told["available"] = True
            return told
        reason = _not_read(state)
        if not reason and reading_age <= STALE_S:
            # Fresh when the state arrived: old only because no newer one came.
            reason = link_why or f"no newer ROBOT_STATE for {age:.1f} s"
        elif not reason:
            reason = _fault(state) or (
                f"the robot app sends no newer reading; motion {state.motion}"
            )
        told["last_read"] = {"age_s": told["pose_age_s"], "why": reason}
        told["why"] = f"the joints were last read {pose_age:.1f} s ago ({reason})"
        return told

    def _draw(self, n: int, q: list[float]) -> tuple[dict[str, Any] | None, str, str]:
        """(the geometry of state n, "", the model's run) or (None, why, the run).

        Each state is placed once.
        """
        with self._draw_lock:
            if self._drawn[0] == n:
                return self._drawn[1:]
            drawn, problem = None, "this viewer draws no live arm"
            if self._geometry is not None:
                try:
                    drawn = self._geometry(q)
                    json.dumps(drawn, allow_nan=False)  # what the page can read
                    problem = ""
                except ArmModelError as cannot:
                    drawn, problem = None, str(cannot)
                except Exception as cannot:
                    _log.exception("the live arm's geometry failed")
                    drawn = None
                    problem = (
                        f"the arm cannot be placed: {type(cannot).__name__}: {cannot}"
                    )
            run = ""
            if self._model_run is not None:
                try:
                    run = str(self._model_run())
                except Exception:
                    _log.exception("the live arm's model did not name its run")
            self._drawn = (n, drawn, problem, run)
            return drawn, problem, run


def no_live_robot() -> dict[str, Any]:
    """The answer of a viewer that has no live arm."""
    return {
        "n": 0,
        "available": False,
        "why": NO_LIVE_ROBOT,
        "age_s": None,
        "per_second": 0,
        "wall_s": time.time(),
        "stale_s": STALE_S,
        "pose_age_s": None,
        "last_read": None,
        "model_run": None,
        "robot": None,
    }


def _pose_of(state: RobotState) -> tuple[list[float], float] | None:
    """(the measured joints, their sample span) of a ROBOT_STATE; None if not whole.

    As the agent program takes the pose it records: the joints in their order,
    and the newest end of a read less the oldest among them.
    """
    joints = sorted(state.joints, key=lambda read: read.joint)
    if not joints or any(
        read.valid is not True or not _is_number(read.value) for read in joints
    ):
        return None
    ends = [read.t_end for read in joints]
    span = max(ends) - min(ends) if all(_is_number(end) for end in ends) else 0.0
    return [float(read.value) for read in joints], span


def _reading_age(state: RobotState) -> float | None:
    """How old the oldest joint reading was when the robot app built the state.

    health["reading_age_s"]; where the readings carry the end of their reads,
    the larger of it and the state's clock_mono_s less the oldest end. None:
    the state does not say, and its arm is not live.
    """
    told = _number_or_none(_health(state).get("reading_age_s"))
    if told is None or told < 0.0:
        return None
    ends = [read.t_end for read in state.joints]
    if ends and _is_number(state.clock_mono_s) and all(_is_number(e) for e in ends):
        told = max(told, state.clock_mono_s - min(ends))
    return float(told)


def _not_read(state: RobotState) -> str:
    """Why the robot app reads no joint, where its ROBOT_STATE says so; else ""."""
    if state.torque_on is not True:
        return TORQUE_OFF_WHY
    reason = state.fault_reason if isinstance(state.fault_reason, str) else ""
    return FEEDBACK_LOST if FEEDBACK_LOST in reason else ""


def _fault(state: RobotState) -> str:
    """The fault a ROBOT_STATE names, as a reason; "" without one."""
    reason = state.fault_reason if isinstance(state.fault_reason, str) else ""
    if state.stop_level != "fault" or not reason:
        return ""
    return f"fault: {printable(reason, _REASON_CHARS)}"


def _health(state: RobotState) -> dict[str, Any]:
    return state.health if isinstance(state.health, dict) else {}


def _state_json(state: RobotState, age_s: float) -> dict[str, Any]:
    """What a ROBOT_STATE that arrived age_s ago says of the arm, for the page."""
    motors = _health(state).get("motors")
    hold = motors.get("state") if isinstance(motors, dict) else None
    torque = torque_text(state.torque_on, hold, state.motion == _RELEASE_FAILED)
    targets = state.target_q if isinstance(state.target_q, list) else []
    joints = []
    for read in sorted(state.joints, key=lambda read: read.joint):
        valid = read.valid is True and _is_number(read.value)
        q = read.value if valid else None
        target = None
        if 0 <= read.joint < len(targets):
            target = _number_or_none(targets[read.joint])
        joints.append(
            {
                "joint": read.joint,
                "valid": valid,
                "q": q,  # rad
                "deg": None if q is None else math.degrees(q),
                "target": target,  # rad: the last target sent
            }
        )
    jaws = state.gripper
    jaw_valid = jaws.valid is True and _is_number(jaws.opening)
    # The jaw reading's time is on the robot app's clock, as clock_mono_s is.
    jaw_age = None
    if jaw_valid and _is_number(jaws.t) and _is_number(state.clock_mono_s):
        jaw_age = age_s + max(0.0, state.clock_mono_s - jaws.t)
    return {
        "seq": state.seq,
        "joints": joints,
        "target_q": [_number_or_none(value) for value in targets] or None,
        # Shares of the jaw travel, 1.0 fully open.
        "gripper": {
            "valid": jaw_valid,
            "opening": jaws.opening if jaw_valid else None,
            "target": _number_or_none(jaws.last_target),
            "load": jaws.load,
            # How old the jaw reading is now; None: the state does not say.
            # No jaw reading is taken during a move or with the motors off.
            "age_s": None if jaw_age is None else round(jaw_age, 3),
            "fresh": jaw_age is not None and jaw_age <= STALE_S,
        },
        "owner": state.owner,
        "stop_level": state.stop_level,
        "motion": state.motion,
        "torque_on": state.torque_on is True,
        "torque": torque.removeprefix("torque "),
        "reading_age_s": _number_or_none(_health(state).get("reading_age_s")),
    }


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _number_or_none(value: Any) -> float | None:
    return value if _is_number(value) else None
