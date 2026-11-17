# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured joint readings: their history, the idle reads and the ROBOT_STATE."""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import logging
import math
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..protocol.carrier import send
from ..protocol.messages import JointSample, ProtocolError, RobotState
from .interfaces import Owner, ReadFailed
from .text import MAX_REASON_CHARS, LineLimit, printable

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from ..protocol.messages import GripperState
    from .clock import Clock
    from .config import RobotAppConfig, StateConfig
    from .interfaces import (
        ArmPort,
        AuthorityState,
        JointReading,
        MotionState,
    )

_log = logging.getLogger("isaaccapture.examples.agentic_remote.state")

# Idle reads wait this long after a target, so a move's own reads have the bus.
IDLE_AFTER_TARGET_S = 0.1
# Slack when a period is compared on a floating-point clock.
_SLACK_S = 1e-6
# A ROBOT_STATE that cannot be sent is logged once per this many seconds.
NOT_SENT_LOG_S = 5.0
_NUMBER = re.compile(r"\d+")


def _t_end(reading: JointReading) -> float:
    return reading.t_end


def _usable(reading: JointReading) -> bool:
    """True for a reading that is marked valid and has a finite value."""
    return bool(reading.valid) and math.isfinite(reading.value)


class JointHistory:
    """The joint readings of the last keep_s seconds, ordered by the end of their read.

    Each joint's newest valid reading is kept however old it gets, so latest() and
    age() show how stale a joint is instead of forgetting it. Thread-safe.
    """

    def __init__(self, n_joints: int, keep_s: float) -> None:
        if n_joints < 1:
            raise ValueError(f"n_joints must be at least 1, got {n_joints!r}")
        if not (math.isfinite(keep_s) and keep_s > 0.0):
            raise ValueError(f"keep_s must be finite and above 0, got {keep_s!r}")
        self._keep_s = float(keep_s)
        self._lock = threading.Lock()
        self._readings: list[JointReading] = []  # sorted by t_end
        self._newest: list[JointReading | None] = [None] * n_joints

    @property
    def n_joints(self) -> int:
        return len(self._newest)

    def add(self, reading: JointReading) -> None:
        """Store a reading; one with a non-finite value is stored as not valid.

        Raises ValueError for a joint index out of range or a non-finite time.
        """
        if not 0 <= reading.joint < len(self._newest):
            raise ValueError(
                f"joint index {reading.joint!r} is outside 0..{len(self._newest) - 1}"
            )
        if not (math.isfinite(reading.t_start) and math.isfinite(reading.t_end)):
            raise ValueError(f"a reading needs finite times, got {reading!r}")
        if reading.valid and not math.isfinite(reading.value):
            reading = dataclasses.replace(reading, valid=False)
        with self._lock:
            readings = self._readings
            if not readings or readings[-1].t_end <= reading.t_end:
                readings.append(reading)
            else:
                bisect.insort_right(readings, reading, key=_t_end)
            cutoff = readings[-1].t_end - self._keep_s
            del readings[: bisect.bisect_left(readings, cutoff, key=_t_end)]
            newest = self._newest[reading.joint]
            if reading.valid and (newest is None or newest.t_end <= reading.t_end):
                self._newest[reading.joint] = reading

    def latest(self) -> list[JointReading | None]:
        """Per joint its newest valid reading, None for a joint that has none."""
        with self._lock:
            return list(self._newest)

    def since(self, t: float) -> list[JointReading]:
        """Kept readings, valid or not, that ended at t or later, oldest first."""
        with self._lock:
            first = bisect.bisect_left(self._readings, t, key=_t_end)
            return self._readings[first:]

    def age(self, now: float) -> float | None:
        """Age of the oldest among the joints' newest valid readings, from its read end.

        None while a joint has no valid reading.
        """
        with self._lock:
            newest = list(self._newest)
        if any(reading is None for reading in newest):
            return None
        return max(0.0, now - min(reading.t_end for reading in newest))


class IdleReader:
    """Reads one joint per period, in turn, while no move is sending targets.

    It keeps the history fresh in a hold and counts the reads that fail; a failed
    read never becomes a value, and its joint is read once more at once. With
    the motors off it reads nothing and counts no lost feedback: nothing holds
    then, and a released driver cannot be read.
    """

    def __init__(
        self, arm: ArmPort, history: JointHistory, config: StateConfig, clock: Clock
    ) -> None:
        if arm.n_joints != history.n_joints:
            raise ValueError(
                f"the arm has {arm.n_joints} joints but the history {history.n_joints}"
            )
        self._arm = arm
        self._history = history
        self._period_s = config.idle_read_period_s
        self._clock = clock
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()  # held while a joint is read
        self._held = 0  # paused() blocks that are open
        self._next_joint = 0
        self._retry: int | None = None  # the joint whose read has just failed
        self._last_read_t: float | None = None
        self._paused = True
        self._reading_since = 0.0  # when reads began, or resumed after a pause
        self._failures = [0] * history.n_joints
        # Per joint while its reads fail: (the time its loss is counted from at the
        # earliest, the end of its newest failed read).
        self._failing: list[tuple[float, float] | None] = [None] * history.n_joints

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Begin no idle read inside the block; one under way ends first.

        A move runs inside it, so no idle read takes the bus, or clock time,
        from its setpoints. The time inside is a pause, not lost feedback.
        """
        with self._read_lock, self._lock:
            self._held += 1
            self._paused = True
        try:
            yield
        finally:
            with self._lock:
                self._held -= 1

    def due_in(self, now: float) -> float | None:
        """Seconds until the next idle read is due, 0.0 when one is due now.

        None while none will be: the motors are off or the reads are paused.
        """
        if not self._arm.torque_on():
            return None
        target = self._arm.last_target()
        with self._lock:
            if self._held:
                return None
            return max(0.0, self._due_at(target, now) - now)

    def _due_at(self, target: tuple[list[float], float] | None, now: float) -> float:
        """When the next read is due, with the lock held; `now` if it is due."""
        due = now
        if target is not None:
            due = max(due, target[1] + IDLE_AFTER_TARGET_S)
        if self._retry is None and self._last_read_t is not None:
            due = max(due, self._last_read_t + self._period_s)
        return due

    def step(self, now: float) -> JointReading | None:
        """Read the next joint if a read is due; the reading if it is good, else None.

        A read is due when the motors are on, the reads are not paused, no
        target was sent in the last IDLE_AFTER_TARGET_S and the last idle read
        began at least idle_read_period_s ago, or failed: its joint is then
        read again at once, a single time.
        """
        with self._read_lock:
            return self._step(now)

    def _step(self, now: float) -> JointReading | None:
        if not self._arm.torque_on():
            with self._lock:
                self._paused = True
                self._retry = None
                self._failing = [None] * len(self._failing)
            return None
        target = self._arm.last_target()
        with self._lock:
            sending = target is not None and (
                now - target[1] < IDLE_AFTER_TARGET_S - _SLACK_S
            )
            if self._held or sending:
                self._paused = True
                return None
            if now < self._due_at(target, now) - _SLACK_S:
                return None
            if self._paused:
                self._paused = False
                self._reading_since = now
            again = self._retry is not None
            if again:
                joint, self._retry = self._retry, None
            else:
                joint = self._next_joint
                self._next_joint = (joint + 1) % len(self._failures)
            self._last_read_t = now
        try:
            reading = self._arm.read_joint(joint)
        except ReadFailed as error:
            _log.debug("idle read of joint %d failed: %s", joint, error)
            reading = None
        else:
            self._history.add(reading)
        good = reading is not None and _usable(reading)
        if not good and not self._arm.torque_on():
            # The motors went off while the joint was read: nothing holds
            # now, and that is no lost feedback.
            with self._lock:
                self._paused = True
                self._retry = None
                self._failing = [None] * len(self._failing)
            return None
        with self._lock:
            if good:
                self._failing[joint] = None
            else:
                self._failures[joint] += 1
                failing = self._failing[joint]
                since = self._reading_since if failing is None else failing[0]
                self._failing[joint] = (since, max(now, self._clock.monotonic()))
                if not again:
                    self._retry = joint
        return reading if good else None

    def failures(self) -> list[int]:
        """Per joint, how many idle reads have failed so far."""
        with self._lock:
            return list(self._failures)

    def consecutive_failures_s(self, now: float) -> float:
        """Seconds the worst joint whose idle reads fail has had no good reading.

        0.0 while no joint is failing. A good reading that another reader stored
        after the failure ends the count. For a joint that began to fail after a
        pause, the time before the idle reads resumed is not counted.
        """
        latest = self._history.latest()
        lost = 0.0
        with self._lock:
            for joint, failing in enumerate(self._failing):
                if failing is None:
                    continue
                since, failed_at = failing
                good = latest[joint]
                if good is not None:
                    if good.t_end > failed_at:
                        continue
                    since = max(since, good.t_end)
                lost = max(lost, now - since)
        return lost


def _text(value: object) -> str:
    """An enum member's value, or the string itself."""
    return str(value.value) if isinstance(value, Enum) else str(value)


def _sample(reading: JointReading) -> JointSample:
    """A reading from the history as a link message part.

    The history holds a non-finite value only as not valid; it is sent as 0.0.
    """
    value = float(reading.value)
    return JointSample(
        joint=int(reading.joint),
        value=value if math.isfinite(value) else 0.0,
        t_start=float(reading.t_start),
        t_end=float(reading.t_end),
        valid=bool(reading.valid),
    )


def _clean(value: Any, name: str, not_finite: list[str]) -> Any:
    """`value` as JSON data; a non-finite number becomes None and `name` is noted."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # a numpy array or scalar
        value = tolist()
    if isinstance(value, float) and not math.isfinite(value):
        if name not in not_finite:
            not_finite.append(name)
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item, name, not_finite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item, name, not_finite) for item in value]
    return value


class StatePublisher:
    """Builds ROBOT_STATE from measurements and sends it newest-wins.

    It never reads a joint itself: the joint values come from the history. A
    non-finite number from a source is sent as unknown and named in
    health["not_finite"], so one bad value cannot end the state stream.
    """

    def __init__(
        self,
        arm: ArmPort,
        history: JointHistory,
        authority_state: Callable[[], AuthorityState],
        move_progress: Callable[[], tuple[str, int, int]],
        motion_state: Callable[[], MotionState],
        gripper_state: Callable[[], GripperState],
        assistance_wait_s: Callable[[], float | None],
        last_result: Callable[[], dict[str, Any] | None],
        session_id: str,
        config: RobotAppConfig,
        clock: Clock,
        read_failures: Callable[[], Sequence[int]] | None = None,
        move_stage: Callable[[], str] | None = None,
        view_age: Callable[[float], float | None] | None = None,
        permit_facts: Callable[[float], dict[str, Any] | None] | None = None,
        permit_refusal: Callable[[str], str | None] | None = None,
        human_stage: Callable[[float], str] | None = None,
        controller_age: Callable[[float], float | None] | None = None,
        human_facts: Callable[[], dict[str, Any]] | None = None,
        contact_hold: Callable[[], bool] | None = None,
        local_program: Callable[[], bool | None] | None = None,
    ) -> None:
        """view_age(now) is the age of the operator's live view, None if unknown.

        permit_facts(now) is the held permit (PermitGate.held_facts) and
        permit_refusal(move_id) why that move's newest permit was refused
        (PermitGate.last_refusal). human_stage(now) is the stage of the human
        program, the workstation program that turns controller data into joint
        targets (HumanInput.stage), controller_age(now) the age of its newest
        controller data (HumanInput.sample_age_s), human_facts() the content
        of health["human"], and contact_hold() whether the arm holds at a
        contact. local_program() is whether a human program is attached at the
        robot app's own socket, None when the app has no such socket: it is
        sent as health["human"]["local_program"], and not at all for None.
        Each of the eight may be left out.
        """
        self._local_program = local_program
        self._move_stage = move_stage
        self._view_age = view_age
        self._permit_facts = permit_facts
        self._permit_refusal = permit_refusal
        self._human_stage = human_stage
        self._controller_age = controller_age
        self._human_facts = human_facts
        self._contact_hold = contact_hold
        self._arm = arm
        self._history = history
        self._authority_state = authority_state
        self._move_progress = move_progress
        self._motion_state = motion_state
        self._gripper_state = gripper_state
        self._assistance_wait_s = assistance_wait_s
        self._last_result = last_result
        self._session_id = session_id
        self._clock = clock
        self._read_failures = read_failures
        self._period_s = config.state.publish_period_s
        self._recent_window_s = config.state.recent_window_s
        self._versions = {
            Owner.AGENT: config.limits.agent.version,
            Owner.HUMAN: config.limits.human.version,
            Owner.RECOVERY: config.limits.recovery.version,
        }
        self._lock = threading.Lock()
        self._seq = 0
        self._missed_deadlines = 0
        self._clamped = 0
        self._last_publish_t: float | None = None
        self._warned: set[str] = set()
        self._not_sent = LineLimit(clock, NOT_SENT_LOG_S)

    def note_missed_deadline(self) -> None:
        """Count a setpoint that was sent late."""
        with self._lock:
            self._missed_deadlines += 1

    def note_clamped(self) -> None:
        """Count a target that was clamped before it was checked."""
        with self._lock:
            self._clamped += 1

    def build(self, now: float) -> RobotState:
        """The ROBOT_STATE for the app-clock time `now`; each call takes the next seq.

        Raises ValueError when `now` is not finite.
        """
        if not math.isfinite(now):
            raise ValueError(f"now must be finite, got {now!r}")
        not_finite: list[str] = []
        # A clock pair read together; the wall time is moved to the instant `now`.
        mono, wall = self._clock.monotonic(), self._clock.wall()
        # Read first: what a command changed is in place before its outcome
        # is, so every value read after this is at least as new as the outcome.
        newest_result = self._last_result()
        authority = self._authority_state()
        move_id, move_index, move_total = self._move_progress() or ("", 0, 0)
        # Read after the progress, and sent only with a move id: a stage is
        # never shown for a move that had not reached it.
        stage = "" if self._move_stage is None else self._move_stage()
        move_stage = str(stage) if move_id else ""
        joints = [
            JointSample(joint, 0.0, 0.0, 0.0, False)
            if reading is None
            else _sample(reading)
            for joint, reading in enumerate(self._history.latest())
        ]
        recent = self._history.since(now - self._recent_window_s)
        target_q, target_t = self._target(not_finite)
        gripper = self._gripper(not_finite)
        wait_s = self._assistance_wait_s()
        if wait_s is not None and not math.isfinite(wait_s):
            not_finite.append("assistance_wait_s")
            wait_s = None
        view_age_s = None if self._view_age is None else self._view_age(now)
        if view_age_s is not None and not math.isfinite(view_age_s):
            not_finite.append("view_age_s")
            view_age_s = None
        teleop_stage = "" if self._human_stage is None else self._human_stage(now)
        controller_age_s = None
        if self._controller_age is not None:
            controller_age_s = self._controller_age(now)
        if controller_age_s is not None and not math.isfinite(controller_age_s):
            not_finite.append("controller_age_s")
            controller_age_s = None
        human = {} if self._human_facts is None else self._human_facts()
        human = _clean(human, "human", not_finite)
        local = None if self._local_program is None else self._local_program()
        if local is not None:
            human = {**human, "local_program": bool(local)}
        contact_hold = False if self._contact_hold is None else self._contact_hold()
        permit = None if self._permit_facts is None else self._permit_facts(now)
        permit = _clean(permit, "permit", not_finite)
        refusal = None
        if self._permit_refusal is not None:
            refusal = self._permit_refusal(move_id)
        last_result = _clean(newest_result, "last_result", not_finite)
        temperatures = _clean(self._arm.temperatures(), "temperatures", not_finite)
        failures = None if self._read_failures is None else self._read_failures()
        health: dict[str, Any] = {}
        # What the arm port's joint readings say of the hold; not sent by an
        # arm port that cannot say it. torque_on is no reading of the motors.
        evidence = getattr(self._arm, "hold_evidence", None)
        if callable(evidence):
            hold, why = evidence()
            reason = printable(why, MAX_REASON_CHARS)
            health["motors"] = {"state": str(hold), "reason": reason}
        with self._lock:
            self._seq += 1
            seq = self._seq
            missed_deadlines, clamped = self._missed_deadlines, self._clamped
            new_names = [name for name in not_finite if name not in self._warned]
            self._warned.update(new_names)
        for name in new_names:
            _log.warning("ROBOT_STATE: %s is not finite and is sent as unknown", name)
        return RobotState(
            seq=seq,
            session_id=self._session_id,
            clock_mono_s=float(now),
            clock_wall_s=wall + (now - mono),
            joints=joints,
            recent=[_sample(reading) for reading in recent],
            target_q=target_q,
            target_t=target_t,
            gripper=gripper,
            owner=_text(authority.owner),
            owner_number=authority.owner_number,
            stop_number=authority.stop_number,
            stop_level=_text(authority.stop_level),
            fault_reason=authority.fault_reason,
            limit_set_version=self._versions.get(authority.owner, ""),
            motion=_text(self._motion_state()),
            move_id=move_id,
            move_index=int(move_index),
            move_total=int(move_total),
            last_result=last_result,
            assistance_wait_s=wait_s,
            torque_on=bool(self._arm.torque_on()),
            view_age_s=view_age_s,
            teleop_stage=str(teleop_stage),
            controller_age_s=controller_age_s,
            health={
                "reading_age_s": self._history.age(now),
                "read_failures": None if failures is None else list(failures),
                "faults": [str(fault) for fault in self._arm.faults()],
                "temperatures": temperatures,
                "missed_deadlines": missed_deadlines,
                "clamped": clamped,
                "not_finite": not_finite,
                # The held permit and why the move's newest permit was refused.
                "permit": permit,
                "permit_refusal": refusal or "",
                # A person driving the arm: the takeover, the human drive and
                # the newest human target; empty without a source.
                "human": human,
                "contact_hold": bool(contact_hold),
                **health,
            },
            move_stage=move_stage,
        )

    def maybe_publish(self, carrier: Carrier, now: float) -> bool:
        """Send a ROBOT_STATE if publish_period_s has passed; True when it was sent.

        Nothing is built or sent on a dead link.
        """
        if not carrier.alive:
            return False
        with self._lock:
            last = self._last_publish_t
            if last is not None and now - last < self._period_s - _SLACK_S:
                return False
            self._last_publish_t = now
        for problem, count in self._not_sent.ended():
            _log.warning(
                "ROBOT_STATE was not sent %d more times: %s",
                count,
                problem or "other problems",
            )
        try:
            send(carrier, self.build(now))
        except ProtocolError as error:
            # A problem is logged when it first shows, then counted. Numbers in
            # it, such as a byte count that differs each time, do not make it new.
            if self._not_sent.allow(_NUMBER.sub("#", str(error))):
                _log.warning("ROBOT_STATE was not sent: %s", error)
            return False
        return True

    def _target(self, not_finite: list[str]) -> tuple[list[float] | None, float | None]:
        """The last target sent and its send time, or (None, None)."""
        target = self._arm.last_target()
        if target is None:
            return None, None
        q, t = [float(value) for value in target[0]], float(target[1])
        if not all(math.isfinite(value) for value in (*q, t)):
            not_finite.append("target")
            return None, None
        return q, t

    def _gripper(self, not_finite: list[str]) -> GripperState:
        """The jaw state with every non-finite number replaced by None."""
        state = self._gripper_state()
        bad = {
            name: None
            for name in ("opening", "t", "last_target")
            if (value := getattr(state, name)) is not None and not math.isfinite(value)
        }
        if not bad:
            return state
        not_finite.append("gripper")
        valid = state.valid and "opening" not in bad
        return dataclasses.replace(state, **bad, valid=valid)
