# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulated arm and a basic checker, for tests and the self-test demo.

SimArm stands in for the motors and BasicChecker for APAI's checker; neither
imports APAI.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from .config import GraspConfig
from .interfaces import (
    ArmFault,
    ContactStop,
    FeedbackLost,
    GripperOutcome,
    JointReading,
    ReadFailed,
    Refusal,
)

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .clock import Clock
    from .config import LimitSet
    from .interfaces import ArmPort
    from .state import JointHistory

    # Answers tip_blocked for a test: (q, region) -> reason or None.
    TipHook = Callable[[list[float], GraspRegion | None], str | None]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.sim")

# The two jaw stiffness states.
CLOSING = "closing"
LIGHT = "light"

JAW_MODES = ("plain", "close_until_contact")

_EPS = 1e-9


def _is_fraction(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return 0.0 <= value <= 1.0  # false for NaN


@dataclass
class JawCommand:
    """One gripper_command call as the simulated arm saw it."""

    opening: float
    max_effort: float
    mode: str
    t: float  # app-clock time the command began
    # "closing" or "light"; follows the jaws while this is the newest command.
    stiffness: str
    status: str = ""  # the outcome's status; empty while the command runs


class SimArm:
    """A software arm with jaws: a physical stand-in with knobs for tests.

    The measured pose approaches the last target with a first-order lag as app-clock
    time passes. Torque is off before connect() and after release_torque(): targets
    are then ignored, though still recorded in `sent`. Every method may be called
    from any thread.
    """

    def __init__(
        self,
        n_joints: int = 6,
        clock: Clock | None = None,
        q0: Sequence[float] | None = None,
        *,
        joint_names: Sequence[str] | None = None,
        read_time_s: float = 0.008,
        time_constant_s: float = 0.05,
        jaw_opening: float = 1.0,
        jaw_speed: float = 1.0,
        jaw_step_s: float = 0.01,
        jaw_timeout_s: float = 3.0,
    ) -> None:
        if clock is None:
            raise TypeError("SimArm needs a clock")
        if n_joints < 1:
            raise ValueError(f"n_joints must be at least 1, got {n_joints!r}")
        if jaw_step_s <= 0.0:
            raise ValueError(f"jaw_step_s must be above 0, got {jaw_step_s!r}")
        self.n_joints = int(n_joints)
        if joint_names is None:
            joint_names = [f"joint{i}" for i in range(1, self.n_joints + 1)]
        self.joint_names = list(joint_names)
        if len(self.joint_names) != self.n_joints:
            raise ValueError(f"joint_names needs {self.n_joints} names")
        self.read_time_s = float(read_time_s)
        self.time_constant_s = float(time_constant_s)
        self.jaw_speed = float(jaw_speed)  # opening travelled per second
        self.jaw_step_s = float(jaw_step_s)
        self.jaw_timeout_s = float(jaw_timeout_s)

        # (q, send time) of every send_target call, also those ignored with torque off.
        self.sent: list[tuple[list[float], float]] = []
        # (measured pose, time) of every hold_measured call that took effect.
        self.holds: list[tuple[list[float], float]] = []
        # The measured pose at each release_torque call.
        self.released_at: list[list[float]] = []
        self.jaw_commands: list[JawCommand] = []
        self.light_holds: list[float] = []  # time of every gripper_light_hold call
        # Called with the opening after each jaw step, on the commanding thread.
        self.on_jaw_step: Callable[[float], None] | None = None

        self._clock = clock
        self._lock = threading.Lock()
        zeros = [0.0] * self.n_joints
        self._q = self._vector("q0", zeros if q0 is None else q0)
        self._t = clock.monotonic()  # the time _q belongs to
        self._target: np.ndarray | None = None
        self._target_t = 0.0
        self._connected = False
        self._torque = False
        self._warned_ignored = False
        self._failing: dict[int, int] = {}  # joint -> reads still to fail
        self._blocks: dict[int, tuple[float, bool]] = {}  # joint -> (at, is upper stop)
        self._faults: list[str] = []
        self._temperatures: list[float] | None = None

        self._jaw = float(jaw_opening)
        self._jaw_target: float | None = None
        self._jaw_referenced = False
        self._jaw_object: float | None = None
        self._jaw_failing = 0
        self._jaw_stiffness = LIGHT

    # ArmPort: motors

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                raise RuntimeError("connect() was already called; it runs once")
            self._advance()
            self._connected = True
            self._torque = True

    def torque_on(self) -> bool:
        with self._lock:
            return self._torque

    def send_target(self, q: Sequence[float]) -> None:
        target = self._vector("q", q)
        with self._lock:
            self._advance()
            self.sent.append((target.tolist(), self._t))
            if not self._torque:
                if not self._warned_ignored:
                    self._warned_ignored = True
                    _log.warning("targets are ignored while torque is off")
                return
            self._target = target
            self._target_t = self._t

    def last_target(self) -> tuple[list[float], float] | None:
        with self._lock:
            if self._target is None:
                return None
            return self._target.tolist(), self._target_t

    def read_joint(self, joint: int) -> JointReading:
        self._check_joint(joint)
        t_start = self._clock.monotonic()
        self._clock.sleep(self.read_time_s)
        with self._lock:
            self._advance()
            if self._failing.get(joint, 0) > 0:
                self._failing[joint] -= 1
                raise ReadFailed(f"joint index {joint}: the read failed")
            return JointReading(joint, float(self._q[joint]), t_start, self._t)

    def read_all(self) -> list[JointReading]:
        return [self.read_joint(joint) for joint in range(self.n_joints)]

    def hold_measured(self) -> None:
        with self._lock:
            self._advance()
            if not self._torque:
                return
            self._target = self._q.copy()
            self._target_t = self._t
            self.holds.append((self._q.tolist(), self._t))

    def release_torque(self) -> None:
        with self._lock:
            self._advance()
            self.released_at.append(self._q.tolist())
            self._torque = False
            self._target = None

    def release_confirmed(self) -> bool:
        with self._lock:
            return bool(self.released_at) and not self._torque

    def faults(self) -> list[str]:
        with self._lock:
            return list(self._faults)

    def temperatures(self) -> list[float] | None:
        with self._lock:
            return None if self._temperatures is None else list(self._temperatures)

    # ArmPort: jaws

    def gripper_referenced(self) -> bool:
        with self._lock:
            return self._jaw_referenced

    def gripper_reference(self, cancel: threading.Event | None = None) -> None:
        with self._lock:
            if not self._torque:
                raise ArmFault("the jaws cannot be referenced with torque off")
            if self._faults:
                raise ArmFault(self._faults[0])
            if cancel is not None and cancel.is_set():
                return  # ended early: the jaws stay as they were
            self._jaw_referenced = True
            if self._jaw_target is None:
                # A driver's reference leaves a jaw target; here, where the jaws are.
                self._jaw_target = self._jaw

    def gripper_read(self) -> float | None:
        with self._lock:
            if not self._jaw_referenced:
                return None
            if self._jaw_failing > 0:
                self._jaw_failing -= 1
                return None
            return self._jaw

    def gripper_last_target(self) -> float | None:
        with self._lock:
            return self._jaw_target

    def gripper_command(
        self, opening: float, max_effort: float, mode: str, cancel: threading.Event
    ) -> GripperOutcome:
        with self._lock:
            record = JawCommand(
                opening, max_effort, mode, self._clock.monotonic(), self._jaw_stiffness
            )
            self.jaw_commands.append(record)
            refusal = self._jaw_refusal(opening, max_effort, mode)
            if refusal is not None:
                record.status = "refused"
                measured = self._jaw if self._jaw_referenced else None
                return GripperOutcome("refused", measured, refusal)
            if cancel.is_set():
                # Cancelled before it began: the target and the stiffness stay.
                record.status = "cancelled"
                return GripperOutcome("cancelled", self._jaw)
            target = float(opening)
            self._jaw_target = target
            # Every accepted command moves the jaws at the closing stiffness.
            self._set_jaw_stiffness(CLOSING)
        # The command's own waiting counts toward the timeout, not the shared clock,
        # so another thread that advances a fake clock cannot time it out.
        steps_left = max(1, math.ceil(self.jaw_timeout_s / self.jaw_step_s - _EPS))
        while True:
            with self._lock:
                status = self._jaw_status(target, cancel, steps_left)
                if status is not None:
                    short_close = self._jaw > target + _EPS
                    if status in ("stalled", "timeout") and short_close:
                        self._set_jaw_stiffness(LIGHT)
                    elif status == "cancelled" and self._jaw < target - _EPS:
                        # An opening ends where the jaws are; a close's target stays.
                        self._jaw_target = self._jaw
                    record.status = status
                    return GripperOutcome(status, self._jaw)
            self._clock.sleep(self.jaw_step_s)
            steps_left -= 1
            with self._lock:
                self._jaw = self._jaw_after_step(target)
                moved_to = self._jaw
            hook = self.on_jaw_step
            if hook is not None:
                hook(moved_to)

    def gripper_light_hold(self) -> None:
        with self._lock:
            self.light_holds.append(self._clock.monotonic())
            self._set_jaw_stiffness(LIGHT)

    # Knobs and readouts for tests

    def pose(self) -> list[float]:
        """The measured pose now, without the cost of a read."""
        with self._lock:
            self._advance()
            return self._q.tolist()

    def set_pose(self, q: Sequence[float]) -> None:
        """Put the arm at q with no target standing."""
        pose = self._vector("q", q)
        with self._lock:
            self._advance()
            self._q = pose
            self._target = None

    def fail_reads(self, joint: int, count: int) -> None:
        """Make the next count reads of this joint raise ReadFailed."""
        self._check_joint(joint)
        with self._lock:
            self._failing[joint] = int(count)

    def block_joint(self, joint: int, at: float | None) -> None:
        """Stop this joint at `at`, coming from the side it is on now; None frees it."""
        self._check_joint(joint)
        with self._lock:
            self._advance()
            if at is None:
                self._blocks.pop(joint, None)
            else:
                self._blocks[joint] = (float(at), bool(self._q[joint] <= at))

    def set_fault(self, text: str) -> None:
        with self._lock:
            self._faults.append(text)

    def clear_faults(self) -> None:
        with self._lock:
            self._faults.clear()

    def set_temperatures(self, values: Sequence[float] | None) -> None:
        with self._lock:
            self._temperatures = None if values is None else [float(v) for v in values]

    def jaw_object(self, opening: float | None) -> None:
        """Put an object between the jaws: a close stalls at this opening."""
        with self._lock:
            self._jaw_object = None if opening is None else float(opening)

    def fail_gripper_reads(self, count: int) -> None:
        """Make the next count jaw reads return None."""
        with self._lock:
            self._jaw_failing = int(count)

    @property
    def jaw_stiffness(self) -> str:
        """The jaw stiffness in force: "closing" or "light"."""
        with self._lock:
            return self._jaw_stiffness

    # Internals; the ones that touch state run with the lock held.

    def _vector(self, name: str, q: Sequence[float]) -> np.ndarray:
        values = np.array(q, dtype=float)
        if values.shape != (self.n_joints,):
            raise ValueError(f"{name} needs {self.n_joints} joint values, got {q!r}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite, got {q!r}")
        return values

    def _check_joint(self, joint: int) -> None:
        if not 0 <= joint < self.n_joints:
            raise IndexError(f"joint index {joint!r} is outside 0..{self.n_joints - 1}")

    def _advance(self) -> None:
        """Bring the measured pose up to the present."""
        now = self._clock.monotonic()
        dt = now - self._t
        if dt <= 0.0:
            return
        self._t = now
        if self._target is None:  # also the case whenever torque is off
            return
        tau = self.time_constant_s
        gain = 1.0 if tau <= 0.0 else -math.expm1(-dt / tau)
        q = self._q + (self._target - self._q) * gain
        for joint, (at, upper) in self._blocks.items():
            q[joint] = min(q[joint], at) if upper else max(q[joint], at)
        self._q = q

    def _jaw_refusal(self, opening: float, max_effort: float, mode: str) -> str | None:
        if not self._torque:
            return "torque is off"
        if not self._jaw_referenced:
            return "the jaws are not referenced"
        if not _is_fraction(opening):
            return f"opening must be between 0 and 1, got {opening!r}"
        if not _is_fraction(max_effort):
            return f"max_effort must be between 0 and 1, got {max_effort!r}"
        if mode not in JAW_MODES:
            return f"unknown mode {mode!r}"
        return None

    def _object_in_the_way(self, target: float) -> float | None:
        """The object's opening when a move to target would close onto it."""
        held = self._jaw_object
        if held is None or target >= held - _EPS or self._jaw < held - _EPS:
            return None
        return held

    def _jaw_status(
        self, target: float, cancel: threading.Event, steps_left: int
    ) -> str | None:
        if cancel.is_set():
            return "cancelled"
        if abs(self._jaw - target) <= _EPS:
            return "reached"
        held = self._object_in_the_way(target)
        if held is not None and self._jaw <= held + _EPS:
            return "stalled"
        if steps_left <= 0:
            return "timeout"
        return None

    def _jaw_after_step(self, target: float) -> float:
        travel = self.jaw_speed * self.jaw_step_s
        if target >= self._jaw:
            return min(self._jaw + travel, target)
        moved = max(self._jaw - travel, target)
        held = self._object_in_the_way(target)
        return moved if held is None else max(moved, held)

    def _set_jaw_stiffness(self, stiffness: str) -> None:
        self._jaw_stiffness = stiffness
        if self.jaw_commands:
            self.jaw_commands[-1].stiffness = stiffness


class BasicContactWatch:
    """Joint reads and the contact check for one move, from motor feedback alone.

    A joint is in contact when two of its readings in a row lie more than tol_rad
    outside the band of setpoints sent over the last lag_steps steps: it trails the
    setpoint sent that many steps earlier.
    """

    def __init__(
        self,
        arm: ArmPort,
        q_start: Sequence[float],
        history: JointHistory | None,
        clock: Clock,
        *,
        reads_per_step: int = 2,
        tol_rad: float = 0.10,
        lag_steps: int = 10,
        feedback_timeout_s: float = 0.35,
    ) -> None:
        self._arm = arm
        self._history = history
        self._clock = clock
        self._reads_per_step = reads_per_step
        self._tol_rad = tol_rad
        self._feedback_timeout_s = feedback_timeout_s
        self._n = arm.n_joints
        # The start pose and the setpoints of the last lag_steps steps, oldest first.
        self._recent: deque[list[float]] = deque(
            [[float(v) for v in q_start]], maxlen=lag_steps + 1
        )
        self._next_joint = 0
        self._off_in_a_row = [0] * self._n
        # End of each joint's last good read; the first step starts the count, so a
        # wait between building the watch and the first setpoint is not lost feedback.
        self._last_good: list[float] | None = None
        self.finished = False

    def after_send(
        self, index: int, q: Sequence[float], deadline_mono: float
    ) -> list[JointReading]:
        self._recent.append([float(v) for v in q])
        if self._last_good is None:
            self._last_good = [self._clock.monotonic()] * self._n
        readings: list[JointReading] = []
        for done in range(self._reads_per_step):
            # A read that would begin after the step's deadline is left out.
            if done > 0 and self._clock.monotonic() >= deadline_mono:
                break
            joint = self._next_joint
            self._next_joint = (joint + 1) % self._n
            try:
                reading = self._arm.read_joint(joint)
            except ReadFailed:
                continue
            readings.append(reading)
            if self._history is not None:
                self._history.add(reading)
            if reading.valid and math.isfinite(reading.value):
                self._last_good[joint] = reading.t_end
                self._check_contact(reading)
        self._check_feedback(self._last_good)
        return readings

    def hold(self) -> None:
        self._arm.hold_measured()

    def finish(self) -> None:
        self.finished = True

    def _check_contact(self, reading: JointReading) -> None:
        joint = reading.joint
        sent = [q[joint] for q in self._recent]
        off = max(min(sent) - reading.value, reading.value - max(sent), 0.0)
        if off <= self._tol_rad:
            self._off_in_a_row[joint] = 0
            return
        self._off_in_a_row[joint] += 1
        if self._off_in_a_row[joint] >= 2:
            raise ContactStop(
                f"contact: joint index {joint} trails its setpoints by {off:.3f} rad "
                f"(limit {self._tol_rad:.3f} rad)"
            )

    def _check_feedback(self, last_good: list[float]) -> None:
        now = self._clock.monotonic()
        for joint, t_good in enumerate(last_good):
            age = now - t_good
            if age > self._feedback_timeout_s:
                raise FeedbackLost(
                    f"joint feedback lost: joint index {joint} has had no good "
                    f"reading for {age:.2f} s"
                )


class BasicChecker:
    """Joint-space checks for one control owner: a stand-in for APAI's checker.

    It knows no kinematics, so the grasp region and the contact spots are only
    stored, and tip_blocked answers through a hook a test installs.
    """

    def __init__(
        self,
        limit_set: LimitSet,
        clock: Clock,
        *,
        watchdog_s: float | None = 5.0,
        reads_per_step: int = 2,
        contact_tol_rad: float = 0.10,
        contact_lag_steps: int = 10,
        feedback_timeout_s: float = 0.35,
        grasp: GraspConfig | None = None,
    ) -> None:
        self.name = limit_set.name
        self._grasp = GraspConfig() if grasp is None else grasp
        self.version = limit_set.version
        self.limit_set = limit_set
        self.watchdog_s = watchdog_s  # None: no watchdog
        self.reads_per_step = reads_per_step
        self.contact_tol_rad = contact_tol_rad
        self.contact_lag_steps = contact_lag_steps
        self.feedback_timeout_s = feedback_timeout_s

        self._clock = clock
        self._lock = threading.Lock()
        self._n = len(limit_set.joint_min)
        self._joint_min = np.array(limit_set.joint_min, dtype=float)
        self._joint_max = np.array(limit_set.joint_max, dtype=float)
        self._envelope: tuple[np.ndarray, np.ndarray] | None = None
        if limit_set.envelope_min is not None and limit_set.envelope_max is not None:
            self._envelope = (
                np.array(limit_set.envelope_min, dtype=float),
                np.array(limit_set.envelope_max, dtype=float),
            )
        self._fed_at: float | None = None
        self._halt_count = 0
        self._halt_reason: str | None = None  # set by halt, cleared by begin_motion
        self._latched = False
        self._latch_reason = ""
        self._in_motion = False
        self._grasp_region: GraspRegion | None = None
        self._contact_spots: list[list[float]] = []
        self._tip_hook: TipHook | None = None

    # Checker

    def feed_watchdog(self, t_mono: float | None = None) -> None:
        now = self._clock.monotonic()
        t = now if t_mono is None else min(float(t_mono), now)
        if not math.isfinite(t):
            return
        with self._lock:
            if self._fed_at is None or t > self._fed_at:
                self._fed_at = t

    def halt_count(self) -> int:
        with self._lock:
            return self._halt_count

    def halt(self, reason: str) -> int:
        with self._lock:
            self._halt_count += 1
            self._halt_reason = reason
            return self._halt_count

    def latch(self, reason: str) -> None:
        with self._lock:
            self._latched = True
            self._latch_reason = reason

    def reset_latch(self) -> None:
        with self._lock:
            self._latched = False
            self._latch_reason = ""

    def latched(self) -> bool:
        with self._lock:
            return self._latched

    def begin_motion(self, halt_count: int) -> None:
        now = self._clock.monotonic()
        with self._lock:
            self._refuse_if_latched()
            if halt_count != self._halt_count:
                raise Refusal(
                    f"halted since this motion was admitted: halt count is "
                    f"{self._halt_count}, the motion quotes {halt_count}"
                )
            if self.watchdog_s is not None:
                if self._fed_at is None:
                    raise Refusal("perception watchdog: never fed")
                age = now - self._fed_at
                if age > self.watchdog_s:
                    raise Refusal(
                        f"perception watchdog: last fed {age:.1f} s ago, "
                        f"limit {self.watchdog_s:g} s"
                    )
            self._halt_reason = None
            self._in_motion = True

    def end_motion(self) -> None:
        with self._lock:
            self._in_motion = False

    def check_setpoint(
        self, q_prev: Sequence[float], q_next: Sequence[float], dt: float
    ) -> None:
        with self._lock:
            self._refuse_if_latched()
            if self._halt_reason is not None:
                raise Refusal(f"halted: {self._halt_reason}")
        self._check_step("setpoint", q_prev, q_next, dt)

    def check_pose(self, q: Sequence[float]) -> None:
        with self._lock:
            self._refuse_if_latched()
        self._check_boxes("pose", self._as_pose("pose", q))

    def check_path(
        self, q_start: Sequence[float], setpoints: Sequence[Sequence[float]], dt: float
    ) -> None:
        with self._lock:
            self._refuse_if_latched()
        if len(setpoints) == 0:
            raise Refusal("the path has no setpoints")
        q_prev = q_start
        for index, q_next in enumerate(setpoints):
            self._check_step(f"setpoint {index}", q_prev, q_next, dt)
            q_prev = q_next

    def check_grasp_region(self, region: GraspRegion) -> None:
        # The table plane of the simulated arm is at z = 0; it has no workspace box.
        problem = self._grasp.problem(region)
        if problem is not None:
            raise Refusal(problem)

    def set_grasp_region(self, region: GraspRegion | None) -> None:
        if region is not None:
            self.check_grasp_region(region)
        with self._lock:
            self._grasp_region = region

    def note_contact(self, q: Sequence[float]) -> None:
        with self._lock:
            self._contact_spots.append([float(v) for v in q])

    def clear_contact_spots(self) -> None:
        with self._lock:
            self._contact_spots.clear()

    def tip_blocked(self, q: Sequence[float], region: GraspRegion | None) -> str | None:
        with self._lock:
            hook = self._tip_hook
        return None if hook is None else hook([float(v) for v in q], region)

    def contact_watch(
        self, arm: ArmPort, q_start: Sequence[float], history: JointHistory | None
    ) -> BasicContactWatch:
        return BasicContactWatch(
            arm,
            q_start,
            history,
            self._clock,
            reads_per_step=self.reads_per_step,
            tol_rad=self.contact_tol_rad,
            lag_steps=self.contact_lag_steps,
            feedback_timeout_s=self.feedback_timeout_s,
        )

    # Knobs and readouts for tests

    def set_tip_hook(self, hook: TipHook | None) -> None:
        """Let hook(q, region) answer tip_blocked; None restores "never blocked"."""
        with self._lock:
            self._tip_hook = hook

    @property
    def grasp_region(self) -> GraspRegion | None:
        with self._lock:
            return self._grasp_region

    @property
    def contact_spots(self) -> list[list[float]]:
        """The poses given to note_contact since the last clear."""
        with self._lock:
            return [list(q) for q in self._contact_spots]

    @property
    def in_motion(self) -> bool:
        """True between begin_motion and end_motion."""
        with self._lock:
            return self._in_motion

    # Internals

    def _refuse_if_latched(self) -> None:
        if self._latched:
            raise Refusal(f"fault-stop latched: {self._latch_reason}")

    def _as_pose(self, what: str, q: Sequence[float]) -> np.ndarray:
        try:
            values = np.array(q, dtype=float)
        except (TypeError, ValueError):
            raise Refusal(f"{what}: not a list of numbers") from None
        if values.shape != (self._n,):
            raise Refusal(f"{what}: needs {self._n} joint values")
        if not np.all(np.isfinite(values)):
            raise Refusal(f"{what}: a value is not finite")
        return values

    def _check_box(
        self, what: str, q: np.ndarray, low: np.ndarray, high: np.ndarray, box: str
    ) -> None:
        outside = np.flatnonzero((q < low - _EPS) | (q > high + _EPS))
        if outside.size:
            joint = int(outside[0])
            raise Refusal(
                f"{what}: joint index {joint} at {q[joint]:.4f} rad is outside the "
                f"{self.name} {box} {low[joint]:.4f}..{high[joint]:.4f}"
            )

    def _check_boxes(self, what: str, q: np.ndarray) -> None:
        self._check_box(what, q, self._joint_min, self._joint_max, "joint limits")
        if self._envelope is not None:
            self._check_box(what, q, *self._envelope, "envelope")

    def _check_step(
        self, what: str, q_prev: Sequence[float], q_next: Sequence[float], dt: float
    ) -> None:
        start = self._as_pose(f"{what}: its start", q_prev)
        end = self._as_pose(what, q_next)
        if not (math.isfinite(dt) and dt > 0.0):
            raise Refusal(f"{what}: the step time must be above 0, got {dt!r}")
        self._check_boxes(what, end)
        if self._envelope is not None:
            # The envelope cannot be re-entered: a step that starts outside is refused.
            self._check_box(f"{what}: its start", start, *self._envelope, "envelope")
        speed = np.abs(end - start) / dt
        fast = np.flatnonzero(speed > self.limit_set.max_joint_speed + _EPS)
        if fast.size:
            joint = int(fast[0])
            raise Refusal(
                f"{what}: joint index {joint} would move at {speed[joint]:.3f} rad/s, "
                f"over the {self.name} speed limit "
                f"{self.limit_set.max_joint_speed:.3f} rad/s"
            )
