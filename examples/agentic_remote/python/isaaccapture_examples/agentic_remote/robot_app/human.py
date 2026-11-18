# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A person drives the arm: the newest human target and the human drive.

A human target is a joint target with its sample age and the owner and stop
numbers it quotes; it says nothing about where it was computed. The sample age
is how old the controller data behind a target are. A human drive is one
stretch of human motion on the Thor.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

from ..protocol.messages import HUMAN_MOVE_ID
from .executor import HOLD_FAILED
from .interfaces import ArmFault, ContactStop, FeedbackLost, Owner, ReadFailed, Refusal

if TYPE_CHECKING:
    from ..protocol.messages import GraspRegion
    from .authority import Authority
    from .clock import Clock
    from .config import RobotAppConfig
    from .interfaces import (
        ArmPort,
        AuthorityState,
        Checker,
        ContactWatch,
        JointReading,
    )
    from .state import JointHistory

    # How a drive ended: (status, reason).
    Ending = tuple[str, str]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.human_drive")

# The speed clamp as a share of limits.human.max_joint_speed.
SPEED_MARGIN = 0.98

# A hold at the measured pose needs a joint read, and one failed read is tried
# again, as after a contact in a move.
_HOLD_ATTEMPTS = 2

# A target's stage says how far the human program, the program that turns
# controller data into joint targets, is with those data (HUMAN_STAGES). Only
# at this one do the data count as fresh.
_STAGE_READY = "ready"

# The gripper tip is low while it is below the table clearance or inside the
# agent's last grasp region. A person lifts it out. The floor is the highest
# tip height measured since the takeover, or since the tip last was not low.
# While the measured tip is low, a human target, and the step toward it, may
# not bring the tip under the floor, within TIP_TOL_M. LIFT begins the refusal.
# An arm settles when it comes to rest under its joint target. A tip that
# settled under the floor was not lowered by a target: from there a drive
# sends nothing until its target is back at the floor.
TIP_TOL_M = 0.001
LIFT = "lift the gripper: "


@dataclass(frozen=True)
class HumanTarget:
    """What the Thor side of the human path takes, whoever made it."""

    seq: int
    owner_number: int
    stop_number: int
    source: str  # HUMAN_SOURCE_TELEOP or HUMAN_SOURCE_SYNTHETIC
    stage: str  # one of HUMAN_STAGES
    note: str
    engaged: bool
    grip_released: bool
    q: tuple[float, ...] | None  # joint target, rad
    # Age of the controller data at arrived_mono, never less than it is: the
    # time a target spent on its way to the Thor side is in it. None: unknown,
    # stale, or not a finite number of 0 or more.
    sample_age_s: float | None
    # The app's monotonic clock when the target reached the Thor side.
    arrived_mono: float
    round_trip_s: float | None = None
    # The jaw opening the trigger asks for, 1.0 fully open to 0.0 closed; None:
    # the trigger is not usable. The human drive does not read it.
    jaw_opening: float | None = None


def _finite(value: object) -> bool:
    """True for a finite real number that is not a bool."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _age_s(target: HumanTarget, now_mono: float) -> float | None:
    """The reported sample age plus the time since arrival; None if unknown."""
    reported = target.sample_age_s
    if not _finite(reported) or reported < 0.0:
        return None
    since = now_mono - target.arrived_mono
    if not _finite(since):
        return None
    return float(reported) + max(0.0, float(since))


def _pose_of(readings: Sequence[JointReading], n: int) -> list[float] | None:
    """The pose of one full read, None unless every joint has a valid finite value."""
    pose: list[float | None] = [None] * n
    for reading in readings:
        if 0 <= reading.joint < n and reading.valid and math.isfinite(reading.value):
            pose[reading.joint] = float(reading.value)
    return None if any(value is None for value in pose) else pose


class HumanInput:
    """The newest human target of this robot connection. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._newest: HumanTarget | None = None

    def put(self, target: HumanTarget) -> None:
        """Keep the last target that arrived; seq is not compared."""
        with self._lock:
            self._newest = target

    def newest(self) -> HumanTarget | None:
        with self._lock:
            return self._newest

    def sample_age_s(self, now_mono: float) -> float | None:
        """The newest target's reported age plus the time since it arrived.

        None without a target and when its age is unknown.
        """
        target = self.newest()
        return None if target is None else _age_s(target, now_mono)

    def stage(self, now_mono: float, keep_s: float) -> str:
        """The newest target's stage; "" when none arrived within keep_s."""
        target = self.newest()
        if target is None:
            return ""
        since = now_mono - target.arrived_mono
        return target.stage if since <= keep_s else ""

    def clear(self) -> None:
        """Forget the newest target."""
        with self._lock:
            self._newest = None


class DriveEnd(NamedTuple):
    """How a human drive ended. Every end is a hold with the motors on."""

    # "released", "disengaged", "stopped", "stale", "view", "refused", "late",
    # "contact" or "failed".
    status: str
    reason: str  # a "failed" reason starts with "fault: "
    sent: int  # setpoints sent
    # The measured end pose, None when it was not read. A drive reads no joint
    # for it: it is the start read's pose while nothing was sent after it.
    end_q: list[float] | None


@dataclass
class _Drive:
    """What one run() call carries from pass to pass."""

    admitted: AuthorityState
    should_stop: Callable[[], str | None]
    # The pose the start read measured, None before it.
    measured: list[float] | None = None
    # The newest reading of each joint: the start read's, then the watch's.
    at: list[float] = field(default_factory=list)
    # Where the next step starts: the measured pose, then the last setpoint sent.
    q_prev: list[float] = field(default_factory=list)
    # This pass's setpoint once it is checked, and the target it follows.
    q_next: list[float] = field(default_factory=list)
    target: HumanTarget | None = None
    fed_seq: int | None = None  # seq of the target the watchdog was last fed with
    begun: bool = False  # the checker's begin_motion passed
    watch: ContactWatch | None = None
    sent: int = 0  # setpoints sent so far
    due: float = 0.0  # when this pass's setpoint is due, on the app's clock
    # This pass sends nothing: the arm has settled under the floor of its
    # gripper tip and the target is not back at the floor.
    waiting: bool = False
    waited: int = 0  # passes that sent nothing so far
    # The highest gripper tip of the poses this drive's steps began at since
    # the tip was low, m; None while it is not low.
    step_top: float | None = None


class HumanDrive:
    """Runs one human drive at a time, toward the newest human target.

    Once per period it asks the stop conditions, the newest target's numbers,
    stage and age and, where the file requires it, the live view's age, moves
    one speed-limited step toward the target, has the human checker check it
    and sends it. While the measured gripper tip is low, a target that would
    bring it under its floor ends the drive, refused; where the arm itself
    settled under the floor, the drive sends nothing until the target is back
    at the floor.
    Every end leaves the last setpoint sent standing, which is a hold with the
    motors on; a contact holds at the measured pose. It never changes the
    control owner, raises no stop, cuts no torque and leaves the jaws alone.
    """

    def __init__(
        self,
        arm: ArmPort,
        authority: Authority,
        checker: Checker,
        history: JointHistory | None,
        config: RobotAppConfig,
        clock: Clock,
        human_input: HumanInput,
        view_age: Callable[[float], float | None],
        *,
        claim_idle: Callable[[], AbstractContextManager[bool]],
        note_clamped: Callable[[], None] | None = None,
        reads_paused: Callable[[], AbstractContextManager[None]] | None = None,
        last_grasp_region: Callable[[], GraspRegion | None] | None = None,
    ) -> None:
        """checker is the human checker; view_age(now) is the live view's age.

        A drive runs inside claim_idle() (MoveExecutor.claim_idle), so no move
        starts meanwhile. note_clamped is called for a step the speed clamp
        cut. From its start read on the drive runs inside reads_paused()
        (IdleReader.paused). last_grasp_region() is the grasp region of the
        agent's last grasp, None when there is none.
        """
        self._last_grasp_region = last_grasp_region or (lambda: None)
        self._arm = arm
        self._authority = authority
        self._checker = checker
        self._history = history
        self._config = config
        self._clock = clock
        self._human_input = human_input
        self._view_age = view_age
        self._claim_idle = claim_idle
        self._note_clamped = note_clamped
        self._reads_paused = reads_paused or contextlib.nullcontext
        self._lock = threading.Lock()  # guards _progress
        self._progress: tuple[str, int, int] | None = None
        # (owner number, floor): the highest gripper tip height measured, m,
        # since that takeover or since the tip last was not low. None while
        # the tip is not low. Only run() uses it, one drive at a time.
        self._tip_floor: tuple[int, float] | None = None

    def progress(self) -> tuple[str, int, int] | None:
        """(HUMAN_MOVE_ID, setpoints sent, 0) while a drive runs, else None."""
        with self._lock:
            return self._progress

    def run(
        self, admitted: AuthorityState, should_stop: Callable[[], str | None]
    ) -> DriveEnd:
        """One drive from its start checks to its end; it never raises a stop.

        admitted holds the owner number and the stop number the drive was
        admitted with; should_stop() gives a reason once the drive must end.
        Nothing is sent before the start checks, the start read and the
        checker's begin have passed.
        """
        with self._claim_idle() as idle:
            if not idle:
                # Also while a torque release holds the claim.
                return DriveEnd("refused", "another move is running", 0, None)
            drive = _Drive(admitted, should_stop)
            self._set_progress(0)
            try:
                try:
                    ended = self._start(drive)
                    if ended is None:
                        # No idle read shares the bus, or the clock, with the
                        # drive, the hold after a contact included.
                        with self._reads_paused():
                            try:
                                ended = self._begin(drive)
                                while ended is None:
                                    ended = self._next_setpoint(drive)
                                    if ended is None and drive.waiting:
                                        self._after_wait(drive)
                                    elif ended is None:
                                        self._arm.send_target(drive.q_next)
                                        ended = self._after_send(drive)
                            except ContactStop as contact:
                                ended = self._after_contact(drive.watch, contact.reason)
                except ContactStop as contact:
                    # Raised before the start read: nothing is paused yet.
                    ended = self._after_contact(drive.watch, contact.reason)
                except FeedbackLost as lost:
                    # Like a driver fault: the caller raises a fault-stop.
                    ended = ("failed", f"fault: {lost.reason}")
                except Refusal as refusal:
                    ended = self._refused(drive, refusal.reason)
                except ReadFailed as failed:
                    ended = ("refused", f"a joint read failed: {failed}")
            except ArmFault as fault:
                ended = ("failed", f"fault: {fault}")
            finally:
                self._leave(drive)
        status, reason = ended
        if drive.waited:
            _log.info(
                "human drive: %d passes sent nothing: the arm had settled and "
                "the target was under the floor of the low gripper tip",
                drive.waited,
            )
        _log.info(
            "human drive: %s, sent %d setpoints (%s)",
            status,
            drive.sent,
            reason or "no reason given",
        )
        # No setpoint went out after the start read: the arm is still there.
        end_q = drive.measured if drive.sent == 0 else None
        return DriveEnd(status, reason, drive.sent, end_q)

    def _start(self, drive: _Drive) -> Ending | None:
        """The checks before the start read; then the watchdog is fed."""
        stop = self._stop_reason(drive)
        if stop is not None:
            return "stopped", stop
        if not self._arm.torque_on():
            return "refused", "the motors are off"
        target, ended = self._fresh_target(drive, self._clock.monotonic())
        if target is None:
            return ended
        # Nothing else feeds the human checker, and its begin needs a feed.
        self._feed(drive, target)
        return None

    def _begin(self, drive: _Drive) -> Ending | None:
        """The start read, the checker's begin and the contact watch."""
        try:
            readings = self._arm.read_all()
        except ReadFailed as failed:
            return "refused", f"the start pose could not be read: {failed}"
        self._keep(readings)
        measured = _pose_of(readings, self._config.n_joints)
        if measured is None:
            return "refused", "the start pose read left a joint out"
        drive.measured = measured
        drive.at = list(measured)
        drive.q_prev = measured  # the first step starts where the arm is
        self._checker.begin_motion(self._checker.halt_count())
        drive.begun = True
        drive.watch = self._checker.contact_watch(self._arm, measured, self._history)
        drive.due = self._clock.monotonic()
        return None

    def _next_setpoint(self, drive: _Drive) -> Ending | None:
        """One pass up to its send: drive.q_next is checked and may go out.

        A late wake-up moves this setpoint and those after it back. The step
        keeps its size, so a late pass is never made up.
        """
        config = self._config.executor
        stop = self._stop_reason(drive)
        if stop is not None:
            return "stopped", stop
        now = self._clock.monotonic()
        target, ended = self._fresh_target(drive, now)
        if target is None:
            return ended
        late = now - drive.due
        if late > config.large_delay_s:
            return self._late(drive.sent, late, "is")
        if late > 0.0:
            drive.due = now
        far = self._step_problem(target.q, drive.q_prev)
        if far is not None:
            return "refused", far
        q_next = self._clamped(target.q, drive.q_prev)
        lower = self._lift_problem(drive, target.q, q_next)
        if lower is not None:
            return "refused", lower
        if not drive.waiting:
            self._checker.check_setpoint(drive.q_prev, q_next, config.period_s)
        # The check took time: what held before it is asked again.
        stop = self._stop_reason(drive)
        if stop is not None:
            return "stopped", stop
        now = self._clock.monotonic()
        target, ended = self._fresh_target(drive, now)
        if target is None:
            return ended
        if now - drive.due > config.large_delay_s:
            return self._late(drive.sent, now - drive.due, "is")
        drive.q_next = q_next
        drive.target = target
        return None

    def _after_send(self, drive: _Drive) -> Ending | None:
        """What follows a send: the count, the joint reads, the feed, the sleep."""
        config = self._config.executor
        index = drive.sent
        behind = self._clock.monotonic() - drive.due
        drive.sent += 1
        drive.q_prev = drive.q_next
        self._set_progress(drive.sent)
        if behind > config.large_delay_s:
            return self._late(index, behind, "went out")
        if behind > config.small_delay_s:
            # A delay, not the usual cost: the next setpoint keeps its distance.
            drive.due += behind
        deadline = drive.due + config.period_s
        self._read(drive, index, drive.q_next, deadline)
        # The checker suspends its watchdog while it is in motion: this feed
        # serves the next drive, and the sample age is the watchdog of this one.
        self._feed(drive, drive.target)
        self._clock.sleep(max(0.0, deadline - self._clock.monotonic()))
        drive.due = deadline
        return None

    def _after_wait(self, drive: _Drive) -> None:
        """What follows a pass that sent nothing: the joint reads, the feed, the sleep.

        The last setpoint sent stands. The watch is given the pose the
        drive's next step begins at: that is all this drive has asked for.
        """
        drive.waited += 1
        deadline = drive.due + self._config.executor.period_s
        self._read(drive, drive.sent, drive.q_prev, deadline)
        self._feed(drive, drive.target)
        self._clock.sleep(max(0.0, deadline - self._clock.monotonic()))
        drive.due = deadline

    def _read(
        self, drive: _Drive, index: int, q: Sequence[float], deadline: float
    ) -> None:
        """One pass's joint reads by the contact watch; drive.at follows them."""
        readings = drive.watch.after_send(index, q, deadline)
        for reading in readings or ():
            known = 0 <= reading.joint < len(drive.at)
            if known and reading.valid and math.isfinite(reading.value):
                drive.at[reading.joint] = float(reading.value)

    def _stop_reason(self, drive: _Drive) -> str | None:
        """Why the drive must stop now, None while it may go on."""
        reason = drive.should_stop()
        if reason is not None:
            return reason or "stop requested"
        now = self._authority.snapshot()
        admitted = drive.admitted
        if now.stop_number != admitted.stop_number:
            return (
                f"stop number is now {now.stop_number}, the drive was admitted "
                f"at {admitted.stop_number}"
            )
        if now.owner_number != admitted.owner_number:
            return (
                f"owner number is now {now.owner_number}, the drive was admitted "
                f"at {admitted.owner_number}"
            )
        if now.owner is not Owner.HUMAN:
            return f"control owner is {now.owner}, not {Owner.HUMAN}"
        return None

    def _fresh_target(
        self, drive: _Drive, now: float
    ) -> tuple[HumanTarget, None] | tuple[None, Ending]:
        """The newest target when a setpoint may follow it now, else the end.

        It must quote the admitted numbers, be engaged with one finite value
        per joint and be fresh (stage "ready", a sample age within the limit),
        and with human.require_live_view the live view must be fresh.
        """
        human = self._config.human
        target = self._human_input.newest()
        if target is None:
            return None, ("stale", "controller data stale (no human target)")
        admitted = drive.admitted
        quoted = (target.owner_number, target.stop_number)
        if quoted != (admitted.owner_number, admitted.stop_number):
            return None, (
                "stopped",
                f"the human target quotes owner number {target.owner_number} and "
                f"stop number {target.stop_number}, the drive was admitted at "
                f"{admitted.owner_number} and {admitted.stop_number}",
            )
        q = target.q
        if not target.engaged or q is None:
            if target.grip_released:
                return None, ("released", "the grip was released")
            note = target.note or "the human program ended the engagement"
            return None, ("disengaged", note)
        if target.stage != _STAGE_READY:
            reason = f"controller data stale (stage {target.stage or 'unknown'})"
            return None, ("stale", reason)
        n = self._config.n_joints
        if len(q) != n:
            reason = f"the target has {len(q)} values, the arm {n} joints"
            return None, ("refused", reason)
        if not all(_finite(value) for value in q):
            reason = "the target holds a value that is not a finite number"
            return None, ("refused", reason)
        age = _age_s(target, now)
        if age is None:
            reason = "controller data stale (the sample age is unknown)"
            return None, ("stale", reason)
        limit = human.sample_age_limit_s
        if age > limit:
            reason = (
                f"controller data stale ({age:.3f} s old, the limit is {limit:g} s)"
            )
            return None, ("stale", reason)
        if human.require_live_view:
            view = self._view_age(now)
            limit = human.view_age_limit_s
            if view is None:
                reason = (
                    "the live view is not fresh (the viewer reports no live picture)"
                )
                return None, ("view", reason)
            if not 0.0 <= view <= limit:
                reason = (
                    f"the live view is not fresh ({view:.3f} s old, the limit "
                    f"is {limit:g} s)"
                )
                return None, ("view", reason)
        return target, None

    def _step_problem(self, q: Sequence[float], q_prev: Sequence[float]) -> str | None:
        """Why the target lies too far from the last setpoint, else None.

        Such a target is refused, not clipped: it is an error, not a fast hand.
        """
        limit = self._config.limits.human.max_step_rad
        joint = max(range(len(q)), key=lambda i: abs(q[i] - q_prev[i]))
        off = abs(q[joint] - q_prev[joint])
        if off <= limit:
            return None
        return (
            f"the target is {off:.3f} rad from the arm on joint index {joint}, "
            f"the limit is {limit:g} rad"
        )

    def _lift_problem(
        self, drive: _Drive, q: Sequence[float], q_next: Sequence[float]
    ) -> str | None:
        """Why the target q, or the step q_next toward it, is refused, else None.

        Only while the measured gripper tip is low; with a checker that knows
        no table it never is. It sets drive.waiting for a pass that sends
        nothing: the arm has settled under the floor, and the target is under
        the floor but not under the tip.
        """
        drive.waiting = False
        region = self._last_grasp_region()
        tip = self._checker.tip_state(drive.at, region)
        if tip is None or tip.low is None:
            self._tip_floor = None
            drive.step_top = None
            return None

        def lowers(what: str, z: float) -> str:
            return f"{LIFT}{tip.low}, and {what} would lower it to z={z:.3f} m"

        owner = drive.admitted.owner_number
        kept = self._tip_floor
        floor = tip.z if kept is None or kept[0] != owner else max(kept[1], tip.z)
        self._tip_floor = (owner, floor)
        lowest = floor - TIP_TOL_M
        target_z = self._tip_z(q, region)
        if target_z < lowest:
            # No target under the floor is followed, so a tip measured under
            # it was not asked for: the arm settled there. A target under
            # that tip is a lowering the person asks for all the same.
            if tip.z < lowest and target_z >= tip.z - TIP_TOL_M:
                drive.waiting = True
                return None
            return lowers("the target", target_z)
        # The steps of a lift from where the arm settled begin under the
        # floor: they are held against the highest of them until they reach it.
        if drive.step_top is None:
            drive.step_top = self._tip_z(drive.q_prev, region)
        base = floor if drive.step_top >= lowest else drive.step_top
        step_z = self._tip_z(q_next, region)
        if step_z < base - TIP_TOL_M:
            return lowers("the step toward the target", step_z)
        drive.step_top = max(drive.step_top, step_z)
        return None

    def _tip_z(self, q: Sequence[float], region: GraspRegion | None) -> float:
        """The gripper tip's height at q, m, from a checker that knows the table."""
        state = self._checker.tip_state(q, region)
        return math.inf if state is None else state.z

    def _clamped(self, q: Sequence[float], q_prev: Sequence[float]) -> list[float]:
        """One step from q_prev toward q, each joint within the speed clamp.

        The clamp stays SPEED_MARGIN under the limit, because a checker may
        refuse a step that rounding put above its speed limit by any amount.
        """
        speed = self._config.limits.human.max_joint_speed
        most = SPEED_MARGIN * speed * self._config.executor.period_s
        q_next: list[float] = []
        clipped = False
        for at, to in zip(q_prev, q):
            if abs(to - at) > most:
                clipped = True
                q_next.append(at + math.copysign(most, to - at))
            else:
                q_next.append(float(to))
        if clipped and self._note_clamped is not None:
            self._note_clamped()
        return q_next

    def _feed(self, drive: _Drive, target: HumanTarget) -> None:
        """Feed the checker's watchdog once per seq, with the controller data's time."""
        if target.seq == drive.fed_seq:
            return
        drive.fed_seq = target.seq
        self._checker.feed_watchdog(target.arrived_mono - target.sample_age_s)

    def _refused(self, drive: _Drive, reason: str) -> Ending:
        """A refusal, reported as a stop when the drive was stopped meanwhile."""
        stop = self._stop_reason(drive)
        return ("refused", reason) if stop is None else ("stopped", stop)

    def _late(self, index: int, by_s: float, what: str) -> Ending:
        """The end of a drive whose setpoint is, or went out, over the large delay."""
        limit = self._config.executor.large_delay_s
        return "late", (
            f"setpoint {index} {what} {by_s:.3f} s late, over the limit of {limit:g} s"
        )

    def _after_contact(self, watch: ContactWatch | None, contact: str) -> Ending:
        """The end after a contact: a hold at the measured pose.

        A hold that raises more than a failed read ends the drive as a fault,
        so that no error of the watch leaves run().
        """
        try:
            return "contact", contact + self._hold_measured(watch)
        except Refusal as refusal:
            # What such a hold sent is not known: the caller raises a fault-stop.
            return "failed", f"fault: {contact}{HOLD_FAILED}{refusal.reason}"

    def _hold_measured(self, watch: ContactWatch | None) -> str:
        """Hold at a freshly measured pose; text for the reason when that failed.

        The drive's watch does it, so the reading is one that watch trusts.
        When it fails nothing is sent and the last setpoint stands.
        """
        if watch is None:
            return ""  # the drive had sent nothing yet: the target before it stands
        error: ReadFailed | None = None
        for _ in range(_HOLD_ATTEMPTS):
            try:
                watch.hold()
            except ReadFailed as failed:
                error = failed
            else:
                return ""
        _log.error("no hold at the measured pose after a contact: %s", error)
        return f"{HOLD_FAILED}{error}; the last setpoint stands"

    def _keep(self, readings: Sequence[JointReading]) -> None:
        if self._history is not None:
            for reading in readings:
                self._history.add(reading)

    def _set_progress(self, sent: int | None) -> None:
        with self._lock:
            self._progress = None if sent is None else (HUMAN_MOVE_ID, sent, 0)

    def _leave(self, drive: _Drive) -> None:
        """What every end owes the checker and the watch."""
        try:
            if drive.begun:
                self._checker.end_motion()
        finally:
            try:
                if drive.watch is not None:
                    drive.watch.finish()
            finally:
                self._set_progress(None)
