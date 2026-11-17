# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The robot app: every module wired together, its threads and its link messages.

See sections 6 and 9 to 13 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import statistics
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..protocol.carrier import send
from ..protocol.messages import (
    CLOSE_JAWS,
    COMMAND_TYPES,
    DRIVER_APAI_MOCK,
    DRIVER_REAL,
    DRIVER_SIM,
    HUMAN_SOURCE_TELEOP,
    MESSAGE_TYPES,
    OPEN_ACTIONS,
    OPEN_JAWS,
    REQUEST_HANDBACK,
    REQUEST_HUMAN,
    REQUEST_MOTORS_OFF,
    REQUEST_MOTORS_ON,
    REQUEST_PARK,
    SIMULATED_DRIVERS,
    CheckerState,
    Event,
    Gripper,
    HumanSetpoint,
    Move,
    Open,
    OwnerRequest,
    Permit,
    ProtocolError,
    Release,
    Reset,
    Result,
    Shutdown,
    Stop,
    ViewStatus,
    Welcome,
    command_id_at_head,
    command_id_in,
    decode,
)
from . import rest as rest_module
from .authority import Authority
from .executor import HOLD_FAILED, MoveExecutor
from .gripper import (
    ALREADY_REFERENCED,
    CLOSE,
    CLOSE_TORQUE,
    CLOSE_UNTIL_CONTACT,
    JAWS_BUSY,
    OPEN,
    PLAIN,
    GripperWorker,
    JawFollower,
    JawState,
)
from .human import DriveEnd, HumanDrive, HumanInput, HumanTarget
from .inbox import Inbox
from .interfaces import (
    ArmFault,
    AuthorityState,
    MotionState,
    Owner,
    ReadFailed,
    Refusal,
    StopLevel,
)
from .permits import FUTURE_TOLERANCE_S, PermitGate
from .rest import ReturnToRest
from .state import IdleReader, JointHistory, StatePublisher
from .stop import StopController
from .text import (
    MAX_ID_CHARS,
    MAX_OUT_CHARS,
    MAX_REASON_CHARS,
    MAX_WORD_CHARS,
    LineLimit,
    printable,
)
from .waits import AssistanceWait

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from ..protocol.messages import GraspRegion, Message
    from .clock import Clock
    from .config import RobotAppConfig
    from .interfaces import (
        ArmPort,
        CheckerSet,
        GripperOutcome,
        JointReading,
        MoveOutcome,
    )

    # Hears every operator event: (kind, text, data).
    EventListener = Callable[[str, str, dict[str, Any]], None]
    # Told the final RESULT of a request made at the Thor terminal.
    DoneFn = Callable[[Result], None]

_log = logging.getLogger("isaaccapture.examples.agentic_remote.app")

# RESULT statuses.
ACCEPTED = "accepted"
REFUSED = "refused"
COMPLETED = "completed"
CANCELLED = "cancelled"
FAILED = "failed"

# Kinds of the operator events the app itself emits; the wait's kinds, the
# stops' and rest.py's pass through unchanged.
EVENT_HELP_NEEDED = "help_needed"
EVENT_OWNER = "owner"
EVENT_HUMAN = "human"
EVENT_CONNECTION = "connection"
EVENT_REFUSED = "refused"
EVENT_SHUTDOWN = "shutdown"
EVENT_JAWS = "jaws"

THOR_TERMINAL = "thor_terminal"
# Joint feedback lost for longer than this in a hold is a fault-stop.
LOST_FEEDBACK_S = 0.35
# After a connect, how long one joint read takes on the motor bus is said
# once, from this many idle reads: five rounds of a six-joint arm.
READ_TIMING_READS = 30
# How many command ids keep their newest RESULT for a repeated command.
RESULT_IDS_KEPT = 256
# What request_rest(None) asks the Thor terminal's confirm().
PATH_CLEAR_PROMPT = "Return to rest: is the path to the rest pose clear?"

# Why nothing moves the arm after a release at rest, and the way on: the
# motors are switched on again without a new start of the robot app.
MOTORS_OFF = (
    "the motors are off; switch them on again: type on at the Thor terminal or "
    "run the console command on"
)
# What request_power_on(None) asks the Thor terminal's confirm(): the question
# of the functional launcher before its motor connect, for this arm.
POWER_ON_PROMPT = (
    "Connect the powered arm again? Stay beside the e-stop. The arm must lie at rest."
)
CONFIRM_POWER_ON = "confirm that the motors may be switched on"
# What ends a typed rest whose arm lies at rest with the motors still on.
AT_REST_MOTORS_ON = "at rest with the motors on"
# A typed rest while a takeover is open ends with the motors on: the
# hand-back follows it.
REST_KEEPS_MOTORS = "a takeover is open: the motors stay on at rest for the hand-back"
# Longest the rest's jaw close waits for the gripper worker, in real time.
_JAW_CLOSE_WAIT_S = 20.0
# A torque release that raised: the driver may have switched some or all of
# the motors off before it failed. RELEASE_FAILED begins the fault-stop's
# reason; MOTORS_UNKNOWN is why nothing moves the arm from then on. NEW_START
# is the way on that works with every driver: the reBot arm's driver takes no
# second release, a quit only tries one, and Ctrl+C is a STOP.
TORQUE_UNKNOWN = "the motors may be on or off"
RELEASE_FAILED = f"torque release failed, {TORQUE_UNKNOWN}"
NEW_START = "end the robot app with SIGTERM and start it again with the arm at rest"
MOTORS_UNKNOWN = f"a torque release failed and {TORQUE_UNKNOWN}; {NEW_START}"
CONFIRM_PATH = rest_module.CONFIRM_PATH
CONFIRM_OPEN = "confirm the jaw opening"
# Why a typed close is refused: the jaws are not proven empty, and the operator
# did not type that they may close on what they hold.
CONFIRM_CLOSE = "confirm the jaw close: the jaws are not proven empty"
CONFIRM_JAWS_EMPTY = "confirm that the jaws are empty"
# The operator's word: typed in place of the proof that the jaws are empty,
# for a return to rest, a torque release, a shutdown or a hand-back. It stands
# until the jaws are next sent a command or the arm gets a new control owner.
JAWS_EMPTY_WORD = (
    "the operator typed that the jaws are empty: taken in place of the "
    "measurement for {what}"
)
# The robot app setting that switches the typed jaw reference on.
REFERENCE_SETTING = "jaws.allow_reference"
REFERENCE_OFF = (
    f"the typed jaw reference is switched off: the robot app setting "
    f"{REFERENCE_SETTING} is false"
)
# What WELCOME.driver may say; "" declares nothing.
DRIVERS = ("", DRIVER_SIM, DRIVER_APAI_MOCK, DRIVER_REAL)
LINK_LOST = "robot connection lost"
# The text of the "connection" event of a new robot connection.
LINK_CONNECTED = (
    "robot connection made: the arm holds as it was; a new connection starts no motion"
)
CLOSING = "the robot app is closing"
NOT_RUNNING = "the robot app is not running"
# Told once for as long as a second human program goes on sending
# HUMAN_SETPOINT link messages: where the second one sends them, then where
# the one that is followed does.
SECOND_HUMAN_PROGRAM = (
    "a second human program sends HUMAN_SETPOINT link messages {here}: they are "
    "dropped while the one {there} sends; one human program at a time"
)
_AT_SOCKET = "at the robot app's socket"
_OVER_LINK = "over the robot connection"
_RACED = "a STOP or an owner change arrived while the request was checked"

# A person drives the arm. A takeover makes a person the control owner; the
# hand-back ends it at rest; from the one to the other a takeover is open. A
# human target is a joint target from a person's hand; a human drive is one
# stretch of human motion.
HUMAN_OFF = "human driving is switched off in this robot app's file (human.enabled)"
TAKEOVER_OPEN = "takeover not finished: give the arm back with right A at rest"
NO_TAKEOVER = "no takeover is open"
NO_CONTROLLER_DATA = "no fresh controller data: "
VIEW_NOT_FRESH = "the live view is not fresh: "
OPEN_THE_VIEW = "; open the live view page in a visible browser tab"
# Said at the start of a robot app whose file lets a person drive without a
# live view (human.enabled true, human.require_live_view false).
NO_LIVE_VIEW = (
    "human driving needs no live view: the person at the controller must see "
    "the arm itself"
)
OUTSIDE_HUMAN_LIMITS = "the arm is outside the human limits: "
# Added to a takeover's refusal when the controller data or the live view were
# fresh before the takeover read the joints and are not after it.
READ_TOOK = "; the joint read took {took:.3f} s"
# What a refused hand-back tells the person to do: by what is not proven.
REST_FIRST = "; type rest first"
OPEN_FIRST = "; type stop, then open, then yes, then hold right A again"
# How rest.py's text begins when the jaws are not proven empty.
_JAWS_NOT_EMPTY = "the jaws are not proven empty"
TAKEN_OVER = (
    "TAKEOVER: a person owns the arm, owner number {number}; release the grip, "
    "then squeeze it to move"
)
# Added to TAKEN_OVER when the gripper tip is low at the takeover: below the
# table clearance or inside the agent's last grasp region.
LOW_TIP = "; the gripper tip is low: only a lift is taken until it is clear"
HANDED_BACK = "takeover ended: the arm is given back; nobody owns it"
# Why an engaged human target moves nothing: the grip was squeezed before the
# person could move the arm, since a drive that the grip did not end, while
# the controller data were away, or while other work ran.
NEEDS_RELEASE = "release the grip, then squeeze it to move"
DRIVE_ENDED = (
    "human drive ended: {why}; the arm holds; release the grip and squeeze again"
)
# Why the robot app itself ends a human drive that is queued or runs: the
# controller data were away between two human targets, or the admission
# refused an engaged target. TARGET_REFUSED is followed by that refusal.
DATA_AWAY = "controller data were away"
TARGET_REFUSED = "an engaged human target was refused: "
# How the reason of a human drive that ended "stopped" begins when no STOP, no
# owner change and no lost robot connection told the operator of it: the two
# above, and the drive's own (human.py) for a human target that quotes other
# numbers than the drive was admitted with.
_UNTOLD_STOPS = (DATA_AWAY, TARGET_REFUSED, "the human target quotes ")
# How the note of a human target begins after its sender ended a drive; the
# sender's copy is NOTE_SQUEEZE_AGAIN of human/logic.py. DRIVE_ENDED says it.
NOTE_SQUEEZE_AGAIN = "release the grip and squeeze again: "
PERSON_KEEPS_ARM = (
    "; the person keeps the arm: squeeze again when the connection is back"
)
# A human target's stage says how far the human program is with its controller
# data (HUMAN_STAGES). Only at this one do the data count as fresh.
STAGE_READY = "ready"
# The statuses of a human drive (DriveEnd.status) the app treats on their own.
_DRIVE_RELEASED = "released"
_DRIVE_STOPPED = "stopped"
_DRIVE_CONTACT = "contact"
_DRIVE_FAILED = "failed"

CLEAR_CONTACT_SPOTS = "clear_contact_spots"
CLOSE_GRASP = "close_grasp"

# The most bytes of payload the app parses: a few kB for every link message but
# a MOVE, whose limit follows from the longest move the configuration allows,
# with room per joint value for 17 digits, a sign, a point and an exponent.
SMALL_PAYLOAD_BYTES = 16 * 1024
MOVE_BASE_BYTES = 8 * 1024
MOVE_VALUE_BYTES = 32
# The most characters kept of a link message's text fields; the others are cut
# like a reason. An id is never cut: a command with an unusable one is refused.
_TEXT_LIMITS = {
    "sender": MAX_WORD_CHARS,
    "request": MAX_WORD_CHARS,
    "action": MAX_WORD_CHARS,
    "mode": MAX_WORD_CHARS,
    "source": MAX_WORD_CHARS,
    "stage": MAX_WORD_CHARS,
}
_ID_FIELDS = ("command_id", "move_id")
# What an operator's command is called when it is refused undecoded.
_OPERATOR_COMMANDS = {
    OwnerRequest.TYPE: "owner request",
    Reset.TYPE: "reset",
    Release.TYPE: "release",
    Open.TYPE: "open",
    Shutdown.TYPE: "shutdown",
}
BAD_STOP = "STOP with a bad payload"

# Longest close() waits, in real time, for one of the app's threads to end.
_JOIN_S = 5.0
# A repeated error of the dispatcher is logged again after this many repeats.
_LOG_EVERY = 1000
# The link is asked whether the pass's time is up after this many link messages.
_TAKEN_PER_CHECK = 256
_NUMBER = re.compile(r"\d+")

_MOVE_STATUS = {
    "completed": COMPLETED,
    "stopped": CANCELLED,
    "refused": REFUSED,
    "late": FAILED,
    "not_settled": FAILED,
    "contact": FAILED,
    "failed": FAILED,
}
_REST_STATUS = {
    rest_module.AT_REST: COMPLETED,
    rest_module.REFUSED: REFUSED,
    rest_module.STOPPED: CANCELLED,
    rest_module.FAILED: FAILED,
}
_JAW_REASON = {
    "stalled": "the jaws stalled short of their target",
    "timeout": "the jaw command timed out",
}
# The jaw modes that close: where the jaws stop is what was asked for, and the
# end of a stage's time is no error.
_CLOSES = (CLOSE_UNTIL_CONTACT, CLOSE, CLOSE_TORQUE)


@dataclass(frozen=True)
class _Ending:
    """How a piece of motion work ended: the content of its final RESULT."""

    status: str
    reason: str = ""
    end_q: list[float] | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    close: bool = False  # the app closes once this was reported


@dataclass(frozen=True)
class _Reply:
    """Where a command's RESULTs go: the link, or a Thor terminal caller."""

    command_id: str
    deliver: Callable[[Result], None]
    announces: bool  # link commands report "accepted" when they are queued

    def accepted(self) -> None:
        if self.announces:
            self.deliver(Result(self.command_id, ACCEPTED))

    def final(
        self,
        status: str,
        reason: str = "",
        end_q: list[float] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.deliver(Result(self.command_id, status, reason, end_q, dict(detail or {})))


@dataclass
class _Job:
    """One piece of work for the motion worker."""

    what: str  # for "busy" refusals and the log
    run: Callable[[_Job], _Ending]
    reply: _Reply | None  # None for the timed return: no command asked for it
    needs_link: bool  # ends in a hold when the robot connection is lost
    is_move: bool = False  # an agent MOVE
    is_human_drive: bool = False
    # Written under the motion lock; read by the move before every setpoint.
    stop_reason: str | None = None

    def should_stop(self) -> str | None:
        return self.stop_reason


def _any_job(job: _Job) -> bool:
    return True


def _a_human_drive(job: _Job) -> bool:
    return job.is_human_drive


def _no_human_drive(job: _Job) -> bool:
    return not job.is_human_drive


class _NoCarrier:
    """Stands for the robot connection while the app has none: a dead link."""

    alive = False

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        return None

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        return None

    def receive(self, timeout_s: float) -> tuple[int, bytes] | None:
        return None

    def close(self) -> None:
        return None


class _Both:
    """Two carriers as one, for a link message that goes to both."""

    def __init__(self, first: Carrier, second: Carrier) -> None:
        self._first, self._second = first, second

    @property
    def alive(self) -> bool:
        return bool(self._first.alive or self._second.alive)

    def send_reliable(self, msg_type: int, payload: bytes) -> None:
        self._first.send_reliable(msg_type, payload)
        self._second.send_reliable(msg_type, payload)

    def send_latest(self, msg_type: int, payload: bytes) -> None:
        self._first.send_latest(msg_type, payload)
        self._second.send_latest(msg_type, payload)


class _MotionBox:
    """The arm's motion state, for rest.py and ROBOT_STATE; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = MotionState.IDLE

    def get(self) -> MotionState:
        with self._lock:
            return self._state

    def set(self, state: MotionState) -> None:
        with self._lock:
            self._state = state


class _Results:
    """The newest RESULT of each of the last command ids; thread-safe.

    A command that has not ended is never forgotten: it is still to get its
    final RESULT, and sent again it must not run a second time.
    """

    def __init__(self, keep: int = RESULT_IDS_KEPT) -> None:
        self._keep = keep
        self._lock = threading.Lock()
        self._by_id: OrderedDict[str, Result] = OrderedDict()

    def get(self, command_id: str) -> Result | None:
        with self._lock:
            return self._by_id.get(command_id)

    def room_for(self, command_id: str) -> bool:
        """False when only unfinished commands are kept and this is a further one."""
        with self._lock:
            if command_id in self._by_id or len(self._by_id) < self._keep:
                return True
            return any(kept.status != ACCEPTED for kept in self._by_id.values())

    def put(self, result: Result, deliver: Callable[[Result], None]) -> bool:
        """Store and deliver a RESULT; False if its command has a final one.

        Delivered under the lock, so RESULTs leave in the order they are stored.
        """
        with self._lock:
            stored = self._by_id.get(result.command_id)
            if stored is not None and stored.status != ACCEPTED:
                return False
            self._by_id[result.command_id] = result
            while len(self._by_id) > self._keep and self._evict():
                pass
            deliver(result)
        return True

    def _evict(self) -> bool:
        """Forget the oldest command that has ended; False when none has."""
        for command_id, result in self._by_id.items():
            if result.status != ACCEPTED:
                del self._by_id[command_id]
                return True
        return False


class _JawReply:
    """A jaw command's on_done: its final RESULT never passes its "accepted"."""

    def __init__(self, reply: _Reply, mode: str) -> None:
        self._reply = reply
        self._mode = mode
        self._lock = threading.Lock()
        self._queued = False
        self._early: GripperOutcome | None = None

    def __call__(self, outcome: GripperOutcome) -> None:
        with self._lock:
            if not self._queued:
                self._early = outcome  # reported by queued()
                return
        self._final(outcome)

    def queued(self, refusal: str | None) -> None:
        """Report "accepted" unless the gripper worker refused the command."""
        with self._lock:
            if refusal is None:
                self._reply.accepted()
            self._queued = True
            early = self._early
        if early is not None:
            self._final(early)
        elif refusal is not None:
            self._reply.final(REFUSED, refusal)

    def _final(self, outcome: GripperOutcome) -> None:
        """The RESULT of a jaw command: the driver's outcome goes into its detail.

        A fault of the driver's that leaves the arm usable ("gripper_fault")
        is "failed" with the driver's own text as the reason.
        """
        status = outcome.status
        if status in ("reached", "sent"):
            result = COMPLETED
        elif status in ("stalled", "timeout") and self._mode in _CLOSES:
            result = COMPLETED  # stalling on the object is what was asked for
        elif status in ("cancelled", "refused"):
            result = status
        else:  # stalled on a plain command, a fault of the jaws, failed
            result = FAILED
        reason = outcome.detail
        if not reason and result == FAILED:
            reason = _JAW_REASON.get(status, f"the jaw command ended {status}")
        detail: dict[str, Any] = {"outcome": status, "opening": outcome.opening}
        skipped = getattr(outcome, "stages_skipped", None)
        if skipped is not None:
            detail["close_stages_skipped"] = skipped
        closed = getattr(outcome, "closed", None)
        if closed is not None:
            detail["closed"] = closed
        self._reply.final(result, reason, None, detail)


def _pose(readings: Sequence[JointReading], n_joints: int) -> list[float]:
    """The pose of one full read; raises ReadFailed unless every joint is valid."""
    pose: list[float | None] = [None] * n_joints
    for reading in readings:
        usable = reading.valid and math.isfinite(reading.value)
        if usable and 0 <= reading.joint < n_joints:
            pose[reading.joint] = float(reading.value)
    for joint, value in enumerate(pose):
        if value is None:
            raise ReadFailed(f"joint index {joint} has no valid reading")
    return pose


def _summary(result: Result) -> dict[str, Any]:
    """A final RESULT as ROBOT_STATE's last_result; a long reason is cut."""
    return {
        "command_id": result.command_id,
        "status": result.status,
        "reason": _cut(result.reason),
    }


def _cut(text: str) -> str:
    """A reason at the length that every RESULT, EVENT and ROBOT_STATE can carry."""
    return text if len(text) <= MAX_OUT_CHARS else printable(text, MAX_OUT_CHARS)


def _declared(driver: str) -> str:
    """How a refusal names what WELCOME declares the arm to be."""
    return f"driver {driver}" if driver else "no driver"


def _span_s(value: float | None) -> float | None:
    """A time span in seconds; None unless it is a finite number of 0 or more."""
    if value is None or isinstance(value, bool):
        return None
    try:
        span = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return span if math.isfinite(span) and span >= 0.0 else None


def _age_on_arrival(
    age_s: float | None, sent_mono_s: float | None, arrived_mono: float
) -> float | None:
    """How old a datum was when its link message arrived, never less than it is.

    age_s is the age its sender measured as it sent. sent_mono_s is that
    moment on this app's monotonic clock as the sender read it off a
    ROBOT_STATE: at or before the true one. The time from then to
    arrived_mono, the way to the robot app and the wait there, is added. None
    when either is unknown or no finite number of 0 or more, as with a
    sent_mono_s that lies after arrived_mono.
    """
    age = _span_s(age_s)
    if age is None or sent_mono_s is None:
        return None
    on_the_way = _span_s(arrived_mono - sent_mono_s)
    return None if on_the_way is None else age + on_the_way


def no_live_view_notice(config: RobotAppConfig) -> str | None:
    """NO_LIVE_VIEW when the file lets a person drive without a live view, else None.

    Any driver takes such a file, the real one included: nothing then shows
    the robot app that the person sees the arm.
    """
    human = config.human
    unseen = human.enabled and not human.require_live_view
    return NO_LIVE_VIEW if unseen else None


def human_target(m: HumanSetpoint, arrived_mono: float) -> HumanTarget:
    """The human target a HUMAN_SETPOINT carries; the only place one is read.

    arrived_mono is the app's monotonic clock when the app took the link
    message up. The sample age is the one the setpoint reports plus the time
    since it was sent (_age_on_arrival); None when that is not known. The
    round trip is the time since the ROBOT_STATE the setpoint was computed on,
    None when it names none or the time is not a finite number of 0 or more.
    """
    round_trip_s = None
    if m.state_clock_mono_s is not None:
        round_trip_s = _span_s(arrived_mono - m.state_clock_mono_s)
    # A human program that does not say when it sent: it sent after the
    # ROBOT_STATE it computed on was made, so the age counts from there.
    sent_mono_s = m.state_clock_mono_s if m.sent_mono_s is None else m.sent_mono_s
    return HumanTarget(
        seq=m.seq,
        owner_number=m.owner_number,
        stop_number=m.stop_number,
        source=m.source,
        stage=m.stage,
        note=m.note,
        engaged=bool(m.engaged),
        grip_released=bool(m.grip_released),
        q=None if m.q is None else tuple(m.q),
        sample_age_s=_age_on_arrival(m.sample_age_s, sent_mono_s, arrived_mono),
        arrived_mono=arrived_mono,
        round_trip_s=round_trip_s,
        jaw_opening=m.jaw_opening,
    )


class RobotApp:
    """The one program that talks to the motors.

    A dispatcher thread receives link messages and runs the supervisor step; a
    motion worker runs moves, human drives, returns to rest and releases; the
    gripper worker runs jaw commands. STOP, fault-stop, the assistance wait's
    step and every start of motion work happen under one reentrant lock, so a
    STOP always wins.
    """

    def __init__(
        self,
        config: RobotAppConfig,
        arm: ArmPort,
        checkers: CheckerSet,
        carrier: Carrier | None,
        clock: Clock,
        *,
        confirm: Callable[[str], bool] | None = None,
        step_period_s: float = 0.01,
        driver: str = "",
        allow_reference: bool = False,
        rest_route: rest_module.RestRoute | None = None,
        rebuild_arm: Callable[[], ArmPort] | None = None,
    ) -> None:
        """confirm(prompt) is the Thor terminal's typed confirmation.

        rest_route is the route of the functional rest; None: a return to
        rest is one straight move. rebuild_arm() makes a new arm port for the
        motors to be switched on again after a release. With it a typed rest
        ends as the functional rest does: the jaws closed, the motors off.
        None: nothing could switch the motors on again without a new start,
        so a typed rest ends at rest with the motors on.
        None means "never confirmed". step_period_s is the longest the
        dispatcher waits for a link message between two supervisor steps.
        carrier None starts the app with no robot connection: it holds and
        serves the Thor terminal until attach_carrier() gives it one. driver
        is what WELCOME declares the arm to be: one of DRIVERS (ValueError
        otherwise). allow_reference is the setting jaws.allow_reference: False
        refuses request_reference().
        """
        if driver not in DRIVERS:
            names = ", ".join(repr(name) for name in DRIVERS)
            raise ValueError(f"driver must be one of {names}, got {driver!r}")
        self._driver = driver
        self._allow_reference = bool(allow_reference)
        if arm.n_joints != config.n_joints:
            raise ValueError(
                f"the arm has {arm.n_joints} joints but the configuration "
                f"{config.n_joints}"
            )
        if not (math.isfinite(step_period_s) and step_period_s > 0.0):
            raise ValueError(f"step_period_s must be above 0, got {step_period_s!r}")
        self._config = config
        self._rest_powers_off = rebuild_arm is not None
        if rebuild_arm is not None:
            arm = rest_module.SwitchedArm(arm, rebuild_arm)
        self._arm = arm
        self._checkers = checkers
        # Swapped only by attach_carrier(), under the app's lock.
        self._carrier: Carrier = _NoCarrier() if carrier is None else carrier
        self._clock = clock
        self._confirm = confirm
        self._step_period_s = float(step_period_s)

        # Serialises STOP, fault-stop, the wait's step and the start of every
        # motion job. Taken before any other lock of the app, never after one.
        self._lock = threading.RLock()
        self._motion_cond = threading.Condition()
        self._jobs: deque[_Job] = deque()
        self._running: _Job | None = None
        self._reporting = False  # a finished job's final RESULT is going out
        self._event_lock = threading.RLock()
        self._event_seq = 0
        self._listeners: list[EventListener] = []
        self._results = _Results()
        self._last_result: dict[str, Any] | None = None
        self._local_ids = 0
        # STOPs asked for by a signal handler: appended without a lock.
        self._stop_requests: deque[tuple[str, str]] = deque()

        self._task_timed_return = True  # of the agent's current task
        self._grasp_region: GraspRegion | None = None  # of the agent's last grasp
        self._contact_hold = False  # the last move ended at a contact
        # True while no carrier is attached, and once the loss of the attached
        # one was handled. Written under the app's lock.
        self._link_lost = carrier is None
        self._state_forced = False  # a ROBOT_STATE is owed to a new carrier
        self._jaws_due = 0.0  # app-clock time of the next jaw reading in a hold
        self._jaw_peek_misses = 0  # jaw peeks in a row without a reading
        # How long each idle joint read since the connect took, in s, until
        # READ_TIMING_READS of them were told; None from then on. Written by
        # the dispatcher, and by the thread that connects before it reads.
        self._read_times: list[float] | None = None
        # The newest VIEW_STATUS of this robot connection and the app-clock
        # time it arrived; one value, so a reader never sees half of a change.
        self._view: tuple[ViewStatus, float] | None = None
        # The four below are written under the app's lock.
        self._takeover_open = False  # a takeover was accepted, no hand-back yet
        # The owner number at which a released grip was noted. While it is not
        # the current one, the next human drive needs a released grip first:
        # the needs-release flag, which every change of the control owner sets.
        # Controller data that were away set it too, and so does an engaged
        # target that arrives while other work runs.
        self._release_seen: int | None = None
        self._drive_ended = ""  # why the last human drive ended; "" while one runs
        self._human_refusal = ""  # why the newest engaged human target moves nothing
        self._human_input = HumanInput()
        self._permit_refusal: str | None = None
        self._dropped: set[str] = set()  # kinds of link messages not taken
        self._inbox = Inbox()  # taken off the link, not yet handled
        # A human program on the Thor, the program that turns a person's
        # controller data into joint targets: its carrier (None: there is no
        # such socket), what was taken off it and not yet handled, and
        # whether such a program is attached now. See attach_local().
        self._local: Carrier | None = None
        self._local_inbox = Inbox()
        self._local_attached: Callable[[], bool] = lambda: False
        # The local carrier while the dispatcher handles a link message that
        # arrived on it; None for one of the robot connection.
        self._arrived_on: Carrier | None = None
        # Of the newest HUMAN_SETPOINT taken: whether it arrived on the local
        # carrier, and when. When a second human program's was last dropped;
        # None: none was since the followed end last changed.
        self._human_from: tuple[bool, float] | None = None
        self._second_at: float | None = None
        self._lines = LineLimit(clock)  # for log lines a link can make repeat
        longest_move = config.executor.max_setpoints * (
            config.n_joints * MOVE_VALUE_BYTES + len("[],")
        )
        self._payload_limits = {Move.TYPE: MOVE_BASE_BYTES + longest_move}
        self._fault_while_latched = ""
        self._errors: dict[str, tuple[str, int]] = {}
        self._published: tuple[Any, ...] | None = None

        self._connect_called = False
        self._connect_ended = False  # the arm's connect() has returned
        self._started = False
        self._closing = threading.Event()
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._pass_cond = threading.Condition()
        self._passes_started = 0
        self._passes_finished = 0

        self._motion = _MotionBox()
        self._authority = Authority()
        self._history = JointHistory(config.n_joints, config.state.history_s)
        self._gate = PermitGate(self._authority, checkers.agent, config.permits, clock)
        self._executor = MoveExecutor(
            arm,
            self._authority,
            self._history,
            config.executor,
            clock,
            first_wait_s=config.permits.first_wait_s,
            on_missed_deadline=lambda: self._publisher.note_missed_deadline(),
            reads_paused=lambda: self._idle.paused(),
            # How many missed motor answers the arm's own driver starts a
            # move over; an arm that names none gets one read.
            start_read_tries=int(getattr(arm, "start_read_tries", 1)),
            # The longest pause between two setpoints the arm's own driver
            # goes on after; an arm that names none gets large_delay_s.
            pause_limit_s=getattr(arm, "move_pause_limit_s", None),
        )
        self._human_drive = HumanDrive(
            arm,
            self._authority,
            checkers.human,
            self._history,
            config,
            clock,
            self._human_input,
            self.view_age_s,
            claim_idle=self._executor.claim_idle,
            note_clamped=lambda: self._publisher.note_clamped(),
            reads_paused=lambda: self._idle.paused(),
            last_grasp_region=lambda: self._grasp_region,
        )
        self._jaw_state = JawState(config.gripper, clock)
        self._worker = GripperWorker(arm, self._jaw_state, self._authority, clock)
        # The jaws under a person's trigger; asked only with human.trigger_jaws.
        self._jaw_follower = JawFollower(
            arm,
            self._jaw_state,
            self._worker,
            self._authority,
            config.human.jaw_match_tol,
            lambda line: self._note(logging.INFO, line, kind=_NUMBER.sub("#", line)),
        )
        self._wait = AssistanceWait(config.waits, clock, self._on_expire)
        self._stops = StopController(
            self._authority,
            checkers,
            self._worker,
            self._wait,
            self._emit,
            self._stop_motion,
        )
        self._worker.on_fault = self._fault
        self._rest = ReturnToRest(
            arm,
            self._authority,
            checkers,
            self._executor,
            self._jaw_state,
            self._history,
            config,
            clock,
            self._emit,
            self._motion,
            lambda: self._grasp_region,
            # While a takeover is open no agent move can end a contact hold, so
            # a return the operator confirmed may start from one.
            lambda: self._contact_hold and not self.takeover_open,
            # rest.py asks the executor itself; a human drive is a move too.
            lambda: self._human_drive.progress() is not None,
            jaws_busy=self._worker.busy,
            route=rest_route,
            close_jaws=self._close_jaws_at_rest,
            reads_paused=lambda: self._idle.paused(),
        )
        self._idle = IdleReader(arm, self._history, config.state, clock)
        self._publisher = StatePublisher(
            arm,
            self._history,
            self._authority.snapshot,
            # A human drive's progress while one runs, else the executor's.
            lambda: self._human_drive.progress() or self._executor.progress(),
            self._motion.get,
            lambda: self._jaw_state.gripper_state(arm),
            lambda: self._wait.remaining(clock.monotonic()),
            lambda: self._last_result,
            self._authority.session_id,
            config,
            clock,
            read_failures=self._idle.failures,
            move_stage=self._executor.stage,
            view_age=self.view_age_s,
            permit_facts=self._gate.held_facts,
            permit_refusal=self._gate.last_refusal,
            human_stage=lambda now: self._human_input.stage(
                now, config.human.stage_keep_s
            ),
            controller_age=self._human_input.sample_age_s,
            human_facts=self._human_facts,
            contact_hold=lambda: self._contact_hold,
            local_program=self._local_program,
        )
        self._handlers: dict[type, Callable[[Any], None]] = {
            Move: self._on_move,
            Permit: self._on_permit,
            Gripper: self._on_gripper,
            CheckerState: self._on_checker_state,
            Stop: self._on_stop,
            OwnerRequest: self._on_owner_request,
            Reset: self._on_reset,
            Release: self._on_release,
            Open: self._on_open,
            Shutdown: self._on_shutdown,
            ViewStatus: self._on_view_status,
            HumanSetpoint: self._on_human_setpoint,
        }
        # Type number -> class of the link messages the app takes.
        self._takes = {cls.TYPE: cls for cls in self._handlers}
        self._supervisor_parts: tuple[tuple[str, Callable[[], None]], ...] = (
            ("the assistance wait", self._step_wait),
            ("the idle joint read", self._step_idle),
            ("the fault poll", self._poll_faults),
            ("the robot connection check", self._check_link),
            ("the jaws under a person's trigger", self._watch_jaws),
            ("the state publisher", self._publish),
            ("the count of repeated log lines", self._flush_lines),
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="robot-app-dispatcher", daemon=True
        )
        self._motion_worker = threading.Thread(
            target=self._motion_loop, name="robot-app-motion", daemon=True
        )

    # Life cycle

    @property
    def session_id(self) -> str:
        return self._authority.session_id

    @property
    def running(self) -> bool:
        """True from start() until the app begins to close."""
        return self._started and not self._closing.is_set()

    @property
    def link_alive(self) -> bool:
        """Whether the robot connection is alive."""
        return bool(self._carrier.alive)

    @property
    def wants_carrier(self) -> bool:
        """Whether attach_carrier() would take a live carrier now.

        True while the app runs and has no carrier, or has handled the loss of
        the one it had.
        """
        return self.running and self._link_lost

    def attach_carrier(self, carrier: Carrier) -> bool:
        """Take a new robot connection; False, and nothing changes, if not taken.

        Taken while wants_carrier holds and the carrier is alive; otherwise the
        caller closes it. The app sends WELCOME with its session id, the event
        "connection" and, from the dispatcher, a ROBOT_STATE. It starts no
        motion and no wait, and sends nothing that was meant for an earlier
        connection. Safe from any thread; it does not touch the arm.
        """
        with self._lock:
            if not (self.wants_carrier and carrier.alive):
                return False
            # The carrier before the flag: the dispatcher reads them unlocked.
            self._carrier = carrier
            self._link_lost = False
            self._send(self._welcome())
            self._state_forced = True
            self._emit(EVENT_CONNECTION, LINK_CONNECTED, {"connected": True})
        _log.info("robot connection made")
        return True

    def attach_local(self, carrier: Carrier, attached: Callable[[], bool]) -> None:
        """Take the carrier of a human program on the Thor; once.

        What that program sends on `carrier` is handled as when it arrives on
        the robot connection, by the same handlers, and also while the robot
        connection is down. The carrier gets WELCOME, every ROBOT_STATE and
        the RESULTs of what arrived on it; the robot connection gets those
        RESULTs too. attached() says whether a human program is attached
        there now; it must not block. The carrier's sends must not block
        either. Safe before and after start(); it does not touch the arm.
        """
        with self._lock:
            if self._local is not None:
                raise RuntimeError("a local carrier was attached before")
            self._local_attached = attached
            self._local = carrier
            self._send(self._welcome(), via=carrier)
            self._state_forced = True  # the dispatcher sends a ROBOT_STATE

    @property
    def motion_state(self) -> MotionState:
        return self._motion.get()

    def snapshot(self) -> AuthorityState:
        """Session id, control owner, owner number, stop number and stop level."""
        return self._authority.snapshot()

    @property
    def takeover_open(self) -> bool:
        """Whether a takeover was accepted and no hand-back has ended it yet.

        A STOP and a lost robot connection leave it open. While it is open no
        agent task is taken.
        """
        return self._takeover_open

    def start(self) -> None:
        """Connect the arm, once, send WELCOME and start the threads.

        The jaws are referenced when the configuration asks for it and the arm
        is measured at rest within rest.start_rest_tol_rad. The motion state is
        "rest_hold" only within rest.rest_tol_rad, where a release is accepted.
        """
        if self._connect_called:
            raise RuntimeError("start() was already called; the arm connects once")
        self._connect_called = True
        self._arm.connect()
        self._connect_ended = True
        state = self._after_connect("at start")
        self._send(self._welcome())
        self._dispatcher.start()
        self._motion_worker.start()
        self._started = True
        _log.info(
            "robot app started: session %s, motion state %s",
            self._authority.session_id,
            state,
        )
        notice = no_live_view_notice(self._config)
        if notice is not None:
            _log.info(notice)

    def _after_connect(self, when: str) -> MotionState:
        """What follows the arm's connect, at start and when the motors are
        switched on again: the pose, the motion state and the jaw reference.
        """
        rest = self._config.rest
        at_rest = releasable = False
        self._read_times = []  # the joint reads of this connect are timed anew
        try:
            off_rest = self._off_rest(self._read_pose())
            at_rest = off_rest[1] <= rest.start_rest_tol_rad
            releasable = off_rest[1] <= rest.rest_tol_rad
            if at_rest and not releasable:
                _log.warning(
                    "the arm lies at rest %s (within %g rad) but joint index "
                    "%d is %.3f rad from the rest pose, outside the %g rad of a "
                    "release: the motion state is hold until a typed rest",
                    when,
                    rest.start_rest_tol_rad,
                    *off_rest,
                    rest.rest_tol_rad,
                )
        except ReadFailed as failed:
            _log.warning("the pose could not be read %s: %s", when, failed)
        if not self._arm.torque_on():
            _log.error("the motors are off after connect()")
            state = MotionState.REST_UNPOWERED if at_rest else MotionState.HOLD
        else:
            state = MotionState.REST_HOLD if releasable else MotionState.HOLD
        self._motion.set(state)
        if self._config.gripper.reference_at_start and at_rest:
            try:
                self._arm.gripper_reference()
            except ArmFault as fault:
                self._fault(f"the jaw reference failed: {fault}")
        # One classification, so ROBOT_STATE has a jaw reading from the start.
        self._jaw_state.classify(self._arm)
        return state

    def close(self) -> None:
        """Stop the threads and the gripper worker; the motors stay on and hold.

        Running motion work ends in a hold and queued work is cancelled. Torque
        is not released here: only a release or a shutdown at rest does that.
        """
        current = threading.current_thread()
        own = current in (self._dispatcher, self._motion_worker)
        # One of the app's own threads never waits for another closer: that
        # closer is waiting for this thread to end.
        if not self._close_lock.acquire(blocking=not own):
            return
        try:
            if self._closed.is_set():
                return
            self._closing.set()
            with self._lock:
                self._wait.cancel(CLOSING)
                self._stop_motion(CLOSING)
            with self._motion_cond:
                self._motion_cond.notify_all()  # the idle motion worker ends
            for thread in (self._dispatcher, self._motion_worker):
                if thread is current or not thread.is_alive():
                    continue
                thread.join(_JOIN_S)
                if thread.is_alive():
                    _log.warning("%s has not ended", thread.name)
            self._worker.close()
            self._closed.set()
        finally:
            self._close_lock.release()
        if self._connect_called and not self._connect_ended:
            # A connect that raised may have enabled the motors before it did;
            # the arm port then still says that its motors are off.
            _log.info(
                "robot app closed; torque on: not known, the arm's connect did "
                "not end: the motors may be on"
            )
            return
        if self._motion.get() is MotionState.RELEASE_FAILED:
            _log.info(
                "robot app closed; torque on: not known, a torque release failed: %s",
                TORQUE_UNKNOWN,
            )
            return
        _log.info("robot app closed; torque on: %s", self._arm.torque_on())

    def wait_closed(self, timeout: float | None = None) -> bool:
        """Wait in real time until the app has closed; False if it has not."""
        return self._closed.wait(timeout)

    def wait_step(self, timeout_s: float = 2.0) -> bool:
        """Wait in real time for one dispatcher pass that began after this call.

        A pass is the link messages that arrived, as many as one step period
        of the app clock has room for, and one supervisor step.
        """
        with self._pass_cond:
            target = self._passes_started + 1
            return self._pass_cond.wait_for(
                lambda: self._passes_finished >= target, timeout_s
            )

    def wait_motion_idle(self, timeout_s: float = 2.0) -> bool:
        """Wait in real time until no motion work is queued, running or reporting."""

        def idle() -> bool:
            return self._running is None and not self._jobs and not self._reporting

        with self._motion_cond:
            return self._motion_cond.wait_for(idle, timeout_s)

    # Calls for the Thor terminal; they work with the robot connection down.

    def add_event_listener(self, listener: EventListener) -> None:
        """Call listener(kind, text, data) for every operator event.

        It runs on the thread that emits the event, often under the app's lock:
        it must not block and must not call back into the app.
        """
        with self._event_lock:
            self._listeners.append(listener)

    def stop(self, reason: str, sender: str = THOR_TERMINAL) -> int:
        """STOP, as the link message does it; returns the new stop number.

        Not for a signal handler: it takes the app's lock. Use request_stop().
        """
        with self._lock:
            return self._stops.stop(reason, sender)

    def request_stop(self, reason: str, sender: str = THOR_TERMINAL) -> None:
        """Ask for a STOP without taking a lock; safe in a signal handler.

        The dispatcher raises it on its next pass.
        """
        self._stop_requests.append((str(reason), str(sender)))

    def jaws_not_proven(self) -> str | None:
        """Why ROBOT_STATE's jaw reading does not show the jaws empty, or None.

        What the Thor terminal asks the operator about before a rest, a
        release, a quit or a close. It reads nothing.
        """
        return self._jaw_state.not_proven(self._arm)

    def request_open(
        self, confirmed: bool, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue an opening of the jaws typed at the Thor terminal; why not, or None.

        It is the driver's own opening. confirmed says that the operator typed
        the confirmation. Nobody may own the arm. on_done is as for
        request_rest().
        """
        return self._request_jaws("open", OPEN_JAWS, confirmed, on_done)

    def request_close(
        self, confirmed: bool, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue a close of the jaws typed at the Thor terminal; why not, or None.

        It is the driver's own close. confirmed is needed while the jaws are
        not proven empty: the operator typed that they may close on what they
        hold. Nobody may own the arm. on_done is as for request_rest().
        """
        return self._request_jaws("close", CLOSE_JAWS, confirmed, on_done)

    def _request_jaws(
        self, what: str, action: str, confirmed: bool, on_done: DoneFn | None
    ) -> str | None:
        reply = self._local_reply(what, on_done)
        with self._lock:
            state = self._authority.snapshot()
            # The app's own numbers: only the fault-stop and the owner count.
            reason = self._not_running() or self._authority.admit(
                state.session_id, state.owner_number, state.stop_number, Owner.NOBODY
            )
            return self._typed_jaws(reply, action, confirmed, reason)

    def request_rest(
        self,
        confirmed: bool | None,
        on_done: DoneFn | None = None,
        jaws_empty: bool = False,
    ) -> str | None:
        """Queue a return to rest typed at the Thor terminal; why not, or None.

        confirmed says that the operator typed that the path is clear; None asks
        confirm() on this thread. jaws_empty says that the operator typed that
        the jaws are empty: it is taken in place of the proof. on_done gets
        the final RESULT exactly once, also for a refusal; it must not block
        or call back into the app.
        """
        if confirmed is None:
            confirmed = self._ask(PATH_CLEAR_PROMPT)
        reply = self._local_reply("rest", on_done)
        with self._lock:
            reason = self._not_running() or (None if confirmed else CONFIRM_PATH)
            # A return would wait behind a jaw reference and begin as it ends.
            reason = reason or self._busy() or self._referencing()
            if reason is None:
                self._take_jaw_word(jaws_empty, "a return to rest")
                stop_number = self._authority.stop_number
                self._queue(
                    _Job(
                        "a return to rest",
                        lambda job: self._run_rest(
                            job, rest_module.TYPED, stop_number, confirmed=True
                        ),
                        reply,
                        needs_link=False,
                    )
                )
        if reason is not None:
            self._refuse(reply, "rest typed at the Thor terminal", reason)
        return reason

    def request_release(
        self, on_done: DoneFn | None = None, jaws_empty: bool = False
    ) -> str | None:
        """Queue a torque release typed at the Thor terminal; why not, or None.

        The motors go off only at measured rest with the jaws proven empty,
        or empty on the operator's typed word (jaws_empty); otherwise they
        stay on. on_done is as for request_rest().
        """
        reply = self._local_reply("release", on_done)
        with self._lock:
            reason = self._not_running() or self._busy()
            if reason is None:
                self._take_jaw_word(jaws_empty, "a torque release")
                stop_number = self._authority.stop_number
                self._queue(
                    _Job(
                        "a torque release",
                        lambda job: self._run_release(job, stop_number),
                        reply,
                        needs_link=False,
                    )
                )
        if reason is not None:
            self._refuse(reply, "release typed at the Thor terminal", reason)
        return reason

    @property
    def allow_reference(self) -> bool:
        """The setting jaws.allow_reference: whether a jaw reference may be typed."""
        return self._allow_reference

    def request_reference(
        self, confirmed: bool, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue a jaw reference typed at the Thor terminal; why not, or None.

        confirmed says that the operator typed that the jaws are empty. Taken
        only with the setting jaws.allow_reference on, nobody owning the arm,
        no fault-stop, the motors on, no motion work or jaw command running or
        waiting, and the jaws not referenced. A STOP ends it "cancelled". It
        leaves "may be holding" as it was. on_done is as for request_rest().
        """
        reply = self._local_reply("reference", on_done)
        with self._lock:
            state = self._authority.snapshot()
            reason = (
                self._not_running()
                or (None if self._allow_reference else REFERENCE_OFF)
                or (None if confirmed else CONFIRM_JAWS_EMPTY)
                or self._motors_off()
                or self._busy()
                or (JAWS_BUSY if self._worker.busy() else None)
                # The app's own numbers: only the fault-stop and the owner count.
                or self._authority.admit(
                    state.session_id,
                    state.owner_number,
                    state.stop_number,
                    Owner.NOBODY,
                )
                or (ALREADY_REFERENCED if self._arm.gripper_referenced() else None)
            )
            if reason is None:
                done = _JawReply(reply, PLAIN)
                reason = self._worker.reference(done)
                done.queued(reason)
                return reason
        self._refuse(reply, "reference typed at the Thor terminal", reason)
        return reason

    def view_age_s(self, now: float) -> float | None:
        """The age of the operator's live view at the app-clock time `now`.

        The age the newest VIEW_STATUS reported plus the time since it was
        sent (_age_on_arrival), never less than the view's age. None when none
        has arrived on this robot connection, when it said the view is not
        live, or when its age or the time it was sent is not known.
        """
        view = self._view
        if view is None:
            return None
        status, arrived = view
        if not status.live:
            return None
        age = _age_on_arrival(status.age_s, status.sent_mono_s, arrived)
        return None if age is None else age + max(0.0, now - arrived)

    def offer_human_target(self, target: HumanTarget) -> None:
        """Take one human target, whoever made it; safe from any thread.

        It is kept as the newest one. A released grip it shows is noted, and
        an engaged one that passes the admission starts one human drive. Why
        an engaged target moves nothing shows in ROBOT_STATE, in
        health["human"]["refusal"]. A human drive that is queued or runs ends
        in a hold when the controller data were away before this target, or
        when the admission refuses it engaged. The jaw opening its trigger
        asks for goes to _follow_trigger(), engaged or not.
        """
        with self._lock:
            now = self._clock.monotonic()
            # Controller data that come back after a gap are a new start: a
            # grip squeezed meanwhile moves nothing until it was released, and
            # no human drive runs on over the gap.
            if self._controller_problem(now) is not None:
                self._release_seen = None
                self._end_human_drive(DATA_AWAY)
                self._jaw_follower.end(DATA_AWAY)
            self._human_input.put(target)
            state = self._authority.snapshot()
            # Why the target is not a person's who owns the arm now.
            not_owner = self._authority.admit(
                state.session_id, target.owner_number, target.stop_number, Owner.HUMAN
            )
            self._follow_trigger(target, not_owner, now)
            if not target.engaged:
                self._human_refusal = ""
                fresh = self._controller_problem(now) is None
                if target.grip_released and fresh and not_owner is None:
                    self._release_seen = state.owner_number
                return
            # A squeeze that began while other work ran moves nothing once that
            # work has ended: a released grip comes first.
            other_work = self._busy(_no_human_drive) or self._referencing()
            if other_work is not None:
                self._release_seen = None
            reason = (
                self._human_off()
                or self._source_problem()
                or not_owner
                or self._motors_off()
                or other_work
            )
            if self._busy(_a_human_drive) is not None:
                if reason is None:
                    self._human_refusal = ""  # the drive under way follows it
                    return
                # No drive follows a target the admission refuses.
                self._release_seen = None
                self._end_human_drive(f"{TARGET_REFUSED}{reason}")
            # A drive starts on the controller data a takeover needs.
            reason = reason or self._controller_problem(now)
            if reason is None and self._release_seen != state.owner_number:
                reason = NEEDS_RELEASE
            reason = reason or self._shape_problem(target)
            if reason is not None:
                if reason != self._human_refusal:
                    line = f"an engaged human target moves nothing: {reason}"
                    # A sample age that differs each time makes no new kind.
                    self._note(logging.INFO, line, kind=_NUMBER.sub("#", line))
                self._human_refusal = reason
                return
            self._human_refusal = ""
            # The numbers the target quoted and admit() found current: the
            # drive ends as soon as one of them changes.
            admitted = AuthorityState(
                state.session_id, Owner.HUMAN, target.owner_number, target.stop_number
            )
            self._queue(
                _Job(
                    "human driving",
                    lambda job: self._run_human_drive(job, admitted),
                    None,
                    needs_link=True,
                    is_human_drive=True,
                )
            )

    def _follow_trigger(
        self, target: HumanTarget, not_owner: str | None, now: float
    ) -> None:
        """The jaw part of a human target: the opening its trigger asks for.

        Under the app's lock, with human.trigger_jaws. not_owner is why the
        target is not a person's who owns the arm now. The jaws follow the
        trigger of a target that passes the checks of an engaged one, the grip
        squeezed or not; a close also needs the live view fresh where the file
        requires it. A target that fails them ends the jaw motion in a light
        hold, and so does one whose opening is no number from 0 to 1. Why the
        trigger moves nothing shows in ROBOT_STATE, in
        health["human"]["jaws"]["refusal"].
        """
        if not self._config.human.trigger_jaws:
            return
        reason = (
            self._human_off()
            or self._source_problem()
            or not_owner
            or self._motors_off()
            # A return to rest or a hand-back proves the jaws empty: no jaw
            # step may follow that proof.
            or self._busy(_no_human_drive)
            or self._referencing()
            or self._controller_problem(now)
        )
        if reason is not None:
            self._jaw_follower.end(reason)
            return
        self._jaw_follower.offer(
            target.jaw_opening,
            target.owner_number,
            target.stop_number,
            no_close=self._view_problem(now),
        )

    def _watch_jaws(self) -> None:
        """End the jaw motion of a trigger whose human targets no longer count.

        No human target arrives to end it when the controller data are away,
        or after a STOP or an owner change.
        """
        if not self._jaw_follower.following:
            return
        with self._lock:
            state = self._authority.snapshot()
            # The app's own numbers: only the fault-stop and the owner count.
            reason = self._authority.admit(
                state.session_id, state.owner_number, state.stop_number, Owner.HUMAN
            ) or self._controller_problem(self._clock.monotonic())
            if reason is not None:
                self._jaw_follower.end(reason)

    def shutdown_moves_arm(self) -> bool:
        """Whether a shutdown asked for now would first move the arm to rest.

        By the motion state: False while the arm holds at rest or the motors
        are off, and after a torque release that failed, when nothing moves
        the arm. Such a shutdown needs the confirmation that the path is clear.
        """
        unmoved = (
            MotionState.REST_HOLD,
            MotionState.REST_UNPOWERED,
            MotionState.RELEASE_FAILED,
        )
        return self._motion.get() not in unmoved

    def request_shutdown(
        self,
        confirmed: bool | None,
        on_done: DoneFn | None = None,
        jaws_empty: bool = False,
    ) -> str | None:
        """Queue a shutdown typed at the Thor terminal; why not, or None.

        The arm returns to rest, torque is released there and the app closes. If
        the return or the release is refused the app keeps running and holding.
        confirmed, jaws_empty and on_done are as for request_rest(); the
        confirmation is needed, and None asks for it, only when the arm has
        to move.
        """
        if confirmed is None:
            confirmed = self.shutdown_moves_arm() and self._ask(PATH_CLEAR_PROMPT)
        reply = self._local_reply("shutdown", on_done)
        with self._lock:
            reason = self._not_running() or self._shutdown_unconfirmed(confirmed)
            reason = reason or self._busy()
            if reason is None:
                self._take_jaw_word(jaws_empty, "a shutdown")
                stop_number = self._authority.stop_number
                self._queue(
                    _Job(
                        "a shutdown",
                        lambda job: self._run_shutdown(job, stop_number, confirmed),
                        reply,
                        needs_link=False,
                    )
                )
        if reason is not None:
            self._refuse(reply, "shutdown typed at the Thor terminal", reason)
        return reason

    @property
    def motors_off_at_rest(self) -> bool:
        """Whether a release at rest switched the motors off: `on` is the way on."""
        return self._motion.get() is MotionState.REST_UNPOWERED

    def request_power_on(
        self, confirmed: bool | None, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue "motors on again" typed at the Thor terminal; why not, or None.

        Taken only after a release at rest. confirmed says that the operator
        typed yes to POWER_ON_PROMPT; None asks confirm() on this thread.
        on_done is as for request_rest().
        """
        if confirmed is None:
            confirmed = self._ask(POWER_ON_PROMPT)
        reply = self._local_reply("on", on_done)
        with self._lock:
            reason = self._power_on_refusal(confirmed)
            if reason is None:
                self._queue(
                    _Job("motors on again", self._run_power_on, reply, needs_link=False)
                )
        if reason is not None:
            self._refuse(reply, "on typed at the Thor terminal", reason)
        return reason

    def _power_on_refusal(self, confirmed: bool) -> str | None:
        """Why the motors are not switched on again; under the app's lock."""
        reason = self._not_running() or (None if confirmed else CONFIRM_POWER_ON)
        if reason is None and self._motion.get() is MotionState.RELEASE_FAILED:
            reason = MOTORS_UNKNOWN
        if reason is None and not self.motors_off_at_rest:
            reason = rest_module.MOTORS_ARE_ON
        state = self._authority.snapshot()
        if reason is None and state.stop_level is StopLevel.FAULT:
            why = f": {state.fault_reason}" if state.fault_reason else ""
            reason = f"fault-stop latched{why}"
        return reason or self._busy() or self._referencing()

    def request_reset(self, on_done: DoneFn | None = None) -> str | None:
        """Clear a latched fault-stop, typed at the Thor terminal; why not, or None.

        As the console's reset: it clears the latch and starts nothing.
        on_done is as for request_rest().
        """
        reply = self._local_reply("reset", on_done)
        with self._lock:
            state = self._authority.snapshot()
            reason = self._not_running() or self._stops.reset(
                state.stop_number, self._arm, state.owner_number
            )
        if reason is not None:
            self._refuse(reply, "reset typed at the Thor terminal", reason)
        else:
            reply.final(COMPLETED)
        return reason

    def end_at_rest(self, timeout_s: float = 30.0) -> str | None:
        """What SIGTERM asks before the app closes; why the motors stay on, or None.

        The end of the functional rest where the arm lies: at measured rest
        the jaws are closed and the motors go off; away from it nothing moves
        and the motors stay on. It waits in real time for the work to end.
        """
        results: list[Result] = []
        reason = self.request_release(results.append)
        if reason is not None:
            return reason
        self.wait_motion_idle(timeout_s)
        if not results:
            return f"the release did not end within {timeout_s:g} s"
        result = results[0]
        return None if result.status == COMPLETED else result.reason

    # Events and RESULTs

    def _emit(self, kind: str, text: str, data: dict[str, Any] | None = None) -> None:
        """Send one EVENT and tell the local listeners; never raises."""
        data = dict(data or {})
        with self._event_lock:
            self._event_seq += 1
            seq = self._event_seq
            self._send(
                Event(seq, kind, text, data),
                lambda: Event(seq, kind, text),
                lambda: Event(seq, kind, _cut(text)),
            )
            for listener in list(self._listeners):
                try:
                    listener(kind, text, data)
                except Exception:
                    _log.exception("an event listener raised")

    def _send(
        self,
        message: Message,
        *plainer: Callable[[], Message],
        via: Carrier | None = None,
    ) -> None:
        """Send a link message; if it cannot be encoded, what `plainer` builds.

        Each of `plainer` is tried in turn until one can be sent. It goes over
        the attached carrier, or over `via`.
        """
        try:
            send(self._carrier if via is None else via, message)
        except ProtocolError as error:
            name = type(message).__name__
            # A byte count that differs each time does not make the line new.
            kind = f"a {name} was not sent: {_NUMBER.sub('#', str(error))}"
            self._note(logging.ERROR, f"a {name} was not sent: {error}", kind=kind)
            if plainer:
                self._send(plainer[0](), *plainer[1:], via=via)

    def _deliver(self, result: Result, via: Carrier | None = None) -> None:
        """Send a RESULT: whole, or without its detail, or with its reason cut."""

        def plain() -> Result:
            return dataclasses.replace(result, end_q=None, detail={})

        def cut() -> Result:
            return dataclasses.replace(plain(), reason=_cut(result.reason))

        to = self._reply_carrier() if via is None else via
        self._send(result, plain, cut, via=to)

    def _reply_carrier(self) -> Carrier:
        """Where the RESULTs of the link message being handled go.

        The robot connection. For a link message of the human program on the
        Thor: its carrier, and the robot connection for the operator console.
        """
        local = self._arrived_on
        return self._carrier if local is None else _Both(local, self._carrier)

    def _local_program(self) -> bool | None:
        """Whether a human program is attached at this app's own socket.

        None without such a socket: ROBOT_STATE then says nothing of it.
        """
        return None if self._local is None else bool(self._local_attached())

    def _send_result(self, result: Result, via: Carrier | None = None) -> None:
        """Store and send a RESULT; a command's second final one is dropped."""

        def deliver(stored: Result) -> None:
            self._deliver(stored, via)

        if not self._results.put(result, deliver):
            if result.status != ACCEPTED:
                _log.error(
                    "command %s already has its final RESULT; dropped %s (%s)",
                    result.command_id,
                    result.status,
                    result.reason,
                )
            return
        if result.status != ACCEPTED:
            self._last_result = _summary(result)

    def _link_reply(self, command_id: str) -> _Reply:
        """Where a link command's RESULTs go: the carrier it arrived on.

        A RESULT that becomes final after that carrier was lost is stored and
        not sent: a later connection gets it only for the same command id.
        """
        arrived_on = self._reply_carrier()

        def deliver(result: Result) -> None:
            self._send_result(result, arrived_on)

        return _Reply(command_id, deliver, announces=True)

    def _local_reply(self, what: str, on_done: DoneFn | None) -> _Reply:
        def deliver(result: Result) -> None:
            self._last_result = _summary(result)
            if on_done is None:
                return
            try:
                on_done(result)
            except Exception:
                _log.exception("the on_done of a Thor terminal request raised")

        with self._event_lock:
            self._local_ids += 1
            command_id = f"thor-terminal-{what}-{self._local_ids}"
        return _Reply(command_id, deliver, announces=False)

    def _refuse(self, reply: _Reply, what: str, reason: str) -> None:
        """Refuse an operator's command and say why at both terminals."""
        self._emit(EVENT_REFUSED, f"{what} refused: {reason}", {"reason": reason})
        reply.final(REFUSED, reason)

    def _ask(self, prompt: str) -> bool:
        """The Thor terminal's typed confirmation; False when there is none."""
        if self._confirm is None:
            return False
        try:
            return bool(self._confirm(prompt))
        except Exception:
            _log.exception("the Thor terminal's confirmation raised")
            return False

    # The dispatcher thread

    def _dispatch_loop(self) -> None:
        while not self._closing.is_set():
            with self._pass_cond:
                self._passes_started += 1
            try:
                if self._link_lost:
                    # A dead link stays dead: only the pace of the steps is kept.
                    self._closing.wait(self._link_wait_s())
                else:
                    self._serve_link()
            except Exception as error:
                self._report("serving the link", error)
                # An error that comes at once must not spin the thread.
                self._closing.wait(self._step_period_s)
            self._guarded("the local link messages", self._serve_local)
            self._guarded("a STOP request", self._serve_stop_requests)
            self._supervisor_step()
            with self._pass_cond:
                self._passes_finished = self._passes_started
                self._pass_cond.notify_all()

    def _serve_link(self) -> None:
        """Handle the link messages that arrived, for up to one step period.

        The link is asked again after each one handled, and a STOP is served
        the moment it is taken off: it waits for the one link message being
        handled and for nothing that merely arrived before it.
        """
        until = self._clock.monotonic() + self._step_period_s
        wait_s = 0.0 if len(self._inbox) else self._link_wait_s()
        handled = 0
        while self._take_from_link(wait_s, until) and not self._closing.is_set():
            if handled and self._clock.monotonic() >= until:
                return  # the supervisor step is due; the rest waits in the inbox
            waiting = self._inbox.take()
            if waiting is None:
                return
            self._guarded("a link message", self._handle_payload, *waiting)
            handled += 1
            wait_s = 0.0

    def _link_wait_s(self) -> float:
        """How long a pass may wait for a link message: at most one step period.

        Less when an idle joint read falls due sooner, so the idle reads keep
        their own period whatever this wait and a read take.
        """
        due_in = self._idle.due_in(self._clock.monotonic())
        return (
            self._step_period_s if due_in is None else min(self._step_period_s, due_in)
        )

    def _take_from_link(self, wait_s: float, until: float) -> bool:
        """Take off the link what waits there; False when the link raised.

        It goes on until the link is dry or the app clock reads `until`, so a
        link that never runs dry cannot keep the supervisor step from running.
        """
        taken = 0
        try:
            received = self._carrier.receive(wait_s)
            while received is not None:
                self._guarded("a link message", self._sort, *received)
                taken += 1
                # The clock is read after a STOP, which takes time to serve, and
                # otherwise once per _TAKEN_PER_CHECK link messages.
                if (received[0] == Stop.TYPE or taken % _TAKEN_PER_CHECK == 0) and (
                    self._closing.is_set() or self._clock.monotonic() >= until
                ):
                    break
                received = self._carrier.receive(0.0)
        except Exception as error:
            self._report("receiving a link message", error)
            # A receive that fails at once must not spin the thread.
            self._closing.wait(self._step_period_s)
            return False
        return True

    def _sort(self, msg_type: int, payload: bytes, inbox: Inbox | None = None) -> None:
        """One link message, just off the link: served, kept for later or dropped.

        A STOP is served now. What the app has no handler for is dropped
        undecoded. A payload over its size limit is refused now, unparsed, and
        so is one the inbox has no room for. inbox is where it is kept; None
        is the robot connection's.
        """
        inbox = self._inbox if inbox is None else inbox
        if msg_type == Stop.TYPE:
            self._handle_payload(msg_type, payload)
        elif msg_type not in self._takes:
            self._not_taken(msg_type)
        elif len(payload) > self._payload_limit(msg_type):
            self._handle_payload(msg_type, payload)  # refused where it is decoded
        elif not inbox.add(msg_type, payload):
            problem = f"busy: {len(inbox)} link messages wait at the robot app"
            self._undecoded(msg_type, self._unread_id(msg_type, payload), problem)

    def _serve_local(self) -> None:
        """Handle what the human program on the Thor sent, for up to one step period.

        Its link messages go through the handlers of the robot connection's,
        a STOP the moment it is taken off its carrier, and they are handled
        with the robot connection down as well.
        """
        local = self._local
        if local is None:
            return
        until = self._clock.monotonic() + self._step_period_s
        self._arrived_on = local
        try:
            taken = 0
            received = local.receive(0.0)
            while received is not None:
                kept = self._local_inbox
                self._guarded("a local link message", self._sort, *received, kept)
                taken += 1
                if (received[0] == Stop.TYPE or taken % _TAKEN_PER_CHECK == 0) and (
                    self._closing.is_set() or self._clock.monotonic() >= until
                ):
                    break
                received = local.receive(0.0)
            handled = 0
            while not self._closing.is_set():
                if handled and self._clock.monotonic() >= until:
                    break  # the supervisor step is due; the rest waits
                waiting = self._local_inbox.take()
                if waiting is None:
                    break
                self._guarded("a local link message", self._handle_payload, *waiting)
                handled += 1
        finally:
            self._arrived_on = None

    def _not_taken(self, msg_type: int) -> None:
        cls = MESSAGE_TYPES.get(msg_type)
        if cls is None:
            problem = f"unknown link message type {msg_type}"
            self._undecoded(msg_type, None, problem)
        elif cls.__name__ not in self._dropped:
            self._dropped.add(cls.__name__)  # logged once per kind
            _log.warning(
                "the robot app takes no %s link message; dropped", cls.__name__
            )

    def _payload_limit(self, msg_type: int) -> int:
        return self._payload_limits.get(msg_type, SMALL_PAYLOAD_BYTES)

    def _unread_id(self, msg_type: int, payload: bytes) -> tuple[str, bool] | None:
        """The command id of a payload that is not decoded; None for no command.

        It is read at the head of the payload, where the protocol module writes
        it; a payload of a few kB is parsed for it when it is not there.
        """
        if msg_type not in COMMAND_TYPES:
            return None
        if len(payload) > SMALL_PAYLOAD_BYTES:
            return command_id_at_head(payload)
        return command_id_in(payload)

    def _decode(self, msg_type: int, payload: bytes) -> Message:
        """Decode a link message the app takes, within this app's own limits."""
        limit = self._payload_limit(msg_type)
        if len(payload) > limit:
            name = self._takes[msg_type].__name__
            error = ProtocolError(
                f"{name}: {len(payload)} bytes is over the limit of {limit} bytes"
            )
            error.command_id = self._unread_id(msg_type, payload)
            raise error
        check = self._move_shape if msg_type == Move.TYPE else None
        return decode(msg_type, payload, check)

    def _move_shape(self, obj: dict[str, Any]) -> str | None:
        """Why a MOVE has more values than a move may: none of them is converted."""
        n = self._config.n_joints
        most = self._config.executor.max_setpoints
        start, rows = obj.get("start_q"), obj.get("setpoints")
        if isinstance(start, list) and len(start) > n:
            return f"start_q needs {n} finite joint values"
        if not isinstance(rows, list):
            return None  # the decoder says what is wrong with it
        if len(rows) > most:
            return f"the move has {len(rows)} setpoints, the limit is {most}"
        for index, row in enumerate(rows):
            if isinstance(row, list) and len(row) > n:
                return f"setpoint {index} needs {n} finite joint values"
        return None

    def _handle_payload(self, msg_type: int, payload: bytes) -> None:
        """Decode one link message the app takes and handle it."""
        try:
            message = self._decode(msg_type, payload)
        except ProtocolError as error:
            if msg_type != Stop.TYPE:
                self._undecoded(msg_type, error.command_id, str(error))
                return
            problem = printable(error, MAX_REASON_CHARS)
            self._note(logging.WARNING, f"{BAD_STOP} is still a STOP: {problem}")
            message = Stop(reason=BAD_STOP, sender="unknown")
        self._handle(message)

    def _undecoded(
        self, msg_type: int, command_id: tuple[str, bool] | None, problem: str
    ) -> None:
        """Drop a link message that was not decoded; a command is refused.

        command_id is (the id as far as it was read, whether that is all of
        it). Nothing else of the payload is used, and nothing of it is run.
        """
        problem = printable(problem, MAX_REASON_CHARS)
        dropped = f"dropped a link message of type {msg_type}: "
        # A count or a place that differs each time does not make the line new.
        kind = dropped + _NUMBER.sub("#", problem)
        if command_id is None:
            self._note(logging.WARNING, dropped + problem, kind=kind)
            return
        text, whole = command_id
        shown = printable(text, MAX_ID_CHARS)
        tail = f" (command {shown}, refused)"
        self._note(logging.WARNING, dropped + problem, tail, kind=kind)
        if not (whole and shown == text):
            self._refuse_command(msg_type, self._unkept_reply(shown), problem)
            return
        stored = self._results.get(shown)
        if stored is None:
            self._refuse_command(msg_type, self._link_reply(shown), problem)
        else:
            self._deliver(stored)  # as a repeated command is answered

    def _refuse_command(self, msg_type: int, reply: _Reply, reason: str) -> None:
        """Refuse a command that no handler has seen, as its handler would."""
        what = _OPERATOR_COMMANDS.get(msg_type)
        if what is None:
            reply.final(REFUSED, reason)
        else:
            self._refuse(reply, what, reason)

    def _note(self, level: int, line: str, tail: str = "", kind: str = "") -> None:
        """Log a line that a link can make repeat: each kind once per second.

        The kind is `kind`, or the line itself; `tail` is written after the
        line and is not part of its kind.
        """
        if self._lines.allow((level, kind or line)):
            _log.log(level, "%s%s", line, tail)

    def _flush_lines(self) -> None:
        """Log how often each line that _note() left out was repeated."""
        for kind, count in self._lines.ended():
            if kind is None:
                _log.warning("%d more lines of other kinds were left out", count)
            else:
                _log.log(kind[0], "%d more: %s", count, kind[1])

    def _guarded(self, what: str, step: Callable[..., None], *args: Any) -> None:
        """Run one step of the dispatcher; an error becomes a fault-stop."""
        try:
            step(*args)
        except Exception as error:
            self._report(what, error)

    def _report(self, what: str, error: Exception) -> None:
        """Log an error of the dispatcher and raise a fault-stop; never raises."""
        text = repr(error)
        last, repeats = self._errors.get(what, ("", 0))
        repeats = repeats + 1 if text == last else 0
        self._errors[what] = (text, repeats)
        if repeats % _LOG_EVERY == 0:
            _log.error("robot app error in %s: %s", what, text, exc_info=error)
        try:
            self._fault(f"robot app error in {what}: {text}")
        except Exception:
            _log.exception("the fault-stop after an error in %s failed", what)

    def _serve_stop_requests(self) -> None:
        while self._stop_requests:
            reason, sender = self._stop_requests.popleft()
            self.stop(reason, sender)

    def _supervisor_step(self) -> None:
        """The wait, one idle read, the fault and connection checks, the state."""
        for what, part in self._supervisor_parts:
            self._guarded(what, part)

    def _step_wait(self) -> None:
        with self._lock:
            now = self._clock.monotonic()
            for kind, text in self._wait.step(now):
                self._emit(kind, text, {"remaining_s": self._wait.remaining(now)})

    def _on_expire(self, timed_return: bool) -> None:
        """Queue the timed return; runs inside the wait's step, under the lock."""
        if not timed_return:
            return
        if not self._carrier.alive:
            self._no_timed_return()
            return
        busy = self._busy()
        if busy is not None:
            # The wait's one attempt is made now, when the bells said so: queued
            # behind other work the arm would move by itself after that work ended.
            data = {"source": rest_module.WAIT, "status": REFUSED, "reason": busy}
            self._emit(rest_module.EVENT_REST, f"no timed return to rest: {busy}", data)
            return
        # Quoted by the return: a STOP from here on keeps the arm where it is.
        stop_number = self._authority.stop_number
        self._queue(
            _Job(
                "the timed return to rest",
                lambda job: self._run_timed_return(job, stop_number),
                None,
                needs_link=True,
            )
        )

    def _step_idle(self) -> None:
        now = self._clock.monotonic()
        reading = self._idle.step(now)
        if reading is not None and self._read_times is not None:
            self._note_read_time(reading.t_end - reading.t_start)
        self._refresh_jaws(now)

    def _note_read_time(self, took_s: float) -> None:
        """Say once after a connect how long one joint read takes on the bus.

        Measured on the first READ_TIMING_READS idle reads, which are made
        anyway: no read is added for it.
        """
        times = self._read_times
        if times is None:
            return
        times.append(took_s)
        if len(times) < READ_TIMING_READS:
            return
        self._read_times = None
        median, worst = statistics.median(times), max(times)
        period = self._config.state.idle_read_period_s
        per_s = "any number of" if median <= 0.0 else f"about {1.0 / median:.0f}"
        _log.info(
            "one joint read takes %.1f ms on the motor bus (median of %d idle "
            "reads; worst %.1f ms): the bus allows %s joint reads a second; "
            "state.idle_read_period_s asks for %.0f",
            1e3 * median,
            len(times),
            1e3 * worst,
            per_s,
            1.0 / period,
        )
        if median > period:
            _log.warning(
                "one joint read takes longer than state.idle_read_period_s "
                "(%g s): the idle reads follow each other without a pause, and "
                "a link message waits for the read under way",
                period,
            )

    def _refresh_jaws(self, now: float) -> None:
        """One jaw reading for each ROBOT_STATE while the arm holds.

        The agent decides from it what the jaws hold: after a lift, before a
        release and between its skills. None is taken while a move sends
        setpoints, a jaw command runs, the motors are off or a joint's idle
        read is failing: the bus is the joints' then. After two readings in a
        row that did not come, the next is taken five periods later, until
        one comes: a jaw motor that stays silent costs each reading its
        timeout, and the joints keep the bus.
        """
        if now < self._jaws_due or self._idle.due_in(now) is None:
            return
        if self._worker.busy():
            return
        if self._idle.consecutive_failures_s(self._clock.monotonic()) > 0.0:
            return
        period = self._config.state.publish_period_s
        self._jaws_due = now + period
        # A pause of the idle reads waits for this read too.
        with self._idle.extra_read() as free:
            if not free or not self._arm.gripper_referenced():
                return  # no reading is taken
            self._jaw_state.refresh(self._arm)
            if self._jaw_state.gripper_state(self._arm).valid:
                self._jaw_peek_misses = 0
                return
            self._jaw_peek_misses += 1
            if self._jaw_peek_misses >= 2:
                # Counted from the end of the reading that did not come.
                self._jaws_due = self._clock.monotonic() + 5.0 * period

    def _poll_faults(self) -> None:
        if self._authority.stop_level is StopLevel.FAULT:
            return  # one fault-stop per fault
        faults = [str(fault) for fault in self._arm.faults()]
        if faults:
            self._fault("; ".join(faults))
            return
        lost = self._idle.consecutive_failures_s(self._clock.monotonic())
        if lost > LOST_FEEDBACK_S:
            self._fault(f"joint feedback lost for {lost:.2f} s in hold")
            # ROBOT_STATE's torque_on says that this program has not released
            # the motors; it is not a reading of them.
            _log.error(
                "joint feedback is lost: whether the motors still hold the arm "
                "cannot be read until the joints answer again; keep a hand on "
                "the arm before a reset"
            )

    def _check_link(self) -> None:
        if self._link_lost or self._carrier.alive:
            return
        with self._lock:
            if self._link_lost or self._carrier.alive:
                return  # a new carrier was attached meanwhile
            self._link_lost = True  # handled once per carrier; a dead link stays dead
            # Told to stop before the view and the human targets go: a human
            # drive then ends on the lost connection, not on what it took away.
            self._stop_motion(LINK_LOST, lambda job: job.needs_link)
            # What came over the lost connection and was not handled goes with it.
            self._inbox = Inbox()
            self._view = None
            # The human targets of the lost connection go too, and a grip that
            # stays squeezed moves nothing on the next one.
            self._human_input.clear()
            self._release_seen = None
            self._human_refusal = ""
            # The trigger must match again; the jaw stop below holds lightly
            # the jaws that followed it. Jaws the agent closed keep their grip.
            followed = self._jaw_follower.following
            self._jaw_follower.end(LINK_LOST, hold=False)
            # A jaw reference was typed at the Thor terminal: it needs no link.
            self._worker.stop_now(keep_reference=True, hold_idle=followed)
            cancelled = self._wait.cancel(LINK_LOST)
            state = self._authority.snapshot()
            if state.owner is Owner.AGENT:
                self._authority.set_owner_if(
                    Owner.NOBODY,
                    LINK_LOST,
                    stop_number=state.stop_number,
                    owner_number=state.owner_number,
                )
            text = f"{LINK_LOST}: the arm holds"
            if state.owner is Owner.AGENT:
                text += "; the agent's ownership ended"
            if state.owner is Owner.HUMAN:
                text += PERSON_KEEPS_ARM
            if cancelled:
                text += "; the assistance wait is cancelled"
            text += "; rest can still be typed at the Thor terminal"
            data = {"owner_before": state.owner.value, "wait_cancelled": cancelled}
            self._emit(EVENT_CONNECTION, text, data)

    def _publish(self) -> None:
        """Send ROBOT_STATE when something a command quotes changed, or it is due."""
        now = self._clock.monotonic()
        key = (
            self._authority.snapshot(),
            self._motion.get(),
            self._wait.active(),
            self._last_result,
            # The agent waits for "armed" before it clears a move's first setpoints.
            self._executor.stage(),
        )
        with self._lock:
            # Cleared before the state is built: a carrier attached after this
            # is owed one again.
            forced, self._state_forced = self._state_forced, False
            carrier = self._carrier
            if self._local is not None:
                # The human program on the Thor gets every ROBOT_STATE too,
                # also with the robot connection down.
                carrier = _Both(carrier, self._local)
        if not carrier.alive or (key == self._published and not forced):
            self._publisher.maybe_publish(carrier, now)
            return
        self._published = key
        self._send(self._publisher.build(now), via=carrier)

    def _fault(self, reason: str) -> None:
        """Raise a fault-stop, unless one is latched already."""
        with self._lock:
            if self._authority.stop_level is not StopLevel.FAULT:
                self._stops.fault(reason, self._way_on(reason))
            elif reason != self._fault_while_latched:
                # One fault-stop per fault; a fault that repeats is logged once.
                self._fault_while_latched = reason
                _log.error("fault while a fault-stop is latched: %s", reason)

    # Link messages

    def _handle(self, message: Message) -> None:
        handler = self._handlers.get(type(message))
        if handler is None:
            self._not_taken(message.TYPE)
            return
        message = self._fitted(message)
        if message is None:
            return  # its id cannot be used: refused
        command_id = getattr(message, "command_id", None)
        if command_id is not None:
            stored = self._results.get(command_id)
            if stored is not None:
                # A repeated command is answered again and not executed again.
                line = f"command {command_id} repeated; RESULT sent again"
                self._note(logging.INFO, line, kind="a command repeated")
                self._deliver(stored)
                return
            if not self._results.room_for(command_id):
                # Taking it would mean forgetting a command that has not ended.
                reason = f"busy: {RESULT_IDS_KEPT} commands have not ended"
                self._refuse_command(
                    message.TYPE, self._unkept_reply(command_id), reason
                )
                return
        try:
            handler(message)
        except Exception as error:
            # The fault-stop first: it stands when the command's RESULT arrives.
            self._report("a link message", error)
            if command_id is not None:
                stored = self._results.get(command_id)
                if stored is None or stored.status == ACCEPTED:
                    reason = f"robot app error: {error!r}"
                    self._send_result(Result(command_id, FAILED, reason))

    def _fitted(self, message: Message) -> Message | None:
        """The link message with its texts printable and cut to their limits.

        An id is never changed: a command whose id is too long or cannot be
        printed is refused here and a permit with such a move id is dropped;
        None is returned then. The session id is only ever compared.
        """
        changes: dict[str, str] = {}
        for item in dataclasses.fields(message):
            value = getattr(message, item.name)
            if not isinstance(value, str) or item.name == "session_id":
                continue
            if item.name in _ID_FIELDS:
                if printable(value, MAX_ID_CHARS) != value:
                    self._unusable_id(message, item.name, value)
                    return None
                continue
            limit = _TEXT_LIMITS.get(item.name, MAX_REASON_CHARS)
            shown = printable(value, limit)
            if shown != value:
                changes[item.name] = shown
        return dataclasses.replace(message, **changes) if changes else message

    def _unusable_id(self, message: Message, name: str, value: str) -> None:
        """Refuse a command, or drop a permit, whose id cannot be used."""
        what = name.replace("_", " ")
        if len(value) > MAX_ID_CHARS:
            problem = f"the {what} is longer than {MAX_ID_CHARS} characters"
        else:
            problem = f"the {what} has a character that cannot be printed"
        shown = printable(value, MAX_ID_CHARS)
        kind = type(message).__name__
        self._note(logging.WARNING, f"{kind} refused: {problem}", f" ({shown})")
        if name == "command_id":
            # Answered under the id as it is shown; nothing is kept under it,
            # so two ids that are shown alike stay two commands.
            self._refuse_command(message.TYPE, self._unkept_reply(shown), problem)

    def _unkept_reply(self, command_id: str) -> _Reply:
        """Where the RESULT of a command goes that the results store does not keep."""

        def deliver(result: Result) -> None:
            self._deliver(result)
            self._last_result = _summary(result)

        return _Reply(command_id, deliver, announces=False)

    def _on_stop(self, m: Stop) -> None:
        self.stop(m.reason, m.sender)

    def _on_view_status(self, m: ViewStatus) -> None:
        # A takeover and every setpoint of a human drive need this view fresh.
        self._view = (m, self._clock.monotonic())

    def _on_human_setpoint(self, m: HumanSetpoint) -> None:
        if m.session_id != self._authority.session_id:
            line = "a HUMAN_SETPOINT of another session was dropped"
            self._note(logging.DEBUG, line)
            return
        if self._second_human_program():
            return
        self.offer_human_target(human_target(m, self._clock.monotonic()))

    def _second_human_program(self) -> bool:
        """Whether the HUMAN_SETPOINT being handled is a second human program's.

        One human program at a time is followed: the one at the robot link
        service, whose link messages come over the robot connection, or the
        one at the robot app's own socket. While one of them has sent within
        human.stage_keep_s, the other's HUMAN_SETPOINTs are dropped. An EVENT
        says so once, and again when the other sends after it was silent for
        human.stage_keep_s or after the followed one changed.
        """
        local = self._arrived_on is not None
        now = self._clock.monotonic()
        last = self._human_from
        keep_s = self._config.human.stage_keep_s
        if last is not None and last[0] != local and now - last[1] <= keep_s:
            dropped_at, self._second_at = self._second_at, now
            if dropped_at is None or now - dropped_at > keep_s:
                ends = (_AT_SOCKET, _OVER_LINK) if local else (_OVER_LINK, _AT_SOCKET)
                text = SECOND_HUMAN_PROGRAM.format(here=ends[0], there=ends[1])
                self._emit(EVENT_REFUSED, text, {"local": local})
            return True
        if last is None or last[0] != local:
            self._second_at = None  # the followed one changed
        self._human_from = (local, now)
        return False

    def _on_permit(self, m: Permit) -> None:
        reason = self._gate.update(m)
        if reason is not None and reason != self._permit_refusal:
            line = f"permit for move {m.move_id} refused: {reason}"
            self._note(logging.INFO, line, kind=f"a permit refused: {reason}")
        self._permit_refusal = reason

    def _on_move(self, m: Move) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            # Read before the admission, so a halt raised after it refuses the move.
            halt_count = self._checkers.agent.halt_count()
            reason = (
                self._authority.admit(
                    m.session_id, m.owner_number, m.stop_number, Owner.AGENT
                )
                or self._motors_off()
                or self._busy()
                or self._referencing()
                or self._expired(m.expires_mono_s)
                or self._region_problem(m.grasp_region)
            )
            evidence_mono: float | None = None
            if reason is None and not self._config.permits.required_for_agent:
                evidence_mono, reason = self._move_evidence(m.evidence_wall_s)
            if reason is not None:
                reply.final(REFUSED, reason)
                return
            # The numbers the MOVE quoted and admit() found current: the move
            # ends as soon as one of them changes.
            admitted = AuthorityState(
                m.session_id, Owner.AGENT, m.owner_number, m.stop_number
            )
            reply.accepted()
            self._queue(
                _Job(
                    "a move",
                    lambda job: self._run_move(
                        job, m, admitted, halt_count, evidence_mono
                    ),
                    reply,
                    needs_link=True,
                    is_move=True,
                )
            )

    def _on_gripper(self, m: Gripper) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = (
                self._authority.admit(
                    m.session_id, m.owner_number, m.stop_number, Owner.AGENT
                )
                or self._motors_off()
                # A return to rest or a release has proven the jaws empty, or
                # will: no jaw command may follow that proof.
                or self._busy(lambda job: not job.is_move)
                or self._referencing()
            )
            if reason is not None:
                reply.final(REFUSED, reason)
                return
            done = _JawReply(reply, m.mode)
            # The worker refuses a GRIPPER whose expiry has passed, now or later.
            done.queued(
                self._worker.submit(
                    m.opening,
                    m.max_effort,
                    m.mode,
                    Owner.AGENT,
                    done,
                    m.expires_mono_s,
                    m.scout_opening,
                )
            )

    def _on_open(self, m: Open) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.NOBODY
            )
            self._typed_jaws(reply, m.action, m.confirmed, reason)

    def _typed_jaws(
        self, reply: _Reply, action: str, confirmed: bool, reason: str | None
    ) -> str | None:
        """Queue a typed opening or close of the jaws; why not, or None.

        Under the app's lock. reason is why the command's numbers are not
        taken, None when they are. The opening is the driver's own, and so is
        the close. An opening needs the operator's typed confirmation; a close
        needs it while the jaws are not proven empty: they may close on what
        they hold. Nobody may own the arm.
        """
        if action not in OPEN_ACTIONS:
            reason = reason or f"unknown action {action!r}"
        elif reason is None and not confirmed:
            if action == OPEN_JAWS:
                reason = CONFIRM_OPEN
            elif self._jaw_state.not_proven(self._arm) is not None:
                reason = CONFIRM_CLOSE
        reason = reason or self._motors_off() or self._busy()
        # It would wait behind a reference and move the jaws as that ends.
        reason = reason or self._referencing()
        if reason is not None:
            self._refuse(reply, "close" if action == CLOSE_JAWS else "open", reason)
            return reason
        if action == CLOSE_JAWS:
            done = _JawReply(reply, CLOSE)
            reason = self._worker.close_jaws(done, Owner.NOBODY)
        else:
            done = _JawReply(reply, OPEN)
            reason = self._worker.open_guarded(done, Owner.NOBODY)
        done.queued(reason)
        return reason

    def _take_jaw_word(self, jaws_empty: bool, what: str) -> None:
        """Take the operator's word that the jaws are empty; under the app's lock."""
        if not jaws_empty:
            return
        self._jaw_state.take_word()
        self._emit(EVENT_JAWS, JAWS_EMPTY_WORD.format(what=what), {"for": what})

    def _on_checker_state(self, m: CheckerState) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.AGENT
            )
            if reason is None and m.action not in (CLEAR_CONTACT_SPOTS, CLOSE_GRASP):
                reason = f"unknown action {m.action!r}"
            if reason is not None:
                reply.final(REFUSED, reason)
                return
            if m.action == CLEAR_CONTACT_SPOTS:
                # The recovery checker keeps the spots the agent checker has.
                self._checkers.agent.clear_contact_spots()
                self._checkers.recovery.clear_contact_spots()
                self._contact_hold = False
            else:
                self._checkers.agent.set_grasp_region(None)
                self._grasp_region = None
            reply.final(COMPLETED)

    def _on_reset(self, m: Reset) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            # admit() refuses everything while a fault-stop is latched, so the
            # reset compares the numbers the RESET quotes itself.
            reason = self._session_mismatch(m.session_id)
            reason = reason or self._stops.reset(
                m.stop_number, self._arm, m.owner_number
            )
        if reason is not None:
            self._refuse(reply, "reset", reason)
        else:
            reply.final(COMPLETED)

    def _on_release(self, m: Release) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, None
            )
            reason = reason or self._busy()
            if reason is not None:
                self._refuse(reply, "release", reason)
                return
            self._take_jaw_word(m.jaws_empty, "a torque release")
            reply.accepted()
            self._queue(
                _Job(
                    "a torque release",
                    lambda job: self._run_release(job, m.stop_number),
                    reply,
                    needs_link=True,
                )
            )

    def _on_shutdown(self, m: Shutdown) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, None
            )
            reason = reason or self._shutdown_unconfirmed(m.confirmed) or self._busy()
            if reason is not None:
                self._refuse(reply, "shutdown", reason)
                return
            self._take_jaw_word(m.jaws_empty, "a shutdown")
            reply.accepted()
            self._queue(
                _Job(
                    "a shutdown",
                    lambda job: self._run_shutdown(job, m.stop_number, m.confirmed),
                    reply,
                    needs_link=True,
                )
            )

    def _on_owner_request(self, m: OwnerRequest) -> None:
        reply = self._link_reply(m.command_id)
        if m.request == "agent":
            reason = self._new_task(m, reply)
        elif m.request == "help_needed":
            reason = self._help_needed(m, reply)
        elif m.request == "nobody":
            reason = self._give_up(m, reply)
        elif m.request == "rest":
            reason = self._typed_rest(m, reply)
        elif m.request == REQUEST_PARK:
            # The agent program's operator typed that the path is clear.
            reason = self._typed_rest(dataclasses.replace(m, confirmed=True), reply)
        elif m.request == REQUEST_MOTORS_OFF:
            reason = self._motors_off_asked(m, reply)
        elif m.request == REQUEST_MOTORS_ON:
            reason = self._motors_on_asked(m, reply)
        elif m.request == REQUEST_HUMAN:
            reason = self._takeover(m, reply)
        elif m.request == REQUEST_HANDBACK:
            reason = self._handback(m, reply)
        else:
            reason = f"unknown request {m.request!r}"
        if reason is not None:
            self._refuse(reply, f"owner request {m.request!r}", reason)

    def _new_task(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "agent": why it is refused, or None once it was answered."""
        with self._lock:
            if self._takeover_open:
                return TAKEOVER_OPEN  # also when a STOP made nobody the owner
            reason = self._session_mismatch(m.session_id) or self._motors_off()
            reason = reason or self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.NOBODY
            )
            reason = reason or self._busy() or self._referencing()
            if reason is not None:
                return reason
            seen = self._authority.snapshot()
        # Read outside the lock: a STOP does not wait for the joint read, and
        # set_owner_if() below lets one that came meanwhile win.
        try:
            pose = self._read_pose()
        except ReadFailed as failed:
            return f"the measured pose could not be read: {failed}"
        try:
            # Judged as a hold where the arm is, as the first setpoint of the
            # task will be: a low gripper tip does not refuse the task.
            self._checkers.agent.check_start_pose(pose)
        except Refusal as refusal:
            return f"type rest first: {refusal.reason}"
        with self._lock:
            # What began or ended while the pose was read.
            reason = self._motors_off() or self._busy() or self._referencing()
            if reason is not None:
                return reason
            number = self._authority.set_owner_if(
                Owner.AGENT,
                "new task",
                stop_number=seen.stop_number,
                owner_number=seen.owner_number,
            )
            if number is None:
                return _RACED
            self._wait.cancel("a new task was accepted")
            # The operator's word was for the jaws as the last owner left them.
            self._jaw_state.clear_word()
            self._task_timed_return = bool(m.timed_return)
            text = f"new task: the agent owns the arm, owner number {number}"
            detail = {"owner": Owner.AGENT.value, "owner_number": number}
            self._emit(EVENT_OWNER, text, dict(detail))
            reply.final(COMPLETED, detail=detail)
        return None

    def _help_needed(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "help_needed": the agent failed while it owned the arm."""
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.AGENT
            )
            if reason is not None:
                return reason
            seen = self._authority.snapshot()
            why = f"help needed: {m.reason.strip() or 'no reason given'}"
            number = self._authority.set_owner_if(
                Owner.NOBODY,
                why,
                stop_number=seen.stop_number,
                owner_number=seen.owner_number,
            )
            if number is None:
                return _RACED
            self._stop_motion(why, lambda job: job.is_move)
            self._worker.stop_now()
            timed_return = self._task_timed_return and self._config.rest.auto_return
            detail = {
                "owner": Owner.NOBODY.value,
                "owner_number": number,
                "timed_return": timed_return,
                "assistance_s": self._config.waits.assistance_s,
            }
            text = f"HELP NEEDED: {why}; the arm holds and nobody owns it"
            self._emit(EVENT_HELP_NEEDED, text, {"reason": m.reason, **detail})
            self._wait.start(m.reason, timed_return)
            reply.final(COMPLETED, detail=detail)
        return None

    def _give_up(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "nobody": the agent's ownership ends in a hold."""
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, None
            )
            if reason is not None:
                return reason
            seen = self._authority.snapshot()
            number = seen.owner_number
            if seen.owner is not Owner.NOBODY:
                if seen.owner is not Owner.AGENT:
                    return f"control owner is {seen.owner}, not agent"
                why = f"the agent gave the arm back: {m.reason.strip() or 'no reason'}"
                number = self._authority.set_owner_if(
                    Owner.NOBODY,
                    why,
                    stop_number=seen.stop_number,
                    owner_number=seen.owner_number,
                )
                if number is None:
                    return _RACED
                self._stop_motion(why, lambda job: job.is_move)
                self._worker.stop_now()
                self._emit(EVENT_OWNER, f"{why}; the arm holds and nobody owns it")
            detail = {"owner": Owner.NOBODY.value, "owner_number": number}
            reply.final(COMPLETED, detail=detail)
        return None

    def _typed_rest(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "rest": the operator typed rest at the console."""
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, None
            )
            if reason is None and not m.confirmed:
                reason = CONFIRM_PATH
            reason = reason or self._busy()
            if reason is not None:
                return reason
            self._take_jaw_word(m.jaws_empty, "a return to rest")
            reply.accepted()
            self._queue(
                _Job(
                    "a return to rest",
                    lambda job: self._run_rest(
                        job, rest_module.TYPED, m.stop_number, confirmed=True
                    ),
                    reply,
                    needs_link=True,
                )
            )
        return None

    def _motors_owner(self, m: OwnerRequest) -> str | None:
        """Why a request about the motors is not taken; under the app's lock.

        Its numbers must stand, and the agent or nobody must own the arm.
        """
        reason = self._authority.admit(
            m.session_id, m.owner_number, m.stop_number, None
        )
        owner = self._authority.snapshot().owner
        if reason is None and owner not in (Owner.AGENT, Owner.NOBODY):
            reason = f"control owner is {owner}"
        return reason

    def _motors_off_asked(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "motors_off": the agent's rest tool has folded the arm.

        That tool found no load before it folded and has closed the jaws
        itself. The motors go off only at measured rest.
        """
        with self._lock:
            reason = self._motors_owner(m) or self._busy()
            if reason is not None:
                return reason
            reply.accepted()
            self._queue(
                _Job(
                    "a torque release",
                    lambda job: self._run_release(
                        job, m.stop_number, jaws_empty=True, close=False
                    ),
                    reply,
                    needs_link=True,
                )
            )
        return None

    def _motors_on_asked(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "motors_on": its sender asked the operator first."""
        with self._lock:
            reason = self._motors_owner(m) or self._power_on_refusal(True)
            if reason is not None:
                return reason
            reply.accepted()
            self._queue(
                _Job("motors on again", self._run_power_on, reply, needs_link=True)
            )
        return None

    def _takeover(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "human": the numbers it quotes, then _grant_takeover()."""
        with self._lock:
            reason = self._human_off() or self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.NOBODY
            )
            if reason is not None:
                return reason
            seen = self._authority.snapshot()
        return self._grant_takeover(seen, m.reason, reply)

    def _handback(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "handback": the numbers it quotes, then _end_takeover()."""
        with self._lock:
            reason = self._no_takeover() or self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, None
            )
            if reason is not None:
                return reason
            seen = self._authority.snapshot()
        return self._end_takeover(seen, m.reason, reply, m.jaws_empty)

    def _grant_takeover(
        self, seen: AuthorityState, reason: str, reply: _Reply
    ) -> str | None:
        """Make a person the control owner; why not, or None once it was answered.

        seen is the authority as the request found it, with nobody the owner;
        reason is the request's own. No link message is read here. No jaw
        command is sent: the jaws stay as they are, and the opening in force
        is noted as the grip level the person's trigger must reach.
        """
        with self._lock:
            problem = self._takeover_problem(self._clock.monotonic(), seen)
            if problem is not None:
                return problem
        # Read outside the lock: a STOP does not wait for the joint read, and
        # set_owner_if() below lets one that came meanwhile win.
        began = self._clock.monotonic()
        try:
            pose = self._read_pose()
        except ReadFailed as failed:
            return f"the measured pose could not be read: {failed}"
        read_s = self._clock.monotonic() - began
        low = self._low_tip(pose)
        try:
            if low:
                # A person lifts such an arm out: the human drive takes no
                # target that lowers the tip while it is low.
                self._checkers.human.check_low_pose(pose)
            else:
                self._checkers.human.check_pose(pose)
        except Refusal as refusal:
            if self._authority.stop_number != seen.stop_number:
                return _RACED  # a fault-stop latches the checker
            return f"{OUTSIDE_HUMAN_LIMITS}{refusal.reason}; type rest"
        with self._lock:
            # What began or ended while the pose was read, and the two ages
            # again: they count when the person becomes the owner.
            problem = self._motors_off() or self._busy() or self._jaws_busy()
            problem = problem or self._ran_out(read_s)
            if problem is not None:
                return problem
            number = self._authority.set_owner_if(
                Owner.HUMAN,
                f"takeover: {reason.strip() or 'no reason given'}",
                stop_number=seen.stop_number,
                owner_number=seen.owner_number,
            )
            if number is None:
                return _RACED
            self._wait.cancel("a person took over")
            self._takeover_open = True
            self._jaw_state.clear_word()  # it was for the jaws before the person
            if self._config.human.trigger_jaws:
                # The grip level: the trigger does nothing until it matches it.
                self._jaw_follower.note(number, seen.stop_number)
            detail = {"owner": Owner.HUMAN.value, "owner_number": number}
            taken = TAKEN_OVER.format(number=number) + (LOW_TIP if low else "")
            self._emit(EVENT_OWNER, taken, dict(detail))
            reply.final(COMPLETED, detail=detail)
        return None

    def _ran_out(self, read_s: float) -> str | None:
        """Why a takeover is refused after its joint read of read_s, else None.

        The controller data and the live view were fresh before the read.
        The dispatcher takes up no link message while it reads, so each is
        older by the read; READ_TOOK says so.
        """
        now = self._clock.monotonic()
        took = READ_TOOK.format(took=read_s)
        problem = self._controller_problem(now)
        if problem is not None:
            return problem + took
        problem = self._source_problem()
        if problem is not None:
            return problem
        problem = self._view_problem(now)
        return None if problem is None else problem + took

    def _low_tip(self, pose: Sequence[float]) -> bool:
        """Whether the gripper tip is low at this pose, as the human checker has it.

        Low is below the table clearance or inside the agent's last grasp
        region. False with a checker that knows no table.
        """
        try:
            tip = self._checkers.human.tip_state(pose, self._grasp_region)
        except Refusal:
            return False  # no pose: check_pose() says so
        return tip is not None and tip.low is not None

    def _end_takeover(
        self,
        seen: AuthorityState,
        reason: str,
        reply: _Reply,
        jaws_empty: bool = False,
    ) -> str | None:
        """Queue the hand-back; why not, or None once it was accepted.

        seen is the authority as the request found it: a person or nobody the
        owner. The takeover ends once the jaws are proven empty, or empty on
        the operator's typed word (jaws_empty), and the arm is measured still
        at rest. Nothing is moved and nothing is cut.
        """
        with self._lock:
            now = self._authority.snapshot()
            problem = self._no_takeover() or self._authority.admit(
                seen.session_id, seen.owner_number, seen.stop_number, None
            )
            if problem is None and now.owner not in (Owner.HUMAN, Owner.NOBODY):
                problem = f"control owner is {now.owner}"
            problem = problem or self._busy()
            problem = problem or (JAWS_BUSY if self._worker.busy() else None)
            if problem is not None:
                return problem
            self._take_jaw_word(jaws_empty, "a hand-back")
            reply.accepted()
            self._queue(
                _Job(
                    "a hand-back",
                    lambda job: self._run_handback(job, now, reason),
                    reply,
                    needs_link=True,
                )
            )
        return None

    # Admission helpers

    def _session_mismatch(self, session_id: str) -> str | None:
        if session_id != self._authority.session_id:
            return "session id does not match"
        return None

    def _motors_off(self) -> str | None:
        """Why nothing may move the arm: its motors are off, or may be."""
        motion = self._motion.get()
        if motion is MotionState.REST_UNPOWERED:
            return MOTORS_OFF
        if motion is MotionState.RELEASE_FAILED:
            return MOTORS_UNKNOWN
        return None

    def _not_running(self) -> str | None:
        return None if self.running else NOT_RUNNING

    def _human_off(self) -> str | None:
        return None if self._config.human.enabled else HUMAN_OFF

    def _no_takeover(self) -> str | None:
        return None if self._takeover_open else NO_TAKEOVER

    def _takeover_problem(
        self, now: float, seen: AuthorityState | None = None
    ) -> str | None:
        """Why a takeover is refused at the app-clock time `now`, else None.

        The first reason, in the order of the refusals. Without `seen`, only
        what needs no joint read and depends neither on the owner nor on
        running work: what ROBOT_STATE shows. With `seen`, the numbers a
        request was made on, also that they stand, that nobody owns the arm
        and that no motion work and no jaw command runs or waits.
        """
        reason = self._human_off()
        if reason is not None:
            return reason
        with self._lock:
            # Without `seen` the app's own numbers: only the fault-stop counts.
            quoted = self._authority.snapshot() if seen is None else seen
            reason = self._authority.admit(
                quoted.session_id,
                quoted.owner_number,
                quoted.stop_number,
                None if seen is None else Owner.NOBODY,
            )
            reason = reason or self._motors_off()
            if reason is None and seen is not None:
                reason = self._busy() or self._jaws_busy()
            return (
                reason
                or self._controller_problem(now)
                or self._source_problem()
                or self._view_problem(now)
            )

    def _controller_problem(self, now: float) -> str | None:
        """Why the newest human target is no fresh controller data, else None."""
        target = self._human_input.newest()
        if target is None:
            return f"{NO_CONTROLLER_DATA}no HUMAN_SETPOINT has arrived"
        if target.stage != STAGE_READY:
            return f"{NO_CONTROLLER_DATA}stage {target.stage or 'unknown'}"
        age = self._human_input.sample_age_s(now)
        limit = self._config.human.sample_age_limit_s
        if age is None:
            return f"{NO_CONTROLLER_DATA}the sample age is unknown"
        if age > limit:
            return f"{NO_CONTROLLER_DATA}{age:.3f} s old, the limit is {limit:g} s"
        return None

    def _source_problem(self) -> str | None:
        """Why the newest human target's source may not drive this arm, else None.

        Only the headset's word may drive an arm that is not simulated.
        """
        target = self._human_input.newest()
        if target is None or target.source == HUMAN_SOURCE_TELEOP:
            return None
        if self._driver in SIMULATED_DRIVERS:
            return None
        return (
            "a synthetic controller may drive only a simulated arm; this robot "
            f"app declares {_declared(self._driver)}"
        )

    def _view_problem(self, now: float) -> str | None:
        """Why the live view is not fresh enough for a person to drive, else None.

        None also where the file does not require it (human.require_live_view).
        """
        human = self._config.human
        if not human.require_live_view:
            return None
        age = self.view_age_s(now)
        if age is None:
            view = self._view
            if view is None:
                why = "no VIEW_STATUS has arrived"
            elif not view[0].live:
                why = "the view is not live"
            elif view[0].sent_mono_s is None:
                why = "the VIEW_STATUS does not say when it was sent"
            else:
                why = "the view's age is unknown"
        elif age > human.view_age_limit_s:
            why = f"{age:.3f} s old, the limit is {human.view_age_limit_s:g} s"
        else:
            return None
        return f"{VIEW_NOT_FRESH}{why}{OPEN_THE_VIEW}"

    def _shape_problem(self, target: HumanTarget) -> str | None:
        """Why an engaged target's joint values cannot be followed, else None."""
        n = self._config.n_joints
        if target.q is None:
            return "the engaged target carries no joint values"
        if len(target.q) != n:
            return f"the target has {len(target.q)} values, the arm {n} joints"
        return None

    def _human_facts(self) -> dict[str, Any]:
        """ROBOT_STATE's health["human"]: the takeover, the drive, the newest target."""
        now = self._clock.monotonic()
        with self._lock:
            target = self._human_input.newest()
            # Controller data that are away set the flag when they come back.
            away = self._controller_problem(now) is not None
            unseen = self._release_seen != self._authority.owner_number
            # While a takeover is open, with human.trigger_jaws: whether the
            # trigger moves the jaws, and why not.
            jaws: dict[str, Any] = {}
            if self._config.human.trigger_jaws and self._takeover_open:
                jaws = {"jaws": self._jaw_follower.facts()}
            return {
                **jaws,
                "enabled": self._config.human.enabled,
                # Whether a takeover and a human drive need a fresh live view.
                "live_view_required": self._config.human.require_live_view,
                "takeover_open": self._takeover_open,
                "driving": self._human_drive.progress() is not None,
                "needs_release": unseen or away,
                "source": "" if target is None else target.source,
                "seq": None if target is None else target.seq,
                "round_trip_s": None if target is None else target.round_trip_s,
                "note": "" if target is None else target.note,
                "ended": self._drive_ended,
                "refusal": self._human_refusal,
                # "" or why a takeover would be refused now, whoever owns the arm.
                "takeover": self._takeover_problem(now) or "",
            }

    def _shutdown_unconfirmed(self, confirmed: bool) -> str | None:
        """Why a shutdown is not taken: it would move the arm and is unconfirmed."""
        if confirmed or not self.shutdown_moves_arm():
            return None
        return CONFIRM_PATH

    def _busy(self, match: Callable[[_Job], bool] = _any_job) -> str | None:
        """Why no new work is taken: motion work that matches is queued or running."""
        with self._motion_cond:
            if self._running is not None and match(self._running):
                return f"busy: {self._running.what} is running"
            for job in self._jobs:
                if match(job):
                    return f"busy: {job.what} is queued"
        return None

    def _referencing(self) -> str | None:
        """Why no task, move, open or jaw command is taken: a jaw reference goes on.

        It waits or runs. The owner check of a MOVE or a GRIPPER comes first
        and refuses both: nobody owns the arm meanwhile.
        """
        return JAWS_BUSY if self._worker.referencing() else None

    def _jaws_busy(self) -> str | None:
        """Why no takeover is taken: a jaw command or a jaw reference goes on.

        The owner change would end a guarded opening with the jaws part open.
        """
        return JAWS_BUSY if self._worker.busy() else None

    def _expired(self, expires_mono_s: float) -> str | None:
        late = self._clock.monotonic() - expires_mono_s
        return None if late < 0.0 else f"the move expired {late:.3f} s ago"

    def _region_problem(self, region: GraspRegion | None) -> str | None:
        """Why a MOVE's grasp region is not taken: it is beyond the Thor's bounds."""
        if region is None:
            return None
        try:
            self._checkers.agent.check_grasp_region(region)
        except Refusal as refusal:
            return refusal.reason
        return None

    def _move_evidence(
        self, evidence_wall_s: float | None
    ) -> tuple[float | None, str | None]:
        """A MOVE's camera evidence when permits are off: (monotonic time, refusal)."""
        if evidence_wall_s is None:
            return None, "the move carries no evidence_wall_s and permits are off"
        now = self._clock.monotonic()
        evidence_mono = evidence_wall_s - (self._clock.wall() - now)
        if evidence_mono > now + FUTURE_TOLERANCE_S:
            return None, "the move's evidence is in the future"
        evidence_mono = min(evidence_mono, now)
        limit = self._config.permits.evidence_age_limit_s
        if now - evidence_mono >= limit:
            age = now - evidence_mono
            return (
                None,
                f"the move's evidence is {age:.2f} s old, the limit is {limit:g} s",
            )
        return evidence_mono, None

    def _read_pose(self) -> list[float]:
        """The pose from one fresh full read, kept in the history; raises ReadFailed."""
        readings = self._arm.read_all()
        for reading in readings:
            self._history.add(reading)
        return _pose(readings, self._arm.n_joints)

    def _off_rest(self, pose: Sequence[float]) -> tuple[int, float]:
        """The joint index farthest from the rest pose, and how far it is, rad."""
        off = [abs(at - goal) for at, goal in zip(pose, self._config.rest.rest_q)]
        joint = max(range(len(off)), key=off.__getitem__)
        return joint, off[joint]

    def _welcome(self) -> Welcome:
        config = self._config
        limits = config.limits
        return Welcome(
            session_id=self._authority.session_id,
            robot_id=config.robot_id,
            profile_id=config.profile_id,
            joint_names=list(config.joint_names),
            joint_min=list(limits.agent.joint_min),
            joint_max=list(limits.agent.joint_max),
            rest_q=list(config.rest.rest_q),
            setpoint_period_s=config.executor.period_s,
            max_move_setpoints=config.executor.max_setpoints,
            limit_set_versions={
                "agent": limits.agent.version,
                "human": limits.human.version,
                "recovery": limits.recovery.version,
            },
            camera_calibration_id=config.camera_calibration_id,
            driver=self._driver,
            agent_max_joint_speed=limits.agent.max_joint_speed,
        )

    # The motion worker

    def _queue(self, job: _Job) -> None:
        with self._motion_cond:
            if not self._closing.is_set():
                self._jobs.append(job)
                self._motion_cond.notify_all()
                return
        self._finish(job, _Ending(CANCELLED, CLOSING))

    def _stop_motion(
        self, reason: str, match: Callable[[_Job], bool] = _any_job
    ) -> None:
        """Stop the running job and drop the queued ones, where they match.

        The running move sees the reason before its next setpoint and ends in a
        hold; a dropped job is reported as cancelled.
        """
        with self._motion_cond:
            dropped = [job for job in self._jobs if match(job)]
            for job in dropped:
                self._jobs.remove(job)
            running = self._running
            if running is not None and match(running) and running.stop_reason is None:
                running.stop_reason = reason
            self._motion_cond.notify_all()
        for job in dropped:
            self._finish(job, _Ending(CANCELLED, reason))

    def _end_human_drive(self, reason: str) -> None:
        """End the human drive that is queued or runs; under the app's lock.

        The running one sees the reason before its next setpoint and ends in
        a hold. A queued one is dropped and told as a drive that ended with
        nothing sent: no STOP tells of it.
        """
        # The motion worker takes a job only under the app's lock, so the
        # drive that is queued here is the one the stop drops.
        with self._motion_cond:
            queued = any(job.is_human_drive for job in self._jobs)
        self._stop_motion(reason, _a_human_drive)
        if queued:
            self._after_drive(DriveEnd(_DRIVE_STOPPED, reason, 0, None))

    def _finish(self, job: _Job, ending: _Ending) -> None:
        _log.info("%s ended %s: %s", job.what, ending.status, ending.reason or "-")
        if job.reply is not None:
            job.reply.final(ending.status, ending.reason, ending.end_q, ending.detail)

    def _motion_loop(self) -> None:
        while True:
            with self._motion_cond:
                while not self._jobs and not self._closing.is_set():
                    self._motion_cond.wait()
            if self._closing.is_set():
                return  # close() cancelled what was queued
            # Under the app's lock a job starts either before a STOP, which then
            # stops it, or after one, which has dropped it.
            with self._lock, self._motion_cond:
                if not self._jobs:
                    continue
                job = self._jobs.popleft()
                self._running = job
            ending = self._run_job(job)
            # No longer busy when the final RESULT goes out, so the command that
            # follows it is not refused; idle only once it has gone out.
            with self._motion_cond:
                self._running = None
                self._reporting = True
            try:
                self._finish(job, ending)
            except Exception as error:
                # The worker lives on: later work is still run and answered.
                _log.exception("the final RESULT of %s raised", job.what)
                self._fault(f"robot app error reporting {job.what}: {error!r}")
            finally:
                with self._motion_cond:
                    self._reporting = False
                    self._motion_cond.notify_all()
            if ending.close:
                self.close()
                return

    def _run_job(self, job: _Job) -> _Ending:
        """Run one job; an error becomes a fault-stop and the worker lives on."""
        try:
            return job.run(job)
        except ArmFault as fault:
            self._fault(str(fault))
            return _Ending(FAILED, f"fault: {fault}")
        except Exception as error:
            _log.exception("%s raised on the motion worker", job.what)
            self._fault(f"robot app error in {job.what}: {error!r}")
            return _Ending(FAILED, f"robot app error: {error!r}")
        finally:
            if self._motion.get() is MotionState.MOVING:
                self._motion.set(MotionState.HOLD)

    def _run_move(
        self,
        job: _Job,
        m: Move,
        admitted: AuthorityState,
        halt_count: int,
        evidence_mono: float | None,
    ) -> _Ending:
        permits = self._config.permits.required_for_agent
        if evidence_mono is not None:
            # Without permits the MOVE's own evidence feeds the watchdog.
            self._checkers.agent.feed_watchdog(evidence_mono)
        if m.grasp_region is not None:
            self._grasp_region = m.grasp_region
        before = self._motion.get()
        self._motion.set(MotionState.MOVING)
        outcome: MoveOutcome | None = None
        try:
            outcome = self._executor.run(
                m.setpoints,
                m.start_q,
                m.start_tol_rad,
                m.period_s,
                Owner.AGENT,
                self._checkers.agent,
                admitted,
                m.command_id,
                self._gate if permits else None,
                m.grasp_region,
                job.should_stop,
                permit_arrived=self._gate.arrived,
                halt_count=halt_count,
                expires_mono_s=m.expires_mono_s,
            )
        finally:
            # With no setpoint sent the arm is where it was and keeps its state.
            unmoved = outcome is not None and outcome.index == 0
            self._motion.set(before if unmoved else MotionState.HOLD)
        self._after_move(outcome)
        reason = outcome.reason
        if outcome.status == "contact":
            # The contact's own text goes into the detail; a hold that was not
            # sent is part of the reason.
            _, failed, why = outcome.reason.partition(HOLD_FAILED)
            reason = f"contact{failed}{why}"
        detail = {
            "outcome": outcome.status,
            "sent": outcome.index,
            "total": len(m.setpoints),
        }
        if outcome.status == "contact":
            detail["contact"] = outcome.reason
        status = _MOVE_STATUS.get(outcome.status, FAILED)
        return _Ending(status, reason, outcome.end_q, detail)

    def _after_move(self, outcome: MoveOutcome) -> None:
        """What the end of an agent or recovery move means for what follows."""
        if outcome.status == "contact":
            # A hold at the measured pose, not latched.
            self._contact_hold = True
            if outcome.end_q is not None:
                self._checkers.agent.note_contact(outcome.end_q)
                self._checkers.recovery.note_contact(outcome.end_q)
            return
        if outcome.index > 0:
            # A contact hold ends when the arm is next moved, however that
            # move ended; the checkers keep their contact spots.
            self._contact_hold = False
            if not self._worker.busy():
                # The ROBOT_STATE that shows the move's end carries a jaw
                # reading newer than its last setpoint.
                self._jaw_state.refresh(self._arm)
        if outcome.reason.startswith("fault: "):
            # A driver fault or joint feedback lost during the move.
            self._fault(outcome.reason.removeprefix("fault: "))

    def _run_human_drive(self, job: _Job, admitted: AuthorityState) -> _Ending:
        """One human drive: every end of it is a hold with the motors on."""
        self._drive_ended = ""
        before = self._motion.get()
        self._motion.set(MotionState.MOVING)
        end: DriveEnd | None = None
        try:
            end = self._human_drive.run(admitted, job.should_stop)
        finally:
            # With no setpoint sent the arm is where it was and keeps its state.
            unmoved = end is not None and end.sent == 0
            self._motion.set(before if unmoved else MotionState.HOLD)
            if end is None:  # an error left the drive: a fault-stop follows
                with self._lock:
                    self._release_seen = None
                    self._drive_ended = _DRIVE_FAILED
        self._after_drive(end)
        detail = {"outcome": end.status, "sent": end.sent}
        return _Ending(end.status, end.reason, end.end_q, detail)

    def _after_drive(self, end: DriveEnd) -> None:
        """What the end of a human drive means for what follows."""
        # The EVENT has the note's first words once: DRIVE_ENDED says them.
        reason = end.reason.removeprefix(NOTE_SQUEEZE_AGAIN)
        why = reason or end.status
        with self._lock:
            if end.status == _DRIVE_CONTACT:
                # A hold at the measured pose, not latched. No contact spot is
                # noted: the spots are the agent's.
                self._contact_hold = True
            elif end.sent > 0:
                self._contact_hold = False  # the arm was moved since the contact
            if end.status != _DRIVE_RELEASED:
                self._release_seen = None  # a released grip comes first
                # The jaw motion ends with the drive: the trigger matches anew.
                self._jaw_follower.end(f"the human drive ended: {why}")
            self._drive_ended = why
        if end.status == _DRIVE_FAILED and end.reason.startswith("fault: "):
            # A driver fault or joint feedback lost during the drive. The
            # reason of another status may be a human program's note.
            self._fault(end.reason.removeprefix("fault: "))
        if end.status == _DRIVE_RELEASED:
            return
        # A STOP, an owner change and a lost robot connection have an EVENT of
        # their own; a drive that ended "stopped" without one has only this.
        if end.status == _DRIVE_STOPPED and not end.reason.startswith(_UNTOLD_STOPS):
            return
        if end.status == _DRIVE_CONTACT and not why.startswith(_DRIVE_CONTACT):
            why = f"{_DRIVE_CONTACT}: {why}"
        data = {"status": end.status, "reason": reason, "sent": end.sent}
        self._emit(EVENT_HUMAN, DRIVE_ENDED.format(why=why), data)

    def _run_handback(self, job: _Job, admitted: AuthorityState, why: str) -> _Ending:
        """The hand-back: the proof of rest, then the takeover ends.

        admitted is the authority as the hand-back was accepted. It reads the
        jaws and the joints and sends nothing.
        """
        problem = self._rest.proven_at_rest()
        with self._lock:
            now = self._authority.snapshot()
            stood = (now.stop_number, now.owner_number) == (
                admitted.stop_number,
                admitted.owner_number,
            )
            stopped = job.should_stop()
            number: int | None = admitted.owner_number
            if not stood:
                problem = _RACED  # the takeover stays open
            elif stopped is not None:
                # The robot connection it came over was lost, or the app closes.
                return _Ending(CANCELLED, stopped)
            elif problem is not None:
                jaws = problem.startswith(_JAWS_NOT_EMPTY)
                problem += OPEN_FIRST if jaws else REST_FIRST
            elif admitted.owner is Owner.HUMAN:
                number = self._authority.set_owner_if(
                    Owner.NOBODY,
                    f"hand-back: {why.strip() or 'no reason given'}",
                    stop_number=admitted.stop_number,
                    owner_number=admitted.owner_number,
                )
                problem = _RACED if number is None else None
            if problem is None:
                self._takeover_open = False
                detail = {"owner": Owner.NOBODY.value, "owner_number": number}
                self._emit(EVENT_OWNER, HANDED_BACK, dict(detail))
                return _Ending(COMPLETED, detail=detail)
        what = f"owner request {REQUEST_HANDBACK!r}"
        self._emit(EVENT_REFUSED, f"{what} refused: {problem}", {"reason": problem})
        return _Ending(REFUSED, problem)

    def _run_rest(
        self, job: _Job, source: str, stop_number: int, confirmed: bool = False
    ) -> _Ending:
        """One return to rest; rest.py tells the operator how it ended.

        confirmed is what the operator's command said of the path.
        """
        if self._motion.get() is MotionState.RELEASE_FAILED:
            # Nothing moves an arm whose motors may be off.
            data = {
                "source": source,
                "status": rest_module.REFUSED,
                "reason": MOTORS_UNKNOWN,
            }
            text = f"return to rest refused: {MOTORS_UNKNOWN}"
            self._emit(rest_module.EVENT_REST, text, data)
            return _Ending(
                REFUSED, MOTORS_UNKNOWN, None, {"outcome": rest_module.REFUSED}
            )
        # Only the agent can clear its contact spots, and it is refused while a
        # takeover is open: "path is clear" is the person's word for them.
        if confirmed and self.takeover_open:
            self._checkers.recovery.clear_contact_spots()
        outcome = self._rest.request(
            source,
            confirmed,
            stop_number_seen=stop_number,
            should_stop=job.should_stop,
        )
        move = self._rest.last_move
        if move is not None:
            self._after_move(move)
        ending = _Ending(
            _REST_STATUS.get(outcome.status, FAILED),
            outcome.reason,
            None if move is None else move.end_q,
            {"outcome": outcome.status},
        )
        if source != rest_module.TYPED or outcome.status != rest_module.AT_REST:
            return ending
        if not self._rest_powers_off:
            return ending
        # A typed rest ends as the functional rest does: jaws closed, motors off.
        if self.takeover_open:
            self._emit(rest_module.EVENT_REST, f"return to rest: {REST_KEEPS_MOTORS}")
            return dataclasses.replace(ending, detail={**ending.detail, "motors": "on"})
        refusal = self._power_down(job, stop_number)
        if refusal is None:
            return dataclasses.replace(
                ending, detail={**ending.detail, "motors": "off"}
            )
        reason = f"{AT_REST_MOTORS_ON}; release refused: {refusal}"
        self._emit(EVENT_REFUSED, f"rest: {reason}", {"reason": refusal})
        return dataclasses.replace(
            ending,
            status=FAILED,
            reason=reason,
            detail={**ending.detail, "motors": "on"},
        )

    def _power_down(
        self,
        job: _Job,
        stop_number: int,
        *,
        jaws_empty: bool = False,
        close: bool = True,
    ) -> str | None:
        """rest.py's power_down(): why the motors stay on, or None when off."""
        if self._motion.get() is MotionState.RELEASE_FAILED:
            # The release that failed had judged the jaws, and the driver has
            # been sent nothing since: it is sent nothing but the release.
            close, jaws_empty = False, True
        try:
            return self._rest.power_down(
                stop_number, job.should_stop, jaws_empty=jaws_empty, close=close
            )
        except ArmFault as fault:
            self._release_raised(fault)
            raise

    def _close_jaws_at_rest(self, should_stop: Callable[[], str | None]) -> str | None:
        """Close the jaws with the driver's close; why they were left, or None.

        The gripper worker's close before a torque release, waited for on
        this thread: the jaws close to their stop and are left in the light
        hold, not pushing, when the motors go off. A reason from should_stop
        ends the wait.
        """
        ended = threading.Event()
        outcomes: list[GripperOutcome] = []

        def done(outcome: GripperOutcome) -> None:
            outcomes.append(outcome)
            ended.set()

        refusal = self._worker.close_at_rest(done)
        if refusal is not None:
            return refusal
        waited = 0.0
        while not ended.wait(0.05):
            waited += 0.05
            stopped = should_stop()
            if stopped is not None:
                return stopped or "stop requested"
            if waited >= _JAW_CLOSE_WAIT_S:
                return f"no end of the jaw close within {_JAW_CLOSE_WAIT_S:g} s"
        outcome = outcomes[0]
        if outcome.status in ("reached", "sent", "stalled", "timeout"):
            return None
        return outcome.detail or f"the jaw close ended {outcome.status}"

    def _run_timed_return(self, job: _Job, stop_number: int) -> _Ending:
        # Told to stop by the loss of the connection it was queued under, it
        # does not begin on a later one.
        if not self._carrier.alive or job.should_stop() == LINK_LOST:
            return _Ending(REFUSED, self._no_timed_return())
        return self._run_rest(job, rest_module.WAIT, stop_number)

    def _no_timed_return(self) -> str:
        reason = f"no timed return to rest: {LINK_LOST}"
        self._emit(rest_module.EVENT_REST, f"{reason}; the arm holds")
        return reason

    def _release_raised(self, fault: ArmFault) -> None:
        """Note a torque release whose driver call raised; the caller passes it on.

        Inside rest.py's release only the arm port's release_torque() raises
        an ArmFault. With the arm port still saying that its motors are on,
        the driver may have switched some or all of them off before it
        failed: the motion state becomes "release_failed" and an ArmFault
        that says so is raised.
        """
        if not self._arm.torque_on():
            return  # the arm port knows that its motors are off
        self._motion.set(MotionState.RELEASE_FAILED)
        _log.error("a torque release raised: %s", TORQUE_UNKNOWN, exc_info=fault)
        raise ArmFault(f"{RELEASE_FAILED}: {fault}") from fault

    def _way_on(self, reason: str) -> str | None:
        """What a fault-stop's text names in place of a reset; None for a reset.

        After a torque release that failed the arm may not hold, and a reset
        does not bring it back.
        """
        if self._motion.get() is not MotionState.RELEASE_FAILED:
            return None
        # The failed release's own reason says that the torque is not known.
        return NEW_START if reason.startswith(RELEASE_FAILED) else MOTORS_UNKNOWN

    def _run_release(
        self,
        job: _Job,
        stop_number: int,
        *,
        jaws_empty: bool = False,
        close: bool = True,
    ) -> _Ending:
        """The end of the functional rest at measured rest: jaws closed, motors off.

        jaws_empty: the agent's rest tool found no load before it folded the
        arm, so the jaws are not judged here. close False: that tool has
        closed them itself.
        """
        reason = self._power_down(job, stop_number, jaws_empty=jaws_empty, close=close)
        if reason is None:
            return _Ending(COMPLETED, detail={"motors": "off"})
        unknown = self._motion.get() is MotionState.RELEASE_FAILED
        left = f"{TORQUE_UNKNOWN}; {NEW_START}" if unknown else "the motors stay on"
        text = f"release refused: {reason}; {left}"
        self._emit(EVENT_REFUSED, text, {"reason": reason})
        return _Ending(REFUSED, reason)

    def _run_power_on(self, job: _Job) -> _Ending:
        """Switch the motors on again after a release at rest.

        The arm connects as it did at the start: the driver refuses an arm it
        does not measure at rest. The jaws are referenced as at the start.
        """
        motion = self._motion.get()
        reason: str | None = None
        if motion is MotionState.RELEASE_FAILED:
            reason = MOTORS_UNKNOWN
        elif motion is not MotionState.REST_UNPOWERED or self._arm.torque_on():
            reason = rest_module.MOTORS_ARE_ON
        else:
            reason = job.should_stop()
        if reason is None:
            try:
                self._arm.connect()
            except ArmFault:
                raise
            except Exception as error:
                _log.warning("the motors were not switched on again", exc_info=True)
                reason = f"{type(error).__name__}: {error}"
        if reason is not None:
            text = rest_module.NOT_ON_AGAIN.format(why=reason)
            self._emit(EVENT_REFUSED, text, {"reason": reason})
            return _Ending(REFUSED, reason)
        state = self._after_connect("as the motors were switched on again")
        self._emit(
            rest_module.EVENT_REST,
            f"{rest_module.MOTORS_ON_AGAIN}: motion {state.value}",
            {"motion": state.value},
        )
        return _Ending(COMPLETED, detail={"motors": "on", "motion": state.value})

    def _run_shutdown(self, job: _Job, stop_number: int, confirmed: bool) -> _Ending:
        """Return to rest, release torque there, then let the app close.

        After a torque release that failed it starts no return to rest: the
        release is tried again, and the app closes only once it has worked.
        """
        ending = _Ending(COMPLETED)
        released = False  # by this shutdown; the motors may have been off before
        motion = self._motion.get()
        unknown = motion is MotionState.RELEASE_FAILED
        if motion is not MotionState.REST_UNPOWERED:
            if not unknown:
                ending = self._run_rest(
                    job, rest_module.SHUTDOWN, stop_number, confirmed
                )
            if ending.status != COMPLETED:
                reason = f"the return to rest ended {ending.status}: {ending.reason}"
                ending = dataclasses.replace(ending, reason=reason)
            else:
                # What told this job to stop since, such as the loss of the
                # robot connection it came over, keeps the motors on.
                refusal = self._power_down(job, stop_number)
                released = refusal is None
                if refusal is not None:
                    where = "at rest with the motors on"
                    if unknown:
                        where = f"a torque release failed and {TORQUE_UNKNOWN}"
                    reason = f"{where}; release refused: {refusal}"
                    ending = dataclasses.replace(ending, status=FAILED, reason=reason)
        if ending.status != COMPLETED:
            text = (
                f"shutdown {ending.status}: {ending.reason}; the robot app keeps "
                "running"
            )
            text += f"; {NEW_START}" if unknown else " and the arm holds"
            self._emit(EVENT_SHUTDOWN, text, {"status": ending.status})
            return ending
        # A release that the driver only sent is not reported as motors off.
        sent_only = released and not self._arm.release_confirmed()
        how = "torque release sent" if sent_only else "the motors are off"
        text = f"shutdown: {how} at rest; the robot app closes"
        self._emit(EVENT_SHUTDOWN, text, {"status": COMPLETED})
        return dataclasses.replace(ending, close=True)
