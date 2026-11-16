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
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..protocol.carrier import send
from ..protocol.messages import (
    COMMAND_TYPES,
    DRIVER_APAI_MOCK,
    DRIVER_REAL,
    DRIVER_SIM,
    MESSAGE_TYPES,
    CheckerState,
    Event,
    Gripper,
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
    CLOSE_UNTIL_CONTACT,
    JAWS_BUSY,
    PLAIN,
    GripperWorker,
    JawState,
)
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
EVENT_CONNECTION = "connection"
EVENT_REFUSED = "refused"
EVENT_SHUTDOWN = "shutdown"

THOR_TERMINAL = "thor_terminal"
# Joint feedback lost for longer than this in a hold is a fault-stop.
LOST_FEEDBACK_S = 0.35
# How many command ids keep their newest RESULT for a repeated command.
RESULT_IDS_KEPT = 256
# What request_rest(None) asks the Thor terminal's confirm().
PATH_CLEAR_PROMPT = "Return to rest: is the path to the rest pose clear?"

MOTORS_OFF = "the motors are off; restart the robot app"
CONFIRM_PATH = rest_module.CONFIRM_PATH
CONFIRM_OPEN = "confirm the jaw opening"
CONFIRM_JAWS_EMPTY = "confirm that the jaws are empty"
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
_RACED = "a STOP or an owner change arrived while the request was checked"

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
    # Written under the motion lock; read by the move before every setpoint.
    stop_reason: str | None = None

    def should_stop(self) -> str | None:
        return self.stop_reason


def _any_job(job: _Job) -> bool:
    return True


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
        status = outcome.status
        if status == "reached":
            result = COMPLETED
        elif status == "stalled" and self._mode == CLOSE_UNTIL_CONTACT:
            result = COMPLETED  # stalling on the object is what was asked for
        elif status in ("cancelled", "refused"):
            result = status
        else:  # stalled on a plain command, timeout, failed
            result = FAILED
        reason = outcome.detail
        if not reason and result == FAILED:
            reason = _JAW_REASON.get(status, f"the jaw command ended {status}")
        detail = {"outcome": status, "opening": outcome.opening}
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


class RobotApp:
    """The one program that talks to the motors.

    A dispatcher thread receives link messages and runs the supervisor step; a
    motion worker runs moves, returns to rest and releases; the gripper worker
    runs jaw commands. STOP, fault-stop, the assistance wait's step and every
    start of motion work happen under one reentrant lock, so a STOP always wins.
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
    ) -> None:
        """confirm(prompt) is the Thor terminal's typed confirmation.

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
        # The newest VIEW_STATUS of this robot connection and the app-clock
        # time it arrived; one value, so a reader never sees half of a change.
        self._view: tuple[ViewStatus, float] | None = None
        self._permit_refusal: str | None = None
        self._dropped: set[str] = set()  # kinds of link messages not taken
        self._inbox = Inbox()  # taken off the link, not yet handled
        self._lines = LineLimit(clock)  # for log lines a link can make repeat
        longest_move = config.executor.max_setpoints * (
            config.n_joints * MOVE_VALUE_BYTES + len("[],")
        )
        self._payload_limits = {Move.TYPE: MOVE_BASE_BYTES + longest_move}
        self._fault_while_latched = ""
        self._errors: dict[str, tuple[str, int]] = {}
        self._published: tuple[Any, ...] | None = None

        self._connect_called = False
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
        )
        self._jaw_state = JawState(config.gripper, clock)
        self._worker = GripperWorker(arm, self._jaw_state, self._authority, clock)
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
            lambda: self._contact_hold,
            lambda: False,  # rest.py asks the executor itself
            jaws_busy=self._worker.busy,
        )
        self._idle = IdleReader(arm, self._history, config.state, clock)
        self._publisher = StatePublisher(
            arm,
            self._history,
            self._authority.snapshot,
            self._executor.progress,
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
        }
        # Type number -> class of the link messages the app takes.
        self._takes = {cls.TYPE: cls for cls in self._handlers}
        self._supervisor_parts: tuple[tuple[str, Callable[[], None]], ...] = (
            ("the assistance wait", self._step_wait),
            ("the idle joint read", self._step_idle),
            ("the fault poll", self._poll_faults),
            ("the robot connection check", self._check_link),
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

    @property
    def motion_state(self) -> MotionState:
        return self._motion.get()

    def snapshot(self) -> AuthorityState:
        """Session id, control owner, owner number, stop number and stop level."""
        return self._authority.snapshot()

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
        rest = self._config.rest
        at_rest = releasable = False
        try:
            off_rest = self._off_rest(self._read_pose())
            at_rest = off_rest[1] <= rest.start_rest_tol_rad
            releasable = off_rest[1] <= rest.rest_tol_rad
            if at_rest and not releasable:
                _log.warning(
                    "the arm lies at rest at start (within %g rad) but joint index "
                    "%d is %.3f rad from the rest pose, outside the %g rad of a "
                    "release: the motion state is hold until a typed rest",
                    rest.start_rest_tol_rad,
                    *off_rest,
                    rest.rest_tol_rad,
                )
        except ReadFailed as failed:
            _log.warning("the pose could not be read at start: %s", failed)
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
        self._send(self._welcome())
        self._dispatcher.start()
        self._motion_worker.start()
        self._started = True
        _log.info(
            "robot app started: session %s, motion state %s",
            self._authority.session_id,
            state,
        )

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

    def request_rest(
        self, confirmed: bool | None, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue a return to rest typed at the Thor terminal; why not, or None.

        confirmed says that the operator typed that the path is clear; None asks
        confirm() on this thread. on_done gets the final RESULT exactly once,
        also for a refusal; it must not block or call back into the app.
        """
        if confirmed is None:
            confirmed = self._ask(PATH_CLEAR_PROMPT)
        reply = self._local_reply("rest", on_done)
        with self._lock:
            reason = self._not_running() or (None if confirmed else CONFIRM_PATH)
            reason = reason or self._busy()
            if reason is None:
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

    def request_release(self, on_done: DoneFn | None = None) -> str | None:
        """Queue a torque release typed at the Thor terminal; why not, or None.

        The motors go off only at measured rest with the jaws proven empty;
        otherwise they stay on. on_done is as for request_rest().
        """
        reply = self._local_reply("release", on_done)
        with self._lock:
            reason = self._not_running() or self._busy()
            if reason is None:
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

        The age the newest VIEW_STATUS reported plus the time since it
        arrived. None when none has arrived on this robot connection, when it
        said the view is not live, or when its age is not 0 or more.
        """
        view = self._view
        if view is None:
            return None
        status, arrived = view
        if not status.live or not status.age_s >= 0.0:
            return None
        return status.age_s + max(0.0, now - arrived)

    def shutdown_moves_arm(self) -> bool:
        """Whether a shutdown asked for now would first move the arm to rest.

        By the motion state: False while the arm holds at rest or the motors
        are off. Such a shutdown needs the confirmation that the path is clear.
        """
        at_rest = (MotionState.REST_HOLD, MotionState.REST_UNPOWERED)
        return self._motion.get() not in at_rest

    def request_shutdown(
        self, confirmed: bool | None, on_done: DoneFn | None = None
    ) -> str | None:
        """Queue a shutdown typed at the Thor terminal; why not, or None.

        The arm returns to rest, torque is released there and the app closes. If
        the return or the release is refused the app keeps running and holding.
        confirmed and on_done are as for request_rest(); the confirmation is
        needed, and None asks for it, only when the arm has to move.
        """
        if confirmed is None:
            confirmed = self.shutdown_moves_arm() and self._ask(PATH_CLEAR_PROMPT)
        reply = self._local_reply("shutdown", on_done)
        with self._lock:
            reason = self._not_running() or self._shutdown_unconfirmed(confirmed)
            reason = reason or self._busy()
            if reason is None:
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

        self._send(result, plain, cut, via=via)

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
        arrived_on = self._carrier

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

    def _sort(self, msg_type: int, payload: bytes) -> None:
        """One link message, just off the link: served, kept for later or dropped.

        A STOP is served now. What the app has no handler for is dropped
        undecoded. A payload over its size limit is refused now, unparsed, and
        so is one the inbox has no room for.
        """
        if msg_type == Stop.TYPE:
            self._handle_payload(msg_type, payload)
        elif msg_type not in self._takes:
            self._not_taken(msg_type)
        elif len(payload) > self._payload_limit(msg_type):
            self._handle_payload(msg_type, payload)  # refused where it is decoded
        elif not self._inbox.add(msg_type, payload):
            problem = f"busy: {len(self._inbox)} link messages wait at the robot app"
            self._undecoded(msg_type, self._unread_id(msg_type, payload), problem)

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
        self._idle.step(self._clock.monotonic())

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

    def _check_link(self) -> None:
        if self._link_lost or self._carrier.alive:
            return
        with self._lock:
            if self._link_lost or self._carrier.alive:
                return  # a new carrier was attached meanwhile
            self._link_lost = True  # handled once per carrier; a dead link stays dead
            # What came over the lost connection and was not handled goes with it.
            self._inbox = Inbox()
            self._view = None
            self._stop_motion(LINK_LOST, lambda job: job.needs_link)
            self._worker.stop_now()
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
        if not carrier.alive or (key == self._published and not forced):
            self._publisher.maybe_publish(carrier, now)
            return
        self._published = key
        self._send(self._publisher.build(now), via=carrier)

    def _fault(self, reason: str) -> None:
        """Raise a fault-stop, unless one is latched already."""
        with self._lock:
            if self._authority.stop_level is not StopLevel.FAULT:
                self._stops.fault(reason)
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
        # Only shown in ROBOT_STATE: nothing acts on the live view's age yet.
        self._view = (m, self._clock.monotonic())

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
            )
            if reason is not None:
                reply.final(REFUSED, reason)
                return
            done = _JawReply(reply, m.mode)
            done.queued(
                self._worker.submit(m.opening, m.max_effort, m.mode, Owner.AGENT, done)
            )

    def _on_open(self, m: Open) -> None:
        reply = self._link_reply(m.command_id)
        with self._lock:
            reason = self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.NOBODY
            )
            if reason is None and not m.confirmed:
                reason = CONFIRM_OPEN
            reason = reason or self._motors_off() or self._busy()
            if reason is not None:
                self._refuse(reply, "open", reason)
                return
            done = _JawReply(reply, "plain")
            done.queued(self._worker.open_guarded(done, Owner.NOBODY))

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
        elif m.request == "human":
            reason = "human path not built yet"
        else:
            reason = f"unknown request {m.request!r}"
        if reason is not None:
            self._refuse(reply, f"owner request {m.request!r}", reason)

    def _new_task(self, m: OwnerRequest, reply: _Reply) -> str | None:
        """OWNER_REQUEST "agent": why it is refused, or None once it was answered."""
        with self._lock:
            reason = self._session_mismatch(m.session_id) or self._motors_off()
            reason = reason or self._authority.admit(
                m.session_id, m.owner_number, m.stop_number, Owner.NOBODY
            )
            reason = reason or self._busy()
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
            self._checkers.agent.check_pose(pose)
        except Refusal as refusal:
            return f"type rest first: {refusal.reason}"
        with self._lock:
            # What began or ended while the pose was read.
            reason = self._motors_off() or self._busy()
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

    # Admission helpers

    def _session_mismatch(self, session_id: str) -> str | None:
        if session_id != self._authority.session_id:
            return "session id does not match"
        return None

    def _motors_off(self) -> str | None:
        if self._motion.get() is MotionState.REST_UNPOWERED:
            return MOTORS_OFF
        return None

    def _not_running(self) -> str | None:
        return None if self.running else NOT_RUNNING

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
        if outcome.reason.startswith("fault: "):
            # A driver fault or joint feedback lost during the move.
            self._fault(outcome.reason.removeprefix("fault: "))

    def _run_rest(
        self, job: _Job, source: str, stop_number: int, confirmed: bool = False
    ) -> _Ending:
        """One return to rest; rest.py tells the operator how it ended.

        confirmed is what the operator's command said of the path.
        """
        outcome = self._rest.request(
            source,
            confirmed,
            stop_number_seen=stop_number,
            should_stop=job.should_stop,
        )
        move = self._rest.last_move
        if move is not None:
            self._after_move(move)
        return _Ending(
            _REST_STATUS.get(outcome.status, FAILED),
            outcome.reason,
            None if move is None else move.end_q,
            {"outcome": outcome.status},
        )

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

    def _run_release(self, job: _Job, stop_number: int) -> _Ending:
        reason = self._rest.release(stop_number, job.should_stop)
        if reason is None:
            return _Ending(COMPLETED)
        text = f"release refused: {reason}; the motors stay on"
        self._emit(EVENT_REFUSED, text, {"reason": reason})
        return _Ending(REFUSED, reason)

    def _run_shutdown(self, job: _Job, stop_number: int, confirmed: bool) -> _Ending:
        """Return to rest, release torque there, then let the app close."""
        ending = _Ending(COMPLETED)
        released = False  # by this shutdown; the motors may have been off before
        if self._motion.get() is not MotionState.REST_UNPOWERED:
            ending = self._run_rest(job, rest_module.SHUTDOWN, stop_number, confirmed)
            if ending.status != COMPLETED:
                reason = f"the return to rest ended {ending.status}: {ending.reason}"
                ending = dataclasses.replace(ending, reason=reason)
            else:
                # What told this job to stop since, such as the loss of the
                # robot connection it came over, keeps the motors on.
                refusal = self._rest.release(stop_number, job.should_stop)
                released = refusal is None
                if refusal is not None:
                    reason = f"at rest with the motors on; release refused: {refusal}"
                    ending = dataclasses.replace(ending, status=FAILED, reason=reason)
        if ending.status != COMPLETED:
            text = (
                f"shutdown {ending.status}: {ending.reason}; the robot app keeps "
                "running and the arm holds"
            )
            self._emit(EVENT_SHUTDOWN, text, {"status": ending.status})
            return ending
        # A release that the driver only sent is not reported as motors off.
        sent_only = released and not self._arm.release_confirmed()
        how = "torque release sent" if sent_only else "the motors are off"
        text = f"shutdown: {how} at rest; the robot app closes"
        self._emit(EVENT_SHUTDOWN, text, {"status": COMPLETED})
        return dataclasses.replace(ending, close=True)
