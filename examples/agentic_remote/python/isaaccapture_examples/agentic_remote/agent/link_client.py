# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's client of the robot link service's local API.

Every command of a task quotes the pinned numbers: the session id, owner number
and stop number the agent got when it became control owner. They are never
refreshed from a newer ROBOT_STATE, so the robot app refuses a command of an
ownership that has ended.

Timeouts are real seconds. Ages of a ROBOT_STATE and the estimate of the Thor's
clocks are taken on the `monotonic` the link was built with.
"""

from __future__ import annotations

import logging
import threading
import time
import types
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..protocol.messages import (
    CheckerState,
    Gripper,
    Move,
    OwnerRequest,
    Permit,
    Result,
    RobotState,
    Welcome,
)
from ..testing.client import ACCEPTED, WorkstationClient

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from ..protocol.messages import GraspRegion, Message

_log = logging.getLogger("isaaccapture.examples.agentic_remote.link_client")

# The reason of the RESULTs the robot link service makes when the robot
# connection is down (LOST in link_service/service.py).
LOST = "robot connection lost"
# Command ids whose RESULTs are kept.
RESULTS_KEPT = 256
# ROBOT_STATEs the estimate of the Thor's monotonic clock looks back over.
OFFSETS_KEPT = 64
# A change of the Thor's wall clock against its monotonic clock above this,
# between two ROBOT_STATEs, counts as a step of the wall clock.
CLOCK_STEP_S = 0.02
# The robot app cuts a reason and a sender to these lengths; so does the link.
MAX_REASON_CHARS = 256
MAX_SENDER_CHARS = 64
# How often a wait looks at the link, in real seconds.
_LOOK_S = 0.1


class LinkLost(RuntimeError):
    """The robot link service or the robot connection is gone."""


class CommandTimeout(RuntimeError):
    """No RESULT in the time given; command_id says for which command."""

    def __init__(self, text: str, command_id: str = "") -> None:
        super().__init__(text)
        self.command_id = command_id


class StaleState(RuntimeError):
    """There is no ROBOT_STATE, or the newest one is older than asked."""


@dataclass(frozen=True)
class Pinned:
    """The numbers of one ownership of the arm."""

    session_id: str
    owner_number: int
    stop_number: int


class _Kept:
    """The RESULTs kept for one command."""

    __slots__ = ("final", "first", "states_at_final")

    def __init__(self) -> None:
        self.first: Result | None = None
        self.final: Result | None = None
        self.states_at_final = 0  # ROBOT_STATEs received before the final RESULT


class AgentLink:
    """One agent program's connection to the robot app.

    Link messages are taken in on the receive thread of the WorkstationClient
    it wraps; the link starts no thread of its own. Every method may be called
    from any thread.
    """

    def __init__(
        self,
        carrier: Carrier,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        name: str = "agent",
        clock_margin_s: float = 0.02,
        state_timeout_s: float = 1.0,
    ) -> None:
        self._carrier = carrier
        self._monotonic = monotonic
        self._clock_margin_s = float(clock_margin_s)
        self._state_timeout_s = float(state_timeout_s)
        self._cond = threading.Condition()
        self._closed = False
        self._welcome: Welcome | None = None
        self._state: RobotState | None = None
        self._state_at = 0.0
        self._states = 0  # ROBOT_STATEs received so far
        self._changes = 0  # ROBOT_STATEs and RESULTs received so far
        # Local arrival time minus clock_mono_s, of the newest ROBOT_STATEs.
        self._offsets: deque[float] = deque(maxlen=OFFSETS_KEPT)
        self._clock_steps = 0
        self._clock_step_at: float | None = None
        self._results: OrderedDict[str, _Kept] = OrderedDict()
        self._pinned: Pinned | None = None
        self._pin_lost = ""  # why, once a RESULT said the robot connection is lost
        self._listeners: tuple[Callable[[RobotState], None], ...] = ()
        self._welcome_listeners: tuple[Callable[[Welcome], None], ...] = ()
        self._client = WorkstationClient(
            carrier,
            types.SimpleNamespace(monotonic=monotonic),
            name=name,
            listeners=(self._take,),
        )

    @classmethod
    def attach(
        cls,
        socket_path: str | None = None,
        *,
        timeout_s: float = 10.0,
        **options: Any,
    ) -> AgentLink:
        """Attach to the robot link service in the role "agent".

        Blocks at most timeout_s. LinkLost when nothing listens at the path, the
        service does not answer or it refuses the client (its reason is kept).
        options are those of the constructor.
        """
        from ..link_service.local import connect_local

        name = options.get("name", "agent")
        try:
            carrier = connect_local(socket_path, "agent", name, timeout_s)
        except OSError as error:  # also LocalRefused and TimeoutError
            where = socket_path or "its default path"
            raise LinkLost(
                f"cannot attach to the robot link service at {where}: {error}"
            ) from error
        return cls(carrier, **options)

    def close(self) -> None:
        """Close the local socket; sends nothing.

        While ROBOT_STATE shows the agent as control owner, the service answers
        the closed socket with its own help request ("agent program gone").
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._client.close()
        self._carrier.close()

    # What the robot app sent

    def wait_ready(self, timeout_s: float) -> Welcome:
        """Wait for the WELCOME and a ROBOT_STATE of its session with every joint valid.

        A robot connection that is down is waited for. LinkLost at once when the
        service is gone, and after timeout_s when the robot connection is still
        down; TimeoutError otherwise.
        """
        end = time.monotonic() + timeout_s
        with self._cond:
            while True:
                gone = self._gone()
                if gone:
                    raise LinkLost(gone)
                if _ready(self._welcome, self._state):
                    return self._welcome
                left = end - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(min(left, _LOOK_S))
            have = "no WELCOME" if self._welcome is None else "a WELCOME"
            state = "no ROBOT_STATE" if self._state is None else "a ROBOT_STATE"
        down = self._down()
        if down:
            raise LinkLost(down)
        raise TimeoutError(
            f"the robot app was not ready within {timeout_s:g} s: {have} and {state} "
            "arrived; every joint needs a valid reading"
        )

    @property
    def welcome(self) -> Welcome | None:
        with self._cond:
            return self._welcome

    def state(self, max_age_s: float | None = None) -> RobotState:
        """The newest ROBOT_STATE; never blocks.

        LinkLost while the link is lost; StaleState before the first one and
        when it is older than max_age_s.
        """
        return self.state_and_arrival(max_age_s)[0]

    def state_and_arrival(
        self, max_age_s: float | None = None
    ) -> tuple[RobotState, float]:
        """The newest ROBOT_STATE and the local monotonic time it arrived.

        Both are of one state. Never blocks; raises as state() does.
        """
        down = self._down()
        if down:
            raise LinkLost(down)
        with self._cond:
            state, at = self._state, self._state_at
        if state is None:
            raise StaleState("no ROBOT_STATE has arrived yet")
        if max_age_s is not None:
            age = self._monotonic() - at
            if age > max_age_s:
                raise StaleState(
                    f"the newest ROBOT_STATE is {age:.2f} s old, over {max_age_s:g} s"
                )
        return state, at

    def state_arrived_s(self) -> float:
        """Local monotonic time the newest ROBOT_STATE arrived; StaleState if none."""
        with self._cond:
            if self._state is None:
                raise StaleState("no ROBOT_STATE has arrived yet")
            return self._state_at

    def add_state_listener(self, fn: Callable[[RobotState], None]) -> None:
        """Call fn(state) for every ROBOT_STATE from now on.

        fn runs on the receive thread and must not block. clock_steps already
        counts a step of the Thor's wall clock that this state showed.
        """
        with self._cond:
            self._listeners = (*self._listeners, fn)

    def add_welcome_listener(self, fn: Callable[[Welcome], None]) -> None:
        """Call fn(welcome) for every WELCOME from now on.

        fn runs on the receive thread, after `welcome` shows that WELCOME, and
        must not block.
        """
        with self._cond:
            self._welcome_listeners = (*self._welcome_listeners, fn)

    def wait_change(self, timeout_s: float) -> None:
        """Return at the next ROBOT_STATE or RESULT, at a lost link or after timeout_s."""
        end = time.monotonic() + timeout_s
        with self._cond:
            seen = self._changes
            while self._changes == seen and not self._down():
                left = end - time.monotonic()
                if left <= 0:
                    return
                self._cond.wait(min(left, _LOOK_S))

    @property
    def service_status(self) -> Any:
        """The service's newest LOCAL_STATUS; None on a carrier that has none."""
        return getattr(self._carrier, "status", None)

    # Ownership

    def pin(self, pinned: Pinned) -> None:
        """Set the numbers every agent command quotes from now on."""
        if not isinstance(pinned, Pinned):
            raise TypeError(f"pin() takes a Pinned, got {type(pinned).__name__}")
        with self._cond:
            self._pinned = pinned
            self._pin_lost = ""

    def unpin(self) -> None:
        with self._cond:
            self._pinned = None
            self._pin_lost = ""

    @property
    def pinned(self) -> Pinned | None:
        with self._cond:
            return self._pinned

    def lost_reason(self) -> str | None:
        """Why the pinned ownership no longer stands; None while it does."""
        down = self._down()
        with self._cond:
            pinned, state, at = self._pinned, self._state, self._state_at
            welcome, pin_lost = self._welcome, self._pin_lost
        if pinned is None:
            return "no ownership is pinned"
        if down:
            return down
        if pin_lost:
            return pin_lost
        if welcome is not None and welcome.session_id != pinned.session_id:
            return (
                f"the robot app has a new session ({welcome.session_id}); the "
                f"ownership is of session {pinned.session_id}"
            )
        if state is None:
            return "no ROBOT_STATE has arrived yet"
        age = self._monotonic() - at
        if age > self._state_timeout_s:
            return (
                f"the newest ROBOT_STATE is {age:.2f} s old, over "
                f"{self._state_timeout_s:g} s"
            )
        if state.session_id != pinned.session_id:
            return (
                f"the robot app has a new session ({state.session_id}); the "
                f"ownership is of session {pinned.session_id}"
            )
        if state.owner != "agent":
            return f"the control owner is {state.owner}, not the agent"
        if state.owner_number != pinned.owner_number:
            return (
                f"the owner number is {state.owner_number}; the ownership has "
                f"{pinned.owner_number}"
            )
        if state.stop_number != pinned.stop_number:
            return (
                f"the stop number is {state.stop_number}; the ownership has "
                f"{pinned.stop_number}"
            )
        if state.stop_level != "none":
            reason = f": {state.fault_reason}" if state.fault_reason else ""
            return f"the stop level is {state.stop_level}{reason}"
        if getattr(self.service_status, "stop_unconfirmed", False):
            return "a STOP was sent and the robot app has not shown it yet"
        return None

    # The Thor's clocks

    def thor_now(self) -> tuple[float, float]:
        """An estimate of the Thor's (monotonic, wall) clocks now; never blocks.

        It runs behind the true clocks by the shortest transit time among the
        newest ROBOT_STATEs, so an expiry made from it ends early, never late.
        StaleState before the first ROBOT_STATE.
        """
        with self._cond:
            state = self._state
            offset = min(self._offsets, default=0.0)
        if state is None:
            raise StaleState("no ROBOT_STATE has arrived yet: no Thor clock")
        mono = self._monotonic() - offset
        return mono, mono + (state.clock_wall_s - state.clock_mono_s)

    def wall_age_s(self, thor_wall_s: float) -> float:
        """The Thor's wall clock now minus thor_wall_s, plus the clock margin."""
        return self.thor_now()[1] - thor_wall_s + self._clock_margin_s

    def wall_to_thor_mono(self, thor_wall_s: float) -> float:
        """A Thor wall time on its monotonic clock, by the newest clock pair alone."""
        with self._cond:
            state = self._state
        if state is None:
            raise StaleState("no ROBOT_STATE has arrived yet: no Thor clock")
        return thor_wall_s - (state.clock_wall_s - state.clock_mono_s)

    @property
    def clock_steps(self) -> int:
        """Times the Thor's wall clock jumped against its monotonic clock."""
        with self._cond:
            return self._clock_steps

    def clock_step_at_s(self) -> float | None:
        """Local monotonic time the newest such jump was seen; None if never."""
        with self._cond:
            return self._clock_step_at

    # Commands

    def new_id(self) -> str:
        """A command id no other command of this link has."""
        return self._client.new_id()

    def request_owner(
        self,
        request: str,
        *,
        reason: str = "",
        timed_return: bool = True,
        quote: Pinned | None = None,
        timeout_s: float = 5.0,
    ) -> Result:
        """OWNER_REQUEST "agent", "help_needed" or "nobody"; blocks for its RESULT.

        It quotes `quote`, else the newest ROBOT_STATE, which must be fresh
        (StaleState). LinkLost; CommandTimeout.
        """
        if quote is None:
            state = self.state(self._state_timeout_s)
            quote = Pinned(state.session_id, state.owner_number, state.stop_number)
        command_id = self.new_id()
        self._send(
            OwnerRequest(
                command_id,
                **_numbers(quote),
                request=request,
                reason=_cut(reason, MAX_REASON_CHARS),
                timed_return=bool(timed_return),
            )
        )
        return self._take_final(command_id, timeout_s)

    def send_move(
        self,
        move_id: str,
        start_q: Sequence[float],
        start_tol_rad: float,
        setpoints: Sequence[Sequence[float]],
        grasp_region: GraspRegion | None,
        evidence_wall_s: float | None,
        expires_in_s: float,
    ) -> None:
        """Send one MOVE with the pinned numbers; never blocks.

        The period is the WELCOME's and the expiry is expires_in_s from now on
        the Thor's monotonic clock. RuntimeError without a pin; LinkLost;
        StaleState before the WELCOME and the first ROBOT_STATE.
        """
        pinned = self._need_pin("MOVE")
        welcome = self.welcome
        if welcome is None:
            raise StaleState("no WELCOME has arrived yet: no setpoint period")
        mono, _ = self.thor_now()
        self._send(
            Move(
                command_id=move_id,
                **_numbers(pinned),
                start_q=_row(start_q),
                start_tol_rad=float(start_tol_rad),
                setpoints=[_row(row) for row in setpoints],
                period_s=welcome.setpoint_period_s,
                grasp_region=grasp_region,
                evidence_wall_s=None
                if evidence_wall_s is None
                else float(evidence_wall_s),
                expires_mono_s=mono + float(expires_in_s),
            )
        )

    def send_permit(
        self,
        move_id: str,
        through_index: int,
        evidence_wall_s: float,
        life_s: float,
        pinned: Pinned | None = None,
    ) -> None:
        """Send one PERMIT; never blocks.

        It quotes `pinned`, the numbers its move began with, else the pinned
        numbers, and ends life_s from now on the Thor's monotonic clock.
        RuntimeError without either; nothing else is raised: a PERMIT that
        cannot go out is dropped, and the robot app then ends the move by
        itself.
        """
        if pinned is None:
            pinned = self._need_pin("PERMIT")
        try:
            if self._down():
                return
            mono, _ = self.thor_now()
            self._client.send(
                Permit(
                    move_id=move_id,
                    **_numbers(pinned),
                    through_index=int(through_index),
                    evidence_wall_s=float(evidence_wall_s),
                    expires_mono_s=mono + float(life_s),
                )
            )
        except Exception as error:
            _log.warning("PERMIT for %s not sent: %s", move_id, error)

    def gripper(
        self, opening: float, max_effort: float, mode: str, timeout_s: float
    ) -> Result:
        """GRIPPER with the pinned numbers; blocks until its final RESULT.

        RuntimeError without a pin; LinkLost; CommandTimeout.
        """
        pinned = self._need_pin("GRIPPER")
        command_id = self.new_id()
        self._send(
            Gripper(
                command_id,
                **_numbers(pinned),
                opening=float(opening),
                max_effort=float(max_effort),
                mode=mode,
            )
        )
        return self._take_final(command_id, timeout_s)

    def checker_state(self, action: str) -> str:
        """Send CHECKER_STATE with the pinned numbers; returns its command id.

        Does not wait. RuntimeError without a pin; LinkLost.
        """
        pinned = self._need_pin("CHECKER_STATE")
        command_id = self.new_id()
        self._send(CheckerState(command_id, **_numbers(pinned), action=action))
        return command_id

    def stop(self, reason: str, sender: str) -> None:
        """Send STOP; never raises. It needs no ownership and has no RESULT."""
        try:
            self._client.stop(
                reason=_cut(reason, MAX_REASON_CHARS),
                sender=_cut(sender, MAX_SENDER_CHARS),
            )
        except Exception:
            _log.exception("STOP could not be sent")

    def wait_first(self, command_id: str, timeout_s: float) -> Result:
        """The first RESULT of this command, "accepted" or final.

        CommandTimeout; LinkLost when the link is lost and no RESULT is here.
        """
        return self._wait_result(command_id, timeout_s, final=False)

    def wait_final(self, command_id: str, timeout_s: float) -> Result:
        """The final RESULT of this command.

        The service's own "robot connection lost" is returned like any other
        final RESULT. CommandTimeout; LinkLost when the link is lost and no
        final RESULT is here (looked at every 0.1 s).
        """
        return self._wait_result(command_id, timeout_s, final=True)

    def final_result(self, command_id: str) -> Result | None:
        """The final RESULT of this command if it is here; never blocks."""
        with self._cond:
            kept = self._results.get(command_id)
            return None if kept is None else kept.final

    def wait_reflected(self, command_id: str, timeout_s: float) -> RobotState | None:
        """A ROBOT_STATE that holds what this command changed; None after timeout_s.

        That is the first one received after the command's final RESULT whose
        last_result names the command, or the second one received after it.
        Never raises: None also when the link is lost.
        """
        end = time.monotonic() + timeout_s
        with self._cond:
            while True:
                kept = self._results.get(command_id)
                if kept is not None and kept.final is not None:
                    since = self._states - kept.states_at_final
                    state = self._state
                    if since >= 2 or (since == 1 and _names(state, command_id)):
                        return state
                left = end - time.monotonic()
                if left <= 0 or self._down():
                    return None
                self._cond.wait(min(left, _LOOK_S))

    def forget(self, command_id: str) -> None:
        """Drop the RESULTs kept for this command."""
        with self._cond:
            self._results.pop(command_id, None)
        self._client.forget(command_id)

    # Internals

    def _gone(self) -> str | None:
        """Why the service cannot be reached; None while it can."""
        if self._closed:
            return "the agent link was closed"
        if not self._carrier.alive:
            why = getattr(self._carrier, "close_reason", "")
            return "the robot link service is gone" + (f": {why}" if why else "")
        return None

    def _down(self) -> str | None:
        """Why the link is lost; None while it stands."""
        gone = self._gone()
        if gone:
            return gone
        status = self.service_status
        if status is not None and not status.robot_connected:
            why = status.close_reason
            return LOST + (f": {why}" if why else ": the robot app is not connected")
        return None

    def _need_pin(self, what: str) -> Pinned:
        pinned = self.pinned
        if pinned is None:
            raise RuntimeError(f"no ownership is pinned; a {what} needs one")
        return pinned

    def _send(self, message: Message) -> None:
        down = self._down()
        if down:
            raise LinkLost(down)
        self._client.send(message)

    def _take_final(self, command_id: str, timeout_s: float) -> Result:
        result = self.wait_final(command_id, timeout_s)
        self.forget(command_id)
        return result

    def _wait_result(self, command_id: str, timeout_s: float, final: bool) -> Result:
        end = time.monotonic() + timeout_s
        with self._cond:
            while True:
                kept = self._results.get(command_id)
                if kept is not None:
                    result = kept.final if final else kept.first
                    if result is not None:
                        return result
                down = self._down()
                if down:
                    raise LinkLost(down)
                left = end - time.monotonic()
                if left <= 0:
                    kind = "final RESULT" if final else "RESULT"
                    raise CommandTimeout(
                        f"no {kind} for {command_id} within {timeout_s:g} s",
                        command_id,
                    )
                self._cond.wait(min(left, _LOOK_S))

    def _take(self, message: Message) -> None:
        """One received link message; runs on the client's receive thread."""
        if isinstance(message, RobotState):
            self._take_state(message)
        elif isinstance(message, Result):
            self._take_result(message)
        elif isinstance(message, Welcome):
            with self._cond:
                self._welcome = message
                listeners = self._welcome_listeners
                self._cond.notify_all()
            for listener in listeners:
                try:
                    listener(message)
                except Exception:
                    _log.exception("a WELCOME listener failed")

    def _take_state(self, state: RobotState) -> None:
        arrived = self._monotonic()
        with self._cond:
            earlier = self._state
            if earlier is not None:
                if (
                    earlier.session_id != state.session_id
                    or state.clock_mono_s < earlier.clock_mono_s
                ):
                    self._offsets.clear()  # another run of the Thor's clock
                was = earlier.clock_wall_s - earlier.clock_mono_s
                now = state.clock_wall_s - state.clock_mono_s
                if abs(now - was) > CLOCK_STEP_S:
                    self._clock_steps += 1
                    self._clock_step_at = arrived
                    _log.warning(
                        "the Thor's wall clock stepped by %+.3f s against its "
                        "monotonic clock",
                        now - was,
                    )
            self._offsets.append(arrived - state.clock_mono_s)
            self._state, self._state_at = state, arrived
            self._states += 1
            self._changes += 1
            listeners = self._listeners
            self._cond.notify_all()
        for listener in listeners:
            try:
                listener(state)
            except Exception:
                _log.exception("a state listener failed")

    def _take_result(self, result: Result) -> None:
        with self._cond:
            kept = self._results.get(result.command_id)
            if kept is None:
                kept = self._results[result.command_id] = _Kept()
                while len(self._results) > RESULTS_KEPT:
                    self._results.popitem(last=False)
            if kept.first is None:
                kept.first = result
            if result.status != ACCEPTED and kept.final is None:
                kept.final = result
                kept.states_at_final = self._states
            if result.reason == LOST and self._pinned is not None:
                self._pin_lost = f"{LOST} (RESULT of {result.command_id})"
            self._changes += 1
            self._cond.notify_all()
        # The link keeps its own; the client's copy would stay for ever.
        self._client.forget(result.command_id)


def _ready(welcome: Welcome | None, state: RobotState | None) -> bool:
    if welcome is None or state is None or state.session_id != welcome.session_id:
        return False
    valid = {sample.joint for sample in state.joints if sample.valid}
    return all(joint in valid for joint in range(len(welcome.joint_names)))


def _names(state: RobotState | None, command_id: str) -> bool:
    """Whether a ROBOT_STATE's last_result is of this command."""
    last = None if state is None else state.last_result
    return isinstance(last, dict) and last.get("command_id") == command_id


def _numbers(pinned: Pinned) -> dict[str, Any]:
    return {
        "session_id": pinned.session_id,
        "owner_number": pinned.owner_number,
        "stop_number": pinned.stop_number,
    }


def _cut(text: str, limit: int) -> str:
    return str(text)[:limit]


def _row(row: Sequence[float]) -> list[float]:
    """A pose as a list of floats; a numpy array is welcome."""
    values = row.tolist() if hasattr(row, "tolist") else row
    return [float(value) for value in values]
