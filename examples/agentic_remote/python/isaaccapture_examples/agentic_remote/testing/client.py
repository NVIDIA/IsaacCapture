# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A workstation-side client of the robot app, for the tests and the self-test demo.

It uses only the protocol package, so the same client can drive a real link.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import threading
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from ..protocol.carrier import receive_message, send
from ..protocol.messages import (
    CheckerState,
    Event,
    Gripper,
    Move,
    Open,
    OwnerRequest,
    Permit,
    Release,
    Reset,
    Result,
    RobotState,
    Shutdown,
    Stop,
    Welcome,
)

if TYPE_CHECKING:
    from ..protocol.carrier import Carrier
    from ..protocol.messages import GraspRegion, Message

_log = logging.getLogger("isaaccapture.examples.agentic_remote.client")

ACCEPTED = "accepted"
# Longest a wait_*() call waits, in real time, unless the caller says otherwise.
WAIT_S = 5.0
# Longest one receive waits, in real time, before the thread looks at close().
_RECEIVE_S = 0.005
# How long a permit of this client lasts unless the caller says otherwise: just
# under the evidence age limit of the robot app's demo configuration (3 s).
PERMIT_LIFE_S = 2.5
# EVENTs kept, and link messages a workstation should never get: the newest.
MESSAGES_KEPT = 1024
# Command ids whose RESULTs are kept, and RESULTs kept for one of them. Over
# that, the oldest command that ended goes first, then the oldest.
RESULTS_KEPT = 1024
RESULTS_PER_COMMAND = 16
# Peak speed of the minimum-jerk time law over its mean speed.
_PEAK_OVER_MEAN = 1.875
_EPS = 1e-9
# Stands for "fill this in from the newest ROBOT_STATE".
_AUTO: Any = object()
# What a command quotes from the newest ROBOT_STATE.
_QUOTED = frozenset({"session_id", "owner_number", "stop_number"})


class Clock(Protocol):
    """What the client needs of a clock: seconds that never go back."""

    def monotonic(self) -> float: ...


class PermitFeed:
    """Keeps permits flowing for one move until its final RESULT arrives."""

    def __init__(self, send_one: Callable[[], None], ended: Callable[[], bool]) -> None:
        self._send_one = send_one
        self._ended = ended
        self._stopped = threading.Event()
        self.sent = 0  # permits sent so far
        self._thread: threading.Thread | None = None

    def start(self, every_s: float) -> None:
        self._send()  # the first one on the caller's thread, before start() returns
        self._thread = threading.Thread(
            target=self._run, args=(every_s,), name="permit-feed", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Send no further permit."""
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(WAIT_S)

    def _run(self, every_s: float) -> None:
        # Paced in real time: the next permit goes out every_s after the last.
        while not self._stopped.wait(every_s):
            if self._ended():
                return
            self._send()

    def _send(self) -> None:
        self._send_one()
        self.sent += 1


class WorkstationClient:
    """Sends commands to a robot app and keeps what it answers.

    A receive thread stores the WELCOME, the newest ROBOT_STATE, the RESULTs by
    command id (of RESULTS_KEPT commands at most; see forget()) and the newest
    MESSAGES_KEPT EVENTs. Commands quote the session id, owner number and stop
    number of the newest ROBOT_STATE unless the caller passes its own.
    """

    def __init__(
        self,
        carrier: Carrier,
        clock: Clock,
        *,
        name: str = "ws",
        quotable: Callable[[RobotState], bool] | None = None,
        listeners: Sequence[Callable[[Message], None]] = (),
    ) -> None:
        """quotable says which ROBOT_STATE a command may take its numbers from.

        A command that fills them in waits for a state it accepts; None takes
        the newest one, which can be older than the last owner change.
        listeners are added before the first link message is received; see
        add_listener().
        """
        self._carrier = carrier
        self._clock = clock
        self._quotable = quotable
        self._cond = threading.Condition()
        self._welcome: Welcome | None = None
        self._state: RobotState | None = None
        self._state_at = 0.0  # this client's clock when the state arrived
        self._results: dict[str, list[Result]] = {}
        self._events: deque[Event] = deque(maxlen=MESSAGES_KEPT)
        self._unexpected: deque[Message] = deque(maxlen=MESSAGES_KEPT)
        self._ids = itertools.count(1)
        self._prefix = f"{name}-{uuid.uuid4().hex[:8]}"
        self._feeds: list[PermitFeed] = []
        self._listeners: tuple[Callable[[Message], None], ...] = tuple(listeners)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._receive_loop, name="workstation-client", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """End the receive thread and the permit feeds; the link is left open."""
        self._closed.set()
        for feed in list(self._feeds):
            feed.stop()
        if self._thread is not threading.current_thread():
            self._thread.join(WAIT_S)

    def add_listener(self, fn: Callable[[Message], None]) -> None:
        """Pass every link message received from now on to fn.

        fn runs on the receive thread, after the client stored the link message
        and outside the client's lock; it must not block.
        """
        with self._cond:
            self._listeners = (*self._listeners, fn)

    def forget(self, command_id: str) -> None:
        """Drop every RESULT kept for this command."""
        with self._cond:
            self._results.pop(command_id, None)

    # What the robot app sent

    @property
    def welcome(self) -> Welcome | None:
        with self._cond:
            return self._welcome

    @property
    def state(self) -> RobotState | None:
        """The newest ROBOT_STATE, None before the first one."""
        with self._cond:
            return self._state

    @property
    def events(self) -> list[Event]:
        """The EVENTs so far, oldest first; the newest MESSAGES_KEPT at most."""
        with self._cond:
            return list(self._events)

    @property
    def unexpected(self) -> list[Message]:
        """Link messages a workstation should never get; the newest MESSAGES_KEPT."""
        with self._cond:
            return list(self._unexpected)

    def results(self, command_id: str) -> list[Result]:
        """Every RESULT for this command so far, oldest first."""
        with self._cond:
            return list(self._results.get(command_id, ()))

    def wait_until(
        self, predicate: Callable[[], bool], timeout_s: float = WAIT_S, what: str = ""
    ) -> None:
        """Wait in real time until the predicate holds.

        It is asked again, under the client's lock, whenever a link message
        arrives.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: bool(predicate()), timeout_s):
                raise TimeoutError(f"{what or 'the condition'} did not come true")

    def wait_welcome(self, timeout_s: float = WAIT_S) -> Welcome:
        with self._cond:
            if not self._cond.wait_for(lambda: self._welcome is not None, timeout_s):
                raise TimeoutError(f"no WELCOME within {timeout_s:g} s")
            return self._welcome

    def wait_state(
        self,
        predicate: Callable[[RobotState], bool] | None = None,
        timeout_s: float = WAIT_S,
    ) -> RobotState:
        """Wait in real time for a ROBOT_STATE the predicate accepts; returns it.

        ROBOT_STATE is newest-wins: states in between may never be seen.
        """

        def found() -> bool:
            state = self._state
            return state is not None and (predicate is None or bool(predicate(state)))

        with self._cond:
            if not self._cond.wait_for(found, timeout_s):
                raise TimeoutError(
                    f"no such ROBOT_STATE within {timeout_s:g} s; newest: "
                    f"{_brief(self._state)}"
                )
            return self._state

    def wait_result(
        self, command_id: str, final: bool = True, timeout_s: float = WAIT_S
    ) -> Result:
        """Wait in real time for a RESULT of this command; returns it.

        final=True waits for the one that is not "accepted"; final=False returns
        the first RESULT, whichever it is.
        """

        def found() -> Result | None:
            for result in self._results.get(command_id, ()):
                if not final or result.status != ACCEPTED:
                    return result
            return None

        with self._cond:
            if not self._cond.wait_for(lambda: found() is not None, timeout_s):
                got = [result.status for result in self._results.get(command_id, ())]
                kind = "final RESULT" if final else "RESULT"
                raise TimeoutError(
                    f"no {kind} for {command_id} within {timeout_s:g} s; got {got}"
                )
            return found()

    def wait_event(
        self,
        kind: str | None = None,
        predicate: Callable[[Event], bool] | None = None,
        timeout_s: float = WAIT_S,
    ) -> Event:
        """Wait in real time for an EVENT of this kind that the predicate accepts."""

        def found() -> Event | None:
            for event in self._events:
                if kind is not None and event.kind != kind:
                    continue
                if predicate is None or predicate(event):
                    return event
            return None

        with self._cond:
            if not self._cond.wait_for(lambda: found() is not None, timeout_s):
                kinds = [event.kind for event in self._events]
                raise TimeoutError(
                    f"no such EVENT ({kind}) within {timeout_s:g} s; got {kinds}"
                )
            return found()

    # The robot app's clocks, as far as ROBOT_STATE tells

    def thor_now(self) -> tuple[float, float]:
        """The robot app's (monotonic, wall) time now, from the newest clock pair."""
        with self._cond:
            state, at = self._state, self._state_at
        if state is None:
            state = self.wait_state()
            with self._cond:
                at = self._state_at
        passed = max(0.0, self._clock.monotonic() - at)
        return state.clock_mono_s + passed, state.clock_wall_s + passed

    # Commands; each returns its command id

    def request_owner(
        self,
        request: str,
        *,
        reason: str = "",
        timed_return: bool = True,
        confirmed: bool = False,
        **quoted: Any,
    ) -> str:
        """OWNER_REQUEST: "agent", "help_needed", "nobody", "rest" or "human"."""
        return self._command(
            OwnerRequest,
            quoted,
            request=request,
            reason=reason,
            timed_return=timed_return,
            confirmed=confirmed,
        )

    def start_task(
        self, timed_return: bool = True, timeout_s: float = WAIT_S
    ) -> Result:
        """Ask for the arm as the agent and wait for the answer.

        Once accepted, also waits for the ROBOT_STATE that shows the agent as
        control owner, so the next command quotes the new owner number.
        """
        command_id = self.request_owner("agent", timed_return=timed_return)
        result = self.wait_result(command_id, timeout_s=timeout_s)
        if result.status == "completed":
            number = result.detail.get("owner_number")
            self.wait_state(
                lambda s: s.owner == "agent" and number in (None, s.owner_number),
                timeout_s,
            )
        return result

    def rest(self, confirmed: bool = True, **quoted: Any) -> str:
        """The operator's typed rest: OWNER_REQUEST "rest"."""
        return self.request_owner("rest", confirmed=confirmed, **quoted)

    def move(
        self,
        setpoints: Sequence[Sequence[float]],
        *,
        start_q: Sequence[float] | None = None,
        start_tol_rad: float = 0.02,
        grasp_region: GraspRegion | None = None,
        evidence_wall_s: float | None = _AUTO,
        expires_in_s: float = 5.0,
        **quoted: Any,
    ) -> str:
        """MOVE. start_q defaults to the measured pose of the newest ROBOT_STATE.

        evidence_wall_s defaults to now and the expiry to expires_in_s from now,
        both on the robot app's clocks. period_s and expires_mono_s can be given.
        """
        mono, wall = self.thor_now()
        fields = {
            "start_q": self.measured_pose() if start_q is None else _rows([start_q])[0],
            "start_tol_rad": start_tol_rad,
            "setpoints": _rows(setpoints),
            "period_s": self.wait_welcome().setpoint_period_s,
            "grasp_region": grasp_region,
            "evidence_wall_s": wall if evidence_wall_s is _AUTO else evidence_wall_s,
            "expires_mono_s": mono + expires_in_s,
        }
        return self._command(Move, quoted, **fields)

    def permit(
        self,
        move_id: str,
        through_index: int,
        *,
        evidence_wall_s: float | None = None,
        expires_in_s: float = PERMIT_LIFE_S,
        **quoted: Any,
    ) -> Permit:
        """PERMIT for a move up to and including through_index; returns it.

        The evidence time defaults to now and the expiry to expires_in_s from
        now, both on the robot app's clocks; expires_mono_s can be given.
        """
        mono, wall = self.thor_now()
        fields = {
            **self._quoted(),
            "evidence_wall_s": wall if evidence_wall_s is None else evidence_wall_s,
            "expires_mono_s": mono + expires_in_s,
            **quoted,
        }
        permit = Permit(move_id=move_id, through_index=int(through_index), **fields)
        send(self._carrier, permit)
        return permit

    def feed_permits(
        self,
        move_id: str,
        through_index: int,
        *,
        every_s: float = 0.05,
        expires_in_s: float = PERMIT_LIFE_S,
    ) -> PermitFeed:
        """Send a permit for the move now and a fresh one every every_s.

        every_s is real time. The feed ends with the move's final RESULT, with
        close() or with stop() on the returned feed.
        """

        def send_one() -> None:
            self.permit(move_id, through_index, expires_in_s=expires_in_s)

        def ended() -> bool:
            if self._closed.is_set():
                return True
            return any(r.status != ACCEPTED for r in self.results(move_id))

        feed = PermitFeed(send_one, ended)
        self._feeds.append(feed)
        feed.start(every_s)
        return feed

    def run_move(
        self,
        setpoints: Sequence[Sequence[float]],
        timeout_s: float = WAIT_S,
        **move_fields: Any,
    ) -> Result:
        """Send a MOVE with permits for all of it and wait for its final RESULT.

        The first permit goes out ahead of the MOVE, so the move never waits.
        """
        command_id = self.new_id()
        feed = self.feed_permits(command_id, len(setpoints) - 1)
        try:
            self.move(setpoints, command_id=command_id, **move_fields)
            return self.wait_result(command_id, timeout_s=timeout_s)
        finally:
            feed.stop()

    def gripper(
        self,
        opening: float,
        max_effort: float = 0.5,
        mode: str = "plain",
        **quoted: Any,
    ) -> str:
        return self._command(
            Gripper, quoted, opening=opening, max_effort=max_effort, mode=mode
        )

    def checker_state(self, action: str, **quoted: Any) -> str:
        return self._command(CheckerState, quoted, action=action)

    def stop(
        self, reason: str = "operator pressed stop", sender: str = "console"
    ) -> None:
        """STOP; it has no command id and no RESULT."""
        send(self._carrier, Stop(reason=reason, sender=sender))

    def reset(self, **quoted: Any) -> str:
        return self._command(Reset, quoted)

    def release(self, **quoted: Any) -> str:
        return self._command(Release, quoted)

    def open_jaws(self, confirmed: bool = True, **quoted: Any) -> str:
        return self._command(Open, quoted, confirmed=confirmed)

    def shutdown(self, confirmed: bool = True, **quoted: Any) -> str:
        """SHUTDOWN; confirmed says that the operator typed that the path is clear."""
        return self._command(Shutdown, quoted, confirmed=confirmed)

    def send(self, message: Message) -> None:
        """Send any link message as it is."""
        send(self._carrier, message)

    def new_id(self) -> str:
        """A command id no other command of this client has."""
        return f"{self._prefix}-{next(self._ids)}"

    # Planning

    def measured_pose(self) -> list[float]:
        """The measured pose of the newest ROBOT_STATE; every joint must be valid."""
        state = self.wait_state()
        bad = [sample.joint for sample in state.joints if not sample.valid]
        if bad:
            raise ValueError(f"ROBOT_STATE has no valid reading for joints {bad}")
        return [sample.value for sample in sorted(state.joints, key=_joint)]

    def plan_straight(
        self, q_from: Sequence[float], q_to: Sequence[float], speed: float
    ) -> list[list[float]]:
        """Setpoints of a straight minimum-jerk joint-space move, one per period.

        speed is the peak joint speed in rad/s; the period is the WELCOME's.
        The last setpoint is q_to and q_from itself is not among them.
        """
        if not (math.isfinite(speed) and speed > 0.0):
            raise ValueError(f"speed must be above 0, got {speed!r}")
        start, goal = _rows([q_from, q_to])
        if len(start) != len(goal):
            raise ValueError("q_from and q_to differ in length")
        period = self.wait_welcome().setpoint_period_s
        travel = max((abs(b - a) for a, b in zip(start, goal)), default=0.0)
        duration = _PEAK_OVER_MEAN * travel / speed
        count = max(1, math.ceil(duration / period - _EPS))
        rows: list[list[float]] = []
        for index in range(1, count):
            t = index / count
            s = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
            rows.append([a + (b - a) * s for a, b in zip(start, goal)])
        rows.append(goal)
        return rows

    # Internals

    def _quoted(self) -> dict[str, Any]:
        """The values of the newest ROBOT_STATE that commands quote."""
        state = self.wait_state(self._quotable)
        return {
            "session_id": state.session_id,
            "owner_number": state.owner_number,
            "stop_number": state.stop_number,
        }

    def _command(self, kind: type, quoted: dict[str, Any], **fields: Any) -> str:
        """Send one command; the caller's quoted values replace the state's."""
        names = {item.name for item in dataclasses.fields(kind)}
        unknown = sorted(set(quoted) - names)
        if unknown:
            raise TypeError(f"{kind.__name__} has no field(s) {', '.join(unknown)}")
        given = {**fields, **quoted}
        missing = (_QUOTED & names) - set(given)
        if missing:
            state = self._quoted()
            given.update({name: state[name] for name in missing})
        if "command_id" not in given:
            given["command_id"] = self.new_id()
        send(self._carrier, kind(**given))
        return given["command_id"]

    def _keep_result(self, result: Result) -> None:
        """Store one RESULT, within the bounds; under the client's lock."""
        kept = self._results.get(result.command_id)
        if kept is None:
            if len(self._results) >= RESULTS_KEPT:
                ended = (
                    command_id
                    for command_id, rows in self._results.items()
                    if any(row.status != ACCEPTED for row in rows)
                )
                del self._results[next(ended, next(iter(self._results)))]
            kept = self._results[result.command_id] = []
        if len(kept) >= RESULTS_PER_COMMAND:
            del kept[0]
        kept.append(result)

    def _receive_loop(self) -> None:
        while not self._closed.is_set():
            try:
                message = receive_message(self._carrier, _RECEIVE_S)
            except Exception:
                _log.exception("receiving a link message failed")
                self._closed.wait(_RECEIVE_S)
                continue
            if message is None:
                continue
            with self._cond:
                if isinstance(message, RobotState):
                    self._state = message
                    self._state_at = self._clock.monotonic()
                elif isinstance(message, Result):
                    self._keep_result(message)
                elif isinstance(message, Event):
                    self._events.append(message)
                elif isinstance(message, Welcome):
                    self._welcome = message
                else:
                    self._unexpected.append(message)
                self._cond.notify_all()
                listeners = self._listeners
            for listener in listeners:
                try:
                    listener(message)
                except Exception:
                    _log.exception("a listener failed on a link message")


def _joint(sample: Any) -> int:
    return sample.joint


def _rows(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Poses as lists of floats; numpy arrays are welcome."""
    out = []
    for row in rows:
        values = row.tolist() if hasattr(row, "tolist") else row
        out.append([float(value) for value in values])
    return out


def _brief(state: RobotState | None) -> str:
    if state is None:
        return "none yet"
    return (
        f"seq {state.seq}, owner {state.owner} ({state.owner_number}), stop number "
        f"{state.stop_number}, stop level {state.stop_level}, motion {state.motion}"
    )
