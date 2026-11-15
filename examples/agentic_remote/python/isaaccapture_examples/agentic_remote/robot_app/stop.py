# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""STOP, fault-stop and reset.

See section 9 of docs/design/claude_agentic_remote_execution.md.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .interfaces import StopLevel
from .text import MAX_OUT_CHARS, MAX_REASON_CHARS, MAX_WORD_CHARS, printable

if TYPE_CHECKING:
    from .authority import Authority
    from .gripper import GripperWorker
    from .interfaces import ArmPort, Checker, CheckerSet
    from .waits import AssistanceWait

_log = logging.getLogger("isaaccapture.examples.agentic_remote.stop")

# Event kinds given to emit_event.
STOP_EVENT = "stop"
FAULT_EVENT = "fault"
RESET_EVENT = "reset"

# A STOP from the agent leaves the assistance wait running. A STOP from any other
# sender, listed here or not, counts as a person's and cancels it.
AGENT_SENDER = "agent"
PERSON_SENDERS = ("console", "thor_terminal", "controller", "unknown")

_CHECKER_NAMES = ("agent", "human", "recovery")

EmitEvent = Callable[[str, str, dict[str, Any]], None]  # (kind, text, data)
RequestMotionStop = Callable[[str], None]  # (reason)


class StopController:
    """Raises STOP and fault-stop, and clears a fault-stop on reset.

    Both levels end in a hold with the motors on: the running move stops at its
    next setpoint and the jaws stay where they are, at the light stiffness.
    Nothing here starts a motion or a wait, opens the jaws or cuts torque.
    Thread-safe; every step of a stop runs even when an earlier one raised.
    """

    def __init__(
        self,
        authority: Authority,
        checkers: CheckerSet,
        gripper_worker: GripperWorker,
        assistance_wait: AssistanceWait,
        emit_event: EmitEvent,
        request_motion_stop: RequestMotionStop,
    ) -> None:
        """emit_event(kind, text, data) carries text for the operator.

        request_motion_stop(reason) makes the running move's stop condition
        return the reason. Both are called from whichever thread raises the stop.
        """
        self._authority = authority
        self._checkers = checkers
        self._gripper = gripper_worker
        self._wait = assistance_wait
        self._emit_event = emit_event
        self._request_motion_stop = request_motion_stop
        # Held while a fault-stop latches and while a reset clears, so that the
        # authority's stop level and the checkers' latches change together. A STOP
        # never waits for it.
        self._lock = threading.RLock()

    def stop(self, reason: str, sender: str) -> int:
        """STOP: hold, the control owner becomes nobody; returns the new stop number.

        A STOP from anyone but the agent also cancels the assistance wait. A
        latched fault-stop stays latched. The reason and the sender may come
        from a link: they are made printable and cut before anything shows them.
        """
        reason = _worded(reason, MAX_REASON_CHARS)
        # Only the word itself is the agent, not a text that reads so once cut.
        by_person = str(sender).strip() != AGENT_SENDER
        sender = printable(sender, MAX_WORD_CHARS).strip() or "unknown"
        if by_person and sender == AGENT_SENDER:
            sender = "unknown"
        problems: list[str] = []
        number = self._authority.raise_stop(reason, sender)
        state = self._authority.snapshot()
        for name, checker in self._named_checkers():
            _attempt(problems, f"the {name} checker's halt", checker.halt, reason)
        why = f"STOP from {sender}: {reason}"
        cancelled = self._stop_move_and_jaws(problems, why, cancel_wait=by_person)
        text = f"{why}; the arm holds, stop number {number}"
        if state.stop_level is StopLevel.FAULT:
            text += f"; the fault-stop stays latched: {state.fault_reason}"
        if cancelled:
            text += "; the assistance wait is cancelled"
        data = {
            "reason": reason,
            "sender": sender,
            "stop_number": number,
            "stop_level": state.stop_level.value,
            "wait_cancelled": cancelled,
        }
        self._emit(STOP_EVENT, text, data, problems)
        return number

    def fault(self, reason: str) -> int:
        """Fault-stop: as a STOP, and latched until reset; returns the new stop number.

        It always cancels the assistance wait. Every call raises the stop number,
        so call it once per fault, not on every poll of a fault that stays.
        """
        reason = _worded(reason, MAX_OUT_CHARS)
        problems: list[str] = []
        with self._lock:
            number = self._authority.raise_fault(reason)
            state = self._authority.snapshot()
            for name, checker in self._named_checkers():
                _attempt(problems, f"the {name} checker's latch", checker.latch, reason)
        why = f"fault-stop: {reason}"
        cancelled = self._stop_move_and_jaws(problems, why, cancel_wait=True)
        text = (
            f"FAULT-STOP: {reason}; the arm holds until a person types reset, "
            f"stop number {number}"
        )
        if state.fault_reason != reason:
            text += f"; latched earlier: {state.fault_reason}"
        if cancelled:
            text += "; the assistance wait is cancelled"
        data = {
            "reason": reason,
            "fault_reason": state.fault_reason,
            "stop_number": number,
            "stop_level": state.stop_level.value,
            "wait_cancelled": cancelled,
        }
        self._emit(FAULT_EVENT, text, data, problems)
        return number

    def reset(
        self, stop_number_seen: int, arm: ArmPort, owner_number_seen: int | None = None
    ) -> str | None:
        """Clear a latched fault-stop; returns why not, or None when cleared.

        Needs a latched fault-stop, the current stop number, the current owner
        number when one is given, and an arm that reports no fault. It clears
        the latch and nothing else.
        """
        refusal = self._reset_refusal(stop_number_seen, owner_number_seen, arm)
        if refusal is None:
            with self._lock:
                refusal = self._clear_latch(stop_number_seen)
        if refusal is not None:
            _log.info("reset refused: %s", refusal)
            return refusal
        text = "fault-stop cleared by reset; the arm holds, nothing was started"
        self._emit(RESET_EVENT, text, {"stop_number": stop_number_seen}, [])
        return None

    def _named_checkers(self) -> list[tuple[str, Checker]]:
        return list(zip(_CHECKER_NAMES, self._checkers.all(), strict=True))

    def _stop_move_and_jaws(
        self, problems: list[str], why: str, cancel_wait: bool
    ) -> bool:
        """Stop the running move and the jaws; True if a running wait was cancelled.

        The wait is cancelled before the jaw stop, which can block for a moment:
        the wait must not run out and ask for a return to rest meanwhile.
        """
        _attempt(problems, "the motion stop request", self._request_motion_stop, why)
        cancelled = False
        if cancel_wait:
            cancel = self._wait.cancel
            cancelled = bool(_attempt(problems, "the wait's cancel", cancel, why))
        _attempt(problems, "the jaw stop", self._gripper.stop_now)
        return cancelled

    def _reset_refusal(
        self, stop_number_seen: int, owner_number_seen: int | None, arm: ArmPort
    ) -> str | None:
        """Why a reset cannot go ahead, None when it can; nothing is changed."""
        state = self._authority.snapshot()
        if state.stop_level is not StopLevel.FAULT:
            return "no fault-stop is latched"
        if stop_number_seen != state.stop_number:
            return (
                f"stop number {stop_number_seen} does not match the current "
                f"{state.stop_number}"
            )
        # Compared here only: while a fault-stop is latched nothing changes the
        # control owner but a STOP, which the authority's own check catches.
        if owner_number_seen is not None and owner_number_seen != state.owner_number:
            return (
                f"owner number {owner_number_seen} does not match the current "
                f"{state.owner_number}"
            )
        try:
            faults = [str(fault) for fault in arm.faults()]
        except Exception as exc:  # an unknown fault state must not clear the latch
            _log.exception("the arm's faults could not be read")
            return f"the arm's faults could not be read: {exc!r}"
        if faults:
            return "the arm still reports a fault: " + "; ".join(faults)
        return None

    def _clear_latch(self, stop_number_seen: int) -> str | None:
        """Clear the checkers' latches, then the authority's; the lock is held.

        The authority decides last: it refuses when a STOP or a fault-stop was
        raised since the reset was checked. Every checker is then latched again.
        """
        problems: list[str] = []
        for name, checker in self._named_checkers():
            what = f"the {name} checker's latch reset"
            _attempt(problems, what, checker.reset_latch)
        if problems:
            refusal: str | None = "; ".join(problems)
        else:
            refusal = self._authority.reset(stop_number_seen)
        if refusal is None:
            return None
        state = self._authority.snapshot()
        if state.stop_level is StopLevel.FAULT:
            reason = state.fault_reason
            for name, checker in self._named_checkers():
                _attempt([], f"the {name} checker's latch", checker.latch, reason)
        return refusal

    def _emit(
        self, kind: str, text: str, data: dict[str, Any], problems: list[str]
    ) -> None:
        if problems:
            text += "; steps that failed: " + "; ".join(problems)
        data["problems"] = list(problems)
        try:
            self._emit_event(kind, text, data)
        except Exception:  # the stop itself is done; only its report is lost
            _log.exception("the %s event could not be emitted: %s", kind, text)


def _worded(reason: object, limit: int) -> str:
    """The reason on one printable line of at most `limit` characters."""
    return printable(str(reason).strip(), limit).strip() or "no reason given"


def _attempt(
    problems: list[str], what: str, step: Callable[..., Any], *args: Any
) -> Any:
    """Run one step; an error is logged and noted in problems, never passed on."""
    try:
        return step(*args)
    except Exception as exc:  # a stop must run to its end
        _log.exception("%s failed", what)
        problems.append(f"{what} failed: {exc!r}")
        return None
