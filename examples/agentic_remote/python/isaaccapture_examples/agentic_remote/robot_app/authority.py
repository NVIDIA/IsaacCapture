# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The authority: session id, control owner, owner number, stop number, stop level."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING

from .interfaces import AuthorityState, Owner, StopLevel

if TYPE_CHECKING:
    from .clock import Clock

_log = logging.getLogger("isaaccapture.examples.agentic_remote.authority")


class Authority:
    """Who may move the arm, and the values a command must quote to be admitted.

    Thread-safe: every read and change happens under one lock, and each change
    returns the number it raised.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        """A new session id, control owner nobody and both numbers at 0.

        clock is accepted because every robot app class takes one; it is not
        used, as the authority keeps no times.
        """
        self._lock = threading.RLock()
        self._session_id = uuid.uuid4().hex
        self._owner = Owner.NOBODY
        self._owner_number = 0
        self._stop_number = 0
        self._stop_level = StopLevel.NONE
        self._fault_reason = ""

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def owner(self) -> Owner:
        with self._lock:
            return self._owner

    @property
    def owner_number(self) -> int:
        with self._lock:
            return self._owner_number

    @property
    def stop_number(self) -> int:
        with self._lock:
            return self._stop_number

    @property
    def stop_level(self) -> StopLevel:
        with self._lock:
            return self._stop_level

    @property
    def fault_reason(self) -> str:
        with self._lock:
            return self._fault_reason

    def snapshot(self) -> AuthorityState:
        """All values as they stood at one moment."""
        with self._lock:
            return AuthorityState(
                session_id=self._session_id,
                owner=self._owner,
                owner_number=self._owner_number,
                stop_number=self._stop_number,
                stop_level=self._stop_level,
                fault_reason=self._fault_reason,
            )

    def admit(
        self,
        session_id: str,
        owner_number: int | None,
        stop_number: int,
        required: Owner | None,
    ) -> str | None:
        """Why a command quoting these values is refused, or None when admitted.

        Checked in this order: session id, fault-stop latched, stop number, control
        owner, owner number. Pass owner_number None for a command that carries
        none, and required None for one any control owner may send.
        """
        with self._lock:
            if session_id != self._session_id:
                return "session id does not match"
            if self._stop_level is StopLevel.FAULT:
                return self._fault_text()
            if stop_number != self._stop_number:
                return _mismatch("stop number", stop_number, self._stop_number)
            if required is not None and self._owner is not Owner(required):
                return f"control owner is {self._owner}, not {Owner(required)}"
            if owner_number is not None and owner_number != self._owner_number:
                return _mismatch("owner number", owner_number, self._owner_number)
            return None

    def set_owner(self, owner: Owner, reason: str) -> int:
        """Change the control owner; returns the new owner number.

        A STOP level ends when anyone but nobody takes the arm. A fault-stop stays.
        """
        owner = Owner(owner)
        with self._lock:
            before, number = self._change_owner(owner)
        _log_owner(before, owner, reason, number)
        return number

    def set_owner_if(
        self, owner: Owner, reason: str, *, stop_number: int, owner_number: int
    ) -> int | None:
        """Change the control owner only while both numbers still stand.

        Returns the new owner number, or None when a STOP, a fault-stop or an
        owner change came first; nothing is changed then.
        """
        owner = Owner(owner)
        with self._lock:
            if (stop_number, owner_number) != (self._stop_number, self._owner_number):
                return None
            before, number = self._change_owner(owner)
        _log_owner(before, owner, reason, number)
        return number

    def raise_stop(self, reason: str, sender: str) -> int:
        """STOP: the control owner becomes nobody; returns the new stop number.

        The stop level becomes STOP, unless a fault-stop is latched: that stays.
        """
        with self._lock:
            number = self._end_ownership()
            if self._stop_level is not StopLevel.FAULT:
                self._stop_level = StopLevel.STOP
        _log.info("STOP from %s: %s; stop number %d", sender, reason, number)
        return number

    def raise_fault(self, reason: str) -> int:
        """Fault-stop: as a STOP, and latched; returns the new stop number.

        A fault raised while one is latched keeps the first reason.
        """
        with self._lock:
            number = self._end_ownership()
            if self._stop_level is not StopLevel.FAULT:
                self._stop_level = StopLevel.FAULT
                self._fault_reason = str(reason)
        _log.error("fault-stop: %s; stop number %d", reason, number)
        return number

    def reset(self, stop_number_seen: int) -> str | None:
        """Clear a latched fault-stop; returns why not, or None when cleared.

        Needs the current stop number. The control owner is nobody afterwards.
        """
        with self._lock:
            if self._stop_level is not StopLevel.FAULT:
                return "no fault-stop is latched"
            if stop_number_seen != self._stop_number:
                return _mismatch("stop number", stop_number_seen, self._stop_number)
            self._stop_level = StopLevel.NONE
            self._fault_reason = ""
            # An owner set while the fault was latched must not outlive it.
            if self._owner is not Owner.NOBODY:
                self._owner = Owner.NOBODY
                self._owner_number += 1
        _log.info("fault-stop cleared by reset")
        return None

    def _change_owner(self, owner: Owner) -> tuple[Owner, int]:
        """Set the owner; returns (owner before, new owner number). Lock held."""
        before = self._owner
        self._owner = owner
        self._owner_number += 1
        if owner is not Owner.NOBODY and self._stop_level is StopLevel.STOP:
            self._stop_level = StopLevel.NONE
        return before, self._owner_number

    def _end_ownership(self) -> int:
        """Raise both numbers and make nobody the owner; the caller holds the lock."""
        self._stop_number += 1
        self._owner = Owner.NOBODY
        self._owner_number += 1
        return self._stop_number

    def _fault_text(self) -> str:
        if self._fault_reason:
            return f"fault-stop latched: {self._fault_reason}"
        return "fault-stop latched"


def _log_owner(before: Owner, owner: Owner, reason: str, number: int) -> None:
    _log.info(
        "control owner %s -> %s (%s); owner number %d", before, owner, reason, number
    )


def _mismatch(what: str, quoted: object, current: int) -> str:
    return f"{what} {quoted} does not match the current {current}"
