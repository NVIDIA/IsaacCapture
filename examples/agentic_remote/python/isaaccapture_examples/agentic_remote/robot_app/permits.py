# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The permit gate: which setpoints of an agent move may be sent, and until when."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .interfaces import Owner

if TYPE_CHECKING:
    from ..protocol.messages import Permit
    from .authority import Authority
    from .clock import Clock
    from .config import PermitsConfig
    from .interfaces import Checker

_log = logging.getLogger("isaaccapture.examples.agentic_remote.permits")

# Camera evidence may lead the app clock by this much before a permit is refused.
FUTURE_TOLERANCE_S = 0.05


@dataclass(frozen=True)
class _Held:
    permit: Permit
    evidence_mono: float  # camera time of the evidence on the app's monotonic clock


class PermitGate:
    """Holds one permit and says whether it covers a setpoint of a move.

    Thread-safe. A permit that passes every check is used whole: its reach,
    expiry and evidence time replace the held permit's. The newest permit
    decides, so one that reaches less far shortens how far the move may go.
    Less cover needs no fresh evidence: a permit of the held move that reaches
    less far and fails a time check lowers the held reach and nothing else. A
    cancel (through_index -1) is such a permit.
    """

    def __init__(
        self,
        authority: Authority,
        agent_checker: Checker,
        config: PermitsConfig,
        clock: Clock,
    ) -> None:
        self._authority = authority
        self._agent_checker = agent_checker
        self._age_limit_s = config.evidence_age_limit_s
        self._clock = clock
        self._lock = threading.Lock()
        self._held: _Held | None = None
        # (move id, reason) of the newest permit that was refused.
        self._refused: tuple[str, str] | None = None
        self._arrived = threading.Event()

    @property
    def arrived(self) -> threading.Event:
        """Set whenever a permit is stored; give it to MoveExecutor.run().

        The move that waits for its first permit clears it.
        """
        return self._arrived

    def update(
        self,
        permit: Permit,
        now_mono: float | None = None,
        wall_minus_mono: float | None = None,
    ) -> str | None:
        """Take in a permit; returns why it was refused, or None.

        wall_minus_mono (app wall clock minus monotonic clock) turns the evidence
        time into monotonic time; both times default to the clock. An accepted
        permit replaces the held one. For the move already held, a permit that
        fails a time check or quotes an older evidence time lowers the held
        reach when it reaches less far (down to -1) and leaves the held expiry
        and evidence as they are; otherwise the first is refused and the second
        ignored without a reason, and the held permit stays. The quoted times
        are compared: two conversions of one quoted time differ by the jitter
        of the clock reads.
        """
        if now_mono is None:
            now_mono = self._clock.monotonic()
        if wall_minus_mono is None:
            wall_minus_mono = self._clock.wall() - self._clock.monotonic()
        reason = self._stale(permit)
        if reason is not None:
            with self._lock:
                self._refused = (permit.move_id, reason)
            return reason
        evidence_mono = permit.evidence_wall_s - wall_minus_mono
        reason = self._untimely(permit, now_mono, evidence_mono)
        # Evidence slightly ahead of the clock must not add to the permit's life.
        evidence_mono = min(evidence_mono, now_mono)
        with self._lock:
            held = self._held
            same_move = held is not None and held.permit.move_id == permit.move_id
            older = same_move and permit.evidence_wall_s < held.permit.evidence_wall_s
            if reason is None and not older:
                # Fed under the lock, so a move that finds this permit covering its
                # first setpoint also finds the watchdog fed.
                self._agent_checker.feed_watchdog(evidence_mono)
                self._held = _Held(permit, evidence_mono)
                taken = "whole"
            elif same_move and -1 <= permit.through_index < held.permit.through_index:
                lowered = replace(held.permit, through_index=permit.through_index)
                self._held = _Held(lowered, held.evidence_mono)
                taken = "reach"
            else:
                taken = ""
                if reason is not None:
                    self._refused = (permit.move_id, reason)
            if taken and self._refused is not None:
                if self._refused[0] == permit.move_id:
                    self._refused = None  # the move has a newer permit than that
        if taken:
            if taken == "reach":
                _log.debug(
                    "took only the reach of a permit for move %s through setpoint "
                    "%d: %s",
                    permit.move_id,
                    permit.through_index,
                    reason or "older evidence",
                )
            self._arrived.set()
            return None
        if reason is None:
            _log.debug(
                "ignored a permit for move %s through setpoint %d: older evidence",
                permit.move_id,
                permit.through_index,
            )
        return reason

    def covers(
        self, move_id: str, index: int, now_mono: float | None = None
    ) -> str | None:
        """Why setpoint index of this move may not be sent now, or None if it may.

        The held permit must be for this move, still quote the authority's current
        values, be unexpired, rest on evidence younger than the limit and reach
        the index.
        """
        if now_mono is None:
            now_mono = self._clock.monotonic()
        with self._lock:
            held = self._held
        if held is None or held.permit.move_id != move_id:
            return "no permit for this move"
        permit = held.permit
        reason = self._stale(permit)
        if reason is not None:
            return reason
        if now_mono >= permit.expires_mono_s:
            return "permit expired"
        if now_mono - held.evidence_mono >= self._age_limit_s:
            return "permit evidence is too old"
        if index > permit.through_index:
            return f"permit reaches setpoint {permit.through_index}, not {index}"
        return None

    def may_renew(self, move_id: str) -> bool:
        """Whether a later permit can still bring this move cover.

        False without a permit for the move, after its cancel (a reach of -1)
        and once the held permit no longer quotes the authority's values. A
        setpoint that is not covered waits only while this is true.
        """
        with self._lock:
            held = self._held
        if held is None or held.permit.move_id != move_id:
            return False
        return held.permit.through_index >= 0 and self._stale(held.permit) is None

    def last_refusal(self, move_id: str) -> str | None:
        """Why the newest refused permit was refused, if it was for this move.

        None once a later permit for the move was taken, or one for another
        move was refused.
        """
        with self._lock:
            refused = self._refused
        return refused[1] if refused is not None and refused[0] == move_id else None

    def held_facts(self, now_mono: float) -> dict[str, Any] | None:
        """The held permit as ROBOT_STATE shows it; None without a permit.

        move_id, through_index, evidence_age_s (as covers() counts it) and
        expires_in_s, which is negative once the permit has expired. The
        permit of a move that has ended stays held until the next one.
        """
        with self._lock:
            held = self._held
        if held is None:
            return None
        return {
            "move_id": held.permit.move_id,
            "through_index": held.permit.through_index,
            "evidence_age_s": now_mono - held.evidence_mono,
            "expires_in_s": held.permit.expires_mono_s - now_mono,
        }

    def clear(self) -> None:
        """Forget the held permit and the last refusal."""
        with self._lock:
            self._held = None
            self._refused = None

    def _untimely(
        self, permit: Permit, now_mono: float, evidence_mono: float
    ) -> str | None:
        """Which time check a permit fails, or None when it passes them all."""
        if not (math.isfinite(permit.expires_mono_s) and math.isfinite(evidence_mono)):
            return "permit times are not finite"
        if permit.expires_mono_s <= now_mono:
            return "permit expired"
        if evidence_mono > now_mono + FUTURE_TOLERANCE_S:
            return "permit evidence is in the future"
        if now_mono - min(evidence_mono, now_mono) >= self._age_limit_s:
            return "permit evidence is too old"
        return None

    def _stale(self, permit: Permit) -> str | None:
        reason = self._authority.admit(
            permit.session_id, permit.owner_number, permit.stop_number, Owner.AGENT
        )
        return None if reason is None else f"permit: {reason}"
