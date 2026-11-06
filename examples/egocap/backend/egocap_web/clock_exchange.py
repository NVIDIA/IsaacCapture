# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Maps a remote machine's monotonic clock onto this one's from four-timestamp exchanges.

The pattern is the one CloudXR uses for its clock sync: this side stamps ``t1`` when it sends a request, the remote
side stamps ``t2`` on receipt and ``t3`` when it replies, and this side stamps ``t4`` on the reply. The round trip
without the remote's processing time is ``(t4 - t1) - (t3 - t2)``, and the remote-minus-local offset is
``((t2 - t1) + (t3 - t4)) / 2``. That point estimate assumes the two directions took equally long; the true offset
can lie anywhere within half the round trip of it, so that half is reported as the mapping's uncertainty and no
smaller bound is ever claimed. All times are integer nanoseconds and the offset is signed.

This module only turns exchanges into a mapping. Sending the requests, and the transport under them, belong to
the caller, and so does the policy for what to do when no mapping is available: a mapping that is missing, stale
or discontinuous means the remote's timestamps cannot be placed on this clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

#: Exchanges whose round trip exceeds this are not used: a slow or queued exchange says little about the clocks.
DEFAULT_MAX_RTT_NS = 250_000_000
#: A mapping older than this (by this side's clock) is stale and is withheld.
DEFAULT_STALE_AFTER_NS = 5_000_000_000
#: The most recent accepted exchanges the best one is chosen from.
DEFAULT_WINDOW = 8
#: Relative clock rate difference tolerated between two mappings before their offsets are judged discontinuous.
DEFAULT_DRIFT_ALLOWANCE = 200e-6
#: Adopted mappings kept as evidence; the newest are the ones a take can be recorded under.
_MAX_HISTORY = 256


@dataclass(frozen=True)
class Exchange:
    """One completed four-timestamp exchange: ``t1``/``t4`` on the local clock, ``t2``/``t3`` on the remote's."""

    t1: int
    t2: int
    t3: int
    t4: int

    @property
    def rtt_ns(self) -> int:
        """Round trip with the remote's own processing time removed."""
        return (self.t4 - self.t1) - (self.t3 - self.t2)

    @property
    def offset_ns(self) -> int:
        """Remote minus local, assuming symmetric transit."""
        return ((self.t2 - self.t1) + (self.t3 - self.t4)) // 2

    @property
    def well_formed(self) -> bool:
        """Time moved forward on both clocks and the round trip is not negative."""
        return self.t4 >= self.t1 and self.t3 >= self.t2 and self.rtt_ns >= 0


@dataclass(frozen=True)
class ClockMapping:
    """An immutable mapping of the remote clock onto the local one; frame conversion only reads this."""

    #: Starts at 1 and changes whenever the estimator starts over or finds a discontinuity.
    epoch: int
    offset_ns: int
    rtt_ns: int
    #: Half the round trip, plus the drift allowed over the exchange's age: the error unknown path asymmetry and
    #: clock rate difference can cause.
    uncertainty_ns: int
    #: Local time at which the exchange behind this mapping completed.
    accepted_at_ns: int

    def to_local_ns(self, remote_ns: int) -> int:
        """The local-clock time of a remote-clock reading."""
        return remote_ns - self.offset_ns


class ClockEstimator:
    """Keeps the best recent exchange as the current mapping and notices when the clocks jump apart."""

    def __init__(
        self,
        *,
        max_rtt_ns: int = DEFAULT_MAX_RTT_NS,
        stale_after_ns: int = DEFAULT_STALE_AFTER_NS,
        window: int = DEFAULT_WINDOW,
        drift_allowance: float = DEFAULT_DRIFT_ALLOWANCE,
    ) -> None:
        self._max_rtt_ns = max_rtt_ns
        self._stale_after_ns = stale_after_ns
        self._drift_allowance = drift_allowance
        self._recent: deque[Exchange] = deque(maxlen=window)
        self._epoch = 1
        self._current: ClockMapping | None = None
        #: Local time of the newest accepted exchange: staleness is about whether replies are still arriving, not
        #: about how old the best exchange in the window is.
        self._newest_ns: int | None = None
        #: Every mapping adopted, oldest first, so a take can record the epochs and quality it was made under.
        self.history: list[ClockMapping] = []  # trimmed to the newest _MAX_HISTORY
        #: Set, and left set, when a new best mapping disagrees with the last by more than their uncertainties and
        #: the allowed drift explain: mappings before it stay as they were, and a take spanning it is not aligned.
        self.discontinuity = False

    def reset(self) -> None:
        """Start over, for a new source session or a restarted remote: the old mapping no longer applies."""
        self._recent.clear()
        self._current = None
        self._newest_ns = None
        self._epoch += 1

    def add(self, exchange: Exchange) -> str | None:
        """Take one exchange into account; returns why it was rejected, or None when it was accepted."""
        if not exchange.well_formed:
            return "malformed exchange"
        if exchange.rtt_ns > self._max_rtt_ns:
            return "round trip too long"
        previous = self._current
        self._newest_ns = exchange.t4
        newest = self._mapping_of(exchange)
        if previous is not None and self._disagrees(previous, newest):
            # Not a refinement: the clocks moved. Judged on the newest exchange, not on the best of the window, so
            # a jump is seen at once even when old exchanges have an equally good round trip. Keep the evidence and
            # start a new epoch from this exchange alone.
            self.discontinuity = True
            self._epoch += 1
            self._recent.clear()
            self._recent.append(exchange)
            candidate = self._mapping_of(exchange)
        else:
            self._recent.append(exchange)
            # The exchange with the smallest uncertainty now: half its round trip plus the drift its age could have
            # added. A low round trip beats a fresher one only while its age stays cheaper; the newest wins ties.
            best = min(self._recent, key=lambda e: (self._uncertainty_ns(e, exchange.t4), -e.t4))
            candidate = self._mapping_of(best, exchange.t4)
        if candidate != previous:
            self._current = candidate
            self.history.append(candidate)
            del self.history[:-_MAX_HISTORY]
        return None

    def mapping(self, now_ns: int) -> ClockMapping | None:
        """The current mapping, or None when there is none yet or no exchange arrived within the stale limit.

        The uncertainty is aged to ``now_ns``: while replies are late the offset's drift allowance keeps growing,
        so a budget check on a returned mapping never sees a figure older than the call. History is not appended.
        """
        current = self._current
        if current is None or self._newest_ns is None or now_ns - self._newest_ns > self._stale_after_ns:
            return None
        aged = current.rtt_ns // 2 + int(self._drift_allowance * max(0, now_ns - current.accepted_at_ns))
        if aged <= current.uncertainty_ns:
            return current
        return replace(current, uncertainty_ns=aged)

    def _uncertainty_ns(self, exchange: Exchange, now_ns: int) -> int:
        """Half the round trip plus the offset drift the exchange's age allows at the allowed relative rate."""
        return exchange.rtt_ns // 2 + int(self._drift_allowance * max(0, now_ns - exchange.t4))

    def _mapping_of(self, exchange: Exchange, now_ns: int | None = None) -> ClockMapping:
        return ClockMapping(
            epoch=self._epoch,
            offset_ns=exchange.offset_ns,
            rtt_ns=exchange.rtt_ns,
            uncertainty_ns=self._uncertainty_ns(exchange, exchange.t4 if now_ns is None else now_ns),
            accepted_at_ns=exchange.t4,
        )

    def _disagrees(self, previous: ClockMapping, candidate: ClockMapping) -> bool:
        elapsed_ns = max(0, candidate.accepted_at_ns - previous.accepted_at_ns)
        allowed = previous.uncertainty_ns + candidate.uncertainty_ns + self._drift_allowance * elapsed_ns
        return abs(candidate.offset_ns - previous.offset_ns) > allowed
