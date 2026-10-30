"""The one timeline every source of a replay session reads.

Tracking rows and camera frames that were recorded together must be shown
together, and all streams must wrap at the same moment, so a session's
sources share one :class:`PlaybackClock` instead of each counting time
from its own ``enter()`` (the adapters enter sequentially, which would
otherwise skew the streams by their startup order).
"""

from __future__ import annotations

from capture_server.acquisition.clock import Clock, MonotonicClock

__all__ = ["PlaybackClock"]


class PlaybackClock:
    """Maps host monotonic time onto a looping episode timeline of ``period`` seconds.

    The origin is fixed at construction (the session's Begin), so every reader -- each on its own adapter
    thread -- sees the same position for the same host instant without any shared mutable state.
    """

    def __init__(self, period: float, *, clock: Clock | None = None) -> None:
        if not period > 0.0:
            raise ValueError(f"PlaybackClock period must be positive, got {period!r}")
        self._period = float(period)
        self._clock = clock or MonotonicClock()
        self._t0 = self._clock.monotonic()

    @property
    def period(self) -> float:
        """Loop length in seconds."""
        return self._period

    def elapsed(self) -> float:
        """Host seconds since this session's timeline started (not wrapped)."""
        return self._clock.monotonic() - self._t0

    def position(self) -> float:
        """Episode-relative position in seconds, wrapped into ``[0, period)``."""
        return self.elapsed() % self._period

    def cycle(self) -> int:
        """How many times the episode has looped so far; a change tells sequential decoders to rewind."""
        return int(self.elapsed() // self._period)
