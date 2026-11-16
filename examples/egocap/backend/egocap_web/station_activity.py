# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station's active/inactive time for today, and the daily activity log it feeds.

An operator marks the station active while it is collecting and inactive while it is blocked. The state and its
intervals are kept by ``egocap.analytics.station_status_log`` in one file per day under the recording output, and
each change also rewrites that station's daily activity log beside it. Both are local files written atomically;
nothing here uploads them.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from egocap.analytics.station_status_log import ACTIVE, INACTIVE, StationStatusTracker

STATES = (ACTIVE, INACTIVE)

#: Bounded so a stray paste cannot bloat the day file every toggle rewrites.
MAX_NOTES_CHARS = 4000

#: The directory, under the recording output, holding the day files and the daily activity logs.
ACTIVITY_DIR = "station_status"


class StationActivityError(ValueError):
    """A refused activity change, with a public reason."""


class StationActivity:
    """Today's activity for any station, behind one lock: the tracker is not safe for concurrent use."""

    def __init__(self, log_dir: Path, clock: Callable[[], float] = time.time) -> None:
        """``clock`` returns epoch seconds; tests pass their own to move through a day."""
        self._tracker = StationStatusTracker(log_dir)
        self._clock = clock
        self._lock = threading.Lock()

    def report(self, station: str) -> dict[str, Any]:
        """Today's state and totals for ``station``, with the open segment counted up to now."""
        with self._lock:
            return self._report(station, self._clock())

    def _report(self, station: str, now: float) -> dict[str, Any]:
        """The wire form of one station's day; the caller holds the lock."""
        totals = self._tracker.compute_totals(station, now=now)
        return {
            "station": station,
            "date": datetime.date.fromtimestamp(now).isoformat(),
            "state": totals["current_state"],
            # None until the first change of the day: no time is counted before the operator starts it.
            "since": totals["current_segment_start_ts"],
            "active_seconds": totals["active_seconds"],
            "inactive_seconds": totals["inactive_seconds"],
            "blocker_count": totals["blocker_count"],
            "notes": self._tracker.get_blocker_notes(station, now=now),
            # The moment these totals were computed, so a page can keep the open segment counting locally.
            "as_of": now,
        }

    def change(self, station: str, *, state: Any = None, notes: Any = None) -> dict[str, Any]:
        """Set the state, the blocker notes, or both, then rewrite the daily log; returns the new report."""
        if state is not None and state not in STATES:
            raise StationActivityError(f"state must be one of {', '.join(STATES)}")
        if notes is not None and (not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARS):
            raise StationActivityError(f"notes must be text of at most {MAX_NOTES_CHARS} characters")
        with self._lock:
            now = self._clock()
            if state is not None:
                self._tracker.set_state(station, state, now=now)
            if notes is not None:
                self._tracker.set_blocker_notes(station, notes, now=now)
            self._write_log(station, now)
            return self._report(station, now)

    def leave(self, station: str) -> None:
        """Stop counting active time for a station that is no longer in use; a no-op unless it is active."""
        with self._lock:
            now = self._clock()
            # Only an open active segment keeps accumulating; an untouched station must stay untouched.
            if self._tracker.get_state(station, now=now) != ACTIVE:
                return
            self._tracker.set_state(station, INACTIVE, now=now)
            self._write_log(station, now)

    def write_log(self, station: str, *, data_seconds: float | None, valid_seconds: float | None) -> str:
        """Rewrite the daily log with freshly measured recorded time; returns the log's file name."""
        with self._lock:
            now = self._clock()
            return self._tracker.save_daily_log(
                station, now=now, data_seconds=data_seconds, valid_seconds=valid_seconds
            ).name

    def _write_log(self, station: str, now: float) -> None:
        """Rewrite the daily log after a change, keeping the recorded time a report last measured."""
        date = datetime.date.fromtimestamp(now).isoformat()
        measured = _measured_seconds(self._tracker.daily_log_path(station, date))
        self._tracker.save_daily_log(station, date_iso=date, now=now, **measured)


def _measured_seconds(log_path: Path) -> dict[str, float | None]:
    """The recorded-time figures already in a daily log, so a state change does not blank them."""
    try:
        previous = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    return {key: _seconds(previous.get(key)) for key in ("data_seconds", "valid_seconds")}


def _seconds(value: Any) -> float | None:
    """A stored duration as a float, or None for anything that is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
