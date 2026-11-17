# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station's time today, logged automatically: when it could record and when it was down, and why.

Nobody marks the station active or blocked. A sampler asks every few seconds whether the station is recording,
ready to record, or down (a device, the capture session or a check is in the way), and the log keeps the day as a
list of segments, one file per day under the recording output. Active and down time are counted between the
first and last take of the day, so a station left powered on outside working hours adds to neither. Each down
period inside that window is a blocker, kept with the reason it began with.

Everything here is a local file written atomically; nothing uploads it.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from egocap.analytics.station_status_log import StationStatusTracker
from egocap.recording.episode_layout import write_json_atomic

logger = logging.getLogger("isaaccapture.egocap.station_uptime")

READY, RECORDING, DOWN = "ready", "recording", "down"

#: Bounded so a stray paste cannot bloat a file that is rewritten through the day.
MAX_NOTES_CHARS = 4000

#: The directory, under the recording output, holding the day timelines, the daily logs and the saved reports.
UPTIME_DIR = "station_status"

#: The reason logged for time the backend itself was not running to observe.
BACKEND_NOT_RUNNING = "the backend was not running"
#: The reason reported once a station's day holds as many segments as it may: its state is no longer followed.
LOG_FULL = "the day's log is full; the station's time is no longer logged"

#: A silence longer than this between two samples means the backend was not running, not that nothing changed.
_GAP_SECONDS = 90.0
#: How often an unchanged state is still written, so a crash loses at most this much of the last segment.
_HEARTBEAT_SECONDS = 30.0
#: A day is a few hundred segments; the bound only stops a flapping device from growing the file without limit.
_MAX_SEGMENTS = 5000


class UptimeError(ValueError):
    """A refused change to the day's notes, with a public reason."""


def notes_problem(notes: Any) -> str | None:
    """Why a value cannot be saved as the day's notes, or None when it can."""
    if not isinstance(notes, str) or len(notes) > MAX_NOTES_CHARS:
        return f"notes must be text of at most {MAX_NOTES_CHARS} characters"
    return None


def _midnight(day: str) -> float:
    """The moment a local calendar day (``YYYY-MM-DD``) begins, in epoch seconds."""
    return datetime.datetime.combine(datetime.date.fromisoformat(day), datetime.time.min).timestamp()


def _day(now: float) -> str:
    """The local calendar day of a moment, as ``YYYY-MM-DD``."""
    return datetime.date.fromtimestamp(now).isoformat()


def _duration(seconds: float) -> str:
    """A duration as ``Hh MMm SSs``, the form the daily log has always used."""
    whole = max(0, int(round(seconds)))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _open(record: dict[str, Any]) -> dict[str, Any] | None:
    """The segment of a station that has not ended yet, or None."""
    segments = record["segments"]
    return segments[-1] if segments and segments[-1]["end"] is None else None


def _full(record: dict[str, Any]) -> bool:
    """Whether a station's day holds as many segments as it may, so nothing more is added to it."""
    return len(record["segments"]) >= _MAX_SEGMENTS


def _took(record: dict[str, Any] | None) -> bool:
    """Whether a station's day has a take in it."""
    return record is not None and any(segment["state"] == RECORDING for segment in record["segments"])


def window_totals(segments: list[dict[str, Any]], now: float) -> dict[str, Any]:
    """Active time, down time and blockers between the first and last take of a day.

    Before the first take there is no window and everything is zero. A take still being recorded extends the
    window to now. Adjacent down segments are one blocker, named by the reason it began with; its ``end`` is
    None while it has not ended, and its seconds then run to now.
    """
    spans = [(s["state"], s["reason"], s["start"], s["end"] if s["end"] is not None else now) for s in segments]
    takes = [span for span in spans if span[0] == RECORDING]
    totals: dict[str, Any] = {"active_seconds": 0.0, "down_seconds": 0.0, "blockers": []}
    if not takes:
        return totals
    window_start, window_end = takes[0][2], takes[-1][3]
    for segment, (state, reason, start, end) in zip(segments, spans):
        # Only the part of a segment inside the working window counts.
        start, end = max(start, window_start), min(end, window_end)
        if end <= start:
            continue
        if state != DOWN:
            totals["active_seconds"] += end - start
            continue
        totals["down_seconds"] += end - start
        blockers = totals["blockers"]
        # Still open only when the segment has no end and the window did not cut it short.
        ended = None if segment["end"] is None and end == now else end
        if blockers and blockers[-1]["end"] == start:
            blockers[-1].update(end=ended, seconds=blockers[-1]["seconds"] + end - start)
        else:
            blockers.append({"reason": reason, "start": start, "end": ended, "seconds": end - start})
    return totals


class UptimeLog:
    """The day's timeline for every station, behind one lock; one writer process is assumed."""

    def __init__(self, directory: Path, clock: Callable[[], float] = time.time) -> None:
        """``clock`` returns epoch seconds; tests pass their own to move through a day."""
        self._directory = directory
        self._clock = clock
        self._lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._written_at = 0.0
        #: Why the station is down right now. A segment keeps the reason it began with; this is the current one.
        self._live_reason: str | None = None
        self._full_warned = False
        #: Stations whose daily log is behind the timeline: a take ended and the log has not been written since.
        self._owed_logs: set[str] = set()

    # --- the day file ----------------------------------------------------------------------------------------

    def _path(self, day: str) -> Path:
        """The timeline file of one day."""
        return self._directory / f"{day}_timeline.json"

    def _load(self, day: str) -> dict[str, Any]:
        """The stored timeline of a day, or an empty one when it is absent or unusable."""
        fresh: dict[str, Any] = {"date": day, "last_sample": None, "stations": {}}
        try:
            stored = json.loads(self._path(day).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return fresh
        stations = stored.get("stations") if isinstance(stored, dict) else None
        if not isinstance(stations, dict) or not all(self._usable(record) for record in stations.values()):
            logger.warning("Ignoring an unusable timeline at %s", self._path(day))
            return fresh
        return {"date": day, "last_sample": stored.get("last_sample"), "stations": stations}

    @staticmethod
    def _usable(record: Any) -> bool:
        """Whether a stored station record has the shape this module writes."""
        if not isinstance(record, dict) or not isinstance(record.get("segments"), list):
            return False
        record.setdefault("notes", "")
        record.setdefault("aliases", [])
        return all(
            isinstance(s, dict) and {"state", "reason", "start", "end"} <= set(s) for s in record["segments"]
        )

    def _today(self, now: float) -> dict[str, Any]:
        """Today's timeline, loaded on first use and rolled over when the local day changes."""
        day = _day(now)
        if self._payload is None:
            self._payload = self._unfinished_before(day) or self._load(day)
        # Before a rollover, so a silence that crossed midnight is down time on the day it began.
        self._bridge_gap(self._payload, now)
        if self._payload["date"] != day:
            self._payload = self._rolled(self._payload, day)
        return self._payload

    def _unfinished_before(self, day: str) -> dict[str, Any] | None:
        """The most recent earlier day's timeline when the backend stopped with a segment still open in it
        and ``day`` has no timeline yet, else None.

        Starting from that day lets the ordinary gap and rollover steps finish it: its open segment is ended,
        its daily log brought up to its midnight, and the state carried into ``day``.
        """
        # A day that already has a file was started by a run that finished the one before it.
        if self._path(day).exists():
            return None
        suffix = "_timeline.json"
        try:
            names = [path.name for path in self._directory.glob(f"*{suffix}")]
        except OSError:
            return None
        # ISO dates sort as text, so the greatest name below ``day`` is the latest earlier day.
        earlier = sorted(name[: -len(suffix)] for name in names if name[: -len(suffix)] < day)
        if not earlier:
            return None
        try:
            datetime.date.fromisoformat(earlier[-1])
        except ValueError:
            # Not a day file of this log; nothing to finish.
            return None
        payload = self._load(earlier[-1])
        return payload if any(_open(record) is not None for record in payload["stations"].values()) else None

    @staticmethod
    def _bridge_gap(payload: dict[str, Any], now: float) -> None:
        """Account for time nothing was sampled (the backend stopped, or its host was suspended): each open
        segment ends at the last sample and the silence since is logged as down."""
        last = payload["last_sample"]
        bridged = False
        for record in payload["stations"].values():
            segment = _open(record)
            # A full day takes no further segment; its open one is left as it is.
            if segment is None or _full(record):
                continue
            seen = last if isinstance(last, (int, float)) else segment["start"]
            # A quick restart or a late sample is not an outage; the segment simply continues.
            if now - seen <= _GAP_SECONDS:
                continue
            segment["end"] = max(seen, segment["start"])
            record["segments"].append(
                {"state": DOWN, "reason": BACKEND_NOT_RUNNING, "start": segment["end"], "end": None}
            )
            bridged = True
        # The silence is accounted for up to now; without this the next read would bridge it again.
        if bridged:
            payload["last_sample"] = now

    def _rolled(self, previous: dict[str, Any], day: str) -> dict[str, Any]:
        """Close the old day at its midnight and carry each open segment into the new one, so time that
        crosses midnight is counted on both days and no state is lost.

        ``previous`` is left as it was: the closed day is a copy, so when its write fails the open segments
        are still there to be carried by the next attempt.
        """
        midnight = _midnight(day)
        # The old day ends at its own midnight, which is earlier than the new day's start after a longer stop.
        # Found by the calendar, not by adding 24 hours: a day on which the clocks change is not that long.
        following = datetime.date.fromisoformat(previous["date"]) + datetime.timedelta(days=1)
        closing = min(midnight, _midnight(following.isoformat()))
        last = previous["last_sample"]
        # A sample later than midnight is a silence already bridged up to then; it must not be bridged again.
        sampled = max(midnight, last) if isinstance(last, (int, float)) else midnight
        closed = copy.deepcopy(previous)
        current: dict[str, Any] = {"date": day, "last_sample": sampled, "stations": {}}
        for name, record in closed["stations"].items():
            segment = _open(record)
            if segment is None:
                continue
            segment["end"] = max(closing, segment["start"])
            carried = {"state": segment["state"], "reason": segment["reason"], "start": midnight, "end": None}
            current["stations"][name] = {"segments": [carried], "notes": "", "aliases": []}
            # The old day's log is otherwise written only when a take ends, so the time after its last take, or
            # the part of a take that crosses midnight, would never reach it.
            if any(part["state"] == RECORDING for part in record["segments"]):
                self._close_log(name, closed, closing)
        self._write(closed)
        # Each day is told once when its log fills up, and owes nothing for the day before.
        self._full_warned = False
        self._owed_logs.clear()
        return current

    def _close_log(self, station: str, payload: dict[str, Any], midnight: float) -> None:
        """Bring a finished day's daily log up to its midnight. A failure is logged, not raised: the rollover
        must go on, or the new day would never start."""
        try:
            self._write_log(station, payload, midnight, self._measured(station, payload["date"]))
        except OSError:
            logger.exception("Could not bring the daily log of %s up to midnight", payload["date"])

    def _write(self, payload: dict[str, Any]) -> None:
        """Replace the day file atomically, once the day has something worth keeping.

        Time is counted from the first take, so until then the timeline lives in memory only: a backend that
        is merely running, or a test that starts one, leaves no file behind.
        """
        self._written_at = self._clock()
        path = self._path(payload["date"])
        worth_keeping = any(record["notes"] or _took(record) for record in payload["stations"].values())
        # A file already written is kept current, or notes cleared before the first take would return.
        if not worth_keeping and not path.exists():
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, payload)

    @staticmethod
    def _claim(payload: dict[str, Any], name: str) -> None:
        """A name that gets a record of its own stops being another station's earlier name: from here its
        takes are that record's, not those of the station renamed away from it."""
        for record in payload["stations"].values():
            if name in record["aliases"]:
                record["aliases"].remove(name)

    @classmethod
    def _record(cls, payload: dict[str, Any], station: str) -> dict[str, Any]:
        """A station's record for the day, created on first use."""
        record = payload["stations"].get(station)
        if record is None:
            cls._claim(payload, station)
            record = payload["stations"][station] = {"segments": [], "notes": "", "aliases": []}
        return record

    # --- writing ---------------------------------------------------------------------------------------------

    def observe(self, station: str, state: str, reason: str | None) -> None:
        """Log what the station in use is doing now. A new segment starts only when the state changes; a down
        period keeps the reason it began with, however the cause shifts while it lasts."""
        with self._lock:
            now = self._clock()
            payload = self._today(now)
            changed = False
            # One station is in use at a time: whatever another station was doing has ended.
            for name, other in payload["stations"].items():
                segment = _open(other)
                if name != station and segment is not None:
                    segment["end"], changed = now, True
            record = self._record(payload, station)
            segment = _open(record)
            took = False
            if (segment is None or segment["state"] != state) and self._has_room(record):
                if segment is not None:
                    segment["end"], took = now, segment["state"] == RECORDING
                record["segments"].append(
                    {"state": state, "reason": reason if state == DOWN else None, "start": now, "end": None}
                )
                changed = True
            self._live_reason = reason if state == DOWN else None
            payload["last_sample"] = now
            # The daily log follows each take, so it is current without anyone saving a report. Owed until
            # it is written: a failed write here is made good at the next one below.
            if took:
                self._owed_logs.add(station)
            if changed or now - self._written_at >= _HEARTBEAT_SECONDS:
                self._write(payload)
                self._settle_logs(payload, now)

    def _settle_logs(self, payload: dict[str, Any], now: float) -> None:
        """Write each daily log that is owed; one that cannot be written stays owed for the next attempt."""
        for station in sorted(self._owed_logs):
            self._write_log(station, payload, now, self._measured(station, payload["date"]))
            self._owed_logs.discard(station)

    def _has_room(self, record: dict[str, Any]) -> bool:
        """Whether a station's day takes another segment. At its bound the day stops being logged, and says so
        once; its open segment is left open, so an unchanged file is not rewritten at every sample."""
        if not _full(record):
            return True
        if not self._full_warned:
            self._full_warned = True
            logger.warning("The day's timeline is full; further state changes today are not logged")
        return False

    def rename(self, previous: str, name: str) -> None:
        """Move today's timeline of a station to its new name, remembering the old one so the day's takes,
        recorded under either name, are reported together."""
        with self._lock:
            now = self._clock()
            payload = self._today(now)
            self._carry_files(previous, name, payload["date"])
            record = payload["stations"].pop(previous, None)
            if record is None:
                return
            existing = payload["stations"].get(name)
            if existing is not None:
                # The new name was used earlier today: the two histories are one station's day.
                record["segments"] = sorted(existing["segments"] + record["segments"], key=lambda s: s["start"])
                record["notes"] = record["notes"] or existing["notes"]
                record["aliases"] = [*existing["aliases"], *record["aliases"]]
            else:
                self._claim(payload, name)
            record["aliases"] = sorted({*record["aliases"], previous} - {name})
            payload["stations"][name] = record
            # A log still owed to the old name is owed to the new one.
            if previous in self._owed_logs:
                self._owed_logs.discard(previous)
                self._owed_logs.add(name)
            self._write(payload)
            # The carried log still names the old station and lacks whatever the new name had done today.
            if self._log_path(name, payload["date"]).is_file():
                self._write_log(name, payload, now, self._measured(name, payload["date"]))

    def _carry_files(self, previous: str, name: str, day: str) -> None:
        """Move the day's daily log and saved report of a renamed station to its new name, so recorded time and
        the saved report follow it and nothing is left to be counted under both names. A file the new name
        already has is kept and the old one removed. A failure is logged, not raised: the timeline still moves.
        """
        for path_of in (self._log_path, self._report_path):
            old, new = path_of(previous, day), path_of(name, day)
            # Two names that differ only in case or punctuation share one file; there is nothing to move.
            if old == new:
                continue
            try:
                if new.exists():
                    old.unlink(missing_ok=True)
                elif old.exists():
                    os.replace(old, new)
            except OSError:
                logger.exception("Could not carry %s over to the station's new name", old.name)

    def set_notes(self, station: str, notes: Any) -> dict[str, Any]:
        """Save the operator's notes for the day; returns the station's report."""
        problem = notes_problem(notes)
        if problem is not None:
            raise UptimeError(problem)
        with self._lock:
            now = self._clock()
            payload = self._today(now)
            self._record(payload, station)["notes"] = notes
            self._write(payload)
            return self._report(station, payload, now)

    # --- reading ---------------------------------------------------------------------------------------------

    def report(self, station: str) -> dict[str, Any]:
        """The station's state now and its time today."""
        with self._lock:
            now = self._clock()
            return self._report(station, self._today(now), now)

    def _report(self, station: str, payload: dict[str, Any], now: float) -> dict[str, Any]:
        """The wire form of one station's day; the caller holds the lock."""
        record = payload["stations"].get(station) or {"segments": [], "notes": "", "aliases": []}
        segment = _open(record)
        state = segment["state"] if segment is not None else DOWN
        # Until the first sample nothing is known; saying "ready" would be a guess.
        reason = (self._live_reason or segment["reason"]) if segment is not None else "not observed yet"
        # A full day no longer follows the station, so its last segment's state would be a guess as well.
        if _full(record):
            state, reason = DOWN, LOG_FULL
        return {
            "station": station,
            "date": payload["date"],
            "state": state,
            "reason": reason if state == DOWN else None,
            "since": segment["start"] if segment is not None else now,
            **window_totals(record["segments"], now),
            "notes": record["notes"],
            # The moment these figures were computed, so a page can keep the open one counting locally.
            "as_of": now,
        }

    def took_today(self, station: str) -> bool:
        """Whether the station has recorded a take today, under this name or one it was renamed from."""
        with self._lock:
            return _took(self._today(self._clock())["stations"].get(station))

    def names(self, station: str) -> frozenset[str]:
        """Every name the station has had today, so takes recorded before a rename are still its own."""
        with self._lock:
            record = self._today(self._clock())["stations"].get(station)
            return frozenset({station, *(record["aliases"] if record is not None else ())})

    # --- the daily log and the saved report ------------------------------------------------------------------

    def _log_path(self, station: str, day: str) -> Path:
        """The station's daily log, named by the library that defines that file."""
        return StationStatusTracker(self._directory).daily_log_path(station, day)

    def _report_path(self, station: str, day: str) -> Path:
        """The station's saved report, beside its daily log so the two sort together."""
        log = self._log_path(station, day)
        return log.with_name(log.name.replace("_activity.json", "_report.json"))

    def _measured(self, station: str, day: str) -> dict[str, float | None]:
        """The recorded-time figures already in a daily log, so a rewrite after a take does not blank them."""
        try:
            previous = json.loads(self._log_path(station, day).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        values = previous if isinstance(previous, dict) else {}
        return {
            key: float(values[key])
            if isinstance(values.get(key), (int, float)) and not isinstance(values[key], bool)
            else None
            for key in ("data_seconds", "valid_seconds")
        }

    def _write_log(
        self, station: str, payload: dict[str, Any], now: float, measured: dict[str, float | None]
    ) -> Path:
        """Rewrite the station's daily log from the timeline and the recorded time last measured."""
        record = payload["stations"].get(station) or {"segments": []}
        totals = window_totals(record["segments"], now)
        hours = {key: None if value is None else round(value / 3600.0, 4) for key, value in measured.items()}
        path = self._log_path(station, payload["date"])
        write_json_atomic(
            path,
            {
                "station": station,
                "date": payload["date"],
                "active_station_time": _duration(totals["active_seconds"]),
                "active_station_time_seconds": totals["active_seconds"],
                "inactive_station_time": _duration(totals["down_seconds"]),
                "inactive_station_time_seconds": totals["down_seconds"],
                "blockers": len(totals["blockers"]),
                "data_hours": hours["data_seconds"],
                "data_seconds": measured["data_seconds"],
                "valid_hours": hours["valid_seconds"],
                "valid_seconds": measured["valid_seconds"],
            },
        )
        return path

    def write_log(self, station: str, *, data_seconds: float | None, valid_seconds: float | None) -> str:
        """Rewrite the daily log with freshly measured recorded time; returns the log's file name."""
        with self._lock:
            now = self._clock()
            measured = {"data_seconds": data_seconds, "valid_seconds": valid_seconds}
            return self._write_log(station, self._today(now), now, measured).name

    def save_report(self, station: str, report: dict[str, Any]) -> str:
        """Write the day's report beside the daily log, atomically; returns the file's name."""
        with self._lock:
            path = self._report_path(station, _day(self._clock()))
            self._directory.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, report)
            return path.name

    def report_saved(self, station: str) -> bool:
        """Whether today's report for the station has been saved."""
        return self._report_path(station, _day(self._clock())).is_file()


class UptimeSampler:
    """Decides what to log from the station's state, asked every few seconds.

    A state must hold for a while before it is logged, so a device that flickers for a second does not become a
    blocker. A take is the exception in both directions: its start and end are logged at once, because they set
    the window everything else is counted in.
    """

    def __init__(self, *, settle_seconds: float = 6.0, monotonic: Callable[[], float] = time.monotonic) -> None:
        """``settle_seconds`` is how long a new state must hold before it replaces the logged one."""
        self._settle_seconds = settle_seconds
        self._monotonic = monotonic
        self._logged: tuple[str, str] | None = None
        self._pending: tuple[str, str] | None = None
        self._pending_since = 0.0

    def settle(self, station: str, state: str, reason: str | None) -> tuple[str, str, str | None]:
        """What to log for the state just seen: that state once it is settled, else the last settled one."""
        seen = (station, state)
        now = self._monotonic()
        if seen != self._pending:
            self._pending, self._pending_since = seen, now
        immediate = (
            self._logged is None
            # Another station, or the start or end of a take, is a fact, not a flicker.
            or seen[0] != self._logged[0]
            or RECORDING in (state, self._logged[1])
        )
        if seen == self._logged or immediate or now - self._pending_since >= self._settle_seconds:
            self._logged = seen
            return station, state, reason
        # Not settled yet: the station is still logged as what it last settled on.
        return self._logged[0], self._logged[1], reason if self._logged[1] == DOWN else None
