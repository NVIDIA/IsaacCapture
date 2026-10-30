"""Per-station, per-day active/inactive time tracker for the egocap GUI.

Each egocap station is tracked independently. The operator toggles a button
between **Active** (green, station is collecting data) and **Inactive** (red,
station is blocked). This module owns:

* In-memory state for the tracker.
* Per-day JSON persistence under
  ``<recording_root>/station_status/<YYYY-MM-DD>.json`` so toggles survive an
  app restart and the EOD report can compute totals.
* A per-station daily activity JSON file written by
  :meth:`StationStatusTracker.save_daily_log`. The file is named
  ``<station>_<YYYY-MM-DD>_activity.json`` (station name sanitized and
  lowercased) and carries the active/inactive station times plus the
  hours of recorded data attributed to the station for that day.

Default state for any new (date, station) pair is **inactive** with no open
segment — i.e. the timer only starts running once the operator first marks the
station active. Active and inactive seconds are accumulated independently.

The tracker writes are atomic (tempfile + ``os.replace``) so a crash mid-write
does not corrupt the daily file.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

# State labels — kept short because they appear in the JSON on disk and in the
# saved daily log file.
ACTIVE = "active"
INACTIVE = "inactive"
_VALID_STATES = (ACTIVE, INACTIVE)
_DEFAULT_STATE = INACTIVE

# Top-level directory under the recording root where daily files live.
_LOG_SUBDIR = "station_status"


def _station_filename_stem(station: str) -> str:
    """Return the lowercase filename stem used for *station*'s daily file.

    Canonical egocap station names look like ``station_01``, already a
    well-formed filename stem once lowercased. Alphanumerics / ``-`` / ``_``
    are kept and lowercased; other characters become ``_``. An empty result
    falls back to ``"station"`` so the path is always well-formed.
    """
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (station or "")).lower().strip("_")
    return cleaned or "station"


def _today_iso(now: float | None = None) -> str:
    """Return today's date as an ISO ``YYYY-MM-DD`` string."""
    ts = now if now is not None else time.time()
    return datetime.date.fromtimestamp(ts).isoformat()


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration as ``Hh MMm SSs``."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _empty_station_record() -> dict:
    return {
        "current_state": _DEFAULT_STATE,
        "current_segment_start_ts": None,
        "intervals": [],
        "blocker_notes": "",
    }


class StationStatusTracker:
    """Persistent tracker of station active/inactive time, per-station per-day.

    Multiple instances pointing at the same ``log_dir`` are safe to use in a
    single-process GUI: every mutation re-reads the latest day's JSON before
    writing back, so concurrent edits in the same process resolve cleanly.
    Cross-process contention (multiple egocap GUIs) is not a supported use-case.
    """

    def __init__(self, log_dir: Path | str):
        self._log_dir = Path(log_dir).expanduser()
        # Cache of the current day's payload to avoid hitting the disk on every
        # ``get_state`` / ``compute_totals`` call.
        self._cached_date: str | None = None
        self._cached_payload: dict | None = None

    # ------------------------------------------------------------------ paths

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def _date_file(self, date_iso: str) -> Path:
        return self._log_dir / f"{date_iso}.json"

    def daily_log_path(self, station: str, date_iso: str | None = None) -> Path:
        """Return the per-station daily activity JSON path.

        Filename format: ``<station>_<YYYY-MM-DD>_activity.json``, with
        *station* sanitized and lowercased.
        """
        date_iso = date_iso or _today_iso()
        stem = _station_filename_stem(station)
        return self._log_dir / f"{stem}_{date_iso}_activity.json"

    # ----------------------------------------------------------- persistence

    def _load_payload(self, date_iso: str) -> dict:
        path = self._date_file(date_iso)
        if not path.exists():
            return {"date": date_iso, "stations": {}}
        try:
            with path.open("r") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {"date": date_iso, "stations": {}}
        if not isinstance(payload, dict):
            return {"date": date_iso, "stations": {}}
        payload.setdefault("date", date_iso)
        stations = payload.setdefault("stations", {})
        if not isinstance(stations, dict):
            payload["stations"] = {}
        return payload

    def _payload_for_today(self, now: float | None = None) -> dict:
        date_iso = _today_iso(now)
        if self._cached_date != date_iso or self._cached_payload is None:
            self._cached_payload = self._load_payload(date_iso)
            self._cached_date = date_iso
        return self._cached_payload

    def _save_payload(self, payload: dict) -> None:
        date_iso = payload.get("date") or _today_iso()
        _atomic_write_json(self._date_file(date_iso), payload)
        # Refresh cache so subsequent reads reflect the new payload, not the
        # version from before we wrote.
        self._cached_date = date_iso
        self._cached_payload = payload

    def _station_record(self, payload: dict, station: str) -> dict:
        stations = payload.setdefault("stations", {})
        record = stations.get(station)
        if not isinstance(record, dict):
            record = _empty_station_record()
            stations[station] = record
            return record
        record.setdefault("current_state", _DEFAULT_STATE)
        record.setdefault("current_segment_start_ts", None)
        record.setdefault("intervals", [])
        record.setdefault("blocker_notes", "")
        if record["current_state"] not in _VALID_STATES:
            record["current_state"] = _DEFAULT_STATE
        return record

    # ----------------------------------------------------------- public API

    def get_state(self, station: str, now: float | None = None) -> str:
        payload = self._payload_for_today(now)
        return self._station_record(payload, station)["current_state"]

    def get_segment_start(self, station: str, now: float | None = None) -> float | None:
        payload = self._payload_for_today(now)
        return self._station_record(payload, station)["current_segment_start_ts"]

    def get_blocker_notes(self, station: str, now: float | None = None) -> str:
        payload = self._payload_for_today(now)
        return self._station_record(payload, station)["blocker_notes"]

    def set_blocker_notes(
        self,
        station: str,
        text: str,
        now: float | None = None,
    ) -> None:
        payload = self._payload_for_today(now)
        record = self._station_record(payload, station)
        record["blocker_notes"] = text or ""
        self._save_payload(payload)

    def set_state(
        self,
        station: str,
        new_state: str,
        now: float | None = None,
    ) -> dict:
        """Force *station* into *new_state* (no-op if already in that state).

        Returns the updated station record (a fresh dict snapshot).
        """
        if new_state not in _VALID_STATES:
            raise ValueError(f"Invalid state {new_state!r}; expected one of {_VALID_STATES}")
        ts = now if now is not None else time.time()
        payload = self._payload_for_today(ts)
        record = self._station_record(payload, station)
        if record["current_state"] == new_state and record["current_segment_start_ts"] is not None:
            return dict(record)
        # Close the currently-open segment, if any.
        if record["current_segment_start_ts"] is not None:
            seg_start = float(record["current_segment_start_ts"])
            if ts > seg_start:
                record["intervals"].append(
                    {
                        "state": record["current_state"],
                        "start_ts": seg_start,
                        "end_ts": ts,
                    }
                )
        record["current_state"] = new_state
        record["current_segment_start_ts"] = ts
        self._save_payload(payload)
        return dict(record)

    def toggle(self, station: str, now: float | None = None) -> dict:
        """Flip active <-> inactive for *station* and persist.

        On the first toggle of the day this opens the first segment in the
        opposite-of-default state (i.e. ``ACTIVE``), which matches the typical
        morning workflow: operator arrives, presses the button to begin
        tracking active time.
        """
        ts = now if now is not None else time.time()
        payload = self._payload_for_today(ts)
        record = self._station_record(payload, station)
        if record["current_segment_start_ts"] is None:
            # First toggle ever today — start an active segment from now,
            # regardless of the recorded default state.
            new_state = ACTIVE
        else:
            new_state = INACTIVE if record["current_state"] == ACTIVE else ACTIVE
        return self.set_state(station, new_state, now=ts)

    def compute_totals(self, station: str, now: float | None = None) -> dict:
        """Return today's totals for *station* without mutating state.

        Includes the currently-open segment (closed virtually at ``now``) so
        the report stays live while the operator watches it.
        """
        ts = now if now is not None else time.time()
        payload = self._payload_for_today(ts)
        record = self._station_record(payload, station)

        intervals = list(record.get("intervals", []))
        # Always represent the open segment so state transitions are counted
        # even if the operator just toggled (zero-duration segment). The
        # ``max(0, ...)`` below keeps duration accounting honest.
        if record["current_segment_start_ts"] is not None:
            seg_start = float(record["current_segment_start_ts"])
            intervals.append(
                {
                    "state": record["current_state"],
                    "start_ts": seg_start,
                    "end_ts": max(seg_start, ts),
                }
            )

        active_seconds = 0.0
        inactive_seconds = 0.0
        # Number of active->inactive transitions today (i.e. blocker events).
        blocker_count = 0
        prev_state: str | None = None
        for interval in intervals:
            duration = max(0.0, float(interval["end_ts"]) - float(interval["start_ts"]))
            if interval["state"] == ACTIVE:
                active_seconds += duration
            elif interval["state"] == INACTIVE:
                inactive_seconds += duration
                if prev_state == ACTIVE:
                    blocker_count += 1
            prev_state = interval["state"]

        return {
            "active_seconds": active_seconds,
            "inactive_seconds": inactive_seconds,
            "blocker_count": blocker_count,
            "current_state": record["current_state"],
            "current_segment_start_ts": record["current_segment_start_ts"],
            "intervals": intervals,
        }

    def save_daily_log(
        self,
        station: str,
        date_iso: str | None = None,
        now: float | None = None,
        data_seconds: float | None = None,
        valid_seconds: float | None = None,
    ) -> Path:
        """Write a per-station daily activity JSON file and return its path.

        The JSON carries the operator-facing daily numbers:

        * ``active_station_time`` — total seconds the station was toggled
          to ``ACTIVE`` today (formatted ``Hh MMm SSs`` and as a raw float
          in ``active_station_time_seconds``).
        * ``inactive_station_time`` — same, for ``INACTIVE`` segments.
        * ``data_hours`` / ``data_seconds`` — total recorded episode
          duration attributed to this station today (all episodes,
          including discarded ones).
        * ``valid_hours`` / ``valid_seconds`` — same total minus episodes
          listed in the session's ``discarded_episode_indices``.

        Pass *data_seconds* / *valid_seconds* from a recording-root walk
        (see ``count_daily_trajectories.compute_station_data_breakdown``).
        When ``None``, the corresponding hours/seconds fields are written
        as ``null`` to mean "not measured".
        """
        date_iso = date_iso or _today_iso(now)
        totals = self.compute_totals(station, now=now)
        log_path = self.daily_log_path(station, date_iso)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def _hours_pair(seconds: float | None) -> tuple[float | None, float | None]:
            if seconds is None:
                return None, None
            secs = float(seconds)
            return secs, round(secs / 3600.0, 4)

        data_secs_out, data_hours_out = _hours_pair(data_seconds)
        valid_secs_out, valid_hours_out = _hours_pair(valid_seconds)

        payload = {
            "station": station,
            "date": date_iso,
            "active_station_time": _format_duration(totals["active_seconds"]),
            "active_station_time_seconds": float(totals["active_seconds"]),
            "inactive_station_time": _format_duration(totals["inactive_seconds"]),
            "inactive_station_time_seconds": float(totals["inactive_seconds"]),
            "data_hours": data_hours_out,
            "data_seconds": data_secs_out,
            "valid_hours": valid_hours_out,
            "valid_seconds": valid_secs_out,
        }
        _atomic_write_json(log_path, payload)
        return log_path

    def list_stations(self, now: float | None = None) -> Iterable[str]:
        payload = self._payload_for_today(now)
        return list(payload.get("stations", {}).keys())

    # Convenience for tests: drop the in-memory cache so the next read hits
    # disk again. The GUI relies on the cache to keep redraws cheap; tests
    # that simulate cross-process edits clear it explicitly.
    def invalidate_cache(self) -> None:
        self._cached_date = None
        self._cached_payload = None


__all__ = [
    "ACTIVE",
    "INACTIVE",
    "StationStatusTracker",
]
