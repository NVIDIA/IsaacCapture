# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One day's takes, read from the recording service's own episode manifests.

Each session directory under the recording output keeps a manifest per take, whatever became of it, so a take
that was rejected, discarded or interrupted is counted here with its actual outcome rather than vanishing from
the day. A take belongs to the day it was recorded on, which is not always the day its session was opened.

Read-only: nothing here writes to a session.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Collection
from pathlib import Path
from typing import Any

from egocap.analytics.count_daily_trajectories import _read_session_station_name, _read_zarr_json
from egocap.recording.capture_qc import QcDecision
from egocap.recording.episode_layout import (
    MANIFEST_DIR,
    EpisodeManifest,
    final_dir,
    manifest_path,
    scan_manifests,
)

#: What became of a take, in the order the page lists them. ``recording`` is the take being written right now.
KEPT, FLAGGED, REJECTED, DISCARDED, PENDING, RECORDING = (
    "kept",
    "flagged",
    "rejected",
    "discarded",
    "pending",
    "recording",
)
DISPOSITIONS = (KEPT, FLAGGED, REJECTED, DISCARDED, PENDING, RECORDING)

_SESSION_STARTED = re.compile(r"^session_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_")


def episode_disposition(manifest: EpisodeManifest, active_episode_id: str | None = None) -> str:
    """What became of one take, from its manifest alone.

    A finalized take is kept, or flagged when capture quality marked its data bad. A take still in an earlier
    storage phase is the one being recorded when its id is ``active_episode_id``, and otherwise an interrupted
    take waiting to be recovered or discarded.
    """
    if manifest.storage_phase == "FINALIZED":
        return FLAGGED if manifest.qc_decision == QcDecision.FLAGGED_BAD.value else KEPT
    if manifest.storage_phase in ("REJECTED", "DISCARDED"):
        return manifest.storage_phase.lower()
    return RECORDING if manifest.episode_id == active_episode_id else PENDING


def _final_attributes(session_path: Path, manifest: EpisodeManifest) -> dict[str, Any]:
    """The attributes of a take's finalized episode; empty when it has none or its location cannot be trusted."""
    try:
        return _read_zarr_json(final_dir(session_path, manifest) / "zarr.json")
    except (ValueError, OSError):
        return {}


def _recorded_date(attributes: dict[str, Any], session_path: Path, manifest: EpisodeManifest) -> str:
    """The day a take was recorded: its finalized episode's own ``recorded_at``, else when its manifest was
    last checkpointed, which for a take that never finalized is when it stopped."""
    recorded_at = attributes.get("recorded_at")
    if isinstance(recorded_at, str) and len(recorded_at) >= 10:
        return recorded_at[:10]
    checkpointed = manifest_path(session_path, manifest.episode_index).stat().st_mtime
    return datetime.date.fromtimestamp(checkpointed).isoformat()


def _location(manifest: EpisodeManifest) -> str:
    """Where a take's data is, relative to the recording output: its routed slot, else inside its session."""
    if manifest.assignment is not None:
        return f"{manifest.assignment.get('group')}/episode_{manifest.assignment.get('slot')}"
    return f"{manifest.session_id}/{manifest.final_path}"


def _session_started(session_id: str) -> str | None:
    """When a session was opened, from its directory name; None for a name of another shape."""
    match = _SESSION_STARTED.match(session_id)
    if match is None:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


def _empty_tally() -> dict[str, dict[str, float]]:
    """A count and a duration for every disposition, so the page never has to guess a missing key."""
    return {name: {"count": 0, "seconds": 0.0} for name in DISPOSITIONS}


def _add(tally: dict[str, dict[str, float]], disposition: str, seconds: float) -> None:
    """Count one take under its disposition."""
    tally[disposition]["count"] += 1
    tally[disposition]["seconds"] += seconds


def _episode_record(manifest: EpisodeManifest, disposition: str, attributes: dict[str, Any]) -> dict[str, Any]:
    """One take as the report lists it. The task is the finalized episode's own when it has one: a task
    corrected after recording is written there, while the manifest keeps what the take was started with."""
    corrected = attributes.get("task_description")
    return {
        "index": manifest.episode_index,
        "episode_id": manifest.episode_id,
        "disposition": disposition,
        "duration_seconds": float(manifest.duration_seconds),
        "task": corrected if isinstance(corrected, str) and corrected else manifest.task,
        "task_details": manifest.task_details,
        "operator": manifest.operator,
        "issues": list(manifest.issues),
        "qc_decision": manifest.qc_decision,
        "location": _location(manifest),
    }


def _session_record(
    session_path: Path, date: str, active_episode_id: str | None
) -> tuple[dict[str, Any], dict[str, dict[str, float]]] | None:
    """One session's takes of ``date`` and their tally, or None when it has none and nothing unreadable."""
    manifests, problems = scan_manifests(session_path)
    tally = _empty_tally()
    episodes = []
    for manifest in sorted(manifests, key=lambda m: m.episode_index):
        try:
            attributes = _final_attributes(session_path, manifest)
            recorded = _recorded_date(attributes, session_path, manifest)
            disposition = episode_disposition(manifest, active_episode_id)
            record = _episode_record(manifest, disposition, attributes)
        except OSError:
            # The manifest vanished between the scan and here (a discard finishing); the next read is correct.
            continue
        except (KeyError, TypeError, ValueError, AttributeError):
            # A manifest that parses but holds values of the wrong shape is one unaccounted take, not a reason
            # to lose the whole day's report.
            problems.append(f"episode_{manifest.episode_index}.json: malformed take record")
            continue
        if recorded != date:
            continue
        _add(tally, disposition, record["duration_seconds"])
        episodes.append(record)
    # An unreadable manifest is a take the day cannot account for; it is reported, never dropped.
    started = _session_started(session_path.name)
    unreadable_today = bool(problems) and (started is None or started[:10] == date)
    if not episodes and not unreadable_today:
        return None
    record = {
        "session_id": session_path.name,
        "station": _read_session_station_name(session_path),
        "started_at": started,
        "by_disposition": tally,
        "episodes": episodes,
        "problems": problems if unreadable_today else [],
    }
    return record, tally


def _session_dirs(root: Path) -> list[Path]:
    """Every recording-service session directly under the recording output, oldest name first."""
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if not entry.name.startswith(".") and (entry / MANIFEST_DIR).is_dir()]


def build_day_report(
    root: Path, date: str, *, stations: Collection[str] | None = None, active_episode_id: str | None = None
) -> dict[str, Any]:
    """Every take recorded on ``date`` (``YYYY-MM-DD``), by session, with totals per disposition.

    ``stations`` limits the report to sessions recorded under one of those station names (a station renamed
    during the day has more than one); None reports every station. ``active_episode_id`` names the take being
    recorded now, if any, so it is not mistaken for an interrupted one.
    """
    totals = _empty_tally()
    sessions = []
    for session_path in _session_dirs(root):
        if stations is not None and _read_session_station_name(session_path) not in stations:
            continue
        found = _session_record(session_path, date, active_episode_id)
        if found is None:
            continue
        record, tally = found
        sessions.append(record)
        for name in DISPOSITIONS:
            totals[name]["count"] += tally[name]["count"]
            totals[name]["seconds"] += tally[name]["seconds"]
    return {
        "date": date,
        "stations": sorted(stations) if stations is not None else None,
        "totals": {
            "episodes": int(sum(entry["count"] for entry in totals.values())),
            "seconds": sum(entry["seconds"] for entry in totals.values()),
            "by_disposition": totals,
        },
        "sessions": sessions,
    }


def recorded_seconds(report: dict[str, Any]) -> tuple[float, float]:
    """A report's recorded time as (everything recorded, kept and unflagged), for the daily activity log."""
    return float(report["totals"]["seconds"]), float(report["totals"]["by_disposition"][KEPT]["seconds"])
