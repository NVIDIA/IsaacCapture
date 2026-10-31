# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Count trajectories collected today (or a specified date), grouped by session.

Data layout (post Studio/timestamp routing):

    <data_dir>/<Studio>/<YYYY-MM-DD-HH-MM-SS>/episode_*/zarr.json

Each ``<Studio>/<timestamp>`` folder is one session. Its own ``zarr.json``
carries ``discarded_episode_indices`` (indices that were discarded). Each
episode's ``zarr.json`` carries ``episode_duration`` (seconds) and
``task_description`` (the instruction string used for that episode).

Library functions only — called by the backend's EOD Report handler. No CLI
entry point; nothing in the web backend needs a command-line wrapper.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def make_session_stats() -> dict:
    """Create a new session stats dict."""
    return {
        "studio": "",
        "timestamp": "",
        "total": 0,
        "excluded": 0,
        "valid": 0,
        "valid_seconds": 0.0,
        "excluded_seconds": 0.0,
        "descriptions": set(),
    }


def _read_zarr_json(path: Path) -> dict:
    """Load a zarr.json file's ``attributes`` dict; return ``{}`` on error."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload.get("attributes") or {}


def _episode_index(episode_dir: Path) -> int | None:
    """Return the integer suffix of an ``episode_N`` folder (or ``None``)."""
    name = episode_dir.name
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (ValueError, IndexError):
        return None


def _episode_recorded_date(
    ep_attrs: dict,
    episode_dir: Path,
    session_dir: Path,
) -> str:
    """Return ``YYYY-MM-DD`` for *episode_dir*.

    Resolution order:
      1. ``recorded_at.txt`` sidecar next to the episode (ISO 8601; first 10 chars = date).
      2. ``recorded_at`` attr on the episode's zarr.json
         (format: ``YYYY-MM-DD; HH-MM-SS``; first 10 chars are the date).
      3. Episode folder's filesystem mtime.
      4. Session folder's name prefix (``YYYY-MM-DD-HH-MM-SS``) — last resort.
    """
    sidecar = episode_dir / "recorded_at.txt"
    if sidecar.is_file():
        try:
            text = sidecar.read_text().strip()
            if len(text) >= 10:
                return text[:10]
        except OSError:
            pass
    recorded_at = ep_attrs.get("recorded_at") or ""
    if len(recorded_at) >= 10:
        return recorded_at[:10]
    try:
        return datetime.fromtimestamp(episode_dir.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        pass
    name = session_dir.name
    if len(name) >= 10 and name[4] == "-" and name[7] == "-":
        return name[:10]
    return ""


def _read_session_station_name(session_dir: Path) -> str | None:
    """Return ``meta.egocap_station_name`` from a session's ``config.yaml``.

    Returns ``None`` if the file is missing, unreadable, lacks the field,
    or PyYAML isn't available. ``RecordingService`` writes this field
    whenever a station label is configured, so it's the canonical way to
    map a session directory back to the egocap station that produced it.
    """
    cfg_path = session_dir / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        import yaml  # imported lazily so this module stays light when unused
    except ImportError:
        return None
    try:
        with cfg_path.open("r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return None
    meta = cfg.get("meta") if isinstance(cfg, dict) else None
    if not isinstance(meta, dict):
        return None
    name = meta.get("egocap_station_name")
    return name if isinstance(name, str) and name else None


def compute_station_data_breakdown(
    data_dir: Path,
    station: str,
    date_str: str,
) -> dict:
    """Sum ``episode_duration`` for one station on *date_str*.

    Walks ``<data_dir>/<Studio>/<timestamp>/episode_*`` and considers every
    episode that satisfies **both**:

    * its session's ``config.yaml`` has
      ``meta.egocap_station_name == station``;
    * it resolves to *date_str* via :func:`_episode_recorded_date`.

    Returns a breakdown so callers can show both totals and the
    valid-only portion::

        {
            "total_seconds": float,    # all matching episodes
            "valid_seconds": float,    # excluding discarded_episode_indices
            "excluded_seconds": float, # only discarded_episode_indices
        }

    Returns zeros when *data_dir* doesn't exist, *station* is empty, or no
    matching episodes are found.
    """
    breakdown = {
        "total_seconds": 0.0,
        "valid_seconds": 0.0,
        "excluded_seconds": 0.0,
    }
    if not station or not data_dir.is_dir():
        return breakdown

    for studio_dir in sorted(data_dir.iterdir()):
        if not studio_dir.is_dir():
            continue
        for session_dir in sorted(studio_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Skip sibling ``<timestamp>_aligned`` folders the background aligner
            # writes — they are processed copies of the raw session, not
            # separately collected data (else each aligned episode double-counts).
            if session_dir.name.endswith("_aligned"):
                continue
            if _read_session_station_name(session_dir) != station:
                continue

            session_attrs = _read_zarr_json(session_dir / "zarr.json")
            discarded = set(session_attrs.get("discarded_episode_indices") or [])

            for episode_dir in sorted(session_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                ep_idx = _episode_index(episode_dir)
                if ep_idx is None:
                    continue
                ep_attrs = _read_zarr_json(episode_dir / "zarr.json")
                if not ep_attrs:
                    continue
                ep_date = _episode_recorded_date(ep_attrs, episode_dir, session_dir)
                if ep_date != date_str:
                    continue
                duration = float(ep_attrs.get("episode_duration") or 0.0)
                breakdown["total_seconds"] += duration
                if ep_idx in discarded:
                    breakdown["excluded_seconds"] += duration
                else:
                    breakdown["valid_seconds"] += duration

    return breakdown


def count_trajectories(
    data_dir: Path,
    date_str: str,
    track_language: bool = False,
) -> dict[str, dict]:
    """Count trajectories for *date_str*, grouped by ``<Studio>/<timestamp>`` session.

    Walks two levels deep (``data_dir/<Studio>/<timestamp>/episode_*``)
    and attributes each episode to the day it was **actually recorded**,
    using the episode's ``recorded_at`` zarr attr. This gives the right
    answer even when a session crosses midnight (e.g. a session folder
    named ``2026-04-16-...`` that contains episodes recorded on
    ``2026-04-17``).

    Discarded episodes (per session root ``discarded_episode_indices``)
    are counted as "excluded". Duration comes from ``episode_duration``
    on each episode's zarr.json.
    """
    sessions: dict[str, dict] = defaultdict(make_session_stats)

    if not data_dir.is_dir():
        return dict(sessions)

    for studio_dir in sorted(data_dir.iterdir()):
        if not studio_dir.is_dir():
            continue
        for session_dir in sorted(studio_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Skip sibling ``<timestamp>_aligned`` folders the background aligner
            # writes — they are processed copies of the raw session, not new
            # trajectories (else each aligned episode double-counts the raw one).
            if session_dir.name.endswith("_aligned"):
                continue

            session_attrs = _read_zarr_json(session_dir / "zarr.json")
            discarded = set(session_attrs.get("discarded_episode_indices") or [])

            # Only accumulate once we see an episode actually recorded
            # on date_str; a session's timestamp folder name prefixing
            # date_str is not by itself enough to include it.
            session_key = f"{studio_dir.name}/{session_dir.name}"

            for episode_dir in sorted(session_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                ep_idx = _episode_index(episode_dir)
                if ep_idx is None:
                    continue

                ep_attrs = _read_zarr_json(episode_dir / "zarr.json")
                if not ep_attrs:
                    continue

                ep_date = _episode_recorded_date(ep_attrs, episode_dir, session_dir)
                if ep_date != date_str:
                    continue

                stats = sessions[session_key]
                stats["studio"] = studio_dir.name
                stats["timestamp"] = session_dir.name

                duration = float(ep_attrs.get("episode_duration") or 0.0)
                description = ep_attrs.get("task_description", "") or ""

                stats["total"] += 1
                if track_language and description:
                    stats["descriptions"].add(description)

                if ep_idx in discarded:
                    stats["excluded"] += 1
                    stats["excluded_seconds"] += duration
                else:
                    stats["valid"] += 1
                    stats["valid_seconds"] += duration

    return dict(sessions)
