# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
EOD **Episode Rename** backfill: per-episode and dataset zarr attrs, ``config.yaml``
``meta``, optional **EOD session folder move** (``_unmatched`` or wrong studio →
``<EOD>/<canonical Studio>/``), and session-level zarr **aggregates** matching
:mod:`route`'s own group attrs (``num_episodes``, ``episode_lengths``,
``total_data_time_hours``, plus seed-style ``discarded_*`` fields).

The backend's end-of-day correction route (``POST /api/day-report/correct``) calls this with a
task only. Studio and relocation are for offline callers: one that also sets "Studio (metadata)"
should call :func:`relocate_session_to_canonical_studio` first, so the whole session tree moves
like a **properly routed** record under ``…/EOD/<Studio>/<timestamp>/``.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import zarr

from egocap.eod.eod_studio_session_move import (
    _episode_index_from_key,
    ensure_dataset_level_seed_fields,
    refresh_session_egocap_zarr_aggregates,
    relocate_session_to_canonical_studio,
)
from egocap.studio.studio_path_names import (
    canonicalize_studio_dir_name,
    sanitize_studio_dir_name,
)
from egocap.studio.task_library_data import TASK_LIBRARY

_EPISODE_RE = re.compile(r"^(\d+)(?:-(\d+))?$")
#: No session holds this many takes; a longer range is a slip of the hand, refused before it is walked.
MAX_EPISODE_SPAN = 10_000


class EpisodeRangeError(ValueError):
    """Raised when an episode-range spec can't be parsed."""


def _parse_episode_arg(spec: str) -> range:
    """Parse an episode-range spec ('N' or 'M-N') into its ascending indices.

    A ``range``, never a list: the bounds are a caller's free text, and a list of them would be as large as
    whatever number was typed.
    """
    s = re.sub(r"\s+", "", (spec or "").strip())
    m = _EPISODE_RE.match(s)
    if not m:
        raise EpisodeRangeError(f"expected 'N' or 'M-N' (e.g. 5 or 2-7), got {spec!r}")
    try:
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
    except ValueError:
        # Python refuses to convert a number of thousands of digits.
        raise EpisodeRangeError("the episode number is too long") from None
    if b < a:
        a, b = b, a
    if b - a >= MAX_EPISODE_SPAN:
        raise EpisodeRangeError(f"a range covers at most {MAX_EPISODE_SPAN} episodes")
    return range(a, b + 1)


def _update_session_config_yaml(
    zarr_root: Path,
    *,
    task_description: str | None,
    task_name: str,
    studio: str,
) -> bool:
    """Returns True if ``config.yaml``'s ``meta`` was updated and written,
    False if it was absent or the write failed -- the caller surfaces this
    on the result rather than letting a failure pass silently. A ``task_description``
    of None leaves the session's task as it is."""
    cfg_p = zarr_root / "config.yaml"
    if not cfg_p.is_file():
        return False
    try:
        with open(cfg_p, encoding="utf-8") as f:
            data: Any = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            data = {}
        meta: dict = data.get("meta")  # type: ignore[assignment]
        if not isinstance(meta, dict):
            meta = {}
        if task_description is not None:
            meta["task_description"] = task_description
        if task_name:
            meta["task_name"] = task_name
        if studio:
            meta["studio"] = studio
        data["meta"] = meta
        _replace_yaml(cfg_p, data)
        return True
    except (OSError, TypeError, yaml.YAMLError):
        return False


def _replace_yaml(path: Path, data: Any) -> None:
    """Replace a YAML file through a temporary file beside it, so a crash never leaves half of one."""
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise


@dataclass
class EpisodeTaskUpdateResult:
    """Outcome of :func:`update_episode_task_description`."""

    zarr_root: Path
    studio_stored: str
    updated_episodes: list[str] = field(default_factory=list)
    missing_episodes: list[str] = field(default_factory=list)
    aggregates_refreshed: bool = False
    config_yaml_updated: bool = False


def _relabel_episode(group: Any, task_description: str, task_name: str, studio: str) -> None:
    """Write a corrected task onto one episode. An episode carries the task under two names; both are kept in
    step, or a reader of the other one would still see the task the correction replaced."""
    group.attrs["task_description"] = task_description
    if "instruction_language" in group.attrs:
        group.attrs["instruction_language"] = task_description
    if task_name:
        group.attrs["task_name"] = task_name
    if studio:
        group.attrs["studio"] = studio


def _update_aligned_copy(
    aligned_root: Path, indices: range, task_description: str, task_name: str, studio: str, whole: bool
) -> None:
    """Apply the same task, name and studio metadata to an aligned copy's episodes and, when the correction
    covers the ``whole`` session, to its root and ``config.yaml``."""
    store = zarr.open(str(aligned_root), mode="a")
    if whole:
        store.attrs["task_description"] = task_description
        if task_name:
            store.attrs["task_name"] = task_name
    if studio:
        store.attrs["studio"] = studio
    for n in indices:
        if f"episode_{n}" in store:
            _relabel_episode(store[f"episode_{n}"], task_description, task_name, studio)
    _update_session_config_yaml(
        aligned_root,
        task_description=task_description if whole else None,
        task_name=task_name if whole else "",
        studio=studio,
    )


def update_episode_task_description(
    zarr_root: Path,
    episodes: str,
    *,
    task_description: str,
    task_name: str = "",
    studio: str = "",
    relocate: bool = False,
) -> EpisodeTaskUpdateResult:
    """Backfill task/studio metadata for the episodes of a recorded session that ``episodes`` names, on the
    session and on its time-aligned copy; episodes it names that do not exist are reported, not created."""
    zarr_root = zarr_root.expanduser().resolve()
    if relocate and not (studio or "").strip():
        raise ValueError("relocate needs a non-empty studio")
    # Parse before relocating: a bad range must fail before the session is
    # physically moved, not after -- otherwise a malformed spec leaves the
    # session relocated with no episode updates applied.
    indices = _parse_episode_arg(episodes)
    known = frozenset(TASK_LIBRARY.keys())
    if relocate and studio:
        zarr_root = relocate_session_to_canonical_studio(zarr_root, studio, known)
    if not zarr_root.is_dir():
        raise FileNotFoundError(f"not a directory: {zarr_root}")

    store = zarr.open(str(zarr_root), mode="a")

    ensure_dataset_level_seed_fields(store)
    keys = list(store.keys())
    # The session-level task describes every episode, so it follows the correction only when the correction
    # covers all of them; correcting some takes of a session must not relabel the rest.
    finalized = {index for index in map(_episode_index_from_key, keys) if index is not None}
    whole = bool(finalized) and all(index in indices for index in finalized)
    if whole:
        store.attrs["task_description"] = task_description
        if task_name:
            store.attrs["task_name"] = task_name
    studio_stored = ""
    if studio:
        # Falls back to the already-truthy `studio` when canonicalization
        # yields nothing, so studio_stored is always truthy here.
        studio_stored = canonicalize_studio_dir_name(sanitize_studio_dir_name(studio), known) or studio
        store.attrs["studio"] = studio_stored

    result = EpisodeTaskUpdateResult(zarr_root=zarr_root, studio_stored=studio_stored)
    for n in indices:
        key = f"episode_{n}"
        if key not in keys:
            result.missing_episodes.append(key)
            continue
        _relabel_episode(store[key], task_description, task_name, studio_stored)
        result.updated_episodes.append(key)

    result.config_yaml_updated = _update_session_config_yaml(
        zarr_root,
        task_description=task_description if whole else None,
        task_name=task_name if whole else "",
        studio=studio_stored,
    )
    # The time-aligned copy beside the session carries the same metadata, so it is updated with it.
    aligned_root = zarr_root.with_name(f"{zarr_root.name}_aligned")
    if aligned_root.is_dir():
        _update_aligned_copy(aligned_root, indices, task_description, task_name, studio_stored, whole)

    try:
        refresh_session_egocap_zarr_aggregates(zarr_root)
        result.aggregates_refreshed = True
    except Exception:  # noqa: BLE001
        result.aggregates_refreshed = False

    return result
