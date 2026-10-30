"""
EOD **Episode Rename** backfill: per-episode and dataset zarr attrs, ``config.yaml``
``meta``, optional **EOD session folder move** (``_unmatched`` or wrong studio →
``<EOD>/<canonical Studio>/``), and session-level zarr **aggregates** matching
:mod:`route`'s own group attrs (``num_episodes``, ``episode_lengths``,
``total_data_time_hours``, plus seed-style ``discarded_*`` fields).

A library function for offline/operator-tool callers, not something the current backend's
own pages invoke: a caller that also sets "Studio (metadata)" should call
:func:`relocate_session_to_canonical_studio` first, so the whole session tree moves like a
**properly routed** record under ``…/EOD/<Studio>/<timestamp>/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import zarr

from egocap.eod.eod_studio_session_move import (
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


class EpisodeRangeError(ValueError):
    """Raised when an episode-range spec can't be parsed."""


def _parse_episode_arg(spec: str) -> list[int]:
    """Parse an episode-range spec ('N' or 'M-N') into a sorted index list."""
    s = re.sub(r"\s+", "", (spec or "").strip())
    m = _EPISODE_RE.match(s)
    if not m:
        raise EpisodeRangeError(f"expected 'N' or 'M-N' (e.g. 5 or 2-7), got {spec!r}")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) is not None else a
    if b < a:
        a, b = b, a
    return list(range(a, b + 1))


def _update_session_config_yaml(
    zarr_root: Path,
    *,
    task_description: str,
    task_name: str,
    studio: str,
) -> bool:
    """Returns True if ``config.yaml``'s ``meta`` was updated and written,
    False if it was absent or the write failed -- the caller surfaces this
    on the result rather than letting a failure pass silently."""
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
        meta["task_description"] = task_description
        if task_name:
            meta["task_name"] = task_name
        if studio:
            meta["studio"] = studio
        data["meta"] = meta
        with open(cfg_p, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        return True
    except (OSError, TypeError, yaml.YAMLError):
        return False


@dataclass
class EpisodeTaskUpdateResult:
    """Outcome of :func:`update_episode_task_description`."""

    zarr_root: Path
    studio_stored: str
    updated_episodes: list[str] = field(default_factory=list)
    missing_episodes: list[str] = field(default_factory=list)
    aggregates_refreshed: bool = False
    config_yaml_updated: bool = False


def update_episode_task_description(
    zarr_root: Path,
    episodes: str,
    *,
    task_description: str,
    task_name: str = "",
    studio: str = "",
    relocate: bool = False,
) -> EpisodeTaskUpdateResult:
    """Backfill task/studio metadata for a recorded session's episodes.

    Replaces the removed CLI's ``main()``: called directly by the backend's
    EOD Report page handler, not spawned as a subprocess.
    """
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
    store.attrs["task_description"] = task_description
    if task_name:
        store.attrs["task_name"] = task_name
    studio_stored = ""
    if studio:
        # Falls back to the already-truthy `studio` when canonicalization
        # yields nothing, so studio_stored is always truthy here.
        studio_stored = canonicalize_studio_dir_name(sanitize_studio_dir_name(studio), known) or studio
        store.attrs["studio"] = studio_stored

    keys = list(store.keys())
    result = EpisodeTaskUpdateResult(zarr_root=zarr_root, studio_stored=studio_stored)
    for n in indices:
        key = f"episode_{n}"
        if key not in keys:
            result.missing_episodes.append(key)
            continue
        g = store[key]
        g.attrs["task_description"] = task_description
        if task_name:
            g.attrs["task_name"] = task_name
        if studio_stored:
            g.attrs["studio"] = studio_stored
        result.updated_episodes.append(key)

    result.config_yaml_updated = _update_session_config_yaml(
        zarr_root,
        task_description=task_description,
        task_name=task_name,
        studio=studio_stored,
    )

    try:
        refresh_session_egocap_zarr_aggregates(zarr_root)
        result.aggregates_refreshed = True
    except Exception:  # noqa: BLE001
        result.aggregates_refreshed = False

    return result
