# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Session folder layout under a recording (EOD) root and zarr health for EOD renames.

Layout: ``<eod_root>/<StudioSegment>/<session_timestamp>/`` (session = zarr + sidecars).
Matches :mod:`route`'s own routing of a kept episode into that layout -- when the user
re-picks a **Studio (metadata)** value in Episode Rename, we may need to **move** the whole
session to ``<eod>/`` + canonical studio + same timestamp.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import zarr

from egocap.recording.constants import PENDING_SUFFIX
from egocap.studio.studio_path_names import (
    canonicalize_studio_dir_name,
    sanitize_studio_dir_name,
)
from egocap.studio.task_library_data import TASK_LIBRARY

_SESSION_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
_UNMATCHED = "_unmatched"


def relocate_session_to_canonical_studio(
    session: Path,
    desired_studio_meta: str,
    known_studios: AbstractSet[str] | None,
) -> Path:
    """If *session* is not already under the canonical theme folder, move
    ``EOD/…/session_ts`` to ``EOD/<canonical_studio>/session_ts``.

    Covers both ``EOD/_unmatched/<ts>`` and ``EOD/Some_Studio/<ts>`` when
    the theme metadata points elsewhere.

    Returns
    -------
    The path to the session directory to use (possibly unchanged). No-op if
    the path does not look like a normal session
    (timestamp folder name) or *desired_studio_meta* is empty.
    """
    if not str(desired_studio_meta or "").strip():
        return session.expanduser().resolve()
    known: AbstractSet[str] = (
        frozenset(known_studios) if known_studios is not None else frozenset(TASK_LIBRARY.keys())
    )

    session = session.expanduser().resolve()
    if not session.is_dir():
        return session
    if not _SESSION_DIR.match(session.name):
        return session
    eod = session.parent.parent
    # Both conditions are defensive and unreachable here: session.is_dir()
    # already succeeded above, which on a normal filesystem guarantees every
    # ancestor directory (including session.parent.parent) exists and is a
    # directory too -- and a resolved path can never equal its own
    # grandparent.
    if not eod.is_dir() or eod == session:  # pragma: no cover
        return session

    want = canonicalize_studio_dir_name(sanitize_studio_dir_name(str(desired_studio_meta).strip()), known)
    if not want or want == _UNMATCHED:
        return session
    have_seg = session.parent.name
    have = canonicalize_studio_dir_name(sanitize_studio_dir_name(have_seg), known)
    dest = eod / want / session.name
    if session.resolve() == dest.resolve():
        return session
    if have == want:
        return session
    if dest.exists():
        raise OSError(f"EOD session relocate: destination already exists (refuse overwrite): {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(session), str(dest))
    return dest


def _episode_index_from_key(key: str) -> int | None:
    if not (isinstance(key, str) and key.startswith("episode_")):
        return None
    if key.endswith(PENDING_SUFFIX):
        return None
    base = key.removeprefix("episode_")
    if not base or not base.isdigit():
        return None
    return int(base)


def ensure_dataset_level_seed_fields(store: Any) -> None:
    """``discarded_episode_indices`` / ``discarded_reasons`` like
    :meth:`record_manager.RecorderManager._save_meta_data` and
    :meth:`RecordManagerEgoCap._seed_target_root`."""
    a = store.attrs
    a.setdefault("discarded_episode_indices", [])
    a.setdefault("discarded_reasons", {})


def refresh_session_egocap_zarr_aggregates(session_path: str | Path) -> None:
    """Recompute ``num_episodes``, ``episode_lengths``, ``total_data_time_hours`` at
    dataset level from final ``episode_N`` groups (parallels
    :class:`RecorderManagerEgoCap` helpers)."""
    p = str(Path(session_path).resolve())
    store = zarr.open(p, mode="a")
    ensure_dataset_level_seed_fields(store)
    finalized: list[str] = []
    try:
        names = list(store.keys())
    except (AttributeError, TypeError):
        names = list(store)
    for k in names:
        if isinstance(k, str) and k.startswith("episode_") and not k.endswith(PENDING_SUFFIX):
            if _episode_index_from_key(k) is not None:
                finalized.append(k)
    num_ep = len(finalized)
    store.attrs["num_episodes"] = num_ep
    episode_lengths: dict = dict(store.attrs.get("episode_lengths", {}) or {})
    for k in finalized:
        idx = _episode_index_from_key(k)
        if idx is None:  # pragma: no cover - defensive: `finalized` only ever
            continue  # contains keys this same function already verified
        epi = store[k]
        elen = epi.attrs.get("episode_length")
        if elen is not None:
            episode_lengths[str(idx)] = elen
    store.attrs["episode_lengths"] = episode_lengths
    total_seconds = sum(float(store[ek].attrs.get("episode_duration", 0.0) or 0.0) for ek in finalized)
    store.attrs["total_data_time_hours"] = round(total_seconds / 3600.0, 6)
