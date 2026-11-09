# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Writes a time-aligned copy of a recorded episode, in the aligned dataset layout downstream readers expect.

The aligned session sits beside the raw one and mirrors it: the same ``episode_<n>`` groups, the same per-source
groups, the same videos. What differs is the time alignment (see :mod:`alignment`): each episode gains
``valid_indices`` and ``valid_timestamps``; each camera gains ``aligned_frame_indices``; each resampled field is
written as ``<field>_interp`` and its raw array is left out (readers use the ``_interp`` name); every other raw
array, the attributes, the session's ``config.yaml`` and sidecars, and the videos are carried over unchanged. The
aligned root's ``zarr.json`` records which source was the index.

This module writes the files; the caller chooses the times, qualification and the staging that makes publication
atomic.
"""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

from .alignment import AlignedEpisode, align_episode, check_continuity, qualified_sources, source_rules

__all__ = [
    "ALIGNED_SESSION_SIDECARS",
    "AlignedWriteError",
    "align_kept_episode",
    "publish_aligned_episode",
    "validate_aligned_episode",
    "write_aligned_episode",
]

#: Session-level files carried into the aligned session when the raw session has them.
ALIGNED_SESSION_SIDECARS = (
    "config.yaml",
    "data_check_config.yaml",
    "head_camera_intrinsics.json",
    "head_camera_extrinsics.json",
    "wrist_calibration_offsets.json",
    "left_wrist_camera_intrinsics.json",
    "right_wrist_camera_intrinsics.json",
    "rectification.json",
)
#: Video container suffixes that are carried into the aligned episode.
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".avi", ".mov")
#: Arrays at most this large are read back by validation to catch stale or empty chunks.
_CHECK_MAX_BYTES = 10 * 1024 * 1024


class AlignedWriteError(RuntimeError):
    """The aligned episode could not be written or did not validate."""


def _copy_attrs(source: Any, target: Any) -> None:
    """Carry every attribute of a zarr group over to the aligned group."""
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _link_or_copy(source: Path, target: Path) -> None:
    """Put a video in the aligned episode: a hard link when the filesystem allows it, else a copy."""
    # A video from an earlier alignment is replaced, not merged with.
    if os.path.lexists(target):
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _check_separate(raw_session: Path, aligned_session: Path) -> None:
    """Refuse an output that is the raw session or shares a tree with it, before any file is touched.

    Re-aligning an episode deletes the aligned copy first; if that copy were the raw episode, the recording
    would be destroyed. Paths are resolved, so a symlink to the raw session is refused too.
    """
    raw, aligned = raw_session.resolve(), aligned_session.resolve()
    if aligned == raw or raw in aligned.parents or aligned in raw.parents:
        raise AlignedWriteError("the aligned session must be separate from the raw session")


def _write_root(raw_session: Path, aligned_session: Path, index_source: str) -> None:
    """Create the aligned root: its attributes, the session sidecars, and the index source in ``zarr.json``."""
    aligned_session.mkdir(parents=True, exist_ok=True)
    aligned_root = zarr.open_group(str(aligned_session), mode="a")
    # The raw session is only read: a raw root with no metadata of its own has no attributes to carry over.
    try:
        raw_attrs = dict(zarr.open_group(str(raw_session), mode="r").attrs)
    except (FileNotFoundError, zarr.errors.GroupNotFoundError):
        raw_attrs = {}
    for key, value in raw_attrs.items():
        aligned_root.attrs[key] = value
    # The index source is a root attribute so a reader knows which camera's frames define the timeline.
    aligned_root.attrs["index_source"] = index_source
    for name in ALIGNED_SESSION_SIDECARS:
        source, target = raw_session / name, aligned_session / name
        # The session config is always refreshed from the raw session, as the aligned dataset format does; any
        # other existing sidecar is left alone, so re-aligning never overwrites what is already there.
        if source.is_file() and (name == "config.yaml" or not target.exists()):
            shutil.copy2(source, target)


def _write_source(raw_group: Any, target_group: Any, aligned_source: Any) -> None:
    """Write one source: its attributes, the raw arrays that are not resampled, and what the alignment adds."""
    _copy_attrs(raw_group, target_group)
    for name in raw_group.array_keys():
        # A resampled field is read as ``<field>_interp``; its raw array is not kept.
        if name not in aligned_source.interpolated:
            target_group[name] = raw_group[name][:]
    for name, values in aligned_source.interpolated.items():
        target_group[f"{name}_interp"] = values
    if aligned_source.aligned_frame_indices is not None:
        target_group["aligned_frame_indices"] = aligned_source.aligned_frame_indices


def validate_aligned_episode(aligned_session: Path, episode_index: int, index_source: str) -> list[str]:
    """Problems with a written aligned episode; empty means it is as expected.

    Checks the arrays every aligned episode has, that the index source has ``aligned_frame_indices``, and that the
    small arrays can be read back and are not empty (a stale chunk from an earlier, differently shaped alignment
    only shows when the data is read).
    """
    problems: list[str] = []
    key = f"episode_{episode_index}"
    root = zarr.open_group(str(aligned_session), mode="r")
    if key not in root:
        return [f"{key} group missing from the aligned session"]
    episode = root[key]
    for required in ("valid_indices", "valid_timestamps"):
        if required not in episode:
            problems.append(f"{key}/{required} missing")
    if index_source not in episode:
        problems.append(f"{key}/{index_source} group missing")
    elif "aligned_frame_indices" not in episode[index_source]:
        problems.append(f"{key}/{index_source}/aligned_frame_indices missing")

    def check(group: Any, prefix: str) -> None:
        """Read each small array under ``group`` and report one that fails or is empty."""
        for name in group.array_keys():
            array = group[name]
            if array.nbytes > _CHECK_MAX_BYTES:
                continue
            try:
                data = np.asarray(array[:])
            except Exception as exc:  # a stale chunk raises something zarr-specific
                problems.append(f"{prefix}/{name} unreadable: {exc}")
                continue
            if data.size == 0:
                problems.append(f"{prefix}/{name} is empty")
        for name in group.group_keys():
            check(group[name], f"{prefix}/{name}")

    check(episode, key)
    return problems


def write_aligned_episode(
    raw_session: Path,
    aligned_session: Path,
    episode_index: int,
    index_source: str,
    *,
    downsample: int = 1,
    qualified: Mapping[str, bool],
    continuity: Mapping[str, Mapping[str, object]] | None,
) -> AlignedEpisode:
    """Align one recorded episode and write it into ``aligned_session``; returns the alignment.

    ``qualified`` says, per source, whether its recorded capture time is a qualified event time; a source that is
    not qualified (or not listed) refuses the alignment, so nothing is written for it. The times aligned on are
    the recorded capture times. ``continuity`` is the episode's stored per-source continuity evidence (None if it
    recorded none): a source whose evidence is missing, unknown or records a break refuses the alignment too.
    Raises ``AlignedWriteError`` if the written episode does not validate.
    """
    _check_separate(raw_session, aligned_session)
    config = yaml.safe_load((raw_session / "config.yaml").read_text(encoding="utf-8")) or {}
    rules = source_rules(config)
    cameras = [e["name"] for e in config.get("video_sources") or []]
    key = f"episode_{episode_index}"
    # The episode group is opened on its own: the raw session's root need not have metadata of its own.
    raw_episode = zarr.open_group(str(raw_session / key), mode="r")

    # Only declared sources the episode actually recorded take part, as the valid range spans what is present.
    present = [name for name in rules if name in raw_episode]
    unqualified = [name for name in [index_source, *present] if not qualified.get(name, False)]
    if unqualified:
        raise AlignedWriteError(f"sources without a qualified event time: {sorted(set(unqualified))}")
    times = {name: raw_episode[name]["capture_time"][:] for name in present}
    # The evidence is checked against the rows actually written: a break at a row the source lacks is refused.
    broken = check_continuity(continuity, [index_source, *present], {name: len(t) for name, t in times.items()})
    if broken:
        raise AlignedWriteError("; ".join(broken))
    fields = {name: {a: raw_episode[name][a][:] for a in raw_episode[name].array_keys()} for name in present}
    aligned = align_episode(
        index_source,
        times,
        [c for c in cameras if c in present],
        fields,
        {name: rules[name] for name in present},
        downsample=downsample,
    )

    _write_root(raw_session, aligned_session, index_source)
    root = zarr.open_group(str(aligned_session), mode="a")
    # A previous alignment of this episode is removed whole, so no chunk of a different shape survives it.
    if key in root:
        del root[key]
    episode = root.require_group(key)
    _copy_attrs(raw_episode, episode)
    episode["valid_indices"] = aligned.valid_indices
    episode["valid_timestamps"] = aligned.valid_timestamps
    for name in present:
        _write_source(raw_episode[name], episode.require_group(name), aligned.sources[name])

    # The videos are the same files: linked, so the aligned episode costs no extra disk.
    raw_dir, aligned_dir = raw_session / key, aligned_session / key
    for video in sorted(raw_dir.iterdir()) if raw_dir.is_dir() else []:
        if video.suffix in _VIDEO_SUFFIXES:
            _link_or_copy(video, aligned_dir / video.name)

    problems = validate_aligned_episode(aligned_session, episode_index, index_source)
    if problems:
        raise AlignedWriteError("; ".join(problems))
    return aligned


def publish_aligned_episode(
    raw_session: Path,
    aligned_session: Path,
    episode_index: int,
    index_source: str,
    *,
    downsample: int = 1,
    qualified: Mapping[str, bool],
    continuity: Mapping[str, Mapping[str, object]] | None,
) -> AlignedEpisode:
    """Write the aligned episode privately, then move it into ``aligned_session`` by renames.

    A reader never sees a half-written episode: it is written and validated in a staging session beside the
    aligned one (same filesystem, so the videos stay linked and the moves are renames). An interrupted run
    leaves either the staging tree, which the next run discards, or, if it stopped between the two renames,
    the previous episode under a ``.replaced`` name, which the next run puts back before anything else.
    Arguments and refusals are those of :func:`write_aligned_episode`.

    Publication contract: one publisher at a time per aligned session (an exclusive lock on a file beside it is
    held throughout, so concurrent runs queue instead of discarding each other's staging). A reader sees the
    previous episode, the new one, or, in the instant between the two renames of a re-publication, no episode;
    never a partial one. The rename pair is crash-recoverable, not a single atomic replacement.
    """
    _check_separate(raw_session, aligned_session)
    key = f"episode_{episode_index}"
    staging = aligned_session.with_name(f".{aligned_session.name}.staging")
    _check_separate(raw_session, staging)
    lock_path = aligned_session.with_name(f".{aligned_session.name}.lock")
    # The lock file sits beside the aligned session, not in it, so a refused run still creates nothing inside.
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish_locked(
            raw_session,
            aligned_session,
            staging,
            key,
            episode_index,
            index_source,
            downsample,
            qualified,
            continuity,
        )


def _publish_locked(
    raw_session: Path,
    aligned_session: Path,
    staging: Path,
    key: str,
    episode_index: int,
    index_source: str,
    downsample: int,
    qualified: Mapping[str, bool],
    continuity: Mapping[str, Mapping[str, object]] | None,
) -> AlignedEpisode:
    """The body of :func:`publish_aligned_episode`, run while its lock is held."""
    final, replaced = aligned_session / key, aligned_session / f".{key}.replaced"

    # Leftovers of an interrupted run: a half-built staging tree is discarded, and an episode set aside by an
    # interrupted move is restored (or dropped if its replacement did arrive).
    shutil.rmtree(staging, ignore_errors=True)
    if replaced.exists():
        if final.exists():
            shutil.rmtree(replaced)
        else:
            os.replace(replaced, final)

    try:
        aligned = write_aligned_episode(
            raw_session,
            staging,
            episode_index,
            index_source,
            downsample=downsample,
            qualified=qualified,
            continuity=continuity,
        )
    except BaseException:
        # Nothing of a failed or refused write reaches the aligned session.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # The aligned root and sidecars are idempotent; the episode moves in with two renames, keeping the old copy
    # until the new one is in place.
    _write_root(raw_session, aligned_session, index_source)
    if final.exists():
        os.replace(final, replaced)
    os.replace(staging / key, final)
    shutil.rmtree(replaced, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    return aligned


def align_kept_episode(
    raw_session: Path,
    episode_index: int,
    index_source: str,
    continuity: Mapping[str, Mapping[str, object]] | None,
    *,
    aligned_session: Path | None = None,
    downsample: int = 1,
) -> AlignedEpisode:
    """Align one kept episode from what the recorder stored about it, and publish the result.

    ``continuity`` is the episode manifest's evidence. Which sources have a qualified event time is read from the
    episode's own ``timing`` attribute, so an episode recorded without that evidence is refused, never assumed
    qualified. The aligned session defaults to ``<raw session name>_aligned`` beside the raw one.
    """
    config = yaml.safe_load((raw_session / "config.yaml").read_text(encoding="utf-8")) or {}
    declared = [e["name"] for section in ("video_sources", "numeric_sources") for e in config.get(section) or []]
    timing = dict(zarr.open_group(str(raw_session / f"episode_{episode_index}"), mode="r").attrs).get("timing")
    return publish_aligned_episode(
        raw_session,
        aligned_session or raw_session.with_name(f"{raw_session.name}_aligned"),
        episode_index,
        index_source,
        downsample=downsample,
        qualified=qualified_sources(timing, declared),
        continuity=continuity,
    )
