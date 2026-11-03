# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Session directory layout, durable manifests and retry-safe episode finalization.

A session directory holds ``episode_<N>.pending`` while an episode is being written and, without routing,
``episode_<N>`` once finalized; with routing a finalized episode lives at ``<Studio>/<timestamp>/episode_<k>``
under the output root instead. The ``.recording/`` sidecar directory holds one ``episode_<N>.json`` manifest per
episode (and, with routing, ``routes.json``). The manifest lives outside the episode so it
stays reachable across the pending-to-final rename; it records what is needed to resume
or safely discard an interrupted episode after a crash.

Every manifest checkpoint is: write a uniquely named temporary file in the sidecar
directory, flush, ``fsync`` the file, atomically replace the manifest, ``fsync`` the
directory. A leftover temporary file is not a committed checkpoint.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
import zarr

from egocap.studio.studio_path_names import sanitize_studio_dir_name

from .constants import PENDING_SUFFIX

MANIFEST_DIR = ".recording"
MANIFEST_SCHEMA_VERSION = 3
#: Older records are decoded onto the current fields when read and rewritten at the current version only when
#: next saved; nothing is migrated at startup.
READABLE_SCHEMA_VERSIONS = (1, 2, 3)
_EPISODE_NAME = re.compile(r"^episode_(\d+)(\.pending)?$")

#: What finalization does with a checked take, recorded durably so recovery never re-derives it from policy:
#: KEEP publishes its output, REMOVE deletes this take's own data (a resolved rejection), RETAIN leaves its data
#: in place for explicit resolution.
FINALIZATION_KEEP = "KEEP"
FINALIZATION_REMOVE = "REMOVE"
FINALIZATION_RETAIN = "RETAIN"

#: Storage phases a manifest can record, in the order an episode passes through them.
PHASES = ("WRITING", "SEALED", "CHECKED", "FINALIZED", "REJECTED", "DISCARDED")


@dataclass
class EpisodeManifest:
    """Recovery metadata for one episode; see the service design for field contracts."""

    session_id: str
    episode_id: str
    episode_index: int
    config_digest: str
    task: str
    operator: str
    sources: list[dict[str, Any]]
    pending_path: str
    final_path: str
    storage_phase: str = "WRITING"
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    quality: dict[str, Any] | None = None
    capture_quality: dict[str, Any] | None = None  # capture-QC report: durable before cleanup/publication
    qc_decision: str | None = None  # a QcDecision value once the checks ran; None on version-1 manifests
    capture_policy: dict[str, Any] | None = None  # the exact policy and algorithm version that made the decision
    finalization: str | None = None  # FINALIZATION_*, recorded with the checked reports
    # Local routing (None when the service is not routed): the studio is frozen when the take starts, and the
    # assignment {"group": "<Studio>/<timestamp>", "slot": k} is made once, durably, just before publication.
    studio: str | None = None
    assignment: dict[str, Any] | None = None
    # Digest of the configuration without the capture-QC policy: what a CHECKED take (its decision is already
    # stored with its own policy) must still match to be finished. None on manifests written before it existed.
    layout_digest: str | None = None
    issues: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    schema_version: int = MANIFEST_SCHEMA_VERSION


def fsync_dir(path: Path) -> None:
    """Flush a directory's entries to stable storage."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(root: Path) -> None:
    """``fsync`` every file under ``root`` and then every directory, deepest first."""
    dirs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        dirs.append(Path(dirpath))
        for name in filenames:
            fd = os.open(Path(dirpath) / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    for directory in reversed(dirs):
        fsync_dir(directory)


def episode_dir_names(index: int) -> tuple[str, str]:
    """Return the (pending, final) directory names for an episode index."""
    return f"episode_{index}{PENDING_SUFFIX}", f"episode_{index}"


def next_episode_index(session_path: Path) -> int:
    """Return the first index not used by any pending/final directory or manifest in the session."""
    used = {-1}
    if session_path.is_dir():
        for entry in session_path.iterdir():
            match = _EPISODE_NAME.match(entry.name)
            if match:
                used.add(int(match.group(1)))
    manifests = session_path / MANIFEST_DIR
    if manifests.is_dir():
        for entry in manifests.glob("episode_*.json"):
            used.add(int(entry.stem.split("_")[1]))
    return max(used) + 1


def manifest_path(session_path: Path, index: int) -> Path:
    """Return the sidecar manifest path for an episode index."""
    return session_path / MANIFEST_DIR / f"episode_{index}.json"


def write_json_atomic(path: Path, data: Any) -> None:
    """Durably write JSON: temp file in the same directory, flush, fsync, atomic replace, directory fsync."""
    directory = path.parent
    directory.mkdir(exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    fsync_dir(directory)


def write_yaml_atomic(path: Path, data: Any) -> None:
    """Durably write YAML: temp file in the same directory, flush, fsync, atomic replace, directory fsync."""
    directory = path.parent
    directory.mkdir(exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    fsync_dir(directory)


def write_manifest(session_path: Path, manifest: EpisodeManifest) -> None:
    """Durably checkpoint ``manifest`` in the session's sidecar directory."""
    write_json_atomic(manifest_path(session_path, manifest.episode_index), asdict(manifest))


#: Sidecar record of the calibration assets preserved in a session: name -> sha256 (null if absent at creation).
CALIBRATION_RECORD = "calibration_assets.json"

#: Session-root (and, once routed, group-root) metadata snapshot: {"meta": {"egocap_station_name": ...}}.
#: Written once at session creation and never rewritten on reopen/recovery (see
#: RecordingService._prepare_session_assets); downstream tooling such as
#: analytics/count_daily_trajectories.py reads it to map a session back to its station.
SESSION_CONFIG_FILE = "config.yaml"


_FINALIZATIONS = (FINALIZATION_KEEP, FINALIZATION_REMOVE, FINALIZATION_RETAIN)


def _decode_older(data: dict[str, Any]) -> dict[str, Any]:
    """Map an older record onto the current fields, once, so everything downstream sees only the current shape.

    The finalization intent did not exist before schema 3: a checked older record gets the intent its stored
    results imply (failed integrity retains; a recorded camera rejection removes; otherwise keep). A record whose
    stored results are missing or damaged is unreadable, not a crash.
    """
    data = dict(data)
    version = data.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION and data.get("quality") and "finalization" not in data:
        try:
            outcome = data["quality"]["outcome"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"schema {version!r} record is missing or damaged: {exc!r}") from exc
        if outcome != "PASSED":
            data["finalization"] = FINALIZATION_RETAIN
        elif data.get("qc_decision") == "REJECTED_CAMERA":  # the one legacy value this decode knows about
            data["finalization"] = FINALIZATION_REMOVE
        else:
            data["finalization"] = FINALIZATION_KEEP
    return data


def read_manifest(path: Path) -> EpisodeManifest:
    """Load a manifest; unsupported schema versions raise so they are retained and reported, never guessed."""
    data = json.loads(path.read_text())
    version = data.get("schema_version")
    if version not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(f"{path}: unsupported manifest schema_version {version!r}")
    manifest = EpisodeManifest(**{**_decode_older(data), "schema_version": MANIFEST_SCHEMA_VERSION})
    # Paths are not caller data: they must be exactly the canonical names for the episode index.
    if not isinstance(manifest.episode_index, int) or manifest.episode_index < 0:
        raise ValueError(f"{path}: invalid episode_index {manifest.episode_index!r}")
    if manifest.storage_phase not in PHASES:
        raise ValueError(f"{path}: unsupported storage_phase {manifest.storage_phase!r}")
    checked_without_intent = bool(manifest.quality) and manifest.finalization is None
    if manifest.finalization not in (None, *_FINALIZATIONS) or checked_without_intent:
        raise ValueError(f"{path}: invalid finalization {manifest.finalization!r} for a checked take")
    expected = episode_dir_names(manifest.episode_index)
    if (manifest.pending_path, manifest.final_path) != expected:
        raise ValueError(f"{path}: non-canonical episode paths {manifest.pending_path!r}, {manifest.final_path!r}")
    return manifest


def scan_manifests(session_path: Path) -> tuple[list[EpisodeManifest], list[str]]:
    """Return every readable manifest in a session plus a diagnostic for each unreadable one."""
    manifests, problems = [], []
    directory = session_path / MANIFEST_DIR
    if directory.is_dir():
        for path in sorted(directory.glob("episode_*.json")):
            try:
                manifests.append(read_manifest(path))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                problems.append(f"{path.name}: {exc}")
    return manifests, problems


#: Root-group attribute that stamps an episode directory with the episode that owns it.
IDENTITY_ATTR = "recording_episode_id"


def read_identity(episode_dir: Path) -> str | None:
    """Return the owning episode id stamped in an episode directory's root metadata, or None."""
    try:
        data = json.loads((episode_dir / "zarr.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("attributes", {}).get(IDENTITY_ATTR)


def _owned_path(session_path: Path, relative: str, expected: str) -> Path:
    """Resolve a manifest path to a direct child of the session directory, refusing anything non-canonical.

    Rejects absolute paths, separators and traversal, names other than the canonical ``expected`` one,
    and symlinks, so a damaged or hostile manifest can never redirect a rename or delete.
    """
    if relative != expected or Path(relative).name != relative or Path(relative).is_absolute():
        raise ValueError(f"manifest path {relative!r} is not the canonical {expected!r}")
    path = session_path / relative
    if path.is_symlink() or path.parent.resolve() != session_path.resolve():
        raise ValueError(f"{path} escapes the session directory or is a symlink")
    return path


#: Group-directory attribute that stamps a routed group with the session and group that own it.
ROUTE_ATTR = "recording_route_id"


def route_id(session_id: str, group: str) -> str:
    """The ownership stamp value of a routed group."""
    return f"{session_id}|{group}"


def destination_path(assignment: dict[str, Any]) -> str:
    """The output-root-relative path of an assigned output: ``<Studio>/<timestamp>/episode_<slot>``."""
    return f"{assignment['group']}/episode_{assignment['slot']}"


def _owned_route_final(session_path: Path, manifest: EpisodeManifest) -> Path:
    """Resolve a routed final location, refusing anything but exactly ``<Studio>/<group>/episode_<slot>``.

    The studio folder must be the sanitized studio frozen in the manifest, the group must carry this session's
    ownership stamp, and no component may be a symlink or escape the output root, so a damaged or hostile
    manifest can never redirect a rename or delete outside the routed tree.
    """
    if manifest.studio is None or manifest.assignment is None:
        raise ValueError("manifest has no routed assignment")
    group, slot = manifest.assignment["group"], manifest.assignment["slot"]
    if not isinstance(slot, int) or slot < 0:
        raise ValueError(f"manifest slot {slot!r} is not a non-negative integer")
    relative = destination_path(manifest.assignment)
    parts = Path(relative).parts
    bad_part = any(p in ("", ".", "..") or "/" in p or "\\" in p for p in parts)
    if len(parts) != 3 or bad_part or Path(relative).is_absolute():
        raise ValueError(f"manifest route {relative!r} is not a three-part relative path")
    if parts[0] != sanitize_studio_dir_name(manifest.studio):
        raise ValueError(f"manifest route {relative!r} is not under its studio {manifest.studio!r}")
    output_root = session_path.parent.resolve()
    group_dir, final = output_root / parts[0] / parts[1], output_root / relative
    for path in (output_root / parts[0], group_dir):
        if path.is_symlink() or (path.exists() and path.resolve().parent != path.parent.resolve()):
            raise ValueError(f"{path} is a symlink or escapes the output root")
    if final.is_symlink():
        raise ValueError(f"{final} is a symlink")
    try:
        stamp = json.loads((group_dir / "zarr.json").read_text()).get("attributes", {}).get(ROUTE_ATTR)
    except (OSError, json.JSONDecodeError):
        stamp = None
    if group_dir.is_dir() and stamp != route_id(manifest.session_id, group):
        raise PermissionError(f"{group_dir} is not a group owned by session {manifest.session_id!r}")
    return final


def final_dir(session_path: Path, manifest: EpisodeManifest) -> Path:
    """The validated final directory of an episode: routed when a slot was assigned, else the session's own."""
    if manifest.assignment is not None:
        return _owned_route_final(session_path, manifest)
    return _owned_path(session_path, manifest.final_path, episode_dir_names(manifest.episode_index)[1])


def write_episode_attributes(episode_dir: Path, attributes: dict[str, Any]) -> None:
    """Merge dataset-level attributes into an episode's root metadata and make the change durable."""
    zarr.open_group(str(episode_dir), mode="r+").attrs.update(attributes)
    fsync_tree(episode_dir)


def finalize_episode_dir(session_path: Path, manifest: EpisodeManifest) -> Path:
    """Rename the pending directory to its final name; a repeat after a crash returns the owned final path.

    The caller must have recorded the intended final location in the manifest before calling. A final
    directory that already exists is accepted only if it carries this episode's identity.
    """
    pending_name, _ = episode_dir_names(manifest.episode_index)
    pending = _owned_path(session_path, manifest.pending_path, pending_name)
    final = final_dir(session_path, manifest)
    if final.exists() and not pending.exists():
        if read_identity(final) != manifest.episode_id:
            raise FileExistsError(f"{final} exists but is not owned by episode {manifest.episode_id!r}")
        return final
    if final.exists():
        raise FileExistsError(f"both {pending} and {final} exist; refusing to overwrite")
    os.rename(pending, final)
    fsync_dir(session_path)
    if final.parent != session_path:
        fsync_dir(final.parent)
    return final


def locate_episode(session_path: Path, manifest: EpisodeManifest) -> str:
    """Return ``"pending"`` or ``"final"`` for the directory that holds this episode, validated before any use.

    The pending name must be a canonical direct child of the session and the final name the canonical session child
    or, when routed, the exact assigned ``<Studio>/<timestamp>/episode_<k>`` (no traversal, absolute path or
    symlink), the
    directory found must carry this episode's identity (an unstamped pending directory is accepted only while
    the manifest is still WRITING, before the stamp could exist, or when it is the empty residue of an interrupted
    removal of a camera-rejected take, whose stamp is deleted last), and pending and final may not both exist.
    Recovery and metadata writes go through this check, so nothing is touched when ownership is unclear.
    """
    pending_name, _ = episode_dir_names(manifest.episode_index)
    pending = _owned_path(session_path, manifest.pending_path, pending_name)
    final = final_dir(session_path, manifest)
    if pending.exists():
        if final.exists():
            raise FileExistsError(f"both {pending} and {final} exist; refusing to choose")
        identity = read_identity(pending)
        # An unstamped directory is accepted only before the stamp exists (WRITING), or when it is empty and the
        # manifest durably says this take's data is being removed: what an interrupted removal leaves after it
        # deletes the stamp last. No other take may treat an emptied directory as its own.
        cleanup_residue = (
            manifest.storage_phase == "CHECKED"
            and manifest.finalization == FINALIZATION_REMOVE
            and not any(pending.iterdir())
        )
        unstamped_ok = manifest.storage_phase == "WRITING" or cleanup_residue
        if identity != manifest.episode_id and not (identity is None and unstamped_ok):
            raise PermissionError(f"{pending} is not owned by episode {manifest.episode_id!r}")
        return "pending"
    if final.exists():
        if read_identity(final) != manifest.episode_id:
            raise FileExistsError(f"{final} exists but is not owned by episode {manifest.episode_id!r}")
        return "final"
    raise FileNotFoundError(f"neither {pending} nor {final} exists")


def remove_owned_episode(session_path: Path, manifest: EpisodeManifest) -> None:
    """Delete only this episode's pending directory; refuses to touch a finalized episode."""
    pending_name, _ = episode_dir_names(manifest.episode_index)
    pending = _owned_path(session_path, manifest.pending_path, pending_name)
    final = final_dir(session_path, manifest)
    if manifest.storage_phase == "FINALIZED" or (final.exists() and manifest.storage_phase != "DISCARDED"):
        raise PermissionError(f"{final} is finalized output and is never deleted by discard")
    if pending.exists():
        locate_episode(session_path, manifest)  # raises unless this episode owns the directory
        # Everything else goes first and the ownership stamp last: if this is interrupted, what is left still
        # proves it is this episode's, so the removal can be retried instead of stranding unowned data.
        marker = pending / "zarr.json"
        for child in pending.iterdir():
            if child == marker:
                continue
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        marker.unlink(missing_ok=True)
        pending.rmdir()
        fsync_dir(session_path)


#: Manifest phases whose pending data still needs a decision (recover or discard).
UNRESOLVED_PHASES = ("WRITING", "SEALED", "CHECKED")


class SessionLock:
    """Exclusive ownership of one session directory, held until explicitly released."""

    def __init__(self, session_path: Path) -> None:
        """Take a non-blocking exclusive lock; raises BlockingIOError if another owner holds the session."""
        directory = session_path / MANIFEST_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self._handle = open(directory / "session.lock", "w")  # noqa: SIM115 - held for the session's lifetime
        try:
            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise

    def release(self) -> None:
        """Release the lock; safe to call twice."""
        if not self._handle.closed:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
