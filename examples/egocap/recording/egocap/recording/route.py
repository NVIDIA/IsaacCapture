# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Local destination routing for kept episodes: studio, timestamped group, output slot, and group totals.

Layout: a kept episode ends at
``<output_root>/<Studio>/<timestamp>/episode_<k>``. A *group* (``<Studio>/<timestamp>``) is created once per
(studio, task, calendar day) for a recording session; ``k`` is the episode's slot in its group, assigned only when
the take is kept, so a rejected or discarded attempt never uses one while a kept-but-flagged episode does. The
group directory carries its own ``zarr.json`` with the native dataset-level attributes (episode count, total hours,
per-episode lengths, and the bad-episode record), which are always recomputed from the finalized episodes
themselves, never incremented, so a retry or a crash cannot double count.

The session's routes live in ``<session>/.recording/routes.json`` next to the manifests. A take's own destination
is the ``assignment`` frozen in its manifest, which is what a recovery uses; ``routes.json`` only records which
group the session has claimed for each (studio, task, day). Everything stays under the output root, on one
filesystem, so publication is a single atomic rename.

Pure helpers over the filesystem: nothing here imports capture, acquisition or device code.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from egocap.studio.studio_lookup import _normalize_desc, lookup_studio
from egocap.studio.studio_path_names import canonicalize_studio_dir_name, sanitize_studio_dir_name

from .episode_layout import MANIFEST_DIR, ROUTE_ATTR, fsync_dir, route_id, write_json_atomic

__all__ = ["RoutingConfig", "assign_output", "refresh_group_totals", "resolve_studio"]

ROUTES_FILE = "routes.json"
_FINAL_EPISODE = re.compile(r"^episode_(\d+)$")


@dataclass(frozen=True)
class RoutingConfig:
    """How a take's studio is chosen. The application resolves modes and themes into these inputs.

    ``session_studio`` overrides everything when set; otherwise the task description is looked up in
    ``studio_by_task`` (description, studio pairs); otherwise ``unmatched_studio`` is used.
    """

    session_studio: str = ""
    studio_by_task: tuple[tuple[str, str], ...] = ()
    known_studios: tuple[str, ...] = ()
    unmatched_studio: str = "_unmatched"

    def problems(self) -> list[str]:
        """Everything wrong with this configuration (empty when usable)."""
        problems = []
        if _unsafe_component(sanitize_studio_dir_name(self.unmatched_studio)):
            problems.append("routing unmatched_studio must be a non-empty, safe studio name")
        for name in (self.session_studio.strip(), *self.known_studios, *(s for _, s in self.studio_by_task)):
            if name and _unsafe_component(sanitize_studio_dir_name(name)):
                problems.append(f"routing studio {name!r} is not a safe directory name")
        for description, studio in self.studio_by_task:
            if not description.strip() or not studio.strip():
                problems.append(f"routing studio_by_task entry {(description, studio)!r} has an empty field")
        keys = [_normalize_desc(d) for d, _ in self.studio_by_task]
        if len(set(keys)) != len(keys):
            problems.append("routing studio_by_task has descriptions that normalize to the same text")
        return problems


def resolve_studio(task: str, config: RoutingConfig) -> str:
    """The studio for a task: the session override, else the task-to-studio map, else the unmatched bucket."""
    known = frozenset(config.known_studios)
    if config.session_studio.strip():
        return canonicalize_studio_dir_name(config.session_studio, known)
    found = lookup_studio(task, {_normalize_desc(d): s for d, s in config.studio_by_task})
    return canonicalize_studio_dir_name(found, known) if found else config.unmatched_studio


def _unsafe_component(name: str) -> bool:
    """True for a name that is not a plain, visible, single directory component (sanitizing alone leaves '..')."""
    return name in ("", ".", "..") or name.startswith(".") or "/" in name or "\\" in name or "\x00" in name


def _load_json(path: Path) -> dict:
    """A JSON object from ``path``; unreadable or malformed metadata fails clearly instead of becoming ``{}``."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is unreadable or not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def _load_routes(routes_file: Path) -> dict[str, str]:
    """The session's recorded routes (``{}`` only when none were ever recorded); any damage is an error."""
    if not routes_file.exists():
        return {}
    routes = _load_json(routes_file)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in routes.items()):
        raise ValueError(f"{routes_file} must map route keys to group names")
    return routes


def _require_group_under_studio(group: str, studio_dir: str) -> None:
    """Refuse a recorded group name that is not exactly ``<studio_dir>/<plain component>``."""
    parts = group.split("/")
    if len(parts) != 2 or parts[0] != studio_dir or _unsafe_component(parts[1]):
        raise ValueError(f"recorded route {group!r} is not a group under studio {studio_dir!r}")


def _refuse_symlink(*paths: Path) -> None:
    """Refuse to write routed output through a symlink, which could point outside the output root."""
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"{path} is a symlink; routed output is never written through one")


def _claim_group(group_dir: Path, attributes: dict) -> None:
    """Create ``group_dir`` and stamp it. ``mkdir`` is the atomic claim: it fails if anyone got there first."""
    os.mkdir(group_dir)
    write_json_atomic(group_dir / "zarr.json", {"attributes": attributes, "zarr_format": 3, "node_type": "group"})
    # The new entry must outlive a crash before a manifest points at it; the stamp write flushed the group itself.
    fsync_dir(group_dir.parent)


def _next_output_index(group_dir: Path, reserved: Iterable[int]) -> int:
    """The first slot after every finalized episode in the group and every slot already promised to a take."""
    used = set(reserved)
    for entry in group_dir.iterdir():
        match = _FINAL_EPISODE.match(entry.name)
        if match:
            used.add(int(match.group(1)))
    return max(used, default=-1) + 1


def assign_output(
    session_path: Path,
    session_id: str,
    studio: str,
    task: str,
    operator: str,
    reserved: Iterable[tuple[str, int]],
    now: datetime.datetime,
) -> tuple[str, int]:
    """The (group, output slot) for a take being kept; creates the group once per (studio, task, day).

    ``reserved`` is every (group, slot) already promised to another take of this session.

    A new group is claimed with an atomic ``mkdir`` (another session choosing the same name gets the next suffix),
    stamped, and only then recorded in ``routes.json``; an existing directory is never adopted just because it is
    there. A recorded route resumes with the same group, and one whose directory exists but is not stamped as this
    session's own refuses. Damaged routes or group metadata fail clearly, and no path is touched before the studio,
    the recorded group and every directory on the way have been validated.
    """
    output_root = session_path.parent
    studio_dir = sanitize_studio_dir_name(studio)
    if _unsafe_component(studio_dir):
        raise ValueError(f"studio {studio!r} is not a safe directory name")
    studio_path = output_root / studio_dir
    # Everything below is validated before the first filesystem mutation.
    _refuse_symlink(studio_path)
    routes_file = session_path / MANIFEST_DIR / ROUTES_FILE
    routes = _load_routes(routes_file)
    key = f"{studio_dir}|{task}|{now.date().isoformat()}"
    group = routes.get(key)
    if group is not None:
        _require_group_under_studio(group, studio_dir)
        _refuse_symlink(output_root / group)
    studio_path.mkdir(exist_ok=True)  # safe if another session creates it first
    fsync_dir(output_root)  # the studio directory's own entry
    attributes = {
        ROUTE_ATTR: "",
        "studio": studio,
        "task_description": task,
        "operator_name": operator,
        "discarded_episode_indices": [],
        "discarded_reasons": {},
    }
    if group is None:
        stamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        candidate, suffix = f"{studio_dir}/{stamp}", 1
        while True:
            if candidate not in routes.values():
                try:
                    _claim_group(
                        output_root / candidate, {**attributes, ROUTE_ATTR: route_id(session_id, candidate)}
                    )
                    break
                except FileExistsError:
                    pass  # another session (or an unrelated directory) holds this name: take the next one
            candidate = f"{studio_dir}/{stamp}_{suffix}"
            suffix += 1
        group = candidate
        routes[key] = group
        write_json_atomic(routes_file, routes)
    elif not (output_root / group).exists():
        # A recorded route whose directory vanished: claim it again under the same name.
        _claim_group(output_root / group, {**attributes, ROUTE_ATTR: route_id(session_id, group)})
    found = _load_json(output_root / group / "zarr.json").get("attributes", {})
    if found.get(ROUTE_ATTR) != route_id(session_id, group):
        raise PermissionError(f"{output_root / group} exists but is not a group owned by session {session_id!r}")
    # The recorded route must still describe this group: a route entry for another task must not alias it.
    if found.get("studio") != studio or found.get("task_description") != task:
        raise PermissionError(
            f"{output_root / group} holds {found.get('task_description')!r} in {found.get('studio')!r}, "
            f"not {task!r} in {studio!r}"
        )
    return group, _next_output_index(output_root / group, (index for g, index in reserved if g == group))


def refresh_group_totals(group_dir: Path) -> dict:
    """Recompute the group's dataset-level attributes from its finalized episodes and persist them.

    Counts and hours include every finalized episode, flagged-bad ones too (saved never means usable); the bad
    record lists them with their reasons. Rejected, discarded and pending takes are not in the group, so they are
    never counted. Derived from the episodes' own attributes, so repeating it changes nothing.
    """
    episodes = {}
    for entry in group_dir.iterdir():
        match = _FINAL_EPISODE.match(entry.name)
        if match and entry.is_dir():
            episodes[int(match.group(1))] = _load_json(entry / "zarr.json").get("attributes", {})
    bad = {k: a for k, a in sorted(episodes.items()) if a.get("bad_episode")}
    stamp_file = group_dir / "zarr.json"
    payload = _load_json(stamp_file)
    attributes = payload.setdefault("attributes", {})
    attributes.update(
        {
            "num_episodes": len(episodes),
            "total_data_time_hours": round(
                sum(a.get("episode_duration", 0.0) for a in episodes.values()) / 3600.0, 6
            ),
            "episode_lengths": {
                str(k): a["episode_length"] for k, a in sorted(episodes.items()) if "episode_length" in a
            },
            "discarded_episode_indices": list(bad),
            "discarded_reasons": {
                str(k): "; ".join(a.get("capture_qc_issues") or []) or "episode flagged bad"
                for k, a in bad.items()
            },
        }
    )
    write_json_atomic(stamp_file, payload)
    return attributes
