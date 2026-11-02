# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deployment station YAML: resolved once at backend startup into one immutable station value.

``EGOCAP_STATION_CONFIG`` points at a YAML file describing this station's identity, operator,
initial task, recording output root and the cameras to request from CaptureServer. Unset
preserves the existing configuration-free development workflow (the deployment station label,
freeform task editing, no cameras requested). Set but unreadable or invalid is a hard
recording-configuration failure, never a silent fallback to a development default -- see
:class:`StationConfigError`. ``operator_name`` never auto-commits an active operator; whether it
even prefills the entry field's draft text is controlled separately by ``auto_fill_operator_name``
(default off), so an unconfigured or explicit-false station always starts with an empty draft that
the operator fills in themselves.

Only the fields in :class:`ResolvedStation` are consumed; everything else in the YAML (hardware,
visualization, or any other section) is read by nobody here and never passed downstream. The
``cameras`` field is itself a small ``name``/``type`` projection of ``video_sources`` -- the
fuller per-source declaration (``frame_type``, ``latency``, ``params``) is reference data for
other tooling, not consumed here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from capture_server import protocol as p

STATION_CONFIG_ENV = "EGOCAP_STATION_CONFIG"
_DEFAULT_FLAT_STATION_ID = "station_01"
#: Matches capture_server.protocol's own BeginRequest.cameras limit -- checked here too so an
#: over-limit YAML is a configuration error, not a later capture-startup failure.
_MAX_CAMERAS = 3


class StationConfigError(ValueError):
    """Raised for a configured-but-invalid station YAML. Must block new recordings, never be
    silently replaced by a development default -- the caller (app.py's lifespan) surfaces this as
    a readiness-check reason that blocks recording admission, NOT a backend startup crash: capture
    preview has nothing to do with recording configuration and must keep working regardless."""


@dataclass(frozen=True)
class ResolvedStation:
    """The one immutable station value startup resolves from YAML; nothing here is mutated after load."""

    station_id: str
    station_label: str | None
    operator_name: str
    auto_fill_operator_name: bool
    task_description: str | None
    data_dir: Path | None
    cameras: tuple[p.CameraSelection, ...]


class _DuplicateKeyLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that refuses a mapping with a repeated key, instead of silently keeping the last one."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set[Any] = set()
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise StationConfigError(f"duplicate key {key!r} in station configuration")
        seen.add(key)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _safe_load_no_duplicates(text: str) -> Any:
    return yaml.load(text, Loader=_DuplicateKeyLoader)


def _str_field(meta: dict, key: str) -> str | None:
    """A ``meta`` field that, if present, must be a string -- never silently coerced from another type."""
    if key not in meta:
        return None
    value = meta[key]
    if not isinstance(value, str):
        raise StationConfigError(f"{key!r} must be a string, got {type(value).__name__}")
    return value


def _bool_field(meta: dict, key: str, default: bool) -> bool:
    """A ``meta`` field that, if present, must be a bool -- never silently coerced from another type."""
    if key not in meta:
        return default
    value = meta[key]
    if not isinstance(value, bool):
        raise StationConfigError(f"{key!r} must be a boolean, got {type(value).__name__}")
    return value


def _resolve_cameras(station_id: str, entry: dict) -> tuple[p.CameraSelection, ...]:
    """The ``name``/``type`` client selection requested from CaptureServer, from ``video_sources``.
    An absent ``video_sources`` key is an explicit empty selection."""
    video_sources = entry.get("video_sources", [])
    if not isinstance(video_sources, list):
        raise StationConfigError(f"station {station_id!r}: 'video_sources' must be a list")
    if len(video_sources) > _MAX_CAMERAS:
        raise StationConfigError(f"station {station_id!r}: video_sources supports at most {_MAX_CAMERAS} entries")
    cameras: list[p.CameraSelection] = []
    seen_names: set[str] = set()
    for i, source in enumerate(video_sources):
        if not isinstance(source, dict):
            raise StationConfigError(f"station {station_id!r}: video_sources[{i}] must be a mapping")
        name = source.get("name")
        if not isinstance(name, str) or not name:
            raise StationConfigError(f"station {station_id!r}: video_sources[{i}].name must be a nonempty string")
        if name in seen_names:
            raise StationConfigError(f"station {station_id!r}: duplicate video source name {name!r}")
        seen_names.add(name)
        source_type = source.get("type")
        if source_type != "oak_camera":
            raise StationConfigError(
                f"station {station_id!r}: video_sources[{i}].type {source_type!r} is not 'oak_camera'"
            )
        cameras.append(p.CameraSelection(name=name, type=p.CameraType.OAK_D_COLOR))
    return tuple(cameras)


def _resolve_one_station(station_id: str, entry: Any, config_dir: Path) -> ResolvedStation:
    """Build a :class:`ResolvedStation` from one station's parsed mapping."""
    if not isinstance(entry, dict):
        raise StationConfigError(f"station {station_id!r} must be a mapping")
    meta = entry.get("meta", {})
    if not isinstance(meta, dict):
        raise StationConfigError(f"station {station_id!r}: 'meta' must be a mapping")
    recording = entry.get("recording", {})
    if not isinstance(recording, dict):
        raise StationConfigError(f"station {station_id!r}: 'recording' must be a mapping")

    data_dir_raw = recording.get("data_dir")
    data_dir: Path | None = None
    if data_dir_raw is not None:
        if not isinstance(data_dir_raw, str) or not data_dir_raw:
            raise StationConfigError(f"station {station_id!r}: recording.data_dir must be a nonempty string")
        candidate = Path(data_dir_raw).expanduser()
        # Relative paths resolve against the configuration file's own directory, not the process cwd.
        data_dir = candidate if candidate.is_absolute() else (config_dir / candidate)

    operator_name = _str_field(meta, "operator_name") or ""

    return ResolvedStation(
        station_id=station_id,
        station_label=_str_field(meta, "egocap_station_name"),
        operator_name=operator_name,
        # Off by default: an unconfigured or explicit-false station never prefills the operator
        # entry field, so a fresh browser always starts with the operator typing their own name.
        auto_fill_operator_name=_bool_field(meta, "auto_fill_operator_name", default=False),
        task_description=_str_field(meta, "task_description"),
        data_dir=data_dir,
        cameras=_resolve_cameras(station_id, entry),
    )


def load_station_config(path: Path | None) -> ResolvedStation | None:
    """Load and validate the station YAML at *path*.

    Returns ``None`` when *path* is ``None`` (``EGOCAP_STATION_CONFIG`` unset): the caller keeps the
    existing configuration-free development workflow in that case. A configured-but-invalid file
    raises :class:`StationConfigError`, which the caller must treat as a hard recording-configuration
    failure (blocking new recording admission) -- never caught here and silently replaced with a
    default, and never treated as a reason to fail backend startup itself.
    """
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StationConfigError(f"cannot read station configuration {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise StationConfigError(f"station configuration {path} is not valid UTF-8: {exc}") from exc
    try:
        data = _safe_load_no_duplicates(text)
    except yaml.YAMLError as exc:
        raise StationConfigError(f"station configuration {path} is not valid YAML: {exc}") from exc
    except (ValueError, TypeError) as exc:
        # Expected read/parse failures PyYAML doesn't wrap as YAMLError itself: an
        # out-of-range date/timestamp scalar (plain ValueError from the stdlib datetime
        # construction) or an unhashable mapping key (TypeError from our own duplicate-key
        # check above) -- both are configuration-content problems, not programming errors, so
        # they belong behind StationConfigError same as any other malformed-file case.
        raise StationConfigError(f"station configuration {path} is not valid: {exc}") from exc
    if not isinstance(data, dict):
        raise StationConfigError(f"station configuration {path} must be a mapping")

    config_dir = path.resolve().parent
    if "stations" in data:
        stations = data["stations"]
        if not isinstance(stations, dict) or not stations:
            raise StationConfigError(f"station configuration {path}: 'stations' must be a nonempty mapping")
        # dict preserves YAML document order (both PyYAML's default loader and ours construct in document
        # order), so the first key is the first entry in file order -- only the first station is read;
        # there is no selector for a file with more than one.
        first_id = next(iter(stations))
        return _resolve_one_station(first_id, stations[first_id], config_dir)

    flat_meta = data.get("meta", {})
    flat_meta = flat_meta if isinstance(flat_meta, dict) else {}
    station_id = _str_field(flat_meta, "egocap_station_name")
    return _resolve_one_station(station_id or _DEFAULT_FLAT_STATION_ID, data, config_dir)


def station_config_path_from_env() -> Path | None:
    """The configured station YAML path, or ``None`` when ``EGOCAP_STATION_CONFIG`` is unset/blank."""
    raw = os.environ.get(STATION_CONFIG_ENV)
    return Path(raw).expanduser() if raw else None
