# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The station registry: the stations an operator can choose between, and which one is in use.

A station is a name plus the hardware that fills its roles: the camera type, which offered camera is the ego view
and which is on each wrist, and the host address last given to the headset. The registry is one JSON file beside
the operator settings, so every browser sees the same stations and they survive a restart. An empty registry is
the configuration-free default: nothing is selected, the deployment's own station label applies, and no station
rule blocks recording.

This module holds the values and their validation only. Applying a selection to the running session lives in
station_service.py, and the HTTP surface in station_routes.py.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import secrets
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .operator_settings import (
    CAMERA_ROLES,
    CAMERA_TYPES,
    OperatorSettingsError,
    parse_camera_roles,
)

logger = logging.getLogger("isaaccapture.egocap.stations")

#: A registry is a site's stations, not a database; the bound keeps the file and the page small.
MAX_STATIONS = 32
MAX_STATION_NAME_CHARS = 64

#: The role every station must fill before it can record: without the operator's view a take has no subject.
REQUIRED_ROLE = "ego"

#: Operator-facing role names for messages; the page keeps its own copy for labels.
_ROLE_NAMES = {"ego": "ego (head)", "wrist_left": "left wrist", "wrist_right": "right wrist"}

#: The fields a create or update may carry. ``control_token`` is the legacy body credential, already checked.
_EDITABLE_FIELDS = ("name", "camera_type", "camera_roles", "headset_address")
_IGNORED_FIELDS = ("control_token",)


class StationError(OperatorSettingsError):
    """A station change that was refused, with a public, path-free reason naming what to fix."""


class StationNotFoundError(StationError):
    """A request named a station id the registry does not hold."""


def _clean_name(raw: Any) -> str:
    """A station name with surrounding and repeated whitespace removed, or a refusal saying what is wrong."""
    if not isinstance(raw, str):
        raise StationError("station name must be text")
    # Names are compared and shown, so "Bay  1" and "Bay 1" must not become two stations.
    name = " ".join(raw.split())
    if name == "":
        raise StationError("station name cannot be empty")
    if len(name) > MAX_STATION_NAME_CHARS:
        raise StationError(f"station name is limited to {MAX_STATION_NAME_CHARS} characters")
    # The name is recorded with every session and used in a log file name; control characters have no place there.
    if not name.isprintable():
        raise StationError("station name cannot contain control characters")
    return name


def _clean_camera_type(raw: Any) -> str | None:
    """A camera type the backend knows, or None for not set."""
    if raw is not None and raw not in CAMERA_TYPES:
        raise StationError(f"camera_type must be one of {', '.join(CAMERA_TYPES)} or null")
    return raw


def _clean_headset_address(raw: Any) -> str | None:
    """The host address last given to the headset, as a canonical IP literal, or None when never set."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise StationError("headset_address must be an IP address or null")
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        raise StationError(f"headset_address {raw.strip()!r} is not an IP address") from None


@dataclass(frozen=True)
class Station:
    """One station: its name and the hardware that fills its roles."""

    station_id: str
    name: str
    camera_type: str | None = None
    #: Which offered camera fills each role; every role is present, None where unassigned.
    camera_roles: dict[str, str | None] = field(default_factory=lambda: {role: None for role in CAMERA_ROLES})
    #: The capture host address last entered in the headset; compared with today's to warn of a change.
    headset_address: str | None = None

    def stored(self) -> dict[str, Any]:
        """The record as written to the registry file."""
        return {
            "id": self.station_id,
            "name": self.name,
            "camera_type": self.camera_type,
            "camera_roles": dict(self.camera_roles),
            "headset_address": self.headset_address,
        }

    def problems(self, offered_cameras: frozenset[str] | None) -> list[str]:
        """Why this station cannot record yet, most important first; empty when it can.

        ``offered_cameras`` is what the running capture session offers, or None while that is not known yet, in
        which case only the assignments themselves are judged.
        """
        found: list[str] = []
        if self.camera_roles[REQUIRED_ROLE] is None:
            found.append(f"no camera is assigned to the {_ROLE_NAMES[REQUIRED_ROLE]} role")
        if offered_cameras is None:
            return found
        # A role pointing at a camera the session does not offer would record nothing for that role.
        for role in CAMERA_ROLES:
            camera = self.camera_roles[role]
            if camera is not None and camera not in offered_cameras:
                found.append(f"camera {camera!r} ({_ROLE_NAMES[role]}) is not offered by the capture session")
        return found


def _parse_station(raw: Any) -> Station:
    """One stored record as a station; a malformed record is an error, never a guess."""
    if not isinstance(raw, dict):
        raise StationError("each station must be an object")
    station_id = raw.get("id")
    if not isinstance(station_id, str) or station_id == "":
        raise StationError("each station needs an id")
    return Station(
        station_id=station_id,
        name=_clean_name(raw.get("name")),
        camera_type=_clean_camera_type(raw.get("camera_type")),
        camera_roles=_station_roles(raw.get("camera_roles")),
        headset_address=_clean_headset_address(raw.get("headset_address")),
    )


def _station_roles(raw: Any) -> dict[str, str | None]:
    """Role assignments for a station, refused as a station error so the route reports one error family."""
    try:
        return parse_camera_roles(raw)
    except OperatorSettingsError as exc:
        raise StationError(exc.public_message) from None


def _checked_fields(changes: Any) -> dict[str, Any]:
    """The editable fields a request names; an unknown field is refused so a typo is not silently dropped."""
    if not isinstance(changes, dict):
        raise StationError("a station change must be an object")
    unknown = sorted(set(changes) - set(_EDITABLE_FIELDS) - set(_IGNORED_FIELDS))
    if unknown:
        raise StationError(f"unknown station field(s): {', '.join(unknown)}")
    return {key: changes[key] for key in _EDITABLE_FIELDS if key in changes}


@dataclass(frozen=True)
class StationRegistry:
    """Every station and the selected one. Immutable: each change returns a new registry."""

    stations: tuple[Station, ...] = ()
    selected_id: str | None = None

    @property
    def selected(self) -> Station | None:
        """The station in use, or None when nothing is selected."""
        return self.find(self.selected_id) if self.selected_id is not None else None

    def find(self, station_id: str) -> Station | None:
        """The station with this id, or None."""
        return next((station for station in self.stations if station.station_id == station_id), None)

    def require(self, station_id: str) -> Station:
        """The station with this id; an unknown id is a refusal the route reports as not found."""
        station = self.find(station_id)
        if station is None:
            raise StationNotFoundError("no such station")
        return station

    def stored(self) -> dict[str, Any]:
        """The registry as written to its file."""
        return {"selected": self.selected_id, "stations": [station.stored() for station in self.stations]}

    def _refuse_duplicate_name(self, name: str, except_id: str | None) -> None:
        """Refuse a name another station already has, ignoring case, so two stations never share a log."""
        for other in self.stations:
            if other.station_id != except_id and other.name.casefold() == name.casefold():
                raise StationError(f"a station named {other.name!r} already exists")

    def added(
        self, fields: Any, new_id: Callable[[], str] = lambda: f"st_{secrets.token_hex(4)}"
    ) -> StationRegistry:
        """This registry with one more station built from ``fields``; the new station is last."""
        if len(self.stations) >= MAX_STATIONS:
            raise StationError(f"a registry holds at most {MAX_STATIONS} stations")
        checked = _checked_fields(fields)
        name = _clean_name(checked.get("name"))
        self._refuse_duplicate_name(name, except_id=None)
        # A colliding id would make one station shadow another; draw until it is free.
        station_id = new_id()
        while self.find(station_id) is not None:
            station_id = new_id()
        station = Station(
            station_id=station_id,
            name=name,
            camera_type=_clean_camera_type(checked.get("camera_type")),
            camera_roles=_station_roles(checked.get("camera_roles")),
            headset_address=_clean_headset_address(checked.get("headset_address")),
        )
        return replace(self, stations=(*self.stations, station))

    def updated(self, station_id: str, changes: Any) -> StationRegistry:
        """This registry with the named fields of one station replaced; a role change may name only some roles."""
        current = self.require(station_id)
        checked = _checked_fields(changes)
        merged = current
        if "name" in checked:
            name = _clean_name(checked["name"])
            self._refuse_duplicate_name(name, except_id=station_id)
            merged = replace(merged, name=name)
        if "camera_type" in checked:
            merged = replace(merged, camera_type=_clean_camera_type(checked["camera_type"]))
        if "camera_roles" in checked:
            roles = checked["camera_roles"]
            if not isinstance(roles, dict):
                raise StationError("camera_roles must be an object of role to camera name or null")
            merged = replace(merged, camera_roles=_station_roles({**current.camera_roles, **roles}))
        if "headset_address" in checked:
            merged = replace(merged, headset_address=_clean_headset_address(checked["headset_address"]))
        return replace(
            self,
            stations=tuple(merged if station.station_id == station_id else station for station in self.stations),
        )

    def removed(self, station_id: str) -> StationRegistry:
        """This registry without one station; removing the selected station leaves nothing selected."""
        self.require(station_id)
        return StationRegistry(
            stations=tuple(station for station in self.stations if station.station_id != station_id),
            selected_id=None if self.selected_id == station_id else self.selected_id,
        )

    def selecting(self, station_id: str | None) -> StationRegistry:
        """This registry with ``station_id`` in use, or nothing in use for None."""
        if station_id is not None:
            self.require(station_id)
        return replace(self, selected_id=station_id)


def parse_registry(raw: Any) -> StationRegistry:
    """A stored object as a registry: unique ids and names, and a selection that names a stored station."""
    if not isinstance(raw, dict):
        raise StationError("the station registry must be an object")
    records = raw.get("stations", [])
    if not isinstance(records, list) or len(records) > MAX_STATIONS:
        raise StationError(f"stations must be a list of at most {MAX_STATIONS}")
    registry = StationRegistry()
    for record in records:
        station = _parse_station(record)
        # A hand-edited file must hold to the same uniqueness the page enforces.
        if registry.find(station.station_id) is not None:
            raise StationError(f"station id {station.station_id!r} appears twice")
        registry._refuse_duplicate_name(station.name, except_id=None)
        registry = replace(registry, stations=(*registry.stations, station))
    selected = raw.get("selected")
    if selected is not None and (not isinstance(selected, str) or registry.find(selected) is None):
        raise StationError("the selected station is not in the registry")
    return replace(registry, selected_id=selected)


class StationStore:
    """The registry file: absent means no stations, and every write replaces it atomically."""

    def __init__(self, path: Path) -> None:
        """Bind the file and the lock that serialises changes within this backend process."""
        self.path = path
        self.lock = threading.RLock()

    def load(self) -> StationRegistry:
        """The stored registry, or an empty one when nothing has been saved."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return StationRegistry()
        except UnicodeError:
            raise StationError("the station registry is not valid UTF-8") from None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raise StationError("the station registry is not valid JSON") from None
        return parse_registry(raw)

    def save(self, registry: StationRegistry) -> None:
        """Write the registry through a uniquely named temporary file, so a crash never leaves half of one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
        ) as handle:
            staged = Path(handle.name)
            try:
                handle.write(json.dumps(registry.stored(), indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(staged, self.path)
            finally:
                staged.unlink(missing_ok=True)

    def change(
        self, apply: Callable[[StationRegistry], StationRegistry]
    ) -> tuple[StationRegistry, StationRegistry]:
        """Apply one change under the lock and persist it; returns the registry before and after.

        An unreadable registry is set aside as ``<name>.invalid`` and replaced, so the page can repair a station
        list without shell access while the bad file stays available to inspect.
        """
        with self.lock:
            try:
                before = self.load()
            except StationError:
                logger.warning("Setting aside an unreadable station registry at %s", self.path)
                os.replace(self.path, self.path.with_name(f"{self.path.name}.invalid"))
                before = StationRegistry()
            after = apply(before)
            self.save(after)
            return before, after


def station_store_path(operator_settings_file: Path) -> Path:
    """The registry file, kept beside the operator settings so both stay out of the recordings browser."""
    return operator_settings_file.with_name("stations.json")
