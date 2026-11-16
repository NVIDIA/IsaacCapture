# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Applies station registry changes to the running session.

The registry (stations.py) only stores values. This service decides when a change is allowed and what it does to
the live session: the station in use cannot change while a take is being recorded, and a change of the station in
use takes effect in process, without a restart. The capture session keeps running; the station label shown and
recorded changes at once, and the recording batch rotates so the next take opens a session carrying the new label.
Which cameras CaptureServer is asked for stays a deployment setting (see station_config.py).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .operator_settings import CAMERA_ROLES, CAMERA_TYPES
from .stations import Station, StationError, StationRegistry, StationStore

logger = logging.getLogger("isaaccapture.egocap.station_service")

#: Shown when a change would alter the station a take in progress is being recorded against.
TAKE_IN_PROGRESS = "finish or discard the current take before changing the station in use"


class StationSession(Protocol):
    """What this service needs from the live session; the application wires the real objects in."""

    def offered_cameras(self) -> frozenset[str] | None:
        """Names of the cameras the capture session offers, or None before a session has been accepted."""

    def change_blocked(self) -> str | None:
        """Why the station in use cannot change now, or None when it can."""

    def show_label(self, label: str) -> None:
        """Publish the station label every viewer sees."""

    async def rotate_batch(self) -> None:
        """End the current recording batch so the next take opens a session with the new label; raises
        :class:`StationError` when the session cannot rotate now."""

    def leave_station(self, label: str) -> None:
        """Close the activity segment of the station being left."""


class StationService:
    """One owner for the station in use: the registry, the label it implies, and the rules for changing either."""

    def __init__(self, store: StationStore, session: StationSession, default_label: str) -> None:
        """``default_label`` is the deployment's own station label, used while no station is selected."""
        self._store = store
        self._session = session
        self._default_label = default_label
        # One change at a time: a rotation awaits, and a second change must not interleave with it.
        self._changing = asyncio.Lock()
        self._warning: str | None = None
        self._registry = self._load()
        self._label = self._label_of(self._registry)

    def _load(self) -> StationRegistry:
        """The stored registry; an unreadable one starts empty with a warning the page shows."""
        try:
            return self._store.load()
        except StationError as exc:
            logger.warning("Station registry is unreadable (%s); starting with no stations", exc.public_message)
            self._warning = "Saved stations are unreadable. Adding a station replaces them."
            return StationRegistry()

    def _label_of(self, registry: StationRegistry) -> str:
        """The label recordings and viewers carry for a registry: the selected station's name, else the default."""
        selected = registry.selected
        return selected.name if selected is not None else self._default_label

    @property
    def registry(self) -> StationRegistry:
        """The registry as last loaded or changed by this process."""
        return self._registry

    def label(self) -> str:
        """The station label in effect now; read each time a recording session opens."""
        return self._label

    def block_reason(self) -> str | None:
        """Why recording is not allowed for station reasons, or None.

        An empty registry blocks nothing: that is the configuration-free default. Once stations exist, one must
        be selected and able to record.
        """
        if not self._registry.stations:
            return None
        selected = self._registry.selected
        if selected is None:
            return "select a station before recording"
        problems = selected.problems(self._session.offered_cameras())
        return f"station {selected.name!r}: {problems[0]}" if problems else None

    def listing(self, *, private: bool) -> dict[str, Any]:
        """The wire form of the registry. ``private`` adds each station's headset address, a LAN detail shown
        only to a caller holding the control token."""
        offered = self._session.offered_cameras()
        stations = []
        for station in self._registry.stations:
            record = {
                "id": station.station_id,
                "name": station.name,
                "camera_type": station.camera_type,
                "camera_roles": dict(station.camera_roles),
                "problems": station.problems(offered),
            }
            if private:
                record["headset_address"] = station.headset_address
            stations.append(record)
        listing: dict[str, Any] = {
            "stations": stations,
            "selected": self._registry.selected_id,
            "roles": list(CAMERA_ROLES),
            "camera_types": list(CAMERA_TYPES),
            "label": self._label,
            "locked_reason": self._session.change_blocked(),
        }
        if self._warning is not None:
            listing["warning"] = self._warning
        return listing

    async def add(self, fields: Any) -> Station:
        """Add a station; nothing about the session in use changes, so this is allowed during a take."""
        async with self._changing:
            after = await self._persist(lambda registry: registry.added(fields), touches_station_in_use=False)
            return after.stations[-1]

    async def update(self, station_id: str, changes: Any) -> None:
        """Change one station; changing the station in use is refused while a take is recorded."""
        async with self._changing:
            in_use = self._registry.selected_id == station_id
            await self._persist(
                lambda registry: registry.updated(station_id, changes), touches_station_in_use=in_use
            )

    async def remove(self, station_id: str) -> None:
        """Delete one station; deleting the station in use leaves nothing selected."""
        async with self._changing:
            in_use = self._registry.selected_id == station_id
            await self._persist(lambda registry: registry.removed(station_id), touches_station_in_use=in_use)

    async def select(self, station_id: str | None) -> None:
        """Put a station in use, or none for None."""
        async with self._changing:
            await self._persist(lambda registry: registry.selecting(station_id), touches_station_in_use=True)

    async def update_selected(self, changes: Any) -> Station | None:
        """Change the station in use, for callers that edit "this station" without naming it; None when
        nothing is selected."""
        async with self._changing:
            selected_id = self._registry.selected_id
            if selected_id is None:
                return None
            after = await self._persist(
                lambda registry: registry.updated(selected_id, changes), touches_station_in_use=True
            )
            return after.selected

    async def _persist(
        self, apply: Callable[[StationRegistry], StationRegistry], *, touches_station_in_use: bool
    ) -> StationRegistry:
        """Persist one change and apply its effect on the live session; the caller holds the change lock."""
        # Refused before anything is written, so a blocked change leaves the file as it was.
        if touches_station_in_use:
            blocked = self._session.change_blocked()
            if blocked is not None:
                raise StationError(blocked)
        # Validated in memory first: a refused change must not have rotated the batch.
        label = self._label_of(apply(self._registry))
        relabelled = label != self._label
        # Rotated before the file is written: a session still open under the old label must never
        # receive a take attributed to the new one, so a refused rotation refuses the whole change.
        if relabelled:
            await self._session.rotate_batch()
        _, after = await asyncio.to_thread(self._store.change, apply)
        self._registry = after
        self._warning = None
        if relabelled:
            previous, self._label = self._label, self._label_of(after)
            # The activity log is a file write; it stays off the event loop.
            await asyncio.to_thread(self._session.leave_station, previous)
            self._session.show_label(self._label)
        return after


def session_adapter(
    *,
    offered_cameras: Callable[[], frozenset[str] | None],
    change_blocked: Callable[[], str | None],
    show_label: Callable[[str], None],
    rotate_batch: Callable[[], Awaitable[None]],
    leave_station: Callable[[str], None],
) -> StationSession:
    """A :class:`StationSession` built from plain callables, so the application can bind the real monitor,
    recording context and activity log without this module importing any of them."""

    class _Adapter:
        """The callables above, under the protocol's method names."""

        def offered_cameras(self) -> frozenset[str] | None:
            """See :class:`StationSession`."""
            return offered_cameras()

        def change_blocked(self) -> str | None:
            """See :class:`StationSession`."""
            return change_blocked()

        def show_label(self, label: str) -> None:
            """See :class:`StationSession`."""
            show_label(label)

        async def rotate_batch(self) -> None:
            """See :class:`StationSession`."""
            await rotate_batch()

        def leave_station(self, label: str) -> None:
            """See :class:`StationSession`."""
            leave_station(label)

    return _Adapter()
