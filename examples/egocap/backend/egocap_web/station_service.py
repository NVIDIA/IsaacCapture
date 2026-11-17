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
#: Why a take is held for the moment a change of the station in use is being applied.
STATION_CHANGING = "the station is being changed"


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

    def renamed(self, previous: str, name: str) -> None:
        """The station in use was renamed: carry its day (time and notes) over to the new name."""


class StationService:
    """One owner for the station in use: the registry, the label it implies, and the rules for changing either."""

    def __init__(self, store: StationStore, session: StationSession, default_label: str) -> None:
        """``default_label`` is the deployment's own station label, used while no station is selected."""
        self._store = store
        self._session = session
        self._default_label = default_label
        # One change at a time: a rotation awaits, and a second change must not interleave with it.
        self._changing = asyncio.Lock()
        # True while a change to the station in use is being applied; no take is admitted meanwhile.
        self._applying = False
        self._warning: str | None = None
        self._registry = self._load()
        self._label = self._label_of(self._registry)

    def _load(self) -> StationRegistry:
        """The stored registry; one that is invalid or cannot be read starts empty with a warning the page
        shows. Capture preview does not depend on stations, so neither stops the backend from starting."""
        try:
            return self._store.load()
        except StationError as exc:
            logger.warning("Station registry is unreadable (%s); starting with no stations", exc.public_message)
            self._warning = "Saved stations are unreadable. Adding a station replaces them."
        except OSError as exc:
            # Not set aside on the next change, unlike an invalid file: it may be valid and merely unreadable.
            logger.warning("Station registry cannot be read (%s); starting with no stations", exc.strerror)
            self._warning = "Saved stations cannot be read. Check the registry file's permissions."
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
        be selected and able to record. While the station in use is being changed nothing may start: the
        take would open a session under the label, or record against the cameras, on their way out.
        """
        if self._applying:
            return STATION_CHANGING
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

    async def remember_headset_address(self, station_id: str, observed: str | None, address: str) -> None:
        """Keep the capture host address a connected headset is evidently using, for the station in use.

        ``station_id`` and ``observed`` are the station the caller looked at and the address it had saved
        then. The caller decided across an await, so both are checked again here: another station put in use
        meanwhile, or an address the operator just entered, is left alone.

        It changes nothing about what is recorded, so unlike an edit of the station in use it is allowed while
        a take is being written.
        """
        async with self._changing:
            selected = self._registry.selected
            if selected is None or selected.station_id != station_id or selected.headset_address != observed:
                return
            await self._persist(
                lambda registry: registry.updated(station_id, {"headset_address": address}),
                touches_station_in_use=False,
            )

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
        candidate = apply(self._registry)
        label = self._label_of(candidate)
        relabelled = label != self._label
        # A rename keeps the same station in use; anything else that changes the label is a different station.
        renamed = (
            relabelled
            and candidate.selected_id is not None
            and candidate.selected_id == self._registry.selected_id
        )
        # Held across the awaits below, for every change to the station in use: a take admitted between them
        # would open a session under the old label, or record against camera assignments being replaced.
        self._applying = touches_station_in_use
        try:
            # Rotated before the file is written: a session still open under the old label must never
            # receive a take attributed to the new one, so a refused rotation refuses the whole change.
            if relabelled:
                await self._session.rotate_batch()
            _, after = await asyncio.to_thread(self._store.change, apply)
            self._registry = after
            self._warning = None
            if relabelled:
                previous, self._label = self._label, self._label_of(after)
                # The label is live from here; what follows only keeps the day's records in step with it.
                self._session.show_label(self._label)
        finally:
            self._applying = False
        if renamed:
            await self._carry_day(previous, self._label)
        return after

    async def _carry_day(self, previous: str, name: str) -> None:
        """Move the renamed station's day to its new name. A failure is logged, not raised: the rename itself
        is saved and live, and refusing it now would leave the page and the registry disagreeing."""
        try:
            await asyncio.to_thread(self._session.renamed, previous, name)
        except OSError:
            logger.exception("Could not carry the day's timeline from %r to %r", previous, name)


def session_adapter(
    *,
    offered_cameras: Callable[[], frozenset[str] | None],
    change_blocked: Callable[[], str | None],
    show_label: Callable[[str], None],
    rotate_batch: Callable[[], Awaitable[None]],
    renamed: Callable[[str, str], None],
) -> StationSession:
    """A :class:`StationSession` built from plain callables, so the application can bind the real monitor,
    recording context and uptime log without this module importing any of them."""

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

        def renamed(self, previous: str, name: str) -> None:
            """See :class:`StationSession`."""
            renamed(previous, name)

    return _Adapter()
