# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for stations: the registry, the station in use, and its time today.

Reading the registry and the station's time is open to any page that can reach the backend, as the operator
settings are. A station's headset address is a LAN detail and the day's notes are the operator's free text, so
both are returned only with the run's control token. Every change needs that token, the same credential that
admits recording commands.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .recording_command import control_token_valid
from .request_body import read_authorized_json
from .station_uptime import notes_problem
from .stations import StationError, StationNotFoundError

logger = logging.getLogger("isaaccapture.egocap.station_routes")

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}
#: A station record is a few short fields; the day's notes are the largest body accepted here.
_MAX_BODY_BYTES = 32 * 1024


def _reply(body: dict[str, Any], status: int = 200) -> JSONResponse:
    """A non-cacheable JSON reply: station state changes under the page and must never be served stale."""
    return JSONResponse(body, status_code=status, headers=_NO_STORE)


def _unavailable() -> JSONResponse:
    """The reply while the application has not finished starting."""
    return _reply({"detail": "stations are not available yet"}, 503)


def _state(request: Request) -> tuple[Any, Any, Any] | None:
    """The station service, uptime log and recording context, or None until the lifespan has set them."""
    state = request.app.state
    names = ("station_service", "station_uptime", "recording_context")
    found = tuple(getattr(state, name, None) for name in names)
    return None if any(item is None for item in found) else found  # type: ignore[return-value]


def _holds_token(request: Request, context: Any) -> bool:
    """Whether a read carries the run's control token, which unlocks the LAN details."""
    return control_token_valid(request.headers.get("x-control-token"), context.control_token)


def _refusal(exc: StationError) -> JSONResponse:
    """A refused station change as its HTTP status: an unknown id is not found, anything else a bad request."""
    return _reply({"detail": exc.public_message}, 404 if isinstance(exc, StationNotFoundError) else 400)


@router.get("/api/stations")
async def list_stations(request: Request) -> JSONResponse:
    """Every station, the one in use, and why each cannot record yet."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, _uptime, context = state
    return _reply(service.listing(private=_holds_token(request, context)))


async def _change(request: Request, change: Any) -> JSONResponse:
    """Run one authorized registry change and answer with the new listing."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, _uptime, context = state
    body, refused = await read_authorized_json(request, context.control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    try:
        await change(service, body)
    except StationError as exc:
        return _refusal(exc)
    except OSError:
        logger.exception("Could not save the station registry")
        return _reply({"detail": "could not save the station registry"}, 503)
    return _reply(service.listing(private=True))


@router.post("/api/stations")
async def add_station(request: Request) -> JSONResponse:
    """Add a station from its name and, optionally, its hardware."""
    return await _change(request, lambda service, body: service.add(body))


@router.post("/api/stations/select")
async def select_station(request: Request) -> JSONResponse:
    """Put one station in use, or none for ``{"id": null}``; refused while a take is being recorded."""

    async def select(service: Any, body: dict[str, Any]) -> None:
        """Validate the id's shape here; whether it names a station is the registry's rule."""
        station_id = body.get("id")
        if station_id is not None and not isinstance(station_id, str):
            raise StationError("id must be a station id or null")
        await service.select(station_id)

    return await _change(request, select)


@router.put("/api/stations/{station_id}")
async def update_station(request: Request, station_id: str) -> JSONResponse:
    """Change the named fields of one station."""
    return await _change(request, lambda service, body: service.update(station_id, body))


@router.delete("/api/stations/{station_id}")
async def delete_station(request: Request, station_id: str) -> JSONResponse:
    """Delete one station. The credential travels in the header: a DELETE carries no body to bound or parse."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, _uptime, context = state
    if not _holds_token(request, context):
        return _reply({"detail": "control token rejected"}, 403)
    try:
        await service.remove(station_id)
    except StationError as exc:
        return _refusal(exc)
    except OSError:
        logger.exception("Could not save the station registry")
        return _reply({"detail": "could not save the station registry"}, 503)
    return _reply(service.listing(private=True))


@router.get("/api/station-uptime")
async def read_station_uptime(request: Request) -> JSONResponse:
    """What the station in use is doing now, and its active and down time between today's takes. The day's
    notes are left empty unless the request carries the control token."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, uptime, context = state
    report = await asyncio.to_thread(uptime.report, service.label())
    return _reply(report if _holds_token(request, context) else {**report, "notes": ""})


@router.post("/api/station-uptime")
async def save_station_notes(request: Request) -> JSONResponse:
    """Save the operator's notes for the station's day. The state itself is observed, never set.

    The body names the station the notes were written for. A draft begun before the station in use changed
    is refused, so it never replaces the notes of the station now in use.
    """
    state = _state(request)
    if state is None:
        return _unavailable()
    service, uptime, context = state
    body, refused = await read_authorized_json(request, context.control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    station, notes = body.get("station"), body.get("notes")
    if not isinstance(station, str):
        return _reply({"detail": "station must name the station the notes are for"}, 400)
    if station != service.label():
        return _reply({"detail": "the station in use changed; reload the report"}, 409)
    problem = notes_problem(notes)
    if problem is not None:
        return _reply({"detail": problem}, 400)
    try:
        # Saved under the name just checked, so a change of station during the write cannot redirect it.
        return _reply(await asyncio.to_thread(uptime.set_notes, station, notes))
    except OSError:
        logger.exception("Could not save the station's notes")
        return _reply({"detail": "could not save the notes"}, 503)
