# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for stations: the registry, the station in use, its activity and the headset address.

Reading the registry and today's activity is open to any page that can reach the backend, as the operator
settings are; a station's headset address and this machine's network addresses are LAN details and need the
run's control token. Every change needs that token, the same credential that admits recording commands.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .host_addresses import address_changed, headset_addresses
from .recording_command import control_token_valid
from .request_body import read_authorized_json
from .station_activity import StationActivityError
from .stations import StationError, StationNotFoundError

logger = logging.getLogger("isaaccapture.egocap.station_routes")

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}
#: A station record is a few short fields; activity notes are the largest body accepted here.
_MAX_BODY_BYTES = 32 * 1024


def _reply(body: dict[str, Any], status: int = 200) -> JSONResponse:
    """A non-cacheable JSON reply: station state changes under the page and must never be served stale."""
    return JSONResponse(body, status_code=status, headers=_NO_STORE)


def _unavailable() -> JSONResponse:
    """The reply while the application has not finished starting."""
    return _reply({"detail": "stations are not available yet"}, 503)


def _state(request: Request) -> tuple[Any, Any, Any] | None:
    """The station service, activity log and recording context, or None until the lifespan has set them."""
    state = request.app.state
    found = tuple(
        getattr(state, name, None) for name in ("station_service", "station_activity", "recording_context")
    )
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
    service, _activity, context = state
    return _reply(service.listing(private=_holds_token(request, context)))


async def _change(request: Request, change: Any) -> JSONResponse:
    """Run one authorized registry change and answer with the new listing."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, _activity, context = state
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
    service, _activity, context = state
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


@router.get("/api/headset-address")
async def read_headset_address(request: Request) -> JSONResponse:
    """The addresses to enter in the headset, the one saved for the station in use, and whether it is stale."""
    state = _state(request)
    settings = getattr(request.app.state, "settings", None)
    if state is None or settings is None:
        return _unavailable()
    service, _activity, context = state
    if not _holds_token(request, context):
        return _reply({"detail": "control token rejected"}, 403)
    # Interface enumeration is a blocking system call, kept off the event loop.
    candidates = await asyncio.to_thread(headset_addresses, settings.capture.capture_server_url)
    selected = service.registry.selected
    saved = selected.headset_address if selected is not None else None
    return _reply(
        {
            "candidates": [candidate.to_json() for candidate in candidates],
            "saved": saved,
            "changed": address_changed(saved, candidates),
        }
    )


@router.get("/api/station-activity")
async def read_station_activity(request: Request) -> JSONResponse:
    """Today's active and inactive time for the station in use."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, activity, _context = state
    return _reply(await asyncio.to_thread(activity.report, service.label()))


@router.post("/api/station-activity")
async def change_station_activity(request: Request) -> JSONResponse:
    """Mark the station in use active or inactive, save its blocker notes, or both."""
    state = _state(request)
    if state is None:
        return _unavailable()
    service, activity, context = state
    body, refused = await read_authorized_json(request, context.control_token, _MAX_BODY_BYTES)
    if body is None:
        return refused  # type: ignore[return-value]
    if "state" not in body and "notes" not in body:
        return _reply({"detail": "name a state, notes, or both"}, 400)
    try:
        report = await asyncio.to_thread(
            lambda: activity.change(service.label(), state=body.get("state"), notes=body.get("notes"))
        )
    except StationActivityError as exc:
        return _reply({"detail": str(exc)}, 400)
    except OSError:
        logger.exception("Could not save station activity")
        return _reply({"detail": "could not save station activity"}, 503)
    return _reply(report)
