# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for the start-of-day flow: where each step stands and the live station checklist.

Reading the flow is open to any page that can reach the backend, as the capture status is. The capture host
address to enter in the headset is a LAN detail, so it is included only when the request carries the run's
control token. Nothing here changes anything: the checklist is observed, not run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .host_addresses import address_changed, headset_addresses
from .recording_command import control_token_valid

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


def _reply(body: dict[str, Any], status: int = 200) -> JSONResponse:
    """A non-cacheable JSON reply: the flow changes under the page."""
    return JSONResponse(body, status_code=status, headers=_NO_STORE)


def _state(request: Request) -> tuple[Any, ...] | None:
    """The services this route reads, or None until the lifespan has set them all."""
    state = request.app.state
    names = ("setup_service", "station_service", "station_uptime", "recording_context", "settings")
    found = tuple(getattr(state, name, None) for name in names)
    return None if any(item is None for item in found) else found


async def headset_address(setup: Any, stations: Any, capture_server_url: str) -> dict[str, Any]:
    """The capture host address to enter in the headset, the one remembered for the station in use, and
    whether the remembered one stopped being valid.

    A connected headset is proof that the address it was given works, so nothing is reported as changed while
    one is connected.
    """
    # Interface enumeration and name resolution are blocking system calls, kept off the event loop.
    candidates = await asyncio.to_thread(headset_addresses, capture_server_url)
    selected = stations.registry.selected
    saved = selected.headset_address if selected is not None else None
    return {
        "address": candidates[0].address if candidates else None,
        "saved": saved,
        "changed": address_changed(saved, candidates) and not setup.headset_connected(),
    }


def _day_so_far(uptime: Any, station: str) -> tuple[bool, bool]:
    """Whether the station has recorded a take today, and whether today's report was saved."""
    return uptime.took_today(station), uptime.report_saved(station)


@router.get("/api/setup")
async def read_setup(request: Request) -> JSONResponse:
    """Each step of the start-of-day flow, the station checklist as it stands now, and what holds a take."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "setup is not available yet"}, 503)
    setup, stations, uptime, context, settings = state
    headset = await headset_address(setup, stations, settings.capture.capture_server_url)
    # Both read the day's files, so they stay off the event loop.
    took_today, report_saved = await asyncio.to_thread(_day_so_far, uptime, stations.label())
    body = setup.status(
        took_today=took_today,
        saved_count=context.saved_count,
        report_saved=report_saved,
        headset_address_changed=headset["changed"],
    )
    holds_token = control_token_valid(request.headers.get("x-control-token"), context.control_token)
    return _reply({**body, "headset": headset if holds_token else None})
