# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for the start-of-day flow: where each step stands, and running the device connection test.

Reading the flow is open to any page that can reach the backend, as the capture status is. Running the test
changes what recording is allowed to do, so it needs the run's control token.
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


def _state(request: Request) -> tuple[Any, Any, Any, Any] | None:
    """The setup service, station service, recording context and settings, or None until the lifespan set them."""
    state = request.app.state
    names = ("setup_service", "station_service", "recording_context", "settings")
    found = tuple(getattr(state, name, None) for name in names)
    return None if any(item is None for item in found) else found  # type: ignore[return-value]


async def _headset_address_changed(stations: Any, settings: Any) -> bool:
    """Whether the address saved for the station in use is no longer one of the capture host's."""
    selected = stations.registry.selected
    saved = selected.headset_address if selected is not None else None
    # Nothing saved means nothing to compare; the interfaces are not even listed.
    if saved is None:
        return False
    candidates = await asyncio.to_thread(headset_addresses, settings.capture.capture_server_url)
    return address_changed(saved, candidates)


async def _status(setup: Any, stations: Any, context: Any, settings: Any) -> JSONResponse:
    """The flow as the page reads it."""
    changed = await _headset_address_changed(stations, settings)
    return _reply(setup.status(saved_count=context.saved_count, headset_address_changed=changed))


@router.get("/api/setup")
async def read_setup(request: Request) -> JSONResponse:
    """Each step of the start-of-day flow, the last connection test, the system checks, and what blocks a take."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "setup is not available yet"}, 503)
    return await _status(*state)


@router.post("/api/setup/connection-test")
async def run_connection_test(request: Request) -> JSONResponse:
    """Test every device the station in use needs now; the result decides whether a take may start."""
    state = _state(request)
    if state is None:
        return _reply({"detail": "setup is not available yet"}, 503)
    setup, stations, context, settings = state
    if not control_token_valid(request.headers.get("x-control-token"), context.control_token):
        return _reply({"detail": "control token rejected"}, 403)
    setup.run_connection_test()
    return await _status(setup, stations, context, settings)
