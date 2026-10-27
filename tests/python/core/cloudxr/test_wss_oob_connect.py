# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``/api/oob/v1/connect``: releasing a deferred headset setup over HTTP."""

from __future__ import annotations

import asyncio
import json

import pytest
from cloudxr_py_test_ns.oob_teleop_hub import OOBControlHub
from cloudxr_py_test_ns.wss import CORS_HEADERS, _make_http_handler, add_cors_headers

TOKEN = "test-control-token"
PATH = "/api/oob/v1/connect"


class FakeRequest:
    def __init__(self, path: str, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {}


def _handler(connect_requested: asyncio.Event | None):
    hub = OOBControlHub(control_token=TOKEN)
    return _make_http_handler(
        "localhost", 49100, hub=hub, connect_requested=connect_requested
    )


@pytest.mark.asyncio
async def test_connect_opens_the_deferred_gate_once() -> None:
    """A retrying orchestrator is told it already succeeded, not that it failed."""
    gate = asyncio.Event()
    handler = _handler(gate)
    request = FakeRequest(PATH, {"X-Control-Token": TOKEN})

    first = await handler(None, request)
    second = await handler(None, request)

    assert gate.is_set()
    assert [r.status_code for r in (first, second)] == [200, 200]
    assert [json.loads(r.body)["alreadyRequested"] for r in (first, second)] == [
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_connect_refusals() -> None:
    """Bad token: 401 and the gate stays shut. No deferral: 409."""
    gate = asyncio.Event()
    handler = _handler(gate)
    for headers in ({}, {"X-Control-Token": "wrong"}):
        response = await handler(None, FakeRequest(PATH, headers))
        assert response.status_code == 401
    assert not gate.is_set()

    response = await _handler(None)(None, FakeRequest(PATH, {"X-Control-Token": TOKEN}))
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_connect_is_a_hub_endpoint() -> None:
    """Like the other hub endpoints: 404 without a hub, no CORS headers."""
    response = await _make_http_handler("localhost", 49100, hub=None)(
        None, FakeRequest(PATH)
    )
    assert response.status_code == 404

    request = FakeRequest(PATH, {"X-Control-Token": TOKEN})
    response = await _handler(asyncio.Event())(None, request)
    add_cors_headers(None, request, response)
    assert not any(k in response.headers for k in CORS_HEADERS)
