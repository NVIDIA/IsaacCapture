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
async def test_connect_opens_the_deferred_gate() -> None:
    gate = asyncio.Event()
    handler = _handler(gate)

    response = await handler(
        None, FakeRequest("/api/oob/v1/connect", {"X-Control-Token": TOKEN})
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "connectRequested": True,
        "alreadyRequested": False,
    }
    assert gate.is_set()


@pytest.mark.asyncio
async def test_connect_is_idempotent() -> None:
    """An orchestrator that retries must not be told it failed."""
    gate = asyncio.Event()
    handler = _handler(gate)
    request = FakeRequest(f"/api/oob/v1/connect?token={TOKEN}")

    await handler(None, request)
    response = await handler(None, request)

    assert response.status_code == 200
    assert json.loads(response.body)["alreadyRequested"] is True
    assert gate.is_set()


@pytest.mark.asyncio
async def test_connect_requires_the_control_token() -> None:
    gate = asyncio.Event()
    handler = _handler(gate)

    for headers in ({}, {"X-Control-Token": "wrong"}):
        response = await handler(None, FakeRequest("/api/oob/v1/connect", headers))
        assert response.status_code == 401

    assert not gate.is_set()


@pytest.mark.asyncio
async def test_connect_without_deferral_is_a_conflict() -> None:
    """Without defer_oob_connect setup already ran; say so rather than pretend."""
    handler = _handler(None)

    response = await handler(
        None, FakeRequest("/api/oob/v1/connect", {"X-Control-Token": TOKEN})
    )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_connect_response_has_no_cors_headers() -> None:
    """Like the other hub endpoints, it must not be usable cross-origin."""
    handler = _handler(asyncio.Event())
    request = FakeRequest("/api/oob/v1/connect", {"X-Control-Token": TOKEN})

    response = await handler(None, request)
    add_cors_headers(None, request, response)

    assert not any(k in response.headers for k in CORS_HEADERS)


@pytest.mark.asyncio
async def test_connect_is_not_served_without_a_hub() -> None:
    handler = _make_http_handler("localhost", 49100, hub=None)

    response = await handler(None, FakeRequest("/api/oob/v1/connect"))

    assert response.status_code == 404
