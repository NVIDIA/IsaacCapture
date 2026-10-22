# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CORS handling for the OOB hub HTTP API in ``wss``."""

from __future__ import annotations

import pytest
from cloudxr_py_test_ns.oob_teleop_hub import OOBControlHub
from cloudxr_py_test_ns.wss import (
    CORS_HEADERS,
    _make_http_handler,
    add_cors_headers,
)
from websockets.datastructures import Headers
from websockets.http11 import Response

TOKEN = "test-control-token"


class FakeRequest:
    def __init__(self, path: str, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {}


def _response() -> Response:
    return Response(200, "OK", Headers(), b"")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/api/oob/v1/state", "/api/oob/v1/config?serverIP=10.0.0.2"]
)
async def test_hub_api_responses_have_no_cors_headers(path: str) -> None:
    hub = OOBControlHub(control_token=TOKEN, initial_config={"serverIP": "10.0.0.1"})
    handler = _make_http_handler("localhost", 49100, hub=hub)
    request = FakeRequest(path, {"X-Control-Token": TOKEN})

    response = await handler(None, request)
    add_cors_headers(None, request, response)

    assert response.status_code == 200
    assert not any(k in response.headers for k in CORS_HEADERS)


@pytest.mark.asyncio
async def test_hub_api_rejection_has_no_cors_headers() -> None:
    hub = OOBControlHub(control_token=TOKEN)
    handler = _make_http_handler("localhost", 49100, hub=hub)
    request = FakeRequest("/api/oob/v1/state")

    response = await handler(None, request)
    add_cors_headers(None, request, response)

    assert response.status_code == 401
    assert "Access-Control-Allow-Origin" not in response.headers


@pytest.mark.parametrize("path", ["/", "/oob/v1/ws", "/client/index.html"])
def test_other_responses_keep_cors_headers(path: str) -> None:
    response = _response()
    add_cors_headers(None, FakeRequest(path), response)
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_hub_api_path_with_traversal_still_skips_cors() -> None:
    response = _response()
    add_cors_headers(None, FakeRequest("/x/../api/oob/v1/state"), response)
    assert "Access-Control-Allow-Origin" not in response.headers
