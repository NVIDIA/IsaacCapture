# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mutation bodies are bounded independently of the transport's declared length."""

import asyncio
import json

import pytest
from starlette.requests import Request

from egocap_web.request_body import read_authorized_json

TOKEN = "t" * 43


def _read(chunks, *, headers=None, limit=128):
    consumed = []
    pending = iter(chunks)

    async def receive():
        chunk = next(pending)
        consumed.append(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    request = Request(
        {"type": "http", "headers": [(key.encode(), value.encode()) for key, value in (headers or {}).items()]},
        receive,
    )
    return asyncio.run(read_authorized_json(request, TOKEN, limit)), consumed


def test_rejected_header_does_not_consume_even_one_chunk():
    (body, response), consumed = _read([b"never read"], headers={"x-control-token": "wrong"})
    assert body is None and response.status_code == 403
    assert consumed == []


def test_declared_oversize_is_rejected_without_consuming_the_body():
    (_, response), consumed = _read([b"never read"], headers={"content-length": "129"})
    assert response.status_code == 413
    assert consumed == []
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("headers", [{}, {"content-length": "1"}, {"x-control-token": TOKEN}])
def test_stream_limit_stops_consuming_before_remaining_chunks(headers):
    (_, response), consumed = _read([b"x" * 64, b"y" * 65, b"never read"], headers=headers)
    assert response.status_code == 413
    assert len(consumed) == 2


@pytest.mark.parametrize("length", ["not a number", "-1"])
def test_invalid_declared_length_is_rejected_without_reading(length):
    (_, response), consumed = _read([b"never read"], headers={"content-length": length})
    assert response.status_code == 400 and consumed == []


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]"])
def test_invalid_json_returns_a_public_client_error(raw):
    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    request = Request({"type": "http", "headers": [(b"x-control-token", TOKEN.encode())]}, receive)
    body, response = asyncio.run(read_authorized_json(request, TOKEN, 128))
    assert body is None and response.status_code == 400


@pytest.mark.parametrize("legacy", [True, False])
def test_exact_limit_accepts_header_and_legacy_body_credentials(legacy):
    expected = {"camera_type": "oak_d"}
    if legacy:
        expected["control_token"] = TOKEN
    raw = json.dumps(expected).encode()

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    headers = [] if legacy else [(b"x-control-token", TOKEN.encode())]
    request = Request({"type": "http", "headers": headers}, receive)
    body, response = asyncio.run(read_authorized_json(request, TOKEN, len(raw)))
    assert response is None and body == expected
