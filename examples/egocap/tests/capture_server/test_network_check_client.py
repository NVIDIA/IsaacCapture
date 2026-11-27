# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CaptureClient.run_network_check: sends the request on the open connection of an accepted session."""

from __future__ import annotations

import json

import pytest

from capture_server import client as c
from capture_server import protocol as p
from capture_server.client import CaptureClient


class _Transport:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)


def _client_with_attempt(*, accepted: bool, ws: object | None) -> CaptureClient:
    client = CaptureClient("ws://127.0.0.1:1")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    attempt = c._Attempt(session_id="session-1", begin=begin, callbacks={}, ws=ws)
    attempt.accepted = accepted
    client._attempt = attempt
    return client


async def test_run_network_check_without_a_session_raises():
    with pytest.raises(RuntimeError, match="accepted capture session"):
        await CaptureClient("ws://127.0.0.1:1").run_network_check()


async def test_run_network_check_before_acceptance_raises():
    client = _client_with_attempt(accepted=False, ws=_Transport())
    with pytest.raises(RuntimeError, match="accepted capture session"):
        await client.run_network_check()
    assert client._attempt.ws.sent == []


async def test_run_network_check_without_a_connection_raises():
    with pytest.raises(RuntimeError, match="accepted capture session"):
        await _client_with_attempt(accepted=True, ws=None).run_network_check()


async def test_run_network_check_sends_the_request_on_the_open_connection():
    ws = _Transport()
    await _client_with_attempt(accepted=True, ws=ws).run_network_check()
    (text,) = ws.sent
    request = p.parse_run_network_check(json.loads(text))
    assert request.session_id
    assert json.loads(text)["type"] == int(p.MessageType.RUN_NETWORK_CHECK)
