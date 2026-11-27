# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from capture_server import protocol as p
from capture_server.client import CaptureClient, ServerStatusQueryError
from egocap_web.app import app
from egocap_web.capture_status import CaptureStatusMonitor
from egocap_web.config import CaptureStatusConfig

URL = "ws://127.0.0.1:9"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.state, "capture_server_url", URL, raising=False)
    monitor = CaptureStatusMonitor(CaptureStatusConfig(capture_server_url=URL), backend_instance_id="b1")
    monkeypatch.setattr(app.state, "monitor", monitor, raising=False)
    return TestClient(app)


def _reply() -> p.ServerStatusReply:
    return p.ServerStatusReply(
        request_id="r-1",
        mode=p.CaptureMode.MOCK,
        session_id=None,
        status=None,
        network_check=None,
        cameras=(p.ServerCamera(name="head", transport="livekit", selected=False, room="r", track="t"),),
    )


def test_the_relay_returns_the_servers_reply(client, monkeypatch):
    urls: list[str] = []
    original_init = CaptureClient.__init__

    def init(self, url, *args, **kwargs):
        urls.append(url)
        original_init(self, url, *args, **kwargs)

    async def query_status(self):
        return _reply()

    monkeypatch.setattr(CaptureClient, "__init__", init)
    monkeypatch.setattr(CaptureClient, "query_status", query_status)
    response = client.get("/api/server-status")
    assert response.status_code == 200
    assert response.json() == _reply().to_json()
    assert response.headers["cache-control"] == "no-store"
    assert urls == [URL]


def test_an_unreachable_server_is_a_503_with_a_detail(client, monkeypatch):
    async def query_status(self):
        raise ServerStatusQueryError("the status query failed: refused")

    monkeypatch.setattr(CaptureClient, "query_status", query_status)
    response = client.get("/api/server-status")
    assert response.status_code == 503
    assert response.json() == {"detail": "CaptureServer did not answer the status query"}
    assert "refused" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_capture_status_is_unchanged_and_never_queries_the_server(client, monkeypatch):
    calls = SimpleNamespace(n=0)

    async def query_status(self):
        calls.n += 1
        return _reply()

    monkeypatch.setattr(CaptureClient, "query_status", query_status)
    response = client.get("/api/capture-status")
    assert response.status_code == 200
    assert response.json() == app.state.monitor.snapshot().to_json()
    assert "cameras" not in response.json()
    assert calls.n == 0


def test_a_reply_with_a_status_carries_the_nanosecond_clock_as_a_decimal_string(client, monkeypatch):
    updated_at_ns = 1795534562681000000
    snapshot = p.StatusSnapshot(
        updated_at_ns=updated_at_ns,
        providers=(
            p.ProviderStatus(
                id="prov",
                type=next(iter(p.ProviderType)),
                plugin=None,
                name="provider",
                status=next(iter(p.ProviderState)),
                reason=next(iter(p.StatusReason)),
                error="",
            ),
        ),
        devices=(
            p.DeviceStatus(
                id="dev",
                path="/dev/x",
                type="camera",
                description="d",
                provider_id="prov",
                status=next(iter(p.DeviceState)),
                reason=next(iter(p.StatusReason)),
                error="",
            ),
        ),
        cameras=(
            p.CameraStatus(
                name="head",
                stream_id=1,
                status=next(iter(p.DeviceState)),
                reason=next(iter(p.StatusReason)),
                error="",
            ),
        ),
        network_check=p.NetworkCheckStatus(
            state=p.NetworkCheckState.READY,
            session_id="s-1",
            generation="gen-1",
            endpoint="LiveKit wss://lk, room r, publisher o",
            legs_tested=("a",),
            legs_not_tested=("b",),
            window_s=1.0,
            valid_for_s=100.0,
        ),
    )
    reply = p.ServerStatusReply(
        request_id="r-2",
        mode=p.CaptureMode.MOCK,
        session_id="s-1",
        status=snapshot,
        network_check=snapshot.network_check,
        cameras=(p.ServerCamera(name="head", transport="livekit", selected=True, room="r", track="t"),),
    )

    async def query_status(self):
        return reply

    monkeypatch.setattr(CaptureClient, "query_status", query_status)
    response = client.get("/api/server-status")
    assert response.status_code == 200
    status = response.json()["status"]
    assert isinstance(status["updated_at_ns"], str)
    assert re.fullmatch(r"\d+", status["updated_at_ns"])
    assert status["updated_at_ns"] == str(updated_at_ns)
    expected = snapshot.to_json()
    for key in ("providers", "devices", "cameras", "network_check"):
        assert status[key] == expected[key]
    assert status["providers"][0]["id"] == "prov"
    assert status["cameras"][0]["name"] == "head"
    assert status["network_check"]["generation"] == "gen-1"
    assert [c["name"] for c in response.json()["cameras"]] == ["head"]
