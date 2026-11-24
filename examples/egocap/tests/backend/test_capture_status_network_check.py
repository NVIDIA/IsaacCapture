# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The status JSON the browser reads carries CaptureServer's network check unchanged."""

from __future__ import annotations

import json

from capture_server import protocol as p
from egocap_web.capture_status import CaptureStatusMonitor
from egocap_web.config import CaptureStatusConfig
from egocap_web.preview_protocol import NativeFacts, build_snapshot
from tests.support.network_check import capture_status, check_status, status_snapshot


def _monitor(check: p.NetworkCheckStatus | None) -> CaptureStatusMonitor:
    monitor = CaptureStatusMonitor(
        CaptureStatusConfig(capture_server_url="ws://127.0.0.1:1"), backend_instance_id="b1"
    )
    snapshot = status_snapshot(check)
    monitor.native = NativeFacts(status=capture_status(snapshot), latest_status=snapshot)
    return monitor


def _preview(native: NativeFacts) -> dict:
    return build_snapshot(
        backend_instance_id="b1",
        revision=1,
        station_label="",
        profile_label="",
        connecting=False,
        native=native,
        error_message=None,
        binary_generation=None,
        tracking_slot=None,
        tracking_diagnostic=None,
        camera_slots_by_name={},
        camera_diagnostics={},
        now_monotonic=0.0,
    )


def test_the_http_status_carries_the_network_check_as_the_protocol_serializes_it():
    check = check_status(p.NetworkCheckState.UNSUITABLE, reason="cam0 is late")
    body = json.loads(json.dumps(_monitor(check).snapshot().to_json()))
    assert body["status"]["network_check"] == check.to_json()
    assert body["status"]["network_check"]["state"] == int(p.NetworkCheckState.UNSUITABLE)


def test_the_http_status_omits_the_network_check_when_none_is_reported():
    assert "network_check" not in _monitor(None).snapshot().to_json()["status"]


def test_the_preview_snapshot_carries_the_network_check():
    check = check_status(p.NetworkCheckState.CHECKING)
    monitor = _monitor(check)
    body = json.loads(json.dumps(_preview(monitor.native)))
    assert body["capture"]["status"]["status"]["network_check"] == check.to_json()


def test_the_preview_snapshot_omits_the_network_check_when_none_is_reported():
    assert "network_check" not in _preview(_monitor(None).native)["capture"]["status"]["status"]


async def test_run_network_check_relays_to_the_capture_client(monkeypatch):
    monitor = _monitor(None)
    calls: list[str] = []

    async def fake() -> None:
        calls.append("run")

    monkeypatch.setattr(monitor._client, "run_network_check", fake)
    await monitor.run_network_check()
    assert calls == ["run"]


async def test_run_network_check_propagates_the_clients_refusal(monkeypatch):
    monitor = _monitor(None)

    async def refuse() -> None:
        raise RuntimeError("run_network_check needs an accepted capture session")

    monkeypatch.setattr(monitor._client, "run_network_check", refuse)
    try:
        await monitor.run_network_check()
    except RuntimeError as exc:
        assert "accepted capture session" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
