# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GET /api/settings: the wire form of a hand-built snapshot, and the real application's own."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from capture_server import protocol as p
from egocap_web.app import app as real_app
from egocap_web.config import CaptureStatusConfig, VisualizationConfig
from egocap_web.settings_routes import EffectiveSettings, router
from egocap_web.station_config import ResolvedStation


def _snapshot(**overrides) -> EffectiveSettings:
    values = dict(
        capture=CaptureStatusConfig(),
        station_path=None,
        station=None,
        station_error=None,
        recording_output_root=Path("/var/egocap/recordings"),
        visualization=VisualizationConfig(),
        visualization_error=None,
        task_file_root=Path("/nonexistent/taskfiles"),
    )
    values.update(overrides)
    return EffectiveSettings(**values)


def _client(settings: EffectiveSettings | None) -> TestClient:
    small = FastAPI()
    small.include_router(router)
    small.state.recording_context = SimpleNamespace(control_token="t" * 43)
    if settings is not None:
        small.state.settings = settings
    return TestClient(small, headers={"X-Control-Token": "t" * 43})


def _clear_env(monkeypatch) -> None:
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "EGOCAP_RECORDINGS_ENABLED",
        "EGOCAP_ALLOWED_ORIGINS",
        "EGOCAP_TRUSTED_HOSTNAMES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_unresolved_settings_answer_503_and_are_never_cached():
    response = _client(None).get("/api/settings")
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_defaults_serialise_as_names_and_booleans(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("EGOCAP_RECORDINGS_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_ALLOWED_ORIGINS", "https://b.example, https://a.example")
    response = _client(_snapshot()).get("/api/settings")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["capture"]["server_url"] == "ws://127.0.0.1:8082"
    assert body["capture"]["sources"] == ["head", "controllers", "hands"]
    assert body["capture"]["plugins"] == [] and body["capture"]["cameras"] == []
    assert body["preview"] == {"camera_sample_hz": 10.0, "tracking_sample_hz": 10.0}
    assert body["station"] == {
        "path": None,
        "error": None,
        "id": None,
        "operator_name": None,
        "auto_fill_operator_name": False,
        "task_description": None,
        "data_dir": None,
    }
    assert body["recording"] == {"output_root": "/var/egocap/recordings"}
    assert body["visualization"]["enabled"] is True
    assert body["visualization"]["host"] == "127.0.0.1" and body["visualization"]["port"] == 8085
    assert body["visualization"]["device"] == "pico-4-ultra"
    # Unused environment secrets must not leak into the deployment snapshot.
    assert "livekit" not in body
    assert "secret" not in response.text and "devkey" not in response.text
    assert body["recordings_browser"] == {"enabled": True}
    assert body["task_files"] == {"dir": "/nonexistent/taskfiles", "available": False}
    assert body["origins"] == {"allowed": ["https://a.example", "https://b.example"], "trusted_hostnames": None}


def test_station_and_view_failures_are_reported_not_hidden(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("EGOCAP_TRUSTED_HOSTNAMES", "Station-PC, 10.0.0.5")
    snapshot = _snapshot(
        station_path=tmp_path / "stations.yaml",
        station_error="station configuration is invalid; see backend logs",
        visualization=None,
        visualization_error="EGOCAP_VISER_DEVICE 'x' is not a known device profile",
        task_file_root=tmp_path,
    )
    body = _client(snapshot).get("/api/settings").json()
    assert body["station"]["path"] == str(tmp_path / "stations.yaml")
    assert body["station"]["error"].startswith("station configuration is invalid")
    assert body["visualization"]["enabled"] is False
    assert body["visualization"]["error"].startswith("EGOCAP_VISER_DEVICE")
    assert body["visualization"]["host"] is None
    assert body["recordings_browser"] == {"enabled": False}
    assert body["task_files"]["available"] is True
    assert body["origins"]["trusted_hostnames"] == ["10.0.0.5", "station-pc"]


def test_a_resolved_station_is_reported_with_its_cameras(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    station = ResolvedStation(
        station_id="station_01",
        station_label="Station 01",
        operator_name="",
        auto_fill_operator_name=True,
        task_description="Fold the towel",
        data_dir=tmp_path / "data",
        cameras=(),
    )
    body = _client(_snapshot(station=station, station_path=Path("s.yaml"))).get("/api/settings").json()
    assert body["station"]["id"] == "station_01"
    assert body["station"]["auto_fill_operator_name"] is True
    assert body["station"]["task_description"] == "Fold the towel"
    assert body["station"]["data_dir"] == str(tmp_path / "data")


def test_configured_cameras_are_reported_by_name_and_type(monkeypatch):
    _clear_env(monkeypatch)
    cameras = (p.CameraSelection(name="head", type=p.CameraType.OAK_D_COLOR),)
    body = _client(_snapshot(capture=CaptureStatusConfig(cameras=cameras))).get("/api/settings").json()
    assert body["capture"]["cameras"] == [{"name": "head", "type": "oak_d_color"}]


def test_the_real_application_reports_what_it_started_with(monkeypatch, unused_tcp_port):
    url = f"ws://127.0.0.1:{unused_tcp_port}"  # nothing listens there; the preview still starts
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", url)
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "0")
    monkeypatch.delenv("EGOCAP_STATION_CONFIG", raising=False)
    with TestClient(real_app) as client:
        response = client.get(
            "/api/settings", headers={"X-Control-Token": real_app.state.recording_context.control_token}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["capture"]["server_url"] == url
        assert body["visualization"]["enabled"] is False and body["visualization"]["error"] is None
        assert body["station"]["path"] is None
        assert body["task_files"]["available"] is True


def test_the_real_application_reports_an_unreadable_view_setting(monkeypatch, unused_tcp_port):
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    # The view must be on for its other settings to be read at all; the bad device then keeps it from starting.
    monkeypatch.setenv("EGOCAP_VISER_ENABLED", "1")
    monkeypatch.setenv("EGOCAP_VISER_DEVICE", "no-such-device")
    with TestClient(real_app) as client:
        body = client.get(
            "/api/settings", headers={"X-Control-Token": real_app.state.recording_context.control_token}
        ).json()
        assert body["visualization"]["enabled"] is False
        assert "no-such-device" in body["visualization"]["error"]


def test_deployment_snapshot_requires_a_valid_token_before_disclosing_any_values():
    client = _client(_snapshot())
    client.headers.clear()
    for headers in ({}, {"X-Control-Token": "x" * 43}, {"X-Control-Token": "null"}):
        response = client.get("/api/settings", headers=headers)
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"detail": "control token rejected"}
