# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Start-of-day routes: reading the flow, running the connection test, and the real application end to end."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web import setup_routes
from egocap_web.app import app as real_app
from egocap_web.host_addresses import HostAddress
from egocap_web.setup_checks import CameraFact, CaptureFacts, DeviceFact
from egocap_web.setup_service import SetupService
from egocap_web.station_routes import router as station_router
from egocap_web.station_service import StationService, session_adapter
from egocap_web.stations import StationStore
from tests.backend.test_preview_routes import _poll_until_sync, _ServerThread

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}


@dataclass
class _Context:
    control_token: str
    saved_count: int = 0


async def _rotate() -> None:
    """Nothing to rotate in these tests."""


def _client(tmp_path, ready: bool = True) -> TestClient:
    small = FastAPI()
    small.include_router(setup_routes.router)
    small.include_router(station_router)
    if not ready:
        return TestClient(small)
    facts = CaptureFacts(
        running=True,
        mock=True,
        session_id="s1",
        requested=frozenset({"headset"}),
        tracking_fresh=True,
        devices=(DeviceFact("headset", "Headset", True),),
        cameras={"cam_head": CameraFact(True, True)},
    )
    stations = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head"}),
            change_blocked=lambda: None,
            show_label=lambda label: None,
            rotate_batch=_rotate,
            leave_station=lambda label: None,
        ),
        "Deployment",
    )
    small.state.station_service = stations
    small.state.station_activity = object()
    small.state.setup_service = SetupService(
        facts=lambda: facts,
        session_id=lambda: "s1",
        stations=stations,
        calibration_dir=lambda: tmp_path / "calibration",
        output_root=tmp_path,
        free_bytes=lambda path: 50 * 1024**3,
    )
    small.state.recording_context = _Context(control_token=TOKEN)
    small.state.settings = SimpleNamespace(capture=SimpleNamespace(capture_server_url="ws://127.0.0.1:8082"))
    return TestClient(small)


def _steps(body: dict) -> dict[str, tuple[str, str | None]]:
    return {step["id"]: (step["status"], step["detail"]) for step in body["steps"]}


def test_the_routes_answer_503_until_the_application_has_started(tmp_path):
    client = _client(tmp_path, ready=False)
    for response in (client.get("/api/setup"), client.post("/api/setup/connection-test")):
        assert response.status_code == 503 and response.headers["cache-control"] == "no-store"


def test_the_flow_is_read_openly_and_the_test_is_run_with_the_token(tmp_path):
    client = _client(tmp_path)
    bench = client.post(
        "/api/stations", json={"name": "Bench A", "camera_roles": {"ego": "cam_head"}}, headers=AUTH
    )
    client.post("/api/stations/select", json={"id": bench.json()["stations"][0]["id"]}, headers=AUTH)
    before = client.get("/api/setup").json()
    assert before["enforced"] is True and before["connection_test"] is None
    assert before["blocked_reason"] == "run the device connection test before recording"
    assert _steps(before)["devices"] == ("todo", "run the connection test")
    assert client.post("/api/setup/connection-test").status_code == 403
    assert client.post("/api/setup/connection-test", headers={"X-Control-Token": "wrong"}).status_code == 403
    assert client.get("/api/setup").json()["connection_test"] is None
    after = client.post("/api/setup/connection-test", headers=AUTH).json()
    assert after["blocked_reason"] is None and after["connection_test"]["passed"] is True
    assert [(item["label"], item["status"]) for item in after["connection_test"]["items"]] == [
        ("Headset", "pass"),
        ("Ego (head): cam_head", "pass"),
        ("Left wrist", "skip"),
        ("Right wrist", "skip"),
    ]
    assert _steps(after) == {
        "station": ("done", "Bench A"),
        "devices": ("done", "connection test passed"),
        "calibration": ("skipped", "mock capture has nothing to calibrate"),
        "record": ("todo", None),
        "wrap_up": ("todo", None),
    }
    client.app.state.recording_context.saved_count = 2
    assert _steps(client.get("/api/setup").json())["record"] == ("done", "2 saved")


def test_a_changed_host_address_shows_on_the_station_step(tmp_path, monkeypatch):
    client = _client(tmp_path)
    calls: list[str] = []

    def candidates(url: str) -> list[HostAddress]:
        calls.append(url)
        return [HostAddress("10.0.0.42", "wlp2s0", "wireless")]

    monkeypatch.setattr(setup_routes, "headset_addresses", candidates)
    fields = {"name": "Bench A", "camera_roles": {"ego": "cam_head"}}
    bench = client.post("/api/stations", json=fields, headers=AUTH).json()["stations"][0]["id"]
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    assert _steps(client.get("/api/setup").json())["station"] == ("done", "Bench A")
    assert calls == []  # nothing saved, so the interfaces are not even listed
    client.put(f"/api/stations/{bench}", json={"headset_address": "10.0.0.7"}, headers=AUTH)
    assert _steps(client.get("/api/setup").json())["station"] == (
        "attention",
        "this host's address changed; update the headset",
    )
    client.put(f"/api/stations/{bench}", json={"headset_address": "10.0.0.42"}, headers=AUTH)
    assert _steps(client.get("/api/setup").json())["station"] == ("done", "Bench A")
    assert calls == ["ws://127.0.0.1:8082"] * 2


def test_a_station_with_a_missing_device_cannot_record_and_says_why(monkeypatch, unused_tcp_port, tmp_path):
    """The real application against the real CaptureServer in its mock mode."""
    from egocap_web import app as app_module

    station_file = tmp_path / "station.yaml"
    station_file.write_text("video_sources:\n- type: oak_camera\n  name: head_human_camera\n", encoding="utf-8")
    monkeypatch.setenv("EGOCAP_STATION_CONFIG", str(station_file))
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.delenv("EGOCAP_OPERATOR_SETTINGS_FILE", raising=False)
    monkeypatch.setattr(app_module, "_recording_output_root", lambda data_dir: tmp_path)
    server = _ServerThread(unused_tcp_port)
    try:
        with TestClient(real_app) as client:
            context = client.app.state.recording_context
            auth = {"X-Control-Token": context.control_token}
            monitor = client.app.state.monitor
            _poll_until_sync(
                lambda: monitor.snapshot().running and monitor.preview.camera("head_human_camera") is not None
            )
            context.coordinator.set_task("Pour water.")

            def add(name: str, ego: str) -> str:
                fields = {"name": name, "camera_roles": {"ego": ego}}
                listing = client.post("/api/stations", json=fields, headers=auth).json()
                return listing["stations"][-1]["id"]

            bench, cart = add("Bench A", "head_human_camera"), add("Cart B", "cart_top_camera")

            # A station whose camera the session does not offer never reaches the recording step.
            client.post("/api/stations/select", json={"id": cart}, headers=auth)
            missing = (
                "station 'Cart B': camera 'cart_top_camera' (ego (head)) is not offered by the capture session"
            )
            assert context.snapshot()["not_ready_reason"] == missing
            failed = client.post("/api/setup/connection-test", headers=auth).json()
            assert failed["connection_test"]["passed"] is False and failed["blocked_reason"] == missing
            ego = next(item for item in failed["connection_test"]["items"] if item["id"] == "camera:ego")
            assert (ego["status"], ego["detail"]) == (
                "fail",
                "camera 'cart_top_camera' is not offered by the capture session",
            )
            assert _steps(failed)["station"] == ("attention", missing)

            # The station whose devices are all present passes the test and may record.
            client.post("/api/stations/select", json={"id": bench}, headers=auth)
            assert context.snapshot()["not_ready_reason"] == "run the device connection test before recording"
            passed = client.post("/api/setup/connection-test", headers=auth).json()
            assert passed["connection_test"]["passed"] is True, passed["connection_test"]["items"]
            assert passed["blocked_reason"] is None
            assert _steps(passed)["devices"] == ("done", "connection test passed")
            assert _steps(passed)["calibration"][0] == "skipped"  # mock capture has nothing to calibrate
            _poll_until_sync(lambda: context.snapshot()["ready_to_record"] is True)
    finally:
        server.stop()
