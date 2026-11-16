# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Start-of-day route: reading the flow and the headset address, and the real application end to end."""

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
from egocap_web.station_uptime import READY, RECORDING, UptimeLog
from egocap_web.stations import StationStore
from tests.backend.test_preview_routes import _poll_until_sync, _ServerThread

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}
BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head"}}


@dataclass
class _Context:
    control_token: str
    saved_count: int = 0


class _World:
    """What a test can change between requests: whether the headset is connected."""

    headset_connected = True


async def _rotate() -> None:
    """Nothing to rotate in these tests."""


def _client(tmp_path, world: _World | None = None, ready: bool = True) -> TestClient:
    world = world or _World()
    small = FastAPI()
    small.include_router(setup_routes.router)
    small.include_router(station_router)
    if not ready:
        return TestClient(small)

    def facts() -> CaptureFacts:
        return CaptureFacts(
            running=True,
            mock=True,
            tracking_fresh=True,
            devices=(DeviceFact("Headset", world.headset_connected, "disconnected", headset=True),),
            cameras={"cam_head": CameraFact(True, True)},
        )

    uptime = UptimeLog(tmp_path / "station_status")
    stations = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head"}),
            change_blocked=lambda: None,
            show_label=lambda label: None,
            rotate_batch=_rotate,
            renamed=uptime.rename,
        ),
        "Deployment",
    )
    small.state.station_service = stations
    small.state.station_uptime = uptime
    small.state.setup_service = SetupService(
        facts=facts,
        stations=stations,
        calibration_dir=lambda: tmp_path / "calibration",
        output_root=tmp_path,
        free_bytes=lambda path: 50 * 1024**3,
        # Every request judges afresh, so a test can change the world between two of them.
        monotonic=iter(range(0, 10_000, 5)).__next__,
    )
    small.state.recording_context = _Context(control_token=TOKEN)
    small.state.settings = SimpleNamespace(capture=SimpleNamespace(capture_server_url="ws://127.0.0.1:8082"))
    return TestClient(small)


def _use_bench(client: TestClient, **fields) -> str:
    listing = client.post("/api/stations", json={**BENCH, **fields}, headers=AUTH).json()
    bench = listing["stations"][0]["id"]
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    return bench


def _steps(body: dict) -> dict[str, tuple[str, str | None]]:
    return {step["id"]: (step["status"], step["detail"]) for step in body["steps"]}


def _host(monkeypatch, *addresses: str) -> list[str]:
    """Make the capture host have these addresses; returns the URLs the route asked about."""
    asked: list[str] = []

    def candidates(url: str) -> list[HostAddress]:
        asked.append(url)
        return [HostAddress(address, "wlp2s0", "wireless") for address in addresses]

    monkeypatch.setattr(setup_routes, "headset_addresses", candidates)
    return asked


def test_the_route_answers_503_until_the_application_has_started(tmp_path):
    response = _client(tmp_path, ready=False).get("/api/setup")
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"


def test_the_flow_and_the_live_checklist_are_read_openly(tmp_path, monkeypatch):
    _host(monkeypatch, "10.0.0.42")
    world = _World()
    client = _client(tmp_path, world)
    _use_bench(client)
    body = client.get("/api/setup").json()
    assert body["enforced"] is True and body["blocked_reason"] is None
    assert _steps(body) == {
        "station": ("done", "Bench A"),
        "devices": ("done", "all checks pass"),
        "record": ("todo", None),
        "wrap_up": ("todo", None),
    }
    assert [(item["label"], item["status"]) for item in body["checks"]] == [
        ("Headset", "pass"),
        ("Ego (head): cam_head", "pass"),
        ("Left wrist", "skip"),
        ("Right wrist", "skip"),
        ("Camera calibration", "skip"),
        ("Free disk space", "pass"),
    ]
    # Nothing is run: the next read simply shows what changed.
    world.headset_connected = False
    held = client.get("/api/setup").json()
    assert held["blocked_reason"] == "Headset: disconnected"
    assert _steps(held)["devices"] == ("attention", "Headset: disconnected")
    # A batch's count alone is not a take today: it may be left from yesterday.
    client.app.state.recording_context.saved_count = 2
    assert _steps(client.get("/api/setup").json())["record"] == ("todo", None)
    uptime = client.app.state.station_uptime
    uptime.observe("Bench A", RECORDING, None)
    uptime.observe("Bench A", READY, None)
    uptime.save_report("Bench A", {})
    done = _steps(client.get("/api/setup").json())
    assert (done["record"], done["wrap_up"]) == (
        ("done", "recorded today · 2 saved in this batch"),
        ("done", "report saved"),
    )
    # A new batch restarts the count; the step stays done for the day.
    client.app.state.recording_context.saved_count = 0
    assert _steps(client.get("/api/setup").json())["record"] == ("done", "recorded today")
    assert not hasattr(client.app.state.setup_service, "run_connection_test")


def test_the_headset_address_is_returned_only_with_the_token(tmp_path, monkeypatch):
    asked = _host(monkeypatch, "10.0.0.42", "192.168.1.10")
    client = _client(tmp_path)
    _use_bench(client)
    assert client.get("/api/setup").json()["headset"] is None
    assert client.get("/api/setup", headers={"X-Control-Token": "wrong"}).json()["headset"] is None
    assert client.get("/api/setup", headers=AUTH).json()["headset"] == {
        "address": "10.0.0.42",
        "saved": None,
        "changed": False,
    }
    assert set(asked) == {"ws://127.0.0.1:8082"}


def test_a_remembered_address_the_host_no_longer_has_needs_attention_unless_the_headset_is_connected(
    tmp_path, monkeypatch
):
    _host(monkeypatch, "10.0.0.42")
    world = _World()
    client = _client(tmp_path, world)
    bench = _use_bench(client, headset_address="10.0.0.7")
    # A connected headset proves the address it was given works, whatever is remembered.
    connected = client.get("/api/setup", headers=AUTH).json()
    assert connected["headset"] == {"address": "10.0.0.42", "saved": "10.0.0.7", "changed": False}
    assert _steps(connected)["station"] == ("done", "Bench A")
    world.headset_connected = False
    stale = client.get("/api/setup", headers=AUTH).json()
    assert stale["headset"]["changed"] is True
    assert _steps(stale)["station"] == ("attention", "this host's address changed; update the headset")
    client.put(f"/api/stations/{bench}", json={"headset_address": "10.0.0.42"}, headers=AUTH)
    assert client.get("/api/setup", headers=AUTH).json()["headset"]["changed"] is False


def test_a_host_with_no_address_reports_none(tmp_path, monkeypatch):
    _host(monkeypatch)
    client = _client(tmp_path)
    assert client.get("/api/setup", headers=AUTH).json()["headset"] == {
        "address": None,
        "saved": None,
        "changed": False,
    }


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
            held = client.get("/api/setup").json()
            assert held["blocked_reason"] == missing and _steps(held)["station"] == ("attention", missing)
            ego = next(item for item in held["checks"] if item["id"] == "camera:ego")
            assert (ego["status"], ego["detail"]) == (
                "fail",
                "camera 'cart_top_camera' is not offered by the capture session",
            )

            # The station whose devices are all present may record, with nothing to run first.
            client.post("/api/stations/select", json={"id": bench}, headers=auth)
            _poll_until_sync(lambda: client.get("/api/setup").json()["blocked_reason"] is None)
            ready = client.get("/api/setup").json()
            assert _steps(ready)["devices"] == ("done", "all checks pass")
            assert {item["status"] for item in ready["checks"] if item["group"] != "system"} <= {"pass", "skip"}
            _poll_until_sync(lambda: context.snapshot()["ready_to_record"] is True)
    finally:
        server.stop()
