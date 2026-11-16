# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Station routes: the token-gated registry, the activity toggle, the headset address, and the real application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web import station_routes
from egocap_web.app import app as real_app
from egocap_web.host_addresses import HostAddress
from egocap_web.operator_settings import OperatorSettingsStore
from egocap_web.operator_settings import router as operator_settings_router
from egocap_web.station_activity import StationActivity
from egocap_web.station_service import TAKE_IN_PROGRESS, StationService, session_adapter
from egocap_web.stations import StationStore

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}
BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}


@dataclass
class _Context:
    control_token: str


class _Live:
    """The live-session facts a test can change between requests."""

    blocked: str | None = None


def _client(tmp_path: Path, live: _Live | None = None, ready: bool = True) -> TestClient:
    live = live or _Live()
    small = FastAPI()
    small.include_router(station_routes.router)
    small.include_router(operator_settings_router)
    if not ready:
        return TestClient(small)
    activity = StationActivity(tmp_path / "station_status")

    async def rotate() -> None:
        """Nothing to rotate in these tests."""

    small.state.station_service = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head", "cam_left"}),
            change_blocked=lambda: live.blocked,
            show_label=lambda label: None,
            rotate_batch=rotate,
            leave_station=activity.leave,
        ),
        "Deployment",
    )
    small.state.station_activity = activity
    small.state.recording_context = _Context(control_token=TOKEN)
    small.state.operator_settings_store = OperatorSettingsStore(tmp_path / "operator_settings.json")
    small.state.settings = SimpleNamespace(capture=SimpleNamespace(capture_server_url="ws://127.0.0.1:8082"))
    return TestClient(small)


def _add(client: TestClient, fields: dict = BENCH) -> str:
    response = client.post("/api/stations", json=fields, headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["stations"][-1]["id"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/stations"),
        ("post", "/api/stations"),
        ("post", "/api/stations/select"),
        ("put", "/api/stations/st_1"),
        ("delete", "/api/stations/st_1"),
        ("get", "/api/headset-address"),
        ("get", "/api/station-activity"),
        ("post", "/api/station-activity"),
    ],
)
def test_every_route_answers_503_until_the_application_has_started(tmp_path, method, path):
    response = getattr(_client(tmp_path, ready=False), method)(path)
    assert response.status_code == 503 and response.headers["cache-control"] == "no-store"


def test_stations_are_added_edited_selected_and_deleted(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/stations").json()["stations"] == []
    bench = _add(client)
    cart = _add(client, {"name": "Cart B", "camera_roles": {"ego": "cam_top"}})
    selected = client.post("/api/stations/select", json={"id": bench}, headers=AUTH).json()
    assert (selected["selected"], selected["label"]) == (bench, "Bench A")
    edited = client.put(f"/api/stations/{cart}", json={"headset_address": "192.168.4.20"}, headers=AUTH).json()
    assert edited["stations"][1]["headset_address"] == "192.168.4.20"
    assert edited["stations"][1]["problems"] == [
        "camera 'cam_top' (ego (head)) is not offered by the capture session"
    ]
    # A second client sees the same registry, read back from the file.
    assert StationStore(tmp_path / "stations.json").load().selected.name == "Bench A"
    remaining = client.delete(f"/api/stations/{bench}", headers=AUTH).json()
    assert [station["name"] for station in remaining["stations"]] == ["Cart B"]
    assert (remaining["selected"], remaining["label"]) == (None, "Deployment")
    assert client.post("/api/stations/select", json={"id": None}, headers=AUTH).status_code == 200


def test_the_headset_address_is_shown_only_to_a_caller_with_the_token(tmp_path):
    client = _client(tmp_path)
    _add(client, {**BENCH, "headset_address": "192.168.4.20"})
    assert "headset_address" not in client.get("/api/stations").json()["stations"][0]
    assert (
        "headset_address"
        not in client.get("/api/stations", headers={"X-Control-Token": "wrong"}).json()["stations"][0]
    )
    assert client.get("/api/stations", headers=AUTH).json()["stations"][0]["headset_address"] == "192.168.4.20"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/stations", BENCH),
        ("post", "/api/stations/select", {"id": None}),
        ("put", "/api/stations/st_1", {"name": "X"}),
        ("post", "/api/station-activity", {"state": "active"}),
    ],
)
def test_a_change_without_the_token_is_refused_and_changes_nothing(tmp_path, method, path, body):
    client = _client(tmp_path)
    assert getattr(client, method)(path, json=body).status_code == 403
    assert getattr(client, method)(path, json=body, headers={"X-Control-Token": "wrong"}).status_code == 403
    assert not (tmp_path / "stations.json").exists() and not (tmp_path / "station_status").exists()


def test_a_delete_needs_the_token_in_its_header(tmp_path):
    client = _client(tmp_path)
    bench = _add(client)
    assert client.delete(f"/api/stations/{bench}").status_code == 403
    assert client.delete(f"/api/stations/{bench}", headers={"X-Control-Token": "wrong"}).status_code == 403
    assert len(client.get("/api/stations").json()["stations"]) == 1


def test_a_refused_change_says_why_and_an_unknown_station_is_not_found(tmp_path):
    client = _client(tmp_path)
    bench = _add(client)
    duplicate = client.post("/api/stations", json={"name": "bench a"}, headers=AUTH)
    assert (duplicate.status_code, duplicate.json()["detail"]) == (400, "a station named 'Bench A' already exists")
    clash = client.put(f"/api/stations/{bench}", json={"camera_roles": {"wrist_right": "cam_head"}}, headers=AUTH)
    assert (clash.status_code, clash.json()["detail"]) == (
        400,
        "camera 'cam_head' cannot fill both ego and wrist_right",
    )
    assert client.post("/api/stations/select", json={"id": 7}, headers=AUTH).json()["detail"] == (
        "id must be a station id or null"
    )
    for response in (
        client.put("/api/stations/st_missing", json={"name": "X"}, headers=AUTH),
        client.delete("/api/stations/st_missing", headers=AUTH),
        client.post("/api/stations/select", json={"id": "st_missing"}, headers=AUTH),
    ):
        assert (response.status_code, response.json()["detail"]) == (404, "no such station")


def test_the_station_in_use_cannot_change_during_a_take(tmp_path):
    live = _Live()
    client = _client(tmp_path, live)
    bench, cart = _add(client), _add(client, {"name": "Cart B"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    live.blocked = TAKE_IN_PROGRESS
    assert client.get("/api/stations").json()["locked_reason"] == TAKE_IN_PROGRESS
    for response in (
        client.post("/api/stations/select", json={"id": cart}, headers=AUTH),
        client.put(f"/api/stations/{bench}", json={"name": "Bench Z"}, headers=AUTH),
        client.delete(f"/api/stations/{bench}", headers=AUTH),
    ):
        assert (response.status_code, response.json()["detail"]) == (400, TAKE_IN_PROGRESS)
    live.blocked = None
    assert client.post("/api/stations/select", json={"id": cart}, headers=AUTH).json()["selected"] == cart


def test_a_registry_that_cannot_be_written_is_a_503(tmp_path, monkeypatch):
    client = _client(tmp_path)
    bench = _add(client)

    def refuse(self, registry):
        raise OSError("disk full")

    monkeypatch.setattr(StationStore, "save", refuse)
    assert client.post("/api/stations", json={"name": "Cart B"}, headers=AUTH).status_code == 503
    assert client.delete(f"/api/stations/{bench}", headers=AUTH).status_code == 503


def test_the_camera_setup_the_page_reads_is_the_station_in_use(tmp_path):
    client = _client(tmp_path)
    saved = client.put("/api/operator-settings", json={"camera_type": "sensing_gmsl"}, headers=AUTH)
    assert saved.json()["camera_type"] == "sensing_gmsl"  # no station in use: the stored value applies
    bench = _add(client, {**BENCH, "camera_type": "oak_d"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    read = client.get("/api/operator-settings").json()
    assert (read["camera_type"], read["camera_roles"]["wrist_left"]) == ("oak_d", "cam_left")
    # An edit of "this station" from the existing settings controls is saved to the station in use.
    changed = client.put("/api/operator-settings", json={"camera_roles": {"wrist_left": None}}, headers=AUTH)
    assert changed.json()["camera_roles"] == {"ego": "cam_head", "wrist_left": None, "wrist_right": None}
    assert client.get("/api/stations").json()["stations"][0]["camera_roles"]["wrist_left"] is None
    refused = client.put("/api/operator-settings", json={"camera_roles": {"wrist_left": "cam_head"}}, headers=AUTH)
    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        "camera 'cam_head' cannot fill both ego and wrist_left",
    )
    # With nothing in use again, the station-independent value is back, untouched by the station's edits.
    client.post("/api/stations/select", json={"id": None}, headers=AUTH)
    assert client.get("/api/operator-settings").json()["camera_type"] == "sensing_gmsl"


def test_invalid_saved_settings_still_show_the_station_in_use(tmp_path):
    client = _client(tmp_path)
    bench = _add(client, {**BENCH, "camera_type": "oak_d"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    (tmp_path / "operator_settings.json").write_text("not json")
    read = client.get("/api/operator-settings").json()
    assert read["camera_type"] == "oak_d" and "invalid" in read["warning"]


def test_the_headset_address_reports_candidates_and_a_stale_saved_address(tmp_path, monkeypatch):
    client = _client(tmp_path)
    seen: list[str] = []

    def candidates(url: str) -> list[HostAddress]:
        seen.append(url)
        return [HostAddress("10.0.0.42", "wlp2s0", "wireless")]

    monkeypatch.setattr(station_routes, "headset_addresses", candidates)
    assert client.get("/api/headset-address").status_code == 403
    nothing_in_use = client.get("/api/headset-address", headers=AUTH).json()
    assert nothing_in_use == {
        "candidates": [{"address": "10.0.0.42", "interface": "wlp2s0", "kind": "wireless"}],
        "saved": None,
        "changed": False,
    }
    bench = _add(client, {**BENCH, "headset_address": "10.0.0.7"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    stale = client.get("/api/headset-address", headers=AUTH).json()
    assert (stale["saved"], stale["changed"]) == ("10.0.0.7", True)
    client.put(f"/api/stations/{bench}", json={"headset_address": "10.0.0.42"}, headers=AUTH)
    assert client.get("/api/headset-address", headers=AUTH).json()["changed"] is False
    assert seen[0] == "ws://127.0.0.1:8082"


def test_activity_is_toggled_for_the_station_in_use(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/station-activity").json()["station"] == "Deployment"
    bench = _add(client)
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    started = client.post("/api/station-activity", json={"state": "active"}, headers=AUTH).json()
    assert (started["station"], started["state"]) == ("Bench A", "active")
    noted = client.post("/api/station-activity", json={"notes": "left glove battery"}, headers=AUTH).json()
    assert (noted["state"], noted["notes"]) == ("active", "left glove battery")
    assert client.get("/api/station-activity").json()["notes"] == "left glove battery"
    # Leaving the station stops its active time, so it does not keep counting for a station nobody is at.
    client.post("/api/stations/select", json={"id": None}, headers=AUTH)
    assert client.get("/api/station-activity").json()["station"] == "Deployment"
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    assert client.get("/api/station-activity").json()["state"] == "inactive"


def test_an_activity_change_must_name_something_valid(tmp_path, monkeypatch):
    client = _client(tmp_path)
    empty = client.post("/api/station-activity", json={}, headers=AUTH)
    assert (empty.status_code, empty.json()["detail"]) == (400, "name a state, notes, or both")
    invalid = client.post("/api/station-activity", json={"state": "paused"}, headers=AUTH)
    assert (invalid.status_code, invalid.json()["detail"]) == (400, "state must be one of active, inactive")

    def refuse(self, station, **_changes):
        raise OSError("disk full")

    monkeypatch.setattr(StationActivity, "change", refuse)
    assert client.post("/api/station-activity", json={"state": "active"}, headers=AUTH).status_code == 503


def test_the_real_application_gates_recording_on_the_station_in_use(monkeypatch, unused_tcp_port, tmp_path):
    from egocap_web import app as app_module

    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.delenv("EGOCAP_OPERATOR_SETTINGS_FILE", raising=False)
    monkeypatch.delenv("EGOCAP_STATION_CONFIG", raising=False)
    monkeypatch.setattr(app_module, "_recording_output_root", lambda data_dir: tmp_path)
    with TestClient(real_app) as client:
        context = client.app.state.recording_context
        auth = {"X-Control-Token": context.control_token}
        # No stations: the configuration-free default, where only capture itself decides readiness.
        assert client.get("/api/stations").json()["label"] == "Station 01"
        assert context.setup_block() is None
        bench = client.post("/api/stations", json=BENCH, headers=auth).json()["stations"][0]["id"]
        assert (tmp_path / ".egocap" / "stations.json").is_file()
        assert context.snapshot()["not_ready_reason"] == "select a station before recording"
        selected = client.post("/api/stations/select", json={"id": bench}, headers=auth).json()
        assert (selected["label"], selected["locked_reason"]) == ("Bench A", None)
        # The label every viewer sees, and the one the next recording session will carry, changed in process.
        assert client.get("/api/capture-status").json()["station_label"] == "Bench A"
        assert client.app.state.station_service.label() == "Bench A"
        # Capture never connected here, so what the session offers is unknown and the station itself is fine.
        assert context.setup_block() is None
        client.put(f"/api/stations/{bench}", json={"camera_roles": {"ego": None}}, headers=auth)
        assert context.snapshot()["not_ready_reason"] == (
            "station 'Bench A': no camera is assigned to the ego (head) role"
        )
        active = client.post("/api/station-activity", json={"state": "active"}, headers=auth).json()
        assert (active["station"], active["state"]) == ("Bench A", "active")
        assert sorted(path.name for path in (tmp_path / "station_status").iterdir())[1].startswith("bench_a_")
        assert client.get("/api/headset-address").status_code == 403
        assert client.get("/api/headset-address", headers=auth).status_code == 200
