# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Station routes: the token-gated registry, the station's observed time and notes, and the real application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap_web import app as app_module
from egocap_web import station_routes
from egocap_web.app import app as real_app
from egocap_web.operator_settings import OperatorSettingsStore
from egocap_web.operator_settings import router as operator_settings_router
from egocap_web.station_service import TAKE_IN_PROGRESS, StationService, session_adapter
from egocap_web.station_uptime import READY, RECORDING, UptimeLog
from egocap_web.stations import StationStore
from recording_service import RecordingPhase
from tests.backend.test_preview_routes import _poll_until_sync

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}
BENCH = {"name": "Bench A", "camera_roles": {"ego": "cam_head", "wrist_left": "cam_left"}}


@dataclass
class _Context:
    control_token: str


class _Live:
    """The live-session facts a test can change between requests."""

    blocked: str | None = None
    #: A stand-in recording coordinator the application's own batch rotation is run against, when set.
    coordinator = None


def _client(tmp_path: Path, live: _Live | None = None, ready: bool = True) -> TestClient:
    live = live or _Live()
    small = FastAPI()
    small.include_router(station_routes.router)
    small.include_router(operator_settings_router)
    if not ready:
        return TestClient(small)
    uptime = UptimeLog(tmp_path / "station_status")

    async def rotate() -> None:
        """Nothing to rotate, unless the test supplies a coordinator for the application's rotation."""
        if live.coordinator is not None:
            await app_module._rotate_batch_for_station(live.coordinator)

    small.state.station_service = StationService(
        StationStore(tmp_path / "stations.json"),
        session_adapter(
            offered_cameras=lambda: frozenset({"cam_head", "cam_left"}),
            change_blocked=lambda: live.blocked,
            show_label=lambda label: None,
            rotate_batch=rotate,
            renamed=uptime.rename,
        ),
        "Deployment",
    )
    small.state.station_uptime = uptime
    small.state.recording_context = _Context(control_token=TOKEN)
    small.state.operator_settings_store = OperatorSettingsStore(tmp_path / "operator_settings.json")
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
        ("get", "/api/station-uptime"),
        ("post", "/api/station-uptime"),
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
    wrong = client.get("/api/stations", headers={"X-Control-Token": "wrong"}).json()
    assert "headset_address" not in wrong["stations"][0]
    assert client.get("/api/stations", headers=AUTH).json()["stations"][0]["headset_address"] == "192.168.4.20"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/stations", BENCH),
        ("post", "/api/stations/select", {"id": None}),
        ("put", "/api/stations/st_1", {"name": "X"}),
        ("post", "/api/station-uptime", {"notes": "x"}),
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
    without_token = client.delete(f"/api/stations/{bench}")
    assert without_token.status_code == 403
    wrong_token = client.delete(f"/api/stations/{bench}", headers={"X-Control-Token": "wrong"})
    assert wrong_token.status_code == 403
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


def test_a_change_that_would_start_a_new_batch_is_refused_while_an_interrupted_take_waits(tmp_path):
    live = _Live()
    client = _client(tmp_path, live)
    bench, cart = _add(client), _add(client, {"name": "Cart B"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    rotations: list[str] = []

    async def new_batch() -> None:
        rotations.append("rotated")

    service = SimpleNamespace(phase=RecordingPhase.RECOVERY_REQUIRED, episode=object())
    live.coordinator = SimpleNamespace(snapshot=lambda: SimpleNamespace(service=service), new_batch=new_batch)
    for response in (
        client.post("/api/stations/select", json={"id": cart}, headers=AUTH),
        client.post("/api/stations/select", json={"id": None}, headers=AUTH),
        client.put(f"/api/stations/{bench}", json={"name": "Bench Z"}, headers=AUTH),
        client.delete(f"/api/stations/{bench}", headers=AUTH),
    ):
        assert (response.status_code, response.json()) == (
            400,
            {"detail": "recover or discard the interrupted take first"},
        )
    listing = client.get("/api/stations").json()
    assert (listing["selected"], listing["label"], rotations) == (bench, "Bench A", [])
    assert StationStore(tmp_path / "stations.json").load().selected.name == "Bench A"
    # What leaves the batch alone is still allowed: the station's cameras, and another station's record.
    kept = client.put(f"/api/stations/{bench}", json={"camera_roles": {"wrist_left": None}}, headers=AUTH)
    removed = client.delete(f"/api/stations/{cart}", headers=AUTH)
    assert (kept.status_code, removed.status_code) == (200, 200)
    # Once the take is recovered or discarded the change goes through.
    service.episode = None
    renamed = client.put(f"/api/stations/{bench}", json={"name": "Bench Z"}, headers=AUTH)
    assert renamed.json()["label"] == "Bench Z" and rotations == ["rotated"]


def test_the_first_station_cannot_be_put_in_use_while_an_interrupted_take_waits(tmp_path):
    live = _Live()
    client = _client(tmp_path, live)
    service = SimpleNamespace(phase=RecordingPhase.RECOVERY_REQUIRED, episode=object())
    live.coordinator = SimpleNamespace(snapshot=lambda: SimpleNamespace(service=service))
    bench = _add(client)  # adding it rotates nothing
    refused = client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        "recover or discard the interrupted take first",
    )
    assert client.get("/api/stations").json()["selected"] is None


def test_a_registry_that_cannot_be_written_is_a_503(tmp_path, monkeypatch):
    client = _client(tmp_path)
    bench = _add(client)

    def refuse(self, registry):
        raise OSError("disk full")

    monkeypatch.setattr(StationStore, "save", refuse)
    assert client.post("/api/stations", json={"name": "Cart B"}, headers=AUTH).status_code == 503
    deleted = client.delete(f"/api/stations/{bench}", headers=AUTH)
    assert deleted.status_code == 503


def test_the_camera_setup_the_page_reads_is_the_station_in_use_and_is_changed_on_the_station(tmp_path):
    client = _client(tmp_path)
    saved = client.put("/api/operator-settings", json={"camera_type": "sensing_gmsl"}, headers=AUTH)
    assert saved.json()["camera_type"] == "sensing_gmsl"  # no station in use: the stored value applies
    bench = _add(client, {**BENCH, "camera_type": "oak_d"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    read = client.get("/api/operator-settings").json()
    assert (read["camera_type"], read["camera_roles"]["wrist_left"]) == ("oak_d", "cam_left")
    # With a station in use its own record holds the camera setup; saving it here would store values nothing reads.
    for body in ({"camera_roles": {"wrist_left": None}}, {"camera_type": None}):
        refused = client.put("/api/operator-settings", json=body, headers=AUTH)
        assert (refused.status_code, refused.json()["detail"]) == (
            400,
            "the station in use holds the camera setup; change it on the station",
        )
    assert client.get("/api/stations").json()["stations"][0]["camera_roles"]["wrist_left"] == "cam_left"
    # Everything else is station-independent and still saved here.
    assert client.put("/api/operator-settings", json={"task_file": None}, headers=AUTH).status_code == 200
    client.post("/api/stations/select", json={"id": None}, headers=AUTH)
    assert client.get("/api/operator-settings").json()["camera_type"] == "sensing_gmsl"


def test_invalid_saved_settings_still_show_the_station_in_use(tmp_path):
    client = _client(tmp_path)
    bench = _add(client, {**BENCH, "camera_type": "oak_d"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    (tmp_path / "operator_settings.json").write_text("not json")
    read = client.get("/api/operator-settings").json()
    assert read["camera_type"] == "oak_d" and "invalid" in read["warning"]


def test_the_stations_time_is_read_openly_and_only_its_notes_can_be_set(tmp_path):
    client = _client(tmp_path)
    uptime = client.app.state.station_uptime
    bench = _add(client)
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    uptime.observe("Bench A", RECORDING, None)
    uptime.observe("Bench A", READY, None)
    report = client.get("/api/station-uptime").json()
    assert (report["station"], report["state"], report["reason"], report["blockers"]) == (
        "Bench A",
        READY,
        None,
        [],
    )
    note = {"station": "Bench A", "notes": "left glove battery"}
    noted = client.post("/api/station-uptime", json=note, headers=AUTH).json()
    assert (noted["state"], noted["notes"]) == (READY, "left glove battery")
    # The state is observed, never set: a request cannot mark the station down.
    still = client.post(
        "/api/station-uptime", json={"station": "Bench A", "state": "down", "notes": "x"}, headers=AUTH
    ).json()
    assert still["state"] == READY
    for notes in (7, "x" * 4001):
        refused = client.post("/api/station-uptime", json={"station": "Bench A", "notes": notes}, headers=AUTH)
        assert (refused.status_code, refused.json()["detail"]) == (
            400,
            "notes must be text of at most 4000 characters",
        )
    assert client.post("/api/station-uptime", json={"station": "Bench A"}, headers=AUTH).status_code == 400
    assert uptime.report("Bench A")["notes"] == "x"


def test_the_days_notes_are_returned_only_with_the_token(tmp_path):
    client = _client(tmp_path)
    bench = _add(client)
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    client.post("/api/station-uptime", json={"station": "Bench A", "notes": "operator was unwell"}, headers=AUTH)
    # The operator's free text: an open read gets the station's time and an empty note.
    for headers in ({}, {"X-Control-Token": "wrong"}):
        opened = client.get("/api/station-uptime", headers=headers).json()
        assert (opened["station"], opened["notes"]) == ("Bench A", "")
    assert client.get("/api/station-uptime", headers=AUTH).json()["notes"] == "operator was unwell"


def test_notes_written_for_another_station_than_the_one_in_use_are_refused_and_nothing_is_written(tmp_path):
    client = _client(tmp_path)
    uptime = client.app.state.station_uptime
    bench, cart = _add(client), _add(client, {"name": "Cart B"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    client.post("/api/station-uptime", json={"station": "Bench A", "notes": "bench notes"}, headers=AUTH)
    # The dialog was opened for Bench A; Cart B was put in use before its draft was saved.
    client.post("/api/stations/select", json={"id": cart}, headers=AUTH)
    client.post("/api/station-uptime", json={"station": "Cart B", "notes": "cart notes"}, headers=AUTH)
    stale = client.post("/api/station-uptime", json={"station": "Bench A", "notes": "a draft"}, headers=AUTH)
    assert (stale.status_code, stale.json()) == (
        409,
        {"detail": "the station in use changed; reload the report"},
    )
    assert uptime.report("Cart B")["notes"] == "cart notes" and uptime.report("Bench A")["notes"] == "bench notes"
    # A save that does not say which station it is for cannot be checked, so it is not accepted either.
    for body in ({"notes": "a draft"}, {"station": 7, "notes": "a draft"}):
        unnamed = client.post("/api/station-uptime", json=body, headers=AUTH)
        assert (unnamed.status_code, unnamed.json()["detail"]) == (
            400,
            "station must name the station the notes are for",
        )
    assert uptime.report("Cart B")["notes"] == "cart notes"


def test_renaming_the_station_in_use_carries_its_time_and_notes(tmp_path):
    client = _client(tmp_path)
    uptime = client.app.state.station_uptime
    bench = _add(client, {**BENCH, "name": "Bnech A"})
    client.post("/api/stations/select", json={"id": bench}, headers=AUTH)
    uptime.observe("Bnech A", RECORDING, None)
    client.post("/api/station-uptime", json={"station": "Bnech A", "notes": "typo in the name"}, headers=AUTH)
    client.put(f"/api/stations/{bench}", json={"name": "Bench A"}, headers=AUTH)
    report = client.get("/api/station-uptime", headers=AUTH).json()
    assert (report["station"], report["state"], report["notes"]) == ("Bench A", RECORDING, "typo in the name")
    assert uptime.names("Bench A") == frozenset({"Bench A", "Bnech A"})


def test_notes_that_cannot_be_written_are_a_503(tmp_path, monkeypatch):
    client = _client(tmp_path)

    def refuse(self, station, notes):
        raise OSError("disk full")

    monkeypatch.setattr(UptimeLog, "set_notes", refuse)
    note = {"station": "Deployment", "notes": "x"}
    assert client.post("/api/station-uptime", json=note, headers=AUTH).status_code == 503


def test_the_real_application_gates_recording_on_the_station_in_use(monkeypatch, unused_tcp_port, tmp_path):
    monkeypatch.setenv("EGOCAP_CAPTURE_SERVER_URL", f"ws://127.0.0.1:{unused_tcp_port}")
    monkeypatch.delenv("EGOCAP_OPERATOR_SETTINGS_FILE", raising=False)
    monkeypatch.delenv("EGOCAP_STATION_CONFIG", raising=False)
    monkeypatch.setenv("EGOCAP_CALIBRATION_DIR", str(tmp_path / "calibration"))
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
        # Capture never connected here, so the station itself is fine and the checklist is what holds a take.
        assert client.app.state.station_service.block_reason() is None
        # The checklist is judged by a background duty, so the new station's result arrives within a moment.
        _poll_until_sync(lambda: (context.setup_block() or "").startswith("Capture session: "))

        def calibration() -> str:
            checks = client.get("/api/setup").json()["checks"]
            return next(item["status"] for item in checks if item["id"] == "calibration")

        # No calibration file exists. That is advisory until the deployment says calibration is required.
        assert calibration() == "warn"
        required = client.put("/api/operator-settings", json={"require_calibration": True}, headers=auth)
        assert required.status_code == 200 and calibration() == "fail"
        client.put("/api/operator-settings", json={"require_calibration": False}, headers=auth)
        assert calibration() == "warn"
        client.put(f"/api/stations/{bench}", json={"camera_roles": {"ego": None}}, headers=auth)
        assert context.snapshot()["not_ready_reason"] == (
            "station 'Bench A': no camera is assigned to the ego (head) role"
        )
        report = client.get("/api/station-uptime").json()
        assert (report["station"], report["active_seconds"]) == ("Bench A", 0.0)
        # A backend that has recorded nothing leaves no timeline behind.
        assert not (tmp_path / "station_status").exists()
