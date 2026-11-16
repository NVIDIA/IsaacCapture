# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-of-day routes: the day's report, saving it, and a task correction that stays inside the recording output."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import zarr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from egocap.recording.episode_layout import ROUTE_ATTR, EpisodeManifest, manifest_path, route_id, write_manifest
from egocap_web import day_report_routes
from egocap_web.station_activity import StationActivity

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}
TODAY = datetime.date.today().isoformat()
SESSION = f"session_{TODAY.replace('-', '')}-081500_abc123"


class _Recorder:
    """The recording facts a test can change: the open session and the take being written."""

    def __init__(self):
        self.session_id: str | None = None
        self.episode_id: str | None = None
        self.operation: str | None = None

    def snapshot(self):
        if self.session_id is None:
            return SimpleNamespace(service=None, active_operation=self.operation)
        episode = SimpleNamespace(episode_id=self.episode_id) if self.episode_id is not None else None
        session = SimpleNamespace(session_id=self.session_id)
        return SimpleNamespace(
            service=SimpleNamespace(session=session, episode=episode), active_operation=self.operation
        )


def _client(tmp_path: Path, ready: bool = True) -> tuple[TestClient, _Recorder]:
    small = FastAPI()
    small.include_router(day_report_routes.router)
    recorder = _Recorder()
    if ready:
        small.state.recording_context = SimpleNamespace(control_token=TOKEN, coordinator=recorder)
        small.state.station_service = SimpleNamespace(label=lambda: "Bench A")
        small.state.station_activity = StationActivity(tmp_path / "station_status")
        small.state.recording_output_root = tmp_path
    return TestClient(small), recorder


def _manifest(
    index: int, phase: str, *, session: str = SESSION, duration: float = 10.0, **fields
) -> EpisodeManifest:
    return EpisodeManifest(
        session_id=session,
        episode_id=f"{session}/episode_{index}",
        episode_index=index,
        config_digest="digest",
        task="Pour water.",
        operator="alex",
        sources=[],
        pending_path=f"episode_{index}.pending",
        final_path=f"episode_{index}",
        storage_phase=phase,
        duration_seconds=duration,
        **fields,
    )


def _session(
    root: Path, name: str = SESSION, station: str = "Bench A", takes: tuple[EpisodeManifest, ...] = ()
) -> Path:
    """A recorder session with real Zarr episode groups for its finalized takes."""
    session = root / name
    (session / ".recording").mkdir(parents=True)
    (session / "config.yaml").write_text(f"meta:\n  egocap_station_name: {station}\n")
    for manifest in takes:
        write_manifest(session, manifest)
        if manifest.storage_phase == "FINALIZED":
            group = zarr.open_group(str(session / manifest.final_path), mode="w")
            group.attrs.update(
                {"recorded_at": f"{TODAY}; 08-20-00", "episode_duration": manifest.duration_seconds}
            )
    return session


def _day(root: Path) -> Path:
    return _session(
        root,
        takes=(
            _manifest(0, "FINALIZED", duration=12.5),
            _manifest(1, "FINALIZED", duration=8.0, qc_decision="FLAGGED_BAD"),
            _manifest(2, "REJECTED", duration=3.0),
            _manifest(3, "SEALED", duration=4.0),
        ),
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/api/day-report"), ("post", "/api/day-report/save"), ("post", "/api/day-report/correct")],
)
def test_every_route_needs_the_application_and_the_token(tmp_path, method, path):
    starting, _ = _client(tmp_path, ready=False)
    assert getattr(starting, method)(path).status_code == 503
    client, _ = _client(tmp_path)
    body = {} if method == "get" else {"json": {}}
    assert getattr(client, method)(path, **body).status_code == 403
    assert getattr(client, method)(path, headers={"X-Control-Token": "wrong"}, **body).status_code == 403


def test_the_report_shows_the_days_takes_with_their_real_dispositions_and_the_activity(tmp_path):
    client, recorder = _client(tmp_path)
    _day(tmp_path)
    _session(tmp_path, f"session_{TODAY.replace('-', '')}-090000_def456", station="Cart B")
    client.app.state.station_activity.change("Bench A", state="active", notes="left glove battery")
    body = client.get("/api/day-report", headers=AUTH).json()
    report = body["report"]
    assert (report["date"], report["station"], body["today"], body["station_in_use"]) == (
        TODAY,
        "Bench A",
        TODAY,
        "Bench A",
    )
    assert {name: entry["count"] for name, entry in report["totals"]["by_disposition"].items()} == {
        "kept": 1,
        "flagged": 1,
        "rejected": 1,
        "discarded": 0,
        "pending": 1,
        "recording": 0,
    }
    assert report["totals"]["seconds"] == 27.5 and [s["session_id"] for s in report["sessions"]] == [SESSION]
    assert (body["activity"]["state"], body["activity"]["notes"]) == ("active", "left glove battery")
    # The take being written now is "recording", not an interrupted one.
    recorder.session_id, recorder.episode_id = SESSION, f"{SESSION}/episode_3"
    live = client.get("/api/day-report", headers=AUTH).json()["report"]
    assert live["sessions"][0]["episodes"][3]["disposition"] == "recording"


def test_the_report_can_be_asked_for_another_station_every_station_or_another_day(tmp_path):
    client, _ = _client(tmp_path)
    _day(tmp_path)
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    _session(tmp_path, other, station="Cart B", takes=(_manifest(0, "FINALIZED", session=other),))
    cart = client.get("/api/day-report", params={"station": "Cart B"}, headers=AUTH).json()
    assert [s["session_id"] for s in cart["report"]["sessions"]] == [other] and cart["activity"] is not None
    everything = client.get("/api/day-report", params={"station": "*"}, headers=AUTH).json()
    assert (everything["report"]["station"], len(everything["report"]["sessions"])) == (None, 2)
    assert everything["activity"] is None  # activity is one station's
    yesterday = client.get("/api/day-report", params={"date": "2020-01-01"}, headers=AUTH).json()
    assert yesterday["report"]["sessions"] == [] and yesterday["activity"] is None  # and today's only
    refused = client.get("/api/day-report", params={"date": "yesterday"}, headers=AUTH)
    assert (refused.status_code, refused.json()["detail"]) == (400, "date must be a day as YYYY-MM-DD")


def test_saving_writes_the_report_and_brings_the_activity_log_up_to_date(tmp_path):
    client, _ = _client(tmp_path)
    _day(tmp_path)
    client.app.state.station_activity.change("Bench A", state="active", notes="left glove battery")
    saved = client.post("/api/day-report/save", json={}, headers=AUTH).json()["saved"]
    assert saved == {"report": f"bench_a_{TODAY}_report.json", "activity_log": f"bench_a_{TODAY}_activity.json"}
    status_dir = tmp_path / "station_status"
    report = json.loads((status_dir / saved["report"]).read_text())
    assert report["report"]["totals"]["episodes"] == 4 and report["activity"]["notes"] == "left glove battery"
    assert datetime.datetime.fromisoformat(report["generated_at"]).date().isoformat() == TODAY
    log = json.loads((status_dir / saved["activity_log"]).read_text())
    assert (log["data_seconds"], log["valid_seconds"]) == (27.5, 12.5)  # everything recorded, and kept unflagged
    assert sorted(path.name for path in status_dir.iterdir()) == [
        f"{TODAY}.json",
        saved["activity_log"],
        saved["report"],
    ]
    # Every station, or another day: a report is written, the one-station activity log is left alone.
    every = client.post("/api/day-report/save", json={"station": "*"}, headers=AUTH).json()["saved"]
    assert every == {"report": f"all_stations_{TODAY}_report.json", "activity_log": None}
    assert client.post("/api/day-report/save", json={"date": "nope"}, headers=AUTH).status_code == 400
    assert client.post("/api/day-report/save", json={"station": 7}, headers=AUTH).json()["detail"] == (
        "station must be a station name"
    )


def test_a_report_that_cannot_be_written_is_a_503(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)

    def refuse(self, station, date, report):
        raise OSError("disk full")

    monkeypatch.setattr(StationActivity, "save_report", refuse)
    assert client.post("/api/day-report/save", json={}, headers=AUTH).status_code == 503


def _correct(client: TestClient, **body):
    return client.post("/api/day-report/correct", json=body, headers=AUTH)


def test_a_correction_rewrites_the_task_of_the_named_episodes_and_shows_in_the_report(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    response = _correct(
        client, session=SESSION, episodes="0-2", task=" Place the cup on the shelf. ", task_name="place_cup"
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "session": SESSION,
        "updated": ["episode_0", "episode_1"],
        "missing": ["episode_2"],
        "studio": None,
        "config_updated": True,
    }
    meta = yaml.safe_load((session / "config.yaml").read_text())["meta"]
    assert (meta["task_description"], meta["task_name"], meta["egocap_station_name"]) == (
        "Place the cup on the shelf.",
        "place_cup",
        "Bench A",
    )
    tasks = [
        e["task"] for e in client.get("/api/day-report", headers=AUTH).json()["report"]["sessions"][0]["episodes"]
    ]
    assert tasks == ["Place the cup on the shelf."] * 2 + ["Pour water."] * 2


@pytest.mark.parametrize(
    ("session", "reason"),
    [
        ("../elsewhere", "the session is outside the recording output"),
        ("/etc", "the session is outside the recording output"),
        ("", "session must be a path under the recording output"),
        (7, "session must be a path under the recording output"),
        (".", "the session is outside the recording output"),
        ("link_out", "the session is outside the recording output"),
        (".egocap", "that directory is not a recorded session"),
        (f"{SESSION}/.recording", "that directory is not a recorded session"),
        ("session_missing", "no such session"),
    ],
)
def test_a_correction_outside_the_recording_output_is_refused(tmp_path, session, reason):
    root = tmp_path / "recordings"
    root.mkdir()
    client, _ = _client(root)
    _day(root)
    (root / ".egocap").mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, root / "link_out")
    refused = _correct(client, session=session, episodes="0", task="x")
    assert (refused.status_code, refused.json()["detail"]) == (400, reason)
    assert list(outside.iterdir()) == []


def test_a_correction_is_refused_for_the_session_being_recorded(tmp_path):
    client, recorder = _client(tmp_path)
    _day(tmp_path)
    recorder.session_id = SESSION
    open_session = _correct(client, session=SESSION, episodes="0", task="x")
    assert (open_session.status_code, open_session.json()["detail"]) == (
        400,
        "this session is still open for recording; start a new batch first",
    )
    # A routed group stamped as the open session's is the same recording.
    group = tmp_path / "Kitchen" / "2026-03-09-08-15-00"
    group.mkdir(parents=True)
    stamp = {"attributes": {ROUTE_ATTR: route_id(SESSION, "Kitchen/2026-03-09-08-15-00")}}
    (group / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", **stamp}))
    assert (
        "still open for recording"
        in _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x").json()["detail"]
    )
    # Another session is not held by it, but nothing is corrected while a take is being written.
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    _session(tmp_path, other, takes=(_manifest(0, "FINALIZED", session=other),))
    assert _correct(client, session=other, episodes="0", task="x").status_code == 200
    for episode, operation in ((f"{SESSION}/episode_3", None), (None, "STOPPING")):
        recorder.episode_id, recorder.operation = episode, operation
        busy = _correct(client, session=other, episodes="0", task="x")
        assert (busy.status_code, busy.json()["detail"]) == (
            400,
            "finish or discard the current take before correcting a session",
        )
    recorder.session_id, recorder.episode_id, recorder.operation = None, None, None
    assert _correct(client, session=SESSION, episodes="0", task="x").status_code == 200


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"task": "   "}, "task cannot be empty"),
        ({"task": 7}, "task must be text"),
        ({"task": "x" * 1001}, "task must be text of at most 1000 characters"),
        ({"studio": ["Kitchen"]}, "studio must be text"),
        ({"episodes": "all"}, "episodes: expected 'N' or 'M-N'"),
        ({"episodes": 3}, "episodes must be a range such as 3 or 2-7"),
        ({"relocate": "yes"}, "episodes must be a range such as 3 or 2-7, and relocate true or false"),
        ({"relocate": True, "studio": "Kitchen"}, "only a session filed under a studio folder can be relocated"),
    ],
)
def test_an_invalid_correction_is_refused_with_the_reason_and_changes_nothing(tmp_path, change, reason):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    before = (session / "config.yaml").read_text()
    refused = _correct(client, **{"session": SESSION, "episodes": "0", "task": "x", **change})
    assert refused.status_code == 400 and reason in refused.json()["detail"]
    assert (session / "config.yaml").read_text() == before


def _legacy_session(root: Path, studio: str, name: str = "2026-03-09-08-15-00") -> Path:
    """A session from before the recording service: a plain Zarr group filed under a studio folder."""
    session = root / studio / name
    group = zarr.open_group(str(session), mode="w")
    group.create_group("episode_0").attrs.update({"episode_duration": 5.0})
    (session / "config.yaml").write_text("meta: {}\n")
    return session


def test_a_legacy_session_is_relocated_under_its_studio_inside_the_recording_output(tmp_path):
    client, _ = _client(tmp_path)
    _legacy_session(tmp_path, "_unmatched")
    moved = _correct(
        client,
        session="_unmatched/2026-03-09-08-15-00",
        episodes="0",
        task="Wash the cup.",
        studio="Kitchen",
        relocate=True,
    )
    assert moved.status_code == 200, moved.text
    assert (moved.json()["session"], moved.json()["studio"]) == ("Kitchen/2026-03-09-08-15-00", "Kitchen")
    assert (tmp_path / "Kitchen" / "2026-03-09-08-15-00" / "episode_0").is_dir()
    assert not (tmp_path / "_unmatched" / "2026-03-09-08-15-00").exists()
    # A second session of that name cannot be moved onto it.
    _legacy_session(tmp_path, "_unmatched")
    clash = _correct(
        client, session="_unmatched/2026-03-09-08-15-00", episodes="0", task="x", studio="Kitchen", relocate=True
    )
    assert (clash.status_code, clash.json()["detail"]) == (
        400,
        "the session could not be relocated: a session of that name is already under the studio",
    )
    missing_studio = _correct(
        client, session="_unmatched/2026-03-09-08-15-00", episodes="0", task="x", relocate=True
    )
    assert missing_studio.json()["detail"] == "a relocation needs a studio"


def test_a_recorder_session_keeps_its_location_and_only_its_labels_change(tmp_path):
    client, _ = _client(tmp_path)
    group = _legacy_session(tmp_path, "Bathroom")
    # Stamped as a routed group of a (closed) recorder session: its place is fixed by that session's manifests.
    stamped = zarr.open_group(str(group), mode="a")
    stamped.attrs[ROUTE_ATTR] = route_id("session_20260309-081500_abc123", "Bathroom/2026-03-09-08-15-00")
    refused = _correct(
        client, session="Bathroom/2026-03-09-08-15-00", episodes="0", task="x", studio="Kitchen", relocate=True
    )
    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        "this recording's location is fixed by its manifests; correct its labels without relocating it",
    )
    assert group.is_dir()
    relabelled = _correct(
        client, session="Bathroom/2026-03-09-08-15-00", episodes="0", task="Wash the cup.", studio="Kitchen"
    )
    assert relabelled.status_code == 200 and relabelled.json()["session"] == "Bathroom/2026-03-09-08-15-00"
    assert relabelled.json()["studio"] == "Kitchen"


def test_an_unreadable_group_stamp_does_not_hold_a_session(tmp_path):
    client, recorder = _client(tmp_path)
    recorder.session_id = SESSION
    group = _legacy_session(tmp_path, "Kitchen")
    (group / "zarr.json").write_text("not json")
    refused = _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x")
    assert "still open for recording" not in refused.text


def test_manifest_checkpoint_times_are_not_touched_by_a_correction(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    before = manifest_path(session, 0).read_text()
    assert _correct(client, session=SESSION, episodes="0", task="Place the cup.").status_code == 200
    assert manifest_path(session, 0).read_text() == before
