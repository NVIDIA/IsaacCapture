# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-of-day routes: today's report, saving it, and a task correction that stays inside the recording output."""

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
from egocap_web.session_locks import ALIGNER, SessionLocks
from egocap_web.station_uptime import DOWN, READY, RECORDING, UptimeLog
from recording_service import RecordingPhase

TOKEN = "t" * 43
AUTH = {"X-Control-Token": TOKEN}
TODAY = datetime.date.today().isoformat()
SESSION = f"session_{TODAY.replace('-', '')}-081500_abc123"


class _Recorder:
    """The recording facts a test can change: the open session and the take being written."""

    def __init__(self, root: Path):
        self.root = root
        self.session_id: str | None = None
        self.episode_id: str | None = None
        self.operation: str | None = None
        self.interrupted = False

    def snapshot(self):
        if self.session_id is None:
            return SimpleNamespace(service=None, active_operation=self.operation)
        episode = SimpleNamespace(episode_id=self.episode_id) if self.episode_id is not None else None
        session = SimpleNamespace(session_id=self.session_id, session_path=self.root / self.session_id)
        if episode is None:
            phase = RecordingPhase.IDLE
        else:
            phase = RecordingPhase.RECOVERY_REQUIRED if self.interrupted else RecordingPhase.RECORDING
        return SimpleNamespace(
            service=SimpleNamespace(session=session, episode=episode, phase=phase),
            active_operation=self.operation,
        )


def _client(tmp_path: Path, ready: bool = True) -> tuple[TestClient, _Recorder]:
    small = FastAPI()
    small.include_router(day_report_routes.router)
    recorder = _Recorder(tmp_path)
    if ready:
        small.state.recording_context = SimpleNamespace(control_token=TOKEN, coordinator=recorder)
        small.state.station_service = SimpleNamespace(label=lambda: "Bench A")
        small.state.station_uptime = UptimeLog(tmp_path / "station_status")
        small.state.recording_output_root = tmp_path
        small.state.session_locks = SessionLocks()
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


def test_the_report_shows_todays_takes_for_the_station_in_use_with_its_observed_time(tmp_path):
    client, recorder = _client(tmp_path)
    _day(tmp_path)
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    _session(tmp_path, other, station="Cart B", takes=(_manifest(0, "FINALIZED", session=other),))
    uptime = client.app.state.station_uptime
    uptime.observe("Bench A", RECORDING, None)
    uptime.observe("Bench A", DOWN, "Headset: disconnected")
    uptime.set_notes("Bench A", "left glove battery")
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
    # Another station's session is not this station's day.
    assert report["totals"]["seconds"] == 27.5 and [s["session_id"] for s in report["sessions"]] == [SESSION]
    assert (body["uptime"]["state"], body["uptime"]["reason"], body["uptime"]["notes"]) == (
        DOWN,
        "Headset: disconnected",
        "left glove battery",
    )
    # The take being written now is "recording", not an interrupted one.
    recorder.session_id, recorder.episode_id = SESSION, f"{SESSION}/episode_3"
    live = client.get("/api/day-report", headers=AUTH).json()["report"]
    assert live["sessions"][0]["episodes"][3]["disposition"] == "recording"
    # Once that take is interrupted nothing is writing it: it waits to be recovered or discarded.
    recorder.interrupted = True
    waiting = client.get("/api/day-report", headers=AUTH).json()["report"]
    assert waiting["sessions"][0]["episodes"][3]["disposition"] == "pending"
    # The report is always today's for the station in use: it takes no day and no station.
    same = client.get("/api/day-report", params={"date": "2020-01-01", "station": "*"}, headers=AUTH).json()
    assert (same["report"]["date"], same["report"]["station"]) == (TODAY, "Bench A")


def test_takes_recorded_before_a_rename_are_still_the_stations_day(tmp_path):
    client, _ = _client(tmp_path)
    _session(tmp_path, station="Bnech A", takes=(_manifest(0, "FINALIZED"),))
    uptime = client.app.state.station_uptime
    uptime.observe("Bnech A", RECORDING, None)
    uptime.rename("Bnech A", "Bench A")
    report = client.get("/api/day-report", headers=AUTH).json()["report"]
    assert (report["station"], report["stations"], report["totals"]["episodes"]) == (
        "Bench A",
        ["Bench A", "Bnech A"],
        1,
    )


def test_saving_writes_the_report_and_brings_the_daily_log_up_to_date(tmp_path):
    client, _ = _client(tmp_path)
    _day(tmp_path)
    uptime = client.app.state.station_uptime
    uptime.observe("Bench A", RECORDING, None)
    uptime.observe("Bench A", READY, None)
    uptime.set_notes("Bench A", "left glove battery")
    assert uptime.report_saved("Bench A") is False
    saved = client.post("/api/day-report/save", json={}, headers=AUTH).json()["saved"]
    assert saved == {"report": f"bench_a_{TODAY}_report.json", "activity_log": f"bench_a_{TODAY}_activity.json"}
    assert uptime.report_saved("Bench A") is True
    status_dir = tmp_path / "station_status"
    report = json.loads((status_dir / saved["report"]).read_text())
    assert report["report"]["totals"]["episodes"] == 4 and report["uptime"]["notes"] == "left glove battery"
    assert datetime.datetime.fromisoformat(report["generated_at"]).date().isoformat() == TODAY
    log = json.loads((status_dir / saved["activity_log"]).read_text())
    assert (log["data_seconds"], log["valid_seconds"]) == (27.5, 12.5)  # everything recorded, and kept unflagged
    assert sorted(path.name for path in status_dir.iterdir()) == [
        f"{TODAY}_timeline.json",
        saved["activity_log"],
        saved["report"],
    ]


def test_a_report_that_cannot_be_written_is_a_503(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)

    def refuse(self, station, report):
        raise OSError("disk full")

    monkeypatch.setattr(UptimeLog, "save_report", refuse)
    assert client.post("/api/day-report/save", json={}, headers=AUTH).status_code == 503


def _correct(client: TestClient, **body):
    return client.post("/api/day-report/correct", json=body, headers=AUTH)


def test_a_correction_rewrites_the_task_of_the_named_takes_and_shows_in_the_report(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    response = _correct(client, session=SESSION, episodes="0-2", task=" Place the cup on the shelf. ")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "session": SESSION,
        "updated": ["episode_0", "episode_1"],
        "missing": ["episode_2"],
        "config_updated": True,
    }
    meta = yaml.safe_load((session / "config.yaml").read_text())["meta"]
    assert (meta["task_description"], meta["egocap_station_name"]) == ("Place the cup on the shelf.", "Bench A")
    tasks = [
        e["task"] for e in client.get("/api/day-report", headers=AUTH).json()["report"]["sessions"][0]["episodes"]
    ]
    assert tasks == ["Place the cup on the shelf."] * 2 + ["Pour water."] * 2
    # Correcting one of the two finalized takes leaves the session's own task as it was.
    _correct(client, session=SESSION, episodes="1", task="Wipe the table.")
    assert yaml.safe_load((session / "config.yaml").read_text())["meta"]["task_description"] == (
        "Place the cup on the shelf."
    )


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
        ("session_missing", "that directory is not a recorded session"),
        # Inside the output but not a session: an episode, the status folder, a studio folder.
        (f"{SESSION}/episode_0", "that directory is not a recorded session"),
        ("station_status", "that directory is not a recorded session"),
        ("Kitchen", "that directory is not a recorded session"),
        ("Kitchen/2026-03-09-08-15-00/episode_0", "that directory is not a recorded session"),
        # A group is known by the recording service's stamp, not by being a Zarr group two levels down.
        ("Kitchen/unstamped", "that directory is not a recorded session"),
        # A time-aligned copy, or an episode of one, is only ever corrected with its session.
        (f"{SESSION}_aligned", "that directory is not a recorded session"),
        (f"{SESSION}_aligned/episode_0", "that directory is not a recorded session"),
        ("Kitchen/2026-03-09-08-15-00_aligned", "that directory is not a recorded session"),
        # Values the filesystem will not even look up.
        ("a\x00b", "that directory is not a recorded session"),
        ("a" * 300, "that directory is not a recorded session"),
        ("a/" * 4000, "that directory is not a recorded session"),
    ],
)
def test_a_correction_of_anything_but_a_recorded_session_inside_the_output_is_refused(tmp_path, session, reason):
    root = tmp_path / "recordings"
    root.mkdir()
    client, _ = _client(root)
    _day(root)
    (root / ".egocap").mkdir()
    (root / "station_status").mkdir()
    stamp = {ROUTE_ATTR: route_id(SESSION, "Kitchen/2026-03-09-08-15-00")}
    routed = zarr.open_group(str(root / "Kitchen" / "2026-03-09-08-15-00"), mode="w", attributes=stamp)
    routed.create_group("episode_0")
    zarr.open_group(str(root / "Kitchen" / "unstamped"), mode="w").create_group("episode_0")
    # Derived copies carry their source's attributes, the stamp included.
    for copy in (f"{SESSION}_aligned", "Kitchen/2026-03-09-08-15-00_aligned"):
        zarr.open_group(str(root / copy), mode="w", attributes=stamp).create_group("episode_0", attributes=stamp)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, root / "link_out")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    refused = _correct(client, session=session, episodes="0", task="x")
    assert (refused.status_code, refused.json()["detail"]) == (400, reason)
    # Nothing was created anywhere, inside the output or out of it.
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


def test_an_aligned_copy_that_leads_out_of_the_output_refuses_the_correction(tmp_path):
    root = tmp_path / "recordings"
    root.mkdir()
    client, _ = _client(root)
    _day(root)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink(outside, root / f"{SESSION}_aligned")
    refused = _correct(client, session=SESSION, episodes="0", task="x")
    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        "the session's aligned copy is outside the recording output",
    )
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
    stamped = zarr.open_group(str(group), mode="w")
    stamped.attrs[ROUTE_ATTR] = route_id(SESSION, "Kitchen/2026-03-09-08-15-00")
    stamped.create_group("episode_0")
    held = _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x")
    assert "still open for recording" in held.json()["detail"]
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
    assert _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x").status_code == 200


def test_a_group_whose_stamp_cannot_be_read_is_not_taken_for_the_open_session(tmp_path):
    client, recorder = _client(tmp_path)
    recorder.session_id = SESSION  # open, but its directory was never created
    group = tmp_path / "Kitchen" / "2026-03-09-08-15-00"
    group.mkdir(parents=True)
    (group / "zarr.json").write_text("not json")
    refused = _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x")
    assert (refused.status_code, refused.json()["detail"]) == (400, "that directory is not a recorded session")
    assert (group / "zarr.json").read_text() == "not json"
    # A stamp of the wrong shape is no stamp either.
    (group / "zarr.json").write_text(json.dumps({"attributes": {ROUTE_ATTR: 7}}))
    assert _correct(client, session="Kitchen/2026-03-09-08-15-00", episodes="0", task="x").status_code == 400


def test_a_group_of_another_session_is_corrected_while_the_open_one_has_no_directory_yet(tmp_path):
    client, recorder = _client(tmp_path)
    recorder.session_id = SESSION  # open, but its directory was never created
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    stamp = {ROUTE_ATTR: route_id(other, "Kitchen/2026-03-09-09-00-00")}
    group = zarr.open_group(str(tmp_path / "Kitchen" / "2026-03-09-09-00-00"), mode="w", attributes=stamp)
    group.create_group("episode_0")
    corrected = _correct(client, session="Kitchen/2026-03-09-09-00-00", episodes="0", task="x")
    assert (corrected.status_code, corrected.json()["updated"]) == (200, ["episode_0"])


def test_metadata_that_cannot_be_rewritten_is_refused_without_naming_a_path(tmp_path, caplog):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    (session / "zarr.json").write_text("not json")
    refused = _correct(client, session=SESSION, episodes="0", task="x")
    assert (refused.status_code, refused.json()["detail"]) == (400, "the session's metadata could not be updated")
    assert "A task correction could not be written" in caplog.text


def test_an_interrupted_take_holds_only_its_own_session(tmp_path):
    client, recorder = _client(tmp_path)
    _day(tmp_path)
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    _session(tmp_path, other, takes=(_manifest(0, "FINALIZED", session=other),))
    recorder.session_id, recorder.episode_id, recorder.interrupted = SESSION, f"{SESSION}/episode_3", True
    # Nothing is being written, so another session can be corrected; the interrupted take's own is still open.
    assert _correct(client, session=other, episodes="0", task="x").status_code == 200
    held = _correct(client, session=SESSION, episodes="0", task="x")
    assert (held.status_code, held.json()["detail"]) == (
        400,
        "this session is still open for recording; start a new batch first",
    )


def test_a_correction_waits_while_a_take_of_that_session_is_being_time_aligned(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    other = f"session_{TODAY.replace('-', '')}-090000_def456"
    _session(tmp_path, other, takes=(_manifest(0, "FINALIZED", session=other),))
    locks = client.app.state.session_locks
    # The aligner holds the session while it copies a take's labels and until the copy is published.
    assert locks.acquire(session.resolve(), ALIGNER) is None
    before = (session / "config.yaml").read_text()
    busy = _correct(client, session=SESSION, episodes="0", task="x")
    assert (busy.status_code, busy.json()["detail"]) == (
        400,
        "time alignment of this take is still running; try again in a moment",
    )
    assert (session / "config.yaml").read_text() == before
    assert _correct(client, session=other, episodes="0", task="x").status_code == 200
    locks.release(session.resolve())
    assert _correct(client, session=SESSION, episodes="0", task="x").status_code == 200


def test_a_correction_holds_its_session_from_its_first_write_to_its_last(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    locks = client.app.state.session_locks
    update = day_report_routes.update_episode_task_description
    during: list[tuple[str | None, int]] = []

    def writing(target, episodes, *, task_description):
        # What the aligner, and a second correction, find while this one is between its files.
        second = _correct(client, session=SESSION, episodes="0", task="other")
        during.append((locks.acquire(target, ALIGNER), second.status_code))
        return update(target, episodes, task_description=task_description)

    monkeypatch.setattr(day_report_routes, "update_episode_task_description", writing)
    first = _correct(client, session=SESSION, episodes="0", task="Wash the cup.")
    assert first.status_code == 200 and during == [("correction", 400)]
    assert zarr.open_group(str(session / "episode_0"), mode="r").attrs["task_description"] == "Wash the cup."
    # Let go when it ends, however it ends.
    assert locks.acquire(session.resolve(), ALIGNER) is None
    locks.release(session.resolve())

    def failing(target, episodes, *, task_description):
        raise OSError("disk full")

    monkeypatch.setattr(day_report_routes, "update_episode_task_description", failing)
    assert _correct(client, session=SESSION, episodes="0", task="x").status_code == 400
    assert locks.acquire(session.resolve(), ALIGNER) is None


def test_two_corrections_of_one_session_never_interleave(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    _day(tmp_path)
    update = day_report_routes.update_episode_task_description
    replies: list[tuple[int, str]] = []

    def writing(target, episodes, *, task_description):
        second = _correct(client, session=SESSION, episodes="0", task="second")
        replies.append((second.status_code, second.json()["detail"]))
        return update(target, episodes, task_description=task_description)

    monkeypatch.setattr(day_report_routes, "update_episode_task_description", writing)
    assert _correct(client, session=SESSION, episodes="0", task="first").status_code == 200
    assert replies == [(400, "another correction of this session is still being written; try again in a moment")]


def test_a_correction_needs_the_session_locks_the_application_sets(tmp_path):
    client, _ = _client(tmp_path)
    _day(tmp_path)
    del client.app.state.session_locks
    assert _correct(client, session=SESSION, episodes="0", task="x").status_code == 503


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"task": "   "}, "task must be text of 1 to 1000 characters"),
        ({"task": 7}, "task must be text of 1 to 1000 characters"),
        ({"task": "x" * 1001}, "task must be text of 1 to 1000 characters"),
        ({"episodes": "all"}, "episodes: expected 'N' or 'M-N'"),
        ({"episodes": 3}, "episodes must be a take or a range, such as 3 or 2-7"),
        ({"episodes": "0-" + "9" * 40}, "episodes must be a take or a range, such as 3 or 2-7"),
        ({"episodes": "0-3000000000"}, "episodes: a range covers at most 10000 episodes"),
    ],
)
def test_an_invalid_correction_is_refused_with_the_reason_and_changes_nothing(tmp_path, change, reason):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    before = (session / "config.yaml").read_text()
    refused = _correct(client, **{"session": SESSION, "episodes": "0", "task": "x", **change})
    assert refused.status_code == 400 and reason in refused.json()["detail"]
    assert (session / "config.yaml").read_text() == before


def test_a_correction_never_moves_a_session_and_ignores_fields_it_does_not_take(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    response = _correct(client, session=SESSION, episodes="0", task="x", studio="..", relocate=True)
    assert response.status_code == 200 and response.json()["session"] == SESSION
    assert session.is_dir() and "studio" not in yaml.safe_load((session / "config.yaml").read_text())["meta"]


def test_manifests_are_not_touched_by_a_correction(tmp_path):
    client, _ = _client(tmp_path)
    session = _day(tmp_path)
    before = manifest_path(session, 0).read_text()
    assert _correct(client, session=SESSION, episodes="0", task="Place the cup.").status_code == 200
    assert manifest_path(session, 0).read_text() == before
