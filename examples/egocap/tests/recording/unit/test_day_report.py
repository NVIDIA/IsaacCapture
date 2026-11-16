# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import os
from pathlib import Path

import pytest

from egocap.analytics import day_report
from egocap.analytics.day_report import build_day_report, episode_disposition, recorded_seconds
from egocap.recording.episode_layout import ROUTE_ATTR, EpisodeManifest, manifest_path, route_id, write_manifest

DAY = "2026-03-09"
SESSION = "session_20260309-081500_abc123"


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


def _checkpoint_on(path: Path, day: str) -> None:
    """Set a manifest's last-checkpoint time to noon on ``day``."""
    noon = datetime.datetime.fromisoformat(f"{day}T12:00:00").timestamp()
    os.utime(path, (noon, noon))


def _session(root: Path, name: str = SESSION, station: str | None = "Bench A") -> Path:
    session = root / name
    (session / ".recording").mkdir(parents=True)
    if station is not None:
        (session / "config.yaml").write_text(f"meta:\n  egocap_station_name: {station}\n")
    return session


def _take(session: Path, manifest: EpisodeManifest, *, day: str = DAY, recorded_at: str | None = None) -> None:
    """Write one take's manifest, checkpointed on ``day``; ``recorded_at`` also writes its finalized episode."""
    write_manifest(session, manifest)
    _checkpoint_on(manifest_path(session, manifest.episode_index), day)
    if recorded_at is not None:
        episode = session / manifest.final_path
        episode.mkdir()
        (episode / "zarr.json").write_text(json.dumps({"attributes": {"recorded_at": recorded_at}}))


def _dispositions(report: dict) -> dict[str, int]:
    return {name: entry["count"] for name, entry in report["totals"]["by_disposition"].items() if entry["count"]}


@pytest.mark.parametrize(
    ("phase", "fields", "expected"),
    [
        ("FINALIZED", {}, "kept"),
        ("FINALIZED", {"qc_decision": "PASSED"}, "kept"),
        ("FINALIZED", {"qc_decision": "FLAGGED_BAD"}, "flagged"),
        ("REJECTED", {"qc_decision": "REJECTED_CAMERA"}, "rejected"),
        ("DISCARDED", {}, "discarded"),
        ("WRITING", {}, "pending"),
        ("SEALED", {}, "pending"),
        ("CHECKED", {"qc_decision": "ERROR"}, "pending"),
    ],
)
def test_a_take_shows_what_actually_became_of_it(phase, fields, expected):
    assert episode_disposition(_manifest(0, phase, **fields)) == expected


def test_the_take_being_recorded_is_not_mistaken_for_an_interrupted_one():
    manifest = _manifest(3, "WRITING")
    assert episode_disposition(manifest, active_episode_id=manifest.episode_id) == "recording"
    assert episode_disposition(manifest, active_episode_id="another/episode_0") == "pending"
    # A finished take keeps its outcome whatever is being recorded now.
    assert episode_disposition(_manifest(3, "FINALIZED"), active_episode_id=manifest.episode_id) == "kept"


def test_totals_come_from_the_manifests_with_each_takes_real_disposition(tmp_path):
    session = _session(tmp_path)
    _take(session, _manifest(0, "FINALIZED", duration=12.5), recorded_at=f"{DAY}; 08-20-00")
    _take(session, _manifest(1, "FINALIZED", duration=8.0, qc_decision="FLAGGED_BAD", issues=["sustained_freeze"]))
    _take(
        session, _manifest(2, "REJECTED", duration=3.0, qc_decision="REJECTED_CAMERA", issues=["camera_min_fps"])
    )
    _take(session, _manifest(3, "DISCARDED", duration=1.5))
    _take(session, _manifest(4, "SEALED", duration=4.0))
    report = build_day_report(tmp_path, DAY)
    assert _dispositions(report) == {"kept": 1, "flagged": 1, "rejected": 1, "discarded": 1, "pending": 1}
    assert report["totals"]["episodes"] == 5 and report["totals"]["seconds"] == 29.0
    assert report["totals"]["by_disposition"]["flagged"] == {"count": 1, "seconds": 8.0}
    assert (report["date"], report["station"]) == (DAY, None)
    (only,) = report["sessions"]
    assert (only["session_id"], only["station"], only["started_at"]) == (SESSION, "Bench A", "2026-03-09T08:15:00")
    assert only["problems"] == [] and only["by_disposition"] == report["totals"]["by_disposition"]
    assert [(e["index"], e["disposition"]) for e in only["episodes"]] == [
        (0, "kept"),
        (1, "flagged"),
        (2, "rejected"),
        (3, "discarded"),
        (4, "pending"),
    ]
    assert only["episodes"][1] == {
        "index": 1,
        "episode_id": f"{SESSION}/episode_1",
        "disposition": "flagged",
        "duration_seconds": 8.0,
        "task": "Pour water.",
        "task_details": "",
        "operator": "alex",
        "issues": ["sustained_freeze"],
        "qc_decision": "FLAGGED_BAD",
        "location": f"{SESSION}/episode_1",
    }
    assert recorded_seconds(report) == (29.0, 12.5)


def test_a_take_belongs_to_the_day_it_was_recorded_not_the_day_its_session_opened(tmp_path):
    session = _session(tmp_path)
    # Finalized after midnight: its own recorded_at outranks the manifest's checkpoint time.
    _take(session, _manifest(0, "FINALIZED"), day="2026-03-10", recorded_at=f"{DAY}; 23-59-30")
    _take(session, _manifest(1, "FINALIZED"), day="2026-03-10", recorded_at="2026-03-10; 00-04-00")
    # Never finalized: the day its manifest was last checkpointed.
    _take(session, _manifest(2, "DISCARDED"), day="2026-03-10")
    assert [e["index"] for e in build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]] == [0]
    assert [e["index"] for e in build_day_report(tmp_path, "2026-03-10")["sessions"][0]["episodes"]] == [1, 2]
    assert build_day_report(tmp_path, "2026-03-11")["sessions"] == []


def test_a_task_corrected_after_recording_is_the_one_reported(tmp_path):
    session = _session(tmp_path)
    _take(session, _manifest(0, "FINALIZED"), recorded_at=f"{DAY}; 08-20-00")
    _take(session, _manifest(1, "FINALIZED"), recorded_at=f"{DAY}; 08-21-00")
    attributes = {"recorded_at": f"{DAY}; 08-21-00", "task_description": "Place the cup on the shelf."}
    (session / "episode_1" / "zarr.json").write_text(json.dumps({"attributes": attributes}))
    tasks = [e["task"] for e in build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]]
    assert tasks == ["Pour water.", "Place the cup on the shelf."]


def test_an_unusable_recorded_at_falls_back_to_the_checkpoint_time(tmp_path):
    session = _session(tmp_path)
    _take(session, _manifest(0, "FINALIZED"), recorded_at="soon")
    _take(session, _manifest(1, "FINALIZED"))
    (session / "episode_1").mkdir()
    (session / "episode_1" / "zarr.json").write_text(json.dumps({"attributes": {"recorded_at": 20260309}}))
    assert [e["index"] for e in build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]] == [0, 1]


def test_a_station_report_holds_only_that_stations_sessions(tmp_path):
    _take(_session(tmp_path), _manifest(0, "FINALIZED"))
    other = "session_20260309-090000_def456"
    _take(_session(tmp_path, other, station="Cart B"), _manifest(0, "FINALIZED", session=other, duration=7.0))
    unlabelled = "session_20260309-100000_0a0a0a"
    _take(_session(tmp_path, unlabelled, station=None), _manifest(0, "FINALIZED", session=unlabelled))
    assert [s["station"] for s in build_day_report(tmp_path, DAY)["sessions"]] == ["Bench A", "Cart B", None]
    cart = build_day_report(tmp_path, DAY, station="Cart B")
    assert [s["session_id"] for s in cart["sessions"]] == [other]
    assert (cart["station"], cart["totals"]["episodes"], cart["totals"]["seconds"]) == ("Cart B", 1, 7.0)


def test_a_routed_take_is_located_at_its_assigned_slot_and_dated_from_it(tmp_path):
    session = _session(tmp_path)
    group = "Kitchen/2026-03-09-08-15-00"
    routed = _manifest(0, "FINALIZED", studio="Kitchen", assignment={"group": group, "slot": 4})
    _take(session, routed, day="2026-03-10")
    episode = tmp_path / group / "episode_4"
    episode.mkdir(parents=True)
    (tmp_path / group / "zarr.json").write_text(json.dumps({"attributes": {ROUTE_ATTR: route_id(SESSION, group)}}))
    (episode / "zarr.json").write_text(json.dumps({"attributes": {"recorded_at": f"{DAY}; 08-30-00"}}))
    (take,) = build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]
    assert take["location"] == f"{group}/episode_4"


def test_a_take_whose_final_location_cannot_be_trusted_is_dated_from_its_manifest(tmp_path):
    session = _session(tmp_path)
    # The group is stamped for another session, so the routed location is refused; the take is still counted.
    group = "Kitchen/2026-03-09-08-15-00"
    _take(session, _manifest(0, "FINALIZED", studio="Kitchen", assignment={"group": group, "slot": 0}))
    (tmp_path / group).mkdir(parents=True)
    (tmp_path / group / "zarr.json").write_text(json.dumps({"attributes": {ROUTE_ATTR: "someone-else|x"}}))
    # A slot that is not a number is refused the same way.
    _take(session, _manifest(1, "FINALIZED", studio="Kitchen", assignment={"group": group, "slot": "x"}))
    assert [e["index"] for e in build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]] == [0, 1]


def test_an_unreadable_manifest_is_reported_for_its_sessions_day_and_never_dropped(tmp_path):
    session = _session(tmp_path)
    (session / ".recording" / "episode_0.json").write_text("not json")
    report = build_day_report(tmp_path, DAY)
    (only,) = report["sessions"]
    assert only["episodes"] == [] and len(only["problems"]) == 1 and "episode_0.json" in only["problems"][0]
    assert report["totals"]["episodes"] == 0
    assert build_day_report(tmp_path, "2026-03-10")["sessions"] == []
    # A session whose name carries no date cannot be placed on a day, so it is reported on every day.
    odd = _session(tmp_path, "legacy_session")
    (odd / ".recording" / "episode_0.json").write_text("not json")
    assert [s["session_id"] for s in build_day_report(tmp_path, "2026-03-10")["sessions"]] == ["legacy_session"]
    assert build_day_report(tmp_path, "2026-03-10")["sessions"][0]["started_at"] is None


def test_readable_takes_of_another_day_do_not_carry_a_sessions_problems(tmp_path):
    session = _session(tmp_path)
    _take(session, _manifest(0, "FINALIZED"), day="2026-03-10")
    (session / ".recording" / "episode_1.json").write_text("not json")
    (next_day,) = build_day_report(tmp_path, "2026-03-10")["sessions"]
    assert [e["index"] for e in next_day["episodes"]] == [0] and next_day["problems"] == []


def test_a_manifest_that_vanishes_during_the_read_is_skipped(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _take(session, _manifest(0, "FINALIZED"))
    _take(session, _manifest(1, "DISCARDED"))
    real = day_report.manifest_path

    def vanishing(session_path, index):
        if index == 1:
            return session_path / ".recording" / "gone.json"
        return real(session_path, index)

    monkeypatch.setattr(day_report, "manifest_path", vanishing)
    assert [e["index"] for e in build_day_report(tmp_path, DAY)["sessions"][0]["episodes"]] == [0]


def test_only_recording_service_sessions_are_read(tmp_path):
    (tmp_path / ".egocap").mkdir()
    (tmp_path / ".egocap" / ".recording").mkdir()
    (tmp_path / "station_status").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert build_day_report(tmp_path, DAY)["sessions"] == []
    assert build_day_report(tmp_path / "missing", DAY) == {
        "date": DAY,
        "station": None,
        "totals": {
            "episodes": 0,
            "seconds": 0,
            "by_disposition": {name: {"count": 0, "seconds": 0.0} for name in day_report.DISPOSITIONS},
        },
        "sessions": [],
    }
