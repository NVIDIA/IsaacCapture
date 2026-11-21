# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from egocap.analytics.station_status_log import ACTIVE, INACTIVE, StationStatusTracker

DAY1 = 1_700_000_000.0  # an arbitrary fixed epoch, used as the "today" anchor


def test_init_and_log_dir(tmp_path):
    tracker = StationStatusTracker(tmp_path / "status")
    assert tracker.log_dir == tmp_path / "status"


def test_daily_log_path_canonical_station(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    path = tracker.daily_log_path("station_01", "2026-01-01")
    assert path.name == "station_01_2026-01-01_activity.json"


def test_daily_log_path_sanitized_station(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    path = tracker.daily_log_path("My Station!", "2026-01-01")
    assert path.name == "my_station_2026-01-01_activity.json"


def test_daily_log_path_default_date_uses_today(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    path = tracker.daily_log_path("station_01")
    assert "_activity.json" in path.name


def test_initial_state_is_inactive_with_no_segment(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    assert tracker.get_state("station_01", now=DAY1) == INACTIVE
    assert tracker.get_segment_start("station_01", now=DAY1) is None
    assert tracker.get_blocker_notes("station_01", now=DAY1) == ""


def test_toggle_first_call_opens_active_segment(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    record = tracker.toggle("station_01", now=DAY1)
    assert record["current_state"] == ACTIVE
    assert record["current_segment_start_ts"] == DAY1


def test_toggle_flips_state_and_closes_prior_segment(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.toggle("station_01", now=DAY1)
    record = tracker.toggle("station_01", now=DAY1 + 100)
    assert record["current_state"] == INACTIVE
    assert record["intervals"][0] == {
        "state": ACTIVE,
        "start_ts": DAY1,
        "end_ts": DAY1 + 100,
    }


def test_set_state_rejects_invalid_state(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    with pytest.raises(ValueError):
        tracker.set_state("station_01", "frozen", now=DAY1)


def test_set_state_noop_if_already_in_state_with_open_segment(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_state("station_01", ACTIVE, now=DAY1)
    record = tracker.set_state("station_01", ACTIVE, now=DAY1 + 50)
    assert record["current_segment_start_ts"] == DAY1  # unchanged


def test_set_state_closes_zero_duration_segment_cleanly(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_state("station_01", ACTIVE, now=DAY1)
    record = tracker.set_state("station_01", INACTIVE, now=DAY1)  # same ts
    assert record["intervals"] == []  # ts == seg_start, not appended


def test_set_blocker_notes(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_blocker_notes("station_01", "waiting on calibration", now=DAY1)
    assert tracker.get_blocker_notes("station_01", now=DAY1) == "waiting on calibration"


def test_set_blocker_notes_none_becomes_empty_string(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_blocker_notes("station_01", None, now=DAY1)
    assert tracker.get_blocker_notes("station_01", now=DAY1) == ""


def test_compute_totals_active_and_inactive_segments(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_state("station_01", ACTIVE, now=DAY1)
    tracker.set_state("station_01", INACTIVE, now=DAY1 + 100)
    tracker.set_state("station_01", ACTIVE, now=DAY1 + 150)
    totals = tracker.compute_totals("station_01", now=DAY1 + 200)
    assert totals["active_seconds"] == 150.0
    assert totals["inactive_seconds"] == 50.0
    assert totals["blocker_count"] == 1


def test_compute_totals_no_activity_yet(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    totals = tracker.compute_totals("station_01", now=DAY1)
    assert totals["active_seconds"] == 0.0
    assert totals["inactive_seconds"] == 0.0
    assert totals["blocker_count"] == 0


def test_save_daily_log_writes_payload_with_data_seconds(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.set_state("station_01", ACTIVE, now=DAY1)
    tracker.set_state("station_01", INACTIVE, now=DAY1 + 3600)
    path = tracker.save_daily_log(
        "station_01",
        date_iso="2026-01-01",
        now=DAY1 + 3600,
        data_seconds=1800.0,
        valid_seconds=1200.0,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active_station_time_seconds"] == 3600.0
    assert payload["data_hours"] == 0.5
    assert payload["data_seconds"] == 1800.0
    assert payload["valid_hours"] == round(1200.0 / 3600.0, 4)  # _hours_pair rounds to 4dp


def test_save_daily_log_none_data_seconds_writes_null(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    path = tracker.save_daily_log("station_01", date_iso="2026-01-01", now=DAY1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["data_hours"] is None
    assert payload["data_seconds"] is None


def test_list_stations(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.toggle("station_01", now=DAY1)
    tracker.toggle("station_02", now=DAY1)
    assert sorted(tracker.list_stations(now=DAY1)) == ["station_01", "station_02"]


def test_persistence_across_restart(tmp_path):
    tracker1 = StationStatusTracker(tmp_path)
    tracker1.toggle("station_01", now=DAY1)
    tracker1.toggle("station_01", now=DAY1 + 100)

    tracker2 = StationStatusTracker(tmp_path)
    totals = tracker2.compute_totals("station_01", now=DAY1 + 100)
    assert totals["active_seconds"] == 100.0


def test_invalidate_cache_forces_disk_reread(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    tracker.toggle("station_01", now=DAY1)  # populates and caches today's payload

    # Simulate a cross-process edit: a second tracker instance over the same
    # log_dir writes to today's file directly, bypassing the first instance's
    # in-memory cache entirely.
    other = StationStatusTracker(tmp_path)
    other.set_blocker_notes("station_01", "set by another process", now=DAY1)

    # Without invalidation the first tracker's cached payload is stale.
    assert tracker.get_blocker_notes("station_01", now=DAY1) == ""
    tracker.invalidate_cache()
    assert tracker.get_blocker_notes("station_01", now=DAY1) == "set by another process"


def test_load_payload_missing_file_returns_default(tmp_path):
    tracker = StationStatusTracker(tmp_path)
    payload = tracker._load_payload("2026-01-01")
    assert payload == {"date": "2026-01-01", "stations": {}}


def test_load_payload_malformed_json_returns_default(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "2026-01-01.json").write_text("not json", encoding="utf-8")
    tracker = StationStatusTracker(tmp_path)
    payload = tracker._load_payload("2026-01-01")
    assert payload == {"date": "2026-01-01", "stations": {}}


def test_load_payload_non_dict_json_returns_default(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "2026-01-01.json").write_text("[1, 2, 3]", encoding="utf-8")
    tracker = StationStatusTracker(tmp_path)
    assert tracker._load_payload("2026-01-01") == {"date": "2026-01-01", "stations": {}}


def test_load_payload_stations_not_dict_resets_to_dict(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "2026-01-01.json").write_text(
        json.dumps({"date": "2026-01-01", "stations": "not a dict"}), encoding="utf-8"
    )
    tracker = StationStatusTracker(tmp_path)
    payload = tracker._load_payload("2026-01-01")
    assert payload["stations"] == {}


def test_atomic_write_json_cleans_up_tempfile_on_failure(tmp_path, monkeypatch):
    import egocap.analytics.station_status_log as mod

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mod.json, "dump", boom)
    tracker = StationStatusTracker(tmp_path)
    with pytest.raises(OSError):
        tracker.toggle("station_01", now=DAY1)
    # No leftover temp file from the aborted write.
    assert list(tmp_path.glob("*.json.*")) == []


def test_atomic_write_json_unlink_failure_in_cleanup_is_swallowed(tmp_path, monkeypatch):
    import egocap.analytics.station_status_log as mod

    def boom_dump(*a, **k):
        raise OSError("disk full")

    def boom_unlink(*a, **k):
        raise OSError("already gone")

    monkeypatch.setattr(mod.json, "dump", boom_dump)
    monkeypatch.setattr(mod.os, "unlink", boom_unlink)
    tracker = StationStatusTracker(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        tracker.toggle("station_01", now=DAY1)


def test_compute_totals_two_inactive_intervals_in_a_row_no_blocker(tmp_path):
    import datetime

    date_str = datetime.date.fromtimestamp(DAY1).isoformat()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / f"{date_str}.json").write_text(
        json.dumps(
            {
                "date": date_str,
                "stations": {
                    "station_01": {
                        "current_state": INACTIVE,
                        "current_segment_start_ts": None,
                        "intervals": [
                            {"state": INACTIVE, "start_ts": DAY1, "end_ts": DAY1 + 5},
                            {"state": INACTIVE, "start_ts": DAY1 + 5, "end_ts": DAY1 + 10},
                        ],
                        "blocker_notes": "",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    tracker = StationStatusTracker(tmp_path)
    totals = tracker.compute_totals("station_01", now=DAY1 + 10)
    assert totals["inactive_seconds"] == 10.0
    assert totals["blocker_count"] == 0  # prev_state was never ACTIVE


def test_compute_totals_interval_with_unrecognized_state_is_ignored(tmp_path):
    import datetime

    date_str = datetime.date.fromtimestamp(DAY1).isoformat()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / f"{date_str}.json").write_text(
        json.dumps(
            {
                "date": date_str,
                "stations": {
                    "station_01": {
                        "current_state": INACTIVE,
                        "current_segment_start_ts": None,
                        "intervals": [
                            {"state": "unknown", "start_ts": DAY1, "end_ts": DAY1 + 10},
                        ],
                        "blocker_notes": "",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    tracker = StationStatusTracker(tmp_path)
    totals = tracker.compute_totals("station_01", now=DAY1 + 10)
    assert totals["active_seconds"] == 0.0
    assert totals["inactive_seconds"] == 0.0


def test_station_record_invalid_state_resets_to_default(tmp_path):
    import datetime

    date_str = datetime.date.fromtimestamp(DAY1).isoformat()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / f"{date_str}.json").write_text(
        json.dumps(
            {
                "date": date_str,
                "stations": {"station_01": {"current_state": "bogus"}},
            }
        ),
        encoding="utf-8",
    )
    tracker = StationStatusTracker(tmp_path)
    assert tracker.get_state("station_01", now=DAY1) == INACTIVE
