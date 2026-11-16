# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Station activity: active and inactive time through a day, and the daily log it feeds."""

from __future__ import annotations

import datetime
import json

import pytest

from egocap_web.station_activity import MAX_NOTES_CHARS, StationActivity, StationActivityError

#: A fixed local morning, so every step of a test lands on one calendar day.
DAY_START = datetime.datetime(2026, 3, 9, 8, 0, 0).timestamp()


class _Clock:
    def __init__(self):
        self.now = DAY_START

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _activity(tmp_path) -> tuple[StationActivity, _Clock]:
    clock = _Clock()
    return StationActivity(tmp_path / "station_status", clock), clock


def _log(tmp_path, station: str = "bench_a") -> dict:
    return json.loads((tmp_path / "station_status" / f"{station}_2026-03-09_activity.json").read_text())


def test_nothing_is_counted_before_the_operator_starts_the_day(tmp_path):
    activity, clock = _activity(tmp_path)
    clock.advance(600)
    assert activity.report("Bench A") == {
        "station": "Bench A",
        "date": "2026-03-09",
        "state": "inactive",
        "since": None,
        "active_seconds": 0.0,
        "inactive_seconds": 0.0,
        "blocker_count": 0,
        "notes": "",
        "as_of": DAY_START + 600,
    }
    assert not (tmp_path / "station_status").exists()  # reading never creates files


def test_the_daily_log_shows_correct_active_and_inactive_time(tmp_path):
    activity, clock = _activity(tmp_path)
    activity.change("Bench A", state="active")
    clock.advance(3600)  # an hour collecting
    blocked = activity.change("Bench A", state="inactive")
    assert (blocked["state"], blocked["since"]) == ("inactive", DAY_START + 3600)
    clock.advance(900)  # a quarter of an hour blocked
    activity.change("Bench A", state="active")
    clock.advance(1800)  # half an hour more, still open
    report = activity.report("Bench A")
    assert (report["active_seconds"], report["inactive_seconds"], report["blocker_count"]) == (5400.0, 900.0, 1)
    # The log on disk was written at the last change; the open segment is closed up to that moment.
    assert _log(tmp_path) == {
        "station": "Bench A",
        "date": "2026-03-09",
        "active_station_time": "1h 00m 00s",
        "active_station_time_seconds": 3600.0,
        "inactive_station_time": "0h 15m 00s",
        "inactive_station_time_seconds": 900.0,
        "data_hours": None,
        "data_seconds": None,
        "valid_hours": None,
        "valid_seconds": None,
    }
    assert sorted(path.name for path in (tmp_path / "station_status").iterdir()) == [
        "2026-03-09.json",
        "bench_a_2026-03-09_activity.json",
    ]


def test_setting_the_same_state_again_does_not_restart_its_segment(tmp_path):
    activity, clock = _activity(tmp_path)
    activity.change("Bench A", state="active")
    clock.advance(60)
    again = activity.change("Bench A", state="active")
    assert again["since"] == DAY_START and again["active_seconds"] == 60.0


def test_notes_are_saved_with_or_without_a_state_change(tmp_path):
    activity, clock = _activity(tmp_path)
    assert activity.change("Bench A", notes="left glove battery")["notes"] == "left glove battery"
    assert activity.report("Bench A")["state"] == "inactive"
    both = activity.change("Bench A", state="active", notes="")
    assert (both["state"], both["notes"]) == ("active", "")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"state": "paused"}, "state must be one of active, inactive"),
        ({"state": True}, "state must be one of"),
        ({"notes": 7}, "notes must be text"),
        ({"notes": "x" * (MAX_NOTES_CHARS + 1)}, f"at most {MAX_NOTES_CHARS} characters"),
    ],
)
def test_an_invalid_change_is_refused_and_writes_nothing(tmp_path, change, reason):
    activity, _ = _activity(tmp_path)
    with pytest.raises(StationActivityError, match=reason):
        activity.change("Bench A", **change)
    assert not (tmp_path / "station_status").exists()


def test_leaving_a_station_stops_its_active_time_and_touches_nothing_else(tmp_path):
    activity, clock = _activity(tmp_path)
    activity.leave("Cart B")  # never started today: stays untouched
    assert not (tmp_path / "station_status").exists()
    activity.change("Bench A", state="active")
    clock.advance(120)
    activity.leave("Bench A")
    clock.advance(600)
    report = activity.report("Bench A")
    assert (report["state"], report["active_seconds"], report["inactive_seconds"]) == ("inactive", 120.0, 600.0)
    activity.leave("Bench A")  # already inactive
    assert activity.report("Bench A")["blocker_count"] == 1


def test_measured_recorded_time_survives_a_later_state_change(tmp_path):
    activity, clock = _activity(tmp_path)
    activity.change("Bench A", state="active")
    clock.advance(600)
    assert activity.write_log("Bench A", data_seconds=540.0, valid_seconds=480.0) == (
        "bench_a_2026-03-09_activity.json"
    )
    assert (_log(tmp_path)["data_seconds"], _log(tmp_path)["valid_hours"]) == (540.0, round(480 / 3600, 4))
    activity.change("Bench A", state="inactive")
    log = _log(tmp_path)
    assert (log["data_seconds"], log["valid_seconds"], log["active_station_time_seconds"]) == (540.0, 480.0, 600.0)


@pytest.mark.parametrize("content", ["not json", "[1]", '{"data_seconds": true, "valid_seconds": "long"}'])
def test_an_unusable_previous_log_is_rewritten_without_recorded_time(tmp_path, content):
    activity, _ = _activity(tmp_path)
    log_dir = tmp_path / "station_status"
    log_dir.mkdir()
    (log_dir / "bench_a_2026-03-09_activity.json").write_text(content)
    activity.change("Bench A", state="active")
    assert (_log(tmp_path)["data_seconds"], _log(tmp_path)["valid_seconds"]) == (None, None)


def test_stations_are_tracked_independently(tmp_path):
    activity, clock = _activity(tmp_path)
    activity.change("Bench A", state="active")
    clock.advance(300)
    activity.change("Cart B", state="active")
    clock.advance(300)
    assert activity.report("Bench A")["active_seconds"] == 600.0
    assert activity.report("Cart B")["active_seconds"] == 300.0
