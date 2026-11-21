# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The uptime log: a station's active and down time, observed through a day and counted between its takes."""

from __future__ import annotations

import datetime
import json
import time

import pytest

from egocap_web import station_uptime
from egocap_web.station_uptime import (
    BACKEND_NOT_RUNNING,
    DOWN,
    LOG_FULL,
    MAX_NOTES_CHARS,
    READY,
    RECORDING,
    UptimeError,
    UptimeLog,
    UptimeSampler,
    window_totals,
)

#: A fixed local morning, so every step of a test lands on one calendar day unless it means to cross midnight.
DAY_START = datetime.datetime(2026, 3, 9, 8, 0, 0).timestamp()


#: How far apart a lasting state is sampled here; well inside what the log takes for a silence.
_SAMPLE_SECONDS = 60.0


class _Clock:
    def __init__(self, start: float = DAY_START):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _log(tmp_path, clock: _Clock | None = None) -> tuple[UptimeLog, _Clock]:
    clock = clock or _Clock()
    return UptimeLog(tmp_path / "station_status", clock), clock


def _day_file(tmp_path, day: str = "2026-03-09") -> dict:
    return json.loads((tmp_path / "station_status" / f"{day}_timeline.json").read_text())


def _run(
    log: UptimeLog, clock: _Clock, steps: list[tuple[str, str | None, float]], station: str = "Bench A"
) -> None:
    """Observe each (state, reason) and let it last the given seconds, sampled through it as the backend does:
    a silence between two samples would be logged as an outage."""
    for state, reason, seconds in steps:
        log.observe(station, state, reason)
        while seconds > _SAMPLE_SECONDS:
            clock.advance(_SAMPLE_SECONDS)
            seconds -= _SAMPLE_SECONDS
            log.observe(station, state, reason)
        clock.advance(seconds)


def test_nothing_is_counted_or_written_before_the_first_take(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(DOWN, "Headset: disconnected", 600), (READY, None, 300)])
    report = log.report("Bench A")
    assert (report["state"], report["reason"], report["since"]) == (READY, None, DAY_START + 600)
    assert (report["active_seconds"], report["down_seconds"], report["blockers"]) == (0.0, 0.0, [])
    assert (report["station"], report["date"], report["notes"], report["as_of"]) == (
        "Bench A",
        "2026-03-09",
        "",
        clock.now,
    )
    # A backend that is merely running leaves no file behind.
    assert not (tmp_path / "station_status").exists()


def test_active_and_down_time_are_counted_between_the_first_and_last_take(tmp_path):
    log, clock = _log(tmp_path)
    _run(
        log,
        clock,
        [
            (DOWN, "Headset: disconnected", 1200),  # before the first take: not counted
            (READY, None, 60),
            (RECORDING, None, 120),
            (READY, None, 300),
            (DOWN, "Left wrist: cam_left: no recent frame", 400),
            (DOWN, "Left wrist: cam_left: unplugged", 320),  # the cause shifted; still one blocker
            (READY, None, 100),
            (RECORDING, None, 60),
            (READY, None, 3000),  # after the last take: not counted
            (DOWN, "Headset: disconnected", 500),
        ],
    )
    report = log.report("Bench A")
    first_take = DAY_START + 1260
    assert (report["active_seconds"], report["down_seconds"]) == (120 + 300 + 100 + 60, 720.0)
    assert report["blockers"] == [
        {
            "reason": "Left wrist: cam_left: no recent frame",
            "start": first_take + 420,
            "end": first_take + 1140,
            "seconds": 720.0,
        }
    ]
    # What it is doing now is reported whatever the window is.
    assert (report["state"], report["reason"]) == (DOWN, "Headset: disconnected")
    stored = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert [segment["state"] for segment in stored] == [
        DOWN,
        READY,
        RECORDING,
        READY,
        DOWN,
        READY,
        RECORDING,
        READY,
        DOWN,
    ]
    assert stored[-1]["end"] is None and stored[4]["reason"] == "Left wrist: cam_left: no recent frame"


def test_a_take_being_recorded_extends_the_window_to_now(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 30), (DOWN, "Capture session: stopped", 90), (RECORDING, None, 45)])
    report = log.report("Bench A")
    assert (report["state"], report["active_seconds"], report["down_seconds"]) == (RECORDING, 75.0, 90.0)
    assert [blocker["seconds"] for blocker in report["blockers"]] == [90.0]


def test_the_current_reason_is_live_while_a_blocker_keeps_the_one_it_began_with(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 10), (DOWN, "Headset: disconnected", 20)])
    log.observe("Bench A", DOWN, "Free disk space: 3.0 GB free; 10 GB needed")
    assert log.report("Bench A")["reason"] == "Free disk space: 3.0 GB free; 10 GB needed"
    assert _day_file(tmp_path)["stations"]["Bench A"]["segments"][-1]["reason"] == "Headset: disconnected"


def test_the_daily_log_is_rewritten_after_each_take_and_keeps_measured_recorded_time(tmp_path):
    log, clock = _log(tmp_path)
    _run(
        log,
        clock,
        [(RECORDING, None, 120), (DOWN, "Headset: disconnected", 60), (RECORDING, None, 30), (READY, None, 5)],
    )
    path = tmp_path / "station_status" / "bench_a_2026-03-09_activity.json"
    assert json.loads(path.read_text()) == {
        "station": "Bench A",
        "date": "2026-03-09",
        "active_station_time": "0h 02m 30s",
        "active_station_time_seconds": 150.0,
        "inactive_station_time": "0h 01m 00s",
        "inactive_station_time_seconds": 60.0,
        "blockers": 1,
        "data_hours": None,
        "data_seconds": None,
        "valid_hours": None,
        "valid_seconds": None,
    }
    assert log.write_log("Bench A", data_seconds=140.0, valid_seconds=120.0) == path.name
    _run(log, clock, [(RECORDING, None, 10), (READY, None, 1)])
    after = json.loads(path.read_text())
    assert (after["data_seconds"], after["valid_seconds"], after["valid_hours"]) == (
        140.0,
        120.0,
        round(120 / 3600, 4),
    )
    assert after["active_station_time_seconds"] == 165.0


@pytest.mark.parametrize("content", ["not json", "[1]", '{"data_seconds": true, "valid_seconds": "long"}'])
def test_an_unusable_previous_log_is_rewritten_without_recorded_time(tmp_path, content):
    log, clock = _log(tmp_path)
    (tmp_path / "station_status").mkdir()
    (tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").write_text(content)
    _run(log, clock, [(RECORDING, None, 10), (READY, None, 1)])
    written = json.loads((tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").read_text())
    assert (written["data_seconds"], written["valid_seconds"]) == (None, None)


def test_only_one_station_is_in_use_so_observing_another_ends_the_first(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 60)])
    _run(log, clock, [(RECORDING, None, 20), (READY, None, 600)], station="Cart B")
    bench = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert bench[-1]["end"] == DAY_START + 90  # closed when Cart B was first observed
    assert log.report("Bench A")["active_seconds"] == 30.0 and log.report("Cart B")["active_seconds"] == 20.0


def test_time_across_midnight_is_counted_on_both_days_and_the_state_carries_over(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 50, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 300), (READY, None, 600)])  # ends at 00:05 the next day
    report = log.report("Bench A")
    midnight = datetime.datetime(2026, 3, 10).timestamp()
    assert (report["date"], report["state"], report["since"]) == ("2026-03-10", READY, midnight)
    assert report["active_seconds"] == 0.0  # the new day has no take yet
    yesterday = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert yesterday[-1] == {"state": READY, "reason": None, "start": late.now - 600, "end": midnight}
    assert window_totals(yesterday, midnight)["active_seconds"] == 300.0
    # A station that was not open at midnight is not carried into the new day.
    assert list(log._payload["stations"]) == ["Bench A"]


def test_a_station_with_nothing_open_at_midnight_is_left_behind(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 50, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 60)])
    _run(log, clock, [(READY, None, 900)], station="Cart B")  # Bench A is closed; Cart B crosses midnight
    assert list(log._payload["stations"]) == ["Cart B"] if log.report("Cart B") else None


def test_a_restart_counts_the_time_nobody_was_watching_as_down(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 40)])
    log.observe("Bench A", READY, None)  # the last sample before the backend stops
    clock.advance(900)
    restarted, _ = _log(tmp_path, clock)
    _run(restarted, clock, [(READY, None, 30), (RECORDING, None, 20), (READY, None, 1)])
    segments = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert [(s["state"], s["reason"]) for s in segments] == [
        (RECORDING, None),
        (READY, None),
        (DOWN, BACKEND_NOT_RUNNING),
        (READY, None),
        (RECORDING, None),
        (READY, None),
    ]
    report = restarted.report("Bench A")
    assert (report["down_seconds"], report["blockers"][0]["reason"]) == (900.0, BACKEND_NOT_RUNNING)


def test_a_restart_while_down_is_one_blocker_not_two(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (DOWN, "Headset: disconnected", 100)])
    log.observe("Bench A", DOWN, "Headset: disconnected")
    clock.advance(500)
    restarted, _ = _log(tmp_path, clock)
    _run(restarted, clock, [(DOWN, "Headset: disconnected", 50), (RECORDING, None, 10), (READY, None, 1)])
    report = restarted.report("Bench A")
    assert report["down_seconds"] == 650.0
    assert [(b["reason"], b["seconds"]) for b in report["blockers"]] == [("Headset: disconnected", 650.0)]


def test_a_quick_restart_is_not_an_outage(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 20)])
    restarted, _ = _log(tmp_path, clock)
    restarted.observe("Bench A", READY, None)
    assert [s["state"] for s in restarted._payload["stations"]["Bench A"]["segments"]] == [RECORDING, READY]


def test_a_restart_with_no_recorded_sample_time_falls_back_to_the_segments_start(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)])
    path = tmp_path / "station_status" / "2026-03-09_timeline.json"
    stored = json.loads(path.read_text())
    stored["last_sample"] = None
    path.write_text(json.dumps(stored))
    clock.advance(600)
    restarted, _ = _log(tmp_path, clock)
    restarted.observe("Bench A", READY, None)
    segments = restarted._payload["stations"]["Bench A"]["segments"]
    assert segments[1]["end"] == segments[1]["start"] and segments[2]["reason"] == BACKEND_NOT_RUNNING


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[1]",
        '{"stations": []}',
        '{"stations": {"Bench A": 5}}',
        '{"stations": {"Bench A": {"segments": [{"state": "ready"}]}}}',
    ],
)
def test_an_unusable_timeline_file_is_ignored(tmp_path, content):
    (tmp_path / "station_status").mkdir()
    (tmp_path / "station_status" / "2026-03-09_timeline.json").write_text(content)
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 10), (READY, None, 1)])
    assert log.report("Bench A")["active_seconds"] == 10.0


def test_a_stored_record_without_notes_or_aliases_is_still_usable(tmp_path):
    (tmp_path / "station_status").mkdir()
    segment = {"state": RECORDING, "reason": None, "start": DAY_START - 100, "end": DAY_START - 40}
    (tmp_path / "station_status" / "2026-03-09_timeline.json").write_text(
        json.dumps({"last_sample": DAY_START - 40, "stations": {"Bench A": {"segments": [segment]}}})
    )
    log, _ = _log(tmp_path)
    assert (log.report("Bench A")["active_seconds"], log.report("Bench A")["notes"]) == (60.0, "")
    assert log.names("Bench A") == frozenset({"Bench A"})


def test_notes_persist_and_are_bounded(tmp_path):
    log, clock = _log(tmp_path)
    assert log.set_notes("Bench A", "Left glove battery swapped.")["notes"] == "Left glove battery swapped."
    reopened, _ = _log(tmp_path, clock)
    assert reopened.report("Bench A")["notes"] == "Left glove battery swapped."
    for bad in (7, None, "x" * (MAX_NOTES_CHARS + 1)):
        with pytest.raises(UptimeError, match=f"at most {MAX_NOTES_CHARS} characters"):
            log.set_notes("Bench A", bad)


def test_a_renamed_station_keeps_its_day_under_the_new_name(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 30)], station="Bnech A")
    log.set_notes("Bnech A", "typo in the name")
    log.rename("Bnech A", "Bench A")
    _run(log, clock, [(RECORDING, None, 40), (READY, None, 1)])
    report = log.report("Bench A")
    assert (report["active_seconds"], report["notes"]) == (60 + 30 + 40, "typo in the name")
    assert log.names("Bench A") == frozenset({"Bench A", "Bnech A"})
    assert "Bnech A" not in _day_file(tmp_path)["stations"]
    log.rename("Never seen", "Whatever")  # nothing to carry
    assert "Whatever" not in log._payload["stations"]


def test_a_rename_onto_a_name_used_earlier_today_joins_the_two_histories(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 10)])
    log.set_notes("Bench A", "morning")
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 10)], station="Bench Z")
    log.rename("Bench Z", "Bench A")
    record = log._payload["stations"]["Bench A"]
    assert [s["state"] for s in record["segments"]] == [RECORDING, READY, RECORDING, READY]
    assert (record["notes"], record["aliases"]) == ("morning", ["Bench Z"])


def test_the_day_stops_being_logged_once_its_bound_is_reached(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(station_uptime, "_MAX_SEGMENTS", 3)
    log, clock = _log(tmp_path)
    _run(
        log,
        clock,
        [(RECORDING, None, 1), (READY, None, 1), (RECORDING, None, 1), (READY, None, 1), (DOWN, "x", 1)],
    )
    assert len(log._payload["stations"]["Bench A"]["segments"]) == 3
    assert caplog.text.count("timeline is full") == 1


def test_a_silence_while_the_backend_stays_up_is_down_time_and_is_bridged_once(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 30)])
    log.observe("Bench A", READY, None)  # the last sample before the host is suspended
    clock.advance(3600)
    # Reading the day first must not bridge the silence a second time when the sampler comes back.
    assert log.report("Bench A")["reason"] == BACKEND_NOT_RUNNING
    assert log.report("Bench A")["since"] == DAY_START + 90
    _run(log, clock, [(READY, None, 10), (RECORDING, None, 60), (READY, None, 1)])
    segments = log._payload["stations"]["Bench A"]["segments"]
    assert [(s["state"], s["reason"]) for s in segments] == [
        (RECORDING, None),
        (READY, None),
        (DOWN, BACKEND_NOT_RUNNING),
        (READY, None),
        (RECORDING, None),
        (READY, None),
    ]
    report = log.report("Bench A")
    assert (report["active_seconds"], report["down_seconds"]) == (60 + 30 + 10 + 60, 3600.0)
    assert [(b["reason"], b["seconds"]) for b in report["blockers"]] == [(BACKEND_NOT_RUNNING, 3600.0)]


def test_a_silence_that_crosses_midnight_is_down_on_the_day_it_began(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 0, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 40)])
    log.observe("Bench A", READY, None)
    asleep = clock.now
    clock.advance(2 * 3600)  # nothing sampled until 01:01 the next day
    log.observe("Bench A", READY, None)
    midnight = datetime.datetime(2026, 3, 10).timestamp()
    yesterday = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert yesterday[-2:] == [
        {"state": READY, "reason": None, "start": asleep - 40, "end": asleep},
        {"state": DOWN, "reason": BACKEND_NOT_RUNNING, "start": asleep, "end": midnight},
    ]
    # The new day carries the outage from midnight, once, and ends it when the station is seen again.
    assert log._payload["stations"]["Bench A"]["segments"] == [
        {"state": DOWN, "reason": BACKEND_NOT_RUNNING, "start": midnight, "end": clock.now},
        {"state": READY, "reason": None, "start": clock.now, "end": None},
    ]


def test_a_silence_across_midnight_read_before_the_sampler_returns_is_bridged_once(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 0, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 40)])
    log.observe("Bench A", READY, None)
    clock.advance(2 * 3600)
    # The page reads the new day before the sampler is back; that read rolls the day over.
    assert log.report("Bench A")["reason"] == BACKEND_NOT_RUNNING
    log.observe("Bench A", READY, None)
    midnight = datetime.datetime(2026, 3, 10).timestamp()
    assert log._payload["stations"]["Bench A"]["segments"] == [
        {"state": DOWN, "reason": BACKEND_NOT_RUNNING, "start": midnight, "end": clock.now},
        {"state": READY, "reason": None, "start": clock.now, "end": None},
    ]


def test_a_take_that_crosses_midnight_brings_the_old_days_log_up_to_midnight(tmp_path, monkeypatch, caplog):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 30, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 600), (READY, None, 900), (RECORDING, None, 900), (READY, None, 1)])
    old_day = json.loads((tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").read_text())
    assert old_day["active_station_time_seconds"] == 1800.0  # 23:30 to midnight, not only to 23:40
    new_day = json.loads((tmp_path / "station_status" / "bench_a_2026-03-10_activity.json").read_text())
    assert new_day["active_station_time_seconds"] == 600.0
    # A take cut short by a silence that lasts past midnight never ends with a sample; its day is closed too.
    cut, clock = _log(tmp_path / "cut", _Clock(datetime.datetime(2026, 3, 9, 23, 0, 0).timestamp()))
    cut.observe("Bench A", RECORDING, None)
    clock.advance(60)
    cut.observe("Bench A", RECORDING, None)
    clock.advance(2 * 3600)
    cut.observe("Bench A", READY, None)
    closed = json.loads((tmp_path / "cut" / "station_status" / "bench_a_2026-03-09_activity.json").read_text())
    assert closed["active_station_time_seconds"] == 60.0
    # A log that cannot be written must not stop the new day from starting.
    again, clock = _log(tmp_path / "again", _Clock(datetime.datetime(2026, 3, 9, 23, 59, 0).timestamp()))
    again.observe("Bench A", RECORDING, None)
    write_log = UptimeLog._write_log

    def refuse_the_old_day(self, station, payload, now, measured):
        if payload["date"] == "2026-03-09":
            raise OSError("disk full")
        return write_log(self, station, payload, now, measured)

    monkeypatch.setattr(UptimeLog, "_write_log", refuse_the_old_day)
    clock.advance(80)
    again.observe("Bench A", RECORDING, None)
    assert again.report("Bench A")["date"] == "2026-03-10" and "up to midnight" in caplog.text


def test_a_renamed_station_takes_its_daily_log_and_saved_report_with_it(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)], station="Bay 1")
    log.write_log("Bay 1", data_seconds=50.0, valid_seconds=40.0)
    log.save_report("Bay 1", {"report": {}})
    log.rename("Bay 1", "Bay 2")
    status = tmp_path / "station_status"
    assert sorted(path.name for path in status.iterdir()) == [
        "2026-03-09_timeline.json",
        "bay_2_2026-03-09_activity.json",
        "bay_2_2026-03-09_report.json",
    ]
    carried = json.loads((status / "bay_2_2026-03-09_activity.json").read_text())
    assert (carried["station"], carried["data_seconds"], carried["active_station_time_seconds"]) == (
        "Bay 2",
        50.0,
        60.0,
    )
    assert log.report_saved("Bay 2") is True
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 1)], station="Bay 2")
    after = json.loads((status / "bay_2_2026-03-09_activity.json").read_text())
    assert (after["data_seconds"], after["valid_seconds"]) == (50.0, 40.0)


def test_a_rename_onto_a_name_with_files_of_its_own_keeps_those_and_drops_the_old_ones(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)], station="Bay 2")
    log.write_log("Bay 2", data_seconds=50.0, valid_seconds=40.0)
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 5)], station="Bay 1")
    log.save_report("Bay 1", {"report": {}})
    log.rename("Bay 1", "Bay 2")
    status = tmp_path / "station_status"
    assert sorted(path.name for path in status.iterdir()) == [
        "2026-03-09_timeline.json",
        "bay_2_2026-03-09_activity.json",
        "bay_2_2026-03-09_report.json",
    ]
    joined = json.loads((status / "bay_2_2026-03-09_activity.json").read_text())
    assert (joined["data_seconds"], joined["active_station_time_seconds"]) == (50.0, 60 + 5 + 30)


def test_a_rename_that_keeps_the_file_name_leaves_the_files_alone(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)], station="bench a")
    log.save_report("bench a", {"report": {}})
    log.rename("bench a", "Bench A")
    assert log.report_saved("Bench A") is True
    assert (tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").is_file()


def test_files_that_cannot_be_carried_are_logged_and_the_timeline_still_moves(tmp_path, monkeypatch, caplog):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)], station="Bay 1")

    replace = station_uptime.os.replace

    def refuse_the_daily_log(old, new):
        if str(old).endswith("bay_1_2026-03-09_activity.json"):
            raise OSError("permission denied")
        replace(old, new)

    monkeypatch.setattr(station_uptime.os, "replace", refuse_the_daily_log)
    log.rename("Bay 1", "Bay 2")
    assert "Could not carry bay_1_2026-03-09_activity.json" in caplog.text
    assert list(log._payload["stations"]) == ["Bay 2"]


def test_a_name_given_to_another_station_stops_being_the_renamed_stations_alias(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)], station="Bay 2")
    log.rename("Bay 2", "Bay 1")
    assert log.names("Bay 1") == frozenset({"Bay 1", "Bay 2"})
    # Another station is now called Bay 2 and records under that name.
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 5)], station="Bay 2")
    assert log.names("Bay 1") == frozenset({"Bay 1"}) and log.names("Bay 2") == frozenset({"Bay 2"})
    # The same holds when a station is renamed onto the freed name rather than first seen under it.
    _run(log, clock, [(RECORDING, None, 30), (READY, None, 5)], station="Bay 3")
    log.rename("Bay 3", "Bay 4")
    log.rename("Bay 1", "Bay 3")
    assert log.names("Bay 4") == frozenset({"Bay 4"})
    assert log.names("Bay 3") == frozenset({"Bay 3", "Bay 1"})


def test_notes_cleared_before_the_first_take_stay_cleared_after_a_restart(tmp_path):
    log, clock = _log(tmp_path)
    log.observe("Bench A", READY, None)
    log.set_notes("Bench A", "gloves missing")
    assert log.set_notes("Bench A", "")["notes"] == ""
    reopened, _ = _log(tmp_path, clock)
    assert reopened.report("Bench A")["notes"] == ""


def test_a_full_day_is_not_rewritten_at_every_sample_and_says_it_is_full(tmp_path, monkeypatch):
    monkeypatch.setattr(station_uptime, "_MAX_SEGMENTS", 3)
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 1), (READY, None, 1), (RECORDING, None, 1)])
    writes: list[str] = []
    write = station_uptime.write_json_atomic
    monkeypatch.setattr(
        station_uptime, "write_json_atomic", lambda path, data: (writes.append(path.name), write(path, data))
    )
    for _ in range(10):
        log.observe("Bench A", READY, None)
        clock.advance(1)
    assert writes == []
    segments = log._payload["stations"]["Bench A"]["segments"]
    assert len(segments) == 3 and segments[-1]["end"] is None
    report = log.report("Bench A")
    assert (report["state"], report["reason"]) == (DOWN, LOG_FULL)
    # A silence adds nothing to a full day either.
    clock.advance(3600)
    log.observe("Bench A", READY, None)
    assert len(log._payload["stations"]["Bench A"]["segments"]) == 3


def test_a_blocker_that_has_not_ended_has_no_end_and_runs_to_now():
    segments = [
        {"state": RECORDING, "reason": None, "start": 0.0, "end": 10.0},
        {"state": DOWN, "reason": "Headset: disconnected", "start": 10.0, "end": None},
        {"state": RECORDING, "reason": None, "start": 20.0, "end": None},
    ]
    assert window_totals(segments, 50.0)["blockers"] == [
        {"reason": "Headset: disconnected", "start": 10.0, "end": None, "seconds": 40.0}
    ]


def test_a_station_took_today_once_a_take_was_logged_and_not_on_the_next_day(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 0, 0).timestamp())
    log, clock = _log(tmp_path, late)
    log.observe("Bench A", READY, None)
    assert log.took_today("Bench A") is False and log.took_today("Cart B") is False
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 5)])
    assert log.took_today("Bench A") is True
    log.rename("Bench A", "Bench B")
    assert log.took_today("Bench B") is True and log.took_today("Bench A") is False
    _run(log, clock, [(READY, None, 3600)], station="Bench B")  # past midnight
    assert log.took_today("Bench B") is False


def test_a_saved_report_sits_beside_the_daily_log(tmp_path):
    log, _ = _log(tmp_path)
    assert log.report_saved("Bench A") is False
    assert log.save_report("Bench A", {"report": {"totals": {}}}) == "bench_a_2026-03-09_report.json"
    assert log.report_saved("Bench A") is True and log.report_saved("Cart B") is False
    assert json.loads((tmp_path / "station_status" / "bench_a_2026-03-09_report.json").read_text()) == {
        "report": {"totals": {}}
    }


def test_an_unchanged_state_is_still_written_now_and_then(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 10), (READY, None, 5)])
    log.observe("Bench A", READY, None)
    assert _day_file(tmp_path)["last_sample"] == DAY_START + 10  # too soon to write again
    clock.advance(40)
    log.observe("Bench A", READY, None)
    assert _day_file(tmp_path)["last_sample"] == DAY_START + 55


def test_a_station_that_was_never_observed_says_so_rather_than_ready(tmp_path):
    log, clock = _log(tmp_path)
    report = log.report("Bench A")
    assert (report["state"], report["reason"], report["since"]) == (DOWN, "not observed yet", clock.now)


# --- the sampler -------------------------------------------------------------------------------------------------


def test_a_state_must_hold_before_it_is_logged_but_a_take_is_logged_at_once():
    clock = _Clock(0.0)
    sampler = UptimeSampler(settle_seconds=6.0, monotonic=clock)
    assert sampler.settle("Bench A", READY, None) == ("Bench A", READY, None)  # the first is logged as it is
    clock.advance(2)
    assert sampler.settle("Bench A", DOWN, "Headset: disconnected") == ("Bench A", READY, None)  # a flicker
    clock.advance(2)
    assert sampler.settle("Bench A", READY, None) == ("Bench A", READY, None)
    clock.advance(2)
    assert sampler.settle("Bench A", DOWN, "Headset: disconnected") == ("Bench A", READY, None)
    clock.advance(6)
    assert sampler.settle("Bench A", DOWN, "Headset: disconnected") == ("Bench A", DOWN, "Headset: disconnected")
    clock.advance(2)
    # While a recovery settles the station is still logged down, with the reason it has now.
    assert sampler.settle("Bench A", READY, None) == ("Bench A", DOWN, None)
    assert sampler.settle("Bench A", RECORDING, None) == ("Bench A", RECORDING, None)  # a take starts: at once
    assert sampler.settle("Bench A", DOWN, "Capture session: stopped") == (
        "Bench A",
        DOWN,
        "Capture session: stopped",
    )
    assert sampler.settle("Cart B", READY, None) == ("Cart B", READY, None)  # another station: at once


# --- a stop that lasts past midnight, and writes that fail -------------------------------------------------------


def _activity(tmp_path, day: str) -> dict:
    return json.loads((tmp_path / "station_status" / f"bench_a_{day}_activity.json").read_text())


def test_a_backend_restarted_after_midnight_finishes_the_day_it_stopped_in(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 40, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 120), (READY, None, 60), (RECORDING, None, 300)])
    log.observe("Bench A", RECORDING, None)  # 23:48, the last sample before the backend stops mid-take
    stopped = clock.now
    midnight = datetime.datetime(2026, 3, 10).timestamp()
    clock.now = datetime.datetime(2026, 3, 10, 8, 0, 0).timestamp()
    restarted, _ = _log(tmp_path, clock)
    restarted.observe("Bench A", READY, None)
    # The old day's take ends where it was last seen, and the rest of that day is the backend not running.
    segments = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert [(s["state"], s["end"]) for s in segments[-2:]] == [(RECORDING, stopped), (DOWN, midnight)]
    assert segments[-1]["reason"] == BACKEND_NOT_RUNNING
    # Its daily log, which only a take's end or a rollover writes, covers the take that never ended.
    assert _activity(tmp_path, "2026-03-09")["active_station_time_seconds"] == 480.0
    # The new day begins with the outage carried over from midnight, then what is observed now.
    today = restarted._payload["stations"]["Bench A"]["segments"]
    assert restarted._payload["date"] == "2026-03-10"
    assert [(s["state"], s["reason"], s["start"]) for s in today] == [
        (DOWN, BACKEND_NOT_RUNNING, midnight),
        (READY, None, clock.now),
    ]
    assert restarted.took_today("Bench A") is False


def test_a_day_left_open_for_several_days_ends_at_its_own_midnight(tmp_path):
    log, clock = _log(tmp_path, _Clock(datetime.datetime(2026, 3, 9, 23, 0, 0).timestamp()))
    _run(log, clock, [(RECORDING, None, 60), (READY, None, 30)])
    log.observe("Bench A", READY, None)
    clock.now = datetime.datetime(2026, 3, 12, 8, 0, 0).timestamp()
    restarted, _ = _log(tmp_path, clock)
    report = restarted.report("Bench A")
    assert (report["date"], report["since"]) == ("2026-03-12", datetime.datetime(2026, 3, 12).timestamp())
    last = _day_file(tmp_path)["stations"]["Bench A"]["segments"][-1]
    assert (last["state"], last["end"]) == (DOWN, datetime.datetime(2026, 3, 10).timestamp())


@pytest.fixture
def clocks_go_back(monkeypatch):
    """A time zone whose 2026-11-01 lasts 25 hours; the process's own zone is restored afterwards."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_a_day_the_clocks_go_back_on_still_ends_at_its_midnight(tmp_path, clocks_go_back):
    log, clock = _log(tmp_path, _Clock(datetime.datetime(2026, 11, 1, 23, 58, 0).timestamp()))
    _run(log, clock, [(RECORDING, None, 60)])
    log.observe("Bench A", READY, None)
    midnight = datetime.datetime(2026, 11, 2).timestamp()
    clock.now = midnight + 10
    assert log.report("Bench A")["date"] == "2026-11-02"
    # 24 hours after the day began is 23:00 on a 25-hour day; the hour after it belongs to the day too.
    assert _day_file(tmp_path, "2026-11-01")["stations"]["Bench A"]["segments"][-1]["end"] == midnight


def test_a_restart_finds_nothing_to_finish_in_a_day_that_was_closed_or_already_followed(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 60)])
    _run(log, clock, [(READY, None, 30)], station="Cart B")  # Bench A's day is closed; Cart B is never written
    stored = _day_file(tmp_path)
    stored["stations"].pop("Cart B")
    (tmp_path / "station_status" / "2026-03-09_timeline.json").write_text(json.dumps(stored))
    clock.now = datetime.datetime(2026, 3, 10, 8, 0, 0).timestamp()
    restarted, _ = _log(tmp_path, clock)
    assert restarted.report("Bench A")["date"] == "2026-03-10" and restarted._payload["stations"] == {}
    assert _day_file(tmp_path) == stored
    # A day that has a file of its own was started by a run that finished the one before it.
    restarted.set_notes("Bench A", "second day")
    again, _ = _log(tmp_path, clock)
    assert again.report("Bench A")["notes"] == "second day"


def test_a_file_that_is_not_a_day_of_this_log_or_a_directory_that_cannot_be_listed_is_not_finished(
    tmp_path, monkeypatch
):
    directory = tmp_path / "station_status"
    directory.mkdir()
    (directory / "0000_timeline.json").write_text("{}")
    log, _ = _log(tmp_path)
    assert log.report("Bench A")["date"] == "2026-03-09" and log._payload["stations"] == {}
    from pathlib import Path

    def unlistable(self, pattern):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "glob", unlistable)
    other, _ = _log(tmp_path)
    assert other.report("Bench A")["date"] == "2026-03-09"


def test_a_rollover_whose_write_fails_still_carries_the_take_into_the_new_day(tmp_path, monkeypatch):
    log, clock = _log(tmp_path, _Clock(datetime.datetime(2026, 3, 9, 23, 59, 0).timestamp()))
    log.observe("Bench A", RECORDING, None)
    midnight = datetime.datetime(2026, 3, 10).timestamp()
    write, failures = UptimeLog._write, [OSError("disk full")]

    def failing_once(self, payload):
        if payload["date"] == "2026-03-09" and failures:
            raise failures.pop()
        write(self, payload)

    monkeypatch.setattr(UptimeLog, "_write", failing_once)
    clock.advance(80)
    with pytest.raises(OSError):
        log.observe("Bench A", RECORDING, None)
    clock.advance(2)
    log.observe("Bench A", RECORDING, None)
    # The take is still open on the new day, from midnight: nothing between midnight and now was lost.
    report = log.report("Bench A")
    assert (report["date"], report["state"], report["since"]) == ("2026-03-10", RECORDING, midnight)
    assert report["active_seconds"] == clock.now - midnight
    assert _day_file(tmp_path)["stations"]["Bench A"]["segments"][-1]["end"] == midnight
    assert _activity(tmp_path, "2026-03-09")["active_station_time_seconds"] == 60.0


@pytest.mark.parametrize("failing_write", [1, 2], ids=["the timeline", "the daily log"])
def test_a_daily_log_that_could_not_be_written_when_a_take_ended_is_written_at_the_next_save(
    tmp_path, monkeypatch, failing_write
):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 120)])
    write, writes = station_uptime.write_json_atomic, [0]

    def failing_once(path, data):
        writes[0] += 1
        if writes[0] == failing_write:
            raise OSError("disk full")
        write(path, data)

    monkeypatch.setattr(station_uptime, "write_json_atomic", failing_once)
    with pytest.raises(OSError):
        log.observe("Bench A", READY, None)  # the take ends; its log cannot be written
    assert not (tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").exists()
    # No further take ends. The next periodic save of the timeline makes the log good.
    clock.advance(10)
    log.observe("Bench A", READY, None)
    assert not (tmp_path / "station_status" / "bench_a_2026-03-09_activity.json").exists()
    clock.advance(30)
    log.observe("Bench A", READY, None)
    assert _activity(tmp_path, "2026-03-09")["active_station_time_seconds"] == 120.0
    assert log._owed_logs == set()


def test_a_daily_log_still_owed_follows_a_rename(tmp_path, monkeypatch):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 120)], station="Bnech A")
    write_log, failures = UptimeLog._write_log, [OSError("disk full")]

    def failing_once(self, *args):
        if failures:
            raise failures.pop()
        return write_log(self, *args)

    monkeypatch.setattr(UptimeLog, "_write_log", failing_once)
    with pytest.raises(OSError):
        log.observe("Bnech A", READY, None)
    log.rename("Bnech A", "Bench A")
    clock.advance(40)
    log.observe("Bench A", READY, None)
    assert _activity(tmp_path, "2026-03-09")["station"] == "Bench A"
    assert not (tmp_path / "station_status" / "bnech_a_2026-03-09_activity.json").exists()


# --- a clock that steps back, and a take a silence cut short -----------------------------------------------------


def test_a_clock_stepped_back_holds_the_log_at_its_last_sample(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(RECORDING, None, 120), (READY, None, 60)])
    log.observe("Bench A", READY, None)
    held = clock.now
    clock.now -= 600
    log.observe("Bench A", DOWN, "headset disconnected")
    # No segment ends before it began, and the report's time does not run backwards.
    segments = _day_file(tmp_path)["stations"]["Bench A"]["segments"]
    assert all(s["end"] is None or s["end"] >= s["start"] for s in segments)
    assert (segments[-1]["start"], log.report("Bench A")["as_of"]) == (held, held)


def test_a_clock_stepped_back_across_midnight_never_rewrites_the_day_already_closed(tmp_path):
    late = _Clock(datetime.datetime(2026, 3, 9, 23, 50, 0).timestamp())
    log, clock = _log(tmp_path, late)
    _run(log, clock, [(RECORDING, None, 300), (READY, None, 600)])  # ends at 00:05 the next day
    log.observe("Bench A", READY, None)
    closed = _day_file(tmp_path)
    clock.now = datetime.datetime(2026, 3, 9, 23, 55, 0).timestamp()
    log.observe("Bench A", DOWN, "headset disconnected")
    assert log.report("Bench A")["date"] == "2026-03-10"
    assert _day_file(tmp_path) == closed


def test_a_take_cut_short_by_a_silence_reaches_the_daily_log(tmp_path):
    log, clock = _log(tmp_path)
    _run(log, clock, [(READY, None, 60), (RECORDING, None, 120)])
    log.observe("Bench A", RECORDING, None)
    # The backend's host sleeps mid-take; the take ends where it was last seen.
    clock.advance(1800)
    log.observe("Bench A", READY, None)
    assert _activity(tmp_path, "2026-03-09")["active_station_time_seconds"] == 120.0
