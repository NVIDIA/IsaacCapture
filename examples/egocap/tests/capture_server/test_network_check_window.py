# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capture_server import protocol as p
from capture_server.network_check import NetworkCheckPolicy, StreamWindow

SEC = 1_000_000_000
FPS = 60.0


def _stream(fps: float | None = FPS) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=16,
        name="Head left",
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        width=6,
        height=4,
        fps=fps,
    )


def _window(window_s: float = 30.0, *, start_ns: int = 0, fps: float | None = FPS, **policy) -> StreamWindow:
    return StreamWindow(
        _stream(fps), NetworkCheckPolicy(window_s=window_s, **policy), start_ns, start_ns + int(window_s * SEC)
    )


def _feed(
    window: StreamWindow,
    *,
    fps: float = FPS,
    begin_s: float = 0.0,
    end_s: float | None = None,
    age_s: float = 0.05,
    first_id: int = 0,
) -> None:
    """Healthy frames at ``fps`` from ``begin_s`` to ``end_s`` (default the window's end), each ``age_s`` old."""
    end_s = window.elapsed_s if end_s is None else end_s
    count = int(round((end_s - begin_s) * fps))
    for i in range(count):
        receive_s = begin_s + i / fps
        window.add_frame(
            frame_id=first_id + i,
            capture_s=window.start_ns / SEC + receive_s - age_s,
            receive_ns=window.start_ns + int(receive_s * SEC),
            age_s=age_s,
            mapping_ok=True,
        )


def _counters(overflow=0, changes=0, missing=0) -> SimpleNamespace:
    return SimpleNamespace(queue_overflow=overflow, publication_changes=changes, missing_frames=missing)


def _finished(window: StreamWindow, **counters) -> p.NetworkCheckCamera:
    window.finish(_counters(**counters))
    return window.result()


def test_healthy_stream_over_the_window_passes():
    window = _window()
    _feed(window)
    camera = _finished(window)
    assert camera.passed is True
    assert camera.reasons == ()
    assert camera.usable_frames == 30 * 60


def test_frames_stamped_ten_seconds_in_the_future_fail():
    window = _window()
    _feed(window, age_s=-10.0)
    camera = _finished(window)
    assert camera.passed is False
    assert any("implausible age" in r for r in camera.reasons)
    assert camera.usable_frames == 0


def test_frames_older_than_the_plausible_limit_fail():
    window = _window()
    _feed(window, age_s=5.0)
    camera = _finished(window)
    assert camera.passed is False
    assert any("implausible age" in r for r in camera.reasons)


def test_silence_for_the_final_800_ms_fails():
    window = _window()
    _feed(window, end_s=29.2)
    camera = _finished(window)
    assert camera.passed is False
    assert any("last" in r and "of the window" in r for r in camera.reasons)
    assert camera.end_silence_ms == pytest.approx(800, abs=20)


def test_arrival_stall_followed_by_a_catch_up_burst_with_continuous_capture_times_fails():
    window = _window(window_s=10.0)
    for i in range(600):
        capture_s = i / FPS
        # Frames 300..329 are held for 500 ms and arrive as a burst; capture times stay evenly spaced.
        if 300 <= i < 330:
            receive_s = 5.0 + 0.5 + (i - 300) * 0.001
        elif i >= 330:
            receive_s = max(capture_s, 5.5 + 30 * 0.001 + (i - 330) * 0.001)
        else:
            receive_s = capture_s
        window.add_frame(
            frame_id=i, capture_s=capture_s, receive_ns=int(receive_s * SEC), age_s=0.05, mapping_ok=True
        )
    camera = _finished(window)
    assert camera.passed is False
    assert any("silence between arrivals" in r for r in camera.reasons)
    assert camera.max_capture_gap_ms == pytest.approx(1000 / FPS, abs=1)


def test_first_frame_one_second_into_the_window_is_start_silence():
    window = _window()
    _feed(window, begin_s=1.0)
    camera = _finished(window)
    assert camera.passed is False
    assert any("first" in r and "of the window" in r for r in camera.reasons)
    assert camera.start_silence_ms == pytest.approx(1000, abs=1)


def test_frames_outside_the_window_are_ignored():
    window = _window(start_ns=100 * SEC)
    for receive_ns in (window.start_ns - 1, window.end_ns + 1, 0):
        window.add_frame(frame_id=None, capture_s=1.0, receive_ns=receive_ns, age_s=0.0, mapping_ok=True)
    assert (window.unique, window.valid) == (0, 0)
    window.add_frame(frame_id=None, capture_s=1.0, receive_ns=window.start_ns, age_s=0.0, mapping_ok=True)
    window.add_frame(frame_id=None, capture_s=1.1, receive_ns=window.end_ns, age_s=0.0, mapping_ok=True)
    assert window.valid == 2


def test_backlog_before_the_window_does_not_shift_the_per_second_buckets():
    window = _window(start_ns=50 * SEC)
    # Backlog: a second's worth of frames received before the window began, then a full healthy window.
    for i in range(60):
        window.add_frame(
            frame_id=i,
            capture_s=i / FPS,
            receive_ns=window.start_ns - SEC + i * 16_000_000,
            age_s=0.05,
            mapping_ok=True,
        )
    _feed(window, first_id=1000)
    assert window._second_counts() == [60] * 30
    assert _finished(window).passed is True


def test_per_second_buckets_count_from_the_window_start_not_the_first_frame():
    window = _window(window_s=10.0)
    # Three frames in the last tenth of second 0, three in the first tenth of second 1.
    for receive_s in (0.91, 0.93, 0.95, 1.01, 1.03, 1.05):
        window.add_frame(
            frame_id=None, capture_s=receive_s, receive_ns=int(receive_s * SEC), age_s=0.05, mapping_ok=True
        )
    assert window._second_counts() == [3, 3, 0, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("repeat_id", [5, 3])
def test_duplicate_or_backwards_frame_ids_are_not_deliveries(repeat_id):
    window = _window(window_s=10.0)
    for i, frame_id in enumerate([1, 2, 3, 4, 5, repeat_id]):
        window.add_frame(
            frame_id=frame_id, capture_s=i / FPS, receive_ns=i * 10_000_000, age_s=0.0, mapping_ok=True
        )
    assert window.unique == 5
    assert window.valid == 5


def test_frames_without_a_capture_time_never_count_as_usable():
    window = _window()
    for i in range(1800):
        window.add_frame(frame_id=i, capture_s=None, receive_ns=i * 16_000_000, age_s=None, mapping_ok=True)
    camera = _finished(window)
    assert window.valid == 0 and window.missing_timestamp == 1800
    assert camera.passed is False
    assert any("without a capture time" in r for r in camera.reasons)


def test_frames_placed_with_an_unusable_mapping_never_count_as_usable():
    window = _window()
    for i in range(1800):
        window.add_frame(frame_id=i, capture_s=i / FPS, receive_ns=i * 16_000_000, age_s=0.05, mapping_ok=False)
    camera = _finished(window)
    assert window.valid == 0 and window.mapping_problems == 1800
    assert camera.passed is False
    assert any("clock mapping" in r for r in camera.reasons)


def test_a_lost_track_fails():
    window = _window()
    _feed(window)
    window.lost = True
    camera = _finished(window)
    assert camera.passed is False
    assert any("track was lost" in r for r in camera.reasons)


def test_receiver_overflow_during_the_window_fails():
    window = _window()
    _feed(window)
    camera = _finished(window, overflow=3)
    assert camera.passed is False
    assert any("dropped 3 frame" in r for r in camera.reasons)


def test_publication_change_during_the_window_fails():
    window = _window()
    _feed(window)
    camera = _finished(window, changes=1)
    assert camera.passed is False
    assert any("publication was replaced" in r for r in camera.reasons)


def test_receiver_counters_from_before_the_window_are_not_attributed_to_it():
    window = _window()
    window.overflow_at_start, window.publication_changes_at_start, window.missing_at_start = 4, 2, 7
    _feed(window)
    camera = _finished(window, overflow=4, changes=2, missing=7)
    assert camera.passed is True
    assert camera.missing_frames == 0


def test_missing_frames_is_unknown_when_no_frame_ids_were_seen():
    window = _window()
    for i in range(1800):
        window.add_frame(frame_id=None, capture_s=i / FPS, receive_ns=i * 16_666_666, age_s=0.05, mapping_ok=True)
    camera = _finished(window, missing=99)
    assert camera.missing_frames is None


def test_missing_frames_is_a_number_possibly_zero_when_ids_were_seen():
    window = _window()
    _feed(window)
    assert _finished(window).missing_frames == 0
    window = _window()
    _feed(window)
    assert _finished(window, missing=12).missing_frames == 12


def test_no_frames_at_all_fails_and_reports_unknowns_not_a_healthy_zero_age():
    window = _window()
    camera = _finished(window)
    assert camera.passed is False
    assert camera.usable_frames == 0
    assert camera.delivered_fps == 0.0
    assert camera.missing_frames is None
    assert camera.max_capture_gap_ms is None
    assert camera.max_arrival_gap_ms is None
    assert (camera.age_median_ms, camera.age_max_ms, camera.age_growth_ms) == (None, None, None)
    assert camera.start_silence_ms == pytest.approx(30_000)
    assert camera.end_silence_ms == pytest.approx(30_000)
    assert "No usable frames received in 30 of 30 one-second samples" in camera.reasons
    assert (
        "Low frame count in 30 of 30 one-second samples: fewer than 52 usable frames per sample "
        "(lowest count: 0 frames)" in camera.reasons
    )


def test_a_few_low_seconds_are_tolerated_up_to_max_low_windows():
    def with_low_seconds(count: int, **policy) -> p.NetworkCheckCamera:
        window = _window(window_s=10.0, **policy)
        frame_id = 0
        for second in range(10):
            rate = 40 if second < count else 60
            for i in range(rate):
                receive_s = second + i / rate
                window.add_frame(
                    frame_id=frame_id,
                    capture_s=receive_s,
                    receive_ns=int(receive_s * SEC),
                    age_s=0.05,
                    mapping_ok=True,
                )
                frame_id += 1
        return _finished(window)

    assert with_low_seconds(2, max_low_windows=2).passed is True
    failing = with_low_seconds(3, max_low_windows=2)
    assert failing.passed is False
    assert (
        "Low frame count in 3 of 10 one-second samples: fewer than 52 usable frames per sample "
        "(lowest count: 40 frames)" in failing.reasons
    )
    assert not any("No usable frames received" in r for r in failing.reasons)
    assert with_low_seconds(0, max_low_windows=0).passed is True
    assert with_low_seconds(1, max_low_windows=0).passed is False


def test_low_sample_message_rounds_up_the_required_frame_count():
    window = _window(window_s=3, fps=20, max_low_windows=0)
    _feed(window, fps=17)
    camera = _finished(window)
    assert (
        "Low frame count in 3 of 3 one-second samples: fewer than 18 usable frames per sample "
        "(lowest count: 17 frames)" in camera.reasons
    )


def test_an_empty_second_is_never_tolerated():
    window = _window(window_s=10.0, max_low_windows=10, min_delivery_fraction=0.01)
    frame_id = 0
    for second in range(10):
        if second == 4:
            continue
        for i in range(60):
            receive_s = second + i / 60
            window.add_frame(
                frame_id=frame_id,
                capture_s=receive_s,
                receive_ns=int(receive_s * SEC),
                age_s=0.05,
                mapping_ok=True,
            )
            frame_id += 1
    camera = _finished(window)
    assert camera.passed is False
    assert "No usable frames received in 1 of 10 one-second samples" in camera.reasons


def test_growing_frame_age_fails():
    window = _window()
    for i in range(1800):
        age = 0.05 + 0.8 * i / 1800
        t = i / FPS
        window.add_frame(frame_id=i, capture_s=t - age, receive_ns=int(t * SEC), age_s=age, mapping_ok=True)
    camera = _finished(window)
    assert camera.passed is False
    assert any("frame age grew" in r for r in camera.reasons)
    assert camera.age_growth_ms > 250


def test_steady_frame_age_does_not_count_as_growth():
    window = _window()
    _feed(window, age_s=0.4)
    camera = _finished(window)
    assert camera.passed is True
    assert camera.age_growth_ms == pytest.approx(0, abs=1e-6)


def test_a_capture_time_gap_over_the_limit_fails_even_when_arrivals_are_even():
    window = _window(window_s=10.0)
    for i in range(600):
        capture_s = i / FPS + (0.4 if i >= 300 else 0.0)
        window.add_frame(
            frame_id=i, capture_s=capture_s, receive_ns=int(i / FPS * SEC), age_s=0.05, mapping_ok=True
        )
    camera = _finished(window)
    assert camera.passed is False
    assert any("gap between capture times" in r for r in camera.reasons)
    assert camera.max_capture_gap_ms == pytest.approx(400 + 1000 / FPS, abs=1)


def test_result_fields_are_numerically_right():
    window = _window(window_s=10.0)
    _feed(window, age_s=0.1)
    camera = _finished(window)
    assert camera.stream_id == 16 and camera.name == "Head left"
    assert camera.expected_fps == FPS
    assert camera.usable_frames == 600
    assert camera.delivered_fps == pytest.approx(60.0)
    assert camera.max_arrival_gap_ms == pytest.approx(1000 / FPS, abs=0.01)
    assert camera.max_capture_gap_ms == pytest.approx(1000 / FPS, abs=0.01)
    assert camera.start_silence_ms == pytest.approx(0.0)
    assert camera.end_silence_ms == pytest.approx(1000 / FPS, abs=0.01)
    assert camera.age_median_ms == pytest.approx(100.0)
    assert camera.age_max_ms == pytest.approx(100.0)


def test_delivered_rate_is_frames_over_the_window_not_over_the_frame_span():
    window = _window(window_s=10.0)
    _feed(window, end_s=5.0)
    assert _finished(window).delivered_fps == pytest.approx(30.0)


def test_too_few_usable_frames_fails_with_the_measured_rate():
    window = _window(window_s=10.0, max_low_windows=100)
    _feed(window, fps=30.0)
    camera = _finished(window)
    assert camera.passed is False
    assert any("usable frames/s arrived against 60 configured" in r for r in camera.reasons)


def test_a_result_with_shrinking_frame_age_is_a_wire_valid_camera():
    window = _window()
    for i in range(1800):
        age = 0.5 - 0.4 * i / 1800
        t = i / FPS
        window.add_frame(frame_id=i, capture_s=t - age, receive_ns=int(t * SEC), age_s=age, mapping_ok=True)
    camera = _finished(window)
    assert camera.passed is True
    assert camera.age_growth_ms < 0
    assert p.NetworkCheckCamera.from_json(camera.to_json()) == camera


def test_wrong_size_frames_are_not_usable_and_the_reason_names_the_offered_size():
    window = _window()
    _feed(window)
    for i in range(5):
        window.add_frame(
            frame_id=10_000 + i,
            capture_s=1.0,
            receive_ns=window.start_ns + SEC,
            age_s=0.05,
            mapping_ok=True,
            size_ok=False,
        )
    camera = _finished(window)
    assert camera.usable_frames == 30 * 60
    assert camera.passed is False
    assert any("5 frame(s) were not the offered 6x4" in r for r in camera.reasons)


def test_a_wrong_size_frame_is_not_also_counted_as_missing_a_capture_time():
    window = _window()
    window.add_frame(
        frame_id=1, capture_s=None, receive_ns=window.start_ns + SEC, age_s=None, mapping_ok=True, size_ok=False
    )
    reasons = _finished(window).reasons
    assert any("not the offered" in r for r in reasons)
    assert not any("without a capture time" in r for r in reasons)
