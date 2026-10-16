# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The playback track, over frames built in memory."""

from __future__ import annotations

import math
from dataclasses import replace

import synth
from full_body_cts.frames import NUM_JOINTS
from full_body_cts.labels import StepTimeline, Step
from full_body_cts.panel.track import (
    LOG_CLOCK,
    SAMPLE_CLOCK,
    TeeSource,
    TrackBuilder,
)
from full_body_cts.profile import FULL_BODY
from full_body_cts.report import Verdict, run


def build(frames: list) -> "object":
    builder = TrackBuilder()
    for frame in frames:
        builder.update(frame)
    return builder.track()


def test_time_starts_at_zero_on_the_clock_the_rate_checks_use():
    track = build(synth.frames(10))
    assert track.clock == SAMPLE_CLOCK
    assert track.times_s[0] == 0.0
    assert track.duration_s == synth.PERIOD_NS * 9 / 1e9


def test_a_frame_without_a_sample_time_falls_back_to_the_log_time():
    frames = [replace(f, sample_time_ns=None) for f in synth.frames(5)]
    assert build(frames).clock == LOG_CLOCK


def test_the_rate_is_the_reciprocal_of_the_interval():
    track = build(synth.frames(5))
    assert math.isnan(track.samples[0].rate_hz)
    assert track.samples[1].rate_hz == 1e9 / synth.PERIOD_NS


def test_an_interval_that_does_not_move_forward_has_no_rate():
    """A non-monotonic timestamp is evidence; the curve shows a break, not a number."""
    frames = synth.frames(3)
    frames[2] = replace(frames[2], sample_time_ns=frames[1].sample_time_ns - 1000)
    assert math.isnan(build(frames).samples[2].rate_hz)


def test_an_invalid_joint_is_held_where_it_was_last_seen():
    first = synth.frame(0)
    assert first.joints is not None
    last_seen = first.joints[-1].position
    garbage = synth.joint(position=(-16363.96, 8123.5, 55.0), is_valid=False)
    track = build([first, synth.with_joint(synth.frame(1), NUM_JOINTS - 1, garbage)])

    held = track.samples[1]
    assert held.valid[-1] is False
    assert held.positions[-1] == last_seen
    assert held.valid_count == NUM_JOINTS - 1


def test_a_joint_never_valid_has_no_position_at_all():
    track = build(synth.frames(4, valid_joints=NUM_JOINTS - 2))
    assert track.samples[0].positions[-1] is None
    assert track.min_valid_count == NUM_JOINTS - 2
    assert track.joints_ever_invalid() == (NUM_JOINTS - 2, NUM_JOINTS - 1)


def test_a_record_carrying_no_pose_holds_every_joint():
    track = build([synth.frame(0), synth.frame(1, has_payload=False)])
    blank = track.samples[1]
    assert blank.valid_count == 0
    assert all(position is not None for position in blank.positions)


def test_the_window_trails_the_playhead():
    track = build(synth.frames(300))
    index = track.index_at(2.0)
    times, rates = track.rate_window(index, 0.5)
    assert times[-1] <= 2.0
    assert 2.0 - times[0] <= 0.5 + synth.PERIOD_NS / 1e9
    assert len(times) == len(rates)


def test_the_window_fills_before_it_scrolls():
    track = build(synth.frames(300))
    times, _ = track.rate_window(0, 0.5)
    assert times[0] == synth.PERIOD_NS / 1e9
    assert 0.4 <= times[-1] <= 0.5


def test_the_window_carries_no_sample_without_a_rate():
    """One NaN anywhere in the series and uPlot draws no curve at all."""
    frames = synth.frames(60)
    frames[30] = replace(frames[30], sample_time_ns=frames[29].sample_time_ns)
    track = build(frames)
    times, rates = track.rate_window(len(frames) - 1, 60.0)
    assert all(rate == rate for rate in rates)
    assert len(times) == len(rates) == len(frames) - 2


def test_seeking_lands_on_the_last_sample_at_or_before_the_time():
    track = build(synth.frames(10))
    period_s = synth.PERIOD_NS / 1e9
    assert track.index_at(-1.0) == 0
    assert track.index_at(period_s * 3.5) == 3
    assert track.index_at(1e6) == 9


def test_an_empty_source_is_a_track_with_nothing_in_it():
    track = build([])
    assert track.samples == ()
    assert track.duration_s == 0.0
    assert track.index_at(1.0) == 0
    assert track.rate_window(0, 1.0) == ([], [])


def test_frames_carry_the_label_of_the_window_they_fall_in():
    frames = synth.frames(60)
    assert frames[0].sample_time_ns is not None
    start = frames[20].sample_time_ns
    end = frames[40].sample_time_ns
    assert start is not None and end is not None
    timeline = StepTimeline(
        steps=(Step(0, "squat", start, end, is_still_window=False),)
    )
    builder = TrackBuilder(timeline)
    for frame in frames:
        builder.update(frame)
    track = builder.track()

    assert track.samples[10].step is None
    assert track.samples[25].step == "squat"
    assert track.samples[50].step is None


def test_the_tee_feeds_the_checks_and_the_track_from_one_pass():
    source = TeeSource(synth.StubSource(synth.frames(400)))
    report = run(source)
    track = source.track()

    assert report.frames == 400
    assert len(track.samples) == 400
    assert track.profile is FULL_BODY
    assert report.verdict in set(Verdict)


def test_the_tee_leaves_a_source_without_a_path_pathless():
    """``report.run`` reads ``path`` with a default; the panel must not forge one."""
    report = run(TeeSource(synth.StubSource(synth.frames(5))))
    assert "None" not in report.source
