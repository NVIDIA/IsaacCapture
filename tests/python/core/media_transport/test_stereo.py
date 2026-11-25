# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal pairing must expose waiting and never pair expired or unrelated eyes."""

import logging
from types import SimpleNamespace

import pytest

from isaac_media_transport.stereo import StereoPairer, camera_specs
from isaac_media_transport.transport import ReceivedFrame, ReceivedPair


def frame(stamp, received, value="image"):
    return ReceivedFrame(value, received, SimpleNamespace(user_timestamp=stamp))


def test_pair_waits_only_for_missing_eye_and_reports_latency(caplog):
    now = [1_000_000_000]
    pairer = StereoPairer(clock=lambda: now[0])
    left = frame(10_000_000_000, now[0], "left")
    pairer.push(0, left)
    assert pairer.pop() is None
    now[0] += 5_000_000
    right = frame(10_002_000_000, now[0], "right")
    pairer.push(1, right)
    assert pairer.pop() == (left, right, now[0])
    assert pairer.counters["waited_left"] == pairer.counters["wait_gt_1ms"] == 1
    assert list(pairer.samples[0]) == [5]
    assert list(pairer.samples[1]) == [0]
    with caplog.at_level(logging.INFO):
        pairer.report("cam", force=True)
    assert "delayed_left=1" in caplog.text
    assert "wait_gt_1ms=100.0%" in caplog.text
    assert "5.00/5.00/5.00" in caplog.text
    assert "coordination adds latency" in caplog.text
    assert pairer.pop() is None


def test_expired_eye_is_not_released_when_partner_arrives_late():
    now = [1_000_000_000]
    pairer = StereoPairer({"max_wait_ms": 8}, clock=lambda: now[0])
    pairer.push(0, frame(10_000_000_000, now[0]))
    now[0] += 9_000_000
    pairer.push(1, frame(10_000_000_000, now[0]))
    assert pairer.pop() is None
    assert pairer.counters["drop_left_timeout"] == 1
    assert list(pairer.dropped_holds[0]) == [9]


def test_skew_drop_and_recovery_do_not_depend_on_frame_sequence():
    now = [1_000_000_000]
    pairer = StereoPairer(clock=lambda: now[0])
    pairer.push(0, frame(10_000_000_000, now[0], "old left"))
    pairer.push(1, frame(10_033_000_000, now[0], "right"))
    assert pairer.pop() is None
    pairer.push(0, frame(10_034_000_000, now[0], "new left"))
    left, right, _ = pairer.pop()
    assert (left.frame, right.frame) == ("new left", "right")
    assert pairer.counters["drop_left_skew"] == 1


def test_queue_is_bounded_and_restart_flushes_both_eyes():
    pairer = StereoPairer({"queue_size": 1}, clock=lambda: 1_000_000_000)
    pairer.push(0, frame(1, 1_000_000_000))
    pairer.push(0, frame(2, 1_000_000_000))
    assert pairer.counters["drop_left_overflow"] == 1
    pairer.reset()
    pairer.push(1, frame(2, 1_000_000_000))
    assert pairer.pop() is None
    assert pairer.counters["drop_left_reset"] == 1


def test_missing_or_reordered_timestamps_have_no_arrival_time_fallback():
    pairer = StereoPairer()
    pairer.push(0, ReceivedFrame("image", 0, None))
    pairer.push(0, frame(2, 0))
    pairer.push(0, frame(1, 0))
    assert pairer.counters["drop_left_missing_timestamp"] == 1
    assert pairer.counters["drop_left_old_timestamp"] == 1


@pytest.mark.parametrize(
    "options", [{"max_wait_ms": 0}, {"max_skew_ms": float("nan")}, {"queue_size": 17}]
)
def test_invalid_pairing_budget_is_rejected(options):
    with pytest.raises(ValueError):
        StereoPairer(options)


def test_stereo_specs_preserve_mono_and_split_without_composition():
    mono = dict(name="cam", type="v4l2")
    assert camera_specs({}, mono) == [mono]
    spec = dict(
        name="cam",
        type="argus",
        stereo=True,
        sensor_id=0,
        width=1920,
        height=1080,
        fps=30,
    )
    eyes = camera_specs({"video_encoder": "gstreamer"}, spec)
    assert [(e["name"], e["sensor_id"], e["stereo"]) for e in eyes] == [
        ("cam.left", 0, False),
        ("cam.right", 1, False),
    ]
    assert all(e["width"] == 1920 for e in eyes)
    assert spec["stereo"]
    with pytest.raises(ValueError, match="distinct"):
        camera_specs(
            {"video_encoder": "gstreamer"}, {**spec, "right": {"sensor_id": 0}}
        )


def test_pair_age_uses_older_received_frame():
    pair = ReceivedPair(frame(100, 10), frame(102, 15), 20)
    assert pair.received_ns == 10
