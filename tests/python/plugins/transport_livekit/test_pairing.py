# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arrival-driven pairing must retain timestamp bounds and release silent eyes."""

import asyncio
import time
from types import SimpleNamespace

from isaac_media_transport.transport import ReceivedFrame
from isaac_transport_livekit.video import VideoReceiver


def receiver():
    return VideoReceiver({}, identity="test", publisher="camera", track=("l", "r"))


def frame(stamp, received, pixels):
    return ReceivedFrame(pixels, received, SimpleNamespace(user_timestamp=stamp))


def test_second_arrival_publishes_without_coordinator_and_preserves_timings():
    r = receiver()
    now = [1_000_000_000]
    r._pairer.clock = lambda: now[0]
    events = []
    r.timing = SimpleNamespace(record=lambda *args, **kw: events.append((args, kw)))
    left = frame(10_000_000_000, now[0], object())
    r._accept_frame("l", left)
    assert r.latest() is None
    now[0] += 4_000_000
    right = frame(10_004_000_000, now[0], object())
    r._accept_frame("r", right)
    result = r.latest()
    assert (result.left, result.right) == (left, right)
    assert list(r._pairer.samples[0]) == [4]
    assert list(r._pairer.samples[1]) == [0]
    assert events == [
        (("pair_ready", "l", 10_000_000_000), {"at_ns": now[0]}),
        (("pair_ready", "r", 10_004_000_000), {"at_ns": now[0]}),
    ]


def test_late_or_wrong_partner_cannot_create_false_low_latency_pair():
    r = receiver()
    now = [1_000_000_000]
    r._pairer.clock = lambda: now[0]
    r._accept_frame("l", frame(10_000_000_000, now[0], "left"))
    now[0] += 26_000_000
    r._accept_frame("r", frame(10_004_000_000, now[0], "late right"))
    assert r.latest() is None
    assert r.pairing_stats()["drop_left_timeout"] == 1
    r._accept_frame("l", frame(10_033_000_000, now[0], "next left"))
    assert r.latest() is None
    assert r.pairing_stats()["drop_right_skew"] == 1
    r._accept_frame("r", frame(10_037_000_000, now[0], "next right"))
    result = r.latest()
    assert (result.left.frame, result.right.frame) == ("next left", "next right")


def test_latest_mailbox_drops_whole_pairs_and_retained_frames_stay_owned():
    r = receiver()
    r._pairer.clock = lambda: 1_000_000_000
    for index in range(3):
        for eye in ("l", "r"):
            r._accept_frame(eye, frame(index + 1, 1_000_000_000, (eye, index)))
    pair = r.latest()
    assert r.frames == 3 and r.overwritten == 2
    assert (pair.left.frame, pair.right.frame) == (("l", 2), ("r", 2))
    r._pairer.reset()
    assert pair.left.frame == ("l", 2)


def test_silent_partner_expires_without_further_arrivals():
    async def run():
        r = receiver()
        r._stop = asyncio.Event()
        r._pairing_changed = asyncio.Event()
        expired = asyncio.Event()
        drop = r._pairer._drop

        def notify_drop(eye, reason):
            drop(eye, reason)
            if reason == "timeout":
                expired.set()

        r._pairer._drop = notify_drop
        r._accept_frame("l", frame(1, time.monotonic_ns(), "left"))
        task = asyncio.create_task(r._coordinate())
        try:
            async with asyncio.timeout(1):
                await expired.wait()
            assert r.pairing_stats()["queued_left"] == 0
            assert r.latest() is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_mono_still_delivers_immediately():
    r = VideoReceiver({}, identity="test", publisher="camera", track="mono")
    image = frame(1, 1, "pixels")
    r._accept_frame("mono", image)
    assert r.latest() is image
