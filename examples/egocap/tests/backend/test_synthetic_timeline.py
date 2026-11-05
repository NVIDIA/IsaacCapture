# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A synthetic event timeline through the real LiveKitRecordingFeed: what stamping error survives recording."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web.livekit_receiver import ReceivedFrame
from egocap_web.livekit_recording import LiveKitRecordingFeed, UnixClockMap

#: Truth event times in this PC's monotonic seconds; tracking samples carry these directly.
TRUTH_S = [100.0, 100.033, 100.066, 100.1, 100.133]
#: Fixed gap between the PC's Unix and monotonic clocks.
UNIX_MINUS_MONO_S = 1_700_000_000.0
MICROSECOND = 1e-6


class PcClocks:
    """This PC's clocks: monotonic is settable, Unix is monotonic plus a settable gap; reads are instantaneous."""

    def __init__(self) -> None:
        self.mono_ns = int(TRUTH_S[0] * 1e9)
        self.gap_ns = int(UNIX_MINUS_MONO_S * 1e9)

    def monotonic_ns(self) -> int:
        return self.mono_ns

    def unix_ns(self) -> int:
        return self.mono_ns + self.gap_ns

    def clock_map(self) -> UnixClockMap:
        return UnixClockMap(monotonic_ns=self.monotonic_ns, unix_ns=self.unix_ns)


def stamp_us(truth_s: float, offset_us: int = 0, drift_us: float = 0.0) -> int:
    """The emulator's capture-basis stamp: truth Unix time plus offset and drift (no pipeline delay)."""
    return round(truth_s * 1e6 + UNIX_MINUS_MONO_S * 1e6 + offset_us + drift_us)


def frame(frame_id: int, user_ts: int, receive_s: float) -> ReceivedFrame:
    """A received frame whose pixels all equal its frame_id, so identity can be read back from the image."""
    return ReceivedFrame(
        stream_id=16,
        image=np.full((2, 2, 3), frame_id, dtype=np.uint8),
        receive_time_ns=round(receive_s * 1e9),
        transport_timestamp_us=0,
        user_timestamp_us=user_ts,
        frame_id=frame_id,
        user_data=None,
    )


class QueueReceiver:
    """Serves the scripted frames in order, then reports the track lost."""

    def __init__(self, frames: list[ReceivedFrame]) -> None:
        self._frames = list(frames)

    async def next_frame_or_lost(self, stream_id: int) -> ReceivedFrame | None:
        return self._frames.pop(0) if self._frames else None


class ListIngress:
    """Accepts every sample."""

    def __init__(self) -> None:
        self.offered: list = []

    def offer(self, item) -> bool:
        self.offered.append(item)
        return True


async def record(frames: list[ReceivedFrame], clocks: PcClocks, receiver=None):
    """Runs the feed over ``frames`` to the end of the script; returns (feed, offered samples)."""
    stream = p.LiveKitCameraStream(
        stream_id=16,
        name="cam",
        url="wss://lk",
        room="r",
        publisher="o",
        track="t",
        timestamp_basis=p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US,
    )
    ingress = ListIngress()
    feed = LiveKitRecordingFeed(
        receiver or QueueReceiver(frames), [stream], ingress, {16: "cam"}, clock_map=clocks.clock_map()
    )
    feed.start()
    for _ in range(20):
        await asyncio.sleep(0)
    await feed.stop()
    return feed, ingress.offered


def residuals(samples, truth_by_id: dict[int, float]) -> list[float]:
    """Recorded capture_time minus the truth time of the same frame."""
    return [s.capture_time - truth_by_id[int(s.image[0, 0, 0])] for s in samples]


TRUTH_BY_ID = {i + 1: t for i, t in enumerate(TRUTH_S)}


async def test_recorded_capture_time_shows_the_injected_offset_and_drift():
    offset_us, drift_ppm = 12_000, 20.0  # ppm is microseconds of error per elapsed second
    run_start_s = 100.0
    truth_by_id = {i + 1: run_start_s + e for i, e in enumerate([0.0, 15.0, 30.0, 45.0, 60.0])}
    elapsed_s = {i: t - run_start_s for i, t in truth_by_id.items()}
    frames = [frame(i, stamp_us(t, offset_us, elapsed_s[i] * drift_ppm), t + 0.1) for i, t in truth_by_id.items()]
    _, samples = await record(frames, PcClocks())
    assert len(samples) == len(truth_by_id)
    got = residuals(samples, truth_by_id)
    for r, i in zip(got, truth_by_id):
        assert r == pytest.approx((offset_us + elapsed_s[i] * drift_ppm) * MICROSECOND, abs=MICROSECOND)
    assert all(b > a for a, b in zip(got, got[1:]))


# Jitter/drop/restart cases: LiveKitRecordingFeed fed manual ReceivedFrames, not the full wire path.


async def test_delivery_jitter_does_not_change_capture_time():
    delays = [0.02, 0.9, 0.001, 0.5, 0.2]
    frames = [frame(i, stamp_us(t), t + d) for (i, t), d in zip(TRUTH_BY_ID.items(), delays)]
    _, samples = await record(frames, PcClocks())
    assert residuals(samples, TRUTH_BY_ID) == pytest.approx([0.0] * len(TRUTH_S), abs=MICROSECOND)


async def test_dropped_frames_leave_the_remaining_frames_capture_time_and_identity():
    kept = [1, 3, 5]
    frames = [frame(i, stamp_us(TRUTH_BY_ID[i], 5_000), TRUTH_BY_ID[i] + 0.1) for i in kept]
    _, samples = await record(frames, PcClocks())
    assert [int(s.image[0, 0, 0]) for s in samples] == kept
    assert residuals(samples, TRUTH_BY_ID) == pytest.approx([0.005] * 3, abs=MICROSECOND)


async def test_publisher_restart_keeps_each_frames_own_capture_time():
    # frame_id restarts at 1 after the third frame; only the id repeats, never the event time.
    ids = [1, 2, 3, 1, 2]
    frames = [frame(fid, stamp_us(t), t + 0.1) for fid, t in zip(ids, TRUTH_S)]
    _, samples = await record(frames, PcClocks())
    assert [s.capture_time for s in samples] == pytest.approx(TRUTH_S, abs=MICROSECOND)


async def test_a_pc_clock_step_between_measurements_fails_the_take_and_stops_recording():
    clocks = PcClocks()

    class SteppingReceiver(QueueReceiver):
        """Steps this PC's Unix clock by 50 ms and ages the pair after the second frame is served."""

        async def next_frame_or_lost(self, stream_id):
            if len(self._frames) == 3:
                clocks.gap_ns += 50_000_000
                clocks.mono_ns += 1_500_000_000
            return await super().next_frame_or_lost(stream_id)

    # After the step the later frames arrive in the shifted monotonic time, so their age stays plausible.
    frames = [frame(i, stamp_us(TRUTH_BY_ID[i]), TRUTH_BY_ID[i] + 0.1) for i in (1, 2)]
    frames += [frame(i, stamp_us(TRUTH_BY_ID[i]), TRUTH_BY_ID[i] + 1.6) for i in (3, 4, 5)]
    feed, samples = await record(frames, clocks, SteppingReceiver(frames))
    assert feed.failure is not None and "clock stepped" in feed.failure
    assert [int(s.image[0, 0, 0]) for s in samples] == [1, 2]
