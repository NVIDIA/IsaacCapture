# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""livekit_recording: Unix-to-monotonic mapping, frame-to-sample conversion and the per-stream feed."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web.livekit_receiver import ReceivedFrame
from egocap_web.livekit_recording import (
    LiveKitRecordingFeed,
    MappedFrame,
    UnixClockMap,
    mapped_frame_to_samples,
)
from recording_service import PixelFormat

CAPTURE = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
PROCESSING = p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US
MONO = p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US


class FakeClocks:
    """Scripted monotonic and Unix clocks; each read pops the next value (the last one repeats)."""

    def __init__(self, monotonic: list[int], unix: list[int]) -> None:
        self._mono, self._unix = list(monotonic), list(unix)
        self.mono_reads = 0
        self.unix_reads = 0

    def monotonic_ns(self) -> int:
        self.mono_reads += 1
        return self._mono.pop(0) if len(self._mono) > 1 else self._mono[0]

    def unix_ns(self) -> int:
        self.unix_reads += 1
        return self._unix.pop(0) if len(self._unix) > 1 else self._unix[0]


def _map(clocks: FakeClocks) -> UnixClockMap:
    return UnixClockMap(monotonic_ns=clocks.monotonic_ns, unix_ns=clocks.unix_ns)


# --- UnixClockMap --------------------------------------------------------------


def test_offset_is_midpoint_of_the_bracket_minus_the_unix_reading():
    # Every attempt brackets a Unix reading of 5000 s between monotonic 100 s and 102 s: midpoint 101 s.
    clocks = FakeClocks(
        monotonic=[0] + [100_000_000_000, 102_000_000_000] * 3 + [103_000_000_000],
        unix=[5_000_000_000_000],
    )
    mapper = _map(clocks)
    assert mapper.monotonic_seconds(5_000_000_000) == pytest.approx(101.0)
    assert mapper.monotonic_seconds(5_001_000_000) == pytest.approx(102.0)  # offset is reused, 1 s later


def test_best_of_three_picks_the_shortest_span():
    # Spans 4 s, 1 s, 2 s; the 1 s bracket is [20 s, 21 s] around unix 1000 s -> offset 20.5 - 1000.
    clocks = FakeClocks(
        monotonic=[
            0,  # age check
            0,
            4_000_000_000,
            20_000_000_000,
            21_000_000_000,
            30_000_000_000,
            32_000_000_000,
            32_000_000_000,  # stamp
        ],
        unix=[1_000_000_000_000, 1_000_000_000_000, 1_000_000_000_000],
    )
    mapper = _map(clocks)
    assert mapper.monotonic_seconds(1_000_000_000) == pytest.approx(20.5)


def test_pair_is_reused_within_one_second_without_remeasuring():
    state = {"mono": 0}
    reads = {"unix": 0}

    def unix() -> int:
        reads["unix"] += 1
        return 100_000_000_000

    mapper = UnixClockMap(monotonic_ns=lambda: state["mono"], unix_ns=unix)
    assert mapper.monotonic_seconds(100_000_000) == pytest.approx(0.0)  # offset -100 s
    assert reads["unix"] == 3
    state["mono"] = 500_000_000
    assert mapper.monotonic_seconds(101_000_000) == pytest.approx(1.0)
    state["mono"] = 1_000_000_000  # age exactly 1 s: still reused
    assert mapper.monotonic_seconds(102_000_000) == pytest.approx(2.0)
    assert reads["unix"] == 3


def test_pair_is_remeasured_after_one_second():
    state = {"mono": 0, "unix": 1_000_000_000_000}
    reads = {"unix": 0}

    def mono() -> int:
        return state["mono"]

    def unix() -> int:
        reads["unix"] += 1
        return state["unix"]

    mapper = UnixClockMap(monotonic_ns=mono, unix_ns=unix)
    assert mapper.monotonic_seconds(1_000_000_000) == pytest.approx(0.0)
    assert reads["unix"] == 3
    state["mono"] = 1_000_000_001  # just over one second
    mapper.monotonic_seconds(1_000_000_000)
    assert reads["unix"] == 6


def test_stepped_unix_clock_is_picked_up_at_the_next_refresh():
    state = {"mono": 0, "unix": 1_000_000_000_000}

    def mono() -> int:
        return state["mono"]

    def unix() -> int:
        return state["unix"]

    mapper = UnixClockMap(monotonic_ns=mono, unix_ns=unix)
    assert mapper.monotonic_seconds(1_000_000_000) == pytest.approx(0.0)
    state["unix"] += 3_000_000_000  # the Unix clock steps forward 3 s
    state["mono"] = 500_000_000
    assert mapper.monotonic_seconds(1_000_000_000) == pytest.approx(0.0)  # not yet noticed
    state["mono"] = 2_000_000_000
    # Unix now reads 1003 s at monotonic 2 s: unix 1000 s is 3 s before that, i.e. -1 s.
    assert mapper.monotonic_seconds(1_000_000_000) == pytest.approx(-1.0)


def test_default_clocks_map_now_to_about_now():
    import time

    mapped = UnixClockMap().monotonic_seconds(time.time_ns() // 1000)
    assert mapped == pytest.approx(time.monotonic(), abs=0.05)


# --- mapped_frame_to_samples ---------------------------------------------------


def _received(
    stream_id: int = 16, user_ts: int | None = 1_000_000, receive_ns: int = 2_500_000_000
) -> ReceivedFrame:
    return ReceivedFrame(
        stream_id=stream_id,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        receive_time_ns=receive_ns,
        transport_timestamp_us=7,
        user_timestamp_us=user_ts,
        frame_id=None,
        user_data=None,
    )


def test_mapped_frame_becomes_one_rgb8_sample_in_seconds():
    frame = _received()
    samples = mapped_frame_to_samples(MappedFrame(frame, "head_left", 1.25))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.source == "head_left"
    assert sample.pixel_format == PixelFormat.RGB8
    assert sample.capture_time == 1.25
    assert sample.receive_time == pytest.approx(2.5)
    assert sample.image is frame.image


def test_mapped_frame_image_is_the_received_image():
    frame = _received()
    assert MappedFrame(frame, "s", 0.0).image is frame.image


# --- LiveKitRecordingFeed ------------------------------------------------------


def _stream(stream_id: int, name: str, basis=CAPTURE) -> p.LiveKitCameraStream:
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=name,
        url="wss://lk",
        room="r",
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
    )


class FakeReceiver:
    """Per-stream scripted frames; an exhausted stream blocks like a quiet camera, or reports loss."""

    def __init__(self, frames: dict[int, list[ReceivedFrame]]) -> None:
        self.frames = {k: list(v) for k, v in frames.items()}
        self.calls: list[int] = []
        self.gates: dict[int, asyncio.Event] = {}
        self.lost: set[int] = set()

    async def next_frame_or_lost(self, stream_id: int) -> ReceivedFrame | None:
        self.calls.append(stream_id)
        gate = self.gates.get(stream_id)
        if gate is not None:
            await gate.wait()
        if self.frames.get(stream_id):
            return self.frames[stream_id].pop(0)
        if stream_id in self.lost:
            return None
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class FakeIngress:
    """Records offers; answers each with the next scripted value (True once the script runs out)."""

    def __init__(self, answers: list[bool] | None = None) -> None:
        self.offered: list[MappedFrame] = []
        self._answers = list(answers or [])

    def offer(self, item: MappedFrame) -> bool:
        self.offered.append(item)
        return self._answers.pop(0) if self._answers else True


class FixedMap(UnixClockMap):
    """Maps unix microseconds to ``us / 1e6 + 10`` seconds."""

    def monotonic_seconds(self, unix_us: int) -> float:
        return unix_us / 1e6 + 10.0


async def _settle(times: int = 10) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def _feed(receiver, streams, ingress, names=None, **kw) -> LiveKitRecordingFeed:
    names = names if names is not None else {s.stream_id: f"src{s.stream_id}" for s in streams}
    return LiveKitRecordingFeed(receiver, streams, ingress, names, clock_map=FixedMap(), **kw)


async def test_capture_stream_frames_are_offered_with_source_and_mapped_capture_time():
    streams = [_stream(16, "Head left")]
    receiver = FakeReceiver({16: [_received(16, 1_000_000), _received(16, 2_000_000)]})
    ingress = FakeIngress()
    feed = _feed(receiver, streams, ingress, {16: "head_left"})
    feed.start()
    await _settle()
    await feed.stop()
    assert [m.source for m in ingress.offered] == ["head_left", "head_left"]
    assert [m.capture_time for m in ingress.offered] == [11.0, 12.0]
    assert feed.failure is None


@pytest.mark.parametrize("basis", [MONO, PROCESSING])
async def test_non_capture_stream_fails_at_once_and_never_offers(basis):
    reasons: list[str] = []
    streams = [_stream(16, "Head left", basis)]
    receiver = FakeReceiver({16: [_received(16)]})
    ingress = FakeIngress()
    feed = _feed(receiver, streams, ingress, on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert ingress.offered == []
    assert receiver.calls == []  # refused before any frame was taken
    assert feed.failure is not None
    assert "Head left" in feed.failure
    assert (
        feed.failure == "Head left: its timestamps are not a capture time, so it cannot be aligned with tracking"
    )
    assert reasons == [feed.failure]


async def test_frame_without_publisher_timestamp_fails():
    reasons: list[str] = []
    streams = [_stream(16, "Head left")]
    receiver = FakeReceiver({16: [_received(16, user_ts=None)]})
    ingress = FakeIngress()
    feed = _feed(receiver, streams, ingress, on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert ingress.offered == []
    assert "without the publisher's capture timestamp" in feed.failure
    assert reasons == [feed.failure]


async def test_refusing_ingress_fails_and_stops_that_stream():
    reasons: list[str] = []
    streams = [_stream(16, "Head left")]
    receiver = FakeReceiver({16: [_received(16, 1_000_000), _received(16, 2_000_000), _received(16, 3_000_000)]})
    ingress = FakeIngress([True, False])
    feed = _feed(receiver, streams, ingress, on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert len(ingress.offered) == 2  # the third frame is never taken
    assert receiver.calls == [16, 16]
    assert "no longer accepting" in feed.failure
    assert reasons == [feed.failure]


async def test_failure_is_reported_once_across_streams():
    reasons: list[str] = []
    streams = [_stream(16, "Head left", MONO), _stream(17, "Head right", MONO)]
    feed = _feed(FakeReceiver({}), streams, FakeIngress(), on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert len(reasons) == 1
    assert "Head left" in feed.failure  # the first reason is kept
    assert "Head right" not in feed.failure


async def test_failure_without_a_callback_is_still_stored():
    feed = _feed(FakeReceiver({}), [_stream(16, "Head left", MONO)], FakeIngress())
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is not None


def test_missing_source_names_raise_value_error():
    streams = [_stream(16, "Head left"), _stream(17, "Head right")]
    with pytest.raises(ValueError, match="Head right"):
        LiveKitRecordingFeed(FakeReceiver({}), streams, FakeIngress(), {16: "a"})


async def test_stop_leaves_no_running_tasks():
    streams = [_stream(16, "Head left"), _stream(17, "Head right")]
    feed = _feed(FakeReceiver({}), streams, FakeIngress())
    before = asyncio.all_tasks()
    feed.start()
    await _settle()
    assert len(asyncio.all_tasks() - before) == 2  # both blocked on a quiet camera
    await feed.stop()
    assert asyncio.all_tasks() == before
    await feed.stop()  # idempotent


async def test_stop_before_start_is_a_no_op():
    feed = _feed(FakeReceiver({}), [_stream(16, "Head left")], FakeIngress())
    await feed.stop()
    assert feed.failure is None


async def test_streams_are_independent():
    streams = [_stream(16, "Head left"), _stream(17, "Head right")]
    receiver = FakeReceiver({16: [_received(16, 1_000_000)], 17: [_received(17, 2_000_000)]})
    receiver.gates[16] = asyncio.Event()  # stream 16 is stuck waiting
    ingress = FakeIngress()
    feed = _feed(receiver, streams, ingress)
    feed.start()
    await _settle()
    assert [m.source for m in ingress.offered] == ["src17"]
    receiver.gates[16].set()
    await _settle()
    await feed.stop()
    assert sorted(m.source for m in ingress.offered) == ["src16", "src17"]


async def test_a_stream_starting_after_another_has_failed_never_takes_a_frame():
    streams = [_stream(16, "Head left"), _stream(17, "Head right")]
    receiver = FakeReceiver({16: [], 17: [_received(17, 2_000_000)]})
    receiver.lost.add(16)
    ingress = FakeIngress()
    feed = _feed(receiver, streams, ingress)
    feed.start()
    await _settle()
    assert feed.failure == "Head left: the LiveKit track was lost"
    assert receiver.calls == [16]
    assert ingress.offered == []
    await feed.stop()


async def test_lost_track_fails_the_take_with_the_stream_name():
    reasons: list[str] = []
    receiver = FakeReceiver({})
    receiver.lost.add(16)
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress(), on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure == "Head left: the LiveKit track was lost"
    assert reasons == [feed.failure]


async def test_frames_queued_before_the_loss_are_all_offered_first():
    receiver = FakeReceiver({16: [_received(16, 1_000_000), _received(16, 2_000_000), _received(16, 3_000_000)]})
    receiver.lost.add(16)
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress)
    feed.start()
    await _settle()
    await feed.stop()
    assert [m.capture_time for m in ingress.offered] == [11.0, 12.0, 13.0]
    assert "track was lost" in feed.failure


async def test_a_stream_still_waiting_on_its_track_does_not_delay_another_that_is_flowing():
    receiver = FakeReceiver({17: [_received(17, 1_000_000)]})
    receiver.gates[16] = asyncio.Event()  # 16 is quiet, and will later be lost
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left"), _stream(17, "Head right")], ingress)
    feed.start()
    await _settle()
    assert [m.source for m in ingress.offered] == ["src17"]
    assert feed.failure is None
    receiver.lost.add(16)
    receiver.gates[16].set()
    await _settle()
    await feed.stop()
    assert feed.failure == "Head left: the LiveKit track was lost"
