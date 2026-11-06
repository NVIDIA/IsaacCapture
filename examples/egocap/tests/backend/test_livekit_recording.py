# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""livekit_recording: Unix-to-monotonic mapping, the per-stream feed."""

from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web import livekit_recording as lrec
from egocap_web.livekit_receiver import ReceivedFrame, StreamCounters
from egocap_web.livekit_recording import (
    LiveKitRecordingFeed,
    UnixClockMap,
    unrecordable_reason,
)
from egocap_web.recording_adapter import RecordingIngress
from recording_service import PixelFormat, SubmissionResult, SubmissionStatus, VideoSample

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


# --- received frames -----------------------------------------------------------


def _received(
    stream_id: int = 16, user_ts: int | None = 1_000_000, receive_ns: int | None = None
) -> ReceivedFrame:
    if receive_ns is None:
        # Arrives 0.1 s after capture on FixedMap's clock (capture = user_ts / 1e6 + 10).
        receive_ns = int(((user_ts or 0) / 1e6 + 10.1) * 1e9)
    return ReceivedFrame(
        stream_id=stream_id,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        receive_time_ns=receive_ns,
        transport_timestamp_us=7,
        user_timestamp_us=user_ts,
        frame_id=None,
        user_data=None,
    )


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
        self.counters: defaultdict[int, StreamCounters] = defaultdict(StreamCounters)
        self.timeouts: list[float | None] = []

    async def next_frame_or_lost(self, stream_id: int, timeout: float | None = None) -> ReceivedFrame | None:
        self.calls.append(stream_id)
        self.timeouts.append(timeout)
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
        self.offered: list[VideoSample] = []
        self._answers = list(answers or [])

    def offer(self, item: VideoSample) -> bool:
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
    assert all(isinstance(m, VideoSample) for m in ingress.offered)
    assert feed.failure is None


async def test_offered_sample_is_rgb8_with_the_frame_image_and_receive_time_in_seconds():
    frame = _received(16, 1_000_000, receive_ns=11_100_000_000)
    ingress = FakeIngress()
    feed = _feed(FakeReceiver({16: [frame]}), [_stream(16, "Head left")], ingress, {16: "head_left"})
    feed.start()
    await _settle()
    await feed.stop()
    (sample,) = ingress.offered
    assert isinstance(sample, VideoSample)
    assert sample.source == "head_left"
    assert sample.pixel_format == PixelFormat.RGB8
    assert sample.image is frame.image
    assert sample.capture_time == 11.0
    assert sample.receive_time == pytest.approx(11.1)


async def test_feed_into_the_real_ingress_converts_and_submits_the_sample():
    submitted: list[VideoSample] = []

    def submit(sample):
        submitted.append(sample)
        return SubmissionResult(SubmissionStatus.ACCEPTED, sample.source, None)

    ingress = RecordingIngress(queue_capacity=4, submit=submit)
    await ingress.start()
    frame = _received(16, 1_000_000)
    feed = _feed(FakeReceiver({16: [frame]}), [_stream(16, "Head left")], ingress, {16: "head_left"})
    feed.start()
    for _ in range(50):
        if submitted:
            break
        await asyncio.sleep(0.01)
    await feed.stop()
    await ingress.stop()
    assert feed.failure is None
    assert ingress.failure is None
    assert [x.source for x in submitted] == ["head_left"]
    assert submitted[0].image is frame.image


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


# --- what the receiver lost during the take -----------------------------------


async def test_overflow_since_the_take_began_fails_the_take_with_the_count():
    reasons: list[str] = []
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]})
    receiver.counters[16].queue_overflow = 4  # before the take: not this take's loss
    receiver.gates[16] = asyncio.Event()
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress, on_failure=reasons.append)
    feed.start()
    receiver.counters[16].queue_overflow += 3
    receiver.gates[16].set()
    await _settle()
    await feed.stop()
    assert ingress.offered == []
    assert feed.failure == "Head left: the recorder fell behind and dropped 3 frame(s)"
    assert reasons == [feed.failure]


async def test_overflow_before_the_take_is_not_held_against_it():
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]})
    receiver.counters[16].queue_overflow = 4
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress)
    feed.start()
    await _settle()
    await feed.stop()
    assert len(ingress.offered) == 1 and feed.failure is None


async def test_a_publication_change_during_the_take_fails_it():
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]})
    receiver.gates[16] = asyncio.Event()
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress)
    feed.start()
    receiver.counters[16].publication_changes += 1
    receiver.gates[16].set()
    await _settle()
    await feed.stop()
    assert ingress.offered == []
    assert feed.failure == "Head left: the camera's publication was replaced during the take"


async def test_silence_fails_the_take_after_the_limit(monkeypatch):
    monkeypatch.setattr(lrec, "_MAX_SILENCE_S", 0.02)

    class Silent(FakeReceiver):
        async def next_frame_or_lost(self, stream_id, timeout=None):
            self.timeouts.append(timeout)
            await asyncio.sleep(timeout)
            raise asyncio.TimeoutError

    reasons: list[str] = []
    receiver = Silent({})
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress(), on_failure=reasons.append)
    feed.start()
    await asyncio.sleep(0.1)
    await feed.stop()
    assert receiver.timeouts == [0.02]
    assert feed.failure == "Head left: no frame for 0 s although the track is still published"
    assert reasons == [feed.failure]


async def test_the_feed_asks_the_receiver_to_time_out_at_the_silence_limit():
    receiver = FakeReceiver({})
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress())
    feed.start()
    await _settle()
    await feed.stop()
    assert receiver.timeouts == [lrec._MAX_SILENCE_S]


async def test_loss_reports_counters_since_the_take_began():
    receiver = FakeReceiver({})
    counters = receiver.counters[16]
    counters.queue_overflow, counters.publication_changes = 2, 1
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress())
    feed.start()
    counters.received, counters.missing_frames, counters.frame_id_resets = 9, 3, 1
    counters.queue_overflow += 5
    counters.publication_changes += 2
    assert feed.loss() == {
        "Head left": {
            "received": 9,
            "queue_overflow": 5,
            "publication_changes": 2,
            "missing_frames": 3,
            "frame_id_resets": 1,
        }
    }
    await feed.stop()


def test_loss_before_start_reports_raw_counters():
    receiver = FakeReceiver({})
    receiver.counters[16].queue_overflow = 2
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress())
    assert feed.loss()["Head left"]["queue_overflow"] == 2


async def test_a_healthy_take_with_stride_two_ids_records_every_frame_with_no_missing():
    import egocap_web.livekit_receiver as lr

    frames = [_received(16, 1_000_000 * i) for i in (1, 2, 3, 4)]
    receiver = FakeReceiver({16: frames})
    counters = receiver.counters[16]
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress)
    feed.start()
    for fid in (2, 4, 6, 8):  # the receiver's own accounting, fed the ids a half-rate publisher numbers
        lr.LiveKitFrameReceiver._account(counters, fid)
    await _settle()
    await feed.stop()
    assert len(ingress.offered) == 4 and feed.failure is None
    assert feed.loss()["Head left"]["missing_frames"] == 0


# --- frame age window ----------------------------------------------------------


@pytest.mark.parametrize("age_s", [-0.04, 0.0, 0.5, 1.99])
async def test_frames_inside_the_age_window_are_offered(age_s):
    frame = _received(16, 1_000_000, receive_ns=int((11.0 + age_s) * 1e9))
    ingress = FakeIngress()
    feed = _feed(FakeReceiver({16: [frame]}), [_stream(16, "Head left")], ingress)
    feed.start()
    await _settle()
    await feed.stop()
    assert len(ingress.offered) == 1
    assert feed.failure is None


@pytest.mark.parametrize(
    ("age_s", "text"), [(3.0, "frame age +3.000 s"), (-0.5, "frame age -0.500 s"), (2.01, "frame age +2.010 s")]
)
async def test_frames_outside_the_age_window_fail_the_feed_and_are_not_offered(age_s, text):
    reasons: list[str] = []
    frame = _received(16, 1_000_000, receive_ns=int((11.0 + age_s) * 1e9))
    ingress = FakeIngress()
    feed = _feed(FakeReceiver({16: [frame]}), [_stream(16, "Head left")], ingress, on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert ingress.offered == []
    assert feed.failure == f"Head left: the publisher's clock disagrees with this host's ({text})"
    assert reasons == [feed.failure]


# --- unrecordable_reason -------------------------------------------------------


def _started(*streams):
    return type("Started", (), {"streams": list(streams)})()


def _geometry(stream_id, basis=CAPTURE, **missing):
    values = {"width": 6, "height": 4, "fps": 30.0} | missing
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room="r",
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        **values,
    )


def test_unrecordable_reason_is_none_without_a_session_or_streams():
    assert unrecordable_reason(None) is None
    assert unrecordable_reason(_started()) is None


def test_unrecordable_reason_is_none_when_every_stream_is_recordable():
    assert unrecordable_reason(_started(_geometry(16), _geometry(17))) is None


def test_unrecordable_reason_names_a_stream_that_is_not_a_capture_time():
    started = _started(_geometry(16), _geometry(17, PROCESSING), _geometry(18, MONO))
    assert (
        unrecordable_reason(started)
        == "LiveKit stream 'cam17': its timestamps are not a capture time, so it cannot be recorded"
    )


@pytest.mark.parametrize("field", ["width", "height", "fps"])
def test_unrecordable_reason_names_a_stream_missing_geometry(field):
    started = _started(_geometry(16, **{field: None}))
    assert (
        unrecordable_reason(started)
        == "LiveKit stream 'cam16': width, height and fps must be configured to record it"
    )


def test_unrecordable_reason_ignores_streams_that_are_not_livekit():
    assert unrecordable_reason(_started(object(), _geometry(16))) is None


@pytest.mark.parametrize(
    "url", ["ws://127.0.0.1:8082", "ws://localhost:8082", "ws://[::1]:8082", "ws://127.0.0.2"]
)
def test_unrecordable_reason_accepts_a_loopback_capture_server(url):
    assert unrecordable_reason(_started(_geometry(16)), url) is None


@pytest.mark.parametrize("url", ["ws://192.168.1.20:8082", "ws://capture.lan:8082", "ws://[2001:db8::1]:8082"])
def test_unrecordable_reason_refuses_a_capture_server_on_another_host(url):
    reason = unrecordable_reason(_started(_geometry(16)), url)
    assert reason is not None and "CaptureServer on this machine" in reason and url in reason


def test_a_remote_capture_server_does_not_affect_a_session_without_livekit_streams():
    assert unrecordable_reason(_started(object()), "ws://192.168.1.20:8082") is None
