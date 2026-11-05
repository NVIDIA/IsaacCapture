# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""livekit_recording: the per-stream feed over the clock-sync mapping."""

from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web import livekit_recording as lrec
from egocap_web.clock_exchange import ClockMapping
from egocap_web.livekit_receiver import LiveKitFrameReceiver, ReceivedFrame, StreamCounters
from egocap_web.livekit_recording import (
    BUDGET_ENV,
    LiveKitRecordingFeed,
    uncertainty_budget_ns,
    unrecordable_reason,
)
from egocap_web.recording_adapter import RecordingIngress
from recording_service import PixelFormat, SubmissionResult, SubmissionStatus, VideoSample

CAPTURE = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US
UNIX_CAPTURE = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
PROCESSING = p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US
MONO = p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US


# --- received frames -----------------------------------------------------------


def _received(
    stream_id: int = 16, user_ts: int | None = 1_000_000, receive_ns: int | None = None
) -> ReceivedFrame:
    if receive_ns is None:
        # Arrives 0.1 s after capture on the fake mapping's clock (capture = user_ts / 1e6 + 10).
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

    def __init__(self, frames: dict[int, list[ReceivedFrame]], clock_sync=None) -> None:
        self.clock_sync = clock_sync if clock_sync is not None else FakeClockSync()
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


class FakeClockSync:
    """Stands in for ClockSyncClient: a fixed mapping of the publisher clock (local = remote + 10 s)."""

    def __init__(self, offset_ns: int = -10_000_000_000, uncertainty_ns: int = 1_000) -> None:
        self.offset_ns = offset_ns
        self.uncertainty_ns = uncertainty_ns
        self.discontinuity = False
        self.available = True

    def mapping(self) -> ClockMapping | None:
        if not self.available:
            return None
        return ClockMapping(
            epoch=1, offset_ns=self.offset_ns, rtt_ns=2_000, uncertainty_ns=self.uncertainty_ns, accepted_at_ns=0
        )


async def _settle(times: int = 10) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def _feed(receiver, streams, ingress, names=None, **kw) -> LiveKitRecordingFeed:
    names = names if names is not None else {s.stream_id: f"src{s.stream_id}" for s in streams}
    return LiveKitRecordingFeed(receiver, streams, ingress, names, **kw)


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


class _FlipAfterFirst(FakeReceiver):
    """Applies ``flip`` to the clock-sync object while the second frame is being served."""

    def __init__(self, frames, clock_sync, flip) -> None:
        super().__init__(frames, clock_sync)
        self._flip = flip

    async def next_frame_or_lost(self, stream_id):
        frame = await super().next_frame_or_lost(stream_id)
        if len(self.calls) == 2:
            self._flip(self.clock_sync)
        return frame


@pytest.mark.parametrize(
    ("flip", "text"),
    [
        (lambda c: setattr(c, "discontinuity", True), "clock mapping jumped"),
        (lambda c: setattr(c, "available", False), "no current clock mapping"),
    ],
)
async def test_a_jumped_or_missing_mapping_fails_the_take_and_records_no_further_frames(flip, text):
    reasons: list[str] = []
    frames = [_received(16, 1_000_000), _received(16, 2_000_000), _received(16, 3_000_000)]
    ingress = FakeIngress()
    # The first frame is mapped before the flip and recorded; the second sees the changed mapping.
    receiver = _FlipAfterFirst({16: frames}, FakeClockSync(), flip)
    feed = LiveKitRecordingFeed(
        receiver, [_stream(16, "Head left")], ingress, {16: "head_left"}, on_failure=reasons.append
    )
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is not None
    assert text in feed.failure
    assert reasons == [feed.failure]
    assert [m.capture_time for m in ingress.offered] == [11.0]


async def test_mapping_offset_places_capture_time_on_the_local_clock():
    clock = FakeClockSync(offset_ns=-10_000_000_000)
    ingress = FakeIngress()
    feed = _feed(FakeReceiver({16: [_received(16, 1_000_000)]}, clock), [_stream(16, "Head left")], ingress)
    feed.start()
    await _settle()
    clock.offset_ns = -20_000_000_000  # a later mapping applies to later frames only
    await feed.stop()
    assert [m.capture_time for m in ingress.offered] == [11.0]


@pytest.mark.parametrize(
    ("value", "expected_ns"),
    [
        (None, 5_000_000),
        ("", 5_000_000),
        ("  ", 5_000_000),
        ("2", 2_000_000),
        ("0.5", 500_000),
        (" 12.5 ", 12_500_000),
    ],
)
def test_uncertainty_budget_defaults_and_overrides(monkeypatch, value, expected_ns):
    if value is None:
        monkeypatch.delenv(BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(BUDGET_ENV, value)
    assert uncertainty_budget_ns() == expected_ns


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan", "-0.5"])
def test_an_invalid_or_non_positive_budget_is_refused_naming_the_variable(monkeypatch, value):
    monkeypatch.setenv(BUDGET_ENV, value)
    with pytest.raises(ValueError, match=BUDGET_ENV):
        uncertainty_budget_ns()


async def test_frames_are_recorded_when_uncertainty_is_within_the_budget():
    ingress = FakeIngress()
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]}, FakeClockSync(uncertainty_ns=3_000_000))
    feed = _feed(receiver, [_stream(16, "Head left")], ingress, max_uncertainty_ns=3_000_000)
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is None
    assert len(ingress.offered) == 1


async def test_uncertainty_over_the_budget_fails_the_take_naming_the_budget(monkeypatch):
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    reasons: list[str] = []
    ingress = FakeIngress()
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]}, FakeClockSync(uncertainty_ns=5_000_001))
    feed = _feed(receiver, [_stream(16, "Head left")], ingress, on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is not None
    assert BUDGET_ENV in feed.failure
    assert "5.00 ms budget" in feed.failure
    assert reasons == [feed.failure]
    assert ingress.offered == []


async def test_the_budget_comes_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv(BUDGET_ENV, "1")
    receiver = FakeReceiver({16: [_received(16, 1_000_000)]}, FakeClockSync(uncertainty_ns=2_000_000))
    feed = _feed(receiver, [_stream(16, "Head left")], FakeIngress())
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is not None
    assert "1.00 ms budget" in feed.failure


async def test_a_mapping_in_budget_at_the_first_frame_ages_over_budget_when_replies_stop(monkeypatch):
    """Real estimator and sync client: only time passing, with no new exchange, pushes the take over budget."""
    from egocap_web.clock_exchange import ClockEstimator, Exchange
    from egocap_web.livekit_clock_sync import ClockSyncClient

    monkeypatch.delenv(BUDGET_ENV, raising=False)
    s = 1_000_000_000
    now = [100 * s]
    # Stale limit raised so the aged figure, not the staleness rule, is what stops the take.
    client = ClockSyncClient(
        None,  # type: ignore[arg-type]  # only mapping() is used; no room is needed
        "o",
        monotonic_ns=lambda: now[0],
        estimator=ClockEstimator(stale_after_ns=600 * s),
    )
    remote_offset_ns = 1_000 * s  # the publisher's clock reads this far ahead of ours
    # A 2 ms round trip: 1 ms uncertainty at the exchange, 5 ms budget.
    t1 = 99 * s
    assert (
        client.estimator.add(
            Exchange(
                t1=t1,
                t2=t1 + 1_000_000 + remote_offset_ns,
                t3=t1 + 1_000_000 + remote_offset_ns,
                t4=t1 + 2_000_000,
            )
        )
        is None
    )

    def frame(capture_local_s: float) -> ReceivedFrame:
        user_ts_us = round((capture_local_s * s + remote_offset_ns) / 1000)
        return _received(16, user_ts_us, receive_ns=round((capture_local_s + 0.1) * s))

    class AgingReceiver(FakeReceiver):
        """Delivers each frame at its own local time: the clock moves, no clock-sync reply arrives."""

        def __init__(self, scripted: list[tuple[int, ReceivedFrame]]) -> None:
            super().__init__({}, client)
            self.scripted = scripted

        async def next_frame_or_lost(self, stream_id: int) -> ReceivedFrame | None:
            if not self.scripted:
                await asyncio.Event().wait()
            at_ns, received = self.scripted.pop(0)
            now[0] = at_ns
            return received

    first, second, third = frame(100.0), frame(129.9), frame(130.0)
    receiver = AgingReceiver([(round(100.1 * s), first), (round(130 * s), second), (round(130.1 * s), third)])
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress, {16: "head_left"}, max_uncertainty_ns=5_000_000)
    feed.start()
    await _settle()
    await feed.stop()

    assert client.mapping() is not None  # not stale: the budget, not staleness, is what failed
    assert [m.capture_time for m in ingress.offered] == [pytest.approx(100.0)]  # the earlier frame stays recorded
    assert feed.failure is not None
    assert BUDGET_ENV in feed.failure
    assert "5.00 ms budget" in feed.failure
    # 1 ms from the round trip plus the default 200 ppm over the 31 s since the exchange.
    assert "uncertain by 7.20 ms" in feed.failure
    assert len(receiver.scripted) == 1  # nothing after the failing frame was taken


def test_a_receiver_without_clock_sync_is_refused():
    receiver = FakeReceiver({})
    receiver.clock_sync = None
    with pytest.raises(ValueError, match="clock-sync"):
        LiveKitRecordingFeed(receiver, [_stream(16, "Head left")], FakeIngress(), {16: "a"})


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


async def test_publisher_monotonic_stream_is_recorded_through_the_clock_mapping():
    ingress = FakeIngress()
    frame = _received(16, 1_000_000, receive_ns=11_100_000_000)
    feed = _feed(FakeReceiver({16: [frame]}), [_stream(16, "Head left", MONO)], ingress, {16: "head_left"})
    feed.start()
    await _settle()
    await feed.stop()
    assert feed.failure is None
    (sample,) = ingress.offered
    assert sample.source == "head_left"
    assert sample.capture_time == 11.0


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


@pytest.mark.parametrize("basis", [PROCESSING, UNIX_CAPTURE])
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
    assert feed.failure == "Head left: its timestamps are not on the publisher's monotonic clock"
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
    streams = [_stream(16, "Head left", PROCESSING), _stream(17, "Head right", PROCESSING)]
    feed = _feed(FakeReceiver({}), streams, FakeIngress(), on_failure=reasons.append)
    feed.start()
    await _settle()
    await feed.stop()
    assert len(reasons) == 1
    assert "Head left" in feed.failure  # the first reason is kept
    assert "Head right" not in feed.failure


async def test_failure_without_a_callback_is_still_stored():
    feed = _feed(FakeReceiver({}), [_stream(16, "Head left", PROCESSING)], FakeIngress())
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
    frames = [_received(16, 1_000_000 * i) for i in (1, 2, 3, 4)]
    receiver = FakeReceiver({16: frames})
    counters = receiver.counters[16]
    ingress = FakeIngress()
    feed = _feed(receiver, [_stream(16, "Head left")], ingress)
    feed.start()
    for fid in (2, 4, 6, 8):  # the receiver's own accounting, fed the ids a half-rate publisher numbers
        LiveKitFrameReceiver._account(counters, fid)
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


def test_unrecordable_reason_names_a_stream_that_is_not_monotonic():
    started = _started(_geometry(16), _geometry(17, PROCESSING), _geometry(18, MONO))
    assert (
        unrecordable_reason(started)
        == "LiveKit stream 'cam17': its timestamps are not on the publisher's monotonic clock, "
        "so it cannot be recorded"
    )


def test_publisher_monotonic_stream_is_recordable():
    assert unrecordable_reason(_started(_geometry(16, MONO), _geometry(17))) is None


@pytest.mark.parametrize("basis", [PROCESSING, UNIX_CAPTURE])
def test_non_monotonic_bases_are_still_refused(basis):
    assert unrecordable_reason(_started(_geometry(16, basis))) == (
        "LiveKit stream 'cam16': its timestamps are not on the publisher's monotonic clock, "
        "so it cannot be recorded"
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
