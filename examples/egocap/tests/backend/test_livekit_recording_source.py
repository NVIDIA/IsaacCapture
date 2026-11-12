# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""recordable_livekit_streams and LiveKitRecordingSource, against a fake frame receiver."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from capture_server import protocol as p
from egocap_web import livekit_recording as lrec
from egocap_web.clock_exchange import ClockMapping
from egocap_web.livekit_receiver import ReceivedFrame, StreamCounters
from egocap_web.livekit_recording import LiveKitRecordingSource, recordable_livekit_streams

CAPTURE = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US
UNIX_CAPTURE = p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
PROCESSING = p.LiveKitTimestampBasis.PUBLISHER_PROCESSING_UNIX_US
MONO = p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US


def livekit_stream(stream_id: int, *, basis=CAPTURE, width=6, height=4, fps=30.0, room="room-a"):
    return p.LiveKitCameraStream(
        stream_id=stream_id,
        name=f"cam{stream_id}",
        url="wss://lk",
        room=room,
        publisher="o",
        track=f"t{stream_id}",
        timestamp_basis=basis,
        width=width,
        height=height,
        fps=fps,
    )


def jpeg_camera(stream_id: int = 1) -> p.CameraStream:
    return p.CameraStream(
        stream_id=stream_id,
        kind=p.StreamKind.IMAGE,
        name="jpeg",
        type=p.CameraType.OAK_D_COLOR,
        required=False,
        width=6,
        height=4,
        fps=30.0,
        codec=p.ImageCodec.JPEG,
        jpeg_quality=80,
        orientation=p.ImageOrientation(),
    )


def frame(stream_id: int, *, width=6, height=4) -> ReceivedFrame:
    return ReceivedFrame(
        stream_id=stream_id,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        # Captured just now on this host's clock (the fake mapping is the identity): passes the frame-age check.
        receive_time_ns=time.monotonic_ns(),
        transport_timestamp_us=1,
        user_timestamp_us=time.monotonic_ns() // 1000,
        frame_id=None,
        user_data=None,
    )


# --- recordable_livekit_streams --------------------------------------------------


def test_no_started_session_has_no_recordable_streams():
    assert recordable_livekit_streams(None) == ()


def test_only_monotonic_streams_with_full_geometry_are_recordable():
    keep = livekit_stream(16)
    frame_ready = livekit_stream(18, basis=MONO)
    streams = [
        keep,
        frame_ready,
        livekit_stream(17, basis=PROCESSING),
        livekit_stream(22, basis=UNIX_CAPTURE),
        livekit_stream(19, width=None),
        livekit_stream(20, height=None),
        livekit_stream(21, fps=None),
        jpeg_camera(),
    ]
    assert recordable_livekit_streams(SimpleNamespace(streams=streams)) == (keep, frame_ready)


# --- LiveKitRecordingSource ------------------------------------------------------


class FakeClockSync:
    """Identity mapping once ``ready``; ``mapping()`` is None before that, like a publisher not yet answering."""

    discontinuity = False

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def mapping(self):
        if not self.ready:
            return None
        return ClockMapping(epoch=1, offset_ns=0, rtt_ns=2_000, uncertainty_ns=1_000, accepted_at_ns=0)


class FakeReceiver:
    """Per-stream queues; a ``None`` item is a lost track, an empty queue is a quiet camera."""

    instances: list[FakeReceiver] = []
    start_error: Exception | None = None
    #: When set, ``start``/``stop`` wait on it, so a test can stall the join or the leave.
    start_gate: asyncio.Event | None = None
    stop_gate: asyncio.Event | None = None
    mapping_ready = True

    def __init__(self, streams, token, clock_sync=False):
        self.clock_sync_requested = clock_sync
        self.clock_sync = FakeClockSync(FakeReceiver.mapping_ready) if clock_sync else None
        self.streams, self.token = list(streams), token
        self.queues = {s.stream_id: asyncio.Queue() for s in streams}
        self.events: list[str] = []
        self.stops = 0
        self.counters = defaultdict(StreamCounters)
        FakeReceiver.instances.append(self)

    async def start(self):
        self.events.append("start")
        if FakeReceiver.start_gate is not None:
            await FakeReceiver.start_gate.wait()
        if FakeReceiver.start_error is not None:
            raise FakeReceiver.start_error

    async def stop(self):
        self.events.append("stop")
        if FakeReceiver.stop_gate is not None:
            await FakeReceiver.stop_gate.wait()
        self.stops += 1

    async def next_frame_or_lost(self, stream_id, timeout=None):
        return await self.queues[stream_id].get()

    def clear_queued(self):
        self.events.append("clear")
        for queue in self.queues.values():
            while not queue.empty():
                queue.get_nowait()


class FakeIngress:
    def __init__(self):
        self.offered = []

    def offer(self, item):
        self.offered.append(item)
        return True


@pytest.fixture(autouse=True)
def fake_receiver(monkeypatch):
    FakeReceiver.instances = []
    FakeReceiver.start_error = None
    FakeReceiver.start_gate = FakeReceiver.stop_gate = None
    FakeReceiver.mapping_ready = True
    monkeypatch.setattr(lrec, "LiveKitFrameReceiver", FakeReceiver)


def make_source(streams, *, timeout=1.0, tokens=None, **kwargs):
    tokens = tokens if tokens is not None else []
    return LiveKitRecordingSource(
        lambda: streams,
        lambda room: tokens.append(room) or f"token-{room}",
        first_frame_timeout_s=timeout,
        **kwargs,
    )


def preload(streams, **frame_kw):
    for stream in streams:
        FakeReceiver.instances[-1].queues[stream.stream_id].put_nowait(frame(stream.stream_id, **frame_kw))


async def prepare_with_frames(source, streams, **frame_kw):
    """Runs prepare() while feeding every stream one first frame as the receiver appears."""
    task = asyncio.ensure_future(source.prepare())
    while not FakeReceiver.instances:
        await asyncio.sleep(0)
    preload(streams, **frame_kw)
    _ = await task


async def test_prepare_with_no_recordable_stream_does_nothing():
    source = make_source([])
    await source.prepare()
    source.start(FakeIngress(), lambda message: None)
    await source.stop()
    assert FakeReceiver.instances == []


async def test_prepare_joins_with_a_token_for_the_first_streams_room_and_takes_every_first_frame():
    streams = [livekit_stream(16, room="room-a"), livekit_stream(17, room="room-b")]
    tokens: list[str] = []
    source = make_source(streams, tokens=tokens)
    await prepare_with_frames(source, streams)
    receiver = FakeReceiver.instances[0]
    assert tokens == ["room-a"]
    assert receiver.token == "token-room-a"
    assert receiver.clock_sync_requested
    assert receiver.events == ["start"]
    assert all(q.empty() for q in receiver.queues.values())
    await source.stop()


async def test_a_wrong_sized_first_frame_fails_naming_the_stream_and_leaves_the_room():
    streams = [livekit_stream(16)]
    source = make_source(streams)
    with pytest.raises(RuntimeError, match=r"cam16: frames are 8x2, expected 6x4"):
        await prepare_with_frames(source, streams, width=8, height=2)
    assert FakeReceiver.instances[0].stops == 1


async def test_a_lost_track_fails_naming_the_stream_and_leaves_the_room():
    streams = [livekit_stream(16)]
    source = make_source(streams)
    task = asyncio.ensure_future(source.prepare())
    while not FakeReceiver.instances:
        await asyncio.sleep(0)
    FakeReceiver.instances[0].queues[16].put_nowait(None)
    with pytest.raises(RuntimeError, match="cam16: the LiveKit track was lost"):
        _ = await task
    assert FakeReceiver.instances[0].stops == 1


async def test_a_quiet_stream_times_out_and_leaves_the_room():
    source = make_source([livekit_stream(16)], timeout=0.01)
    with pytest.raises(RuntimeError, match="sent no frame"):
        await source.prepare()
    assert FakeReceiver.instances[0].stops == 1


async def test_no_clock_mapping_before_the_timeout_fails_naming_clock_sync_and_leaves_the_room():
    FakeReceiver.mapping_ready = False
    streams = [livekit_stream(16)]
    source = make_source(streams, timeout=0.2)
    with pytest.raises(RuntimeError, match="clock-sync"):
        await prepare_with_frames(source, streams)
    assert FakeReceiver.instances[0].stops == 1


async def test_a_mapping_that_arrives_late_still_lets_the_take_start():
    FakeReceiver.mapping_ready = False
    streams = [livekit_stream(16)]
    source = make_source(streams, timeout=2.0)
    task = asyncio.ensure_future(prepare_with_frames(source, streams))
    await asyncio.sleep(0.1)
    assert not task.done()
    FakeReceiver.instances[0].clock_sync.ready = True
    _ = await task
    await source.stop()


async def test_a_mapping_over_the_uncertainty_budget_refuses_the_start_with_the_reason(monkeypatch):
    monkeypatch.setenv(lrec.BUDGET_ENV, "0.0005")  # 500 ns, below the fake mapping's 1000 ns
    streams = [livekit_stream(16)]
    source = make_source(streams, timeout=0.2)
    with pytest.raises(RuntimeError, match=f"uncertain by .*{lrec.BUDGET_ENV}"):
        await prepare_with_frames(source, streams)
    assert FakeReceiver.instances[0].stops == 1


async def test_a_mapping_that_already_jumped_refuses_the_start_with_the_reason(monkeypatch):
    monkeypatch.setattr(FakeClockSync, "discontinuity", True)
    streams = [livekit_stream(16)]
    source = make_source(streams, timeout=0.2)
    with pytest.raises(RuntimeError, match="jumped"):
        await prepare_with_frames(source, streams)
    assert FakeReceiver.instances[0].stops == 1


async def test_a_bad_budget_setting_fails_prepare_before_the_room_is_joined(monkeypatch):
    monkeypatch.setenv(lrec.BUDGET_ENV, "not-a-number")
    source = make_source([livekit_stream(16)])
    with pytest.raises(ValueError, match=lrec.BUDGET_ENV):
        await source.prepare()
    assert FakeReceiver.instances == []


async def test_a_failing_join_leaves_the_room_and_propagates():
    FakeReceiver.start_error = ConnectionError("no route")
    source = make_source([livekit_stream(16)])
    with pytest.raises(ConnectionError, match="no route"):
        await source.prepare()
    assert FakeReceiver.instances[0].stops == 1


async def test_cancelling_prepare_leaves_the_room():
    source = make_source([livekit_stream(16)], timeout=30.0)
    task = asyncio.ensure_future(source.prepare())
    while not FakeReceiver.instances:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task
    assert FakeReceiver.instances[0].stops == 1


async def test_start_drops_the_backlog_before_feeding_and_names_sources_after_the_streams():
    streams = [livekit_stream(16)]
    source = make_source(streams)
    await prepare_with_frames(source, streams)
    receiver = FakeReceiver.instances[0]
    receiver.queues[16].put_nowait(frame(16))  # acquired while storage was opening
    ingress = FakeIngress()
    source.start(ingress, lambda message: None)
    await asyncio.sleep(0)
    assert "clear" in receiver.events
    assert ingress.offered == []
    receiver.queues[16].put_nowait(frame(16))
    for _ in range(5):
        await asyncio.sleep(0)
    assert [m.source for m in ingress.offered] == ["cam16"]
    await source.stop()


async def test_a_feed_failure_reaches_the_failure_callback():
    streams = [livekit_stream(16)]
    source = make_source(streams)
    await prepare_with_frames(source, streams)
    reasons: list[str] = []
    source.start(FakeIngress(), reasons.append)
    FakeReceiver.instances[0].queues[16].put_nowait(None)
    for _ in range(5):
        await asyncio.sleep(0)
    assert reasons == ["cam16: the LiveKit track was lost"]
    await source.stop()


async def test_stop_is_idempotent_and_safe_before_prepare():
    source = make_source([livekit_stream(16)])
    await source.stop()
    streams = [livekit_stream(16)]
    await prepare_with_frames(source, streams)
    source.start(FakeIngress(), lambda message: None)
    await source.stop()
    await source.stop()
    assert FakeReceiver.instances[0].stops == 1


async def test_a_stalled_leave_does_not_block_teardown_and_finishes_in_the_background():
    streams = [livekit_stream(16)]
    source = make_source(streams, leave_timeout_s=0.01)
    await prepare_with_frames(source, streams)
    FakeReceiver.stop_gate = asyncio.Event()
    await asyncio.wait_for(source.stop(), 1)
    receiver = FakeReceiver.instances[0]
    assert receiver.stops == 0
    FakeReceiver.stop_gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert receiver.stops == 1


async def test_a_late_join_fails_the_start_without_cancelling_it_and_leaves_once_it_ends():
    FakeReceiver.start_gate = asyncio.Event()
    source = make_source([livekit_stream(16)], join_timeout_s=0.01)
    with pytest.raises(RuntimeError, match="joining the LiveKit room took longer"):
        await source.prepare()
    receiver = FakeReceiver.instances[0]
    # The join is still running and has not been cancelled or disconnected underneath.
    assert receiver.events == ["start"]
    FakeReceiver.start_gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert receiver.events == ["start", "stop"]
    await source.stop()
    assert receiver.stops == 1


async def test_cancelling_prepare_during_the_join_does_not_cancel_the_join():
    FakeReceiver.start_gate = asyncio.Event()
    source = make_source([livekit_stream(16)])
    task = asyncio.ensure_future(source.prepare())
    while not FakeReceiver.instances:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await task
    receiver = FakeReceiver.instances[0]
    assert receiver.events == ["start"]
    FakeReceiver.start_gate.set()
    for _ in range(5):
        await asyncio.sleep(0)
    assert receiver.events == ["start", "stop"]


# --- timing_evidence ---------------------------------------------------------------


def history_of(count: int) -> list[ClockMapping]:
    return [
        ClockMapping(
            epoch=1 + i // 10, offset_ns=-i, rtt_ns=2_000 + i, uncertainty_ns=1_000 + i, accepted_at_ns=100 * i
        )
        for i in range(count)
    ]


def clock_sync_with(history, *, discontinuity=False, rejected=0, current=True, trimmed=0):
    return SimpleNamespace(
        estimator=SimpleNamespace(history=history, trimmed_mappings=trimmed),
        discontinuity=discontinuity,
        rejected=rejected,
        mapping=lambda: history[-1] if current else None,
    )


async def prepared_source(streams, **kwargs):
    source = make_source(streams, **kwargs)
    await prepare_with_frames(source, streams)
    return source, FakeReceiver.instances[0]


def test_a_source_that_never_prepared_has_no_timing_evidence():
    assert make_source([livekit_stream(16)]).timing_evidence() is None


async def test_without_clock_sync_there_is_no_timing_evidence():
    streams = [livekit_stream(16)]
    source, receiver = await prepared_source(streams)
    receiver.clock_sync = None
    assert source.timing_evidence() is None


async def test_timing_evidence_describes_the_mappings_the_take_was_placed_under(monkeypatch):
    monkeypatch.delenv(lrec.BUDGET_ENV, raising=False)
    streams = [livekit_stream(16), livekit_stream(17, basis=MONO)]
    source, receiver = await prepared_source(streams)
    history = history_of(3)
    receiver.clock_sync = clock_sync_with(history, discontinuity=True, rejected=4, current=False)

    evidence = source.timing_evidence()

    assert evidence == {
        "clock_mapping": {
            "method": "four_timestamp_exchange",
            "local_clock": "pc_monotonic",
            "remote_clock": "publisher_monotonic",
            "streams": [
                {
                    "name": "cam16",
                    "timestamp_basis": "SENSOR_CAPTURE_MONOTONIC_US",
                    "time_meaning": "sensor_capture",
                },
                {"name": "cam17", "timestamp_basis": "PUBLISHER_MONOTONIC_US", "time_meaning": "frame_ready"},
            ],
            "mappings": [
                {
                    "epoch": m.epoch,
                    "offset_ns": m.offset_ns,
                    "rtt_ns": m.rtt_ns,
                    "uncertainty_ns": m.uncertainty_ns,
                    "accepted_at_ns": m.accepted_at_ns,
                }
                for m in history
            ],
            "mappings_omitted_oldest": 0,
            "uncertainty_budget_ns": 5_000_000,
            "discontinuity": True,
            "exchanges_rejected": 4,
            "mapping_current_at_stop": False,
        },
        "event_time": {
            "cam16": {"meaning": "sensor_capture", "qualified": True},
            "cam17": {"meaning": "frame_ready", "qualified": False},
        },
        "tracking_semantics": lrec._TRACKING_SEMANTICS,
    }
    assert json.loads(json.dumps(evidence)) == evidence


async def test_event_time_is_keyed_by_recorded_source_and_qualifies_only_sensor_capture(monkeypatch):
    streams = [livekit_stream(16), livekit_stream(17, basis=MONO)]
    source, receiver = await prepared_source(streams)
    receiver.clock_sync = clock_sync_with(history_of(1))
    event_time = source.timing_evidence()["event_time"]
    # Recorded sources are named after their streams; the key is that source name, never a stream id.
    assert {k: v["qualified"] for k, v in event_time.items()} == {
        "cam16": True,
        "cam17": False,
    }
    assert event_time["cam16"]["meaning"] == "sensor_capture" and event_time["cam17"]["meaning"] == "frame_ready"
    assert 16 not in event_time and 17 not in event_time


def test_a_tracking_latency_makes_that_source_a_marked_estimate_and_others_stay_unqualified():
    event_time = lrec.tracking_event_time({"pico_controller": 0.02})
    assert event_time["pico_controller"] == {
        "meaning": "host_poll_minus_latency",
        "estimated": True,
        "latency_s": 0.02,
        "qualified": True,
    }
    assert event_time["manus_glove"] == {"meaning": "host_poll_time", "qualified": False}


def test_a_zero_latency_is_still_a_qualified_estimate():
    assert lrec.tracking_event_time({"manus_glove": 0.0})["manus_glove"]["latency_s"] == 0.0
    assert lrec.tracking_event_time({"manus_glove": 0.0})["manus_glove"]["qualified"] is True


async def test_the_sources_timing_evidence_lists_only_its_cameras_not_tracking():
    source, receiver = await prepared_source([livekit_stream(16)])
    receiver.clock_sync = clock_sync_with(history_of(1))
    assert set(source.timing_evidence()["event_time"]) == {"cam16"}


async def test_timing_evidence_records_the_configured_uncertainty_budget(monkeypatch):
    monkeypatch.setenv(lrec.BUDGET_ENV, "2.5")
    source, receiver = await prepared_source([livekit_stream(16)])
    receiver.clock_sync = clock_sync_with(history_of(1))
    assert source.timing_evidence()["clock_mapping"]["uncertainty_budget_ns"] == 2_500_000


async def test_timing_evidence_carries_only_the_newest_mappings():
    source, receiver = await prepared_source([livekit_stream(16)])
    history = history_of(lrec._MAX_EVIDENCE_MAPPINGS + 36)
    receiver.clock_sync = clock_sync_with(history)

    clock_mapping = source.timing_evidence()["clock_mapping"]
    mappings = clock_mapping["mappings"]

    assert [m["accepted_at_ns"] for m in mappings] == [
        m.accepted_at_ns for m in history[-lrec._MAX_EVIDENCE_MAPPINGS :]
    ]
    assert clock_mapping["mappings_omitted_oldest"] == 36


async def test_timing_evidence_counts_mappings_the_estimator_already_trimmed():
    source, receiver = await prepared_source([livekit_stream(16)])
    receiver.clock_sync = clock_sync_with(history_of(lrec._MAX_EVIDENCE_MAPPINGS + 4), trimmed=10)

    assert source.timing_evidence()["clock_mapping"]["mappings_omitted_oldest"] == 14


async def test_an_exchange_reported_again_with_more_uncertainty_is_listed_once_as_first_reported():
    source, receiver = await prepared_source([livekit_stream(16)])
    first = ClockMapping(epoch=1, offset_ns=5, rtt_ns=2_000, uncertainty_ns=1_000, accepted_at_ns=100)
    later_report = dataclasses.replace(first, uncertainty_ns=1_500)
    newer = ClockMapping(epoch=1, offset_ns=6, rtt_ns=2_000, uncertainty_ns=1_000, accepted_at_ns=200)
    other_epoch = dataclasses.replace(first, epoch=2, offset_ns=9)
    receiver.clock_sync = clock_sync_with([first, later_report, newer, other_epoch])

    mappings = source.timing_evidence()["clock_mapping"]["mappings"]

    assert [(m["epoch"], m["accepted_at_ns"], m["uncertainty_ns"]) for m in mappings] == [
        (1, 100, 1_000),
        (2, 100, 1_000),
        (1, 200, 1_000),
    ]
