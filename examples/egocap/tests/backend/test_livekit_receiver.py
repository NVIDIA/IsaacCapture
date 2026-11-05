# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LiveKitFrameReceiver against a fake ``livekit.rtc``: subscription, frame accounting, queues, teardown."""

from __future__ import annotations

import asyncio
import itertools
import time
from types import SimpleNamespace

import numpy as np
import pytest
from livekit import rtc as real_rtc

from capture_server import protocol as p
from egocap_web import livekit_receiver as lr

URL = "ws://lk.test"
ROOM = "room"
PUBLISHER = "orin"
TOKEN = "tok"


class FakeTrack:
    """A remote video track; the test feeds its events through ``push``."""

    _sids = itertools.count(1)

    def __init__(self, kind: str = "video") -> None:
        self.kind = kind
        self.sid = f"TR_{next(FakeTrack._sids)}"
        self.events: asyncio.Queue = asyncio.Queue()

    def push(self, event: object) -> None:
        self.events.put_nowait(event)


class FakeVideoStream:
    """Async iterator over a track's pushed events that blocks (cancellably) while there are none."""

    instances: list[FakeVideoStream] = []

    def __init__(self, track: FakeTrack, capacity: int = 0, format: object = None) -> None:
        self.track, self.capacity, self.format = track, capacity, format
        self.closed = 0
        FakeVideoStream.instances.append(self)

    def __aiter__(self) -> FakeVideoStream:
        return self

    async def __anext__(self) -> object:
        item = await self.track.events.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self.closed += 1


class FakeRoom:
    """Records handlers and connect/disconnect calls; ``emit`` plays an SDK event."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.connected_with: tuple[str, str] | None = None
        self.connect_options: object = None
        self.disconnects = 0
        self.remote_participants: dict[str, object] = {}

    def on(self, name: str, handler: object) -> None:
        self.handlers[name] = handler

    async def connect(self, url: str, token: str, options: object = None) -> None:
        self.connected_with = (url, token)
        self.connect_options = options

    async def disconnect(self) -> None:
        self.disconnects += 1

    def emit(self, name: str, *args: object) -> None:
        self.handlers[name](*args)


FAKE_RTC = SimpleNamespace(
    Room=FakeRoom,
    VideoStream=FakeVideoStream,
    RoomOptions=lambda **fields: SimpleNamespace(**fields),
    TrackKind=SimpleNamespace(KIND_VIDEO="video"),
    VideoBufferType=SimpleNamespace(RGB24="rgb24"),
)


def participant(identity: str = PUBLISHER, publications: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, track_publications=publications or {})


def publication(name: str, track: FakeTrack | None, kind: str = "video") -> SimpleNamespace:
    """A remote publication; ``subscribed_calls`` records each ``set_subscribed`` the receiver makes."""
    calls: list[bool] = []
    return SimpleNamespace(name=name, track=track, kind=kind, subscribed_calls=calls, set_subscribed=calls.append)


def event(
    value: int = 0,
    *,
    frame_id: int | None = None,
    user_timestamp: int | None = None,
    user_data: bytes | None = None,
    no_metadata: bool = False,
    timestamp_us: int = 111,
    shape: tuple[int, int] = (2, 3),
) -> SimpleNamespace:
    height, width = shape
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    fields = {}
    if frame_id is not None:
        fields["frame_id"] = frame_id
    if user_timestamp is not None:
        fields["user_timestamp"] = user_timestamp
    if user_data is not None:
        fields["user_data"] = user_data
    metadata = None if no_metadata else real_rtc.FrameMetadata(**fields)
    frame = SimpleNamespace(data=memoryview(pixels), width=width, height=height)
    return SimpleNamespace(frame=frame, timestamp_us=timestamp_us, metadata=metadata, pixels=pixels)


def camera(stream_id: int, track: str, **overrides: str) -> p.LiveKitCameraStream:
    fields = {"url": URL, "room": ROOM, "publisher": PUBLISHER, **overrides}
    return p.LiveKitCameraStream(stream_id=stream_id, name=f"cam{stream_id}", track=track, **fields)


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeVideoStream.instances = []
    monkeypatch.setattr(lr, "rtc", FAKE_RTC)


@pytest.fixture
async def receiver():
    # Deeper than the default queues so these tests exercise accounting, not the memory-budget-sized back-pressure.
    rx = lr.LiveKitFrameReceiver([camera(1, "left"), camera(2, "right")], TOKEN, queue_frames=8)
    await rx.start()
    yield rx
    await rx.stop()


def room_of(rx: lr.LiveKitFrameReceiver) -> FakeRoom:
    return rx._room


def subscribe(rx: lr.LiveKitFrameReceiver, name: str = "left", who: str = PUBLISHER) -> FakeTrack:
    track = FakeTrack()
    room_of(rx).emit("track_subscribed", track, publication(name, track), participant(who))
    return track


# ---- constructor ----


def test_constructor_requires_a_stream():
    with pytest.raises(ValueError, match="at least one"):
        lr.LiveKitFrameReceiver([], TOKEN)


@pytest.mark.parametrize(
    "other",
    [camera(2, "b", room="other"), camera(2, "b", publisher="someone"), camera(2, "b", url="ws://other")],
)
def test_constructor_rejects_streams_from_different_room_publisher_or_url(other):
    with pytest.raises(ValueError, match="share a URL, room and publisher"):
        lr.LiveKitFrameReceiver([camera(1, "a"), other], TOKEN)


@pytest.mark.parametrize("queue_frames", [0, -1])
def test_constructor_rejects_queue_smaller_than_one(queue_frames):
    with pytest.raises(ValueError, match="queue_frames"):
        lr.LiveKitFrameReceiver([camera(1, "a")], TOKEN, queue_frames=queue_frames)


# ---- start and subscription ----


async def test_start_connects_with_url_and_token(receiver):
    assert room_of(receiver).connected_with == (URL, TOKEN)
    assert not receiver.failed.is_set()
    assert not receiver.counters[1].subscribed


async def test_start_adopts_tracks_published_before_joining(monkeypatch):
    left, right = FakeTrack(), FakeTrack()

    class Prejoined(FakeRoom):
        def __init__(self) -> None:
            super().__init__()
            self.remote_participants = {
                "x": participant(
                    PUBLISHER,
                    {
                        "a": publication("left", left),
                        "b": publication("right", right),
                        "c": publication("left", None),
                    },
                )
            }

    monkeypatch.setattr(lr, "rtc", SimpleNamespace(**{**vars(FAKE_RTC), "Room": Prejoined}))
    rx = lr.LiveKitFrameReceiver([camera(1, "left"), camera(2, "right")], TOKEN)
    await rx.start()
    try:
        assert rx.counters[1].subscribed and rx.counters[2].subscribed
        left.push(event(7, frame_id=1))
        assert (await asyncio.wait_for(rx.next_frame(1), 1)).frame_id == 1
        stream = FakeVideoStream.instances[0]
        assert stream.capacity == lr.SDK_QUEUE_FRAMES and stream.format == "rgb24"
    finally:
        await rx.stop()


async def test_track_subscribed_starts_reading(receiver):
    track = subscribe(receiver, "left")
    assert receiver.counters[1].subscribed and not receiver.counters[2].subscribed
    track.push(event(3, frame_id=1))
    assert (await asyncio.wait_for(receiver.next_frame(1), 1)).frame_id == 1


async def test_other_participant_is_ignored(receiver):
    track = subscribe(receiver, "left", who="intruder")
    track.push(event())
    await settle()
    assert not receiver.counters[1].subscribed
    assert receiver.pending(1) == 0 and not FakeVideoStream.instances


async def test_unoffered_track_name_is_ignored(receiver):
    track = subscribe(receiver, "depth")
    track.push(event())
    await settle()
    assert not any(c.subscribed for c in receiver.counters.values())
    assert not FakeVideoStream.instances


async def test_non_video_track_is_ignored(receiver):
    track = FakeTrack(kind="audio")
    room_of(receiver).emit("track_subscribed", track, publication("left", track), participant())
    assert not receiver.counters[1].subscribed


# ---- frames ----


async def test_frames_arrive_in_order_with_image_and_metadata(receiver):
    track = subscribe(receiver)
    before = time.monotonic_ns()
    track.push(event(10, frame_id=1, user_timestamp=1000, user_data=b"\x01\x02", timestamp_us=5, shape=(4, 6)))
    track.push(event(20, frame_id=2, user_timestamp=2000, user_data=b"", timestamp_us=6, shape=(4, 6)))
    first = await asyncio.wait_for(receiver.next_frame(1), 1)
    second = await asyncio.wait_for(receiver.next_frame(1), 1)
    after = time.monotonic_ns()
    assert first.stream_id == 1
    assert first.image.shape == (4, 6, 3) and first.image.dtype == np.uint8
    assert (first.image == 10).all() and (second.image == 20).all()
    assert (first.frame_id, second.frame_id) == (1, 2)
    assert (first.user_timestamp_us, second.user_timestamp_us) == (1000, 2000)
    assert (first.transport_timestamp_us, second.transport_timestamp_us) == (5, 6)
    assert first.user_data == b"\x01\x02" and second.user_data == b""
    assert before <= first.receive_time_ns <= second.receive_time_ns <= after
    assert receiver.counters[1].received == 2


async def test_absent_metadata_fields_are_none(receiver):
    track = subscribe(receiver)
    track.push(event(1, frame_id=4))
    frame = await asyncio.wait_for(receiver.next_frame(1), 1)
    assert frame.frame_id == 4
    assert frame.user_timestamp_us is None and frame.user_data is None


async def test_missing_metadata_object_gives_none_fields_and_no_loss_accounting(receiver):
    track = subscribe(receiver)
    track.push(event(1, no_metadata=True))
    track.push(event(2, no_metadata=True))
    frames = [await asyncio.wait_for(receiver.next_frame(1), 1) for _ in range(2)]
    assert all(f.frame_id is None and f.user_timestamp_us is None and f.user_data is None for f in frames)
    counters = receiver.counters[1]
    assert (counters.received, counters.missing_frames, counters.frame_id_resets) == (2, 0, 0)


async def test_successive_frames_hold_their_own_distinct_images(receiver):
    track = subscribe(receiver)
    for value, fid in ((9, 1), (50, 2), (200, 3)):
        track.push(event(value, frame_id=fid, shape=(2, 3)))
    frames = [await asyncio.wait_for(receiver.next_frame(1), 1) for _ in range(3)]
    for frame, value in zip(frames, (9, 50, 200)):
        assert frame.image.shape == (2, 3, 3) and frame.image.dtype == np.uint8
        assert (frame.image == value).all()
    assert not np.shares_memory(frames[0].image, frames[1].image)
    assert not np.shares_memory(frames[1].image, frames[2].image)


async def test_streams_are_queued_independently(receiver):
    subscribe(receiver, "left").push(event(1, frame_id=1))
    subscribe(receiver, "right").push(event(2, frame_id=1))
    assert (await asyncio.wait_for(receiver.next_frame(2), 1)).image[0, 0, 0] == 2
    assert (await asyncio.wait_for(receiver.next_frame(1), 1)).image[0, 0, 0] == 1


# ---- loss accounting ----


async def _drain(rx: lr.LiveKitFrameReceiver, count: int) -> None:
    for _ in range(count):
        await asyncio.wait_for(rx.next_frame(1), 1)


async def test_frame_id_jump_counts_missing_frames_but_first_frame_is_not_a_loss(receiver):
    track = subscribe(receiver)
    for fid in (100, 101, 104):
        track.push(event(frame_id=fid))
    await _drain(receiver, 3)
    counters = receiver.counters[1]
    assert counters.missing_frames == 2
    assert counters.frame_id_resets == 0 and counters.last_frame_id == 104


async def test_repeated_or_decreasing_frame_id_is_a_reset_not_a_loss(receiver):
    track = subscribe(receiver)
    for fid in (5, 6, 6, 2, 3):
        track.push(event(frame_id=fid))
    await _drain(receiver, 5)
    counters = receiver.counters[1]
    assert counters.frame_id_resets == 2
    assert counters.missing_frames == 0 and counters.last_frame_id == 3


# ---- bounded queue ----


async def test_full_queue_counts_overflow_keeps_oldest_and_never_blocks():
    rx = lr.LiveKitFrameReceiver([camera(1, "left")], TOKEN, queue_frames=2)
    await rx.start()
    try:
        track = subscribe(rx)
        for fid in range(1, 6):
            track.push(event(fid, frame_id=fid))
        await settle()
        counters = rx.counters[1]
        assert counters.received == 5 and counters.queue_overflow == 3
        assert counters.missing_frames == 0
        assert rx.pending(1) == 2
        kept = [await asyncio.wait_for(rx.next_frame(1), 1) for _ in range(2)]
        assert [f.frame_id for f in kept] == [1, 2]
        # The reader is still alive after overflow.
        track.push(event(9, frame_id=6))
        assert (await asyncio.wait_for(rx.next_frame(1), 1)).frame_id == 6
    finally:
        await rx.stop()


async def test_pending_reports_queued_counts(receiver):
    track = subscribe(receiver)
    assert receiver.pending(1) == 0 and receiver.pending(2) == 0
    for fid in (1, 2, 3):
        track.push(event(frame_id=fid))
    await settle()
    assert receiver.pending(1) == 3 and receiver.pending(2) == 0
    await receiver.next_frame(1)
    assert receiver.pending(1) == 2


# ---- detach and republish ----


async def test_track_unsubscribed_detaches_and_cancels_reader(receiver):
    track = subscribe(receiver)
    await settle()
    reader = receiver._readers[1]
    room_of(receiver).emit("track_unsubscribed", track, publication("left", track), participant())
    await settle()
    assert not receiver.counters[1].subscribed
    assert reader.cancelled() and 1 not in receiver._readers
    assert FakeVideoStream.instances[0].closed == 1


async def test_unsubscribe_of_other_participant_or_track_is_ignored(receiver):
    track = subscribe(receiver)
    room_of(receiver).emit("track_unsubscribed", track, publication("left", track), participant("intruder"))
    room_of(receiver).emit("track_unsubscribed", track, publication("depth", track), participant())
    assert receiver.counters[1].subscribed and 1 in receiver._readers


async def test_publisher_disconnect_detaches_every_stream(receiver):
    subscribe(receiver, "left")
    subscribe(receiver, "right")
    await settle()
    room_of(receiver).emit("participant_disconnected", participant("intruder"))
    assert receiver.counters[1].subscribed and receiver.counters[2].subscribed
    room_of(receiver).emit("participant_disconnected", participant())
    await settle()
    assert not receiver.counters[1].subscribed and not receiver.counters[2].subscribed
    assert not receiver._readers
    assert [s.closed for s in FakeVideoStream.instances] == [1, 1]
    assert not receiver.failed.is_set()


async def test_republication_replaces_reader_and_resets_frame_numbering(receiver):
    old = subscribe(receiver)
    old.push(event(1, frame_id=500))
    await _drain(receiver, 1)
    old_reader = receiver._readers[1]
    new = subscribe(receiver)
    await settle()
    assert old_reader.cancelled() and receiver._readers[1] is not old_reader
    assert receiver.counters[1].last_frame_id is None
    new.push(event(2, frame_id=1))
    frame = await asyncio.wait_for(receiver.next_frame(1), 1)
    assert frame.frame_id == 1
    counters = receiver.counters[1]
    assert counters.missing_frames == 0 and counters.frame_id_resets == 0 and counters.subscribed
    # The replaced track no longer feeds the queue.
    old.push(event(3, frame_id=501))
    await settle()
    assert receiver.pending(1) == 0


# ---- failure and stop ----


async def test_room_disconnected_sets_failed_and_detaches(receiver):
    subscribe(receiver, "left")
    subscribe(receiver, "right")
    room_of(receiver).emit("disconnected", "SOME_REASON")
    await settle()
    assert receiver.failed.is_set()
    assert not any(c.subscribed for c in receiver.counters.values())
    assert not receiver._readers


async def test_stop_cancels_readers_disconnects_once_and_keeps_queued_frames():
    rx = lr.LiveKitFrameReceiver([camera(1, "left")], TOKEN)
    await rx.start()
    room = room_of(rx)
    track = subscribe(rx)
    track.push(event(4, frame_id=1))
    await settle()
    reader = rx._readers[1]
    await rx.stop()
    await rx.stop()
    assert reader.cancelled() and not rx._readers
    assert room.disconnects == 1
    assert FakeVideoStream.instances[0].closed == 1
    assert not rx.counters[1].subscribed
    assert rx.pending(1) == 1
    assert (await asyncio.wait_for(rx.next_frame(1), 1)).frame_id == 1


async def test_stop_before_start_is_harmless():
    rx = lr.LiveKitFrameReceiver([camera(1, "left")], TOKEN)
    await rx.stop()
    assert not rx.failed.is_set()


# ---- next_frame_or_lost ----


def _only_test_tasks() -> set:
    return asyncio.all_tasks() - {asyncio.current_task()}


async def test_queued_frames_are_drained_before_none(receiver):
    track = subscribe(receiver)
    track.push(event(1, frame_id=1))
    track.push(event(2, frame_id=2))
    await settle()
    room_of(receiver).emit("track_unsubscribed", track, publication("left", track), participant())
    first = await asyncio.wait_for(receiver.next_frame_or_lost(1), 1)
    second = await asyncio.wait_for(receiver.next_frame_or_lost(1), 1)
    assert (first.frame_id, second.frame_id) == (1, 2)
    assert await asyncio.wait_for(receiver.next_frame_or_lost(1), 1) is None


async def test_frame_is_returned_while_the_track_is_live(receiver):
    track = subscribe(receiver)
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    assert not waiter.done()
    track.push(event(5, frame_id=9))
    assert (await asyncio.wait_for(waiter, 1)).frame_id == 9


async def test_loss_while_waiting_returns_none(receiver):
    track = subscribe(receiver)
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    assert not waiter.done()
    room_of(receiver).emit("track_unsubscribed", track, publication("left", track), participant())
    assert await asyncio.wait_for(waiter, 1) is None


def _unsubscribe(rx, track):
    room_of(rx).emit("track_unsubscribed", track, publication("left", track), participant())


def _publisher_leaves(rx, track):
    room_of(rx).emit("participant_disconnected", participant())


def _room_disconnected(rx, track):
    room_of(rx).emit("disconnected", "SOME_REASON")


@pytest.mark.parametrize("lose", [_unsubscribe, _publisher_leaves, _room_disconnected])
async def test_each_kind_of_loss_wakes_a_waiter_on_an_attached_stream(receiver, lose):
    track = subscribe(receiver)
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    lose(receiver, track)
    assert await asyncio.wait_for(waiter, 1) is None


async def test_room_disconnect_marks_a_never_attached_stream_lost(receiver):
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(2))
    await settle()
    _room_disconnected(receiver, None)
    assert await asyncio.wait_for(waiter, 1) is None


async def test_unsubscribe_of_a_never_attached_track_is_ignored(receiver):
    # Only the active track's unsubscribe is a loss; one for a track that was never read says nothing.
    track = FakeTrack()
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    _unsubscribe(receiver, track)
    await settle()
    assert not waiter.done()
    waiter.cancel()


async def test_publisher_leaving_marks_a_never_attached_stream_lost(receiver):
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    _publisher_leaves(receiver, None)
    assert await asyncio.wait_for(waiter, 1) is None


async def test_republication_clears_the_lost_state(receiver):
    track = subscribe(receiver)
    _unsubscribe(receiver, track)
    assert await asyncio.wait_for(receiver.next_frame_or_lost(1), 1) is None
    again = subscribe(receiver)
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    assert not waiter.done()
    again.push(event(4, frame_id=1))
    assert (await asyncio.wait_for(waiter, 1)).frame_id == 1


async def test_no_helper_tasks_are_left_after_returning(receiver):
    track = subscribe(receiver)
    await settle()
    baseline = _only_test_tasks()
    # Returned because of a frame.
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    track.push(event(1, frame_id=1))
    await asyncio.wait_for(waiter, 1)
    await settle()
    assert _only_test_tasks() == baseline
    # Returned because of the loss.
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    _unsubscribe(receiver, track)
    await asyncio.wait_for(waiter, 1)
    await settle()
    assert _only_test_tasks() - baseline == set()


async def test_cancelling_a_waiter_leaves_no_helper_tasks(receiver):
    subscribe(receiver)
    await settle()
    baseline = _only_test_tasks()
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await waiter
    await settle()
    assert _only_test_tasks() == baseline


async def test_publisher_leaving_before_any_track_marks_every_stream_lost(receiver):
    waiters = [asyncio.ensure_future(receiver.next_frame_or_lost(i)) for i in (1, 2)]
    await settle()
    assert not any(w.done() for w in waiters)
    _publisher_leaves(receiver, None)
    assert [await asyncio.wait_for(w, 1) for w in waiters] == [None, None]


async def test_a_different_participant_disconnecting_changes_nothing(receiver):
    track = subscribe(receiver)
    waiters = [asyncio.ensure_future(receiver.next_frame_or_lost(i)) for i in (1, 2)]
    await settle()
    room_of(receiver).emit("participant_disconnected", participant("intruder"))
    await settle()
    assert not any(w.done() for w in waiters)
    assert receiver.counters[1].subscribed
    track.push(event(3, frame_id=1))
    assert (await asyncio.wait_for(waiters[0], 1)).frame_id == 1
    waiters[1].cancel()


async def test_feed_on_a_never_attached_stream_fails_when_the_publisher_leaves(monkeypatch):
    from egocap_web.livekit_recording import LiveKitRecordingFeed

    capture = p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US
    stream = camera(1, "left", timestamp_basis=capture)
    rx = lr.LiveKitFrameReceiver([stream], TOKEN)
    await rx.start()
    stub = SimpleNamespace(mapping=lambda: None, discontinuity=False, stop=lambda: asyncio.sleep(0))
    rx.clock_sync = stub  # the feed needs a clock-sync object to exist
    reasons: list[str] = []
    feed = LiveKitRecordingFeed(
        rx, [stream], SimpleNamespace(offer=lambda item: True), {1: "src"}, on_failure=reasons.append
    )
    try:
        feed.start()
        await settle()
        assert feed.failure is None
        _publisher_leaves(rx, None)
        await settle()
        assert feed.failure == "cam1: the LiveKit track was lost"
        assert reasons == [feed.failure]
    finally:
        await feed.stop()
        await rx.stop()


async def test_clear_queued_drops_every_queued_frame(receiver):
    track = subscribe(receiver)
    track.push(event(3, frame_id=1))
    track.push(event(3, frame_id=2))
    await settle()
    assert receiver.pending(1) > 0
    receiver.clear_queued()
    assert receiver.pending(1) == receiver.pending(2) == 0


# ---- reader failure ----


async def test_a_failing_reader_marks_its_stream_lost_and_logs(receiver, caplog):
    track = subscribe(receiver, "left")
    bad = event(1, frame_id=1)
    bad.frame.height = 99  # the buffer no longer matches the stated size
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    track.push(bad)
    assert await asyncio.wait_for(waiter, 1) is None
    assert not receiver.counters[1].subscribed
    assert "reader for stream 1 failed" in caplog.text
    assert FakeVideoStream.instances[0].closed == 1


async def test_a_failing_reader_leaves_the_other_stream_running(receiver):
    left, right = subscribe(receiver, "left"), subscribe(receiver, "right")
    bad = event(1)
    bad.frame.width = 99
    left.push(bad)
    right.push(event(5, frame_id=1))
    assert (await asyncio.wait_for(receiver.next_frame(2), 1)).image[0, 0, 0] == 5
    assert receiver.counters[2].subscribed and not receiver.counters[1].subscribed


# ---- selective subscription ----


def _prejoined_rtc(monkeypatch, participants: dict):
    class Prejoined(FakeRoom):
        def __init__(self) -> None:
            super().__init__()
            self.remote_participants = participants

    monkeypatch.setattr(lr, "rtc", SimpleNamespace(**{**vars(FAKE_RTC), "Room": Prejoined}))


async def test_connect_disables_auto_subscribe(receiver):
    assert room_of(receiver).connect_options.auto_subscribe is False


async def test_only_the_offered_publication_is_subscribed_among_existing_ones(monkeypatch):
    offered = publication("left", None)
    unrelated = publication("depth", None)
    audio = publication("left", None, kind="audio")
    foreign = publication("left", None)
    _prejoined_rtc(
        monkeypatch,
        {
            "a": participant(PUBLISHER, {"1": offered, "2": unrelated, "3": audio}),
            "b": participant("intruder", {"4": foreign}),
        },
    )
    rx = lr.LiveKitFrameReceiver([camera(1, "left")], TOKEN)
    await rx.start()
    try:
        assert offered.subscribed_calls == [True]
        assert unrelated.subscribed_calls == [] and audio.subscribed_calls == [] and foreign.subscribed_calls == []
    finally:
        await rx.stop()


async def test_a_later_publication_is_subscribed_only_when_offered(receiver):
    room = room_of(receiver)
    offered = publication("right", None)
    unrelated = publication("depth", None)
    foreign = publication("right", None)
    audio = publication("right", None, kind="audio")
    room.emit("track_published", offered, participant())
    room.emit("track_published", unrelated, participant())
    room.emit("track_published", foreign, participant("intruder"))
    room.emit("track_published", audio, participant())
    assert offered.subscribed_calls == [True]
    assert unrelated.subscribed_calls == foreign.subscribed_calls == audio.subscribed_calls == []


# ---- replacement and stale events ----


async def test_stale_unsubscribe_after_replacement_does_not_detach_the_replacement(receiver):
    old = subscribe(receiver)
    new = subscribe(receiver)
    await settle()
    reader = receiver._readers[1]
    _unsubscribe(receiver, old)
    await settle()
    assert receiver.counters[1].subscribed and receiver._readers[1] is reader and not reader.done()
    new.push(event(1, frame_id=1))
    assert (await asyncio.wait_for(receiver.next_frame_or_lost(1), 1)).frame_id == 1


async def test_unsubscribe_of_the_active_track_still_detaches(receiver):
    subscribe(receiver)
    new = subscribe(receiver)
    await settle()
    _unsubscribe(receiver, new)
    await settle()
    assert not receiver.counters[1].subscribed and 1 not in receiver._readers and 1 not in receiver._active


async def test_replacement_counts_a_publication_change_and_drops_the_old_queue(receiver):
    old = subscribe(receiver)
    assert receiver.counters[1].publication_changes == 0  # the first attach is not a change
    for fid in (1, 2, 3):
        old.push(event(frame_id=fid))
    await settle()
    assert receiver.pending(1) == 3
    subscribe(receiver)
    assert receiver.counters[1].publication_changes == 1
    assert receiver.pending(1) == 0
    assert receiver.counters[1].received == 3


async def test_a_different_sid_without_a_reader_is_a_change_too(receiver):
    track = subscribe(receiver)
    reader = receiver._readers.pop(1)
    reader.cancel()
    other = FakeTrack()
    room_of(receiver).emit("track_subscribed", other, publication("left", other), participant())
    assert receiver.counters[1].publication_changes == 1
    assert receiver._active[1] == other.sid != track.sid


async def test_reattaching_after_a_detach_is_not_a_publication_change(receiver):
    track = subscribe(receiver)
    _unsubscribe(receiver, track)
    subscribe(receiver)
    assert receiver.counters[1].publication_changes == 0


# ---- the SDK stream ending ----


async def test_a_normally_ended_reader_marks_the_stream_lost_and_wakes_a_waiter(receiver):
    track = subscribe(receiver)
    track.push(event(1, frame_id=1))
    await settle()
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    assert (await asyncio.wait_for(waiter, 1)).frame_id == 1
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1))
    await settle()
    assert not waiter.done()
    track.push(StopAsyncIteration)  # the fake stream raises it from __anext__
    assert await asyncio.wait_for(waiter, 1) is None
    assert 1 not in receiver._readers


async def test_next_frame_or_lost_times_out_when_nothing_arrives(receiver):
    subscribe(receiver)
    await settle()
    baseline = _only_test_tasks()
    with pytest.raises(asyncio.TimeoutError):
        await receiver.next_frame_or_lost(1, timeout=0.01)
    await settle()
    assert _only_test_tasks() == baseline


async def test_timeout_does_not_fire_when_a_frame_arrives_in_time(receiver):
    track = subscribe(receiver)
    waiter = asyncio.ensure_future(receiver.next_frame_or_lost(1, timeout=1))
    await settle()
    track.push(event(1, frame_id=1))
    assert (await asyncio.wait_for(waiter, 1)).frame_id == 1


# ---- frame-id stride ----


async def _ids(rx, ids):
    track = subscribe(rx)
    for fid in ids:
        track.push(event(frame_id=fid))
    await _drain(rx, len(ids))
    return rx.counters[1]


async def test_a_constant_stride_of_two_is_not_loss(receiver):
    counters = await _ids(receiver, (2, 4, 6, 8))
    assert counters.missing_frames == 0 and counters.id_step == 2


async def test_one_dropped_frame_at_stride_two_is_one_missing(receiver):
    counters = await _ids(receiver, (2, 4, 6, 10, 12))
    assert counters.missing_frames == 1


async def test_a_gap_of_three_at_stride_one_is_two_missing(receiver):
    counters = await _ids(receiver, (1, 2, 5))
    assert counters.missing_frames == 2 and counters.id_step == 1


async def test_a_reset_is_counted_and_not_loss_with_a_stride(receiver):
    counters = await _ids(receiver, (10, 12, 14, 2, 4))
    assert counters.frame_id_resets == 1 and counters.missing_frames == 0


async def test_the_step_is_learned_from_the_smallest_delta(receiver):
    # A first gap of two steps looks like stride 4 until a smaller step is seen; later gaps use the smaller one.
    counters = await _ids(receiver, (4, 8, 10, 12))
    assert counters.id_step == 2 and counters.missing_frames == 0
