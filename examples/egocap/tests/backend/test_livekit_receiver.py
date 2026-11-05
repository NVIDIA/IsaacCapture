# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LiveKitFrameReceiver against a fake ``livekit.rtc``: subscription, frame accounting, queues, teardown."""

from __future__ import annotations

import asyncio
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

    def __init__(self, kind: str = "video") -> None:
        self.kind = kind
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
        return await self.track.events.get()

    async def aclose(self) -> None:
        self.closed += 1


class FakeRoom:
    """Records handlers and connect/disconnect calls; ``emit`` plays an SDK event."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.connected_with: tuple[str, str] | None = None
        self.disconnects = 0
        self.remote_participants: dict[str, object] = {}

    def on(self, name: str, handler: object) -> None:
        self.handlers[name] = handler

    async def connect(self, url: str, token: str) -> None:
        self.connected_with = (url, token)

    async def disconnect(self) -> None:
        self.disconnects += 1

    def emit(self, name: str, *args: object) -> None:
        self.handlers[name](*args)


FAKE_RTC = SimpleNamespace(
    Room=FakeRoom,
    VideoStream=FakeVideoStream,
    TrackKind=SimpleNamespace(KIND_VIDEO="video"),
    VideoBufferType=SimpleNamespace(RGB24="rgb24"),
)


def participant(identity: str = PUBLISHER, publications: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, track_publications=publications or {})


def publication(name: str, track: FakeTrack | None) -> SimpleNamespace:
    return SimpleNamespace(name=name, track=track)


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
    rx = lr.LiveKitFrameReceiver([camera(1, "left"), camera(2, "right")], TOKEN)
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
        assert stream.capacity == lr._SDK_QUEUE_FRAMES and stream.format == "rgb24"
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
