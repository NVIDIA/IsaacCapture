# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Receives the LiveKit camera video a session offers, for recording.

One room connection serves every offered stream. Each frame arrives with the publisher's own metadata (the
application timestamp, the frame id and the raw user data) and the time this process took it from the SDK, and is
put on a bounded per-stream queue. Nothing here converts those publisher times to this machine's clock or writes
anything: that belongs to the recorder. A full queue never blocks the receiver or hides a loss; the frame is
dropped and counted, and a jump in the publisher's frame id is counted as missing frames.

This is separate from the browser's own connection: closing the browser does not touch this one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from capture_server import protocol as p

try:
    from livekit import rtc
except ImportError:  # the optional "livekit" extra is not installed
    rtc = None

logger = logging.getLogger("isaaccapture.egocap.livekit_receiver")

LIVEKIT_MISSING = "receiving LiveKit video needs the livekit package: install the egocap 'livekit' extra"


def livekit_available() -> bool:
    """Whether the optional ``livekit`` package is installed, so LiveKit video can be received."""
    return rtc is not None


#: Frames a stream may hold between the SDK and the consumer before further frames are dropped and counted.
#: Decoded frames are megabytes each, so the queues are short and sized into the recorder's memory budget.
DEFAULT_QUEUE_FRAMES = 2
#: The SDK keeps only this many undelivered frames per track and drops the oldest silently; its drops show up
#: as a jump in the publisher's frame id, which is counted below.
SDK_QUEUE_FRAMES = 2
#: Frames one stream can pin at once with default queues: both queues plus the one the reader is converting.
RECEIVER_FRAMES_PER_STREAM = DEFAULT_QUEUE_FRAMES + SDK_QUEUE_FRAMES + 1


@dataclass(frozen=True)
class ReceivedFrame:
    """One decoded frame and everything the publisher attached to it."""

    stream_id: int
    #: Height x width x 3, uint8, RGB. Views the SDK frame's own buffer, which the SDK builds per frame and never
    #: reuses, so no further copy is made here.
    image: np.ndarray
    #: ``time.monotonic_ns()`` of this process when the frame was taken from the SDK, not a capture time.
    receive_time_ns: int
    #: The WebRTC event timestamp, which the SDK documents as different from the publisher's own metadata.
    transport_timestamp_us: int
    #: The publisher's application timestamp; its clock is named by the stream's ``timestamp_basis``.
    user_timestamp_us: int | None
    frame_id: int | None
    user_data: bytes | None


@dataclass
class StreamCounters:
    """What one stream received and what was lost on the way to the consumer."""

    received: int = 0
    #: Frames dropped because the consumer's queue was full.
    queue_overflow: int = 0
    #: Frames the publisher numbered but that never arrived (SDK drops, network loss).
    missing_frames: int = 0
    #: Times the publisher's frame id went backwards or repeated: a restarted publisher.
    frame_id_resets: int = 0
    #: Whether the stream's track is currently subscribed.
    subscribed: bool = False
    last_frame_id: int | None = field(default=None, repr=False)


class LiveKitFrameReceiver:
    """Joins the room of the offered streams and queues each stream's frames until the consumer takes them."""

    def __init__(
        self,
        streams: Sequence[p.LiveKitCameraStream],
        token: str,
        *,
        queue_frames: int = DEFAULT_QUEUE_FRAMES,
    ) -> None:
        # The package is optional; fail here with the fix rather than later with an AttributeError.
        if rtc is None:
            raise RuntimeError(LIVEKIT_MISSING)
        if not streams:
            raise ValueError("at least one LiveKit camera stream is required")
        if len({(s.url, s.room, s.publisher) for s in streams}) != 1:
            raise ValueError("all streams of one receiver must share a URL, room and publisher")
        if queue_frames < 1:
            raise ValueError("queue_frames must be at least 1")
        self._streams = {s.stream_id: s for s in streams}
        self._by_track = {s.track: s for s in streams}
        self._url, self._publisher = streams[0].url, streams[0].publisher
        self._token = token
        self._room: rtc.Room | None = None
        self._queues = {s.stream_id: asyncio.Queue[ReceivedFrame](maxsize=queue_frames) for s in streams}
        self.counters = {s.stream_id: StreamCounters() for s in streams}
        # Set when a stream's track goes away or the room is lost, so a waiting consumer can fail instead of
        # waiting forever; cleared when the track is subscribed again.
        self._lost = {s.stream_id: asyncio.Event() for s in streams}
        self._readers: dict[int, asyncio.Task[None]] = {}
        # Set once the room connection is lost for good; the SDK's own reconnects do not set it.
        self.failed = asyncio.Event()

    async def start(self) -> None:
        """Connect and begin receiving; frames of tracks published before or after joining are both picked up."""
        room = rtc.Room()
        self._room = room
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_disconnected", self._on_participant_disconnected)
        room.on("disconnected", self._on_disconnected)
        _ = await room.connect(self._url, self._token)
        # Tracks published before we joined arrive here rather than through track_subscribed.
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track is not None:
                    self._attach(participant, publication.track, publication.name)

    async def stop(self) -> None:
        """Leave the room and stop every reader; frames still queued stay available to the consumer."""
        room, self._room = self._room, None
        for task in self._readers.values():
            task.cancel()
        for task in list(self._readers.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._readers.clear()
        if room is not None:
            await room.disconnect()
        for counters in self.counters.values():
            counters.subscribed = False

    async def next_frame(self, stream_id: int) -> ReceivedFrame:
        """The oldest queued frame of ``stream_id``, waiting for one if none is queued."""
        return await self._queues[stream_id].get()

    async def next_frame_or_lost(self, stream_id: int) -> ReceivedFrame | None:
        """The oldest queued frame of ``stream_id``, or None once its track is lost and nothing is queued.

        Frames already queued when the track goes away are still returned first, so a consumer drains what
        arrived before it learns of the loss.
        """
        queue = self._queues[stream_id]
        if not queue.empty():
            return queue.get_nowait()
        get = asyncio.ensure_future(queue.get())
        lost = asyncio.ensure_future(self._lost[stream_id].wait())
        try:
            await asyncio.wait({get, lost}, return_when=asyncio.FIRST_COMPLETED)
            # A frame that arrived together with the loss is still a frame the consumer is owed.
            return get.result() if get.done() else None
        finally:
            for task in (get, lost):
                if not task.done():
                    task.cancel()

    def clear_queued(self) -> None:
        """Drop every queued frame, so a take starts with frames acquired after it began."""
        for queue in self._queues.values():
            while not queue.empty():
                queue.get_nowait()

    def pending(self, stream_id: int) -> int:
        """How many frames of ``stream_id`` are queued for the consumer."""
        return self._queues[stream_id].qsize()

    def _attach(self, participant: rtc.RemoteParticipant, track: rtc.Track, name: str) -> None:
        """Start reading ``track`` if it is one of the offered streams, from the offered publisher."""
        stream = self._by_track.get(name)
        if participant.identity != self._publisher or stream is None or track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        # A republished track replaces the old reader; its frames must not interleave with the new ones.
        previous = self._readers.pop(stream.stream_id, None)
        if previous is not None:
            previous.cancel()
        self.counters[stream.stream_id].subscribed = True
        self._lost[stream.stream_id].clear()
        # A new publication restarts the publisher's numbering, so a jump from the old track is not a loss.
        self.counters[stream.stream_id].last_frame_id = None
        self._readers[stream.stream_id] = asyncio.get_running_loop().create_task(
            self._read(stream.stream_id, track)
        )

    def _detach(self, stream_id: int) -> None:
        """Stop reading ``stream_id``: its track went away."""
        task = self._readers.pop(stream_id, None)
        if task is not None:
            task.cancel()
        self._mark_lost(stream_id)

    def _mark_lost(self, stream_id: int) -> None:
        """Record that ``stream_id`` can deliver no more frames, without cancelling the calling reader task."""
        self.counters[stream_id].subscribed = False
        self._lost[stream_id].set()

    def _on_track_subscribed(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        self._attach(participant, track, publication.name)

    def _on_track_unsubscribed(
        self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        stream = self._by_track.get(publication.name)
        if stream is not None and participant.identity == self._publisher:
            self._detach(stream.stream_id)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._publisher:
            # Every offered stream, not only those already being read: one that never appeared is lost too.
            for stream_id in self._streams:
                self._detach(stream_id)

    def _on_disconnected(self, *_: object) -> None:
        # Not a reconnect (the SDK reports those separately): the room is gone, and the recorder must find out.
        for stream_id in self._streams:
            self._detach(stream_id)
        self.failed.set()

    async def _read(self, stream_id: int, track: rtc.Track) -> None:
        """Take frames of one track from the SDK, convert them, and queue them for the consumer."""
        counters = self.counters[stream_id]
        this_task = asyncio.current_task()
        video = rtc.VideoStream(track, capacity=SDK_QUEUE_FRAMES, format=rtc.VideoBufferType.RGB24)
        try:
            async for event in video:
                received_at = time.monotonic_ns()
                frame = event.frame
                image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
                metadata = event.metadata
                has_meta = metadata is not None
                frame_id = metadata.frame_id if has_meta and metadata.HasField("frame_id") else None
                self._account(counters, frame_id)
                item = ReceivedFrame(
                    stream_id=stream_id,
                    image=image,
                    receive_time_ns=received_at,
                    transport_timestamp_us=event.timestamp_us,
                    user_timestamp_us=metadata.user_timestamp
                    if has_meta and metadata.HasField("user_timestamp")
                    else None,
                    frame_id=frame_id,
                    user_data=bytes(metadata.user_data) if has_meta and metadata.HasField("user_data") else None,
                )
                try:
                    self._queues[stream_id].put_nowait(item)
                except asyncio.QueueFull:
                    # Never block the reader or skip silently: the consumer is behind, and the loss is counted.
                    counters.queue_overflow += 1
        except Exception:
            # A dead reader would leave the consumer waiting forever; marking the stream lost lets the take settle.
            # Only the reader still registered for the stream may do so: a replaced one is already superseded.
            logger.exception("LiveKit reader for stream %d failed", stream_id)
            if self._readers.get(stream_id) is this_task:
                self._readers.pop(stream_id)
                self._mark_lost(stream_id)
        finally:
            await video.aclose()

    @staticmethod
    def _account(counters: StreamCounters, frame_id: int | None) -> None:
        """Count a received frame, and any frames the publisher numbered that did not arrive before it."""
        counters.received += 1
        if frame_id is None:
            return
        last = counters.last_frame_id
        if last is not None:
            if frame_id <= last:
                # The publisher restarted its numbering: a new run, not a loss.
                counters.frame_id_resets += 1
            elif frame_id > last + 1:
                counters.missing_frames += frame_id - last - 1
        counters.last_frame_id = frame_id
