# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Feeds received LiveKit camera frames into a take's recording ingress.

A frame is recorded only when its timestamp is a capture time (``SENSOR_CAPTURE_UNIX_US``), which can then be put
on this machine's monotonic clock, the same clock the tracking samples use. A stream stamped with the publisher's
monotonic clock or with its processing time is not a capture time, so the feed stops with that reason instead of
recording a guessed one. Arrival time is never used as capture time.

This assumes the backend runs on the machine whose monotonic clock CaptureServer reports: tracking times come from
CaptureServer, and the two agree only when both processes share a clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from capture_server import protocol as p
from recording_service import PixelFormat, VideoSample

from .livekit_receiver import LiveKitFrameReceiver
from .recording_adapter import RecordingIngress

logger = logging.getLogger("isaaccapture.egocap.livekit_recording")

#: How long a measured pair of clock readings is reused. Unix and monotonic time only drift apart slowly, and a
#: step of the Unix clock is caught within this interval.
_PAIR_MAX_AGE_S = 1.0
#: Readings per pair; the one read in the shortest span is the most trustworthy.
_PAIR_ATTEMPTS = 3


class UnixClockMap:
    """Converts Unix-time microseconds to this process's monotonic seconds, from a recently measured pair.

    The pair is two clock readings taken back to back and refreshed about once a second, from the consumer's
    thread and never from an SDK callback. It converts on this machine only: it does not detect a step of the
    Unix clock, and says nothing about whether the publisher's clock agrees with this one.
    """

    def __init__(
        self,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        unix_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._monotonic_ns = monotonic_ns
        self._unix_ns = unix_ns
        self._offset_s = 0.0
        self._measured_at_ns: int | None = None

    def monotonic_seconds(self, unix_us: int) -> float:
        """The monotonic-clock time, in seconds, of the Unix time ``unix_us`` (microseconds)."""
        now = self._monotonic_ns()
        if self._measured_at_ns is None or now - self._measured_at_ns > _PAIR_MAX_AGE_S * 1e9:
            self._measure()
        return unix_us / 1e6 + self._offset_s

    def _measure(self) -> None:
        """Take the pair read in the shortest span: the Unix reading is taken between two monotonic ones."""
        best: tuple[int, float] | None = None
        for _ in range(_PAIR_ATTEMPTS):
            before = self._monotonic_ns()
            unix = self._unix_ns()
            after = self._monotonic_ns()
            span = after - before
            if best is None or span < best[0]:
                # The Unix reading happened somewhere in [before, after]; its midpoint is the best estimate.
                best = (span, ((before + after) / 2 - unix) / 1e9)
        assert best is not None
        self._offset_s = best[1]
        self._measured_at_ns = self._monotonic_ns()


class LiveKitRecordingFeed:
    """Moves every offered stream's frames from the receiver into the ingress of one take.

    Stops, with the reason in ``failure``, at the first frame that cannot be recorded honestly or when the ingress
    stops accepting; the recorder settles the take from that, so a source never goes missing silently.
    """

    def __init__(
        self,
        receiver: LiveKitFrameReceiver,
        streams: Sequence[p.LiveKitCameraStream],
        ingress: RecordingIngress,
        source_names: Mapping[int, str],
        *,
        clock_map: UnixClockMap | None = None,
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        missing = [s.name for s in streams if s.stream_id not in source_names]
        if missing:
            raise ValueError(f"no recording source for LiveKit stream(s): {missing}")
        self._receiver = receiver
        self._streams = {s.stream_id: s for s in streams}
        self._ingress = ingress
        self._source_names = dict(source_names)
        self._clock_map = clock_map or UnixClockMap()
        self._on_failure = on_failure
        self._tasks: list[asyncio.Task[None]] = []
        self.failure: str | None = None

    def start(self) -> None:
        """Begin feeding; one task per stream so a slow stream never delays another."""
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._run(stream_id)) for stream_id in self._streams]

    async def stop(self) -> None:
        """Stop feeding. Frames already offered to the ingress stay with it."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    def _fail(self, reason: str) -> None:
        """Record the first reason feeding had to stop, once, and tell the recorder."""
        if self.failure is None:
            self.failure = reason
            logger.error("LiveKit recording feed stopped: %s", reason)
            if self._on_failure is not None:
                self._on_failure(reason)

    async def _run(self, stream_id: int) -> None:
        """Feed one stream until the take fails or the feed is stopped."""
        stream = self._streams[stream_id]
        source = self._source_names[stream_id]
        if stream.timestamp_basis is not p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US:
            # Checked before any frame arrives: this stream cannot become recordable by waiting.
            self._fail(
                f"{stream.name}: its timestamps are not a capture time, so it cannot be aligned with tracking"
            )
            return
        while self.failure is None:
            frame = await self._receiver.next_frame_or_lost(stream_id)
            if frame is None:
                # Whatever was queued has been taken; the camera is gone, and a recorded source must not vanish.
                self._fail(f"{stream.name}: the LiveKit track was lost")
                return
            if frame.user_timestamp_us is None:
                self._fail(f"{stream.name}: a frame arrived without the publisher's capture timestamp")
                return
            capture_time = self._clock_map.monotonic_seconds(frame.user_timestamp_us)
            sample = VideoSample(
                source=source,
                image=frame.image,
                pixel_format=PixelFormat.RGB8,
                capture_time=capture_time,
                receive_time=frame.receive_time_ns / 1e9,
            )
            if not self._ingress.offer(sample):
                # The ingress refuses once it has failed or stopped; it reports its own reason.
                self._fail(f"{stream.name}: the recording ingress is no longer accepting frames")
                return


def recordable_livekit_streams(started: Any) -> tuple[p.LiveKitCameraStream, ...]:
    """The LiveKit streams of a started session that a recording may include.

    A stream is recordable when its timestamps are a capture time and its geometry is configured: the recording
    schema fixes each source's size and rate when the session opens. The rest remain preview-only.
    """
    if started is None:
        return ()
    return tuple(
        s
        for s in started.streams
        if isinstance(s, p.LiveKitCameraStream)
        and s.timestamp_basis is p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US
        and None not in (s.width, s.height, s.fps)
    )


class LiveKitRecordingSource:
    """The recorder's LiveKit video for one take: joins before the take, feeds during it, leaves after it.

    The recordable streams are read when the take is prepared, because the session offers them only once it has
    started. The first frame of each stream is checked against its configured size. The browser's own connection
    is separate and unaffected.
    """

    def __init__(
        self,
        recordable_streams: Callable[[], Sequence[p.LiveKitCameraStream]],
        make_token: Callable[[str], str],
        *,
        first_frame_timeout_s: float = 10.0,
    ) -> None:
        self._recordable_streams = recordable_streams
        self._make_token = make_token
        self._first_frame_timeout_s = first_frame_timeout_s
        self._streams: tuple[p.LiveKitCameraStream, ...] = ()
        self._receiver: LiveKitFrameReceiver | None = None
        self._feed: LiveKitRecordingFeed | None = None

    async def prepare(self) -> None:
        """Join the room and wait for a usable first frame from every stream; raises if one never arrives.

        Runs before the take's storage opens, so a missing camera fails the start instead of a half-written take.
        Does nothing when the session offers no recordable stream.
        """
        self._streams = tuple(self._recordable_streams())
        if not self._streams:
            return
        receiver = LiveKitFrameReceiver(self._streams, self._make_token(self._streams[0].room))
        self._receiver = receiver
        try:
            await receiver.start()
            for stream in self._streams:
                frame = await asyncio.wait_for(
                    receiver.next_frame_or_lost(stream.stream_id), self._first_frame_timeout_s
                )
                if frame is None:
                    raise RuntimeError(f"{stream.name}: the LiveKit track was lost before recording started")
                if frame.image.shape[:2] != (stream.height, stream.width):
                    raise RuntimeError(
                        f"{stream.name}: frames are {frame.image.shape[1]}x{frame.image.shape[0]}, "
                        f"expected {stream.width}x{stream.height}"
                    )
        except asyncio.TimeoutError as exc:
            await self.stop()
            raise RuntimeError("a LiveKit stream sent no frame before the recording could start") from exc
        except BaseException:
            await self.stop()
            raise

    def start(self, ingress: RecordingIngress, on_failure: Callable[[str], None]) -> None:
        """Begin feeding the take's ingress with frames acquired from now on."""
        if self._receiver is None:
            return
        # Frames queued while storage was opening were acquired before the take started.
        self._receiver.clear_queued()
        source_names = {s.stream_id: s.name for s in self._streams}
        self._feed = LiveKitRecordingFeed(
            self._receiver, self._streams, ingress, source_names, on_failure=on_failure
        )
        self._feed.start()

    async def stop(self) -> None:
        """Stop feeding and leave the room. Safe to call at any point, including after a failed prepare."""
        feed, receiver = self._feed, self._receiver
        self._feed = self._receiver = None
        if feed is not None:
            await feed.stop()
        if receiver is not None:
            await receiver.stop()
