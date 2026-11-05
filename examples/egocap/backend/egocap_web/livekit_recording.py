# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Feeds received LiveKit camera frames into a take's recording ingress.

A frame is recorded only when its capture time can be put on this machine's monotonic clock, the same clock the
tracking samples use. That is possible when the publisher's application timestamp is Unix time on a host kept in
step with this one (``PUBLISHER_UNIX_US``). A stream whose timestamp is on the publisher's own monotonic clock has
no honest mapping, so the feed stops with that reason instead of recording a guessed time. Arrival time is never
used as capture time.

This assumes the backend runs on the machine whose monotonic clock CaptureServer reports: tracking times come from
CaptureServer, and the two agree only when both processes share a clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from capture_server import protocol as p
from recording_service import PixelFormat, VideoSample

from .livekit_receiver import LiveKitFrameReceiver, ReceivedFrame
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
    thread and never from an SDK callback. Between refreshes the two clocks run at the same rate, so the offset
    holds; a step of the Unix clock shows up at the next refresh.
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


@dataclass(frozen=True)
class MappedFrame:
    """A received frame with the recording source it belongs to and its capture time on this machine's clock."""

    frame: ReceivedFrame
    source: str
    capture_time: float

    @property
    def image(self) -> np.ndarray:
        """The frame's pixels, so the ingress can size its queue by bytes."""
        return self.frame.image


def mapped_frame_to_samples(item: MappedFrame) -> list[VideoSample]:
    """The recording sample for one mapped frame (the ingress's ``convert``)."""
    return [
        VideoSample(
            source=item.source,
            image=item.frame.image,
            pixel_format=PixelFormat.RGB8,
            capture_time=item.capture_time,
            receive_time=item.frame.receive_time_ns / 1e9,
        )
    ]


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
        if stream.timestamp_basis is not p.LiveKitTimestampBasis.PUBLISHER_UNIX_US:
            # Checked before any frame arrives: this stream cannot become recordable by waiting.
            self._fail(
                f"{stream.name}: timestamps are on the publisher's own clock and cannot be aligned with tracking"
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
            if not self._ingress.offer(MappedFrame(frame, source, capture_time)):
                # The ingress refuses once it has failed or stopped; it reports its own reason.
                self._fail(f"{stream.name}: the recording ingress is no longer accepting frames")
                return
