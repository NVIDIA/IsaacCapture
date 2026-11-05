# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Feeds received LiveKit camera frames into a take's recording ingress.

A frame is recorded only when its timestamp is a capture time (``SENSOR_CAPTURE_UNIX_US``), which can then be put
on this machine's monotonic clock, the same clock the tracking samples use. A stream stamped with the publisher's
monotonic clock or with its processing time is not a capture time, so the feed stops with that reason instead of
recording a guessed one. Arrival time is never used as capture time.

Tracking times come from CaptureServer's monotonic clock, which agrees with this process's only when both run on
one machine, so recording LiveKit video is refused unless CaptureServer's URL is a loopback address.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from capture_server import protocol as p
from recording_service import PixelFormat, VideoSample

from .livekit_receiver import LIVEKIT_MISSING, LiveKitFrameReceiver, livekit_available
from .recording_adapter import RecordingIngress

logger = logging.getLogger("isaaccapture.egocap.livekit_recording")

#: How long a measured pair of clock readings is reused. Unix and monotonic time only drift apart slowly (a slewing
#: clock moves them well under a millisecond per second), so a larger change between two measurements is a step.
_PAIR_MAX_AGE_S = 1.0
#: An offset change below this is measurement noise or slew; above it, and above a few times the bracket's own
#: uncertainty, the Unix clock stepped and the take's capture times can no longer be trusted.
_STEP_FLOOR_S = 0.005
_STEP_UNCERTAINTY_FACTOR = 4
#: A frame older than this when it arrives, or stamped further than the lead below in the future, means the
#: publisher's clock and this host's disagree grossly. This catches gross disagreement only, not millisecond error.
_MAX_FRAME_AGE_S = 2.0
_MAX_FRAME_LEAD_S = 0.05
#: How long a published stream may send nothing before the take ends. Normal frame gaps are tens of milliseconds.
_MAX_SILENCE_S = 5.0
#: Readings per pair; the one read in the shortest span is the most trustworthy.
_PAIR_ATTEMPTS = 3
#: How long the room join may take before the take's start is refused.
_JOIN_TIMEOUT_S = 15.0
#: How long leaving the room may take before teardown moves on and lets the leave finish in the background.
_LEAVE_TIMEOUT_S = 10.0


class UnixClockMap:
    """Converts Unix-time microseconds to this process's monotonic seconds, from a recently measured pair.

    The pair is two clock readings taken back to back and refreshed about once a second, from the consumer's
    thread and never from an SDK callback. It converts on this machine only and says nothing about whether the
    publisher's clock agrees with this one. It keeps the uncertainty of its pair (half the bracketing span) and
    flags a discontinuity of the Unix clock between two measurements (a heuristic, not proof of a step or an accuracy
    bound): `step_detected` stays set, because samples already mapped are never remapped with a newer offset, and a
    map covers one take. Frames mapped between a step and its detection are already recorded with the old offset.
    """

    step_detected = False
    uncertainty_s: float | None = None

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
        span_ns, offset_s = best
        uncertainty_s = span_ns / 2e9
        if self._measured_at_ns is not None:
            bound_s = max(_STEP_FLOOR_S, _STEP_UNCERTAINTY_FACTOR * max(uncertainty_s, self.uncertainty_s or 0.0))
            if abs(offset_s - self._offset_s) > bound_s:
                self.step_detected = True
        self._offset_s = offset_s
        self.uncertainty_s = uncertainty_s
        self._measured_at_ns = self._monotonic_ns()


def is_loopback_url(url: str) -> bool:
    """Whether ``url`` names this machine: ``localhost`` or a loopback IP literal."""
    host = urlparse(url).hostname or ""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def unrecordable_reason(started: Any, capture_server_url: str | None = None) -> str | None:
    """Why a started session's LiveKit streams cannot all be recorded, naming the first that cannot; else None.

    A stream that cannot be recorded is never left out of a take silently: recording is refused until it is
    removed from the streams file or made recordable. When ``capture_server_url`` is given and is not a
    loopback address, every LiveKit stream is refused because the clocks cannot be assumed to be shared.
    """
    if started is None:
        return None
    for stream in started.streams:
        if not isinstance(stream, p.LiveKitCameraStream):
            continue
        # Without the optional package no LiveKit stream can be received, so say so before anything else.
        if not livekit_available():
            return f"LiveKit stream {stream.name!r}: {LIVEKIT_MISSING}"
        if capture_server_url is not None and not is_loopback_url(capture_server_url):
            return (
                f"LiveKit stream {stream.name!r}: recording needs CaptureServer on this machine "
                f"(its clock must match tracking's), but it is at {capture_server_url!r}"
            )
        if stream.timestamp_basis is not p.LiveKitTimestampBasis.SENSOR_CAPTURE_UNIX_US:
            return (
                f"LiveKit stream {stream.name!r}: its timestamps are not a capture time, so it cannot be recorded"
            )
        if None in (stream.width, stream.height, stream.fps):
            return f"LiveKit stream {stream.name!r}: width, height and fps must be configured to record it"
    return None


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
        self._baseline: dict[int, tuple[int, int]] = {}

    def loss(self) -> dict[str, dict[str, int]]:
        """What the receiver lost on the way to this take, per stream name (counted since the take began)."""
        out: dict[str, dict[str, int]] = {}
        for stream_id, stream in self._streams.items():
            counters = self._receiver.counters[stream_id]
            overflow, changes = self._baseline.get(stream_id, (0, 0))
            out[stream.name] = {
                "received": counters.received,
                "queue_overflow": counters.queue_overflow - overflow,
                "publication_changes": counters.publication_changes - changes,
                "missing_frames": counters.missing_frames,
                "frame_id_resets": counters.frame_id_resets,
            }
        return out

    def start(self) -> None:
        """Begin feeding; one task per stream so a slow stream never delays another."""
        for stream_id in self._streams:
            counters = self._receiver.counters[stream_id]
            self._baseline[stream_id] = (counters.queue_overflow, counters.publication_changes)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._run(stream_id)) for stream_id in self._streams]

    async def stop(self) -> None:
        """Stop feeding. Frames already offered to the ingress stay with it."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                _ = await task
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
            try:
                frame = await self._receiver.next_frame_or_lost(stream_id, timeout=_MAX_SILENCE_S)
            except asyncio.TimeoutError:
                # A track can stay published and send nothing; a take must not wait on it forever.
                self._fail(
                    f"{stream.name}: no frame for {_MAX_SILENCE_S:.0f} s although the track is still published"
                )
                return
            if frame is None:
                # Whatever was queued has been taken; the camera is gone, and a recorded source must not vanish.
                self._fail(f"{stream.name}: the LiveKit track was lost")
                return
            # What the receiver lost since this take began: video that was not recorded must not look like video
            # that was, so a gap or a replaced publication ends the take with its reason.
            counters = self._receiver.counters[stream_id]
            overflow, changes = self._baseline[stream_id]
            if counters.queue_overflow > overflow:
                self._fail(
                    f"{stream.name}: the recorder fell behind and dropped "
                    f"{counters.queue_overflow - overflow} frame(s)"
                )
                return
            if counters.publication_changes > changes:
                self._fail(f"{stream.name}: the camera's publication was replaced during the take")
                return
            if frame.user_timestamp_us is None:
                self._fail(f"{stream.name}: a frame arrived without the publisher's capture timestamp")
                return
            capture_time = self._clock_map.monotonic_seconds(frame.user_timestamp_us)
            if self._clock_map.step_detected:
                # Frames already recorded keep their mapping; none after the step is recorded with a guessed one.
                self._fail(
                    f"{stream.name}: this host's clock stepped during the take, so capture times are untrustworthy"
                )
                return
            receive_time = frame.receive_time_ns / 1e9
            age = receive_time - capture_time
            if not -_MAX_FRAME_LEAD_S <= age <= _MAX_FRAME_AGE_S:
                # A coarse validity check: a capture time far from this host's clock is not a capture time here.
                self._fail(
                    f"{stream.name}: the publisher's clock disagrees with this host's (frame age {age:+.3f} s)"
                )
                return
            sample = VideoSample(
                source=source,
                image=frame.image,
                pixel_format=PixelFormat.RGB8,
                capture_time=capture_time,
                receive_time=receive_time,
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
        join_timeout_s: float = _JOIN_TIMEOUT_S,
        leave_timeout_s: float = _LEAVE_TIMEOUT_S,
    ) -> None:
        self._recordable_streams = recordable_streams
        self._make_token = make_token
        self._first_frame_timeout_s = first_frame_timeout_s
        self._join_timeout_s = join_timeout_s
        self._leave_timeout_s = leave_timeout_s
        # Room joins and leaves that outlived their deadline; held so they are not garbage-collected mid-flight.
        self._background: set[asyncio.Future[None]] = set()
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
            await self._join(receiver)
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

    async def _join(self, receiver: LiveKitFrameReceiver) -> None:
        """Run ``receiver.start()`` with a deadline on the wait, never on the join itself.

        Cancelling ``Room.connect`` can strand the SDK's room-event handshake, so a join that is late (or whose
        caller is cancelled) keeps running and the receiver is retired once it finishes.
        """
        join = asyncio.ensure_future(receiver.start())
        try:
            _ = await asyncio.wait({join}, timeout=self._join_timeout_s)
        except BaseException:
            self._retire_after_join(join, receiver)
            raise
        if not join.done():
            self._retire_after_join(join, receiver)
            raise RuntimeError(f"joining the LiveKit room took longer than {self._join_timeout_s:g} s")
        # A join that failed raises here, with the receiver still ours to stop.
        join.result()

    def _retire_after_join(self, join: asyncio.Future[None], receiver: LiveKitFrameReceiver) -> None:
        """Hand an unfinished join to a background task that leaves the room as soon as the join ends."""
        if join.done():
            return
        # This source no longer owns the receiver: stop() must not leave a room that is still being joined.
        self._receiver = None
        self._background.add(join)

        def _leave(done: asyncio.Future[None]) -> None:
            self._background.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.warning("late LiveKit room join failed: %s", done.exception())
            self._track(asyncio.ensure_future(receiver.stop()), "leaving the room after a late join")

        join.add_done_callback(_leave)

    def _track(self, task: asyncio.Future[None], what: str) -> None:
        """Keep a background task referenced until it ends, logging a failure instead of losing it."""
        self._background.add(task)

        def _done(finished: asyncio.Future[None]) -> None:
            self._background.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                logger.warning("%s failed: %s", what, finished.exception())

        task.add_done_callback(_done)

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
            # A stalled room leave must not block retiring the take's ingress; it is left to finish on its own.
            leave = asyncio.ensure_future(receiver.stop())
            self._track(leave, "leaving the LiveKit room")
            _ = await asyncio.wait({leave}, timeout=self._leave_timeout_s)
            if not leave.done():
                logger.warning(
                    "leaving the LiveKit room took longer than %g s; continuing teardown", self._leave_timeout_s
                )
