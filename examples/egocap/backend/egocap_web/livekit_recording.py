# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Feeds received LiveKit camera frames into a take's recording ingress.

A frame is recorded only when its timestamp is on the publisher's monotonic clock: the time the frame was ready
to encode (``PUBLISHER_MONOTONIC_US``) or the sensor's capture event (``SENSOR_CAPTURE_MONOTONIC_US``). The
saved time means whichever the stream declares; frame-ready time is later than capture by the publisher's
capture and conversion delay, and neither is proof of physical alignment with tracking. It is placed on this
machine's monotonic clock, the one the tracking samples use, through the mapping the receiver's clock-sync
client builds from four-timestamp exchanges with the same publisher. There is one path: a stream with any other
basis, a missing or stale mapping, or a mapping that jumped during the take stops the feed with that reason
instead of recording a guessed time. Arrival time is never used as capture time. A mapping carries its
uncertainty (half the exchange round trip), not an accuracy claim about the sensor.

Tracking times come from CaptureServer's monotonic clock, which agrees with this process's only when both run on
one machine, so recording LiveKit video is refused unless CaptureServer's URL is a loopback address.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from capture_server import protocol as p
from recording_service import PixelFormat, VideoSample

from .livekit_receiver import LIVEKIT_MISSING, LiveKitFrameReceiver, livekit_available
from .recording_adapter import MANUS_GLOVE_SOURCE, PICO_CONTROLLER_SOURCE, RecordingIngress

logger = logging.getLogger("isaaccapture.egocap.livekit_recording")

#: A frame older than this when it arrives, or stamped further than the lead below in the future, means the
#: publisher's clock and this host's disagree grossly. This catches gross disagreement only, not millisecond error.
_MAX_FRAME_AGE_S = 2.0
_MAX_FRAME_LEAD_S = 0.05
#: How long a published stream may send nothing before the take ends. Normal frame gaps are tens of milliseconds.
_MAX_SILENCE_S = 5.0
#: How long the room join may take before the take's start is refused.
_JOIN_TIMEOUT_S = 15.0
#: How long leaving the room may take before teardown moves on and lets the leave finish in the background.
_LEAVE_TIMEOUT_S = 10.0
#: Most mappings recorded as a take's timing evidence, the newest ones; the estimator may have adopted more.
_MAX_EVIDENCE_MAPPINGS = 64

# The bases recorded, and what each one's time means in the saved take.
_RECORDABLE_BASES: dict[p.LiveKitTimestampBasis, str] = {
    p.LiveKitTimestampBasis.PUBLISHER_MONOTONIC_US: "frame_ready",
    p.LiveKitTimestampBasis.SENSOR_CAPTURE_MONOTONIC_US: "sensor_capture",
}

BUDGET_ENV = "EGOCAP_CLOCK_UNCERTAINTY_BUDGET_MS"
# A frame at 30 fps is 33 ms; a mapping uncertain by more than a few ms would blur which camera frame matches a
# tracking sample. This default is a policy choice, not a measurement; set the environment variable to change it.
_DEFAULT_BUDGET_MS = 5.0


# What each tracking source's saved time means today; none is a physical acquisition time.
_TRACKING_SEMANTICS: dict[str, str] = {
    "recorded_time": "the host's poll time for every tracking source, less a configured latency where one is set",
    "controllers": "device value is the pose-evaluation time (a predicted XrTime), not an acquisition time",
    "manus_gloves": "device value is host monotonic time at push; the SDK publishTime is not used",
    "physical_alignment": "unavailable: no tracking source carries a mapped device measurement time",
}


def _tracking_event_time(latency: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """What each tracking source's saved time is, and whether it is a qualified event time.

    A source with a configured latency is saved as its host poll time minus that latency: an estimate, marked as
    one with the latency used, never a measurement of the device. One without a latency is only the host poll
    time and stays unqualified.
    """
    return {
        name: {
            "meaning": "host_poll_minus_latency",
            "estimated": True,
            "latency_s": latency[name],
            "qualified": True,
        }
        if name in latency
        else {"meaning": "host_poll_time", "qualified": False}
        for name in (PICO_CONTROLLER_SOURCE, MANUS_GLOVE_SOURCE)
    }


def uncertainty_budget_ns() -> int:
    """The largest clock-mapping uncertainty a take may be recorded with, from the environment or the default."""
    raw = os.environ.get(BUDGET_ENV)
    if raw is None or not raw.strip():
        return int(_DEFAULT_BUDGET_MS * 1e6)
    try:
        ms = float(raw)
    except ValueError:
        raise ValueError(f"{BUDGET_ENV} must be a number of milliseconds, got {raw!r}") from None
    if not ms > 0:
        raise ValueError(f"{BUDGET_ENV} must be positive, got {raw!r}")
    return int(ms * 1e6)


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
        if stream.timestamp_basis not in _RECORDABLE_BASES:
            return (
                f"LiveKit stream {stream.name!r}: its timestamps are not on the publisher's monotonic clock, "
                "so it cannot be recorded"
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
        on_failure: Callable[[str], None] | None = None,
        max_uncertainty_ns: int | None = None,
    ) -> None:
        missing = [s.name for s in streams if s.stream_id not in source_names]
        if missing:
            raise ValueError(f"no recording source for LiveKit stream(s): {missing}")
        self._receiver = receiver
        self._streams = {s.stream_id: s for s in streams}
        self._ingress = ingress
        self._source_names = dict(source_names)
        # The receiver was started with clock sync, so its client maps the publisher's clock onto this one.
        clock_sync = receiver.clock_sync
        if clock_sync is None:
            raise ValueError("the receiver has no clock-sync client; start it with clock_sync=True")
        self._clock = clock_sync
        self._on_failure = on_failure
        self._max_uncertainty_ns = uncertainty_budget_ns() if max_uncertainty_ns is None else max_uncertainty_ns
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

    def _clock_problem(self) -> str:
        """What is wrong with the publisher's clock mapping right now, as a suffix for a failure reason (or "")."""
        if self._clock.discontinuity:
            return "; the publisher's clock mapping jumped"
        mapping = self._clock.mapping()
        if mapping is None:
            return "; no current clock mapping from the publisher (its clock-sync replies stopped)"
        if mapping.uncertainty_ns > self._max_uncertainty_ns:
            return f"; the clock mapping is uncertain by {mapping.uncertainty_ns / 1e6:.2f} ms, over the budget"
        return ""

    async def _run(self, stream_id: int) -> None:
        """Feed one stream until the take fails or the feed is stopped."""
        stream = self._streams[stream_id]
        source = self._source_names[stream_id]
        if stream.timestamp_basis not in _RECORDABLE_BASES:
            # Checked before any frame arrives: this stream cannot become recordable by waiting.
            self._fail(f"{stream.name}: its timestamps are not on the publisher's monotonic clock")
            return
        while self.failure is None:
            try:
                frame = await self._receiver.next_frame_or_lost(stream_id, timeout=_MAX_SILENCE_S)
            except asyncio.TimeoutError:
                # A track can stay published and send nothing; a take must not wait on it forever. With no frame
                # to carry the check, the clock mapping is judged here too, so a dead publisher says that it is.
                self._fail(
                    f"{stream.name}: no frame for {_MAX_SILENCE_S:.0f} s although the track is still published"
                    + self._clock_problem()
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
                self._fail(f"{stream.name}: a frame arrived without the publisher's frame timestamp")
                return
            # A mapping that jumped during the take, or went missing or stale, ends it: frames already recorded
            # keep the mapping they were made under, and none after is recorded with a guessed one.
            if self._clock.discontinuity:
                self._fail(f"{stream.name}: the publisher's clock mapping jumped during the take")
                return
            mapping = self._clock.mapping()
            if mapping is None:
                self._fail(
                    f"{stream.name}: no current clock mapping from the publisher (its clock-sync replies stopped)"
                )
                return
            if mapping.uncertainty_ns > self._max_uncertainty_ns:
                self._fail(
                    f"{stream.name}: the clock mapping is uncertain by {mapping.uncertainty_ns / 1e6:.2f} ms, "
                    f"over the {self._max_uncertainty_ns / 1e6:.2f} ms budget ({BUDGET_ENV})"
                )
                return
            capture_time = mapping.to_local_ns(frame.user_timestamp_us * 1000) / 1e9
            receive_time = frame.receive_time_ns / 1e9
            age = receive_time - capture_time
            if not -_MAX_FRAME_LEAD_S <= age <= _MAX_FRAME_AGE_S:
                # A coarse validity check: a frame time far from this host's clock is not a frame time here.
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
                # The mapping that placed this frame, read once here: a later refresh must not relabel it.
                continuity=("clock_mapping", mapping.epoch),
            )
            if not self._ingress.offer(sample):
                # The ingress refuses once it has failed or stopped; it reports its own reason.
                self._fail(f"{stream.name}: the recording ingress is no longer accepting frames")
                return


def recordable_livekit_streams(started: Any) -> tuple[p.LiveKitCameraStream, ...]:
    """The LiveKit streams of a started session that a recording may include.

    A stream is recordable when its timestamps are on the publisher's monotonic clock (frame-ready or sensor
    capture time) and its geometry is configured: the recording
    schema fixes each source's size and rate when the session opens. The rest remain preview-only.
    """
    if started is None:
        return ()
    return tuple(
        s
        for s in started.streams
        if isinstance(s, p.LiveKitCameraStream)
        and s.timestamp_basis in _RECORDABLE_BASES
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
        tracking_latency: Mapping[str, float] | None = None,
    ) -> None:
        self._tracking_latency = tracking_latency or {}
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
        # Read from the environment once in prepare(), so the feed and the saved evidence use the same figure.
        self._budget_ns = int(_DEFAULT_BUDGET_MS * 1e6)

    async def prepare(self) -> None:
        """Join the room and wait for a usable first frame from every stream; raises if one never arrives.

        Runs before the take's storage opens, so a missing camera fails the start instead of a half-written take.
        Does nothing when the session offers no recordable stream.
        """
        self._streams = tuple(self._recordable_streams())
        if not self._streams:
            return
        # A bad budget setting fails the start here, before the room is joined and storage and ingress are live.
        self._budget_ns = uncertainty_budget_ns()
        receiver = LiveKitFrameReceiver(self._streams, self._make_token(self._streams[0].room), clock_sync=True)
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
            await self._wait_for_mapping(receiver)
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

    def _mapping_problem(self, receiver: LiveKitFrameReceiver) -> str | None:
        """Why the publisher's clock mapping cannot be used to start a take right now, or None when it can."""
        clock_sync = receiver.clock_sync
        mapping = None if clock_sync is None else clock_sync.mapping()
        if clock_sync is None or mapping is None:
            return "the publisher did not answer clock-sync requests, so its frame times cannot be placed"
        if clock_sync.discontinuity:
            return "the publisher's clock mapping jumped before the take started"
        if mapping.uncertainty_ns > self._budget_ns:
            return (
                f"the clock mapping is uncertain by {mapping.uncertainty_ns / 1e6:.2f} ms, "
                f"over the {self._budget_ns / 1e6:.2f} ms budget ({BUDGET_ENV})"
            )
        return None

    async def _wait_for_mapping(self, receiver: LiveKitFrameReceiver) -> None:
        """Wait for a usable clock mapping: present, continuous and within the uncertainty budget.

        A mapping the feed would reject on its first frame refuses the start instead, with the reason, rather
        than opening a take that ends at once. The mapping may improve while waiting, so the check repeats until
        the deadline.
        """
        deadline = asyncio.get_running_loop().time() + self._first_frame_timeout_s
        while (problem := self._mapping_problem(receiver)) is not None:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(problem)
            await asyncio.sleep(0.05)

    def start(self, ingress: RecordingIngress, on_failure: Callable[[str], None]) -> None:
        """Begin feeding the take's ingress with frames acquired from now on."""
        if self._receiver is None:
            return
        # Frames queued while storage was opening were acquired before the take started.
        self._receiver.clear_queued()
        source_names = {s.stream_id: s.name for s in self._streams}
        self._feed = LiveKitRecordingFeed(
            self._receiver,
            self._streams,
            ingress,
            source_names,
            on_failure=on_failure,
            max_uncertainty_ns=self._budget_ns,
        )
        self._feed.start()

    def timing_evidence(self) -> dict[str, Any] | None:
        """The clock mappings this take's video was placed on the PC's clock with, for the take's attributes.

        None when there is no clock sync (nothing LiveKit was recorded). Read before ``stop()``, which leaves the
        room. A long take keeps only the newest mappings; ``mappings_omitted_oldest`` counts the rest.
        """
        receiver = self._receiver
        clock_sync = None if receiver is None else receiver.clock_sync
        if clock_sync is None:
            return None
        # The estimator re-reports one exchange with a growing age allowance while no newer one arrives; keep the
        # first report of each exchange.
        unique = {(m.epoch, m.accepted_at_ns): m for m in reversed(clock_sync.estimator.history)}
        ordered = sorted(unique.values(), key=lambda m: (m.accepted_at_ns, m.epoch))
        mappings = ordered[-_MAX_EVIDENCE_MAPPINGS:]
        return {
            "clock_mapping": {
                "method": "four_timestamp_exchange",
                "local_clock": "pc_monotonic",
                "remote_clock": "publisher_monotonic",
                "streams": [
                    {
                        "name": s.name,
                        "timestamp_basis": s.timestamp_basis.name,
                        "time_meaning": _RECORDABLE_BASES[s.timestamp_basis],
                    }
                    for s in self._streams
                ],
                "mappings": [
                    {
                        "epoch": m.epoch,
                        "offset_ns": m.offset_ns,
                        "rtt_ns": m.rtt_ns,
                        "uncertainty_ns": m.uncertainty_ns,
                        "accepted_at_ns": m.accepted_at_ns,
                    }
                    for m in mappings
                ],
                # A long take keeps only the newest mappings; this counts the older ones left out.
                "mappings_omitted_oldest": clock_sync.estimator.trimmed_mappings + len(ordered) - len(mappings),
                "uncertainty_budget_ns": self._budget_ns,
                "discontinuity": clock_sync.discontinuity,
                "exchanges_rejected": clock_sync.rejected,
                "mapping_current_at_stop": clock_sync.mapping() is not None,
            },
            # Per recorded source: what its saved time means and whether it is a qualified event time. A camera is
            # qualified only when its time is the sensor's capture event; a frame-ready time is not.
            "event_time": {
                # A stream is recorded under its own name (see start()), which is the key readers look it up by.
                **{
                    s.name: {
                        "meaning": _RECORDABLE_BASES[s.timestamp_basis],
                        "qualified": _RECORDABLE_BASES[s.timestamp_basis] == "sensor_capture",
                    }
                    for s in self._streams
                },
                **_tracking_event_time(self._tracking_latency),
            },
            # Tracking samples carry the host's poll time for every source; device measurement times are unmapped.
            "tracking_time": "host_poll",
            "tracking_semantics": _TRACKING_SEMANTICS,
        }

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
