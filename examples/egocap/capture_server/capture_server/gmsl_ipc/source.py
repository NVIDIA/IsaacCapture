# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A ``Pipeline``-shaped camera source serving live GMSL frames from the station's frame feed,
for :class:`~capture_server.acquisition.depthai.DepthAIAdapter`.

Like the replay source, this reproduces the small SDK surface the adapter drives
(``node``/``CameraBoardSocket``/``ImgFrame``/``VideoEncoderProperties`` and a ``tryGet()`` queue)
so the adapter's validation, pacing and status logic are reused unchanged.

Each pipeline runs one encoder thread that reacts to the feed's notices: as soon as a frame is
announced it is copied to host, converted NV12/RGBA8 to BGR, optionally downscaled, JPEG-encoded
and appended to a short bounded queue. The adapter's poll (``tryGet``) then takes the oldest pending
frame, or waits a fraction of a frame period for a late one. Encoding ahead of the poll is what makes
a true 60 Hz possible: the adapter polls once per fixed tick, and a poll that has to *find* a frame
announced a millisecond after the previous tick ends up delivering only every other frame when the
publisher's cadence drifts against the poll clock. The queue is bounded to ``QUEUE_DEPTH`` frames
(the oldest is dropped on overflow), so a consumer that stops polling never sees stale frames and
the extra latency is at most ``QUEUE_DEPTH - 1`` frame periods.

The wire protocol caps one image message at 2 MiB, which a full-resolution head camera can exceed on
a detailed scene, so a preview size smaller than the sensor's is configurable per camera.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

import cv2
import numpy as np

from capture_server.acquisition.clock import Clock, MonotonicClock

# SDK-shaped enums and the encoded-frame container, reproduced in the mock module and reused unchanged.
from capture_server.mock.mock_depthai import CameraBoardSocket, EncodedFrame, ImgFrame, VideoEncoderProperties

from .feed import FeedClient, FrameNotice, StreamInfo

__all__ = [
    "QUEUE_DEPTH",
    "CameraBoardSocket",
    "EncodedFrame",
    "ImgFrame",
    "MessageQueue",
    "Pipeline",
    "VideoEncoderProperties",
    "decode_to_bgr",
    "node",
]

#: Encoded frames kept ahead of the adapter's poll. Two absorbs the publisher's cadence jitter against
#: the poll clock (a late frame plus the next one) without ever delivering more than one frame late.
QUEUE_DEPTH = 2
#: Fraction of a frame period a poll may wait for a late frame before reporting an empty poll. Waiting
#: longer would push the adapter past its next tick, which it then skips; shorter would miss frames
#: that are merely a few milliseconds behind the poll clock.
_GRACE_FRACTION = 0.5
#: How long the encoder thread sleeps per wait before re-checking for shutdown.
_WORKER_WAIT_S = 0.25


def decode_to_bgr(pixels: np.ndarray, stream: StreamInfo) -> np.ndarray:
    """BGR image from one slot's host bytes, per the stream's advertised pixel format and size.

    The slot is ``rows x pitch``: NV12 is ``height * 3 / 2`` rows of ``width`` bytes (Y plane then
    interleaved UV), RGBA8 is ``height`` rows of ``width * 4`` bytes. The pitch may exceed the width,
    so the view is cropped before conversion.
    """
    if stream.pixel_format == "nv12":
        rows = stream.height * 3 // 2
        return cv2.cvtColor(np.ascontiguousarray(pixels[:rows, : stream.width]), cv2.COLOR_YUV2BGR_NV12)
    if stream.pixel_format == "rgba8":
        flat = pixels[: stream.height, : stream.width * 4]
        return cv2.cvtColor(np.ascontiguousarray(flat).reshape(stream.height, stream.width, 4), cv2.COLOR_RGBA2BGR)
    raise ValueError(f"unsupported frame feed pixel format {stream.pixel_format!r} for stream {stream.name!r}")


class MessageQueue:
    """The nonblocking ``tryGet()`` queue the adapter polls; each poll asks the pipeline for the newest frame."""

    def __init__(self, *, fill: Any = None) -> None:
        self._fill = fill
        self._closed = False
        self._blocking = False

    def setMaxSize(self, max_size: int) -> None:
        """Accepted for API shape; at most one frame is ever pending here by construction."""

    def setBlocking(self, blocking: bool) -> None:
        """Record the requested policy; ``tryGet`` is nonblocking regardless."""
        self._blocking = bool(blocking)

    def getBlocking(self) -> bool:
        """The last requested blocking policy."""
        return self._blocking

    def has(self) -> bool:
        """Whether a poll could yield a frame (consumes nothing)."""
        return not self._closed and self._fill is not None

    def tryGet(self) -> EncodedFrame | None:
        """The newest unserved frame, or None when nothing new has been announced."""
        if self._closed or self._fill is None:
            return None
        return self._fill()

    def close(self) -> None:
        """Stop yielding frames."""
        self._closed = True


class _Output:
    """An output port whose queue delivers the pipeline's frames."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def createOutputQueue(self) -> MessageQueue:
        """The adapter's host-side queue, fed by the owning pipeline."""
        return self._pipeline._create_output_queue()


class _CameraNode:
    """Accepts the adapter's camera configuration; the feed fixes the real geometry."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def build(self, socket: CameraBoardSocket) -> _CameraNode:
        """Bind the (nominal) board socket."""
        return self

    def requestOutput(
        self, size: tuple[int, int], *, type: str | None = None, fps: float | None = None
    ) -> _Output:
        """The adapter asks for the accepted stream's size; frames are served at the pipeline's output size,
        which the server configured from the same station entry, and the adapter checks they match."""
        return _Output(self._pipeline)


class _VideoEncoderNode:
    """Accepts the adapter's encoder configuration: only MJPEG is meaningful, at the requested quality."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self.out: _Output | None = None

    def build(
        self,
        input_output: _Output,
        *,
        frameRate: float,
        profile: VideoEncoderProperties.Profile,
        quality: int = 80,
    ) -> _VideoEncoderNode:
        """Wire the camera output through; the JPEG quality and frame rate are the accepted stream's."""
        if profile != VideoEncoderProperties.Profile.MJPEG:
            raise ValueError(f"frame feed VideoEncoder only supports MJPEG, got {profile!r}")
        self.out = input_output
        self._pipeline._jpeg_quality = int(quality)
        # The poll grace is derived from the stream's rate: the adapter polls at exactly this cadence.
        self._pipeline._frame_rate = float(frameRate)
        return self


class node:
    """Mirrors ``dai.node`` -- the namespace ``Pipeline.create()`` takes its node type from."""

    Camera = _CameraNode
    VideoEncoder = _VideoEncoderNode


class Pipeline:
    """Serves one feed stream for the lifetime of one camera stream; context-manager shaped like the SDK.

    ``feed`` is shared by every camera of a session (one subscription); this pipeline only reads its own
    stream from it. ``output_size`` is the served ``(width, height)``; a size smaller than the sensor's
    downscales each frame before encoding. ``tryGet`` runs on the adapter's dedicated thread and the
    encoder on this pipeline's own thread, so the pending queue is guarded by one condition variable.
    """

    def __init__(
        self,
        *,
        feed: FeedClient,
        stream: str,
        output_size: tuple[int, int] | None = None,
        jpeg_quality: int = 80,
        clock: Clock | None = None,
    ) -> None:
        self._feed = feed
        self._stream = stream
        self._output_size = output_size
        self._jpeg_quality = int(jpeg_quality)
        self._clock = clock or MonotonicClock()
        self._frame_rate = 60.0
        self._info: StreamInfo | None = None
        self._running = False
        self._queue: MessageQueue | None = None
        self._delivered_seq: int | None = None
        self._pending: collections.deque[EncodedFrame] = collections.deque(maxlen=QUEUE_DEPTH)
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None
        # An encoder failure is parked here and raised from the adapter's next poll, so the camera is
        # reported failed on the thread that owns status instead of dying silently in the background.
        self._failure: Exception | None = None

    def __enter__(self) -> Pipeline:
        """Resolve this stream in the feed's catalog; an unsubscribed name fails here, where the adapter
        maps it to a camera start failure."""
        info = self._feed.catalog.get(self._stream)
        if info is None:
            raise RuntimeError(f"stream {self._stream!r} is not in the frame feed subscription")
        self._info = info
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop serving; the shared feed is closed by its owner, not per stream."""
        self.stop()
        return None

    def create(self, node_type: type) -> _CameraNode | _VideoEncoderNode:
        """The two node types the adapter's graph uses."""
        if node_type is node.Camera:
            return _CameraNode(self)
        if node_type is node.VideoEncoder:
            return _VideoEncoderNode(self)
        raise ValueError(f"frame feed Pipeline does not implement node type {node_type!r}")

    def _create_output_queue(self) -> MessageQueue:
        """The one queue that yields this stream's frames."""
        self._queue = MessageQueue(fill=self._next_frame)
        return self._queue

    def start(self) -> None:
        """Start the encoder thread; frames are encoded as the feed announces them."""
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._encode_loop, name=f"gmsl-encode-{self._stream}", daemon=True)
        self._worker.start()

    def isRunning(self) -> bool:
        """Whether ``start()`` has been called and ``stop()`` has not."""
        return self._running

    def stop(self) -> None:
        """Stop encoding and serving; joins the encoder thread so no frame is produced after this returns."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        worker, self._worker = self._worker, None
        if worker is not None and worker is not threading.current_thread():
            # Bounded: the worker checks ``_running`` between waits and between frames; a join that still
            # times out means a stuck slot copy, which the adapter's own close bound then reports.
            worker.join(timeout=2.0)
        if self._queue is not None:
            self._queue.close()

    def wait(self) -> None:
        """Block until the encoder thread has exited (after ``stop``); the feed's reader thread is the feed's."""
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def _encode_loop(self) -> None:
        """Encode every frame the feed announces for this stream, newest first when behind, until stopped."""
        encoded_seq: int | None = None
        try:
            while self._running:
                notice = self._feed.wait_for_newer(self._stream, encoded_seq, _WORKER_WAIT_S)
                if notice is None:
                    continue
                frame = self._encode(notice)
                encoded_seq = notice.seq
                if frame is None:
                    continue
                with self._cond:
                    # deque(maxlen=QUEUE_DEPTH) drops the oldest pending frame when the consumer is behind,
                    # which keeps the bounded-latency promise of the module docstring.
                    self._pending.append(frame)
                    self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 -- surfaced from the next poll, where the adapter maps it
            with self._cond:
                self._failure = exc
                self._cond.notify_all()

    def _encode(self, notice: FrameNotice) -> EncodedFrame | None:
        """JPEG for one announced frame, or None when its slot was already overwritten by the publisher."""
        assert self._info is not None
        pixels = self._feed.read_slot(self._stream, notice.slot)
        # The ring may have wrapped onto this slot while it was being copied (the publisher overwrites a slot
        # every ``slot_count`` frames); a notice that far behind the newest means the bytes are not this
        # frame's. Skip it rather than serve a torn image.
        newest = self._feed.latest(self._stream)
        if newest is not None and newest.seq - notice.seq >= self._info.slot_count:
            return None
        bgr = decode_to_bgr(pixels, self._info)
        if self._output_size is not None and self._output_size != (bgr.shape[1], bgr.shape[0]):
            bgr = cv2.resize(bgr, self._output_size, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            raise RuntimeError(f"{self._stream}: could not JPEG-encode frame {notice.seq}")
        return EncodedFrame(
            data=encoded.tobytes(),
            width=int(bgr.shape[1]),
            height=int(bgr.shape[0]),
            sequence_num=notice.seq,
            # Informational only: the adapter stamps its own host-poll time per the wire protocol.
            timestamp_s=_frame_time_s(notice, self._clock),
        )

    def _next_frame(self) -> EncodedFrame | None:
        """The oldest pending encoded frame, waiting up to half a frame period for a late one; None if none.

        Called on the adapter's thread at its poll cadence. A parked encoder failure is raised here so the
        adapter reports the camera as failed.
        """
        if not self._running or self._info is None:
            return None
        grace_s = _GRACE_FRACTION / self._frame_rate if self._frame_rate > 0 else 0.0
        with self._cond:
            if not self._pending and self._failure is None and self._running:
                self._cond.wait(grace_s)
            if self._failure is not None:
                failure, self._failure = self._failure, None
                raise failure
            if not self._pending:
                return None
            frame = self._pending.popleft()
        self._delivered_seq = frame.getSequenceNum()
        return frame


def _frame_time_s(notice: FrameNotice, clock: Clock) -> float:
    """The sensor-side timestamp in seconds when the receiver relayed one, else the publish time."""
    sensor_ns = notice.metadata.get("sensor_timestamp_ns")
    if isinstance(sensor_ns, (int, float)) and sensor_ns > 0:
        return float(sensor_ns) / 1e9
    if notice.timestamp_ns > 0:
        return notice.timestamp_ns / 1e9
    return clock.monotonic()
