# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A ``Pipeline``-shaped camera source that plays back one recorded MP4,
for :class:`~capture_server.acquisition.depthai.DepthAIAdapter`.

The adapter builds ``Camera -> VideoEncoder -> encoded host queue`` through
an ``sdk`` module's ``node``/``CameraBoardSocket``/``ImgFrame``/
``VideoEncoderProperties`` namespace and then only ever polls the queue's
``tryGet()``. Here the "encoder" output is the recorded frame due at the
session's shared :class:`~capture_server.replay.playback.PlaybackClock`
position, decoded from the MP4 and re-encoded as JPEG.

Frames are decoded strictly sequentially (``grab`` forward to the due
frame, then ``retrieve``); the file is never seeked. OpenCV's frame seek
fails on these recordings for the last ~1000 frames of a file, so a
seeking reader would lose the end of every episode. Looping therefore
reopens the file and decodes from the first frame again.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from capture_server.acquisition.clock import Clock, MonotonicClock

# SDK-shaped enums and the encoded-frame container, reproduced in the mock module and reused unchanged.
from capture_server.mock.mock_depthai import CameraBoardSocket, EncodedFrame, ImgFrame, VideoEncoderProperties

from .episode import CameraTrack
from .playback import PlaybackClock

__all__ = [
    "CameraBoardSocket",
    "EncodedFrame",
    "ImgFrame",
    "MessageQueue",
    "Pipeline",
    "ReplayFrame",
    "VideoEncoderProperties",
    "node",
]


class _Transformation:
    """The slice of the SDK's frame transformation the adapter reads: the 3x3 intrinsic matrix."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = tuple(tuple(float(v) for v in row) for row in matrix)

    def getIntrinsicMatrix(self) -> tuple[tuple[float, ...], ...]:
        """Recorded per-frame intrinsics as rows of floats."""
        return self._matrix


class ReplayFrame(EncodedFrame):
    """An encoded replay frame that also carries the recorded intrinsics, when the recorder wrote them."""

    def __init__(self, *, intrinsics: np.ndarray, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._transformation = _Transformation(intrinsics)

    def getTransformation(self) -> _Transformation:
        """The frame's recorded camera geometry; the adapter reports its intrinsic matrix per stream."""
        return self._transformation


class MessageQueue:
    """The nonblocking, bounded ``tryGet()`` queue the adapter polls; every poll asks the pipeline for the
    frame due now (``fill``), so there is no catch-up burst -- overdue frames are skipped, not replayed."""

    def __init__(self, *, fill: Callable[[], EncodedFrame | None] | None = None) -> None:
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
        """Whether a poll right now would yield a frame (consumes nothing)."""
        return not self._closed and self._fill is not None

    def tryGet(self) -> EncodedFrame | None:
        """The frame due at the current playback position, or None when none is new since the last poll."""
        if self._closed or self._fill is None:
            return None
        return self._fill()

    def close(self) -> None:
        """Stop yielding frames."""
        self._closed = True


class _Output:
    """An output port whose queue delivers the pipeline's replay frames."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def createOutputQueue(self) -> MessageQueue:
        """The adapter's host-side queue, fed by the owning pipeline."""
        return self._pipeline._create_output_queue()


class _CameraNode:
    """Accepts the adapter's camera configuration; the recording fixes the real geometry."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def build(self, socket: CameraBoardSocket) -> _CameraNode:
        """Bind the (nominal) board socket."""
        return self

    def requestOutput(
        self, size: tuple[int, int], *, type: str | None = None, fps: float | None = None
    ) -> _Output:
        """Record the requested size; frames are served at the recorded size and the adapter checks they match."""
        self._pipeline._requested_size = (int(size[0]), int(size[1]))
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
        """Wire the camera output through; the JPEG quality is the one the accepted stream asked for."""
        if profile != VideoEncoderProperties.Profile.MJPEG:
            raise ValueError(f"replay VideoEncoder only supports MJPEG, got {profile!r}")
        self.out = input_output
        self._pipeline._jpeg_quality = int(quality)
        return self


class node:
    """Mirrors ``dai.node`` -- the namespace ``Pipeline.create()`` takes its node type from."""

    Camera = _CameraNode
    VideoEncoder = _VideoEncoderNode


class Pipeline:
    """Owns one recorded camera's decoder for the lifetime of one stream; context-manager shaped like the SDK.

    Created and used on the adapter's single worker thread, so the OpenCV capture needs no locking.
    """

    def __init__(
        self,
        *,
        track: CameraTrack,
        playback: PlaybackClock,
        jpeg_quality: int = 80,
        clock: Clock | None = None,
    ) -> None:
        self._track = track
        self._playback = playback
        self._jpeg_quality = int(jpeg_quality)
        self._clock = clock or MonotonicClock()
        self._capture: Any = None
        self._running = False
        self._requested_size: tuple[int, int] | None = None
        self._queue: MessageQueue | None = None
        # Decoder position bookkeeping: the index of the last frame decoded, the last one handed out, and the
        # loop cycle the decoder currently belongs to.
        self._decoded = -1
        self._delivered = -1
        self._cycle = 0

    def __enter__(self) -> Pipeline:
        """Open the MP4; an unopenable file fails here, where the adapter maps it to a camera start failure."""
        self._open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop and release the decoder."""
        self.stop()
        self._release()
        return None

    def _open(self) -> None:
        """(Re)open the file and rewind the bookkeeping; the current loop cycle is recorded so the first poll
        after a wrap does not reopen again."""
        self._release()
        capture = cv2.VideoCapture(str(self._track.path))
        if not capture.isOpened():
            raise RuntimeError(f"{self._track.name}: could not open {self._track.path}")
        self._capture = capture
        self._decoded = -1
        self._delivered = -1
        self._cycle = self._playback.cycle()

    def _release(self) -> None:
        """Release the OpenCV capture if one is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def create(self, node_type: type) -> _CameraNode | _VideoEncoderNode:
        """The two node types the adapter's graph uses."""
        if node_type is node.Camera:
            return _CameraNode(self)
        if node_type is node.VideoEncoder:
            return _VideoEncoderNode(self)
        raise ValueError(f"replay Pipeline does not implement node type {node_type!r}")

    def _create_output_queue(self) -> MessageQueue:
        """The one queue that yields replay frames."""
        self._queue = MessageQueue(fill=self._next_due_frame)
        return self._queue

    def start(self) -> None:
        """Begin serving frames on polls."""
        self._running = True

    def isRunning(self) -> bool:
        """Whether ``start()`` has been called and ``stop()`` has not."""
        return self._running

    def stop(self) -> None:
        """Stop serving frames; the queue reports nothing further."""
        self._running = False
        if self._queue is not None:
            self._queue.close()

    def wait(self) -> None:
        """Nothing runs in the background."""
        return None

    def _next_due_frame(self) -> EncodedFrame | None:
        """The recorded frame due at the current playback position, if it is newer than the last one served.

        Decoding only moves forward: frames between the last decoded one and the due one are grabbed and
        dropped (never seeked over), and a wrap of the shared timeline reopens the file from its first frame.
        """
        if not self._running or self._capture is None:
            return None
        cycle = self._playback.cycle()
        position = self._playback.position()
        if cycle != self._cycle:
            self._open()
        target = self._track.latest_frame(position)
        if target < 0 or target == self._delivered:
            return None
        if target < self._decoded:
            # Defensive: a timeline wrap that slipped past the cycle check still must not try to seek back.
            self._open()
        while self._decoded < target:
            if not self._capture.grab():
                raise RuntimeError(f"{self._track.name}: could not decode frame {self._decoded + 1}")
            self._decoded += 1
        ok, bgr = self._capture.retrieve()
        if not ok or bgr is None:
            raise RuntimeError(f"{self._track.name}: could not decode frame {target}")
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            raise RuntimeError(f"{self._track.name}: could not JPEG-encode frame {target}")
        self._delivered = target
        # The host-poll timestamp is stamped by the adapter; this one only describes the frame itself.
        common = {
            "data": encoded.tobytes(),
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "sequence_num": target,
            "timestamp_s": self._clock.monotonic(),
        }
        if self._track.intrinsics is not None:
            return ReplayFrame(intrinsics=self._track.intrinsics[target], **common)
        return EncodedFrame(**common)
