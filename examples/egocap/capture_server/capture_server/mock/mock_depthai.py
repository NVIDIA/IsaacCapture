"""A synthetic approximation of the public DepthAI v3 API subset the
production :mod:`capture_server.acquisition.depthai` adapter calls,
grounded against the public `luxonis/depthai-core
<https://github.com/luxonis/depthai-core>`_ repository at pinned commit
``8c0d8eca65dda9e272f93606263d0cf0569c85f2`` (reconfirm before relying on
this file beyond this package's own needs):

- ``include/depthai/pipeline/node/Camera.hpp`` /
  ``bindings/python/src/pipeline/node/CameraBindings.cpp`` --
  ``Pipeline.create(Camera).build(socket)``, ``.requestOutput((w, h),
  type=...)``.
- ``include/depthai/pipeline/node/VideoEncoder.hpp`` /
  ``bindings/python/src/pipeline/node/VideoEncoderBindings.cpp`` --
  ``Pipeline.create(VideoEncoder).build(output, frameRate=, profile=)``,
  reproduced per ``examples/python/VideoEncoder/video_encode.py``'s
  ``Camera -> VideoEncoder -> encoded host queue`` construction path.
- ``bindings/python/src/MessageQueueBindings.cpp`` -- ``has()``/
  ``tryGet()``/``get()``/``close()``/``setMaxSize()``/``setBlocking()``,
  the queue the adapter polls non-blockingly (``tryGet``) rather than the
  example's blocking ``get()``.
- ``bindings/python/src/pipeline/datatype/EncodedFrameBindings.cpp`` /
  ``BufferBindings.cpp`` -- ``EncodedFrame.getData()``/``getWidth()``/
  ``getHeight()``/``getSequenceNum()``/``getTimestamp()``.

This mock only implements the color-MJPEG Camera -> VideoEncoder ->
encoded-queue path -- not arbitrary DepthAI graphs, other sensors, XLink
transport or firmware. It must not import ``capture_server.protocol`` and
does not; it returns SDK-shaped values the production adapter translates.
All scenario/fault state lives on the :class:`Pipeline` instance that owns
it -- there is no process-global mutable registry.

Synthetic JPEG generation below is not adapted from any existing static-
GIF/Tkinter-viewer mock -- it is a pure function of (device identity,
frame number), like the deterministic motion math in
``mock_isaac_capture.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from PIL import Image

from capture_server.acquisition.clock import Clock, MonotonicClock

__all__ = [
    "CameraBoardSocket",
    "DeviceInfo",
    "EncodedFrame",
    "KNOWN_FAULTS",
    "MessageQueue",
    "Pipeline",
    "VideoEncoderProperties",
    "available_devices",
    "node",
]

# Named camera-side faults, scoped to this file: all scenario state
# belongs to the created mock API/session instance, never a process-
# global mutable fault registry. Each Pipeline owns its own
# active set, settable at construction and live via set_faults().
KNOWN_FAULTS = frozenset({"camera_required_failure", "camera_optional_failure"})


class CameraBoardSocket(str, Enum):
    CAM_A = "CAM_A"
    CAM_B = "CAM_B"
    CAM_C = "CAM_C"
    CAM_D = "CAM_D"


class VideoEncoderProperties:
    class Profile(str, Enum):
        MJPEG = "MJPEG"
        H264_MAIN = "H264_MAIN"
        H265_MAIN = "H265_MAIN"


@dataclass(frozen=True)
class DeviceInfo:
    """A simulated device identity -- mirrors the small subset of
    ``dai.DeviceInfo`` the adapter's device-selection call reads."""

    mxid: str
    name: str


def available_devices(device_ids: tuple[str, ...]) -> tuple[DeviceInfo, ...]:
    """Enumerate configured simulated device identities. A requested
    device not in this list is not silently substituted -- see
    :class:`Pipeline`'s ``device_id``/``available_device_ids`` arguments."""
    return tuple(DeviceInfo(mxid=d, name=d) for d in device_ids)


class EncodedFrame:
    """A complete encoded frame -- SDK-shaped owned byte-array semantics,
    like the real ``EncodedFrame``'s ``getData()``."""

    def __init__(self, *, data: bytes, width: int, height: int, sequence_num: int, timestamp_s: float) -> None:
        self._data = data
        self._width = width
        self._height = height
        self._sequence_num = sequence_num
        self._timestamp_s = timestamp_s

    def getData(self) -> bytes:
        return self._data

    def getWidth(self) -> int:
        return self._width

    def getHeight(self) -> int:
        return self._height

    def getSequenceNum(self) -> int:
        return self._sequence_num

    def getTimestamp(self) -> float:
        return self._timestamp_s


class MessageQueue:
    """The small real ``dai.MessageQueue`` subset the adapter actually
    uses: ``tryGet()``/``has()``/``setMaxSize()``/``setBlocking()``/
    ``close()``. The production adapter explicitly configures this exact
    narrow policy right after ``createOutputQueue()`` -- ``setMaxSize(1)``
    and ``setBlocking(False)`` -- a bounded nonblocking latest-frame
    queue; this mock supports exactly that configured policy rather than
    silently pretending an unconfigured/unsupported one is honored.

    Independently paced, like a real encoder's output queue: a new frame
    only becomes available once ``interval`` has genuinely elapsed since
    the last one, regardless of how often ``tryGet()`` is called --
    polling an empty queue must not manufacture a new frame. There is no
    adapter-facing trigger method; ``fill`` (bound by :class:`Pipeline` at
    queue-creation time) is this queue's own private frame source,
    returning ``None`` while the pipeline isn't running.

    Finite retention, actually bounded by ``setMaxSize``: each poll
    generates at most ONE new frame (regardless of how large the gap
    since the last poll was -- overdue slots are skipped to "now" in one
    jump, never replayed one generated frame per call, the same "no
    catch-up bursts" policy the production adapter's own deadline pacing
    uses) and appends it to a small buffer, dropping the oldest buffered
    frame once ``max_size`` would be exceeded."""

    def __init__(
        self,
        *,
        fill: Callable[[], "EncodedFrame | None"] | None = None,
        interval: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._fill = fill
        self._interval = interval
        self._clock = clock or MonotonicClock()
        self._next_due: float | None = None
        self._closed = False
        self._max_size = 1
        self._blocking = False
        self._buffer: list[EncodedFrame] = []

    def setMaxSize(self, max_size: int) -> None:
        self._max_size = max(1, max_size)

    def setBlocking(self, blocking: bool) -> None:
        self._blocking = bool(blocking)

    def getBlocking(self) -> bool:
        return self._blocking

    def _advance(self) -> None:
        if self._closed or self._fill is None:
            return
        now = self._clock.monotonic()
        if self._next_due is None:
            self._next_due = now
        if self._next_due > now:
            return
        while self._next_due <= now:
            self._next_due += self._interval
        frame = self._fill()
        if frame is None:
            return
        if len(self._buffer) >= self._max_size:
            self._buffer.pop(0)  # bounded: drop-oldest once at capacity
        self._buffer.append(frame)

    def has(self) -> bool:
        self._advance()
        return bool(self._buffer)

    def tryGet(self) -> EncodedFrame | None:
        self._advance()
        if not self._buffer:
            return None
        return self._buffer.pop(0)

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()


@dataclass(frozen=True)
class _Resolution:
    width: int
    height: int


class _Output:
    def __init__(self, *, pipeline: "Pipeline", resolution: _Resolution) -> None:
        self._pipeline = pipeline
        self.resolution = resolution

    def createOutputQueue(self) -> MessageQueue:
        return self._pipeline._create_output_queue(resolution=self.resolution)


class _CameraNode:
    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline
        self.socket: CameraBoardSocket | None = None

    def build(self, socket: CameraBoardSocket) -> "_CameraNode":
        self.socket = socket
        self._pipeline._bind_camera(socket)
        return self

    def requestOutput(
        self, size: tuple[int, int], *, type: str | None = None, fps: float | None = None
    ) -> _Output:
        width, height = size
        return _Output(pipeline=self._pipeline, resolution=_Resolution(width, height))


class _VideoEncoderNode:
    def __init__(self, pipeline: "Pipeline") -> None:
        self._pipeline = pipeline
        self.out: _Output | None = None

    def build(
        self,
        input_output: _Output,
        *,
        frameRate: float,
        profile: "VideoEncoderProperties.Profile",
        quality: int = 80,
    ) -> "_VideoEncoderNode":
        if profile != VideoEncoderProperties.Profile.MJPEG:
            raise ValueError(f"mock VideoEncoder only supports MJPEG, got {profile!r}")
        self.out = input_output
        self._pipeline._bind_encoder(self, frame_rate=frameRate, resolution=input_output.resolution)
        return self


class ImgFrame:
    """Mirrors ``dai.ImgFrame``'s type namespace (the adapter requests NV12 output)."""

    class Type:
        """The frame types the adapter requests."""

        NV12 = "NV12"


class node:
    """Mirrors ``dai.node`` -- the namespace ``Pipeline.create()`` expects
    its node-type argument from (``pipeline.create(dai.node.Camera)``)."""

    Camera = _CameraNode
    VideoEncoder = _VideoEncoderNode


def _color_for_frame(seed: int, frame_num: int) -> tuple[int, int, int]:
    mixed = (seed * 2_654_435_761 + frame_num * 40_503) & 0xFFFFFF
    return (mixed & 0xFF, (mixed >> 8) & 0xFF, (mixed >> 16) & 0xFF)


def _jpeg_bytes(*, width: int, height: int, seed: int, frame_num: int, quality: int = 80) -> bytes:
    color = _color_for_frame(seed, frame_num)
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class Pipeline:
    """Owns the ``Camera -> VideoEncoder -> encoded host queue`` graph for
    one simulated device. Context-manager-shaped (``with Pipeline(...) as
    pipeline:``) like the real ``dai.Pipeline``.

    ``required`` here is deliberately NOT how the real/mock camera reports
    failure consequences: mock camera failure has no knowledge of the
    protocol's required-camera StopReason -- the common camera policy
    handles it. This mock only ever reports whether a
    frame is available this poll or raises for an injected failure; the
    production adapter decides what a required vs. optional camera's
    failure means."""

    def __init__(
        self,
        *,
        device_id: str,
        available_device_ids: tuple[str, ...] | None = None,
        seed: int = 0,
        faults: set[str] | frozenset[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._requested_device_id = device_id
        self._available = {d.mxid: d for d in available_devices(available_device_ids or (device_id,))}
        self._seed = seed
        self._faults: set[str] = set(faults) if faults else set()
        self._clock = clock or MonotonicClock()
        self._running = False
        self._bound_socket: CameraBoardSocket | None = None
        self._encoder: _VideoEncoderNode | None = None
        self._frame_rate: float = 30.0
        self._resolution = _Resolution(1280, 720)
        self._queue: MessageQueue | None = None
        self._frame_num = 0

    def set_faults(self, names: set[str] | frozenset[str] | None) -> None:
        """Replace this pipeline's active fault set (``None``/empty means
        none). Live: takes effect on the next frame this pipeline's own
        output queue generates, including on an already-running
        pipeline."""
        self._faults = set() if names is None else set(names)

    def active_faults(self) -> set[str]:
        return set(self._faults)

    def __enter__(self) -> "Pipeline":
        if self._requested_device_id not in self._available:
            raise RuntimeError(
                f"mock device {self._requested_device_id!r} not found "
                f"(available: {sorted(self._available)}) -- not silently substituting another device"
            )
        if "camera_required_failure" in self._faults:
            raise RuntimeError(f"mock camera_required_failure fault: {self._requested_device_id} failed to start")
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()
        return None

    def create(self, node_type: type) -> "_CameraNode | _VideoEncoderNode":
        if node_type is node.Camera:
            return _CameraNode(self)
        if node_type is node.VideoEncoder:
            return _VideoEncoderNode(self)
        raise ValueError(f"mock Pipeline does not implement node type {node_type!r}")

    def _bind_camera(self, socket: CameraBoardSocket) -> None:
        self._bound_socket = socket

    def _bind_encoder(self, encoder: _VideoEncoderNode, *, frame_rate: float, resolution: _Resolution) -> None:
        self._encoder = encoder
        self._frame_rate = frame_rate
        self._resolution = resolution

    def _create_output_queue(self, *, resolution: _Resolution) -> MessageQueue:
        # Only the encoder's output queue is meaningful for this adapter's
        # color-MJPEG path; the camera's own (unencoded) output queue is
        # accepted but never produces frames here, matching "only this
        # adapter's color-MJPEG path is required."
        if self._encoder is not None and resolution == self._resolution:
            queue = MessageQueue(
                fill=self._produce_frame_payload,
                interval=1.0 / self._frame_rate,
                clock=self._clock,
            )
            self._queue = queue
            return queue
        return MessageQueue(clock=self._clock)

    def start(self) -> None:
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._queue is not None:
            self._queue.close()

    def wait(self) -> None:
        return None

    def _produce_frame_payload(self) -> EncodedFrame | None:
        """This queue's own private frame source -- called by
        :class:`MessageQueue` itself, on its own pacing, never by the
        production adapter directly (there is no adapter-facing trigger
        method; the real SDK's adapter only ever calls ``tryGet()``)."""
        if not self._running:
            return None
        if "camera_optional_failure" in self._faults:
            raise RuntimeError(
                f"mock camera_optional_failure fault: no frame this poll ({self._requested_device_id})"
            )
        self._frame_num += 1
        data = _jpeg_bytes(
            width=self._resolution.width,
            height=self._resolution.height,
            seed=self._seed,
            frame_num=self._frame_num,
        )
        return EncodedFrame(
            data=data,
            width=self._resolution.width,
            height=self._resolution.height,
            sequence_num=self._frame_num,
            timestamp_s=self._clock.monotonic(),
        )

    def getCameraIntrinsics(
        self, socket: CameraBoardSocket, width: int, height: int
    ) -> tuple[tuple[float, ...], ...]:
        """Configured intrinsics for a simulated device -- a plausible
        pinhole-camera matrix, not invented extrinsics."""
        if socket != self._bound_socket:
            raise RuntimeError(f"no calibration for unbound socket {socket!r}")
        focal = float(max(width, height))
        return (
            (focal, 0.0, width / 2.0),
            (0.0, focal, height / 2.0),
            (0.0, 0.0, 1.0),
        )
