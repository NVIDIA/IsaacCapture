# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The production DepthAI acquisition adapter: owns stable camera
selection, camera/encoder configuration, the single rate-paced loop that
polls the encoded-frame queue, and resource release for one configured
camera stream.

Like :mod:`capture_server.acquisition.isaac_capture`, this adapter owns
its own acquisition cadence: camera queues are independently paced by
their camera/encoder, and polling an empty queue must not manufacture a
new frame. This file is an ordinary adapter, not a mock helper: everything
mock-specific (synthetic JPEG content, native-style device inventory,
injected failures) lives in
:mod:`capture_server.mock.mock_depthai`; everything here is
SDK-shape-agnostic composition/mapping the real adapter sharing this
same class would also need.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

from capture_server import protocol as p

from ._thread_runner import ThreadRunner
from .clock import Clock, MonotonicClock
from .errors import CameraSourceError

__all__ = ["DepthAIAdapter"]


# How long a started camera may produce nothing before it counts as failed (at least five frame periods,
# so a very low configured fps is not mistaken for a stalled camera).
_NO_FRAME_TIMEOUT_S = 5.0


class DepthAIAdapter:
    """Wraps one ``Pipeline``-shaped object for the lifetime of one
    configured camera stream. Identical code path for mock and real
    SDKs: ``sdk`` supplies only the real, SDK-shared
    ``node``/``VideoEncoderProperties`` construction surface used to build
    the ``Camera -> VideoEncoder -> encoded queue`` graph; ``pipeline_
    factory`` is a ``Callable[[], Pipeline]`` that returns a ready
    -to-enter pipeline. Launch composition supplies ``pipeline_factory``
    -- for mock mode a closure/partial over ``mock_depthai.Pipeline`` with
    device id/seed/fault options already bound; for real mode,
    a pipeline bound to the station's physical device. This class never sees or branches on
    seed/faults -- those are mock-only concepts that belong entirely
    inside ``mock_depthai.py``."""

    def __init__(
        self,
        *,
        sdk: ModuleType,
        pipeline_factory: Callable[[], Any],
        camera: p.CameraStreamConfig,
        clock: Clock | None = None,
    ) -> None:
        self._sdk = sdk
        self._pipeline_factory = pipeline_factory
        self._camera = camera
        self._interval = 1.0 / camera.fps
        self._clock = clock or MonotonicClock()
        self._runner = ThreadRunner()
        self._pipeline: Any = None
        self._queue: Any = None
        # True only once pipeline.__enter__() itself returns -- close()
        # must never pair __exit__ with an __enter__ that failed or never
        # ran, the same contract a `with` block enforces.
        self._pipeline_entered = False
        self._deadline: float | None = None
        self._last_frame_s = 0.0
        self._intrinsics: tuple[tuple[float, float, float], ...] | None = None
        # Pessimistic until a real success is actually confirmed -- never
        # default to a healthy status before startup has genuinely
        # finished (see camera_status()).
        self._degraded = True
        self._last_error = "camera not yet started"

    @property
    def stream_id(self) -> int:
        return self._camera.stream_id

    @property
    def required(self) -> bool:
        return self._camera.required

    def _enter_sync(self) -> Any:
        try:
            # Creating the pipeline binds the physical device, so it can fail (device absent or busy).
            pipeline = self._pipeline_factory()
            # Record ownership BEFORE any further native work, on this same
            # worker thread -- if the asyncio await wrapping this call is
            # cancelled (or this entry only partially succeeds) after this
            # point, close() must still be able to find and release whatever
            # was actually constructed, rather than seeing a still-None field
            # set only by an enter() continuation that may never run.
            self._pipeline = pipeline
            pipeline.__enter__()
            self._pipeline_entered = True
            camera_node = pipeline.create(self._sdk.node.Camera).build(self._sdk.CameraBoardSocket.CAM_A)
            # The camera's own output FPS and NV12 format are set here (the encoder's FPS alone does not
            # configure acquisition); NV12 is what the on-device MJPEG encoder takes.
            output = camera_node.requestOutput(
                (self._camera.width, self._camera.height),
                type=self._sdk.ImgFrame.Type.NV12,
                fps=self._camera.fps,
            )
            encoder = pipeline.create(self._sdk.node.VideoEncoder).build(
                output,
                frameRate=self._camera.fps,
                profile=self._sdk.VideoEncoderProperties.Profile.MJPEG,
                quality=self._camera.jpeg_quality,
            )
            # The queue is a LOCAL until start() has actually succeeded --
            # self._queue (and therefore _poll_sync's "never started, poll
            # forever returns None" vs. "poll the real queue" branch) must
            # never be trusted/published on a pipeline that only partially
            # started, or an empty poll afterward would clear _degraded
            # and falsely report CONNECTED for a camera that never ran.
            queue = encoder.out.createOutputQueue()
            # Explicit policy, not an assumed SDK default: bounded,
            # nonblocking, latest-frame. tryGet() is already nonblocking
            # regardless of this flag, but setting it makes the policy
            # this adapter actually wants unambiguous rather than
            # relying on whatever the SDK's own default queue behavior
            # happens to be.
            queue.setMaxSize(1)
            queue.setBlocking(False)
            pipeline.start()
            if not pipeline.isRunning():
                raise RuntimeError("pipeline did not report running after start()")
        except Exception as exc:  # noqa: BLE001 -- native entry failure, mapped uniformly
            self._last_error = str(exc) or type(exc).__name__
            raise CameraSourceError(self._last_error, camera_name=self._camera.name) from exc
        else:
            self._queue = queue
            self._last_frame_s = self._clock.monotonic()
        return pipeline

    def _frame_intrinsics(self, frame: Any) -> tuple[tuple[float, float, float], ...] | None:
        """The intrinsics the delivered image itself carries, or None -- never inferred from the sensor.

        A real frame reports the transformation from sensor to output, so its matrix describes exactly this
        stream's geometry. The mock frame has no transformation and yields None.
        """
        import math

        get_transformation = getattr(frame, "getTransformation", None)
        if get_transformation is None:
            return None
        try:
            rows = tuple(tuple(float(v) for v in row) for row in get_transformation().getIntrinsicMatrix())
        except Exception:  # noqa: BLE001 -- optional metadata; absence is reported, not fatal
            return None
        if len(rows) != 3 or any(len(r) != 3 or not all(math.isfinite(v) for v in r) for r in rows):
            return None
        return rows

    async def enter(self) -> None:
        # The deadline is set FIRST and unconditionally: Acquisition.enter
        # tolerates an optional camera's startup failure and keeps going,
        # so next_camera_jpeg() must still be callable afterward (it then
        # just returns None forever -- see _poll_sync's self._queue is
        # None check) rather than asserting on a deadline that only a
        # successful entry would otherwise have set.
        self._deadline = self._clock.monotonic() + self._interval
        await self._runner.run(self._enter_sync)  # self._pipeline is set inside _enter_sync itself
        self._degraded = False
        self._last_error = ""

    def _next_deadline(self, deadline: float) -> float:
        """Advance *deadline* to the next tick strictly after "now",
        skipping any ticks that are overdue rather than bursting to catch
        up. Same policy as :meth:`IsaacCaptureAdapter._next_deadline`."""
        now = self._clock.monotonic()
        while deadline <= now:
            deadline += self._interval
        return deadline

    def _poll_sync(self) -> tuple[int, bytes] | None:
        if self._queue is None:
            # Entry never actually succeeded (an optional camera's
            # startup failure that Acquisition.enter chose to tolerate
            # rather than end the session) -- nothing to poll, forever.
            return None
        try:
            frame = self._queue.tryGet()
        except Exception as exc:  # noqa: BLE001 -- native acquisition failure, mapped uniformly
            self._degraded = True
            self._last_error = str(exc) or type(exc).__name__
            raise CameraSourceError(self._last_error, camera_name=self._camera.name) from exc
        if frame is None:
            silent_s = self._clock.monotonic() - self._last_frame_s
            if silent_s > max(_NO_FRAME_TIMEOUT_S, 5 * self._interval):
                # A started pipeline that stops producing is a failed camera, never "connected" forever.
                self._degraded = True
                self._last_error = f"no frame from the camera for {silent_s:.1f}s"
                raise CameraSourceError(self._last_error, camera_name=self._camera.name)
            return None
        self._last_frame_s = self._clock.monotonic()
        self._intrinsics = self._frame_intrinsics(frame)
        # KNOWN LIMITATION, real hardware only: this size-one, non-blocking
        # queue can be overwritten by a newer encoded frame before this
        # poll runs. An overwritten frame never reaches this method at
        # all, so it is invisible to every counter this adapter reports --
        # it is not the same event as an empty poll (no frame ready) and
        # is not covered by this adapter's own sequence numbering (which
        # only advances on frames actually delivered here). The native
        # frame type exposes getSequenceNum() for its own encoder-side
        # count, which would detect a gap, but folding that into
        # produced/dropped would redefine produced from "delivered to
        # this adapter" to "generated by the camera" -- a protocol-level
        # semantics change, not something to invent unreviewed here.
        # Cannot be exercised against the mock queue, which never
        # overwrites a pending frame.
        # Stamp the host-sampling timestamp HERE, at the real tryGet()
        # poll that produced this frame -- the wire protocol's
        # sample_time_semantics is HOST_POLL (the timestamp marks the
        # host sampling step, not a hardware exposure time or SDK
        # callback time), so capture_time_ns must mark this sampling
        # step, not whenever the caller later publishes the frame after
        # validation/encoding/thread-hop.
        sample_time_ns = int(self._clock.monotonic() * 1_000_000_000)
        # Bounds-check BEFORE copying: inspect dimensions/length first (a
        # native buffer's length is readable without materializing an
        # owned copy of its contents) so an oversized/malformed buffer is
        # rejected without ever being copied.
        try:
            dims_error = self._validate_dimensions(frame)
            if dims_error is None:
                # getData() is called exactly ONCE -- the real binding returns a native SDK buffer, not
                # guaranteed-owned Python bytes, and two getData() calls need not return the same buffer.
                # One memoryview wraps that single result so the buffer whose length is validated is
                # the exact buffer that gets copied.
                view = memoryview(frame.getData())
        except Exception as exc:  # noqa: BLE001 -- a native frame-read failure is a camera failure
            self._degraded = True
            self._last_error = f"could not read the camera frame: {exc or type(exc).__name__}"
            raise CameraSourceError(self._last_error, camera_name=self._camera.name) from exc
        if dims_error is not None:
            self._degraded = True
            self._last_error = dims_error
            raise CameraSourceError(dims_error, camera_name=self._camera.name)
        size_error = self._validate_size(view.nbytes)
        if size_error is not None:
            self._degraded = True
            self._last_error = size_error
            raise CameraSourceError(size_error, camera_name=self._camera.name)
        data = bytes(view)  # always allocates fresh, owned storage
        marker_error = self._validate_markers(data)
        if marker_error is not None:
            self._degraded = True
            self._last_error = marker_error
            raise CameraSourceError(marker_error, camera_name=self._camera.name)
        self._degraded = False
        self._last_error = ""
        return sample_time_ns, data

    def _validate_dimensions(self, frame: Any) -> str | None:
        width, height = frame.getWidth(), frame.getHeight()
        if width != self._camera.width or height != self._camera.height:
            return (
                f"frame dimensions {width}x{height} do not match the accepted "
                f"stream's {self._camera.width}x{self._camera.height}"
            )
        return None

    # The wire protocol's JPEG payload ceiling is a hard 2 MiB -- the
    # complete binary image message is at most 2 MiB plus the 32-byte
    # header, and CaptureClient enforces exactly that limit on its own
    # receive side (``max_size``), so anything this adapter accepts above
    # it would get the connection closed with code 1009 on the client.
    _SPEC_MAX_ENCODED_BYTES = 2 * 1024 * 1024

    def _max_encoded_bytes(self) -> int:
        """Raw RGB size plus a fixed header-overhead margin, same as
        before -- a legitimately valid small JPEG's header/metadata
        overhead can exceed its own tiny raw-pixel count, so a bare
        raw-pixel bound would wrongly reject small-but-valid frames.
        Still clamped to the hard 2 MiB spec ceiling above: at larger
        resolutions (720p+) the raw-pixel-based estimate alone can exceed
        what the client will actually accept, which is the bug this
        clamp fixes -- the margin only ever matters for small frames
        where it stays below 2 MiB on its own."""
        return min(
            self._camera.width * self._camera.height * 3 + 65536,
            self._SPEC_MAX_ENCODED_BYTES,
        )

    def _validate_size(self, size: int) -> str | None:
        if size == 0:
            return "empty encoded frame"
        max_bytes = self._max_encoded_bytes()
        if size > max_bytes:
            return f"encoded frame of {size} bytes exceeds the {max_bytes}-byte ceiling for this stream"
        return None

    @staticmethod
    def _validate_markers(data: bytes) -> str | None:
        """Encoded-sanity check only -- SOI/EOI presence, not a pixel or
        tracking-style decode."""
        if data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
            return "encoded frame is not a complete JPEG (missing SOI/EOI markers)"
        return None

    async def next_camera_jpeg(self) -> tuple[int, bytes] | None:
        """Owns this camera's single acquisition cadence: sleeps until the
        next monotonic deadline (skipping overdue ticks), then polls the
        encoded queue -- ``MessageQueue.tryGet()``, the real SDK's own
        nonblocking call -- on this adapter's dedicated thread. There is
        no adapter-side trigger call; the mock queue paces/fills itself
        independently, matching how a real encoder would only have a new
        frame ready at its own cadence regardless of poll frequency.
        Returns ``None`` for an empty poll -- polling an empty queue must
        not manufacture a new frame -- which the caller must not treat as
        a failure. On a real frame, returns the
        host-monotonic sampling timestamp paired with the encoded JPEG
        bytes -- see ``_poll_sync`` for where that timestamp is taken."""
        assert self._deadline is not None, "next_camera_jpeg() called before enter()"
        import asyncio

        await asyncio.sleep(max(0.0, self._deadline - self._clock.monotonic()))
        self._deadline = self._next_deadline(self._deadline)
        return await self._runner.run(self._poll_sync)

    def camera_status(self) -> p.CameraStatus:
        return p.CameraStatus(
            name=self._camera.name,
            stream_id=self._camera.stream_id,
            status=p.DeviceState.DEGRADED if self._degraded else p.DeviceState.CONNECTED,
            reason=p.StatusReason.NO_CURRENT_DATA if self._degraded else p.StatusReason.NONE,
            error=self._last_error if self._degraded else "",
            intrinsics=self._intrinsics,
        )

    async def close(self) -> bool:
        def _close() -> None:
            # Read at EXECUTION time (on this adapter's single-worker
            # executor, strictly after any earlier-queued _enter_sync has
            # finished) -- never a value captured eagerly when close()
            # was first awaited, which could still be None if entry's own
            # job hadn't run yet.
            pipeline = self._pipeline
            try:
                if pipeline is not None and self._pipeline_entered:
                    pipeline.__exit__(None, None, None)
            finally:
                # Dropping the last references is what releases the bound device.
                self._pipeline = None
                self._queue = None

        return await self._runner.close(_close)
