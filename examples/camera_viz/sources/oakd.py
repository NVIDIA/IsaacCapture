# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OAK-D source for camera_viz (mono / stereo / stereo_rgb).

DepthAI hands one ``dai.Device`` per physical OAK-D, so stereo can't be
two independent ``FrameSource``s. The pattern here:

  * ``_OakdDevice`` owns the ``dai.Device`` + ``dai.Pipeline`` + one
    producer thread. It exposes per-stream publish slots (``left`` /
    ``right`` / ``rgb``).
  * ``OakdSource`` is a thin, reference-counted handle over one slot.
    ``acquire()`` on the first source opens the device with the union of
    every stream the YAML asked for; ``release()`` on the last source
    closes it.

Stereo defaults to GRAY8 over USB to halve bandwidth (OAK-D's stereo
sensors are monochrome anyway); the GPU broadcasts it to RGBA. RGB
streams come over as planar BGR888p; depthai's host-side
``getCvFrame()`` packs them into HxWx3 BGR before we upload + GPU-swap
to RGBA.

Reconnect uses ``dai.Device.getAllAvailableDevices()`` as a pre-check
before constructing ``dai.Device(...)`` — otherwise a missing device
would block USB enumeration for ~10s.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

import numpy as np

from pipeline import Frame, FrameSource, SourceSpec
from ._helpers import alloc_pinned_host, notify, notify_verbose

logger = logging.getLogger("isaaccapture.camera_viz.sources.oakd")

RECONNECT_DELAY_S = 5.0


@dataclass
class _StreamSpec:
    """Static config for one OAK-D stream. The frame_type drives both the
    depthai pipeline node config and the GPU upload path."""

    name: str  # "left" | "right" | "rgb" | "mono"
    socket: str  # "LEFT" | "RIGHT" | "RGB"
    width: int
    height: int
    fps: int
    frame_type: str  # "gray" | "bgr"


@dataclass
class _StreamSlot:
    """Per-stream triple-buffered output + pinned host staging.

    Producer thread (one per device) writes into ``gpu_buffers[write_idx]``;
    consumer (renderer thread) reads via ``latest()``. ``cu_stream`` is a
    non-blocking CUDA stream so each stream's H2D + convert can issue
    concurrently with the others on the same producer thread.
    """

    spec: _StreamSpec
    gpu_buffers: list  # 3 × cupy ndarray, HxWx4 RGBA8 (triple-buffer mailbox)
    host_staging: np.ndarray  # HxW (gray) or HxWx3 (bgr)
    gpu_landing: object  # cupy ndarray matching host_staging shape
    cu_stream: object  # cupy.cuda.Stream
    write_idx: int = 0
    publish_idx: int = -1
    consumed_idx: int = -2
    lock: threading.Lock = field(default_factory=threading.Lock)
    queue: object = None  # depthai output queue

    def upload_and_convert(self, frame_np: np.ndarray) -> None:
        """Memcpy SDK frame into pinned host, then async H2D + GPU convert
        into the current write buffer. Caller must sync the stream before
        publishing so the consumer can read from any stream."""
        np.copyto(self.host_staging, frame_np)
        buf = self.gpu_buffers[self.write_idx]
        with self.cu_stream:
            self.gpu_landing.set(self.host_staging)
            if self.spec.frame_type == "gray":
                # Broadcast HxW gray to RGB; alpha pre-set to 255.
                buf[..., 0] = self.gpu_landing
                buf[..., 1] = self.gpu_landing
                buf[..., 2] = self.gpu_landing
            else:
                # BGR (HxWx3) → RGB via channel reverse into RGBA's first 3.
                buf[..., :3] = self.gpu_landing[..., ::-1]

    def publish(self) -> None:
        with self.lock:
            self.publish_idx = self.write_idx
        self.write_idx = (self.write_idx + 1) % len(self.gpu_buffers)

    def latest(self) -> Optional[Frame]:
        with self.lock:
            if self.publish_idx < 0 or self.publish_idx == self.consumed_idx:
                return None
            idx = self.publish_idx
            self.consumed_idx = idx
        return Frame(
            image=self.gpu_buffers[idx],
            timestamp_ns=time.monotonic_ns(),
            source_id=self.spec.name,
            stream=0,
        )


class _OakdDevice:
    """Owns one dai.Device + dai.Pipeline + one producer thread.

    Reference-counted by ``OakdSource`` handles. Thread-safe ``acquire`` /
    ``release`` — the first acquire opens the device, the last release
    closes it.
    """

    _SOCKET_MAP: ClassVar[dict[str, str]] = {
        "RGB": "CAM_A",
        "CAM_A": "CAM_A",
        "LEFT": "CAM_B",
        "CAM_B": "CAM_B",
        "RIGHT": "CAM_C",
        "CAM_C": "CAM_C",
    }

    def __init__(self, device_id: str, streams: List[_StreamSpec]) -> None:
        try:
            import cupy as cp  # noqa: F401  — pre-checked here so __init__ fails loudly
            import depthai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "OakdSource requires CuPy + depthai. "
                "Install via `uv pip install cupy-cuda12x depthai`."
            ) from e

        self._device_id = device_id
        self._stream_specs = streams
        self._slots: dict[str, _StreamSlot] = {}
        self._lock = threading.Lock()  # protects refcount + lifecycle
        self._refcount = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._device = None
        self._pipeline = None
        self._connected = False
        self._last_reconnect_attempt_s = 0.0
        self._reconnect_count = 0
        self._frame_counts: dict[str, int] = {s.name: 0 for s in streams}

        # Pre-allocate slots up front so OakdSource handles can be queried
        # for ``spec`` (resolution + format) before the device opens.
        self._build_slots()

    def _build_slots(self) -> None:
        import cupy as cp

        for s in self._stream_specs:
            if s.frame_type == "gray":
                staging = alloc_pinned_host((s.height, s.width), np.uint8)
                landing = cp.empty((s.height, s.width), dtype=cp.uint8)
            else:  # bgr (HxWx3)
                staging = alloc_pinned_host((s.height, s.width, 3), np.uint8)
                landing = cp.empty((s.height, s.width, 3), dtype=cp.uint8)
            gpu_buffers = [
                cp.empty((s.height, s.width, 4), dtype=cp.uint8) for _ in range(3)
            ]
            for b in gpu_buffers:
                b[..., 3] = 255
            self._slots[s.name] = _StreamSlot(
                spec=s,
                gpu_buffers=gpu_buffers,
                host_staging=staging,
                gpu_landing=landing,
                cu_stream=cp.cuda.Stream(non_blocking=True),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    def acquire(self) -> None:
        with self._lock:
            self._refcount += 1
            if self._refcount > 1:
                return
            # First acquire: start the producer thread; the device opens
            # lazily inside the loop so a missing camera doesn't crash
            # the runner's start().
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._produce_loop,
                name=f"oakd_{self._device_id or 'auto'}",
                daemon=False,
            )
            self._thread.start()

    def release(self) -> None:
        with self._lock:
            self._refcount -= 1
            if self._refcount > 0:
                return
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join()
        self._close_device()

    def latest(self, stream_name: str) -> Optional[Frame]:
        slot = self._slots.get(stream_name)
        if slot is None:
            return None
        return slot.latest()

    # ── Device open / close (called only from producer thread) ────────

    def _is_device_available(self) -> bool:
        """Cheap pre-check before ``dai.Device(...)``. Constructing the
        device for a non-enumerated camera blocks ~10s; this returns
        immediately."""
        import depthai as dai

        try:
            available = dai.Device.getAllAvailableDevices()
        except Exception:
            return False
        if not self._device_id:
            return bool(available)
        return any(self._device_id in str(d) for d in available)

    def _open_device(self) -> bool:
        import depthai as dai

        # Map our socket strings to the depthai enum.
        socket_enum = {
            "CAM_A": dai.CameraBoardSocket.CAM_A,
            "CAM_B": dai.CameraBoardSocket.CAM_B,
            "CAM_C": dai.CameraBoardSocket.CAM_C,
        }
        frame_type_enum = {
            "gray": dai.ImgFrame.Type.GRAY8,
            "bgr": dai.ImgFrame.Type.BGR888p,
        }

        device_info = (
            dai.DeviceInfo(self._device_id)
            if self._device_id
            else dai.Device.getAllAvailableDevices()[0]
        )
        device = dai.Device(device_info)

        # SUPER = USB 3.0+; HIGH / FULL / LOW = USB 2, silently caps fps
        # under stereo / BGR / 720p+ workloads.
        try:
            speed_name = getattr(device.getUsbSpeed(), "name", "?")
        except Exception:
            speed_name = "?"
        if speed_name in ("HIGH", "FULL", "LOW"):
            notify(
                "oakd",
                f"USB {speed_name} (NOT USB 3) — high fps / stereo / BGR will drop frames.",
            )
        else:
            notify("oakd", f"USB {speed_name}")

        pipeline = dai.Pipeline(device)
        for s in self._stream_specs:
            socket_key = self._SOCKET_MAP[s.socket.upper()]
            cam = pipeline.create(dai.node.Camera).build(socket_enum[socket_key])
            out = cam.requestOutput(
                (s.width, s.height), type=frame_type_enum[s.frame_type], fps=s.fps
            )
            self._slots[s.name].queue = out.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()
        self._device = device
        self._pipeline = pipeline
        return True

    def _close_device(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        for slot in self._slots.values():
            slot.queue = None
        self._connected = False

    # ── Producer loop ─────────────────────────────────────────────────

    def _produce_loop(self) -> None:
        import cupy as cp
        import depthai as dai  # local import keeps OAK-D-less envs runnable

        # Pin to the GPU our slot buffers + per-slot CUDA streams were
        # allocated on at __init__ time. On multi-GPU hosts VizSession may
        # have picked a non-default Vulkan adapter and this producer thread
        # otherwise defaults to GPU 0.
        first_slot = next(iter(self._slots.values()))
        device_id = int(first_slot.gpu_buffers[0].device.id)
        with cp.cuda.Device(device_id):
            self._produce_loop_inner(dai)

    def _produce_loop_inner(self, dai) -> None:
        first_frame_seen = False
        opening_notified = False
        unavailable_notified = False
        # Periodic actual-vs-requested fps; flags USB / VPU throttling.
        _FPS_REPORT_S = 5.0
        last_fps_report_at = time.monotonic()
        last_fps_counts = {s.name: 0 for s in self._stream_specs}
        while not self._stop.is_set():
            if not self._connected:
                now = time.monotonic()
                if now - self._last_reconnect_attempt_s < RECONNECT_DELAY_S:
                    self._stop.wait(timeout=0.1)
                    continue
                self._last_reconnect_attempt_s = now
                if not self._is_device_available():
                    if not unavailable_notified:
                        notify("oakd", "device not visible on USB; waiting")
                        unavailable_notified = True
                    continue
                if not opening_notified:
                    notify("oakd", "opening...")
                    opening_notified = True
                try:
                    self._connected = self._open_device()
                except Exception as e:
                    notify("oakd", f"open failed ({e})", logging.ERROR)
                    self._close_device()
                    self._reconnect_count += 1
                    continue
                notify("oakd", "connected")
                first_frame_seen = False
                opening_notified = False
                unavailable_notified = False

            try:
                self._pipeline.processTasks()
            except Exception as e:
                notify("oakd", f"pipeline error ({e}); reconnecting", logging.ERROR)
                self._close_device()
                self._reconnect_count += 1
                continue

            emitted_any = False
            try:
                for stream_spec in self._stream_specs:
                    slot = self._slots[stream_spec.name]
                    queue = slot.queue
                    if queue is None or not queue.has():
                        continue
                    msg = queue.get()
                    if not isinstance(msg, dai.ImgFrame):
                        continue
                    if stream_spec.frame_type == "gray":
                        frame = (
                            msg.getFrame()
                        )  # HxW uint8, zero-copy view into SDK buffer
                    else:
                        frame = (
                            msg.getCvFrame()
                        )  # HxWx3 BGR uint8 (planar → packed on host)
                    if frame is None:
                        continue
                    slot.upload_and_convert(frame)
                    slot.cu_stream.synchronize()
                    slot.publish()
                    self._frame_counts[stream_spec.name] += 1
                    emitted_any = True
            except Exception as e:
                notify("oakd", f"frame error ({e}); reconnecting", logging.ERROR)
                self._close_device()
                self._reconnect_count += 1
                continue

            if emitted_any and not first_frame_seen:
                first_frame_seen = True
                notify("oakd", "streaming")
            elif not emitted_any:
                # No queue had data this tick — brief yield to avoid burning CPU.
                self._stop.wait(timeout=0.001)

            # Periodic actual-vs-target fps; <80% flagged as throttled.
            now = time.monotonic()
            elapsed = now - last_fps_report_at
            if elapsed >= _FPS_REPORT_S and first_frame_seen:
                parts = []
                throttled = False
                for s in self._stream_specs:
                    delta = self._frame_counts[s.name] - last_fps_counts[s.name]
                    actual = delta / elapsed
                    parts.append(f"{s.name}={actual:.1f}/{s.fps}")
                    if actual < 0.8 * s.fps:
                        throttled = True
                    last_fps_counts[s.name] = self._frame_counts[s.name]
                tag = " ⚠ throttled" if throttled else ""
                notify_verbose("oakd", f"fps {' '.join(parts)}{tag}")
                last_fps_report_at = now


# ── Mode → streams config ──────────────────────────────────────────────


def _streams_for_mode(
    mode: str,
    width: int,
    height: int,
    fps: int,
    camera_socket: str,
    rgb_width: int,
    rgb_height: int,
    rgb_fps: int,
) -> List[_StreamSpec]:
    """Build the OAK-D stream layout for the requested mode."""
    mode = mode.lower()
    if mode == "mono":
        return [_StreamSpec("mono", camera_socket, width, height, fps, "bgr")]
    if mode == "stereo":
        # Mono sensors → GRAY8 over USB; halves bandwidth vs BGR.
        return [
            _StreamSpec("left", "LEFT", width, height, fps, "gray"),
            _StreamSpec("right", "RIGHT", width, height, fps, "gray"),
        ]
    if mode == "stereo_rgb":
        return [
            _StreamSpec("left", "LEFT", width, height, fps, "gray"),
            _StreamSpec("right", "RIGHT", width, height, fps, "gray"),
            _StreamSpec(
                "rgb",
                "RGB",
                rgb_width or width,
                rgb_height or height,
                rgb_fps or fps,
                "bgr",
            ),
        ]
    raise ValueError(
        f"OakdSource: unknown mode {mode!r} (expected mono | stereo | stereo_rgb)"
    )


class OakdSource(FrameSource):
    """Per-stream handle over a shared :class:`_OakdDevice`.

    Instantiate via :meth:`OakdSource.build`, which configures the device
    for all streams the requested mode needs and returns a list of handles
    (1 for mono, 2 for stereo, 3 for stereo_rgb).
    """

    def __init__(self, device: _OakdDevice, stream_name: str, spec: SourceSpec) -> None:
        self._device = device
        self._stream_name = stream_name
        self._spec = spec

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    @property
    def stream_name(self) -> str:
        return self._stream_name

    def start(self) -> None:
        self._device.acquire()

    def stop(self) -> None:
        self._device.release()

    def latest(self) -> Optional[Frame]:
        return self._device.latest(self._stream_name)

    @classmethod
    def build(
        cls,
        base_name: str,
        mode: str = "mono",
        device_id: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        camera_socket: str = "RGB",
        rgb_width: int = 0,
        rgb_height: int = 0,
        rgb_fps: int = 0,
    ) -> List["OakdSource"]:
        streams = _streams_for_mode(
            mode, width, height, fps, camera_socket, rgb_width, rgb_height, rgb_fps
        )
        device = _OakdDevice(device_id, streams)
        return [
            cls(
                device,
                s.name,
                SourceSpec(f"{base_name}_{s.name}", s.width, s.height, "rgba8"),
            )
            for s in streams
        ]
