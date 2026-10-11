# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EGO stereo source for camera_viz.

One source owns a native Capture instance and a producer thread. Each Frame
contains paired ColorLeft/ColorRight GPU RGBA8 images. The binding rejects
missing eyes or device timestamps more than 2 ms apart; publication uses
host monotonic time sampled after payload extraction, not camera exposure time.

MJPEG uses SDK conversion to host RGB, pinned staging, and CUDA upload.
H.264/H.265 host payloads feed one shared-codec NVDEC decoder per eye.
Both paths finish GPU writes before publishing reusable triple buffers.
The viewer supplies its CUDA device ID; buffers and both decoders use that GPU.

Open failures retry; capture-read exceptions reopen and reset the decoders.
Incomplete pairs and SDK timeouts are skipped. Other capture/decode failures
reach the viewer through latest() and exit through SDK cleanup. stop() requests
exit and waits up to five seconds, retaining the handle while the producer is live.

This source opens the camera independently of the recording plugin.
Sensor recording, control restoration, and MCAP output belong to that plugin;
run preview and recording separately because each needs device ownership.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from pipeline import Frame, FrameSource, SourceSpec
from ._helpers import alloc_pinned_host, notify, notify_verbose
from ._nv_decode import NvVideoDecoder

logger = logging.getLogger("isaaccapture.camera_viz.sources.ego")


class EgoSource(FrameSource):
    """One physical Ego producing timestamp-matched left/right GPU RGBA8."""

    def __init__(
        self,
        name: str,
        device_uid: str = "",
        width: int = 1600,
        height: int = 1300,
        fps: int = 30,
        format: str = "h264",
        gpu_id: int = 0,
        *,
        renderer_cuda_device_id: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("EGO width, height, and fps must be positive")
        self._spec = SourceSpec(name=name, width=width, height=height)
        self._device_uid = device_uid
        self._fps = fps
        self._format = format.lower()
        if self._format not in ("mjpg", "h264", "h265"):
            raise ValueError("EGO format must be mjpg, h264, or h265")
        self._requested_gpu_id = gpu_id
        self._renderer_cuda_device_id = renderer_cuda_device_id
        self._gpu_id = gpu_id
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._publish_idx = -1
        self._consumed_idx = -2
        self._timestamp_ns = 0
        self._write_idx = 0
        self._buffers_left = []
        self._buffers_right = []
        self._capture = None
        self._incomplete_pairs = 0
        self._frames = 0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        if self._thread is not None:
            return
        gpu_id = self._requested_gpu_id
        renderer_gpu = self._renderer_cuda_device_id
        if gpu_id < 0 or (renderer_gpu is not None and renderer_gpu < 0):
            raise ValueError("EGO requires a nonnegative CUDA device ID")
        if renderer_gpu is not None:
            # Zero retains the source's auto-selection contract, not a pin to GPU 0.
            if gpu_id != 0 and gpu_id != renderer_gpu:
                raise ValueError(
                    f"EGO source {self._spec.name!r} requests CUDA GPU {gpu_id}, "
                    f"but the renderer uses CUDA GPU {renderer_gpu}; "
                    "set gpu_id to 0 to follow the renderer"
                )
            gpu_id = renderer_gpu
        try:
            import cupy as cp
            from ego_preview import Capture  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "EgoSource requires CuPy and the native EGO binding. "
                "Run `camera_viz.sh setup --with-ego "
                "--sdk-root PATH`."
            ) from exc
        if renderer_gpu is None and gpu_id == 0:
            gpu_id = int(cp.cuda.runtime.getDevice())
        self._gpu_id = gpu_id
        with self._lock:
            self._error = None
            self._publish_idx = -1
            self._consumed_idx = -2
            self._timestamp_ns = 0
        self._write_idx = 0
        self._incomplete_pairs = 0
        self._frames = 0
        with cp.cuda.Device(self._gpu_id):
            buffers_left = [
                cp.empty((self._spec.height, self._spec.width, 4), dtype=cp.uint8)
                for _ in range(3)
            ]
            buffers_right = [
                cp.empty((self._spec.height, self._spec.width, 4), dtype=cp.uint8)
                for _ in range(3)
            ]
            for buffer in buffers_left + buffers_right:
                buffer[..., 3] = 255
        self._buffers_left, self._buffers_right = buffers_left, buffers_right
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._produce_loop, name=f"ego_{self._spec.name}", daemon=False
        )
        try:
            self._thread.start()
        except Exception:
            # An interrupt may arrive after the thread starts; retain it for stop().
            self._thread = None
            self._buffers_left, self._buffers_right = [], []
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Match NV's source stop budget; a live SDK call owns its handle.
                logger.warning(
                    "EGO producer did not exit within 5s; retaining SDK handle"
                )
                return
            self._thread = None
        self._capture = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    f"EGO source {self._spec.name!r} failed: {self._error}"
                ) from self._error
            if self._publish_idx < 0 or self._publish_idx == self._consumed_idx:
                return None
            index = self._publish_idx
            self._consumed_idx = index
            timestamp_ns = self._timestamp_ns
        return Frame(
            image=self._buffers_left[index],
            image_right=self._buffers_right[index],
            timestamp_ns=timestamp_ns,
            source_id=self._spec.name,
            stream=0,
        )

    def _produce_loop(self) -> None:
        try:
            self._capture_loop()
        except BaseException as exc:
            # Report before SDK cleanup, which may block while latest() is polled.
            self._record_error(exc)
            logger.exception("EGO producer stopped after a capture or decode failure")
        finally:
            try:
                self._close_capture()
            except BaseException as exc:
                self._record_error(exc)
                logger.exception("EGO SDK close failed")

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            # A cleanup failure must not replace the original producer failure.
            if self._error is None:
                self._error = exc
        self._stop.set()

    def _close_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.close()

    def _capture_loop(self) -> None:
        import cupy as cp
        from ego_preview import Capture

        with cp.cuda.Device(self._gpu_id):
            stream_left = cp.cuda.Stream(non_blocking=True)
            stream_right = cp.cuda.Stream(non_blocking=True)
            landing_left = landing_right = None
            staging_left = staging_right = None
            decoder_left = decoder_right = None
            if self._format == "mjpg":
                shape = (self._spec.height, self._spec.width, 3)
                staging_left = alloc_pinned_host(shape, np.uint8)
                staging_right = alloc_pinned_host(shape, np.uint8)
                landing_left = cp.empty(shape, dtype=cp.uint8)
                landing_right = cp.empty(shape, dtype=cp.uint8)
            else:
                decoder_left = NvVideoDecoder(
                    self._spec.width,
                    self._spec.height,
                    gpu_id=self._gpu_id,
                    codec=self._format,
                )
                decoder_right = NvVideoDecoder(
                    self._spec.width,
                    self._spec.height,
                    gpu_id=self._gpu_id,
                    codec=self._format,
                )

            last_report = time.monotonic()
            while not self._stop.is_set():
                if self._capture is None:
                    try:
                        notify("ego", "opening...")
                        self._capture = Capture(
                            self._device_uid,
                            self._spec.width,
                            self._spec.height,
                            self._fps,
                            self._format,
                        )
                        notify("ego", "connected")
                    except Exception as exc:
                        notify("ego", f"open failed ({exc}); retrying", logging.WARNING)
                        self._stop.wait(5.0)
                        continue
                try:
                    pair = self._capture.next_pair(100)
                except Exception as exc:
                    notify(
                        "ego",
                        f"capture failed ({exc}); reconnecting",
                        logging.WARNING,
                    )
                    self._close_capture()
                    if decoder_left:
                        decoder_left.reset()
                        decoder_right.reset()
                    continue
                if pair is None:
                    self._incomplete_pairs += 1
                    continue

                index = self._write_idx
                if self._format == "mjpg":
                    np.copyto(
                        staging_left,
                        np.frombuffer(pair["left"], dtype=np.uint8).reshape(
                            self._spec.height, self._spec.width, 3
                        ),
                    )
                    np.copyto(
                        staging_right,
                        np.frombuffer(pair["right"], dtype=np.uint8).reshape(
                            self._spec.height, self._spec.width, 3
                        ),
                    )
                    with stream_left:
                        landing_left.set(staging_left)
                        self._buffers_left[index][..., :3] = landing_left
                    with stream_right:
                        landing_right.set(staging_right)
                        self._buffers_right[index][..., :3] = landing_right
                    stream_left.synchronize()
                    stream_right.synchronize()
                    produced = True
                else:
                    produced_left = decoder_left.decode(
                        pair["left"], self._buffers_left[index]
                    )
                    produced_right = decoder_right.decode(
                        pair["right"], self._buffers_right[index]
                    )
                    produced = produced_left and produced_right
                if not produced:
                    self._incomplete_pairs += 1
                    continue
                with self._lock:
                    self._publish_idx = index
                    self._timestamp_ns = int(pair["timestamp_ns"])
                self._write_idx = (index + 1) % len(self._buffers_left)
                self._frames += 1

                now = time.monotonic()
                if now - last_report >= 5.0:
                    notify_verbose(
                        "ego",
                        f"frames={self._frames} incomplete={self._incomplete_pairs}",
                    )
                    last_report = now
