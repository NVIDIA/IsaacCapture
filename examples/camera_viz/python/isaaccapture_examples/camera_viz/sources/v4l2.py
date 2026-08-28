# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V4L2 (UVC webcam) source for camera_viz.

Drives any V4L2-compatible device via OpenCV's V4L2 backend. The hot
path is: ``cap.read()`` into pinned host memory → async ``cudaMemcpy``
into a pre-allocated GPU BGR landing buffer → GPU BGR→RGBA channel
reverse into the RGBA output buffer. Everything pre-allocated; no
per-frame allocation.

CAP_PROP_BUFFERSIZE is forced to 1 so the kernel-side V4L2 buffer
doesn't queue stale frames — the consumer always sees the latest
captured frame, not whatever sat in the queue since the last grab.

Hardware-decoded MJPG via NVDEC isn't in scope here; for sub-millisecond
decode at 4K, replace the OpenCV backend with a native NVDEC source.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._helpers import PolledSource


class V4l2Source(PolledSource):
    """Single-stream V4L2 capture (UVC webcam, /dev/video*)."""

    _kind = "v4l2"

    def __init__(
        self,
        name: str,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fourcc: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, width=width, height=height, staging_channels=3)
        self._device = device
        self._fps = float(fps)
        self._fourcc = fourcc

        # Dedicated GPU landing zone for the contiguous H2D copy. The base
        # class's RGBA output buffer is strided over 4 channels, so a direct
        # ``set()`` into it would force a slower pitched memcpy. One H2D
        # into ``_gpu_bgr`` + one GPU strided copy out is faster overall and
        # keeps the alpha channel untouched.
        self._gpu_bgr = self._cp.empty((height, width, 3), dtype=self._cp.uint8)

        self._cv2 = None  # imported lazily in _open_device
        self._cap = None

    def _open_device(self) -> bool:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as e:
                raise RuntimeError(
                    "V4l2Source requires opencv-python. Install via "
                    "`uv pip install opencv-python`."
                ) from e
            self._cv2 = cv2

        cv2 = self._cv2
        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        # FOURCC must be set BEFORE width/height/fps — many drivers ignore
        # subsequent property changes once the format is locked in.
        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._spec.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._spec.height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Drop kernel-side V4L2 queue depth — we want the latest frame, not
        # a stale one. Some drivers ignore this; OpenCV's V4L2 backend honors
        # it. Worth setting either way.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self._spec.width or actual_h != self._spec.height:
            cap.release()
            raise ValueError(
                f"V4l2Source '{self._spec.name}': requested "
                f"{self._spec.width}x{self._spec.height}, got {actual_w}x{actual_h}"
            )

        self._cap = cap
        return True

    def _close_device(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _grab(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            # Genuine read failure — caller treats this as fatal and triggers
            # reconnect. cv2 returns (False, None) on disconnect / EOF / sleep.
            raise RuntimeError("V4L2 read returned no frame")
        # Host→host memcpy into pinned staging. ~0.2ms at 1080p; the price
        # of OpenCV's API not letting us pass a pre-allocated dst.
        self._host_staging[...] = frame
        return self._host_staging

    def _upload_and_convert(self, gpu_buf) -> None:
        # Async H2D + GPU channel reverse (BGR → RGB). Alpha was set to 255
        # at construction and isn't touched here.
        with self._stream:
            self._gpu_bgr.set(self._host_staging)
            gpu_buf[..., :3] = self._gpu_bgr[..., ::-1]
