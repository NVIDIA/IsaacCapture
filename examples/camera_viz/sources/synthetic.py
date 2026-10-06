# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-resident synthetic source for camera_viz.

Generates an animated test pattern entirely on the GPU via CuPy — no
H2D / D2H copies, so it doubles as a sanity check for the zero-copy
hot path before real cameras (M7b) land.

Uses double-buffering so the producer thread can update the next frame
while the renderer's ``cudaMemcpy2D`` is still consuming the current
one. The vendor-SDK sources will model the same pattern.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from pipeline import Frame, FrameSource, SourceSpec


class SyntheticSource(FrameSource):
    """Two-buffer GPU source emitting an animated RGBA8 pattern."""

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fps: float = 60.0,
        hue_speed_hz: float = 0.25,
    ) -> None:
        # CuPy is required for GPU-resident sources. We import lazily so
        # camera_viz can advertise this source's existence on a CuPy-less
        # box without crashing module import.
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "SyntheticSource requires CuPy (cupy-cuda12x). Install via "
                "`uv pip install cupy-cuda12x` or skip this source."
            ) from e

        self._cp = cp
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._frame_interval_s = 1.0 / fps if fps > 0.0 else 0.0
        self._hue_speed_hz = hue_speed_hz

        # Triple-buffer — matches the other sources' mailbox depth so
        # the consumer's async copy has at least one producer cycle to
        # finish before the producer wraps back to the same buffer.
        self._buffers = [cp.zeros((height, width, 4), dtype=cp.uint8) for _ in range(3)]
        self._write_idx = 0
        self._publish_idx: int = -1  # -1 = nothing published yet
        self._consumed_idx: int = -2  # track what `latest()` last returned
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0_ns = 0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._t0_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._produce_loop, name=f"synth_{self._spec.name}", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            if self._publish_idx < 0 or self._publish_idx == self._consumed_idx:
                return None
            idx = self._publish_idx
            self._consumed_idx = idx
        return Frame(
            image=self._buffers[idx],
            timestamp_ns=time.monotonic_ns(),
            source_id=self._spec.name,
            stream=0,
        )

    def _produce_loop(self) -> None:
        cp = self._cp
        h, w = self._spec.height, self._spec.width
        # Pin this producer thread to the GPU the pre-allocated buffers
        # live on. On multi-GPU hosts VizSession picks the Vulkan adapter
        # (potentially non-default), the buffers land on that device, and
        # the producer thread otherwise defaults to GPU 0 — every kernel
        # then fires CuPy's cross-device peer-access fallback and warns.
        with cp.cuda.Device(int(self._buffers[0].device.id)):
            # Precompute coordinate grids once; reused every frame for the pattern.
            y_grid = cp.arange(h, dtype=cp.float32).reshape(h, 1)
            x_grid = cp.arange(w, dtype=cp.float32).reshape(1, w)
            diag = (x_grid + y_grid) / float(w + h)  # 0..1 sweep

            while not self._stop.is_set():
                t = (time.monotonic_ns() - self._t0_ns) * 1e-9
                phase = (t * self._hue_speed_hz) % 1.0
                # Three offset sinusoids over the diagonal — readable on any HMD.
                r = (cp.sin((diag + phase) * 6.2831853) * 127.0 + 128.0).astype(
                    cp.uint8
                )
                g = (
                    cp.sin((diag + phase + 0.3333) * 6.2831853) * 127.0 + 128.0
                ).astype(cp.uint8)
                b = (
                    cp.sin((diag + phase + 0.6667) * 6.2831853) * 127.0 + 128.0
                ).astype(cp.uint8)

                buf = self._buffers[self._write_idx]
                buf[..., 0] = r
                buf[..., 1] = g
                buf[..., 2] = b
                buf[..., 3] = 255
                # Block until the kernel writes are visible to the renderer's
                # consumer copy (which runs on stream 0). Without this the
                # async kernel could still be in flight when submit() issues
                # its cudaMemcpy2D and we'd see torn frames.
                cp.cuda.Stream.null.synchronize()

                with self._lock:
                    self._publish_idx = self._write_idx
                self._write_idx = (self._write_idx + 1) % len(self._buffers)

                if self._frame_interval_s > 0.0:
                    # Coarse pacing — CuPy fill kernels at 1080p are well under
                    # the budget, so a simple sleep keeps us at target fps.
                    time.sleep(self._frame_interval_s)


class SyntheticStereoSource(FrameSource):
    """Stereo variant of SyntheticSource with a horizontal pixel
    disparity between eyes; for hardware-free stereo testing."""

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fps: float = 60.0,
        hue_speed_hz: float = 0.25,
        disparity_px: int = 20,
    ) -> None:
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "SyntheticStereoSource requires CuPy (cupy-cuda12x). Install via "
                "`uv pip install cupy-cuda12x` or skip this source."
            ) from e

        self._cp = cp
        self._spec = SourceSpec(
            name=name, width=width, height=height, pixel_format="rgba8"
        )
        self._frame_interval_s = 1.0 / fps if fps > 0.0 else 0.0
        self._hue_speed_hz = hue_speed_hz
        self._disparity_px = int(disparity_px)

        # Lock-step per-eye triple-buffer; latest() returns matching pairs.
        self._left = [cp.zeros((height, width, 4), dtype=cp.uint8) for _ in range(3)]
        self._right = [cp.zeros((height, width, 4), dtype=cp.uint8) for _ in range(3)]
        self._write_idx = 0
        self._publish_idx: int = -1
        self._consumed_idx: int = -2
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0_ns = 0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._t0_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._produce_loop,
            name=f"synth_stereo_{self._spec.name}",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest(self) -> Optional[Frame]:
        with self._lock:
            if self._publish_idx < 0 or self._publish_idx == self._consumed_idx:
                return None
            idx = self._publish_idx
            self._consumed_idx = idx
        return Frame(
            image=self._left[idx],
            image_right=self._right[idx],
            timestamp_ns=time.monotonic_ns(),
            source_id=self._spec.name,
            stream=0,
        )

    def _produce_loop(self) -> None:
        cp = self._cp
        h, w = self._spec.height, self._spec.width
        with cp.cuda.Device(int(self._left[0].device.id)):
            y_grid = cp.arange(h, dtype=cp.float32).reshape(h, 1)
            x_grid_l = cp.arange(w, dtype=cp.float32).reshape(1, w)
            # Right-eye x shifted by disparity_px → visible parallax.
            x_grid_r = (cp.arange(w, dtype=cp.float32) - self._disparity_px).reshape(
                1, w
            )
            diag_l = (x_grid_l + y_grid) / float(w + h)
            diag_r = (x_grid_r + y_grid) / float(w + h)

            # Defined once, above the loop: taking phase as a parameter avoids
            # rebuilding a closure over it on every generated frame.
            def fill(buf, diag, phase):
                r = (cp.sin((diag + phase) * 6.2831853) * 127.0 + 128.0).astype(
                    cp.uint8
                )
                g = (
                    cp.sin((diag + phase + 0.3333) * 6.2831853) * 127.0 + 128.0
                ).astype(cp.uint8)
                b = (
                    cp.sin((diag + phase + 0.6667) * 6.2831853) * 127.0 + 128.0
                ).astype(cp.uint8)
                buf[..., 0] = r
                buf[..., 1] = g
                buf[..., 2] = b
                buf[..., 3] = 255

            while not self._stop.is_set():
                t = (time.monotonic_ns() - self._t0_ns) * 1e-9
                phase = (t * self._hue_speed_hz) % 1.0

                fill(self._left[self._write_idx], diag_l, phase)
                fill(self._right[self._write_idx], diag_r, phase)
                cp.cuda.Stream.null.synchronize()

                with self._lock:
                    self._publish_idx = self._write_idx
                self._write_idx = (self._write_idx + 1) % len(self._left)

                if self._frame_interval_s > 0.0:
                    time.sleep(self._frame_interval_s)
