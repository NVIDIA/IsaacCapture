# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Jetson Argus camera source for low-latency local capture."""

from __future__ import annotations

import threading
from typing import Optional, Sequence

from ..pipeline import Frame, FrameSource, SourceSpec

from ._helpers import notify


class _CudaDevice:
    def __init__(self, gpu_id: int) -> None:
        self.id = int(gpu_id)


class _CudaFrameView:
    """Small CUDA-array-interface wrapper around native-owned device memory."""

    def __init__(
        self, data: int, width: int, height: int, pitch: int, gpu_id: int
    ) -> None:
        self._iface = {
            "shape": (int(height), int(width), 4),
            "strides": (int(pitch), 4, 1),
            "typestr": "|u1",
            "data": (int(data), False),
            "version": 3,
        }
        self.device = _CudaDevice(gpu_id)

    @property
    def __cuda_array_interface__(self):
        return self._iface


class ArgusSource(FrameSource):
    """Argus STREAM_TYPE_EGL -> CUDA -> RGBA8 source.

    The native module owns Argus/CUDA resources and publishes a latest-frame
    mailbox backed by CUDA device memory. Python exposes those pointers via
    ``__cuda_array_interface__`` so QuadLayer.submit() and the native encoder
    can consume them without a copy or an isaaccapture.viz dependency.
    """

    _kind = "argus"

    def __init__(
        self,
        name: str,
        sensor_ids: Sequence[int],
        width: int,
        height: int,
        sensor_mode: int = 0,
        fps: float = 30.0,
        gpu_id: int = 0,
        full_range: bool = False,
        swap_uv: bool = False,
        acquire_timeout_ms: int = 0xFFFFFFFF,
        repeat_capture: bool = True,
    ) -> None:
        try:
            from ..argus import ArgusCamera, ArgusConfig
        except ImportError as e:
            raise RuntimeError(
                "Argus source requires the native module. Build it with "
                "`examples/camera_viz/python/isaaccapture_examples/camera_viz/argus/build.sh` on the Jetson."
            ) from e

        sensor_ids = [int(s) for s in sensor_ids]
        if len(sensor_ids) not in (1, 2):
            raise ValueError("ArgusSource requires one or two sensor ids")

        cfg = ArgusConfig()
        cfg.name = name
        cfg.sensor_ids = sensor_ids
        cfg.sensor_mode = int(sensor_mode)
        cfg.width = int(width)
        cfg.height = int(height)
        cfg.fps = float(fps)
        cfg.gpu_id = int(gpu_id)
        cfg.full_range = bool(full_range)
        cfg.swap_uv = bool(swap_uv)
        cfg.acquire_timeout_ms = int(acquire_timeout_ms)
        cfg.repeat_capture = bool(repeat_capture)

        self._spec = SourceSpec(
            name=name, width=int(width), height=int(height), pixel_format="rgba8"
        )
        self._camera = ArgusCamera(cfg)
        self._gpu_id = int(gpu_id)
        self._stereo = len(sensor_ids) == 2

        self._lock = threading.Lock()
        self._start_refs = 0
        self._cached_view = None
        self._paired_consumed_sequence = 0
        self._eye_consumed_sequence = {"left": 0, "right": 0}

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        self._acquire_start_ref()

    def stop(self) -> None:
        self._release_start_ref()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            view = self._fresh_view_locked("paired")
            if view is None:
                return None
            self._paired_consumed_sequence = int(view.sequence)
            left = self._make_cuda_view(int(view.left_ptr), int(view.left_pitch))
            right = None
            if bool(view.stereo):
                right = self._make_cuda_view(int(view.right_ptr), int(view.right_pitch))
            return Frame(
                image=left,
                image_right=right,
                timestamp_ns=int(view.timestamp_ns),
                source_id=self._spec.name,
                stream=0,
            )

    def eye_source(self, eye: str, name: Optional[str] = None) -> FrameSource:
        if not self._stereo:
            raise ValueError("ArgusSource.eye_source() requires a stereo ArgusSource")
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        return _ArgusEyeSource(
            parent=self, eye=eye, name=name or f"{self._spec.name}_{eye}"
        )

    def _acquire_start_ref(self) -> None:
        with self._lock:
            if self._start_refs == 0:
                notify(self._kind, f"opening {self._spec.name}...")
                try:
                    self._camera.start()
                except Exception:
                    self._start_refs = 0
                    raise
                notify(self._kind, f"{self._spec.name} streaming")
            self._start_refs += 1

    def _release_start_ref(self) -> None:
        with self._lock:
            if self._start_refs <= 0:
                return
            self._start_refs -= 1
            if self._start_refs == 0:
                self._camera.stop()
                self._cached_view = None
                self._paired_consumed_sequence = 0
                self._eye_consumed_sequence = {"left": 0, "right": 0}

    def _latest_eye(self, eye: str, source_id: str) -> Optional[Frame]:
        with self._lock:
            view = self._fresh_view_locked(eye)
            if view is None:
                return None
            self._eye_consumed_sequence[eye] = int(view.sequence)
            if eye == "left":
                image = self._make_cuda_view(int(view.left_ptr), int(view.left_pitch))
            else:
                image = self._make_cuda_view(int(view.right_ptr), int(view.right_pitch))
            return Frame(
                image=image,
                timestamp_ns=int(view.timestamp_ns),
                source_id=source_id,
                stream=0,
            )

    def _fresh_view_locked(self, consumer: str):
        cached = self._cached_view
        if cached is not None:
            seq = int(cached.sequence)
            if consumer == "paired" and self._paired_consumed_sequence != seq:
                return cached
            if (
                consumer in ("left", "right")
                and self._eye_consumed_sequence[consumer] != seq
            ):
                return cached

        view = self._camera.latest()
        if view is None:
            return None
        self._cached_view = view
        return view

    def _make_cuda_view(self, data: int, pitch: int) -> _CudaFrameView:
        return _CudaFrameView(
            data=data,
            width=self._spec.width,
            height=self._spec.height,
            pitch=pitch,
            gpu_id=self._gpu_id,
        )


class _ArgusEyeSource(FrameSource):
    def __init__(self, parent: ArgusSource, eye: str, name: str) -> None:
        self._parent = parent
        self._eye = eye
        self._spec = SourceSpec(
            name=name,
            width=parent.spec.width,
            height=parent.spec.height,
            pixel_format=parent.spec.pixel_format,
        )

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        self._parent._acquire_start_ref()

    def latest(self) -> Optional[Frame]:
        return self._parent._latest_eye(self._eye, self._spec.name)

    def stop(self) -> None:
        self._parent._release_start_ref()
