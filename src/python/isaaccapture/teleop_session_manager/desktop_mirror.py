# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration, cadence and copy completion for the owned desktop mirror."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DesktopMirrorConfig:
    """Configuration for the optional, application-side OVUI mirror."""

    max_fps: float = 30.0
    window_width: int = 960
    title: str = "Isaac Teleop — render products"
    initial_status_text: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_fps) or self.max_fps <= 0.0:
            raise ValueError("desktop mirror max_fps must be finite and positive")
        if self.window_width <= 0:
            raise ValueError("desktop mirror window_width must be positive")
        if not self.title:
            raise ValueError("desktop mirror title must be non-empty")


@dataclass(frozen=True)
class DesktopMirrorStatus:
    """Thread-safe snapshot of mirror lifecycle and latest-frame behavior."""

    thread_id: int | None = None
    window_opened: bool = False
    closed: bool = False
    error: str | None = None
    presented_frames: int = 0
    last_sequence: int = 0
    main_thread_fps: float = 0.0
    status_text: str = ""
    render_thread_fps: float = 0.0


class _CadenceSampler:
    """Measure recent host and render cadence from monotonic samples."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_ns = int(window_s * 1_000_000_000)
        self._host_samples: deque[int] = deque()
        self._render_samples: deque[int] = deque()

    def record_host_frame(self, now_ns: int) -> None:
        self._host_samples.append(now_ns)
        while (
            len(self._host_samples) > 2
            and self._host_samples[1] <= now_ns - self._window_ns
        ):
            self._host_samples.popleft()

    def record_render_frame(self, now_ns: int) -> None:
        self._render_samples.append(now_ns)
        while (
            len(self._render_samples) > 2
            and self._render_samples[1] <= now_ns - self._window_ns
        ):
            self._render_samples.popleft()

    @staticmethod
    def _event_rate(samples: deque[int] | list[int]) -> float:
        if len(samples) < 2:
            return 0.0
        elapsed_s = (samples[-1] - samples[0]) / 1_000_000_000
        return 0.0 if elapsed_s <= 0.0 else (len(samples) - 1) / elapsed_s

    def rates(self, now_ns: int | None = None) -> tuple[float, float]:
        def recent_rate(samples: deque[int]) -> float:
            if now_ns is not None and (
                not samples or now_ns - samples[-1] > self._window_ns
            ):
                return 0.0
            return self._event_rate(samples)

        return recent_rate(self._host_samples), recent_rate(self._render_samples)


class _InputCopyCompletion:
    """Fence OVUI 0.2 input uploads, not its unrelated output-streaming API.

    Both standalone backends copy the input on CUDA's legacy stream. Vulkan
    synchronizes internally, OpenGL only queues map/copy/unmap. An event recorded
    after set_bytes_data_from_gpu covers the last source read in both paths,
    including resize and a Python exception after native submission. Only the
    UI worker waits; OVXR skips captures when its optional slots are leased.
    """

    def __init__(self, device: int) -> None:
        from cuda.bindings import runtime

        self._cuda = runtime
        self._check(runtime.cudaSetDevice(device))
        self._event = self._check(
            runtime.cudaEventCreateWithFlags(runtime.cudaEventDisableTiming)
        )

    @staticmethod
    def _check(result):
        if int(result[0]):
            raise RuntimeError(f"OVUI input-copy completion failed: CUDA {result[0]}")
        return result[1] if len(result) > 1 else None

    def wait(self) -> None:
        self._check(
            self._cuda.cudaEventRecord(self._event, self._cuda.cudaStreamLegacy)
        )
        self._check(self._cuda.cudaEventSynchronize(self._event))

    def close(self) -> None:
        if self._event is not None:
            self._check(self._cuda.cudaEventDestroy(self._event))
            self._event = None
