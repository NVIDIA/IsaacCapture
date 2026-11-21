#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone AVS1 receiver and CUDA IPC publisher.

The sender emits one AVS1 bundle per TCP record:

    uint32_be_record_bytes || AVS1 bundle

This process connects to the sender, decodes or copies configured streams into
GPU-resident rings, and publishes those rings over CUDA IPC for same-host
consumers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import yaml

from avs_wire import (
    CODEC_HEVC,
    CODEC_RAW_NV12,
    HEADER as AVS1_HEADER,
    Avs1Bundle,
    Avs1Sample,
    Avs1Stream,
    parse_avs1_record,
    sensor_sof_lag_ns,
    sensor_sof_ns,
)
from cuda_ipc import CudaIpcFramePublisher
from nv_video_decode import NvVideoDecoder

logger = logging.getLogger("isaaccapture.camera_streamer.receiver")

_SOCK_TIMEOUT_S = 0.5
# If no bytes arrive for this long the link is treated as dead and torn down so
# the produce loop reconnects. A half-open peer -- e.g. the AGX rebooted -- leaves
# the PC socket ESTABLISHED with data that never comes; without this bound recv
# just keeps timing out on that dead socket forever and the camera never returns
# until the receiver is killed by hand. Far above a 60 fps inter-frame gap (~17ms)
# so a healthy stream never trips it.
_STALL_TIMEOUT_S = 3.0
# Grace for a FRESH connection before the stall watchdog arms. The AGX sender's
# NVENC/capture-domain cold start takes ~2-3 s to emit its first frame bundle;
# a 3 s stall timeout applied from connect fires DURING that warmup, before any
# data has ever arrived, and tears the link down in a reconnect loop that never
# lets video through. So until the first bundle is received we allow this longer
# window; once the stream is flowing, _STALL_TIMEOUT_S governs a real stall.
_COLD_START_TIMEOUT_S = 15.0
_CONNECT_WAIT_LOG_INTERVAL_S = 5.0
_DEFAULT_MAX_RECORD_BYTES = 256 * 1024 * 1024
_DEFAULT_STATS_LOG_INTERVAL_S = 5.0
_DEFAULT_RECEIVE_FPS_WARNING_INTERVAL_S = 10.0
# AVS1 device index a sender writes when it has no camera identity to report.
_UNSPECIFIED_DEVICE_INDEX = 0xFFFFFFFF
_DEFAULT_RING_DEPTH = 6
_LATENCY_CSV_FLUSH_INTERVAL_S = 1.0
_OUTPUT_RGBA8 = "rgba8"
_OUTPUT_NV12 = "nv12"
_NV12_TO_RGBA_SRC = r"""
extern "C" __global__
void nv12_to_rgba(const unsigned char* nv12, unsigned char* out, int W, int H,
                  int rotation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const unsigned char* Yp = nv12;
    const unsigned char* UVp = nv12 + W * H;
    float Y = (float)Yp[y * W + x];
    int uv = (y >> 1) * W + (x & ~1);
    float U = (float)UVp[uv] - 128.0f;
    float V = (float)UVp[uv + 1] - 128.0f;
    float Yn = (Y - 16.0f) * 1.16438f;
    float R = Yn + 1.79274f * V;
    float G = Yn - 0.21325f * U - 0.53291f * V;
    float B = Yn + 2.11240f * U;
    int ox = rotation == 180 ? W - 1 - x : x;
    int oy = rotation == 180 ? H - 1 - y : y;
    int o = (oy * W + ox) * 4;
    out[o + 0] = (unsigned char)fminf(fmaxf(R, 0.0f), 255.0f);
    out[o + 1] = (unsigned char)fminf(fmaxf(G, 0.0f), 255.0f);
    out[o + 2] = (unsigned char)fminf(fmaxf(B, 0.0f), 255.0f);
    out[o + 3] = 255;
}
"""
_NV12_ROTATE_180_SRC = r"""
extern "C" __global__
void nv12_rotate_180(const unsigned char* src, unsigned char* dst, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        dst[(H - 1 - y) * W + W - 1 - x] = src[y * W + x];
    }
    if (x < W / 2 && y < H / 2) {
        int sx = x * 2;
        int dx = W - 2 - sx;
        int dy = H / 2 - 1 - y;
        const unsigned char* src_uv = src + W * H;
        unsigned char* dst_uv = dst + W * H;
        dst_uv[dy * W + dx] = src_uv[y * W + sx];
        dst_uv[dy * W + dx + 1] = src_uv[y * W + sx + 1];
    }
}
"""


def _read_wall_clock_ns() -> tuple[int, str]:
    # Use the chrony-disciplined Unix wall clock for cross-host latency.
    return int(time.time_ns()), "realtime"


def _ns_delta_ms(later_ns: Optional[int], earlier_ns: Optional[int]) -> Optional[float]:
    if later_ns is None or earlier_ns is None:
        return None
    return (int(later_ns) - int(earlier_ns)) / 1.0e6


def _parse_optional_gpu_id(value: object, source: str = "gpu_id") -> Optional[int]:
    if value is None:
        return None
    try:
        gpu_id = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{source} must be an integer or null, got {value!r}") from e
    if gpu_id < 0:
        raise ValueError(f"{source} must be >= 0 or null, got {gpu_id}")
    return gpu_id


def _parse_stats_log_interval_s(
    value: object,
    source: str = "receiver.stats_log_interval_s",
) -> float:
    try:
        interval_s = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{source} must be a number >= 0, got {value!r}") from e
    if not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError(f"{source} must be a finite number >= 0, got {value!r}")
    return interval_s


@dataclass(frozen=True)
class ReceiveFpsWarningConfig:
    threshold_fps: float
    interval_s: float
    # For discovered streams: warn below this fraction of each stream's own rate.
    fraction: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.threshold_fps > 0 or self.fraction > 0


def _resolve_receive_fps_warning_config(
    receiver_cfg: dict[str, Any],
) -> ReceiveFpsWarningConfig:
    threshold_fps = _parse_stats_log_interval_s(
        receiver_cfg.get("receive_fps_warning_threshold", 0.0),
        "receiver.receive_fps_warning_threshold",
    )
    interval_s = _parse_stats_log_interval_s(
        receiver_cfg.get(
            "receive_fps_warning_interval_s",
            _DEFAULT_RECEIVE_FPS_WARNING_INTERVAL_S,
        ),
        "receiver.receive_fps_warning_interval_s",
    )
    fraction = _parse_stats_log_interval_s(
        receiver_cfg.get("receive_fps_warning_fraction", 0.0),
        "receiver.receive_fps_warning_fraction",
    )
    if fraction > 1:
        raise ValueError("receiver.receive_fps_warning_fraction must be 0..1")
    if (threshold_fps > 0 or fraction > 0) and interval_s <= 0:
        raise ValueError(
            "receiver.receive_fps_warning_interval_s must be > 0 when "
            "a receive FPS warning is enabled"
        )
    return ReceiveFpsWarningConfig(
        threshold_fps=threshold_fps,
        interval_s=interval_s,
        fraction=fraction,
    )


def _resolve_stats_log_interval_s(
    receiver_cfg: dict[str, Any],
    cli_value: Optional[float],
) -> float:
    if cli_value is not None:
        return _parse_stats_log_interval_s(cli_value, "--stats-log-interval-s")
    if "stats_log_interval_s" in receiver_cfg:
        return _parse_stats_log_interval_s(receiver_cfg["stats_log_interval_s"])
    # Backward compatibility for configs created before the interval setting.
    if not bool(receiver_cfg.get("log_timing", True)):
        return 0.0
    return _DEFAULT_STATS_LOG_INTERVAL_S


class PeriodicStats:
    def __init__(self, label: str, interval_s: float = 5.0) -> None:
        self._label = label
        self._interval_s = float(interval_s)
        if self._interval_s <= 0:
            raise ValueError("PeriodicStats interval_s must be > 0")
        self._lock = threading.Lock()
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self._n = 0
        self._sum_ms = 0.0
        self._max_ms = 0.0
        self._sof_delta_n = 0
        self._sof_delta_sum_ns = 0
        self._sof_delta_min_ns: Optional[int] = None
        self._sof_delta_max_ns: Optional[int] = None
        self._latency_stats: dict[str, list[float]] = {}
        self._stream_counts: dict[str, int] = {}
        self._window_start = now

    def add(
        self,
        stream_name: str,
        dt_s: float,
        sof_delta_ns: Optional[int] = None,
        latencies_ms: Optional[dict[str, Optional[float]]] = None,
    ) -> None:
        dt_ms = dt_s * 1000.0
        log_line = None
        with self._lock:
            self._n += 1
            self._stream_counts[stream_name] = (
                self._stream_counts.get(stream_name, 0) + 1
            )
            self._sum_ms += dt_ms
            self._max_ms = max(self._max_ms, dt_ms)
            if sof_delta_ns is not None:
                delta_ns = int(sof_delta_ns)
                self._sof_delta_n += 1
                self._sof_delta_sum_ns += delta_ns
                if self._sof_delta_min_ns is None:
                    self._sof_delta_min_ns = delta_ns
                    self._sof_delta_max_ns = delta_ns
                else:
                    self._sof_delta_min_ns = min(self._sof_delta_min_ns, delta_ns)
                    assert self._sof_delta_max_ns is not None
                    self._sof_delta_max_ns = max(self._sof_delta_max_ns, delta_ns)
            if latencies_ms:
                for name, value in latencies_ms.items():
                    if value is None:
                        continue
                    bucket = self._latency_stats.setdefault(name, [0.0, 0.0, 0.0])
                    bucket[0] += 1.0
                    bucket[1] += float(value)
                    bucket[2] = max(bucket[2], float(value))
            now = time.monotonic()
            elapsed = now - self._window_start
            if elapsed >= self._interval_s and self._n > 0:
                mean = self._sum_ms / self._n
                fps = self._n / elapsed
                stream_parts = [
                    f"{name}={count / elapsed:.1f}fps"
                    for name, count in sorted(self._stream_counts.items())
                ]
                log_line = (
                    f"{self._label}: {fps:.1f} streams/s, {mean:.2f} ms mean, "
                    f"{self._max_ms:.2f} ms max (n={self._n} over {elapsed:.1f}s)"
                )
                if stream_parts:
                    log_line += ", " + " ".join(stream_parts)
                if self._sof_delta_n > 0:
                    assert self._sof_delta_min_ns is not None
                    assert self._sof_delta_max_ns is not None
                    mean_delta = self._sof_delta_sum_ns / self._sof_delta_n
                    log_line += (
                        f", sof_delta={mean_delta:+.0f} ns mean "
                        f"[{self._sof_delta_min_ns:+d},"
                        f"{self._sof_delta_max_ns:+d}]"
                    )
                else:
                    log_line += ", sof_delta=n/a"
                if self._latency_stats:
                    parts = []
                    for name in sorted(self._latency_stats):
                        count, total, max_value = self._latency_stats[name]
                        if count > 0:
                            parts.append(f"{name}={total / count:.2f}/{max_value:.2f}")
                    if parts:
                        log_line += ", latency_ms(avg/max) " + " ".join(parts)
                self._reset(now)
        if log_line is not None:
            logger.info(log_line)


class ReceiveFpsMonitor:
    """Warn when a configured logical AVS stream arrives below the target FPS."""

    def __init__(
        self,
        stream_names: Iterable[str],
        config: ReceiveFpsWarningConfig,
        clock: Callable[[], float] = time.monotonic,
        thresholds: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not config.enabled or config.interval_s <= 0:
            raise ValueError("ReceiveFpsMonitor requires an enabled config")
        names = tuple(sorted(set(stream_names)))
        if not names:
            raise ValueError("ReceiveFpsMonitor requires at least one stream")
        self._stream_names = names
        self._threshold_fps = config.threshold_fps
        # Per-stream targets, for streams whose rates differ.
        self._thresholds = dict(thresholds) if thresholds is not None else None
        self._interval_s = config.interval_s
        self._clock = clock
        self._active = False
        self._window_start = 0.0
        self._counts = dict.fromkeys(self._stream_names, 0)

    def start(self) -> None:
        self._active = True
        self._reset(self._clock())

    def stop(self) -> None:
        self._active = False

    def observe(self, stream_names: Iterable[str]) -> None:
        if not self._active:
            return
        for name in stream_names:
            if name in self._counts:
                self._counts[name] += 1
        self._maybe_warn(self._clock())

    def poll(self) -> None:
        if self._active:
            self._maybe_warn(self._clock())

    def _reset(self, now: float) -> None:
        self._window_start = now
        self._counts = dict.fromkeys(self._stream_names, 0)

    def _maybe_warn(self, now: float) -> None:
        elapsed = now - self._window_start
        if elapsed < self._interval_s:
            return
        rates = {name: self._counts[name] / elapsed for name in self._stream_names}
        if self._thresholds is not None:
            details = " ".join(
                f"{name}={fps:.1f}fps(target {self._thresholds[name]:.1f})"
                for name, fps in rates.items()
                if fps < self._thresholds.get(name, 0.0)
            )
            if details:
                logger.warning(
                    "AVS1 receive FPS below target over %.1fs: %s", elapsed, details
                )
            self._reset(now)
            return
        low_rates = {
            name: fps for name, fps in rates.items() if fps < self._threshold_fps
        }
        if low_rates:
            details = " ".join(
                f"{name}={fps:.1f}fps" for name, fps in low_rates.items()
            )
            logger.warning(
                "AVS1 receive FPS below %.1ffps target over %.1fs: %s",
                self._threshold_fps,
                elapsed,
                details,
            )
        self._reset(now)


@dataclass(frozen=True)
class ReceivedRecord:
    payload: bytes
    receiver_rx_prefix_done_ns: int
    receiver_rx_payload_done_ns: int
    receiver_rx_prefix_done_mono_ns: int
    receiver_rx_payload_done_mono_ns: int
    receiver_clock_domain: str


@dataclass(frozen=True)
class StreamConfig:
    index: int
    name: str
    stereo: bool
    width: int
    height: int
    output_format: str
    ring_depth: int
    ipc_names: tuple[str, ...]
    full_range: bool = False


def discovered_stream_configs(
    streams: Iterable[Avs1Stream],
    output_format: str = _OUTPUT_NV12,
    ring_depth: int = _DEFAULT_RING_DEPTH,
) -> list[StreamConfig]:
    """Stream configs for a sender layout, taken from one bundle's descriptors.

    A sender that reports each camera's config position in the AVS1 device index
    gets names that survive a disabled camera (cam0, cam2); otherwise names
    follow the wire index.
    """
    streams = list(streams)
    positions = [stream.samples[0].device_index for stream in streams]
    by_position = _UNSPECIFIED_DEVICE_INDEX not in positions and len(
        set(positions)
    ) == len(positions)
    configs = []
    for stream, position in zip(streams, positions):
        name = f"cam{position}" if by_position else f"stream{stream.index}"
        configs.append(
            StreamConfig(
                index=stream.index,
                name=name,
                stereo=stream.stereo,
                width=stream.width,
                height=stream.height,
                output_format=output_format,
                ring_depth=ring_depth,
                ipc_names=(f"{name}.left", f"{name}.right")
                if stream.stereo
                else (name,),
            )
        )
    return configs


class LatencyCsv:
    """One row per received stream frame: sender send time, receiver receive time.

    The two times come from different machines' clocks; their difference is a
    latency only while those clocks are synchronized.
    """

    FIELDS = (
        "stream",
        "session_id",
        "bundle_sequence",
        "stream_sequence",
        "frame_number",
        "sender_send_ns",
        "receiver_receive_ns",
        "receive_minus_send_ms",
        "sender_clock",
        "receiver_clock",
    )

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)
        self._last_flush = time.monotonic()

    def write(
        self,
        name: str,
        bundle: Avs1Bundle,
        stream: Avs1Stream,
        received: ReceivedRecord,
    ) -> None:
        send_ns = bundle.sender_tcp_enqueue_ns
        receive_ns = received.receiver_rx_payload_done_ns
        delta_ms = _ns_delta_ms(receive_ns, send_ns)
        self._writer.writerow(
            (
                name,
                bundle.session_id,
                bundle.sequence,
                stream.sequence,
                stream.samples[0].frame_number,
                "" if send_ns is None else send_ns,
                receive_ns,
                "" if delta_ms is None else f"{delta_ms:.3f}",
                bundle.sender_clock_domain or "",
                received.receiver_clock_domain,
            )
        )
        now = time.monotonic()
        if now - self._last_flush >= _LATENCY_CSV_FLUSH_INTERVAL_S:
            self._file.flush()
            self._last_flush = now

    def close(self) -> None:
        self._file.close()


class _StreamRuntime:
    def __init__(self, config: StreamConfig, gpu_id: int) -> None:
        import cupy as cp

        self.config = config
        self.gpu_id = int(gpu_id)
        self.camera_count = 2 if config.stereo else 1
        self.write_index = 0
        self.slot_generations = [
            [0 for _ in range(config.ring_depth)] for _ in range(self.camera_count)
        ]
        self.decoders: list[Optional[NvVideoDecoder]] = [None] * self.camera_count
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"camera_receiver_stream_{config.name}",
        )
        self.block = (16, 16, 1)
        self.grid = (
            (config.width + 15) // 16,
            (config.height + 15) // 16,
            1,
        )
        with cp.cuda.Device(gpu_id):
            if config.output_format == _OUTPUT_RGBA8:
                self.output_buffers = [
                    [
                        cp.empty((config.height, config.width, 4), dtype=cp.uint8)
                        for _ in range(config.ring_depth)
                    ]
                    for _ in range(self.camera_count)
                ]
                for camera_buffers in self.output_buffers:
                    for buf in camera_buffers:
                        buf[..., 3] = 255
                self.raw_buffers = [
                    cp.empty(config.width * config.height * 3 // 2, dtype=cp.uint8)
                    for _ in range(self.camera_count)
                ]
                self.raw_kernel = cp.RawKernel(_NV12_TO_RGBA_SRC, "nv12_to_rgba")
            elif config.output_format == _OUTPUT_NV12:
                self.output_buffers = [
                    [
                        cp.empty((config.height * 3 // 2, config.width), dtype=cp.uint8)
                        for _ in range(config.ring_depth)
                    ]
                    for _ in range(self.camera_count)
                ]
                self.raw_buffers = [
                    cp.empty(config.width * config.height * 3 // 2, dtype=cp.uint8)
                    for _ in range(self.camera_count)
                ]
                self.raw_kernel = cp.RawKernel(_NV12_ROTATE_180_SRC, "nv12_rotate_180")
            else:
                raise ValueError(f"unsupported output format {config.output_format!r}")

    def reset_decoders(self) -> None:
        for index, decoder in enumerate(self.decoders):
            if decoder is not None:
                try:
                    decoder.reset()
                except Exception:
                    logger.debug(
                        "decoder reset failed for %s[%d]",
                        self.config.name,
                        index,
                        exc_info=True,
                    )
                self.decoders[index] = None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
        self.reset_decoders()

    def next_generation(self, camera_index: int, slot: int) -> int:
        self.slot_generations[camera_index][slot] += 1
        return self.slot_generations[camera_index][slot]


class Avs1TcpReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        gpu_id: Optional[int],
        max_record_bytes: int,
        stats_log_interval_s: float,
        receive_fps_warning: ReceiveFpsWarningConfig,
        ipc_publisher: CudaIpcFramePublisher,
        streams: list[StreamConfig],
        discover: bool = False,
        latency_csv: Optional[LatencyCsv] = None,
    ) -> None:
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "camera_receiver requires CuPy. Run ./setup.sh from camera_receiver."
            ) from e
        if discover and streams:
            raise ValueError("stream discovery takes no configured streams")
        if not discover and not streams:
            raise ValueError("camera_receiver requires at least one AVS stream")
        indices = [stream.index for stream in streams]
        if len(indices) != len(set(indices)):
            raise ValueError("AVS stream indexes must be unique")
        if max_record_bytes <= AVS1_HEADER.size:
            raise ValueError(
                f"max_record_bytes must exceed AVS1 header size, got {max_record_bytes}"
            )

        parsed_gpu_id = _parse_optional_gpu_id(gpu_id)
        self._gpu_device_id = (
            int(cp.cuda.runtime.getDevice()) if parsed_gpu_id is None else parsed_gpu_id
        )
        self._host = str(host)
        self._port = int(port)
        self._max_record_bytes = int(max_record_bytes)
        self._ipc_publisher = ipc_publisher
        self._runtimes = {
            stream.index: _StreamRuntime(stream, self._gpu_device_id)
            for stream in streams
        }
        self._sample_pool: Optional[ThreadPoolExecutor] = None
        self._header_buf = bytearray(4)
        self._record_buf = bytearray()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._conn: Optional[socket.socket] = None
        self._stats = None
        if stats_log_interval_s > 0:
            self._stats = PeriodicStats(
                "decode[avs1]",
                interval_s=stats_log_interval_s,
            )
        self._receive_fps_warning = receive_fps_warning
        self._receive_fps_monitor = None
        if receive_fps_warning.threshold_fps > 0 and self._runtimes:
            self._receive_fps_monitor = ReceiveFpsMonitor(
                (runtime.config.name for runtime in self._runtimes.values()),
                receive_fps_warning,
            )
        self._session_id: Optional[int] = None
        self._discover = bool(discover)
        self._layout_key: Optional[tuple] = None
        # Rings of streams that left the layout. Consumers may still hold their
        # exported handles, so the memory is never freed while this process runs.
        self._retired: list[_StreamRuntime] = []
        self._latency_csv = latency_csv

        for runtime in self._runtimes.values():
            self._register_runtime(runtime)

    def _register_runtime(self, runtime: _StreamRuntime) -> None:
        cfg = runtime.config
        layout = _ipc_layout(cfg.output_format, cfg.width, cfg.height)
        for camera_index, ipc_name in enumerate(cfg.ipc_names):
            self._ipc_publisher.register_stream(
                name=ipc_name,
                width=cfg.width,
                height=cfg.height,
                pixel_format=cfg.output_format,
                gpu_id=self._gpu_device_id,
                buffers=runtime.output_buffers[camera_index],
                layout=layout,
            )

    def _resize_sample_pool(self) -> None:
        workers = sum(runtime.camera_count for runtime in self._runtimes.values())
        previous = self._sample_pool
        self._sample_pool = ThreadPoolExecutor(
            max_workers=max(1, workers),
            thread_name_prefix="camera_receiver_sample",
        )
        if previous is not None:
            previous.shutdown(wait=True)

    def _apply_discovered_layout(self, bundle: Avs1Bundle) -> None:
        """Build or rebuild the stream runtimes to match the sender's layout."""
        configs = discovered_stream_configs(bundle.streams)
        key = tuple((c.index, c.name, c.stereo, c.width, c.height) for c in configs)
        if key == self._layout_key:
            return
        # Only this thread submits stream work, so once each queue drains nothing
        # is using the sample pool or the runtimes being replaced.
        for runtime in self._runtimes.values():
            runtime.executor.submit(lambda: None).result()

        def identity(cfg: StreamConfig) -> tuple:
            return (cfg.name, cfg.stereo, cfg.width, cfg.height)

        reusable = {identity(rt.config): rt for rt in self._runtimes.values()}
        runtimes: dict[int, _StreamRuntime] = {}
        created: list[_StreamRuntime] = []
        for cfg in configs:
            runtime = reusable.pop(identity(cfg), None)
            if runtime is None:
                runtime = _StreamRuntime(cfg, self._gpu_device_id)
                created.append(runtime)
            else:
                # Same stream, possibly at a new wire index.
                runtime.config = cfg
            runtimes[cfg.index] = runtime
        # Unregister before registering: a stream that changed size keeps its name.
        for runtime in reusable.values():
            runtime.shutdown()
            for ipc_name in runtime.config.ipc_names:
                self._ipc_publisher.unregister_stream(ipc_name)
            self._retired.append(runtime)
        for runtime in created:
            self._register_runtime(runtime)
        self._runtimes = runtimes
        self._layout_key = key
        self._resize_sample_pool()

        if self._receive_fps_monitor is not None:
            self._receive_fps_monitor.stop()
            self._receive_fps_monitor = None
        warning = self._receive_fps_warning
        if warning.enabled:
            rates = {
                cfg.name: stream.fps_numerator / stream.fps_denominator
                for cfg, stream in zip(configs, bundle.streams)
            }
            self._receive_fps_monitor = ReceiveFpsMonitor(
                rates,
                warning,
                thresholds={
                    name: max(warning.threshold_fps, warning.fraction * fps)
                    for name, fps in rates.items()
                },
            )
            self._receive_fps_monitor.start()
        self._ipc_publisher.streams_changed(
            order=[name for cfg in configs for name in cfg.ipc_names]
        )
        logger.info(
            "AVS1 layout from sender: %s",
            ", ".join(
                f"{cfg.name} {'stereo' if cfg.stereo else 'mono'} "
                f"{cfg.width}x{cfg.height}"
                for cfg in configs
            ),
        )

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def socket_path(self) -> str:
        return self._ipc_publisher.socket_path

    def start(self) -> None:
        if self._thread is not None:
            return
        self._resize_sample_pool()
        self._ipc_publisher.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._produce_loop,
            name="camera_receiver_avs1",
            daemon=False,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.debug(
                    "failed to close AVS1 connection during shutdown", exc_info=True
                )
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("receiver thread did not exit within 5s")
            else:
                self._thread = None
        for runtime in self._runtimes.values():
            runtime.shutdown()
        if self._sample_pool is not None:
            self._sample_pool.shutdown(wait=False)
            self._sample_pool = None
        self._ipc_publisher.stop()
        if self._latency_csv is not None:
            self._latency_csv.close()

    def wait(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            time.sleep(0.2)

    def _produce_loop(self) -> None:
        import cupy as cp

        with cp.cuda.Device(self._gpu_device_id):
            while not self._stop.is_set():
                conn = self._connect()
                if conn is None:
                    break
                self._conn = conn
                # Not yet warm: the sender may still be in cold-start. Armed to
                # the tight stall timeout only after the first bundle arrives.
                self._link_warm = False
                if self._receive_fps_monitor is not None:
                    self._receive_fps_monitor.start()
                try:
                    while not self._stop.is_set():
                        received = self._read_record(conn)
                        if received is None:
                            break
                        self._link_warm = True
                        bundle = parse_avs1_record(received.payload)
                        self._dispatch_bundle(received, bundle)
                except Exception as e:
                    if not self._stop.is_set():
                        logger.warning("receive/parse error (%s); reconnecting", e)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        logger.debug(
                            "failed to close AVS1 connection after receive loop",
                            exc_info=True,
                        )
                    self._conn = None
                    if self._receive_fps_monitor is not None:
                        self._receive_fps_monitor.stop()
                    self._session_id = None
                    for runtime in self._runtimes.values():
                        runtime.executor.submit(runtime.reset_decoders)

    def _connect(self) -> Optional[socket.socket]:
        next_wait_log_at = 0.0
        while not self._stop.is_set():
            try:
                conn = socket.create_connection(
                    (self._host, self._port), timeout=_SOCK_TIMEOUT_S
                )
                conn.settimeout(_SOCK_TIMEOUT_S)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Enable TCP keepalive with a short probe schedule so the kernel
                # also detects a silently-dead peer (the AGX rebooted mid-stream,
                # leaving this socket ESTABLISHED but never sending a FIN/RST).
                # Belt-and-suspenders with the app-level _STALL_TIMEOUT_S below.
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                for _opt, _val in (
                    ("TCP_KEEPIDLE", 2),
                    ("TCP_KEEPINTVL", 1),
                    ("TCP_KEEPCNT", 3),
                ):
                    if hasattr(socket, _opt):
                        conn.setsockopt(socket.IPPROTO_TCP, getattr(socket, _opt), _val)
                logger.info("connected to AVS1 sender %s:%d", self._host, self._port)
                return conn
            except OSError as e:
                now = time.monotonic()
                if now >= next_wait_log_at:
                    logger.info(
                        "waiting for AVS1 sender %s:%d (%s). Please check that "
                        "the robot is powered on and wait at least 2 minutes after "
                        "powering it on.",
                        self._host,
                        self._port,
                        e,
                    )
                    next_wait_log_at = now + _CONNECT_WAIT_LOG_INTERVAL_S
                deadline = now + 1.0
                while time.monotonic() < deadline and not self._stop.is_set():
                    time.sleep(0.05)
        return None

    def _read_record(self, conn: socket.socket) -> Optional[ReceivedRecord]:
        header = memoryview(self._header_buf)
        if not self._recv_exact(conn, header):
            return None
        receiver_rx_prefix_done_mono_ns = time.monotonic_ns()
        receiver_rx_prefix_done_ns, receiver_clock_domain = _read_wall_clock_ns()
        (length,) = struct.unpack("!I", header)
        if length == 0:
            raise ValueError("zero-length AVS1 record")
        if length > self._max_record_bytes:
            raise ValueError(
                f"AVS1 record too large: {length} bytes > {self._max_record_bytes}"
            )
        if len(self._record_buf) < length:
            self._record_buf = bytearray(length)
        view = memoryview(self._record_buf)[:length]
        if not self._recv_exact(conn, view):
            return None
        receiver_rx_payload_done_mono_ns = time.monotonic_ns()
        receiver_rx_payload_done_ns, receiver_payload_clock_domain = (
            _read_wall_clock_ns()
        )
        if receiver_payload_clock_domain != receiver_clock_domain:
            receiver_clock_domain = receiver_payload_clock_domain
        return ReceivedRecord(
            payload=bytes(view),
            receiver_rx_prefix_done_ns=receiver_rx_prefix_done_ns,
            receiver_rx_payload_done_ns=receiver_rx_payload_done_ns,
            receiver_rx_prefix_done_mono_ns=receiver_rx_prefix_done_mono_ns,
            receiver_rx_payload_done_mono_ns=receiver_rx_payload_done_mono_ns,
            receiver_clock_domain=receiver_clock_domain,
        )

    def _dispatch_bundle(self, received: ReceivedRecord, bundle: Avs1Bundle) -> None:
        if self._session_id != bundle.session_id:
            if self._session_id is not None:
                logger.info(
                    "AVS1 session changed: %s -> %s; resetting decoders",
                    self._session_id,
                    bundle.session_id,
                )
                for runtime in self._runtimes.values():
                    runtime.executor.submit(runtime.reset_decoders)
            self._session_id = bundle.session_id

        if self._discover:
            self._apply_discovered_layout(bundle)
        dispatch_items: list[tuple[Avs1Stream, _StreamRuntime]] = []
        for stream in bundle.streams:
            runtime = self._runtimes.get(stream.index)
            if runtime is None or not stream.present:
                continue
            dispatch_items.append((stream, runtime))
            if self._latency_csv is not None:
                self._latency_csv.write(runtime.config.name, bundle, stream, received)
        if self._receive_fps_monitor is not None:
            self._receive_fps_monitor.observe(
                runtime.config.name for _, runtime in dispatch_items
            )
        for stream, runtime in dispatch_items:
            runtime.executor.submit(
                self._process_stream_safe,
                received,
                bundle,
                stream,
                runtime,
            )

    def _process_stream_safe(
        self,
        received: ReceivedRecord,
        bundle: Avs1Bundle,
        stream: Avs1Stream,
        runtime: _StreamRuntime,
    ) -> None:
        try:
            self._process_stream(received, bundle, stream, runtime)
        except Exception as exc:
            logger.warning(
                "AVS1 stream %s processing failed: %s",
                runtime.config.name,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )

    def _process_stream(
        self,
        received: ReceivedRecord,
        bundle: Avs1Bundle,
        stream: Avs1Stream,
        runtime: _StreamRuntime,
    ) -> None:
        import cupy as cp

        self._validate_stream_descriptor(stream, runtime.config)
        slot = runtime.write_index
        t0 = time.monotonic()
        receiver_process_submit_mono_ns = time.monotonic_ns()
        receiver_process_submit_ns, _ = _read_wall_clock_ns()
        assert self._sample_pool is not None
        futures: list[Future[bool]] = [
            self._sample_pool.submit(
                self._process_sample, runtime, stream, sample, slot
            )
            for sample in stream.samples
        ]
        # Finish every eye before a failure can allow slot reuse or decoder reset.
        wait(futures)
        if not all(future.result() for future in futures):
            return
        receiver_process_done_mono_ns = time.monotonic_ns()
        receiver_process_done_ns, _ = _read_wall_clock_ns()

        # The native decoder is synchronous. Raw NV12 copies / conversion kernels
        # run on CuPy's current stream, so sync before announcing IPC visibility.
        with cp.cuda.Device(self._gpu_device_id):
            cp.cuda.get_current_stream().synchronize()

        timestamp_ns = time.monotonic_ns()
        receiver_ipc_publish_mono_ns = timestamp_ns
        receiver_ipc_publish_ns, _ = _read_wall_clock_ns()
        latencies = self._receiver_latencies(
            bundle=bundle,
            stream=stream,
            received=received,
            receiver_process_submit_ns=receiver_process_submit_ns,
            receiver_process_done_ns=receiver_process_done_ns,
            receiver_ipc_publish_ns=receiver_ipc_publish_ns,
            receiver_process_submit_mono_ns=receiver_process_submit_mono_ns,
            receiver_process_done_mono_ns=receiver_process_done_mono_ns,
            receiver_ipc_publish_mono_ns=receiver_ipc_publish_mono_ns,
        )

        for sample in stream.samples:
            generation = runtime.next_generation(sample.camera_index, slot)
            metadata = self._frame_metadata(
                bundle=bundle,
                stream=stream,
                sample=sample,
                runtime=runtime,
                received=received,
                receiver_process_submit_ns=receiver_process_submit_ns,
                receiver_process_done_ns=receiver_process_done_ns,
                receiver_ipc_publish_ns=receiver_ipc_publish_ns,
                receiver_process_submit_mono_ns=receiver_process_submit_mono_ns,
                receiver_process_done_mono_ns=receiver_process_done_mono_ns,
                receiver_ipc_publish_mono_ns=receiver_ipc_publish_mono_ns,
                latencies=latencies,
            )
            self._ipc_publisher.publish(
                runtime.config.ipc_names[sample.camera_index],
                slot=slot,
                slot_generation=generation,
                seq=stream.sequence,
                timestamp_ns=timestamp_ns,
                metadata=metadata,
            )

        runtime.write_index = (slot + 1) % runtime.config.ring_depth
        if self._stats is not None:
            self._stats.add(
                runtime.config.name,
                time.monotonic() - t0,
                stream.tsc_delta_ns,
                latencies,
            )

    def _process_sample(
        self,
        runtime: _StreamRuntime,
        stream: Avs1Stream,
        sample: Avs1Sample,
        slot: int,
    ) -> bool:
        import cupy as cp
        import numpy as np

        cfg = runtime.config
        output = runtime.output_buffers[sample.camera_index][slot]
        with cp.cuda.Device(self._gpu_device_id):
            if stream.codec == CODEC_HEVC:
                decoder = runtime.decoders[sample.camera_index]
                if decoder is None:
                    decoder = NvVideoDecoder(
                        width=stream.width,
                        height=stream.height,
                        full_range=cfg.full_range,
                        gpu_id=self._gpu_device_id,
                        codec="hevc",
                        low_latency=True,
                    )
                    runtime.decoders[sample.camera_index] = decoder
                if cfg.output_format == _OUTPUT_NV12:
                    return bool(
                        decoder.decode_nv12(
                            sample.payload, output, rotation=stream.rotation
                        )
                    )
                return bool(
                    decoder.decode(sample.payload, output, rotation=stream.rotation)
                )

            if stream.codec != CODEC_RAW_NV12:
                raise ValueError(f"unsupported AVS1 codec {stream.codec}")

            raw_host = np.frombuffer(sample.payload, dtype=np.uint8)
            if cfg.output_format == _OUTPUT_NV12:
                if stream.rotation == 0:
                    output.set(raw_host.reshape(output.shape))
                    return True
                raw = runtime.raw_buffers[sample.camera_index]
                raw.set(raw_host)
                assert runtime.raw_kernel is not None
                runtime.raw_kernel(
                    runtime.grid,
                    runtime.block,
                    (raw, output, stream.width, stream.height),
                )
                return True

            raw = runtime.raw_buffers[sample.camera_index]
            raw.set(raw_host)
            assert runtime.raw_kernel is not None
            runtime.raw_kernel(
                runtime.grid,
                runtime.block,
                (raw, output, stream.width, stream.height, stream.rotation),
            )
            return True

    def _validate_stream_descriptor(
        self, stream: Avs1Stream, cfg: StreamConfig
    ) -> None:
        if stream.width != cfg.width or stream.height != cfg.height:
            raise ValueError(
                f"AVS1 stream {stream.index} size {stream.width}x{stream.height} "
                f"does not match config {cfg.width}x{cfg.height}"
            )
        if stream.stereo != cfg.stereo:
            raise ValueError(
                f"AVS1 stream {stream.index} stereo={stream.stereo} "
                f"does not match config stereo={cfg.stereo}"
            )
        if len(stream.samples) != (2 if cfg.stereo else 1):
            raise ValueError(f"AVS1 stream {stream.index} sample count mismatch")

    def _receiver_latencies(
        self,
        bundle: Avs1Bundle,
        stream: Avs1Stream,
        received: ReceivedRecord,
        receiver_process_submit_ns: int,
        receiver_process_done_ns: int,
        receiver_ipc_publish_ns: int,
        receiver_process_submit_mono_ns: int,
        receiver_process_done_mono_ns: int,
        receiver_ipc_publish_mono_ns: int,
    ) -> dict[str, Optional[float]]:
        local_rx_to_process_submit_ms = _ns_delta_ms(
            receiver_process_submit_mono_ns, received.receiver_rx_payload_done_mono_ns
        )
        local_rx_to_process_done_ms = _ns_delta_ms(
            receiver_process_done_mono_ns, received.receiver_rx_payload_done_mono_ns
        )
        local_rx_to_ipc_publish_ms = _ns_delta_ms(
            receiver_ipc_publish_mono_ns, received.receiver_rx_payload_done_mono_ns
        )
        latencies: dict[str, Optional[float]] = {
            "rx_to_process_submit": local_rx_to_process_submit_ms,
            "rx_to_process_done": local_rx_to_process_done_ms,
            "rx_to_ipc_publish": local_rx_to_ipc_publish_ms,
        }
        if bundle.sender_clock_domain == received.receiver_clock_domain:
            process_done_ns = max(
                (
                    int(sample.sender_process_done_ns)
                    for sample in stream.samples
                    if sample.sender_process_done_ns is not None
                ),
                default=None,
            )
            capture_ready_ns = _stream_capture_ready_ns(bundle, stream)
            sof_ns = _stream_sensor_sof_ns(bundle, stream)
            process_submit_ns = min(
                (
                    int(sample.sender_process_submit_ns)
                    for sample in stream.samples
                    if sample.sender_process_submit_ns is not None
                ),
                default=None,
            )
            cross_host_values = {
                "sender_capture_ready_to_rx": _ns_delta_ms(
                    received.receiver_rx_payload_done_ns, capture_ready_ns
                ),
                "sender_process_submit_to_rx": _ns_delta_ms(
                    received.receiver_rx_payload_done_ns, process_submit_ns
                ),
                "sender_process_done_to_rx": _ns_delta_ms(
                    received.receiver_rx_payload_done_ns, process_done_ns
                ),
                "sender_tcp_enqueue_to_rx": _ns_delta_ms(
                    received.receiver_rx_payload_done_ns, bundle.sender_tcp_enqueue_ns
                ),
                "sender_tcp_enqueue_to_ipc_publish": _ns_delta_ms(
                    receiver_ipc_publish_ns, bundle.sender_tcp_enqueue_ns
                ),
                "sender_capture_ready_to_ipc_publish": _ns_delta_ms(
                    receiver_ipc_publish_ns, capture_ready_ns
                ),
                "sensor_sof_to_rx": _ns_delta_ms(
                    received.receiver_rx_payload_done_ns, sof_ns
                ),
                "sensor_sof_to_ipc_publish": _ns_delta_ms(
                    receiver_ipc_publish_ns, sof_ns
                ),
            }
            latencies.update(cross_host_values)
        return latencies

    def _frame_metadata(
        self,
        bundle: Avs1Bundle,
        stream: Avs1Stream,
        sample: Avs1Sample,
        runtime: _StreamRuntime,
        received: ReceivedRecord,
        receiver_process_submit_ns: int,
        receiver_process_done_ns: int,
        receiver_ipc_publish_ns: int,
        receiver_process_submit_mono_ns: int,
        receiver_process_done_mono_ns: int,
        receiver_ipc_publish_mono_ns: int,
        latencies: dict[str, Optional[float]],
    ) -> dict[str, object]:
        sender_capture_ready_ns = sample.sender_capture_ready_ns
        if sender_capture_ready_ns is None:
            sender_capture_ready_ns = bundle.sender_capture_ready_ns
        sof_lag_ns = sensor_sof_lag_ns(bundle, sample)
        sample_sof_ns = sensor_sof_ns(bundle, sample)
        metadata: dict[str, object] = {
            "source_format": "avs1",
            "avs1_version": 1,
            "avs1_flags": bundle.flags,
            "session_id": bundle.session_id,
            "bundle_sequence": bundle.sequence,
            "sender_sequence": stream.sequence,
            "stream_index": stream.index,
            "stream_name": runtime.config.name,
            "stream_sequence": stream.sequence,
            "stream_stereo": stream.stereo,
            "stream_codec": stream.codec_name,
            "stream_rotation": stream.rotation,
            "stream_fps_numerator": stream.fps_numerator,
            "stream_fps_denominator": stream.fps_denominator,
            "camera_index": sample.camera_index,
            "role": sample.role_name,
            "device_index": sample.device_index,
            "sensor_mode_index": sample.sensor_mode_index,
            "frame_number": sample.frame_number,
            "width": stream.width,
            "height": stream.height,
            "source_pixel_format": "nv12",
            "ipc_pixel_format": runtime.config.output_format,
            "sender_clock": "tegra_tsc_ns" if sample.sof_tsc_ns is not None else None,
            "sender_wall_clock": bundle.sender_clock_domain,
            "sof_tsc_ns": sample.sof_tsc_ns,
            "sof_tsc_valid": sample.sof_tsc_ns is not None,
            "frame_timestamp_ns": sample.frame_timestamp_ns,
            "sensor_timestamp_ns": sample.frame_timestamp_ns,
            "sender_capture_ready_ns": sender_capture_ready_ns,
            "sender_capture_ready_tsc_ns": bundle.sender_capture_ready_tsc_ns,
            # Sensor start-of-frame in the sender wall domain (``sender_wall_clock``);
            # None when the sender TSC anchor is absent or the SOF->ready lag is
            # out of range.
            "sensor_sof_ns": sample_sof_ns,
            # Same value under the key the CUDA IPC consumer tools and other viewers
            # already read for their "sof_to_consumer_ms" column.
            "sof_wall_clock_ns": sample_sof_ns,
            "sensor_sof_lag_ms": (
                sof_lag_ns / 1.0e6 if sof_lag_ns is not None else None
            ),
            "sender_process_submit_ns": sample.sender_process_submit_ns,
            "sender_process_done_ns": sample.sender_process_done_ns,
            "sender_tcp_enqueue_ns": bundle.sender_tcp_enqueue_ns,
            "receiver_clock": received.receiver_clock_domain,
            "receiver_rx_prefix_done_ns": received.receiver_rx_prefix_done_ns,
            "receiver_rx_payload_done_ns": received.receiver_rx_payload_done_ns,
            "receiver_rx_prefix_done_mono_ns": received.receiver_rx_prefix_done_mono_ns,
            "receiver_rx_payload_done_mono_ns": received.receiver_rx_payload_done_mono_ns,
            "receiver_process_submit_ns": receiver_process_submit_ns,
            "receiver_process_done_ns": receiver_process_done_ns,
            "receiver_ipc_publish_ns": receiver_ipc_publish_ns,
            "receiver_process_submit_mono_ns": receiver_process_submit_mono_ns,
            "receiver_process_done_mono_ns": receiver_process_done_mono_ns,
            "receiver_ipc_publish_mono_ns": receiver_ipc_publish_mono_ns,
        }
        if stream.stereo and len(stream.samples) == 2:
            metadata["left_sof_tsc_ns"] = stream.samples[0].sof_tsc_ns
            metadata["right_sof_tsc_ns"] = stream.samples[1].sof_tsc_ns
            metadata["tsc_delta_ns"] = stream.tsc_delta_ns
            metadata["left_frame_timestamp_ns"] = stream.samples[0].frame_timestamp_ns
            metadata["right_frame_timestamp_ns"] = stream.samples[1].frame_timestamp_ns
            metadata["left_sensor_sof_ns"] = sensor_sof_ns(bundle, stream.samples[0])
            metadata["right_sensor_sof_ns"] = sensor_sof_ns(bundle, stream.samples[1])
        for name, value in latencies.items():
            metadata[f"{name}_ms"] = value
        return metadata

    def _recv_exact(self, conn: socket.socket, view: memoryview) -> bool:
        got = 0
        total = len(view)
        last_progress = time.monotonic()
        # Tight stall timeout only once the stream is flowing; a fresh, not-yet-
        # warm link gets the cold-start grace so the sender's NVENC warmup does
        # not look like a dead peer.
        stall_timeout = (
            _STALL_TIMEOUT_S
            if getattr(self, "_link_warm", False)
            else _COLD_START_TIMEOUT_S
        )
        while got < total:
            if self._stop.is_set():
                return False
            try:
                n = conn.recv_into(view[got:], total - got)
            except socket.timeout:
                if self._receive_fps_monitor is not None:
                    self._receive_fps_monitor.poll()
                # Bound the stall: a half-open peer (the AGX rebooted) leaves the
                # socket ESTABLISHED and recv just keeps timing out with no data.
                # Give up so the produce loop reconnects instead of waiting on a
                # dead socket forever; the reconnect loop then re-establishes the
                # link as soon as the sender is listening again, in any order.
                if time.monotonic() - last_progress > stall_timeout:
                    logger.warning(
                        "no data from %s:%d for %.1fs; treating the link as dead "
                        "and reconnecting",
                        self._host,
                        self._port,
                        stall_timeout,
                    )
                    return False
                continue
            except OSError:
                return False
            if n == 0:
                return False
            got += n
            last_progress = time.monotonic()
        return True


def _stream_capture_ready_ns(bundle: Avs1Bundle, stream: Avs1Stream) -> Optional[int]:
    sample_values = [
        int(sample.sender_capture_ready_ns)
        for sample in stream.samples
        if sample.sender_capture_ready_ns is not None
    ]
    if sample_values:
        return max(sample_values)
    return bundle.sender_capture_ready_ns


def _stream_sensor_sof_ns(bundle: Avs1Bundle, stream: Avs1Stream) -> Optional[int]:
    # Earliest exposure of the stream's samples (stereo pairs differ by ~us).
    sample_values = [
        int(sof)
        for sof in (sensor_sof_ns(bundle, sample) for sample in stream.samples)
        if sof is not None
    ]
    if sample_values:
        return min(sample_values)
    return None


def _ipc_layout(output_format: str, width: int, height: int) -> dict[str, Any]:
    if output_format == _OUTPUT_RGBA8:
        return {
            "planes": [
                {
                    "name": "rgba",
                    "offset": 0,
                    "width": width,
                    "height": height,
                    "pitch": width * 4,
                    "channels": 4,
                }
            ]
        }
    if output_format == _OUTPUT_NV12:
        y_bytes = width * height
        return {
            "planes": [
                {
                    "name": "y",
                    "offset": 0,
                    "width": width,
                    "height": height,
                    "pitch": width,
                },
                {
                    "name": "uv",
                    "offset": y_bytes,
                    "width": width,
                    "height": height // 2,
                    "pitch": width,
                },
            ]
        }
    raise ValueError(f"unsupported output_format {output_format!r}")


def _parse_stream_config(raw: dict[str, Any]) -> StreamConfig:
    index = int(raw["index"])
    name = str(raw.get("name", f"stream{index}"))
    stream_type = str(
        raw.get("type", "stereo" if raw.get("stereo", False) else "mono")
    ).lower()
    stereo = bool(raw.get("stereo", stream_type == "stereo"))
    width = int(raw["width"])
    height = int(raw["height"])
    output_format = str(raw.get("output_format", _OUTPUT_NV12)).lower()
    if output_format not in (_OUTPUT_NV12, _OUTPUT_RGBA8):
        raise ValueError(
            f"streams[{index}].output_format must be nv12|rgba8, got {output_format!r}"
        )
    ring_depth = int(raw.get("ring_depth", 6))
    if ring_depth < 3:
        raise ValueError(f"streams[{index}].ring_depth must be >= 3")
    ipc_cfg = raw.get("ipc", {})
    if ipc_cfg is None:
        ipc_cfg = {}
    if not isinstance(ipc_cfg, dict):
        raise ValueError(f"streams[{index}].ipc must be a mapping")
    if stereo:
        left = str(ipc_cfg.get("left", f"{name}.left"))
        right = str(ipc_cfg.get("right", f"{name}.right"))
        ipc_names = (left, right)
    else:
        mono = str(ipc_cfg.get("mono", name))
        ipc_names = (mono,)
    return StreamConfig(
        index=index,
        name=name,
        stereo=stereo,
        width=width,
        height=height,
        output_format=output_format,
        ring_depth=ring_depth,
        ipc_names=ipc_names,
        full_range=bool(raw.get("full_range", False)),
    )


def _configured_streams(cfg: dict) -> list[StreamConfig]:
    """The config's streams; an absent list means discover them from the sender."""
    streams_raw = cfg.get("streams")
    if streams_raw is None:
        return []
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ValueError(
            "streams must be a non-empty list; omit it to discover streams "
            "from the sender"
        )
    streams = [_parse_stream_config(raw) for raw in streams_raw]
    ipc_names = [name for stream in streams for name in stream.ipc_names]
    if len(ipc_names) != len(set(ipc_names)):
        raise ValueError("CUDA IPC stream names must be unique")
    return streams


def _build_receiver(cfg: dict, args: argparse.Namespace) -> Avs1TcpReceiver:
    receiver_cfg = cfg.get("receiver", {})
    ipc_cfg = cfg.get("ipc", {})
    streams = _configured_streams(cfg)

    env_host = os.environ.get("CAMERA_RECEIVER_HOST") or None
    env_port = os.environ.get("CAMERA_RECEIVER_PORT") or None
    host = str(args.host or env_host or receiver_cfg.get("host", "127.0.0.1"))
    port = int(
        args.port
        if args.port is not None
        else (env_port or receiver_cfg.get("port", 5000))
    )
    gpu_id = _parse_optional_gpu_id(
        args.gpu_id if args.gpu_id is not None else receiver_cfg.get("gpu_id"),
        "receiver.gpu_id",
    )
    socket_path = str(
        args.socket
        or ipc_cfg.get("socket_path", "/tmp/camera_receiver_avs1_cuda_ipc.sock")
    )
    publisher = CudaIpcFramePublisher(
        socket_path=socket_path,
        owner_name=str(ipc_cfg.get("owner_name", "camera_receiver_avs1")),
        queue_size=int(ipc_cfg.get("queue_size", 1024)),
        dynamic_streams=not streams,
    )
    return Avs1TcpReceiver(
        host=host,
        port=port,
        gpu_id=gpu_id,
        max_record_bytes=int(
            receiver_cfg.get("max_record_bytes", _DEFAULT_MAX_RECORD_BYTES)
        ),
        stats_log_interval_s=_resolve_stats_log_interval_s(
            receiver_cfg,
            args.stats_log_interval_s,
        ),
        receive_fps_warning=_resolve_receive_fps_warning_config(receiver_cfg),
        ipc_publisher=publisher,
        streams=streams,
        discover=not streams,
        latency_csv=LatencyCsv(Path(args.latency_csv)) if args.latency_csv else None,
    )


def _dry_run_summary(
    cfg: dict,
    args: argparse.Namespace,
) -> tuple[
    str,
    int,
    str,
    list[StreamConfig],
    float,
    ReceiveFpsWarningConfig,
]:
    receiver_cfg = cfg.get("receiver", {})
    ipc_cfg = cfg.get("ipc", {})
    streams = _configured_streams(cfg)
    env_host = os.environ.get("CAMERA_RECEIVER_HOST") or None
    env_port = os.environ.get("CAMERA_RECEIVER_PORT") or None
    host = str(args.host or env_host or receiver_cfg.get("host", "127.0.0.1"))
    port = int(
        args.port
        if args.port is not None
        else (env_port or receiver_cfg.get("port", 5000))
    )
    socket_path = str(
        args.socket
        or ipc_cfg.get("socket_path", "/tmp/camera_receiver_avs1_cuda_ipc.sock")
    )
    stats_log_interval_s = _resolve_stats_log_interval_s(
        receiver_cfg,
        args.stats_log_interval_s,
    )
    receive_fps_warning = _resolve_receive_fps_warning_config(receiver_cfg)
    return (
        host,
        port,
        socket_path,
        streams,
        stats_log_interval_s,
        receive_fps_warning,
    )


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    default_config = Path(__file__).resolve().parent / "configs" / "receiver.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--host", help="override receiver.host or CAMERA_RECEIVER_HOST")
    parser.add_argument(
        "--port", type=int, help="override receiver.port or CAMERA_RECEIVER_PORT"
    )
    parser.add_argument("--socket", help="override ipc.socket_path")
    parser.add_argument("--gpu-id", type=int, help="override receiver.gpu_id")
    parser.add_argument(
        "--stats-log-interval-s",
        type=float,
        help="seconds between periodic stats summaries; 0 disables them",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="parse and instantiate only"
    )
    parser.add_argument(
        "--latency-csv",
        help="write each stream frame's sender send and receiver receive time",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config(Path(args.config))
    if args.dry_run:
        (
            host,
            port,
            socket_path,
            streams,
            stats_log_interval_s,
            receive_fps_warning,
        ) = _dry_run_summary(cfg, args)
        stats_summary = (
            "disabled"
            if stats_log_interval_s == 0
            else f"every {stats_log_interval_s:g}s"
        )
        fps_warning_summary = (
            "disabled"
            if not receive_fps_warning.enabled
            else (
                (
                    f"below {receive_fps_warning.fraction:g} of each stream's rate "
                    if not streams and receive_fps_warning.fraction > 0
                    else f"below {receive_fps_warning.threshold_fps:g}fps "
                )
                + f"every {receive_fps_warning.interval_s:g}s"
            )
        )
        print(
            "camera_receiver: config ok; "
            f"sender={host}:{port}, socket={socket_path}, "
            "streams="
            + (",".join(stream.name for stream in streams) or "discovered from sender")
            + ", "
            f"periodic_stats={stats_summary}, "
            f"receive_fps_warning={fps_warning_summary}",
            flush=True,
        )
        return 0

    receiver = _build_receiver(cfg, args)

    stop_event = threading.Event()

    def _on_signal(signum, _frame):
        logger.info("stopping on signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    receiver.start()
    try:
        logger.info(
            "camera_receiver running: sender=%s:%d, socket=%s",
            receiver.host,
            receiver.port,
            receiver.socket_path,
        )
        receiver.wait(stop_event)
    finally:
        receiver.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
