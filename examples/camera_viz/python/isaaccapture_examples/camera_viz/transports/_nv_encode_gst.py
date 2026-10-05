# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GStreamer-based NVENC H.264 encoder — Jetson alternative to ``NvH264Encoder``.

    appsrc(NV12 sysmem) ! nvv4l2h264enc ! h264parse ! appsink

Jetson L4T R35+: ``nvv4l2h264enc`` (V4L2 M2M NVENC). Falls back to
``nvh264enc`` (desktop GstCUDA) if the Jetson plugin is missing.

RGBA → NV12 conversion runs on GPU via the shared CuPy kernel; result is
downloaded to a pinned host buffer and pushed to ``appsrc`` (D2H cost
~1-3 ms at 720p).
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from ._nv12_convert import build_rgba_to_nv12_kernel, launch_rgba_to_nv12

logger = logging.getLogger("isaaccapture.camera_viz.transports._nv_encode_gst")

# Encoder element candidates in priority order. nvv4l2h264enc is Jetson;
# nvh264enc is the desktop GstCUDA path (gst-plugins-bad with NVIDIA
# headers). x264enc is a CPU fallback — listed for completeness, but if
# you're using the GStreamer backend you almost certainly want hardware.
_ENCODER_CANDIDATES = ("nvv4l2h264enc", "nvh264enc", "x264enc")


# Module-level GStreamer init guard — shared with the sender so a process
# that imports both this and rtp_h264_sender doesn't double-init.
_GST_INIT_LOCK = threading.Lock()
_GST_INITIALIZED = False


def _ensure_gst_initialized() -> None:
    global _GST_INITIALIZED
    with _GST_INIT_LOCK:
        if _GST_INITIALIZED:
            return
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except ImportError as e:
            raise RuntimeError(
                "GstNvH264Encoder requires PyGObject + GStreamer. Install "
                "via your distro: e.g. `apt install python3-gi gstreamer1.0-tools "
                "gstreamer1.0-plugins-{good,bad}` on Ubuntu, plus the Jetson "
                "NVIDIA plugins on L4T."
            ) from e
        Gst.init(None)
        _GST_INITIALIZED = True


def _select_encoder_element():
    """Pick the best available H.264 encoder element. Returns (name, factory)."""
    from gi.repository import Gst

    for name in _ENCODER_CANDIDATES:
        factory = Gst.ElementFactory.find(name)
        if factory is not None:
            return name, factory
    raise RuntimeError(
        f"GstNvH264Encoder: no usable encoder plugin (tried {_ENCODER_CANDIDATES}). "
        "On Jetson, install `gstreamer1.0-plugins-nvidia-l4t`; on desktop, "
        "install gst-plugins-bad with NVIDIA codec headers."
    )


# Per-plugin property formatting. Profile is left as plugin default
# (typically high) so behavior matches the native NVENC path.
def _encoder_args(name: str, *, bitrate: int, gop: int) -> str:
    if name == "nvv4l2h264enc":
        # Jetson V4L2 M2M NVENC. control-rate=1 → CBR. preset-level=1 →
        # ultra-fast (lowest latency). insert-sps-pps=true lets the
        # receiver sync mid-stream.
        return (
            f"{name} bitrate={bitrate} iframeinterval={gop} "
            f"insert-sps-pps=true control-rate=1 preset-level=1 "
            f"maxperf-enable=true"
        )
    if name == "nvh264enc":
        return (
            f"{name} bitrate={bitrate // 1000} gop-size={gop} "
            f"preset=low-latency-hq rc-mode=cbr"
        )
    if name == "x264enc":
        return (
            f"{name} bitrate={bitrate // 1000} key-int-max={gop} "
            f"tune=zerolatency speed-preset=ultrafast"
        )
    raise RuntimeError(f"GstNvH264Encoder: unknown encoder {name!r}")


class GstNvH264Encoder:
    """RGBA8 GPU input → NVENC H.264 bytes via a self-contained GStreamer
    pipeline. Jetson-friendly drop-in for :class:`NvH264Encoder`.

    Same API as ``NvH264Encoder`` so :class:`RtpH264Sender` can take
    either backend interchangeably. The pipeline is built lazily on the
    first ``encode()`` so a never-firing sender doesn't grab NVENC slots.
    """

    def __init__(
        self,
        width: int,
        height: int,
        bitrate: int = 15_000_000,
        fps: int = 30,
        gop: Optional[int] = None,
        gpu_id: int = 0,  # unused on Jetson (single NVENC); accepted for API parity
    ) -> None:
        # ULL defaults: 15 Mbps, IDR every 5 s (fps*5), CBR rate control,
        # no B-frames. Same on both encoder backends.
        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._fps = fps
        self._gop = gop if gop is not None else fps * 5
        self._gpu_id = gpu_id

        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._kernel = None
        self._stream = None
        self._nv12_gpu = None  # CuPy NV12 staging (output of GPU kernel)
        self._nv12_host = None  # numpy NV12 host buffer (D2H target → appsrc)
        self._pts_ns = 0

    def _ensure_initialized(self, rgba_device_id: int) -> None:
        if self._pipeline is not None:
            return
        _ensure_gst_initialized()
        import cupy as cp
        from gi.repository import Gst

        # Pre-allocate GPU NV12 staging on the input's device. NV12 is
        # 1.5 bytes/pixel — one contiguous (H*1.5)xW uint8 buffer with
        # Y at [0..h*w] and UV at [h*w..h*w*1.5].
        with cp.cuda.Device(rgba_device_id):
            self._nv12_gpu = cp.empty(
                (self._height * 3 // 2, self._width), dtype=cp.uint8
            )
            self._stream = cp.cuda.Stream(non_blocking=True)
        self._kernel = build_rgba_to_nv12_kernel()

        # Host buffer for the D2H download — plain numpy is fine here;
        # the pinned-memory perf win is marginal vs the actual GPU→CPU
        # transfer at 720p (~1 ms vs ~0.7 ms pinned).
        self._nv12_host = np.empty((self._height * 3 // 2, self._width), dtype=np.uint8)

        # Build the pipeline. appsrc emits NV12 sysmem at the configured
        # caps; encoder picks the best available plugin; appsink hands us
        # H.264 Annex-B bytes one access-unit at a time.
        encoder_name, _ = _select_encoder_element()
        logger.info("GstNvH264Encoder: using %s", encoder_name)
        encoder_args = _encoder_args(encoder_name, bitrate=self._bitrate, gop=self._gop)
        elements = [
            (
                f"appsrc name=src is-live=true do-timestamp=true format=time "
                f"block=false "
                f"caps=video/x-raw,format=NV12,width={self._width},"
                f"height={self._height},framerate={self._fps}/1"
            ),
            encoder_args,
            "h264parse config-interval=-1",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            "appsink name=sink emit-signals=false sync=false max-buffers=4 drop=false",
        ]
        pipeline_str = " ! ".join(elements)
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        self._appsink = self._pipeline.get_by_name("sink")
        if self._pipeline is None or self._appsrc is None or self._appsink is None:
            raise RuntimeError("GstNvH264Encoder: failed to parse encoder pipeline")
        rc = self._pipeline.set_state(Gst.State.PLAYING)
        if rc == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GstNvH264Encoder: pipeline failed to PLAY")
        self._Gst = Gst

    def encode(self, rgba) -> List[bytes]:
        """Encode one GPU-resident RGBA8 frame. Returns a list of
        Annex-B H.264 packets (empty list while the encoder is spinning
        up, possibly >1 once the pipeline catches up)."""
        rgba_device_id = int(rgba.device.id)
        self._ensure_initialized(rgba_device_id)

        # GPU RGBA → NV12, then D2H download to host buffer.
        h, w = self._height, self._width
        launch_rgba_to_nv12(
            self._kernel,
            rgba,
            self._nv12_gpu[:h, :],
            self._nv12_gpu[h:, :],
            stream=self._stream,
            width=w,
            height=h,
        )
        # Synchronise then download. The host buffer is plain numpy, so
        # cp.asnumpy is the right call; pinning is a marginal perf knob.
        import cupy as cp

        self._stream.synchronize()
        cp.asnumpy(self._nv12_gpu, out=self._nv12_host)

        # Push to appsrc as a Gst.Buffer wrapping our host bytes.
        Gst = self._Gst
        buf = Gst.Buffer.new_wrapped(self._nv12_host.tobytes())
        buf.pts = self._pts_ns
        buf.duration = int(1e9 / max(self._fps, 1))
        self._pts_ns += buf.duration
        if self._appsrc.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            raise RuntimeError("GstNvH264Encoder: appsrc push-buffer rejected")

        # Drain whatever H.264 access units are ready right now. Encoder
        # is typically 1-2 frames behind, so the first few encode() calls
        # may return [] until the pipeline warms up.
        packets: List[bytes] = []
        while True:
            sample = self._appsink.emit("try-pull-sample", 0)
            if sample is None:
                break
            gst_buf = sample.get_buffer()
            ok, info = gst_buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                packets.append(bytes(info.data))
            finally:
                gst_buf.unmap(info)
        return packets

    def end_of_stream(self) -> List[bytes]:
        """Send EOS + drain any buffered packets the encoder still has."""
        if self._pipeline is None or self._appsrc is None or self._appsink is None:
            return []
        Gst = self._Gst
        self._appsrc.emit("end-of-stream")
        packets: List[bytes] = []
        while True:
            sample = self._appsink.emit("try-pull-sample", 100 * Gst.MSECOND)
            if sample is None:
                break
            gst_buf = sample.get_buffer()
            ok, info = gst_buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                packets.append(bytes(info.data))
            finally:
                gst_buf.unmap(info)
        return packets

    def reset(self) -> None:
        """Tear the pipeline down so the next encode() rebuilds it fresh."""
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsrc = None
        self._appsink = None
