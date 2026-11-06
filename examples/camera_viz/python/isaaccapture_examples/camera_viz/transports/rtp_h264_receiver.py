# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GStreamer RTP H.264 receiver — emits Annex-B byte-stream packets.

Wraps the same pipeline shape camera_streamer uses, tuned for VR teleop:

    udpsrc port=N buffer-size=B caps="application/x-rtp,...H264"
      ! rtpjitterbuffer latency=0 drop-on-latency=true do-retransmission=false
      ! rtph264depay
      ! h264parse config-interval=-1
      ! video/x-h264,stream-format=byte-stream,alignment=au
      ! appsink emit-signals=false sync=false max-buffers=1 drop=true

Notes on the low-latency knobs:

* ``rtpjitterbuffer latency=0`` disables the buffering window. Late
  packets are dropped on arrival instead of waiting. For ultra-low-latency
  teleop, a few lost frames are preferable to 50-100 ms of jitter cushion.
* ``do-retransmission=false`` skips RTCP NACK loops — they cost a round
  trip and we'd rather get the next frame than re-request the dropped one.
* ``appsink ... max-buffers=1 drop=true`` keeps only the freshest packet;
  matches the FrameSource mailbox contract.
* ``h264parse config-interval=-1`` re-emits SPS/PPS at every keyframe so
  a late-joining decoder can recover without waiting for the next IDR.

This module is transport-only — it hands raw H.264 byte-stream packets
to whoever asks via :meth:`try_pull_packet`. NVDEC integration is in
``sources/rtp_h264.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger("isaaccapture.camera_viz.transports.rtp_h264_receiver")

# Module-level guard so ``Gst.init`` runs exactly once across all
# transport instances (GStreamer is not happy about double init).
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
                "RTP H.264 transport requires PyGObject + GStreamer. Install "
                "via your distro: e.g. `apt install python3-gi gstreamer1.0-tools "
                "gstreamer1.0-plugins-{good,bad}` on Ubuntu."
            ) from e
        Gst.init(None)
        _GST_INITIALIZED = True


class RtpH264Receiver:
    """Non-blocking RTP H.264 receiver over UDP. Emits Annex-B packets."""

    def __init__(
        self,
        port: int,
        buffer_size: int = 212992,
        latency_ms: int = 0,
        max_buffers: int = 1,
    ) -> None:
        _ensure_gst_initialized()
        from gi.repository import Gst

        self._Gst = Gst
        self._port = port
        self._buffer_size = buffer_size
        self._latency_ms = latency_ms
        self._max_buffers = max_buffers
        self._pipeline = None
        self._appsink = None

    def start(self) -> bool:
        """Build + start the pipeline. Returns True on success, False if
        the pipeline failed to enter PLAYING (port busy, missing plugin)."""
        Gst = self._Gst
        elements = [
            (
                f"udpsrc port={self._port} buffer-size={self._buffer_size} "
                'caps="application/x-rtp,media=video,encoding-name=H264,payload=96"'
            ),
            f"rtpjitterbuffer latency={self._latency_ms} drop-on-latency=true do-retransmission=false",
            "rtph264depay",
            "h264parse config-interval=-1",
            "video/x-h264,stream-format=byte-stream,alignment=au",
            f"appsink name=sink emit-signals=false sync=false max-buffers={self._max_buffers} drop=true",
        ]
        pipeline_str = " ! ".join(elements)
        try:
            self._pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            logger.warning("RtpH264Receiver: Gst.parse_launch failed: %s", e)
            return False
        self._appsink = self._pipeline.get_by_name("sink")
        if self._pipeline is None or self._appsink is None:
            self._pipeline = None
            self._appsink = None
            return False
        rc = self._pipeline.set_state(Gst.State.PLAYING)
        if rc == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None
            return False
        return True

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
            self._appsink = None

    def try_pull_packet(self) -> Optional[bytes]:
        """Non-blocking poll. Returns the next Annex-B H.264 access unit
        as bytes, or None if no packet is available right now."""
        if self._appsink is None:
            return None
        sample = self._appsink.emit("try-pull-sample", 0)
        if not sample:
            return None
        buf = sample.get_buffer()
        if not buf:
            return None
        ok, info = buf.map(self._Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            # ``info.data`` is a memoryview into the GstBuffer; we must copy
            # before unmap or the bytes become invalid mid-decode.
            return bytes(info.data)
        finally:
            buf.unmap(info)
