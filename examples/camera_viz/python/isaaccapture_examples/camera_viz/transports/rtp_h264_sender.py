# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GStreamer RTP H.264 sender — NVENC + RTP packetize + udpsink.

Mirrors the receiver: GStreamer owns RTP transport, NVENC owns the
codec. Pipeline:

    appsrc is-live=true do-timestamp=true format=time
      ! h264parse config-interval=-1
      ! rtph264pay pt=96 config-interval=-1 mtu=1400
      ! udpsink host=<host> port=<port> sync=false async=false

* ``h264parse config-interval=-1`` re-emits SPS/PPS at every keyframe
  so late-joining receivers can re-init their decoder without waiting
  for a fresh stream.
* ``mtu=1400`` keeps each RTP packet under typical Ethernet MTU so
  routers don't fragment (fragmentation is a latency killer over WiFi).
* ``udpsink sync=false async=false`` skips clock sync — we want to push
  packets out as soon as NVENC produces them; no pacing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .rtp_h264_receiver import _ensure_gst_initialized

logger = logging.getLogger("isaaccapture.camera_viz.transports.rtp_h264_sender")

RECONNECT_DELAY_S = 2.0


class RtpH264Sender:
    """RGBA → NVENC → RTP → UDP. Pulls from a ``FrameSource`` via its
    ``latest()`` mailbox and ships each new frame out as an RTP H.264
    stream."""

    def __init__(
        self,
        source,
        encoder,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: int = 30,
        mtu: int = 1400,
    ) -> None:
        # fps must be > 0 — _push_packet does ``int(1e9 / fps)`` for the
        # buffer duration, which would ZeroDivisionError or go negative
        # otherwise. Validate up front so the failure mode is a clear
        # ValueError at construction rather than a cryptic crash on the
        # first encoded frame.
        if fps <= 0:
            raise ValueError(f"RtpH264Sender: fps must be > 0, got {fps}")

        _ensure_gst_initialized()
        from gi.repository import Gst

        self._Gst = Gst
        self._source = source
        self._host = host
        self._port = port
        self._width = width
        self._height = height
        self._fps = fps
        self._mtu = mtu
        # Encoder is injected; see transports._encoder_factory.make_encoder().
        self._encoder = encoder

        self._pipeline = None
        self._appsrc = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._last_reconnect_attempt_s = 0.0
        self._reconnect_count = 0
        self._frame_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._source.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._send_loop, name=f"rtp_h264_send_{self._port}", daemon=False
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    # Don't null _thread — is_alive() must keep reporting
                    # the live thread for the supervisor, and the non-
                    # daemon thread will still block process exit. Fall
                    # through to teardown anyway: setting the GStreamer
                    # pipeline state to NULL often unblocks a wedged send
                    # loop, and we never want to leak the source either.
                    logger.warning("RtpH264Sender: send thread did not exit within 5s")
                else:
                    self._thread = None
        finally:
            self._teardown_pipeline()
            try:
                self._source.stop()
            except Exception:
                pass

    def is_alive(self) -> bool:
        """True while the send loop thread is running. Returns False if
        ``start()`` hasn't been called, after ``stop()``, or if the loop
        died with an uncaught exception. Supervisors should poll this
        to detect post-startup crashes."""
        return self._thread is not None and self._thread.is_alive()

    def _build_pipeline(self) -> bool:
        Gst = self._Gst
        elements = [
            "appsrc name=src is-live=true do-timestamp=true format=time block=false",
            "h264parse config-interval=-1",
            f"rtph264pay pt=96 config-interval=-1 mtu={self._mtu}",
            f"udpsink host={self._host} port={self._port} sync=false async=false",
        ]
        try:
            self._pipeline = Gst.parse_launch(" ! ".join(elements))
        except Exception as e:
            logger.warning("RtpH264Sender: parse_launch failed (%s)", e)
            return False
        self._appsrc = self._pipeline.get_by_name("src")
        if self._pipeline is None or self._appsrc is None:
            self._pipeline = None
            self._appsrc = None
            return False
        # Set caps so h264parse / rtph264pay know the byte-stream format.
        caps = Gst.Caps.from_string(
            f"video/x-h264,stream-format=byte-stream,alignment=au,"
            f"width={self._width},height={self._height},framerate={self._fps}/1"
        )
        self._appsrc.set_property("caps", caps)
        rc = self._pipeline.set_state(Gst.State.PLAYING)
        if rc == Gst.StateChangeReturn.FAILURE:
            self._teardown_pipeline()
            return False
        return True

    def _teardown_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception:
                pass
            self._pipeline = None
            self._appsrc = None
        self._connected = False

    def _push_packet(self, packet: bytes) -> bool:
        Gst = self._Gst
        buf = Gst.Buffer.new_wrapped(packet)
        # No manual PTS — appsrc ``do-timestamp=true`` stamps each buffer
        # with the pipeline clock at push time, which is what
        # rtpjitterbuffer expects. Setting buf.pts here would fight the
        # appsrc and introduce drift between our synthetic monotonic
        # counter and the actual arrival cadence.
        flow = self._appsrc.emit("push-buffer", buf)
        return flow == Gst.FlowReturn.OK

    # fps log cadence + consecutive-encode-fail threshold for hard restart.
    _FPS_REPORT_S = 5.0
    _ENCODE_FAIL_THRESHOLD = 30

    def _send_loop(self) -> None:
        # Idle poll interval. ``FrameSource.latest()`` returns None when no
        # NEW frame has arrived since the previous call, so the loop only
        # idles when there's nothing to ship — it does NOT spin on a steady
        # stream. The interval bounds the wakeup latency between "frame
        # published" and "encoder sees it": at 1 ms the average add is
        # 0.5 ms; the old 1/fps (33 ms @ 30 fps) added ~16 ms on average
        # for nothing. CPU cost of a sleeping thread polling at 1 kHz is
        # negligible (~0% on any modern host).
        idle_poll_s = 0.001

        # Pin to the source's GPU once we see the first frame. The encoder
        # is lazy-init'd inside encode() based on the input device, so the
        # outer context just guards our own CUDA ops (Stream.set, kernel
        # launches via with self._stream:) from drifting.
        import cupy as cp

        device_pinned = False
        consecutive_encode_failures = 0
        last_fps_report_at = time.monotonic()
        last_fps_count = 0

        while not self._stop.is_set():
            if not self._connected:
                now = time.monotonic()
                if now - self._last_reconnect_attempt_s < RECONNECT_DELAY_S:
                    self._stop.wait(timeout=0.1)
                    continue
                self._last_reconnect_attempt_s = now
                try:
                    self._connected = self._build_pipeline()
                except Exception as e:
                    logger.warning("RtpH264Sender: pipeline build failed (%s)", e)
                    self._connected = False
                if not self._connected:
                    self._reconnect_count += 1
                    continue
                logger.info(
                    "RtpH264Sender: sending to %s:%d (%dx%d@%dfps)%s",
                    self._host,
                    self._port,
                    self._width,
                    self._height,
                    self._fps,
                    f" (reconnect #{self._reconnect_count})"
                    if self._reconnect_count
                    else "",
                )

            frame = self._source.latest()
            if frame is None:
                self._stop.wait(timeout=idle_poll_s)
                continue

            # First-frame device pin: capture whichever GPU the source's
            # buffers live on (set by VizSession's Vulkan adapter choice)
            # and lock this thread to that device for the rest of the run.
            if not device_pinned:
                cp.cuda.runtime.setDevice(int(frame.image.device.id))
                device_pinned = True

            try:
                packets = self._encoder.encode(frame.image)
                consecutive_encode_failures = 0
            except Exception as e:
                consecutive_encode_failures += 1
                logger.warning(
                    "RtpH264Sender: encode failed (%s); resetting encoder (%d/%d)",
                    e,
                    consecutive_encode_failures,
                    self._ENCODE_FAIL_THRESHOLD,
                )
                self._encoder.reset()
                # Persistent failures = wedged NVENC; let the supervisor restart.
                if consecutive_encode_failures >= self._ENCODE_FAIL_THRESHOLD:
                    raise RuntimeError(
                        f"RtpH264Sender: encode failed {consecutive_encode_failures} "
                        f"times in a row; surfacing to supervisor for full restart"
                    ) from e
                continue

            for pkt in packets:
                if not self._push_packet(pkt):
                    logger.warning(
                        "RtpH264Sender: push-buffer returned non-OK; reconnecting"
                    )
                    self._teardown_pipeline()
                    break
            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - last_fps_report_at
            if elapsed >= self._FPS_REPORT_S:
                actual = (self._frame_count - last_fps_count) / elapsed
                tag = " ⚠ throttled" if actual < 0.8 * self._fps else ""
                logger.info(
                    "RtpH264Sender :%d fps=%.1f/%d%s",
                    self._port,
                    actual,
                    self._fps,
                    tag,
                )
                last_fps_report_at = now
                last_fps_count = self._frame_count
