# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Jetson camera hardware encoding with complete access units for LiveKit."""

import asyncio
from collections import deque
from fractions import Fraction
import logging
import threading
import time

from isaac_media_transport.transport import get_transport
from .gst_sources import camera_source
from isaac_media_transport.stereo import camera_specs
from .gst_latency import GstTiming
from isaac_media_transport.latency import get_recorder

logger = logging.getLogger("isaaccapture.remote_assistance.gst_camera")


def parameters(settings, spec):
    codec = str(settings.get("codec", "h264")).lower()
    if codec not in ("h264", "h265"):
        raise ValueError("gstreamer encoder supports h264 or h265")
    camera_source(spec)
    width, height = int(spec["width"]), int(spec["height"])
    fps = Fraction(str(spec["fps"])).limit_denominator(1001)
    bitrate = int(float(settings.get("bitrate_mbps", 15)) * 1_000_000)
    if min(width, height, fps, bitrate) <= 0 or width % 2 or height % 2:
        raise ValueError("Positive fps/bitrate and even positive dimensions required")
    return codec, width, height, fps, bitrate


class AccessUnits:
    """After overflow, discard dependent frames until a fresh keyframe arrives."""

    def __init__(self, limit=4):
        self.items = deque()
        self.limit = limit
        self.wait_key = True
        self.dropped = 0
        self.lock = threading.Lock()

    def push(self, item):
        with self.lock:
            overflow = len(self.items) >= self.limit
            if overflow:
                self.dropped += len(self.items)
                self.items.clear()
                self.wait_key = True
            if self.wait_key and not item[1]:
                self.dropped += 1
            else:
                self.wait_key = False
                self.items.append(item)
            return overflow

    def reset(self):
        with self.lock:
            self.dropped += len(self.items)
            self.items.clear()
            self.wait_key = True

    def pop(self):
        with self.lock:
            return self.items.popleft() if self.items else None


class GstCapture:
    def __init__(self, settings, spec):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstVideo
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "Install python3-gi, gir1.2-gstreamer-1.0 and gir1.2-gst-plugins-base-1.0; "
                "use bindings built for the bridge's Python version (see README)"
            ) from exc
        Gst.init(None)
        self.Gst, self.GstVideo = Gst, GstVideo
        codec, width, height, fps, bitrate = parameters(settings, spec)
        self.max_bitrate = bitrate
        self.units = AccessUnits()
        self.error = None
        self.force_key = threading.Event()
        self.count = self.bytes = 0
        self.last_frame = time.monotonic()
        adapter = camera_source(spec)
        self.adapter = adapter
        self.startup_timeout = getattr(adapter, "startup_timeout", 10)
        self.diagnostics = settings.get("_capture_endpoint")
        self.track_name = spec["name"]
        capture_branch = ""
        fork = ""
        if self.diagnostics:
            fork = "tee name=diagnostic_tee ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! "
            capture_branch = (
                " diagnostic_tee. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
                "valve name=diagnostic_valve drop=true ! nvvidconv ! video/x-raw,format=RGBA ! "
                "appsink name=diagnostic_raw emit-signals=true sync=false async=false max-buffers=1 drop=true"
            )
        self.pipeline = Gst.parse_launch(
            adapter.pipeline(width, height, f"{fps.numerator}/{fps.denominator}")
            + fork
            + f"nvv4l2{codec}enc name=encoder bitrate={bitrate} iframeinterval={max(1, round(fps))} "
            f"insert-sps-pps=true ! {codec}parse config-interval=-1 ! "
            f"video/x-{codec},stream-format=byte-stream,alignment=au ! "
            "appsink name=output emit-signals=true sync=false max-buffers=2 drop=false"
            + capture_branch
        )
        adapter.configure(self.pipeline.get_by_name("camera"))
        self.encoder = self.pipeline.get_by_name("encoder")
        if hasattr(adapter, "configure_encoder"):
            adapter.configure_encoder(self.encoder)
        # Disable reordering where the platform exposes this property.
        if self.encoder.find_property("num-B-Frames"):
            self.encoder.set_property("num-B-Frames", 0)
        if self.encoder.find_property("idrinterval"):
            self.encoder.set_property("idrinterval", max(1, round(fps)))
        self.pipeline.get_by_name("output").connect("new-sample", self._sample)
        self.timing = GstTiming(settings, self.track_name, self.pipeline, Gst)
        if self.diagnostics:
            valve = self.pipeline.get_by_name("diagnostic_valve")
            self.diagnostics.register(
                self.track_name,
                stage="source-before-jetson-encoder-after-isp-flip",
                width=width,
                height=height,
                fps=fps,
                on_active=lambda active: valve.set_property("drop", not active),
            )
            self.pipeline.get_by_name("diagnostic_raw").connect(
                "new-sample", self._raw_sample
            )
        self.bus = self.pipeline.get_bus()
        self.bitrate = bitrate
        self.rate_warning = False
        self.fps_warning = False
        self.fps = float(fps)

    def _sample(self, sink):
        Gst = self.Gst
        try:
            sample = sink.emit("pull-sample")
            access_unit_ns = time.monotonic_ns() if self.timing.enabled else None
            buf = sample.get_buffer()
            if buf.pts == Gst.CLOCK_TIME_NONE:
                raise RuntimeError("Encoded frame has no capture timestamp")
            running = (
                self.pipeline.get_clock().get_time() - self.pipeline.get_base_time()
            )
            stamp = (time.monotonic_ns() - running + buf.pts) // 1000
            key = not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
            payload = buf.extract_dup(0, buf.get_size())
            if not payload:
                return Gst.FlowReturn.OK
            stamp, _source_index = (
                self.adapter.frame_metadata(buf.pts)
                if hasattr(self.adapter, "frame_metadata")
                else (stamp, self.count)
            )
            self.timing.complete(buf.pts, stamp * 1000, access_unit_ns)
            # Input rate limiting is not packet loss: encoded IDs must stay contiguous
            # so GPU receivers only reset on gaps after encoding/publication.
            item = (payload, key, stamp, self.count)
            self.last_frame = time.monotonic()
            self.count += 1
            self.bytes += len(payload)
            if self.units.push(item):
                self.force_key.set()
            return Gst.FlowReturn.OK
        except Exception as exc:
            self.error = exc
            return Gst.FlowReturn.ERROR

    def _raw_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not self.diagnostics.wanted(self.track_name):
            return self.Gst.FlowReturn.OK
        try:
            info = self.GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            buf = sample.get_buffer()
            running = (
                self.pipeline.get_clock().get_time() - self.pipeline.get_base_time()
            )
            stamp_ns = time.monotonic_ns() - running + buf.pts

            def pixels():
                raw = buf.extract_dup(0, buf.get_size())
                layout = self.GstVideo.buffer_get_video_meta(buf)
                stride = layout.stride[0] if layout else info.stride[0]
                offset = layout.offset[0] if layout else info.offset[0]
                return b"".join(
                    raw[offset + row * stride : offset + row * stride + info.width * 4]
                    for row in range(info.height)
                )

            self.diagnostics.offer(
                self.track_name,
                pixels,
                dict(
                    width=info.width,
                    height=info.height,
                    source_timestamp_ns=stamp_ns,
                    gst_pts_ns=buf.pts,
                    frame_id=None,
                ),
            )
        except Exception:
            # Diagnostics must not stop the primary media pipeline.
            logger.exception("Raw diagnostic frame capture failed")
        return self.Gst.FlowReturn.OK

    def start(self):
        self.last_frame = time.monotonic()
        if (
            self.pipeline.set_state(self.Gst.State.PLAYING)
            == self.Gst.StateChangeReturn.FAILURE
        ):
            raise RuntimeError("GStreamer camera failed to start")
        if hasattr(self.adapter, "start"):
            try:
                self.adapter.start(self.pipeline, self.Gst)
            except BaseException:
                self.pipeline.set_state(self.Gst.State.NULL)
                raise

    def check(self):
        msg = self.bus.timed_pop_filtered(
            0, self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS
        )
        if msg:
            if msg.type == self.Gst.MessageType.ERROR:
                error, debug = msg.parse_error()
                raise RuntimeError(f"GStreamer: {error}; {debug}")
            raise RuntimeError("Camera pipeline ended unexpectedly")
        if self.error:
            raise RuntimeError("Encoded camera capture failed") from self.error
        adapter_error = getattr(getattr(self, "adapter", None), "error", None)
        if adapter_error is not None:
            raise RuntimeError("Camera source failed") from adapter_error
        timeout = self.startup_timeout if self.count == 0 else 10
        if time.monotonic() - self.last_frame > timeout:
            raise RuntimeError(f"No encoded camera frames for {timeout} seconds")

    def feedback(self, feedback):
        if feedback.keyframe_requested or self.force_key.is_set():
            self.force_key.clear()
            event = self.GstVideo.video_event_new_upstream_force_key_unit(
                self.Gst.CLOCK_TIME_NONE, True, 0
            )
            if not self.encoder.get_static_pad("src").send_event(event):
                logger.warning(
                    "Encoder rejected keyframe request; periodic IDR remains enabled"
                )
        if feedback.rate_control is not None:
            requested_fps = feedback.rate_control.framerate_fps
            if 0 < requested_fps < self.fps and not self.fps_warning:
                logger.warning(
                    "LiveKit requests lower fps; this experimental pipeline keeps capture fps fixed"
                )
                self.fps_warning = True
            requested_bitrate = feedback.rate_control.target_bitrate_bps
            if requested_bitrate <= 0:
                return
            target = min(self.max_bitrate, requested_bitrate)
            prop = self.encoder.find_property("bitrate")
            if prop.flags & self.Gst.PARAM_MUTABLE_PLAYING:
                if target != self.bitrate:
                    self.encoder.set_property("bitrate", target)
                    self.bitrate = target
                    logger.debug("Encoder bitrate target %d bps", target)
            elif not self.rate_warning:
                logger.warning(
                    "Encoder lacks runtime bitrate updates; fixed bitrate requires sufficient network capacity"
                )
                self.rate_warning = True

    def stop(self):
        try:
            if hasattr(self.adapter, "stop"):
                self.adapter.stop()
        finally:
            self.pipeline.set_state(self.Gst.State.NULL)


async def publish_encoded_camera(settings, spec, stop):
    specs = camera_specs(settings, spec)
    session = get_transport(settings).video_sender(settings, stop)
    try:
        await session.start()
        async with asyncio.TaskGroup() as tasks:
            for child in specs:
                tasks.create_task(_publish_track(settings, child, stop, session))
    finally:
        await session.close()


async def _publish_track(settings, spec, stop, session):
    timing = get_recorder(settings)
    codec, width, height, fps, _bitrate = parameters(settings, spec)
    capture = GstCapture(settings, spec)
    backend = f"NVIDIA nvv4l2{codec}enc"
    source = None
    accepted = rejected = 0
    try:
        source = await session.publish(spec, encoded=True)
        startup = asyncio.create_task(asyncio.to_thread(capture.start))
        try:
            await asyncio.shield(startup)
        except BaseException:
            # Cancellation cannot stop a worker: join it before changing pipeline state.
            await asyncio.gather(startup, return_exceptions=True)
            raise
        logger.info(
            "Requested %s -> %s pre-encoded transport, %dx%d at %s fps; track=%s",
            backend,
            settings.get("transport", "livekit"),
            width,
            height,
            fps,
            spec["name"],
        )
        report = feedback_at = time.monotonic()
        old_count = old_bytes = 0
        while not stop.is_set():
            capture.check()
            item = capture.units.pop()
            if item:
                timing.record("publish_begin", spec["name"], item[2] * 1000)
                if source.capture_encoded(*item):
                    accepted += 1
                else:
                    rejected += 1
                    capture.units.reset()
                    capture.force_key.set()
            now = time.monotonic()
            if now - feedback_at >= 0.1:
                capture.feedback(source.feedback())
                feedback_at = now
            if now - report >= 5:
                elapsed = now - report
                logger.info(
                    "Encoded %.1f fps, %.2f Mbps payload; %s accepted=%d rejected=%d queue_dropped=%d track=%s",
                    (capture.count - old_count) / elapsed,
                    (capture.bytes - old_bytes) * 8 / elapsed / 1e6,
                    settings.get("transport", "livekit"),
                    accepted,
                    rejected,
                    capture.units.dropped,
                    spec["name"],
                )
                old_count, old_bytes, report = capture.count, capture.bytes, now
            await asyncio.sleep(0.001)
    finally:
        try:
            await asyncio.to_thread(capture.stop)
        finally:
            if source is not None:
                await source.close()
