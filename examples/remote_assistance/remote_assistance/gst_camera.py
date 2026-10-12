# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Jetson V4L2 hardware encoding with complete access units for LiveKit."""

import asyncio
from collections import deque
from fractions import Fraction
import logging
import threading
import time

from livekit import rtc

from .connection import connect, disconnect
from .encoded_source import EncodedSource

logger = logging.getLogger("isaaccapture.remote_assistance.gst_camera")


def parameters(settings, spec):
    codec = str(settings.get("codec", "h264")).lower()
    if codec not in ("h264", "h265"):
        raise ValueError("gstreamer encoder supports h264 or h265")
    if spec.get("type") != "v4l2" or spec.get("stereo", False):
        raise ValueError("gstreamer encoder requires a mono V4L2 camera")
    if spec.get("fourcc", "YUYV") != "YUYV":
        raise ValueError("gstreamer camera currently requires fourcc: YUYV")
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
                "expose system dist-packages to this Python 3.12 venv (see README)"
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
        self.pipeline = Gst.parse_launch(
            f"v4l2src name=camera do-timestamp=true ! "
            f"video/x-raw,format=YUY2,width={width},height={height},framerate={fps.numerator}/{fps.denominator} ! "
            f"videoconvert ! video/x-raw,format=I420 ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2{codec}enc name=encoder bitrate={bitrate} iframeinterval={max(1, round(fps))} "
            f"insert-sps-pps=true ! {codec}parse config-interval=-1 ! "
            f"video/x-{codec},stream-format=byte-stream,alignment=au ! "
            "appsink name=output emit-signals=true sync=false max-buffers=2 drop=false"
        )
        self.pipeline.get_by_name("camera").set_property("device", str(spec["device"]))
        self.encoder = self.pipeline.get_by_name("encoder")
        # Disable reordering where the platform exposes this property.
        if self.encoder.find_property("num-B-Frames"):
            self.encoder.set_property("num-B-Frames", 0)
        if self.encoder.find_property("idrinterval"):
            self.encoder.set_property("idrinterval", max(1, round(fps)))
        self.pipeline.get_by_name("output").connect("new-sample", self._sample)
        self.bus = self.pipeline.get_bus()
        self.bitrate = bitrate
        self.rate_warning = False
        self.fps_warning = False
        self.fps = float(fps)

    def _sample(self, sink):
        Gst = self.Gst
        try:
            sample = sink.emit("pull-sample")
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
            item = (payload, key, stamp, self.count)
            self.count += 1
            self.bytes += len(payload)
            self.last_frame = time.monotonic()
            if self.units.push(item):
                self.force_key.set()
            return Gst.FlowReturn.OK
        except Exception as exc:
            self.error = exc
            return Gst.FlowReturn.ERROR

    def start(self):
        if (
            self.pipeline.set_state(self.Gst.State.PLAYING)
            == self.Gst.StateChangeReturn.FAILURE
        ):
            raise RuntimeError("GStreamer camera failed to start")

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
        if time.monotonic() - self.last_frame > 10:
            raise RuntimeError("No encoded camera frames for 10 seconds")

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
        if feedback.HasField("rate_control"):
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
        self.pipeline.set_state(self.Gst.State.NULL)


async def publish_encoded_camera(settings, spec, stop):
    if settings.get("video_encoder") == "depthai":
        from .oak_camera import OakCapture, parameters as oak_parameters

        codec, width, height, fps, bitrate = oak_parameters(settings, spec)
        capture = OakCapture(settings, spec)
        backend = "OAK onboard H.264"
    else:
        codec, width, height, fps, bitrate = parameters(settings, spec)
        capture = GstCapture(settings, spec)
        backend = f"NVIDIA nvv4l2{codec}enc"
    source = None
    room = rtc.Room()
    accepted = rejected = 0
    try:
        source = EncodedSource(width, height, codec)
        track = rtc.LocalVideoTrack.create_video_track(spec["name"], source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_codec=rtc.VideoCodec.Value(codec.upper()),
            video_encoding=rtc.VideoEncoding(
                max_framerate=float(fps), max_bitrate=bitrate
            ),
            frame_metadata_features=[
                rtc.FrameMetadataFeature.FMF_FRAME_ID,
                rtc.FrameMetadataFeature.FMF_USER_TIMESTAMP,
            ],
        )
        options.video_encoder = (
            options.DESCRIPTOR.fields_by_name["video_encoder"]
            .enum_type.values_by_name["ENCODER_BACKEND_PRE_ENCODED"]
            .number
        )
        room.on("disconnected", lambda *_: stop.set())
        await connect(room, settings, settings.get("publisher", "robot-camera"))
        await room.local_participant.publish_track(track, options)
        await asyncio.to_thread(capture.start)
        logger.info(
            "Requested %s -> LiveKit PRE_ENCODED, %dx%d at %s fps",
            backend,
            width,
            height,
            fps,
        )
        report = feedback_at = time.monotonic()
        old_count = old_bytes = 0
        while not stop.is_set():
            capture.check()
            item = capture.units.pop()
            if item:
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
                    "Encoded %.1f fps, %.2f Mbps payload; LiveKit accepted=%d rejected=%d queue_dropped=%d",
                    (capture.count - old_count) / elapsed,
                    (capture.bytes - old_bytes) * 8 / elapsed / 1e6,
                    accepted,
                    rejected,
                    capture.units.dropped,
                )
                old_count, old_bytes, report = capture.count, capture.bytes, now
            await asyncio.sleep(0.001)
    finally:
        await asyncio.to_thread(capture.stop)
        await disconnect(room)
        if source is not None:
            await source.aclose()
