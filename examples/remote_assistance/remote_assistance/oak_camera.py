# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RVC2 OAK color camera with onboard H.264 encoding (DepthAI v2)."""

import logging
import re
import threading
import time

from .gst_camera import AccessUnits

logger = logging.getLogger("isaaccapture.remote_assistance.oak_camera")


def parameters(settings, spec):
    width, height, fps = int(spec["width"]), int(spec["height"]), float(spec["fps"])
    bitrate = int(float(settings.get("bitrate_mbps", 15)) * 1_000_000)
    if settings.get("codec", "h264").lower() != "h264":
        raise ValueError("OAK backend currently supports only h264")
    if spec.get("type") != "depthai" or spec.get("stereo", False):
        raise ValueError("OAK backend requires a mono type: depthai camera")
    modes = {
        (1280, 720): (30, 60),
        (1920, 1080): (30, 60),
        (3840, 2160): (30,),
        (2016, 1520): (30, 60),
    }
    if width % 32:
        raise ValueError(
            "OAK H.264 width must be divisible by 32; use 2016x1520 for the 2024x1520 sensor mode"
        )
    if fps not in modes.get((width, height), ()):
        raise ValueError(
            "Unsupported OAK resolution/fps; use 720p/1080p/2016x1520 at 30/60 or 4K at 30"
        )
    if not 0 < bitrate <= 20_000_000:
        raise ValueError("OAK bitrate_mbps must be greater than 0 and at most 20")
    return "h264", width, height, fps, bitrate


class H264AccessUnits:
    """Cache parameter sets and recover at IDR after encoded sequence gaps."""

    def __init__(self):
        self.headers = {}
        self.sequence = None
        self.units = AccessUnits()

    def push(self, payload, stamp, sequence):
        if self.sequence is not None and sequence != self.sequence + 1:
            self.units.reset()
        self.sequence = sequence
        nals = [nal for nal in re.split(b"\x00\x00(?:\x00)?\x01", payload) if nal]
        types = {nal[0] & 31 for nal in nals}
        for nal in nals:
            kind = nal[0] & 31
            if kind in (7, 8):
                self.headers[kind] = b"\x00\x00\x00\x01" + nal
        if not types.intersection((1, 5)):
            return
        key = 5 in types
        if key:
            if not all(kind in self.headers for kind in (7, 8)):
                self.units.reset()
                return
            payload = (
                b"".join(self.headers[k] for k in (7, 8) if k not in types) + payload
            )
        self.units.push((payload, key, stamp, sequence))


def build_pipeline(dai, settings, spec):
    _, width, height, fps, bitrate = parameters(settings, spec)
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.ColorCamera)
    camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    resolution = dai.ColorCameraProperties.SensorResolution
    if width == 3840:
        camera.setResolution(resolution.THE_4_K)
    elif width == 2016:
        # Encoder width must be a multiple of 32; video crops eight sensor columns.
        camera.setResolution(resolution.THE_2024X1520)
    else:
        camera.setResolution(resolution.THE_1080_P)
    if width == 1280:
        camera.setIspScale(2, 3)
    camera.setVideoSize(width, height)
    camera.setFps(fps)
    camera.initialControl.setAutoExposureLimit(int(1_000_000 / fps))
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(
        fps, dai.VideoEncoderProperties.Profile.H264_BASELINE
    )
    encoder.setBitrate(bitrate)
    encoder.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.CBR)
    encoder.setNumBFrames(0)
    encoder.setKeyframeFrequency(int(fps / 2))
    camera.video.link(encoder.input)
    output = pipeline.create(dai.node.XLinkOut)
    output.setStreamName("encoded")
    encoder.bitstream.link(output.input)
    return pipeline


class OakCapture:
    def __init__(self, settings, spec):
        parameters(settings, spec)
        self.settings, self.spec = settings, spec
        self.stream = H264AccessUnits()
        self.units = self.stream.units
        self.force_key = threading.Event()
        self.count = self.bytes = 0
        self.device = None
        self.last_frame = time.monotonic()
        self.warned = False

    def start(self):
        try:
            import depthai as dai
        except ImportError as exc:
            raise RuntimeError(
                "Install remote_assistance[oak] on the Robot Site"
            ) from exc
        if dai.__version__ != "2.31.0.0":
            raise RuntimeError("OAK backend requires depthai==2.31.0.0")
        pipeline = build_pipeline(dai, self.settings, self.spec)
        if self.spec.get("mxid"):
            self.device = dai.Device(pipeline, dai.DeviceInfo(self.spec["mxid"]))
        else:
            self.device = dai.Device(pipeline)
        self.queue = self.device.getOutputQueue("encoded", maxSize=4, blocking=False)
        self.clock_offset = time.monotonic() - dai.Clock.now().total_seconds()
        self.last_frame = time.monotonic()
        logger.info("OAK %s, USB %s", self.device.getMxId(), self.device.getUsbSpeed())
        logger.warning(
            "Experimental OAK sender: fixed bitrate/fps; periodic keyframes every 0.5s"
        )

    def check(self):
        # Sequence gaps from the bounded SDK queue invalidate dependent frames.
        for _ in range(4):
            frame = self.queue.tryGet()
            if frame is None:
                break
            payload = frame.getData().tobytes()
            stamp = int(
                (frame.getTimestamp().total_seconds() + self.clock_offset) * 1e6
            )
            self.stream.push(payload, stamp, frame.getSequenceNum())
            self.count += 1
            self.bytes += len(payload)
            self.last_frame = time.monotonic()
        if time.monotonic() - self.last_frame > 10:
            raise RuntimeError("No OAK encoded frames for 10 seconds")

    def feedback(self, feedback):
        # v2 VideoEncoder has no runtime control input; periodic IDR is the fallback.
        if feedback.keyframe_requested or self.force_key.is_set():
            self.force_key.clear()
        if feedback.HasField("rate_control") and not self.warned:
            logger.warning(
                "OAK rate adaptation unavailable; ensure network capacity for configured bitrate"
            )
            self.warned = True

    def stop(self):
        if self.device is not None:
            self.device.close()
