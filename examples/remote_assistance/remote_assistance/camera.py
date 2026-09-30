# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-side CPU camera capture and LiveKit video publication."""

import asyncio
import logging
import threading
import time

import numpy as np
from livekit import rtc

from .connection import connect, disconnect

logger = logging.getLogger("isaaccapture.remote_assistance.camera")


class Capture:
    """Keep only the latest captured RGBA image; no GPU or XR dependencies."""

    def __init__(self, spec):
        self.spec = spec
        self.error = None
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="camera-capture", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(3)
        if self._thread.is_alive():
            # Never release a VideoCapture while its native read() is still running.
            logger.warning(
                "Camera read is blocked; its daemon thread retains the device"
            )

    def latest(self):
        if self.error:
            raise RuntimeError("Camera capture failed") from self.error
        with self._lock:
            result, self._latest = self._latest, None
        return result

    def _run(self):
        cap = None
        try:
            width, height = int(self.spec["width"]), int(self.spec["height"])
            fps = float(self.spec["fps"])
            if min(width, height, fps) <= 0:
                raise ValueError("Camera dimensions and fps must be positive")
            kind = self.spec["type"]
            if kind == "v4l2":
                import cv2

                cap = cv2.VideoCapture(
                    self.spec.get("device", "/dev/video0"), cv2.CAP_V4L2
                )
                if not cap.isOpened():
                    raise RuntimeError(
                        "Could not open camera; check device access/ownership"
                    )
                fourcc = self.spec.get("fourcc", "MJPG")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                actual = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                if actual != (width, height):
                    raise ValueError(
                        f"Requested {width}x{height}, camera negotiated {actual}"
                    )
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                if abs(actual_fps - fps) > 0.5:
                    raise ValueError(
                        f"Requested {fps} fps, camera negotiated {actual_fps}"
                    )
                logger.info(
                    "V4L2 negotiated %sx%s at %.2f fps (%s)",
                    width,
                    height,
                    actual_fps,
                    fourcc,
                )
            elif kind != "synthetic":
                raise ValueError(
                    "LiveKit camera sender currently supports v4l2 or synthetic"
                )
            index = 0
            deadline = time.monotonic()
            while not self._stop.is_set():
                if cap is not None:
                    ok, bgr = cap.read()
                    captured_ns = time.monotonic_ns()
                    if not ok:
                        raise RuntimeError("V4L2 read failed")
                    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                else:
                    self._stop.wait(max(0, deadline - time.monotonic()))
                    captured_ns = time.monotonic_ns()
                    rgba = np.zeros((height, width, 4), dtype=np.uint8)
                    rgba[..., 3] = 255
                    rgba[..., 0] = 220 if (index // max(1, round(fps))) % 2 else 30
                    x = (index * 8) % width
                    rgba[:, x : min(x + 32, width), 1:3] = 220
                    deadline = max(deadline + 1 / fps, time.monotonic())
                with self._lock:
                    self._latest = (rgba, captured_ns, index)
                index += 1
        except Exception as exc:
            self.error = exc
            logger.exception("Capture stopped")
        finally:
            if cap is not None:
                cap.release()


async def publish_camera(settings, spec, stop):
    if spec.get("stereo", False):
        raise ValueError("LiveKit camera prototype requires a mono camera")
    room = rtc.Room()
    capture = Capture(spec)
    source = rtc.VideoSource(int(spec["width"]), int(spec["height"]))
    track = rtc.LocalVideoTrack.create_video_track(spec["name"], source)
    codec = str(settings.get("codec", "h264")).upper()
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=False,
        video_codec=rtc.VideoCodec.Value(codec),
        video_encoding=rtc.VideoEncoding(
            max_framerate=float(spec["fps"]),
            max_bitrate=int(float(settings.get("bitrate_mbps", 15)) * 1_000_000),
        ),
        frame_metadata_features=[
            rtc.FrameMetadataFeature.FMF_FRAME_ID,
            rtc.FrameMetadataFeature.FMF_USER_TIMESTAMP,
        ],
    )
    backend = "ENCODER_BACKEND_" + str(settings.get("video_encoder", "auto")).upper()
    options.video_encoder = (
        options.DESCRIPTOR.fields_by_name["video_encoder"]
        .enum_type.values_by_name[backend]
        .number
    )
    room.on("disconnected", lambda *_: stop.set())
    started = False
    try:
        await connect(room, settings, settings.get("publisher", "robot-camera"))
        await room.local_participant.publish_track(track, options)
        capture.start()
        started = True
        count = 0
        last_report = time.monotonic()
        logger.info("Publishing %s using %s, encoder=%s", spec["name"], codec, backend)
        while not stop.is_set():
            frame = capture.latest()
            if frame is None:
                await asyncio.sleep(0.001)
                continue
            rgba, captured_ns, index = frame
            source.capture_frame(
                rtc.VideoFrame(
                    rgba.shape[1],
                    rgba.shape[0],
                    rtc.VideoBufferType.RGBA,
                    memoryview(rgba),
                ),
                timestamp_us=captured_ns // 1000,
                metadata=rtc.FrameMetadata(
                    frame_id=index % (2**32), user_timestamp=captured_ns
                ),
            )
            count += 1
            now = time.monotonic()
            if now - last_report >= 5:
                logger.info(
                    "Submitted %.1f fps to LiveKit encoder", count / (now - last_report)
                )
                count = 0
                last_report = now
            await asyncio.sleep(0)
    finally:
        if started:
            await asyncio.to_thread(capture.stop)
        await disconnect(room)
        await source.aclose()
