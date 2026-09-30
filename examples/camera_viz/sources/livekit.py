# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LiveKit video ingestion into the existing Televiz FrameSource interface."""

import logging
import time

import numpy as np

from pipeline import Frame, FrameSource, SourceSpec

logger = logging.getLogger("isaaccapture.camera_viz.livekit")


class LiveKitSource(FrameSource):
    def __init__(self, cam, settings):
        try:
            import cupy as cp
            from remote_assistance.video import VideoReceiver
        except ImportError as exc:
            raise RuntimeError(
                "Install examples/remote_assistance into the camera_viz environment; see its README"
            ) from exc
        if cam.get("stereo", False):
            raise ValueError("LiveKit source currently supports mono cameras")
        self._cp = cp
        self._spec = SourceSpec(cam["name"], int(cam["width"]), int(cam["height"]))
        self._receiver = VideoReceiver(
            settings,
            identity=f"{settings.get('viewer_identity', 'edge-video')}-{cam['name']}",
            publisher=settings.get("publisher", "robot-camera"),
            track=cam["name"],
        )
        self._stale_ns = int(float(settings.get("video_stale_ms", 1000)) * 1e6)
        if self._stale_ns <= 0:
            raise ValueError("video_stale_ms must be positive")
        self._last_frame_ns = 0
        self._blanked = False
        self._warned_sizes = set()

    @property
    def spec(self):
        return self._spec

    def start(self):
        self._receiver.start()

    def stop(self):
        self._receiver.stop()

    def latest(self):
        received = self._receiver.latest()
        now = time.monotonic_ns()
        if received is None or now - received.received_ns > self._stale_ns:
            if self._blanked or now - self._last_frame_ns <= self._stale_ns:
                return None
            image = self._cp.zeros(
                (self.spec.height, self.spec.width, 4), dtype=self._cp.uint8
            )
            image[..., 3] = 255
            self._blanked = True
        else:
            frame = received.frame
            rgba = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 4
            )
            if (frame.width, frame.height) != (self.spec.width, self.spec.height):
                import cv2

                size = (frame.width, frame.height)
                if size not in self._warned_sizes:
                    logger.warning(
                        "LiveKit delivered %s; resizing to configured %sx%s",
                        size,
                        self.spec.width,
                        self.spec.height,
                    )
                    self._warned_sizes.add(size)
                rgba = cv2.resize(rgba, (self.spec.width, self.spec.height))
            image = self._cp.asarray(rgba)
            self._last_frame_ns = received.received_ns
            self._blanked = False
        # Consumer owns this allocation until submit completes; no producer can overwrite it.
        self._cp.cuda.get_current_stream().synchronize()
        return Frame(image=image, timestamp_ns=now, source_id=self.spec.name)
