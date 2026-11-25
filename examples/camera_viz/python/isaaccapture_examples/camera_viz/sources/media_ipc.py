# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional media-transport CUDA IPC source for the existing GPU renderer."""

from ..pipeline import Frame, FrameSource, SourceSpec
from ._helpers import notify


class MediaIpcSource(FrameSource):
    def __init__(self, settings, camera):
        from isaac_media_transport.latency import (
            ViewerTiming,
            for_process,
            get_recorder,
        )
        from isaac_media_transport.gpu_ipc import IpcReceiver
        from isaac_media_transport.stereo import stereo_tracks

        settings = for_process({**settings, "decoded_handoff": "ipc"}, "viewer")
        self.latency_reporter = ViewerTiming(settings)
        self._timing = get_recorder(settings)
        self._spec = SourceSpec(
            camera["name"], int(camera["width"]), int(camera["height"])
        )
        self._stereo = bool(camera.get("stereo", False))
        tracks = stereo_tracks(camera) if self._stereo else camera["name"]
        self._tracks = tracks if self._stereo else (tracks,)
        self._receiver = IpcReceiver(
            settings,
            identity="camera-viz",
            publisher=settings.get("publisher"),
            track=tracks,
        )

    @property
    def spec(self):
        return self._spec

    def start(self):
        self._receiver.start()

    def stop(self):
        self._receiver.stop()
        self.latency_reporter.close()

    def report_status(self):
        self.latency_reporter.report(lambda text: notify("latency", text))

    def latest(self):
        from isaac_media_transport.transport import ReceivedPair

        sample = self._receiver.latest()
        if sample is None:
            return None
        if isinstance(sample, ReceivedPair) != self._stereo:
            raise ValueError(
                "IPC stereo layout does not match the camera configuration"
            )
        eyes = (sample.left, sample.right) if self._stereo else (sample,)
        for track, eye in zip(self._tracks, eyes):
            if eye.metadata:
                self._timing.record("viewer_take", track, eye.metadata.user_timestamp)
        for eye in eyes:
            f = eye.frame
            if f.memory != "cuda" or (f.width, f.height) != (
                self.spec.width,
                self.spec.height,
            ):
                raise ValueError(
                    "IPC frame must match the configured CUDA RGBA dimensions"
                )

        # IPC returns owned, completed GPU copies; retain the arrays through rendering.
        def submitted(begin_ns, end_ns):
            self.latency_reporter.observe(sample.received_ns, begin_ns, end_ns)
            for track, eye in zip(self._tracks, eyes):
                if eye.metadata:
                    stamp = eye.metadata.user_timestamp
                    self._timing.record(
                        "viewer_submit_begin", track, stamp, at_ns=begin_ns
                    )
                    self._timing.record("viewer_submit_end", track, stamp, at_ns=end_ns)

        for track, eye in zip(self._tracks, eyes):
            if eye.metadata:
                self._timing.record("viewer_ready", track, eye.metadata.user_timestamp)
        return Frame(
            on_submit=submitted
            if self._timing.enabled or self.latency_reporter.enabled
            else None,
            image=eyes[0].frame.data,
            image_right=eyes[1].frame.data if self._stereo else None,
            timestamp_ns=sample.paired_ns if self._stereo else sample.received_ns,
            source_id=self.spec.name,
        )
