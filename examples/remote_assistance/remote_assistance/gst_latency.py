# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional buffer-boundary timing without mapping or copying camera pixels."""

from collections import OrderedDict
import threading
import time

from isaac_media_transport.latency import get_recorder


class GstTiming:
    def __init__(self, settings, track, pipeline, gst):
        self.recorder = get_recorder(settings)
        self.enabled = self.recorder.enabled
        self.track = track
        self.pending = OrderedDict()
        self.lock = threading.Lock()
        if self.enabled:
            for element, pad, stage in (
                ("camera", "src", "capture_available"),
                ("encoder", "sink", "encoder_submit"),
                ("encoder", "src", "encoded_ready"),
            ):
                target = pipeline.get_by_name(element)
                target.get_static_pad(pad).add_probe(
                    gst.PadProbeType.BUFFER, self._probe, (stage, gst)
                )

    def _probe(self, _pad, info, context):
        stage, gst = context
        now = time.monotonic_ns()
        buffer = info.get_buffer()
        if buffer is not None and buffer.pts != gst.CLOCK_TIME_NONE:
            with self.lock:
                stamps = self.pending.setdefault(int(buffer.pts), {})
                stamps.setdefault(stage, now)
                while len(self.pending) > 256:
                    self.pending.popitem(last=False)
        return gst.PadProbeReturn.OK

    def complete(self, pts, source_timestamp_ns, access_unit_ns):
        if not self.enabled:
            return
        with self.lock:
            stamps = self.pending.pop(int(pts), {})
        for stage, stamp in stamps.items():
            self.recorder.record(stage, self.track, source_timestamp_ns, at_ns=stamp)
        self.recorder.record(
            "access_unit_ready", self.track, source_timestamp_ns, at_ns=access_unit_ns
        )
