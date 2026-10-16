# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side desktop preview of an existing stereo source.

The adapter delegates capture start and stop to its wrapped FrameSource.
It copies left/right GPU RGBA8 images into a reusable image of twice the width,
placing the left eye first and preserving the original pair timestamp.
Copying follows the producer stream and completes before publication.

The viewer enables this adapter only in window mode; XR uses separate eye images.
The wrapped source remains responsible for device capture, pairing, and decoding.
"""

from __future__ import annotations

from pipeline import Frame, FrameSource, SourceSpec


class StereoSideBySideSource(FrameSource):
    """Expose a stereo pair as one RGBA image, with left then right."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        spec = source.spec
        self._spec = SourceSpec(
            f"{spec.name}.sbs", spec.width * 2, spec.height, spec.pixel_format
        )
        self._buffers = None
        self._index = 0

    @property
    def spec(self) -> SourceSpec:
        return self._spec

    def start(self) -> None:
        self._source.start()

    def stop(self) -> None:
        self._source.stop()

    def latest(self) -> Frame | None:
        frame = self._source.latest()
        if frame is None:
            return None
        if frame.image_right is None:
            raise RuntimeError("Stereo SBS debug mode requires a paired source")
        import cupy as cp

        width = self._source.spec.width
        shape = (self.spec.height, width, 4)
        if frame.image.shape != shape or frame.image_right.shape != shape:
            raise ValueError(
                "stereo debug requires matching left/right RGBA dimensions"
            )
        with cp.cuda.Device(frame.image.device.id):
            # Copy on the producer stream; preserve the Frame stream contract.
            stream = (
                cp.cuda.ExternalStream(frame.stream)
                if frame.stream
                else cp.cuda.Stream.null
            )
            with stream:
                if self._buffers is None:
                    self._buffers = [
                        cp.empty((self.spec.height, self.spec.width, 4), dtype=cp.uint8)
                        for _ in range(3)
                    ]
                image = self._buffers[self._index]
                image[:, :width] = frame.image
                image[:, width:] = frame.image_right
            stream.synchronize()
        self._index = (self._index + 1) % len(self._buffers)
        return Frame(image, frame.timestamp_ns, self.spec.name, stream=frame.stream)
