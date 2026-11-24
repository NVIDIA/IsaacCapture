# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""An RGB8 sample reaches the store as contiguous uint8 BGR without touching the sample's own array."""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from recording_service import PixelFormat, VideoSample
from recording_service.recording_service import RecordingService


class _Store:
    def __init__(self):
        self.frames = []

    def append_video(self, source, frame, capture_time, receive_time, intrinsics):
        self.frames.append(frame)


def test_read_only_rgb_sample_is_written_as_contiguous_bgr_and_left_unchanged():
    raw = bytes(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18,
        ]
    )  # fmt: skip
    image = np.frombuffer(raw, np.uint8).reshape(2, 3, 3)
    expected = np.array(
        [
            [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
            [[12, 11, 10], [15, 14, 13], [18, 17, 16]],
        ],
        np.uint8,
    )
    take = SimpleNamespace(store=_Store(), written=defaultdict(int), span={})

    RecordingService._write(take, VideoSample("cam", image, PixelFormat.RGB8, 1.0, 1.0, None))

    (out,) = take.store.frames
    assert np.array_equal(out, expected)
    assert out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert image.tobytes() == raw
