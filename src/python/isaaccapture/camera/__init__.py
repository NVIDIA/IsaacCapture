# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera frames from the Isaac Capture camera plugin, without a copy.

    from isaaccapture.camera import CameraReader

    with CameraReader.open("head") as reader:
        for frames in reader:
            left = frames.image(0).array()  # (height, width, 4), read-only, the plugin's own buffer

One plugin process owns the cameras; any number of readers read its buffers in place. An
array keeps its frame set held, so it shows that frame for as long as it exists; a reader may
hold as many frame sets at a time as its window. The plugin ships in this package under
``plugins/camera`` and is started like every other plugin, through the plugin manager. Linux
only.
"""

from ._camera import (
    CameraDisconnected,
    CameraReader,
    CameraReaderError,
    CameraWindowExhausted,
    FrameSet,
    FrameTimeline,
    GroupDescription,
    GroupListing,
    ImageView,
    StreamDescription,
)

__all__ = [
    "CameraDisconnected",
    "CameraReader",
    "CameraReaderError",
    "CameraWindowExhausted",
    "FrameSet",
    "FrameTimeline",
    "GroupDescription",
    "GroupListing",
    "ImageView",
    "StreamDescription",
]
