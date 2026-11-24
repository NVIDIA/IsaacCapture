# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera frames from the Isaac Capture camera plugins, without a copy.

    from isaaccapture.camera import CameraReader

    with CameraReader.open("head") as reader:
        for frames in reader:
            left = frames.image(0).array()  # (height, width, 4), read-only, the plugin's own buffer

Each camera vendor has its own plugin, a process that owns its cameras; any number of readers
read its buffers in place. A reader finds the plugin that serves its camera. An array keeps
its frame set held, so it shows that frame for as long as it exists; a reader may hold as many
frame sets at a time as its window. A camera whose frames are on a GPU (``stream.on_gpu``)
is read on the device, also without a copy: ``cupy.asarray(frames.image(0).cuda_array())``,
or ``torch.as_tensor`` of the same. The synthetic camera plugin ships in this package under
``plugins/camera_synthetic`` and is started like every other plugin, through the plugin
manager. Linux only.
"""

from ._camera import (
    CameraDisconnected,
    CameraReader,
    CameraReaderError,
    CameraWindowExhausted,
    CudaArray,
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
    "CudaArray",
    "FrameSet",
    "FrameTimeline",
    "GroupDescription",
    "GroupListing",
    "ImageView",
    "StreamDescription",
]
