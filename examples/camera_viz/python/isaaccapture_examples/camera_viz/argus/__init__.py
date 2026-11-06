# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Argus camera source - pybind11 module re-exports.

Build the .so with ``argus/build.sh``. This source is Jetson-only and
requires libargus, EGL, and CUDA from JetPack.
"""

from __future__ import annotations

try:
    from ._camera_viz_argus import ArgusCamera, ArgusConfig, FrameView
except ImportError as e:
    raise ImportError(
        "camera_viz native Argus source not built. Run "
        "`examples/camera_viz/python/isaaccapture_examples/camera_viz/argus/build.sh` on the Jetson."
    ) from e

__all__ = ["ArgusCamera", "ArgusConfig", "FrameView"]
