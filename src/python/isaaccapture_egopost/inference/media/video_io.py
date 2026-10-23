# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Shared video API for EgoHand and the standalone detector.

The release runtime uses restricted FFmpeg; the default runtime uses OpenCV.
Backend imports are lazy so importing this API never probes a video or GPU.
"""

from importlib import import_module
from pathlib import Path

from .runtime_profile import get_runtime_profile

FPS_FALLBACK = 30.0


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.mp4"))


def _backend():
    name = get_runtime_profile().video_backend
    return import_module(f"._{name}_video", __package__)


def get_video_meta(video_path: str) -> dict:
    return _backend().get_video_meta(video_path)


def read_video_frames_batched(
    video_path: str, batch_size: int, max_frames: int | None = None
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_frames is not None and max_frames <= 0:
        return
    yield from _backend().read_video_frames_batched(video_path, batch_size, max_frames)
