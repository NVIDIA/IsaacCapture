# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""OpenCV implementation of the batched video API."""

from math import isfinite

import cv2

FPS_FALLBACK = 30.0


def get_video_meta(video_path: str) -> dict:
    """Return fps, width, height, num_frames without decoding any frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not isfinite(fps) or fps <= 0:
        fps = FPS_FALLBACK
    meta = {
        "fps": fps,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def read_video_frames_batched(
    video_path: str,
    batch_size: int,
    max_frames: int | None = None,
):
    """Generator that yields lists of (frame_idx, BGR_frame), batch_size at a time.

    Each yielded list has exactly batch_size entries, except possibly the last
    which may be smaller. Stops early when max_frames is reached.
    """
    cap = cv2.VideoCapture(video_path)
    idx = 0
    batch = []
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            batch.append((idx, frame))
            idx += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        cap.release()
