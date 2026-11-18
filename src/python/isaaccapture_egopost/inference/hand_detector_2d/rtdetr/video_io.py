# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Video discovery and batched frame decoding for the standalone bbox detector.

``find_videos``, ``get_video_meta`` and ``read_video_frames_batched`` are copies
of the identically-named helpers in ``inference/scripts/run_egohand_inference_pipeline.py``,
not imports of them. This folder is meant to be copied out and run on its own,
so it may not import from the rest of the repo, and the pipeline is left
untouched rather than made to depend inward on a detector folder.

That makes these a deliberate duplicate, with the usual cost: if the pipeline
changes how it batches or where ``max_frames`` cuts, change it here too. What
must not drift is the frame numbering. Indices are the decode order from the
start of the video, zero-based; the pipeline turns them into ``f"{idx:06d}"``
image IDs, and ``SavedBBoxDetector`` looks boxes up by that exact string. A
mismatch here does not raise -- it silently finds no boxes.

Depends on opencv only.
"""

from math import isfinite
from pathlib import Path

import cv2

FPS_FALLBACK = 30.0


def find_videos(input_path: Path) -> list[Path]:
    """A single MP4, or every MP4 directly under a folder, in sorted order."""
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.mp4"))


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
