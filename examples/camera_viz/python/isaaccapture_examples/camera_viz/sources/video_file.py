# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video-file replay source for camera_viz.

Plays a recording through the same FrameSource contract as a live
camera — preview / Televiz testing when no hardware is attached, and a
deterministic stand-in for a camera on the RTP sender side.

Decode is OpenCV's FFmpeg backend (any container / codec it reads),
same hot path as :mod:`sources.v4l2`: ``cap.read()`` → pinned host
staging → async H2D → GPU BGR→RGBA. Frames are paced against an
absolute monotonic schedule so long clips don't drift.

``stereo: true`` treats the file as side-by-side (a ZED recording's
natural layout) and splits each frame into per-eye halves on the GPU —
one source emits both eyes, so eye sync is frame-perfect. Like
SyntheticStereoSource, this doesn't expose per-eye streams, so stereo
replay is viewer-only (camera_streamer rejects it loudly).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ._helpers import PolledSource, alloc_pinned_host, notify

# Playback rate used when the container reports a missing / absurd FPS.
_FALLBACK_FPS = 30.0


def _probe(path: Path) -> tuple:
    """Return (width, height, fps) of ``path``. Raises ValueError when the
    file is missing or unreadable — a configuration error, not a device
    blip, so it fails at build time rather than retrying forever."""
    import cv2

    if not path.is_file():
        raise ValueError(f"video source: no such file: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(
                f"video source: cannot open {path} — unsupported container/"
                "codec for OpenCV's FFmpeg backend?"
            )
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if w <= 0 or h <= 0:
            # Some containers only report dimensions after a decode.
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"video source: {path} contains no decodable frames")
            h, w = frame.shape[:2]
    finally:
        cap.release()
    return w, h, fps


class VideoFileSource(PolledSource):
    """File playback as a FrameSource; mono or side-by-side stereo."""

    _kind = "video"

    def __init__(
        self,
        name: str,
        path: str,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        loop: bool = True,
        stereo: bool = False,
    ) -> None:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "VideoFileSource requires opencv-python. Install via "
                "`uv pip install opencv-python`."
            ) from e
        self._cv2 = cv2
        self._path = Path(path)

        native_w, native_h, native_fps = _probe(self._path)
        eyes = 2 if stereo else 1
        if stereo and native_w % 2:
            raise ValueError(
                f"video source '{name}': stereo needs an even frame width to "
                f"split side-by-side, got {native_w}"
            )

        # Output size: YAML width/height win; otherwise the file's native
        # size (per eye when stereo).
        out_w = int(width) if width else native_w // eyes
        out_h = int(height) if height else native_h

        play_fps = float(fps) if fps else native_fps
        if not math.isfinite(play_fps) or not 0.0 < play_fps <= 1000.0:
            notify(
                self._kind,
                f"'{name}': no usable FPS in {self._path.name} "
                f"({native_fps!r}); playing at {_FALLBACK_FPS:g}",
            )
            play_fps = _FALLBACK_FPS
        self._frame_interval_s = 1.0 / play_fps
        self._loop = bool(loop)
        self._stereo = bool(stereo)

        super().__init__(name=name, width=out_w, height=out_h, staging_channels=3)
        cp = self._cp

        # The decoded frame is the full (SBS when stereo) picture; staging
        # and the GPU BGR landing zone are sized to it. The base class's
        # per-eye staging is replaced — its (h, w, 3) allocation only fits
        # the mono layout.
        self._frame_w = out_w * eyes
        if stereo:
            self._host_staging = alloc_pinned_host((out_h, self._frame_w, 3), np.uint8)
            # Right-eye output buffers, rotated in lock-step with the base
            # class's (left-eye) triple-buffer.
            self._gpu_right = [
                cp.empty((out_h, out_w, 4), dtype=cp.uint8) for _ in range(3)
            ]
            for buf in self._gpu_right:
                buf[..., 3] = 255
        self._gpu_bgr = cp.empty((out_h, self._frame_w, 3), dtype=cp.uint8)

        self._cap = None
        self._finished = False
        self._deadline = 0.0

    def _open_device(self) -> bool:
        cap = self._cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        self._finished = False
        self._deadline = time.monotonic()
        return True

    def _close_device(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _grab(self) -> Optional[np.ndarray]:
        if self._finished:
            # loop: false and the clip has ended — hold the last published
            # frame. Idle here (not None-spin) until stop().
            self._stop.wait(timeout=0.25)
            return None

        wait = self._deadline - time.monotonic()
        if wait > 0 and self._stop.wait(timeout=wait):
            return None

        cv2 = self._cv2
        ret, frame = self._cap.read()
        if not ret or frame is None:
            if not self._loop:
                self._finished = True
                notify(self._kind, "playback finished (holding last frame)")
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Rewind failed — reopen the file via the reconnect path.
                raise RuntimeError("rewind failed")

        self._deadline += self._frame_interval_s
        now = time.monotonic()
        if self._deadline < now:
            # Fell behind (slow decode, stall) — resync rather than burst.
            self._deadline = now + self._frame_interval_s

        h, fw = self._host_staging.shape[:2]
        if frame.shape[0] != h or frame.shape[1] != fw:
            frame = cv2.resize(frame, (fw, h), interpolation=cv2.INTER_LINEAR)
        self._host_staging[...] = frame
        return self._host_staging

    def _upload_and_convert(self, gpu_buf) -> None:
        # Async H2D + GPU channel reverse (BGR → RGB); alpha stays 255 from
        # construction. Stereo slices the SBS landing zone into eyes.
        with self._stream:
            self._gpu_bgr.set(self._host_staging)
            if not self._stereo:
                gpu_buf[..., :3] = self._gpu_bgr[..., ::-1]
                return
            w = self._spec.width
            gpu_buf[..., :3] = self._gpu_bgr[:, :w, ::-1]
            self._gpu_right[self._write_idx][..., :3] = self._gpu_bgr[:, w:, ::-1]

    def latest(self):
        frame = super().latest()
        if frame is not None and self._stereo:
            # _consumed_idx is the slot latest() just returned; only the
            # consumer thread reads it, so no extra locking needed.
            frame.image_right = self._gpu_right[self._consumed_idx]
        return frame
