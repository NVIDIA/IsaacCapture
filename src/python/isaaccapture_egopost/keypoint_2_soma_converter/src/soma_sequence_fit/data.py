# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Strict input contracts and deterministic frame indexing."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np


@dataclass
class InputData:
    points: np.ndarray
    confidence: np.ndarray
    K: np.ndarray
    timestamps: np.ndarray
    width: int
    height: int
    fps: float
    video: Path
    points_file: Path

    @property
    def nframes(self):
        return len(self.points)


def load_inputs(video, points_file):
    video, points_file = Path(video), Path(points_file)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
        raise ValueError("Invalid video metadata.")
    with np.load(points_file, allow_pickle=False) as d:
        for key in ("points3d", "K", "coordinate_system", "units"):
            if key not in d:
                raise ValueError(f"Missing NPZ array: {key}")
        if str(d["coordinate_system"].item()) != "opencv_camera":
            raise ValueError(
                "Convert points to opencv_camera coordinates: x right, y down, z forward."
            )
        units = str(d["units"].item())
        if units not in ("m", "cm", "mm"):
            raise ValueError("units must be m, cm, or mm.")
        p = (
            np.array(d["points3d"], dtype=np.float32)
            * {"m": 1, "cm": 0.01, "mm": 0.001}[units]
        )
        if p.ndim != 4 or p.shape[1:] != (2, 21, 3) or len(p) < 1:
            raise ValueError("points3d must have shape (T, 2, 21, 3), left then right.")
        c = (
            np.array(d["confidence"], dtype=np.float32)
            if "confidence" in d
            else np.ones(p.shape[:-1], np.float32)
        )
        if (
            c.shape != p.shape[:-1]
            or not np.isfinite(c).all()
            or np.any((c < 0) | (c > 1))
        ):
            raise ValueError("confidence must be finite (T,2,21) in [0,1].")
        c[~np.isfinite(p).all(-1) | (p[..., 2] <= 0)] = 0
        p[c <= 0] = np.nan
        K = np.array(d["K"], dtype=np.float32)
        if K.shape == (3, 3):
            K = np.broadcast_to(K, (len(p), 3, 3)).copy()
        if K.shape != (len(p), 3, 3) or not np.isfinite(K).all():
            raise ValueError("K must be finite (3,3) or (T,3,3).")
        if (
            np.any(K[:, 0, 0] <= 0)
            or np.any(K[:, 1, 1] <= 0)
            or not np.allclose(K[:, 2], [0, 0, 1])
        ):
            raise ValueError("Invalid pinhole intrinsics.")
        ts = (
            np.array(d["timestamps"], dtype=np.float64)
            if "timestamps" in d
            else np.arange(len(p)) / fps
        )
        if (
            ts.shape != (len(p),)
            or not np.isfinite(ts).all()
            or np.any(np.diff(ts) <= 0)
        ):
            raise ValueError("timestamps must be finite and strictly increasing.")
        if "distortion" in d and np.any(np.abs(d["distortion"]) > 1e-12):
            raise ValueError(
                "Undistort video and update K first; nonzero distortion is not accepted."
            )
    return InputData(p, c, K, ts, width, height, fps, video, points_file)


def video_frames(data):
    """Decode sequentially and reject silent frame count mismatches."""
    cap = cv2.VideoCapture(str(data.video))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n >= data.nframes:
                raise ValueError("Video has more decoded frames than points3d.")
            yield n, frame
            n += 1
    finally:
        cap.release()
    if n != data.nframes:
        raise ValueError(f"Decoded {n} frames, but points3d has {data.nframes} frames.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
