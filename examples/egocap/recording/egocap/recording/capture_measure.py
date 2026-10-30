# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure numeric measurements used by the capture-quality checks.

Each function reads arrays and returns numbers (or a problem description); none opens files or decides what a
result means. The decisions live in :mod:`capture_qc`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "average_fps",
    "longest_still_seconds",
    "pico_body_problem",
    "stream_liveness",
    "total_frame_drops",
    "windowed_frame_drops",
]


def average_fps(timestamps: np.ndarray) -> float:
    """Average rate over the recording, ``(frames - 1) / duration``; fewer than two frames counts as 0 fps."""
    if len(timestamps) < 2 or timestamps[-1] - timestamps[0] <= 0:
        return 0.0
    return float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))


def longest_still_seconds(timestamps: np.ndarray, signature: np.ndarray, eps: float) -> float:
    """Longest window over which consecutive-frame max-abs change stays at or below ``eps``."""
    n = min(len(timestamps), len(signature))
    if n < 2:
        return 0.0
    ts = np.asarray(timestamps[:n], dtype=np.float64)
    sig = np.asarray(signature[:n], dtype=np.float64).reshape(n, -1)
    still = np.max(np.abs(np.diff(sig, axis=0)), axis=1) <= eps
    longest, start = 0.0, None
    for i, is_still in enumerate(still):
        if is_still:
            start = i if start is None else start
        elif start is not None:
            longest = max(longest, ts[i] - ts[start])
            start = None
    if start is not None:
        longest = max(longest, ts[n - 1] - ts[start])
    return float(longest)


def stream_liveness(arr: np.ndarray, std_eps: float, frac_identical: float) -> dict:
    """Whether one stream is frozen for the whole episode (the native dead-hand measurement).

    Frozen means the mean per-element standard deviation over time is below ``std_eps``, or at least
    ``frac_identical`` of the frames equal the first frame. A frozen ``(*, 4, 4)`` stream starting at the identity
    matrix is the canonical sensor dropout (``at_identity``). Fewer than two frames cannot be judged: not frozen.
    """
    frames = arr.shape[0] if arr.ndim >= 1 else 0
    if frames < 2:
        return {"frozen": False, "at_identity": False, "std_mean": 0.0, "frac_identical": 0.0, "frames": frames}
    flat = np.nan_to_num(arr.reshape(frames, -1).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    std_mean = float(flat.std(axis=0).mean())
    frac = float(np.mean(np.all(flat == flat[0:1], axis=1)))
    frozen = bool(std_mean < std_eps or frac >= frac_identical)
    at_identity = bool(frozen and arr.shape[-2:] == (4, 4) and np.allclose(arr.reshape(-1, 4, 4)[0], np.eye(4)))
    return {
        "frozen": frozen,
        "at_identity": at_identity,
        "std_mean": std_mean,
        "frac_identical": round(frac, 4),
        "frames": frames,
    }


def windowed_frame_drops(timestamps: np.ndarray, expected_fps: float, window_s: float) -> int:
    """The most frames dropped within any rolling ``window_s`` window.

    Each gap longer than one expected period counts ``round(gap / period) - 1`` missed frames, so occasional
    hiccups age out and only a real burst or stall is large.
    """
    if len(timestamps) < 2 or expected_fps <= 0:
        return 0
    period = 1.0 / expected_fps
    missed = np.maximum(np.round(np.diff(timestamps) / period).astype(int) - 1, 0)
    ends = np.asarray(timestamps[1:], dtype=np.float64)  # when each gap is observed
    worst = running = left = 0
    for right in range(len(ends)):
        running += int(missed[right])
        while ends[right] - ends[left] > window_s:
            running -= int(missed[left])
            left += 1
        worst = max(worst, running)
    return worst


def total_frame_drops(timestamps: np.ndarray, expected_fps: float) -> int:
    """Lifetime count of dropped frames over the whole recording (measured and reported, never gating)."""
    if len(timestamps) < 2 or expected_fps <= 0:
        return 0
    period = 1.0 / expected_fps
    return int(
        np.maximum(np.round(np.diff(np.asarray(timestamps, dtype=np.float64)) / period).astype(int) - 1, 0).sum()
    )


def pico_body_problem(
    poses: np.ndarray, valid: np.ndarray | None, max_invalid: int, max_identity: int, identity_atol: float
) -> str | None:
    """The first problem with Pico full-body tracking, or None. ``poses`` is ``(T, 24, 4, 4)``."""
    if poses.ndim != 4 or poses.shape[1:] != (24, 4, 4):
        return f"expected shape (T, 24, 4, 4), got {tuple(poses.shape)}"
    if poses.shape[0] == 0:
        return "no body-joint frames recorded"
    if not np.isfinite(poses).all():
        return "body-joint poses contain non-finite values"
    near_identity = np.isclose(poses, np.eye(4, dtype=poses.dtype), atol=identity_atol, rtol=0.0)
    identity_frames = int(np.all(near_identity, axis=(1, 2, 3)).sum())
    if valid is not None:
        if valid.shape[0] != poses.shape[0]:
            return f"validity length {valid.shape[0]} does not match {poses.shape[0]} body-joint frames"
        invalid = int((~valid.astype(bool)).sum())
        if invalid > max_invalid:
            return f"{invalid} invalid body-joint frame(s) > allowed {max_invalid}"
    if identity_frames > max_identity:
        return f"{identity_frames} all-identity body-joint frame(s) > allowed {max_identity}"
    return None
