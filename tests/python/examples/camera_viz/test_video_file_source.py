# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the video-file replay source (``type: video``).

Clips are synthesized in tmp_path with cv2.VideoWriter — no binary
fixtures in the repo. Probe / path-resolution tests run everywhere;
playback tests need cupy + a CUDA device and skip cleanly without one.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from isaaccapture_examples.camera_viz.sources import (  # noqa: E402
    build_local_camera,
    resolve_video_paths,
)


def _cuda_available() -> bool:
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _cuda_available(), reason="needs cupy + a CUDA device"
)


def _write_clip(path, frames, fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("cv2.VideoWriter cannot encode mp4v here")
    for f in frames:
        vw.write(f)
    vw.release()


def _solid(h: int, w: int, bgr) -> np.ndarray:
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[...] = bgr
    return frame


def _collect(source, n: int, timeout_s: float = 10.0) -> list:
    """Poll latest() until ``n`` frames arrive or the timeout hits."""
    frames = []
    deadline = time.monotonic() + timeout_s
    while len(frames) < n and time.monotonic() < deadline:
        f = source.latest()
        if f is not None:
            frames.append(f)
        time.sleep(0.002)
    return frames


# ── build-time behavior (no GPU needed) ─────────────────────────────


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        build_local_camera(
            {"name": "x", "type": "video", "path": str(tmp_path / "nope.mp4")}
        )


def test_resolve_video_paths_anchors_to_yaml_dir(tmp_path):
    # An already-absolute path must be left untouched. Build it from
    # tmp_path so it's absolute on any OS (a POSIX "/abs" string is not
    # absolute on Windows and would get re-anchored to the drive).
    abs_path = str(tmp_path / "elsewhere" / "clip.mp4")
    cfg = {
        "cameras": [
            {"type": "video", "path": "clip.mp4"},
            {"type": "video", "path": abs_path},
            {"type": "v4l2", "device": "/dev/video0"},
        ]
    }
    resolve_video_paths(cfg, tmp_path)
    assert cfg["cameras"][0]["path"] == str(tmp_path / "clip.mp4")
    assert cfg["cameras"][1]["path"] == abs_path
    assert "path" not in cfg["cameras"][2]


# ── playback (GPU) ───────────────────────────────────────────────────


@requires_gpu
def test_probe_size_playback_and_color(tmp_path):
    """Omitting width/height sizes the source from the file; frames come
    out RGBA with BGR→RGB applied."""
    import cupy as cp

    clip = tmp_path / "blue.mp4"
    _write_clip(clip, [_solid(48, 64, (255, 0, 0))] * 10, fps=60.0)  # BGR blue

    (source,) = build_local_camera(
        {"name": "replay", "type": "video", "path": str(clip)}
    )
    assert source.spec.width == 64
    assert source.spec.height == 48

    with source:
        frames = _collect(source, 3)
    assert len(frames) == 3
    img = cp.asnumpy(frames[-1].image)
    assert img.shape == (48, 64, 4)
    assert img[..., 2].mean() > 200  # blue landed in the R..B order's B slot
    assert img[..., 0].mean() < 60  # red channel stays dark
    assert (img[..., 3] == 255).all()


@requires_gpu
def test_loop_wraps_past_end_of_file(tmp_path):
    clip = tmp_path / "short.mp4"
    _write_clip(clip, [_solid(32, 32, (0, 255, 0))] * 6, fps=60.0)

    (source,) = build_local_camera(
        {"name": "looper", "type": "video", "path": str(clip), "fps": 60}
    )
    with source:
        # 6-frame clip: anything past 6 proves the rewind worked.
        frames = _collect(source, 20)
    assert len(frames) == 20


@requires_gpu
def test_no_loop_holds_after_last_frame(tmp_path):
    clip = tmp_path / "once.mp4"
    _write_clip(clip, [_solid(32, 32, (0, 0, 255))] * 5, fps=100.0)

    (source,) = build_local_camera(
        {
            "name": "oneshot",
            "type": "video",
            "path": str(clip),
            "loop": False,
            "fps": 100,
        }
    )
    with source:
        frames = _collect(source, 5, timeout_s=5.0)
        assert 1 <= len(frames) <= 5
        # Clip is done — no new frames may appear, and the source must
        # stay alive (holding) rather than reconnect-spinning.
        time.sleep(0.3)
        assert source.latest() is None


@requires_gpu
def test_stereo_sbs_splits_eyes(tmp_path):
    """SBS file → per-eye halves: left green, right red, both eyes in
    one Frame."""
    import cupy as cp

    h, eye_w = 32, 48
    sbs = np.empty((h, eye_w * 2, 3), dtype=np.uint8)
    sbs[:, :eye_w] = (0, 255, 0)  # BGR green left
    sbs[:, eye_w:] = (0, 0, 255)  # BGR red right
    clip = tmp_path / "sbs.mp4"
    _write_clip(clip, [sbs] * 10, fps=60.0)

    (source,) = build_local_camera(
        {"name": "sbs", "type": "video", "path": str(clip), "stereo": True}
    )
    assert source.spec.width == eye_w
    assert source.spec.height == h

    with source:
        frames = _collect(source, 3)
    assert len(frames) == 3
    left = cp.asnumpy(frames[-1].image)
    right = cp.asnumpy(frames[-1].image_right)
    assert left.shape == right.shape == (h, eye_w, 4)
    assert left[..., 1].mean() > 200  # green
    assert right[..., 0].mean() > 200  # red
    assert (right[..., 3] == 255).all()


@requires_gpu
def test_resize_override(tmp_path):
    clip = tmp_path / "big.mp4"
    _write_clip(clip, [_solid(48, 64, (128, 128, 128))] * 5, fps=60.0)

    (source,) = build_local_camera(
        {
            "name": "small",
            "type": "video",
            "path": str(clip),
            "width": 32,
            "height": 24,
        }
    )
    assert source.spec.width == 32
    assert source.spec.height == 24
    with source:
        frames = _collect(source, 2)
    assert len(frames) == 2
    assert frames[-1].image.shape == (24, 32, 4)
