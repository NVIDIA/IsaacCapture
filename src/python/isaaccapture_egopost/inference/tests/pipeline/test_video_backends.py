# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from inference.media import _ffmpeg_video as ff


def test_probe_uses_r_frame_rate_when_avg_is_invalid(monkeypatch):
    payload = (
        '{"streams":[{"codec_name":"mpeg4","width":64,"height":48,'
        '"avg_frame_rate":"0/0","r_frame_rate":"60/1"}]}'
    )
    monkeypatch.setattr(
        ff.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, payload, ""),
    )
    assert ff.probe_video_stream("x.mp4").fps == 60.0


def test_read_exact_times_out_on_stalled_pipe():
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb") as stream:
            with pytest.raises(TimeoutError):
                ff._read_exact(stream, 16, inactivity_timeout=0.1)
    finally:
        os.close(write_fd)


def test_read_exact_collects_full_payload():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"abcdef")
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as stream:
        assert ff._read_exact(stream, 6, inactivity_timeout=1) == b"abcdef"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize(
    ("rotation", "expected_shape"),
    [(0, (32, 64, 3)), (90, (64, 32, 3)), (180, (32, 64, 3)), (270, (64, 32, 3))],
)
def test_rotated_video_has_same_orientation_in_both_backends(
    tmp_path, rotation, expected_shape
):
    from inference.media import _opencv_video as cvv

    base = tmp_path / "base.mp4"
    path = tmp_path / "rot.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x32:rate=5:duration=1",
            "-c:v",
            "mpeg4",
            str(base),
        ],
        check=True,
    )
    # Display-matrix side data: OpenCV applies it by default, FFmpeg must too.
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            str(rotation),
            "-i",
            str(base),
            "-c",
            "copy",
            str(path),
        ],
        check=True,
    )
    cv_meta = cvv.get_video_meta(str(path))
    ff_meta = ff.get_video_meta(str(path))
    assert (cv_meta["width"], cv_meta["height"]) == (
        ff_meta["width"],
        ff_meta["height"],
    )
    _, cv_frame = next(cvv.read_video_frames_batched(str(path), 1))[0]
    _, ff_frame = next(ff.read_video_frames_batched(str(path), 1))[0]
    assert cv_frame.shape == ff_frame.shape == expected_shape
    # Same content, not just same shape: a flipped or mis-turned frame differs.
    assert np.abs(cv_frame.astype(int) - ff_frame.astype(int)).mean() < 2
