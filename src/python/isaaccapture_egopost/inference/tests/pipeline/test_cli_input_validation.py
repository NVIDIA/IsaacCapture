# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Bad inputs fail early with one line, not a traceback after the model loads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.scripts import run_egohand_inference_pipeline as script


def _argv(monkeypatch, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_egohand_inference_pipeline.py",
            "--input",
            "in.mp4",
            "--output_dir",
            "out",
            *extra,
        ],
    )


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_max_seconds_must_be_positive(monkeypatch, capsys, value):
    _argv(monkeypatch, "--max_seconds", value)
    with pytest.raises(SystemExit) as exc:
        script.parse_args()
    assert exc.value.code == 2
    assert "--max_seconds must be positive" in capsys.readouterr().err


def test_missing_input_is_not_reported_as_empty_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_egohand_inference_pipeline.py",
            "--input",
            str(tmp_path / "absent.mp4"),
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit, match="Input not found"):
        script.main()


def test_unwritable_output_dir_fails_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(script.os, "access", lambda *_: False)
    with pytest.raises(SystemExit, match="not writable"):
        script.prepare_output_dir(tmp_path / "out")


def test_output_dir_is_created(tmp_path):
    script.prepare_output_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()


def test_frame_batching_conflicts_with_whole_clip_detector():
    with pytest.raises(SystemExit, match="--frame_batch_size cannot be used"):
        script.check_frame_batching(SimpleNamespace(requires_full_clip=True), 64)
    script.check_frame_batching(SimpleNamespace(requires_full_clip=True), None)
    script.check_frame_batching(SimpleNamespace(requires_full_clip=False), 64)


def test_malformed_detector_yaml(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("detector_backend: [saved_bboxes\n")
    with pytest.raises(script.DetectorConfigError, match="Invalid YAML"):
        script._load_detector_config(str(cfg))


def test_missing_detector_config_file(tmp_path):
    with pytest.raises(script.DetectorConfigError, match="Cannot read"):
        script._load_detector_config(str(tmp_path / "none.yaml"))


def test_unknown_detector_backend_has_no_chained_traceback():
    with pytest.raises(script.DetectorConfigError, match="'nope'") as exc:
        script._build_detector_from_config({"detector_backend": "nope"}, "cpu", 1)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_missing_gpu_is_reported_before_anything_else(monkeypatch):
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="--gpus all"):
        script.check_gpu_visible("cuda")
    script.check_gpu_visible("cpu")


@pytest.mark.parametrize(
    "error",
    [
        script.VIDEO_INPUT_ERRORS[0]("Unsupported video codec 'mjpeg'"),
        script.VIDEO_INPUT_ERRORS[1]("ffprobe failed for /in/a.mp4: moov atom"),
    ],
)
def test_unreadable_video_is_one_line_naming_the_file(monkeypatch, error):
    def fail(_path):
        raise error

    monkeypatch.setattr(script, "check_video_readable", fail)
    with pytest.raises(SystemExit) as exc:
        script.check_videos_readable([Path("/in/a.mp4")])
    assert str(exc.value).count("/in/a.mp4") == 1
    assert str(error) in str(exc.value)


def test_unsupported_codec_is_rejected_by_the_ffmpeg_profile(monkeypatch):
    from inference.media import video_io

    monkeypatch.setenv("EGOHAND_RUNTIME_PROFILE", "release")
    stream = SimpleNamespace(codec_name="mjpeg")
    monkeypatch.setattr(
        "inference.media._ffmpeg_video.probe_video_stream", lambda _p: stream
    )
    with pytest.raises(video_io.UnsupportedVideoCodecError):
        video_io.check_video_readable("clip.mp4")
