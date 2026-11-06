# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Bad inputs fail early with one line, not a traceback after the model loads."""

from __future__ import annotations

import sys
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


def test_missing_input_is_not_reported_as_empty_folder(monkeypatch, tmp_path, capsys):
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
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == script.EXIT_INVALID_INPUT
    assert "Input not found" in capsys.readouterr().err


def test_unwritable_output_dir_fails_before_inference(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script.os, "access", lambda *_: False)
    with pytest.raises(SystemExit) as exc:
        script.prepare_output_dir(tmp_path / "out")
    assert exc.value.code == script.EXIT_INVALID_INPUT
    assert "not writable" in capsys.readouterr().err


def test_output_dir_is_created(tmp_path):
    script.prepare_output_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()


def test_frame_batching_conflicts_with_whole_clip_detector():
    with pytest.raises(SystemExit) as exc:
        script.check_frame_batching(SimpleNamespace(requires_full_clip=True), 64)
    assert exc.value.code == script.EXIT_INVALID_INPUT
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


def test_missing_gpu_exits_with_runtime_code(monkeypatch, capsys):
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        script.check_gpu_visible("cuda")
    assert exc.value.code == script.EXIT_RUNTIME_ERROR
    assert "--gpus all" in capsys.readouterr().err
    script.check_gpu_visible("cpu")


def _patch_frames(monkeypatch, *frames):
    """Make the backend yield ``frames`` batches, tracking whether it is closed."""
    state = {"closed": False}

    def fake(_path, _batch_size, _max_frames=None):
        try:
            yield from frames
        finally:
            state["closed"] = True

    monkeypatch.setattr(script, "read_video_frames_batched", fake)
    return state


def test_unreadable_video_exits_with_runtime_code(monkeypatch, capsys):
    state = _patch_frames(monkeypatch, [(0, object())])
    with pytest.raises(SystemExit) as exc:
        script.check_video_decodable("clip.mp4", {"width": 0, "height": 0})
    assert exc.value.code == script.EXIT_RUNTIME_ERROR
    assert "Cannot read video clip.mp4" in capsys.readouterr().err
    assert not state["closed"]  # a 0x0 video is rejected without opening it
    script.check_video_decodable("clip.mp4", {"width": 640, "height": 480})
    assert state["closed"]


def test_video_with_size_but_no_readable_frame_exits_with_runtime_code(
    monkeypatch, capsys
):
    state = _patch_frames(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        script.check_video_decodable("clip.mp4", {"width": 640, "height": 480})
    assert exc.value.code == script.EXIT_RUNTIME_ERROR
    assert "Cannot read video clip.mp4" in capsys.readouterr().err
    assert state["closed"]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            script.UnsupportedVideoCodecError("Unsupported video codec 'mjpeg'"),
            script.EXIT_INVALID_INPUT,
        ),
        (
            script.VideoDecodeError("ffprobe failed for a.mp4: moov atom not found\nx"),
            script.EXIT_RUNTIME_ERROR,
        ),
        (
            script.VideoDecoderUnavailableError("A GPU is required"),
            script.EXIT_RUNTIME_ERROR,
        ),
    ],
)
def test_video_backend_errors_exit_with_one_line(monkeypatch, capsys, error, code):
    def fail(_path):
        raise error

    monkeypatch.setattr(script, "get_video_meta", fail)
    with pytest.raises(SystemExit) as exc:
        script.probe_video("a.mp4")
    assert exc.value.code == code
    err = capsys.readouterr().err
    assert "a.mp4" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


def test_unsupported_codec_found_while_reading_first_frame(monkeypatch, capsys):
    def fake(_path, _batch_size, _max_frames=None):
        raise script.UnsupportedVideoCodecError("Unsupported video codec 'vp8'")
        yield

    monkeypatch.setattr(script, "read_video_frames_batched", fake)
    monkeypatch.setattr(
        script, "get_video_meta", lambda _p: {"width": 64, "height": 48}
    )
    with pytest.raises(SystemExit) as exc:
        script.probe_video("a.webm")
    assert exc.value.code == script.EXIT_INVALID_INPUT
    assert "vp8" in capsys.readouterr().err


def test_script_reads_video_only_through_the_backend():
    # The release image's OpenCV has no video I/O, so a direct cv2.VideoCapture
    # would reject every input there.
    assert "VideoCapture" not in open(script.__file__).read()
