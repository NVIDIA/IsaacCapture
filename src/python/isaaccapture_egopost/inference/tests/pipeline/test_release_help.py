# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import pytest

from inference.cli import parse_args


def _help(monkeypatch, capsys, profile: str) -> str:
    monkeypatch.setenv("EGOHAND_RUNTIME_PROFILE", profile)
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"], epilog="source epilog")
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_release_help_describes_the_release_image(monkeypatch, capsys):
    out = _help(monkeypatch, capsys, "release")

    for stale in (
        "source epilog",
        "huggingface.co/nvidia/SOMA-X",
        "Skip .mp4 outputs",
        "Disables video/image saving",
        "configs/rtdetr.yaml",
        "--draw_skeleton",
    ):
        assert stale not in out
    assert "overlay" not in out.lower()

    for needed in (
        "/workspace/artifacts/rtdetr.yaml",
        "bbox_smoothing",
        "saved_bboxes",
        "run_hand_detector.py",
        '"model": "PINHOLE"',
    ):
        assert needed in out


def test_full_help_keeps_the_source_text(monkeypatch, capsys):
    out = _help(monkeypatch, capsys, "full")

    assert "source epilog" in out
    assert "huggingface.co/nvidia/SOMA-X" in out
    assert "Skip .mp4 outputs" in out
