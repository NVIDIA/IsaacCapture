# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from __future__ import annotations

import sys

from inference.scripts.run_egohand_inference_pipeline import parse_args


def _parse(monkeypatch, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_egohand_inference_pipeline.py",
            "--input",
            "input.mp4",
            "--egohand_ckpt",
            "model.ckpt",
            "--output_dir",
            "output",
            *extra,
        ],
    )
    return parse_args()


def test_dino_compile_is_opt_in(monkeypatch):
    assert _parse(monkeypatch).dino_compile is False
    assert _parse(monkeypatch, "--dino_compile").dino_compile is True


def test_dino_compile_uses_the_configured_egohand_batch(monkeypatch):
    args = _parse(monkeypatch, "--batch_size", "640", "--dino_compile")

    assert args.batch_size == 640
    assert args.dino_compile is True


def test_smoothing_remains_enabled_by_default_and_can_be_disabled(monkeypatch):
    assert _parse(monkeypatch).postprocessing_smooth is True
    assert (
        _parse(monkeypatch, "--no-postprocessing_smooth").postprocessing_smooth is False
    )
    assert _parse(monkeypatch, "--postprocessing_smooth").postprocessing_smooth is True
