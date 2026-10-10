# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DEFAULT_FIXTURES, index_entries

from hand_cts.cli import main

ENTRIES = {Path(e["filename"]).stem: e for e in index_entries()}


def test_list_checks_needs_no_recording(capsys):
    assert main(["--list-checks"]) == 0
    assert "pinch.finger_identity" in capsys.readouterr().out


@pytest.mark.skipif(not ENTRIES, reason=f"run {DEFAULT_FIXTURES}/generate.sh")
@pytest.mark.parametrize(
    "stem, status",
    [("golden_full_script", 0), ("defect_centimetres", 1), ("retake_shallow_fist", 2)],
)
def test_exit_status_follows_the_verdict(stem, status, fixture_dir, capsys):
    path = fixture_dir / ENTRIES[stem]["filename"]
    assert main([str(path), "--json"]) == status
    assert json.loads(capsys.readouterr().out)["verdict"] in ("pass", "fail", "retake")


@pytest.mark.skipif(not ENTRIES, reason=f"run {DEFAULT_FIXTURES}/generate.sh")
def test_without_labels_the_verdict_is_still_reached(fixture_dir, tmp_path, capsys):
    source = fixture_dir / ENTRIES["golden_full_script"]["filename"]
    copy = tmp_path / "unlabelled.mcap"
    copy.write_bytes(source.read_bytes())
    assert main([str(copy)]) == 0
    assert "no motion-step labels" in capsys.readouterr().out
