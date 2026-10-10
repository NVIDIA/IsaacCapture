# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from hand_cts.labels import StepTimeline
from steps import Window

FIXTURES = Path(__file__).resolve().parents[2] / "oracle" / "fixtures"

make_labels = pytest.importorskip("make_labels")


@pytest.fixture
def golden(tmp_path: Path) -> tuple[Path, list[Window]]:
    source = FIXTURES / "golden_full_script.mcap"
    if not source.is_file():
        pytest.skip(
            "no fixture set; run src/python/isaaccapture/cts/hand/oracle/generate.sh"
        )
    recording = tmp_path / "take-hand.mcap"
    shutil.copy(source, recording)
    timeline = StepTimeline.beside(source)
    windows = [
        Window(s.index, s.label, s.start_ns, s.end_ns, s.is_still_window)
        for s in timeline.steps
    ]
    return recording, windows


def test_sidecar_round_trips_through_the_checker_loader(golden):
    recording, windows = golden
    sidecar, checks = make_labels.write(recording, windows)
    assert sidecar is not None
    loaded = StepTimeline.beside(recording)
    assert [(s.label, s.start_ns, s.end_ns) for s in loaded.steps] == [
        (w.label, w.start_ns, w.end_ns) for w in windows
    ]
    payload = json.loads(sidecar.read_text())
    assert "provisional" not in payload
    assert payload["records"]["left"] == payload["records"]["right"] > 0
    assert all(ok for _, ok, _ in checks), checks


def test_a_sidecar_with_a_retired_key_still_loads(golden):
    recording, windows = golden
    sidecar, _ = make_labels.write(recording, windows)
    payload = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**payload, "provisional": True}))
    loaded = StepTimeline.beside(recording)
    assert [s.label for s in loaded.steps] == [w.label for w in windows]


def test_a_mislabelled_window_is_reported(golden):
    recording, windows = golden
    fist = next(i for i, w in enumerate(windows) if w.label == "fist")
    opened = next(w for w in windows if w.label == "flat_on_table_open")
    windows[fist] = replace(
        windows[fist], start_ns=opened.start_ns, end_ns=opened.end_ns
    )
    _, checks = make_labels.write(recording, windows)
    bad = [name for name, ok, _ in checks if not ok]
    assert "segmentation.labelled_step_actually_performed" in bad


def test_no_windows_writes_no_sidecar(golden):
    recording, _ = golden
    sidecar, _ = make_labels.write(recording, [])
    assert sidecar is None
    assert StepTimeline.beside(recording) is None
