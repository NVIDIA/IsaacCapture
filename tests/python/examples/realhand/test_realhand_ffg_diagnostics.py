# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw diagnostics retain all independent device inputs without selecting poses."""

import sys

import numpy as np
import pytest
from isaaccapture_examples.teleop import realhand_ffg_diagnostics as _DIAGNOSTICS


def test_pipeline_records_hands_controllers_and_raw_sensors():
    spec = _DIAGNOSTICS.build_pipeline().output_types()
    assert set(spec) == {
        "hand_left",
        "hand_right",
        "controller_left",
        "controller_right",
        "raw_left",
        "raw_right",
    }


def test_periodic_phase_summary_does_not_invent_a_half_turn():
    values = np.tile(np.array((6.27, 0.01, 0.02)), 20)
    samples = np.repeat(values[:, None], 21, axis=1)
    summary = _DIAGNOSTICS.summarize_samples(samples)
    np.testing.assert_allclose(summary["median_rad"], 2 * np.pi + 0.01)
    assert max(summary["robust_sigma_rad"]) < 0.04


@pytest.mark.parametrize(
    "values", (np.zeros((29, 21)), np.zeros((40, 20)), np.full((40, 21), np.nan))
)
def test_incomplete_or_invalid_raw_capture_is_rejected(values):
    with pytest.raises(ValueError):
        _DIAGNOSTICS.summarize_samples(values)


def test_repeated_open_pose_can_measure_baseline_drift():
    labels = [label for _, label in _DIAGNOSTICS.POSES]
    assert labels[0] == "open"
    assert labels[-1] == "open_end"
    assert labels[1:5] == ["fist", "thumb_curl", "thumb_opposition", "spread"]
    assert labels[5:9] == ["thumb_index", "thumb_middle", "thumb_ring", "thumb_little"]


def test_diagnostic_refuses_to_overwrite_existing_take(monkeypatch, tmp_path):
    recording = tmp_path / "existing.mcap"
    recording.write_bytes(b"preserve this")
    calibration = tmp_path / "calibration.yml"
    calibration.write_text("format: placeholder\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostic",
            "--output",
            str(recording),
            "--calibration",
            str(calibration),
            "--plugin-path",
            str(tmp_path),
            "--ports",
            "/dev/ttyUSB0,/dev/ttyUSB1",
        ],
    )
    with pytest.raises(SystemExit) as failed:
        _DIAGNOSTICS.main()
    assert failed.value.code == 2
    assert recording.read_bytes() == b"preserve this"
