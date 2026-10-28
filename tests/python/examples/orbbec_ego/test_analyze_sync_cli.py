# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from isaaccapture_examples.orbbec_ego import analyze_sync


def _report(status="validated", incomplete=0):
    return {
        "alignment": {"status": status},
        "coverage": {
            "window_count": 2,
            "complete_windows": 2 - incomplete,
            "incomplete_windows": incomplete,
        },
        "windows": [{"start_time_ns": 100, "end_time_ns": 200}],
        "physical_error_measurements": None,
    }


def test_requested_scope_and_bounds_reach_audit(tmp_path, monkeypatch):
    recording = tmp_path / "recording.mcap"
    alignment = tmp_path / "alignment.json"
    output = tmp_path / "audit.json"
    calls = []

    def audit(path, **kwargs):
        calls.append((path, kwargs))
        return _report()

    monkeypatch.setattr(analyze_sync, "analyze_recording_sync", audit)
    assert (
        analyze_sync.main(
            [
                str(recording),
                "--required-streams",
                "right_rgb",
                "gyro",
                "--reference-stream",
                "right_rgb",
                "--alignment",
                str(alignment),
                "--epoch",
                "3",
                "--start-time-ns",
                "100",
                "--end-time-ns",
                "200",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [
        (
            recording,
            {
                "required_streams": ["right_rgb", "gyro"],
                "alignment": alignment,
                "reference_stream": "right_rgb",
                "epoch": 3,
                "start_time_ns": 100,
                "end_time_ns": 200,
            },
        )
    ]
    assert json.loads(output.read_text()) == _report()


@pytest.mark.parametrize(
    "status,incomplete", [("unverified", 0), ("invalid", 0), ("validated", 1)]
)
def test_failed_acceptance_still_preserves_report(
    tmp_path, monkeypatch, status, incomplete
):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        analyze_sync,
        "analyze_recording_sync",
        lambda *_args, **_kwargs: _report(status, incomplete),
    )
    assert (
        analyze_sync.main(
            [
                "take.mcap",
                "--required-streams",
                "left_rgb",
                "right_rgb",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert json.loads(output.read_text()) == _report(status, incomplete)


def test_output_is_not_overwritten_or_audited(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    output.write_text("retained evidence")

    def unexpected(*_args, **_kwargs):
        pytest.fail("Existing output should be rejected before loading a recording")

    monkeypatch.setattr(analyze_sync, "analyze_recording_sync", unexpected)
    with pytest.raises(SystemExit, match="2"):
        analyze_sync.main(
            ["take.mcap", "--required-streams", "left_rgb", "--output", str(output)]
        )
    assert output.read_text() == "retained evidence"


def test_required_streams_cannot_be_implicit(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        analyze_sync.main(["take.mcap", "--output", str(tmp_path / "audit.json")])


def test_missing_recording_error_creates_no_report(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"

    def audit(*_args, **_kwargs):
        raise ValueError("Required stream is absent: gyro")

    monkeypatch.setattr(analyze_sync, "analyze_recording_sync", audit)
    with pytest.raises(SystemExit, match="2"):
        analyze_sync.main(
            [
                "take.mcap",
                "--required-streams",
                "left_rgb",
                "gyro",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
