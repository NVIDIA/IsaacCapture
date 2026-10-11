# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest

from isaaccapture.ego.calibration import (
    PairedTimeEvent,
    VALIDATION_DURATION_NS,
    evidence_sha256,
    fit_time_alignment,
    make_recording_alignment,
)


ORIGIN = 1_700_000_000_000_000_000


def event(name, source, reference, uncertainty=1000):
    return PairedTimeEvent(name, source, reference, uncertainty, f"raw-{name}.sha256")


def raw_events(offset=20_000_000, validation_error=0):
    training = [
        event(f"cal-{i}", ORIGIN + i * 10**9, ORIGIN + i * 10**9 + offset)
        for i in (0, 1, 2)
    ]
    held_out = [
        event(
            f"val-{i}",
            ORIGIN + i * 10**9,
            ORIGIN + i * 10**9 + offset + validation_error,
        )
        for i in (10, 450, 890)
    ]
    return training, held_out


def fit_raw(**kwargs):
    training, held_out = raw_events(**kwargs)
    return fit_time_alignment(
        training,
        held_out,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
    )


def test_integer_anchors_preserve_ns_at_epoch_scale():
    result = fit_raw(offset=20_000_001)
    assert result.status == "validated"
    assert result.reference_time(ORIGIN + 123) == ORIGIN + 20_000_124
    assert result.max_abs_error_ns == 0
    assert result.drift_ppm == 0
    assert result.validation_start_ns == ORIGIN
    assert result.validation_end_ns == ORIGIN + VALIDATION_DURATION_NS
    assert result.validation_report()["validation_end_ns"] == result.validation_end_ns


def test_training_fit_does_not_hide_independent_validation_error():
    result = fit_raw(validation_error=20_000_000)
    assert result.status == "unverified"
    assert result.max_abs_error_ns == 20_000_000
    assert "physical_error_and_uncertainty_exceed_half_frame" in result.reasons


def test_exposure_and_event_uncertainty_counts_against_half_frame():
    training, held_out = raw_events()
    held_out[0] = event(
        "val-10", held_out[0].source_value, held_out[0].reference_time_ns, 17_000_000
    )
    result = fit_time_alignment(
        training,
        held_out,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
    )
    assert result.status == "unverified"
    assert result.uncertainty_ns == 17_000_000


def test_rejects_validation_data_reused_for_calibration():
    training, _ = raw_events()
    with pytest.raises(ValueError, match="disjoint"):
        fit_time_alignment(
            training,
            training,
            fps=30,
            validation_start_ns=ORIGIN,
            validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
        )


def test_short_or_nonuniform_validation_is_not_certified():
    training, _ = raw_events()
    short = [
        event(f"val-{i}", ORIGIN + i * 10**9, ORIGIN + i * 10**9 + 20_000_000)
        for i in (10, 20, 30)
    ]
    result = fit_time_alignment(
        training,
        short,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + 60 * 10**9,
    )
    assert result.status == "unverified"
    assert "validation_shorter_than_15_minutes" in result.reasons
    assert "missing_beginning_middle_or_end_validation" in result.reasons
    late = [
        event(f"late-{i}", ORIGIN + i * 10**9, ORIGIN + i * 10**9 + 20_000_000)
        for i in (100, 200, 300)
    ]
    result = fit_time_alignment(
        training,
        late,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
    )
    assert "missing_beginning_middle_or_end_validation" in result.reasons


@pytest.mark.parametrize(
    "duration, status",
    [(900_000_000_000, "validated"), (899_999_999_999, "unverified")],
)
def test_fifteen_minute_validation_boundary(duration, status):
    training, held_out = raw_events()
    result = fit_time_alignment(
        training,
        held_out,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + duration,
    )
    assert result.status == status
    assert result.held_out_duration_ns == duration
    assert result.reasons == (
        () if status == "validated" else ("validation_shorter_than_15_minutes",)
    )


def test_audio_period_fits_drift_instead_of_assuming_48000_hz():
    period = 20_834
    training = [
        event(f"cal-{i}", i * 48_000, ORIGIN + i * 48_000 * period) for i in (0, 1, 2)
    ]
    held_out = [
        event(f"val-{i}", i * 48_000, ORIGIN + i * 48_000 * period)
        for i in (10, 450, 890)
    ]
    result = fit_time_alignment(
        training,
        held_out,
        fps=30,
        validation_start_ns=ORIGIN,
        validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
        source_unit="samples",
        sample_rate_hz=48_000,
    )
    assert result.status == "validated"
    assert result.scale == period
    assert result.drift_ppm == pytest.approx(32)
    assert result.reference_time(48_000) == ORIGIN + 48_000 * period


@pytest.mark.parametrize("value", [True, "30", 0, -1, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(value):
    training, held_out = raw_events()
    with pytest.raises(ValueError, match="fps"):
        fit_time_alignment(
            training,
            held_out,
            fps=value,
            validation_start_ns=ORIGIN,
            validation_end_ns=ORIGIN + VALIDATION_DURATION_NS,
        )


def test_recording_model_is_bound_to_original_mcap_and_media(tmp_path):
    recording = tmp_path / "capture.mcap"
    recording.write_bytes(b"recording")
    media = tmp_path / "raw" / "left.h264"
    media.parent.mkdir()
    media.write_bytes(b"original-video")
    result = make_recording_alignment(
        recording,
        model_id="test-physical-events",
        identity={"serial": "test", "firmware": "1", "sdk_version": "2.9"},
        profile={"video": {"format": "H264", "width": 1600, "height": 1300, "fps": 30}},
        epochs=[
            {
                "epoch": 0,
                "valid_from_ns": ORIGIN,
                "valid_to_ns": ORIGIN + VALIDATION_DURATION_NS,
                "streams": {"gyro": fit_raw()},
            }
        ],
        media_paths=[media],
    )
    assert result["status"] == "validated"
    assert result["recording_fingerprint"]["mcap_sha256"] == evidence_sha256(recording)
    assert result["recording_fingerprint"]["media_sha256"] == {
        "raw/left.h264": evidence_sha256(media)
    }
    assert (
        result["epochs"][0]["streams"]["gyro"]["reference_anchor_ns"]
        == ORIGIN + 20_000_000
    )
    json.dumps(result, allow_nan=False)


def test_failed_physical_validation_cannot_emit_validated_document(tmp_path):
    recording = tmp_path / "capture.mcap"
    recording.write_bytes(b"recording")
    result = make_recording_alignment(
        recording,
        model_id="failed-events",
        identity={"serial": "test", "firmware": "1", "sdk_version": "2.9"},
        profile={"video": {"fps": 30}},
        epochs=[
            {
                "epoch": 0,
                "valid_from_ns": ORIGIN,
                "valid_to_ns": ORIGIN + VALIDATION_DURATION_NS,
                "streams": {"gyro": fit_raw(validation_error=20_000_000)},
            }
        ],
    )
    assert result["status"] == "unverified"
    assert result["validation"]["max_abs_error_ns"] == 20_000_000


def test_combined_error_budget_and_audio_continuity_are_not_certified(tmp_path):
    recording = tmp_path / "capture.mcap"
    recording.write_bytes(b"recording")
    arguments = {
        "model_id": "physical-events",
        "identity": {"serial": "test", "firmware": "1", "sdk_version": "2.9"},
        "profile": {"video": {"fps": 30}},
    }
    epoch = {
        "epoch": 0,
        "valid_from_ns": ORIGIN,
        "valid_to_ns": ORIGIN + VALIDATION_DURATION_NS,
        "streams": {
            "gyro": replace(fit_raw(), max_abs_error_ns=12_000_000),
            "accel": replace(fit_raw(), uncertainty_ns=8_000_000),
        },
    }
    result = make_recording_alignment(recording, epochs=[epoch], **arguments)
    assert result["status"] == "unverified"
    assert (
        "physical_error_and_uncertainty_exceed_half_frame"
        in result["validation"]["reasons"]
    )
    assert [report["stream"] for report in result["validation"]["stream_reports"]] == [
        "gyro",
        "accel",
    ]
    epoch["streams"] = {
        "audio": [
            {
                "segment_id": 0,
                "first_sample_index": 0,
                "sample_count": 48000,
                "fit": replace(
                    fit_raw(), source_unit="samples", scale=1_000_000_000 / 48000
                ),
            }
        ],
    }
    result = make_recording_alignment(recording, epochs=[epoch], **arguments)
    assert result["status"] == "unverified"
    assert result["validation"]["reasons"] == ["audio_continuity_unverified"]
    epoch["streams"]["audio"][0]["continuity_verified"] = "true"
    with pytest.raises(ValueError, match="boolean"):
        make_recording_alignment(recording, epochs=[epoch], **arguments)


def test_overlapping_effective_ranges_are_rejected(tmp_path):
    recording = tmp_path / "capture.mcap"
    recording.write_bytes(b"recording")
    epoch = {
        "epoch": 0,
        "valid_from_ns": ORIGIN,
        "valid_to_ns": ORIGIN + VALIDATION_DURATION_NS,
        "streams": {"gyro": fit_raw()},
    }
    with pytest.raises(ValueError, match="effective interval"):
        make_recording_alignment(
            recording,
            model_id="overlap",
            identity={"serial": "test", "firmware": "1", "sdk_version": "2.9"},
            profile={"video": {"fps": 30}},
            epochs=[epoch, epoch],
        )


@pytest.mark.parametrize("extend_start", [False, True])
def test_recording_model_cannot_extend_beyond_independent_evidence(
    tmp_path, extend_start
):
    recording = tmp_path / "capture.mcap"
    recording.write_bytes(b"recording")
    begin = ORIGIN - 1 if extend_start else ORIGIN
    end = ORIGIN + VALIDATION_DURATION_NS + (0 if extend_start else 1)
    with pytest.raises(ValueError, match="independent validation bounds"):
        make_recording_alignment(
            recording,
            model_id="overextended",
            identity={"serial": "test", "firmware": "1", "sdk_version": "2.9"},
            profile={"video": {"fps": 30}},
            epochs=[
                {
                    "epoch": 0,
                    "valid_from_ns": begin,
                    "valid_to_ns": end,
                    "streams": {"gyro": fit_raw()},
                }
            ],
        )
