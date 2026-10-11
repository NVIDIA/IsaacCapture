# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from isaaccapture_examples.mcap_record_replay import calibrate_ego_sync


ORIGIN = 1_700_000_000_000_000_000
DURATION = 1800 * 10**9


def write_manifest(tmp_path, *, audio=False, error_ns=0):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "metadata.mcap").write_bytes(b"original-mcap")
    (capture / "raw").mkdir()
    (capture / "raw" / "ColorLeft.h264").write_bytes(b"original-video")
    (tmp_path / "events.csv").write_text("independent physical event annotations\n")

    def event(prefix, second):
        return {
            "event_id": f"{prefix}-{second}",
            "source_value": second * 48_000 if audio else ORIGIN + second * 10**9,
            "reference_time_ns": ORIGIN
            + second * 10**9
            + (error_ns if prefix == "val" else 0),
            "uncertainty_ns": 1000,
            "evidence": "events.csv",
        }

    clock = {
        "calibration_events": [event("cal", second) for second in (0, 1, 2)],
        "validation_events": [event("val", second) for second in (10, 900, 1790)],
        "validation_start_ns": ORIGIN,
        "validation_end_ns": ORIGIN + DURATION,
    }
    if audio:
        clock.update(segment_id=7, first_sample_index=0, sample_count=1800 * 48_000)
        streams = {"audio": {"segments": [clock]}}
    else:
        streams = {"gyro": clock}
    manifest = {
        "version": 1,
        "model_id": "independent-events",
        "recording": "capture/metadata.mcap",
        "media_paths": ["capture/raw/ColorLeft.h264"],
        "identity": {
            "serial": "test-device",
            "firmware": "1.1.1",
            "sdk_version": "2.9.0",
        },
        "profile": {
            "video": {"format": "h264", "width": 1600, "height": 1300, "fps": 30},
            "imu_rate_hz": 1000,
            "audio": {"sample_rate_hz": 48_000, "channels": 1, "bits_per_sample": 16},
        },
        "epochs": [
            {
                "epoch": 0,
                "valid_from_ns": ORIGIN,
                "valid_to_ns": ORIGIN + DURATION,
                "streams": streams,
            }
        ],
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest))


def test_manifest_paths_resolve_independently_of_working_directory(
    tmp_path, monkeypatch
):
    path, _ = write_manifest(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    result = calibrate_ego_sync.calibrate_manifest(path)
    assert result["status"] == "validated"
    assert result["epochs"][0]["streams"]["gyro"]["scale"] == 1
    assert result["recording_fingerprint"] == {
        "mcap_sha256": hashlib.sha256(b"original-mcap").hexdigest(),
        "media_sha256": {
            "raw/ColorLeft.h264": hashlib.sha256(b"original-video").hexdigest()
        },
    }
    evidence_hash = hashlib.sha256((tmp_path / "events.csv").read_bytes()).hexdigest()
    assert result["validation"]["evidence"] == [f"sha256:{evidence_hash}"]
    assert result["validation"]["stream_reports"][0]["stream"] == "gyro"


def test_cli_writes_new_validated_document(tmp_path, capsys):
    path, _ = write_manifest(tmp_path)
    output = tmp_path / "alignment.json"
    assert calibrate_ego_sync.main([str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "validated"
    assert "status=validated" in capsys.readouterr().out


def test_failed_independent_validation_is_saved_but_returns_failure(tmp_path):
    path, _ = write_manifest(tmp_path, error_ns=20_000_000)
    output = tmp_path / "alignment.json"
    assert calibrate_ego_sync.main([str(path), "--output", str(output)]) == 2
    result = json.loads(output.read_text())
    assert result["status"] == "unverified"
    assert result["validation"]["max_abs_error_ns"] == 20_000_000


@pytest.mark.parametrize("continuity", [None, True])
def test_audio_needs_independent_continuity_proof(tmp_path, continuity):
    path, manifest = write_manifest(tmp_path, audio=True)
    segment = manifest["epochs"][0]["streams"]["audio"]["segments"][0]
    if continuity is not None:
        segment["continuity_verified"] = continuity
    save_manifest(path, manifest)
    output = tmp_path / "alignment.json"
    assert calibrate_ego_sync.main([str(path), "--output", str(output)]) == (
        0 if continuity else 2
    )
    result = json.loads(output.read_text())
    assert result["epochs"][0]["streams"]["audio"]["segments"][0][
        "period_ns"
    ] == pytest.approx(10**9 / 48_000)
    assert ("audio_continuity_unverified" in result["validation"]["reasons"]) == (
        not continuity
    )


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_refuses_existing_output_without_changing_it(tmp_path, symlink):
    path, _ = write_manifest(tmp_path)
    output = tmp_path / "alignment.json"
    if symlink:
        output.symlink_to(tmp_path / "missing-target.json")
    else:
        output.write_text("keep this calibration")
    with pytest.raises(SystemExit) as failure:
        calibrate_ego_sync.main([str(path), "--output", str(output)])
    assert failure.value.code == 2
    if symlink:
        assert output.is_symlink()
        assert not output.exists()
    else:
        assert output.read_text() == "keep this calibration"


@pytest.mark.parametrize("bad_key", ["duplicate", "nonfinite"])
def test_manifest_rejects_ambiguous_json(tmp_path, bad_key):
    path = tmp_path / "events.json"
    path.write_text(
        '{"version":1,"version":2}' if bad_key == "duplicate" else '{"version":NaN}'
    )
    with pytest.raises(ValueError, match=r"Duplicate|Nonfinite"):
        calibrate_ego_sync.calibrate_manifest(path)


@pytest.mark.parametrize("coordinate", ["version", "fps", "source_value"])
def test_manifest_rejects_boolean_integer_coordinates(tmp_path, coordinate):
    path, manifest = write_manifest(tmp_path)
    if coordinate == "version":
        manifest["version"] = True
    elif coordinate == "fps":
        manifest["profile"]["video"]["fps"] = True
    else:
        manifest["epochs"][0]["streams"]["gyro"]["calibration_events"][0][
            "source_value"
        ] = True
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="integer"):
        calibrate_ego_sync.calibrate_manifest(path)


def test_missing_raw_evidence_does_not_create_a_model(tmp_path):
    path, _ = write_manifest(tmp_path)
    (tmp_path / "events.csv").unlink()
    output = tmp_path / "alignment.json"
    with pytest.raises(SystemExit) as failure:
        calibrate_ego_sync.main([str(path), "--output", str(output)])
    assert failure.value.code == 2
    assert not output.exists()


def test_audio_annotations_must_belong_to_declared_sample_segment(tmp_path):
    path, manifest = write_manifest(tmp_path, audio=True)
    manifest["epochs"][0]["streams"]["audio"]["segments"][0]["sample_count"] = 48_000
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="outside its declared sample segment"):
        calibrate_ego_sync.calibrate_manifest(path)


def test_effective_epoch_cannot_exceed_validation_interval(tmp_path):
    path, manifest = write_manifest(tmp_path)
    manifest["epochs"][0]["valid_to_ns"] = ORIGIN + DURATION + 1
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="independent validation bounds"):
        calibrate_ego_sync.calibrate_manifest(path)


def test_effective_epoch_can_select_part_of_validated_interval(tmp_path):
    path, manifest = write_manifest(tmp_path)
    manifest["epochs"][0]["valid_from_ns"] = ORIGIN + 60 * 10**9
    manifest["epochs"][0]["valid_to_ns"] = ORIGIN + 900 * 10**9
    save_manifest(path, manifest)
    result = calibrate_ego_sync.calibrate_manifest(path)
    assert result["status"] == "validated"
    mapping = result["epochs"][0]["streams"]["gyro"]
    assert mapping["validation_start_ns"] == ORIGIN
    assert mapping["validation_end_ns"] == ORIGIN + DURATION


def test_audio_continuity_flag_must_be_boolean(tmp_path):
    path, manifest = write_manifest(tmp_path, audio=True)
    manifest["epochs"][0]["streams"]["audio"]["segments"][0]["continuity_verified"] = (
        "true"
    )
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match="continuity_verified must be a boolean"):
        calibrate_ego_sync.calibrate_manifest(path)
