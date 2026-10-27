# SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete MCAP wrappers with independently generated FlatBuffers."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np
from mcap.writer import CompressionType, Writer

from isaaccapture.ego.sync import (
    analyze_recording_sync,
    iter_sync_windows,
    recording_fingerprint,
)

ORIGIN = 1_700_000_000_000_000_001
PERIOD = 10_000_000
IDENTITY = {"serial": "test-device", "firmware": "1.1.1", "sdk_version": "2.9.0"}
IDENTITY["sdk_binary_sha256"] = "12" * 32
PROFILE = {
    "video": {"format": "h264", "width": 16, "height": 12, "fps": 100},
    "imu_rate_hz": 1000,
    "audio": {"sample_rate_hz": 1000, "channels": 1, "bits_per_sample": 16},
}
ALL = ("left_rgb", "right_rgb", "accel", "gyro", "audio")


@pytest.fixture
def recording(tmp_path):
    for parent in Path(__file__).resolve().parents:
        if (parent / "repo_paths.py").is_file():
            sys.path.insert(0, str(parent))
            break
    from repo_paths import repo_root

    compiler = os.environ.get("EGO_TEST_FLATC")
    if not compiler:
        pytest.fail("EGO_TEST_FLATC must name the build's flatc executable")
    source = repo_root() / "src/core/schema/fbs"
    schema_paths = {}
    schema_data = {}
    serial = 0

    def compile_record(name, data):
        nonlocal serial
        family = {
            "OrbbecEgoFrameMetadataRecord": "orbbec_ego_camera",
            "OrbbecEgoEncodedVideoFrameRecord": "orbbec_ego_camera",
            "OrbbecEgoImuBatchRecord": "orbbec_ego_imu",
            "OrbbecEgoAudioChunkRecord": "orbbec_ego_audio",
            "OrbbecEgoPcmAudioChunkRecord": "orbbec_ego_audio",
            "OrbbecEgoCalibrationRecord": "orbbec_ego_calibration",
        }[name]
        if name not in schema_paths:
            schema_paths[name] = source / f"{family}.fbs"
            subprocess.run(
                [
                    compiler,
                    "--binary",
                    "--schema",
                    "--root-type",
                    f"core.{name}",
                    "-I",
                    str(source),
                    "-o",
                    str(tmp_path),
                    str(schema_paths[name]),
                ],
                check=True,
                capture_output=True,
            )
            schema_data[name] = (tmp_path / f"{family}.bfbs").read_bytes()
        serial += 1
        document = tmp_path / f"record_{serial}.json"
        document.write_text(json.dumps(data))
        subprocess.run(
            [
                compiler,
                "--binary",
                "--strict-json",
                "--root-type",
                f"core.{name}",
                "-I",
                str(source),
                "-o",
                str(tmp_path),
                str(schema_paths[name]),
                str(document),
            ],
            check=True,
            capture_output=True,
        )
        return (tmp_path / f"record_{serial}.bin").read_bytes(), schema_data[name]

    def create(
        *,
        embedded=False,
        legacy_audio=False,
        epochs=(0,),
        frame_ids=(0, 1, 2, 3),
        skip_gyro_sample=None,
        audio=True,
        duplicate_observation=True,
        right_offset_ns=300_000,
        right_sequence_offset=0,
        audio_chunks=((0, 17), (17, 23)),
        raw_regression=False,
        capture_profile=False,
        capture_profile_change=None,
        bridge_change=None,
        bridge_snapshots=(),
        recorded_identity=None,
        local_transition=False,
        pcm_timing_mismatch=False,
        imu_samples=40,
    ):
        root = tmp_path / f"capture_{serial}"
        root.mkdir()
        path = root / "metadata.mcap"
        schemas, channels = {}, {}
        video_bytes = {"ColorLeft": bytearray(), "ColorRight": bytearray()}
        wav = bytearray(44)
        with path.open("wb") as output:
            writer = Writer(output, compression=CompressionType.NONE)
            writer.start()

            def add(topic, name, data, raw):
                wrapped = {
                    "data": data,
                    "timestamp": {
                        "available_time_local_common_clock": raw + 90_000_000,
                        "sample_time_local_common_clock": raw
                        + (
                            90_000_000
                            if local_transition and raw < ORIGIN + PERIOD
                            else 40_000_000
                        ),
                        "sample_time_raw_device_clock": raw,
                    },
                }
                payload, bfbs = compile_record(name, wrapped)
                if name not in schemas:
                    schemas[name] = writer.register_schema(
                        f"core.{name}", "flatbuffer", bfbs
                    )
                if topic not in channels:
                    channels[topic] = writer.register_channel(
                        topic, "flatbuffer", schemas[name]
                    )
                # Arrival/log order is deliberately unrelated to the per-sample clock.
                writer.add_message(
                    channels[topic],
                    log_time=serial,
                    publish_time=serial + 1,
                    data=payload,
                )

            for epoch in epochs:
                baseline = {
                    "version": 1,
                    "scope": "recording",
                    "status": "unverified",
                    "identity": (recorded_identity or IDENTITY).copy(),
                    "profile": PROFILE,
                    "epochs": [],
                }
                if capture_profile:
                    baseline["capture_profile"] = {
                        "version": 1,
                        "scope": "capture_profile",
                        "status": "validated",
                        "reference_stream": "left_rgb",
                        "model_id": "imported-profile",
                        "max_capture_duration_ns": 1_800_000_000_000,
                        "identity": IDENTITY.copy(),
                        "profile": PROFILE,
                        "validation": {
                            "method": "held_out_physical_events",
                            "max_abs_error_ns": 1000,
                            "uncertainty_ns": 1000,
                            "held_out_duration_ns": 1_800_000_000_000,
                            "evidence": ["independent evidence"],
                        },
                        "streams": {
                            stream: {
                                "offset_ns": -2_000_000
                                if stream in ("accel", "gyro")
                                else 0,
                                "scale": 1.0,
                            }
                            for stream in ALL
                            if stream != "audio"
                        },
                    }
                    if capture_profile_change:
                        capture_profile_change(baseline["capture_profile"])
                    baseline["host_bridge"] = {
                        "epoch": epoch,
                        "reference_anchor_ns": ORIGIN,
                        "local_anchor_ns": ORIGIN + 123_000_000,
                        "first_mapped_raw_ns": {
                            "left_rgb": ORIGIN,
                            "right_rgb": ORIGIN + right_offset_ns,
                            "accel": ORIGIN + 2_000_000,
                            "gyro": ORIGIN + 2_000_000,
                        },
                        "valid_from_ns": ORIGIN,
                        "valid_to_ns": ORIGIN
                        + baseline["capture_profile"]["max_capture_duration_ns"],
                        "status": "active",
                        "host_absolute_accuracy": "unverified",
                    }
                    if bridge_change:
                        bridge_change(baseline["host_bridge"])
                calibration_data = {
                    "device_uid": "usb-1",
                    "capture_epoch": epoch,
                    "color_left": {
                        "width": 16,
                        "height": 12,
                        "fx": 10.5,
                        "fy": 10.6,
                        "cx": 8.0,
                        "cy": 6.0,
                        "distortion_model": 5,
                        "distortion": [0.1, 0.2, 0.3, 0.4],
                    },
                    "color_right": {
                        "width": 16,
                        "height": 12,
                        "fx": 10.4,
                        "fy": 10.7,
                        "cx": 8.0,
                        "cy": 6.0,
                    },
                    "left_to_right": {
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation_mm": [-120, 0, 0],
                    },
                    "accel_intrinsics": [1, 2, 3],
                    "gyro_intrinsics": [4, 5, 6],
                    "accel_to_left": {
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation_mm": [1, 2, 3],
                    },
                    "gyro_to_left": {
                        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "translation_mm": [4, 5, 6],
                    },
                    "raw_alignment_yaml": "factory: spatial",
                    "raw_imu_yaml": "factory: imu",
                    "time_alignment": {"json": json.dumps(baseline)},
                }
                add(
                    "orbbec_ego_calibration/Calibration",
                    "OrbbecEgoCalibrationRecord",
                    calibration_data,
                    ORIGIN,
                )
                for bridge in bridge_snapshots:
                    baseline["host_bridge"].update(bridge)
                    calibration_data["time_alignment"] = {"json": json.dumps(baseline)}
                    add(
                        "orbbec_ego_calibration/Calibration",
                        "OrbbecEgoCalibrationRecord",
                        calibration_data,
                        ORIGIN,
                    )
                for side, discriminator in (
                    ("ColorLeft", "ColorLeft"),
                    ("ColorRight", "ColorRight"),
                ):
                    for frame_id in frame_ids:
                        raw = (
                            ORIGIN
                            + frame_id * PERIOD
                            + (right_offset_ns if side == "ColorRight" else 0)
                        )
                        metadata = {
                            "stream": discriminator,
                            "sequence_number": frame_id
                            + (right_sequence_offset if side == "ColorRight" else 0),
                            "width": 16,
                            "height": 12,
                            "fps": 100,
                            "pixel_format": "H264",
                            "encoded_bytes": 4,
                            "capture_epoch": epoch,
                            "sdk_metadata": [{"key": 19, "value": 123456789}],
                        }
                        add(
                            f"orbbec_ego_metadata_{side.lower()}/FrameMetadata",
                            "OrbbecEgoFrameMetadataRecord",
                            metadata,
                            raw,
                        )
                        if duplicate_observation:
                            add(
                                f"session_observed/orbbec_ego_metadata_{side.lower()}/FrameMetadata",
                                "OrbbecEgoFrameMetadataRecord",
                                metadata,
                                raw,
                            )
                        if embedded:
                            media = {
                                key: value
                                for key, value in metadata.items()
                                if key not in ("encoded_bytes", "sdk_metadata")
                            }
                            media["encoded_data"] = [0, 0, 1, frame_id % 256]
                            add(
                                f"orbbec_ego_media/{side}",
                                "OrbbecEgoEncodedVideoFrameRecord",
                                media,
                                raw,
                            )
                        else:
                            video_bytes[side].extend(bytes([0, 0, 1, frame_id % 256]))
                for sensor in ("Accel", "Gyro"):
                    samples = []
                    for index in range(imu_samples):
                        if sensor == "Gyro" and index == skip_gyro_sample:
                            continue
                        raw = ORIGIN + index * 1_000_000 + 2_000_000
                        if raw_regression and sensor == "Gyro" and index == 15:
                            raw -= 2_000_000
                        samples.append(
                            {
                                "x_si": index + 0.25,
                                "y_si": 2.5,
                                "z_si": -3.5,
                                "temperature_c": 28.5,
                                "sample_time_local_common_clock_ns": raw + 70_000_000,
                                "sample_time_raw_device_clock_ns": raw,
                            }
                        )
                    add(
                        f"orbbec_ego_imu_{sensor.lower()}/Imu",
                        "OrbbecEgoImuBatchRecord",
                        {
                            "sensor": sensor,
                            "sequence_number": 0,
                            "sample_rate_hz": 1000,
                            "full_scale": 24,
                            "samples": samples,
                            "capture_epoch": epoch,
                        },
                        ORIGIN + 80_000_000,
                    )
                if audio:
                    for sequence, (first, count) in enumerate(audio_chunks):
                        data = {
                            "sequence_number": sequence,
                            "sample_rate_hz": 1000,
                            "channel_count": 1,
                            "bits_per_sample": 16,
                            "sample_count": count,
                            "wav_data_offset": len(wav),
                            "byte_count": count * 2,
                            "capture_epoch": epoch,
                        }
                        if not legacy_audio:
                            data["timing"] = {
                                "segment_id": epoch,
                                "first_sample_index": first,
                                "continuity_known": False,
                            }
                        add(
                            "orbbec_ego_audio/Audio",
                            "OrbbecEgoAudioChunkRecord",
                            data,
                            0,
                        )
                        if embedded:
                            pcm = {
                                key: value
                                for key, value in data.items()
                                if key not in ("wav_data_offset", "byte_count")
                            }
                            pcm["pcm_data"] = [0, 1] * count
                            if pcm_timing_mismatch:
                                pcm["timing"] = {
                                    **pcm["timing"],
                                    "first_sample_index": first + 1,
                                }
                            add(
                                "orbbec_ego_media/Audio",
                                "OrbbecEgoPcmAudioChunkRecord",
                                pcm,
                                0,
                            )
                        wav.extend(b"\x00\x01" * count)
            writer.finish()
        if not embedded:
            (root / "raw").mkdir()
            for side, data in video_bytes.items():
                (root / "raw" / f"{side}.h264").write_bytes(data)
            if audio:
                (root / "Audio.wav").write_bytes(wav)
        model = {
            "version": 1,
            "scope": "recording",
            "status": "validated",
            "reference_stream": "left_rgb",
            "model_id": "independent-test-model",
            "identity": IDENTITY.copy(),
            "profile": PROFILE.copy(),
            "recording_fingerprint": recording_fingerprint(path),
            "validation": {
                "method": "independent_physical_events",
                "max_abs_error_ns": 1000,
                "uncertainty_ns": 1000,
                "held_out_duration_ns": 1_800_000_000_000,
                "evidence": ["sha256:test-evidence"],
            },
            "epochs": [],
        }
        for epoch in epochs:
            maps = {
                stream: {
                    "source_anchor_ns": ORIGIN
                    + (2_000_000 if stream in ("accel", "gyro") else 0),
                    "reference_anchor_ns": ORIGIN,
                    "scale": 1.0,
                    "validation_start_ns": ORIGIN,
                    "validation_end_ns": ORIGIN + 1_800_000_000_000,
                }
                for stream in ALL
                if stream != "audio"
            }
            maps["audio"] = {
                "segments": [
                    {
                        "segment_id": epoch,
                        "first_sample_index": 0,
                        "sample_count": 40,
                        "reference_anchor_ns": ORIGIN,
                        "period_ns": 1_000_000,
                        "continuity_verified": True,
                        "validation_start_ns": ORIGIN,
                        "validation_end_ns": ORIGIN + 1_800_000_000_000,
                    }
                ]
            }
            model["epochs"].append(
                {
                    "epoch": epoch,
                    "valid_from_ns": ORIGIN,
                    "valid_to_ns": ORIGIN + (max(frame_ids) + 1) * PERIOD,
                    "streams": maps,
                }
            )
        return path, model

    return create


@pytest.mark.parametrize("embedded", [False, True])
def test_full_record_samples_media_and_half_open_windows(recording, embedded):
    path, model = recording(embedded=embedded)
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert len(windows) == 3
    assert all(window.complete for window in windows)
    assert windows[0].start_time_ns == ORIGIN
    assert windows[0].accel[0].time_ns == ORIGIN
    assert windows[0].accel[0].record.sample_time_raw_ns == ORIGIN + 80_000_000
    assert windows[0].accel[0].sample_time_raw_ns == ORIGIN + 2_000_000
    assert windows[0].accel[0].x_si == 0.25
    assert len(windows[0].gyro) == 10
    assert windows[0].gyro[-1].time_ns < windows[1].start_time_ns
    assert windows[0].left_rgb.sdk_metadata == ((19, 123456789),)
    assert [value.sample_count for value in windows[1].audio] == [7, 3]
    assert windows[1].audio[0].sample_offset == 10
    assert windows[1].audio[1].sample_offset == 0
    assert windows[0].left_rgb.media.storage == ("mcap" if embedded else "sidecar")
    assert windows[0].audio[0].media.byte_count == 20
    with pytest.raises(FrozenInstanceError):
        windows[0].complete = False
    report = analyze_recording_sync(path, required_streams=ALL, alignment=model)
    assert report["physical_error_measurements"] is None
    assert report["coverage"]["streams"]["gyro"]["unassigned_samples"] == 10
    assert len(report["coverage"]["unpaired_video"]["left_rgb"]) == 1
    assert report["ignored_noncanonical_records"] == 8
    assert report["five_stream_verified"]
    assert "x_si" not in report["windows"][0]["gyro"][0]
    assert report["windows"][0]["gyro"][0]["sample_count"] == 10
    assert (
        report["coverage"]["unassigned_sample_ranges"]["gyro"][0]["first_sample_index"]
        == 30
    )
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("missing_bounds", [False, True])
def test_external_model_cannot_extrapolate_unmeasured_interval(
    recording, missing_bounds
):
    path, model = recording()
    mapping = model["epochs"][0]["streams"]["gyro"]
    if missing_bounds:
        del mapping["validation_end_ns"]
    else:
        mapping["validation_end_ns"] = ORIGIN + 2 * PERIOD
    with pytest.raises(
        ValueError, match=r"validation_end_ns|independent validation bounds"
    ):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_audio_outside_independent_validation_is_unassigned(recording):
    path, model = recording()
    audio = model["epochs"][0]["streams"]["audio"]["segments"][0]
    audio["validation_start_ns"] = ORIGIN + PERIOD
    audio["validation_end_ns"] = ORIGIN + 2 * PERIOD
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert [window.complete for window in windows] == [False, True, False]
    assert windows[0].reasons == ("missing_audio",)
    assert windows[2].reasons == ("missing_audio",)
    report = analyze_recording_sync(path, required_streams=ALL, alignment=model)
    assert not report["five_stream_verified"]
    assert report["coverage"]["streams"]["audio"]["assigned_samples"] == 10
    assert report["coverage"]["streams"]["audio"]["unassigned_samples"] == 30


def test_audio_validation_boundary_is_not_hidden_by_sample_tolerance(recording):
    path, model = recording()
    audio = model["epochs"][0]["streams"]["audio"]["segments"][0]
    audio["validation_end_ns"] = ORIGIN + 2 * PERIOD - 1
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert windows[0].complete
    assert not windows[1].complete
    assert "audio_validation_boundary" in windows[1].reasons


def test_stereo_sequence_origins_do_not_override_sample_time_pairing(recording):
    path, model = recording(right_sequence_offset=1)
    windows = list(
        iter_sync_windows(
            path, required_streams=("left_rgb", "right_rgb"), alignment=model
        )
    )
    assert all(window.complete for window in windows)
    assert len({window.right_rgb.record.record_index for window in windows}) == len(
        windows
    )
    for window in windows:
        assert (
            window.right_rgb.record.sequence_number
            == window.left_rgb.record.sequence_number + 1
        )
        assert window.right_rgb.time_ns - window.left_rgb.time_ns == 300_000


def test_unrequested_imu_validation_bounds_do_not_block_stereo(recording):
    path, model = recording()
    del model["epochs"][0]["streams"]["gyro"]["validation_end_ns"]
    windows = list(
        iter_sync_windows(
            path, required_streams=("left_rgb", "right_rgb"), alignment=model
        )
    )
    assert all(window.complete for window in windows)


def test_legacy_audit_is_not_a_sync_certificate(recording):
    path, _ = recording(legacy_audio=True)
    report = analyze_recording_sync(path)
    assert report["alignment"]["status"] == "unverified"
    assert report["windows"] == []
    assert report["streams"]["gyro"]["samples"] == 40
    with pytest.raises(ValueError, match="validated"):
        list(iter_sync_windows(path, required_streams=ALL))


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda model: model["identity"].update(serial="other"), "serial"),
        (
            lambda model: model["recording_fingerprint"].update(mcap_sha256="wrong"),
            "fingerprint",
        ),
        (
            lambda model: model["validation"].update(max_abs_error_ns=9_000_000),
            "half-frame",
        ),
        (
            lambda model: model["validation"].update(held_out_duration_ns=1),
            "held_out_duration",
        ),
        (
            lambda model: model["epochs"][0]["streams"]["gyro"].update(
                scale=float("nan")
            ),
            "JSON compliant",
        ),
    ],
)
def test_reject_untrusted_or_mismatched_models(recording, change, match):
    path, model = recording()
    change(model)
    with pytest.raises(ValueError, match=match):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_epoch_barrier_and_query(recording):
    path, model = recording(epochs=(0, 1))
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert len(windows) == 6
    assert [window.epoch for window in windows] == [0, 0, 0, 1, 1, 1]
    selected = list(
        iter_sync_windows(
            path,
            required_streams=ALL,
            alignment=model,
            epoch=1,
            start_time_ns=ORIGIN + PERIOD,
            end_time_ns=ORIGIN + 3 * PERIOD,
        )
    )
    assert len(selected) == 2
    assert all(window.epoch == 1 for window in selected)


def test_gaps_and_missing_stereo_are_incomplete(recording):
    path, model = recording(
        frame_ids=(0, 1, 3), skip_gyro_sample=15, right_offset_ns=3_000_000
    )
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert not any(window.complete for window in windows)
    assert "missing_right_rgb" in windows[0].reasons
    assert "reference_gap" in windows[1].reasons
    assert "gyro_gap" in windows[1].reasons
    assert windows[1].end_time_ns == ORIGIN + 2 * PERIOD


def test_audio_continuity_and_required_subsets(recording):
    path, model = recording()
    model["epochs"][0]["streams"]["audio"]["segments"][0]["continuity_verified"] = False
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert all("audio_continuity_unverified" in window.reasons for window in windows)
    stereo = list(
        iter_sync_windows(
            path, required_streams=("left_rgb", "right_rgb"), alignment=model
        )
    )
    assert all(window.complete for window in stereo)


def test_absent_required_stream_fails_but_audit_succeeds(recording):
    path, model = recording(audio=False)
    with pytest.raises(ValueError, match="Required stream is absent: audio"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    report = analyze_recording_sync(path)
    assert not report["streams"]["audio"]["available"]
    assert report["alignment"]["status"] == "unverified"


def test_model_segment_barrier(recording):
    path, model = recording()
    first = model["epochs"][0]
    second = json.loads(json.dumps(first))
    first["valid_to_ns"] = ORIGIN + 2 * PERIOD
    second["valid_from_ns"] = ORIGIN + 2 * PERIOD
    second["model_id"] = "next"
    model["epochs"].append(second)
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert len(windows) == 2
    assert [window.model_id for window in windows] == ["independent-test-model", "next"]
    assert [window.model_segment for window in windows] == [0, 1]


@pytest.mark.parametrize("invalid_interval", ["empty", "overlap"])
def test_model_validity_intervals_are_checked(recording, invalid_interval):
    path, model = recording()
    if invalid_interval == "empty":
        model["epochs"][0]["valid_to_ns"] = model["epochs"][0]["valid_from_ns"]
    else:
        model["epochs"].append(json.loads(json.dumps(model["epochs"][0])))
    with pytest.raises(ValueError, match="nonempty and nonoverlapping"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


@pytest.mark.parametrize("invalid_audio_range", ["empty", "overlap"])
def test_audio_sample_ranges_are_checked(recording, invalid_audio_range):
    path, model = recording()
    segments = model["epochs"][0]["streams"]["audio"]["segments"]
    if invalid_audio_range == "empty":
        segments[0]["sample_count"] = 0
        expected = "sample_count must be an integer >= 1"
    else:
        segments.append(json.loads(json.dumps(segments[0])))
        expected = "Overlapping audio sample ranges"
    with pytest.raises(ValueError, match=expected):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_saved_capture_profile_maps_four_streams_without_host_accuracy_claim(recording):
    path, _ = recording(capture_profile=True, epochs=(0, 1))
    windows = list(iter_sync_windows(path, required_streams=ALL[:-1]))
    assert len(windows) == 6
    assert all(window.complete for window in windows)
    assert windows[0].accel[0].time_ns == ORIGIN
    assert windows[0].accel[0].sample_time_local_ns != ORIGIN
    with pytest.raises(ValueError, match="missing required stream audio"):
        list(iter_sync_windows(path, required_streams=ALL))


@pytest.mark.parametrize("embedded_profile", [False, True])
@pytest.mark.parametrize("duration", [900_000_000_000, 899_999_999_999])
def test_offline_model_fifteen_minute_validation_boundary(
    recording, embedded_profile, duration
):
    def change(profile):
        profile["max_capture_duration_ns"] = 900_000_000_000
        profile["validation"]["held_out_duration_ns"] = duration

    path, model = recording(
        capture_profile=embedded_profile,
        capture_profile_change=change,
    )
    model["validation"]["held_out_duration_ns"] = duration
    kwargs = {} if embedded_profile else {"alignment": model}
    if duration < 900_000_000_000:
        with pytest.raises(ValueError, match="held_out_duration_ns"):
            list(iter_sync_windows(path, required_streams=ALL[:-1], **kwargs))
    else:
        windows = list(iter_sync_windows(path, required_streams=ALL[:-1], **kwargs))
        assert len(windows) == 3
        assert all(window.complete for window in windows)


@pytest.mark.parametrize(
    "change, expected",
    [
        (
            lambda profile: profile["identity"].update(sdk_binary_sha256="34" * 32),
            "identity mismatch",
        ),
        (
            lambda profile: profile["identity"].pop("sdk_binary_sha256"),
            "requires sdk_binary_sha256",
        ),
        (
            lambda profile: profile["streams"]["gyro"].update(scale=1.00001),
            "unit stream scale",
        ),
        (
            lambda profile: profile.update(max_capture_duration_ns=0),
            "max_capture_duration_ns",
        ),
        (
            lambda profile: profile.update(max_capture_duration_ns=True),
            "max_capture_duration_ns",
        ),
        (
            lambda profile: profile.update(max_capture_duration_ns=1_800_000_000_001),
            "duration exceeds",
        ),
    ],
)
def test_reusable_profile_rejects_unverified_sdk_drift_and_duration(
    recording, change, expected
):
    path, _ = recording(capture_profile=True, capture_profile_change=change)
    with pytest.raises(ValueError, match=expected):
        list(iter_sync_windows(path, required_streams=ALL[:-1]))


def test_capture_bridge_cannot_extend_the_verified_profile_duration(recording):
    path, _ = recording(
        capture_profile=True,
        bridge_change=lambda bridge: bridge.update(
            valid_to_ns=bridge["valid_to_ns"] + PERIOD
        ),
    )
    with pytest.raises(ValueError, match="validity must match its verified duration"):
        list(iter_sync_windows(path, required_streams=ALL[:-1]))


def test_expired_profile_keeps_only_its_trusted_interval(recording):
    path, _ = recording(
        capture_profile=True,
        capture_profile_change=lambda profile: profile.update(
            max_capture_duration_ns=2 * PERIOD + PERIOD // 2
        ),
        bridge_change=lambda bridge: bridge.update(
            status="expired", expired_at_reference_ns=ORIGIN + 3 * PERIOD
        ),
    )
    windows = list(iter_sync_windows(path, required_streams=ALL[:-1]))
    assert len(windows) == 2
    assert all(window.complete for window in windows)
    assert windows[-1].end_time_ns == ORIGIN + 2 * PERIOD
    report = analyze_recording_sync(path, required_streams=ALL[:-1])
    assert report["coverage"]["streams"]["gyro"]["unassigned_samples"] == 20
    assert len(report["coverage"]["unpaired_video"]["left_rgb"]) == 2


def test_first_mapped_boundaries_exclude_the_initial_arrival_fallback(recording):
    path, _ = recording(
        capture_profile=True,
        bridge_change=lambda bridge: bridge["first_mapped_raw_ns"].update(
            left_rgb=ORIGIN + PERIOD, right_rgb=ORIGIN + PERIOD + 300_000
        ),
        local_transition=True,
    )
    windows = list(iter_sync_windows(path, required_streams=ALL[:-1]))
    assert len(windows) == 2
    assert windows[0].start_time_ns == ORIGIN + PERIOD
    assert all(window.complete for window in windows)


def test_first_mapped_boundaries_can_grow_across_calibration_snapshots(recording):
    path, _ = recording(
        capture_profile=True,
        bridge_change=lambda bridge: bridge.update(first_mapped_raw_ns={}),
        bridge_snapshots=(
            {"first_mapped_raw_ns": {"left_rgb": ORIGIN}},
            {
                "first_mapped_raw_ns": {
                    "left_rgb": ORIGIN,
                    "right_rgb": ORIGIN + 300_000,
                    "accel": ORIGIN + 2_000_000,
                    "gyro": ORIGIN + 2_000_000,
                }
            },
        ),
    )
    windows = list(iter_sync_windows(path, required_streams=ALL[:-1]))
    assert len(windows) == 3
    assert all(window.complete for window in windows)


def test_first_mapped_boundary_cannot_change_within_an_epoch(recording):
    path, _ = recording(
        capture_profile=True,
        bridge_snapshots=({"first_mapped_raw_ns": {"left_rgb": ORIGIN + PERIOD}},),
    )
    with pytest.raises(ValueError, match="Conflicting first mapped"):
        list(iter_sync_windows(path, required_streams=ALL[:-1]))


@pytest.mark.parametrize(
    "change, expected",
    [
        (
            lambda bridge: bridge.update(
                status="expired", expired_at_reference_ns=ORIGIN + PERIOD
            ),
            "expiry contradicts",
        ),
        (
            lambda bridge: bridge.update(expired_at_reference_ns=bridge["valid_to_ns"]),
            "expiry contradicts",
        ),
        (
            lambda bridge: bridge["first_mapped_raw_ns"].update(
                gyro=bridge["valid_to_ns"] + 2_000_000
            ),
            "First mapped sample is outside",
        ),
    ],
)
def test_capture_bridge_rejects_inconsistent_trust_boundaries(
    recording, change, expected
):
    path, _ = recording(capture_profile=True, bridge_change=change)
    with pytest.raises(ValueError, match=expected):
        list(iter_sync_windows(path, required_streams=ALL[:-1]))


def test_expired_bridge_cannot_reactivate_without_a_new_epoch(recording):
    path, _ = recording(
        capture_profile=True,
        bridge_change=lambda bridge: bridge.update(status="expired"),
        bridge_snapshots=({"status": "active"},),
    )
    with pytest.raises(ValueError, match="cannot become active"):
        list(iter_sync_windows(path, required_streams=ALL[:-1]))


def test_external_alignment_checks_sdk_digest_when_present(recording):
    path, model = recording()
    model["identity"]["sdk_binary_sha256"] = "34" * 32
    with pytest.raises(ValueError, match="SDK binary digest does not match"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_recording_bound_alignment_can_read_legacy_identity_without_sdk_digest(
    recording,
):
    identity = {
        key: value for key, value in IDENTITY.items() if key != "sdk_binary_sha256"
    }
    path, model = recording(recorded_identity=identity)
    del model["identity"]["sdk_binary_sha256"]
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert len(windows) == 3
    assert all(window.complete for window in windows)


def test_local_clock_transition_does_not_hide_a_valid_raw_time_relation(recording):
    path, _ = recording(capture_profile=True, local_transition=True)
    windows = list(iter_sync_windows(path, required_streams=ALL[:-1]))
    assert len(windows) == 3
    assert all(window.complete for window in windows)
    assert (
        windows[0].left_rgb.record.sample_time_local_ns
        > windows[1].left_rgb.record.sample_time_local_ns
    )
    assert windows[0].start_time_ns < windows[1].start_time_ns


def test_recording_bound_clock_drift_remains_supported(recording):
    path, model = recording()
    model["epochs"][0]["streams"]["gyro"]["scale"] = 1.00001
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    assert len(windows) == 3
    assert all(window.complete for window in windows)
    assert windows[0].gyro[1].time_ns == ORIGIN + 1_000_010


def test_populated_spatial_calibration_is_decoded_with_timing_provenance(recording):
    from mcap.reader import make_reader
    from isaaccapture.ego._records import RecordDecoder

    path, _ = recording()
    with path.open("rb") as source:
        schema, channel, message = next(
            make_reader(source).iter_messages(log_time_order=False)
        )
    assert channel.topic == "orbbec_ego_calibration/Calibration"
    data = RecordDecoder(schema.data, schema.name).decode(message.data)["data"]
    assert data["color_left"]["width"] == 16
    assert data["color_left"]["distortion"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert data["left_to_right"]["rotation"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert data["gyro_to_left"]["translation_mm"] == [4, 5, 6]
    assert json.loads(data["time_alignment"]["json"])["status"] == "unverified"


def test_right_window_reference_uses_left_model_clock(recording):
    path, model = recording()
    model["epochs"][0]["streams"]["right_rgb"]["reference_anchor_ns"] -= 300_000
    windows = list(
        iter_sync_windows(
            path, required_streams=ALL, alignment=model, reference_stream="right_rgb"
        )
    )
    assert len(windows) == 3
    assert all(window.complete for window in windows)
    assert windows[0].start_time_ns == ORIGIN
    assert windows[0].right_rgb.record.sample_time_raw_ns == ORIGIN + 300_000


def test_clock_regression_is_reported_instead_of_hidden_by_sort(recording):
    path, model = recording(raw_regression=True)
    with pytest.raises(ValueError, match="nonmonotonic raw"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    report = analyze_recording_sync(path)
    assert "nonmonotonic raw" in report["issues"][0]
    assert not report["five_stream_verified"]


def test_unrequested_imu_clock_failure_does_not_block_stereo(recording):
    path, model = recording(raw_regression=True)
    windows = list(iter_sync_windows(path, required_streams=ALL[:2], alignment=model))
    assert all(window.complete for window in windows)
    assert not any(window.accel or window.gyro or window.audio for window in windows)
    report = analyze_recording_sync(path, required_streams=ALL[:2], alignment=model)
    assert report["alignment"]["status"] == "validated"
    assert "nonmonotonic raw" in report["issues"][0]


def test_embedded_audio_timing_must_match_its_index(recording):
    with pytest.raises(
        ValueError, match="Embedded audio disagrees with its metadata: timing"
    ):
        recording(embedded=True, pcm_timing_mismatch=True)


def test_zero_length_audio_is_rejected(recording):
    with pytest.raises(
        ValueError, match="Audio sample count, rate, and channels must be positive"
    ):
        recording(audio_chunks=((0, 0),))


def test_imu_vectors_are_immutable_views_without_per_sample_dicts(recording):
    from mcap.reader import make_reader
    from isaaccapture.ego._records import RecordDecoder

    path, _ = recording(imu_samples=2000)
    with path.open("rb") as source:
        for schema, channel, message in make_reader(source).iter_messages():
            if channel.topic == "orbbec_ego_imu_accel/Imu":
                record = RecordDecoder(schema.data, schema.name).decode(message.data)
                samples = record["data"]["samples"]
                assert isinstance(samples, np.ndarray)
                assert samples.shape == (2000,)
                assert samples.nbytes == 2000 * 48
                assert not samples.flags.owndata
                assert not samples.flags.writeable
                assert (
                    samples[1999]["sample_time_raw_device_clock_ns"]
                    == ORIGIN + 2_001_000_000
                )
                with pytest.raises(ValueError, match="read-only"):
                    samples[0]["x_si"] = 0
                break
        else:
            pytest.fail("Fixture did not contain an IMU record")


def test_continuity_flag_is_not_a_truthy_string(recording):
    path, model = recording()
    model["epochs"][0]["streams"]["audio"]["segments"][0]["continuity_verified"] = (
        "false"
    )
    with pytest.raises(ValueError, match="continuity_verified must be a boolean"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_sidecar_hash_and_external_path_are_checked(recording, tmp_path):
    path, model = recording()
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(json.dumps(model))
    assert (
        len(
            list(
                iter_sync_windows(path, required_streams=ALL, alignment=alignment_path)
            )
        )
        == 3
    )
    media = path.parent / "raw/ColorLeft.h264"
    media.write_bytes(b"xxxx" * 4)
    with pytest.raises(ValueError, match="fingerprint"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=alignment_path))


@pytest.mark.parametrize(
    "stream, side", [("left_rgb", "ColorLeft"), ("right_rgb", "ColorRight")]
)
@pytest.mark.parametrize("trailing_bytes", [1, 30_395])
def test_sidecar_tail_is_reported_without_changing_associations(
    recording, stream, side, trailing_bytes
):
    path, model = recording()
    windows = list(iter_sync_windows(path, required_streams=ALL, alignment=model))
    media = path.parent / "raw" / f"{side}.h264"
    with media.open("ab") as output:
        output.write(bytes(trailing_bytes))

    with pytest.raises(ValueError, match="fingerprint"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))

    model["recording_fingerprint"] = recording_fingerprint(path)
    assert (
        list(iter_sync_windows(path, required_streams=ALL, alignment=model)) == windows
    )
    report = analyze_recording_sync(path, required_streams=ALL, alignment=model)
    assert report["alignment"]["status"] == "validated"
    assert report["issues"] == []
    assert report["streams"][stream]["sidecar_trailing_bytes"] == trailing_bytes
    for name in ("left_rgb", "right_rgb"):
        assert report["streams"][name]["records"] == 4
        assert report["streams"][name]["missing_media_records"] == 0
        if name != stream:
            assert report["streams"][name]["sidecar_trailing_bytes"] == 0
    assert report["coverage"]["window_count"] == len(windows)
    assert report["coverage"]["complete_windows"] == len(windows)


@pytest.mark.parametrize(
    "stream, side", [("left_rgb", "ColorLeft"), ("right_rgb", "ColorRight")]
)
def test_sidecar_truncation_still_rejects_missing_indexed_bytes(
    recording, stream, side
):
    path, model = recording()
    media = path.parent / "raw" / f"{side}.h264"
    with media.open("r+b") as output:
        output.truncate(media.stat().st_size - 1)
    model["recording_fingerprint"] = recording_fingerprint(path)

    report = analyze_recording_sync(path, required_streams=ALL, alignment=model)
    assert report["alignment"]["status"] == "invalid"
    assert report["streams"][stream]["missing_media_records"] == 1
    assert report["streams"][stream]["sidecar_trailing_bytes"] == 0
    assert report["coverage"]["complete_windows"] == 0
    with pytest.raises(ValueError, match="shorter than its metadata index"):
        list(iter_sync_windows(path, required_streams=ALL, alignment=model))


def test_truncated_flatbuffer_is_rejected(recording):
    from mcap.reader import make_reader
    from isaaccapture.ego._records import RecordDecoder

    path, _ = recording()
    with path.open("rb") as source:
        schema, _, message = next(make_reader(source).iter_messages())
    decoder = RecordDecoder(schema.data, schema.name)
    with pytest.raises(ValueError, match="Truncated or invalid FlatBuffer"):
        decoder.decode(message.data[:15])
