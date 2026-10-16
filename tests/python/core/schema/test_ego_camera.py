# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    EgoFrameMetadata,
    EgoFrameMetadataRecord,
    EgoAudioChunk,
    EgoAudioChunkRecord,
    EgoCalibration,
    EgoCalibrationRecord,
    EgoCameraIntrinsics,
    EgoCameraStream,
    EgoCaptureHealth,
    EgoRecoveryStage,
    EgoDevicePropertyValue,
    EgoDiagnosticEvent,
    EgoSdkReport,
    EgoCaptureSummary,
    EgoRecoveryEvent,
    EgoDeviceSnapshot,
    EgoDeviceState,
    EgoDeviceStateRecord,
    EgoEncodedVideoFrame,
    EgoEncodedVideoFrameRecord,
    EgoExtrinsics,
    EgoFrameMetadataEntry,
    EgoImuBatch,
    EgoImuBatchRecord,
    EgoImuSample,
    EgoImuSensor,
    EgoPcmAudioChunk,
    EgoPcmAudioChunkRecord,
    EgoPixelFormat,
)


def test_camera_payload_and_retained_record_views():
    metadata = EgoFrameMetadata(
        stream=EgoCameraStream.ColorRight,
        sequence_number=2**64 - 1,
        width=1600,
        height=1300,
        fps=30,
        pixel_format=EgoPixelFormat.H265,
        encoded_bytes=123,
        sdk_metadata=[EgoFrameMetadataEntry(1, 99)],
        capture_epoch=3,
    )
    assert metadata.stream == EgoCameraStream.ColorRight
    assert metadata.sequence_number == 2**64 - 1
    assert metadata.sdk_metadata[0].value == 99
    assert "ColorRight" in repr(metadata)
    with pytest.raises(AttributeError):
        metadata.capture_epoch = 4

    record = EgoFrameMetadataRecord(metadata, DeviceDataTimestamp(1, 2, 3))
    payload = record.data
    timestamp = record.timestamp
    del metadata, record
    gc.collect()
    assert payload.capture_epoch == 3
    assert timestamp.sample_time_raw_device_clock == 3
    assert EgoFrameMetadataRecord(None, DeviceDataTimestamp(1, 2, 3)).data is None


def test_auxiliary_payloads_and_nested_calibration():
    imu = EgoImuBatch(
        sensor=EgoImuSensor.Gyro,
        sample_rate_hz=1000,
        samples=[EgoImuSample(1, 2, 3, 24, 100, 200)],
        capture_epoch=2,
    )
    assert imu.samples[0].z_si == 3
    assert imu.samples[0].sample_time_raw_device_clock_ns == 200
    audio = EgoAudioChunk(sample_rate_hz=48000, wav_data_offset=44, capture_epoch=2)
    assert audio.wav_data_offset == 44
    intrinsics = EgoCameraIntrinsics(width=1600, fx=800, distortion=[1, 2])
    calibration = EgoCalibration(
        device_uid="ego",
        color_left=intrinsics,
        left_to_right=EgoExtrinsics(rotation=[1, 0, 0], translation_mm=[60, 0, 0]),
        capture_epoch=2,
    )
    left = calibration.color_left
    del calibration, intrinsics
    gc.collect()
    assert left.width == 1600
    assert left.distortion == [1, 2]
    assert EgoCalibration().color_left is None

    state = EgoDeviceState(
        "ego",
        2,
        100,
        EgoCaptureSummary(
            EgoCaptureHealth.Warning, "queue warning", 4096, 3482, 0, 0, False
        ),
    )
    assert state.event.capture_health == EgoCaptureHealth.Warning
    assert not hasattr(state.event, "properties")
    assert not hasattr(state.event, "stage")


def test_embedded_video_and_pcm_record_views():
    video = EgoEncodedVideoFrame(
        stream=EgoCameraStream.ColorRight,
        encoded_data=[0, 0, 0, 1, 0x65],
        capture_epoch=2,
    )
    audio = EgoPcmAudioChunk(
        sample_rate_hz=48000, pcm_data=[1, 0, 2, 0], capture_epoch=2
    )
    timestamp = DeviceDataTimestamp(100, 90, 3)
    video_record = EgoEncodedVideoFrameRecord(video, timestamp)
    audio_record = EgoPcmAudioChunkRecord(audio, timestamp)
    assert video_record.data.encoded_data == [0, 0, 0, 1, 0x65]
    assert audio_record.data.pcm_data == [1, 0, 2, 0]
    assert audio_record.timestamp.sample_time_local_common_clock == 90


def test_no_tracked_wrapper_compatibility_layer():
    import isaaccapture.schema as schema

    assert not hasattr(schema, "EgoFrameMetadataTrackedT")
    assert not hasattr(schema, "EgoAudioChunkTrackedT")


@pytest.mark.parametrize(
    ("payload_type", "record_type"),
    [
        (EgoFrameMetadata, EgoFrameMetadataRecord),
        (EgoImuBatch, EgoImuBatchRecord),
        (EgoAudioChunk, EgoAudioChunkRecord),
        (EgoCalibration, EgoCalibrationRecord),
        (EgoDeviceState, EgoDeviceStateRecord),
        (EgoEncodedVideoFrame, EgoEncodedVideoFrameRecord),
        (EgoPcmAudioChunk, EgoPcmAudioChunkRecord),
    ],
)
def test_every_record_maps_absence_and_retains_immutable_payload(
    payload_type, record_type
):
    timestamp = DeviceDataTimestamp(100, 90, 80)
    assert record_type(None, timestamp).data is None
    payload = (
        payload_type("ego", 3, 90, EgoSdkReport(0, 0, 0, 0, 0))
        if payload_type is EgoDeviceState
        else payload_type(capture_epoch=3)
    )
    record = record_type(payload, timestamp)
    payload = record.data
    del record
    gc.collect()
    assert payload.capture_epoch == 3
    assert not hasattr(payload, "data")
    with pytest.raises(AttributeError):
        payload.capture_epoch = 4


@pytest.mark.parametrize(
    "payload,tag,field,value,absent",
    [
        (
            EgoSdkReport(7, 2, 0, 0, 123),
            EgoDiagnosticEvent.SdkReport,
            "error_flags",
            0,
            "imu_temperature_c",
        ),
        (
            EgoCaptureSummary(EgoCaptureHealth.Healthy, "", 4096, 1, 0, 0, True),
            EgoDiagnosticEvent.CaptureSummary,
            "dropped_events",
            0,
            "work_mode",
        ),
        (
            EgoRecoveryEvent(EgoRecoveryStage.Recovered, "stream resumed", 4),
            EgoDiagnosticEvent.RecoveryEvent,
            "stage",
            EgoRecoveryStage.Recovered,
            "capture_health",
        ),
        (
            EgoDeviceSnapshot(0.0, []),
            EgoDiagnosticEvent.DeviceSnapshot,
            "imu_temperature_c",
            0.0,
            "storage_free_bytes",
        ),
    ],
)
def test_typed_events_own_only_their_fields(payload, tag, field, value, absent):
    state = EgoDeviceState("ego", 3, 100, payload)
    assert state.diagnostic_version == 2
    assert state.event_type == tag
    assert state.observed_at_ns == 100
    assert getattr(state.event, field) == value
    assert not hasattr(state.event, absent)
    assert not hasattr(state, "valid_fields")
    with pytest.raises(AttributeError):
        setattr(state.event, field, value)


def test_snapshot_optional_temperature_and_retained_union_view():
    snapshot = EgoDeviceSnapshot(None, [EgoDevicePropertyValue(279, 0)])
    state = EgoDeviceState("ego", 2, 100, snapshot)
    record = EgoDeviceStateRecord(state, DeviceDataTimestamp(110, 100, 0))
    event = record.data.event
    del record, state, snapshot
    gc.collect()
    assert event.imu_temperature_c is None
    assert event.properties[0].value == 0
    assert EgoDeviceSnapshot(0.0, []).imu_temperature_c == 0.0
    assert EgoSdkReport(1, 0, 0, 17, 0).error_flags == 17
    assert EgoSdkReport(2, 0, 0, 0, 0).error_flags == 0


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf")])
def test_snapshot_rejects_nonfinite_temperature(temperature):
    with pytest.raises(ValueError):
        EgoDeviceSnapshot(temperature, [])


def test_event_constructor_requires_its_observations():
    for cls in (EgoSdkReport, EgoCaptureSummary, EgoRecoveryEvent, EgoDeviceSnapshot):
        with pytest.raises(TypeError):
            cls()
    with pytest.raises(TypeError):
        EgoDeviceState("ego", 0, 100, None)
    with pytest.raises(TypeError):
        EgoDeviceState("ego", 0, 100, EgoCalibration())
    with pytest.raises(ValueError):
        EgoDeviceState("ego", 0, 0, EgoSdkReport(0, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        EgoDeviceState("", 0, 100, EgoSdkReport(0, 0, 0, 0, 0))
